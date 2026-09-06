//! Naive x86-64 code generation implementation.
//!
//! This module implements a straightforward, correctness-focused code generation
//! strategy for x86-64. It prioritizes simplicity and correctness over performance.

use alloc::collections::btree_map::BTreeMap;
use portal_solutions_asm_x86_64::RegisterClass;
use portal_solutions_asm_x86_64::out::arg::{ArgKind, MemArg, MemArgKind};
use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::ops::ProbeTableConfig;
use portal_solutions_blitz_common::shard::{CallTarget, SecondCtxConfig};
use portal_solutions_blitz_common::wasm_encoder::{
    self, Catch, FuncType, Instruction,
    reencode::{self as reencode, Reencode},
};

/// Static Context Register (SCR) — r14 on x86-64.
///
/// Holds a pointer to the cross-shard function-pointer table when sharding is
/// active. See `docs/second-context-register.md`.
pub const SCR: Reg = Reg(14);

/// How WASM linear-memory addresses are translated to host addresses by the
/// load/store lowering.
///
/// The naive backend historically used [`MemBase::Raw`]: the WASM address is
/// used directly as a host pointer, so the runtime (or emulator) must map linear
/// memory such that the WASM offset equals the host virtual address. That works
/// under Unicorn and for runtimes that can `mmap` at a fixed VA, but not for an
/// ordinary OS process. [`MemBase::WasmMemSymbol`] matches the C backend: each
/// access is computed as `__wasm_mem + (uint32_t)addr`, where `__wasm_mem` is a
/// `uint8_t*` the runtime defines. The full-binary recompiler selects the symbol
/// mode; existing tests keep the default raw mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemBase {
    /// WASM address used directly as a host pointer (default; legacy behavior).
    /// Equivalent to a numeric memory base of zero — no `__wasm_mem` add.
    #[default]
    Raw,
    /// Address as `__wasm_mem + (uint32_t)addr`, matching the C backend.
    WasmMemSymbol,
}

impl MemBase {
    /// True when the memory's host base is zero (identity / [`Self::Raw`]).
    pub fn is_zero(self) -> bool {
        matches!(self, Self::Raw)
    }
}

use crate::{
    out::{Writer, arg::Arg},
    *,
};

/// Sharding state carried in [`State`] when cross-shard call dispatch is needed.
///
/// Holds a reference to the [`ShardMap`] for the duration of the compilation,
/// allowing [`call_target`][NaiveShardState::call_target] to classify each call
/// instruction without unsafe code.
///
/// [`ShardMap`]: portal_solutions_blitz_common::shard::ShardMap
#[derive(Clone, Copy)]
pub struct NaiveShardState<'a> {
    pub config: SecondCtxConfig,
    /// Shard index of the function currently being compiled.
    pub current_shard: usize,
    pub imports_len: u32,
    pub map: &'a (dyn portal_solutions_blitz_common::shard::ShardMap + 'a),
}

impl<'a> NaiveShardState<'a> {
    pub fn new(
        config: SecondCtxConfig,
        current_shard: usize,
        imports_len: u32,
        map: &'a (dyn portal_solutions_blitz_common::shard::ShardMap + 'a),
    ) -> Self {
        Self {
            config,
            current_shard,
            imports_len,
            map,
        }
    }

    /// Classify a call to `callee_fn` (WASM-space function index).
    pub fn call_target(&self, callee_fn: u32) -> CallTarget {
        if callee_fn < self.imports_len {
            return CallTarget::Import;
        }
        let callee_shard = self.map.shard_for(callee_fn);
        if callee_shard == self.current_shard {
            CallTarget::Local
        } else {
            CallTarget::CrossShard {
                table_slot: callee_fn,
            }
        }
    }
}

/// State tracker for x86-64 code generation.
///
/// The lifetime `'a` is the lifetime of the [`ShardMap`] reference held in
/// [`shard`][State::shard].  When sharding is not active (`shard` is `None`)
/// the lifetime has no constraints and can be `'static` or elided.
///
/// [`ShardMap`]: portal_solutions_blitz_common::shard::ShardMap
#[derive(Default)]
pub struct State<'a> {
    pub local_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub label_index: usize,
    pub if_stack: Vec<Endable>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` so probes can be emitted after
    /// the function-entry label is placed (ensuring every call — linear or
    /// via label-jump — passes through the counter and handler-dispatch check).
    pub probes: Option<ProbeTableConfig>,
    /// Next probe id to assign (function entry consumes probe 0; each
    /// loop/block consumes the next).  See `emit_probe_site`.
    pub next_probe_id: u32,
    /// Present when sharding is active. Used to classify `Call` instructions
    /// as intra-shard (direct label) or cross-shard (SCR-relative indirect).
    pub shard: Option<NaiveShardState<'a>>,
    /// Default how linear-memory load/store addresses are translated for every
    /// memory index. Defaults to [`MemBase::Raw`] (legacy raw-pointer behavior).
    pub mem_base: MemBase,
    /// Per-`memory_index` overrides of [`Self::mem_base`]. A [`MemBase::Raw`]
    /// (zero-base) entry skips the `__wasm_mem` add for that memory only.
    pub mem_base_by_index: BTreeMap<u32, MemBase>,
}

impl State<'_> {
    /// Resolve the memory base for `memory_index` (override or default).
    pub fn mem_base_for(&self, memory_index: u32) -> MemBase {
        self.mem_base_by_index
            .get(&memory_index)
            .copied()
            .unwrap_or(self.mem_base)
    }
}

/// Magic sentinel pushed onto the CTX stack to mark a TryTable frame.
/// Chosen to be unlikely to occur as a real label address.
pub const TRYTABLE_SENTINEL: u64 = 0xE4C3_E4C3_E4C3_E4C3;

/// Represents a control flow structure that needs an end marker.
pub enum Endable {
    /// A branch target.
    Br,
    /// An if statement with its label index.
    If { idx: usize },
    /// A try_table block.
    ///
    /// `exit_idx`          — label placed at the start of the body (branch target for `br N`).
    /// `dispatch_idx`      — label for the exception dispatch stub (jumped to by `throw`).
    /// `after_dispatch_idx`— label placed after the dispatch stub (normal fall-through exit).
    /// `catches`           — catch clauses, cloned from the TryTable instruction.
    TryTable {
        exit_idx: usize,
        dispatch_idx: usize,
        after_dispatch_idx: usize,
        catches: alloc::boxed::Box<[Catch]>,
    },
}

/// Extension trait for x86-64 code writers.
///
/// Provides methods for generating x86-64 assembly code for WASM operations,
/// including branches, calls, and instruction handling.
pub trait WriterExt<Context>: Writer<X64Label, Context> {
    /// Generates code for a branch instruction.
    ///
    /// Emits x86-64 assembly to jump to the target label specified by the
    /// relative depth in the control flow stack.
    ///
    /// # Arguments
    ///
    /// * `arch` - The x86-64 architecture variant
    /// * `state` - Current compilation state
    /// * `relative_depth` - Depth of the target label in control flow stack
    fn br(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        relative_depth: u32,
    ) -> Result<(), Self::Error> {
        self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        for _ in 0..=relative_depth {
            self.pop(ctx, arch, &Reg(0))?;
            self.pop(ctx, arch, &Reg(1))?;
        }
        self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        self.mov(ctx, arch, &RSP, &Reg(1))?;
        self.jmp(ctx, arch, &Reg(0))?;
        Ok(())
    }

    /// Raw-byte emission helper for instructions the Writer doesn't model
    /// (ROL/ROR/LZCNT/TZCNT/POPCNT).  `db` is on WriterCore, available on
    /// every concrete writer (IcedWriter etc.).
    fn raw(&mut self, ctx: &mut Context, arch: X64Arch, bytes: &[u8]) -> Result<(), Self::Error> {
        self.db(ctx, arch, bytes)
    }

    /// Signed/unsigned integer compare helper (phase B, spec tests).
    ///
    /// Pops rhs then lhs, compares, and pushes 1/0. `is32` truncates
    /// operands to 32 bits for i32 comparisons first. Uses RAX/RDX/RCX.
    fn int_cmp(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        cc: ConditionCode,
        is32: bool,
    ) -> Result<(), Self::Error> {
        self.pop(ctx, arch, &Reg(0))?; // rhs
        self.pop(ctx, arch, &Reg(1))?; // lhs
        if is32 {
            // Truncate both operands in place (writing r32 zero-extends).
            let r0 = MemArgKind::NoMem(ArgKind::Reg {
                reg: Reg(0),
                size: MemorySize::_32,
            });
            let r1 = MemArgKind::NoMem(ArgKind::Reg {
                reg: Reg(1),
                size: MemorySize::_32,
            });
            self.mov(ctx, arch, &r0, &r0)?;
            self.mov(ctx, arch, &r1, &r1)?;
        }
        self.cmp(ctx, arch, &Reg(1), &Reg(0))?; // flags: lhs vs rhs
        self.mov64(ctx, arch, &Reg(2), 0)?;
        self.mov64(ctx, arch, &Reg(0), 1)?;
        self.cmovcc(ctx, arch, cc, &Reg(2), &Reg(0))?; // if cc: result = 1
        self.push(ctx, arch, &Reg(2))
    }

    /// Rotate helper (phase B). Pops count then value, pushes the rotated
    /// value. ROL/ROR mask the count to the operand width (mod 32/64),
    /// matching WASM rotl/rotr semantics.
    fn int_rot(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        op: &Instruction<'_>,
        left: bool,
    ) -> Result<(), Self::Error> {
        let is32 = matches!(op, Instruction::I32Rotl | Instruction::I32Rotr);
        self.pop(ctx, arch, &Reg(1))?; // count
        self.pop(ctx, arch, &Reg(0))?; // value
        // Count must be in CL (CL = RCX low byte). Reg(1) == RCX already.
        // ROL/ROR r64, CL : 48 D3 /0|/1 with modrm C1 (mod=11, rm=001)
        // ROL/ROR r32, CL :    D3 /0|/1 with modrm C1 (32-bit operand size)
        // Encodings (opcode low 3 bits select reg field of modrm):
        //   ROL: D3 C1 (32) / 48 D3 C1 (64); ROR: D3 C9 (32) / 48 D3 C9 (64)
        if is32 {
            let b: [u8; 2] = if left { [0xD3, 0xC1] } else { [0xD3, 0xC9] };
            self.raw(ctx, arch, &b)?;
            // Zero-extend the 32-bit result.
            let r = MemArgKind::NoMem(ArgKind::Reg {
                reg: Reg(0),
                size: MemorySize::_32,
            });
            self.mov(ctx, arch, &r, &r)?;
        } else {
            let b: [u8; 3] = if left {
                [0x48, 0xD3, 0xC1]
            } else {
                [0x48, 0xD3, 0xC9]
            };
            self.raw(ctx, arch, &b)?;
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Sign-extend the low 8/16 bits of the top of stack (phase B).
    /// `is64` selects the i64 (REX.W MOVSX) result form.
    fn sign_extend(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        width: MemorySize,
        is64: bool,
    ) -> Result<(), Self::Error> {
        self.pop(ctx, arch, &Reg(0))?;
        // MOVSX r64, r/m8 : 48 0F BE C0   MOVSX r32, r/m8 : 0F BE C0
        // MOVSX r64, r/m16: 48 0F BF C0   MOVSX r32, r/m16: 0F BF C0
        // (dst = same register as src's full form: Reg(0) = RAX/EAX)
        let opbyte: u8 = match width {
            MemorySize::_8 => 0xBE,
            _ => 0xBF,
        };
        if is64 {
            let b = [0x48, 0x0F, opbyte, 0xC0];
            self.raw(ctx, arch, &b)?;
        } else {
            // 32-bit dest: movsx eax, al/al zero-extends? No — movsx r32, r/m8
            // sign-extends into 32 bits, and writing eax zero-extends to rax,
            // giving the correct i32 value in the 64-bit register.
            let b = [0x0F, opbyte, 0xC0];
            self.raw(ctx, arch, &b)?;
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Emit a control-flow probe site (`TailTakeover` binding) for a
    /// loop/block header, consuming the next `probe_id`.  No-op when probes
    /// are disabled.
    ///
    /// Placed after the site's entry label and CTX-frame push, so the
    /// specialization tail-jump (if installed) inherits the operand-stack /
    /// CTX-frame layout of the generic site entry (see the blitz-specialize
    /// stack-state contract).  Uses `Reg(2)` (RDX) as scratch.
    fn emit_control_flow_probe(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
            let probe_id = state.next_probe_id;
            state.next_probe_id += 1;
            let mut bw = crate::codegen::BlitzW::new(self, ctx, arch);
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                probe_id,
                2,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Emit the optional control-flow probe at a function boundary.
    ///
    /// # Placement
    /// - **NaiveAbi**: call from `emit_start_body` / the `StartBody` handler,
    ///   *after* the function-entry label has been placed.  Use `scratch = Reg(2)`
    ///   (RDX); Reg(0)/Reg(1) hold old-CTX and return-addr needed by StartBody.
    /// - **SysVAbi**: call from inside the `StartFn` handler, *after* `set_label`
    ///   but *before* `push rbp`.  Use `scratch = Reg(0)` (RAX); SysV arg
    ///   registers (Reg 1/2/6/7/8/9) are untouched.
    ///
    /// The tail-jump transfers control to the outer-JIT specialisation with the
    /// current register/stack state fully intact for the ABI in question.
    /// Generates x86-64 assembly code for a machine operator.
    ///
    /// Main entry point for translating WASM machine operators into x86-64
    /// assembly. Handles all WASM operations including arithmetic, memory access,
    /// control flow, and function calls.
    ///
    /// # Arguments
    ///
    /// * `arch` - The x86-64 architecture variant
    /// * `state` - Current compilation state
    /// * `func_imports` - Information about imported functions
    /// * `op` - The machine operator to translate
    /// * `rewriter` - Re-encoder for instruction format conversion
    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Self: Sized,
        Err: From<Self::Error> + From<reencode::Error<E>>,
    {
        if target != state.body {
            // First-instruction guard: see comment in `_handle_op` below.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jmp_label(
                    ctx,
                    arch,
                    X64Label::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )
                .map_err(Err::from)?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, X64Label::Indexed { idx })
                        .map_err(Err::from)?;
                }
            }
        }
        //Stack Frame: r&Reg::CTX[&Reg(0)] => local variable frame
        match op {
            MachOperator::StartFn {
                id,
                data:
                    FnData {
                        num_params: params,
                        num_returns,
                        control_depth,
                        probes,
                        ..
                    },
            } => {
                state.local_count = *params;
                state.num_returns = *num_returns;
                state.control_depth = *control_depth;
                state.probes = *probes;
                self.pop(ctx, arch, &Reg(1)).map_err(Err::from)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(1),
                        offset: None,
                        disp: 0u32.wrapping_sub(*params as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )
                .map_err(Err::from)?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)
                    .map_err(Err::from)?;
                self.set_label(ctx, arch, X64Label::Func { r#fn: *id })
                    .map_err(Err::from)?;
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    state.local_count += 1;
                    self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                }
            }
            MachOperator::StartBody => {
                state.next_probe_id = 1;
                if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW::new(self, ctx, arch);
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw,
                        cfg.table_base_off,
                        0,
                        2,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    )
                    .map_err(Err::from)?;
                }
                self.push(ctx, arch, &Reg(1)).map_err(Err::from)?;
                self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(state.control_depth as u32 * 16),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )
                .map_err(Err::from)?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)
                    .map_err(Err::from)?;
                self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                for _ in 0..state.control_depth {
                    for _ in 0..2 {
                        self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                    }
                }
            }
            MachOperator::Instruction { op, .. } => self
                ._handle_op(ctx, arch, state, func_imports, sigs, tags, op, target)
                .map_err(Err::from)?,
            MachOperator::Operator { op, annot } => match match op.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            } {
                op => self
                    ._handle_op(
                        ctx,
                        arch,
                        state,
                        func_imports,
                        sigs,
                        tags,
                        &rewriter.instruction(op.clone())?,
                        target,
                    )
                    .map_err(Err::from)?,
            },
            // EndBody and any future meta-ops are no-ops at the assembly level.
            _ => {} // non-instruction MachOperators (meta-ops)
        }
        Ok(())
    }
    /// Apply the [`MemBase::WasmMemSymbol`] address transform to a load/store
    /// effective address already held in `addr` (i.e. `wasm_addr + memarg.offset`):
    /// wrap it to 32 bits and add the `__wasm_mem_N` base pointer, leaving the final
    /// host address in `addr`. `scratch` is clobbered. No-op for [`MemBase::Raw`].
    fn apply_mem_base(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
        addr: Reg,
        scratch: Reg,
        memory_index: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // Zero-base / Raw: guest address is already the host pointer.
        if state.mem_base_for(memory_index).is_zero() {
            return Ok(());
        }
        // addr := (uint32_t)addr — writing the 32-bit subregister zero-extends.
        let addr32 = MemArgKind::NoMem(ArgKind::Reg {
            reg: addr,
            size: MemorySize::_32,
        });
        self.mov(ctx, arch, &addr32, &addr32)?;
        // scratch := __wasm_mem_N (load the base pointer value).
        let sym = if memory_index == 0 {
            "__wasm_mem".into()
        } else {
            alloc::format!("__wasm_mem_{memory_index}")
        };
        self.lea_label(ctx, arch, &scratch, X64Label::External { name: sym })?;
        self.mov(
            ctx,
            arch,
            &scratch,
            &MemArgKind::Mem {
                base: scratch,
                offset: None,
                disp: 0,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
        )?;
        // addr := addr + scratch.
        self.lea(
            ctx,
            arch,
            &addr,
            &MemArgKind::Mem {
                base: addr,
                offset: Some((scratch, 1)),
                disp: 0,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
        )?;
        Ok(())
    }

    /// Compute the effective address `[popped_addr + offset]` into RAX, applying
    /// the memory base. Leaves the address in RAX (Reg(0)); clobbers Reg(1).
    /// When `offset == 0`, skips the `mov64`+`lea` materialization.
    fn mem_effective_addr(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
        offset: u64,
        memory_index: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if offset != 0 {
            self.mov64(ctx, arch, &Reg(1), offset)?;
            self.lea(
                ctx,
                arch,
                &Reg(0),
                &MemArgKind::Mem {
                    base: Reg(0),
                    offset: Some((Reg(1), 1)),
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                },
            )?;
        }
        self.apply_mem_base(ctx, arch, state, Reg(0), Reg(1), memory_index)
    }

    /// Width-generic load. `access` is the load width; `signed` sign-extends a
    /// sub-word load; `to64` selects i64 vs i32 result (i32 stays zero-extended
    /// in the high 32 bits — the backend relies on that for full-register compares).
    fn mem_load(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
        offset: u64,
        memory_index: u32,
        access: MemorySize,
        signed: bool,
        to64: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.pop(ctx, arch, &Reg(0))?; // address
        self.mem_effective_addr(ctx, arch, state, offset, memory_index)?;
        let mem = MemArgKind::Mem {
            base: Reg(0),
            offset: None,
            disp: 0,
            size: access,
            reg_class: RegisterClass::Gpr,
            segment: Default::default(),
        };
        match (access, signed) {
            (MemorySize::_64, _) => self.mov(ctx, arch, &Reg(0), &mem)?,
            (MemorySize::_32, false) => {
                // mov eax, dword [addr] — zero-extends to rax.
                let eax = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(0),
                    size: MemorySize::_32,
                });
                self.mov(ctx, arch, &eax, &mem)?;
            }
            (_, false) => self.movzx(ctx, arch, &Reg(0), &mem)?, // byte/word zero-extend
            (_, true) => {
                self.movsx(ctx, arch, &Reg(0), &mem)?; // sign-extend access-width to 64
                if !to64 {
                    // i32 result: keep low 32 (sign-extended within 32), zero upper 32.
                    let eax = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &eax, &eax)?;
                }
            }
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Width-generic store (writes the low `access` bits of the value).
    fn mem_store(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
        offset: u64,
        memory_index: u32,
        access: MemorySize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.pop(ctx, arch, &Reg(2))?; // value → RDX
        self.pop(ctx, arch, &Reg(0))?; // addr → RAX
        self.mem_effective_addr(ctx, arch, state, offset, memory_index)?;
        let mem = MemArgKind::Mem {
            base: Reg(0),
            offset: None,
            disp: 0,
            size: access,
            reg_class: RegisterClass::Gpr,
            segment: Default::default(),
        };
        let val = MemArgKind::NoMem(ArgKind::Reg {
            reg: Reg(2),
            size: access,
        });
        self.mov(ctx, arch, &mem, &val)
    }

    // ---- floating-point helpers (bit-threading via GP<->XMM moves) ----
    // FP values ride as raw bits on the GP operand stack. RAX=Reg(0)/RCX=Reg(1)/
    // RDX=Reg(2) are GP scratch; XMM0=Reg(0)/XMM1=Reg(1) are the FP scratch (a
    // distinct register file, so the shared Reg index does not collide). The FP
    // emitter methods pick the file (xmm vs gpr) per operand.

    /// Pop two F64 operands, `op(XMM0, XMM1)` (x86 2-operand: `XMM0 op= XMM1`),
    /// push the XMM0 result bits. `f32` selects the single-precision moves.
    fn fp_binop<F>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        f32: bool,
        f: F,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, X64Arch, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.pop(ctx, arch, &Reg(1))?; // rhs bits → RCX
        self.pop(ctx, arch, &Reg(0))?; // lhs bits → RAX
        if f32 {
            self.fmov_gp_to_s(ctx, arch, &Reg(0), &Reg(0))?; // XMM0 = lhs
            self.fmov_gp_to_s(ctx, arch, &Reg(1), &Reg(1))?; // XMM1 = rhs
            f(self, ctx, arch, Reg(0), Reg(1))?;
            self.fmov_s_to_gp(ctx, arch, &Reg(0), &Reg(0))?; // RAX = XMM0
        } else {
            self.fmov_gp_to_d(ctx, arch, &Reg(0), &Reg(0))?;
            self.fmov_gp_to_d(ctx, arch, &Reg(1), &Reg(1))?;
            f(self, ctx, arch, Reg(0), Reg(1))?;
            self.fmov_d_to_gp(ctx, arch, &Reg(0), &Reg(0))?;
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Pop one FP operand, `op(XMM0, XMM1)` (`XMM0 = f(XMM1)`), push XMM0 bits.
    fn fp_unop<F>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        f32: bool,
        f: F,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, X64Arch, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.pop(ctx, arch, &Reg(0))?; // operand bits → RAX
        if f32 {
            self.fmov_gp_to_s(ctx, arch, &Reg(1), &Reg(0))?; // XMM1 = operand
            f(self, ctx, arch, Reg(0), Reg(1))?; // XMM0 = f(XMM1)
            self.fmov_s_to_gp(ctx, arch, &Reg(0), &Reg(0))?;
        } else {
            self.fmov_gp_to_d(ctx, arch, &Reg(1), &Reg(0))?;
            f(self, ctx, arch, Reg(0), Reg(1))?;
            self.fmov_d_to_gp(ctx, arch, &Reg(0), &Reg(0))?;
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Load the two FP comparison operands into XMM0 (=a, lower) and XMM1 (=b,
    /// top), as bits. Leaves RAX/RCX free for the boolean materialization.
    fn fp_cmp_load(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        f32: bool,
    ) -> Result<(), Self::Error> {
        self.pop(ctx, arch, &Reg(0))?; // b bits → RAX
        self.pop(ctx, arch, &Reg(1))?; // a bits → RCX
        if f32 {
            self.fmov_gp_to_s(ctx, arch, &Reg(0), &Reg(1))?; // XMM0 = a
            self.fmov_gp_to_s(ctx, arch, &Reg(1), &Reg(0))?; // XMM1 = b
        } else {
            self.fmov_gp_to_d(ctx, arch, &Reg(0), &Reg(1))?;
            self.fmov_gp_to_d(ctx, arch, &Reg(1), &Reg(0))?;
        }
        Ok(())
    }

    /// `a OP b` for the ordered relational compares (lt/le/gt/ge): false on NaN.
    /// `swap` exchanges the `ucomisd` operands so `lt`/`le` reuse the
    /// above/above-or-equal conditions; `cc` is `A` (strict) or `NB` (or-equal).
    fn fp_cmp_rel(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        f32: bool,
        swap: bool,
        cc: ConditionCode,
    ) -> Result<(), Self::Error> {
        self.fp_cmp_load(ctx, arch, f32)?;
        if swap {
            if f32 {
                self.fcmp_s(ctx, arch, &Reg(1), &Reg(0))?;
            } else {
                self.fcmp(ctx, arch, &Reg(1), &Reg(0))?;
            }
        } else if f32 {
            self.fcmp_s(ctx, arch, &Reg(0), &Reg(1))?;
        } else {
            self.fcmp(ctx, arch, &Reg(0), &Reg(1))?;
        }
        // mov does not disturb the ucomisd flags.
        self.mov64(ctx, arch, &Reg(0), 0)?;
        self.mov64(ctx, arch, &Reg(2), 1)?;
        self.cmovcc(ctx, arch, cc, &Reg(0), &Reg(2))?;
        self.push(ctx, arch, &Reg(0))
    }

    /// `a == b` / `a != b` with WASM NaN semantics (eq false on NaN; ne true on
    /// NaN). `ucomisd` sets ZF for equal-OR-unordered and PF for unordered, so
    /// the parity flag distinguishes the NaN case.
    fn fp_cmp_eq(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        f32: bool,
        want_eq: bool,
    ) -> Result<(), Self::Error> {
        self.fp_cmp_load(ctx, arch, f32)?;
        if f32 {
            self.fcmp_s(ctx, arch, &Reg(0), &Reg(1))?;
        } else {
            self.fcmp(ctx, arch, &Reg(0), &Reg(1))?;
        }
        if want_eq {
            self.mov64(ctx, arch, &Reg(0), 0)?;
            self.mov64(ctx, arch, &Reg(2), 1)?;
            self.cmovcc(ctx, arch, ConditionCode::E, &Reg(0), &Reg(2))?; // ZF? 1:0
            self.mov64(ctx, arch, &Reg(2), 0)?;
            self.cmovcc(ctx, arch, ConditionCode::P, &Reg(0), &Reg(2))?; // unordered → 0
        } else {
            self.mov64(ctx, arch, &Reg(0), 1)?;
            self.mov64(ctx, arch, &Reg(2), 0)?;
            self.cmovcc(ctx, arch, ConditionCode::E, &Reg(0), &Reg(2))?; // ZF? 0:1
            self.mov64(ctx, arch, &Reg(2), 1)?;
            self.cmovcc(ctx, arch, ConditionCode::P, &Reg(0), &Reg(2))?; // unordered → 1
        }
        self.push(ctx, arch, &Reg(0))
    }

    /// Pop one value into RAX, run `f` (producing the result in RAX via XMM0/XMM1
    /// scratch), and push RAX. Used for all int<->fp and f32<->f64 conversions.
    fn fp_convert<F>(&mut self, ctx: &mut Context, arch: X64Arch, f: F) -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, X64Arch) -> Result<(), Self::Error>,
    {
        self.pop(ctx, arch, &Reg(0))?;
        f(self, ctx, arch)?;
        self.push(ctx, arch, &Reg(0))
    }

    fn _handle_op(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if target != state.body {
            // On the very first instruction `state.body == 0` is the Default
            // value rather than a real prior body.  Emitting a skip jump
            // here references a `_idx_0` label that is never set, leaving
            // a dangling relocation (jump-to-self on AArch64, no-op-fallthrough
            // on x86 — but no useful skip semantics either way).
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jmp_label(
                    ctx,
                    arch,
                    X64Label::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, X64Label::Indexed { idx })?;
                }
            }
        }
        match op {
            Instruction::I32Const(value) => {
                self.mov64(ctx, arch, &Reg(0), *value as u32 as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64Const(value) => {
                self.mov64(ctx, arch, &Reg(0), *value as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F32Const(value) => {
                self.mov64(ctx, arch, &Reg(0), value.bits() as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F64Const(value) => {
                self.mov64(ctx, arch, &Reg(0), value.bits())?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64ReinterpretF64
            | Instruction::F64ReinterpretI64
            | Instruction::I32ReinterpretF32
            | Instruction::F32ReinterpretI32 => {}
            Instruction::I32Add | Instruction::I64Add => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                if let Instruction::I32Add = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Sub | Instruction::I64Sub => {
                self.pop(ctx, arch, &Reg(0))?; // Reg(0) = b (subtrahend, top of stack)
                self.pop(ctx, arch, &Reg(1))?; // Reg(1) = a (minuend)
                self.not(ctx, arch, &Reg(0))?; // ~b; result = ~b + a + 1 = a - b
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                if let Instruction::I32Sub = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Mul | Instruction::I64Mul => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.mul(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Mul = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivU | Instruction::I64DivU => {
                // WASM stack: [dividend, divisor]. Pop divisor→RCX, dividend→RAX.
                // x86-64 div: rdx:rax / rcx → quotient in rax, remainder in rdx.
                self.pop(ctx, arch, &Reg(1))?; // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?; // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX (high half of dividend)
                self.div(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivU = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivS | Instruction::I64DivS => {
                self.pop(ctx, arch, &Reg(1))?; // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?; // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX (sign-extend not available yet)
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivS = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32RemU | Instruction::I64RemU => {
                self.pop(ctx, arch, &Reg(1))?; // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?; // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX
                self.div(ctx, arch, &Reg(0), &Reg(1))?; // remainder → RDX
                if let Instruction::I32RemU = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(2),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(2))?; // push RDX (remainder)
            }
            Instruction::I32RemS | Instruction::I64RemS => {
                self.pop(ctx, arch, &Reg(1))?; // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?; // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?; // remainder → RDX
                if let Instruction::I32RemS = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(2),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(2))?; // push RDX (remainder)
            }
            Instruction::I32And | Instruction::I64And => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.and(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32And = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Or | Instruction::I64Or => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.or(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Or = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Xor | Instruction::I64Xor => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Xor = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Shl | Instruction::I64Shl => {
                // x86-64 shl count must be in CL (low byte of RCX).
                // Pass count as 8-bit so text-asm emits "cl" and IcedWriter uses CL.
                let cl = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(1),
                    size: MemorySize::_8,
                });
                self.pop(ctx, arch, &Reg(1))?; // shift count → RCX
                self.pop(ctx, arch, &Reg(0))?; // value → RAX
                self.shl(ctx, arch, &Reg(0), &cl)?;
                if let Instruction::I32Shl = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32ShrU | Instruction::I64ShrU => {
                let cl = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(1),
                    size: MemorySize::_8,
                });
                self.pop(ctx, arch, &Reg(1))?; // shift count → RCX
                self.pop(ctx, arch, &Reg(0))?; // value → RAX
                self.shr(ctx, arch, &Reg(0), &cl)?;
                if let Instruction::I32ShrU = op {
                    {
                        let r = MemArgKind::NoMem(ArgKind::Reg {
                            reg: Reg(0),
                            size: MemorySize::_32,
                        });
                        self.mov(ctx, arch, &r, &r)?;
                    }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32WrapI64 => {
                self.pop(ctx, arch, &Reg(0))?;
                {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &r, &r)?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            // Sign-extend the low 32 bits to 64 (MOVSXD).
            // Both sign-extend the low 32 bits to 64.
            Instruction::I64ExtendI32S | Instruction::I64Extend32S => {
                self.pop(ctx, arch, &Reg(0))?;
                let dst = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(0),
                    size: MemorySize::_64,
                });
                let src = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(0),
                    size: MemorySize::_32,
                });
                self.movsx(ctx, arch, &dst, &src)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            // Zero-extend the low 32 bits to 64 (writing a 32-bit reg zero-extends).
            Instruction::I64ExtendI32U => {
                self.pop(ctx, arch, &Reg(0))?;
                {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &r, &r)?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Eqz | Instruction::I64Eqz => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                // CMOV has no immediate form: materialize 1 in a scratch reg.
                // (`mov` does not disturb the flags set by `cmp0`.)
                self.mov64(ctx, arch, &Reg(2), 1)?;
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &Reg(2))?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I32Eq | Instruction::I64Eq => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.not(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov64(ctx, arch, &Reg(1), 0)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(2), 1)?;
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &Reg(2))?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I32Ne | Instruction::I64Ne => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.not(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov64(ctx, arch, &Reg(1), 1)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(2), 0)?;
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &Reg(2))?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I64Load(memarg) | Instruction::F64Load(memarg) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mem_effective_addr(ctx, arch, state, memarg.offset, memarg.memory_index)?;
                // Dereference: load 64-bit value from [rax] into rax.
                self.mov(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64Store(memarg) | Instruction::F64Store(memarg) => {
                self.pop(ctx, arch, &Reg(2))?; // value → RDX
                self.pop(ctx, arch, &Reg(0))?; // addr → RAX
                self.mem_effective_addr(ctx, arch, state, memarg.offset, memarg.memory_index)?;
                // Store 64-bit value from RDX to [RAX].
                self.mov(
                    ctx,
                    arch,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                    &Reg(2),
                )?;
            }
            Instruction::I32Load(memarg) | Instruction::F32Load(memarg) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mem_effective_addr(ctx, arch, state, memarg.offset, memarg.memory_index)?;
                // Dereference: load 32-bit value into eax (zero-extends to rax).
                // Previously this was `mov rax, rax`, a register-to-register
                // no-op that left the *address* in rax instead of loading the
                // value at that address.
                let eax = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(0),
                    size: MemorySize::_32,
                });
                self.mov(
                    ctx,
                    arch,
                    &eax,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_32,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
            }
            // i32.store and i64.store32 both write the low 32 bits to memory.
            Instruction::I32Store(memarg)
            | Instruction::I64Store32(memarg)
            | Instruction::F32Store(memarg) => {
                self.pop(ctx, arch, &Reg(2))?; // value → RDX
                self.pop(ctx, arch, &Reg(0))?; // addr → RAX
                self.mem_effective_addr(ctx, arch, state, memarg.offset, memarg.memory_index)?;
                // Store 32-bit value from EDX to [RAX].
                let edx = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(2),
                    size: MemorySize::_32,
                });
                self.mov(
                    ctx,
                    arch,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_32,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                    &edx,
                )?;
            }

            // ---- sub-word loads (zero/sign-extended) ----
            Instruction::I32Load8U(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_8,
                false,
                false,
            )?,
            Instruction::I32Load8S(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_8,
                true,
                false,
            )?,
            Instruction::I32Load16U(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_16,
                false,
                false,
            )?,
            Instruction::I32Load16S(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_16,
                true,
                false,
            )?,
            Instruction::I64Load8U(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_8,
                false,
                true,
            )?,
            Instruction::I64Load8S(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_8,
                true,
                true,
            )?,
            Instruction::I64Load16U(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_16,
                false,
                true,
            )?,
            Instruction::I64Load16S(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_16,
                true,
                true,
            )?,
            Instruction::I64Load32U(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_32,
                false,
                true,
            )?,
            Instruction::I64Load32S(m) => self.mem_load(
                ctx,
                arch,
                state,
                m.offset,
                m.memory_index,
                MemorySize::_32,
                true,
                true,
            )?,

            // ---- sub-word stores ----
            Instruction::I32Store8(m) => {
                self.mem_store(ctx, arch, state, m.offset, m.memory_index, MemorySize::_8)?
            }
            Instruction::I32Store16(m) => {
                self.mem_store(ctx, arch, state, m.offset, m.memory_index, MemorySize::_16)?
            }
            Instruction::I64Store8(m) => {
                self.mem_store(ctx, arch, state, m.offset, m.memory_index, MemorySize::_8)?
            }
            Instruction::I64Store16(m) => {
                self.mem_store(ctx, arch, state, m.offset, m.memory_index, MemorySize::_16)?
            }

            // ---- arithmetic shift right (sar) ----
            Instruction::I32ShrS | Instruction::I64ShrS => {
                let cl = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(1),
                    size: MemorySize::_8,
                });
                self.pop(ctx, arch, &Reg(1))?; // shift count → RCX
                self.pop(ctx, arch, &Reg(0))?; // value → RAX
                if let Instruction::I32ShrS = op {
                    // 32-bit sar replicates bit 31 and zero-extends the result.
                    let eax = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.sar(ctx, arch, &eax, &cl)?;
                } else {
                    self.sar(ctx, arch, &Reg(0), &cl)?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }

            // ---- select: c ? a : b (pop c, b, a) ----
            Instruction::Select | Instruction::TypedSelect { .. } => {
                self.pop(ctx, arch, &Reg(1))?; // condition → RCX
                self.pop(ctx, arch, &Reg(2))?; // b (false value) → RDX
                self.pop(ctx, arch, &Reg(0))?; // a (true value) → RAX
                self.cmp0(ctx, arch, &Reg(1))?;
                // if condition == 0, take b.
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(0), &Reg(2))?;
                self.push(ctx, arch, &Reg(0))?;
            }

            Instruction::LocalGet(local_index) => {
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.pop(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::LocalTee(local_index) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::LocalSet(local_index) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
            }
            Instruction::Return => {
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    // &Reg(0),
                    // (state.local_count + 3) as isize * 8,
                    // None,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0u32.wrapping_sub(8),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov(ctx, arch, &RSP, &Reg(0))?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                for a in 0..state.num_returns {
                    self.mov(ctx, arch, &Reg(2), &Reg(1))?;
                    self.push(ctx, arch, &Reg(2))?;
                }
                self.push(ctx, arch, &Reg(0))?;
                self.ret(ctx, arch)?;
            }
            Instruction::Br(relative_depth) => {
                self.br(ctx, arch, state, *relative_depth)?;
            }
            Instruction::BrIf(relative_depth) => {
                let i = state.label_index;
                state.label_index += 1;
                self.pop(ctx, arch, &Reg(0))?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: i })?;
                self.br(ctx, arch, state, *relative_depth)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
            }
            Instruction::BrTable(targets, default) => {
                for relative_depth in targets.iter().cloned() {
                    let i = state.label_index;
                    state.label_index += 1;
                    self.pop(ctx, arch, &Reg(0))?;
                    self.cmp0(ctx, arch, &Reg(0))?;
                    self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: i })?;
                    self.br(ctx, arch, state, relative_depth)?;
                    self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                    self.lea(
                        ctx,
                        arch,
                        &Reg(0),
                        &MemArgKind::Mem {
                            base: Reg(0),
                            offset: None,
                            disp: 0xffff_ffff,
                            size: MemorySize::_64,
                            reg_class: RegisterClass::Gpr,
                            segment: Default::default(),
                        },
                    )?;
                    self.push(ctx, arch, &Reg(0))?;
                }
                self.pop(ctx, arch, &Reg(0))?;
                self.br(ctx, arch, state, *default)?;
            }
            Instruction::Block(blockty) => {
                state.if_stack.push(Endable::Br);
                let i = state.label_index;
                state.label_index += 1;
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                // for _ in &Reg(0)..=(*relative_depth) {
                self.push(ctx, arch, &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
                // }
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::If(blockty) => {
                let i = state.label_index;
                state.label_index += 3;
                state.if_stack.push(Endable::If { idx: i });
                self.pop(ctx, arch, &Reg(2))?;
                self.cmp0(ctx, arch, &Reg(2))?;
                self.jcc_label(
                    ctx,
                    arch,
                    ConditionCode::E,
                    X64Label::Indexed { idx: i + 1 },
                )?;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
            }
            Instruction::Else => {
                let Endable::If { idx: i } = state.if_stack.last().unwrap() else {
                    todo!()
                };
                let i = *i;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i + 2 })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })?;
            }
            Instruction::Loop(blockty) => {
                state.if_stack.push(Endable::Br);
                let i = state.label_index;
                state.label_index += 1;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                // for _ in &Reg(0)..=(*relative_depth) {
                self.push(ctx, arch, &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
                // }
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::End => {
                // Function-level End (if_stack empty) is a no-op: the function
                // return path already cleaned up the frame.
                if let Some(top) = state.if_stack.pop() {
                    self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                    match top {
                        Endable::Br => {
                            self.pop(ctx, arch, &Reg(0))?;
                            self.pop(ctx, arch, &Reg(1))?;
                        }
                        Endable::If { idx: i } => {
                            self.set_label(ctx, arch, X64Label::Indexed { idx: i + 2 })?;
                        }
                        Endable::TryTable {
                            exit_idx: _,
                            dispatch_idx,
                            after_dispatch_idx,
                            catches,
                        } => {
                            // Normal fall-through: pop CTX frame (same as Block).
                            self.pop(ctx, arch, &Reg(0))?; // exit_label
                            self.pop(ctx, arch, &Reg(1))?; // old_RSP
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            // Software EH stack: normal (non-throwing) exit —
                            // discard our frame. Local `Throw` and
                            // `__wasm_exn_propagate` each pop their own
                            // frame right before jumping into a dispatch
                            // stub, so this is the only path that needs to
                            // pop here.
                            self.lea_label(
                                ctx,
                                arch,
                                &Reg(0),
                                X64Label::External {
                                    name: "__wasm_eh_pop".into(),
                                },
                            )?;
                            self.call(ctx, arch, &Reg(0))?;
                            // Jump over dispatch stub.
                            self.jmp_label(
                                ctx,
                                arch,
                                X64Label::Indexed {
                                    idx: after_dispatch_idx,
                                },
                            )?;

                            // Dispatch stub: entered when throw jumps here.
                            self.set_label(ctx, arch, X64Label::Indexed { idx: dispatch_idx })?;
                            // Restore operand stack to TryTable entry RSP.
                            // The CTX frame still has old_RSP; re-read from CTX stack.
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            self.pop(ctx, arch, &Reg(0))?; // exit_label (discard)
                            self.pop(ctx, arch, &Reg(1))?; // old_RSP
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            self.mov(ctx, arch, &RSP, &Reg(1))?; // restore operand RSP
                            // r2 = thrown tag index (set by Throw instruction).
                            for catch in catches.iter() {
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else {
                                            0
                                        };
                                        let skip_lbl = state.label_index;
                                        state.label_index += 1;
                                        self.mov64(ctx, arch, &Reg(0), *tag as u64)?;
                                        self.cmp(ctx, arch, &Reg(2), &Reg(0))?;
                                        self.jcc_label(
                                            ctx,
                                            arch,
                                            ConditionCode::NE,
                                            X64Label::Indexed { idx: skip_lbl },
                                        )?;
                                        // Tag matched: push exception values (r3..r(2+arity))
                                        for i in (0..arity).rev() {
                                            self.push(ctx, arch, &Reg(3 + i as u8))?;
                                        }
                                        self.br(ctx, arch, state, *label)?;
                                        self.set_label(
                                            ctx,
                                            arch,
                                            X64Label::Indexed { idx: skip_lbl },
                                        )?;
                                    }
                                    Catch::All { label } => {
                                        self.br(ctx, arch, state, *label)?;
                                    }
                                    Catch::OneRef { .. } | Catch::AllRef { .. } => {
                                        // exnref deferred — fall through to unhandled
                                    }
                                }
                            }
                            // No catch matched: propagate via CTX chain.
                            self.lea_label(
                                ctx,
                                arch,
                                &Reg(0),
                                X64Label::External {
                                    name: alloc::format!("__wasm_exn_propagate"),
                                },
                            )?;
                            self.jmp(ctx, arch, &Reg(0))?;
                            self.set_label(
                                ctx,
                                arch,
                                X64Label::Indexed {
                                    idx: after_dispatch_idx,
                                },
                            )?;
                            return Ok(());
                        }
                    }
                    self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                }
            }
            // ---- exception handling ------------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else {
                    0
                };
                // Static dispatch: does the innermost TryTable's if_stack
                // entry (a purely compile-time, same-function lookup) have a
                // dispatch stub for us? Resolved before marshalling
                // tag/values below since `__wasm_eh_pop` is a real SysV
                // call and would otherwise clobber caller-saved RDX/RBX etc.
                let local_dispatch = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_idx, .. } => Some(*dispatch_idx),
                    _ => None,
                });
                if local_dispatch.is_some() {
                    // Software EH stack: this try_table will handle the
                    // throw locally (bypassing __wasm_exn_propagate
                    // entirely), so pop its frame ourselves — propagate
                    // only pops when *it* dispatches (see TryTable::End).
                    self.lea_label(
                        ctx,
                        arch,
                        &Reg(0),
                        X64Label::External {
                            name: "__wasm_eh_pop".into(),
                        },
                    )?;
                    self.call(ctx, arch, &Reg(0))?;
                }
                // Tag index → r2; exception values → r3, r4, r5 (up to arity 3).
                self.mov64(ctx, arch, &Reg(2), *tag_index as u64)?;
                for i in 0..arity.min(3) {
                    self.pop(ctx, arch, &Reg(3 + i as u8))?;
                }
                if let Some(dispatch_idx) = local_dispatch {
                    self.jmp_label(ctx, arch, X64Label::Indexed { idx: dispatch_idx })?;
                } else {
                    // No intra-function handler: propagate via the software
                    // EH stack (see `speet_rt::generate_exn_tu`).
                    self.lea_label(
                        ctx,
                        arch,
                        &Reg(0),
                        X64Label::External {
                            name: alloc::format!("__wasm_exn_propagate"),
                        },
                    )?;
                    self.jmp(ctx, arch, &Reg(0))?;
                }
            }
            Instruction::ThrowRef => todo!("exnref deferred"),
            // ---- TryTable block ---------------------------------------------
            Instruction::TryTable(blockty, catches) => {
                let exit_idx = state.label_index;
                let dispatch_idx = state.label_index + 1;
                let after_dispatch_idx = state.label_index + 2;
                state.label_index += 3;
                state.if_stack.push(Endable::TryTable {
                    exit_idx,
                    dispatch_idx,
                    after_dispatch_idx,
                    catches: catches
                        .iter()
                        .cloned()
                        .collect::<alloc::vec::Vec<_>>()
                        .into_boxed_slice(),
                });
                // Push CTX frame (same as Block): old_RSP + exit_label.
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: exit_idx })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(1))?; // old_RSP
                self.push(ctx, arch, &Reg(0))?; // exit_label
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: exit_idx })?;
                // Software EH stack: __wasm_eh_push(dispatch_addr, current
                // operand-stack RSP) — a real SysV call (like `MemoryInit`),
                // marshalling args into RDI/RSI explicitly. This is what
                // `__wasm_exn_propagate` walks on a cross-function throw,
                // instead of a CTX chain — see `speet_rt::generate_exn_tu`'s
                // doc for why. The CTX frame above still handles the
                // same-function fast path unchanged.
                self.lea_label(ctx, arch, &Reg(7), X64Label::Indexed { idx: dispatch_idx })?; // RDI = dispatch addr
                self.mov(ctx, arch, &Reg(6), &RSP)?; // RSI = saved operand-stack RSP
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_eh_push".into(),
                    },
                )?;
                self.call(ctx, arch, &Reg(0))?;
            }
            Instruction::Call(function_index) => {
                let fn_idx = *function_index;
                // Classify the call when sharding is active.
                let target = state.shard.as_ref().map(|s| s.call_target(fn_idx));
                match target {
                    Some(CallTarget::CrossShard { table_slot }) => {
                        // Cross-shard: load fn ptr from [SCR + table_slot * 8].
                        self.mov(
                            ctx,
                            arch,
                            &Reg(0),
                            &MemArgKind::Mem {
                                base: SCR,
                                offset: None,
                                disp: table_slot.wrapping_mul(8),
                                size: MemorySize::_64,
                                reg_class: RegisterClass::Gpr,
                                segment: Default::default(),
                            },
                        )?;
                        self.call(ctx, arch, &Reg(0))?;
                    }
                    _ => {
                        // Import or local (or no sharding): existing label-call path.
                        match func_imports.get(fn_idx as usize) {
                            Some((module, name)) => {
                                let sym = alloc::format!("{module}__{name}");
                                self.lea_label(
                                    ctx,
                                    arch,
                                    &Reg(0),
                                    X64Label::External { name: sym },
                                )?;
                                self.call(ctx, arch, &Reg(0))?;
                            }
                            None => {
                                let idx = fn_idx - func_imports.len() as u32;
                                self.lea_label(ctx, arch, &Reg(0), X64Label::Func { r#fn: idx })?;
                                self.call(ctx, arch, &Reg(0))?;
                            }
                        }
                    }
                }
            }
            // ---- memory.size ------------------------------------------------
            // Load __wasm_mem_pages (32-bit global) and push as i64 onto the WASM stack.
            // The concrete writer must resolve X64Label::External symbols.
            Instruction::MemorySize(_) => {
                // Get address of the pages-count global into Reg(0).
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_mem_pages".into(),
                    },
                )?;
                // Load the 32-bit value (zero-extend to 64 bits).
                // Dest must be eax (the 32-bit half) for the assembler to accept
                // `mov eax, dword ptr [rax]` — writing eax auto-zero-extends rax.
                let eax = MemArgKind::NoMem(ArgKind::Reg {
                    reg: Reg(0),
                    size: MemorySize::_32,
                });
                self.mov(
                    ctx,
                    arch,
                    &eax,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_32,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
            }
            // ---- memory.grow ------------------------------------------------
            // delta is on the WASM stack top.  Call __wasm_memory_grow using the
            // same blitz-x86-64 WASM calling convention as regular function calls:
            // the callee pops the hardware return address, accesses delta via its
            // frame pointer, and pushes old_pages before returning.
            Instruction::MemoryGrow(_) => {
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_memory_grow".into(),
                    },
                )?;
                self.call(ctx, arch, &Reg(0))?;
            }
            // ---- memory.init ------------------------------------------------
            // Pop (dest, src_offset, len) off the WASM stack (len on top) and
            // call a plain-C-ABI helper the runtime shim provides:
            // `void __wasm_memory_init_copy(uint32_t dest_off, const void
            // *seg_base, uint32_t src_off, uint32_t len)`. Each data segment's
            // bytes are embedded by the shim as a separate symbol
            // `__wasm_data_seg_{data_index}` — `data_index` is a compile-time
            // instruction operand, so the symbol name (and therefore which
            // segment gets copied) is resolved at this call site, not at
            // runtime. This is a real SysV C call (unlike `MemoryGrow` above,
            // which reuses this backend's WASM-internal calling convention) —
            // marshal args into RDI/RSI/RDX/RCX explicitly rather than
            // relying on whatever's already on the native stack.
            Instruction::MemoryInit { data_index, .. } => {
                self.pop(ctx, arch, &Reg(1))?; // RCX ← len (4th SysV arg)
                self.pop(ctx, arch, &Reg(2))?; // RDX ← src_offset (3rd SysV arg)
                self.pop(ctx, arch, &Reg(7))?; // RDI ← dest_offset (1st SysV arg)
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(6),
                    X64Label::External {
                        name: alloc::format!("__wasm_data_seg_{data_index}"),
                    },
                )?; // RSI ← segment base (2nd SysV arg)
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_memory_init_copy".into(),
                    },
                )?;
                self.call(ctx, arch, &Reg(0))?;
            }
            // memory.copy: (dest, src, len) → `__wasm_memory_copy` (memmove).
            Instruction::MemoryCopy { .. } => {
                self.pop(ctx, arch, &Reg(2))?; // RDX ← len
                self.pop(ctx, arch, &Reg(6))?; // RSI ← src_offset
                self.pop(ctx, arch, &Reg(7))?; // RDI ← dest_offset
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_memory_copy".into(),
                    },
                )?;
                self.call(ctx, arch, &Reg(0))?;
            }
            // memory.fill: (dest, val, len) → `__wasm_memory_fill` (memset).
            Instruction::MemoryFill(_) => {
                self.pop(ctx, arch, &Reg(2))?; // RDX ← len
                self.pop(ctx, arch, &Reg(6))?; // RSI ← val
                self.pop(ctx, arch, &Reg(7))?; // RDI ← dest_offset
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_memory_fill".into(),
                    },
                )?;
                self.call(ctx, arch, &Reg(0))?;
            }
            // `data.drop` is a compile-time no-op here: this AOT backend
            // never re-runs a `memory.init` for the same `data_index` after a
            // `data.drop` (each generated data-init function initializes
            // every segment exactly once, by construction — see
            // `speet-recompile`'s data-init function generator), so there is
            // no runtime segment-liveness state to actually drop.
            Instruction::DataDrop(_) => {}
            // `unreachable` traps. Emit HLT — privileged in user mode, so it
            // faults deterministically rather than executing past the trap.
            Instruction::Unreachable => {
                self.hlt(ctx, arch)?;
            }
            // True-tail `return_call`: tear down the NaiveAbi CTX frame then
            // `jmp` to the callee (never `call`+`ret`, which nests a frame per
            // hop — speet emits long `return_call` chains that must stay O(1)).
            // Arg marshalling matches `Call` above (none on the NaiveAbi path;
            // AllStack marshalling lives in `sysv.rs`).
            Instruction::ReturnCall(function_index) => {
                let fn_idx = *function_index;
                // Resolve callee into Reg(3) (Return teardown clobbers 0/1/2).
                let call_target = state.shard.as_ref().map(|s| s.call_target(fn_idx));
                match call_target {
                    Some(CallTarget::CrossShard { table_slot }) => {
                        self.mov(
                            ctx,
                            arch,
                            &Reg(3),
                            &MemArgKind::Mem {
                                base: SCR,
                                offset: None,
                                disp: table_slot.wrapping_mul(8),
                                size: MemorySize::_64,
                                reg_class: RegisterClass::Gpr,
                                segment: Default::default(),
                            },
                        )?;
                    }
                    _ => match func_imports.get(fn_idx as usize) {
                        Some((module, name)) => {
                            let sym = alloc::format!("{module}__{name}");
                            self.lea_label(ctx, arch, &Reg(3), X64Label::External { name: sym })?;
                        }
                        None => {
                            let idx = fn_idx - func_imports.len() as u32;
                            self.lea_label(ctx, arch, &Reg(3), X64Label::Func { r#fn: idx })?;
                        }
                    },
                }
                // Frame teardown (same shape as `Return`, without result shuffle).
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0u32.wrapping_sub(8),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov(ctx, arch, &RSP, &Reg(0))?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                // Reg(0) holds the original return address; leave it for the
                // callee's `ret`, then jump (no link).
                self.push(ctx, arch, &Reg(0))?;
                self.jmp(ctx, arch, &Reg(3))?;
            }
            // ---- F64 arithmetic (XMM0 op= XMM1) ----
            Instruction::F64Add => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fadd(c, a, &d, &s))?
            }
            Instruction::F64Sub => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fsub(c, a, &d, &s))?
            }
            Instruction::F64Mul => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fmul(c, a, &d, &s))?
            }
            Instruction::F64Div => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fdiv(c, a, &d, &s))?
            }
            Instruction::F64Min => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fmin(c, a, &d, &s))?
            }
            Instruction::F64Max => {
                self.fp_binop(ctx, arch, false, |w, c, a, d, s| w.fmax(c, a, &d, &s))?
            }
            Instruction::F64Sqrt => {
                self.fp_unop(ctx, arch, false, |w, c, a, d, s| w.fsqrt(c, a, &d, &s))?
            }
            // ---- F32 arithmetic ----
            Instruction::F32Add => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fadd_s(c, a, &d, &s))?
            }
            Instruction::F32Sub => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fsub_s(c, a, &d, &s))?
            }
            Instruction::F32Mul => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fmul_s(c, a, &d, &s))?
            }
            Instruction::F32Div => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fdiv_s(c, a, &d, &s))?
            }
            Instruction::F32Min => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fmin_s(c, a, &d, &s))?
            }
            Instruction::F32Max => {
                self.fp_binop(ctx, arch, true, |w, c, a, d, s| w.fmax_s(c, a, &d, &s))?
            }
            Instruction::F32Sqrt => {
                self.fp_unop(ctx, arch, true, |w, c, a, d, s| w.fsqrt_s(c, a, &d, &s))?
            }

            // ---- FP abs/neg via GP bit masking (exact, no SSE constant pool) ----
            Instruction::F64Abs => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0x7fff_ffff_ffff_ffff)?;
                self.and(ctx, arch, &Reg(0), &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F64Neg => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0x8000_0000_0000_0000)?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F32Abs => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0x7fff_ffff)?;
                self.and(ctx, arch, &Reg(0), &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F32Neg => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0x8000_0000)?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
            }

            // ---- copysign: GP bit-mask on the sign bit (bits live in RAX) ----
            Instruction::F64Copysign => {
                self.pop(ctx, arch, &Reg(1))?; // y (sign source)
                self.pop(ctx, arch, &Reg(0))?; // x
                self.mov64(ctx, arch, &Reg(2), 0x7fff_ffff_ffff_ffff)?;
                self.and(ctx, arch, &Reg(0), &Reg(2))?; // x & ~sign
                self.mov64(ctx, arch, &Reg(2), 0x8000_0000_0000_0000)?;
                self.and(ctx, arch, &Reg(1), &Reg(2))?; // y & sign
                self.eor(ctx, arch, &Reg(0), &Reg(1))?; // combine
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F32Copysign => {
                self.pop(ctx, arch, &Reg(1))?; // y
                self.pop(ctx, arch, &Reg(0))?; // x
                self.mov64(ctx, arch, &Reg(2), 0x7fff_ffff)?;
                self.and(ctx, arch, &Reg(0), &Reg(2))?;
                self.mov64(ctx, arch, &Reg(2), 0x8000_0000)?;
                self.and(ctx, arch, &Reg(1), &Reg(2))?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
            }

            // ---- FP compares (false on NaN except ne) ----
            Instruction::F64Eq => self.fp_cmp_eq(ctx, arch, false, true)?,
            Instruction::F64Ne => self.fp_cmp_eq(ctx, arch, false, false)?,
            Instruction::F64Gt => self.fp_cmp_rel(ctx, arch, false, false, ConditionCode::A)?,
            Instruction::F64Ge => self.fp_cmp_rel(ctx, arch, false, false, ConditionCode::NB)?,
            Instruction::F64Lt => self.fp_cmp_rel(ctx, arch, false, true, ConditionCode::A)?,
            Instruction::F64Le => self.fp_cmp_rel(ctx, arch, false, true, ConditionCode::NB)?,
            Instruction::F32Eq => self.fp_cmp_eq(ctx, arch, true, true)?,
            Instruction::F32Ne => self.fp_cmp_eq(ctx, arch, true, false)?,
            Instruction::F32Gt => self.fp_cmp_rel(ctx, arch, true, false, ConditionCode::A)?,
            Instruction::F32Ge => self.fp_cmp_rel(ctx, arch, true, false, ConditionCode::NB)?,
            Instruction::F32Lt => self.fp_cmp_rel(ctx, arch, true, true, ConditionCode::A)?,
            Instruction::F32Le => self.fp_cmp_rel(ctx, arch, true, true, ConditionCode::NB)?,

            // ---- conversions: int -> fp (signed; bits already in RAX) ----
            Instruction::F64ConvertI32S => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_d_w(c, a, &Reg(0), &Reg(0))?;
                w.fmov_d_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            Instruction::F64ConvertI64S => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_d_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_d_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            Instruction::F32ConvertI32S => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_s_w(c, a, &Reg(0), &Reg(0))?;
                w.fmov_s_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            Instruction::F32ConvertI64S => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_s_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_s_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            // i32-unsigned: zero-extend to 64 bits, then use the signed r64 convert.
            Instruction::F64ConvertI32U => self.fp_convert(ctx, arch, |w, c, a| {
                w.mov64(c, a, &Reg(1), 0xffff_ffff)?;
                w.and(c, a, &Reg(0), &Reg(1))?;
                w.scvtf_d_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_d_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            Instruction::F32ConvertI32U => self.fp_convert(ctx, arch, |w, c, a| {
                w.mov64(c, a, &Reg(1), 0xffff_ffff)?;
                w.and(c, a, &Reg(0), &Reg(1))?;
                w.scvtf_s_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_s_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            // i64-unsigned: use the signed convert (exact for values < 2^63).
            Instruction::F64ConvertI64U => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_d_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_d_to_gp(c, a, &Reg(0), &Reg(0))
            })?,
            Instruction::F32ConvertI64U => self.fp_convert(ctx, arch, |w, c, a| {
                w.scvtf_s_x(c, a, &Reg(0), &Reg(0))?;
                w.fmov_s_to_gp(c, a, &Reg(0), &Reg(0))
            })?,

            // ---- conversions: fp -> int (truncating; unsigned uses the signed
            // trunc, exact for in-range values) ----
            Instruction::I32TruncF64S | Instruction::I32TruncF64U => {
                self.fp_convert(ctx, arch, |w, c, a| {
                    w.fmov_gp_to_d(c, a, &Reg(0), &Reg(0))?;
                    w.fcvtzs_w_d(c, a, &Reg(0), &Reg(0))
                })?
            }
            Instruction::I64TruncF64S | Instruction::I64TruncF64U => {
                self.fp_convert(ctx, arch, |w, c, a| {
                    w.fmov_gp_to_d(c, a, &Reg(0), &Reg(0))?;
                    w.fcvtzs_x_d(c, a, &Reg(0), &Reg(0))
                })?
            }
            Instruction::I32TruncF32S | Instruction::I32TruncF32U => {
                self.fp_convert(ctx, arch, |w, c, a| {
                    w.fmov_gp_to_s(c, a, &Reg(0), &Reg(0))?;
                    w.fcvtzs_w_s(c, a, &Reg(0), &Reg(0))
                })?
            }
            Instruction::I64TruncF32S | Instruction::I64TruncF32U => {
                self.fp_convert(ctx, arch, |w, c, a| {
                    w.fmov_gp_to_s(c, a, &Reg(0), &Reg(0))?;
                    w.fcvtzs_x_s(c, a, &Reg(0), &Reg(0))
                })?
            }

            // ---- conversions: f32 <-> f64 ----
            Instruction::F32DemoteF64 => self.fp_convert(ctx, arch, |w, c, a| {
                w.fmov_gp_to_d(c, a, &Reg(0), &Reg(0))?;
                w.fcvt_s_d(c, a, &Reg(1), &Reg(0))?;
                w.fmov_s_to_gp(c, a, &Reg(0), &Reg(1))
            })?,
            Instruction::F64PromoteF32 => self.fp_convert(ctx, arch, |w, c, a| {
                w.fmov_gp_to_s(c, a, &Reg(0), &Reg(0))?;
                w.fcvt_d_s(c, a, &Reg(1), &Reg(0))?;
                w.fmov_d_to_gp(c, a, &Reg(0), &Reg(1))
            })?,

            // drop: pop one value and discard it (no push back).
            Instruction::Drop => self.pop(ctx, arch, &Reg(0))?,

            // ---- MVP+ additions for the spec-test suite (phase B of
            // docs/native-spectests-plan.md).  All operate on the raw
            // operand stack (RSP); values live in full 64-bit registers
            // with i32 results zero-extended by writing the 32-bit
            // subregister. ----
            Instruction::Nop => {}

            // ---- integer comparisons: pop rhs, pop lhs, cmp, cmov/setcc.
            // Signed uses L/G/LE/GE; unsigned uses B/A/BE/NB. i32 forms
            // compare the 32-bit halves (32-bit cmp zero-extends the flag
            // result the same way the 64-bit one does, but the operands
            // must be truncated: writing r32 truncates the register).
            Instruction::I32LtS | Instruction::I64LtS => {
                self.int_cmp(ctx, arch, ConditionCode::L, true)?
            }
            Instruction::I32GtS | Instruction::I64GtS => {
                self.int_cmp(ctx, arch, ConditionCode::G, true)?
            }
            Instruction::I32LeS | Instruction::I64LeS => {
                self.int_cmp(ctx, arch, ConditionCode::NG, true)?
            }
            Instruction::I32GeS | Instruction::I64GeS => {
                self.int_cmp(ctx, arch, ConditionCode::NL, true)?
            }
            Instruction::I32LtU | Instruction::I64LtU => {
                self.int_cmp(ctx, arch, ConditionCode::B, false)?
            }
            Instruction::I32GtU | Instruction::I64GtU => {
                self.int_cmp(ctx, arch, ConditionCode::A, false)?
            }
            Instruction::I32LeU | Instruction::I64LeU => {
                self.int_cmp(ctx, arch, ConditionCode::NA, false)?
            }
            Instruction::I32GeU | Instruction::I64GeU => {
                self.int_cmp(ctx, arch, ConditionCode::NB, false)?
            }

            // ---- clz / ctz / popcnt.  LZCNT/TZCNT/POPCNT encode as
            // rep-prefixed BSF/BSR and F3-prefixed POP; the encoder emits
            // them directly.  ctz(0) must be the bit width (not undefined
            // as BSF gives) — handle with a conditional select; clz(0) is
            // 64 for LZCNT (no fixup needed on 64-bit regs).
            Instruction::I32Clz | Instruction::I64Clz => {
                self.pop(ctx, arch, &Reg(0))?;
                if let Instruction::I32Clz = op {
                    // Zero-extend the operand so clz counts from bit 31.
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &r, &r)?;
                    // LZCNT eax, eax : F3 0F BD C0; clz32 = lzcnt64 - 32 for
                    // nonzero inputs, and lzcnt(0)=64 so 64-32=32 = correct.
                    self.raw(ctx, arch, &[0xF3, 0x0F, 0xBD, 0xC0])?;
                    self.mov64(ctx, arch, &Reg(1), 32)?;
                    self.sub(ctx, arch, &Reg(0), &Reg(1))?;
                } else {
                    // LZCNT rax, rax : F3 48 0F BD C0. clz(0)=64 per spec.
                    self.raw(ctx, arch, &[0xF3, 0x48, 0x0F, 0xBD, 0xC0])?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Ctz | Instruction::I64Ctz => {
                self.pop(ctx, arch, &Reg(0))?;
                // TZCNT rax, rax : F3 48 0F BC C0. tzcnt(0)=64 per spec.
                self.raw(ctx, arch, &[0xF3, 0x48, 0x0F, 0xBC, 0xC0])?;
                if let Instruction::I32Ctz = op {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &r, &r)?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Popcnt | Instruction::I64Popcnt => {
                self.pop(ctx, arch, &Reg(0))?;
                // POPCNT rax, rax : F3 48 0F B8 C0.
                self.raw(ctx, arch, &[0xF3, 0x48, 0x0F, 0xB8, 0xC0])?;
                if let Instruction::I32Popcnt = op {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_32,
                    });
                    self.mov(ctx, arch, &r, &r)?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }

            // ---- rotates: wasm rotl/rotr mask the count to the width
            // (6 bits for i32, 6 for i64 — both 64); ROL/ROR already mask
            // the count to the operand width.
            Instruction::I32Rotl | Instruction::I64Rotl => self.int_rot(ctx, arch, op, true)?,
            Instruction::I32Rotr | Instruction::I64Rotr => self.int_rot(ctx, arch, op, false)?,

            // ---- sign extends: (i32/i64).extend8_s / extend16_s;
            // i64.extend32_s shares the I64ExtendI32S arm (MOVSXD).
            Instruction::I32Extend8S => self.sign_extend(ctx, arch, MemorySize::_8, false)?,
            Instruction::I32Extend16S => self.sign_extend(ctx, arch, MemorySize::_16, false)?,
            Instruction::I64Extend8S => self.sign_extend(ctx, arch, MemorySize::_8, true)?,
            Instruction::I64Extend16S => self.sign_extend(ctx, arch, MemorySize::_16, true)?,

            // ---- globals: __wasm_globals is a plain u64 array in the
            // runtime data area (resolved via External label).  i32 globals
            // are stored zero-extended in 64-bit slots.
            Instruction::GlobalGet(global_index) => {
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_globals".into(),
                    },
                )?;
                self.mov64(ctx, arch, &Reg(1), (*global_index as u64) * 8)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::GlobalSet(global_index) => {
                self.pop(ctx, arch, &Reg(1))?; // value
                self.lea_label(
                    ctx,
                    arch,
                    &Reg(0),
                    X64Label::External {
                        name: "__wasm_globals".into(),
                    },
                )?;
                self.mov64(ctx, arch, &Reg(2), (*global_index as u64) * 8)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(2), 1)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                self.mov(
                    ctx,
                    arch,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                    &Reg(1),
                )?;
            }

            other => panic!("unimplemented WASM instruction in x86-64 naive _handle_op: {other:?}"),
        };
        Ok(())
    }
}
impl<T: Writer<X64Label, Context> + ?Sized, Context> WriterExt<Context> for T {}

/// Emit one-instruction jump stubs for each exported function.
///
/// Each stub emits an `External` label followed by a `jmp_label` to the
/// function's internal label. The caller provides `exports` as a list of
/// `(internal_id, export_name)` where `internal_id` is the WASM function index
/// minus the import count (i.e. 0-based within the internal function space).
pub fn emit_export_dispatchers<W, Ctx>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: X64Arch,
    exports: &[(u32, &str)],
) -> Result<(), W::Error>
where
    W: WriterExt<Ctx>,
{
    for (id, name) in exports {
        w.set_label(
            ctx,
            arch,
            X64Label::External {
                name: (*name).into(),
            },
        )?;
        w.jmp_label(ctx, arch, X64Label::Func { r#fn: *id })?;
    }
    Ok(())
}

#[cfg(test)]
mod membase_tests {
    use super::*;
    use crate::{X64Arch, X64Label};
    use alloc::string::String;
    use alloc::vec::Vec;
    use portal_solutions_asm_x86_64::out::iced::IcedWriter;
    use portal_solutions_blitz_common::wasm_encoder::MemArg;

    /// Assemble a single `I64Load` and return the unresolved external symbol
    /// names produced (i.e. labels never internally defined).
    fn load_externals(mem_base: MemBase, memory_index: u32) -> Vec<String> {
        let mut out = IcedWriter::<X64Label>::new(0x1000);
        let mut ctx = ();
        let mut state = State {
            mem_base,
            ..State::default()
        };
        let op = Instruction::I64Load(MemArg {
            offset: 0,
            align: 3,
            memory_index,
        });
        WriterExt::_handle_op(
            &mut out,
            &mut ctx,
            X64Arch::default(),
            &mut state,
            &[],
            &[],
            &[],
            &op,
            0,
        )
        .unwrap();
        let (_bytes, _labels, relocs) = out.into_parts_with_relocs();
        relocs
            .into_iter()
            .filter_map(|r| match r.label {
                X64Label::External { name } => Some(name),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn raw_mode_emits_no_wasm_mem_reference() {
        // Default raw-pointer addressing references no runtime memory symbol.
        assert!(
            !load_externals(MemBase::Raw, 0)
                .iter()
                .any(|n| n == "__wasm_mem")
        );
    }

    #[test]
    fn wasm_mem_symbol_mode_references_base() {
        // Symbol mode loads the `__wasm_mem` base, producing exactly one such ref.
        let externs = load_externals(MemBase::WasmMemSymbol, 0);
        assert_eq!(externs.iter().filter(|n| *n == "__wasm_mem").count(), 1);
    }

    #[test]
    fn wasm_mem_symbol_mode_uses_per_index_base() {
        let externs = load_externals(MemBase::WasmMemSymbol, 1);
        assert_eq!(externs.iter().filter(|n| *n == "__wasm_mem_1").count(), 1);
    }

    #[test]
    fn per_index_raw_skips_wasm_mem_even_when_default_is_symbol() {
        let mut out = IcedWriter::<X64Label>::new(0x1000);
        let mut ctx = ();
        let mut by_index = BTreeMap::new();
        by_index.insert(1, MemBase::Raw);
        let mut state = State {
            mem_base: MemBase::WasmMemSymbol,
            mem_base_by_index: by_index,
            ..State::default()
        };
        let op = Instruction::I64Load(MemArg {
            offset: 0,
            align: 3,
            memory_index: 1,
        });
        WriterExt::_handle_op(
            &mut out,
            &mut ctx,
            X64Arch::default(),
            &mut state,
            &[],
            &[],
            &[],
            &op,
            0,
        )
        .unwrap();
        let (_bytes, _labels, relocs) = out.into_parts_with_relocs();
        let externs: Vec<_> = relocs
            .into_iter()
            .filter_map(|r| match r.label {
                X64Label::External { name } => Some(name),
                _ => None,
            })
            .collect();
        assert!(!externs.iter().any(|n| n.starts_with("__wasm_mem")));
    }

    #[test]
    fn offset_zero_skips_lea_materialization() {
        let mut with_off = IcedWriter::<X64Label>::new(0x1000);
        let mut zero_off = IcedWriter::<X64Label>::new(0x1000);
        let mut ctx = ();
        let mut s1 = State {
            mem_base: MemBase::Raw,
            ..State::default()
        };
        let mut s2 = State {
            mem_base: MemBase::Raw,
            ..State::default()
        };
        let op_off = Instruction::I64Load(MemArg {
            offset: 8,
            align: 3,
            memory_index: 0,
        });
        let op_zero = Instruction::I64Load(MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        });
        WriterExt::_handle_op(
            &mut with_off,
            &mut ctx,
            X64Arch::default(),
            &mut s1,
            &[],
            &[],
            &[],
            &op_off,
            0,
        )
        .unwrap();
        WriterExt::_handle_op(
            &mut zero_off,
            &mut ctx,
            X64Arch::default(),
            &mut s2,
            &[],
            &[],
            &[],
            &op_zero,
            0,
        )
        .unwrap();
        assert!(with_off.into_bytes().len() > zero_off.into_bytes().len());
    }

    #[test]
    fn wasm_mem_symbol_mode_emits_more_code() {
        // The base+wrap transform adds instructions over raw addressing.
        let mut raw = IcedWriter::<X64Label>::new(0x1000);
        let mut sym = IcedWriter::<X64Label>::new(0x1000);
        let mut ctx = ();
        let op = Instruction::I64Load(MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        });
        let mut s_raw = State {
            mem_base: MemBase::Raw,
            ..State::default()
        };
        let mut s_sym = State {
            mem_base: MemBase::WasmMemSymbol,
            ..State::default()
        };
        WriterExt::_handle_op(
            &mut raw,
            &mut ctx,
            X64Arch::default(),
            &mut s_raw,
            &[],
            &[],
            &[],
            &op,
            0,
        )
        .unwrap();
        WriterExt::_handle_op(
            &mut sym,
            &mut ctx,
            X64Arch::default(),
            &mut s_sym,
            &[],
            &[],
            &[],
            &op,
            0,
        )
        .unwrap();
        assert!(sym.into_bytes().len() > raw.into_bytes().len());
    }
}
