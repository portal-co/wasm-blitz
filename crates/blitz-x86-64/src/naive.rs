//! Naive x86-64 code generation implementation.
//!
//! This module implements a straightforward, correctness-focused code generation
//! strategy for x86-64. It prioritizes simplicity and correctness over performance.

use alloc::collections::btree_map::BTreeMap;
use portal_solutions_asm_x86_64::RegisterClass;
use portal_solutions_asm_x86_64::out::arg::{ArgKind, MemArg, MemArgKind};
use portal_solutions_blitz_common::ops::TracingConfig;
use portal_solutions_blitz_common::wasm_encoder::{self, Catch, FuncType, Instruction, reencode::{self as reencode, Reencode}};

use crate::{
    out::{Writer, arg::Arg},
    *,
};

/// State tracker for x86-64 code generation.
///
/// Maintains information about the current function being compiled,
/// including local variables, control flow, and labels.
#[derive(Default)]
pub struct State {
    pub local_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub label_index: usize,
    pub if_stack: Vec<Endable>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` so tracing can be emitted after
    /// the function-entry label is placed (ensuring every call — linear or
    /// via label-jump — passes through the counter and specialisation check).
    pub tracing: Option<TracingConfig>,
    /// Next trace-site id to assign (function entry consumes site 0; each
    /// loop/block consumes the next).  See `emit_jit_preamble` / Item 1.
    pub next_site_id: u32,
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
        state: &mut State,
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

    /// Emit a tracing/specialization preamble for a loop/block control-flow
    /// site, consuming the next `site_id`.  No-op when tracing is disabled.
    ///
    /// Placed after the site's entry label and CTX-frame push, so the
    /// specialization tail-jump (if installed) inherits the operand-stack /
    /// CTX-frame layout of the generic site entry (see the blitz-specialize
    /// stack-state contract).  Uses `Reg(2)` (RDX) as scratch.
    fn emit_trace_site(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.tracing.as_ref().copied().filter(|c| c.enabled) {
            let site_id = state.next_site_id;
            state.next_site_id += 1;
            let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch };
            portal_solutions_blitz_codegen::emit_jit_preamble(
                &mut bw, cfg.table_base_off, site_id, 2, &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Emit the optional tracing preamble at a function boundary.
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
        state: &mut State,
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
                ).map_err(Err::from)?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, X64Label::Indexed { idx }).map_err(Err::from)?;
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
                        tracing,
                        ..
                    },
            } => {
                state.local_count = *params;
                state.num_returns = *num_returns;
                state.control_depth = *control_depth;
                state.tracing = *tracing;
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
                ).map_err(Err::from)?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX).map_err(Err::from)?;
                self.set_label(ctx, arch, X64Label::Func { r#fn: *id }).map_err(Err::from)?;
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    state.local_count += 1;
                    self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                }
            }
            MachOperator::StartBody => {
                state.next_site_id = 1;
                if let Some(cfg) = state.tracing.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch };
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, cfg.table_base_off, 0,
                        2, &mut state.label_index,
                    ).map_err(Err::from)?;
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
                ).map_err(Err::from)?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX).map_err(Err::from)?;
                self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                for _ in 0..state.control_depth {
                    for _ in 0..2 {
                        self.push(ctx, arch, &Reg(0)).map_err(Err::from)?;
                    }
                }
            }
            MachOperator::Instruction { op, .. } => {
                self._handle_op(ctx, arch, state, func_imports, sigs, tags, op, target).map_err(Err::from)?
            }
            MachOperator::Operator { op, annot } => match match op.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            } {
                op => self._handle_op(
                    ctx,
                    arch,
                    state,
                    func_imports,
                    sigs,
                    tags,
                    &rewriter.instruction(op.clone())?,
                    target,
                ).map_err(Err::from)?,
            },
            // EndBody and any future meta-ops are no-ops at the assembly level.
            _ => {} // non-instruction MachOperators (meta-ops)

        }
        Ok(())
    }
    fn _handle_op(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
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
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Sub | Instruction::I64Sub => {
                self.pop(ctx, arch, &Reg(0))?;   // Reg(0) = b (subtrahend, top of stack)
                self.pop(ctx, arch, &Reg(1))?;   // Reg(1) = a (minuend)
                self.not(ctx, arch, &Reg(0))?;   // ~b; result = ~b + a + 1 = a - b
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
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Mul | Instruction::I64Mul => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.mul(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Mul = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivU | Instruction::I64DivU => {
                // WASM stack: [dividend, divisor]. Pop divisor→RCX, dividend→RAX.
                // x86-64 div: rdx:rax / rcx → quotient in rax, remainder in rdx.
                self.pop(ctx, arch, &Reg(1))?;   // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?;   // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX (high half of dividend)
                self.div(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivU = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivS | Instruction::I64DivS => {
                self.pop(ctx, arch, &Reg(1))?;   // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?;   // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX (sign-extend not available yet)
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivS = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32RemU | Instruction::I64RemU => {
                self.pop(ctx, arch, &Reg(1))?;   // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?;   // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX
                self.div(ctx, arch, &Reg(0), &Reg(1))?; // remainder → RDX
                if let Instruction::I32RemU = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(2))?; // push RDX (remainder)
            }
            Instruction::I32RemS | Instruction::I64RemS => {
                self.pop(ctx, arch, &Reg(1))?;   // divisor → RCX
                self.pop(ctx, arch, &Reg(0))?;   // dividend → RAX
                self.mov64(ctx, arch, &Reg(2), 0)?; // zero RDX
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?; // remainder → RDX
                if let Instruction::I32RemS = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(2))?; // push RDX (remainder)
            }
            Instruction::I32And | Instruction::I64And => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.and(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32And = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Or | Instruction::I64Or => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.or(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Or = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Xor | Instruction::I64Xor => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Xor = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Shl | Instruction::I64Shl => {
                // x86-64 shl count must be in CL (low byte of RCX).
                // Pass count as 8-bit so text-asm emits "cl" and IcedWriter uses CL.
                let cl = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(1), size: MemorySize::_8 });
                self.pop(ctx, arch, &Reg(1))?;   // shift count → RCX
                self.pop(ctx, arch, &Reg(0))?;   // value → RAX
                self.shl(ctx, arch, &Reg(0), &cl)?;
                if let Instruction::I32Shl = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32ShrU | Instruction::I64ShrU => {
                let cl = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(1), size: MemorySize::_8 });
                self.pop(ctx, arch, &Reg(1))?;   // shift count → RCX
                self.pop(ctx, arch, &Reg(0))?;   // value → RAX
                self.shr(ctx, arch, &Reg(0), &cl)?;
                if let Instruction::I32ShrU = op {
                    { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32WrapI64 => {
                self.pop(ctx, arch, &Reg(0))?;
                { let r = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 }); self.mov(ctx, arch, &r, &r)?; }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Eqz | Instruction::I64Eqz => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &1u64)?;
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
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &1u64)?;
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
                self.cmovcc(ctx, arch, ConditionCode::E, &Reg(1), &0u64)?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I64Load(memarg) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
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
                // Dereference: load 64-bit value from [rax] into rax.
                self.mov(ctx, arch, &Reg(0), &MemArgKind::Mem {
                    base: Reg(0),
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                })?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64Store(memarg) => {
                self.pop(ctx, arch, &Reg(2))?;  // value → RDX
                self.pop(ctx, arch, &Reg(0))?;  // addr → RAX
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
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
                // Store 64-bit value from RDX to [RAX].
                self.mov(ctx, arch, &MemArgKind::Mem {
                    base: Reg(0),
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                }, &Reg(2))?;
            }
            Instruction::I32Load(memarg) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 1)),
                        disp: 0,
                        size: MemorySize::_32,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                // Dereference: load 32-bit value into eax (zero-extends to rax).
                // Previously this was `mov rax, rax`, a register-to-register
                // no-op that left the *address* in rax instead of loading the
                // value at that address.
                let eax = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 });
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
            Instruction::I32Store(memarg) => {
                self.pop(ctx, arch, &Reg(2))?;  // value → RDX
                self.pop(ctx, arch, &Reg(0))?;  // addr → RAX
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
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
                // Store 32-bit value from EDX to [RAX].
                let edx = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_32 });
                self.mov(ctx, arch, &MemArgKind::Mem {
                    base: Reg(0),
                    offset: None,
                    disp: 0,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                }, &edx)?;
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
                self.emit_trace_site(ctx, arch, state)?;
            }
            Instruction::If(blockty) => {
                let i = state.label_index;
                state.label_index += 3;
                state.if_stack.push(Endable::If { idx: i });
                self.pop(ctx, arch, &Reg(2))?;
                self.cmp0(ctx, arch, &Reg(2))?;
                self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: i + 1 })?;
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
                self.emit_trace_site(ctx, arch, state)?;
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
                        Endable::TryTable { exit_idx: _, dispatch_idx, after_dispatch_idx, catches } => {
                            // Normal fall-through: pop CTX frame (same as Block).
                            self.pop(ctx, arch, &Reg(0))?;  // exit_label
                            self.pop(ctx, arch, &Reg(1))?;  // old_RSP
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            // Jump over dispatch stub.
                            self.jmp_label(ctx, arch, X64Label::Indexed { idx: after_dispatch_idx })?;

                            // Dispatch stub: entered when throw jumps here.
                            self.set_label(ctx, arch, X64Label::Indexed { idx: dispatch_idx })?;
                            // Restore operand stack to TryTable entry RSP.
                            // The CTX frame still has old_RSP; re-read from CTX stack.
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            self.pop(ctx, arch, &Reg(0))?;  // exit_label (discard)
                            self.pop(ctx, arch, &Reg(1))?;  // old_RSP
                            self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                            self.mov(ctx, arch, &RSP, &Reg(1))?;  // restore operand RSP
                            // r2 = thrown tag index (set by Throw instruction).
                            for catch in catches.iter() {
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else { 0 };
                                        let skip_lbl = state.label_index;
                                        state.label_index += 1;
                                        self.mov64(ctx, arch, &Reg(0), *tag as u64)?;
                                        self.cmp(ctx, arch, &Reg(2), &Reg(0))?;
                                        self.jcc_label(ctx, arch, ConditionCode::NE, X64Label::Indexed { idx: skip_lbl })?;
                                        // Tag matched: push exception values (r3..r(2+arity))
                                        for i in (0..arity).rev() {
                                            self.push(ctx, arch, &Reg(3 + i as u8))?;
                                        }
                                        self.br(ctx, arch, state, *label)?;
                                        self.set_label(ctx, arch, X64Label::Indexed { idx: skip_lbl })?;
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
                            self.lea_label(ctx, arch, &Reg(0), X64Label::External {
                                name: alloc::format!("__wasm_exn_propagate"),
                            })?;
                            self.jmp(ctx, arch, &Reg(0))?;
                            self.set_label(ctx, arch, X64Label::Indexed { idx: after_dispatch_idx })?;
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
                } else { 0 };
                // Tag index → r2; exception values → r3, r4, r5 (up to arity 3).
                self.mov64(ctx, arch, &Reg(2), *tag_index as u64)?;
                for i in 0..arity.min(3) {
                    self.pop(ctx, arch, &Reg(3 + i as u8))?;
                }
                // Static dispatch: jump to innermost TryTable's dispatch stub if present.
                if let Some(dispatch_idx) = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_idx, .. } => Some(*dispatch_idx),
                    _ => None,
                }) {
                    self.jmp_label(ctx, arch, X64Label::Indexed { idx: dispatch_idx })?;
                } else {
                    // No intra-function handler: propagate via CTX chain.
                    self.lea_label(ctx, arch, &Reg(0), X64Label::External {
                        name: alloc::format!("__wasm_exn_propagate"),
                    })?;
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
                    catches: catches.iter().cloned().collect::<alloc::vec::Vec<_>>().into_boxed_slice(),
                });
                // Push CTX frame (same as Block): old_RSP + exit_label.
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: exit_idx })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(1))?;  // old_RSP
                self.push(ctx, arch, &Reg(0))?;  // exit_label
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: exit_idx })?;
            }
            Instruction::Call(function_index) => {
                match func_imports.get(*function_index as usize) {
                    Some((module, name)) => {
                        let sym = alloc::format!("{module}__{name}");
                        self.lea_label(ctx, arch, &Reg(0), X64Label::External { name: sym })?;
                        self.call(ctx, arch, &Reg(0))?;
                    }
                    None => {
                        let idx = *function_index - func_imports.len() as u32;
                        self.lea_label(ctx, arch, &Reg(0), X64Label::Func { r#fn: idx })?;
                        self.call(ctx, arch, &Reg(0))?;
                    }
                }
            }
            // ---- memory.size ------------------------------------------------
            // Load __wasm_mem_pages (32-bit global) and push as i64 onto the WASM stack.
            // The concrete writer must resolve X64Label::External symbols.
            Instruction::MemorySize(_) => {
                // Get address of the pages-count global into Reg(0).
                self.lea_label(ctx, arch, &Reg(0), X64Label::External { name: "__wasm_mem_pages".into() })?;
                // Load the 32-bit value (zero-extend to 64 bits).
                // Dest must be eax (the 32-bit half) for the assembler to accept
                // `mov eax, dword ptr [rax]` — writing eax auto-zero-extends rax.
                let eax = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 });
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
                self.lea_label(ctx, arch, &Reg(0), X64Label::External { name: "__wasm_memory_grow".into() })?;
                self.call(ctx, arch, &Reg(0))?;
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
        w.set_label(ctx, arch, X64Label::External { name: (*name).into() })?;
        w.jmp_label(ctx, arch, X64Label::Func { r#fn: *id })?;
    }
    Ok(())
}
