//! Naive RISC-V codegen (incremental port)

#![allow(dead_code)]

/// Feature-gated trace macro.  Enable with `--features portal-solutions-blitz-riscv64/log`.
#[cfg(feature = "log")]
macro_rules! trace {
    ($($arg:tt)*) => { eprintln!("[riscv64-naive] {}", format_args!($($arg)*)) };
}
#[cfg(not(feature = "log"))]
macro_rules! trace {
    ($($arg:tt)*) => {};
}

use crate::RiscvLabel;
use alloc::{collections::BTreeMap, vec::Vec};
use portal_solutions_asm_riscv64::ConditionCode;
use portal_solutions_asm_riscv64::RiscV64Arch;
use portal_solutions_asm_riscv64::out::Writer;

use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::ops::{MachOperator, ProbeTableConfig};
use portal_solutions_blitz_common::shard::{CallTarget, SecondCtxConfig};
use portal_solutions_blitz_common::wasm_encoder;
use portal_solutions_blitz_common::wasm_encoder::reencode::{self as reencode, Reencode};

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv64::RegisterClass;
use portal_solutions_asm_riscv64::out::arg::{ArgKind, MemArgKind};

use portal_solutions_asm_regalloc as regalloc;
use portal_solutions_asm_riscv64 as asm_riscv;
use portal_solutions_asm_riscv64::regalloc as riscv_regalloc;

/// Static Context Register (SCR) — S10 (x26) on RISC-V 64.
///
/// Callee-saved; holds the cross-shard function-pointer table pointer when
/// sharding is active. See `docs/second-context-register.md`.
pub const SCR: Reg = Reg(26);

/// Host address calculation for WASM linear-memory accesses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemBase {
    #[default]
    Raw,
    WasmMemSymbol,
}

impl MemBase {
    pub fn is_zero(self) -> bool {
        matches!(self, Self::Raw)
    }
}

/// Inter-function convention for the SysV entry path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallAbi {
    #[default]
    RegSysv,
    AllStack,
}

/// Sharding state for RISC-V 64 functions — same design as x86-64/AArch64.
#[derive(Clone, Copy)]
pub struct NaiveShardState<'a> {
    pub config: SecondCtxConfig,
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
        Self { config, current_shard, imports_len, map }
    }

    pub fn call_target(&self, callee_fn: u32) -> CallTarget {
        if callee_fn < self.imports_len {
            return CallTarget::Import;
        }
        let callee_shard = self.map.shard_for(callee_fn);
        if callee_shard == self.current_shard {
            CallTarget::Local
        } else {
            CallTarget::CrossShard { table_slot: callee_fn }
        }
    }
}

#[derive(Default)]
pub struct State<'a> {
    pub label_index: usize,
    pub local_count: usize,
    /// Number of incoming parameters, used to bound true tail-call overwrites.
    pub param_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub if_stack: Vec<Endable>,
    pub regalloc: Option<regalloc::RegAlloc<riscv_regalloc::RegKind, 32, Frames<riscv_regalloc::RegKind, 32>>>,
    pub body: u32,
    pub body_labels: alloc::collections::BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` for RISC-V NaiveAbi — not actually
    /// needed here since the label is placed before frame setup, but kept for
    /// consistency with the x86-64 backend.  Preamble is emitted in `StartFn`.
    pub probes: Option<ProbeTableConfig>,
    /// Next probe id to assign (function entry = probe 0; each loop/block
    /// consumes the next).  See `emit_probe_site`.
    pub next_probe_id: u32,
    /// Total frame size in bytes, set by SysV `StartFn` to locate the RA/FP
    /// save slots at the bottom of the frame (`[FP - sysv_frame_sz]` = RA).
    pub sysv_frame_sz: i32,
    /// How mid-function probe sites reach the runtime probe-table base.  The
    /// NaiveAbi keeps the default (CTX-relative); the SysV ABI sets this to a
    /// frame slot after spilling its virtual-param base register.
    pub probe_base: crate::codegen::ProbeBase,
    /// Present when sharding is active. SCR (S10/x26) is saved in the SysV
    /// frame. Naive functions use SCR read-only (the runtime sets it).
    pub shard: Option<NaiveShardState<'a>>,
    /// Embedder-requested probes at arbitrary instruction indices, in addition
    /// to the function-entry/loop/block probes above.  `None` → zero overhead,
    /// identical codegen to today.
    pub probe_plan: Option<portal_solutions_blitz_common::ops::ProbePlan>,
    /// Ordinal index of the next dispatched instruction (0 = the first real
    /// WASM operator after locals), used to look up `probe_plan` entries.
    pub op_index: usize,
    /// Default linear-memory host base policy.
    pub mem_base: MemBase,
    /// Per-memory-index overrides.
    pub mem_base_by_index: BTreeMap<u32, MemBase>,
    /// SysV internal-call convention.
    pub call_abi: CallAbi,
    /// Number of imported functions in the WASM index space.
    pub n_imports: u32,
    /// Function parameter/result arities, imports first.
    pub call_params: Vec<u32>,
    pub call_results: Vec<u32>,
    /// Type parameter/result arities for indirect calls.
    pub sig_params: Vec<u32>,
    pub sig_results: Vec<u32>,
}

impl State<'_> {
    pub fn mem_base_for(&self, memory_index: u32) -> MemBase {
        self.mem_base_by_index
            .get(&memory_index)
            .copied()
            .unwrap_or(self.mem_base)
    }
}

/// Register frames for the RISC-V regalloc, sized for 32 int/32 float regs.
///
/// Shared adapter — see `portal_solutions_blitz_codegen::regalloc_adapter`.
/// The x86-64 `fast` backend uses the same adapter with its own `RegKind`.
pub use portal_solutions_blitz_codegen::regalloc_adapter::Frames;

pub enum Endable {
    /// `Block`/`Loop`/`If` — see `portal_solutions_blitz_codegen::control_flow`.
    Std(portal_solutions_blitz_codegen::control_flow::Frame),
    TryTable {
        exit_idx: usize,
        dispatch_idx: usize,
        after_dispatch_idx: usize,
        catches: alloc::boxed::Box<[portal_solutions_blitz_common::wasm_encoder::Catch]>,
    },
}

pub trait WriterExt<Context>: Writer<RiscvLabel, Context> {
    /// Emit a control-flow probe site (`TailTakeover` binding) for a
    /// loop/block header, consuming the next `probe_id`.  No-op when probes
    /// are disabled.
    ///
    /// Flushes regalloc first so the operand stack is materialised at the site
    /// (matching the generic-entry layout the specialization tail-jump expects).
    /// Uses t0 (Reg 5) as scratch, t1 (Reg 6) as the `inc_mem64` scratch.
    fn emit_control_flow_probe(&mut self, ctx: &mut Context, arch: RiscV64Arch, state: &mut State<'_>)
        -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
            if let Some(ralloc) = state.regalloc.as_mut() {
                let it = ralloc.flush();
                emit_cmds(self, ctx, arch, it)?;
                ralloc.tos = None;
            }
            let probe_id = state.next_probe_id;
            state.next_probe_id += 1;
            let probe_base = state.probe_base;
            let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: 6, probe_base };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw, cfg.table_base_off, probe_id, 5,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Apply a linear-memory base and a full-width WASM memarg offset to `addr`.
    /// RISC-V load/store displacements are signed 12-bit, so larger offsets are
    /// materialized in the address register instead of truncating the u64 offset.
    fn apply_mem_base(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        base: MemBase,
        addr: Reg,
        scratch: Reg,
        memory_index: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if base.is_zero() {
            return Ok(());
        }
        // addr := (uint32_t)addr.  RV64 has no dedicated zext.w in WriterCore,
        // so use the canonical pair of variable shifts.
        self.li(ctx, arch, &scratch, 32)?;
        self.sll(ctx, arch, &addr, &addr, &scratch)?;
        self.srl(ctx, arch, &addr, &addr, &scratch)?;
        let sym = if memory_index == 0 {
            "__wasm_mem".into()
        } else {
            alloc::format!("__wasm_mem_{memory_index}")
        };
        self.la_label(ctx, arch, &scratch, RiscvLabel::External { name: sym })?;
        self.ld(ctx, arch, &scratch, &MemArgKind::Mem {
            base: ArgKind::Reg { reg: scratch, size: MemorySize::_64 },
            offset: None, disp: 0, size: MemorySize::_64, reg_class: RegisterClass::Gpr,
        })?;
        self.add(ctx, arch, &addr, &addr, &scratch)
    }

    fn mem_add_offset(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        addr: Reg,
        scratch: Reg,
        offset: u64,
    ) -> Result<i32, Self::Error>
    where
        Self: Sized,
    {
        if offset <= 2047 {
            return Ok(offset as i32);
        }
        self.li(ctx, arch, &scratch, offset)?;
        self.add(ctx, arch, &addr, &addr, &scratch)?;
        Ok(0)
    }

    fn br(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        relative_depth: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // flush regalloc before branching; reset TOS to avoid stale chain pointers
        if let Some(ralloc) = state.regalloc.as_mut() {
            let it = ralloc.flush();
            emit_cmds(self, ctx, arch, it)?;
            ralloc.tos = None;
        }
        self.br_after_flush(ctx, arch, &state.if_stack, relative_depth)
    }

    /// The label-resolution half of [`Self::br`], without the regalloc flush.
    ///
    /// Split out so `BrTable` (which flushes once up front, then resolves
    /// each arm via [`portal_solutions_blitz_codegen::emit_br_table`]) can
    /// reuse it without re-flushing per arm and without needing `&mut State`
    /// inside the `resolve` closure (see the `BrTable` match arm below).
    fn br_after_flush(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        if_stack: &[Endable],
        relative_depth: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let mut depth = relative_depth as usize;
        for entry in if_stack.iter().rev() {
            if depth == 0 {
                // Which label to jump to is the same rule for Block/Loop/If
                // regardless of ISA — see Frame::branch_target(). TryTable
                // stays local (its exit label isn't part of the shared Frame).
                let target_idx = match entry {
                    Endable::Std(frame) => frame.branch_target(),
                    Endable::TryTable { exit_idx, .. } => *exit_idx,
                };
                let lbl = RiscvLabel::Indexed { idx: target_idx };
                self.jal_label(ctx, arch, &portal_solutions_blitz_common::asm::Reg(0), lbl)?;
                return Ok(());
            }
            depth -= 1;
        }
        Ok(())
    }
    /// Emit the optional tracing preamble.
    ///
    /// - **NaiveAbi**: call in `StartFn` after `set_label`, before frame setup.
    ///   Use `scratch = Reg(5)` (t0); RISC-V NaiveAbi passes args on the WASM stack.
    /// - **SysVAbi**: call in `StartFn` after `set_label`, before `addi sp, sp, -N`.
    ///   Use `scratch = Reg(5)` (t0); SysV arg regs a0–a7 (Reg 10–17) are untouched.
    fn handle_op_<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[portal_solutions_blitz_common::wasm_encoder::FuncType],
        tags: &[u32],
        op: &portal_solutions_blitz_common::wasm_encoder::Instruction<'_>,
        _rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        trace!("handle_op_ enter: target={target} body={}", state.body);
        if target != state.body {
            // First-instruction guard: state.body == 0 is the Default value,
            // not a real prior body.  See aarch64/naive.rs for the rationale.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jal_label(
                    ctx,
                    arch,
                    &Reg(0),
                    RiscvLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, RiscvLabel::Indexed { idx })?;
                }
            }
        }
        use portal_solutions_blitz_common::wasm_encoder::Instruction;
        match op {
            Instruction::I32Const(v) => {
                let v = *v as u64;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::push_const(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, reg| rw.writer.li(rw.ctx, rw.arch, &Reg(reg), v),
                )?;
            }
            Instruction::I64Const(v) => {
                let v = *v as u64;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::push_const(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, reg| rw.writer.li(rw.ctx, rw.arch, &Reg(reg), v),
                )?;
            }
            Instruction::LocalGet(local_index) => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::push_local(
                    &mut rw, riscv_regalloc::RegKind::Int, *local_index,
                )?;
            }
            Instruction::LocalSet(local_index) => {
                // pop_to_local transitions TOS from Stack → Local(N), marking the register as
                // holding local N's value. No memory write yet; flush() or eviction emits SetLocal.
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::pop_to_local(
                    &mut rw, riscv_regalloc::RegKind::Int, *local_index,
                )?;
            }
            Instruction::LocalTee(local_index) => {
                let fp = Reg(8);
                let tmp = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &tmp, &spmem)?;
                let disp = -((*local_index as i32 + 1) * 8);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: fp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &tmp, &mem)?;
                // push tmp back
                let sp = Reg(2);
                self.addi(ctx, arch, &sp, &sp, -8)?;
                let spmem2 = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &tmp, &spmem2)?;
            }
            Instruction::I64Load(memarg) | Instruction::F64Load(memarg) => {
                let base = state.mem_base_for(memarg.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dest, addr| {
                        rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dest), memarg.memory_index)?;
                        let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dest), memarg.offset)?;
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 },
                            offset: None, disp, size: MemorySize::_64, reg_class: RegisterClass::Gpr,
                        };
                        rw.writer.ld(rw.ctx, rw.arch, &Reg(dest), &mem)
                    },
                )?;
            }
            Instruction::I32Load(memarg) | Instruction::F32Load(memarg) => {
                let base = state.mem_base_for(memarg.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dest, addr| {
                        rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dest), memarg.memory_index)?;
                        let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dest), memarg.offset)?;
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 },
                            offset: None, disp, size: MemorySize::_32, reg_class: RegisterClass::Gpr,
                        };
                        rw.writer.lw(rw.ctx, rw.arch, &Reg(dest), &mem)
                    },
                )?;
            }
            Instruction::I32Store(memarg) | Instruction::I64Store32(memarg) | Instruction::F32Store(memarg) => {
                let base = state.mem_base_for(memarg.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::store(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, val, addr| {
                        rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(5), memarg.memory_index)?;
                        let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(5), memarg.offset)?;
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 },
                            offset: None, disp, size: MemorySize::_32, reg_class: RegisterClass::Gpr,
                        };
                        rw.writer.sw(rw.ctx, rw.arch, &Reg(val), &mem)
                    },
                )?;
            }
            Instruction::I64Store(memarg) | Instruction::F64Store(memarg) => {
                let base = state.mem_base_for(memarg.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::store(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, val, addr| {
                        rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(5), memarg.memory_index)?;
                        let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(5), memarg.offset)?;
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 },
                            offset: None, disp, size: MemorySize::_64, reg_class: RegisterClass::Gpr,
                        };
                        rw.writer.sd(rw.ctx, rw.arch, &Reg(val), &mem)
                    },
                )?;
            }
            Instruction::I32Load8S(m) | Instruction::I64Load8S(m) => {
                let base = state.mem_base_for(m.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(&mut rw, riscv_regalloc::RegKind::Int, |rw, dst, addr| {
                    rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dst), m.memory_index)?;
                    let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dst), m.offset)?;
                    rw.writer.lb(rw.ctx, rw.arch, &Reg(dst), &MemArgKind::Mem { base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 }, offset: None, disp, size: MemorySize::_8, reg_class: RegisterClass::Gpr })
                })?;
            }
            Instruction::I32Load16S(m) | Instruction::I64Load16S(m) => {
                let base = state.mem_base_for(m.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(&mut rw, riscv_regalloc::RegKind::Int, |rw, dst, addr| {
                    rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dst), m.memory_index)?;
                    let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dst), m.offset)?;
                    rw.writer.lh(rw.ctx, rw.arch, &Reg(dst), &MemArgKind::Mem { base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 }, offset: None, disp, size: MemorySize::_16, reg_class: RegisterClass::Gpr })
                })?;
            }
            Instruction::I32Load8U(m) | Instruction::I64Load8U(m) | Instruction::I32Load16U(m) | Instruction::I64Load16U(m) | Instruction::I64Load32U(m) => {
                let (width, shift) = match op {
                    Instruction::I32Load8U(_) | Instruction::I64Load8U(_) => (MemorySize::_8, 56),
                    Instruction::I32Load16U(_) | Instruction::I64Load16U(_) => (MemorySize::_16, 48),
                    _ => (MemorySize::_32, 32),
                };
                let base = state.mem_base_for(m.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(&mut rw, riscv_regalloc::RegKind::Int, |rw, dst, addr| {
                    rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dst), m.memory_index)?;
                    let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dst), m.offset)?;
                    let mem = MemArgKind::Mem { base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 }, offset: None, disp, size: width, reg_class: RegisterClass::Gpr };
                    if width == MemorySize::_8 { rw.writer.lb(rw.ctx, rw.arch, &Reg(dst), &mem)?; }
                    else if width == MemorySize::_16 { rw.writer.lh(rw.ctx, rw.arch, &Reg(dst), &mem)?; }
                    else { rw.writer.lw(rw.ctx, rw.arch, &Reg(dst), &mem)?; }
                    rw.writer.li(rw.ctx, rw.arch, &Reg(5), shift)?;
                    rw.writer.sll(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(5))?;
                    rw.writer.srl(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(5))
                })?;
            }
            Instruction::I64Load32S(m) => {
                let base = state.mem_base_for(m.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::load(&mut rw, riscv_regalloc::RegKind::Int, |rw, dst, addr| {
                    rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(dst), m.memory_index)?;
                    let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(dst), m.offset)?;
                    rw.writer.lw(rw.ctx, rw.arch, &Reg(dst), &MemArgKind::Mem { base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 }, offset: None, disp, size: MemorySize::_32, reg_class: RegisterClass::Gpr })
                })?;
            }
            Instruction::I32Store8(m) | Instruction::I64Store8(m) | Instruction::I32Store16(m) | Instruction::I64Store16(m) => {
                let width = match op { Instruction::I32Store8(_) | Instruction::I64Store8(_) => MemorySize::_8, _ => MemorySize::_16 };
                let base = state.mem_base_for(m.memory_index);
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::store(&mut rw, riscv_regalloc::RegKind::Int, |rw, val, addr| {
                    rw.writer.apply_mem_base(rw.ctx, rw.arch, base, Reg(addr), Reg(5), m.memory_index)?;
                    let disp = rw.writer.mem_add_offset(rw.ctx, rw.arch, Reg(addr), Reg(5), m.offset)?;
                    let mem = MemArgKind::Mem { base: ArgKind::Reg { reg: Reg(addr), size: MemorySize::_64 }, offset: None, disp, size: width, reg_class: RegisterClass::Gpr };
                    if width == MemorySize::_8 { rw.writer.sb(rw.ctx, rw.arch, &Reg(val), &mem) } else { rw.writer.sh(rw.ctx, rw.arch, &Reg(val), &mem) }
                })?;
            }
            // I32Add is the same as I64Add at the regalloc/register level — RISC-V add
            // operates on 64-bit registers; the lower 32 bits give the correct i32 result.
            Instruction::I32Add |
            Instruction::I64Add => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.add(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Sub |
            Instruction::I64Sub => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.sub(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Mul |
            Instruction::I64Mul => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.mul(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32And |
            Instruction::I64And => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.and(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Or |
            Instruction::I64Or => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.or(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Xor |
            Instruction::I64Xor => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.xor(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Shl |
            Instruction::I64Shl => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.sll(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32ShrS |
            Instruction::I64ShrS => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.sra(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32ShrU |
            Instruction::I64ShrU => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.srl(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Eq |
            Instruction::I64Eq => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::EQ, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32Ne |
            Instruction::I64Ne => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::NE, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32LtS |
            Instruction::I64LtS => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::LT, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32LtU |
            Instruction::I64LtU => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::LTU, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32GtS |
            Instruction::I64GtS => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::LT, &rb, &ra, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32GtU |
            Instruction::I64GtU => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::LTU, &rb, &ra, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32LeS |
            Instruction::I64LeS => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::GT, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32LeU |
            Instruction::I64LeU => {
                let label_index = &mut state.label_index;
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::compare(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    move |rw, dest, ta, tb| {
                        let (ra, rb) = (Reg(ta), Reg(tb));
                        let dest = Reg(dest);
                        let i = *label_index;
                        *label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        rw.writer.bcond_label(rw.ctx, rw.arch, ConditionCode::GTU, &ra, &rb, lbl_true.clone())?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 0)?;
                        rw.writer.jal_label(rw.ctx, rw.arch, &portal_solutions_blitz_common::asm::Reg(0), lbl_end.clone())?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_true)?;
                        rw.writer.li(rw.ctx, rw.arch, &dest, 1)?;
                        rw.writer.set_label(rw.ctx, rw.arch, lbl_end)?;
                        Ok(())
                    },
                )?;
            }
            Instruction::I32DivU | Instruction::I64DivU => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::regalloc_frontend::binop(
                    &mut rw, riscv_regalloc::RegKind::Int,
                    |rw, dst, rhs| rw.writer.divu(rw.ctx, rw.arch, &Reg(dst), &Reg(dst), &Reg(rhs)),
                )?;
            }
            Instruction::I32Eqz | Instruction::I64Eqz => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                let (ta, cmds_a) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds_a)?;
                let ra = Reg(ta.reg);
                // `seqz` is the RISC-V pseudo-op `sltiu rd, rs, 1`; WriterCore
                // has no immediate SLT form, so emit its equivalent branch form
                // without destroying the input before the comparison.
                let i = state.label_index;
                state.label_index += 2;
                let yes = RiscvLabel::Indexed { idx: i };
                let done = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(ctx, arch, ConditionCode::EQ, &ra, &Reg(0), yes.clone())?;
                self.li(ctx, arch, &ra, 0)?;
                self.jal_label(ctx, arch, &Reg(0), done.clone())?;
                self.set_label(ctx, arch, yes)?;
                self.li(ctx, arch, &ra, 1)?;
                self.set_label(ctx, arch, done)?;
                let it = state.regalloc.as_mut().unwrap()
                    .push_existing(regalloc::Target { reg: ta.reg, kind: ta.kind });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32WrapI64 => {
                // No-op at the register level; lower 32 bits are already the i32 value.
            }
            Instruction::Br(relative_depth) => {
                self.br(ctx, arch, state, *relative_depth)?;
            }
            Instruction::BrIf(relative_depth) => {
                // flush regalloc before conditional branch; reset TOS
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let i = state.label_index;
                state.label_index += 1;
                let skip = RiscvLabel::Indexed { idx: i };
                let tmp = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &tmp, &spmem)?;
                self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                self.bcond_label(
                    ctx,
                    arch,
                    ConditionCode::EQ,
                    &tmp,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    skip.clone(),
                )?;
                self.br(ctx, arch, state, *relative_depth)?;
                self.set_label(ctx, arch, skip)?;
            }
            Instruction::BrTable(targets, default) => {
                // flush regalloc once, up front — emit_br_table's resolve
                // closure calls br_after_flush (no per-arm re-flush needed).
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let idx_reg = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &idx_reg, &spmem)?;
                self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                // Shared decrement-approach br_table (branch_zero_label + reg_decrement
                // per arm) — see portal_solutions_blitz_codegen::emit_br_table.
                let mut bw = crate::codegen::BlitzW::new(self, ctx, arch, 6);
                portal_solutions_blitz_codegen::emit_br_table(
                    &mut bw,
                    idx_reg.0,
                    &targets[..],
                    *default,
                    &mut state.label_index,
                    |w, relative_depth| {
                        w.writer.br_after_flush(w.ctx, w.arch, &state.if_stack, relative_depth)
                    },
                )?;
            }
            Instruction::Block(_blockty) => {
                // Do NOT emit the label here: Br(N) to a Block is a forward branch to
                // the block's End, so the label must only be placed at End time.
                let frame = portal_solutions_blitz_codegen::control_flow::open_block(&mut state.label_index);
                state.if_stack.push(Endable::Std(frame));
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::If(_blockty) => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                let frame = portal_solutions_blitz_codegen::control_flow::open_if(&mut rw, &mut state.label_index)?;
                state.if_stack.push(Endable::Std(frame));
            }
            Instruction::Else => {
                let frame = match state.if_stack.last() {
                    Some(Endable::Std(frame)) => *frame,
                    _ => panic!("Else without If"),
                };
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                portal_solutions_blitz_codegen::control_flow::emit_else(&mut rw, &frame)?;
            }
            Instruction::Loop(_blockty) => {
                let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                let frame = portal_solutions_blitz_codegen::control_flow::open_loop(&mut rw, &mut state.label_index)?;
                state.if_stack.push(Endable::Std(frame));
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::End => {
                // Function-level End (empty if_stack) is a no-op: the function
                // return path already cleaned up the frame.
                if let Some(top) = state.if_stack.pop() {
                    match top {
                        Endable::Std(frame) => {
                            let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
                            portal_solutions_blitz_codegen::control_flow::close_frame(&mut rw, frame)?;
                        }
                        Endable::TryTable { exit_idx, dispatch_idx, after_dispatch_idx, catches } => {
                            if let Some(ralloc) = state.regalloc.as_mut() {
                                let it = ralloc.flush();
                                emit_cmds(self, ctx, arch, it)?;
                            }
                            let ra = portal_solutions_blitz_common::asm::Reg(0);
                            // Normal path: jump over dispatch stub.
                            self.jal_label(ctx, arch, &ra, RiscvLabel::Indexed { idx: after_dispatch_idx })?;
                            // Dispatch stub.
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: dispatch_idx })?;
                            for catch in catches.iter() {
                                use portal_solutions_blitz_common::wasm_encoder::Catch;
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else { 0 };
                                        let skip_idx = state.label_index;
                                        state.label_index += 1;
                                        // T0 = thrown tag (set by Throw), compare with this tag
                                        self.addi(ctx, arch, &Reg(11), &Reg(0), *tag as i32)?; // a1 = tag
                                        // branch-not-equal to skip label
                                        // Use bne: bne T0, a1, skip
                                        self.jal_label(ctx, arch, &ra, RiscvLabel::Indexed { idx: skip_idx })?; // placeholder: need bne
                                        // Push exception values (a2..a(1+arity))
                                        for i in (0..arity.min(3)).rev() {
                                            push(self, ctx, arch, Reg(12 + i as u8))?;
                                        }
                                        self.br(ctx, arch, state, *label)?;
                                        self.set_label(ctx, arch, RiscvLabel::Indexed { idx: skip_idx })?;
                                    }
                                    Catch::All { label } => {
                                        self.br(ctx, arch, state, *label)?;
                                    }
                                    Catch::OneRef { .. } | Catch::AllRef { .. } => {}
                                }
                            }
                            self.jal_label(ctx, arch, &Reg(1), RiscvLabel::External { name: "__wasm_exn_propagate".into() })?;
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: after_dispatch_idx })?;
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: exit_idx })?;
                        }
                    }
                    // No SP cleanup here: the regalloc manages the operand stack, and
                    // the SysV frame (via FP) restores SP correctly at function return.
                    // Running control_space+A0 restoration on every End would corrupt
                    // the stack for nested blocks (each End would undo too much).
                }
            }
            Instruction::Call(function_index) => {
                trace!("handle_op_: Call({function_index}) flush start, regalloc={}", state.regalloc.is_some());
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let mut n = 0usize;
                    for cmd in ralloc.flush() {
                        trace!("handle_op_: Call flush cmd #{n}");
                        n += 1;
                        trace!("handle_op_: Call flush cmd #{n} process_cmd start");
                        riscv_regalloc::process_cmd(self, ctx, arch, &cmd)?;
                        trace!("handle_op_: Call flush cmd #{n} process_cmd done");
                    }
                    trace!("handle_op_: Call flush done ({n} cmds)");
                }
                // Use ra (x1) as the link register so the callee can return correctly.
                let ra = Reg(1);
                let target = state.shard.as_ref().map(|s| s.call_target(*function_index));
                match target {
                    Some(CallTarget::CrossShard { table_slot }) => {
                        // Cross-shard: load fn ptr from [SCR + table_slot * 8] into t0.
                        let t0 = Reg(5);
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: SCR, size: MemorySize::_64 },
                            offset: None,
                            disp: table_slot as i32 * 8,
                            size: MemorySize::_64,
                            reg_class: RegisterClass::Gpr,
                        };
                        self.ld(ctx, arch, &t0, &mem)?;
                        self.jalr(ctx, arch, &ra, &t0, 0)?;
                    }
                    _ => {
                        match func_imports.get(*function_index as usize) {
                            Some((module, name)) => {
                                let sym = alloc::format!("{module}__{name}");
                                self.jal_label(ctx, arch, &ra, RiscvLabel::External { name: sym })?;
                            }
                            None => {
                                let idx = *function_index - func_imports.len() as u32;
                                self.jal_label(ctx, arch, &ra, RiscvLabel::Func { r#fn: idx })?;
                            }
                        }
                    }
                }
            }
            Instruction::Return => {
                // flush regalloc before return
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                // function epilogue: restore sp from fp, restore saved fp, return
                let sp = Reg(2);
                let fp = Reg(8);
                // set sp = fp
                self.mv(ctx, arch, &sp, &fp)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: sp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                let saved_fp = Reg(10);
                self.ld(ctx, arch, &saved_fp, &mem)?;
                self.addi(ctx, arch, &sp, &sp, 8)?;
                self.mv(ctx, arch, &fp, &saved_fp)?;
                self.ret(ctx, arch)?;
            }
            // ---- memory.size ------------------------------------------------
            // Load the 32-bit `__wasm_mem_pages` global and push as i64.
            // The concrete writer must resolve RiscvLabel::External symbols.
            Instruction::MemorySize(_) => {
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                let dest = Reg(10);
                // Load the *address* of __wasm_mem_pages into dest using the
                // `la` pseudo-instruction.  Using `jal` here would jump to the
                // data symbol and execute its bytes as instructions.
                self.la_label(ctx, arch, &dest, RiscvLabel::External { name: "__wasm_mem_pages".into() })?;
                // Dereference: load 32-bit page count, zero-extend to 64 bits.
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: dest, size: MemorySize::_64 },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &dest, &mem)?;
                // Push onto WASM stack.
                push(self, ctx, arch, dest)?;
            }
            // ---- memory.grow ------------------------------------------------
            // delta is already on the WASM stack top. Call __wasm_memory_grow
            // using the same WASM calling convention as regular function calls:
            // the callee pops the return address, accesses delta via its frame
            // pointer, and pushes old_pages before returning.
            Instruction::MemoryGrow(_) => {
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                // `jal ra, __wasm_memory_grow` is the canonical RISC-V direct
                // call: it transfers control to the function and stores PC+4
                // in `ra` for the callee to return through.  Previously this
                // emitted `jal a0, sym; call a0`, which jumped to the function
                // and then re-called the now-corrupted a0 (an infinite loop).
                let ra = Reg(1);
                self.jal_label(ctx, arch, &ra, RiscvLabel::External { name: "__wasm_memory_grow".into() })?;
            }
            // `__wasm_memory_init_copy(dest_offset, seg_base, src_offset, len)`
            // uses the normal RISC-V psABI, unlike the internal WASM grow helper.
            Instruction::MemoryInit { data_index, .. } => {
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                pop(self, ctx, arch, &Reg(13))?; // a3 = len
                pop(self, ctx, arch, &Reg(12))?; // a2 = source offset
                pop(self, ctx, arch, &Reg(10))?; // a0 = destination offset
                self.la_label(ctx, arch, &Reg(11), RiscvLabel::External {
                    name: alloc::format!("__wasm_data_seg_{data_index}"),
                })?;
                self.la_label(ctx, arch, &Reg(5), RiscvLabel::External {
                    name: "__wasm_memory_init_copy".into(),
                })?;
                self.jalr(ctx, arch, &Reg(1), &Reg(5), 0)?;
            }
            // AOT data segments are immutable symbols; dropping their runtime
            // liveness is intentionally a no-op, like the other backends.
            Instruction::DataDrop(_) => {}
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else { 0 };
                // T0 (a0/x10) = tag index, T2..T(1+arity) = exception values
                self.addi(ctx, arch, &Reg(10), &Reg(0), *tag_index as i32)?; // a0 = tag
                for i in 0..arity.min(3) {
                    pop(self, ctx, arch, &Reg(12 + i as u8))?; // a2, a3, a4
                }
                if let Some(dispatch_idx) = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_idx, .. } => Some(*dispatch_idx),
                    _ => None,
                }) {
                    self.jal_label(ctx, arch, &Reg(0), RiscvLabel::Indexed { idx: dispatch_idx })?;
                } else {
                    self.jal_label(ctx, arch, &Reg(1), RiscvLabel::External { name: "__wasm_exn_propagate".into() })?;
                }
            }
            Instruction::ThrowRef => { /* exnref deferred */ }
            // drop: pop one value and discard it (no push back).
            Instruction::Drop => pop(self, ctx, arch, &Reg(5))?,
            Instruction::TryTable(_, catches) => {
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
                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: exit_idx })?;
            }
            other => panic!("unimplemented WASM instruction in RISC-V naive handle_op: {other:?}"),
        }
        Ok(())
    }
    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[portal_solutions_blitz_common::wasm_encoder::FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        trace!("handle_op enter: target={target} body={}", state.body);
        if target != state.body {
            // First-instruction guard: state.body == 0 is the Default value,
            // not a real prior body.  See aarch64/naive.rs for the rationale.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jal_label(
                    ctx,
                    arch,
                    &Reg(0),
                    RiscvLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                ).map_err(Err::from)?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, RiscvLabel::Indexed { idx }).map_err(Err::from)?;
                }
            }
        }
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.regalloc = None;
                state.probes = data.probes;
                state.next_probe_id = 1;

                self.set_label(ctx, arch, RiscvLabel::Func { r#fn: *id }).map_err(Err::from)?;

                // Function-entry probe: after label, before frame setup.
                // Scratch: t0 (Reg(5)) + t1 (Reg(6)) — caller-saved, not NaiveAbi arg regs.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW::new(self, ctx, arch, 6);
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw, cfg.table_base_off, 0, 5,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                let sp = Reg(2);
                let fp = Reg(8);

                self.addi(ctx, arch, &sp, &sp, -8).map_err(Err::from)?;
                let push_mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: sp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &fp, &push_mem).map_err(Err::from)?;

                self.mv(ctx, arch, &fp, &sp).map_err(Err::from)?;

                let locals_slots =
                    (state.local_count as i32) + (state.control_depth as i32) * 2 + 4;
                let alloc_bytes = locals_slots * 8;
                if alloc_bytes > 0 {
                    self.addi(ctx, arch, &sp, &sp, -alloc_bytes).map_err(Err::from)?;
                }

                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);

                    let (ridx, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(riscv_regalloc::RegKind::Int)
                        .unwrap_or_else(|e| panic!("regalloc push error: {e:?}"));
                    emit_cmds(self, ctx, arch, cmds).map_err(Err::from)?;
                    let phys = Reg(ridx as u8);
                    self.li(ctx, arch, &phys, 0u64).map_err(Err::from)?;
                }

                Ok(())
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    let sp = Reg(2);
                    self.addi(ctx, arch, &sp, &sp, -8).map_err(Err::from)?;
                    let mem = MemArgKind::Mem {
                        base: ArgKind::Reg {
                            reg: sp,
                            size: MemorySize::_64,
                        },
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    };
                    let zero = ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_64,
                    };
                    self.sd(ctx, arch, &zero, &mem).map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }
            MachOperator::EndBody => Ok(()),

            MachOperator::StartBody => Ok(()),
            MachOperator::Instruction { op, .. } => {
                self.handle_op_(ctx, arch, state, func_imports, sigs, tags, op, rewriter, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op, .. } => {
                if let Some(op) = op {
                    trace!("handle_op: Operator dispatch start rewriting op");
                    let insn = rewriter.instruction(op.clone())?;
                    trace!("handle_op: Operator rewritten, calling handle_op_");
                    let r = self.handle_op_(
                        ctx,
                        arch,
                        state,
                        func_imports,
                        sigs,
                        tags,
                        &insn,
                        rewriter,
                        target,
                    ).map_err(Into::into);
                    trace!("handle_op: Operator handle_op_ returned");
                    r
                } else {
                    Ok(())
                }
            }
            other => panic!("unimplemented WASM instruction in RISC-V naive handle_op_: {other:?}"),
        }
    }
}

impl<T: Writer<RiscvLabel, Context> + ?Sized, Context> WriterExt<Context> for T {}

pub(crate) fn emit_cmds<
    E: core::error::Error,
    Context,
    W: asm_riscv::out::Writer<RiscvLabel, Context, Error = E>,
>(
    writer: &mut W,
    ctx: &mut Context,
    arch: asm_riscv::RiscV64Arch,
    mut it: impl Iterator<Item = regalloc::Cmd<riscv_regalloc::RegKind>>,
) -> Result<(), E> {
    while let Some(cmd) = it.next() {
        riscv_regalloc::process_cmd(writer, ctx, arch, &cmd)?;
    }
    Ok(())
}

/// Flush the register allocator, emitting any spill/restore instructions.
/// Call this before emitting a return epilogue in the SysV backend.
pub fn flush_regalloc<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
) -> Result<(), W::Error> {
    if let Some(ralloc) = state.regalloc.as_mut() {
        let it = ralloc.flush();
        emit_cmds(w, ctx, arch, it)?;
    }
    Ok(())
}

/// Registers the lazy register allocator currently holds a live value in (a
/// stack element or a local variable), without flushing or mutating it.
///
/// `RegAlloc`'s `frames`/`tos` fields and the `RegAllocFrame`/`Target` types
/// are public, so this reads the allocator's bookkeeping directly rather
/// than needing any new query on `portal-solutions-asm-regalloc` itself.
pub(crate) fn regalloc_occupied(
    ralloc: &regalloc::RegAlloc<riscv_regalloc::RegKind, 32, Frames<riscv_regalloc::RegKind, 32>>,
) -> Vec<regalloc::Target<riscv_regalloc::RegKind>> {
    [riscv_regalloc::RegKind::Int, riscv_regalloc::RegKind::Float]
        .into_iter()
        .flat_map(|kind| {
            ralloc.frames[kind].iter().enumerate().filter_map(move |(i, f)| match f {
                regalloc::RegAllocFrame::Stack { .. } | regalloc::RegAllocFrame::Local(_) => {
                    Some(regalloc::Target { reg: i as u8, kind })
                }
                regalloc::RegAllocFrame::Reserved | regalloc::RegAllocFrame::Empty => None,
            })
        })
        .collect()
}

/// Emit a `Call`-bound probe site in **Passive** mode.
///
/// Unlike [`WriterExt::emit_control_flow_probe`] (Active — flushes the whole
/// register allocator first, materialising the operand stack to memory, as
/// `TailTakeover` requires), this saves and restores exactly the physical
/// registers the allocator currently has occupied around the probe call,
/// leaving its `frames`/`tos` bookkeeping untouched so codegen continues
/// exactly where it left off. A no-op extra cost when no allocator is active.
///
/// Save/restore reuses the allocator's own `Cmd::Push`/`Cmd::Pop` codegen
/// (the same instructions `flush_regalloc` emits per register) — restored in
/// reverse order for correct LIFO stack discipline — so this never touches
/// `ralloc` itself: the `Vec` of occupied targets is just data, not state.
pub fn emit_passive_call_probe<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
) -> Result<(), W::Error> {
    let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) else {
        return Ok(());
    };
    let occupied = state.regalloc.as_ref().map(regalloc_occupied).unwrap_or_default();
    emit_cmds(w, ctx, arch, occupied.iter().cloned().map(regalloc::Cmd::Push))?;

    let probe_id = state.next_probe_id;
    state.next_probe_id += 1;
    let probe_base = state.probe_base;
    let mut bw = crate::codegen::BlitzW { writer: w, ctx, arch, scratch2: 6, probe_base };
    portal_solutions_blitz_codegen::emit_probe_site(
        &mut bw, cfg.table_base_off, probe_id, 5,
        portal_solutions_blitz_codegen::ProbeBinding::Call,
        &mut state.label_index,
    )?;

    emit_cmds(w, ctx, arch, occupied.iter().rev().cloned().map(regalloc::Cmd::Pop))?;
    Ok(())
}

/// Pop the top of the WASM operand stack (via regalloc) into `dest_reg`.
/// If the regalloc is active, uses it; otherwise falls back to memory pop.
pub fn pop_regalloc_to<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
    dest: portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    if let Some(ralloc) = state.regalloc.as_mut() {
        let (target, cmds) = ralloc.pop(riscv_regalloc::RegKind::Int);
        emit_cmds(w, ctx, arch, cmds)?;
        let phys = portal_solutions_blitz_common::asm::Reg(target.reg);
        if phys != dest {
            w.mv(ctx, arch, &dest, &phys)?;
        }
    } else {
        // No regalloc: value is on the memory stack.
        pop(w, ctx, arch, &dest)?;
    }
    Ok(())
}

// Lightweight helpers
pub fn emit_li<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    reg: portal_solutions_blitz_common::asm::Reg,
    val: u64,
) -> Result<(), W::Error> {
    // materialize immediate into `reg` using `li`
    w.li(ctx, arch, &reg, val)
}

pub fn push<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // decrement sp and store register
    let sp = portal_solutions_blitz_common::asm::Reg(2);
    w.addi(ctx, arch, &sp, &sp, -8)?;
    let mem = MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: sp,
            size: MemorySize::_64,
        },
        offset: None,
        disp: 0,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    };
    w.sd(ctx, arch, &r, &mem)
}

pub fn pop<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // load from sp then increment sp
    let sp = portal_solutions_blitz_common::asm::Reg(2);
    let mem = MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: sp,
            size: MemorySize::_64,
        },
        offset: None,
        disp: 0,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    };
    w.ld(ctx, arch, r, &mem)?;
    w.addi(ctx, arch, &sp, &sp, 8)
}

pub fn set_label<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    l: RiscvLabel,
) -> Result<(), W::Error> {
    w.set_label(ctx, arch, l)
}

pub fn lea_label<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
    l: RiscvLabel,
) -> Result<(), W::Error> {
    // emit address of label into r using jal + ld pattern isn't available here; instead use jal_label provided by Writer
    w.jal_label(ctx, arch, &*r, l)
}

pub fn call<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // assume r contains function address; perform jalr x1, r
    w.jalr(ctx, arch, &portal_solutions_blitz_common::asm::Reg(1), r, 0)
}

/// Emit one-instruction jump stubs for each exported function.
///
/// Each stub emits an `External` label followed by a `jal_label` with `x0`
/// (the pseudo-jump, equivalent to `j target`) to the function's internal label.
pub fn emit_export_dispatchers<W, Ctx>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: RiscV64Arch,
    exports: &[(u32, &str)],
) -> Result<(), W::Error>
where
    W: Writer<RiscvLabel, Ctx>,
{
    for (id, name) in exports {
        w.set_label(ctx, arch, RiscvLabel::External { name: (*name).into() })?;
        // jal x0, target = unconditional jump (no link)
        w.jal_label(ctx, arch, &portal_solutions_blitz_common::asm::Reg(0),
            RiscvLabel::Func { r#fn: *id })?;
    }
    Ok(())
}

pub fn ret<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
) -> Result<(), W::Error> {
    // return via jalr x0, ra (x1)
    w.jalr(
        ctx,
        arch,
        &portal_solutions_blitz_common::asm::Reg(0),
        &portal_solutions_blitz_common::asm::Reg(1),
        0,
    )
}
