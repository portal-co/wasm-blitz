//! RISC-V Linux SysV ABI code generation.
//!
//! Generates RISC-V 32-bit functions following the RISC-V psABI (ILP32 variant),
//! directly callable from C and other SysV-conforming code.
//!
//! # Calling Convention (RISC-V psABI ILP32)
//!
//! Integer arguments: A0–A7 (Reg 10–17); overflow on stack
//! Return value: A0 (single), A0+A1 (pair)
//! Callee-saved: S0 (fp), S1–S11, RA (in the prologue)
//!
//! # Design
//!
//! The SysV entry copies A0–A7 into the standard FP-relative local frame so that
//! the naive backend's arithmetic, memory, and control-flow handlers continue to
//! work unchanged.  Only `StartFn`, local variable access, and `Return` differ.

use alloc::vec::Vec;
extern crate alloc;

use portal_solutions_asm_riscv32::RiscV32Arch;
use portal_solutions_asm_riscv32::out::Writer;
use portal_solutions_asm_regalloc as regalloc;
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::{MachOperator, ProbeMode, ProbePlacement},
    wasm_encoder::{Instruction, reencode::{self as reencode, Reencode}},
};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv32::{RegisterClass, out::arg::{ArgKind, MemArgKind}};

use crate::RiscvLabel;
use crate::codegen::ProbeBase;
use crate::naive::{State, WriterExt as NaiveExt, flush_regalloc, push, pop, pop_regalloc_to, SCR};

/// Public SysV-facing state and policy names, matching the other native backends.
pub use crate::naive::{CallAbi, MemBase, State as SysVState};

/// Blitz register number of the RISC-V psABI **probe-base virtual parameter**
/// (`t2` / x7).
///
/// `t2` is caller-saved and never a positional argument register (a0–a7), nor
/// the probe-preamble scratch (t0/t1), so the runtime can pass the per-function
/// probe-table base in it.  Read directly at the function-entry site; spilled to
/// an fp-relative frame slot for mid-function (loop/block) sites.
pub const PROBE_BASE_REG: u8 = 7;

// Argument registers in RISC-V psABI order: a0–a7
const ARG_REGS: [Reg; 8] = [
    Reg(10), Reg(11), Reg(12), Reg(13), Reg(14), Reg(15), Reg(16), Reg(17),
];
// Return register
const A0: Reg = Reg(10);
const A1: Reg = Reg(11);
// Frame pointer
const FP: Reg = Reg(8);   // s0
// Stack pointer
const SP: Reg = Reg(2);
// Return address
const RA: Reg = Reg(1);
const T0: Reg = Reg(5);
const T1: Reg = Reg(6);
const T2: Reg = Reg(7);
const T3: Reg = Reg(28);

/// 8-byte WASM operand / local slot access (soft-expanded `ld`/`sd` on RV32).
fn mem64(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_32 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// 4-byte host pointer / fn-ptr table access (ILP32).
fn mem32(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_32 },
        offset: None,
        disp,
        size: MemorySize::_32,
        reg_class: RegisterClass::Gpr,
    }
}

/// Extension trait for generating RISC-V psABI-compatible functions.
pub trait SysVWriterExt<Context>: Writer<RiscvLabel, Context> + NaiveExt<Context> {

    fn sysv_call_target(
        state: &State<'_>,
        func_imports: &[(&str, &str)],
        idx: u32,
    ) -> (RiscvLabel, u32, u32, bool) {
        let wi = idx as usize;
        let is_import = idx < state.n_imports;
        let arity = state.call_params.get(wi).copied().unwrap_or(0);
        let results = state.call_results.get(wi).copied().unwrap_or(0);
        let label = if is_import {
            let (m, n) = func_imports[wi];
            RiscvLabel::External { name: alloc::format!("{m}__{n}") }
        } else {
            RiscvLabel::Indexed { idx: (idx - state.n_imports) as usize | (1 << 28) }
        };
        (label, arity, results, is_import)
    }

    /// Pop the table index and resolve `__wasm_table[index]` into t0.
    fn sysv_load_indirect_target(&mut self, ctx: &mut Context, arch: RiscV32Arch) -> Result<(), Self::Error>
    where Self: Sized {
        pop(self, ctx, arch, &T0)?;
        self.la_label(ctx, arch, &T1, RiscvLabel::External { name: "__wasm_table".into() })?;
        // ILP32 fn-ptr table: stride 4.
        self.li(ctx, arch, &T2, 2)?;
        self.sll(ctx, arch, &T0, &T0, &T2)?;
        self.add(ctx, arch, &T1, &T1, &T0)?;
        self.lw(ctx, arch, &T0, &mem32(T1, 0))
    }

    /// Marshal a normal call. Internal targets receive every parameter in an
    /// AllStack outgoing area; imports retain the RISC-V psABI register split.
    fn sysv_emit_marshalled_call(
        &mut self, ctx: &mut Context, arch: RiscV32Arch,
        target: Option<RiscvLabel>, target_reg: Option<Reg>,
        arity: u32, results: u32, is_import: bool,
    ) -> Result<(), Self::Error>
    where Self: Sized {
        self.mv(ctx, arch, &T3, &SP)?;
        let stack_args = if is_import { arity.saturating_sub(8) } else { arity };
        // One additional slot preserves the operand base across the call. Round
        // to 16 bytes so host imports retain psABI stack alignment.
        let bytes = (((stack_args + 1) * 8 + 15) & !15) as i32;
        if is_import {
            for i in 0..arity.min(8) {
                self.ld(ctx, arch, &ARG_REGS[i as usize], &mem64(T3, ((arity - 1 - i) * 8) as i32))?;
            }
        }
        self.addi(ctx, arch, &SP, &SP, -bytes)?;
        for i in (if is_import { 8 } else { 0 })..arity {
            self.ld(ctx, arch, &T1, &mem64(T3, ((arity - 1 - i) * 8) as i32))?;
            let dst = if is_import { (i - 8) * 8 } else { i * 8 };
            self.sd(ctx, arch, &T1, &mem64(SP, dst as i32))?;
        }
        self.sd(ctx, arch, &T3, &mem64(SP, (stack_args * 8) as i32))?;
        if let Some(reg) = target_reg {
            self.jalr(ctx, arch, &RA, &reg, 0)?;
        } else {
            self.jal_label(ctx, arch, &RA, target.unwrap())?;
        }
        self.ld(ctx, arch, &T3, &mem64(SP, (stack_args * 8) as i32))?;
        // Discard exactly the consumed operand args; padding is discarded too.
        self.addi(ctx, arch, &SP, &T3, (arity * 8) as i32)?;
        if results > 1 { push(self, ctx, arch, A1)?; }
        if results > 0 { push(self, ctx, arch, A0)?; }
        Ok(())
    }

    /// Emit a genuine internal AllStack tail call. The current prologue sets FP
    /// to the incoming caller SP, therefore incoming parameters start at FP+0.
    fn sysv_emit_tail_call(
        &mut self, ctx: &mut Context, arch: RiscV32Arch,
        target: Option<RiscvLabel>, target_reg: Option<Reg>, arity: u32,
        frame_sz: i32, shard: bool,
    ) -> Result<(), Self::Error>
    where Self: Sized {
        self.mv(ctx, arch, &T3, &SP)?;
        for i in 0..arity {
            self.ld(ctx, arch, &T1, &mem64(T3, ((arity - 1 - i) * 8) as i32))?;
            self.sd(ctx, arch, &T1, &mem64(FP, (i * 8) as i32))?;
        }
        // Preserve the incoming SP (our FP) so the callee's prologue sees args
        // at [SP+i*8]. Restore RA/old-FP from our frame, then SP := incoming.
        self.mv(ctx, arch, &T2, &FP)?;
        self.addi(ctx, arch, &SP, &FP, -frame_sz)?;
        self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
        self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
        if shard {
            self.lw(ctx, arch, &SCR, &mem32(SP, 24))?;
        }
        self.mv(ctx, arch, &SP, &T2)?;
        if let Some(reg) = target_reg {
            self.jalr(ctx, arch, &Reg(0), &reg, 0)
        } else {
            self.jal_label(ctx, arch, &Reg(0), target.unwrap())
        }
    }

    /// Load local N from the FP-relative frame into `dest`.
    fn sysv_load_local(&mut self, ctx: &mut Context, arch: RiscV32Arch, dest: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        // Frame layout (from FP downward):
        //   [FP-8]  = local 0    ← same as naive, regalloc-compatible
        //   [FP-16] = local 1
        //   ...
        //   [SP+8]  = saved old-FP  ← bottom of frame
        //   [SP+0]  = saved RA
        let disp = -((n as i32 + 1) * 8);
        self.ld(ctx, arch, &dest, &mem64(FP, disp))
    }

    /// Store `src` into local N in the FP-relative frame.
    fn sysv_store_local(&mut self, ctx: &mut Context, arch: RiscV32Arch, src: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.sd(ctx, arch, &src, &mem64(FP, disp))
    }

    /// Handle an instruction with RISC-V SysV local/return semantics.
    fn sysv_handle_insn<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV32Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        match op {
            Instruction::Call(idx) if state.call_abi == CallAbi::AllStack => {
                flush_regalloc(self, ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                self.sysv_emit_marshalled_call(ctx, arch, Some(label), None, arity, results, is_import)
            }
            Instruction::ReturnCall(idx) if state.call_abi == CallAbi::AllStack => {
                flush_regalloc(self, ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                if is_import || arity as usize > state.param_count {
                    // Host-ABI import / oversized arity: marshalled call then
                    // epilogue (not nested return_call→call→return for internals).
                    self.sysv_emit_marshalled_call(
                        ctx, arch, Some(label), None, arity, results, is_import,
                    )?;
                    self.sysv_handle_insn(
                        ctx, arch, state, func_imports, &Instruction::Return, rewriter, target,
                    )
                } else {
                    self.sysv_emit_tail_call(
                        ctx, arch, Some(label), None, arity, state.sysv_frame_sz, state.shard.is_some(),
                    )
                }
            }
            Instruction::CallIndirect { type_index, .. } if state.call_abi == CallAbi::AllStack => {
                flush_regalloc(self, ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.sysv_load_indirect_target(ctx, arch)?;
                self.sysv_emit_marshalled_call(ctx, arch, None, Some(T0), arity, results, false)
            }
            Instruction::ReturnCallIndirect { type_index, .. } if state.call_abi == CallAbi::AllStack => {
                flush_regalloc(self, ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                assert!(arity as usize <= state.param_count, "tail callee arity exceeds incoming AllStack area");
                self.sysv_load_indirect_target(ctx, arch)?;
                self.sysv_emit_tail_call(
                    ctx, arch, None, Some(T0), arity, state.sysv_frame_sz, state.shard.is_some(),
                )
            }
            // Local access: delegate to naive which uses regalloc with [FP-(n+1)*8] offsets.
            // SysV frame places locals at the same offsets, so this is correct.
            Instruction::LocalGet(_) | Instruction::LocalSet(_) | Instruction::LocalTee(_) => {
                self.handle_op_(ctx, arch, state, func_imports, &[], &[], op, rewriter, target)
            }

            // Return: always emit RISC-V SysV epilogue.
            // flush_regalloc spills any register-held values to the memory stack,
            // after which they are readable with the normal memory pop.
            //
            // WASM multi-value results are held in *declaration* order:
            // result 0 first/deepest, result n-1 last/top (see the WASM
            // spec, and the matching fix in blitz-aarch64/blitz-x86-64's
            // `sysv_emit_epilogue`). The RISC-V psABI can carry at most two
            // results back through registers (A0, A1); pop-and-drop any
            // results beyond index 1 first (there's no register for them),
            // so the *last* two pops land (result 1, result 0) into
            // (A1, A0) — not the two most-recently-pushed values.
            //
            // NOTE: unlike blitz-aarch64/blitz-x86-64, this fix is
            // currently untested — speet's `drive.rs` doesn't wire RISC-V
            // through wasm-blitz's native codegen yet (only through the
            // WASM-interpreter path, which is unaffected by this bug: an
            // interpreter reads all `n` results directly regardless of
            // this backend's 2-register cap). Add a native RISC-V sysv
            // test exercising `num_returns > 1` before relying on this in
            // a real native RISC-V build.
            Instruction::Return => {
                // Use pop_regalloc_to: pops from regalloc (if active) or memory stack.
                let scratch = Reg(6); // t1: caller-saved, not PROBE_BASE_REG (t2/Reg(7))
                for _ in 0..state.num_returns.saturating_sub(2) {
                    pop_regalloc_to(self, ctx, arch, state, scratch)?;
                }
                if state.num_returns > 1 {
                    pop_regalloc_to(self, ctx, arch, state, A1)?;
                }
                if state.num_returns > 0 {
                    pop_regalloc_to(self, ctx, arch, state, A0)?;
                }
                flush_regalloc(self, ctx, arch, state)?;
                // Restore RA/old-FP from the frame, then SP := incoming SP (FP).
                self.mv(ctx, arch, &T2, &FP)?;
                self.addi(ctx, arch, &SP, &FP, -state.sysv_frame_sz)?;
                self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
                self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
                if state.shard.is_some() {
                    self.lw(ctx, arch, &SCR, &mem32(SP, 24))?;
                }
                self.mv(ctx, arch, &SP, &T2)?;
                self.jalr(ctx, arch, &Reg(0), &RA, 0)
            }
            // Function-level End (empty if_stack) acts as implicit return.
            // See the matching multi-value-order comment on `Return` above.
            Instruction::End if state.if_stack.is_empty() => {
                flush_regalloc(self, ctx, arch, state)?;
                let scratch = Reg(6); // t1
                for _ in 0..state.num_returns.saturating_sub(2) {
                    pop(self, ctx, arch, &scratch)?;
                }
                if state.num_returns > 1 {
                    pop(self, ctx, arch, &A1)?;
                }
                if state.num_returns > 0 {
                    pop(self, ctx, arch, &A0)?;
                }
                self.mv(ctx, arch, &T2, &FP)?;
                self.addi(ctx, arch, &SP, &FP, -state.sysv_frame_sz)?;
                self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
                self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
                if state.shard.is_some() {
                    self.lw(ctx, arch, &SCR, &mem32(SP, 24))?;
                }
                self.mv(ctx, arch, &SP, &T2)?;
                self.jalr(ctx, arch, &Reg(0), &RA, 0)
            }

            // Everything else: naive handler
            other => self.handle_op_(ctx, arch, state, func_imports, &[], &[], other, rewriter, target),
        }
    }

    /// Handle a `MachOperator` using the RISC-V SysV ABI.
    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV32Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.param_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                // Allocation state is per-function. Keeping a prior function's
                // local/register mapping corrupts the next AllStack callee.
                state.regalloc = None;
                state.probes = data.probes;
                state.next_probe_id = 1; // probe 0 is the function entry below
                state.probe_plan = data.probe_plan.clone();
                state.op_index = 0;

                self.set_label(ctx, arch, RiscvLabel::Indexed {
                    idx: *id as usize | (1 << 28),
                }).map_err(Err::from)?;

                // Function-entry probe (probe 0): probe-table base arrives in the
                // virtual-param register t2; read it directly before frame setup
                // so a0–a7 are intact for the tail-jump.  Scratch t0/t1.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW {
                        writer: self, ctx, arch, scratch2: 6,
                        probe_base: ProbeBase::Reg(PROBE_BASE_REG),
                    };
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw, cfg.table_base_off, 0, 5,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                // Frame layout (from FP downward):
                //   [FP-(1*8)]  = local 0   ← same offsets as naive, regalloc-compatible
                //   ...
                //   [SP+24] = saved SCR (s10)    ← extra slot when sharding active
                //   [SP+16] = spilled probe base  ← extra slot for mid-function sites
                //   [SP+8]  = saved old-FP   ← bottom two slots for callee-saves
                //   [SP+0]  = saved RA
                let shard_extra = if state.shard.is_some() { 1 } else { 0 };
                let frame_slots = data.num_params + 2 + state.control_depth * 2 + 3 + shard_extra;
                let frame_sz = (frame_slots * 8) as i32;
                state.sysv_frame_sz = frame_sz;
                self.addi(ctx, arch, &SP, &SP, -frame_sz).map_err(Err::from)?;
                self.sd(ctx, arch, &RA, &mem64(SP, 0)).map_err(Err::from)?;
                self.sd(ctx, arch, &FP, &mem64(SP, 8)).map_err(Err::from)?;
                if state.shard.is_some() {
                    // SCR holds a host pointer (ILP32).
                    self.sw(ctx, arch, &SCR, &mem32(SP, 24)).map_err(Err::from)?;
                }
                self.addi(ctx, arch, &FP, &SP, frame_sz).map_err(Err::from)?;

                // Spill the virtual-param base (t2) to the [SP+16] slot and point
                // mid-function sites at it (fp-relative). Host pointer: ILP32 `sw`.
                if data.probes.as_ref().map(|c| c.enabled).unwrap_or(false) {
                    let disp = -frame_sz + 16;
                    state.probe_base = ProbeBase::FrameSlot(disp);
                    self.sw(ctx, arch, &Reg(PROBE_BASE_REG), &mem32(FP, disp)).map_err(Err::from)?;
                }

                match state.call_abi {
                    CallAbi::RegSysv => {
                        for i in 0..data.num_params.min(8) {
                            self.sysv_store_local(ctx, arch, ARG_REGS[i], i).map_err(Err::from)?;
                        }
                        // psABI overflow arguments begin at caller SP, which is FP
                        // after this backend's prologue.
                        for i in 8..data.num_params {
                            self.ld(ctx, arch, &T0, &mem64(FP, ((i - 8) * 8) as i32)).map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, T0, i).map_err(Err::from)?;
                        }
                    }
                    CallAbi::AllStack => {
                        for i in 0..data.num_params {
                            self.ld(ctx, arch, &T0, &mem64(FP, (i * 8) as i32)).map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, T0, i).map_err(Err::from)?;
                        }
                    }
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.li(ctx, arch, &A0, 0).map_err(Err::from)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, A0, state.local_count).map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }

            MachOperator::StartBody | MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before).map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, rewriter, target).map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After).map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before).map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, rewriter, target).map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After).map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }

    /// Emit any embedder-requested probes (`state.probe_plan`) attached to
    /// the instruction at `state.op_index` with the given `placement`.
    ///
    /// Unlike x86-64/AArch64, RISC-V has a real lazy register allocator, so
    /// `ProbeSpec::mode` matters here: `Active` flushes it first (same
    /// sequence `flush_regalloc`/`emit_control_flow_probe` use); `Passive`
    /// saves/restores exactly the registers it currently has occupied
    /// (`crate::naive::regalloc_occupied`) around the probe instead, leaving
    /// its `frames`/`tos` bookkeeping untouched. Reuses the mid-function
    /// probe-base addressing (`ProbeBase::FrameSlot`, already live in
    /// `state.probe_base` once `StartFn` has run) and the t0/t1 scratch
    /// registers the function-entry probe above already establishes.
    fn sysv_emit_indexed_probes(
        &mut self,
        ctx: &mut Context,
        arch: RiscV32Arch,
        state: &mut State<'_>,
        placement: ProbePlacement,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let (Some(plan), Some(cfg)) = (state.probe_plan.as_ref(), state.probes) else {
            return Ok(());
        };
        if !cfg.enabled {
            return Ok(());
        }
        let specs: Vec<_> = plan.at(state.op_index, placement).copied().collect();
        for spec in specs {
            let passive_occupied = match spec.mode {
                ProbeMode::Active => {
                    if let Some(ralloc) = state.regalloc.as_mut() {
                        let it = ralloc.flush();
                        crate::naive::emit_cmds(self, ctx, arch, it)?;
                        ralloc.tos = None;
                    }
                    None
                }
                ProbeMode::Passive => {
                    let occ = state.regalloc.as_ref().map(crate::naive::regalloc_occupied).unwrap_or_default();
                    crate::naive::emit_cmds(self, ctx, arch, occ.iter().cloned().map(regalloc::Cmd::Push))?;
                    Some(occ)
                }
            };
            let probe_base = state.probe_base;
            let mut bw = crate::codegen::BlitzW {
                writer: self, ctx, arch, scratch2: 6, probe_base,
            };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw, cfg.table_base_off, spec.probe_id, 5, spec.binding, &mut state.label_index,
            )?;
            if let Some(occ) = passive_occupied {
                crate::naive::emit_cmds(self, ctx, arch, occ.iter().rev().cloned().map(regalloc::Cmd::Pop))?;
            }
        }
        Ok(())
    }
}

impl<T: Writer<RiscvLabel, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}
