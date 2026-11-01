//! RISC-V Linux SysV ABI code generation.
//!
//! Generates RISC-V 64-bit functions following the RISC-V psABI (LP64 variant),
//! directly callable from C and other SysV-conforming code.
//!
//! # Calling Convention (RISC-V psABI LP64)
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

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_regalloc as regalloc;
use portal_solutions_asm_riscv64::RiscV64Arch;
use portal_solutions_asm_riscv64::out::Writer;
use portal_solutions_asm_riscv64::{
    RegisterClass,
    out::arg::{ArgKind, MemArgKind},
};
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::{MachOperator, ProbeMode, ProbePlacement},
    wasm_encoder::{
        Instruction,
        reencode::{self as reencode, Reencode},
    },
};

use crate::RiscvLabel;
use crate::codegen::ProbeBase;
use crate::naive::{SCR, State, WriterExt as NaiveExt, flush_regalloc, pop, push};

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
    Reg(10),
    Reg(11),
    Reg(12),
    Reg(13),
    Reg(14),
    Reg(15),
    Reg(16),
    Reg(17),
];
// Return register
const A0: Reg = Reg(10);
const A1: Reg = Reg(11);
// Frame pointer
const FP: Reg = Reg(8); // s0
// Stack pointer
const SP: Reg = Reg(2);
// Return address
const RA: Reg = Reg(1);
const T0: Reg = Reg(5);
const T1: Reg = Reg(6);
const T2: Reg = Reg(7);
const T3: Reg = Reg(28);

fn mem64(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
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
            RiscvLabel::External {
                name: alloc::format!("{m}__{n}"),
            }
        } else {
            RiscvLabel::Indexed {
                idx: (idx - state.n_imports) as usize | (1 << 28),
            }
        };
        (label, arity, results, is_import)
    }

    /// Pop the table index and resolve `__wasm_table[index]` into t0.
    fn sysv_load_indirect_target(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        pop(self, ctx, arch, &T0)?;
        self.la_label(
            ctx,
            arch,
            &T1,
            RiscvLabel::External {
                name: "__wasm_table".into(),
            },
        )?;
        self.li(ctx, arch, &T2, 3)?;
        self.sll(ctx, arch, &T0, &T0, &T2)?;
        self.add(ctx, arch, &T1, &T1, &T0)?;
        self.ld(ctx, arch, &T0, &mem64(T1, 0))
    }

    /// Marshal a normal call. Internal targets receive every parameter in an
    /// AllStack outgoing area; imports retain the RISC-V psABI register split.
    fn sysv_emit_marshalled_call(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        target: Option<RiscvLabel>,
        target_reg: Option<Reg>,
        arity: u32,
        results: u32,
        is_import: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.mv(ctx, arch, &T3, &SP)?;
        let slots = if is_import {
            arity
        } else {
            // Internal AllStack: the callee's epilogue writes its results back
            // into the incoming-args area, so size for max(arity, results).
            arity.max(results)
        };
        // One additional slot preserves the operand base across the call. Round
        // to 16 bytes so host imports retain psABI stack alignment.
        let bytes = (((slots + 1) * 8 + 15) & !15) as i32;
        if is_import {
            for i in 0..arity.min(8) {
                self.ld(
                    ctx,
                    arch,
                    &ARG_REGS[i as usize],
                    &mem64(T3, ((arity - 1 - i) * 8) as i32),
                )?;
            }
        }
        self.addi(ctx, arch, &SP, &SP, -bytes)?;
        for i in (if is_import { 8 } else { 0 })..arity {
            self.ld(ctx, arch, &T1, &mem64(T3, ((arity - 1 - i) * 8) as i32))?;
            let dst = if is_import { (i - 8) * 8 } else { i * 8 };
            self.sd(ctx, arch, &T1, &mem64(SP, dst as i32))?;
        }
        self.sd(ctx, arch, &T3, &mem64(SP, (slots * 8) as i32))?;
        if let Some(reg) = target_reg {
            self.jalr(ctx, arch, &RA, &reg, 0)?;
        } else {
            // JAL is only ±1MB; a megabinary's functions can be farther
            // apart. `la` (AUIPC+ADDI, ±2GB) + jalr reaches any target.
            self.la_label(ctx, arch, &T1, target.unwrap())?;
            self.jalr(ctx, arch, &RA, &T1, 0)?;
        }
        self.ld(ctx, arch, &T3, &mem64(SP, (slots * 8) as i32))?;
        if !is_import && results > 0 && results <= arity {
            // The callee left result i at [SP + i*8]. Copy result i to its
            // final operand-stack slot [T3 + (arity-1-i)*8]; source and
            // destination are disjoint when results <= arity. Result 0 lands
            // deepest, result n-1 on top.
            for i in 0..results {
                self.ld(ctx, arch, &T1, &mem64(SP, (i * 8) as i32))?;
                self.sd(ctx, arch, &T1, &mem64(T3, ((arity - 1 - i) * 8) as i32))?;
            }
            // Discard args, keep results: net SP change is (arity - results)*8.
            self.addi(ctx, arch, &SP, &T3, ((arity - results) * 8) as i32)?;
        } else {
            // Legacy register path (imports, or results > arity which cannot
            // be recovered from the outgoing area without an overlap).
            self.addi(ctx, arch, &SP, &T3, (arity * 8) as i32)?;
            if results > 1 {
                push(self, ctx, arch, A1)?;
            }
            if results > 0 {
                push(self, ctx, arch, A0)?;
            }
        }
        Ok(())
    }

    /// Emit a genuine internal AllStack tail call. The current prologue sets FP
    /// to the incoming caller SP, therefore incoming parameters start at FP+0.
    fn sysv_emit_tail_call(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        target: Option<RiscvLabel>,
        target_reg: Option<Reg>,
        arity: u32,
        frame_sz: i32,
        shard: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
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
            self.ld(ctx, arch, &SCR, &mem64(SP, 24))?;
        }
        self.mv(ctx, arch, &SP, &T2)?;
        if let Some(reg) = target_reg {
            self.jalr(ctx, arch, &Reg(0), &reg, 0)
        } else {
            // Same far-range concern as the call path: la + jalr (±2GB).
            self.la_label(ctx, arch, &T1, target.unwrap())?;
            self.jalr(ctx, arch, &Reg(0), &T1, 0)
        }
    }

    /// Load local N from the FP-relative frame into `dest`.
    fn sysv_load_local(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        dest: Reg,
        n: usize,
    ) -> Result<(), Self::Error> {
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
    fn sysv_store_local(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        src: Reg,
        n: usize,
    ) -> Result<(), Self::Error> {
        let disp = -((n as i32 + 1) * 8);
        self.sd(ctx, arch, &src, &mem64(FP, disp))
    }

    /// Handle an instruction with RISC-V SysV local/return semantics.
    fn sysv_handle_insn<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
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
                self.sysv_emit_marshalled_call(
                    ctx,
                    arch,
                    Some(label),
                    None,
                    arity,
                    results,
                    is_import,
                )
            }
            Instruction::ReturnCall(idx) if state.call_abi == CallAbi::AllStack => {
                flush_regalloc(self, ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                if is_import || arity as usize > state.param_count {
                    // Host-ABI import / oversized arity: marshalled call then
                    // epilogue (not nested return_call→call→return for internals).
                    self.sysv_emit_marshalled_call(
                        ctx,
                        arch,
                        Some(label),
                        None,
                        arity,
                        results,
                        is_import,
                    )?;
                    self.sysv_handle_insn(
                        ctx,
                        arch,
                        state,
                        func_imports,
                        &Instruction::Return,
                        rewriter,
                        target,
                    )
                } else {
                    self.sysv_emit_tail_call(
                        ctx,
                        arch,
                        Some(label),
                        None,
                        arity,
                        state.sysv_frame_sz,
                        state.shard.is_some(),
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
            Instruction::ReturnCallIndirect { type_index, .. }
                if state.call_abi == CallAbi::AllStack =>
            {
                flush_regalloc(self, ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                assert!(
                    arity as usize <= state.param_count,
                    "tail callee arity exceeds incoming AllStack area"
                );
                self.sysv_load_indirect_target(ctx, arch)?;
                self.sysv_emit_tail_call(
                    ctx,
                    arch,
                    None,
                    Some(T0),
                    arity,
                    state.sysv_frame_sz,
                    state.shard.is_some(),
                )
            }
            // Local access: delegate to naive which uses regalloc with [FP-(n+1)*8] offsets.
            // SysV frame places locals at the same offsets, so this is correct.
            Instruction::LocalGet(_) | Instruction::LocalSet(_) | Instruction::LocalTee(_) => self
                .handle_op_(
                    ctx,
                    arch,
                    state,
                    func_imports,
                    &[],
                    &[],
                    op,
                    rewriter,
                    target,
                ),

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
                // Flush first so every result is on the memory stack; the
                // AllStack result-return copy below reads them by SP offset.
                flush_regalloc(self, ctx, arch, state)?;
                // AllStack: copy *all* results into the incoming-args area
                // ([FP + i*8], dead by now) so the caller can recover them
                // onto its operand stack. The caller sizes its outgoing area
                // for max(arity, results) slots (see sysv_emit_marshalled_call).
                if matches!(state.call_abi, CallAbi::AllStack) && state.num_returns > 0 {
                    let n = state.num_returns;
                    let scratch = Reg(6); // t1
                    for i in 0..n {
                        let src = ((n - 1 - i) * 8) as i32;
                        self.ld(ctx, arch, &scratch, &mem64(SP, src))?;
                        self.sd(ctx, arch, &scratch, &mem64(FP, (i * 8) as i32))?;
                    }
                }
                // Use pop_regalloc_to: pops from regalloc (if active) or memory stack.
                let scratch = Reg(6); // t1: caller-saved, not PROBE_BASE_REG (t2/Reg(7))
                for _ in 0..state.num_returns.saturating_sub(2) {
                    pop(self, ctx, arch, &scratch)?;
                }
                if state.num_returns > 1 {
                    pop(self, ctx, arch, &A1)?;
                }
                if state.num_returns > 0 {
                    pop(self, ctx, arch, &A0)?;
                }
                // Restore RA/old-FP from the frame, then SP := incoming SP (FP).
                self.mv(ctx, arch, &T2, &FP)?;
                self.addi(ctx, arch, &SP, &FP, -state.sysv_frame_sz)?;
                self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
                self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
                if state.shard.is_some() {
                    self.ld(ctx, arch, &SCR, &mem64(SP, 24))?;
                }
                self.mv(ctx, arch, &SP, &T2)?;
                self.jalr(ctx, arch, &Reg(0), &RA, 0)
            }
            // Function-level End (empty if_stack) acts as implicit return.
            // See the matching multi-value-order comment on `Return` above.
            Instruction::End if state.if_stack.is_empty() => {
                flush_regalloc(self, ctx, arch, state)?;
                // AllStack: copy *all* results into the incoming-args area
                // (see the matching note on `Return` above).
                if matches!(state.call_abi, CallAbi::AllStack) && state.num_returns > 0 {
                    let n = state.num_returns;
                    let scratch = Reg(6); // t1
                    for i in 0..n {
                        let src = ((n - 1 - i) * 8) as i32;
                        self.ld(ctx, arch, &scratch, &mem64(SP, src))?;
                        self.sd(ctx, arch, &scratch, &mem64(FP, (i * 8) as i32))?;
                    }
                }
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
                    self.ld(ctx, arch, &SCR, &mem64(SP, 24))?;
                }
                self.mv(ctx, arch, &SP, &T2)?;
                self.jalr(ctx, arch, &Reg(0), &RA, 0)
            }

            // Everything else: naive handler
            other => self.handle_op_(
                ctx,
                arch,
                state,
                func_imports,
                &[],
                &[],
                other,
                rewriter,
                target,
            ),
        }
    }

    /// Handle a `MachOperator` using the RISC-V SysV ABI.
    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
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

                self.set_label(
                    ctx,
                    arch,
                    RiscvLabel::Indexed {
                        idx: *id as usize | (1 << 28),
                    },
                )
                .map_err(Err::from)?;

                // Function-entry probe (probe 0): probe-table base arrives in the
                // virtual-param register t2; read it directly before frame setup
                // so a0–a7 are intact for the tail-jump.  Scratch t0/t1.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW {
                        writer: self,
                        ctx,
                        arch,
                        scratch2: 6,
                        probe_base: ProbeBase::Reg(PROBE_BASE_REG),
                    };
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw,
                        cfg.table_base_off,
                        0,
                        5,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    )
                    .map_err(Err::from)?;
                }

                // Frame layout (from FP downward):
                //   [FP-(1*8)]  = local 0   ← same offsets as naive, regalloc-compatible
                //   ...
                //   [SP+24] = saved SCR (s10)    ← extra slot when sharding active
                //   [SP+16] = spilled probe base  ← extra slot for mid-function sites
                //   [SP+8]  = saved old-FP   ← bottom two slots for callee-saves
                //   [SP+0]  = saved RA
                let shard_extra = if state.shard.is_some() { 1 } else { 0 };
                let frame_slots = data.num_params
                    + data.num_locals
                    + 2
                    + state.control_depth * 2
                    + 3
                    + shard_extra;
                let frame_sz = (frame_slots * 8) as i32;
                state.sysv_frame_sz = frame_sz;
                self.addi(ctx, arch, &SP, &SP, -frame_sz)
                    .map_err(Err::from)?;
                self.sd(ctx, arch, &RA, &mem64(SP, 0)).map_err(Err::from)?;
                self.sd(ctx, arch, &FP, &mem64(SP, 8)).map_err(Err::from)?;
                if state.shard.is_some() {
                    self.sd(ctx, arch, &SCR, &mem64(SP, 24))
                        .map_err(Err::from)?;
                }
                self.addi(ctx, arch, &FP, &SP, frame_sz)
                    .map_err(Err::from)?;

                // Spill the virtual-param base (t2) to the [SP+16] slot and point
                // mid-function sites at it (fp-relative).
                if data.probes.as_ref().map(|c| c.enabled).unwrap_or(false) {
                    let disp = -frame_sz + 16;
                    state.probe_base = ProbeBase::FrameSlot(disp);
                    self.sd(ctx, arch, &Reg(PROBE_BASE_REG), &mem64(FP, disp))
                        .map_err(Err::from)?;
                }

                match state.call_abi {
                    CallAbi::RegSysv => {
                        for i in 0..data.num_params.min(8) {
                            self.sysv_store_local(ctx, arch, ARG_REGS[i], i)
                                .map_err(Err::from)?;
                        }
                        // psABI overflow arguments begin at caller SP, which is FP
                        // after this backend's prologue.
                        for i in 8..data.num_params {
                            self.ld(ctx, arch, &T0, &mem64(FP, ((i - 8) * 8) as i32))
                                .map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, T0, i).map_err(Err::from)?;
                        }
                    }
                    CallAbi::AllStack => {
                        for i in 0..data.num_params {
                            self.ld(ctx, arch, &T0, &mem64(FP, (i * 8) as i32))
                                .map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, T0, i).map_err(Err::from)?;
                        }
                    }
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.li(ctx, arch, &A0, 0).map_err(Err::from)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, A0, state.local_count)
                        .map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }

            MachOperator::StartBody | MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before)
                    .map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, rewriter, target)
                    .map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After)
                    .map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator {
                op: Some(op_wasm), ..
            } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before)
                    .map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, rewriter, target)
                    .map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After)
                    .map_err(Err::from)?;
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
        arch: RiscV64Arch,
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
                    let occ = state
                        .regalloc
                        .as_ref()
                        .map(crate::naive::regalloc_occupied)
                        .unwrap_or_default();
                    crate::naive::emit_cmds(
                        self,
                        ctx,
                        arch,
                        occ.iter().cloned().map(regalloc::Cmd::Push),
                    )?;
                    Some(occ)
                }
            };
            let probe_base = state.probe_base;
            let mut bw = crate::codegen::BlitzW {
                writer: self,
                ctx,
                arch,
                scratch2: 6,
                probe_base,
            };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                spec.probe_id,
                5,
                spec.binding,
                &mut state.label_index,
            )?;
            if let Some(occ) = passive_occupied {
                crate::naive::emit_cmds(
                    self,
                    ctx,
                    arch,
                    occ.iter().rev().cloned().map(regalloc::Cmd::Pop),
                )?;
            }
        }
        Ok(())
    }
}

impl<T: Writer<RiscvLabel, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context>
    for T
{
}

// ---------------------------------------------------------------------------
// Code sharing (v1: SysV + rewindable binary writer, disabled by default)
// Parity with blitz-x86-64's and blitz-aarch64's `sysv.rs`;
// see docs/generic-code-sharing-plan.md.
// ---------------------------------------------------------------------------

use portal_solutions_asm_writer::RewindableCodeWriter;
use portal_solutions_blitz_codegen::code_sharing::{CodeSharingConfig, FunctionStaging};

/// WASM operand-stack delta for region membership: only regalloc-covered,
/// straight-line, label-free integer/local ops are eligible inside a v1
/// candidate region (see `docs/generic-code-sharing-plan.md` eligibility).
fn sysv_share_region_delta(insn: &Instruction<'_>) -> Option<i32> {
    use Instruction as I;
    match insn {
        I::I32Const(_) | I::I64Const(_) | I::LocalGet(_) => Some(1),
        I::LocalSet(_) => Some(-1),
        I::I32Add
        | I::I64Add
        | I::I32Sub
        | I::I64Sub
        | I::I32And
        | I::I64And
        | I::I32Or
        | I::I64Or
        | I::I32Xor
        | I::I64Xor
        | I::I32Mul
        | I::I64Mul
        | I::I32Shl
        | I::I64Shl
        | I::I32ShrU
        | I::I64ShrU
        | I::I32ShrS
        | I::I64ShrS => Some(-1),
        _ => None,
    }
}

/// Statically known operand-stack deltas used only for tracking (a superset of
/// region eligibility). `None` means unknown: depth tracking stops until the
/// next function boundary, conservatively disabling further candidates in this
/// function.
fn sysv_share_track_delta(insn: &Instruction<'_>) -> Option<i32> {
    use Instruction as I;
    if let Some(d) = sysv_share_region_delta(insn) {
        return Some(d);
    }
    match insn {
        I::LocalTee(_) | I::Nop => Some(0),
        I::F32Const(_) | I::F64Const(_) | I::GlobalGet(_) | I::MemorySize(_) => Some(1),
        I::GlobalSet(_) | I::Drop => Some(-1),
        I::Select => Some(-2),
        I::MemoryGrow(_) => Some(0),
        I::MemoryFill(_) => Some(-3),
        I::MemoryCopy { .. } => Some(-3),
        _ => None,
    }
}

/// Per-compilation state for the RV64 SysV code-sharing collector.
///
/// Disabled by default ([`CodeSharingConfig::default`]); with the default
/// config, [`SysVSharingWriterExt::sysv_handle_op_sharing`] is a pass-through
/// to [`SysVWriterExt::sysv_handle_op`] and output is byte-identical.
pub struct Rv64Sharing<B> {
    /// Per-function staging collector (recreated at each function end).
    pub staging: FunctionStaging<RiscvLabel, B>,
    /// WASM operand-stack depth when statically known; candidates may open
    /// only at `Some(0)`. `pub` for WasmSink wrapper lifecycle resets.
    pub depth: Option<i32>,
    /// Anchor label index of the currently open candidate region.
    pub open_anchor: Option<usize>,
    /// Next helper-label ordinal (see `codegen::RV64_HELPER_LABEL_BASE`).
    pub next_helper: usize,
    /// The active sharing policy (`pub` for wrapper constructors).
    pub config: CodeSharingConfig,
}

impl<B> Rv64Sharing<B> {
    pub fn new(config: CodeSharingConfig) -> Self {
        Self {
            staging: FunctionStaging::new(config, crate::codegen::RV64_SYSV_OUTLINE_COST),
            depth: Some(0),
            open_anchor: None,
            next_helper: 0,
            config,
        }
    }
    /// Enabled v1 configuration with the default size thresholds.
    pub fn enabled() -> Self {
        Self::new(CodeSharingConfig {
            enabled: true,
            ..CodeSharingConfig::default()
        })
    }
}

impl<B> Default for Rv64Sharing<B> {
    fn default() -> Self {
        Self::new(CodeSharingConfig::default())
    }
}

/// Code-sharing entry point parallel to [`SysVWriterExt::sysv_handle_op`].
///
/// Mirrors the x86-64/AArch64 v1 vertical slice: around maximal runs of
/// eligible straight-line ops that start and end with an empty WASM operand
/// stack, the region is anchored, emitted normally, and staged as an opaque
/// blob. At the function-level `End` (or the next `StartFn`), profitable
/// repeated classes are replaced by `jal ra` to one out-of-line helper emitted
/// after the epilogue; singletons and unprofitable classes are restored inline.
pub trait SysVSharingWriterExt<Context>: SysVWriterExt<Context> {
    fn sysv_handle_op_sharing<E, Err, B>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        sharing: &mut Rv64Sharing<B>,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<<Self as portal_solutions_asm_riscv64::out::WriterCore<Context>>::Error>
            + From<reencode::Error<E>>
            + From<<Self as RewindableCodeWriter<RiscvLabel>>::Error>,
        Self: Sized + RewindableCodeWriter<RiscvLabel, Blob = B>,
    {
        // Function boundary: reset per-function staging. A non-empty staging
        // here means the previous body lacked a function-level End; finalize
        // anyway so staged blobs are never dropped with their code missing.
        if matches!(op, MachOperator::StartFn { .. }) {
            sharing.depth = Some(0);
            sharing.open_anchor = None;
            if sharing.staging.staged_count() > 0 {
                let staging = core::mem::replace(
                    &mut sharing.staging,
                    FunctionStaging::new(sharing.config, crate::codegen::RV64_SYSV_OUTLINE_COST),
                );
                let mut sw = crate::codegen::SharingW {
                    writer: &mut *self,
                    ctx: &mut *ctx,
                    arch,
                    next_helper: sharing.next_helper,
                };
                staging.finalize::<_, _, _, Err>(&mut sw)?;
                sharing.next_helper = sw.next_helper;
            }
            return self.sysv_handle_op::<E, Err>(
                ctx,
                arch,
                state,
                func_imports,
                op,
                rewriter,
                target,
            );
        }
        if !sharing.config.enabled {
            return self.sysv_handle_op::<E, Err>(
                ctx,
                arch,
                state,
                func_imports,
                op,
                rewriter,
                target,
            );
        }

        // Normalize raw wasmparser operators (the WasmSink path) through the
        // rewriter so classification sees the same instruction either way.
        let owned_insn;
        let insn_opt: Option<&Instruction<'_>> = match op {
            MachOperator::Instruction { op: insn, .. } => Some(insn),
            MachOperator::Operator { op: Some(o), .. } => {
                owned_insn = rewriter.instruction(o.clone())?;
                Some(&owned_insn)
            }
            _ => None,
        };
        // Instrumentation is dynamic context outside the v1 contract.
        let probes_active =
            state.probes.as_ref().is_some_and(|c| c.enabled) || state.probe_plan.is_some();
        let region_delta = if probes_active {
            None
        } else {
            insn_opt.and_then(sysv_share_region_delta)
        };
        let is_function_end =
            matches!(insn_opt, Some(Instruction::End)) && state.if_stack.is_empty();

        // Close an open candidate at any op that cannot join the region.
        if region_delta.is_none()
            && let Some(anchor) = sharing.open_anchor.take()
            && sharing.depth == Some(0)
        {
            flush_regalloc(self, ctx, arch, state).map_err(Err::from)?;
            sharing
                .staging
                .stage_candidate(
                    &mut *self,
                    RiscvLabel::Indexed { idx: anchor },
                    crate::codegen::RV64_SYSV_SHARE_CONTRACT,
                )
                .map_err(Err::from)?;
            // Canonical register state: after the flush everything is
            // spilled, so resetting the allocator discards no values
            // and makes the next region's register choices identical
            // to the first occurrence's.
            state.regalloc = None;
        }

        // Open a candidate at an eligible op with a provably empty WASM stack.
        // The body-switch guard in naive's handler emits a jump when
        // `target != state.body`; never open across that boundary.
        if sharing.open_anchor.is_none()
            && region_delta.is_some()
            && sharing.depth == Some(0)
            && target == state.body
        {
            flush_regalloc(self, ctx, arch, state).map_err(Err::from)?;
            // Canonical register state at the anchor: flush + reset so every
            // candidate region is compiled from the same allocator state and
            // identical WASM yields byte-identical code.
            state.regalloc = None;
            let anchor = state.label_index;
            state.label_index += 1;
            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: anchor })
                .map_err(Err::from)?;
            sharing.open_anchor = Some(anchor);
        }

        self.sysv_handle_op::<E, Err>(ctx, arch, state, func_imports, op, rewriter, target)?;

        // Track the WASM stack depth across instructions with statically
        // known deltas. Non-instruction operators (locals, body markers) are
        // depth-neutral and must not reset tracking.
        if let Some(insn) = insn_opt {
            sharing.depth = match (sharing.depth, sysv_share_track_delta(insn)) {
                (Some(d), Some(delta)) => Some(d + delta),
                _ => None,
            };
        }

        if is_function_end {
            // Helpers are emitted after the epilogue the End arm just emitted,
            // reachable only through the forward call fixups placed above.
            let staging = core::mem::replace(
                &mut sharing.staging,
                FunctionStaging::new(sharing.config, crate::codegen::RV64_SYSV_OUTLINE_COST),
            );
            let mut sw = crate::codegen::SharingW {
                writer: &mut *self,
                ctx: &mut *ctx,
                arch,
                next_helper: sharing.next_helper,
            };
            staging.finalize::<_, _, _, Err>(&mut sw)?;
            sharing.next_helper = sw.next_helper;
        }
        Ok(())
    }
}

impl<T: SysVWriterExt<Context> + ?Sized, Context> SysVSharingWriterExt<Context> for T {}

#[cfg(test)]
mod sysv_sharing_tests {
    use super::*;
    use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
    use portal_solutions_blitz_common::ops::FnData;
    use portal_solutions_blitz_common::wasm_encoder::reencode::RoundtripReencoder;
    use portal_solutions_blitz_common::wasmparser::ValType;

    /// Test error joining the writer's `Infallible` core errors, reencode
    /// errors, and rewind-transaction errors.
    #[derive(Debug)]
    enum TestErr {
        Fmt,
        Reencode,
        Rewind,
    }
    impl From<core::convert::Infallible> for TestErr {
        fn from(x: core::convert::Infallible) -> Self {
            match x {}
        }
    }
    impl From<reencode::Error<core::convert::Infallible>> for TestErr {
        fn from(_: reencode::Error<core::convert::Infallible>) -> Self {
            TestErr::Reencode
        }
    }
    impl From<portal_solutions_asm_riscv64::out::rv_asm_backend::RvRewindError> for TestErr {
        fn from(_: portal_solutions_asm_riscv64::out::rv_asm_backend::RvRewindError) -> Self {
            TestErr::Rewind
        }
    }

    fn test_ops() -> [MachOperator<'static>; 13] {
        let mut data = FnData::default();
        data.num_params = 0;
        data.num_returns = 0;
        [
            MachOperator::StartFn { id: 0, data },
            MachOperator::Local {
                count: 1,
                ty: ValType::I64,
            },
            MachOperator::StartBody,
            MachOperator::Instruction {
                op: Instruction::I64Const(40),
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::I64Const(2),
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::I64Add,
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::LocalSet(0),
                annot: (),
            },
            // MemorySize + Drop: depth-neutral, but not region-eligible, so it
            // separates the two identical candidate runs.
            MachOperator::Instruction {
                op: Instruction::MemorySize(0),
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::Drop,
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::I64Const(40),
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::I64Const(2),
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::I64Add,
                annot: (),
            },
            MachOperator::Instruction {
                op: Instruction::LocalSet(0),
                annot: (),
            },
        ]
    }

    fn compile_plain() -> Vec<u8> {
        let mut w = RvAsmWriter::<RiscvLabel>::new();
        let mut ctx = ();
        let mut state = State::default();
        for op in test_ops().iter().chain(
            [MachOperator::Instruction {
                op: Instruction::End,
                annot: (),
            }]
            .iter(),
        ) {
            w.sysv_handle_op::<core::convert::Infallible, TestErr>(
                &mut ctx,
                RiscV64Arch::default(),
                &mut state,
                &[],
                op,
                &mut RoundtripReencoder,
                0,
            )
            .unwrap();
        }
        w.into_bytes()
    }

    type SharingBlob = <RvAsmWriter<RiscvLabel> as RewindableCodeWriter<RiscvLabel>>::Blob;

    fn compile_sharing() -> (Vec<u8>, alloc::collections::BTreeMap<RiscvLabel, usize>) {
        let mut w = RvAsmWriter::<RiscvLabel>::new();
        let mut ctx = ();
        let mut state = State::default();
        let mut sharing: Rv64Sharing<SharingBlob> = Rv64Sharing::enabled();
        for op in test_ops().iter().chain(
            [MachOperator::Instruction {
                op: Instruction::End,
                annot: (),
            }]
            .iter(),
        ) {
            w.sysv_handle_op_sharing::<core::convert::Infallible, TestErr, SharingBlob>(
                &mut ctx,
                RiscV64Arch::default(),
                &mut state,
                &mut sharing,
                &[],
                op,
                &mut RoundtripReencoder,
                0,
            )
            .unwrap();
        }
        w.into_parts()
    }

    #[test]
    fn repeated_region_is_outlined_and_smaller() {
        let plain = compile_plain();
        let (shared, labels) = compile_sharing();
        assert!(
            shared.len() < plain.len(),
            "shared {} must be smaller than plain {}",
            shared.len(),
            plain.len()
        );
        // One helper entry label from the helper namespace was placed.
        let helper_idx = crate::codegen::RV64_HELPER_LABEL_BASE;
        assert!(labels.contains_key(&RiscvLabel::Indexed { idx: helper_idx }));
        let helper_off = labels[&RiscvLabel::Indexed { idx: helper_idx }];
        assert!(helper_off > 0 && helper_off + 8 <= shared.len());
        // The helper body ends in `ret` = `jalr x0, ra, 0` (0x00008067 LE).
        assert_eq!(&shared[shared.len() - 4..], &[0x67, 0x80, 0x00, 0x00]);
        // Exactly two `jal ra` instructions (opcode 0x6F, rd=1) target it.
        let jal_offsets: Vec<usize> = (0..shared.len() / 4)
            .map(|i| i * 4)
            .filter(|&off| {
                let w = u32::from_le_bytes([
                    shared[off],
                    shared[off + 1],
                    shared[off + 2],
                    shared[off + 3],
                ]);
                (w & 0x7F) == 0x6F && ((w >> 7) & 0x1F) == 1
            })
            .collect();
        assert_eq!(jal_offsets.len(), 2);
        for off in jal_offsets {
            let w = u32::from_le_bytes([
                shared[off],
                shared[off + 1],
                shared[off + 2],
                shared[off + 3],
            ]);
            let imm = (((w >> 31) & 1) << 20)
                | (((w >> 12) & 0xFF) << 12)
                | (((w >> 20) & 1) << 11)
                | (((w >> 21) & 0x3FF) << 1);
            let disp = ((imm << 11) as i32 >> 11) as i64;
            assert_eq!(off as i64 + disp, helper_off as i64);
        }
    }

    #[test]
    fn disabled_sharing_is_byte_identical() {
        let mut w = RvAsmWriter::<RiscvLabel>::new();
        let mut ctx = ();
        let mut state = State::default();
        let mut sharing: Rv64Sharing<SharingBlob> = Rv64Sharing::default();
        for op in test_ops().iter().chain(
            [MachOperator::Instruction {
                op: Instruction::End,
                annot: (),
            }]
            .iter(),
        ) {
            w.sysv_handle_op_sharing::<core::convert::Infallible, TestErr, SharingBlob>(
                &mut ctx,
                RiscV64Arch::default(),
                &mut state,
                &mut sharing,
                &[],
                op,
                &mut RoundtripReencoder,
                0,
            )
            .unwrap();
        }
        assert_eq!(w.into_bytes(), compile_plain());
    }
}
