//! AAPCS64 (AArch64 System V) ABI code generation.
//!
//! This module provides a `SysVWriterExt` trait that generates AArch64 functions
//! following the AAPCS64 standard calling convention, making them directly callable
//! from C and other System V–conforming code.
//!
//! # Calling Convention (AAPCS64)
//!
//! - Arguments: X0–X7 (up to 8 integer/pointer arguments)
//! - Return value: X0 (single), X0 + X1 (pair)
//! - Callee-saved: X19–X28, X29 (FP), X30 (LR), SP
//! - Frame: `stp x29, x30, [sp, #-16]!; mov x29, sp`
//!
//! The SysV entry code copies X0–X7 into the WASM local frame so that the
//! naive backend's local-variable handlers work unchanged.  This is the "slower"
//! path mentioned in docs/abi.md — the overhead is the register-to-stack copy at
//! the function boundary.

use alloc::vec::Vec;
extern crate alloc;

use portal_solutions_asm_aarch64::{
    out::{
        arg::{AddressingMode, ArgKind, MemArgKind},
        Writer, WriterCore,
    },
    AArch64Arch, RegisterClass,
};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::MachOperator,
    wasm_encoder::{FuncType, reencode::Reencode},
};

use crate::naive::{CallAbi, State, WriterExt, SCR};
use crate::AArch64Label;
use crate::codegen::TraceBase;
use crate::{FP, LR, SP};

/// Blitz register number of the AAPCS64 **trace-base virtual parameter** (`x12`).
///
/// `x12` is caller-saved and never a positional argument register (X0–X7), nor
/// the trace-preamble scratch (x9/x10), so the runtime can pass the per-function
/// trace-table base in it.  Read directly at the function-entry site; spilled to
/// an FP-relative frame slot for mid-function (loop/block) sites.
pub const TRACE_BASE_REG: u8 = 12;

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}
fn mem_pre(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PreIndex,
    }
}
fn mem_post(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PostIndex,
    }
}
fn mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::Offset,
    }
}

/// AAPCS64 argument registers in order (X0–X7).
const ARG_REGS: [Reg; 8] = [
    Reg(0), Reg(1), Reg(2), Reg(3), Reg(4), Reg(5), Reg(6), Reg(7),
];

use portal_solutions_blitz_common::wasm_encoder::{Instruction, reencode};

/// Extension trait for generating AAPCS64-compatible functions.
///
/// Delegates all instruction-level code to the naive `WriterExt::handle_insn`;
/// only the function boundary (StartFn / Return) is different.
pub trait SysVWriterExt<Context>: Writer<AArch64Label, Context> + WriterExt<Context> {
    /// Resolve a call/return_call target index to its label + (arity, results, is_import).
    /// Mirrors the x86-64 `sysv_call_target`; internal targets use the SysV entry
    /// label scheme (`Indexed { local_id + 0x80000000 }`, see `StartFn`).
    fn sysv_call_target(
        state: &State<'_>,
        func_imports: &[(&str, &str)],
        idx: u32,
    ) -> (AArch64Label, u32, u32, bool) {
        let widx = idx as usize;
        let is_import = idx < state.n_imports;
        let arity = state.call_params.get(widx).copied().unwrap_or(0);
        let results = state.call_results.get(widx).copied().unwrap_or(0);
        let label = if is_import {
            let (m, n) = func_imports[widx];
            AArch64Label::External { name: alloc::format!("{m}__{n}") }
        } else {
            AArch64Label::Indexed { idx: (idx - state.n_imports) as usize + 0x80000000 }
        };
        (label, arity, results, is_import)
    }

    /// Marshal `arity` operand-stack arguments per AAPCS64 and call `target`.
    ///
    /// Operand stack (SP-based) on entry: `param_{arity-1}` at `[sp]`, …, `param 0`
    /// at `[sp + (arity-1)*8]`. The first 8 params go in X0–X7; params 8+ are
    /// written to the 16-byte-aligned outgoing stack at `[sp_call + (i-8)*8]`,
    /// matching the SysV prologue's incoming-arg reads. Because the callee derives
    /// its frame from the same aligned `sp`, this is correct for both internal
    /// (blitz) and import (C ABI host) targets. `x15` holds the operand base across
    /// the call (saved in a stack slot); `x9`/`x10` are scratch. `results` (0..2)
    /// values are pushed back onto the operand stack from X0/X1.
    fn sysv_emit_marshalled_call(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        target: AArch64Label,
        arity: u32,
        results: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let base = Reg(15);
        let s9 = Reg(9);
        let s10 = Reg(10);
        let lit = |v: u64| MemArgKind::NoMem(ArgKind::Lit(v));

        // base = operand stack pointer.
        self.mov(ctx, arch, &reg(base), &reg(SP))?;
        // First min(arity, 8) args -> X0..X7.
        for i in 0..arity.min(8) {
            let disp = ((arity - 1 - i) * 8) as i32;
            self.ldr(ctx, arch, &reg(ARG_REGS[i as usize]), &mem_base_disp(base, disp))?;
        }
        // Reserve a 16-byte-aligned outgoing region: stack overflow args (i>=8)
        // plus one slot to preserve `base` across the call.
        let stack_args = arity.saturating_sub(8);
        let needed = (stack_args as u64) * 8 + 8;
        self.sub(ctx, arch, &reg(s9), &reg(base), &lit(needed))?;
        self.and(ctx, arch, &reg(s9), &reg(s9), &lit((-16i64) as u64))?;
        self.mov(ctx, arch, &reg(SP), &reg(s9))?;
        // Spill args 8.. to [sp + (i-8)*8].
        for i in 8..arity {
            let src = ((arity - 1 - i) * 8) as i32;
            self.ldr(ctx, arch, &reg(s10), &mem_base_disp(base, src))?;
            let dst = ((i - 8) * 8) as i32;
            self.str(ctx, arch, &reg(s10), &mem_base_disp(SP, dst))?;
        }
        // Save the operand base above the stack args, then call.
        let base_slot = (stack_args * 8) as i32;
        self.str(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
        self.adr_label(ctx, arch, &reg(s9), target)?;
        self.bl(ctx, arch, &reg(s9))?;
        // Restore base, pop all args (operand sp = base + arity*8).
        self.ldr(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
        self.add(ctx, arch, &reg(SP), &reg(base), &lit((arity as u64) * 8))?;
        // Push results from X0/X1.
        if results > 1 { self.wasm_push(ctx, arch, Reg(1))?; }
        if results > 0 { self.wasm_push(ctx, arch, Reg(0))?; }
        Ok(())
    }

    /// Emit the AAPCS64 epilogue (restore frame) and return. Shared by `Return`,
    /// function-level `End`, and the tail of `ReturnCall`.
    fn sysv_emit_epilogue(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if state.num_returns > 0 { self.wasm_pop(ctx, arch, Reg(0))?; }
        if state.num_returns > 1 { self.wasm_pop(ctx, arch, Reg(1))?; }
        self.mov(ctx, arch, &reg(SP), &reg(FP))?;
        self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
        if state.shard.is_some() {
            self.ldp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_post(SP, 16))?;
        }
        self.ret(ctx, arch)
    }

    /// Handle an instruction, using the AAPCS64 Return epilogue instead of the naive one.
    fn sysv_handle_insn(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        match op {
            // ---- local access (AAPCS64 FP-relative frame) ------------------
            // The SysV prologue stores args at [FP - (n+1)*8], which is exactly
            // the layout that naive's load_local/store_local use, so we can
            // delegate directly.  Reg(9) = x9, a caller-saved scratch register.
            Instruction::LocalGet(idx) => {
                self.load_local(ctx, arch, Reg(9), *idx as usize)?;
                self.wasm_push(ctx, arch, Reg(9))
            }
            Instruction::LocalSet(idx) => {
                self.wasm_pop(ctx, arch, Reg(9))?;
                self.store_local(ctx, arch, Reg(9), *idx as usize)
            }
            Instruction::LocalTee(idx) => {
                // Peek at the top of the WASM stack without popping.
                // SP = Reg(31) in AArch64.
                self.ldr(ctx, arch, &reg(Reg(9)), &mem_base_disp(Reg(31), 0))?;
                self.store_local(ctx, arch, Reg(9), *idx as usize)
            }

            // ---- Calls: marshal operand-stack args per AAPCS64 (AllStack mode) ----
            Instruction::Call(idx) if state.call_abi == CallAbi::AllStack => {
                let (label, arity, results, _is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                self.sysv_emit_marshalled_call(ctx, arch, label, arity, results)
            }
            Instruction::ReturnCall(idx) if state.call_abi == CallAbi::AllStack => {
                let (label, arity, results, _is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                self.sysv_emit_marshalled_call(ctx, arch, label, arity, results)?;
                // Tail: return our (= callee's) results to our caller (AAPCS64 epilogue).
                self.sysv_emit_epilogue(ctx, arch, state)
            }

            // ---- Return: always emit AAPCS64 epilogue regardless of block depth ----
            Instruction::Return => self.sysv_emit_epilogue(ctx, arch, state),
            // ---- Function-level End (empty if_stack) ----
            Instruction::End if state.if_stack.is_empty() => {
                self.sysv_emit_epilogue(ctx, arch, state)
            }
            other => self.handle_insn(ctx, arch, state, func_imports, &[], &[], other, target),
        }
    }

    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
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
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.tracing = data.tracing;
                state.next_site_id = 1; // site 0 is the function entry below

                self.set_label(ctx, arch, AArch64Label::Indexed { idx: *id as usize + 0x80000000 })
                    .map_err(Err::from)?;

                // Function-entry site (site 0): trace-table base arrives in the
                // virtual-param register x12; read it directly before the frame
                // is built so the tail-jump delivers X0–X7 intact.  Scratch x9/x10.
                if let Some(cfg) = data.tracing.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW {
                        writer: self, ctx, arch, scratch2: 10,
                        trace_base: TraceBase::Reg(TRACE_BASE_REG),
                    };
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, cfg.table_base_off, 0,
                        9, &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                // Save SCR (X27) in a 16-byte aligned pair before FP+LR.
                if state.shard.is_some() {
                    self.stp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_pre(SP, -16)).map_err(Err::from)?;
                }
                self.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP)).map_err(Err::from)?;

                // One extra slot (frame bottom) holds the spilled trace base for
                // mid-function sites.
                let locals_slots = data.num_params as i64 + state.control_depth as i64 * 2 + 3;
                let size = MemArgKind::NoMem(ArgKind::Lit((locals_slots * 8) as u64));
                self.sub(ctx, arch, &reg(SP), &reg(SP), &size).map_err(Err::from)?;

                // Spill the virtual-param base (x12) to the bottom frame slot and
                // point mid-function sites at it.
                if data.tracing.as_ref().map(|c| c.enabled).unwrap_or(false) {
                    let disp = -(locals_slots as i32 * 8);
                    state.trace_base = TraceBase::FrameSlot(disp);
                    self.str(ctx, arch, &reg(Reg(TRACE_BASE_REG)), &mem_base_disp(FP, disp))
                        .map_err(Err::from)?;
                }

                for i in 0..data.num_params.min(8) {
                    let disp = -((i as i32 + 1) * 8);
                    self.str(ctx, arch, &reg(ARG_REGS[i]), &mem_base_disp(FP, disp))
                        .map_err(Err::from)?;
                }
                // Params 9+ (index >= 8) are passed by the caller on the stack,
                // above the saved FP/LR pair (and saved SCR/x9 pair if sharding).
                // With `mov FP, SP` taken after those stores, incoming arg `i` is
                // at [FP + 16 + scr_extra + (i-8)*8]. Copy each into its local
                // slot so functions with >8 params receive all their arguments.
                let scr_extra: i32 = if state.shard.is_some() { 16 } else { 0 };
                for i in 8..data.num_params {
                    let src_disp = 16 + scr_extra + ((i as i32 - 8) * 8);
                    self.ldr(ctx, arch, &reg(Reg(9)), &mem_base_disp(FP, src_disp))
                        .map_err(Err::from)?;
                    self.store_local(ctx, arch, Reg(9), i).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov_imm(ctx, arch, &reg(Reg(9)), 0).map_err(Err::from)?;
                for _ in 0..*count {
                    state.local_count += 1;
                    self.store_local(ctx, arch, Reg(9), state.local_count - 1)
                        .map_err(Err::from)?;
                }
                Ok(())
            }
            MachOperator::StartBody | MachOperator::EndBody => Ok(()),
            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Writer<AArch64Label, Context> + WriterExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sysv_manyarg_tests {
    use super::*;
    use portal_solutions_asm_aarch64::out::bin::AArch64Writer;

    /// Drive `sysv_emit_marshalled_call` and return (code bytes, target relocs).
    fn marshal(arity: u32, results: u32, target: AArch64Label) -> (Vec<u8>, Vec<AArch64Label>) {
        let mut w = AArch64Writer::<AArch64Label>::new();
        let mut ctx = ();
        SysVWriterExt::sysv_emit_marshalled_call(
            &mut w, &mut ctx, AArch64Arch::default(), target, arity, results,
        )
        .unwrap();
        let (bytes, _labels, relocs) = w.into_parts_with_relocs();
        (bytes, relocs.into_iter().map(|r| r.label).collect())
    }

    #[test]
    fn marshalled_call_targets_label_and_grows_with_arity() {
        let (small, rel) = marshal(4, 1, AArch64Label::Indexed { idx: 0x8000_0000 });
        let (big, _) = marshal(20, 1, AArch64Label::Indexed { idx: 0x8000_0000 });
        // Exactly one relocation — the call branch — pointing at the requested target.
        assert_eq!(rel.len(), 1);
        assert!(matches!(&rel[0], AArch64Label::Indexed { idx } if *idx == 0x8000_0000));
        // >8 args spill to the stack, so more code is emitted than for 4 args.
        assert!(big.len() > small.len());
    }

    #[test]
    fn no_stack_spill_within_eight_args() {
        // 8 args fit entirely in X0..X7; 12 args spill 4 → strictly more code.
        let (eight, _) = marshal(8, 0, AArch64Label::External { name: "libc__f".into() });
        let (twelve, _) = marshal(12, 0, AArch64Label::External { name: "libc__f".into() });
        assert!(twelve.len() > eight.len());
    }

    #[test]
    fn import_target_emits_external_reloc() {
        let (_b, rel) = marshal(3, 1, AArch64Label::External { name: "env__write".into() });
        assert!(rel
            .iter()
            .any(|l| matches!(l, AArch64Label::External { name } if name.as_str() == "env__write")));
    }
}
