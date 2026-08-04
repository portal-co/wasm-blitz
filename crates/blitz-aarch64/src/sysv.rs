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
    ops::{MachOperator, ProbePlacement},
    wasm_encoder::{FuncType, reencode::Reencode},
};

use crate::naive::{State, WriterExt, SCR, WASM_SLOT};
use crate::AArch64Label;

// Re-exported (renamed to match `blitz-x86-64::sysv::{SysVState, CallAbi}`'s
// naming) so callers building a C-callable (AAPCS64) entry never need to
// reach into `crate::naive` themselves — that module's *name* is
// historically a source of ABI confusion here: `naive::State`/
// `naive::CallAbi` are what this file's `SysVWriterExt` actually configures
// to produce genuinely SysV/AAPCS64-compliant code (see this file's module
// doc), despite living in a module called "naive". `naive` is slated for a
// deeper split/removal (see `docs/naive-abi-deprecation.md`); until then,
// `sysv::{SysVState, CallAbi, MemBase}` is the name a caller should reach
// for instead of `naive::{State, CallAbi, MemBase}`.
pub use crate::naive::{CallAbi, MemBase, State as SysVState};
use crate::codegen::ProbeBase;
use crate::{FP, LR, SP};

/// Blitz register number of the AAPCS64 **probe-base virtual parameter** (`x12`).
///
/// `x12` is caller-saved and never a positional argument register (X0–X7), nor
/// the probe-preamble scratch (x9/x10), so the runtime can pass the per-function
/// probe-table base in it.  Read directly at the function-entry site; spilled to
/// an FP-relative frame slot for mid-function (loop/block) sites.
pub const PROBE_BASE_REG: u8 = 12;

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
    ///
    /// Pop the `call_indirect` table index and load the function pointer from
    /// `__wasm_table[index]` into x14 (a register the marshalling preserves).
    fn sysv_load_indirect_target(&mut self, ctx: &mut Context, arch: AArch64Arch) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let idx = Reg(14);
        let tbl = Reg(13);
        self.wasm_pop(ctx, arch, idx)?; // table index → x14
        crate::load_label_addr(self, ctx, arch, &reg(tbl), AArch64Label::External { name: "__wasm_table".into() })?;
        // x14 := *(x13 + x14*8)
        let lit = |v: u64| MemArgKind::NoMem(ArgKind::Lit(v));
        self.lsl(ctx, arch, &reg(idx), &reg(idx), &lit(3))?;
        self.add(ctx, arch, &reg(tbl), &reg(tbl), &reg(idx))?;
        self.ldr(ctx, arch, &reg(idx), &mem_base_disp(tbl, 0))
    }

    /// Marshal `arity` operand-stack args and call `target`.
    ///
    /// - `is_import`: AAPCS64 host import — first ≤8 params in X0–X7, rest on stack.
    /// - otherwise (`CallAbi::AllStack` internal): write *all* params to
    ///   `[SP_call + i*8]` so the callee prologue reads `param i` at
    ///   `[FP + 16 + scr_extra + i*8]` (mirrors x86-64 AllStack).
    fn sysv_emit_marshalled_call(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        target: Option<AArch64Label>,
        target_reg: Option<Reg>,
        arity: u32,
        results: u32,
        is_import: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let base = Reg(15);
        let s9 = Reg(9);
        let s10 = Reg(10);
        let lit = |v: u64| MemArgKind::NoMem(ArgKind::Lit(v));

        self.mov(ctx, arch, &reg(base), &reg(SP))?;
        if is_import {
            // AAPCS64: first 8 integer args in registers; args 9+ on the stack.
            for i in 0..arity.min(8) {
                let disp = (arity - 1 - i) as i32 * WASM_SLOT;
                self.ldr(ctx, arch, &reg(ARG_REGS[i as usize]), &mem_base_disp(base, disp))?;
            }
            let stack_args = arity.saturating_sub(8);
            let needed = (stack_args as u64) * 8 + 8;
            self.sub(ctx, arch, &reg(s9), &reg(base), &lit(needed))?;
            self.mov_imm(ctx, arch, &reg(s10), (-16i64) as u64)?;
            self.and(ctx, arch, &reg(s9), &reg(s9), &reg(s10))?;
            self.mov(ctx, arch, &reg(SP), &reg(s9))?;
            for i in 8..arity {
                let src = (arity - 1 - i) as i32 * WASM_SLOT;
                self.ldr(ctx, arch, &reg(s10), &mem_base_disp(base, src))?;
                let dst = ((i - 8) * 8) as i32;
                self.str(ctx, arch, &reg(s10), &mem_base_disp(SP, dst))?;
            }
            let base_slot = (stack_args * 8) as i32;
            self.str(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
            if let Some(r) = target_reg {
                self.bl(ctx, arch, &reg(r))?;
            } else {
                crate::load_label_addr(self, ctx, arch, &reg(s9), target.unwrap())?;
                self.bl(ctx, arch, &reg(s9))?;
            }
            self.ldr(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
            self.add(ctx, arch, &reg(SP), &reg(base), &lit(arity as u64 * WASM_SLOT as u64))?;
        } else {
            // Internal AllStack: every param on the outgoing stack.
            let needed = (arity as u64 + 1) * 8;
            self.sub(ctx, arch, &reg(s9), &reg(base), &lit(needed))?;
            self.mov_imm(ctx, arch, &reg(s10), (-16i64) as u64)?;
            self.and(ctx, arch, &reg(s9), &reg(s9), &reg(s10))?;
            self.mov(ctx, arch, &reg(SP), &reg(s9))?;
            for i in 0..arity {
                let src = (arity - 1 - i) as i32 * WASM_SLOT;
                self.ldr(ctx, arch, &reg(s10), &mem_base_disp(base, src))?;
                self.str(ctx, arch, &reg(s10), &mem_base_disp(SP, (i * 8) as i32))?;
            }
            let base_slot = (arity * 8) as i32;
            self.str(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
            if let Some(r) = target_reg {
                self.bl(ctx, arch, &reg(r))?;
            } else {
                crate::load_label_addr(self, ctx, arch, &reg(s9), target.unwrap())?;
                self.bl(ctx, arch, &reg(s9))?;
            }
            self.ldr(ctx, arch, &reg(base), &mem_base_disp(SP, base_slot))?;
            self.add(ctx, arch, &reg(SP), &reg(base), &lit(arity as u64 * WASM_SLOT as u64))?;
        }
        if results > 1 { self.wasm_push(ctx, arch, Reg(1))?; }
        if results > 0 { self.wasm_push(ctx, arch, Reg(0))?; }
        Ok(())
    }

    /// Emit a **true tail call** to an internal AllStack `target` with `arity` args.
    ///
    /// Overwrites our own incoming AllStack arg slots (`[FP + 16 + scr_extra +
    /// i*8]` for every `i`), restores the caller's frame, and `b`/`br`s so the
    /// callee returns straight to our caller. O(1) machine stack across
    /// `return_call` chains. Requires `arity <= param_count`.
    fn sysv_emit_tail_call(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        target: Option<AArch64Label>,
        target_reg: Option<Reg>,
        arity: u32,
        shard: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let base = Reg(15);
        let s9 = Reg(9);
        self.mov(ctx, arch, &reg(base), &reg(SP))?;
        let scr_extra: i32 = if shard { 16 } else { 0 };
        for i in 0..arity {
            let src = (arity - 1 - i) as i32 * WASM_SLOT;
            self.ldr(ctx, arch, &reg(s9), &mem_base_disp(base, src))?;
            let dst = 16 + scr_extra + (i as i32 * 8);
            self.str(ctx, arch, &reg(s9), &mem_base_disp(FP, dst))?;
        }
        self.mov(ctx, arch, &reg(SP), &reg(FP))?;
        self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
        if shard {
            self.ldp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_post(SP, 16))?;
        }
        if let Some(r) = target_reg {
            self.br(ctx, arch, &reg(r))
        } else {
            self.b_label(ctx, arch, target.unwrap())
        }
    }

    /// Emit the AAPCS64 epilogue (restore frame) and return. Shared by `Return`,
    /// function-level `End`, and the tail of `ReturnCall`.
    ///
    /// WASM multi-value results are pushed on the operand stack in
    /// *declaration* order: result 0 is pushed first (deepest), result
    /// `n-1` last (on top) — see the WASM spec. AAPCS64 can carry at most
    /// two results back through registers (X0, X1), so this reads result 0
    /// and result 1 directly by their known offset from `SP`
    /// (`(n-1)*WASM_SLOT` and `(n-2)*WASM_SLOT` respectively) instead of
    /// just popping the top two values in stack order — popping in stack
    /// order would put the *last*-declared result in X0 instead of the
    /// first, which is wrong whenever a caller outside this WASM module
    /// (e.g. a C entry shim reading a recompiled guest's `main` return
    /// value) reads X0 expecting "result 0". Any results beyond index 1
    /// are simply skipped (there is no native register for them); `mov
    /// sp, fp` below discards the whole operand stack unconditionally, so
    /// leaving them unread is not a leak.
    /// Flush `state.regalloc` to the real SP-based stack, if any values are
    /// currently register-held. Must be called before any of this file's own
    /// code that touches SP or a fixed register directly — `sysv_handle_insn`
    /// intercepts `LocalGet`/`LocalSet`/`LocalTee`/`Return`/`End`/the `Call`
    /// variants *before* they ever reach `naive`'s `_handle_op` (where the
    /// regalloc-covered arms and their own flush pre-check live), so none of
    /// that protection applies here without this being called explicitly.
    fn sysv_flush_regalloc(&mut self, ctx: &mut Context, arch: AArch64Arch, state: &mut State<'_>)
        -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let mut rw = crate::codegen::RegAllocW { writer: self, ctx, arch, regalloc: &mut state.regalloc };
        portal_solutions_blitz_codegen::control_flow::ControlFlowWriter::flush(&mut rw)
    }

    fn sysv_emit_epilogue(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // Every result this reads comes straight off [SP]; flush first so a
        // regalloc-held result isn't silently skipped.
        self.sysv_flush_regalloc(ctx, arch, state)?;
        let n = state.num_returns;
        if n > 0 {
            let disp = (n - 1) as i32 * WASM_SLOT;
            self.ldr(ctx, arch, &reg(Reg(0)), &mem_base_disp(SP, disp))?;
        }
        if n > 1 {
            let disp = (n - 2) as i32 * WASM_SLOT;
            self.ldr(ctx, arch, &reg(Reg(1)), &mem_base_disp(SP, disp))?;
        }
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
            // the layout that naive's load_local/store_local use — LocalGet/
            // LocalSet fall through to the `other` arm below, which reaches
            // naive's regalloc-covered versions of them directly. LocalTee
            // isn't regalloc-covered in naive either, so it stays a raw
            // override here (flushed first, like everything else in this
            // match — see `sysv_flush_regalloc`'s doc comment for why).
            Instruction::LocalTee(idx) => {
                // Peek at the top of the WASM stack without popping.
                // SP = Reg(31) in AArch64.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                self.ldr(ctx, arch, &reg(Reg(9)), &mem_base_disp(Reg(31), 0))?;
                self.store_local(ctx, arch, Reg(9), *idx as usize)
            }

            // ---- Calls: marshal operand-stack args (AllStack mode) ----
            Instruction::Call(idx) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                self.sysv_emit_marshalled_call(ctx, arch, Some(label), None, arity, results, is_import)
            }
            Instruction::ReturnCall(idx) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                if is_import || arity as usize > state.param_count {
                    // Fake tail: a real call then the AAPCS64 epilogue. Required for
                    // C-ABI imports and a safe fallback when the callee wants more
                    // args than our incoming-arg frame holds.
                    self.sysv_emit_marshalled_call(
                        ctx, arch, Some(label), None, arity, results, is_import,
                    )?;
                    self.sysv_emit_epilogue(ctx, arch, state)
                } else {
                    // True tail call to an internal function: O(1) machine stack for
                    // long `return_call` chains.
                    self.sysv_emit_tail_call(ctx, arch, Some(label), None, arity, state.shard.is_some())
                }
            }

            // ---- Indirect calls: target = __wasm_table[index] ----
            Instruction::CallIndirect { type_index, .. } if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.sysv_load_indirect_target(ctx, arch)?; // index → x14 (fn ptr)
                // Indirect targets are always internal guest functions (AllStack).
                self.sysv_emit_marshalled_call(ctx, arch, None, Some(Reg(14)), arity, results, false)
            }
            Instruction::ReturnCallIndirect { type_index, .. } if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.sysv_load_indirect_target(ctx, arch)?; // index → x14 (fn ptr)
                if arity as usize > state.param_count {
                    self.sysv_emit_marshalled_call(
                        ctx, arch, None, Some(Reg(14)), arity, results, false,
                    )?;
                    self.sysv_emit_epilogue(ctx, arch, state)
                } else {
                    self.sysv_emit_tail_call(ctx, arch, None, Some(Reg(14)), arity, state.shard.is_some())
                }
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
                state.param_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.probes = data.probes;
                state.next_probe_id = 1; // probe 0 is the function entry below
                state.probe_plan = data.probe_plan.clone();
                state.op_index = 0;
                // Register allocation is per-function (see naive::State::regalloc's
                // doc comment) — sysv has its own StartFn separate from naive's.
                state.regalloc = None;

                self.set_label(ctx, arch, AArch64Label::Indexed { idx: *id as usize + 0x80000000 })
                    .map_err(Err::from)?;

                // Function-entry probe (probe 0): probe-table base arrives in the
                // virtual-param register x12; read it directly before the frame
                // is built so the tail-jump delivers X0–X7 intact.  Scratch x9/x10.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW {
                        writer: self, ctx, arch, scratch2: 10,
                        probe_base: ProbeBase::Reg(PROBE_BASE_REG),
                    };
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw, cfg.table_base_off, 0, 9,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                // Save SCR (X27) in a 16-byte aligned pair before FP+LR.
                if state.shard.is_some() {
                    self.stp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_pre(SP, -16)).map_err(Err::from)?;
                }
                self.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP)).map_err(Err::from)?;

                // One extra slot (frame bottom) holds the spilled probe base for
                // mid-function sites.
                let locals_slots = data.num_params as i64 + state.control_depth as i64 * 2 + 3;
                // Round the frame to 16 bytes so SP stays 16-byte aligned.
                let frame_bytes = (locals_slots as u64 * 8 + 15) & !15;
                let size = MemArgKind::NoMem(ArgKind::Lit(frame_bytes));
                self.sub(ctx, arch, &reg(SP), &reg(SP), &size).map_err(Err::from)?;

                // Spill the virtual-param base (x12) to the bottom frame slot and
                // point mid-function sites at it.
                if data.probes.as_ref().map(|c| c.enabled).unwrap_or(false) {
                    let disp = -(locals_slots as i32 * 8);
                    state.probe_base = ProbeBase::FrameSlot(disp);
                    self.str(ctx, arch, &reg(Reg(PROBE_BASE_REG)), &mem_base_disp(FP, disp))
                        .map_err(Err::from)?;
                }

                let scr_extra: i32 = if state.shard.is_some() { 16 } else { 0 };
                match state.call_abi {
                    CallAbi::RegSysv => {
                        for i in 0..data.num_params.min(8) {
                            let disp = -((i as i32 + 1) * 8);
                            self.str(ctx, arch, &reg(ARG_REGS[i]), &mem_base_disp(FP, disp))
                                .map_err(Err::from)?;
                        }
                        // Params 9+ arrive on the stack above saved FP/LR (+ SCR).
                        for i in 8..data.num_params {
                            let src_disp = 16 + scr_extra + ((i as i32 - 8) * 8);
                            self.ldr(ctx, arch, &reg(Reg(9)), &mem_base_disp(FP, src_disp))
                                .map_err(Err::from)?;
                            self.store_local(ctx, arch, Reg(9), i).map_err(Err::from)?;
                        }
                    }
                    CallAbi::AllStack => {
                        // All params arrive on the stack; param `i` at
                        // [FP + 16 + scr_extra + i*8].
                        for i in 0..data.num_params {
                            let src_disp = 16 + scr_extra + (i as i32 * 8);
                            self.ldr(ctx, arch, &reg(Reg(9)), &mem_base_disp(FP, src_disp))
                                .map_err(Err::from)?;
                            self.store_local(ctx, arch, Reg(9), i).map_err(Err::from)?;
                        }
                    }
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
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before).map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, target).map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After).map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before).map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, target).map_err(Err::from)?;
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
    /// Reuses the same mid-function probe-base addressing
    /// (`ProbeBase::FrameSlot`, already live in `state.probe_base` once
    /// `StartFn` has run) and scratch registers (x9/x10) that the
    /// function-entry probe above already establishes. `ProbeMode`
    /// (Active/Passive) makes no codegen difference here: AArch64 has no
    /// register allocator to disturb between ops, so both are already as
    /// non-disturbing as possible.
    fn sysv_emit_indexed_probes(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
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
            let mut bw = crate::codegen::BlitzW {
                writer: self, ctx, arch, scratch2: 10,
                probe_base: state.probe_base,
            };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw, cfg.table_base_off, spec.probe_id, 9, spec.binding, &mut state.label_index,
            )?;
        }
        Ok(())
    }
}

impl<T: Writer<AArch64Label, Context> + WriterExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sysv_manyarg_tests {
    use super::*;
    use portal_solutions_asm_aarch64::out::bin::AArch64Writer;

    /// Drive `sysv_emit_marshalled_call` and return (code bytes, target relocs).
    fn marshal(
        arity: u32,
        results: u32,
        target: AArch64Label,
        is_import: bool,
    ) -> (Vec<u8>, Vec<AArch64Label>) {
        let mut w = AArch64Writer::<AArch64Label>::new();
        let mut ctx = ();
        SysVWriterExt::sysv_emit_marshalled_call(
            &mut w, &mut ctx, AArch64Arch::default(), Some(target), None, arity, results, is_import,
        )
        .unwrap();
        let (bytes, _labels, relocs) = w.into_parts_with_relocs();
        (bytes, relocs.into_iter().map(|r| r.label).collect())
    }

    #[test]
    fn marshalled_call_targets_label_and_grows_with_arity() {
        let (small, rel) = marshal(4, 1, AArch64Label::Indexed { idx: 0x8000_0000 }, false);
        let (big, _) = marshal(20, 1, AArch64Label::Indexed { idx: 0x8000_0000 }, false);
        // Exactly one relocation — the call branch — pointing at the requested target.
        assert_eq!(rel.len(), 1);
        assert!(matches!(&rel[0], AArch64Label::Indexed { idx } if *idx == 0x8000_0000));
        // AllStack internals put every arg on the stack — more arity → more code.
        assert!(big.len() > small.len());
    }

    #[test]
    fn import_spill_grows_past_eight_args() {
        // Import AAPCS64: 8 args fit in X0..X7; 12 args spill 4 → strictly more code.
        let (eight, _) = marshal(8, 0, AArch64Label::External { name: "libc__f".into() }, true);
        let (twelve, _) = marshal(12, 0, AArch64Label::External { name: "libc__f".into() }, true);
        assert!(twelve.len() > eight.len());
    }

    #[test]
    fn import_target_emits_external_reloc() {
        let (_b, rel) = marshal(3, 1, AArch64Label::External { name: "env__write".into() }, true);
        assert!(rel
            .iter()
            .any(|l| matches!(l, AArch64Label::External { name } if name.as_str() == "env__write")));
    }
}
