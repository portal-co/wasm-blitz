//! [`BackendAbi`] implementations for RISC-V 64.
//!
//! Provides `impl BackendAbi<W, Context> for NaiveAbi` (blitz WASM ABI) and
//! `impl BackendAbi<W, Context> for SysVAbi` (RISC-V psABI LP64).

extern crate alloc;

use portal_solutions_asm_riscv64::{RiscV64Arch, RegisterClass, out::arg::{ArgKind, MemArgKind}};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_blitz_common::{
    abi::BackendAbi,
    asm::Reg,
    ops::FnData,
    wasm_encoder::FuncType,
};

use crate::RiscvLabel;
use crate::naive::{State as NaiveState, WriterExt as NaiveWriterExt, push, pop};
use crate::sysv::SysVWriterExt;

// ---------------------------------------------------------------------------
// Local ABI discriminant types
// ---------------------------------------------------------------------------

/// Blitz WASM calling convention for RISC-V 64.
pub struct NaiveAbi;
/// RISC-V psABI LP64 calling convention.
pub struct SysVAbi;

// RISC-V psABI argument registers: A0–A7
const ARG_REGS: [Reg; 8] = [
    Reg(10), Reg(11), Reg(12), Reg(13), Reg(14), Reg(15), Reg(16), Reg(17),
];
const A0: Reg = Reg(10);
const A1: Reg = Reg(11);
const FP: Reg = Reg(8);  // s0 / frame pointer
const SP: Reg = Reg(2);
const RA: Reg = Reg(1);

fn mem64(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — blitz WASM ABI for RISC-V 64
//
// Uses FP-relative frame access (same layout as SysV) so that emit_local_get/set
// can be implemented without mutating state (required by the BackendAbi trait).
// The only difference from SysVAbi is the function label and calling convention.
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for NaiveAbi
where
    W: NaiveWriterExt<Context>,
{
    type Error = W::Error;
    type State = NaiveState;
    type Arch = RiscV64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut NaiveState,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;

        w.set_label(ctx, arch, RiscvLabel::Func { r#fn: id })?;

        // Standard RISC-V prologue: save FP and set up frame
        w.addi(ctx, arch, &SP, &SP, -8)?;
        w.sd(ctx, arch, &FP, &mem64(SP, 0))?;
        w.mv(ctx, arch, &FP, &SP)?;

        // Allocate space for locals + control-flow slots
        let locals_slots = state.local_count as i32 + state.control_depth as i32 * 2 + 4;
        if locals_slots > 0 {
            w.addi(ctx, arch, &SP, &SP, -(locals_slots * 8))?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut NaiveState,
    ) -> Result<(), W::Error> {
        w.li(ctx, arch, &A0, 0)?;
        let disp = -((state.local_count as i32 + 1) * 8);
        w.sd(ctx, arch, &A0, &mem64(FP, disp))?;
        state.local_count += 1;
        Ok(())
    }

    fn emit_start_body(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: RiscV64Arch,
        _state: &mut NaiveState,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        let disp = -((n as i32 + 1) * 8);
        w.ld(ctx, arch, &A0, &mem64(FP, disp))?;
        push(w, ctx, arch, A0)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        pop(w, ctx, arch, &A0)?;
        let disp = -((n as i32 + 1) * 8);
        w.sd(ctx, arch, &A0, &mem64(FP, disp))
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        // Peek at stack top without popping
        w.ld(ctx, arch, &A0, &mem64(SP, 0))?;
        let disp = -((n as i32 + 1) * 8);
        w.sd(ctx, arch, &A0, &mem64(FP, disp))
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        _sigs: &[FuncType],
        _fsigs: &[u32],
    ) -> Result<(), W::Error> {
        match func_imports.get(fn_idx as usize) {
            Some((module, name)) => {
                let sym = alloc::format!("{module}__{name}");
                w.jal_label(ctx, arch, &A0, RiscvLabel::External { name: sym })?;
                w.call(ctx, arch, &A0)
            }
            None => {
                let idx = fn_idx - func_imports.len() as u32;
                w.jal_label(ctx, arch, &A0, RiscvLabel::Func { r#fn: idx })?;
                w.call(ctx, arch, &A0)
            }
        }
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &NaiveState,
    ) -> Result<(), W::Error> {
        if state.num_returns > 0 {
            pop(w, ctx, arch, &A0)?;
        }
        // Restore frame: sp = fp, reload saved fp, return
        w.mv(ctx, arch, &SP, &FP)?;
        w.ld(ctx, arch, &FP, &mem64(SP, 0))?;
        w.addi(ctx, arch, &SP, &SP, 8)?;
        w.jalr(ctx, arch, &Reg(0), &RA, 0)
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — RISC-V psABI LP64 calling convention
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for SysVAbi
where
    W: SysVWriterExt<Context>,
{
    type Error = W::Error;
    type State = NaiveState;
    type Arch = RiscV64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut NaiveState,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;

        // SysV function label (offset avoids collision with naive Func labels)
        w.set_label(ctx, arch, RiscvLabel::Indexed {
            idx: id as usize | (1 << 28),
        })?;

        // Standard RISC-V SysV prologue
        let frame_slots = data.num_params + 2 + state.control_depth * 2 + 2;
        let frame_sz = (frame_slots * 8) as i32;
        w.addi(ctx, arch, &SP, &SP, -frame_sz)?;
        // Save RA and old FP
        w.sd(ctx, arch, &RA, &mem64(SP, frame_sz - 8))?;
        w.sd(ctx, arch, &FP, &mem64(SP, frame_sz - 16))?;
        // FP = old SP (top of frame)
        w.addi(ctx, arch, &FP, &SP, frame_sz)?;

        // Copy argument registers into FP-relative local frame
        for i in 0..data.num_params.min(8) {
            w.sysv_store_local(ctx, arch, ARG_REGS[i], i)?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut NaiveState,
    ) -> Result<(), W::Error> {
        w.li(ctx, arch, &A0, 0)?;
        w.sysv_store_local(ctx, arch, A0, state.local_count)?;
        state.local_count += 1;
        Ok(())
    }

    fn emit_start_body(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: RiscV64Arch,
        _state: &mut NaiveState,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        w.sysv_load_local(ctx, arch, A0, n as usize)?;
        push(w, ctx, arch, A0)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        pop(w, ctx, arch, &A0)?;
        w.sysv_store_local(ctx, arch, A0, n as usize)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        n: u32,
    ) -> Result<(), W::Error> {
        // Peek at stack top without popping
        w.ld(ctx, arch, &A0, &mem64(SP, 0))?;
        w.sysv_store_local(ctx, arch, A0, n as usize)
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        _state: &NaiveState,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        sigs: &[FuncType],
        fsigs: &[u32],
    ) -> Result<(), W::Error> {
        // Determine param/result counts from the type table
        let (n_params, n_results) = match fsigs.get(fn_idx as usize) {
            Some(&sig_idx) => {
                let sig = &sigs[sig_idx as usize];
                (sig.params().len(), sig.results().len())
            }
            None => (0, 0),
        };

        // Pop WASM stack args into RISC-V argument registers.
        // WASM stack top = last argument; pop highest-indexed reg first.
        for i in (0..n_params.min(8)).rev() {
            pop(w, ctx, arch, &ARG_REGS[i])?;
        }

        // Call target
        if let Some((module, name)) = func_imports.get(fn_idx as usize) {
            let sym = alloc::format!("{module}__{name}");
            w.jal_label(ctx, arch, &A0, RiscvLabel::External { name: sym })?;
            w.call(ctx, arch, &A0)?;
        } else {
            let local_id = fn_idx - func_imports.len() as u32;
            w.jal_label(ctx, arch, &A0, RiscvLabel::Indexed {
                idx: local_id as usize | (1 << 28),
            })?;
            w.call(ctx, arch, &A0)?;
        }

        // Push return values onto WASM stack.
        // result1 (A1) goes deeper, result0 (A0) on top.
        if n_results > 1 {
            push(w, ctx, arch, A1)?;
        }
        if n_results > 0 {
            push(w, ctx, arch, A0)?;
        }
        Ok(())
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &NaiveState,
    ) -> Result<(), W::Error> {
        if state.num_returns > 0 {
            pop(w, ctx, arch, &A0)?;
        }
        if state.num_returns > 1 {
            pop(w, ctx, arch, &A1)?;
        }
        // SysV epilogue: restore SP, RA, FP
        w.mv(ctx, arch, &SP, &FP)?;
        w.ld(ctx, arch, &RA, &mem64(FP, -8))?;
        w.ld(ctx, arch, &FP, &mem64(FP, 0))?;
        w.addi(ctx, arch, &SP, &SP, 16)?;
        w.jalr(ctx, arch, &Reg(0), &RA, 0)
    }
}
