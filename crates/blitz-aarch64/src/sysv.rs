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

use crate::naive::{State, WriterExt, SCR};
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

            // ---- Return: always emit AAPCS64 epilogue regardless of block depth ----
            Instruction::Return => {
                if state.num_returns > 0 { self.wasm_pop(ctx, arch, Reg(0))?; }
                if state.num_returns > 1 { self.wasm_pop(ctx, arch, Reg(1))?; }
                self.mov(ctx, arch, &reg(SP), &reg(FP))?;
                self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
                // Restore SCR if sharding active (Reg(9) = x9 gets discarded — OK).
                if state.shard.is_some() {
                    self.ldp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_post(SP, 16))?;
                }
                self.ret(ctx, arch)
            }
            // ---- Function-level End (empty if_stack) ----
            Instruction::End if state.if_stack.is_empty() => {
                if state.num_returns > 0 { self.wasm_pop(ctx, arch, Reg(0))?; }
                if state.num_returns > 1 { self.wasm_pop(ctx, arch, Reg(1))?; }
                self.mov(ctx, arch, &reg(SP), &reg(FP))?;
                self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
                if state.shard.is_some() {
                    self.ldp(ctx, arch, &reg(SCR), &reg(Reg(9)), &mem_post(SP, 16))?;
                }
                self.ret(ctx, arch)
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
