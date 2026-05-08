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

use portal_solutions_asm_riscv64::RiscV64Arch;
use portal_solutions_asm_riscv64::out::Writer;
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::MachOperator,
    wasm_encoder::{Instruction, reencode::Reencode},
};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv64::{RegisterClass, out::arg::{ArgKind, MemArgKind}};

use crate::RiscvLabel;
use crate::naive::{State, WriterExt as NaiveExt, push, pop};

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

fn mem64(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// Extension trait for generating RISC-V psABI-compatible functions.
pub trait SysVWriterExt<Context>: Writer<RiscvLabel, Context> + NaiveExt<Context> {

    /// Load local N from the FP-relative frame into `dest`.
    fn sysv_load_local(&mut self, ctx: &mut Context, arch: RiscV64Arch, dest: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        // Local n at [fp - (n+1)*8]
        let disp = -((n as i32 + 1) * 8);
        self.ld(ctx, arch, &dest, &mem64(FP, disp))
    }

    /// Store `src` into local N in the FP-relative frame.
    fn sysv_store_local(&mut self, ctx: &mut Context, arch: RiscV64Arch, src: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.sd(ctx, arch, &src, &mem64(FP, disp))
    }

    /// Handle an instruction with RISC-V SysV local/return semantics.
    fn sysv_handle_insn<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        portal_solutions_blitz_common::wasm_encoder::reencode::Error<E>: Into<Self::Error>,
        Self::Error: From<core::fmt::Error>,
        Self: Sized,
    {
        match op {
            // Local access: use FP-relative frame
            Instruction::LocalGet(n) => {
                self.sysv_load_local(ctx, arch, A0, *n as usize)?;
                push(self, ctx, arch, A0)
            }
            Instruction::LocalSet(n) => {
                pop(self, ctx, arch, &A0)?;
                self.sysv_store_local(ctx, arch, A0, *n as usize)
            }
            Instruction::LocalTee(n) => {
                // Peek at top of WASM stack without popping
                self.ld(ctx, arch, &A0, &mem64(SP, 0))?;
                self.sysv_store_local(ctx, arch, A0, *n as usize)
            }

            // Return: pop result into a0, then restore and ret
            Instruction::Return => {
                if state.num_returns > 0 {
                    pop(self, ctx, arch, &A0)?;
                }
                if state.num_returns > 1 {
                    pop(self, ctx, arch, &A1)?;
                }
                // Epilogue: restore SP, RA, FP
                self.mv(ctx, arch, &SP, &FP)?;
                self.ld(ctx, arch, &RA, &mem64(FP, -8))?;  // restored RA
                self.ld(ctx, arch, &FP, &mem64(FP, 0))?;   // restored old FP
                self.addi(ctx, arch, &SP, &SP, 16)?;        // pop saved RA+FP
                self.jalr(ctx, arch, &Reg(0), &RA, 0)       // ret
            }

            // Everything else: naive handler
            other => self.handle_op_(ctx, arch, state, func_imports, other, rewriter, target),
        }
    }

    /// Handle a `MachOperator` using the RISC-V SysV ABI.
    fn sysv_handle_op<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        portal_solutions_blitz_common::wasm_encoder::reencode::Error<E>: Into<Self::Error>,
        Self::Error: From<core::fmt::Error>,
        Self: Sized,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;

                // SysV function label (use offset to differentiate from naive)
                self.set_label(ctx, arch, RiscvLabel::Indexed {
                    idx: *id as usize | (1 << 28),
                })?;

                // Standard RISC-V SysV prologue
                // Allocate frame: RA + old FP + params
                let frame_slots = data.num_params + 2 + state.control_depth * 2 + 2;
                let frame_sz = (frame_slots * 8) as i32;
                self.addi(ctx, arch, &SP, &SP, -frame_sz)?;
                // Save RA and old FP
                self.sd(ctx, arch, &RA, &mem64(SP, frame_sz - 8))?;
                self.sd(ctx, arch, &FP, &mem64(SP, frame_sz - 16))?;
                // FP = old SP (top of frame)
                self.addi(ctx, arch, &FP, &SP, frame_sz)?;

                // Copy argument registers into local frame
                for i in 0..data.num_params.min(8) {
                    self.sysv_store_local(ctx, arch, ARG_REGS[i], i)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.li(ctx, arch, &A0, 0)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, A0, state.local_count)?;
                    state.local_count += 1;
                }
                Ok(())
            }

            MachOperator::StartBody | MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, rewriter, target)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone()).map_err(|e| e.into())?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, rewriter, target)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Writer<RiscvLabel, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}
