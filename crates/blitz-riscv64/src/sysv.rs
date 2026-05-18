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
    ops::{MachOperator, TracingHooks},
    wasm_encoder::{Instruction, reencode::{self as reencode, Reencode}},
};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv64::{RegisterClass, out::arg::{ArgKind, MemArgKind}};

use crate::RiscvLabel;
use crate::naive::{State, WriterExt as NaiveExt, flush_regalloc, push, pop, pop_regalloc_to};

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
        Self: Sized,
    {
        match op {
            // Local access: delegate to naive which uses regalloc with [FP-(n+1)*8] offsets.
            // SysV frame places locals at the same offsets, so this is correct.
            Instruction::LocalGet(_) | Instruction::LocalSet(_) | Instruction::LocalTee(_) => {
                self.handle_op_(ctx, arch, state, func_imports, &[], &[], op, rewriter, target)
            }

            // Return: always emit RISC-V SysV epilogue.
            // flush_regalloc spills any register-held values to the memory stack,
            // after which they are readable with the normal memory pop.
            Instruction::Return => {
                // Use pop_regalloc_to: pops from regalloc (if active) or memory stack.
                if state.num_returns > 0 {
                    pop_regalloc_to(self, ctx, arch, state, A0)?;
                }
                if state.num_returns > 1 {
                    pop_regalloc_to(self, ctx, arch, state, A1)?;
                }
                flush_regalloc(self, ctx, arch, state)?;
                // Restore: RA at [FP - frame_sz + 0], old-FP at [FP - frame_sz + 8]
                self.addi(ctx, arch, &SP, &FP, -state.sysv_frame_sz)?;
                self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
                self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
                self.jalr(ctx, arch, &Reg(0), &RA, 0)
            }
            // Function-level End (empty if_stack) acts as implicit return.
            Instruction::End if state.if_stack.is_empty() => {
                flush_regalloc(self, ctx, arch, state)?;
                if state.num_returns > 0 {
                    pop(self, ctx, arch, &A0)?;
                }
                if state.num_returns > 1 {
                    pop(self, ctx, arch, &A1)?;
                }
                self.addi(ctx, arch, &SP, &FP, -state.sysv_frame_sz)?;
                self.ld(ctx, arch, &RA, &mem64(SP, 0))?;
                self.ld(ctx, arch, &FP, &mem64(SP, 8))?;
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
        arch: RiscV64Arch,
        state: &mut State,
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

                self.set_label(ctx, arch, RiscvLabel::Indexed {
                    idx: *id as usize | (1 << 28),
                }).map_err(Err::from)?;

                // Trace preamble: after label, before frame setup so SysV arg
                // regs (a0–a7, Reg 10–17) are intact for the outer-JIT tail-jump.
                if let Some(hooks) = data.tracing.as_ref() {
                    let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: 6 };
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, hooks.counter as u64, hooks.specialization as u64,
                        5, &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                // Frame layout (from FP downward):
                //   [FP-(1*8)]  = local 0   ← same offsets as naive, regalloc-compatible
                //   [FP-(2*8)]  = local 1
                //   ...
                //   [SP+8] = saved old-FP   ← bottom two slots for callee-saves
                //   [SP+0] = saved RA
                let frame_slots = data.num_params + 2 + state.control_depth * 2 + 2;
                let frame_sz = (frame_slots * 8) as i32;
                state.sysv_frame_sz = frame_sz;
                self.addi(ctx, arch, &SP, &SP, -frame_sz).map_err(Err::from)?;
                self.sd(ctx, arch, &RA, &mem64(SP, 0)).map_err(Err::from)?;
                self.sd(ctx, arch, &FP, &mem64(SP, 8)).map_err(Err::from)?;
                self.addi(ctx, arch, &FP, &SP, frame_sz).map_err(Err::from)?;

                for i in 0..data.num_params.min(8) {
                    self.sysv_store_local(ctx, arch, ARG_REGS[i], i).map_err(Err::from)?;
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
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, rewriter, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, rewriter, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Writer<RiscvLabel, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}
