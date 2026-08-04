//! AAPCS (ARM SysV / soft-float ILP32) entry path.
//!
//! Prologue: push `{fp, lr}` equivalent, `mov fp, sp`.
//! Return: i64 in `r0:r1` (low:high), then epilogue + `bx lr`.

use crate::naive::{State, WriterExt as NaiveExt, WASM_SLOT};
use crate::ArmLabel;
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_arm::{
    ArmArch,
    out::{
        Writer,
        arg::{ArgKind, MemArgKind},
    },
};
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::MachOperator,
    wasm_encoder::{
        Instruction,
        reencode::{self as reencode, Reencode},
    },
};

/// Re-exported so callers can use `sysv::SysVState` like other backends.
pub use crate::naive::State as SysVState;

const FP: Reg = Reg(11);
const SP: Reg = Reg(13);
const LR: Reg = Reg(14);
const R0: Reg = Reg(0);
const R1: Reg = Reg(1);

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: r,
        size: MemorySize::_32,
    })
}
fn lit(v: u64) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Lit(v))
}
fn mem(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_32,
        },
        offset: None,
        disp,
        size: MemorySize::_32,
    }
}

/// Extension trait for AAPCS-compatible functions.
pub trait SysVWriterExt<Context>: Writer<ArmLabel, Context> + NaiveExt<Context> {
    fn sysv_emit_epilogue(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        state: &mut State<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // i64 result in r0:r1 (low:high).
        if state.num_returns > 0 {
            self.pop_i64(ctx, arch, R0, R1)?;
        }
        self.mov(ctx, arch, &reg(SP), &reg(FP))?;
        self.ldr(ctx, arch, &reg(FP), &mem(SP, 0))?;
        self.ldr(ctx, arch, &reg(LR), &mem(SP, 4))?;
        self.add(ctx, arch, &reg(SP), &reg(SP), &lit(8))?;
        self.ret(ctx, arch)
    }

    fn sysv_handle_insn<E>(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
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
            Instruction::Return => self.sysv_emit_epilogue(ctx, arch, state),
            other => {
                self.handle_op_(ctx, arch, state, func_imports, &[], &[], other, rewriter, target)
            }
        }
    }

    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
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

                self.set_label(ctx, arch, ArmLabel::Func { r#fn: *id })
                    .map_err(Err::from)?;

                // AAPCS: push {fp, lr}; mov fp, sp
                self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(8))
                    .map_err(Err::from)?;
                self.str(ctx, arch, &reg(FP), &mem(SP, 0))
                    .map_err(Err::from)?;
                self.str(ctx, arch, &reg(LR), &mem(SP, 4))
                    .map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP))
                    .map_err(Err::from)?;

                let locals_slots =
                    (state.local_count as i32) + (state.control_depth as i32) * 2 + 4;
                let alloc_bytes = locals_slots * WASM_SLOT;
                if alloc_bytes > 0 {
                    self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(alloc_bytes as u64))
                        .map_err(Err::from)?;
                }
                Ok(())
            }
            MachOperator::Local { count, .. } => {
                for _ in 0..*count {
                    self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(WASM_SLOT as u64))
                        .map_err(Err::from)?;
                    self.mov_imm(ctx, arch, &reg(R0), 0).map_err(Err::from)?;
                    self.str(ctx, arch, &reg(R0), &mem(SP, 0))
                        .map_err(Err::from)?;
                    self.str(ctx, arch, &reg(R0), &mem(SP, 4))
                        .map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }
            MachOperator::StartBody | MachOperator::EndBody => Ok(()),
            MachOperator::Instruction { op: insn, .. } => self
                .sysv_handle_insn(ctx, arch, state, func_imports, insn, rewriter, target)
                .map_err(Err::from),
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

impl<T: Writer<ArmLabel, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context>
    for T
{
}
