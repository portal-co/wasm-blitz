//! i686 System V ABI entry path.
//!
//! Prologue: `push ebp; mov ebp, esp; sub esp, N`.
//! Return: i64 in `edx:eax` (high:low), then `leave; ret`.

use crate::naive::{State, WriterExt as NaiveExt, WASM_SLOT};
use crate::I686Label;
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_x86::{
    X86Arch,
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

const EAX: Reg = Reg(0);
const EDX: Reg = Reg(2);
const ESP: Reg = Reg(4);
const EBP: Reg = Reg(5);

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: r,
        size: MemorySize::_32,
    })
}
fn lit(v: u64) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Lit(v))
}

/// Extension trait for i686 SysV-compatible functions.
pub trait SysVWriterExt<Context>: Writer<I686Label, Context> + NaiveExt<Context> {
    fn sysv_emit_epilogue(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
        state: &mut State<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // i64 in edx:eax (high:low).
        if state.num_returns > 0 {
            self.pop_i64(ctx, arch, EAX, EDX)?;
        }
        self.leave(ctx, arch)?;
        self.ret(ctx, arch)
    }

    fn sysv_handle_insn<E>(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
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
        arch: X86Arch,
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

                self.set_label(ctx, arch, I686Label::Func { r#fn: *id })
                    .map_err(Err::from)?;

                self.push(ctx, arch, &reg(EBP)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(EBP), &reg(ESP))
                    .map_err(Err::from)?;

                let locals_slots =
                    (state.local_count as i32) + (state.control_depth as i32) * 2 + 4;
                let alloc_bytes = locals_slots * WASM_SLOT;
                if alloc_bytes > 0 {
                    self.sub(ctx, arch, &reg(ESP), &lit(alloc_bytes as u64))
                        .map_err(Err::from)?;
                }
                Ok(())
            }
            MachOperator::Local { count, .. } => {
                for _ in 0..*count {
                    self.push(ctx, arch, &lit(0)).map_err(Err::from)?;
                    self.push(ctx, arch, &lit(0)).map_err(Err::from)?;
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

impl<T: Writer<I686Label, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context>
    for T
{
}
