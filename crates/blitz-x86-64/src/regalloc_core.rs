//! Regalloc-backed WASM instruction dispatch for x86-64, shared by every ABI
//! that wants register-allocated data flow instead of `naive.rs`'s pure
//! push/pop-every-value stack machine.
//!
//! Unlike `naive.rs`/`fast.rs` (the abandoned attempt — see
//! `docs/regalloc-unification-plan.md`), this module has no `State` of its
//! own and no control-flow handling. It exposes [`handle_op`], a free
//! function that tries to handle one WASM instruction through
//! `portal_solutions_blitz_codegen::regalloc_frontend` and reports whether it
//! did, so a caller (`sysv.rs`) can keep its own control-flow arms and fall
//! back to `naive::_handle_op` for anything not yet covered here — as long as
//! it flushes the same regalloc state first (see `crate::codegen::RegAllocW`'s
//! `ControlFlowWriter::flush`), since `naive.rs`'s fallback path assumes every
//! operand is already on the real RSP-based stack.

use alloc::vec::Vec;
use portal_solutions_asm_regalloc::{Cmd, RegAlloc};
use portal_solutions_asm_x86_64::{
    RegisterClass, X64Arch,
    out::{
        Writer, WriterCore,
        arg::{ArgKind, MemArgKind},
    },
    regalloc as x86_regalloc,
};
use portal_solutions_blitz_common::asm::{Reg, common::mem::MemorySize};
use portal_solutions_blitz_common::wasm_encoder::Instruction;

/// Shared regalloc `Frames` adapter — see `portal_solutions_blitz_codegen::regalloc_adapter`.
pub use portal_solutions_blitz_codegen::regalloc_adapter::Frames;

/// Convenience alias for the x86-64 regalloc-core's allocator state.
pub type X64RegAlloc = RegAlloc<x86_regalloc::RegKind, 32, Frames<x86_regalloc::RegKind, 32>>;

pub(crate) fn emit_cmds<E, Context, W>(
    writer: &mut W,
    ctx: &mut Context,
    arch: X64Arch,
    it: impl Iterator<Item = Cmd<x86_regalloc::RegKind>>,
) -> Result<(), E>
where
    E: core::error::Error,
    W: WriterCore<Context, Error = E>,
{
    for cmd in it {
        x86_regalloc::process_cmd(writer, ctx, arch, &cmd, None)?;
    }
    Ok(())
}

/// Try to handle `op` through the regalloc-backed core. Returns `Ok(true)` if
/// it did (nothing more for the caller to do for this instruction).
///
/// `regalloc` is the same `Option<X64RegAlloc>` the caller's `flush()` (via
/// `RegAllocW::ControlFlowWriter`) operates on — this function lazily
/// initializes it on first use, exactly like every other backend in this
/// unification effort.
pub fn handle_op<W, Context>(
    writer: &mut W,
    ctx: &mut Context,
    arch: X64Arch,
    regalloc: &mut Option<X64RegAlloc>,
    op: &Instruction<'_>,
) -> Result<bool, W::Error>
where
    W: WriterCore<Context> + Writer<crate::X64Label, Context>,
{
    use portal_solutions_blitz_codegen::regalloc_frontend::{
        binop, pop_to_local, push_const, push_local,
    };

    let mut rw = crate::codegen::RegAllocW {
        writer,
        ctx,
        arch,
        regalloc,
    };
    match op {
        Instruction::I32Const(v) => {
            let v = *v as u32 as u64;
            push_const(&mut rw, x86_regalloc::RegKind::Int, |rw, reg| {
                rw.writer.mov64(rw.ctx, rw.arch, &Reg(reg), v)
            })?;
            Ok(true)
        }
        Instruction::I64Const(v) => {
            let v = *v as u64;
            push_const(&mut rw, x86_regalloc::RegKind::Int, |rw, reg| {
                rw.writer.mov64(rw.ctx, rw.arch, &Reg(reg), v)
            })?;
            Ok(true)
        }
        Instruction::LocalGet(n) => {
            push_local(&mut rw, x86_regalloc::RegKind::Int, *n)?;
            Ok(true)
        }
        Instruction::LocalSet(n) => {
            pop_to_local(&mut rw, x86_regalloc::RegKind::Int, *n)?;
            Ok(true)
        }
        Instruction::I32Add | Instruction::I64Add => {
            let is32 = matches!(op, Instruction::I32Add);
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                rw.writer.lea(
                    rw.ctx,
                    rw.arch,
                    &Reg(dst),
                    &MemArgKind::Mem {
                        base: Reg(dst),
                        offset: Some((Reg(rhs), 1)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                if is32 {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(dst),
                        size: MemorySize::_32,
                    });
                    rw.writer.mov(rw.ctx, rw.arch, &r, &r)?;
                }
                Ok(())
            })?;
            Ok(true)
        }
        Instruction::I32Sub | Instruction::I64Sub => {
            let is32 = matches!(op, Instruction::I32Sub);
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                // dst holds `a` (lhs), rhs holds `b`. ~b + a + 1 == a - b.
                rw.writer.not(rw.ctx, rw.arch, &Reg(rhs))?;
                rw.writer.lea(
                    rw.ctx,
                    rw.arch,
                    &Reg(dst),
                    &MemArgKind::Mem {
                        base: Reg(dst),
                        offset: Some((Reg(rhs), 1)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                        segment: Default::default(),
                    },
                )?;
                if is32 {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(dst),
                        size: MemorySize::_32,
                    });
                    rw.writer.mov(rw.ctx, rw.arch, &r, &r)?;
                }
                Ok(())
            })?;
            Ok(true)
        }
        // Bitwise ops — two-operand form, no RAX/RDX clobber (unlike `mul`).
        Instruction::I32And | Instruction::I64And => {
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                rw.writer.and(rw.ctx, rw.arch, &Reg(dst), &Reg(rhs))
            })?;
            Ok(true)
        }
        Instruction::I32Or | Instruction::I64Or => {
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                rw.writer.or(rw.ctx, rw.arch, &Reg(dst), &Reg(rhs))
            })?;
            Ok(true)
        }
        Instruction::I32Xor | Instruction::I64Xor => {
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                rw.writer.eor(rw.ctx, rw.arch, &Reg(dst), &Reg(rhs))
            })?;
            Ok(true)
        }
        // Two-operand `imul` — no RAX/RDX implicit clobber (unlike one-operand `mul`).
        Instruction::I32Mul | Instruction::I64Mul => {
            let is32 = matches!(op, Instruction::I32Mul);
            binop(&mut rw, x86_regalloc::RegKind::Int, |rw, dst, rhs| {
                rw.writer.mul(rw.ctx, rw.arch, &Reg(dst), &Reg(rhs))?;
                if is32 {
                    let r = MemArgKind::NoMem(ArgKind::Reg {
                        reg: Reg(dst),
                        size: MemorySize::_32,
                    });
                    rw.writer.mov(rw.ctx, rw.arch, &r, &r)?;
                }
                Ok(())
            })?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
