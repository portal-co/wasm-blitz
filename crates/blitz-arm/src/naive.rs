//! Thin naive AArch32 (ILP32) codegen — stack-based WASM operand stack.
//!
//! # Host pointer stride
//!
//! `HOST_PTR_STRIDE = 4`: cross-shard / table function-pointer loads use
//! 32-bit host addresses. WASM operand and local slots stay 8 bytes
//! (`WASM_SLOT`) so i64 values fit in a single slot (low word at `[sp]`,
//! high word at `[sp+4]`).

#![allow(dead_code)]

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
        self,
        reencode::{self as reencode, Reencode},
        Instruction,
    },
};

/// Host pointer / fn-ptr table stride (ILP32).
pub const HOST_PTR_STRIDE: i32 = 4;
/// WASM operand / local slot size in bytes.
pub const WASM_SLOT: i32 = 8;

const R0: Reg = Reg(0);
const R1: Reg = Reg(1);
const R2: Reg = Reg(2);
const R3: Reg = Reg(3);
/// AAPCS frame pointer (r11).
const FP: Reg = Reg(11);
const SP: Reg = Reg(13);
const LR: Reg = Reg(14);
/// Static Context Register — r10 (callee-saved). Host pointer tables ×4.
pub const SCR: Reg = Reg(10);

#[derive(Default)]
pub struct State<'a> {
    pub label_index: usize,
    pub local_count: usize,
    pub param_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub body: u32,
    pub _phantom: core::marker::PhantomData<&'a ()>,
}

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

pub trait WriterExt<Context>: Writer<ArmLabel, Context> {
    fn push_i64(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        low: Reg,
        high: Reg,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(WASM_SLOT as u64))?;
        self.str(ctx, arch, &reg(low), &mem(SP, 0))?;
        self.str(ctx, arch, &reg(high), &mem(SP, 4))
    }

    fn pop_i64(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        low: Reg,
        high: Reg,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.ldr(ctx, arch, &reg(low), &mem(SP, 0))?;
        self.ldr(ctx, arch, &reg(high), &mem(SP, 4))?;
        self.add(ctx, arch, &reg(SP), &reg(SP), &lit(WASM_SLOT as u64))
    }

    fn push_const_i64(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        value: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let low = value as u32 as u64;
        let high = (value >> 32) as u32 as u64;
        self.mov_imm(ctx, arch, &reg(R0), low)?;
        self.mov_imm(ctx, arch, &reg(R1), high)?;
        self.push_i64(ctx, arch, R0, R1)
    }

    fn handle_op_<E>(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        _sigs: &[wasm_encoder::FuncType],
        _tags: &[u32],
        op: &Instruction<'_>,
        _rewriter: &mut (dyn Reencode<Error = E> + '_),
        _target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        match op {
            Instruction::I32Const(v) => {
                self.push_const_i64(ctx, arch, *v as u32 as u64)?;
            }
            Instruction::I64Const(v) => {
                self.push_const_i64(ctx, arch, *v as u64)?;
            }
            Instruction::I32Add => {
                // pop b, pop a, push a+b (low words); high cleared.
                self.pop_i64(ctx, arch, R2, R3)?; // b
                self.pop_i64(ctx, arch, R0, R1)?; // a
                self.add(ctx, arch, &reg(R0), &reg(R0), &reg(R2))?;
                self.mov_imm(ctx, arch, &reg(R1), 0)?;
                self.push_i64(ctx, arch, R0, R1)?;
            }
            Instruction::I32Load(memarg) => {
                self.pop_i64(ctx, arch, R0, R1)?; // address in r0
                if memarg.offset != 0 {
                    self.mov_imm(ctx, arch, &reg(R2), memarg.offset as u32 as u64)?;
                    self.add(ctx, arch, &reg(R0), &reg(R0), &reg(R2))?;
                }
                self.ldr(ctx, arch, &reg(R0), &mem(R0, 0))?;
                self.mov_imm(ctx, arch, &reg(R1), 0)?;
                self.push_i64(ctx, arch, R0, R1)?;
            }
            Instruction::I32Store(memarg) => {
                self.pop_i64(ctx, arch, R2, R3)?; // value
                self.pop_i64(ctx, arch, R0, R1)?; // address
                if memarg.offset != 0 {
                    self.mov_imm(ctx, arch, &reg(R1), memarg.offset as u32 as u64)?;
                    self.add(ctx, arch, &reg(R0), &reg(R0), &reg(R1))?;
                }
                self.str(ctx, arch, &reg(R2), &mem(R0, 0))?;
            }
            Instruction::Call(function_index) | Instruction::ReturnCall(function_index) => {
                match func_imports.get(*function_index as usize) {
                    Some((module, name)) => {
                        let sym = alloc::format!("{module}__{name}");
                        self.bl_label(ctx, arch, ArmLabel::External { name: sym })?;
                    }
                    None => {
                        let idx = *function_index - func_imports.len() as u32;
                        self.bl_label(ctx, arch, ArmLabel::Func { r#fn: idx })?;
                    }
                }
            }
            Instruction::Return => {
                // Move results into r0/r1 then tear down the naive frame.
                if state.num_returns > 0 {
                    self.pop_i64(ctx, arch, R0, R1)?;
                }
                // SP := FP; restore saved FP; pop LR; bx lr
                self.mov(ctx, arch, &reg(SP), &reg(FP))?;
                self.ldr(ctx, arch, &reg(FP), &mem(SP, 0))?;
                self.ldr(ctx, arch, &reg(LR), &mem(SP, 4))?;
                self.add(ctx, arch, &reg(SP), &reg(SP), &lit(8))?;
                self.ret(ctx, arch)?;
            }
            Instruction::End => {
                // Function-level End is a no-op; mach_operators injects Return.
            }
            Instruction::Drop => {
                self.add(ctx, arch, &reg(SP), &reg(SP), &lit(WASM_SLOT as u64))?;
            }
            other => {
                // Thin Phase-1: leave a clear panic for unimplemented ops.
                panic!("unimplemented WASM instruction in ARM naive handle_op: {other:?}");
            }
        }
        Ok(())
    }

    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: ArmArch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[wasm_encoder::FuncType],
        tags: &[u32],
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

                // push lr: str lr, [sp, #-4]!  →  sub sp,#4; str lr,[sp]
                // then save fp and establish frame:
                //   sub sp,#4; str fp,[sp]; mov fp,sp
                self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(4))
                    .map_err(Err::from)?;
                self.str(ctx, arch, &reg(LR), &mem(SP, 0))
                    .map_err(Err::from)?;
                self.sub(ctx, arch, &reg(SP), &reg(SP), &lit(4))
                    .map_err(Err::from)?;
                self.str(ctx, arch, &reg(FP), &mem(SP, 0))
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
            MachOperator::Instruction { op, .. } => self
                .handle_op_(ctx, arch, state, func_imports, sigs, tags, op, rewriter, target)
                .map_err(Err::from),
            MachOperator::Operator { op, .. } => {
                if let Some(op) = op {
                    let insn = rewriter.instruction(op.clone())?;
                    self.handle_op_(
                        ctx,
                        arch,
                        state,
                        func_imports,
                        sigs,
                        tags,
                        &insn,
                        rewriter,
                        target,
                    )
                    .map_err(Into::into)
                } else {
                    Ok(())
                }
            }
            other => panic!("unimplemented MachOperator in ARM naive handle_op: {other:?}"),
        }
    }
}

impl<T: Writer<ArmLabel, Context> + ?Sized, Context> WriterExt<Context> for T {}
