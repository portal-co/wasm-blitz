//! Thin naive i686 (ILP32) codegen — stack-based WASM operand stack.
//!
//! # Host pointer stride
//!
//! `HOST_PTR_STRIDE = 4`: SCR / table function-pointer loads use 32-bit host
//! addresses. WASM operand and local slots stay 8 bytes (`WASM_SLOT`).

#![allow(dead_code)]

use crate::I686Label;
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_x86::{
    RegisterClass, X86Arch,
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

/// Host pointer / fn-ptr table stride (ILP32). SCR note: pointer tables ×4.
pub const HOST_PTR_STRIDE: i32 = 4;
/// WASM operand / local slot size in bytes.
pub const WASM_SLOT: i32 = 8;

const EAX: Reg = Reg(0);
const ECX: Reg = Reg(1);
const EDX: Reg = Reg(2);
const ESP: Reg = Reg(4);
const EBP: Reg = Reg(5);
/// Static Context Register — esi (callee-saved). Host pointer tables ×4.
pub const SCR: Reg = Reg(6);

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

fn mem(base: Reg, disp: u32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_32,
        },
        offset: None,
        disp,
        size: MemorySize::_32,
        reg_class: RegisterClass::Gpr,
    }
}

pub trait WriterExt<Context>: Writer<I686Label, Context> {
    /// Push an 8-byte WASM slot: high word first, then low (so `[esp]=low`).
    fn push_i64(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
        low: Reg,
        high: Reg,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.push(ctx, arch, &reg(high))?;
        self.push(ctx, arch, &reg(low))
    }

    fn pop_i64(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
        low: Reg,
        high: Reg,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.pop(ctx, arch, &reg(low))?;
        self.pop(ctx, arch, &reg(high))
    }

    fn push_const_i64(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
        value: u64,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let low = value as u32 as u64;
        let high = (value >> 32) as u32 as u64;
        self.push(ctx, arch, &lit(high))?;
        self.push(ctx, arch, &lit(low))
    }

    fn handle_op_<E>(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
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
                self.pop_i64(ctx, arch, ECX, EDX)?; // b
                self.pop_i64(ctx, arch, EAX, EDX)?; // a (edx clobbered then cleared)
                self.add(ctx, arch, &reg(EAX), &reg(ECX))?;
                self.xor(ctx, arch, &reg(EDX), &reg(EDX))?;
                self.push_i64(ctx, arch, EAX, EDX)?;
            }
            Instruction::I32Load(memarg) => {
                self.pop_i64(ctx, arch, EAX, EDX)?;
                if memarg.offset != 0 {
                    self.add(ctx, arch, &reg(EAX), &lit(memarg.offset as u32 as u64))?;
                }
                self.mov(ctx, arch, &reg(EAX), &mem(EAX, 0))?;
                self.xor(ctx, arch, &reg(EDX), &reg(EDX))?;
                self.push_i64(ctx, arch, EAX, EDX)?;
            }
            Instruction::I32Store(memarg) => {
                self.pop_i64(ctx, arch, ECX, EDX)?; // value
                self.pop_i64(ctx, arch, EAX, EDX)?; // address
                if memarg.offset != 0 {
                    self.add(ctx, arch, &reg(EAX), &lit(memarg.offset as u32 as u64))?;
                }
                self.mov(ctx, arch, &mem(EAX, 0), &reg(ECX))?;
            }
            Instruction::Call(function_index) | Instruction::ReturnCall(function_index) => {
                match func_imports.get(*function_index as usize) {
                    Some((module, name)) => {
                        let sym = alloc::format!("{module}__{name}");
                        self.call_label(ctx, arch, I686Label::External { name: sym })?;
                    }
                    None => {
                        let idx = *function_index - func_imports.len() as u32;
                        self.call_label(ctx, arch, I686Label::Func { r#fn: idx })?;
                    }
                }
            }
            Instruction::Return => {
                if state.num_returns > 0 {
                    // i64 in edx:eax (high:low)
                    self.pop_i64(ctx, arch, EAX, EDX)?;
                }
                self.leave(ctx, arch)?;
                self.ret(ctx, arch)?;
            }
            Instruction::End => {}
            Instruction::Drop => {
                self.add(ctx, arch, &reg(ESP), &lit(WASM_SLOT as u64))?;
            }
            other => {
                panic!("unimplemented WASM instruction in i686 naive handle_op: {other:?}");
            }
        }
        Ok(())
    }

    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X86Arch,
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

                self.set_label(ctx, arch, I686Label::Func { r#fn: *id })
                    .map_err(Err::from)?;

                // push ebp; mov ebp, esp; sub esp for frame
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
            other => panic!("unimplemented MachOperator in i686 naive handle_op: {other:?}"),
        }
    }
}

impl<T: Writer<I686Label, Context> + ?Sized, Context> WriterExt<Context> for T {}
