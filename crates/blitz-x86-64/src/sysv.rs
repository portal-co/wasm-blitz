//! System V AMD64 ABI code generation for the x86-64 backend.
//!
//! This module provides a `SysVWriterExt` trait that generates x86-64 functions
//! following the System V AMD64 ABI, making them directly callable from C and
//! other System V–conforming code.
//!
//! # Calling Convention (System V AMD64)
//!
//! Integer arguments: RDI, RSI, RDX, RCX, R8, R9 (then stack, right-to-left)
//! Return value: RAX (single), RAX + RDX (pair)
//! Frame: `push rbp; mov rbp, rsp; sub rsp, N`
//!
//! Register index mapping in the blitz-x86-64 backend:
//!   Reg(0)=rax  Reg(1)=rcx  Reg(2)=rdx  Reg(3)=rbx
//!   Reg(4)=rsp  Reg(5)=rbp  Reg(6)=rsi  Reg(7)=rdi
//!   Reg(8)=r8   Reg(9)=r9
//!
//! # Design
//!
//! The SysV entry copies register arguments into a standard rbp-relative local
//! frame so that arithmetic, memory, and control-flow handlers (which use RSP as
//! the WASM operand stack) continue to work unchanged.  Only `StartFn`, local
//! variable access, and `Return` differ from the naive backend.  This is the
//! "full rewrite ABI" described in docs/abi.md.

use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
extern crate alloc;

use portal_solutions_asm_x86_64::{
    out::{Writer, arg::{ArgKind, MemArg, MemArgKind}},
    ConditionCode, RegisterClass, X64Arch,
};
use portal_solutions_blitz_common::{
    asm::Reg,
    asm::common::mem::MemorySize,
    ops::{FnData, MachOperator},
    wasm_encoder::{self, FuncType, Instruction, reencode::Reencode},
};

use crate::{X64Label, RSP};
use crate::naive::WriterExt as NaiveExt;

// ---- register shortcuts ----
const RAX: Reg = Reg(0);
const RBP: Reg = Reg(5);

/// Integer argument registers in System V AMD64 ABI order.
const ARG_REGS: [Reg; 6] = [
    Reg(7), // rdi
    Reg(6), // rsi
    Reg(2), // rdx
    Reg(1), // rcx
    Reg(8), // r8
    Reg(9), // r9
];

fn mem64(base: Reg, disp: u32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// State tracker for System V x86-64 code generation.
#[derive(Default)]
pub struct SysVState {
    pub param_count: usize,
    pub ret_count: usize,
    pub local_count: usize,
    pub label_index: usize,
    pub if_stack: Vec<crate::naive::Endable>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
}

/// Extension trait for generating System V AMD64-compatible functions.
///
/// All arithmetic, memory, and control-flow instructions delegate to the naive
/// backend's `_handle_op`.  Only the function boundary and local variable access
/// are different.
pub trait SysVWriterExt<Context>: Writer<X64Label, Context> + NaiveExt<Context> {

    /// Load local variable `n` from the rbp-relative frame into register `dest`.
    fn sysv_load_local(&mut self, ctx: &mut Context, arch: X64Arch, dest: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        // Local n is at [rbp - (n+1)*8]
        let disp = 0u32.wrapping_sub(((n as isize + 1) * 8) as u32);
        self.mov(ctx, arch, &dest, &mem64(RBP, disp))
    }

    /// Store `src` into local variable `n` in the rbp-relative frame.
    fn sysv_store_local(&mut self, ctx: &mut Context, arch: X64Arch, src: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = 0u32.wrapping_sub(((n as isize + 1) * 8) as u32);
        self.mov(ctx, arch, &mem64(RBP, disp), &src)
    }

    /// Returns the value at top of WASM operand stack (RSP) without popping.
    fn sysv_peek(&mut self, ctx: &mut Context, arch: X64Arch, dest: Reg)
        -> Result<(), Self::Error>
    {
        self.mov(ctx, arch, &dest, &mem64(RSP, 0))
    }

    /// Handle an instruction using the SysV ABI (overrides local access and Return).
    fn sysv_handle_insn(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // Redirect body-skip jump if needed (same as naive _handle_op preamble)
        if target != state.body {
            let skip_idx = *state.body_labels.entry(state.body).or_insert_with(|| {
                state.label_index += 1;
                state.label_index - 1
            });
            self.jmp_label(ctx, arch, X64Label::Indexed { idx: skip_idx })?;
            state.body = target;
            if let Some(idx) = state.body_labels.remove(&state.body) {
                self.set_label(ctx, arch, X64Label::Indexed { idx })?;
            }
        }

        match op {
            // ---- local access (SysV rbp-relative frame) ----
            Instruction::LocalGet(n) => {
                self.sysv_load_local(ctx, arch, RAX, *n as usize)?;
                self.push(ctx, arch, &RAX)
            }
            Instruction::LocalSet(n) => {
                self.pop(ctx, arch, &RAX)?;
                self.sysv_store_local(ctx, arch, RAX, *n as usize)
            }
            Instruction::LocalTee(n) => {
                // Peek (don't pop), store
                self.sysv_peek(ctx, arch, RAX)?;
                self.sysv_store_local(ctx, arch, RAX, *n as usize)
            }

            // ---- Return: pop result into rax, then standard epilogue ----
            Instruction::Return => {
                // Pop single result (if any) into rax
                if state.ret_count > 0 {
                    self.pop(ctx, arch, &RAX)?;
                }
                // Second return value into rdx (for 2-return functions)
                if state.ret_count > 1 {
                    self.pop(ctx, arch, &Reg(2))?;  // rdx
                }
                // Standard x86-64 epilogue
                self.mov(ctx, arch, &RSP, &RBP)?;  // rsp = rbp
                self.pop(ctx, arch, &RBP)?;         // pop rbp
                self.ret(ctx, arch)
            }

            // ---- Everything else: naive _handle_op ----
            other => {
                // Build a temporary naive::State to delegate to _handle_op.
                // We re-use label_index and if_stack from SysVState.
                let mut naive_state = crate::naive::State {
                    local_count: state.local_count,
                    num_returns: state.ret_count,
                    control_depth: 0,
                    label_index: state.label_index,
                    if_stack: core::mem::take(&mut state.if_stack),
                    body: state.body,
                    body_labels: core::mem::take(&mut state.body_labels),
                };
                let result = self._handle_op(ctx, arch, &mut naive_state, func_imports, other, target);
                state.label_index = naive_state.label_index;
                state.if_stack = naive_state.if_stack;
                state.body = naive_state.body;
                state.body_labels = naive_state.body_labels;
                result
            }
        }
    }

    /// Handle a `MachOperator` using the SysV ABI.
    fn sysv_handle_op<E>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), portal_solutions_blitz_common::HandleOpError<E>>
    where
        Self::Error: Into<portal_solutions_blitz_common::HandleOpError<E>>,
        Self: Sized,
    {
        use portal_solutions_blitz_common::wasm_encoder;

        match op {
            MachOperator::StartFn { id, data } => {
                state.param_count = data.num_params;
                state.ret_count = data.num_returns;
                state.local_count = data.num_params;

                self.set_label(ctx, arch, X64Label::Indexed {
                    idx: *id as usize | (1 << 28),
                }).map_err(Into::into)?;

                self.push(ctx, arch, &RBP).map_err(Into::into)?;
                self.mov(ctx, arch, &RBP, &RSP).map_err(Into::into)?;

                let frame_sz = (data.num_params + 16) * 8;
                let frame_sz = (frame_sz + 15) & !15;
                self.mov64(ctx, arch, &RAX, frame_sz as u64).map_err(Into::into)?;
                self.sub(ctx, arch, &RSP, &RAX).map_err(Into::into)?;

                for i in 0..data.num_params.min(6) {
                    self.sysv_store_local(ctx, arch, ARG_REGS[i], i).map_err(Into::into)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov64(ctx, arch, &RAX, 0).map_err(Into::into)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, RAX, state.local_count).map_err(Into::into)?;
                    state.local_count += 1;
                }
                Ok(())
            }

            MachOperator::StartBody | MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, target)
                    .map_err(Into::into)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, target)
                    .map_err(Into::into)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Writer<X64Label, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}
