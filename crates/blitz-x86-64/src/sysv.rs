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
    ops::{FnData, MachOperator, TracingHooks},
    wasm_encoder::{self, FuncType, Instruction, reencode::{self as reencode, Reencode}},
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

/// One entry in the SysV control-flow stack (no CTX required).
#[derive(Clone, Copy)]
pub enum SysVCtrl {
    /// `loop`: label at the top (already set); `Br` jumps back.
    Loop(usize),
    /// `block`: label at the exit (not yet set); `Br` jumps forward, `End` sets it.
    Block(usize),
    /// `if`: labels `base`, `base+1` (else), `base+2` (after). `else_seen` tracks Else.
    If { base: usize, else_seen: bool },
}

impl SysVCtrl {
    /// The label that a `Br` targeting this block should jump to.
    fn br_target(self) -> usize {
        match self {
            SysVCtrl::Loop(top) => top,
            SysVCtrl::Block(exit) => exit,
            SysVCtrl::If { base, .. } => base + 2,
        }
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
    pub ctrl_stack: Vec<SysVCtrl>,
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
            // First-instruction guard: see naive.rs.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
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

            // ---- Return: always emit SysV epilogue regardless of block depth ----
            Instruction::Return => {
                if state.ret_count > 0 { self.pop(ctx, arch, &RAX)?; }
                if state.ret_count > 1 { self.pop(ctx, arch, &Reg(2))?; }
                self.mov(ctx, arch, &RSP, &RBP)?;
                self.pop(ctx, arch, &RBP)?;
                self.ret(ctx, arch)
            }
            // ---- Function-level End (empty ctrl_stack) ----
            Instruction::End if state.if_stack.is_empty() => {
                if state.ret_count > 0 { self.pop(ctx, arch, &RAX)?; }
                if state.ret_count > 1 { self.pop(ctx, arch, &Reg(2))?; }
                self.mov(ctx, arch, &RSP, &RBP)?;
                self.pop(ctx, arch, &RBP)?;
                self.ret(ctx, arch)
            }

            // ---- Control flow — CTX-free (no naive delegation) ----

            Instruction::Loop(_) => {
                let i = state.label_index;
                state.label_index += 1;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                state.if_stack.push(crate::naive::Endable::Br);
                state.ctrl_stack.push(SysVCtrl::Loop(i));
                Ok(())
            }
            Instruction::Block(_) => {
                let i = state.label_index;
                state.label_index += 1;
                state.if_stack.push(crate::naive::Endable::Br);
                state.ctrl_stack.push(SysVCtrl::Block(i));
                Ok(())
            }
            Instruction::If(_) => {
                let i = state.label_index;
                state.label_index += 3;
                self.pop(ctx, arch, &RAX)?;
                self.cmp0(ctx, arch, &RAX)?;
                self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: i + 1 })?;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                state.if_stack.push(crate::naive::Endable::If { idx: i });
                state.ctrl_stack.push(SysVCtrl::If { base: i, else_seen: false });
                Ok(())
            }
            Instruction::Else => {
                let Some(SysVCtrl::If { base: i, else_seen }) = state.ctrl_stack.last_mut() else {
                    return Ok(());
                };
                let i = *i;
                *else_seen = true;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i + 2 })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })
            }
            Instruction::End if !state.if_stack.is_empty() => {
                if state.ctrl_stack.is_empty() {
                    // TryTable's End: ctrl_stack has no entry for it (naive tracks it in
                    // if_stack only). Delegate to naive so TryTable can emit its dispatch stub.
                    let other = Instruction::End;
                    let mut naive_state = crate::naive::State {
                        local_count: state.local_count,
                        num_returns: state.ret_count,
                        control_depth: 0,
                        label_index: state.label_index,
                        if_stack: core::mem::take(&mut state.if_stack),
                        body: state.body,
                        body_labels: core::mem::take(&mut state.body_labels),
                        tracing: None,
                    };
                    let result = self._handle_op(ctx, arch, &mut naive_state, func_imports, &[], &[], &other, target);
                    state.label_index = naive_state.label_index;
                    state.if_stack = naive_state.if_stack;
                    state.body = naive_state.body;
                    state.body_labels = naive_state.body_labels;
                    return result;
                }
                state.if_stack.pop();
                let ctrl = state.ctrl_stack.pop().unwrap();
                match ctrl {
                    SysVCtrl::Loop(_) => Ok(()),
                    SysVCtrl::Block(exit) => {
                        self.set_label(ctx, arch, X64Label::Indexed { idx: exit })
                    }
                    SysVCtrl::If { base: i, else_seen } => {
                        if !else_seen {
                            self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })?;
                        }
                        self.set_label(ctx, arch, X64Label::Indexed { idx: i + 2 })
                    }
                }
            }
            Instruction::Br(n) => {
                let n = *n as usize;
                if let Some(ctrl) = state.ctrl_stack.len().checked_sub(n + 1)
                    .and_then(|idx| state.ctrl_stack.get(idx))
                    .copied()
                {
                    self.jmp_label(ctx, arch, X64Label::Indexed { idx: ctrl.br_target() })
                } else {
                    // Br targeting the function block = return
                    if state.ret_count > 0 { self.pop(ctx, arch, &RAX)?; }
                    if state.ret_count > 1 { self.pop(ctx, arch, &Reg(2))?; }
                    self.mov(ctx, arch, &RSP, &RBP)?;
                    self.pop(ctx, arch, &RBP)?;
                    self.ret(ctx, arch)
                }
            }
            Instruction::BrIf(n) => {
                let n = *n as usize;
                let skip = state.label_index;
                state.label_index += 1;
                self.pop(ctx, arch, &RAX)?;
                self.cmp0(ctx, arch, &RAX)?;
                if let Some(ctrl) = state.ctrl_stack.len().checked_sub(n + 1)
                    .and_then(|idx| state.ctrl_stack.get(idx))
                    .copied()
                {
                    self.jcc_label(ctx, arch, ConditionCode::NE, X64Label::Indexed { idx: ctrl.br_target() })?;
                } else {
                    // BrIf targeting the function block = conditional return
                    self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: skip })?;
                    if state.ret_count > 0 { self.pop(ctx, arch, &RAX)?; }
                    if state.ret_count > 1 { self.pop(ctx, arch, &Reg(2))?; }
                    self.mov(ctx, arch, &RSP, &RBP)?;
                    self.pop(ctx, arch, &RBP)?;
                    self.ret(ctx, arch)?;
                }
                self.set_label(ctx, arch, X64Label::Indexed { idx: skip })
            }
            Instruction::BrTable(targets, default) => {
                // Pop selector once; for each target: if selector==0 branch, else decrement.
                self.pop(ctx, arch, &RAX)?;
                let ctrl = &state.ctrl_stack;
                macro_rules! do_br {
                    ($depth:expr) => {{
                        let n = $depth as usize;
                        let tgt = ctrl.len().checked_sub(n + 1).and_then(|i| ctrl.get(i)).map(|c| c.br_target());
                        if let Some(lbl) = tgt {
                            self.jmp_label(ctx, arch, X64Label::Indexed { idx: lbl })?;
                        } else {
                            self.mov(ctx, arch, &RSP, &RBP)?;
                            self.pop(ctx, arch, &RBP)?;
                            self.ret(ctx, arch)?;
                        }
                    }};
                }
                for (arm_idx, &depth) in targets.iter().enumerate() {
                    let skip = state.label_index;
                    state.label_index += 1;
                    self.cmp0(ctx, arch, &RAX)?;
                    self.jcc_label(ctx, arch, ConditionCode::NE, X64Label::Indexed { idx: skip })?;
                    do_br!(depth);
                    self.set_label(ctx, arch, X64Label::Indexed { idx: skip })?;
                    if arm_idx + 1 < targets.len() {
                        self.lea(ctx, arch, &RAX, &MemArgKind::Mem {
                            base: RAX, offset: None, disp: 0u32.wrapping_sub(1),
                            size: MemorySize::_64, reg_class: RegisterClass::Gpr,
                        })?;
                    }
                }
                do_br!(*default);
                Ok(())
            }

            // ---- Everything else: naive _handle_op (no CTX ops in these paths) ----
            other => {
                let mut naive_state = crate::naive::State {
                    local_count: state.local_count,
                    num_returns: state.ret_count,
                    control_depth: 0,
                    label_index: state.label_index,
                    if_stack: core::mem::take(&mut state.if_stack),
                    body: state.body,
                    body_labels: core::mem::take(&mut state.body_labels),
                    tracing: None,
                };
                let result = self._handle_op(ctx, arch, &mut naive_state, func_imports, &[], &[], other, target);
                state.label_index = naive_state.label_index;
                state.if_stack = naive_state.if_stack;
                state.body = naive_state.body;
                state.body_labels = naive_state.body_labels;
                result
            }
        }
    }

    /// Handle a `MachOperator` using the SysV ABI.
    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
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
                }).map_err(Err::from)?;

                if let Some(hooks) = data.tracing.as_ref() {
                    let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch };
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, hooks.counter as u64, hooks.specialization as u64,
                        0, &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                self.push(ctx, arch, &RBP).map_err(Err::from)?;
                self.mov(ctx, arch, &RBP, &RSP).map_err(Err::from)?;

                let frame_sz = (data.num_params + 16) * 8;
                let frame_sz = (frame_sz + 15) & !15;
                self.mov64(ctx, arch, &RAX, frame_sz as u64).map_err(Err::from)?;
                self.sub(ctx, arch, &RSP, &RAX).map_err(Err::from)?;

                for i in 0..data.num_params.min(6) {
                    self.sysv_store_local(ctx, arch, ARG_REGS[i], i).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov64(ctx, arch, &RAX, 0).map_err(Err::from)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, RAX, state.local_count).map_err(Err::from)?;
                    state.local_count += 1;
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

impl<T: Writer<X64Label, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context> for T {}
