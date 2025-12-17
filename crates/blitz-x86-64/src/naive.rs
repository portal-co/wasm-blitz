//! Naive x86-64 code generation implementation.
//!
//! This module implements a straightforward, correctness-focused code generation
//! strategy for x86-64. It prioritizes simplicity and correctness over performance.

use alloc::collections::btree_map::BTreeMap;
use portal_solutions_asm_x86_64::RegisterClass;
use portal_solutions_asm_x86_64::out::arg::{MemArg, MemArgKind};
use portal_solutions_blitz_common::wasm_encoder::{self, Instruction, reencode::Reencode};

use crate::{
    out::{Writer, arg::Arg},
    *,
};

/// State tracker for x86-64 code generation.
///
/// Maintains information about the current function being compiled,
/// including local variables, control flow, and labels.
#[derive(Default)]
pub struct State {
    local_count: usize,
    num_returns: usize,
    control_depth: usize,
    label_index: usize,
    if_stack: Vec<Endable>,
    body: u32,
    body_labels: BTreeMap<u32, usize>,
}

/// Represents a control flow structure that needs an end marker.
enum Endable {
    /// A branch target.
    Br,
    /// An if statement with its label index.
    If { idx: usize },
}

/// Extension trait for x86-64 code writers.
///
/// Provides methods for generating x86-64 assembly code for WASM operations,
/// including branches, calls, and instruction handling.
pub trait WriterExt<Context>: Writer<X64Label, Context> {
    /// Generates code for a branch instruction.
    ///
    /// Emits x86-64 assembly to jump to the target label specified by the
    /// relative depth in the control flow stack.
    ///
    /// # Arguments
    ///
    /// * `arch` - The x86-64 architecture variant
    /// * `state` - Current compilation state
    /// * `relative_depth` - Depth of the target label in control flow stack
    fn br(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
        relative_depth: u32,
    ) -> Result<(), Self::Error> {
        self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        for _ in 0..=relative_depth {
            self.pop(ctx, arch, &Reg(0))?;
            self.pop(ctx, arch, &Reg(1))?;
        }
        self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        self.mov(ctx, arch, &RSP, &Reg(1))?;
        self.jmp(ctx, arch, &Reg(0))?;
        Ok(())
    }

    /// Generates code for a higher-order call (indirect call).
    ///
    /// Emits x86-64 assembly for calling a function through a function pointer,
    /// managing the return address and stack properly.
    ///
    /// # Arguments
    ///
    /// * `arch` - The x86-64 architecture variant
    /// * `state` - Current compilation state
    fn hcall(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
    ) -> Result<(), Self::Error> {
        self.pop(ctx, arch, &Reg(1))?;
        let i = state.label_index;
        state.label_index += 1;
        self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
        self.push(ctx, arch, &Reg(0))?;
        self.push(ctx, arch, &Reg(1))?;
        self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
        self.xchg(ctx, arch, &Reg(0), &RSP)?;
        self.ret(ctx, arch)?;
        self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
        Ok(())
    }

    /// Generates x86-64 assembly code for a machine operator.
    ///
    /// Main entry point for translating WASM machine operators into x86-64
    /// assembly. Handles all WASM operations including arithmetic, memory access,
    /// control flow, and function calls.
    ///
    /// # Arguments
    ///
    /// * `arch` - The x86-64 architecture variant
    /// * `state` - Current compilation state
    /// * `func_imports` - Information about imported functions
    /// * `op` - The machine operator to translate
    /// * `rewriter` - Re-encoder for instruction format conversion
    fn handle_op<E>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        wasm_encoder::reencode::Error<E>: Into<Self::Error>,
    {
        if target != state.body {
            self.jmp_label(
                ctx,
                arch,
                X64Label::Indexed {
                    idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                        state.label_index += 1;
                        return state.label_index - 1;
                    }),
                },
            )?;
            state.body = target;
            self.set_label(
                ctx,
                arch,
                X64Label::Indexed {
                    idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                        state.label_index += 1;
                        return state.label_index - 1;
                    }),
                },
            )?;
        }
        //Stack Frame: r&Reg::CTX[&Reg(0)] => local variable frame
        match op {
            MachOperator::StartFn {
                id,
                data:
                    FnData {
                        num_params: params,
                        num_returns,
                        control_depth,
                        ..
                    },
            } => {
                state.local_count = *params;
                state.num_returns = *num_returns;
                state.control_depth = *control_depth;
                self.pop(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(1),
                        offset: None,
                        disp: 0u32.wrapping_sub(*params as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.set_label(ctx, arch, X64Label::Func { r#fn: *id })?;
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    state.local_count += 1;
                    self.push(ctx, arch, &Reg(0))?;
                }
            }
            MachOperator::StartBody => {
                self.push(ctx, arch, &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(state.control_depth as u32 * 16),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.push(ctx, arch, &Reg(0))?;
                for _ in 0..state.control_depth {
                    for _ in 0..2 {
                        self.push(ctx, arch, &Reg(0))?;
                    }
                }
            }
            MachOperator::Instruction { op, .. } => {
                self._handle_op(ctx, arch, state, func_imports, op, target)?
            }
            MachOperator::Operator { op, annot } => match match op.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            } {
                op => self._handle_op(
                    ctx,
                    arch,
                    state,
                    func_imports,
                    &rewriter.instruction(op.clone()).map_err(|e| e.into())?,
                    target,
                )?,
            },
            _ => todo!(),
        }
        Ok(())
    }
    fn _handle_op(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error> {
        if target != state.body {
            self.jmp_label(
                ctx,
                arch,
                X64Label::Indexed {
                    idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                        state.label_index += 1;
                        return state.label_index - 1;
                    }),
                },
            )?;
            state.body = target;
            self.set_label(
                ctx,
                arch,
                X64Label::Indexed {
                    idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                        state.label_index += 1;
                        return state.label_index - 1;
                    }),
                },
            )?;
        }
        match op {
            Instruction::I32Const(value) => {
                self.mov64(ctx, arch, &Reg(0), *value as u32 as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64Const(value) => {
                self.mov64(ctx, arch, &Reg(0), *value as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F32Const(value) => {
                self.mov64(ctx, arch, &Reg(0), value.bits() as u64)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::F64Const(value) => {
                self.mov64(ctx, arch, &Reg(0), value.bits())?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64ReinterpretF64
            | Instruction::F64ReinterpretI64
            | Instruction::I32ReinterpretF32
            | Instruction::F32ReinterpretI32 => {}
            Instruction::I32Add | Instruction::I64Add => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                if let Instruction::I32Add = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Sub | Instruction::I64Sub => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.not(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                if let Instruction::I32Sub = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Mul | Instruction::I64Mul => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.mul(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Mul = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivU | Instruction::I64DivU => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.div(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivU = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32DivS | Instruction::I64DivS => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32DivS = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32RemU | Instruction::I64RemU => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.div(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32RemU = op {
                    self.u32(ctx, arch, &Reg(3))?;
                }
                self.push(ctx, arch, &Reg(3))?;
            }
            Instruction::I32RemS | Instruction::I64RemS => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.idiv(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32RemS = op {
                    self.u32(ctx, arch, &Reg(3))?;
                }
                self.push(ctx, arch, &Reg(3))?;
            }
            Instruction::I32And | Instruction::I64And => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.and(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32And = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Or | Instruction::I64Or => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.or(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Or = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Xor | Instruction::I64Xor => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.eor(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Xor = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Shl | Instruction::I64Shl => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.shl(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32Shl = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32ShrU | Instruction::I64ShrU => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.shr(ctx, arch, &Reg(0), &Reg(1))?;
                if let Instruction::I32ShrU = op {
                    self.u32(ctx, arch, &Reg(0))?;
                }
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32WrapI64 => {
                self.pop(ctx, arch, &Reg(0))?;
                self.u32(ctx, arch, &Reg(0))?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I32Eqz | Instruction::I64Eqz => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), 0)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.cmovcc64(ctx, arch, ConditionCode::E, &Reg(1), &1u64)?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I32Eq | Instruction::I64Eq => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.not(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.mov64(ctx, arch, &Reg(1), 0)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.cmovcc64(ctx, arch, ConditionCode::E, &Reg(1), &1u64)?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I32Ne | Instruction::I64Ne => {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
                self.not(ctx, arch, &Reg(1))?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 1,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.mov64(ctx, arch, &Reg(1), 1)?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.cmovcc64(ctx, arch, ConditionCode::E, &Reg(1), &0u64)?;
                self.push(ctx, arch, &Reg(1))?;
            }
            Instruction::I64Load(memarg) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.mov(ctx, arch, &Reg(0), &Reg(0))?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::I64Store(memarg) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                self.mov64(ctx, arch, &Reg(1), memarg.offset)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: Some((Reg(1), 0)),
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &Reg(2), &Reg(0))?;
                // self.push(ctx,arch,&Reg(0))?;
            }
            Instruction::LocalGet(local_index) => {
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.pop(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::LocalTee(local_index) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.push(ctx, arch, &Reg(0))?;
            }
            Instruction::LocalSet(local_index) => {
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.push(ctx, arch, &Reg(0))?;
                self.lea(
                    ctx,
                    arch,
                    &RSP,
                    &MemArgKind::Mem {
                        base: RSP,
                        offset: None,
                        disp: 0u32.wrapping_sub(((*local_index as i32 as isize + 1) * 8) as u32),
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
            }
            Instruction::Return => {
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.lea(
                    ctx,
                    arch,
                    &Reg(0),
                    // &Reg(0),
                    // (state.local_count + 3) as isize * 8,
                    // None,
                    &MemArgKind::Mem {
                        base: Reg(0),
                        offset: None,
                        disp: (state.local_count + 3 * 8) as u32,
                        size: MemorySize::_8,
                        reg_class: RegisterClass::Gpr,
                    },
                )?;
                self.mov(ctx, arch, &RSP, &Reg(0))?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                for a in 0..state.num_returns {
                    self.mov(ctx, arch, &Reg(2), &Reg(1))?;
                    self.push(ctx, arch, &Reg(2))?;
                }
                self.push(ctx, arch, &Reg(0))?;
                self.ret(ctx, arch)?;
            }
            Instruction::Br(relative_depth) => {
                self.br(ctx, arch, state, *relative_depth)?;
            }
            Instruction::BrIf(relative_depth) => {
                let i = state.label_index;
                state.label_index += 1;
                self.lea_label(ctx, arch, &Reg(1), X64Label::Indexed { idx: i })?;
                self.pop(ctx, arch, &Reg(0))?;
                self.cmp0(ctx, arch, &Reg(0))?;
                self.jcc(ctx, arch, ConditionCode::E, &Reg(1))?;
                self.br(ctx, arch, state, *relative_depth)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
            }
            Instruction::BrTable(targets, default) => {
                for relative_depth in targets.iter().cloned() {
                    let i = state.label_index;
                    state.label_index += 1;
                    self.lea_label(ctx, arch, &Reg(1), X64Label::Indexed { idx: i })?;
                    self.pop(ctx, arch, &Reg(0))?;
                    self.cmp0(ctx, arch, &Reg(0))?;
                    self.jcc(ctx, arch, ConditionCode::E, &Reg(1))?;
                    self.br(ctx, arch, state, relative_depth)?;
                    self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                    self.lea(
                        ctx,
                        arch,
                        &Reg(0),
                        &MemArgKind::Mem {
                            base: Reg(0),
                            offset: None,
                            disp: 0xffff_ffff,
                            size: MemorySize::_64,
                            reg_class: RegisterClass::Gpr,
                        },
                    )?;
                    self.push(ctx, arch, &Reg(0))?;
                }
                self.pop(ctx, arch, &Reg(0))?;
                self.br(ctx, arch, state, *default)?;
            }
            Instruction::Block(blockty) => {
                state.if_stack.push(Endable::Br);
                let i = state.label_index;
                state.label_index += 1;
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                // for _ in &Reg(0)..=(*relative_depth) {
                self.push(ctx, arch, &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
                // }
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
            }
            Instruction::If(blockty) => {
                let i = state.label_index;
                state.label_index += 3;
                state.if_stack.push(Endable::If { idx: i });
                self.pop(ctx, arch, &Reg(2))?;
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
                self.lea_label(ctx, arch, &Reg(1), X64Label::Indexed { idx: i + 1 })?;
                self.cmp0(ctx, arch, &Reg(2))?;
                self.jcc(ctx, arch, ConditionCode::E, &Reg(1))?;
                self.jmp(ctx, arch, &Reg(0))?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
            }
            Instruction::Else => {
                let Endable::If { idx: i } = state.if_stack.last().unwrap() else {
                    todo!()
                };
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i + 2 })?;
                self.jmp(ctx, arch, &Reg(0))?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })?;
            }
            Instruction::Loop(blockty) => {
                state.if_stack.push(Endable::Br);
                let i = state.label_index;
                state.label_index += 1;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.lea_label(ctx, arch, &Reg(0), X64Label::Indexed { idx: i })?;
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                // for _ in &Reg(0)..=(*relative_depth) {
                self.push(ctx, arch, &Reg(1))?;
                self.push(ctx, arch, &Reg(0))?;
                // }
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
            }
            Instruction::End => {
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
                // for _ in &Reg(0)..=(*relative_depth) {
                match state.if_stack.pop().unwrap() {
                    Endable::Br => {
                        self.pop(ctx, arch, &Reg(0))?;
                        self.pop(ctx, arch, &Reg(1))?;
                    }
                    Endable::If { idx: i } => {
                        self.set_label(ctx, arch, X64Label::Indexed { idx: i + 2 })?;
                    }
                }
                // }
                self.xchg(ctx, arch, &RSP, &Reg::CTX)?;
            }
            Instruction::Call(function_index) => match func_imports.get(*function_index as usize) {
                Some(("blitz", h)) if h.starts_with("hypercall") => {
                    self.hcall(ctx, arch, state)?;
                }
                _ => {
                    let function_index = *function_index - func_imports.len() as u32;
                    self.lea_label(
                        ctx,
                        arch,
                        &Reg(0),
                        X64Label::Func {
                            r#fn: function_index,
                        },
                    )?;
                    self.call(ctx, arch, &Reg(0))?;
                }
            },
            _ => {}
        };
        Ok(())
    }
}
impl<T: Writer<X64Label, Context> + ?Sized, Context> WriterExt<Context> for T {}
