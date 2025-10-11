use crate::{
    out::{Arg, Writer},
    *,
};

#[derive(Default)]
pub struct State {
    local_count: usize,
    num_returns: usize,
    control_depth: usize,
    label_index: usize,
    if_stack: Vec<Endable>,
}
// #[derive(Clone)]
enum Endable {
    Br,
    If { idx: usize },
}
pub trait WriterExt: Writer {
    fn br(&mut self, state: &mut State, relative_depth: u32) -> Result<(), Self::Error> {
        self.xchg(&RSP, &Reg(15), Some(8))?;
        for _ in 0..=relative_depth {
            self.pop(&Reg(0))?;
            self.pop(&Reg(1))?;
        }
        self.xchg(&RSP, &Reg(15), Some(8))?;
        self.mov(&RSP, &Reg(1), None)?;
        self.jmp(&Reg(0))?;
        Ok(())
    }
    fn hcall(&mut self, state: &mut State) -> Result<(), Self::Error> {
        self.pop(&Reg(1))?;
        let i = state.label_index;
        state.label_index += 1;
        self.lea_label(&Reg(0), &X64Label::Indexed { idx: i })?;
        self.push(&Reg(0))?;
        self.push(&Reg(1))?;
        self.mov(&Reg(0), &Reg(15), Some(-8))?;
        self.xchg(&Reg(0), &RSP, Some(0))?;
        self.ret()?;
        self.set_label(&X64Label::Indexed { idx: i })?;
        Ok(())
    }
    fn handle_op(
        &mut self,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
    ) -> Result<(), Self::Error> {
        //Stack Frame: r&Reg(15)[&Reg(0)] => local variable frame
        match op {
            MachOperator::StartFn {
                id: f,
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
                self.pop(&Reg(1))?;
                self.lea(&Reg(0), &Reg(1), -(*params as isize), None)?;
                self.xchg(&Reg(0), &Reg(15), Some(0))?;
                self.set_label(&X64Label::Func { r#fn: *f })?;
            }
            MachOperator::Local { count: a, ty: b } => {
                for _ in 0..*a {
                    state.local_count += 1;
                    self.push(&Reg(0))?;
                }
            }
            MachOperator::StartBody => {
                self.push(&Reg(1))?;
                self.push(&Reg(0))?;
                self.lea(&Reg(0), &RSP, -(state.control_depth as isize * 16), None)?;
                self.xchg(&Reg(0), &Reg(15), Some(8))?;
                self.push(&Reg(0))?;
                for _ in 0..state.control_depth {
                    for _ in 0..2 {
                        self.push(&Reg(0))?;
                    }
                }
            }
            MachOperator::Operator { op, annot } => match match op.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            } {
                op => match op {
                    Operator::I32Const { value } => {
                        self.mov64(&Reg(0), *value as u32 as u64)?;
                        self.push(&Reg(0))?;
                    }
                    Operator::I64Const { value } => {
                        self.mov64(&Reg(0), *value as u64)?;
                        self.push(&Reg(0))?;
                    }
                    Operator::F32Const { value } => {
                        self.mov64(&Reg(0), value.bits() as u64)?;
                        self.push(&Reg(0))?;
                    }
                    Operator::F64Const { value } => {
                        self.mov64(&Reg(0), value.bits())?;
                        self.push(&Reg(0))?;
                    }
                    Operator::I64ReinterpretF64
                    | Operator::F64ReinterpretI64
                    | Operator::I32ReinterpretF32
                    | Operator::F32ReinterpretI32 => {}
                    Operator::I32Add | Operator::I64Add => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.lea(&Reg(0), &Reg(0), 0, Some((&Reg(1), 0)))?;
                        if let Operator::I32Add = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Sub | Operator::I64Sub => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.not(&Reg(1))?;
                        self.lea(&Reg(0), &Reg(0), 1, Some((&Reg(1), 0)))?;
                        if let Operator::I32Sub = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Mul | Operator::I64Mul => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.mul(&Reg(0), &Reg(1))?;
                        if let Operator::I32Mul = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32DivU | Operator::I64DivU => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.div(&Reg(0), &Reg(1))?;
                        if let Operator::I32DivU = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32DivS | Operator::I64DivS => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.idiv(&Reg(0), &Reg(1))?;
                        if let Operator::I32DivS = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32RemU | Operator::I64RemU => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.div(&Reg(0), &Reg(1))?;
                        if let Operator::I32RemU = op {
                            self.u32(&Reg(7))?;
                        }
                        self.push(&Reg(7))?;
                    }
                    Operator::I32RemS | Operator::I64RemS => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.idiv(&Reg(0), &Reg(1))?;
                        if let Operator::I32RemS = op {
                            self.u32(&Reg(7))?;
                        }
                        self.push(&Reg(7))?;
                    }
                    Operator::I32And | Operator::I64And => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.and(&Reg(0), &Reg(1))?;
                        if let Operator::I32And = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Or | Operator::I64Or => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.or(&Reg(0), &Reg(1))?;
                        if let Operator::I32Or = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Xor | Operator::I64Xor => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.eor(&Reg(0), &Reg(1))?;
                        if let Operator::I32Xor = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Shl | Operator::I64Shl => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.shl(&Reg(0), &Reg(1))?;
                        if let Operator::I32Shl = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32ShrU | Operator::I64ShrU => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.shr(&Reg(0), &Reg(1))?;
                        if let Operator::I32ShrU = op {
                            self.u32(&Reg(0))?;
                        }
                        self.push(&Reg(0))?;
                    }
                    Operator::I32WrapI64 => {
                        self.pop(&Reg(0))?;
                        self.u32(&Reg(0))?;
                        self.push(&Reg(0))?;
                    }
                    Operator::I32Eqz | Operator::I64Eqz => {
                        self.pop(&Reg(0))?;
                        self.mov64(&Reg(1), 0)?;
                        self.cmp0(&Reg(0))?;
                        self.cmovz64(&Reg(1), 1)?;
                        self.push(&Reg(1))?;
                    }
                    Operator::I32Eq | Operator::I64Eq => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.not(&Reg(1))?;
                        self.lea(&Reg(0), &Reg(0), 1, Some((&Reg(1), 1)))?;
                        self.mov64(&Reg(1), 0)?;
                        self.cmp0(&Reg(0))?;
                        self.cmovz64(&Reg(1), 1)?;
                        self.push(&Reg(1))?;
                    }
                    Operator::I32Ne | Operator::I64Ne => {
                        self.pop(&Reg(0))?;
                        self.pop(&Reg(1))?;
                        self.not(&Reg(1))?;
                        self.lea(&Reg(0), &Reg(0), 1, Some((&Reg(1), 1)))?;
                        self.mov64(&Reg(1), 1)?;
                        self.cmp0(&Reg(0))?;
                        self.cmovz64(&Reg(1), 0)?;
                        self.push(&Reg(1))?;
                    }
                    Operator::I64Load { memarg } => {
                        self.pop(&Reg(0))?;
                        self.mov64(&Reg(1), memarg.offset)?;
                        self.lea(&Reg(0), &Reg(0), 0, Some((&Reg(1), 1)))?;
                        self.mov(&Reg(0), &Reg(0), Some(0))?;
                        self.push(&Reg(0))?;
                    }
                    Operator::I64Store { memarg } => {
                        self.pop(&Reg(2))?;
                        self.pop(&Reg(0))?;
                        self.mov64(&Reg(1), memarg.offset)?;
                        self.lea(&Reg(0), &Reg(0), 0, Some((&Reg(1), 1)))?;
                        self.xchg(&Reg(2), &Reg(0), Some(0))?;
                        // self.push(&Reg(0))?;
                    }
                    Operator::LocalGet { local_index } => {
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                        self.lea(&RSP, &RSP, -((*local_index as i32 as isize) * 8), None)?;
                        self.pop(&Reg(0))?;
                        self.lea(&RSP, &RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                        self.push(&Reg(0))?;
                    }
                    Operator::LocalTee { local_index } => {
                        self.pop(&Reg(0))?;
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                        self.lea(&RSP, &RSP, -((*local_index as i32 as isize) * 8), None)?;
                        self.push(&Reg(0))?;
                        self.lea(&RSP, &RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                        self.push(&Reg(0))?;
                    }
                    Operator::LocalSet { local_index } => {
                        self.pop(&Reg(0))?;
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                        self.lea(&RSP, &RSP, -((*local_index as i32 as isize) * 8), None)?;
                        self.push(&Reg(0))?;
                        self.lea(&RSP, &RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                        self.xchg(&RSP, &Reg(15), Some(0))?;
                    }
                    Operator::Return => {
                        self.mov(&Reg(1), &RSP, None)?;
                        self.mov(&Reg(0), &Reg(15), Some(0))?;
                        self.lea(&Reg(0), &Reg(0), (state.local_count + 3) as isize * 8, None)?;
                        self.mov(&RSP, &Reg(0), None)?;
                        self.pop(&Reg(0))?;
                        self.xchg(&Reg(0), &Reg(15), Some(8))?;
                        self.pop(&Reg(0))?;
                        self.xchg(&Reg(0), &Reg(15), Some(0))?;
                        self.pop(&Reg(0))?;
                        for a in 0..state.num_returns {
                            self.mov(&Reg(2), &Reg(1), Some(-(a as isize * 8)))?;
                            self.push(&Reg(2))?;
                        }
                        self.push(&Reg(0))?;
                        self.ret()?;
                    }
                    Operator::Br { relative_depth } => {
                        self.br(state, *relative_depth)?;
                    }
                    Operator::BrIf { relative_depth } => {
                        let i = state.label_index;
                        state.label_index += 1;
                        self.lea_label(&Reg(1), &X64Label::Indexed { idx: i })?;
                        self.pop(&Reg(0))?;
                        self.cmp0(&Reg(0))?;
                        self.jz(&Reg(1))?;
                        self.br(state, *relative_depth)?;
                        self.set_label(&X64Label::Indexed { idx: i })?;
                    }
                    Operator::BrTable { targets } => {
                        for relative_depth in targets.targets().flatten() {
                            let i = state.label_index;
                            state.label_index += 1;
                            self.lea_label(&Reg(1), &X64Label::Indexed { idx: i })?;
                            self.pop(&Reg(0))?;
                            self.cmp0(&Reg(0))?;
                            self.jz(&Reg(1))?;
                            self.br(state, relative_depth)?;
                            self.set_label(&X64Label::Indexed { idx: i })?;
                            self.lea(&Reg(0), &Reg(0), -1, None)?;
                            self.push(&Reg(0))?;
                        }
                        self.pop(&Reg(0))?;
                        self.br(state, targets.default())?;
                    }
                    Operator::Block { blockty } => {
                        state.if_stack.push(Endable::Br);
                        let i = state.label_index;
                        state.label_index += 1;
                        self.lea_label(&Reg(0), &X64Label::Indexed { idx: i })?;
                        self.mov(&Reg(1), &RSP, None)?;
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                        // for _ in &Reg(0)..=(*relative_depth) {
                        self.push(&Reg(1))?;
                        self.push(&Reg(0))?;
                        // }
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                        self.set_label(&X64Label::Indexed { idx: i })?;
                    }
                    Operator::If { blockty } => {
                        let i = state.label_index;
                        state.label_index += 3;
                        state.if_stack.push(Endable::If { idx: i });
                        self.pop(&Reg(2))?;
                        self.lea_label(&Reg(0), &X64Label::Indexed { idx: i })?;
                        self.lea_label(&Reg(1), &X64Label::Indexed { idx: i + 1 })?;
                        self.cmp0(&Reg(2))?;
                        self.jz(&Reg(1))?;
                        self.jmp(&Reg(0))?;
                        self.set_label(&X64Label::Indexed { idx: i })?;
                    }
                    Operator::Else => {
                        let Endable::If { idx: i } = state.if_stack.last().unwrap() else {
                            todo!()
                        };
                        self.lea_label(&Reg(0), &X64Label::Indexed { idx: i + 2 })?;
                        self.jmp(&Reg(0))?;
                        self.set_label(&X64Label::Indexed { idx: i + 1 })?;
                    }
                    Operator::Loop { blockty } => {
                        state.if_stack.push(Endable::Br);
                        let i = state.label_index;
                        state.label_index += 1;
                        self.set_label(&X64Label::Indexed { idx: i })?;
                        self.lea_label(&Reg(0), &X64Label::Indexed { idx: i })?;
                        self.mov(&Reg(1), &RSP, None)?;
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                        // for _ in &Reg(0)..=(*relative_depth) {
                        self.push(&Reg(1))?;
                        self.push(&Reg(0))?;
                        // }
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                    }
                    Operator::End => {
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                        // for _ in &Reg(0)..=(*relative_depth) {
                        match state.if_stack.pop().unwrap() {
                            Endable::Br => {
                                self.pop(&Reg(0))?;
                                self.pop(&Reg(1))?;
                            }
                            Endable::If { idx: i } => {
                                self.set_label(&X64Label::Indexed { idx: i + 2 })?;
                            }
                        }
                        // }
                        self.xchg(&RSP, &Reg(15), Some(8))?;
                    }
                    Operator::Call { function_index } => {
                        match func_imports.get(*function_index as usize) {
                            Some(("blitz", h)) if h.starts_with("hypercall") => {
                                self.hcall(state)?;
                            }
                            _ => {
                                let function_index = *function_index - func_imports.len() as u32;
                                self.lea_label(
                                    &Reg(0),
                                    &X64Label::Func {
                                        r#fn: function_index,
                                    },
                                )?;
                                self.call(&Reg(0))?;
                            }
                        }
                    }
                    _ => {}
                },
            },
            _ => todo!(),
        }
        Ok(())
    }
}
impl<T: Writer + ?Sized> WriterExt for T {}
