#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use portal_solutions_blitz_common::{MachOperator, wasmparser::Operator};
pub use wasm_encoder;
use wasm_encoder::reencode::Reencode;
#[derive(Default)]
pub struct MachTracker {
    funcs: Vec<wasm_encoder::Function>,
    locals: Vec<(u32, wasm_encoder::ValType)>,
    dce_stack: Vec<bool>,
}
impl MachTracker {
    pub fn current(&mut self) -> Option<&mut wasm_encoder::Function> {
        return self.funcs.last_mut();
    }
}
pub trait ReencodeExt: Reencode {
    fn mach_instruction(
        &mut self,
        a: &MachOperator<'_>,
        state: &mut MachTracker,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        match a {
            MachOperator::StartFn { id, data } => {}
            MachOperator::Local(a, b) => {
                state.locals.push((*a, self.val_type(b.clone())?));
            }
            MachOperator::StartBody => {
                state
                    .funcs
                    .push(wasm_encoder::Function::new(state.locals.drain(..)));
            }
            MachOperator::EndBody => {}
            MachOperator::Operator(o) => {
                let mut f = state.funcs.last_mut().unwrap();
                match o {
                    Operator::Else => {
                        if let Some(a) = state.dce_stack.last_mut() {
                            *a = false
                        }
                    }
                    Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
                        state.dce_stack.push(false);
                    }
                    Operator::End => {
                        state.dce_stack.pop();
                    }
                    Operator::Br { .. }
                    | Operator::BrTable { .. }
                    | Operator::Return
                    | Operator::ReturnCall { .. }
                    | Operator::ReturnCallIndirect { .. }
                    | Operator::ReturnCallRef { .. }
                    | Operator::Unreachable => {
                        if let Some(a) = state.dce_stack.last_mut() {
                            *a = true
                        }
                    }
                    o => {
                        if state.dce_stack.iter().any(|a| *a) {
                            return Ok(());
                        } else {
                        }
                    }
                };
                f.instruction(&self.instruction(o.clone())?);
            }
            _ => todo!(),
        };
        Ok(())
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
