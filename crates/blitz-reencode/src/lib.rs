#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use portal_solutions_blitz_common::{MachOperator, wasmparser::Operator};
pub use wasm_encoder;
use wasm_encoder::{CodeSection, reencode::Reencode};
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
    pub fn on_code_section(&self, code: &mut CodeSection) {
        for f in self.funcs.iter() {
            code.function(f);
        }
    }
}
pub fn do_mach_instruction<E>(
    r: &mut (impl Reencode<Error = E> + ?Sized),
    a: &MachOperator<'_>,
    state: &mut MachTracker,
) -> Result<(), wasm_encoder::reencode::Error<E>> {
    match a {
        MachOperator::StartFn { id, data } => {}
        MachOperator::Local { count: a, ty: b } => {
            state.locals.push((*a, r.val_type(b.clone())?));
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
            f.instruction(&r.instruction(o.clone())?);
        }
        _ => todo!(),
    };
    Ok(())
}
pub trait ReencodeExt: Reencode {
    fn mach_instruction(
        &mut self,
        a: &MachOperator<'_>,
        state: &mut MachTracker,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        do_mach_instruction(self, a, state)
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
