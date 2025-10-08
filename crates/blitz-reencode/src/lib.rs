#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use portal_solutions_blitz_common::MachOperator;
pub use wasm_encoder;
use wasm_encoder::reencode::Reencode;
#[derive(Default)]
pub struct MachTracker {
    funcs: Vec<wasm_encoder::Function>,
    locals: Vec<(u32, wasm_encoder::ValType)>,
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
                f.instruction(&self.instruction(o.clone())?);
            }
            _ => todo!(),
        };
        Ok(())
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
