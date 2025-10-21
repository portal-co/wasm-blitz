use crate::*;
use portal_solutions_blitz_common::dce::{DceStack, dce};
#[derive(Default)]
pub struct MachTracker {
    funcs: Vec<wasm_encoder::Function>,
    locals: Vec<(u32, wasm_encoder::ValType)>,
    dce_stack: DceStack,
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
pub fn do_mach_instruction<E, A>(
    r: &mut (impl Reencode<Error = E> + ?Sized),
    a: &MachOperator<'_, A>,
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
        MachOperator::EndBody => {
            state.dce_stack = Default::default();
        }
        MachOperator::Operator { op: o, .. } => {
            let Some(o) = o.as_ref() else {
                return Ok(());
            };
            let mut f = state.funcs.last_mut().unwrap();
            if !dce(&mut state.dce_stack, &o) {
                f.instruction(&r.instruction(o.clone())?);
            }
        }
        _ => todo!(),
    };
    Ok(())
}
