//! State tracking for machine instruction re-encoding.
//!
//! This module provides the `MachTracker` type which maintains state during
//! the re-encoding process, tracking functions, local variables, and dead
//! code elimination state.

use crate::*;
use alloc::vec::Drain;
use portal_solutions_blitz_common::dce::{DceStack, dce, dce_instr};
use wasm_encoder::Function;
use wax_core::build::InstructionSink;

/// Tracks state during machine instruction re-encoding.
///
/// Maintains collections of functions being generated, local variable
/// declarations, and dead code elimination state.
#[derive(Default)]
pub struct MachTracker<S> {
    funcs: Vec<S>,
    locals: Vec<(u32, wasm_encoder::ValType)>,
    dce_stack: DceStack,
}

impl<S> MachTracker<S> {
    /// Returns a mutable reference to the current function being tracked.
    ///
    /// Returns `None` if no function is currently being processed.
    pub fn current(&mut self) -> Option<&mut S> {
        return self.funcs.last_mut();
    }
}

impl MachTracker<Function> {
    /// Writes all tracked functions to a code section.
    ///
    /// Iterates through all accumulated functions and adds them to the
    /// provided code section.
    ///
    /// # Arguments
    ///
    /// * `code` - The code section to write functions to
    pub fn on_code_section(&self, code: &mut CodeSection) {
        for f in self.funcs.iter() {
            code.function(f);
        }
    }
}

/// Processes a machine instruction during re-encoding.
///
/// Core function that handles individual machine operators, applying dead code
/// elimination and writing instructions to the appropriate instruction sink.
///
/// # Arguments
///
/// * `r` - The re-encoder to use for instruction conversion
/// * `a` - The machine operator to process
/// * `state` - Current tracking state
/// * `create` - Factory function for creating instruction sinks
///
/// # Returns
///
/// Result indicating success or a re-encoding error.
pub fn do_mach_instruction<E, A,S: InstructionSink<E>>(
    r: &mut (impl Reencode<Error = E> + ?Sized),
    a: &MachOperator<'_, A>,
    state: &mut MachTracker<S>,
    create: &mut (dyn FnMut(Drain<'_,(u32,wasm_encoder::ValType)>) -> S + '_),
) -> Result<(), wasm_encoder::reencode::Error<E>> {
    match a {
        MachOperator::StartFn { id, data } => {}
        MachOperator::Local { count: a, ty: b } => {
            state.locals.push((*a, r.val_type(b.clone())?));
        }
        MachOperator::StartBody => {
            state
                .funcs
                .push(create(state.locals.drain(..)));
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
                f.instruction(&r.instruction(o.clone())?).map_err(|e|wasm_encoder::reencode::Error::UserError(e))?;
            }
        }
        MachOperator::Instruction { op, .. } => {
            let mut f = state.funcs.last_mut().unwrap();
            if !dce_instr(&mut state.dce_stack, op) {
                f.instruction(op).map_err(|e|wasm_encoder::reencode::Error::UserError(e))?;
            }
        }
        _ => todo!(),
    };
    Ok(())
}
