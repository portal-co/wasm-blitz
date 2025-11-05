#![no_std]
extern crate alloc;
use crate::tracker::{MachTracker, do_mach_instruction};
use alloc::vec::{Drain, Vec};
use portal_solutions_blitz_common::{ops::MachOperator, wasmparser::Operator};
pub use wasm_encoder;
use wasm_encoder::{CodeSection, reencode::Reencode};
use wax_core::build::InstructionSink;
pub mod tracker;
pub trait ReencodeExt: Reencode {
    fn mach_instruction<A, S: InstructionSink<Self::Error>>(
        &mut self,
        a: &MachOperator<'_, A>,
        state: &mut MachTracker<S>,
        create: &mut (dyn FnMut(Drain<'_, (u32, wasm_encoder::ValType)>) -> S + '_),
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        do_mach_instruction(self, a, state, create)
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
