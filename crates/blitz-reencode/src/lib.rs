#![no_std]
extern crate alloc;
use crate::tracker::{MachTracker, do_mach_instruction};
use alloc::vec::Vec;
use portal_solutions_blitz_common::{ops::MachOperator, wasmparser::Operator};
pub use wasm_encoder;
use wasm_encoder::{CodeSection, reencode::Reencode};
pub mod tracker;
pub trait ReencodeExt: Reencode {
    fn mach_instruction<A>(
        &mut self,
        a: &MachOperator<'_, A>,
        state: &mut MachTracker,
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        do_mach_instruction(self, a, state)
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
