//! WebAssembly re-encoding utilities for wasm-blitz.
//!
//! This crate provides functionality to transform and re-encode WebAssembly
//! instructions. It acts as a bridge between the WASM parser and the various
//! code generation backends.
//!
//! # Features
//!
//! - Instruction tracking during compilation
//! - Format conversion between different WASM representations
//! - Integration with wasm-encoder for output generation
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_reencode::ReencodeExt;
//! use wasm_encoder::reencode::Reencode;
//!
//! // Use ReencodeExt to transform WASM instructions
//! ```

#![no_std]
extern crate alloc;
use crate::tracker::{MachTracker, do_mach_instruction};
use alloc::vec::{Drain, Vec};
use portal_solutions_blitz_common::{ops::MachOperator, wasmparser::Operator};
pub use wasm_encoder;
use wasm_encoder::{CodeSection, reencode::Reencode};
use wax_core::build::InstructionSink;

/// Instruction tracking utilities.
///
/// Provides state tracking for machine instructions during re-encoding.
pub mod tracker;

/// Extension trait for re-encoding machine operators.
///
/// This trait extends the `Reencode` trait with functionality specific to
/// handling machine operators during the re-encoding process.
pub trait ReencodeExt: Reencode {
    /// Re-encodes a machine operator instruction.
    ///
    /// Converts a machine operator to its encoded form, tracking state
    /// and managing instruction sinks appropriately.
    ///
    /// # Arguments
    ///
    /// * `a` - The machine operator to re-encode
    /// * `state` - The current tracking state
    /// * `create` - Factory function for creating instruction sinks
    ///
    /// # Returns
    ///
    /// Result indicating success or a re-encoding error.
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
