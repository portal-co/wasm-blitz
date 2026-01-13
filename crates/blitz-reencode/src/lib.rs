//! Re-encodes blitz-generated instruction streams back into WebAssembly.
//!
//! This crate provides functionality to convert the blitz-common machine operator
//! representation back into standard WebAssembly binary format. While other backends
//! (JavaScript, x86-64, etc.) use `blitz-common` directly to generate their target
//! code, this crate is specifically for producing WASM output from the blitz IR.
//!
//! # Purpose
//!
//! This crate enables:
//! - Converting blitz machine operators back to WASM bytecode
//! - Applying optimizations before final WASM encoding
//! - Integration with the wasm-encoder library for output generation
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_reencode::ReencodeExt;
//! use wasm_encoder::reencode::Reencode;
//!
//! // Re-encode machine operators back to WASM
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
    fn mach_instruction<A, Context, S: InstructionSink<Context, Self::Error>>(
        &mut self,
        ctx: &mut Context,
        a: &MachOperator<'_, A>,
        state: &mut MachTracker<S>,
        create: &mut (dyn FnMut(Drain<'_, (u32, wasm_encoder::ValType)>) -> S + '_),
    ) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        do_mach_instruction(self, ctx, a, state, create)
    }
}
impl<T: Reencode + ?Sized> ReencodeExt for T {}
