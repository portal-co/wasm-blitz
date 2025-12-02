//! Common utilities and types for the wasm-blitz compiler.
//!
//! This crate provides the foundational types and utilities used across all wasm-blitz
//! compilation targets. It includes:
//!
//! - Machine operator abstractions for representing WASM instructions
//! - Dead code elimination (DCE) optimization passes
//! - Assembly abstractions for various target architectures
//! - Label and display utilities for code generation
//!
//! # Features
//!
//! - `asm`: Enables assembly-related functionality (enabled by default)
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_common::ops::MachOperator;
//! use portal_solutions_blitz_common::dce;
//! ```

#![no_std]
#[doc(hidden)]
pub extern crate alloc;
#[doc(hidden)]
pub mod __ {
    pub use core;
    pub use wax_core;
}
use alloc::vec::Vec;
use wax_core::build::InstructionSink;

/// Converts an iterator into a vector and returns its IntoIter.
///
/// This utility function collects items from an iterator into a vector,
/// then immediately converts it to an IntoIter. Useful for type conversions
/// and iterator transformations.
pub fn vecced<T>(a: impl Iterator<Item = T>) -> alloc::vec::IntoIter<T> {
    a.collect::<Vec<T>>().into_iter()
}
use core::{
    fmt::{Display, Formatter},
    mem::{transmute, transmute_copy},
    str::MatchIndices,
};
pub use wasm_encoder;
pub use wasmparser;
use wasmparser::{BinaryReaderError, FuncType, FunctionBody, Operator, ValType};

use crate::ops::MachOperator;

/// Dead code elimination module.
///
/// Provides utilities for removing unreachable or unused code from WASM functions.
pub mod dce;

/// Label trait for code generation targets.
///
/// Represents labels that can be used for jumps and branches in generated code.
/// The trait provides a method to extract the raw label value if types match.
pub trait Label<X: Clone + 'static>: Display {
    /// Attempts to extract the raw label value if the type matches.
    ///
    /// Returns `Some(X)` if this label is of type X, otherwise `None`.
    fn raw(&self) -> Option<X> {
        if typeid::of::<Self>() == typeid::of::<X>() {
            let this: &X = unsafe { transmute_copy(&self) };
            Some(this.clone())
        } else {
            None
        }
    }
}
impl<T: Display + ?Sized, X: Clone + 'static> Label<X> for T {}

/// A wrapper around a closure that implements Display.
///
/// This allows formatting logic to be captured as a closure and used
/// wherever Display is required.
#[derive(Clone, Copy)]
pub struct DisplayFn<'a>(pub &'a (dyn Fn(&mut Formatter) -> core::fmt::Result + 'a));

impl<'a> Display for DisplayFn<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        (self.0)(f)
    }
}

/// Assembly abstractions for code generation.
///
/// Available when the `asm` feature is enabled.
#[cfg(feature = "asm")]
pub mod asm;

/// Machine operator definitions and utilities.
///
/// Defines the intermediate representation used for WASM instructions.
pub mod ops;

/// Compiler optimization passes.
///
/// Contains various optimization and transformation passes for WASM code.
pub mod passes;
