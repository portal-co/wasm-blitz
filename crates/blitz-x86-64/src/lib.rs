//! x86-64 code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into native
//! x86-64 machine code. It uses a naive code generation strategy that prioritizes
//! correctness and simplicity.
//!
//! # Features
//!
//! - Direct emission of x86-64 assembly instructions
//! - Stack-based execution model matching WASM semantics
//! - Support for function calls, branches, and control flow
//! - Register allocation for local variables
//!
//! # Architecture
//!
//! The x86-64 backend uses the following conventions:
//!
//! - Stack pointer (RSP) for the execution stack
//! - Context register for local variable frame pointer
//! - Dedicated registers for temporary values
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_x86_64::{WriterExt, X64Label};
//!
//! // Use WriterExt to generate x86-64 code for WASM operations
//! ```

#![no_std]
use alloc::vec::Vec;
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
use portal_solutions_blitz_common::{
    asm::Reg,
    asm::common::mem::MemorySize,
    ops::{FnData, MachOperator},
    wasmparser::Operator,
};
extern crate alloc;
pub use portal_solutions_asm_x86_64::*;
/// The stack pointer register (RSP).
const RSP: Reg = Reg(4);

/// Label types for x86-64 code generation.
///
/// Labels are used to mark locations in the generated code for jumps and branches.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum X64Label {
    /// An indexed label for control flow within a function.
    Indexed { idx: usize },
    /// A function entry point label.
    Func { r#fn: u32 },
}

impl Display for X64Label {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            X64Label::Indexed { idx } => write!(f, "_idx_{idx}"),
            X64Label::Func { r#fn } => write!(f, "f{}", r#fn),
        }
    }
}

/// Label trait specialization for x86-64.
///
/// This trait extends the common Label trait with x86-64 specific functionality.
pub trait Label: portal_solutions_blitz_common::Label<X64Label> {}
impl<T: portal_solutions_blitz_common::Label<X64Label> + ?Sized> Label for T {}

/// Naive code generation implementation.
///
/// Contains the naive (straightforward) code generation strategy for x86-64.
pub mod naive;
