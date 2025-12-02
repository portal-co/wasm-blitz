//! RISC-V 64-bit code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into native
//! RISC-V 64-bit machine code. The backend targets the RV64IMAFD instruction set
//! (compatible with [rv-utils](https://github.com/portal-co/rv-utils)).
//!
//! # Status
//!
//! **Work in Progress**: This backend is currently under development.
//!
//! # Features
//!
//! - Native RISC-V 64-bit instruction generation
//! - RV64I base integer instruction set
//! - M extension (integer multiplication and division)
//! - A extension (atomic instructions)
//! - F extension (single-precision floating-point)
//! - D extension (double-precision floating-point)
//!
//! # Architecture Support
//!
//! This backend targets:
//! - RV64IMAFD instruction set
//! - SiFive processors
//! - StarFive processors
//! - Other RISC-V 64-bit compliant processors

#![no_std]
use core::{
    error::Error,
    fmt::{Formatter, Write},
};
extern crate alloc;
