//! RISC-V 64-bit code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into native
//! RISC-V 64-bit machine code. The backend targets the RV64GC instruction set.
//!
//! # Features
//!
//! - Native RISC-V 64-bit instruction generation
//! - Support for RV64I base integer instruction set
//! - Support for standard extensions (M, A, F, D, C)
//!
//! # Architecture Support
//!
//! This backend supports:
//! - RV64GC (General-purpose + Compressed instructions)
//! - SiFive processors
//! - StarFive processors
//! - Other RISC-V 64-bit compliant processors

#![no_std]
use core::{
    error::Error,
    fmt::{Formatter, Write},
};
extern crate alloc;
