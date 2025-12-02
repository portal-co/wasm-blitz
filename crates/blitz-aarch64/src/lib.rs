//! AArch64 (ARM64) code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into native
//! AArch64 (64-bit ARM) machine code. The backend targets modern ARM processors
//! including Apple Silicon, AWS Graviton, and other ARM64 platforms.
//!
//! # Features
//!
//! - Native AArch64 instruction generation
//! - Support for ARM64 calling conventions
//! - Optimized for modern ARM processors
//!
//! # Architecture Support
//!
//! This backend supports ARMv8-A and later architectures, including:
//! - Apple M1/M2/M3 processors
//! - AWS Graviton processors
//! - Qualcomm Snapdragon 8cx and later
//! - Other ARMv8+ compliant processors

#![no_std]
use core::{
    error::Error,
    fmt::{Formatter, Write},
};
extern crate alloc;
