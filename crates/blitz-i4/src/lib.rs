//! Common backend logic for 4-byte instruction architectures.
//!
//! This crate provides shared backend infrastructure for architectures that use
//! 4-byte fixed-length instructions, specifically AArch64 (ARM64) and RISC-V 64-bit.
//! The name "i4" refers to the 4-byte instruction size these architectures have in common.
//!
//! # Purpose
//!
//! The i4 backend logic enables:
//! - Shared code generation patterns between AArch64 and RISC-V 64
//! - Common optimizations for fixed 4-byte instruction architectures
//! - Reduced code duplication between similar backends

#![no_std]
extern crate alloc;
