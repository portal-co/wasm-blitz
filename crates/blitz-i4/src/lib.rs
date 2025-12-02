//! Intermediate representation utilities for wasm-blitz.
//!
//! This crate provides internal intermediate representation (IR) utilities
//! that are used during the compilation process. The i4 IR serves as a
//! bridge between the high-level WASM operations and the low-level
//! machine-specific code generation.
//!
//! # Purpose
//!
//! The i4 IR enables:
//! - Platform-independent optimizations
//! - Analysis passes before code generation
//! - Transformation of WASM operations into a form suitable for various backends

#![no_std]
extern crate alloc;
