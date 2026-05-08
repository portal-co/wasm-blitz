//! AArch64 (ARM64) code generation backend for wasm-blitz.
//!
//! This crate compiles WebAssembly bytecode into native AArch64 machine code.
//! It targets ARMv8-A and later (Apple Silicon, AWS Graviton, etc.).
//!
//! # Calling Convention (blitz WASM ABI)
//!
//! See `docs/abi.md` in the workspace root for a complete description.
//! Summary:
//! - SP  = Reg(31)/sp  — WASM operand stack pointer
//! - FP  = Reg(29)/x29 — frame pointer (callee-saved in prologue)
//! - LR  = Reg(30)/x30 — link register (callee-saved in prologue)
//! - Locals at `[FP − (N+1)*8]`
//! - Function call: `adr_label dest, fn_N` + `bl dest`
//! - Return: `ret` (branches to LR)

#![no_std]
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
extern crate alloc;

use portal_solutions_blitz_common::asm::Reg;

pub use portal_solutions_asm_aarch64::*;

/// The stack pointer register for AArch64.
pub const SP: Reg = Reg(31);
/// The frame pointer register (x29).
pub const FP: Reg = Reg(29);
/// The link register (x30).
pub const LR: Reg = Reg(30);

/// Label types for AArch64 code generation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum AArch64Label {
    /// An indexed label for control flow within a function.
    Indexed { idx: usize },
    /// A function entry point label.
    Func { r#fn: u32 },
    /// An external symbol resolved by the linker/loader at runtime.
    /// Used for imports (`{module}__{name}`), `__wasm_mem_pages`, and `__wasm_memory_grow`.
    External { name: alloc::string::String },
}

impl Display for AArch64Label {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            AArch64Label::Indexed { idx } => write!(f, "_idx_{idx}"),
            AArch64Label::Func { r#fn } => write!(f, "f{}", r#fn),
            AArch64Label::External { name } => write!(f, "{name}"),
        }
    }
}

/// Label trait specialization for AArch64.
pub trait Label: portal_solutions_blitz_common::Label<AArch64Label> {}
impl<T: portal_solutions_blitz_common::Label<AArch64Label> + ?Sized> Label for T {}

/// Naive code generation implementation.
pub mod naive;
/// System V (AAPCS64) ABI code generation.
pub mod sysv;
