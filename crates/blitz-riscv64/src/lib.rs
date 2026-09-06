//! RISC-V 64-bit code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into native
//! RISC-V 64-bit machine code. The backend targets the RV64 instruction set and
//! reuses the asm-arch crate for instruction emission.

#![no_std]
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
extern crate alloc;

use portal_solutions_blitz_common::asm::Reg;

pub use portal_solutions_asm_riscv64::*;

/// The stack pointer register for RISC-V backed writers.
const SP: Reg = Reg(2);

/// Label types for RISC-V code generation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum RiscvLabel {
    /// An indexed label for control flow within a function.
    Indexed { idx: usize },
    /// A function entry point label.
    Func { r#fn: u32 },
    /// An external symbol that the linker/loader resolves at runtime.
    /// Used for imports (`{module}__{name}`), `__wasm_mem_pages`, and `__wasm_memory_grow`.
    External { name: alloc::string::String },
    /// An ambient symbol referencing a pre-existing, unrecompiled native library.
    /// Rendered as `__ambient_{name}`. Resolved at link/load time via
    /// [`wax_core::build::AmbientInfo`].
    Ambient { name: alloc::string::String },
}

impl Display for RiscvLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            RiscvLabel::Indexed { idx } => write!(f, "_idx_{idx}"),
            RiscvLabel::Func { r#fn } => write!(f, "f{}", r#fn),
            RiscvLabel::External { name } => write!(f, "{name}"),
            RiscvLabel::Ambient { name } => write!(f, "__ambient_{name}"),
        }
    }
}

/// Label trait specialization for RISC-V.
pub trait Label: portal_solutions_blitz_common::Label<RiscvLabel> {}
impl<T: portal_solutions_blitz_common::Label<RiscvLabel> + ?Sized> Label for T {}

/// BackendAbi implementations for RISC-V 64.
pub mod abi;
/// Naive code generation implementation (stub).
///
/// A minimal, correctness-first implementation will be placed here. For now this
/// module is a placeholder modeled after the x86-64 backend structure.
pub mod codegen;
pub mod naive;
/// `wax-core` `InstructionSink` / `OperatorSink` impls for the RISC-V 64 backend.
pub mod sink;
pub mod sysv;
