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

/// Load the address of `label` into `dest`.
///
/// Internal labels (`Func`/`Indexed`) resolve within the same code buffer, so a
/// single PC-relative `ADR` (±1MB) suffices. External/ambient symbols become
/// relocations, and AArch64 Mach-O cannot relocate a plain `ADR` — they require
/// an `ADRP`+`ADD` pair (`PAGE21` + `PAGEOFF12`, ±4GB). Use this everywhere an
/// external symbol address is materialized so the emitted instruction matches
/// the relocation the object writer produces.
pub fn load_label_addr<W, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: AArch64Arch,
    dest: &(dyn portal_solutions_asm_aarch64::out::arg::MemArg + '_),
    label: AArch64Label,
) -> Result<(), W::Error>
where
    W: portal_solutions_asm_aarch64::out::Writer<AArch64Label, Context>,
{
    match &label {
        AArch64Label::External { .. } | AArch64Label::Ambient { .. } => {
            w.adrp_label(ctx, arch, dest, label.clone())?;
            w.add_lo12_label(ctx, arch, dest, dest, label)
        }
        _ => w.adr_label(ctx, arch, dest, label),
    }
}

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
    /// An ambient symbol referencing a pre-existing, unrecompiled native library.
    /// Rendered as `__ambient_{name}`. Resolved at link/load time via
    /// [`wax_core::build::AmbientInfo`].
    Ambient { name: alloc::string::String },
}

impl Display for AArch64Label {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            AArch64Label::Indexed { idx } => write!(f, "_idx_{idx}"),
            AArch64Label::Func { r#fn } => write!(f, "f{}", r#fn),
            AArch64Label::External { name } => write!(f, "{name}"),
            AArch64Label::Ambient { name } => write!(f, "__ambient_{name}"),
        }
    }
}

/// Label trait specialization for AArch64.
pub trait Label: portal_solutions_blitz_common::Label<AArch64Label> {}
impl<T: portal_solutions_blitz_common::Label<AArch64Label> + ?Sized> Label for T {}

/// [`blitz_codegen::BlitzWriter`] implementation for AArch64.
pub mod codegen;
/// Naive code generation implementation.
pub mod naive;
/// System V (AAPCS64) ABI code generation.
pub mod sysv;
/// BackendAbi implementations for AArch64.
pub mod abi;
/// LFI-compliant sandboxed code generation for AArch64.
pub mod lfi;
/// `wax-core` `InstructionSink` / `OperatorSink` impls for the AArch64 backend.
pub mod sink;
