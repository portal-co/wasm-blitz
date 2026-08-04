//! AArch32 (ARMv7-A / A32) ILP32 code generation backend for wasm-blitz.
//!
//! Thin Phase-1 backend: stack-based naive lowering plus an AAPCS SysV entry
//! path. Host pointer tables use a 4-byte stride (`HOST_PTR_STRIDE`); WASM
//! operand/local slots remain 8 bytes (`WASM_SLOT`).

#![no_std]
extern crate alloc;

use core::fmt::{Display, Formatter};

pub use portal_solutions_asm_arm::*;

/// Label types for AArch32 code generation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum ArmLabel {
    /// An indexed label for control flow within a function.
    Indexed { idx: usize },
    /// A function entry point label.
    Func { r#fn: u32 },
    /// An external symbol that the linker/loader resolves at runtime.
    External { name: alloc::string::String },
    /// An ambient symbol referencing a pre-existing native library.
    Ambient { name: alloc::string::String },
}

impl Display for ArmLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ArmLabel::Indexed { idx } => write!(f, "_idx_{idx}"),
            ArmLabel::Func { r#fn } => write!(f, "f{}", r#fn),
            ArmLabel::External { name } => write!(f, "{name}"),
            ArmLabel::Ambient { name } => write!(f, "__ambient_{name}"),
        }
    }
}

/// Label trait specialization for AArch32.
pub trait Label: portal_solutions_blitz_common::Label<ArmLabel> {}
impl<T: portal_solutions_blitz_common::Label<ArmLabel> + ?Sized> Label for T {}

pub mod naive;
pub mod sysv;
