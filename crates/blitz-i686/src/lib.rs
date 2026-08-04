//! i686 (IA-32) ILP32 code generation backend for wasm-blitz.
//!
//! Thin Phase-1 backend: stack-based naive lowering plus a SysV entry path.
//! Host pointer tables use a 4-byte stride (`HOST_PTR_STRIDE`); WASM
//! operand/local slots remain 8 bytes (`WASM_SLOT`).

#![no_std]
extern crate alloc;

use core::fmt::{Display, Formatter};

pub use portal_solutions_asm_x86::*;

/// Label types for i686 code generation.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum I686Label {
    /// An indexed label for control flow within a function.
    Indexed { idx: usize },
    /// A function entry point label.
    Func { r#fn: u32 },
    /// An external symbol that the linker/loader resolves at runtime.
    External { name: alloc::string::String },
    /// An ambient symbol referencing a pre-existing native library.
    Ambient { name: alloc::string::String },
}

impl Display for I686Label {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            I686Label::Indexed { idx } => write!(f, "_idx_{idx}"),
            I686Label::Func { r#fn } => write!(f, "f{}", r#fn),
            I686Label::External { name } => write!(f, "{name}"),
            I686Label::Ambient { name } => write!(f, "__ambient_{name}"),
        }
    }
}

/// Label trait specialization for i686.
pub trait Label: portal_solutions_blitz_common::Label<I686Label> {}
impl<T: portal_solutions_blitz_common::Label<I686Label> + ?Sized> Label for T {}

pub mod naive;
pub mod sysv;
