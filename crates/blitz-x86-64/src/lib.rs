#![no_std]
use alloc::vec::Vec;
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
use portal_solutions_blitz_common::{
    MemorySize,
    asm::Reg,
    ops::{FnData, MachOperator},
    wasmparser::Operator,
};
extern crate alloc;
static REG_NAMES: &'static [&'static str; 8] =
    &["rax", "rbx", "rcx", "rsp", "rbp", "rsi", "rdi", "rdx"];
static REG_NAMES_32: &'static [&'static str; 8] =
    &["eax", "ebx", "ecx", "esp", "ebp", "esi", "edi", "edx"];
static REG_NAMES_16: &'static [&'static str; 8] = &["ax", "bx", "cx", "sp", "bp", "si", "di", "dx"];
static REG_NAMES_8: &'static [&'static str; 8] =
    &["al", "bl", "cl", "spl", "bpl", "sil", "dil", "dl"];
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct X64Arch {
    pub apx: bool,
}
#[derive(Clone)]
#[non_exhaustive]
pub struct RegFormatOpts {
    pub arch: X64Arch,
    pub size: MemorySize,
}
impl RegFormatOpts {
    pub fn default_with_arch(arch: X64Arch) -> Self {
        Self::default_with_arch_and_size(arch, Default::default())
    }
    pub fn default_with_arch_and_size(arch: X64Arch, size: MemorySize) -> Self {
        Self { arch, size }
    }
}
impl Default for RegFormatOpts {
    fn default() -> Self {
        Self::default_with_arch(Default::default())
    }
}
const RSP: Reg = Reg(3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum X64Label {
    Indexed { idx: usize },
    Func { r#fn: u32 },
}
impl Display for X64Label {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            X64Label::Indexed { idx } => write!(f, "_idx_{idx}"),
            X64Label::Func { r#fn } => write!(f, "f{}", r#fn),
        }
    }
}
pub trait Label: portal_solutions_blitz_common::Label<X64Label> {}
impl<T: portal_solutions_blitz_common::Label<X64Label> + ?Sized> Label for T {}
pub mod naive;
pub mod out;
pub mod reg;
