#![no_std]

use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};

use alloc::vec::Vec;
use portal_solutions_blitz_common::{
    MemorySize,
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
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct RegFormatOpts {
    pub arch: X64Arch,
    pub size: MemorySize,
}
impl RegFormatOpts {
    pub fn default_with_arch(arch: X64Arch) -> Self {
        Self {
            arch,
            size: Default::default(),
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Reg(pub u8);
impl Reg {
    pub const CTX: Reg = Reg(255);
    pub fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result {
        let idx = (self.0 as usize) % (if opts.arch.apx { 32 } else { 16 });
        if idx < 8 {
            write!(
                f,
                "{}",
                &(match &opts.size {
                    MemorySize::_8 => REG_NAMES_8,
                    MemorySize::_16 => REG_NAMES_16,
                    MemorySize::_32 => REG_NAMES_32,
                    MemorySize::_64 => REG_NAMES,
                })[idx]
            )
        } else {
            write!(
                f,
                "r{idx}{}",
                match &opts.size {
                    MemorySize::_8 => "b",
                    MemorySize::_16 => "w",
                    MemorySize::_32 => "d",
                    MemorySize::_64 => "",
                }
            )
        }
    }
    pub fn display<'a>(&'a self, opts: RegFormatOpts) -> RegDisplay {
        RegDisplay { reg: *self, opts }
    }
}
impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format(f, &Default::default())
    }
}
pub struct RegDisplay {
    reg: Reg,
    opts: RegFormatOpts,
}
impl Display for RegDisplay {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.reg.format(f, &self.opts)
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
