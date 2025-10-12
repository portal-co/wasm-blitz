#![no_std]

use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};

use alloc::vec::Vec;
use portal_solutions_blitz_common::{FnData, MachOperator, wasmparser::Operator};
extern crate alloc;
static REG_NAMES: &'static [&'static str; 8] =
    &["rax", "rbx", "rcx", "rsp", "rbp", "rsi", "rdi", "rdx"];
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct X64Arch {
    pub apx: bool,
}
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct RegFormatOpts {
    pub arch: X64Arch,
}
impl RegFormatOpts {
    pub fn default_with_arch(arch: X64Arch) -> Self {
        Self { arch }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Reg(pub u8);
impl Reg {
    pub const CTX: Reg = Reg(255);
    pub fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result {
        let idx = (self.0 as usize) % (if opts.arch.apx { 32 } else { 16 });
        if idx < 8 {
            write!(f, "{}", &REG_NAMES[idx])
        } else {
            write!(f, "r{idx}")
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
