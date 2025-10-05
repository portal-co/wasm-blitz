#![no_std]

use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};

use alloc::vec::Vec;
use portal_solutions_blitz_common::{FnData, MachOperator, wasmparser::Operator};
extern crate alloc;
static reg_names: &'static [&'static str; 16] = &[
    "rax", "rbx", "rcx", "rsp", "rbp", "rsi", "rdi", "rdx", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct RegFormatOpts {}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Reg(pub u8);
impl Reg {
    pub fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result {
        write!(f, "{}", &reg_names[(self.0 as usize) % 16])
    }
    pub fn display<'a>(&'a self, opts: RegFormatOpts) -> RegDisplay<'a> {
        RegDisplay { reg: self, opts }
    }
}
pub struct RegDisplay<'a> {
    reg: &'a Reg,
    opts: RegFormatOpts,
}
impl Display for RegDisplay<'_> {
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

pub mod nieve;
