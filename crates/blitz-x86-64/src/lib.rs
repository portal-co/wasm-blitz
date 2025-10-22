#![no_std]
use alloc::vec::Vec;
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
use portal_solutions_blitz_common::{
    asm::common::mem::MemorySize,
    asm::Reg,
    ops::{FnData, MachOperator},
    wasmparser::Operator,
};
extern crate alloc;
pub use portal_solutions_asm_x86_64::*;
const RSP: Reg = Reg(4);
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
