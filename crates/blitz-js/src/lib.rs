#![no_std]
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};

use portal_solutions_blitz_common::{MachOperator, wasmparser::ValType};
extern crate alloc;
pub fn push(w: &mut impl Write, a: &(dyn Display + '_)) -> core::fmt::Result {
    write!(w, "(stack=[...stack,{a}])")
}
pub fn pop(w: &mut impl Write) -> core::fmt::Result {
    write!(w, "(([...stack,tmp]=stack),tmp)")
}
pub trait JsWrite: Write {
    fn on_mach(&mut self, m: &MachOperator<'_>) -> core::fmt::Result {
        match m {
            MachOperator::StartFn { id, data } => write!(self, "function ${id}(...locals){{let stack=[],tmp;"),
            MachOperator::Local(a, b) => write!(
                self,
                "locals=[...locals,{}];",
                match b {
                    ValType::F32 | ValType::F64 => "0",
                    _ => "0n",
                }
            ),
            MachOperator::StartBody => Ok(()),

            MachOperator::Operator(o) => match o{
                _ => todo!()
            },
            MachOperator::EndBody => write!(self, "}}"),
            _ => todo!(),
        }
    }
}
impl<T: Write + ?Sized> JsWrite for T {}
