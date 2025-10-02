#![no_std]
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
#[doc(hidden)]
pub mod __ {
    pub use portal_solutions_blitz_common::DisplayFn;
}
use portal_solutions_blitz_common::{
    DisplayFn, MachOperator,
    wasmparser::{Operator, ValType},
};
extern crate alloc;
pub fn push(w: &mut (impl Write + ?Sized), a: &(dyn Display + '_)) -> core::fmt::Result {
    write!(w, "(stack=[...stack,{a}])")
}
pub fn pop(w: &mut (impl Write + ?Sized)) -> core::fmt::Result {
    write!(w, "(([...stack,tmp]=stack),tmp)")
}
#[macro_export]
macro_rules! pop {
    () => {
        $crate::__::DisplayFn(&|f| $crate::pop(f))
    };
}
pub trait JsWrite: Write {
    fn on_mach(&mut self, m: &MachOperator<'_>) -> core::fmt::Result {
        match m {
            MachOperator::StartFn { id, data } => {
                write!(
                    self,
                    "function ${id}(...locals){{let stack=[],tmp,mask32=0xffff_ffffn,mask64=(mask32<<32n)|mask32;"
                )
            }
            MachOperator::Local(a, b) => write!(
                self,
                "locals=[...locals,{}];",
                match b {
                    ValType::F32 | ValType::F64 => "0",
                    _ => "0n",
                }
            ),
            MachOperator::StartBody => Ok(()),

            MachOperator::Operator(o) => {
                match o {
                    Operator::I64Const { value } => push(self, &format_args!("{}n", *value as u64)),
                    Operator::I32Const { value } => {
                        push(self, &format_args!("{}n", *value as u32 as u64))
                    }
                    Operator::I64Eqz | Operator::I32Eqz => {
                        push(self, &format_args!("({}===0n?1n:0n)", pop!()))
                    }
                    _ => todo!(),
                }?;
                write!(self, ";")?;
                Ok(())
            }
            MachOperator::EndBody => write!(self, "}}"),
            _ => todo!(),
        }
    }
}
impl<T: Write + ?Sized> JsWrite for T {}
