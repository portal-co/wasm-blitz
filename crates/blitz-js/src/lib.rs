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
    write!(w, "(tmp={a},stack=[...stack,tmp],tmp)")
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
    fn call(&mut self, function_index: &(dyn Display + '_)) -> core::fmt::Result {
        write!(
            self,
            "args=[];for(let i = 0;i < {function_index}.__sig.params;i++)args=[...args,{}];stack=[...stack,{function_index}(...args)]",
            pop!()
        )
    }
    fn on_mach(
        &mut self,
        func_imports: &[(&str, &str)],
        m: &MachOperator<'_>,
    ) -> core::fmt::Result {
        match m {
            MachOperator::StartFn { id, data } => {
                let id = *id + func_imports.len() as u32;
                write!(
                    self,
                    "Object.defineProperty(${id},'__sig',{{value:Object.freeze({{params:{},rets:{}}}),enumerable:false,configurable:false,writable:false}});function ${id}(...locals){{let stack=[],tmp,mask32=0xffff_ffffn,mask64=(mask32<<32n)|mask32,{{params,rets}}=${id}.__sig,tmp_locals=[],args=[];for(let i = 0; i < params;i++)tmp_locals=[...tmp_locals,locals[locals.length - params + i]];locals=tmp_locals;",
                    data.num_params, data.num_returns
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
                    Operator::Return => {
                        write!(
                            self,
                            "tmp_locals=[];for(let i = 0; i < rets;i++)tmp_locals=[...tmp_locals,stack[stack.length-rets+i]];return tmp_locals;"
                        )
                    }
                    Operator::Call { function_index } => {
                        self.call(&format_args!("${function_index}"))
                    }
                    Operator::LocalGet { local_index } => {
                        push(self, &format_args!("locals[{local_index}]"))
                    }
                    Operator::LocalSet { local_index } => {
                        write!(self, "locals[{local_index}={}", pop!())
                    }
                    Operator::LocalTee { local_index } => {
                        push(self, &format_args!("locals[{local_index}={}", pop!()))
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
