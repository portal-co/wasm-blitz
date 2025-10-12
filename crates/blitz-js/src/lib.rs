#![no_std]
use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};
#[doc(hidden)]
pub mod __ {
    pub use portal_solutions_blitz_common::DisplayFn;
}
use alloc::vec::Vec;
use portal_solutions_blitz_common::{
    DisplayFn, MachOperator,
    wasmparser::{Operator, ValType},
};
extern crate alloc;
const STACK_WEAVE: &'static str = "($$stack_restore_symbol_iterator ?? (a=>a))";
pub fn push(w: &mut (impl Write + ?Sized), a: &(dyn Display + '_)) -> core::fmt::Result {
    write!(w, "(tmp={a},stack=[...{STACK_WEAVE}(stack),tmp],tmp)")
}
pub fn pop(w: &mut (impl Write + ?Sized)) -> core::fmt::Result {
    write!(w, "(([...stack,tmp]={STACK_WEAVE}(stack)),tmp)")
}
#[macro_export]
macro_rules! pop {
    () => {
        $crate::__::DisplayFn(&|f| $crate::pop(f))
    };
}
#[derive(Default)]
pub struct State {
    stack: Vec<Frame>,
}
enum Frame {
    Block,
    Loop,
    If,
}
pub trait JsWrite: Write {
    fn call(&mut self, function_index: &(dyn Display + '_)) -> core::fmt::Result {
        write!(
            self,
            "args=[];
            for(let i = 0;i < {function_index}.__sig.params;i++)args=[...{STACK_WEAVE}(args),{}];
            tmp_locals=[...{STACK_WEAVE}({function_index}(...args))];
            if(tmp_locals.length==={function_index}.__sig.rets){{stack=[...{STACK_WEAVE}(stack),...{STACK_WEAVE}(tmp_locals)];}}else{{for(let i = 0;i < {function_index}.__sig.rets;i++)stack=[...{STACK_WEAVE}(stack),tmp_locals[i]];}};",
            pop!()
        )
    }
    fn br(&mut self, state: &State, idx: u32) -> core::fmt::Result {
        let (idx, frame) = state
            .stack
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, a)| !matches!(a, Frame::If))
            .nth(idx as usize)
            .unwrap();
        let idx = idx + 1;
        match frame {
            Frame::Block => write!(self, "{{stack=[];break l{idx};}}"),
            Frame::Loop => write!(self, "{{stack=[];continue l{idx};}}"),
            _ => todo!(),
        }
    }
    fn on_mach<Annot>(
        &mut self,
        func_imports: &[(&str, &str)],
        state: &mut State,
        m: &MachOperator<'_, Annot>,
    ) -> core::fmt::Result {
        match m {
            MachOperator::StartFn { id, data } => {
                let id = *id + func_imports.len() as u32;
                write!(
                    self,
                    "
                    Object.defineProperty(${id},'__sig',{{
                        value:Object.freeze({{
                            params:{},
                            rets:{}
                        }}),
                        enumerable:false,
                        configurable:false,
                        writable:false
                    }});
                    function ${id}(...locals){{let stack=[],tmp,mask32=0xffff_ffffn,mask64=(mask32<<32n)|mask32,{{params,rets}}=${id}.__sig,tmp_locals=[],args=[];if(locals.length!==params){{for(let i = 0; i < params;i++)tmp_locals=[...{STACK_WEAVE}(tmp_locals),locals[locals.length - params + i]];locals=tmp_locals;}};const toInt=(a,b)=>BigInt.asIntN(b,a);const toUint=(a,b)=>BigInt.asUintN(b,a)",
                    data.num_params, data.num_returns
                )
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    write!(
                        self,
                        "locals=[...{STACK_WEAVE}(locals),{}];",
                        match ty {
                            ValType::F32 | ValType::F64 => "0",
                            _ => "0n",
                        }
                    )?
                }
                Ok(())
            }
            MachOperator::StartBody => Ok(()),

            MachOperator::Operator { op, annot } => {
                let Some(op) = op.as_ref() else{
                    return Ok(());
                };
                match op {
                    Operator::I64Const { value } => push(self, &format_args!("{}n", *value as u64)),
                    Operator::I32Const { value } => {
                        push(self, &format_args!("{}n", *value as u32 as u64))
                    }
                    Operator::I64Eqz | Operator::I32Eqz => {
                        push(self, &format_args!("({}===0n?1n:0n)", pop!()))
                    }
                    Operator::I32Add => push(
                        self,
                        &format_args!("((a={},b={0})=>(a+b)&mask32)()", pop!()),
                    ),
                    Operator::I32Sub => push(
                        self,
                        &format_args!("((a={},b={0})=>toUint((a-b)&mask32,32))()", pop!()),
                    ),
                    Operator::I32Mul => push(
                        self,
                        &format_args!("((a={},b={0})=>(a*b)&mask32)()", pop!()),
                    ),
                    Operator::I32DivU => push(
                        self,
                        &format_args!("((a={},b={0})=>(a/b)&mask32)()", pop!()),
                    ),
                    Operator::I32RemU => push(
                        self,
                        &format_args!("((a={},b={0})=>(a%b)&mask32)()", pop!()),
                    ),
                    Operator::I32DivS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},32),b=toInt({0},32))=>toUint((a/b)&mask32))()",
                            pop!()
                        ),
                    ),
                    Operator::I32RemS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},32),b=toInt({0},32))=>toUint((a%b)&mask32))()",
                            pop!()
                        ),
                    ),
                    Operator::I32Shl => push(
                        self,
                        &format_args!("((a={},b={0}%32n)=>(a<<b)&mask32)()", pop!()),
                    ),
                    Operator::I32ShrU => push(
                        self,
                        &format_args!("((a={},b={0}%32n)=>(a>>b)&mask32)()", pop!()),
                    ),
                    Operator::I32ShrS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},32),b={0}%32n)=>toUint((a>>b)&mask32),32)()",
                            pop!()
                        ),
                    ),
                    Operator::I32Rotl => push(
                        self,
                        &format_args!("((a={},b={0}%32n)=>((a<<b)|(a>>(32n-b)))&mask32)()", pop!()),
                    ),
                    Operator::I32Rotr => push(
                        self,
                        &format_args!("((a={},b={0}%32n)=>((a>>b)|(a<<(32n-b)))&mask32)()", pop!()),
                    ),

                    // 64 bit
                    Operator::I64Add => push(
                        self,
                        &format_args!("((a={},b={0})=>(a+b)&mask64)()", pop!()),
                    ),
                    Operator::I64Sub => push(
                        self,
                        &format_args!("((a={},b={0})=>toUint((a-b)&mask64,64))()", pop!()),
                    ),
                    Operator::I64Mul => push(
                        self,
                        &format_args!("((a={},b={0})=>(a*b)&mask64)()", pop!()),
                    ),
                    Operator::I64DivU => push(
                        self,
                        &format_args!("((a={},b={0})=>(a/b)&mask64)()", pop!()),
                    ),
                    Operator::I64RemU => push(
                        self,
                        &format_args!("((a={},b={0})=>(a%b)&mask64)()", pop!()),
                    ),
                    Operator::I64DivS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},64),b=toInt({0},64))=>toUint((a/b)&mask64))()",
                            pop!()
                        ),
                    ),
                    Operator::I64RemS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},64),b=toInt({0},64))=>toUint((a%b)&mask64))()",
                            pop!()
                        ),
                    ),
                    Operator::I64Shl => push(
                        self,
                        &format_args!("((a={},b={0}%64n)=>(a<<b)&mask64)()", pop!()),
                    ),
                    Operator::I64ShrU => push(
                        self,
                        &format_args!("((a={},b={0}%64n)=>(a>>b)&mask64)()", pop!()),
                    ),
                    Operator::I64ShrS => push(
                        self,
                        &format_args!(
                            "((a=toInt({},64),b={0}%64n)=>toUint((a>>b)&mask64),64)()",
                            pop!()
                        ),
                    ),
                    Operator::I64Rotl => push(
                        self,
                        &format_args!("((a={},b={0}%64n)=>((a<<b)|(a>>(64n-b)))&mask64)()", pop!()),
                    ),
                    Operator::I64Rotr => push(
                        self,
                        &format_args!("((a={},b={0}%64n)=>((a>>b)|(a<<(64n-b)))&mask64)()", pop!()),
                    ),
                    //
                    Operator::Return => {
                        write!(
                            self,
                            "if(stack.length===rets)return stack;tmp_locals=[];for(let i = 0; i < rets;i++)tmp_locals=[...{STACK_WEAVE}(tmp_locals),stack[stack.length-rets+i]];return tmp_locals;"
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
                    Operator::Block { blockty } => {
                        state.stack.push(Frame::Block);
                        write!(self, "l{}: for(;;){{", state.stack.len())
                    }
                    Operator::Loop { blockty } => {
                        state.stack.push(Frame::Loop);
                        write!(self, "l{}: for(;;){{", state.stack.len())
                    }
                    Operator::If { blockty } => {
                        state.stack.push(Frame::If);
                        write!(self, "if({}){{", pop!())
                    }
                    Operator::Else => {
                        write!(self, "}}else{{")
                    }
                    Operator::End => {
                        let s = state.stack.pop();
                        match s.unwrap() {
                            Frame::Block | Frame::Loop => write!(self, "break;")?,
                            _ => {}
                        }
                        write!(self, "}}")
                    }
                    Operator::Br { relative_depth } => self.br(state, *relative_depth),
                    Operator::BrIf { relative_depth } => write!(
                        self,
                        "if({}!==0n){}",
                        pop!(),
                        DisplayFn(&|f| f.br(state, *relative_depth))
                    ),
                    Operator::BrTable { targets } => {
                        write!(self, "{}", pop!())?;
                        for t in targets.targets().flatten() {
                            write!(
                                self,
                                "if(tmp===0n){{{}}};tmp--;",
                                DisplayFn(&|f| f.br(state, t))
                            )?;
                        }
                        self.br(state, targets.default())?;
                        Ok(())
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
