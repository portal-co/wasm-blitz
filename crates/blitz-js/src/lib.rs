#![no_std]
use core::{
    cell::OnceCell,
    error::Error,
    fmt::{Display, Formatter, Write},
};
#[doc(hidden)]
pub mod __ {
    pub use portal_solutions_blitz_common::DisplayFn;
}
use alloc::vec::Vec;
use portal_solutions_blitz_common::{
    DisplayFn,
    ops::MachOperator,
    wasm_encoder::{BlockType, FuncType, Instruction, reencode::Reencode},
    wasmparser::{Operator, ValType},
};
use spin::Mutex;
extern crate alloc;
const STACK_WEAVE: &'static str = "($$stack_restore_symbol_iterator ?? (a=>a))";
pub fn push(
    state: &State,
    w: &mut (impl Write + ?Sized),
    a: &(dyn Display + '_),
) -> core::fmt::Result {
    write!(w, "(tmp={a},stack=[...{STACK_WEAVE}(stack),tmp],tmp)")?;
    if let Some(o) = state.opt() {
        let mut o = o.lock();
        o.depth += 1;
    }
    Ok(())
}
pub fn pop(state: &State, w: &mut (impl Write + ?Sized)) -> core::fmt::Result {
    write!(w, "(([...stack,tmp]={STACK_WEAVE}(stack)),tmp)")?;
    if let Some(o) = state.opt() {
        let mut o = o.lock();
        o.depth -= 1;
    }
    Ok(())
}
#[macro_export]
macro_rules! pop {
    ($state:ident) => {
        $crate::__::DisplayFn(&|f| match $state {
            ref state => $crate::pop(state, f),
        })
    };
}
#[derive(Default)]
#[non_exhaustive]
pub struct State {
    stack: Vec<Frame>,
    opt_state: OnceCell<Mutex<OptState>>,
}
#[derive(Default)]
#[non_exhaustive]
pub struct OptState {
    depth: usize,
}
impl State {
    pub fn enable_opt(&self, opt: impl FnOnce() -> OptState) {
        self.opt_state.get_or_init(|| Mutex::new(opt()));
    }
    fn opt(&self) -> Option<&Mutex<OptState>> {
        self.opt_state.get()
    }
}
enum Frame {
    Block(BlockType),
    Loop(BlockType),
    If,
}
pub trait JsWrite: Write {
    fn call(&mut self, state: &State, function_index: &(dyn Display + '_)) -> core::fmt::Result {
        write!(
            self,
            "args=[];
            for(let i = 0;i < {function_index}.__sig.params;i++)args=[...{STACK_WEAVE}(args),{}];
            tmp_locals=[...{STACK_WEAVE}({function_index}(...args))];
            if(tmp_locals.length==={function_index}.__sig.rets){{stack=[...{STACK_WEAVE}(stack),...{STACK_WEAVE}(tmp_locals)];}}else{{for(let i = 0;i < {function_index}.__sig.rets;i++)stack=[...{STACK_WEAVE}(stack),tmp_locals[i]];}};",
            pop!(state)
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
            Frame::Block(_) => write!(self, "{{stack=[];break l{idx};}}"),
            Frame::Loop(_) => write!(self, "{{stack=[];continue l{idx};}}"),
            _ => todo!(),
        }
    }
    fn on_op(
        &mut self,
        sigs: &[FuncType],
        func_imports: &[(&str, &str)],
        state: &mut State,
        op: &Instruction<'_>,
    ) -> core::fmt::Result {
        match op {
            Instruction::I64Const(value) => push(state, self, &format_args!("{}n", *value as u64)),
            Instruction::I32Const(value) => {
                push(state, self, &format_args!("{}n", *value as u32 as u64))
            }
            Instruction::I64Eqz | Instruction::I32Eqz => {
                push(state, self, &format_args!("({}===0n?1n:0n)", pop!(state)))
            }
            Instruction::I32Add => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a+b)&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32Sub => push(
                state,
                self,
                &format_args!(
                    "((a={},b={})=>toUint((a-b)&mask32,32))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Mul => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a*b)&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32DivU => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a/b)&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32RemU => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a%b)&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32DivS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},32),b=toInt({},32))=>toUint((a/b)&mask32))()",
                    // pop!()
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32RemS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},32),b=toInt({},32))=>toUint((a%b)&mask32))()",
                    // pop!()
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Shl => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%32n)=>(a<<b)&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32ShrU => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%32n)=>(a>>b)&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32ShrS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},32),b={}%32n)=>toUint((a>>b)&mask32),32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Rotl => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%32n)=>((a<<b)|(a>>(32n-b)))&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Rotr => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%32n)=>((a>>b)|(a<<(32n-b)))&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // 64 bit
            Instruction::I64Add => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a+b)&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64Sub => push(
                state,
                self,
                &format_args!(
                    "((a={},b={})=>toUint((a-b)&mask64,64))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64Mul => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a*b)&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64DivU => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a/b)&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64RemU => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a%b)&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64DivS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},64),b=toInt({},64))=>toUint((a/b)&mask64))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64RemS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},64),b=toInt({},64))=>toUint((a%b)&mask64))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64Shl => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%64n)=>(a<<b)&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64ShrU => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%64n)=>(a>>b)&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64ShrS => push(
                state,
                self,
                &format_args!(
                    "((a=toInt({},64),b={}%64n)=>toUint((a>>b)&mask64),64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64Rotl => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%64n)=>((a<<b)|(a>>(64n-b)))&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64Rotr => push(
                state,
                self,
                &format_args!(
                    "((a={},b={}%64n)=>((a>>b)|(a<<(64n-b)))&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            //
            Instruction::Return => {
                write!(
                    self,
                    "if(stack.length===rets)return stack;tmp_locals=[];for(let i = 0; i < rets;i++)tmp_locals=[...{STACK_WEAVE}(tmp_locals),stack[stack.length-rets+i]];return tmp_locals;"
                )
            }
            Instruction::Call(function_index) => {
                self.call(state, &format_args!("${function_index}"))
            }
            Instruction::LocalGet(local_index) => {
                push(state, self, &format_args!("locals[{local_index}]"))
            }
            Instruction::LocalSet(local_index) => {
                write!(self, "locals[{local_index}={}", pop!(state))
            }
            Instruction::LocalTee(local_index) => push(
                state,
                self,
                &format_args!("locals[{local_index}={}", pop!(state)),
            ),
            Instruction::Block(blockty) => {
                state.stack.push(Frame::Block(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Result(
                            val_type,
                        ) => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                            sigs[*f as usize].params().len()
                        }
                    };
                }
                write!(self, "l{}: for(;;){{", state.stack.len())
            }
            Instruction::Loop(blockty) => {
                state.stack.push(Frame::Loop(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Result(
                            val_type,
                        ) => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                            sigs[*f as usize].params().len()
                        }
                    };
                }
                write!(self, "l{}: for(;;){{", state.stack.len())
            }
            Instruction::If(blockty) => {
                state.stack.push(Frame::If);
                write!(self, "if({}){{", pop!(state))
            }
            Instruction::Else => {
                write!(self, "}}else{{")
            }
            Instruction::End => {
                let s = state.stack.pop();
                match s.unwrap() {
                    Frame::Block(blockty) => {
                        write!(self, "break;")?;
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty{
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Result(val_type) => 1,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                            };
                        }
                    }
                    Frame::Loop(blockty) => {
                        write!(self, "break;")?;
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty{
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Result(val_type) => 1,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                            };
                        }
                    }
                    _ => {}
                }
                write!(self, "}}")
            }
            Instruction::Br(relative_depth) => self.br(state, *relative_depth),
            Instruction::BrIf(relative_depth) => write!(
                self,
                "if({}!==0n){}",
                pop!(state),
                DisplayFn(&|f| f.br(state, *relative_depth))
            ),
            Instruction::BrTable(targets, default) => {
                write!(self, "{}", pop!(state))?;
                for t in targets.iter().cloned() {
                    write!(
                        self,
                        "if(tmp===0n){{{}}};tmp--;",
                        DisplayFn(&|f| f.br(state, t))
                    )?;
                }
                self.br(state, *default)?;
                Ok(())
            }
            _ => todo!(),
        }?;
        Ok(())
    }
    fn on_mach<Annot>(
        &mut self,
        sigs: &[FuncType],
        func_imports: &[(&str, &str)],
        state: &mut State,
        m: &MachOperator<'_, Annot>,
        r: &mut impl Reencode,
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
            MachOperator::Instruction { op, annot } => {
                self.on_op(sigs, func_imports, state, op)?;
                write!(self, ";")?;
                Ok(())
            }
            MachOperator::Operator { op, annot } => {
                let Some(op) = op.as_ref() else {
                    return Ok(());
                };
                let Ok(op) = r.instruction(op.clone()) else {
                    return Ok(());
                };
                self.on_op(sigs, func_imports, state, &op)?;
                write!(self, ";")?;
                Ok(())
            }
            MachOperator::EndBody => write!(self, "}}"),
            _ => todo!(),
        }
    }
}
impl<T: Write + ?Sized> JsWrite for T {}
