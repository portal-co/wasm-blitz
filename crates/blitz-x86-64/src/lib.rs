#![no_std]

use core::{
    error::Error,
    fmt::{Display, Formatter, Write},
};

use alloc::vec::Vec;
use portal_solutions_blitz_common::{MachOperator, wasmparser::Operator};
extern crate alloc;
static reg_names: &'static [&'static str; 16] = &[
    "rax", "rbx", "rcx", "rsp", "rbp", "rsi", "rdi", "rdx", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

const RSP: u8 = 3;
pub trait Writer {
    type Error: Error;
    fn set_label(&mut self, s: &(dyn Display + '_)) -> Result<(), Self::Error>;
    fn xchg(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(), Self::Error>;
    fn mov(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(), Self::Error>;
    fn push(&mut self, op: u8) -> Result<(), Self::Error>;
    fn pop(&mut self, op: u8) -> Result<(), Self::Error>;
    fn call(&mut self, op: u8) -> Result<(), Self::Error>;
    fn jmp(&mut self, op: u8) -> Result<(), Self::Error>;
    fn cmp0(&mut self, op: u8) -> Result<(), Self::Error>;
    fn cmovz64(&mut self, op: u8, val: u64) -> Result<(), Self::Error>;
    fn jz(&mut self, op: u8) -> Result<(), Self::Error>;
    fn u32(&mut self, op: u8) -> Result<(), Self::Error>;
    fn not(&mut self, op: u8) -> Result<(), Self::Error>;
    fn lea(
        &mut self,
        dest: u8,
        src: u8,
        offset: isize,
        off_reg: Option<(u8, usize)>,
    ) -> Result<(), Self::Error>;
    fn lea_label(&mut self, dest: u8, label: &(dyn Display + '_)) -> Result<(), Self::Error>;
    fn get_ip(&mut self) -> Result<(), Self::Error>;
    fn ret(&mut self) -> Result<(), Self::Error>;
    fn mov64(&mut self, r: u8, val: u64) -> Result<(), Self::Error>;
    fn mul(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
    fn div(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
    fn idiv(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
    fn and(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
    fn or(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
    fn eor(&mut self, a: u8, b: u8) -> Result<(), Self::Error>;
}
#[derive(Default)]
pub struct State {
    local_count: usize,
    num_returns: usize,
    control_depth: usize,
    label_index: usize,
    if_stack: Vec<Endable>,
}
// #[derive(Clone)]
enum Endable {
    Br,
    If { idx: usize },
}
pub trait WriterExt: Writer {
    fn br(&mut self, state: &mut State, relative_depth: u32) -> Result<(), Self::Error> {
        self.xchg(RSP, 15, Some(8))?;
        for _ in 0..=relative_depth {
            self.pop(0)?;
            self.pop(1)?;
        }
        self.xchg(RSP, 15, Some(8))?;
        self.mov(RSP, 1, None)?;
        self.jmp(0)?;
        Ok(())
    }
    fn handle_op(&mut self, state: &mut State, op: &MachOperator<'_>) -> Result<(), Self::Error> {
        //Stack Frame: r15[0] => local variable frame
        match op {
            MachOperator::StartFn {
                id: f,
                num_params: params,
                num_returns,
                control_depth,
            } => {
                state.local_count = *params;
                state.num_returns = *num_returns;
                state.control_depth = *control_depth;
                self.pop(1)?;
                self.lea(0, 3, -(*params as isize), None)?;
                self.xchg(0, 15, Some(0))?;
                self.set_label(&format_args!("f{f}"))?;
            }
            MachOperator::Local(a, b) => {
                state.local_count += 1;
                self.push(0)?;
            }
            MachOperator::StartBody => {
                self.push(1)?;
                self.push(0)?;
                self.lea(0, RSP, -(state.control_depth as isize * 16), None)?;
                self.xchg(0, 15, Some(8))?;
                self.push(0)?;
                for _ in 0..state.control_depth {
                    for _ in 0..2 {
                        self.push(0)?;
                    }
                }
            }
            MachOperator::Operator(op) => match op {
                Operator::I32Const { value } => {
                    self.mov64(0, *value as u32 as u64)?;
                    self.push(0)?;
                }
                Operator::I64Const { value } => {
                    self.mov64(0, *value as u64)?;
                    self.push(0)?;
                }
                Operator::I32Add | Operator::I64Add => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.lea(0, 0, 0, Some((1, 0)))?;
                    if let Operator::I32Add = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32Sub | Operator::I64Sub => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.not(1)?;
                    self.lea(0, 0, 1, Some((1, 0)))?;
                    if let Operator::I32Sub = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32Mul | Operator::I64Mul => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.mul(0, 1)?;
                    if let Operator::I32Mul = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32DivU | Operator::I64DivU => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.div(0, 1)?;
                    if let Operator::I32DivU = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32DivS | Operator::I64DivS => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.idiv(0, 1)?;
                    if let Operator::I32DivS = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32RemU | Operator::I64RemU => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.div(0, 1)?;
                    if let Operator::I32RemU = op {
                        self.u32(7)?;
                    }
                    self.push(7)?;
                }
                Operator::I32RemS | Operator::I64RemS => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.idiv(0, 1)?;
                    if let Operator::I32RemS = op {
                        self.u32(7)?;
                    }
                    self.push(7)?;
                }
                Operator::I32And | Operator::I64And => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.and(0, 1)?;
                    if let Operator::I32And = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32Or | Operator::I64Or => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.or(0, 1)?;
                    if let Operator::I32Or = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32Xor | Operator::I64Xor => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.eor(0, 1)?;
                    if let Operator::I32Xor = op {
                        self.u32(0)?;
                    }
                    self.push(0)?;
                }
                Operator::I32Eqz | Operator::I64Eqz => {
                    self.pop(0)?;
                    self.mov64(1, 0)?;
                    self.cmp0(0)?;
                    self.cmovz64(1, 1)?;
                    self.push(1)?;
                }
                Operator::I32Eq | Operator::I64Eq => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.not(1)?;
                    self.lea(0, 0, 1, Some((1, 0)))?;
                    self.mov64(1, 0)?;
                    self.cmp0(0)?;
                    self.cmovz64(1, 1)?;
                    self.push(1)?;
                }
                Operator::I32Ne | Operator::I64Ne => {
                    self.pop(0)?;
                    self.pop(1)?;
                    self.not(1)?;
                    self.lea(0, 0, 1, Some((1, 0)))?;
                    self.mov64(1, 1)?;
                    self.cmp0(0)?;
                    self.cmovz64(1, 0)?;
                    self.push(1)?;
                }
                Operator::LocalGet { local_index } => {
                    self.xchg(RSP, 15, Some(0))?;
                    self.lea(RSP, RSP, -((*local_index as i32 as isize) * 8), None)?;
                    self.pop(0)?;
                    self.lea(RSP, RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                    self.xchg(RSP, 15, Some(0))?;
                    self.push(0)?;
                }
                Operator::LocalTee { local_index } => {
                    self.pop(0)?;
                    self.xchg(RSP, 15, Some(0))?;
                    self.lea(RSP, RSP, -((*local_index as i32 as isize) * 8), None)?;
                    self.push(0)?;
                    self.lea(RSP, RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                    self.xchg(RSP, 15, Some(0))?;
                    self.push(0)?;
                }
                Operator::LocalSet { local_index } => {
                    self.pop(0)?;
                    self.xchg(RSP, 15, Some(0))?;
                    self.lea(RSP, RSP, -((*local_index as i32 as isize) * 8), None)?;
                    self.push(0)?;
                    self.lea(RSP, RSP, ((*local_index as i32 as isize + 1) * 8), None)?;
                    self.xchg(RSP, 15, Some(0))?;
                }
                Operator::Return => {
                    self.mov(1, RSP, None)?;
                    self.mov(0, 15, Some(0))?;
                    self.lea(0, 0, (state.local_count + 3) as isize * 8, None)?;
                    self.mov(RSP, 0, None)?;
                    self.pop(0)?;
                    self.xchg(0, 15, Some(8))?;
                    self.pop(0)?;
                    self.xchg(0, 15, Some(0))?;
                    self.pop(0)?;
                    for a in 0..state.num_returns {
                        self.mov(2, 1, Some(-(a as isize * 8)))?;
                        self.push(2)?;
                    }
                    self.push(0)?;
                    self.ret()?;
                }
                Operator::Br { relative_depth } => {
                    self.br(state, *relative_depth)?;
                }
                Operator::BrIf { relative_depth } => {
                    let i = state.label_index;
                    state.label_index += 1;
                    self.lea_label(1, &format_args!("_end_{i}"))?;
                    self.br(state, *relative_depth)?;
                    self.set_label(&format_args!("_end_{i}"))?;
                }
                Operator::BrTable { targets } => {
                    for relative_depth in targets.targets().flatten() {
                        let i = state.label_index;
                        state.label_index += 1;
                        self.lea_label(1, &format_args!("_end_{i}"))?;
                        self.pop(0)?;
                        self.cmp0(0)?;
                        self.jz(1)?;
                        self.br(state, relative_depth)?;
                        self.set_label(&format_args!("_end_{i}"))?;
                        self.lea(0, 0, -1, None)?;
                        self.push(0)?;
                    }
                    self.pop(0)?;
                    self.br(state, targets.default())?;
                }
                Operator::Block { blockty } => {
                    state.if_stack.push(Endable::Br);
                    let i = state.label_index;
                    state.label_index += 1;
                    self.lea_label(0, &format_args!("_end_{i}"))?;
                    self.mov(1, RSP, None)?;
                    self.xchg(RSP, 15, Some(8))?;
                    // for _ in 0..=(*relative_depth) {
                    self.push(1)?;
                    self.push(0)?;
                    // }
                    self.xchg(RSP, 15, Some(8))?;
                    self.set_label(&format_args!("_end_{i}"))?;
                }
                Operator::If { blockty } => {
                    let i = state.label_index;
                    state.label_index += 3;
                    state.if_stack.push(Endable::If { idx: i });
                    self.pop(2)?;
                    self.lea_label(0, &format_args!("_end_{i}"))?;
                    self.lea_label(1, &format_args!("_end_{}", i + 1))?;
                    self.cmp0(2)?;
                    self.jz(1)?;
                    self.jmp(0)?;
                    self.set_label(&format_args!("_end_{i}"))?;
                }
                Operator::Else => {
                    let Endable::If { idx: i } = state.if_stack.last().unwrap() else {
                        todo!()
                    };
                    self.lea_label(0, &format_args!("_end_{}", i + 2))?;
                    self.jmp(0)?;
                    self.set_label(&format_args!("_end_{}", i + 1))?;
                }
                Operator::Loop { blockty } => {
                    state.if_stack.push(Endable::Br);
                    let i = state.label_index;
                    state.label_index += 1;
                    self.set_label(&format_args!("_end_{i}"))?;
                    self.lea_label(0, &format_args!("_end_{i}"))?;
                    self.mov(1, RSP, None)?;
                    self.xchg(RSP, 15, Some(8))?;
                    // for _ in 0..=(*relative_depth) {
                    self.push(1)?;
                    self.push(0)?;
                    // }
                    self.xchg(RSP, 15, Some(8))?;
                }
                Operator::End => {
                    self.xchg(RSP, 15, Some(8))?;
                    // for _ in 0..=(*relative_depth) {
                    match state.if_stack.pop().unwrap() {
                        Endable::Br => {
                            self.pop(0)?;
                            self.pop(1)?;
                        }
                        Endable::If { idx: i } => {
                            self.set_label(&format_args!("_end_{}", i + 2))?;
                        }
                    }
                    // }
                    self.xchg(RSP, 15, Some(8))?;
                }
                Operator::Call { function_index } => {
                    self.lea_label(0, &format_args!("f{function_index}"))?;
                    self.call(0)?;
                }
                _ => todo!(),
            },
            _ => todo!(),
        }
        Ok(())
    }
}
impl<T: Writer + ?Sized> WriterExt for T {}
macro_rules! writers {
    ($($ty:ty),*) => {
        const _: () = {
            $(impl Writer for $ty {
                type Error = core::fmt::Error;
                fn set_label(&mut self, s: &(dyn Display + '_)) -> Result<(), Self::Error> {
                    write!(self, "{s}:\n")
                }
                fn xchg(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(),Self::Error>{
                    let dest = &reg_names[(dest & 15) as usize];
                    let src = &reg_names[(src & 15) as usize];
                    write!(self,"xchg {dest}, ")?;
                    match mem{
                        None => write!(self,"{src}\n"),
                        Some(i) => write!(self,"qword ptr [{src}+{i}]\n")
                    }
                }
                fn push(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"push {op}\n")
                }
                fn pop(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"pop {op}\n")
                }
                fn call(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"call {op}\n")
                }
                 fn jmp(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"jmp {op}\n")
                }
                fn cmp0(&mut self, op: u8) -> Result<(),Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"cmp {op}, 0\n")
                }
                fn cmovz64(&mut self, op: u8,val:u64) -> Result<(), Self::Error>{
                     let op = &reg_names[(op & 15) as usize];
                    write!(self,"cmovz {op}, {val}\n")
                }
                fn jz(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"jz {op}\n")
                }
                fn u32(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"and {op}, 0xffffffff\n")
                }
                fn lea(&mut self, dest: u8, src: u8, offset: isize, off_reg: Option<(u8,usize)>) -> Result<(),Self::Error>{
                    let dest = &reg_names[(dest & 15) as usize];
                    let src = &reg_names[(src & 15) as usize];
                    write!(self,"lea {dest}, [{src}")?;
                    if let Some((r,m)) = off_reg{
                        let r = &reg_names[(r & 15) as usize];
                        write!(self,"+{r}*{m}")?;
                    }
                    write!(self,"+{offset}]\n")
                }
                fn mov(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(), Self::Error>{
                     let dest = &reg_names[(dest & 15) as usize];
                    let src = &reg_names[(src & 15) as usize];
                    write!(self,"mov {dest}, ")?;
                    match mem{
                        None => write!(self,"{src}\n"),
                        Some(i) => write!(self,"qword ptr [{src}+{i}]\n")
                    }
                }
                fn lea_label(&mut self, dest: u8, label: &(dyn Display + '_)) -> Result<(),Self::Error>{
                    let dest = &reg_names[(dest & 15) as usize];
                    write!(self,"lea {dest}, {label}\n")
                }
                fn get_ip(&mut self) -> Result<(),Self::Error>{
                //   let dest = &reg_names[(dest & 15) as usize];
                    write!(self,"call 1f\n1:\n")
                }
                fn ret(&mut self) -> Result<(), Self::Error>{
                    write!(self,"ret\n")
                }
                fn mov64(&mut self, r: u8, val: u64) -> Result<(),Self::Error>{
                    let r = &reg_names[(r & 15) as usize];
                    write!(self,"mov {r}, {val}\n")
                }
                fn not(&mut self, op: u8) -> Result<(), Self::Error>{
                    let op = &reg_names[(op & 15) as usize];
                    write!(self,"not {op}\n")
                }
                fn mul(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"mul {a},{b}\n")
                }
                fn div(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"div {a},{b}\n")
                }
                fn idiv(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"idiv {a},{b}\n")
                }
                fn and(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"and {a},{b}\n")
                }
                fn or(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"or {a},{b}\n")
                }
                fn eor(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    let a = &reg_names[(a & 15) as usize];
                    let b = &reg_names[(b & 15) as usize];
                    write!(self,"eor {a},{b}\n")
                }
            })*
        };
    };
}
macro_rules! writer_dispatch {
    ($( [ $($t:tt)* ] $ty:ty => $e:ty),*) => {
        const _: () = {
            $(impl<$($t)*> Writer for $ty{
                type Error = $e;

                fn set_label(&mut self, s: &(dyn Display + '_)) -> Result<(), Self::Error> {
                    Writer::set_label(&mut **self, s)
                }

                fn xchg(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(), Self::Error> {
                    Writer::xchg(&mut **self, dest, src, mem)
                }

                fn push(&mut self, op: u8) -> Result<(), Self::Error> {
                    Writer::push(&mut **self, op)
                }

                fn pop(&mut self, op: u8) -> Result<(), Self::Error> {
                    Writer::pop(&mut **self, op)
                }
                fn call(&mut self, op: u8) -> Result<(), Self::Error>{
                    Writer::call(&mut **self,op)
                }
                fn jmp(&mut self, op: u8) -> Result<(), Self::Error>{
                    Writer::jmp(&mut **self,op)
                }
                fn cmp0(&mut self, op: u8) -> Result<(),Self::Error>{
                    Writer::cmp0(&mut **self,op)
                }
                fn cmovz64(&mut self, op: u8,val:u64) -> Result<(), Self::Error>{
                    Writer::cmovz64(&mut **self,op,val)
                }
                fn jz(&mut self, op: u8) -> Result<(), Self::Error>{
                    Writer::jz(&mut **self,op)
                }

                fn lea(
                    &mut self,
                    dest: u8,
                    src: u8,
                    offset: isize,
                    off_reg: Option<(u8, usize)>,
                ) -> Result<(), Self::Error> {
                    Writer::lea(&mut **self, dest, src, offset, off_reg)
                }

                fn lea_label(&mut self, dest: u8, label: &(dyn Display + '_)) -> Result<(), Self::Error> {
                    Writer::lea_label(&mut **self, dest, label)
                }
                fn get_ip(&mut self) -> Result<(), Self::Error>{
                    Writer::get_ip(&mut **self)
                }
                fn ret(&mut self) -> Result<(), Self::Error>{
                    Writer::ret(&mut **self)
                }
                fn mov64(&mut self, r: u8, val: u64) -> Result<(),Self::Error>{
                    Writer::mov64(&mut **self,r,val)
                }
                fn mov(&mut self, dest: u8, src: u8, mem: Option<isize>) -> Result<(), Self::Error>{
                    Writer::mov(&mut **self,dest,src,mem)
                }
                fn u32(&mut self, op: u8) -> Result<(), Self::Error>{
                    Writer::u32(&mut **self,op)
                }
                fn not(&mut self, op: u8) -> Result<(), Self::Error>{
                    Writer::not(&mut **self,op)
                }
                fn mul(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::mul(&mut **self,a,b)
                }
                fn div(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::div(&mut **self,a,b)
                }
                fn idiv(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::idiv(&mut **self,a,b)
                }
                fn and(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::and(&mut **self,a,b)
                }
                fn or(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::or(&mut **self,a,b)
                }
                fn eor(&mut self, a: u8, b: u8) -> Result<(), Self::Error>{
                    Writer::eor(&mut **self,a,b)
                }
            })*
        };
    };
}
writers!(Formatter<'_>, (dyn Write + '_));
writer_dispatch!([ T: Writer + ?Sized ] &'_ mut T => T::Error);
