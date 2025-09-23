#![no_std]

use core::{
    error::Error,
    fmt::{Formatter, Write},
};
extern crate alloc;
static reg_names: &'static [&'static str; 16] = &[
    "rax", "rbx", "rcx", "rsp", "rbp", "rsi", "rdi", "rdx", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

const RSP: u8 = 3;
pub trait Writer {
    type Error: Error;
    fn set_label(&mut self, s: &str) -> Result<(), Self::Error>;
    fn xchg(&mut self, dest: u8, src: u8, mem: Option<usize>) -> Result<(), Self::Error>;
    fn push(&mut self, op: u8) -> Result<(), Self::Error>;
    fn pop(&mut self, op: u8) -> Result<(), Self::Error>;
    fn call(&mut self, op: u8) -> Result<(), Self::Error>;
    fn lea(
        &mut self,
        dest: u8,
        src: u8,
        offset: isize,
        off_reg: Option<(u8, usize)>,
    ) -> Result<(), Self::Error>;
    fn lea_label(&mut self, dest: u8, label: &str) -> Result<(), Self::Error>;
    fn get_ip(&mut self) -> Result<(), Self::Error>;
    fn ret(&mut self) -> Result<(), Self::Error>;
}
pub trait WriterExt: Writer {}
impl<T: Writer + ?Sized> WriterExt for T {}
macro_rules! writers {
    ($($ty:ty),*) => {
        const _: () = {
            $(impl Writer for $ty {
                type Error = core::fmt::Error;
                fn set_label(&mut self, s: &str) -> Result<(), Self::Error> {
                    write!(self, "{s}:\n")
                }
                fn xchg(&mut self, dest: u8, src: u8, mem: Option<usize>) -> Result<(),Self::Error>{
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
                 fn lea_label(&mut self, dest: u8, label: &str) -> Result<(),Self::Error>{
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
            })*
        };
    };
}
macro_rules! writer_dispatch {
    ($( [ $($t:tt)* ] $ty:ty => $e:ty),*) => {
        const _: () = {
            $(impl<$($t)*> Writer for $ty{
                type Error = $e;

                fn set_label(&mut self, s: &str) -> Result<(), Self::Error> {
                    Writer::set_label(&mut **self, s)
                }

                fn xchg(&mut self, dest: u8, src: u8, mem: Option<usize>) -> Result<(), Self::Error> {
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

                fn lea(
                    &mut self,
                    dest: u8,
                    src: u8,
                    offset: isize,
                    off_reg: Option<(u8, usize)>,
                ) -> Result<(), Self::Error> {
                    Writer::lea(&mut **self, dest, src, offset, off_reg)
                }

                fn lea_label(&mut self, dest: u8, label: &str) -> Result<(), Self::Error> {
                    Writer::lea_label(&mut **self, dest, label)
                }
                fn get_ip(&mut self) -> Result<(), Self::Error>{
                    Writer::get_ip(&mut **self)
                }
                fn ret(&mut self) -> Result<(), Self::Error>{
                    Writer::ret(&mut **self)
                }
            })*
        };
    };
}
writers!(Formatter<'_>, (dyn Write + '_));
writer_dispatch!([ T: Writer + ?Sized ] &'_ mut T => T::Error);
