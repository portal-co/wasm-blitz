use super::*;
macro_rules! writers {
    ($($ty:ty),*) => {
        const _: () = {
            $(impl Writer for $ty {
                type Error = core::fmt::Error;
                fn set_label(&mut self, s: &(dyn Label + '_)) -> Result<(), Self::Error> {
                    write!(self, "{s}:\n")
                }
                fn xchg(&mut self, dest: &(dyn Arg + '_), src: &(dyn Arg + '_), mem: Option<isize>) -> Result<(),Self::Error>{
                    let dest = dest.reg().display(RegFormatOpts::default());
                    let src = src.reg().display(RegFormatOpts::default());
                    write!(self,"xchg {dest}, ")?;
                    match mem{
                        None => write!(self,"{src}\n"),
                        Some(i) => write!(self,"qword ptr [{src}+{i}]\n")
                    }
                }
                fn push(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"push {op}\n")
                }
                fn pop(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"pop {op}\n")
                }
                fn call(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"call {op}\n")
                }
                 fn jmp(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"jmp {op}\n")
                }
                fn cmp0(&mut self, op: &(dyn Arg + '_)) -> Result<(),Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"cmp {op}, 0\n")
                }
                fn cmovz64(&mut self, op: &(dyn Arg + '_),val:u64) -> Result<(), Self::Error>{
                     let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"cmovz {op}, {val}\n")
                }
                fn jz(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"jz {op}\n")
                }
                fn u32(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"and {op}, 0xffffffff\n")
                }
                fn lea(&mut self, dest: &(dyn Arg + '_), src: &(dyn Arg + '_), offset: isize, off_reg: Option<(&(dyn Arg + '_),usize)>) -> Result<(),Self::Error>{
                    let dest = dest.reg().display(RegFormatOpts::default());
                    let src = src.reg().display(RegFormatOpts::default());
                    write!(self,"lea {dest}, [{src}")?;
                    if let Some((r,m)) = off_reg{
                        let r = r.reg().display(RegFormatOpts::default());
                        write!(self,"+{r}*{m}")?;
                    }
                    write!(self,"+{offset}]\n")
                }
                fn mov(&mut self, dest: &(dyn Arg + '_), src: &(dyn Arg + '_), mem: Option<isize>) -> Result<(), Self::Error>{
                     let dest = dest.reg().display(RegFormatOpts::default());
                    let src = src.reg().display(RegFormatOpts::default());
                    write!(self,"mov {dest}, ")?;
                    match mem{
                        None => write!(self,"{src}\n"),
                        Some(i) => write!(self,"qword ptr [{src}+{i}]\n")
                    }
                }
                fn lea_label(&mut self, dest: &(dyn Arg + '_), label: &(dyn Label + '_)) -> Result<(),Self::Error>{
                    let dest = dest.reg().display(RegFormatOpts::default());
                    write!(self,"lea {dest}, {label}\n")
                }
                fn get_ip(&mut self) -> Result<(),Self::Error>{
                //   let dest = dest.reg().display(RegFormatOpts::default());
                    write!(self,"call 1f\n1:\n")
                }
                fn ret(&mut self) -> Result<(), Self::Error>{
                    write!(self,"ret\n")
                }
                fn mov64(&mut self, r: &(dyn Arg + '_), val: u64) -> Result<(),Self::Error>{
                    let r = r.reg().display(RegFormatOpts::default());
                    write!(self,"mov {r}, {val}\n")
                }
                fn not(&mut self, op: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let op = op.reg().display(RegFormatOpts::default());
                    write!(self,"not {op}\n")
                }
                fn mul(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"mul {a},{b}\n")
                }
                fn div(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"div {a},{b}\n")
                }
                fn idiv(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"idiv {a},{b}\n")
                }
                fn and(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"and {a},{b}\n")
                }
                fn or(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"or {a},{b}\n")
                }
                fn eor(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"eor {a},{b}\n")
                }
                fn shl(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"shl {a},{b}\n")
                }
                fn shr(&mut self, a: &(dyn Arg + '_), b: &(dyn Arg + '_)) -> Result<(), Self::Error>{
                    let a = a.reg().display(RegFormatOpts::default());
                    let b = b.reg().display(RegFormatOpts::default());
                    write!(self,"shr {a},{b}\n")
                }
            })*
        };
    };
}
writers!(Formatter<'_>, (dyn Write + '_));
