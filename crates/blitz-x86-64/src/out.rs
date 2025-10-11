use crate::*;
pub trait Writer {
    type Error: Error;
    fn set_label(&mut self, s: &(dyn Label + '_)) -> Result<(), Self::Error>;
    fn xchg(&mut self, dest: Reg, src: Reg, mem: Option<isize>) -> Result<(), Self::Error>;
    fn mov(&mut self, dest: Reg, src: Reg, mem: Option<isize>) -> Result<(), Self::Error>;
    fn push(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn pop(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn call(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn jmp(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn cmp0(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn cmovz64(&mut self, op: Reg, val: u64) -> Result<(), Self::Error>;
    fn jz(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn u32(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn not(&mut self, op: Reg) -> Result<(), Self::Error>;
    fn lea(
        &mut self,
        dest: Reg,
        src: Reg,
        offset: isize,
        off_reg: Option<(Reg, usize)>,
    ) -> Result<(), Self::Error>;
    fn lea_label(&mut self, dest: Reg, label: &(dyn Label + '_)) -> Result<(), Self::Error>;
    fn get_ip(&mut self) -> Result<(), Self::Error>;
    fn ret(&mut self) -> Result<(), Self::Error>;
    fn mov64(&mut self, r: Reg, val: u64) -> Result<(), Self::Error>;
    fn mul(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn div(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn idiv(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn and(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn or(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn eor(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn shl(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
    fn shr(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>;
}

macro_rules! writer_dispatch {
    ($( [ $($t:tt)* ] $ty:ty => $e:ty),*) => {
        const _: () = {
            $(impl<$($t)*> Writer for $ty{
                type Error = $e;

                fn set_label(&mut self, s: &(dyn Label + '_)) -> Result<(), Self::Error> {
                    Writer::set_label(&mut **self, s)
                }

                fn xchg(&mut self, dest: Reg, src: Reg, mem: Option<isize>) -> Result<(), Self::Error> {
                    Writer::xchg(&mut **self, dest, src, mem)
                }

                fn push(&mut self, op: Reg) -> Result<(), Self::Error> {
                    Writer::push(&mut **self, op)
                }

                fn pop(&mut self, op: Reg) -> Result<(), Self::Error> {
                    Writer::pop(&mut **self, op)
                }
                fn call(&mut self, op: Reg) -> Result<(), Self::Error>{
                    Writer::call(&mut **self,op)
                }
                fn jmp(&mut self, op: Reg) -> Result<(), Self::Error>{
                    Writer::jmp(&mut **self,op)
                }
                fn cmp0(&mut self, op: Reg) -> Result<(),Self::Error>{
                    Writer::cmp0(&mut **self,op)
                }
                fn cmovz64(&mut self, op: Reg,val:u64) -> Result<(), Self::Error>{
                    Writer::cmovz64(&mut **self,op,val)
                }
                fn jz(&mut self, op: Reg) -> Result<(), Self::Error>{
                    Writer::jz(&mut **self,op)
                }

                fn lea(
                    &mut self,
                    dest: Reg,
                    src: Reg,
                    offset: isize,
                    off_reg: Option<(Reg, usize)>,
                ) -> Result<(), Self::Error> {
                    Writer::lea(&mut **self, dest, src, offset, off_reg)
                }

                fn lea_label(&mut self, dest: Reg, label: &(dyn Label + '_)) -> Result<(), Self::Error> {
                    Writer::lea_label(&mut **self, dest, label)
                }
                fn get_ip(&mut self) -> Result<(), Self::Error>{
                    Writer::get_ip(&mut **self)
                }
                fn ret(&mut self) -> Result<(), Self::Error>{
                    Writer::ret(&mut **self)
                }
                fn mov64(&mut self, r: Reg, val: u64) -> Result<(),Self::Error>{
                    Writer::mov64(&mut **self,r,val)
                }
                fn mov(&mut self, dest: Reg, src: Reg, mem: Option<isize>) -> Result<(), Self::Error>{
                    Writer::mov(&mut **self,dest,src,mem)
                }
                fn u32(&mut self, op: Reg) -> Result<(), Self::Error>{
                    Writer::u32(&mut **self,op)
                }
                fn not(&mut self, op: Reg) -> Result<(), Self::Error>{
                    Writer::not(&mut **self,op)
                }
                fn mul(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::mul(&mut **self,a,b)
                }
                fn div(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::div(&mut **self,a,b)
                }
                fn idiv(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::idiv(&mut **self,a,b)
                }
                fn and(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::and(&mut **self,a,b)
                }
                fn or(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::or(&mut **self,a,b)
                }
                fn eor(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::eor(&mut **self,a,b)
                }
                fn shl(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::shl(&mut **self,a,b)
                }
                fn shr(&mut self, a: Reg, b: Reg) -> Result<(), Self::Error>{
                    Writer::shr(&mut **self,a,b)
                }
            })*
        };
    };
}
writer_dispatch!([ T: Writer + ?Sized ] &'_ mut T => T::Error);
