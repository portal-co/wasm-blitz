#![no_std]
#[doc(hidden)]
pub extern crate alloc;
#[doc(hidden)]
pub mod __ {
    pub use core;
}
use alloc::vec::Vec;
use core::{
    fmt::{Display, Formatter},
    mem::{transmute, transmute_copy},
    str::MatchIndices,
};
pub use wasmparser;
pub use wasm_encoder;
use wasmparser::{BinaryReaderError, FuncType, FunctionBody, Operator, ValType};
pub mod dce;
pub trait Label<X: Clone + 'static>: Display {
    fn raw(&self) -> Option<X> {
        if typeid::of::<Self>() == typeid::of::<X>() {
            let this: &X = unsafe { transmute_copy(&self) };
            Some(this.clone())
        } else {
            None
        }
    }
}
impl<T: Display + ?Sized, X: Clone + 'static> Label<X> for T {}
#[derive(Clone, Copy)]
pub struct DisplayFn<'a>(pub &'a (dyn Fn(&mut Formatter) -> core::fmt::Result + 'a));
impl<'a> Display for DisplayFn<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        (self.0)(f)
    }
}
#[cfg(feature = "asm")]
pub mod asm;
pub mod ops;
pub mod passes;
