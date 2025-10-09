#![no_std]
#[doc(hidden)]
pub extern crate alloc;
#[doc(hidden)]
pub mod __{
    pub use core;
}
use core::{
    fmt::{Display, Formatter},
    mem::{transmute, transmute_copy},
    str::MatchIndices,
};

use alloc::vec::Vec;
pub use wasmparser;
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
pub fn mach_operators<'a>(
    code: &[FunctionBody<'a>],
    sigs_per: &[u32],
    sigs: &[FuncType],
) -> impl Iterator<Item = MachOperator<'a>> {
    return code
        .iter()
        .zip(sigs_per.iter().cloned().map(|a| &sigs[a as usize]))
        .enumerate()
        .flat_map(|(i, (a, sig))| {
            let v = a.get_operators_reader()?;
            let l = a.get_locals_reader()?;
            Ok::<_, BinaryReaderError>(
                [MachOperator::StartFn {
                    id: i as u32,
                    data: FnData {
                        num_params: sig.params().len(),
                        num_returns: sig.results().len(),
                        control_depth: control_depth(a),
                    },
                }]
                .into_iter()
                .map(Ok)
                .chain(
                    l.into_iter()
                        .map(|a| a.map(|(a, b)| MachOperator::Local { count: a, ty: b })),
                )
                .chain([MachOperator::StartBody].map(Ok))
                .chain(
                    v.into_iter()
                        .map(|v| v.map(|op| MachOperator::Operator { op: Some(op), annot: () })),
                )
                .chain(
                    [
                        MachOperator::Operator {
                            op: Some(Operator::Return),
                            annot: (),
                        },
                        MachOperator::EndBody,
                    ]
                    .map(Ok),
                ),
            )
        })
        .flatten()
        .flatten();
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FnData {
    pub num_params: usize,
    pub num_returns: usize,
    pub control_depth: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MachOperator<'a, Annot = ()> {
    Operator { op: Option<Operator<'a>>, annot: Annot },
    Local { count: u32, ty: ValType },
    StartFn { id: u32, data: FnData },
    StartBody,
    EndBody,
}
impl<'a, Annot> MachOperator<'a, Annot> {
    pub fn map<Annot2, E>(
        self,
        f: &mut (dyn FnMut(Annot) -> Result<Annot2, E> + '_),
    ) -> Result<MachOperator<'a, Annot2>, E> {
        Ok(match self {
            MachOperator::Operator { op, annot } => MachOperator::Operator {
                op,
                annot: f(annot)?,
            },
            MachOperator::Local { count, ty } => MachOperator::Local { count, ty },
            MachOperator::StartFn { id, data } => MachOperator::StartFn { id, data },
            MachOperator::StartBody => MachOperator::StartBody,
            MachOperator::EndBody => MachOperator::EndBody,
        })
    }
    pub fn as_ref<'b>(&'b self) -> MachOperator<'a, &'b Annot> {
        match self {
            MachOperator::Operator { op, annot } => MachOperator::Operator {
                op: op.clone(),
                annot,
            },
            MachOperator::Local { count, ty } => MachOperator::Local {
                count: *count,
                ty: *ty,
            },
            MachOperator::StartFn { id, data } => MachOperator::StartFn {
                id: *id,
                data: data.clone(),
            },
            MachOperator::StartBody => MachOperator::StartBody,
            MachOperator::EndBody => MachOperator::EndBody,
        }
    }
    pub fn as_mut<'b>(&'b mut self) -> MachOperator<'a, &'b mut Annot> {
        match self {
            MachOperator::Operator { op, annot } => MachOperator::Operator {
                op: op.clone(),
                annot,
            },
            MachOperator::Local { count, ty } => MachOperator::Local {
                count: *count,
                ty: *ty,
            },
            MachOperator::StartFn { id, data } => MachOperator::StartFn {
                id: *id,
                data: data.clone(),
            },
            MachOperator::StartBody => MachOperator::StartBody,
            MachOperator::EndBody => MachOperator::EndBody,
        }
    }
}
pub fn control_depth(a: &FunctionBody<'_>) -> usize {
    let mut cur: usize = 0;
    let mut max: usize = 0;
    for op in a.get_operators_reader().into_iter().flatten().flatten() {
        match op {
            Operator::Block { .. } | Operator::Loop { .. } | Operator::If { .. } => {
                cur += 1;
                max = max.max(cur);
            }
            Operator::End => {
                cur -= 1;
            }
            _ => {}
        }
    }
    return max;
}
#[derive(Clone)]
pub struct ScanMach<T, F, D> {
    wrapped: T,
    handler: F,
    userdata: D,
    data: FnData,
    locals: u32,
}
impl<
    'a,
    A,
    I: Iterator<Item = MachOperator<'a, A>>,
    T,
    F: FnMut(&mut FnData, u32, MachOperator<'a, A>, &mut D) -> T,
    D,
> Iterator for ScanMach<I, F, D>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let o = self.wrapped.next()?;
        if let MachOperator::StartFn { id, data } = &o {
            self.data = data.clone();
            self.locals = 0;
            return Some((self.handler)(
                &mut self.data,
                self.locals,
                o,
                &mut self.userdata,
            ));
        }
        if let MachOperator::Local { count: a, ty: b } = &o {
            self.locals += *a;
        }
        let mut tmp = self.data.clone();
        return Some((self.handler)(&mut tmp, self.locals, o, &mut self.userdata));
    }
}
pub trait IteratorExt: Iterator {
    fn scan_mach<'a, F: FnMut(&mut FnData, u32, MachOperator<'a>, &mut D) -> T, T, D>(
        self,
        handler: F,
        userdata: D,
    ) -> ScanMach<Self, F, D>
    where
        Self: Sized,
    {
        ScanMach {
            wrapped: self,
            handler,
            userdata,
            data: Default::default(),
            locals: 0,
        }
    }
}
impl<T: Iterator + ?Sized> IteratorExt for T {}
pub mod passes;
