use alloc::boxed::Box;
use wasm_encoder::Instruction;

use crate::*;
#[macro_export]
macro_rules! simple_op_match {
    ($a:expr => {$($i:ident $({$a2:ident => |$extra:pat_param|$ev:expr})? => |$annot:pat_param|$e:expr),*} |$b:pat_param| $c:expr) => {
        match $a{
            $(
                $crate::MachOperator::Instruction{annot: $annot, op: $crate::wasm_encoder::Instruction::$i$(($a2))?} => $e,
                $crate::MachOperator::Operator{annot: $annot, op: $crate::__::core::option::Option::Some($crate::wasmparser::Operator::$i $({$a2})?)} => match match ($($a2)?){
                    ($($extra)?) => ($($ev)?)
                }{
                    ($($a2)?) => $e
                }
            ),*,
            $b => $c
        }
    };
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub struct WasmInfo {
    pub offset: usize,
}
pub trait FromWasmInfo {
    fn from_wasm_info(info: WasmInfo) -> Self;
}
impl FromWasmInfo for () {
    fn from_wasm_info(value: WasmInfo) -> Self {
        ()
    }
}
impl<T: FromWasmInfo> FromWasmInfo for Option<T> {
    fn from_wasm_info(value: WasmInfo) -> Self {
        Some(T::from_wasm_info(value))
    }
}
impl FromWasmInfo for WasmInfo {
    fn from_wasm_info(info: WasmInfo) -> Self {
        info
    }
}
#[derive(Clone, Debug)]
pub enum InstructionOrOperator<'a> {
    Instruction(Instruction<'a>),
    Operator(Operator<'a>),
}
// pub trait FuncRewriter<E>:
//     for<'c, 'b> FnMut(
//     u32,
//     &'c [u32],
//     &'c [FuncType],
//     &'c Operator<'b>,
// ) -> Box<dyn Iterator<Item = Result<InstructionOrOperator<'b>, E>> + 'c>
// {
// }
// impl<
//     E,
//     T: for<'c, 'b> FnMut(
//             u32,
//             &'c [u32],
//             &'c [FuncType],
//             &'c Operator<'b>,
//         )
//             -> Box<dyn Iterator<Item = Result<InstructionOrOperator<'b>, E>> + 'c>
//         + ?Sized,
// > FuncRewriter<E> for T
// {
// }
pub fn mach_operators<'a, 'b, Annot: FromWasmInfo, E: From<BinaryReaderError>>(
    code: &[FunctionBody<'a>],
    sigs_per: &[u32],
    sigs: &[FuncType],
    imports: u32,
    // mut func_rewriter: Option<&'b mut (dyn FuncRewriter<E> + '_)>,
) -> impl Iterator<Item = Result<MachOperator<'a, Annot>, E>> {
    return code
        .iter()
        .zip(
            sigs_per
                .iter()
                .cloned()
                .skip(imports as usize)
                .map(|a| &sigs[a as usize]),
        )
        .enumerate()
        .flat_map(move |(i, (a, sig))| {
            // let mut func_rewriter = match &mut func_rewriter {
            //     None => None,
            //     Some(a) => Some(match &mut **a {
            //         b => unsafe { transmute::<_, &'b mut (dyn FuncRewriter<E> + '_)>(b) },
            //     }),
            // };
            let v = a.get_operators_reader()?;
            let l = a.get_locals_reader()?;
            Ok::<_, E>(
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
                .chain(l.into_iter().map(|a| {
                    a.map(|(a, b)| MachOperator::Local { count: a, ty: b })
                        .map_err(E::from)
                }))
                .chain([MachOperator::StartBody].map(Ok))
                .chain(v.into_iter_with_offsets().flat_map(
                    move |v: Result<(Operator<'_>, usize), BinaryReaderError>| {
                        [v.map(|(op, offset)| MachOperator::Operator {
                            op: Some(op),
                            annot: Annot::from_wasm_info(WasmInfo { offset }),
                        })
                        .map_err(E::from)]
                        .into_iter()
                        .collect::<Vec<_>>()
                    },
                ))
                .chain(
                    [
                        MachOperator::Operator {
                            op: Some(Operator::Return),
                            annot: Annot::from_wasm_info(WasmInfo {
                                offset: a.range().end,
                            }),
                        },
                        MachOperator::EndBody,
                    ]
                    .map(Ok),
                ),
            )
        })
        .flatten();
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FnData {
    pub num_params: usize,
    pub num_returns: usize,
    pub control_depth: usize,
}
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MachOperator<'a, Annot = ()> {
    Operator {
        op: Option<Operator<'a>>,
        annot: Annot,
    },
    Instruction {
        op: Instruction<'a>,
        annot: Annot,
    },
    Trap {
        conditional: bool,
        annot: Annot,
    },
    Local {
        count: u32,
        ty: ValType,
    },
    StartFn {
        id: u32,
        data: FnData,
    },
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
            MachOperator::Instruction { op, annot } => MachOperator::Instruction {
                op,
                annot: f(annot)?,
            },
            MachOperator::Trap { conditional, annot } => MachOperator::Trap {
                conditional,
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
            MachOperator::Instruction { op, annot } => MachOperator::Instruction {
                op: op.clone(),
                annot,
            },
            MachOperator::Trap { conditional, annot } => MachOperator::Trap {
                conditional: *conditional,
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
            MachOperator::Instruction { op, annot } => MachOperator::Instruction {
                op: op.clone(),
                annot,
            },
            MachOperator::Trap { conditional, annot } => MachOperator::Trap {
                conditional: *conditional,
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
// impl<
//     'a,
//     A,
//     I: Iterator<Item = MachOperator<'a, A>>,
//     T,
//     F: FnMut(&mut FnData, u32, MachOperator<'a, A>, &mut D) -> T,
//     D,
// > Iterator for ScanMach<I, F, D>
// {
//     type Item = T;
//     fn next(&mut self) -> Option<Self::Item> {
//         let o = self.wrapped.next()?;
//         if let MachOperator::StartFn { id, data } = &o {
//             self.data = data.clone();
//             self.locals = 0;
//             return Some((self.handler)(
//                 &mut self.data,
//                 self.locals,
//                 o,
//                 &mut self.userdata,
//             ));
//         }
//         if let MachOperator::Local { count: a, ty: b } = &o {
//             self.locals += *a;
//         }
//         let mut tmp = self.data.clone();
//         return Some((self.handler)(&mut tmp, self.locals, o, &mut self.userdata));
//     }
// }
impl<
    'a,
    A,
    E,
    I: Iterator<Item = Result<MachOperator<'a, A>, E>>,
    T,
    F: FnMut(&mut FnData, u32, MachOperator<'a, A>, &mut D) -> T,
    D,
> Iterator for ScanMach<I, F, D>
{
    type Item = Result<T, E>;
    fn next(&mut self) -> Option<Self::Item> {
        let o = self.wrapped.next()?;
        match o {
            Ok(o) => {
                if let MachOperator::StartFn { id, data } = &o {
                    self.data = data.clone();
                    self.locals = 0;
                    return Some(Ok((self.handler)(
                        &mut self.data,
                        self.locals,
                        o,
                        &mut self.userdata,
                    )));
                }
                if let MachOperator::Local { count: a, ty: b } = &o {
                    self.locals += *a;
                }
                let mut tmp = self.data.clone();
                return Some(Ok((self.handler)(
                    &mut tmp,
                    self.locals,
                    o,
                    &mut self.userdata,
                )));
            }
            Err(e) => return Some(Err(e)),
        }
    }
}
pub trait IteratorExt: Iterator {
    fn scan_mach<'a, A, F: FnMut(&mut FnData, u32, MachOperator<'a, A>, &mut D) -> T, T, D>(
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
