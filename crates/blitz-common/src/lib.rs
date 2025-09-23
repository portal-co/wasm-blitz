#![no_std]
extern crate alloc;
pub use wasmparser;
use wasmparser::{BinaryReaderError, FunctionBody, Operator, ValType};
pub fn mach_operators<'a>(code: &[FunctionBody<'a>]) -> impl Iterator<Item = MachOperator<'a>> {
    return code
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            let v = a.get_operators_reader()?;
            let l = a.get_locals_reader()?;
            Ok::<_, BinaryReaderError>(
                [MachOperator::StartFn(i as u32)]
                    .into_iter()
                    .map(Ok)
                    .chain(
                        l.into_iter()
                            .map(|a| a.map(|(a, b)| MachOperator::Local(a, b))),
                    )
                    .chain([MachOperator::StartBody].map(Ok))
                    .chain(v.into_iter().map(|v| v.map(MachOperator::Operator)))
                    .chain([MachOperator::EndBody].map(Ok)),
            )
        })
        .flatten()
        .flatten();
}
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MachOperator<'a> {
    Operator(Operator<'a>),
    Local(u32, ValType),
    StartFn(u32),
    StartBody,
    EndBody,
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
