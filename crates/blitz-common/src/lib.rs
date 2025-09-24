#![no_std]
extern crate alloc;
pub use wasmparser;
use wasmparser::{BinaryReaderError, FuncType, FunctionBody, Operator, ValType};
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
                    num_params: sig.params().len(),
                    num_returns: sig.results().len(),
                    control_depth: control_depth(a)
                }]
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
    StartFn {
        id: u32,
        num_params: usize,
        num_returns: usize,
        control_depth: usize,
    },
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
