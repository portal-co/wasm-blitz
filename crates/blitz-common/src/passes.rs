use alloc::vec::Vec;

use crate::*;
pub fn load_coalescing<'a>(
    a: impl Iterator<Item = MachOperator<'a>>,
) -> impl Iterator<Item = MachOperator<'a>> {
    return a
        .scan_mach(
            |d, l, o, x| match o {
                MachOperator::StartBody => [
                    MachOperator::Local(2, ValType::I64),
                    MachOperator::StartBody,
                ]
                .into_iter()
                .collect::<Vec<_>>(),
                MachOperator::Operator(o) => match o {
                    Operator::I64Load8U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I64Load16U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xffff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I64Load32U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xffff_ffff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Load8U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I32WrapI64,
                        Operator::I32Const { value: 0xff },
                        Operator::I32Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Load16U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I32WrapI64,
                        Operator::I32Const { value: 0xffff },
                        Operator::I32Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Load { memarg } => {
                        [Operator::I64Load { memarg }, Operator::I32WrapI64]
                            .into_iter()
                            .map(|v| MachOperator::Operator(v))
                            .collect()
                    }
                    Operator::I64Store8 { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::I64Const { value: !0xff },
                        Operator::I64And,
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I64Store16 { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::I64Const { value: !0xffff },
                        Operator::I64And,
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I64Store32 { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::I64Const {
                            value: !0xffff_ffff,
                        },
                        Operator::I64And,
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Store8 { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::I64ExtendI32U,
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::I64Const { value: !0xff },
                        Operator::I64And,
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Store16 { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::I64ExtendI32U,
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::I64Const { value: !0xffff },
                        Operator::I64And,
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    Operator::I32Store { memarg } => [
                        Operator::LocalSet { local_index: l },
                        Operator::I64ExtendI32U,
                        Operator::LocalTee { local_index: l + 1 },
                        Operator::LocalGet { local_index: l + 1 },
                        Operator::I64Load {
                            memarg: memarg.clone(),
                        },
                        Operator::LocalGet { local_index: l },
                        Operator::I64Or,
                        Operator::I64Store { memarg },
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator(v))
                    .collect(),
                    o => [MachOperator::Operator(o)].into_iter().collect::<Vec<_>>(),
                },
                o => [o].into_iter().collect::<Vec<_>>(),
            },
            (),
        )
        .flatten();
}
