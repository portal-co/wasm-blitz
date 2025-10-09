use alloc::vec::Vec;

use crate::*;
#[macro_export]
macro_rules! dce_pass {
    ($a:expr) => {
        match match $a {
            a => $crate::IteratorExt::scan_mach(
                a,
                |_, _, o, dce_stack| match o {
                    $crate::MachOperator::Operator { op, annot }
                        if $crate::dce::dce(dce_stack,$crate::core::option::Option::as_ref(op)?) =>
                    {
                        $crate::core::option::Option::None
                    }

                    o => {
                        if let $crate::MachOperator::EndBody = &o {
                            *dce_stack = $crate::dce::DceStack::new();
                        }
                        $crate::core::option::Option::Some(o)
                    }
                },
                $crate::dce::DceStack::new(),
            ),
        } {
            a => $crate::__::core::iter::Iterator::flatten(a),
        }
    };
}
pub fn load_coalescing<'a>(
    a: impl Iterator<Item = MachOperator<'a>>,
) -> impl Iterator<Item = MachOperator<'a>> {
    return a
        .scan_mach(
            |d, l, o, x| match o {
                MachOperator::StartBody => [
                    MachOperator::Local {
                        count: 2,
                        ty: ValType::I64,
                    },
                    MachOperator::StartBody,
                ]
                .into_iter()
                .collect::<Vec<_>>(),
                MachOperator::Operator { op: Some(o), annot } => match o {
                    Operator::I64Load8U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    Operator::I64Load16U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xffff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    Operator::I64Load32U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I64Const { value: 0xffff_ffff },
                        Operator::I64Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    Operator::I32Load8U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I32WrapI64,
                        Operator::I32Const { value: 0xff },
                        Operator::I32Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    Operator::I32Load16U { memarg } => [
                        Operator::I64Load { memarg },
                        Operator::I32WrapI64,
                        Operator::I32Const { value: 0xffff },
                        Operator::I32Add,
                    ]
                    .into_iter()
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    Operator::I32Load { memarg } => {
                        [Operator::I64Load { memarg }, Operator::I32WrapI64]
                            .into_iter()
                            .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
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
                    .map(|v| MachOperator::Operator { op: Some(v), annot })
                    .collect(),
                    o => [MachOperator::Operator { op: Some(o), annot }]
                        .into_iter()
                        .collect::<Vec<_>>(),
                },
                o => [o].into_iter().collect::<Vec<_>>(),
            },
            (),
        )
        .flatten();
}
