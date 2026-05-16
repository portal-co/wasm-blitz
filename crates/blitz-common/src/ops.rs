//! Machine operator definitions and utilities.
//!
//! This module provides the core intermediate representation for WASM operations
//! during compilation. The `MachOperator` type represents individual operations
//! that can be either WASM operators or encoded instructions.

use alloc::boxed::Box;
use wasm_encoder::{Instruction, reencode};

use crate::*;

/// Combined error type for backend `handle_op` / `sysv_handle_op` methods.
///
/// Backends that emit assembly text have `Self::Error = fmt::Error` for writer
/// calls, while also needing to propagate re-encoder errors from
/// `Reencode::instruction`.  This enum unifies both so callers do not need to
/// constrain `wasm_encoder::reencode::Error<E>: Into<Self::Error>`.
pub enum HandleOpError<E> {
    /// An error from a writer method (e.g. `fmt::write!` failing).
    Fmt(core::fmt::Error),
    /// An error produced by the WASM re-encoder.
    Reencode(reencode::Error<E>),
}

impl<E> core::fmt::Debug for HandleOpError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HandleOpError::Fmt(e) => write!(f, "HandleOpError::Fmt({e:?})"),
            HandleOpError::Reencode(_) => f.write_str("HandleOpError::Reencode(..)"),
        }
    }
}

impl<E> From<core::fmt::Error> for HandleOpError<E> {
    fn from(e: core::fmt::Error) -> Self {
        HandleOpError::Fmt(e)
    }
}

impl<E> From<reencode::Error<E>> for HandleOpError<E> {
    fn from(e: reencode::Error<E>) -> Self {
        HandleOpError::Reencode(e)
    }
}

/// Macro for pattern matching on machine operators.
///
/// This macro simplifies matching on both `Instruction` and `Operator` variants
/// of `MachOperator`, handling the conversion between the two representations.
///
/// # Example
///
/// ```ignore
/// simple_op_match!(op => {
///     I32Add => |annot| handle_add(annot),
///     I32Sub => |annot| handle_sub(annot)
/// } |other| handle_default(other))
/// ```
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

/// Information about a WASM instruction's location in the binary.
///
/// This struct carries metadata about where an instruction appears in the
/// original WASM bytecode, useful for debugging and error reporting.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub struct WasmInfo {
    /// Byte offset in the WASM binary where this instruction appears.
    pub offset: usize,
}

/// Trait for types that can be constructed from WASM metadata.
///
/// This trait allows various annotation types to be created from
/// `WasmInfo` during the compilation process.
pub trait FromWasmInfo {
    /// Creates an instance from WASM metadata.
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

/// Represents either an encoded instruction or a parsed operator.
///
/// This enum allows code to work with WASM operations in either their
/// encoded or decoded form.
#[derive(Clone, Debug)]
pub enum InstructionOrOperator<'a> {
    /// An encoded WASM instruction.
    Instruction(Instruction<'a>),
    /// A parsed WASM operator.
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

/// Converts WASM function bodies into a stream of machine operators.
///
/// This function parses WASM function bytecode and produces an iterator of
/// `MachOperator` items that represent each operation in a form suitable for
/// code generation backends.
///
/// # Arguments
///
/// * `code` - Array of WASM function bodies to process
/// * `sigs_per` - Function signature indices for each function
/// * `sigs` - Array of function type signatures
/// * `imports` - Number of imported functions (to offset function IDs)
///
/// # Returns
///
/// An iterator yielding `MachOperator` items or errors during parsing.
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
                        tracing: None,
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

/// Tracing and outer-JIT specialization hooks embedded into the generated function prologue.
///
/// Both pointers are baked as 64-bit immediates into the generated machine code at
/// compile time; the outer JIT reads/writes through them at runtime.  The caller must
/// keep the pointed-to storage alive for the entire lifetime of the generated code.
///
/// # Safety
/// Raw pointers — caller is responsible for alignment, validity, and any required
/// synchronisation with the outer JIT.
#[derive(Debug, Clone, Copy)]
pub struct TracingHooks {
    /// Pointer to this function's `u64` invocation counter.
    ///
    /// The generated prologue performs a non-atomic load → increment → store on
    /// every function entry.  Approximate counting is intentional; the outer JIT
    /// does not need exact values.
    pub counter: *mut u64,
    /// Pointer to this function's specialization fn-pointer slot.
    ///
    /// At runtime the generated prologue loads `*specialization`.  If the value is
    /// non-null it is treated as a function pointer and control is transferred there
    /// via a tail-jump (no frame has been set up yet, so the caller's register/stack
    /// state is fully intact for whichever ABI is in use).
    pub specialization: *const *const (),
}

// SAFETY: the outer JIT that owns and writes these slots is responsible for
// any required memory ordering; blitz only embeds the addresses.
unsafe impl Send for TracingHooks {}
unsafe impl Sync for TracingHooks {}

/// Metadata about a WASM function.
///
/// Contains information needed by code generators about the function's
/// signature and structure.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct FnData {
    /// Number of parameters the function accepts.
    pub num_params: usize,
    /// Number of return values the function produces.
    pub num_returns: usize,
    /// Maximum nesting depth of control flow structures in the function.
    pub control_depth: usize,
    /// Optional tracing/specialization hooks.  `None` → no tracing code emitted
    /// (zero overhead on the fast path).
    pub tracing: Option<TracingHooks>,
}

impl PartialEq for FnData {
    fn eq(&self, other: &Self) -> bool {
        self.num_params == other.num_params
            && self.num_returns == other.num_returns
            && self.control_depth == other.control_depth
    }
}
impl Eq for FnData {}

/// A machine-level operation in the compilation pipeline.
///
/// `MachOperator` represents individual operations during compilation, with an
/// optional annotation for carrying metadata. This is the primary IR type
/// used throughout wasm-blitz.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MachOperator<'a, Annot = ()> {
    /// A WASM operator (decoded form).
    Operator {
        op: Option<Operator<'a>>,
        annot: Annot,
    },
    /// An encoded WASM instruction.
    Instruction { op: Instruction<'a>, annot: Annot },
    /// A trap instruction (unconditional or conditional).
    Trap { conditional: bool, annot: Annot },
    /// Declaration of local variables.
    Local { count: u32, ty: ValType },
    /// Start of a function definition.
    StartFn { id: u32, data: FnData },
    /// Start of function body (after locals).
    StartBody,
    /// End of function body.
    EndBody,
}

impl<'a, Annot> MachOperator<'a, Annot> {
    /// Maps the annotation type to a different type.
    ///
    /// Transforms the annotation while preserving the operation itself.
    /// Useful for threading different types of metadata through the pipeline.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to transform the annotation
    ///
    /// # Returns
    ///
    /// A new `MachOperator` with transformed annotation, or an error.
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

    /// Creates a reference to this operator with a borrowed annotation.
    ///
    /// Useful for temporarily working with the annotation without taking ownership.
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

    /// Creates a mutable reference to this operator with a mutably borrowed annotation.
    ///
    /// Allows mutation of the annotation without taking ownership of the operator.
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

/// Calculates the maximum control flow nesting depth in a function.
///
/// Scans through the function's operators to determine the deepest nesting
/// of blocks, loops, and if statements. This information is used by code
/// generators to allocate appropriate stack space.
///
/// # Arguments
///
/// * `a` - The function body to analyze
///
/// # Returns
///
/// The maximum nesting depth of control flow structures.
pub fn control_depth(a: &FunctionBody<'_>) -> usize {
    let mut cur: usize = 0;
    let mut max: usize = 0;
    for op in a.get_operators_reader().into_iter().flatten().flatten() {
        match op {
            Operator::Block { .. }
            | Operator::Loop { .. }
            | Operator::If { .. }
            | Operator::TryTable { .. } => {
                cur += 1;
                max = max.max(cur);
            }
            Operator::End => {
                cur = cur.saturating_sub(1);
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
