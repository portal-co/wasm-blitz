//! Machine operator definitions and utilities.
//!
//! This module provides the core intermediate representation for WASM operations
//! during compilation. The `MachOperator` type represents individual operations
//! that can be either WASM operators or encoded instructions.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use wasm_encoder::{reencode, Instruction};

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

impl<E> From<core::convert::Infallible> for HandleOpError<E> {
    fn from(x: core::convert::Infallible) -> Self {
        match x {}
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
    code.iter()
        .zip(
            sigs_per
                .iter()
                .cloned()
                .skip(imports as usize)
                .map(|a| &sigs[a as usize]),
        )
        .enumerate()
        .flat_map(move |(i, (body, sig))| {
            mach_operators_for_function::<Annot, E>(body, sig, i as u32)
        })
}

/// Convert exactly one WASM function body into machine operators.
///
/// This is the demand-driven entry point for backends: callers select a
/// requested/reachable function and invoke this function for that body only.
/// [`mach_operators`] remains the compatibility materializer for callers that
/// intentionally compile every supplied body.
pub fn mach_operators_for_function<'a, Annot: FromWasmInfo, E: From<BinaryReaderError>>(
    body: &FunctionBody<'a>,
    sig: &FuncType,
    function_id: u32,
) -> impl Iterator<Item = Result<MachOperator<'a, Annot>, E>> {
    (|| -> Result<_, E> {
        let operators = body.get_operators_reader()?;
        let locals = body.get_locals_reader()?;
        Ok::<_, E>(
            [MachOperator::StartFn {
                id: function_id,
                data: FnData {
                    num_params: sig.params().len(),
                    num_returns: sig.results().len(),
                    control_depth: control_depth(body),
                    probes: None,
                    probe_plan: None,
                },
            }]
            .into_iter()
            .map(Ok)
            .chain(locals.into_iter().map(|local| {
                local
                    .map(|(count, ty)| MachOperator::Local { count, ty })
                    .map_err(E::from)
            }))
            .chain([MachOperator::StartBody].map(Ok))
            .chain(operators.into_iter_with_offsets().map(
                |operator: Result<(Operator<'_>, usize), BinaryReaderError>| {
                    operator
                        .map(|(op, offset)| MachOperator::Operator {
                            op: Some(op),
                            annot: Annot::from_wasm_info(WasmInfo { offset }),
                        })
                        .map_err(E::from)
                },
            ))
            .chain(
                [
                    MachOperator::Operator {
                        op: Some(Operator::Return),
                        annot: Annot::from_wasm_info(WasmInfo {
                            offset: body.range().end,
                        }),
                    },
                    MachOperator::EndBody,
                ]
                .map(Ok),
            ),
        )
    })()
    .into_iter()
    .flatten()
}

/// One entry in the runtime-owned *probe table*.
///
/// The runtime allocates a contiguous `[ProbeSlot]` per instrumented function
/// (sized by [`ProbeTableConfig::num_probes`]) and is responsible for zeroing it
/// and for any memory ordering.  The generated code reaches the table through a
/// CTX-relative base pointer (see [`ProbeTableConfig::table_base_off`]) and
/// indexes it by a compile-time `probe_id` — **no absolute address is baked into
/// the code**, so the compilation and runtime environments can be fully
/// separated.
///
/// Layout is `#[repr(C)]` so the generated code and the runtime agree on field
/// offsets: `counter` at `+0`, `handler` at `+8`.
#[derive(Debug, Default)]
#[repr(C)]
pub struct ProbeSlot {
    /// Approximate invocation counter for this probe.
    ///
    /// The generated preamble performs a non-atomic load → increment → store on
    /// every entry.  Approximate counting is intentional; the outer JIT does not
    /// need exact values.
    pub counter: u64,
    /// Handler code pointer for this probe.
    ///
    /// At runtime the generated preamble loads this field.  If non-null it is
    /// dispatched to per the probe's compile-time binding mode: either a
    /// tail-jump (control never returns to the probe site — used for
    /// specialization opt-entry, with the operand-stack / CTX-frame state intact
    /// for this site, see the blitz-specialize stack-state contract) or a real
    /// call-and-return (control resumes at the probe site once the handler
    /// returns).
    pub handler: *const (),
}

/// Compile-time configuration for emitting probe sites in a function, without
/// baking any runtime addresses.
///
/// The generated code references the runtime [`ProbeSlot`] table indirectly: it
/// loads the table base from a fixed, CTX-relative slot ([`Self::table_base_off`])
/// and indexes by a per-probe id.  This keeps compilation independent of the
/// runtime address space.
#[derive(Debug, Clone, Copy)]
pub struct ProbeTableConfig {
    /// Whether probe code should be emitted at all.  `false` → zero overhead,
    /// identical codegen to `probes: None`.
    pub enabled: bool,
    /// Number of probes in this function (function entry = probe 0, then one per
    /// loop/block, for the auto-identified control-flow probe set).  The runtime
    /// uses this to size the [`ProbeSlot`] table.  Computed by
    /// [`probe_site_count`].
    pub num_probes: u32,
    /// Fixed, CTX-relative offset at which the runtime stores the base pointer of
    /// this function's [`ProbeSlot`] table.  Loaded by the preamble; never an
    /// absolute address.
    pub table_base_off: i32,
}

/// Where, relative to the instruction it's attached to, a probe fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbePlacement {
    /// Fires before the instruction's own codegen runs.
    Before,
    /// Fires after the instruction's own codegen runs.
    After,
}

/// How a probe's live state is preserved across a `Call`-bound dispatch.
/// `TailTakeover` always needs the canonical materialized layout (Active),
/// since the tail target inherits the generic site's stack-state contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeMode {
    /// Force the canonical layout first (e.g. flush a backend's register
    /// allocator, if it has one). The only valid mode for `TailTakeover`.
    Active,
    /// Save/restore only the registers a backend's allocator currently has
    /// live, without forcing a flush — cheaper, and what makes a `Call`
    /// probe genuinely non-disturbing when dropped into the middle of an
    /// expression. Identical to `Active` on backends with no allocator
    /// (x86-64, AArch64).
    Passive,
}

/// One embedder-requested probe: which runtime [`ProbeSlot`] it uses
/// (`probe_id`), how it's dispatched, how live state is preserved across it,
/// and where it fires relative to its instrumented instruction.
#[derive(Debug, Clone, Copy)]
pub struct ProbeSpec {
    pub probe_id: u32,
    pub binding: portal_solutions_blitz_codegen::ProbeBinding,
    pub mode: ProbeMode,
    pub placement: ProbePlacement,
}

/// An embedder's placement of probes within one function.
///
/// Probes are addressed by **ordinal instruction index** within the
/// function body (0 = the first real WASM operator after locals/`StartBody`)
/// rather than WASM bytecode offset: the streaming [`mach_operators`]
/// pipeline only threads a byte offset through when the embedder picks an
/// `Annot` that carries one ([`FromWasmInfo`]), while the ordinal index is
/// always available to a backend's dispatcher regardless of `Annot`. An
/// embedder that wants byte-offset addressing can build the index↔offset
/// mapping itself the same way [`mach_operators`] does
/// (`get_operators_reader().into_iter_with_offsets()`).
#[derive(Debug, Clone, Default)]
pub struct ProbePlan {
    /// Fires at the function-entry site, before any frame setup — the
    /// earliest possible point, structurally distinct from "before
    /// instruction 0" (which runs after the frame is built).
    pub entry: Vec<ProbeSpec>,
    /// Probes keyed by the ordinal index of the instruction they attach to.
    pub by_index: BTreeMap<usize, Vec<ProbeSpec>>,
}

impl ProbePlan {
    /// Probes at `idx` with the given placement, if any.
    pub fn at(&self, idx: usize, placement: ProbePlacement) -> impl Iterator<Item = &ProbeSpec> {
        self.by_index
            .get(&idx)
            .into_iter()
            .flatten()
            .filter(move |s| s.placement == placement)
    }

    /// Reproduces today's auto-identified control-flow probe set as data: a
    /// `TailTakeover`/`Active` probe at function entry, then one per
    /// `Block`/`Loop` header in source order — matching [`probe_site_count`]'s
    /// numbering exactly, and the configuration `blitz-specialize`'s opt/deopt
    /// machinery installs into. Backends still emit this set via their own
    /// dedicated entry/loop/block codegen (not by consulting a `ProbePlan`);
    /// this constructor demonstrates that today's behavior is fully
    /// expressible as one instance of the general probe model.
    pub fn control_flow_sites(body: &wasmparser::FunctionBody<'_>) -> ProbePlan {
        let site = |probe_id: u32| ProbeSpec {
            probe_id,
            binding: portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
            mode: ProbeMode::Active,
            placement: ProbePlacement::Before,
        };
        let mut plan = ProbePlan {
            entry: alloc::vec![site(0)],
            by_index: BTreeMap::new(),
        };
        let mut next_probe = 1u32;
        for (idx, op) in body
            .get_operators_reader()
            .into_iter()
            .flatten()
            .flatten()
            .enumerate()
        {
            if matches!(
                op,
                wasmparser::Operator::Block { .. } | wasmparser::Operator::Loop { .. }
            ) {
                plan.by_index.entry(idx).or_default().push(site(next_probe));
                next_probe += 1;
            }
        }
        plan
    }
}

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
    /// Optional probe-table configuration.  `None` (or `enabled: false`) → no
    /// probe code emitted (zero overhead on the fast path).
    pub probes: Option<ProbeTableConfig>,
    /// Optional embedder-requested probes at arbitrary instruction indices,
    /// in addition to the auto-identified control-flow probes above.  `None`
    /// → no additional probe code emitted.
    pub probe_plan: Option<ProbePlan>,
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

/// Counts the number of auto-identified control-flow probe sites in a function.
///
/// Site `0` is the function entry; every `Block` and `Loop` entry adds one more.
/// This is the size of the runtime [`ProbeSlot`] table the function needs and is
/// stored in [`ProbeTableConfig::num_probes`].  The per-site `probe_id` assigned
/// by the backends increments in the same source order this function scans.
///
/// Note: `If` and `TryTable` are *not* counted as probe sites (they are not
/// jump-back targets the way loops are, and `If` does not push a CTX frame the
/// preamble can rely on); keep this in sync with the backends' site emission.
pub fn probe_site_count(a: &FunctionBody<'_>) -> u32 {
    let mut count: u32 = 1; // function entry = site 0
    for op in a.get_operators_reader().into_iter().flatten().flatten() {
        match op {
            Operator::Block { .. } | Operator::Loop { .. } => {
                count += 1;
            }
            _ => {}
        }
    }
    count
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
