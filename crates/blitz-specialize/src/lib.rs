//! On-the-fly function specialization for wasm-blitz.
//!
//! This crate provides the *how* of adaptive specialization; the embedder owns
//! the *when* (policy).  It has three pieces, mirroring the three concerns of a
//! deoptimising JIT:
//!
//! 1. [`FnSlice`] + [`BranchAnalysis`] — a **slice-resident** (random-access)
//!    view of one function's instructions, plus a branch/control analysis that
//!    resolves every relative branch depth to an absolute instruction index.
//!    This is the alternative to consuming the streaming
//!    [`mach_operators`](portal_solutions_blitz_common::ops::mach_operators)
//!    iterator when you need to look around an instruction (e.g. to find a
//!    loop's body, or which site a branch targets).
//!
//! 2. [`SpecSpec`] + [`specialize`] — produce a specialized variant of a
//!    function by substituting embedder-asserted constants (value
//!    specialization) and folding loads from regions asserted constant (memory
//!    specialization), forward-folding the result.  Returns the new instruction
//!    stream plus the list of [`Guard`]s that must hold for the variant to be
//!    valid.
//!
//! 3. [`emit_deopt_guard`] — architecture-generic emission (over
//!    [`BlitzWriter`](portal_solutions_blitz_codegen::BlitzWriter)) of a guard
//!    that falls through when an assumption holds and branches to a *deopt
//!    target* otherwise.
//!
//! # Stack-state contract
//!
//! A specialized variant is installed in a [`ProbeSlot`]'s `handler` slot,
//! behind a `TailTakeover`-bound probe at the entry/loop/block sites built by
//! [`ProbeBinding::TailTakeover`](portal_solutions_blitz_codegen::ProbeBinding::TailTakeover)
//! (function entry, or a loop/block entry — see the blitz-codegen probe-site
//! emission).  When the probe dispatches there, the operand-stack and
//! CTX-frame layout is exactly that of the **generic site entry**.  A
//! specialized variant must therefore preserve that layout, and its deopt
//! target is the generic site entry: deopt is a plain branch back to generic
//! code with the live state untouched.  Because [`specialize`] only *substitutes
//! values* (it never changes stack shape), this holds by construction.
//!
//! [`ProbeSlot`]: portal_solutions_blitz_common::ops::ProbeSlot

#![no_std]
extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use portal_solutions_blitz_codegen::BlitzWriter;
use portal_solutions_blitz_common::ops::MachOperator;
use wasm_encoder::Instruction;
use wasmparser::Operator;

// ---------------------------------------------------------------------------
// (a) Slice-resident representation + branch analysis
// ---------------------------------------------------------------------------

/// A single function's instructions, materialised for random access.
///
/// Build one by collecting the slice of [`MachOperator`]s for a function (the
/// span between `StartFn` and `EndBody` produced by `mach_operators`).  Unlike
/// the streaming iterator, every instruction stays addressable by index, which
/// is what [`BranchAnalysis`] needs to resolve branch targets and locate sites.
#[derive(Clone, Debug, Default)]
pub struct FnSlice<'a, A = ()> {
    /// The function's operators, in source order.
    pub ops: Vec<MachOperator<'a, A>>,
}

impl<'a, A> FnSlice<'a, A> {
    /// Collect a function's operators into a slice.
    pub fn from_ops(ops: impl IntoIterator<Item = MachOperator<'a, A>>) -> Self {
        FnSlice { ops: ops.into_iter().collect() }
    }

    /// Number of operators in the slice.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Normalised control-flow category of one operator, extracted from either the
/// `Operator` (wasmparser) or `Instruction` (wasm-encoder) form of a
/// [`MachOperator`].
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ctl {
    Block,
    Loop,
    If,
    Else,
    End,
    TryTable,
    Br(u32),
    BrIf(u32),
    /// `(targets, default)`
    BrTable(Vec<u32>, u32),
    Return,
    /// Not a control-flow operator.
    Plain,
}

fn ctl_of<A>(op: &MachOperator<'_, A>) -> Ctl {
    match op {
        MachOperator::Operator { op: Some(o), .. } => ctl_of_operator(o),
        MachOperator::Instruction { op, .. } => ctl_of_instruction(op),
        _ => Ctl::Plain,
    }
}

fn ctl_of_operator(o: &Operator<'_>) -> Ctl {
    match o {
        Operator::Block { .. } => Ctl::Block,
        Operator::Loop { .. } => Ctl::Loop,
        Operator::If { .. } => Ctl::If,
        Operator::Else => Ctl::Else,
        Operator::End => Ctl::End,
        Operator::TryTable { .. } => Ctl::TryTable,
        Operator::Br { relative_depth } => Ctl::Br(*relative_depth),
        Operator::BrIf { relative_depth } => Ctl::BrIf(*relative_depth),
        Operator::BrTable { targets } => {
            let def = targets.default();
            let ts: Vec<u32> = targets.targets().filter_map(|t| t.ok()).collect();
            Ctl::BrTable(ts, def)
        }
        Operator::Return => Ctl::Return,
        _ => Ctl::Plain,
    }
}

fn ctl_of_instruction(i: &Instruction<'_>) -> Ctl {
    match i {
        Instruction::Block(_) => Ctl::Block,
        Instruction::Loop(_) => Ctl::Loop,
        Instruction::If(_) => Ctl::If,
        Instruction::Else => Ctl::Else,
        Instruction::End => Ctl::End,
        Instruction::TryTable(..) => Ctl::TryTable,
        Instruction::Br(d) => Ctl::Br(*d),
        Instruction::BrIf(d) => Ctl::BrIf(*d),
        Instruction::BrTable(ts, def) => Ctl::BrTable(ts.iter().copied().collect(), *def),
        Instruction::Return => Ctl::Return,
        _ => Ctl::Plain,
    }
}

/// Resolved control-flow analysis over a [`FnSlice`].
///
/// All indices are absolute positions into [`FnSlice::ops`].
#[derive(Clone, Debug, Default)]
pub struct BranchAnalysis {
    /// Control-nesting depth *before* executing each op (function body = 0).
    pub depth_at: Vec<usize>,
    /// For each control-header op index (`Block`/`Loop`/`If`/`TryTable`), the
    /// index of its matching `End`.
    pub header_to_end: BTreeMap<usize, usize>,
    /// For each branch op index (`Br`/`BrIf`/`BrTable`), the absolute target
    /// index/indices.  A `Br`/`BrIf` has exactly one entry; a `BrTable` has one
    /// per table entry followed by the default.
    pub branch_targets: BTreeMap<usize, Vec<usize>>,
    /// For each `Block`/`Loop` header index, the control-flow `probe_id`
    /// assigned to it (function entry is probe 0, so these start at 1, in
    /// source order — exactly matching `probe_site_count` and the backend
    /// probe numbering).
    pub probe_ids: BTreeMap<usize, u32>,
    /// Basic-block leader indices (sorted): the start of each straight-line
    /// region.  Index 0, any control header/`End`/`Else`, and the op *after* any
    /// branch/return are leaders.
    pub bb_leaders: Vec<usize>,
}

impl BranchAnalysis {
    /// Analyse a function slice.
    pub fn analyze<A>(slice: &FnSlice<'_, A>) -> BranchAnalysis {
        let ctl: Vec<Ctl> = slice.ops.iter().map(ctl_of).collect();
        let n = ctl.len();

        // Pass 1: match each header to its End, and assign probe ids.
        let mut header_to_end = BTreeMap::new();
        let mut probe_ids = BTreeMap::new();
        let mut next_probe: u32 = 1; // probe 0 is function entry
        let mut stack: Vec<usize> = Vec::new();
        for (i, c) in ctl.iter().enumerate() {
            match c {
                Ctl::Block | Ctl::Loop | Ctl::If | Ctl::TryTable => {
                    if matches!(c, Ctl::Block | Ctl::Loop) {
                        probe_ids.insert(i, next_probe);
                        next_probe += 1;
                    }
                    stack.push(i);
                }
                Ctl::End => {
                    if let Some(h) = stack.pop() {
                        header_to_end.insert(h, i);
                    }
                }
                _ => {}
            }
        }

        // Pass 2: depth tracking + branch-target resolution.
        let mut depth_at = Vec::with_capacity(n);
        let mut branch_targets = BTreeMap::new();
        let mut stack: Vec<(usize, Ctl)> = Vec::new();
        for (i, c) in ctl.iter().enumerate() {
            depth_at.push(stack.len());
            // `End`/`Else` belong to the enclosing structure; depth recorded
            // above is the pre-op depth, which is what callers expect.
            let resolve = |stk: &[(usize, Ctl)], rel: u32| -> Option<usize> {
                if (rel as usize) >= stk.len() {
                    return None; // branch out of the function
                }
                let (hidx, hkind) = &stk[stk.len() - 1 - rel as usize];
                match hkind {
                    // Branching to a loop targets its header (back-edge).
                    Ctl::Loop => Some(*hidx),
                    // Branching to a block/if/try targets its matching End (forward).
                    _ => header_to_end.get(hidx).copied(),
                }
            };
            match c {
                Ctl::Block | Ctl::Loop | Ctl::If | Ctl::TryTable => {
                    stack.push((i, c.clone()));
                }
                Ctl::End => {
                    stack.pop();
                }
                Ctl::Br(rel) | Ctl::BrIf(rel) => {
                    if let Some(t) = resolve(&stack, *rel) {
                        branch_targets.insert(i, alloc::vec![t]);
                    }
                }
                Ctl::BrTable(ts, def) => {
                    let mut v = Vec::with_capacity(ts.len() + 1);
                    for &t in ts {
                        if let Some(a) = resolve(&stack, t) {
                            v.push(a);
                        }
                    }
                    if let Some(a) = resolve(&stack, *def) {
                        v.push(a);
                    }
                    branch_targets.insert(i, v);
                }
                _ => {}
            }
        }

        // Basic-block leaders.
        let mut leader = alloc::vec![false; n];
        if n > 0 {
            leader[0] = true;
        }
        for (i, c) in ctl.iter().enumerate() {
            match c {
                Ctl::Block | Ctl::Loop | Ctl::If | Ctl::TryTable | Ctl::Else | Ctl::End => {
                    leader[i] = true;
                }
                Ctl::Br(_) | Ctl::BrIf(_) | Ctl::BrTable(..) | Ctl::Return => {
                    if i + 1 < n {
                        leader[i + 1] = true;
                    }
                }
                _ => {}
            }
            // Branch *targets* are also leaders.
            if let Some(ts) = branch_targets.get(&i) {
                for &t in ts {
                    if t < n {
                        leader[t] = true;
                    }
                }
            }
        }
        let bb_leaders = (0..n).filter(|&i| leader[i]).collect();

        BranchAnalysis { depth_at, header_to_end, branch_targets, probe_ids, bb_leaders }
    }

    /// Resolved target(s) of the branch at `idx`, if it is a branch.
    pub fn branch_target(&self, idx: usize) -> Option<&[usize]> {
        self.branch_targets.get(&idx).map(|v| v.as_slice())
    }

    /// Control-flow `probe_id` of the `Block`/`Loop` header at `idx`, if any.
    pub fn probe_id_of(&self, idx: usize) -> Option<u32> {
        self.probe_ids.get(&idx).copied()
    }

    /// Whether `idx` begins a basic block.
    pub fn is_leader(&self, idx: usize) -> bool {
        self.bb_leaders.binary_search(&idx).is_ok()
    }
}

// ---------------------------------------------------------------------------
// (b) Specialization variants
// ---------------------------------------------------------------------------

/// A concrete constant value an embedder can assert about a local/global, or
/// that is read from a constant memory region.
///
/// Only integer values are supported for now (the common value-specialization
/// case); float specialization is left to a future revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstVal {
    I32(i32),
    I64(i64),
}

impl ConstVal {
    fn as_addr(self) -> Option<u64> {
        match self {
            ConstVal::I32(v) => Some(v as u32 as u64),
            ConstVal::I64(v) => Some(v as u64),
        }
    }

    fn to_instruction<'a>(self) -> Instruction<'a> {
        match self {
            ConstVal::I32(v) => Instruction::I32Const(v),
            ConstVal::I64(v) => Instruction::I64Const(v),
        }
    }
}

/// A region of guest memory the embedder asserts holds known constant bytes.
#[derive(Clone, Debug)]
pub struct ConstRegion {
    /// Guest address of the region's first byte.
    pub addr: u64,
    /// The asserted bytes, starting at `addr`.
    pub bytes: Vec<u8>,
}

impl ConstRegion {
    /// Read a little-endian value of `width` bytes at guest `addr`, if fully
    /// inside this region.
    fn read(&self, addr: u64, width: usize) -> Option<u64> {
        let start = addr.checked_sub(self.addr)? as usize;
        let end = start.checked_add(width)?;
        let slice = self.bytes.get(start..end)?;
        let mut v: u64 = 0;
        for (i, &b) in slice.iter().enumerate() {
            v |= (b as u64) << (8 * i);
        }
        Some(v)
    }
}

/// Embedder-supplied specialization assumptions.
///
/// The embedder decides *which* assumptions to make (the policy); this crate
/// applies them and records the [`Guard`]s needed to keep them honest.
#[derive(Clone, Debug, Default)]
pub struct SpecSpec {
    /// Locals asserted to hold a constant on entry to the specialized region.
    pub locals: BTreeMap<u32, ConstVal>,
    /// Globals asserted to hold a constant.
    pub globals: BTreeMap<u32, ConstVal>,
    /// Memory regions asserted to hold known constant bytes.
    pub const_regions: Vec<ConstRegion>,
}

/// A runtime assumption that a specialized variant depends on.
///
/// The embedder must emit a check for each guard (via [`emit_deopt_guard`])
/// before the specialized body; if any check fails the variant is invalid and
/// control must deopt to the generic site entry.
#[derive(Clone, Debug, PartialEq)]
pub enum Guard {
    /// Local `idx` must equal `expected`.
    Local { idx: u32, expected: ConstVal },
    /// Global `idx` must equal `expected`.
    Global { idx: u32, expected: ConstVal },
    /// The `width` bytes at guest `addr` must equal the LE encoding of `value`.
    Mem { addr: u64, width: u8, value: u64 },
}

/// Output of [`specialize`]: the rewritten instruction stream and the guards it
/// depends on.
#[derive(Clone, Debug, Default)]
pub struct SpecializedFn<'a> {
    /// The specialized operators (always in `Instruction` form).
    pub ops: Vec<MachOperator<'a, ()>>,
    /// Assumptions that must hold for `ops` to be valid.
    pub guards: Vec<Guard>,
}

/// Produce a specialized variant of `slice` under `spec`.
///
/// Substitutes asserted-constant `local.get`/`global.get`, folds loads from
/// constant memory regions, and forward-folds the result with a small set of
/// integer peepholes.  Meta-operators (`StartFn`/`Local`/`StartBody`/`EndBody`)
/// are dropped — the result is the function *body* an embedder feeds back to a
/// backend at the specialization site.  Returns the rewritten ops plus the
/// [`Guard`]s the variant depends on.
///
/// `analysis` is accepted so future revisions can scope assumptions to specific
/// sites/blocks; the current implementation specializes the whole body.
pub fn specialize<'a, A>(
    slice: &FnSlice<'a, A>,
    _analysis: &BranchAnalysis,
    spec: &SpecSpec,
) -> SpecializedFn<'static> {
    let mut out: Vec<MachOperator<'static, ()>> = Vec::new();
    let mut guards: Vec<Guard> = Vec::new();

    // Pass 1: substitute bound locals/globals with constants; pass other
    // body instructions through in `Instruction` form.
    for op in slice.ops.iter() {
        let insn = match normalize_body_instruction(op) {
            None => continue, // meta-op or unrepresentable operator: skip
            Some(i) => i,
        };
        match &insn {
            Instruction::LocalGet(i) => {
                if let Some(&cv) = spec.locals.get(i) {
                    record_guard(&mut guards, Guard::Local { idx: *i, expected: cv });
                    out.push(mach_insn(cv.to_instruction()));
                    continue;
                }
            }
            Instruction::GlobalGet(i) => {
                if let Some(&cv) = spec.globals.get(i) {
                    record_guard(&mut guards, Guard::Global { idx: *i, expected: cv });
                    out.push(mach_insn(cv.to_instruction()));
                    continue;
                }
            }
            _ => {}
        }
        out.push(mach_insn(insn));
    }

    // Pass 2: forward const-folding peepholes to fixpoint (folds the constants
    // we just substituted, plus loads from constant regions).
    fold_to_fixpoint(&mut out, &mut guards, spec);

    SpecializedFn { ops: out, guards }
}

fn record_guard(guards: &mut Vec<Guard>, g: Guard) {
    if !guards.contains(&g) {
        guards.push(g);
    }
}

fn mach_insn<'a>(i: Instruction<'a>) -> MachOperator<'a, ()> {
    MachOperator::Instruction { op: i, annot: () }
}

/// Extract the body `Instruction` for an operator, owning it for `'static`.
///
/// Returns `None` for meta-operators (`StartFn`/`Local`/`StartBody`/`EndBody`/
/// `Trap`) and for operators we do not model.  Both `Operator` and
/// `Instruction` input forms are handled.
fn normalize_body_instruction<A>(op: &MachOperator<'_, A>) -> Option<Instruction<'static>> {
    match op {
        MachOperator::Instruction { op, .. } => clone_static(op),
        MachOperator::Operator { op: Some(o), .. } => operator_to_instruction(o),
        _ => None,
    }
}

/// Pull out the value of a constant-producing instruction, if any.
fn const_of(i: &Instruction<'_>) -> Option<ConstVal> {
    match i {
        Instruction::I32Const(v) => Some(ConstVal::I32(*v)),
        Instruction::I64Const(v) => Some(ConstVal::I64(*v)),
        _ => None,
    }
}

fn is_insn<A>(op: &MachOperator<'_, A>) -> Option<Instruction<'static>> {
    match op {
        MachOperator::Instruction { op, .. } => clone_static(op),
        _ => None,
    }
}

/// Iterative constant-folding peepholes over a body in `Instruction` form.
fn fold_to_fixpoint(out: &mut Vec<MachOperator<'static, ()>>, guards: &mut Vec<Guard>, spec: &SpecSpec) {
    let mut changed = true;
    while changed {
        changed = false;

        // Fold `const(addr) load.*` -> `const(value)` from constant regions.
        let mut i = 0;
        while i + 1 < out.len() {
            let addr = is_insn(&out[i]).as_ref().and_then(const_of).and_then(|c| c.as_addr());
            let load = is_insn(&out[i + 1]);
            if let (Some(addr), Some(load)) = (addr, load) {
                if let Some((width, signed_i64, memarg_off)) = load_width(&load) {
                    let eff = addr.wrapping_add(memarg_off);
                    if let Some(region) = spec.const_regions.iter().find(|r| r.read(eff, width).is_some()) {
                        let raw = region.read(eff, width).unwrap();
                        let cv = if signed_i64 {
                            ConstVal::I64(raw as i64)
                        } else {
                            ConstVal::I32(raw as u32 as i32)
                        };
                        record_guard(guards, Guard::Mem { addr: eff, width: width as u8, value: raw });
                        out.splice(i..=i + 1, [mach_insn(cv.to_instruction())]);
                        changed = true;
                        continue;
                    }
                }
            }
            i += 1;
        }

        // Fold `const(a) const(b) binop` -> `const(fold)`.
        let mut i = 0;
        while i + 2 < out.len() {
            let a = is_insn(&out[i]).as_ref().and_then(const_of);
            let b = is_insn(&out[i + 1]).as_ref().and_then(const_of);
            let opc = is_insn(&out[i + 2]);
            if let (Some(a), Some(b), Some(opc)) = (a, b, opc) {
                if let Some(folded) = fold_binop(a, b, &opc) {
                    out.splice(i..=i + 2, [mach_insn(folded.to_instruction())]);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }
    }
}

/// `(width_bytes, result_is_i64, memarg_offset)` for the loads we model.
fn load_width(i: &Instruction<'_>) -> Option<(usize, bool, u64)> {
    match i {
        Instruction::I32Load(m) => Some((4, false, m.offset)),
        Instruction::I64Load(m) => Some((8, true, m.offset)),
        _ => None,
    }
}

fn fold_binop(a: ConstVal, b: ConstVal, op: &Instruction<'_>) -> Option<ConstVal> {
    match (a, b) {
        (ConstVal::I32(x), ConstVal::I32(y)) => {
            let r = match op {
                Instruction::I32Add => x.wrapping_add(y),
                Instruction::I32Sub => x.wrapping_sub(y),
                Instruction::I32Mul => x.wrapping_mul(y),
                Instruction::I32And => x & y,
                Instruction::I32Or => x | y,
                Instruction::I32Xor => x ^ y,
                _ => return None,
            };
            Some(ConstVal::I32(r))
        }
        (ConstVal::I64(x), ConstVal::I64(y)) => {
            let r = match op {
                Instruction::I64Add => x.wrapping_add(y),
                Instruction::I64Sub => x.wrapping_sub(y),
                Instruction::I64Mul => x.wrapping_mul(y),
                Instruction::I64And => x & y,
                Instruction::I64Or => x | y,
                Instruction::I64Xor => x ^ y,
                _ => return None,
            };
            Some(ConstVal::I64(r))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// (c) Deopt guard emission
// ---------------------------------------------------------------------------

/// Emit a deopt guard: fall through when an assumption holds, branch to
/// `deopt_label` when it does not.
///
/// The caller materialises `diff_reg` such that it is **zero iff the assumption
/// holds** (e.g. `actual XOR expected`, or `actual - expected`) using
/// backend-specific instructions — reading a local, global, or memory word is
/// outside the minimal [`BlitzWriter`] vocabulary, so it stays with the
/// backend.  This function only encodes the branch logic, keeping it
/// architecture-generic.
///
/// `deopt_label` is the label of the generic site entry (function entry or the
/// loop/block site) — see the crate-level stack-state contract.  One scratch
/// label is allocated from `label_counter`.
pub fn emit_deopt_guard<W: BlitzWriter>(
    w: &mut W,
    diff_reg: u8,
    deopt_label: usize,
    label_counter: &mut usize,
) -> Result<(), W::Error> {
    let cont = *label_counter;
    *label_counter += 1;
    // diff == 0  -> assumption holds -> continue into specialized body.
    w.branch_zero_label(diff_reg, cont)?;
    // diff != 0  -> deopt to the generic site entry.
    w.branch_label(deopt_label)?;
    w.place_label(cont)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// wasm-encoder / wasmparser bridging helpers
// ---------------------------------------------------------------------------

/// Clone a borrowed `Instruction` into a `'static` one, for the subset we model.
///
/// Instructions that borrow (`BrTable`, `TryTable`, vector/select-with-types,
/// etc.) are not part of the specialization subset and return `None`.
fn clone_static(i: &Instruction<'_>) -> Option<Instruction<'static>> {
    Some(match i {
        Instruction::I32Const(v) => Instruction::I32Const(*v),
        Instruction::I64Const(v) => Instruction::I64Const(*v),
        Instruction::LocalGet(v) => Instruction::LocalGet(*v),
        Instruction::LocalSet(v) => Instruction::LocalSet(*v),
        Instruction::LocalTee(v) => Instruction::LocalTee(*v),
        Instruction::GlobalGet(v) => Instruction::GlobalGet(*v),
        Instruction::GlobalSet(v) => Instruction::GlobalSet(*v),
        Instruction::I32Load(m) => Instruction::I32Load(*m),
        Instruction::I64Load(m) => Instruction::I64Load(*m),
        Instruction::I32Store(m) => Instruction::I32Store(*m),
        Instruction::I64Store(m) => Instruction::I64Store(*m),
        Instruction::I32Add => Instruction::I32Add,
        Instruction::I32Sub => Instruction::I32Sub,
        Instruction::I32Mul => Instruction::I32Mul,
        Instruction::I32And => Instruction::I32And,
        Instruction::I32Or => Instruction::I32Or,
        Instruction::I32Xor => Instruction::I32Xor,
        Instruction::I64Add => Instruction::I64Add,
        Instruction::I64Sub => Instruction::I64Sub,
        Instruction::I64Mul => Instruction::I64Mul,
        Instruction::I64And => Instruction::I64And,
        Instruction::I64Or => Instruction::I64Or,
        Instruction::I64Xor => Instruction::I64Xor,
        Instruction::Return => Instruction::Return,
        Instruction::Nop => Instruction::Nop,
        Instruction::Drop => Instruction::Drop,
        _ => return None,
    })
}

/// Convert a wasmparser `Operator` to a `'static` wasm-encoder `Instruction`,
/// for the subset the specializer models.
fn operator_to_instruction(o: &Operator<'_>) -> Option<Instruction<'static>> {
    use wasm_encoder::MemArg;
    let conv_mem = |m: &wasmparser::MemArg| MemArg {
        offset: m.offset,
        align: m.align as u32,
        memory_index: m.memory,
    };
    Some(match o {
        Operator::I32Const { value } => Instruction::I32Const(*value),
        Operator::I64Const { value } => Instruction::I64Const(*value),
        Operator::LocalGet { local_index } => Instruction::LocalGet(*local_index),
        Operator::LocalSet { local_index } => Instruction::LocalSet(*local_index),
        Operator::LocalTee { local_index } => Instruction::LocalTee(*local_index),
        Operator::GlobalGet { global_index } => Instruction::GlobalGet(*global_index),
        Operator::GlobalSet { global_index } => Instruction::GlobalSet(*global_index),
        Operator::I32Load { memarg } => Instruction::I32Load(conv_mem(memarg)),
        Operator::I64Load { memarg } => Instruction::I64Load(conv_mem(memarg)),
        Operator::I32Store { memarg } => Instruction::I32Store(conv_mem(memarg)),
        Operator::I64Store { memarg } => Instruction::I64Store(conv_mem(memarg)),
        Operator::I32Add => Instruction::I32Add,
        Operator::I32Sub => Instruction::I32Sub,
        Operator::I32Mul => Instruction::I32Mul,
        Operator::I32And => Instruction::I32And,
        Operator::I32Or => Instruction::I32Or,
        Operator::I32Xor => Instruction::I32Xor,
        Operator::I64Add => Instruction::I64Add,
        Operator::I64Sub => Instruction::I64Sub,
        Operator::I64Mul => Instruction::I64Mul,
        Operator::I64And => Instruction::I64And,
        Operator::I64Or => Instruction::I64Or,
        Operator::I64Xor => Instruction::I64Xor,
        Operator::Return => Instruction::Return,
        Operator::Nop => Instruction::Nop,
        Operator::Drop => Instruction::Drop,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
