//! Shared WASM data-flow dispatch for regalloc-backed backends.
//!
//! Every regalloc-backed backend (`blitz-riscv64::naive`, and eventually
//! `blitz-x86-64::fast` and an AArch64 equivalent) hand-writes the same
//! "pop operand(s) -> emit any spill/reload commands -> do the op -> push
//! result -> emit any spill/reload commands" dance for every WASM
//! const/local-access/binary-op instruction. This module provides that
//! sequencing once, generic over a minimal [`RegAllocWriter`] a backend
//! implements to supply the genuinely arch-specific pieces: how to
//! initialize an empty allocator (reserved-register layout differs per
//! arch) and how to turn a batch of [`Cmd`]s into instructions (`process_cmd`
//! itself, and any extra state like x86's `StackManager`, differ per arch).
//!
//! The actual instruction selection (e.g. `add` vs. `lea`+`not` for
//! subtraction) is *not* covered here — callers supply it as a closure
//! operating on already-allocated register numbers, since that's exactly
//! where backends legitimately differ (see `docs/regalloc-unification-plan.md`
//! for why control-flow and data-flow shape is shared while instruction
//! selection stays per-arch).

use crate::regalloc_adapter::Frames;
use alloc::vec::Vec;
use portal_solutions_asm_regalloc::{Cmd, RegAlloc};

/// What a regalloc-backed backend must supply for the free functions below.
///
/// `K` is the arch's register-kind enum (e.g. `RegKind::{Int,Float}`); `N` is
/// the number of physical registers per kind.
pub trait RegAllocWriter<K, const N: usize>
where
    K: Clone + Eq + TryFrom<usize>,
{
    type Error;

    /// Mutable access to the lazily-initialized allocator.
    fn regalloc_mut(&mut self) -> &mut Option<RegAlloc<K, N, Frames<K, N>>>;

    /// Construct a fresh, empty allocator. Called once, the first time any
    /// of the free functions below is used on a given `regalloc_mut()`.
    /// Arch-specific because the reserved/available register set differs
    /// (e.g. RISC-V reserves x0/ra/sp/gp/tp/fp; x86-64 reserves RSP/RBP/CTX).
    fn init_regalloc(&self) -> RegAlloc<K, N, Frames<K, N>>;

    /// Turn a batch of spill/reload commands into real instructions
    /// (wraps the arch's own `process_cmd`, plus any extra state it needs,
    /// e.g. x86-64's `StackManager`).
    fn emit_regalloc_cmds(&mut self, cmds: Vec<Cmd<K>>) -> Result<(), Self::Error>;
}

fn ensure_regalloc<W, K, const N: usize>(w: &mut W)
where
    W: RegAllocWriter<K, N> + ?Sized,
    K: Clone + Eq + TryFrom<usize>,
{
    if w.regalloc_mut().is_none() {
        let r = w.init_regalloc();
        *w.regalloc_mut() = Some(r);
    }
}

/// `I32Const`/`I64Const`/`F32Const`/`F64Const`-shaped dispatch: allocate a
/// register for a new stack value, then materialize an immediate into it.
pub fn push_const<W, K, const N: usize>(
    w: &mut W,
    kind: K,
    emit_imm: impl FnOnce(&mut W, u8) -> Result<(), W::Error>,
) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let (reg, cmds) = w
        .regalloc_mut()
        .as_mut()
        .unwrap()
        .push(kind)
        .unwrap_or_else(|_| panic!("regalloc error: kind conversion failed"));
    let cmds: Vec<_> = cmds.collect();
    w.emit_regalloc_cmds(cmds)?;
    emit_imm(w, reg)
}

/// `LocalGet`-shaped dispatch: push the value already resident in a local
/// onto the virtual stack (reusing its register if still live, otherwise
/// reloading it).
pub fn push_local<W, K, const N: usize>(w: &mut W, kind: K, local_index: u32) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let cmds = w
        .regalloc_mut()
        .as_mut()
        .unwrap()
        .push_local(kind, local_index)
        .unwrap_or_else(|_| panic!("regalloc error: kind conversion failed"));
    let cmds: Vec<_> = cmds.collect();
    w.emit_regalloc_cmds(cmds)
}

/// `LocalSet`-shaped dispatch: pop the top of the virtual stack into a local
/// (no memory write yet — deferred to `flush`/eviction, matching the
/// allocator's own laziness).
pub fn pop_to_local<W, K, const N: usize>(w: &mut W, kind: K, local_index: u32) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let cmds = w.regalloc_mut().as_mut().unwrap().pop_local(kind, local_index);
    let cmds: Vec<_> = cmds.collect();
    w.emit_regalloc_cmds(cmds)
}

/// The "pop rhs, pop lhs, compute in place, push result" shape shared by
/// every WASM binary op (`I32Add`, `I64And`, comparisons, ...). `emit_op`
/// receives `(dst = lhs's register, rhs's register)` and is expected to
/// leave the result in `dst`; the allocator is told that register now holds
/// the (single) result.
pub fn binop<W, K, const N: usize>(
    w: &mut W,
    kind: K,
    emit_op: impl FnOnce(&mut W, u8, u8) -> Result<(), W::Error>,
) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let (rhs, cmds1) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds1: Vec<_> = cmds1.collect();
    w.emit_regalloc_cmds(cmds1)?;
    let (lhs, cmds2) = w.regalloc_mut().as_mut().unwrap().pop(kind);
    let cmds2: Vec<_> = cmds2.collect();
    w.emit_regalloc_cmds(cmds2)?;
    emit_op(w, lhs.reg, rhs.reg)?;
    let cmds3: Vec<_> = w.regalloc_mut().as_mut().unwrap().push_existing(lhs).collect();
    w.emit_regalloc_cmds(cmds3)
}

/// The "pop rhs, pop lhs, allocate a *new* dest register, compute" shape used
/// by comparison ops (`I32Lt*`, `I32Eq`, ...) — unlike [`binop`], the result
/// needs a register distinct from either operand (typically materialized via
/// a branch + `li 0`/`li 1` sequence that still needs both operands live).
/// `emit_cmp` receives `(dest, lhs, rhs)`.
pub fn compare<W, K, const N: usize>(
    w: &mut W,
    kind: K,
    emit_cmp: impl FnOnce(&mut W, u8, u8, u8) -> Result<(), W::Error>,
) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let (rhs, cmds1) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds1: Vec<_> = cmds1.collect();
    w.emit_regalloc_cmds(cmds1)?;
    let (lhs, cmds2) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds2: Vec<_> = cmds2.collect();
    w.emit_regalloc_cmds(cmds2)?;
    let (dest, cmds3) = w
        .regalloc_mut()
        .as_mut()
        .unwrap()
        .push(kind)
        .unwrap_or_else(|_| panic!("regalloc error: kind conversion failed"));
    let cmds3: Vec<_> = cmds3.collect();
    w.emit_regalloc_cmds(cmds3)?;
    emit_cmp(w, dest, lhs.reg, rhs.reg)
}
