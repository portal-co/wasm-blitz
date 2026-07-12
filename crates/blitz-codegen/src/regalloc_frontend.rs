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
    let guard = guard_reg(w, &rhs);
    let (lhs, cmds2) = w.regalloc_mut().as_mut().unwrap().pop(kind);
    let cmds2: Vec<_> = cmds2.collect();
    unguard_reg(w, &rhs, guard);
    w.emit_regalloc_cmds(cmds2)?;
    emit_op(w, lhs.reg, rhs.reg)?;
    let cmds3: Vec<_> = w.regalloc_mut().as_mut().unwrap().push_existing(lhs).collect();
    w.emit_regalloc_cmds(cmds3)
}

/// Temporarily reserve `t`'s register so a subsequent `pop()` can't select
/// the same slot.
///
/// The allocator's `pop()` frees a register the instant it hands out that
/// register's value (whether the value was already chained via a prior
/// `push`, or freshly materialized from the real stack because nothing was
/// chained — `tos == None`). In the freshly-materialized case specifically,
/// nothing else marks that register "still in use", so a *second* `pop()`
/// call made before the first value is consumed will see the same register
/// as the first available slot and pick it again — its real pop instruction
/// then clobbers the first value before any caller reads it. This is exactly
/// the shape `binop`/`compare`/`store` need (two pops before either value is
/// used), and is exactly what happens whenever the allocator's `tos` is
/// `None` for both — e.g. right after a `flush()`, or (on a backend that
/// falls back to a non-regalloc-aware sibling for some instructions, as
/// `blitz-x86-64::sysv` does) immediately after that sibling pushes a raw,
/// untracked value. Reserving `t`'s register in between makes the second
/// `pop()` skip it and materialize a genuinely different one.
fn guard_reg<W, K, const N: usize>(w: &mut W, t: &portal_solutions_asm_regalloc::Target<K>) -> portal_solutions_asm_regalloc::RegAllocFrame<K>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    core::mem::replace(
        &mut w.regalloc_mut().as_mut().unwrap().frames[t.kind.clone()][t.reg as usize],
        portal_solutions_asm_regalloc::RegAllocFrame::Reserved,
    )
}

/// Undo [`guard_reg`], restoring whatever was there before (always `Empty`
/// in practice, since `pop()` only ever hands back a register it just freed).
fn unguard_reg<W, K, const N: usize>(
    w: &mut W,
    t: &portal_solutions_asm_regalloc::Target<K>,
    prev: portal_solutions_asm_regalloc::RegAllocFrame<K>,
) where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    w.regalloc_mut().as_mut().unwrap().frames[t.kind.clone()][t.reg as usize] = prev;
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
    let guard = guard_reg(w, &rhs);
    let (lhs, cmds2) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds2: Vec<_> = cmds2.collect();
    unguard_reg(w, &rhs, guard);
    w.emit_regalloc_cmds(cmds2)?;
    // Guard both operands so `dest` (needed distinct — see doc comment) can't
    // alias either: `push`'s "first empty slot" scan would otherwise happily
    // hand back whichever of rhs/lhs's now-freed registers sorts lowest.
    let guard_rhs = guard_reg(w, &rhs);
    let guard_lhs = guard_reg(w, &lhs);
    let (dest, cmds3) = w
        .regalloc_mut()
        .as_mut()
        .unwrap()
        .push(kind)
        .unwrap_or_else(|_| panic!("regalloc error: kind conversion failed"));
    let cmds3: Vec<_> = cmds3.collect();
    unguard_reg(w, &rhs, guard_rhs);
    unguard_reg(w, &lhs, guard_lhs);
    w.emit_regalloc_cmds(cmds3)?;
    emit_cmp(w, dest, lhs.reg, rhs.reg)
}

/// `I32Load`/`I64Load`-shaped dispatch: pop an address, allocate a new dest
/// register, and load into it. `emit_load` receives `(dest, addr)`.
pub fn load<W, K, const N: usize>(
    w: &mut W,
    kind: K,
    emit_load: impl FnOnce(&mut W, u8, u8) -> Result<(), W::Error>,
) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let (addr, cmds1) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds1: Vec<_> = cmds1.collect();
    w.emit_regalloc_cmds(cmds1)?;
    let (dest, cmds2) = w
        .regalloc_mut()
        .as_mut()
        .unwrap()
        .push(kind)
        .unwrap_or_else(|_| panic!("regalloc error: kind conversion failed"));
    let cmds2: Vec<_> = cmds2.collect();
    w.emit_regalloc_cmds(cmds2)?;
    emit_load(w, dest, addr.reg)
}

/// `I32Store`/`I64Store`-shaped dispatch: pop the value then the address
/// (WASM stack order puts the address below the value, so the value pops
/// first) and store — no push, both operands are consumed. `emit_store`
/// receives `(val, addr)`.
pub fn store<W, K, const N: usize>(
    w: &mut W,
    kind: K,
    emit_store: impl FnOnce(&mut W, u8, u8) -> Result<(), W::Error>,
) -> Result<(), W::Error>
where
    W: RegAllocWriter<K, N>,
    K: Clone + Eq + TryFrom<usize>,
{
    ensure_regalloc(w);
    let (val, cmds1) = w.regalloc_mut().as_mut().unwrap().pop(kind.clone());
    let cmds1: Vec<_> = cmds1.collect();
    w.emit_regalloc_cmds(cmds1)?;
    let guard = guard_reg(w, &val);
    let (addr, cmds2) = w.regalloc_mut().as_mut().unwrap().pop(kind);
    let cmds2: Vec<_> = cmds2.collect();
    unguard_reg(w, &val, guard);
    w.emit_regalloc_cmds(cmds2)?;
    emit_store(w, val.reg, addr.reg)
}
