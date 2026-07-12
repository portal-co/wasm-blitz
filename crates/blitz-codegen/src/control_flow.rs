//! Shared compile-time-label-stack control-flow resolution.
//!
//! `blitz-riscv64::naive` and `blitz-aarch64::naive` already agree on this
//! model (`Endable`/`if_stack`, near-identical shape in both crates): each
//! structured WASM construct (`block`/`loop`/`if`) pushes a frame recording
//! which label(s) resolve a `br`/`br_if`/`br_table` targeting it, and where
//! `end`/`else` should place labels. This module provides that bookkeeping
//! once, generic over a minimal [`ControlFlowWriter`] trait.
//!
//! [`ControlFlowWriter`] deliberately does *not* extend [`crate::BlitzWriter`]
//! even though `branch_label`/`branch_zero_label`/`place_label` mean the same
//! thing in both: `BlitzWriter`'s other nine methods (probe-site/decrement
//! primitives) are irrelevant here, and a backend's control-flow adapter
//! (e.g. one already built around a regalloc-state reference for [`Self::flush`])
//! is often a different wrapper type than its `BlitzWriter` adapter — forcing
//! the supertrait would mean implementing nine unreachable stub methods on it.
//!
//! `TryTable`/exception handling is deliberately excluded — the catch-arm
//! arity/scratch-register conventions are arch-specific and higher-risk to
//! generalize; it stays a backend-local frame kind layered on top of this
//! (see `docs/regalloc-unification-plan.md`).

/// One entry in the compile-time structured-control-flow stack.
///
/// Mirrors `Endable` (minus `TryTable`) as duplicated in
/// `blitz-riscv64::naive` and `blitz-aarch64::naive`.
#[derive(Clone, Copy)]
pub enum Frame {
    /// `end`: placed by [`close_frame`]. A `br` to this frame is a forward
    /// jump, so `end` is only placed once the block's contents are done.
    Block { end: usize },
    /// `head`: placed immediately by [`open_loop`] — the loop itself is the
    /// branch target for a `br` to depth 0 inside it.
    Loop { head: usize },
    /// `then`/`else_`/`end`, allocated as one contiguous run by [`open_if`].
    /// `then` is placed immediately (so the true-condition path falls
    /// through); `else_` is placed by [`emit_else`] (or, for an `if` with no
    /// `else`, by [`close_frame`]); `end` is placed by [`close_frame`].
    If { then: usize, else_: usize, end: usize },
}

impl Frame {
    /// The label a `br`/`br_if`/`br_table` targeting this frame jumps to.
    pub fn branch_target(&self) -> usize {
        match *self {
            Frame::Block { end } => end,
            Frame::Loop { head } => head,
            Frame::If { end, .. } => end,
        }
    }
}

/// What a control-flow resolver needs from a backend.
pub trait ControlFlowWriter {
    type Error;

    /// Unconditional branch to the label with the given index.
    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error>;

    /// Branch to `label_idx` if `reg` is zero.
    fn branch_zero_label(&mut self, reg: u8, label_idx: usize) -> Result<(), Self::Error>;

    /// Place a label with the given index at the current output position.
    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error>;

    /// Emit whatever a backend needs before a structural branch or label
    /// placement crosses a control-flow boundary — for regalloc-backed
    /// backends, flushing any register-held operand-stack values to memory
    /// (and resetting TOS) so every path into a label sees consistent state.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Pop the next value (the WASM condition operand for `if`/`br_if`),
    /// returning which register it landed in. Always called immediately
    /// after [`Self::flush`], so backends may read it directly off a real
    /// memory stack rather than through a register allocator.
    fn pop_cond(&mut self) -> Result<u8, Self::Error>;
}

/// Open a `block`: allocate its `end` label. Does not place it — see
/// [`Frame::Block`].
pub fn open_block(label_counter: &mut usize) -> Frame {
    let end = *label_counter;
    *label_counter += 1;
    Frame::Block { end }
}

/// Open a `loop`: allocate and immediately place its `head` label.
pub fn open_loop<W: ControlFlowWriter>(w: &mut W, label_counter: &mut usize) -> Result<Frame, W::Error> {
    let head = *label_counter;
    *label_counter += 1;
    w.place_label(head)?;
    Ok(Frame::Loop { head })
}

/// Open an `if`: flush, pop the condition, allocate `then`/`else_`/`end`,
/// branch to `else_` if the condition is zero, then place `then`.
pub fn open_if<W: ControlFlowWriter>(w: &mut W, label_counter: &mut usize) -> Result<Frame, W::Error> {
    w.flush()?;
    let then = *label_counter;
    let else_ = then + 1;
    let end = then + 2;
    *label_counter += 3;
    let cond = w.pop_cond()?;
    w.branch_zero_label(cond, else_)?;
    w.place_label(then)?;
    Ok(Frame::If { then, else_, end })
}

/// Emit an `else`: flush, jump over the else-arm to `end`, then place
/// `else_` (the start of the else-arm).
pub fn emit_else<W: ControlFlowWriter>(w: &mut W, frame: &Frame) -> Result<(), W::Error> {
    let Frame::If { else_, end, .. } = *frame else {
        panic!("Else without a matching If frame")
    };
    w.flush()?;
    w.branch_label(end)?;
    w.place_label(else_)
}

/// Close a frame at `end`: flush, then place whichever label(s) `end`
/// resolves. For `If`, `else_` is placed unconditionally (a harmless no-op
/// via backward-patching if [`emit_else`] already placed it — needed only
/// when there was no `else` arm at all, so the false-condition branch still
/// has a valid, resolved target).
pub fn close_frame<W: ControlFlowWriter>(w: &mut W, frame: Frame) -> Result<(), W::Error> {
    w.flush()?;
    match frame {
        Frame::Block { end } => w.place_label(end),
        Frame::Loop { .. } => Ok(()),
        Frame::If { else_, end, .. } => {
            w.place_label(else_)?;
            w.place_label(end)
        }
    }
}

/// Resolve a `br` to `relative_depth`: flush, then jump to that frame's
/// branch target. A depth with no matching frame (malformed input) is a
/// silent no-op, matching the existing per-arch behavior.
pub fn branch_to_depth<W: ControlFlowWriter>(
    w: &mut W,
    frames: &[Frame],
    relative_depth: u32,
) -> Result<(), W::Error> {
    w.flush()?;
    resolve_depth(w, frames, relative_depth)
}

/// The label-resolution half of [`branch_to_depth`], without the flush.
///
/// Split out so `br_table` (which flushes once up front, then resolves each
/// arm) can reuse it without re-flushing per arm.
pub fn resolve_depth<W: ControlFlowWriter>(
    w: &mut W,
    frames: &[Frame],
    relative_depth: u32,
) -> Result<(), W::Error> {
    if let Some(frame) = frames.iter().rev().nth(relative_depth as usize) {
        w.branch_label(frame.branch_target())?;
    }
    Ok(())
}

/// Resolve a `br_if` to `relative_depth`: flush, pop the condition, and
/// branch to the target frame only if it's non-zero (skipping over an
/// inner [`branch_to_depth`] call otherwise — matching the existing
/// per-arch shape, including its harmless double-flush: `branch_to_depth`
/// flushes again internally, which is a no-op once already flushed).
pub fn branch_if_to_depth<W: ControlFlowWriter>(
    w: &mut W,
    label_counter: &mut usize,
    frames: &[Frame],
    relative_depth: u32,
) -> Result<(), W::Error> {
    w.flush()?;
    let cond = w.pop_cond()?;
    let skip = *label_counter;
    *label_counter += 1;
    w.branch_zero_label(cond, skip)?;
    branch_to_depth(w, frames, relative_depth)?;
    w.place_label(skip)
}
