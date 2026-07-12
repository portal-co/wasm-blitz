//! Shared WASM instruction codegen primitives for blitz backends.
//!
//! Defines [`BlitzWriter`] — a minimal set of architecture primitives needed
//! to express complex WASM instructions generically — plus ready-made
//! implementations of those instructions that work for any conforming backend.
//!
//! # Why this crate exists
//!
//! Each blitz-* backend (x86-64, AArch64, RISC-V) previously had its own
//! copy of instructions like `BrTable` and the JIT tracing preamble. These
//! copies diverged with each edit. This crate provides single implementations
//! that all three backends (and both ABIs) share, eliminating the fan-out.
//!
//! # Usage
//!
//! Wrap your arch-specific writer + context in a [`BlitzWriter`] impl, then
//! call the free functions in this crate:
//!
//! ```ignore
//! blitz_codegen::emit_probe_site(&mut bw, base_off, probe_id, scratch, binding, &mut label_counter)?;
//! blitz_codegen::emit_br_table(&mut bw, selector_reg, targets, default, &mut label_counter, resolve)?;
//! ```

#![no_std]
extern crate alloc;

pub mod control_flow;
pub mod regalloc_adapter;
pub mod regalloc_frontend;

/// How a probe site dispatches to its handler once the handler pointer is
/// found to be non-null.
///
/// This is a compile-time choice (it changes which instructions get emitted),
/// not something the runtime can toggle — only *whether* a handler is
/// installed is runtime-switchable (a null handler pointer means "skip").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeBinding {
    /// Call the handler and fall through to the rest of the function once it
    /// returns. The handler is an ordinary function under a minimal,
    /// register-only probe ABI (see the wasm-blitz ABI reference).
    Call,
    /// Unconditionally transfer control to the handler with the current
    /// stack-state-contract intact; the probe site never regains control.
    /// Used for specialization opt-entry.
    TailTakeover,
}

/// Minimal architecture primitives for shared WASM instruction codegen.
///
/// Implementors bind together the arch-specific writer, its context, and the
/// arch descriptor (e.g. `X64Arch`) in a concrete struct, then implement this
/// trait so the free functions in this crate can generate correct code for any
/// architecture.
pub trait BlitzWriter {
    type Error;

    // ---- branches ---------------------------------------------------------

    /// Unconditional branch to the label with the given index.
    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error>;

    /// Branch to `label_idx` if `reg` is zero (equal to zero).
    /// Used by both `emit_br_table` (decrement approach) and `emit_probe_site`.
    fn branch_zero_label(&mut self, reg: u8, label_idx: usize) -> Result<(), Self::Error>;

    /// Indirect branch through the value in `reg`. Never returns to the
    /// instruction after it — used for [`ProbeBinding::TailTakeover`].
    fn branch_reg(&mut self, reg: u8) -> Result<(), Self::Error>;

    /// Indirect call through the value in `reg`; falls through to the next
    /// instruction once the callee returns — used for [`ProbeBinding::Call`].
    ///
    /// The callee is expected to follow the minimal, register-only probe ABI
    /// (no stack args/results) and to leave the real stack pointer net
    /// unchanged, so this is always safe to splice in regardless of what the
    /// surrounding code is using the stack for.
    fn call_reg(&mut self, reg: u8) -> Result<(), Self::Error>;

    // ---- label placement --------------------------------------------------

    /// Place a label with the given index at the current output position.
    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error>;

    // ---- BrTable helper ---------------------------------------------------

    /// Decrement `reg` by 1 in place.
    ///
    /// Used between arms in `emit_br_table` (decrement approach): the selector
    /// starts at N-1 and is decremented to compare against 0 for each arm.
    fn reg_decrement(&mut self, reg: u8) -> Result<(), Self::Error>;

    // ---- immediate / memory materialisation (JIT preamble) ---------------

    /// Load a 64-bit absolute address or immediate into `dest`.
    fn load_u64_imm(&mut self, dest: u8, imm: u64) -> Result<(), Self::Error>;

    /// Atomicity-free increment of the 64-bit value at the address in `ptr_reg`.
    /// Used for approximate invocation counters; data races are acceptable.
    fn inc_mem64(&mut self, ptr_reg: u8) -> Result<(), Self::Error>;

    /// Load the 64-bit value at the address stored in `src` into `dest`.
    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error>;

    // ---- CTX-relative trace-table access (JIT preamble) ------------------

    /// Load the runtime-provided probe-table base pointer into `dest`.
    ///
    /// The base is stored by the runtime at a fixed, CTX-relative slot
    /// `base_off` (see `ProbeTableConfig::table_base_off`).  This is how the
    /// preamble reaches its [`ProbeSlot`](../portal_solutions_blitz_common/ops/struct.ProbeSlot.html)
    /// table without any absolute address being baked into the code.
    fn load_probe_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error>;

    /// Atomicity-free increment of the 64-bit value at `[ptr_reg + disp]`.
    fn inc_mem64_disp(&mut self, ptr_reg: u8, disp: i32) -> Result<(), Self::Error>;

    /// Load the 64-bit value at `[src + disp]` into `dest`.
    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error>;
}

/// Size in bytes of one `ProbeSlot` entry (`counter: u64` + `handler: ptr`).
pub const PROBE_SLOT_SIZE: i32 = 16;
/// Byte offset of the `handler` pointer within a `ProbeSlot`.
pub const PROBE_SLOT_HANDLER_OFF: i32 = 8;

// ---------------------------------------------------------------------------
// Shared instruction implementations
// ---------------------------------------------------------------------------

/// Emit one probe site.
///
/// Reaches the runtime [`ProbeSlot`] table through a CTX-relative base
/// (`base_off`, see `ProbeTableConfig::table_base_off`), indexes it by
/// `probe_id`, increments that probe's invocation counter, then checks its
/// handler code pointer.  If non-null it dispatches per `binding`; otherwise
/// (or once a [`ProbeBinding::Call`] handler returns) it falls through to
/// `body_label` (placed by this function).
///
/// No absolute address is baked into the generated code — only the structural
/// `base_off` and `probe_id` — so compilation is independent of the runtime
/// address space.
///
/// `scratch` is a caller-provided scratch register (by blitz register number).
/// Architectures that need a second scratch register for `inc_mem64_disp` store
/// it in their [`BlitzWriter`] implementation — callers don't need to know.
///
/// The `label_counter` is incremented once by this function to allocate the
/// body label.
///
/// This function only encodes the "is a handler installed, and how do we
/// reach it" logic. Callers emitting a [`ProbeBinding::Call`] probe are
/// responsible for the surrounding save/restore of any state the call would
/// otherwise disturb (the Active/Passive distinction — see the wasm-blitz
/// probes design), since that is backend-specific and outside the minimal
/// [`BlitzWriter`] vocabulary.
pub fn emit_probe_site<W: BlitzWriter>(
    w: &mut W,
    base_off: i32,
    probe_id: u32,
    scratch: u8,
    binding: ProbeBinding,
    label_counter: &mut usize,
) -> Result<(), W::Error> {
    let body_label = *label_counter;
    *label_counter += 1;

    let slot_off = probe_id as i32 * PROBE_SLOT_SIZE;

    // 1. Load runtime probe-table base (no baked address).
    w.load_probe_base(scratch, base_off)?;

    // 2. Increment this probe's invocation counter (non-atomic, approximate).
    w.inc_mem64_disp(scratch, slot_off)?;

    // 3. Load handler code-ptr; dispatch if non-null.
    w.load_mem64_disp(scratch, scratch, slot_off + PROBE_SLOT_HANDLER_OFF)?;
    w.branch_zero_label(scratch, body_label)?;
    match binding {
        ProbeBinding::TailTakeover => w.branch_reg(scratch)?,
        ProbeBinding::Call => w.call_reg(scratch)?,
    }
    w.place_label(body_label)?;
    Ok(())
}

/// Emit a `br_table` using the decrement approach.
///
/// The `selector_reg` must already be loaded with the table index. This
/// function consumes the register (it is decremented in place).
///
/// Algorithm (works naturally on all three architectures):
/// 1. For each arm i: if selector is zero, branch to arm i's target; else decrement.
/// 2. Fall through to `resolve(w, default)`.
/// 3. After each arm label: `resolve(w, targets[i])`.
///
/// Architecture fit:
/// - x86-64: `test + je` (single flag check) + `add reg, -1`
/// - AArch64: `cbz` (compact zero-test-and-branch) + `sub reg, reg, 1`
/// - RISC-V: `beq reg, x0, label` (natural two-register form) + `addi reg, reg, -1`
///
/// `resolve` emits whatever code branches to the given relative depth.
/// `label_counter` is advanced by `targets.len()` to allocate per-arm labels.
pub fn emit_br_table<W, E>(
    w: &mut W,
    selector_reg: u8,
    targets: &[u32],
    default: u32,
    label_counter: &mut usize,
    mut resolve: impl FnMut(&mut W, u32) -> Result<(), E>,
) -> Result<(), E>
where
    W: BlitzWriter<Error = E>,
{
    // Pre-allocate one label per arm.
    let arm_label_base = *label_counter;
    *label_counter += targets.len();

    // Decrement approach: branch if zero, then decrement, repeat.
    for (arm_idx, _) in targets.iter().enumerate() {
        w.branch_zero_label(selector_reg, arm_label_base + arm_idx)?;
        if arm_idx + 1 < targets.len() {
            w.reg_decrement(selector_reg)?;
        }
    }

    // No arm matched: resolve the default.
    resolve(w, default)?;

    // Emit each arm: place label then resolve the arm's target.
    for (arm_idx, &target) in targets.iter().enumerate() {
        w.place_label(arm_label_base + arm_idx)?;
        resolve(w, target)?;
    }
    Ok(())
}
