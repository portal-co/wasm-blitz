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
//! blitz_codegen::emit_jit_preamble(&mut bw, counter_addr, spec_addr, scratch, &mut label_counter)?;
//! blitz_codegen::emit_br_table(&mut bw, selector_reg, targets, default, &mut label_counter, resolve)?;
//! ```

#![no_std]

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
    /// Used by both `emit_br_table` (decrement approach) and `emit_jit_preamble`.
    fn branch_zero_label(&mut self, reg: u8, label_idx: usize) -> Result<(), Self::Error>;

    /// Indirect branch through the value in `reg`.
    fn branch_reg(&mut self, reg: u8) -> Result<(), Self::Error>;

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
}

// ---------------------------------------------------------------------------
// Shared instruction implementations
// ---------------------------------------------------------------------------

/// Emit a JIT tracing preamble.
///
/// Increments the invocation counter at `counter_addr`, then checks the
/// specialisation function pointer at `spec_addr`. If non-null it tail-jumps
/// there; otherwise it falls through to `body_label` (placed by this function).
///
/// `scratch` is a caller-provided scratch register (by blitz register number).
/// Architectures that need a second scratch register for `inc_mem64` store it
/// in their [`BlitzWriter`] implementation — callers don't need to know.
///
/// The `label_counter` is incremented once by this function to allocate the
/// body label.
pub fn emit_jit_preamble<W: BlitzWriter>(
    w: &mut W,
    counter_addr: u64,
    spec_addr: u64,
    scratch: u8,
    label_counter: &mut usize,
) -> Result<(), W::Error> {
    let body_label = *label_counter;
    *label_counter += 1;

    // 1. Increment invocation counter (non-atomic, approximate).
    w.load_u64_imm(scratch, counter_addr)?;
    w.inc_mem64(scratch)?;

    // 2. Load specialisation fn-ptr; tail-jump if non-null.
    w.load_u64_imm(scratch, spec_addr)?;
    w.load_mem64(scratch, scratch)?;
    w.branch_zero_label(scratch, body_label)?;
    w.branch_reg(scratch)?;
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
