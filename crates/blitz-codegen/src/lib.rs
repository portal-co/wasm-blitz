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

    // ---- CTX-relative trace-table access (JIT preamble) ------------------

    /// Load the runtime-provided trace-table base pointer into `dest`.
    ///
    /// The base is stored by the runtime at a fixed, CTX-relative slot
    /// `base_off` (see `TracingConfig::table_base_off`).  This is how the
    /// preamble reaches its [`TraceSite`](../portal_solutions_blitz_common/ops/struct.TraceSite.html)
    /// table without any absolute address being baked into the code.
    fn load_trace_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error>;

    /// Atomicity-free increment of the 64-bit value at `[ptr_reg + disp]`.
    fn inc_mem64_disp(&mut self, ptr_reg: u8, disp: i32) -> Result<(), Self::Error>;

    /// Load the 64-bit value at `[src + disp]` into `dest`.
    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error>;
}

/// Size in bytes of one `TraceSite` entry (`counter: u64` + `specialization: ptr`).
pub const TRACE_SITE_SIZE: i32 = 16;
/// Byte offset of the `specialization` pointer within a `TraceSite`.
pub const TRACE_SITE_SPEC_OFF: i32 = 8;

// ---------------------------------------------------------------------------
// Shared instruction implementations
// ---------------------------------------------------------------------------

/// Emit a JIT tracing preamble for one trace site.
///
/// Reaches the runtime [`TraceSite`] table through a CTX-relative base
/// (`base_off`, see `TracingConfig::table_base_off`), indexes it by `site_id`,
/// increments that site's invocation counter, then checks its specialisation
/// code pointer.  If non-null it tail-jumps there (with the operand-stack /
/// CTX-frame state intact for this site); otherwise it falls through to
/// `body_label` (placed by this function).
///
/// No absolute address is baked into the generated code — only the structural
/// `base_off` and `site_id` — so compilation is independent of the runtime
/// address space.
///
/// `scratch` is a caller-provided scratch register (by blitz register number).
/// Architectures that need a second scratch register for `inc_mem64_disp` store
/// it in their [`BlitzWriter`] implementation — callers don't need to know.
///
/// The `label_counter` is incremented once by this function to allocate the
/// body label.
pub fn emit_jit_preamble<W: BlitzWriter>(
    w: &mut W,
    base_off: i32,
    site_id: u32,
    scratch: u8,
    label_counter: &mut usize,
) -> Result<(), W::Error> {
    let body_label = *label_counter;
    *label_counter += 1;

    let site_off = site_id as i32 * TRACE_SITE_SIZE;

    // 1. Load runtime trace-table base (no baked address).
    w.load_trace_base(scratch, base_off)?;

    // 2. Increment this site's invocation counter (non-atomic, approximate).
    w.inc_mem64_disp(scratch, site_off)?;

    // 3. Load specialisation code-ptr; tail-jump if non-null.
    w.load_mem64_disp(scratch, scratch, site_off + TRACE_SITE_SPEC_OFF)?;
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
