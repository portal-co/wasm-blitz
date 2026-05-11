//! ABI strategy trait and calling-convention marker types.
//!
//! [`BackendAbi`] is a strategy trait that abstracts over how a code-generation
//! backend handles function prologues, local variable access, calls, and returns.
//! Concrete implementations live in each backend crate:
//!
//! - `blitz-x86-64`: `impl BackendAbi<W, Ctx> for NaiveAbi` and `for SysVAbi`
//! - `blitz-aarch64`: same pattern
//! - `blitz-riscv64`: same pattern
//!
//! # Design
//!
//! All methods are free functions (no `&self` receiver) — the two ZST marker types
//! [`NaiveAbi`] and [`SysVAbi`] carry no state. The writer `W` and its context
//! `Context` are passed in on every call, keeping the API composable without
//! requiring trait objects.
//!
//! # Why `emit_call` takes `sigs` / `fsigs`
//!
//! The existing `sysv_handle_op` / `sysv_handle_insn` helpers do not receive the
//! full function-type table, so they cannot perform register-level argument
//! marshalling for `Call` instructions.  [`BackendAbi::emit_call`] takes the full
//! type information and is the correct replacement call-site.

use crate::ops::FnData;
use crate::wasm_encoder::FuncType;

// ---------------------------------------------------------------------------
// Marker ZSTs
// ---------------------------------------------------------------------------

/// Marker for the blitz-internal "naive" stack-based calling convention.
///
/// In this convention arguments and return values are passed on the WASM
/// operand stack.  No platform ABI register marshalling is performed.
pub struct NaiveAbi;

/// Marker for the platform System V (POSIX) calling convention.
///
/// - x86-64: SysV AMD64 (RDI/RSI/RDX/RCX/R8/R9, return in RAX[+RDX])
/// - AArch64: AAPCS64 (X0–X7, return in X0[+X1])
/// - RISC-V 64: RISC-V psABI LP64 (A0–A7, return in A0[+A1])
pub struct SysVAbi;

// ---------------------------------------------------------------------------
// BackendAbi trait
// ---------------------------------------------------------------------------

/// Strategy trait for ABI-specific code generation.
///
/// Implementors decide how function boundaries, local variable access, calls,
/// and returns are compiled.  The trait is generic over the writer `W` and its
/// context `Context` so each backend can impose its own writer-trait bounds in
/// the `impl` block without polluting the common definition.
///
/// All methods are free (no `&self`); the marker ZST is used only as a type
/// discriminant via the `impl BackendAbi<W,Ctx> for SysVAbi` form.
pub trait BackendAbi<W: ?Sized, Context> {
    /// Error type forwarded from the underlying writer.
    type Error;

    /// Per-function mutable state managed by the ABI implementation.
    ///
    /// Must implement [`Default`] so callers can create fresh instances without
    /// knowing the concrete state layout.
    type State: Default;

    /// Architecture descriptor (e.g. `X64Arch`, `AArch64Arch`, `RiscV64Arch`).
    ///
    /// Must be `Copy + Default` so callers can construct it cheaply.
    type Arch: Copy + Default;

    // ---- function boundary -------------------------------------------------

    /// Emit the function prologue.
    ///
    /// Called when a [`crate::ops::MachOperator::StartFn`] operator is seen.
    /// Sets up the calling-convention frame and emits the function label.
    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &mut Self::State,
        id: u32,
        data: &FnData,
    ) -> Result<(), Self::Error>;

    /// Emit initialisation for one new local variable slot.
    ///
    /// Called once per slot for each [`crate::ops::MachOperator::Local`] count.
    /// `state` must have been set up by a prior [`Self::emit_prologue`] call.
    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &mut Self::State,
    ) -> Result<(), Self::Error>;

    /// Emit the start-of-body code (after all locals have been declared).
    ///
    /// Called when a [`crate::ops::MachOperator::StartBody`] operator is seen.
    fn emit_start_body(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &mut Self::State,
    ) -> Result<(), Self::Error>;

    // ---- local variable access ---------------------------------------------

    /// Emit `local.get n` — push local `n` onto the WASM operand stack.
    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &Self::State,
        n: u32,
    ) -> Result<(), Self::Error>;

    /// Emit `local.set n` — pop the operand stack into local `n`.
    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &Self::State,
        n: u32,
    ) -> Result<(), Self::Error>;

    /// Emit `local.tee n` — copy the top of the operand stack into local `n`
    /// without consuming it.
    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &Self::State,
        n: u32,
    ) -> Result<(), Self::Error>;

    // ---- call / return -----------------------------------------------------

    /// Emit a direct `call fn_idx` with full ABI register marshalling.
    ///
    /// `sigs` is the complete function-type table; `fsigs` maps each WASM
    /// function index to its type index.  The implementation uses these to
    /// determine how many arguments to pop and how many results to push.
    ///
    /// `func_imports` maps import indices to `(module, name)` pairs so the
    /// implementation can emit the correct external symbol for imports versus
    /// a local label for internal functions.
    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &Self::State,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        sigs: &[FuncType],
        fsigs: &[u32],
    ) -> Result<(), Self::Error>;

    /// Emit a `return` instruction, including the ABI-specific epilogue.
    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: Self::Arch,
        state: &Self::State,
    ) -> Result<(), Self::Error>;
}
