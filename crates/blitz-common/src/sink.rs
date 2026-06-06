//! Bridge types connecting `wax-core` instruction sinks to `asm-arch` writers.
//!
//! [`WaxHandle`] is used as the `Context` type parameter in wax-core traits;
//! it bundles the asm-arch writer together with its context so any sink can
//! drive emission through a single `&mut Context` reference.
//!
//! [`WasmSink`] holds per-function WASM compilation state and provides
//! lifecycle helpers that delegate to [`crate::abi::BackendAbi`].  Each
//! backend crate (`blitz-x86-64`, `blitz-riscv64`, `blitz-aarch64`) adds
//! `InstructionSink` / `OperatorSink` impls for its own `WriterExt` and
//! `State` types.

use crate::abi::BackendAbi;
use crate::ops::FnData;
use crate::wasm_encoder::FuncType;
use alloc::{collections::BTreeMap, string::String, vec::Vec};

// ---------------------------------------------------------------------------
// WaxHandle — the wax-core Context type
// ---------------------------------------------------------------------------

/// A handle bundling an asm-arch writer with its context.
///
/// Use this as the `Context` type parameter when implementing or calling
/// wax-core traits (`InstructionSink<WaxHandle<W, AsmCtx>, E>`, etc.).
///
/// Having both `writer` and `asm_ctx` in one struct allows any sink to
/// split-borrow them:
///
/// ```text
/// ctx.writer.some_method(&mut ctx.asm_ctx, ...)
/// ```
///
/// This enables "parallel emission" — the context carries everything needed
/// to emit assembly while the sink owns only WASM-level state.
pub struct WaxHandle<W, AsmCtx> {
    /// The asm-arch writer that emits machine instructions.
    pub writer: W,
    /// The context passed through every writer method.
    pub asm_ctx: AsmCtx,
    /// Name → address registry for ambient (unrecompiled) library symbols.
    ///
    /// Populated by the caller before code generation. The addresses are used
    /// at link/load time to resolve `Ambient { name }` labels emitted by
    /// [`AmbientSink`] implementations — they are never inlined as immediate
    /// values in generated machine code.
    pub ambient_addrs: BTreeMap<String, u64>,
}

impl<W, AsmCtx> WaxHandle<W, AsmCtx> {
    pub fn new(writer: W, asm_ctx: AsmCtx) -> Self {
        Self { writer, asm_ctx, ambient_addrs: BTreeMap::new() }
    }
}

impl<W, AsmCtx> wax_core::build::AmbientInfo for WaxHandle<W, AsmCtx> {
    fn lookup_ambient_addr(&self, name: &str) -> Option<u64> {
        self.ambient_addrs.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// WasmSink — holds WASM compilation state, drives BackendAbi lifecycle
// ---------------------------------------------------------------------------

/// A sink wrapper that owns per-function WASM compilation state.
///
/// Backend crates implement `InstructionSink<WaxHandle<W, AsmCtx>, E>` and
/// `OperatorSink<WaxHandle<W, AsmCtx>, E>` on this type, routing each
/// operator/instruction through the backend's `WriterExt::handle_op`.
///
/// Function-level lifecycle (prologue, locals, body start) is driven via the
/// `start_fn`, `new_local`, and `start_body` helpers, which call through
/// `BackendAbi`.
pub struct WasmSink<State, Arch> {
    /// Per-function backend state (e.g. local counts, control-flow stack).
    pub state: State,
    /// Architecture descriptor forwarded to every writer call.
    pub arch: Arch,
    /// Import table: each entry is `(module, name)` for an imported function.
    pub func_imports: Vec<(String, String)>,
    /// Function-type table (all types in the module), used by exception dispatch.
    pub sigs: Vec<FuncType>,
    /// Exception tag table: maps tag index → function-type index in `sigs`.
    pub tags: Vec<u32>,
    /// Current function index, updated by `start_fn`.
    pub target: u32,
}

impl<State: Default, Arch: Copy + Default> WasmSink<State, Arch> {
    pub fn new(arch: Arch) -> Self {
        Self {
            state: State::default(),
            arch,
            func_imports: Vec::new(),
            sigs: Vec::new(),
            tags: Vec::new(),
            target: 0,
        }
    }
}

impl<S, Arch: Copy> WasmSink<S, Arch> {
    /// Emit the function prologue via `Abi::emit_prologue` and record `id` as
    /// the current compilation target.
    ///
    /// The lifetime `'a` is the lifetime of any references held inside `S`
    /// (e.g. a borrowed [`crate::shard::ShardMap`]).  For states that hold no
    /// references, `'a` is inferred as `'static`.
    pub fn start_fn<'a, W, AsmCtx, Abi>(
        &'a mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        id: u32,
        data: &FnData,
    ) -> Result<(), Abi::Error>
    where
        Abi: BackendAbi<W, AsmCtx, State<'a> = S, Arch = Arch> + 'a,
    {
        self.target = id;
        Abi::emit_prologue(&mut ctx.writer, &mut ctx.asm_ctx, self.arch, &mut self.state, id, data)
    }

    /// Emit initialisation for one new local variable slot.
    pub fn new_local<'a, W, AsmCtx, Abi>(
        &'a mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
    ) -> Result<(), Abi::Error>
    where
        Abi: BackendAbi<W, AsmCtx, State<'a> = S, Arch = Arch> + 'a,
    {
        Abi::emit_new_local(&mut ctx.writer, &mut ctx.asm_ctx, self.arch, &mut self.state)
    }

    /// Emit start-of-body code (called after all locals are declared).
    pub fn start_body<'a, W, AsmCtx, Abi>(
        &'a mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
    ) -> Result<(), Abi::Error>
    where
        Abi: BackendAbi<W, AsmCtx, State<'a> = S, Arch = Arch> + 'a,
    {
        Abi::emit_start_body(&mut ctx.writer, &mut ctx.asm_ctx, self.arch, &mut self.state)
    }

    /// Emit a function return.
    pub fn emit_return<'a, W, AsmCtx, Abi>(
        &'a mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
    ) -> Result<(), Abi::Error>
    where
        Abi: BackendAbi<W, AsmCtx, State<'a> = S, Arch = Arch> + 'a,
    {
        Abi::emit_return(&mut ctx.writer, &mut ctx.asm_ctx, self.arch, &self.state)
    }

    /// Borrow the stored function-type table.
    pub fn sigs_ref(&self) -> &[FuncType] {
        &self.sigs
    }

    /// Borrow the stored exception tag table (tag index → type index).
    pub fn tags_ref(&self) -> &[u32] {
        &self.tags
    }

    /// Build a temporary `Vec<(&str, &str)>` from the stored import table.
    ///
    /// This is a convenience used by the per-backend `InstructionSink` /
    /// `OperatorSink` impls to satisfy `handle_op`'s `&[(&str, &str)]` parameter
    /// without requiring the caller to manage lifetimes.
    pub fn imports_ref(&self) -> Vec<(&str, &str)> {
        self.func_imports
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect()
    }
}
