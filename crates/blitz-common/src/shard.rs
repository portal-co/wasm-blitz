//! Backend sharding: splitting a compiled WASM module across multiple output writers.
//!
//! # Overview
//!
//! A *shard* is one output file/buffer produced by the compiler. Instead of the
//! default single-writer pipeline, sharding routes each function's emitted code
//! to one of N writers based on a [`ShardMap`].
//!
//! Cross-shard calls differ by backend:
//!
//! - **Text backends (C, JS/ESM):** forward declarations / ESM `import` statements;
//!   the linker or JS module system resolves the symbols at load time.
//! - **Native backends (x86-64, AArch64, RISC-V 64, NaiveAbi and SysVAbi only):**
//!   indirect calls through a flat function-pointer table pointed to by the Static
//!   Context Register (SCR, see `docs/second-context-register.md`).
//!
//! # Usage
//!
//! ```ignore
//! let shard_map = RoundRobinShardMap { n: 2 };
//! let mut coord = ShardCoordinator::new(&shard_map, imports_len);
//! let mut writers = VecWriterSet(vec![String::new(); 2]);
//!
//! for op in ops {
//!     if let MachOperator::StartFn { id, .. } = &op { coord.on_start_fn(*id); }
//!     Backend::on_mach(writers.get_mut(coord.current_shard), ..., &op, ...)?;
//! }
//! ```

use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// ShardMap
// ---------------------------------------------------------------------------

/// Maps a WASM-space function index (imports + local functions) to a shard index.
///
/// Implement this trait to control how functions are distributed across shards.
pub trait ShardMap {
    /// Returns the shard index for the given WASM-space function index.
    ///
    /// `fn_idx` includes imports: slot 0 is the first import (or first local
    /// function if there are no imports).
    fn shard_for(&self, fn_idx: u32) -> usize;

    /// Total number of shards.
    fn shard_count(&self) -> usize;
}

/// A no-op shard map: every function goes to shard 0.
///
/// Use this as a compatibility shim when sharding is disabled — the
/// coordinator and helpers still work but produce a single output.
pub struct SingleShard;

impl ShardMap for SingleShard {
    #[inline]
    fn shard_for(&self, _fn_idx: u32) -> usize { 0 }
    #[inline]
    fn shard_count(&self) -> usize { 1 }
}

/// Round-robin shard map: `fn_idx % n`.
pub struct RoundRobinShardMap {
    pub n: usize,
}

impl ShardMap for RoundRobinShardMap {
    #[inline]
    fn shard_for(&self, fn_idx: u32) -> usize {
        (fn_idx as usize) % self.n
    }
    #[inline]
    fn shard_count(&self) -> usize { self.n }
}

/// Explicit per-function shard assignment.
///
/// `map[fn_idx]` gives the shard for that WASM-space function index.
/// Panics on out-of-bounds access; the caller must ensure the vec covers all
/// functions.
pub struct ExplicitShardMap(pub Vec<usize>);

impl ShardMap for ExplicitShardMap {
    #[inline]
    fn shard_for(&self, fn_idx: u32) -> usize {
        self.0[fn_idx as usize]
    }
    fn shard_count(&self) -> usize {
        self.0.iter().copied().max().map_or(1, |m| m + 1)
    }
}

// ---------------------------------------------------------------------------
// CallTarget
// ---------------------------------------------------------------------------

/// How a native-backend call instruction should be compiled in a sharded module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallTarget {
    /// Callee is in the same shard — emit a direct label call (unchanged).
    Local,
    /// Callee is in a different shard — load its pointer from the SCR table.
    ///
    /// `table_slot` is the WASM-space function index used as the table offset:
    /// `SCR + table_slot * 8`.
    CrossShard { table_slot: u32 },
    /// Callee is an import — emit an external label call (unchanged).
    Import,
}

/// Classify a call to `callee_fn` (WASM-space) from a function in `caller_shard`.
pub fn classify_call(
    map: &dyn ShardMap,
    caller_shard: usize,
    callee_fn: u32,
    imports_len: u32,
) -> CallTarget {
    if callee_fn < imports_len {
        CallTarget::Import
    } else {
        let callee_shard = map.shard_for(callee_fn);
        if callee_shard == caller_shard {
            CallTarget::Local
        } else {
            CallTarget::CrossShard { table_slot: callee_fn }
        }
    }
}

// ---------------------------------------------------------------------------
// WriterSet
// ---------------------------------------------------------------------------

/// A collection of per-shard writers.
pub trait WriterSet {
    type Writer;

    fn get_mut(&mut self, shard: usize) -> &mut Self::Writer;
    fn shard_count(&self) -> usize;
}

/// A `Vec`-backed writer set.
pub struct VecWriterSet<W>(pub Vec<W>);

impl<W> WriterSet for VecWriterSet<W> {
    type Writer = W;

    #[inline]
    fn get_mut(&mut self, shard: usize) -> &mut W { &mut self.0[shard] }
    #[inline]
    fn shard_count(&self) -> usize { self.0.len() }
}

/// A single-writer wrapper that implements `WriterSet` (always returns shard 0).
pub struct SingleWriterSet<'a, W>(pub &'a mut W);

impl<'a, W> WriterSet for SingleWriterSet<'a, W> {
    type Writer = W;

    #[inline]
    fn get_mut(&mut self, _shard: usize) -> &mut W { self.0 }
    #[inline]
    fn shard_count(&self) -> usize { 1 }
}

// ---------------------------------------------------------------------------
// ShardCoordinator
// ---------------------------------------------------------------------------

/// Tracks which shard is currently being compiled and classifies call targets.
///
/// The compile loop calls [`on_start_fn`] when it sees a `StartFn` operator;
/// thereafter [`current_shard`] gives the writer to use for all subsequent
/// operators in that function.
pub struct ShardCoordinator<'a, S: ShardMap> {
    pub map: &'a S,
    pub imports_len: u32,
    /// Shard index for the function currently being compiled.
    pub current_shard: usize,
    /// Local function index of the function currently being compiled
    /// (`wasm_fn_idx - imports_len`).
    pub current_local_fn: u32,
}

impl<'a, S: ShardMap> ShardCoordinator<'a, S> {
    pub fn new(map: &'a S, imports_len: u32) -> Self {
        Self { map, imports_len, current_shard: 0, current_local_fn: 0 }
    }

    /// Update state when a `StartFn { id, .. }` operator is seen.
    ///
    /// Returns the shard index that should receive this function's output.
    pub fn on_start_fn(&mut self, fn_idx: u32) -> usize {
        self.current_local_fn = fn_idx.saturating_sub(self.imports_len);
        self.current_shard = self.map.shard_for(fn_idx);
        self.current_shard
    }

    /// Classify a call to `callee_fn` (WASM-space function index).
    pub fn call_target(&self, callee_fn: u32) -> CallTarget {
        classify_call(self.map, self.current_shard, callee_fn, self.imports_len)
    }
}

// ---------------------------------------------------------------------------
// ShardConfig / SecondCtxConfig
// ---------------------------------------------------------------------------

/// Configuration for sharded native-backend compilation.
///
/// When this is `Some` in `emit_prologue` / `emit_call`, the backend:
///
/// 1. Saves the Static Context Register (SCR) in the prologue.
/// 2. Emits indirect loads for cross-shard calls via `[SCR + table_slot * 8]`.
/// 3. Restores SCR in the epilogue.
///
/// The SCR must be pre-populated by the caller before the first entry into
/// sharded code.  See `docs/second-context-register.md` for the full design.
#[derive(Clone, Copy, Debug)]
pub struct ShardConfig {
    /// Number of imported functions (table slots 0..imports_len are for imports,
    /// though typically only local function slots are used for cross-shard calls).
    pub imports_len: u32,
    /// Total number of WASM-space functions (imports + locals).  Used for
    /// bounds documentation; not enforced at runtime in generated code.
    pub total_fns: u32,
}

/// Signals to the ABI prologue/epilogue that the Static Context Register (SCR)
/// must be saved and restored.
///
/// This is a superset of [`ShardConfig`]: any feature that uses the SCR should
/// produce a `SecondCtxConfig`.  When multiple features are active, SCR will
/// point to a composite struct (see `docs/second-context-register.md`).
///
/// When `None`, no SCR save/restore is emitted; the register is invisible to
/// the compiled function.
#[derive(Clone, Copy, Debug)]
pub struct SecondCtxConfig {
    pub shard: ShardConfig,
}

impl SecondCtxConfig {
    pub fn for_shard(cfg: ShardConfig) -> Self {
        Self { shard: cfg }
    }
}
