//! C backend sharding support.
//!
//! Provides helpers to emit cross-shard `extern` declarations so that each
//! shard file can call functions defined in other shards without linker errors.
//! The C linker resolves the symbols when the shards are compiled together.

use core::fmt::Write;
use portal_solutions_blitz_common::{
    shard::ShardMap,
    wasm_encoder::FuncType,
};

/// Emit `extern` declarations for all local functions **not** in `my_shard`.
///
/// Call this once per shard, before emitting that shard's function bodies, so
/// the C compiler knows the signatures of cross-shard callees.
///
/// Each declaration has the form:
///
/// ```c
/// static const struct { int params; int rets; } __sig_N = { .params=P, .rets=R };
/// extern uint64_t* fn_N(uint64_t* restrict);
/// ```
///
/// `imports_len` is the number of imported functions; local functions start at
/// WASM index `imports_len`.
pub fn c_emit_cross_shard_decls<W: Write>(
    w: &mut W,
    my_shard: usize,
    sigs: &[FuncType],
    fsigs: &[u32],
    imports_len: u32,
    shard_map: &dyn ShardMap,
) -> core::fmt::Result {
    let local_fn_count = fsigs.len().saturating_sub(imports_len as usize);
    for local_idx in 0..local_fn_count {
        let wasm_idx = imports_len as usize + local_idx;
        if shard_map.shard_for(wasm_idx as u32) == my_shard {
            continue; // defined in this shard, no extern needed
        }
        let sig = &sigs[fsigs[wasm_idx] as usize];
        write!(
            w,
            "static const struct{{int params;int rets;}}__sig_{wasm_idx}={{.params={0},.rets={1}}};",
            sig.params().len(),
            sig.results().len()
        )?;
        write!(w, "extern uint64_t*fn_{wasm_idx}(uint64_t*restrict);\n")?;
    }
    Ok(())
}
