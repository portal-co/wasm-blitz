//! JS backend sharding support (ESM).
//!
//! When sharding is active, each shard is an ES module. Functions defined in a
//! shard are exported with `export { $N as $N };`; functions defined in other
//! shards are imported at the top of the shard file via ESM `import` statements.
//!
//! # Naming convention
//!
//! The JS backend names every function `$N` where N is the WASM-space function
//! index (0-based, includes imports). These are valid JS identifiers and are
//! used directly as both export and import names.
//!
//! # Example (2-shard module: fn_0 in shard 0, fn_1 in shard 1)
//!
//! shard_0.mjs:
//! ```js
//! import { $1 } from './shard_1.mjs';
//! function $0(...locals) { ... $1(...); ... }
//! export { $0 };
//! ```
//!
//! shard_1.mjs:
//! ```js
//! import { $0 } from './shard_0.mjs';
//! function $1(...locals) { ... }
//! export { $1 };
//! ```

use core::fmt::Write;
use portal_solutions_blitz_common::shard::ShardMap;

/// Emit ESM `import` statements for all local functions **not** in `my_shard`.
///
/// Call this once per shard, at the very top of the output (before any function
/// bodies), so that cross-shard functions are in scope when the shard's
/// functions are defined.
///
/// `shard_paths[k]` is the module specifier string (e.g. `"./shard_1.mjs"`) for
/// shard `k`. The slice must have at least `shard_map.shard_count()` entries.
///
/// `local_fn_count` is the number of locally-defined functions (excluding
/// imports); `imports_len` is the WASM-space index of the first local function.
pub fn js_emit_cross_shard_imports<W: Write>(
    w: &mut W,
    my_shard: usize,
    imports_len: u32,
    local_fn_count: u32,
    shard_map: &dyn ShardMap,
    shard_paths: &[&str],
) -> core::fmt::Result {
    let n_shards = shard_map.shard_count();

    // Collect, per foreign shard, which function identifiers to import.
    // We use a fixed-capacity approach: build groups per shard.
    // For no_std compat we iterate shard by shard.
    for shard_idx in 0..n_shards {
        if shard_idx == my_shard {
            continue;
        }
        let path = shard_paths[shard_idx];
        // Collect the set of $N identifiers from this shard.
        let mut first = true;
        let mut any = false;
        // Two-pass: first check if there's anything to import.
        for local in 0..local_fn_count {
            let wasm_idx = imports_len + local;
            if shard_map.shard_for(wasm_idx) == shard_idx {
                any = true;
                break;
            }
        }
        if !any { continue; }

        write!(w, "import {{")?;
        for local in 0..local_fn_count {
            let wasm_idx = imports_len + local;
            if shard_map.shard_for(wasm_idx) != shard_idx {
                continue;
            }
            if !first { write!(w, ",")?; }
            write!(w, "${wasm_idx}")?;
            first = false;
        }
        write!(w, "}} from '{path}';\n")?;
    }
    Ok(())
}

/// Emit ESM `export` statements for all local functions in `my_shard`.
///
/// Complements [`js_emit_cross_shard_imports`]: this makes the shard's own
/// functions visible to other shards that import them.
///
/// Call this once per shard, after all function bodies have been emitted.
pub fn js_emit_shard_exports<W: Write>(
    w: &mut W,
    my_shard: usize,
    imports_len: u32,
    local_fn_count: u32,
    shard_map: &dyn ShardMap,
) -> core::fmt::Result {
    let mut first = true;
    let mut any = false;
    for local in 0..local_fn_count {
        if shard_map.shard_for(imports_len + local) == my_shard {
            any = true;
            break;
        }
    }
    if !any { return Ok(()); }

    write!(w, "export {{")?;
    for local in 0..local_fn_count {
        let wasm_idx = imports_len + local;
        if shard_map.shard_for(wasm_idx) != my_shard {
            continue;
        }
        if !first { write!(w, ",")?; }
        write!(w, "${wasm_idx}")?;
        first = false;
    }
    write!(w, "}};\n")?;
    Ok(())
}
