# Known gap: `__wasm_exn_propagate` is referenced but never implemented

**Status:** confirmed missing, blocks any cross-function WASM exception path on
all three NaiveAbi native targets (x86-64, AArch64, RISC-V 64).

## What's missing

`docs/abi.md` ("NaiveAbi native backends", `throw` steps 3-6) documents the intended
algorithm for `throw`: walk the compile-time CTX stack for a matching `TryTable`
frame in the *current* function, and if none is found, fall through to
**cross-function propagation** — load the caller's saved CTX and keep walking, finally
calling `__wasm_unhandled_exception` if the root frame is reached with no handler.

The per-function, compile-time part is implemented (`if_stack` scan in
`Instruction::Throw`'s codegen). The **cross-function part is not**:

- `crates/blitz-x86-64/src/naive.rs` (~1330, ~1359), `crates/blitz-aarch64/src/naive.rs`
  (~911, ~935), and `crates/blitz-riscv64/src/naive.rs` (~1472, ~1613) all emit a jump
  to an `External` label named `__wasm_exn_propagate` whenever `throw` (or an
  uncaught-in-this-function `try_table` exit) can't find a handler in the current
  function's own `if_stack`.
- That symbol is **never defined** anywhere in this repo, nor in any consumer repo
  (checked `speet`, which is the only current embedder). It is not emitted by any
  codegen pass, and it is not provided by any runtime shim (contrast with
  `speet-rt/runtime.c`, which *does* define `__wasm_mem`/`__wasm_memory_grow` for the
  symbols those backends expect the embedder to supply).
- `__wasm_unhandled_exception` (the final "no handler anywhere" trap mentioned in step 6
  of the doc) isn't referenced by any codegen path either — there's no fallback wired up
  at all for that case yet.
- SysVAbi exception handling is separately, explicitly deferred
  (`todo!("SysVAbi exception handling requires platform unwinder — deferred")`) — that
  gap is already documented in `docs/abi.md` and is not the subject of this note.

## Why this matters for speculative calls

`speet`'s speculative-call lowering (`crates/helper/yecta/SPECULATIVE_CALLS.md` in the
`speet` repo) relies *structurally* on cross-function exception propagation: the
`try_table`/catch is established by the **caller**'s generated function around a real
(non-tail) `call`, while the matching `throw` — fired when a callee's actual return
doesn't match the speculated `expected_ra` — executes inside the **callee**'s generated
function (or several `return_call` hops further down the callee's own chain). The
callee's own `if_stack`, built while compiling just that function, has no knowledge of
the caller's `try_table`, so every such escape *always* takes the cross-function path.

Concretely: **any guest call whose actual return doesn't match the statically-assumed
`expected_ra` will jump to an undefined `__wasm_exn_propagate` symbol** when compiled
through wasm-blitz's NaiveAbi backends — this fails to link (undefined external) or, if
some default/weak resolution papers over it, traps or corrupts state at runtime. This
is not a rare edge case for any e2e test that turns speculative calls on: it's the path
taken by every non-ABI-compliant return (longjmp, stack manipulation, or simply any
guest control flow the static `expected_ra` guess gets wrong).

## What needs to happen before speculative-call e2e tests can pass here

Implement `__wasm_exn_propagate` (and ideally `__wasm_unhandled_exception`) per the
algorithm already specified in `docs/abi.md` steps 3-6: walking from the current CTX
frame into the **caller's saved CTX** (stored at a fixed offset in the current frame
base, per the doc) and re-scanning for a `TRYTABLE_SENTINEL` frame, repeating until a
handler is found or the root frame is reached. This is shared, target-specific runtime
logic (one implementation per ISA, matching each backend's CTX frame layout) — it
belongs either as a synthesized stub emitted once per compiled module (alongside
existing module-level scaffolding) or as a hand-written asm routine supplied by the
embedding runtime, mirroring how `speet-rt/runtime.c` supplies the memory-growth shim
today.

Until this lands, any new e2e test that exercises speculative calls end-to-end through
wasm-blitz's native targets is expected to fail — that is the point of adding such a
test now (it documents and pins down this gap rather than silently skipping it).
