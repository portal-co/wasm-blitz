# `__wasm_exn_propagate` — cross-function exception propagation

**Status:** implemented (software EH stack). Phase 1 of the shared-gaps plan:
working link + unhandled-trap + full `take_handler` dispatch are all in place on
all three NaiveAbi native targets (x86-64, AArch64, RISC-V 64).

## What this solves

`docs/abi.md` ("NaiveAbi native backends") documents the two-tier `throw`
algorithm: first scan the compile-time CTX `if_stack` for a matching
`TryTable` frame in the *current* function (unchanged, still CTX-based); if
none is found, fall through to **cross-function propagation**.

Previously, `crates/blitz-x86-64/src/naive.rs`, `crates/blitz-aarch64/src/naive.rs`,
and `crates/blitz-riscv64/src/naive.rs` all emitted a jump to an `External`
label named `__wasm_exn_propagate` whenever `throw` (or an uncaught-in-this-
function `try_table` dispatch) couldn't find a handler in the current
function's own `if_stack` — but that symbol was never defined anywhere,
including in `speet` (the only current embedder). Compiling any code path
that took the cross-function throw route failed to link.

## The fix: a global software EH stack

Rather than walking a per-ISA CTX/frame chain across function boundaries
(which would need every native frame — including ones from unrelated,
non-blitz-generated callers — to cooperate), cross-function propagation now
goes through one **global, ISA-agnostic software stack** of
`{dispatch_addr, saved_sp}` frames, owned by the embedding runtime:

- **`try_table` entry**: in addition to the existing CTX push (still used for
  the same-function fast path), calls `__wasm_eh_push(dispatch_addr, sp)` —
  a real ABI call, marshalling into the two argument registers per target.
- **`try_table` normal (non-throwing) exit**: calls `__wasm_eh_pop()` to
  discard its frame.
- **Local `throw`** (handler found in the current function's `if_stack`):
  calls `__wasm_eh_pop()` itself, immediately before jumping to its own
  dispatch stub — the dispatch stub is never re-entered by
  `__wasm_exn_propagate` in this case, so exactly one of "dispatch stub start"
  / "`Throw`'s local-jump site" pops the frame that `try_table` pushed.
- **Unmatched catch / no local handler anywhere in `if_stack`**: instead of a
  `call`, a bare **jump** to `__wasm_exn_propagate` (never returns to the
  throw site).
- **`__wasm_exn_propagate`** (defined once, in `speet-rt`, see below): calls
  `__wasm_eh_take(&dispatch_out, &sp_out)`. If it returns 0 (stack empty),
  falls through to `__wasm_unhandled_exception` (traps). Otherwise, overwrites
  the hardware SP with `sp_out` and jumps to `dispatch_out` — reusing the
  *caller's* (or some ancestor's) dispatch stub, which re-runs that frame's
  own `Catch::One`/`Catch::All` matching against the still-live tag/value
  registers.

This is implemented per the two-file split anticipated in the original gap
note — one implementation per ISA for the push/pop/jump call sites (`naive.rs`
in each `blitz-*` backend), plus one shared runtime routine supplied by the
embedder:

- `crates/os/speet-rt/src/exn.rs` (`speet` repo) generates a small C + inline-
  asm translation unit defining `__wasm_eh_push`, `__wasm_eh_pop`,
  `__wasm_eh_take`, `__wasm_unhandled_exception`, and the
  `__wasm_exn_propagate` trampoline itself (per-target raw assembly, since it
  restores SP and performs a bare indirect jump — not expressible as a
  normal-returning C function). Linked in by `speet-runtime`'s
  `crates/os/speet-runtime/src/link.rs`.

## Why this matters for speculative calls

`speet`'s speculative-call lowering (`crates/helper/yecta/SPECULATIVE_CALLS.md`
in the `speet` repo) relies *structurally* on cross-function exception
propagation: the `try_table`/catch is established by the **caller**'s
generated function around a real (non-tail) `call`, while the matching
`throw` — fired when a callee's actual return doesn't match the speculated
`expected_ra` — executes inside the **callee**'s generated function (or
several `return_call` hops further down the callee's own chain). The callee's
own `if_stack`, built while compiling just that function, has no knowledge of
the caller's `try_table`, so every such escape *always* takes the
cross-function path described above — this is exactly the path this fix
makes work.

## Remaining follow-up (not blocking, out of scope for Phase 1)

- `blitz-tests` execution harnesses that link and run real native code
  (rather than only asserting on emitted assembly text) need the four symbols
  above available at link time; tests that don't pull in `speet-rt` should add
  minimal stubs (no-op `push`/`pop`, `take` always returning 0, `unhandled`
  trapping) if/when they start exercising cross-function throws under
  Unicorn/native execution.
- SysVAbi **codegen** still falls through to this NaiveAbi software EH path
  (supported short-term). A true DWARF / `_Unwind_RaiseException` SysV unwind
  path remains deferred — see *SysVAbi exception handling* in `docs/abi.md`.
