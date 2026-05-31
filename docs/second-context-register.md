# Second Context Register (SCR)

## Overview

Each native backend reserves two special registers for runtime context:

| Register | Name | x86-64 | AArch64 | RISC-V 64 | Purpose |
|----------|------|---------|---------|-----------|---------|
| `Reg::CTX` (`Reg(255)`) | Frame / CTX | r15 | x28 | s11 | WASM control-frame stack (NaiveAbi) / trace-table pointer (JIT entry) |
| SCR | Static Context | r14 | x27 | s10 | Module-level static data, saved/restored only when a feature uses it |

`Reg::CTX` is the existing register used by NaiveAbi as a shadow stack pointer (via `xchg RSP, r15`) and by the JIT preamble as the trace-table base at function entry.

The **Static Context Register (SCR)** is the complement: a second callee-saved register that holds a pointer to static, module-level data throughout the entire function body (not just at entry). It is only saved and restored in the prologue/epilogue when at least one active feature requires it. Non-sharded, non-SCR functions leave it untouched.

## Current use: sharding (first consumer)

When sharding is enabled, SCR points to a flat array of function pointers for the entire module:

```
SCR → [fn_ptr_0, fn_ptr_1, ..., fn_ptr_N-1]   (each entry is 8 bytes / *const ())
```

- **Intra-shard calls** use direct labels as before — no table lookup.
- **Cross-shard calls** emit `mov scratch, [SCR + callee_fn_idx * 8]; call scratch`.

The caller (runtime/embedder) is responsible for populating this table and loading SCR before the first call into sharded code. SCR is callee-saved, so it propagates through the call graph automatically once set.

`ShardConfig` in `blitz-common::shard` is the type-level signal that SCR is needed:

```rust
pub struct ShardConfig {
    pub imports_len: u32,
    pub total_fns: u32,
}
```

## Future uses (planned)

The SCR design is intentionally open-ended. Future features that need module-level data throughout a function body can extend the SCR contract without adding more reserved registers:

- **JIT trace-table (planned):** Currently the trace-table pointer is only accessible at function entry via CTX. Moving it to SCR (or a composite struct behind SCR) would let specialisation checks happen mid-function.
- **Host memory mirror base:** A base pointer for sandboxed linear-memory access, avoiding a load from a global on every memory operation.
- **Thread-local storage base:** For WASM threads, a per-thread context pointer accessible in generated code.

## Composite SCR struct

When multiple features are active simultaneously, SCR points to a composite context struct whose layout is determined at compile time by the active feature set:

```rust
// Example (future): sharding + JIT
#[repr(C)]
struct CompositeCtx {
    shard_table: *const *const (),   // offset 0: cross-shard fn pointers
    trace_table: *const TraceSite,   // offset 8: JIT trace sites
}
```

Callers populate the relevant fields. Each feature accesses its own offset; unused fields are zero.

## Activation

SCR is activated by `SecondCtxConfig` in `blitz-common::shard`. The prologue:

1. Saves SCR to the native stack (push).
2. Does **not** initialise SCR — the caller has already set it.
3. Epilogue restores SCR (pop).

When `SecondCtxConfig` is `None` (no feature uses SCR), no save/restore is emitted and the register is invisible to the compiled function.

## Per-arch scratch for indirect calls

Cross-shard call sequences load the target through a scratch register before calling:

| Arch | Scratch |
|------|---------|
| x86-64 | rax (`Reg(0)`) — loaded after arg marshalling |
| AArch64 | x16 (intra-procedure-call scratch, IP0) |
| RISC-V 64 | t0 (`Reg(5)`) |
