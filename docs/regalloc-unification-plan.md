# Unify regalloc-based backends behind a shared WASM frontend

> Shadow plan doc — mirrors the plan approved 2026-07-11. Update this file (not
> just chat/session state) as increments land or the design changes, so any
> agent/session can pick up where a previous one left off.

## Context

`wasm-blitz` compiles WASM to native code for three ISAs (x86-64, AArch64, RISC-V). Each has grown its own copy of the WASM-instruction-to-machine-code translation logic, but the copies aren't just differently-encoded versions of the same algorithm — they've diverged along **two independent axes**:

- **Control-flow model**: how `Block`/`Loop`/`If`/`Else`/`End`/`Br`/`BrIf`/`BrTable` get resolved to concrete labels/branches.
- **Data-flow model**: how WASM's operand stack (pushes/pops) and locals get realized in machine state — a real memory stack for every value, vs. a register allocator that keeps values in physical registers and only spills to memory when it runs out.

Concretely, per backend (as of the start of this effort):

| Backend | Control flow | Data flow |
|---|---|---|
| `blitz-x86-64/src/naive.rs` | runtime CTX-stack trick (`xchg RSP,CTX` + push saved-RSP/return-label) | pure stack (push/pop every value) |
| `blitz-x86-64/src/fast.rs` | same runtime CTX-stack trick, copy-pasted from naive | regalloc (`portal_solutions_asm_regalloc::RegAlloc`) + `StackManager` spill fallback — but incomplete (catch-all `_ => {}` no-op for most instructions) |
| `blitz-aarch64/src/naive.rs` | compile-time label stack (`Endable::Block{end_lbl}` etc., no runtime CTX manipulation) | pure stack (`wasm_push`/`wasm_pop`) — zero regalloc usage anywhere in the crate |
| `blitz-riscv64/src/naive.rs` | compile-time label stack (`Endable::Block{idx}` etc. — same shape as aarch64's) | regalloc + spill, most mature/complete instance |

RISC-V's `naive.rs` and AArch64's `naive.rs` already agree on the control-flow model. RISC-V's `naive.rs` and x86-64's `fast.rs` already agree on the data-flow model. x86-64's `naive.rs` is the outlier on both axes.

**Target end-state**: every non-legacy backend uses compile-time-label-stack control flow + regalloc-backed data flow (RISC-V's `naive.rs` is effectively the reference implementation already). x86-64's current runtime-stack-based `naive.rs` is explicitly **not** part of this unification — it stays independent and is the actual long-term deprecation target once the regalloc-based path is mature enough to replace it.

The asm-arch layer (`/Users/g/Code-local/portal-hot/asm-arch`, resolved locally via workspace `[patch]`) already provides symmetric regalloc primitives for all three ISAs — `RegKind` (byte-for-byte identical `Int=0`/`Float=1` enum in `asm-x86-64`, `asm-riscv64`, and **`asm-aarch64`**), `init_regalloc::<N>()`, and `process_cmd()`. AArch64 already has these upstream; `blitz-aarch64` simply never wired them up. This meaningfully de-risks the AArch64 increment below — it's wiring existing primitives, not building regalloc support from scratch.

## Design

Two shared pieces in `crates/blitz-codegen` (the crate already holds one instance of this exact pattern: `emit_probe_site`/`emit_br_table` in `src/lib.rs`).

### 1. Generic regalloc adapter — **done**, see `crates/blitz-codegen/src/regalloc_adapter.rs`

`RegAlloc<K, N, I>` (from `portal_solutions_asm_regalloc`) is generic over an arch's `RegKind` and a `Frames` wrapper, but `Frames`/`Index`/`IndexMut`/`Length` were hand-copy-pasted per arch (`blitz-riscv64/src/naive.rs`, `blitz-x86-64/src/fast.rs`). Replaced with one generic `Frames<K, N>`, bounded only on what `RegAlloc` itself already requires of `K` (`Clone + Eq + TryFrom<usize>`) — no per-arch trait impl needed, avoiding orphan-rule issues since `Frames` is the local (blitz-codegen-owned) type.

`blitz-riscv64::naive::Frames` now `pub use`s the generic version (a `pub use`, not `pub type`, because tuple-struct constructor syntax `Frames(...)` requires the name to resolve in the *value* namespace too, which a type alias doesn't provide).

### 2. Shared WASM frontend — **not started**

A minimal backend trait (in the spirit of `BlitzWriter`) exposing what a regalloc-backed backend needs:
- Acquiring/releasing a physical register for a pushed/popped value (`push_value`/`push_local`/`pop_value`/`pop_local` → reg + spill `cmds` to emit)
- Instruction-selection callback for binary/unary ops on already-allocated registers
- Label/branch primitives for the compile-time label-stack control-flow model (`open_block`/`open_loop`/`open_if`/`close`/`branch_depth`), mirroring `Endable`/`if_stack` from `blitz-riscv64/src/naive.rs` and `blitz-aarch64/src/naive.rs` (already near-identical shape).

Free functions implementing, once: the const/arithmetic/comparison/local-access dispatch (regalloc pop/push sequencing + spill emission, generic over instruction selection), and structured control-flow resolution (`Block`/`Loop`/`If`/`Else`/`End`/`Br`/`BrIf`/`BrTable` → label placement/branches).

Each arch's existing `codegen.rs` `BlitzW`-style wrapper gains an impl of the new trait, supplying only genuinely arch-specific pieces (which physical registers exist, how to encode ops on allocated registers, how to encode a conditional branch).

## Sequencing (risk-ordered, not a single PR)

1. **Generic regalloc adapter** — done (commit `93a1ce0`).
2. **Port `blitz-riscv64/src/naive.rs` onto the new shared frontend** (once built) — it already implements the target model on both axes, so this should be a close-to-mechanical extraction with no intended behavior change. Cheapest way to prove the abstraction before other backends depend on it.
3. **Complete x86-64's `fast.rs`** — migrate its control flow off the copy-pasted runtime-CTX-stack trick onto the shared compile-time-label-stack frontend, replace its ad-hoc per-arm regalloc dance with the shared dispatch, and fill in the `_ => {}` gap (most instructions are currently silently dropped). `fast.rs` has **zero existing test coverage** (`grep -c "fast::" crates/blitz-tests/tests/e2e.rs` → 0) — this increment must add new tests, not just rerun the existing suite.
4. **New AArch64 regalloc-backed backend** — net-new integration (an AArch64 `RegKind`/instruction-selection adapter for the shared frontend), reusing the already-existing upstream `asm-aarch64` regalloc primitives. Its control-flow model already matches the target, so only the data-flow half is new. Recommend landing as a new `blitz-aarch64/src/fast.rs` sibling (mirroring x86-64's naming) rather than modifying `naive.rs` in place, to keep the stable path stable while the new one is proven.

**Explicitly out of scope, indefinitely**: `blitz-x86-64/src/naive.rs` is not touched or migrated by any step above — it's the long-term deprecation target, but actually removing it is future work. `sysv.rs`/`lfi.rs` per arch, float ops, and memory load/store instruction selection are left for follow-up once the integer/control-flow core is proven.

## Verification

`crates/blitz-tests/tests/e2e.rs` (~6700 lines) executes emitted code via `unicorn-engine` across x86-64/aarch64/riscv64, plus a clang-assembled textual-asm path. Baseline before this effort started: `cargo test -p portal-solutions-blitz-tests` → 361 passed, 0 failed. Run `cargo build --workspace` and the full test suite before/after each step above; diff pass/fail sets.
