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

### 2. Shared WASM frontend

Split into two independent pieces (control flow and data flow don't need the
same trait — see `handle_op_`'s `flush`-before-jump calls, the only coupling
point, which stays a plain sequential call from each match arm rather than
forcing one mega-trait):

**Data flow — done**, see `crates/blitz-codegen/src/regalloc_frontend.rs`.
`RegAllocWriter<K, N>` trait (`regalloc_mut`, `init_regalloc`, `emit_regalloc_cmds`
— the three genuinely arch-specific pieces: reserved-register layout,
`process_cmd` plus any extra state like x86's `StackManager`) plus free
functions `push_const`, `push_local`, `pop_to_local`, `binop`, `compare`
covering the "pop → emit spill/reload cmds → op → push → emit spill/reload
cmds" shape (`compare` differs from `binop` only in allocating a *new* dest
register rather than reusing one operand's in place — comparisons need a
register distinct from either operand for their branch+`li 0`/`li 1`
sequence). Instruction selection (e.g. `add` vs `lea`+`not`-for-subtract, or
which `ConditionCode`/operand order a comparison uses) stays a
caller-supplied closure over allocated register numbers — that's where
backends legitimately differ, not something to force-unify.

**Control flow — done**, see `crates/blitz-codegen/src/control_flow.rs`. A
`Frame` enum (`Block{end}`/`Loop{head}`/`If{then,else_,end}`) plus free
functions `open_block`/`open_loop`/`open_if`/`emit_else`/`close_frame`/
`branch_to_depth`/`resolve_depth`/`branch_if_to_depth`. `ControlFlowWriter`
is a **standalone** trait (`branch_label`/`branch_zero_label`/`place_label`
plus two new hooks, `flush`/`pop_cond`) rather than a `BlitzWriter`
supertrait — despite the first three methods meaning the same thing in
both traits, a backend's control-flow adapter is often a different wrapper
type than its `BlitzWriter` adapter (RISC-V's is `RegAllocW`, which needs a
`regalloc` field for `flush` that `BlitzW` has no reason to carry), and
forcing the supertrait would mean implementing nine unreachable stub
methods just to satisfy it. `TryTable`/`Throw`/`ThrowRef` (exception
handling) are deliberately excluded — the catch-arm arity/scratch-register
conventions are genuinely arch-specific and higher-risk to generalize;
`Endable::TryTable` stays a RISC-V-local case, resolved outside the shared
resolver (`Endable::{Block,Loop,If}` collapse into `Endable::Std(Frame)`).
`Br`/`BrIf`/`BrTable`'s own resolution (`WriterExt::br`/`br_after_flush`)
deliberately stays hand-rolled rather than routed through
`branch_to_depth`/`branch_if_to_depth`: `if_stack: Vec<Endable>` mixes
`Frame`-representable frames with the arch-local `TryTable` frame, and
`resolve_depth` takes `&[Frame]` — it can't represent that mix generically
without either forcing `TryTable` into `Frame` (rejected above) or an
enum-of-enums, so `br_after_flush`'s per-arm lookup was simplified via
`Frame::branch_target()` in place instead of being replaced outright.

Each arch's existing `codegen.rs` gains small adapter structs (mirroring
`BlitzW`) implementing these traits, supplying only the arch-specific pieces.

## Sequencing (risk-ordered, not a single PR)

1. **Generic regalloc adapter** — done (commit `93a1ce0`).
2. **Wire RISC-V's `BrTable` onto the pre-existing (previously-unused-anywhere) `emit_br_table`** — done (commit `aec082f`). Also split `naive::WriterExt::br` into `br`/`br_after_flush` so `BrTable`'s `resolve` closure could call the label-resolution half without re-flushing per arm or needing `&mut State` inside the closure.
3. **Shared data-flow dispatch (`regalloc_frontend`), ported RISC-V's `I32Const`/`I64Const`/`LocalGet`/`LocalSet`/`I32Add`|`I64Add`/`I32Sub`|`I64Sub`** — done (commit `cc8e9ec`).
4. **Ported RISC-V's remaining `Mul`/`DivU`/`And`/`Or`/`Xor`/`Shl`/`ShrS`/`ShrU` (via `binop`) and `Eq`/`Ne`/`LtS`/`LtU`/`GtS`/`GtU`/`LeS`/`LeU` (via the new `compare`)** — done (commit `24d2c47`). `I32Eqz`/`I64Eqz` intentionally left untouched (pre-existing broken/TODO'd code, unrelated to this refactor); `LocalTee` and `I32Load`/`I64Load`/`I32Store`/`I64Store` on RISC-V still hand-roll their own shape — follow-up, not yet done (memory ops in particular need a new shared shape, since `binop`/`compare` don't cover address computation).
5. **Build the shared control-flow resolver** (`Frame`/`open_block`/`open_loop`/`open_if`/`emit_else`/`close_frame`/`branch_to_depth`/`resolve_depth`/`branch_if_to_depth`) and port RISC-V's `Block`/`Loop`/`If`/`Else`/`End` onto it — done (commit `711d37b`). `Br`/`BrIf`/`BrTable` intentionally kept hand-rolled (see design section above for why); `TryTable` stays local, as planned.
6. **Complete x86-64's `fast.rs`** — migrate its control flow off the copy-pasted runtime-CTX-stack trick onto the shared resolver from step 4, replace its ad-hoc per-arm regalloc dance with `regalloc_frontend`, and fill in the `_ => {}` gap (most instructions are currently silently dropped). `fast.rs` has **zero existing test coverage** (`grep -c "fast::" crates/blitz-tests/tests/e2e.rs` → 0) — this increment must add new tests, not just rerun the existing suite. Not started.
7. **New AArch64 regalloc-backed backend** — net-new integration (an AArch64 `RegKind`/instruction-selection adapter for `regalloc_frontend`), reusing the already-existing upstream `asm-aarch64` regalloc primitives. Its control-flow model already matches the target, so only the data-flow half is new. Recommend landing as a new `blitz-aarch64/src/fast.rs` sibling (mirroring x86-64's naming) rather than modifying `naive.rs` in place, to keep the stable path stable while the new one is proven. Not started.

**Explicitly out of scope, indefinitely**: `blitz-x86-64/src/naive.rs` is not touched or migrated by any step above — it's the long-term deprecation target, but actually removing it is future work. `sysv.rs`/`lfi.rs` per arch, float ops, and memory load/store instruction selection are left for follow-up once the integer/control-flow core is proven.

## Verification

`crates/blitz-tests/tests/e2e.rs` (~6700 lines) executes emitted code via `unicorn-engine` across x86-64/aarch64/riscv64, plus a clang-assembled textual-asm path. Baseline before this effort started: `cargo test -p portal-solutions-blitz-tests` → 361 passed, 0 failed. Every step above so far has been verified against the same 361/361 (no regressions, no newly-skipped tests). Run `cargo build --workspace` and the full test suite before/after each step above; diff pass/fail sets.
