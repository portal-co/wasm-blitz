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
| `blitz-x86-64/src/sysv.rs` | compile-time label stack (own `SysVCtrl`/`ctrl_stack`, CTX-free), delegates unhandled ops to `naive::_handle_op` | pure stack (delegated to naive) — zero regalloc usage |
| `blitz-x86-64/src/lfi.rs` | delegates to naive/sysv (sandboxed variant, same shapes as whichever it wraps) | pure stack — zero regalloc usage |
| `blitz-x86-64/src/fast.rs` | **dropped, see below** | — |
| `blitz-aarch64/src/naive.rs` | compile-time label stack (`Endable::Block{end_lbl}` etc., no runtime CTX manipulation) | pure stack (`wasm_push`/`wasm_pop`) — zero regalloc usage anywhere in the crate |
| `blitz-riscv64/src/naive.rs` | compile-time label stack (`Endable::Block{idx}` etc. — same shape as aarch64's) | regalloc + spill, **fully migrated onto the shared frontend** (reference implementation) |

RISC-V's `naive.rs` and AArch64's `naive.rs` already agree on the control-flow model (compile-time label stack). x86-64's `sysv.rs` *also* already uses that model (its own `SysVCtrl`, independent of the CTX trick `naive.rs`/`fast.rs` used) — it just has no regalloc-backed data flow yet, since it delegates to `naive::_handle_op`'s pure stack machine for everything it doesn't override. x86-64's `naive.rs` remains the sole outlier on both axes and is explicitly **not** part of this unification — long-term deprecation target once the regalloc-based path is mature enough to replace it.

**`fast.rs` dropped (2026-07-12).** It was investigated as the vehicle for x86-64 regalloc unification (steps below originally said "complete fast.rs"), but turned out to be dead code: `grep -rn "fast::"` outside the file itself is empty — no ABI marker, no `sink.rs` wiring, nothing instantiates it. Worse, its `Block`/`Loop`/`Br` copy the *same* runtime-CTX-stack trick as `naive.rs`, which the existing test suite already documents as unable to execute standalone in Unicorn without host-runtime scaffolding (`assert_native_naive_smoke`'s x86-64 arm only checks "output non-empty") — so completing it would also mean inventing new test infrastructure just to verify it, for a backend nothing calls. Decision: leave `fast.rs` as-is, add it to the eventual naive.rs-adjacent deprecation list (do not extend it further), and pursue x86-64 regalloc unification through `sysv.rs`/`lfi.rs` instead — those *are* reachable and *do* have real Unicorn-execution test coverage today (`assert_native_sysv_const` and friends).

**Revised target end-state**: every reachable, testable backend (RISC-V naive fully done; x86-64 sysv/lfi and a new AArch64 backend still to do) uses compile-time-label-stack control flow + regalloc-backed data flow. x86-64 `naive.rs` and the abandoned `fast.rs` are the only things left permanently outside this.

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
4. **Ported RISC-V's remaining `Mul`/`DivU`/`And`/`Or`/`Xor`/`Shl`/`ShrS`/`ShrU` (via `binop`) and `Eq`/`Ne`/`LtS`/`LtU`/`GtS`/`GtU`/`LeS`/`LeU` (via the new `compare`)** — done (commit `24d2c47`).
5. **Build the shared control-flow resolver** (`Frame`/`open_block`/`open_loop`/`open_if`/`emit_else`/`close_frame`/`branch_to_depth`/`resolve_depth`/`branch_if_to_depth`) and port RISC-V's `Block`/`Loop`/`If`/`Else`/`End` onto it — done (commit `711d37b`). `Br`/`BrIf`/`BrTable` intentionally kept hand-rolled (see design section above for why); `TryTable` stays local, as planned.
6. **Ported RISC-V's `I32Load`/`I64Load`/`I32Store`/`I64Store`** via new `load`/`store` free functions in `regalloc_frontend` (pop addr → new dest → load; pop val → pop addr → store, no push) — done (commit `37aaef3`). **This completes the RISC-V `naive.rs` migration** onto the shared abstractions for everything that fits them. Deliberately left untouched, each for a specific reason (see that commit message): `LocalTee` (raw-stack peek/writeback, not regalloc-based, no cross-crate duplication yet to justify a new shape for it), `I32Eqz`/`I64Eqz` (pre-existing broken/TODO'd code), `Call`/`Return`/`MemorySize`/`MemoryGrow` (ABI/calling-convention boundary, not WASM-operator data/control flow), `Throw`/`ThrowRef`/`TryTable` (exception handling, already excluded).
7. **`fast.rs` dropped** — see the Context section above. Not touched further; added to the deprecation list alongside `naive.rs`.
8. **Give x86-64 a regalloc-backed data-flow core, reached through `sysv.rs`** — net-new (there is no existing regalloc-based x86-64 core to extract from, now that `fast.rs` is dropped): a new module (e.g. `blitz-x86-64/src/regalloc_core.rs`, mirroring `blitz-riscv64::naive`'s shape) implementing the shared instruction set via `regalloc_frontend`/`control_flow`, with an x86-64 `RegAllocW`/`ControlFlowWriter` adapter in `codegen.rs` (mirroring RISC-V's). `sysv.rs`'s `sysv_handle_op` stops delegating its "everything else" fallback to `naive::_handle_op` (pure stack) and delegates to this new core instead. `sysv.rs` already uses the compile-time label-stack control-flow model independently (its own `SysVCtrl`/`ctrl_stack`) — confirm during implementation whether that can adopt `control_flow::Frame` directly or needs the same "wrap it, don't force it" treatment RISC-V's `TryTable` got. Verify against the existing `assert_native_sysv_const`-style real-execution tests plus new ones for the newly-regalloc-backed ops. Not started.
9. **`lfi.rs` picks up the same core once step 8 lands** — `lfi.rs` already delegates to naive/sysv rather than duplicating their instruction dispatch, so it should mostly inherit step 8's work; confirm during implementation whether LFI's sandboxing constraints (verified by `lfi-verify --arch x64`) impose any additional restrictions on which registers/instructions the regalloc core may use. Not started.
10. **New AArch64 regalloc-backed backend** — net-new integration (an AArch64 `RegKind`/instruction-selection adapter for `regalloc_frontend`), reusing the already-existing upstream `asm-aarch64` regalloc primitives. Its control-flow model already matches the target, so only the data-flow half is new. Not started.

**Explicitly out of scope, indefinitely**: `blitz-x86-64/src/naive.rs` and the now-dropped `blitz-x86-64/src/fast.rs` are not touched or migrated by any step above. Float ops are left for follow-up once each integer/control-flow core is proven.

## Verification

`crates/blitz-tests/tests/e2e.rs` (~6700 lines) executes emitted code via `unicorn-engine` across x86-64/aarch64/riscv64, plus a clang-assembled textual-asm path. Baseline before this effort started: `cargo test -p portal-solutions-blitz-tests` → 361 passed, 0 failed. Every step above so far has been verified against the same 361/361 (no regressions, no newly-skipped tests). Run `cargo build --workspace` and the full test suite before/after each step above; diff pass/fail sets.
