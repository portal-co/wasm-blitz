# Plan — WebAssembly spec-test suite (conformance test + CI)

> Shadow plan doc — update this file (not just chat/session state) as increments
> land or the design changes, so any agent/session can pick up where a previous
> one left off.

## Context

wasm-blitz currently tests with hand-built modules in
`crates/blitz-tests/tests/e2e.rs` (wasm-encoder → backends → node / clang /
Unicorn). Hand-written tests prove *some* behaviors but not conformance. The
official [WebAssembly spec test suite](https://github.com/webassembly/spec)
(`test/core/*.wast`) is the industry-standard conformance corpus (~50k+
assertions) and is the fastest way to find backend divergences — especially in
the "boring" corners: operand order, NaN payloads, trap conditions, bounds
checks.

Goal: a `cargo test -p portal-solutions-blitz-tests --test spectests` target
that runs the spec suite against every backend that can execute it, plus a CI
workflow that runs it on every push/PR with a pinned suite revision and a
tracked baseline of known failures.

Non-goal: making wasm-blitz pass 100% of the suite in one step. The suite is
large; this plan is about the *harness + CI*, with an explicit baseline/xpass
mechanism so partial conformance is still enforced, not drifted.

## What we are running

The spec suite ships as `.wast` scripts: a mix of module definitions and
assertion commands:

| Command | Meaning | Harness action |
|---|---|---|
| `(module ...)` | define + validate + instantiate module | compile, keep as "current" |
| `(register "name" $id)` | make current module an importable instance | add to host registry |
| `(invoke "export" args...)` | call exported function | execute, discard result |
| `(get "export")` | read exported global | read value |
| `(assert_return (invoke ...) results...)` | call, compare results | execute + compare |
| `(assert_trap (invoke ...) "msg")` | call must trap | execute, expect trap |
| `(assert_exhaustion ...)` | call must exhaust stack | execute, expect stack trap |
| `(assert_invalid module "msg")` | binary must fail validation | feed binary to pipeline, expect reject |
| `(assert_malformed module "msg")` | binary must fail decode | feed binary to pipeline, expect reject |
| `(assert_unlinkable ...)` | instantiation must fail | instantiate, expect link error |
| `(assert_return (get ...))` | global value check | read + compare |

Two deliberate scope cuts (standard practice for compilers):

- `assert_malformed` with **quote** (text-format) modules tests the spec's text
  parser, not ours — **skip** (our pipeline starts from binary). Binary
  (`module binary ...`) malformed assertions are kept.
- Trap **messages** are matched loosely at first (trap vs no-trap), with a
  later phase tightening to per-trap-kind matching. Message-exact matching is
  where most engines cheat, and it is not where wasm-blitz's risk is.

## Design

### New test target

```
crates/blitz-tests/
  tests/
    spectests.rs          # #[test] drivers, one per (suite file × backend)
    spec/
      mod.rs              # wast driving, assertion interpretation, host registry
      baseline.toml       # tracked known-failure list (see Baseline below)
      README.md           # how to run, how to re-baseline
```

Dependency to add (dev-dependency of `blitz-tests` only — `blitz-common` stays
`no_std` and untouched):

```toml
wast = "240"   # matches wasmparser/wasm-encoder 0.240 in the workspace
```

`wast` parses `.wast` (text + binary + assertion commands) and can encode
modules to binary via `wasm_encoder` — the same encoder family the rest of the
workspace already uses, so the compiled-pipeline entry points
(`mach_operators` → `dce_pass!` → backend `on_mach`) are reused as-is.

### Suite acquisition

- CI: `actions/checkout` of `webassembly/spec` at a **pinned commit** (sha, not
  branch) into a job-local path; the test reads
  `BLITZ_SPEC_DIR` env var, else `SPECTESTS_DIR`, else skips with a loud
  warning (mirrors the `assemble_or_skip` fail-soft pattern for missing
  cross-toolchains).
- Local dev: `make spec-fetch`-style script (plain `git clone --depth 1` into
  `target/spec`, idempotent). No vendoring of the suite into the repo.
- Sub-suite selection: `test/core/*.wast` only for phase 1; proposal
  directories (`test/core/gc/`, `tail-call/`, `exception-handling/`, …) are
  opt-in per-backend as features land (see Feature gating).

### Backend execution matrix

Reuse the e2e.rs execution paths verbatim where possible:

| Backend | Execution | Spectest feasibility |
|---|---|---|
| `blitz-js` | `node` | **Primary target.** Full module semantics available in JS; traps surface as thrown JS errors; `spectest` import object trivial to synthesize. |
| `blitz-c` | `clang` native compile + run | **Primary target.** Trap propagation already uses `setjmp`/`longjmp`; the driver links a small C `main` that registers host functions. |
| `blitz-x86-64` (sysv) | clang `-target x86_64` + Unicorn | **Secondary.** Only tests that are expressible in the current standalone-Unicorn smoke shape run (no host imports at first — see risks). |
| other native (`aarch64`, `riscv64`, ILP32) | `assemble_or_skip` | Phase 3; soft-skip when clang triple missing (existing pattern, per AGENTS.md). |
| `blitz-jvm`, `blitz-ppc64` | — | Out of scope for phase 1; jvm once it has an execution story. |

Per (file, backend) we get one `#[test]` per wast file (e.g.
`spectests::js::i32`, `spectests::c::memory_grow`) rather than per assertion —
keeps test-count manageable (~80 files × 2 backends in phase 1) while
`PORTAL_LOG_JSON=1` events carry per-assertion detail for diagnosis.

### Host imports (`spectest` module)

The suite imports from module `"spectest"`: `print*` (6 arities),
`global_i32/i64/f32/f64`, `table` (funcref, 10×20), `memory` (1 page, 2 max),
plus `(register)`-created instances. The harness synthesizes this per backend:

- JS: a plain JS object; `print*` push a marker to an output array the driver
  inspects (some `assert_return` cases check side effects via print ordering).
- C: hand-written `__spectest_print*` etc. compiled into the driver binary.
- Native/Unicorn: phase 3 — requires host-callback support in the Unicorn
  runner; until then, modules importing `"spectest"` soft-skip with a counted
  reason (not silent).

### Value comparison rules

- i32/i64: exact.
- f32/f64: respect the spec's NaN patterns — `wast`'s `NanPattern::Canonical`
  / `Arithmetic` / `Value`. Implement bitwise comparison: `Value(x)` → bit
  equality; `Canonical` → ± canonical NaN; `Arithmetic` → any NaN (the
  permissive end; tighten per-op later if the spec test data demands it — most
  arithmetic ops legitimately allow arithmetic NaNs).
- `ref.null`/`ref.func`/`ref.extern` results: phase 2 (needs GC/function-references).

### Feature gating

The core suite has absorbed several formerly-proposal features. Map them to
`wasmparser` feature flags and to per-file backend capability:

- **Phase 1 (core MVP)**: i32/i64/f32/f64 arithmetic, control flow, locals,
  memory, data, br_table, call/call_indirect. Files requiring
  bulk-memory/multi-memory/tail-call/reference-types beyond what the backends
  already handle start in the baseline.
- **Phase 2**: bulk memory (`memory_copy/fill/init` — already has e2e
  coverage), reference types / funcref tables (already partially emitted via
  `js_emit_funcref_table` / `c_emit_funcref_table`), sign-extension ops,
  multi-value, multi-memory, tail calls (`return_call` — e2e-tested).
- **Phase 3**: GC, function-references, exception-handling (e2e already emits
  tag/exn scaffolding), memory64, relaxed-simd, custom-page-sizes,
  extended-const — each enabled per backend only after its e2e smoke exists.

Mechanism: a `features.rs`-style table in `tests/spec/` mapping wast file →
required feature set; files whose features a backend lacks are auto-skipped
with a reported count, not baseline entries.

### Baseline management (the important part)

`crates/blitz-tests/tests/spec/baseline.toml`:

```toml
# reason: required so no entry is ever cargo-culted without explanation
[[failures]]
file   = "float_memory.wast"
test   = "js"                 # backend id: js | c | x86_64-sysv | ...
assert = 47                   # index of the failing assertion within the file
reason = "f32.load NaN pattern mishandled in js backend"
```

Semantics, enforced by the harness itself:

- A failure **on** the baseline → test reports known-failure (does not fail CI).
- A failure **not** on the baseline → hard failure (regression).
- A baseline entry whose assertion now **passes** → hard failure ("stale
  entry — remove it"), so the baseline can only shrink. This is the ratchet.
- Baseline entries are keyed by assertion index within a file, which is stable
  for a pinned suite commit; bumping the suite commit requires re-baselining
  (documented in `tests/spec/README.md`, tool-assisted: harness prints the new
  baseline diff on `BLITZ_SPEC_REBASELINE=1`).

A summary is printed at the end of the run: per backend — passed / failed
(known) / failed (new) / skipped-with-reason counts.

### CI workflow

New: `.github/workflows/ci.yml` (repo currently has none) plus
`.github/workflows/spectests.yml`.

```yaml
# spectests.yml (shape)
on: [push, pull_request]
jobs:
  spectests:
    strategy:
      matrix:
        os: [ubuntu-latest]        # macos-latest added in phase 2
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with: { repository: webassembly/spec, ref: <pinned-sha>, path: spec }
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-node@v4     # JS backend executor
      - uses: KyleMayes/install-llvm-action@v2   # clang with all targets (C + native assembly)
      - run: cargo test -p portal-solutions-blitz-tests --test spectests
        env:
          BLITZ_SPEC_DIR: ${{ github.workspace }}/spec
          PORTAL_LOG_JSON: "1"          # compression-aware structured output (AGENTS.md)
          PORTAL_LOG_BAT: "1"
```

Notes:

- **No QEMU in CI**: the JS and C backends run natively on the ubuntu runner;
  native backends run under Unicorn (a library, host-neutral). QEMU remains
  reserved for the macOS-VM Linux-coverage case per global AGENTS.md (software
  emulation only, fail-closed on missing prerequisites) and is *not* needed by
  this plan.
- `assemble_or_skip` semantics carry over: a missing/featureless clang must
  soft-skip native-backend files, never fail CI (ILP32 triples etc.).
- A weekly scheduled job (`schedule:` cron) against the spec suite's `main`
  branch, reporting-only (no PR blocking) to surface upstream drift early.
- Job summary step parses the harness's final summary (the harness writes a
  machine-readable `BLITZ_SPEC_SUMMARY=path` JSON when requested) into a
  GitHub step summary table — pass/fail/skip per backend visible on every PR.

### Observability

Follow the compression-aware logging convention (AGENTS.md): all harness
diagnostics go through the existing `log.rs` `LlmtrimLogger`
(`PORTAL_LOG_JSON=1`, `PORTAL_LOG_BAT=1`), one batch per (file × backend) test
with per-assertion events (`phase:"assert"`, fields: file, index, kind, ok).
When unset, plain output only on failure.

## Phases

1. **Harness skeleton** — DONE. `wast =240.0.0` driver, JS backend, 16
   phase-1 files (`const`, `local_get`, `block`, `loop`, `if`, `br`, `br_if`,
   `br_table`, `call`, `labels`, `forward`, `fac`, `stack`, `int_exprs`,
   `int_literals`, `left-to-right`), per-file `#[test]`s, persistent per-file
   node session (JSON line protocol over stdin/stdout, `BLITZ_SPEC_DUMP_JS=1`
   dumps generated JS), baseline ratchet with 3 seeded known failures
   (multi-value br carry; loop back-edge br_if condition loss in `fac`),
   baseline.toml hand-parsed (no serde dep). Suite pinned at
   WebAssembly/spec 37d6b05914b6833330001cea9f051b97f98af5b8.
   *Done: 16/16 files green under the ratchet (16 pass, 3 known-fail).*
   Backend additions required to get there: Drop/Nop/Unreachable/Select,
   comparisons (i32/i64 signed+unsigned), bitwise ops, clz/ctz/popcnt,
   globals (GlobalGet/GlobalSet via `$g_N`), float consts/arith/comparisons/
   conversions/stores/loads, wrap/extend/sign-ext, div/rem with spec trap
   semantics (`__udiv`/`__urem`/`__idivS`/`__srem` in `js_module_preamble`),
   function-level `br` label (Frame::Function early-return).
2. **C backend + full core file set** — all of `test/core/*.wast` for js+c,
   value/NaN comparison complete, `register`/host-registry support, stale-entry
   ratchet enforced.
3. **Native backends** — x86-64 sysv under Unicorn for files expressible
   standalone; ILP32 soft-skips per existing pattern; host imports behind
   Unicorn callback support.
4. **Proposals** — per-backend opt-in for bulk-memory, reference-types,
   tail-call, multi-memory, sign-extension, multi-value.
5. **Tightening** — trap-message matching where cheap, arithmetic-NaN
   narrowing, macOS CI leg, scheduled upstream-drift job → eventually flip
   "known failures" reporting into tracked issues per backend.

## Risks / open questions

- **`wast` crate version alignment**: must track the workspace's
  wasmparser/wasm-encoder 0.240 line (`wast = "240"`). If the pinned spec suite
  uses newer proposals that 0.240 cannot parse, phase 4 is blocked until the
  workspace bumps — acceptable; the pin makes it visible.
- **Instantiation vs compilation**: wasm-blitz's pipeline compiles function
  bodies; the suite also stresses *linking* and *instantiation* semantics
  (imports, tables, element segments, `assert_unlinkable`). Backends emit some
  of this (`js_emit_imports`, `c_emit_import_decls`, funcref tables) but
  data/elem bounds checking at instantiation time may be a real gap the suite
  will find — that's the point; expect phase-2 baseline weight there.
- **Unicorn host calls**: `assert_return` needs host `spectest` imports for
  many files; until Unicorn callback plumbing exists, those files soft-skip on
  native — counted, never silent.
- **Suite pin rotation**: bumping the pinned sha re-baselines everything; keep
  bumps deliberate (one PR per bump, diff visible via the re-baseline output).
- **Runtime cost**: full core suite × 2 backends under Unicorn could be slow;
  phase 3 may need `--ignored`-style splitting or per-file test parallelism
  (cargo does this natively; keep one `#[test]` per file, not one big test).

## Files touched (summary)

| Path | Change |
|---|---|
| `crates/blitz-tests/Cargo.toml` | add `wast = "240"` dev-dependency |
| `crates/blitz-tests/tests/spectests.rs` | new — per-file × backend `#[test]`s |
| `crates/blitz-tests/tests/spec/mod.rs` | new — wast driver, comparison, baseline engine, logging |
| `crates/blitz-tests/tests/spec/baseline.toml` | new — tracked known failures |
| `crates/blitz-tests/tests/spec/README.md` | new — run/re-baseline instructions |
| `.github/workflows/spectests.yml` | new — pinned suite, node + llvm, JSON logging |
| `.github/workflows/ci.yml` | new — existing `cargo test -p portal-solutions-blitz-tests` (e2e) gate |
| `README.md`, `AGENTS.md` | document the spectest target and env vars |
