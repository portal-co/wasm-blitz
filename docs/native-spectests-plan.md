# Plan — Native-backend spectests (imports, MVP+)

> Shadow plan doc — update this file as increments land or the design changes,
> so any agent/session can pick up where a previous one left off.
>
> Companion to `docs/spectests-plan.md` (phases 1–4 done: JS + C backends run
> the phase-1 core file set under the baseline ratchet, 37/37 tests green).
> This plan covers spectest-plan **phase 3 properly**: the native backends
> (x86-64, aarch64, riscv64) under Unicorn, with **imports** — the piece the
> original phase-3 scope explicitly deferred ("host imports behind Unicorn
> callback support").

## Goal

Run the same phase-1 core wast file set (`fac`, `br`, `br_if`, `br_table`,
`call`, `const`, `local_get`, `block`, `loop`, `if`, `labels`, `forward`,
`stack`, `int_exprs`, `int_literals`, `left-to-right`) against the native
backends under Unicorn, with:

- **imports** — the `spectest` host module (print stubs, globals) and any
  import shape the file set needs, via a Unicorn-visible host-call mechanism;
- **MVP+ instruction parity** with what the JS/C backends already support for
  those files (globals, sign-extends, popcnt/clz/ctz, rotations, float
  nearest/trunc unops, trunc-sat);
- the same **baseline ratchet** semantics as the JS/C legs (new failures and
  stale entries both fail; entries keyed per backend).

Non-goals: SIMD/threads proposals; ILP32 arches (riscv32/arm/i686 keep the
existing `assemble_or_skip` soft-skip per AGENTS.md); native execution of
files using tables of funcrefs beyond what `CallIndirect` already supports;
100% core-suite coverage (this plan targets the phase-1 file set first, then
per-directory opt-in via `features.rs` `PROPOSAL_DIRS`).

## Current state (measured, on this tree)

| Capability | x86-64 | aarch64 | riscv64 | Needed by phase-1 file set |
|---|---|---|---|---|
| AllStack binary emit (no clang) | ✅ IcedWriter | ✅ bin AArch64Writer | ✅ RvAsmWriter (`into_parts` → `Vec<u8>`; verified in asm-arch checkout) | — |
| Imports (`External {m}__{n}` labels) | ✅ sysv.rs:534 | ✅ | ✅ sysv.rs:90 | fac/print variants, spectest stubs |
| Memory (`__wasm_mem_pages` / `__wasm_memory_grow` externals) | ✅ | ✅ | ✅ | left-to-right, stack |
| GlobalGet / GlobalSet | ❌ | ❌ | ❌ | fac, left-to-right, others |
| clz / ctz / popcnt | ❌ | ❌ | ❌ | int_exprs |
| rotl / rotr | ❌ | ❌ | ❌ | int_exprs |
| I32/I64Extend8S/16S(/32S) | partial | partial | ❌ | int_exprs |
| F32/F64 nearest / trunc unops | ❌ | ❌ | ❌ | (float files, phase B+) |
| trunc-sat | ❌ | ❌ | ❌ | (conversions, later) |
| Floats (all ops) | ✅ | ✅ | ❌ entirely | (float files, phase C) |
| div_s / rem_s spec traps | x86: partial (e2e-tested) | ✅ | ❌ (only div_u) | fac, int_exprs |
| Unreachable/trap convention under Unicorn | ad-hoc | ad-hoc | ad-hoc | br, fac, int_exprs traps |

Execution helpers that already exist in `crates/blitz-tests/tests/e2e.rs` and
will be **promoted to shared code** (not duplicated):

- `compile_allstack_binary(wasm, arch)` — AllStack-flavoured SysV compile to
  machine code (binary writers for x86-64/aarch64; riscv64 text asm).
- `run_allstack_entry(arch, code, entry, args, count)` — Unicorn invocation
  with AllStack stack marshalling (x86: params at `[rsp+8+i*8]`; aarch64/
  riscv64: params at `[sp+i*8]`); per-arch register conventions.
- `import_stub_add_one(arch)` — proves the **AllStack import ABI**: the callee
  pops its args from the emulated stack and pushes results back. This ABI is
  the contract the spectest host stubs will implement.

## Design

### 1. Host imports via sentinel trampolines (no Unicorn `mem_read` gymnastics)

Each import gets an assembly-level trampoline emitted into the code blob at a
known **sentinel guest address**:

```
<import k trampoline>:      ; AllStack ABI: args on emulated stack
  jmp IMPORT_SENTINEL_BASE + k*16     ; never returns; hook resumes emu
```

The Unicorn runner registers one `code_hook` for the sentinel page and
dispatches on the sentinel offset:

- The hook reads args from the emulated stack (same layout the e2e import
  stubs already consume), calls the Rust host-function closure
  (`spectest.print_i32`, `global_i32` getter, …), pushes results, and resumes
  emulation at the saved return address (which the trampoline preserved).
- A **trap result** from a host closure stops emulation with a distinguished
  error, which the runner maps to `ExecError::Trap`.

Why this shape: it is identical in spirit to the C backend's approach (host
functions are plain native calls), needs no cross-language FFI inside guest
code, and reuses the proven AllStack import ABI. No dependency on Unicorn's
`unicorn_engine::call`/stack juggling beyond the existing hook API.

Scope of spectest host functions for the phase-1 file set:
`print*` (no-op, optionally logged via `PORTAL_LOG_JSON`),
`global_i32`/`global_i64` (666), `global_f32`/`global_f64` (666.6 bit
patterns). `spectest.table` / `spectest.memory` imports → skip (table ops are
not in the phase-1 file set).

### 2. Guest runtime page

A read-write guest page (`RUNTIME_BASE`) laid out by the runner before each
invoke, holding the state the backends reference through `External` symbols:

| Symbol | Layout | Notes |
|---|---|---|
| `__wasm_mem_pages` | 1 × u32 | current linear-memory size in pages |
| `__wasm_memory_grow` | sentinel trampoline | hook implements `memory.grow` on the runner's guest memory mapping |
| `__wasm_globals` | 1024 × u64 | matches the C backend's convention (`__wasm_globals[1024]`) — new GlobalGet/GlobalSet emission targets this |
| `__wasm_table` | deferred | skip files needing it (same rule as C leg) |
| linear memory | `MEM_BASE`, initial pages × 64 KiB | memarg bounds checks are the backend's job (audit: see risks) |

### 3. Trap convention

Unify on one sentinel external, `__wasm_trap` (kind in a register / on stack):

- Backend `Unreachable`, div/rem trap checks, and load bounds checks call it.
- The hook stops emulation → runner maps to `ExecError::Trap(kind)`.
- `assert_exhaustion` = Unicorn instruction-count cap hit → `Trap("call stack
  exhausted")`. (Keep per-invoke caps generous; fac-rec 25 needs ~10⁵.)
- Audit existing `Unreachable`/trap emissions per arch and unify them onto
  `__wasm_trap` — today they are ad-hoc (`int3`, abort paths, `Throw`
  plumbing), which is invisible to the Unicorn runner.

### 4. Instruction parity work (per arch, measured above)

All additions land in the existing `naive.rs`/`sysv.rs` match arms per crate,
following the established per-crate patterns. Verify each with a hand-built
e2e-style unit test **in the backend crate's own test path or e2e.rs** before
the spectest leg relies on it:

1. **x86-64 (primary)** — GlobalGet/Set via `__wasm_globals` page; clz/ctz/
   popcnt (`lzcnt/tzcnt/popcnt` or BSR fallback); rotl/rotr (`rol/ror`);
   sign-extends (`movsx`); float nearest/trunc via `roundss/roundps` with the
   spec's mode bits; trunc-sat via compare+select sequences.
2. **aarch64 (second)** — same set via `clz/rbit`-based ctz/popcount
   sequences, `ror`, `sxtb/sxth/sxtw`, `frintn/frintz`, fmin/fmax already
   present.
3. **riscv64 (third)** — the big one: entire float block (Unicorn's
   qemu-riscv64 emulates the F/D extensions, so hardware FP instructions are
   fine — spike-verify early), div_s/rem_s with trap checks, select, sign-
   extends, rotations via shift-pair sequences, clz/ctz/pcnt (`clz/ctz/
   pcnt` in Zbb, or shift-or fallback if the emulated CPU lacks Zbb).


### 5. Harness integration

- New `tests/spec/native_exec.rs`: sentinel-trampoline emitter, runtime-page
  builder, Unicorn runner with hook, host-function registry.
- `tests/spec/mod.rs`: `Backend::Native(NativeArch)` added to the `Backend`
  enum; `run_wast_file_backend` gains the native path. Eligibility gate per
  module (skip reasons, never silent): typed tables, func imports other than
  spectest print/global set, tables/elem, multiple memories.
- `baseline.toml`: the `backend` field (currently informational) becomes part
  of the ratchet key (`file + idx + backend`), with a migration pass over the
  3 existing entries. Native entries are added only for genuine native-codegen
  bugs, same rules as JS/C.
- `tests/spectests.rs`: one `#[test]` per (phase-1 file × arch), 48 tests at
  full rollout; soft-skip when the arch's assembler/writer is unavailable.

### 6. CI

`ci.yml`/`spectests.yml` need no suite-side change (Unicorn is a library, and
all three arches emit binary machine code — no cross-clang anywhere in the
native spectest path). Native spectests run on ubuntu-latest only (same as
today); macOS hosts run the same tests locally since nothing shells out to an
assembler.

## Phases (each ends committed, spectests + e2e green)

**A. Runtime scaffolding** — native_exec.rs (sentinel ABI, runtime page,
trap convention, runner); promote `compile_allstack_binary`/
`run_allstack_entry` into shared code; smoke: a hand-built module with an
import + a trap + memory.grow runs under x86-64 and aarch64 via binary
writers. *Exit: 3 arches × smoke green.*

**B. x86-64 MVP+ parity** — instruction gaps from the table above; per-op
unit tests; wire `Backend::Native(X86_64)` into the harness for the phase-1
file set; seed native baseline entries. *Exit: phase-1 set green under
ratchet on x86-64.*

**C. aarch64, then riscv64** — same treatment; riscv64 floats spike first.
*Exit: phase-1 set green under ratchet on all available arches; unavailable
arches counted skips.*

**D. File-set expansion + docs** — extend toward the full core set via
`features.rs` opt-in as coverage lands; trap-message tightening where cheap;
update spectests-plan.md phase-3 entry to point here.

## Risks / open questions

- **Unicorn hook semantics**: resuming emulation from a code hook at an
  arbitrary address must be verified on unicorn-engine 2.x Rust bindings
  early (phase A spike). Fallback: sentinel trampolines `ret` into a per-
  import magic return address observed via a code hook, without manual PC
  writes.
- **Load/store bounds checks**: JS/C backends gained bounds behaviour via
  their runtimes; the native backends' memarg handling must be audited —
  missing bounds checks will show up as Unicorn faults (guest memory
  protection), which the runner should classify as traps, not crashes, and
  the resulting baseline entries are real bugs to fix in the backends.
- **Float rounding modes** on riscv64 (FRM register) — spec expects
  round-to-nearest-even by default; ensure emission doesn't set dynamic
  modes.
- **aarch64 text-asm fragility on the macOS host** (the 5 pre-existing e2e
  failures) is bypassed by using the binary AArch64 writer; do not regress
  to text+clang for aarch64. Same applies to riscv64 (`RvAsmWriter` is
  binary, verified).
- **Instruction-count caps**: too low turns long-running asserts into false
  exhaustion traps; too slow turns the suite into a CI problem. Keep caps
  per-invoke configurable via env (`BLITZ_SPEC_NATIVE_CAP`), default ~10⁷.
- **Baseline migration**: making `backend` part of the key is a one-time
  ratchet change — do it in its own commit so the diff is reviewable.

## Status (updated as phases land)

- **Phase A — DONE** (commit `5b58dfc`): `tests/spec/native_exec.rs` runtime —
  AllStack compile + Unicorn runner, sentinel import trampolines, runtime data
  area, trap classification, three x86-64 smokes.
- **Phase B — DONE** (commits `6dee8eb`, `118cb6e`): x86-64 MVP+ instruction
  parity in `blitz-x86-64` (comparisons, clz/ctz/popcnt, rotations, extends,
  globals, copysign); `Backend::NativeX86` wired into the harness; backend key
  added to the baseline ratchet (own commit); native-x86 baseline entries.
- **Phase C — DONE** (commits `feb887c`, phase-C-step-2):
  - aarch64: MVP+ arms in `blitz-aarch64` (nop, i32 rem, SWAR
    clz/ctz/popcnt, shift-pair rotates/extends, globals, copysign — the
    asm-aarch64 Writer has no raw-word escape hatch, so everything is built
    from existing ALU primitives); AArch64 compile arm in `native_exec.rs`
    with post-assembly ADRP/ADD relocation patching and a B-trampoline stub
    region. The sentinel page moved to `0x1100_0000` — AArch64 `B` imm26 is
    relative to the instruction's own address with ±128MB range, and the old
    `0x4010_0000` page was out of reach.
  - riscv64: MVP+ integer arms in `blitz-riscv64` (nop, ebreak-unreachable,
    div/rem family, ge_s/ge_u, extends, SWAR clz/ctz/popcnt, globals);
    `emit_const` replaces the vendored `li` medium path (i32 overflow on
    values with low-12 ≥ 0x800); consts flush the allocator and push
    directly (scratch registers may alias allocator regs). `uses_floats`
    module gate skips float files on riscv64 (backend has no float support).
- **Phase D — DONE**: file-set expansion stays opt-in via `features.rs`
  `PROPOSAL_DIRS`; native baselines record the residual backend bugs (see
  baseline.toml `native-*` entries); this file documents the landed state.

Suite totals after phase D: **87 spectest tests green** (16 files × 3 native
arches + JS + C legs + smokes), e2e unchanged at 389 pass / 5 pre-existing
aarch64 host-clang failures.

Residual native backend bugs (baselined, tracked for backend work):
- All arches: multi-value br carries, deep br/br_if depths, missing
  div/rem-by-zero spec traps, exhaustion classification.
- riscv64 additionally: nested loop+if+br miscompile (returns 1), float ops
  entirely.
