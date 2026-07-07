# wasm-blitz ABI Reference

This document describes every calling convention used by the wasm-blitz backends.
Each backend has a **native WASM-stack ABI** (the default, optimised for JIT execution)
and a **System V compatible ABI** (callable from standard C / host code, slightly slower
because of argument-register marshalling at the function boundary).

---

## blitz WASM ABI — x86-64 naive (`crates/blitz-x86-64/src/naive.rs`)

### Registers
| Register | Role |
|----------|------|
| `RSP` / Reg(4) | WASM operand stack pointer (grows down, same as hardware stack) |
| `Reg::CTX` / Reg(255) | Frame pointer — points to base of the local variable area |
| Reg(0)–Reg(3) | Caller-saved temporaries used during instruction emission |

### Function entry
Functions are entered via a hardware `call` instruction from the caller.  At the entry
point the stack looks like:
```
[RSP]     = return address          ← pushed by `call`
[RSP+8]   = param_0
[RSP+16]  = param_1
...
```
`StartFn` handler:
1. `pop Reg(1)` — pop return address into Reg(1)
2. `lea Reg(0), [Reg(1) − params*8]` — compute base of parameter area
3. `xchg Reg(0), CTX` — CTX ← frame pointer, Reg(0) ← old CTX (saved for callee)

`StartBody` handler then:
1. Pushes return address and old CTX onto the WASM stack (for `Return` to use)
2. Reserves `control_depth * 2` slots for the control-flow continuation stack
3. Updates CTX to point to the top of the control-flow stack

### Local variable access
Local N is at `[saved_RA − (N+1)*8]`, accessed by temporarily swapping `RSP ↔ CTX`,
adjusting RSP, and using `pop`/`push`.

### Return
1. Restore RSP to the base of the control-flow stack (via CTX computation)
2. Pop old CTX and return address
3. Push return values onto the WASM stack
4. `ret` — jumps back to the saved return address

### Function calls
- Compiled WASM calls: `lea_label(Reg(0), Func{fn})` then `call Reg(0)`
- Hypercalls: `hcall` mechanism (push return label + fn addr, swap CTX/RSP, `ret`)

### External symbols
`X64Label::External { name: &'static str }` — the concrete `Writer` implementation
must map external names to their runtime addresses (e.g. via relocations or direct
address embedding).  Used for `__wasm_mem_pages` and `__wasm_memory_grow`.

---

## blitz WASM ABI — RISC-V 64 naive (`crates/blitz-riscv64/src/naive.rs`)

### Registers
| Register | ABI name | Role |
|----------|----------|------|
| Reg(2) / x2 | sp | WASM operand stack pointer |
| Reg(8) / s0 | fp/s0 | Frame pointer (callee-saved) |
| Reg(1) / ra | ra | Link register (return address) |
| Reg(10) / a0 | a0 | Temporary for function address during calls |
| Reg(9) / s1 | s1 | Context / regalloc state |

### Function entry (prologue)
```asm
.Lfn_N:
    addi sp, sp, -8
    sd   s0, 0(sp)        ; save old frame pointer
    mv   s0, sp           ; FP ← current SP
    addi sp, sp, -(locals_slots * 8)   ; allocate locals
```
`locals_slots = params + local_count + control_depth*2 + 4`

Parameters are already on the WASM stack above SP when the call was made via `jal`.

### Local variable access
Local N at `[s0 − (N+1)*8]` (FP-relative negative offset).

### Return (epilogue)
```asm
    mv   sp, s0           ; restore SP to FP
    ld   s0, 0(sp)        ; restore old FP
    addi sp, sp, 8
    jalr x0, ra, 0        ; ret — jumps to saved return address in ra
```

### Function calls
- `jal_label(Reg(10), Func{fn})` — loads function address into a0
- `call(Reg(10))` — `jalr ra, a0, 0` (ra ← return address, jump to a0)

### External symbols
`RiscvLabel::External { name: &'static str }` — resolved by the concrete `Writer`.
Used for `__wasm_mem_pages` (load 32-bit page count) and `__wasm_memory_grow`
(call using the WASM calling convention).

---

## blitz WASM ABI — AArch64 naive (`crates/blitz-aarch64/src/naive.rs`)

### Registers
| Register | ABI name | Role |
|----------|----------|------|
| Reg(31) / sp | sp | WASM operand stack pointer |
| Reg(29) / x29 | fp | Frame pointer (callee-saved) |
| Reg(30) / x30 | lr | Link register (callee-saved in prologue) |
| Reg(0)–Reg(3) | x0–x3 | Caller-saved temporaries |

### Function entry (prologue)
```asm
.Lfn_N:
    stp  x29, x30, [sp, #-16]!   ; save FP and LR (pre-decrement SP)
    mov  x29, sp                   ; FP ← SP
    sub  sp, sp, #(locals_slots * 8)  ; allocate locals
```

### Local variable access
Local N at `[x29, #-(N+1)*8]` (FP-relative negative offset, same layout as RISC-V).

### Return (epilogue)
```asm
    mov  sp, x29              ; restore SP to FP
    ldp  x29, x30, [sp], #16  ; restore FP and LR (post-increment SP)
    ret                        ; branches to LR
```

### Function calls
- `adr_label(Reg(0), Func{fn})` — loads function address into x0
- `bl(Reg(0))` — branch-with-link (LR ← return address, jump to x0)

### External symbols
`AArch64Label::External { name: &'static str }` — resolved by the concrete `Writer`.

---

## blitz C ABI (`crates/blitz-c/src/lib.rs`)

### Function signature
```c
static uint64_t* fn_N(uint64_t* restrict locals_in);
```
- `locals_in[0..params]` — parameter values
- Returns pointer to `__rets_N` (module-scope static buffer), `__rets_N[0..rets]`

### Memory globals
```c
static uint8_t  *__wasm_mem       = 0;    /* pointer to linear memory bytes */
static uint32_t  __wasm_mem_pages = 0;    /* current page count (1 page = 64 KiB) */
/* Grow hook — caller must provide this symbol: */
extern uint32_t __wasm_memory_grow(uint32_t delta, uint8_t** mem, uint32_t* pages);
```

### Data initialisation
`c_emit_data_segments()` emits `__wasm_init_data()` which uses `memcpy` to apply
active data segments.  Caller must invoke it after allocating `__wasm_mem`.

---

## blitz C SysV ABI (`CWriteSysV` trait)

### Function signature
```c
/* 0 returns */  void     fn_N_sysv(uint64_t arg0, ..., uint64_t argN);
/* 1 return  */  uint64_t fn_N_sysv(uint64_t arg0, ..., uint64_t argN);
/* 2+ returns*/  uint64_t fn_N_sysv(uint64_t arg0, ..., uint64_t argN, uint64_t* extra);
```
- First return value is the direct return value
- Additional return values written through `extra[0], extra[1], ...`
- Internally wraps `fn_N` (the blitz C ABI function)

---

## blitz JS ABI (`crates/blitz-js/src/lib.rs`)

### Function signature
```js
function $N(...locals)   // individual BigInt arguments
```
- Takes BigInt values for each WASM parameter
- Returns a single BigInt (1-return function) or an Array of BigInt (multi-return)
- Already System V compatible in spirit — no separate SysV mode

### Memory globals
```js
var $mem    = new Uint8Array(0);      // linear memory byte buffer
var $mem_dv = new DataView($mem.buffer);  // DataView for typed access
```
`memory.grow` resizes `$mem` and reassigns `$mem_dv` in-place.
`js_apply_data_segments()` emits `$mem.set(bytes, offset)` calls.

---

## System V x86-64 ABI (`crates/blitz-x86-64/src/sysv.rs`)

Follows the **AMD64 System V ABI**.  The internal WASM operand stack still uses the
hardware stack (RSP), but the function boundary uses standard register conventions.

### Entry registers
| Argument index | Register |
|----------------|----------|
| 0 | RDI |
| 1 | RSI |
| 2 | RDX |
| 3 | RCX |
| 4 | R8 |
| 5 | R9 |
| ≥6 | pushed right-to-left by caller |

### Return registers
- 1 return value: RAX
- 2 return values: RAX (first), RDX (second)

### Prologue
```asm
push rbp
mov  rbp, rsp
sub  rsp, frame_sz      ; frame_sz = (params + local_slots) * 8 aligned to 16
mov  [rbp-8],  rdi      ; arg 0 → local 0
mov  [rbp-16], rsi      ; arg 1 → local 1
; ...
; CTX ← rbp (so naive local-access code works correctly)
```

### Epilogue
```asm
pop  rax                ; result from WASM stack → rax
leave                   ; mov rsp, rbp; pop rbp
ret
```

---

## RISC-V Linux SysV ABI (`crates/blitz-riscv64/src/sysv.rs`)

Follows the **RISC-V psABI** (LP64 variant).

### Entry registers (A0–A7 / x10–x17)
Up to 8 integer arguments in a0–a7; overflow on stack.

### Return registers
- 1 return value: a0 (x10)
- 2 return values: a0 + a1

### Prologue
```asm
addi sp, sp, -(frame_sz)
sd   ra, (frame_sz-8)(sp)
sd   s0, (frame_sz-16)(sp)
addi s0, sp, frame_sz   ; FP ← old SP
sd   a0, -8(s0)         ; arg 0 → local 0
sd   a1, -16(s0)        ; arg 1 → local 1
; ...
```

### Epilogue
```asm
ld   a0, 0(sp)           ; result from WASM stack → a0
ld   ra, (frame_sz-8)(s0)
ld   s0, (frame_sz-16)(s0)
addi sp, sp, frame_sz
ret                       ; jalr x0, ra
```

---

## AAPCS64 / AArch64 SysV ABI (`crates/blitz-aarch64/src/sysv.rs`)

Follows **AAPCS64** (Procedure Call Standard for the Arm 64-bit Architecture).

### Entry registers (X0–X7)
Up to 8 integer arguments in x0–x7; overflow on stack.

### Return registers
- 1 return value: x0
- 2 return values: x0 + x1

### Prologue
```asm
stp  x29, x30, [sp, #-16]!
mov  x29, sp
sub  sp, sp, #frame_sz
str  x0, [x29, #-8]     ; arg 0 → local 0
str  x1, [x29, #-16]    ; arg 1 → local 1
; ...
```

### Epilogue
```asm
ldr  x0, [sp]            ; result from WASM stack → x0
mov  sp, x29
ldp  x29, x30, [sp], #16
ret
```

---

## Imports and Exports

### Import convention

When a WASM module imports a function `(module, name)`, blitz generates a call to
an external symbol named `{module}__{name}` (double-underscore separator).

**Assembly backends** (x86-64, RISC-V, AArch64):
- The external symbol must be a function that follows the **blitz WASM ABI** for that
  architecture (i.e. it reads/writes the hardware operand stack in the same way as
  internal functions).
- The symbol is loaded via a label reference (`lea_label` / `jal_label` / `adr_label`)
  then called via `call` / `jal(Reg(10))` / `bl`.
- The `External { name: String }` label variant carries the symbol name.

**C backend**:
- `c_emit_import_decls(w, imports, sigs, fsigs)` emits a `__sig_N` struct and a
  function-pointer variable `fn_N` for each import (N = WASM function index).
- The caller must set `fn_N = <host_impl>;` before invoking any WASM function that
  calls the import. The host implementation must follow the blitz C ABI:
  `uint64_t* host_fn(uint64_t* restrict args)`.

**JS backend**:
- `js_emit_imports(w, imports)` emits `var $N;` for each import.
- The caller must assign `$N = hostFunction;` before calling. The host function must
  have `Object.defineProperty($N, '__sig', { value: { params: P, rets: R } })` set,
  and return `[...results]`.

### Export convention

**Assembly backends**:
- `emit_export_dispatchers(w, ctx, arch, exports)` emits a one-instruction stub per
  export: `External { name: export_name }` label + unconditional jump to
  `Func { fn: internal_id }`.
- `exports` is a list of `(internal_id, export_name)` where `internal_id` is the WASM
  function index minus the import count.

**C backend**:
- `c_emit_exports(w, exports)` emits an alias function:
  `uint64_t* <name>(uint64_t* restrict __in) { return fn_N(__in); }`
- `exports` is a list of `(wasm_function_index, name)` where `wasm_function_index`
  includes the import count.

**JS backend**:
- `js_emit_exports(w, exports)` emits `var <name> = $N;` for each export.
- Same index convention as C: full WASM index including imports.

### Symbol naming example

Given a module with:
- Import: `("env", "log")` → assembly symbol `env__log`, C/JS variable `fn_0` / `$0`
- Internal function 0 (WASM index 1) exported as `"run"` → assembly label `run`,
  C alias `uint64_t* run(...)`, JS alias `var run=$1;`

---

## Exception Handling ABI

Implements the [WebAssembly Exception Handling proposal](https://github.com/WebAssembly/exception-handling).

### Supported instructions

| Instruction | Status |
|-------------|--------|
| `throw { tag_index }` | Implemented (all backends) |
| `try_table { catches }` with `Catch::One` | Implemented |
| `try_table { catches }` with `Catch::All` | Implemented |
| `throw_ref` | Deferred — see **exnref deferral** below |
| `Catch::OneRef` / `Catch::AllRef` | Deferred — see **exnref deferral** below |

---

### JS backend (`crates/blitz-js/src/lib.rs`)

Exception handling maps directly to JavaScript `try`/`catch`/`throw`:

**`throw { tag_index }`** (arity = number of tag params):
```javascript
throw {__wasm_tag: <tag_index>n, __wasm_vals: [<pop arity values>]};
```

**`try_table { catches }`** (block label `lN`):
```javascript
lN: try {
  /* body */
} catch(__wasm_e) {
  // Catch::One { tag, label }:
  if (__wasm_e?.__wasm_tag === <tag>n) { <push vals>; {break l<label>;} }
  // Catch::All { label }:
  { {break l<label>;} }
  throw __wasm_e;  // no match — rethrow
}
```

`Frame::TryTable` acts like `Frame::Block` for `br`: branching out emits `break lN`.

---

### C backend (`crates/blitz-c/src/lib.rs`)

Uses POSIX `setjmp`/`longjmp` for non-local control flow. Module-level globals
(emitted by `c_module_preamble`):

```c
#include <setjmp.h>
typedef struct { uint32_t tag; uint64_t vals[64]; int nvals; } __wasm_exn_t;
static __wasm_exn_t __wasm_exn;           /* current in-flight exception */
static jmp_buf __wasm_exn_jmp[64];        /* handler stack */
static int __wasm_exn_d = -1;             /* active handler depth (-1 = none) */
```

**`throw { tag_index }`** (arity N):
```c
__wasm_exn.tag = <tag>; __wasm_exn.nvals = N;
__wasm_exn.vals[0] = <pop>; /* ... for each value */
if (__wasm_exn_d >= 0) longjmp(__wasm_exn_jmp[__wasm_exn_d], 1);
abort();  /* no handler — trap */
```

**`try_table { catches }`** (exit label `blk_e_N`):
```c
{ __wasm_exn_d++;
  if (!setjmp(__wasm_exn_jmp[__wasm_exn_d])) {
    /* body */
    __wasm_exn_d--;   /* normal exit */
  } else { __wasm_exn_d--;
    /* Catch::One { tag, label }: */
    if (__wasm_exn.tag == <tag>) { <push vals>; goto blk_e_<label>; }
    /* Catch::All { label }: */
    { goto blk_e_<label>; }
    if (__wasm_exn_d >= 0) longjmp(__wasm_exn_jmp[__wasm_exn_d], 1); abort();
  }
  blk_e_N: ; }
```

`br` targeting a `TryTable` frame emits `__wasm_exn_d--; goto blk_e_N;` to
clean up the handler depth before jumping out of the try body.

**Important:** `catch_all` must target a label whose block type is `Empty`
(provides 0 values). Targeting a block with a non-empty result type is a WASM
type error. Use an outer `Block(Empty)` wrapper and push the result value after
both blocks exit.

---

### NaiveAbi native backends (x86-64, AArch64, RISC-V 64)

Exception handling for the NaiveAbi custom calling convention uses the **CTX
stack** — the same mechanism used for control flow block frames. Platform
unwinding (`_Unwind_RaiseException` / DWARF) is **not** used; see *SysVAbi
deferral* below.

#### CTX stack layout for `try_table`

On `try_table` entry, three words are pushed onto the CTX stack (after
`xchg RSP ↔ CTX`):

```
[CTX+0]  dispatch_label_addr   ← address of the exception dispatch stub
[CTX+8]  old_RSP               ← operand stack to restore when catch fires
[CTX+16] TRYTABLE_SENTINEL     ← 0xE4C3_E4C3_E4C3_E4C3 (identifies TryTable frames)
```

#### `throw` — NaiveAbi

1. Pop `arity` values from the operand stack into scratch registers (Reg(3)..Reg(2+arity)).
2. Save `tag_index` into a context-relative slot.
3. Walk the CTX stack backward looking for a `TRYTABLE_SENTINEL` frame.
4. On match: restore `RSP` from `old_RSP`, jump to `dispatch_label_addr`.
5. If CTX stack exhausted: load the **caller's saved CTX** (stored at a fixed
   offset in the current frame base) and continue scanning — this is the
   **cross-function propagation** path.
6. If the root frame is reached with no handler: call `__wasm_unhandled_exception`
   (traps with `ud2` / `unimp` / `ebreak`).

#### `try_table` start — NaiveAbi

```asm
lea_label r0, <dispatch_idx>   ; dispatch handler address
mov r2, RSP                     ; save current operand stack pointer
xchg RSP, CTX
push TRYTABLE_SENTINEL
push r2                          ; old_RSP
push r0                          ; dispatch label
xchg RSP, CTX
<exit_label>:                    ; fall-through to body
```

#### `try_table` end — NaiveAbi

Normal exit (no exception): pops the three CTX slots and falls through to the
exit label (same pattern as `Block`):

```asm
xchg RSP, CTX
pop r0   ; dispatch label (discard)
pop r1   ; old_RSP (discard)
pop r0   ; TRYTABLE_SENTINEL (discard)
xchg RSP, CTX
<exit_label>:   ; placed here
```

#### Exception dispatch stub — NaiveAbi

Placed at `dispatch_idx` label. Compares the saved tag index against each
`Catch::One { tag }` clause; on match, restores the operand stack (from
`old_RSP`), pushes exception values, and jumps to the catch target label via
the standard `br`-style CTX restore + jump. `Catch::All` always matches.

---

### SysVAbi deferral

SysVAbi exception handling is **not implemented**. It requires DWARF `.eh_frame`
personality routines and `_Unwind_RaiseException` integration, which is
non-trivial and architecturally independent work.

All `BackendAbi` exception methods on SysVAbi variants panic with:
```
todo!("SysVAbi exception handling requires platform unwinder — deferred; see docs/abi.md")
```

---

---

## Probes

### Overview

**Probes** are the one general mechanism wasm-blitz uses for every out-of-line callout
into a function: lightweight instrumentation, debugger/profiler hooks, and the
specialization opt-entry handoff that used to be called "tracing" are all just
different configurations of the same underlying machinery. A probe:

- never disturbs the function's existing stack-frame/locals layout,
- is dispatched per a compile-time [`ProbeBinding`]:
  - `TailTakeover` — an unconditional indirect jump that permanently hands off control
    (never returns to the probe site). This is what the old "tracing preamble" did, and
    is what specialization opt-entry still uses.
  - `Call` — a real call-and-return through a minimal, register-only ABI (see *Probe
    ABI contract* below). Execution resumes at the probe site once the handler returns.
- preserves live state across a `Call` per a compile-time [`ProbeMode`]:
  - `Active` — force the canonical materialized layout first (flush a backend's
    register allocator, if it has one). The only valid mode for `TailTakeover`.
  - `Passive` — save/restore only the registers a backend's allocator currently has
    live, without forcing a flush. Identical to `Active` on backends with no allocator
    (x86-64, AArch64); meaningfully cheaper on RISC-V, which does have one.
- fires at a compile-time [`ProbePlacement`] (`Before`/`After`) relative to the
  instruction it's attached to.

An embedder describes all of this as a [`ProbePlan`] (`blitz-common/src/ops.rs`),
carried into codegen via `FnData.probe_plan`, addressed by **ordinal instruction
index** within the function body (0 = the first real WASM operator after
locals/`StartBody`) rather than WASM bytecode offset — the streaming `mach_operators`
pipeline only threads a byte offset through when the embedder picks an `Annot` that
carries one, while the ordinal index is always available to a backend's dispatcher
regardless of `Annot`. `ProbePlan::control_flow_sites(&body)` reproduces today's
auto-identified site set (function entry + every `Block`/`Loop` header,
`TailTakeover`/`Active`) as data, proving that configuration is fully expressible as
one instance of the general model — see *Specialisation + deopt* below for how that
configuration is actually still emitted (via dedicated codegen, not yet by consuming a
`ProbePlan` directly).

### Compile/runtime separation — CTX-relative probe table

**No runtime address is baked into the generated code.**  Instead the runtime owns a
contiguous `[ProbeSlot]` table per instrumented function and the code reaches it
indirectly:

```rust
#[repr(C)]
pub struct ProbeSlot {       // blitz-common/src/ops.rs
    pub counter: u64,            // +0 — approximate hit counter
    pub handler: *const (),      // +8 — null = disabled (skip)
}

pub struct ProbeTableConfig {
    pub enabled: bool,
    pub num_probes: u32,         // sizes the runtime's [ProbeSlot] table
    pub table_base_off: i32,     // CTX-relative slot holding the table base ptr
}
```

The runtime writes the table's base pointer into a fixed, CTX-relative slot at
`table_base_off` before the first guest entry.  Compilation emits only that structural
offset and a per-probe `probe_id`; the entry for a probe is `base + probe_id * 16`
(`PROBE_SLOT_SIZE`), with the counter at `+0` and the handler pointer at `+8`
(`PROBE_SLOT_HANDLER_OFF`).  This lets a module be compiled in one process/address
space and run in another. This layout, and the table's runtime contract, are
unchanged from the original "tracing" design — only the names generalised.

### Emitted instruction sequence (pseudo-code), per probe

```
   load  scratch, [CTX + table_base_off]        ; runtime probe-table base
   inc   qword [scratch + probe_id*16 + 0]       ; counter++ (non-atomic)
   load  scratch, [scratch + probe_id*16 + 8]    ; handler ptr
   if scratch == 0: jump body                    ; null → run baseline, zero overhead
   ; ProbeBinding::TailTakeover:
   jump  scratch                                 ; non-null → tail-jump, never returns
   ; ProbeBinding::Call:
   call  scratch                                  ; non-null → call, falls through after
body:
   ; ... baseline body continues ...
```

This is `emit_probe_site(w, base_off, probe_id, scratch, binding, label_counter)` in
`blitz-codegen`, built on the `BlitzWriter` primitives `load_probe_base`,
`inc_mem64_disp`, `load_mem64_disp`, `branch_reg` (`TailTakeover`), and `call_reg`
(`Call`).

### Probe ABI contract (`Call` binding)

A `Call`-bound probe handler is an ordinary function under a minimal, **register-only**
calling convention: no stack arguments or results. This is what makes splicing a call
in always safe regardless of what the surrounding code is using the stack for — the
call/`ret` pair nets to zero effect on the stack pointer, and the callee only ever
touches memory below its own call-time stack pointer (exactly like calling any ordinary
leaf function). The handler must never clobber:

- the CTX/frame pointer register,
- the real stack pointer (safe by construction, as above),
- any ABI virtual parameter currently live in a register at that point (e.g. the SysV
  probe-base register before it's spilled — already excluded from every probe site's
  own scratch-register choice, so the same exclusion applies here for free).

This is a convention, not new code: the existing per-backend scratch-register choices
for probe sites already respect it (e.g. `crates/blitz-x86-64/src/naive.rs` uses
`Reg(2)`/RDX, never CTX/RSP/SCR).

### Active vs Passive mode

x86-64 and AArch64 fully materialise the WASM operand stack to memory between every op
(a pure stack-machine codegen style), so there is no backend register state alive
across an op boundary other than the pinned registers above — `Active` and `Passive`
are identical there, and a probe can always use plain registers as scratch with no
extra save/restore.

RISC-V 64 is the exception: it runs a real lazy register allocator
(`portal_solutions_asm_regalloc`, sibling repo `asm-arch/crates/asm-regalloc`) that
keeps operand-stack values resident in physical registers across multiple ops, only
spilling (`RegAlloc::flush()`) at branches and control-flow probe sites. For a `Call`
probe on RISC-V:

- **Active** (`emit_control_flow_probe`, used for all control-flow sites) flushes the
  allocator first — the same `ralloc.flush()` + `ralloc.tos = None` sequence used
  today, materialising the operand stack to its canonical layout. Required for
  `TailTakeover` (the tail target assumes that layout).
- **Passive** (`naive::emit_passive_call_probe`) does **not** flush. It reads which
  physical registers the allocator currently has occupied directly off its public
  `frames`/`tos` bookkeeping (no new query was added to `asm-regalloc` itself — `Target`,
  `RegAllocFrame`, and `Cmd` are all already public), then `push`/`pop`s exactly those
  registers around the call (reusing the allocator's own `Cmd::Push`/`Cmd::Pop` codegen,
  popped in reverse for correct LIFO discipline) — *without* touching the allocator's
  `frames`/`tos` state, so codegen continues exactly where it left off. This is what
  makes a probe genuinely non-disturbing even when dropped into the middle of an
  expression: the allocator never even notices the probe ran, including when the probe
  site's own scratch register happens to collide with a register the allocator
  currently has a live value in (the save/restore handles that case transparently).

`crates/blitz-tests/tests/e2e.rs::test_passive_call_probe_preserves_regalloc_riscv64`
exercises exactly this collision under Unicorn: it allocates a live register (which
lands on the same register `emit_passive_call_probe` uses as its own scratch), runs a
`Call`-bound probe, and asserts the live value survives untouched.

**Active mode's real constraint on RISC-V — at most one pending value.** `flush()`
resets the allocator to "nothing live" (`tos = None`, all frames `Empty`), and the
*first* subsequent `pop()` correctly falls back to popping off the native stack — but
because that first fallback pop sets `tos = Some(..)` again, a *second* consecutive
`pop()` sees a `tos` it mistakes for an already-resident value and returns the same
register without emitting another native-stack pop. This is harmless at control-flow
headers, where WASM's validation rules guarantee at most one value is live across the
boundary (matching what `emit_control_flow_probe` has always assumed), but it means
Active mode is **not valid** at an arbitrary mid-expression point with two or more
pending operands (e.g. probing immediately before a binary op, with both operands
already evaluated). Passive mode has no such restriction — it never touches
`tos`/`frames`, so it's the correct choice for points like that. This was found, not
designed: `test_indexed_call_probe_fires_inside_loop_riscv64_active`/`_passive`
(`crates/blitz-tests/tests/e2e.rs`) probe the same loop body at two points — one with a
single pending value (valid for Active) and one with two (only valid for Passive) — to
pin down exactly where the line is. It's a constraint on `pop()`/`flush()`'s interaction
in `asm-regalloc`, not something this layer works around.

### Arbitrary insertion points (`ProbePlan`)

Beyond the auto-identified control-flow set, an embedder can place `ProbeSpec`s at any
instruction's ordinal index via `ProbePlan.by_index: BTreeMap<usize, Vec<ProbeSpec>>`
(`Before`/`After`), in addition to (or instead of) the `entry` probe.  This is wired into
the SysV dispatcher of all three architectures — `sysv_emit_indexed_probes` plus
`State::{probe_plan, op_index}` (`SysVState` on x86-64; the shared `naive::State` on
AArch64/RISC-V, since those two already delegate SysV instruction dispatch to the naive
lowering) in each backend's `crates/blitz-*/src/sysv.rs`: every
`MachOperator::Instruction`/`Operator` dispatch checks the plan for entries at the
current `op_index` before/after running the instruction's own codegen, reusing the same
mid-function probe-base addressing (`ProbeBase::FrameSlot`) the control-flow probes
already establish. RISC-V's version additionally branches on `ProbeSpec::mode`
(`Active`/`Passive`) per probe — see the previous section for what that means and its
one real constraint. AArch64/x86-64 ignore `mode` (no allocator to disturb either way).
The NaiveAbi/LFI dispatchers don't yet consume `ProbePlan` — the pattern to mirror there
is identical, just at the naive `handle_op`/`_handle_op` entry point instead of the SysV
one.

**Important:** indices are positions in the stream actually fed to the dispatcher.
If a pass like `dce_pass!` removes operators between building a `ProbePlan` and
compiling, indices must be computed against that *same* post-pass stream, not the raw
`get_operators_reader()` output — `ProbePlan::control_flow_sites` and any embedder
building a plan by hand should keep this in mind.

`crates/blitz-tests/tests/e2e.rs::test_indexed_call_probe_fires_inside_loop_{x86_64,
aarch64,riscv64_active,riscv64_passive}` place a `Call`-bound probe on a plain `I32Add`
(or the instruction right after it, for the RISC-V Active case — see above) inside a
loop body — not a control-flow header — on every architecture, and verify it fires once
per iteration without disturbing the loop's result, alongside the existing entry/loop
control-flow probes still firing correctly in the same function.

### Specialisation + deopt (`crates/blitz-specialize`)

The `blitz-specialize` crate builds specialised variants to install into a
[`ProbeSlot`]'s `handler` slot, behind a `TailTakeover`-bound probe at the entry/loop/
block sites `ProbePlan::control_flow_sites` describes:

- `FnSlice` + `BranchAnalysis` give a slice-resident (random-access) view of one
  function with every relative branch depth resolved to an absolute index, and map each
  `Block`/`Loop` header to its `probe_id`.
- `specialize(slice, analysis, SpecSpec)` substitutes embedder-asserted constant locals
  / globals (value specialisation) and folds loads from regions asserted constant
  (memory specialisation), returning the rewritten body plus the `Guard`s it depends on.
- `emit_deopt_guard(w, diff_reg, deopt_label, label_counter)` emits the guard branch:
  the caller materialises `diff_reg` to be zero iff the assumption holds; the guard
  falls through when it holds and branches to `deopt_label` (the **generic site entry**)
  otherwise.  Deopt fires whenever any guard fails — i.e. whenever the specialised code
  would be invalid. `emit_deopt_guard` is a bare branch with no callout, so it is not
  itself a probe and is unaffected by any of the above.

### Architecture / ABI specifics

| Backend   | ABI     | Scratch reg(s) | Site emission |
|-----------|---------|----------------|---------------|
| x86-64    | NaiveAbi | Reg(2) (RDX)  | entry in `StartBody`; loop/block via `emit_control_flow_probe` |
| x86-64    | SysVAbi  | Reg(0) (RAX)  | entry + loop/block via `sysv_emit_control_flow_probe`; arbitrary indices via `sysv_emit_indexed_probes` |
| AArch64   | NaiveAbi | x9+x10 (T0/T1)| entry in `StartFn`; loop/block via `emit_control_flow_probe` |
| AArch64   | SysVAbi  | x9+x10 (T0/T1)| entry + loop/block (delegated naive `emit_control_flow_probe`); arbitrary indices via `sysv_emit_indexed_probes` |
| RISC-V 64 | NaiveAbi | t0+t1 (Reg 5/6)| entry in `StartFn`; loop/block via `emit_control_flow_probe` (Active); arbitrary `Call` probes via `emit_passive_call_probe` (Active or Passive) |
| RISC-V 64 | SysVAbi  | t0+t1 (Reg 5/6)| entry + loop/block (delegated naive `emit_control_flow_probe`); arbitrary indices via `sysv_emit_indexed_probes` (Active or Passive per `ProbeSpec::mode` — see *Active vs Passive mode* for Active's one constraint) |

On the **NaiveAbi** path (`naive.rs`, also used by the LFI ABI) the base is the CTX
frame pointer.  All three **SysVAbi** backends support mid-function sites via the
virtual-parameter convention below.

**SysVAbi probe-base — the virtual function parameter**:
the SysV frame is not set up at the function-entry preamble and the NaiveAbi CTX frame
pointer does not exist, so the probe-table base is passed as a **virtual function
parameter** in a reserved register (`PROBE_BASE_REG` in each backend's `sysv.rs`):

| Arch     | Virtual-param reg | Frame-pointer for the spill slot |
|----------|-------------------|----------------------------------|
| x86-64   | `r11`             | `rbp` (bottom frame slot)        |
| AArch64  | `x12`             | `x29`/FP (bottom frame slot)     |
| RISC-V 64| `t2` (x7)         | `s0`/fp (`[sp+16]` slot)         |

Each is caller-saved and never a positional argument register (nor the probe-preamble
scratch), so the runtime can set it without disturbing the function's real arguments.

- **Probe 0 (function entry)** reads the base directly from the virtual-param register,
  before any frame is built, so the specialization tail-jump still delivers the SysV
  argument registers intact (`codegen::ProbeBase::Reg`).
- **`StartFn`** then spills it to a dedicated extra frame slot and records the FP-relative
  displacement.  For the x86-64 ABI this lives in `SysVState::probe_base_disp`; the
  AArch64/RISC-V SysV backends reuse the **naive lowering** and configure it via the
  `naive::State::probe_base` field (`codegen::ProbeBase::FrameSlot`).
- **Mid-function sites (loop/block, and arbitrary indices on x86-64)** reload the base
  from the frame slot since the virtual-param register is clobbered by the body.  A
  mid-function `TailTakeover` stub is entered with the operand stack live and the frame
  set up, so to return from the whole function it must tear the frame down itself (on
  x86-64, `mov rsp,rbp; pop rbp; ret`); a `Call`-bound probe just needs an ordinary
  `ret`, per the *Probe ABI contract* above.

The *table layout* (`ProbeSlot`, `probe_id` indexing) is identical across all ABIs; only
the base-load mechanism differs (`codegen::ProbeBase::{CtxSlot, Reg, FrameSlot}`).
The runtime contract is exercised end-to-end for **all three arches** in the e2e suite
under Unicorn (`run_native_sysv_traced`): it installs a zeroed `[ProbeSlot]` table,
passes its base in the virtual-param register, and verifies per-probe counter increments
(including the mid-function loop probe) and entry-probe `TailTakeover` tail-jumps; the
x86-64 tests additionally cover a loop-probe specialization with frame teardown, a
`Call`-bound probe at function entry (`test_call_probe_*`), and the arbitrary-index/
Passive-mode tests described above.

### Stack-state contract

The `TailTakeover` tail-jump is a bare indirect branch with the operand-stack /
CTX-frame state of the **generic probe site** intact, so a specialised variant must
preserve that layout and its deopt target is the generic site entry (a plain branch
back, live state untouched).  Per-site live state:

- **Function entry (probe 0)** — as before frame setup:
  - **NaiveAbi (x86-64)**: Reg(0) = old CTX, Reg(1) = return address, CTX = frame ptr,
    RSP = WASM operand stack.
  - **SysVAbi (x86-64)**: SysV arg registers (RDI/RSI/RDX/RCX/R8/R9) intact, no frame.
  - **AArch64 (both ABIs)**: LR = return addr, SP = caller/WASM stack, X0–X7 intact.
  - **RISC-V 64 (both ABIs)**: RA = return addr, A0–A7 intact, no frame.
- **Loop/block sites** — the CTX frame for the site has been pushed and the operand
  stack is materialised (RISC-V flushes regalloc first); the specialised variant inherits
  exactly that state.  Because `specialize` only substitutes values (never changing stack
  shape), layout compatibility holds by construction.

A `Call`-bound probe has a much weaker contract by design (see *Probe ABI contract*
above): it only needs the pinned registers preserved, not the full generic-site layout,
which is exactly what makes it usable at arbitrary mid-expression points where the
"generic site entry" layout doesn't even apply.

[`ProbeBinding`]: ../crates/blitz-codegen/src/lib.rs
[`ProbeMode`]: ../crates/blitz-common/src/ops.rs
[`ProbePlacement`]: ../crates/blitz-common/src/ops.rs
[`ProbePlan`]: ../crates/blitz-common/src/ops.rs
[`ProbeSlot`]: ../crates/blitz-common/src/ops.rs

### exnref deferral

The `exnref` type and associated instructions are **not implemented**:

- `throw_ref` → `todo!("exnref deferred")`
- `Catch::OneRef` → `todo!("exnref catch deferred")`
- `Catch::AllRef` → `todo!("exnref catch deferred")`

These require first-class `exnref` values on the WASM stack, which in turn
need reference type support in the value representation. Deferred until
reference types land in the type system.

