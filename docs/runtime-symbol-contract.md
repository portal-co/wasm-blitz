# blitz runtime-symbol contract (verified against the busybox megabinary)

A blitz AllStack-compiled module with memories / globals / data segments /
a function table / wasi imports references a fixed set of **external symbols**.
A consumer that JITs or links the emitted code (e.g. speet's
`compile_wasm_to_object` driver) must provide every one of these, with the
semantics below. This contract was verified by running the full 227k-function
busybox megabinary under the Unicorn JIT harness
(`megabinary_unicorn_watch_sp` in `crates/blitz-tests/tests/e2e.rs`), which
executes `busybox echo hi` to a correct `fd_write` of `"hi\n"` and
`proc_exit(0)`.

## Memories

For a module with N memories, blitz emits one **base cell** symbol per memory
(`blitz-aarch64/src/naive.rs:651`): memory 0 uses `__wasm_mem`, memory `k`
uses `__wasm_mem_{k}`. Each cell is 8 bytes holding the **host base address**
of that memory's linear region. Every load/store reads the cell fresh and adds
the (32-bit-wrapped) guest address.

- `__wasm_mem` — `*mut u8` cell: base of memory 0 (the guest's data image).
- `__wasm_mem_{k}` — `*mut u8` cell: base of memory `k`.
- `__wasm_mem_pages` — `u32` cell: memory 0's current page count (64KiB pages).
  Read by `memory.size`; updated by `__wasm_memory_grow`.
- `__wasm_mem_pages_{k}` — `u32` cell: memory `k`'s current page count. Read by
  `memory.size k`; updated by the memory-`k` grow helper.

Multi-memory routing is per-access by `memory_index`, so each memory must be a
distinct region. (In the megabinary: mem0 = guest-private, mem1 = the
os-emulation/host-shared memory where the wasi iovec is staged, mem2 =
handler-statics. The wasi import implementations must read the iovec from
**mem1**, not mem0 — getting this wrong surfaces as reading the debug syscall
ring's `x8` value at `mem0[0x200]`.)

## Globals

- `__wasm_global_{k}` — an 8-byte cell holding global `k`'s current value.
  `global.get`/`global.set` load the cell address and read/write through it.
  Initialize each cell to the global's wasm init value.

## Data segments (for `memory.init`)

- `__wasm_data_seg_{d}` — the **raw bytes** of passive data segment `d`, placed
  at the symbol's address (the symbol IS the data, not a pointer to it).
  `memory.init` passes this address as the source to the init helper.

## Function table

- `__wasm_table` — the `funcref` table: an array of 8-byte host code addresses,
  indexed by the guest's computed table index. `call_indirect` /
  `return_call_indirect` do `ldr target, [__wasm_table + idx*8]; br/blr`. The
  consumer fills it (the megabinary uses an identity map of wasm-func-index →
  host address for the populated range).

## Runtime helper functions (AAPCS64, called via the symbol address)

These are *callable*, not data. Give each a real implementation:

- `u32 __wasm_memory_grow(u32 delta, u8** mem_cell, u32* pages_cell)` —
  `memory.grow` on memory 0. On success: make `*pages_cell += delta` bytes of
  the region usable, update `*mem_cell` if the base moved, return the OLD page
  count. On failure return `u32::MAX` (`-1`).
  - **aarch64** marshals X0=delta, X1=&`__wasm_mem[_k]`, X2=&`__wasm_mem_pages[_k]`
    and always calls the one `__wasm_memory_grow` symbol — the cells, not the
    symbol, select the memory.
  - **x86-64 / riscv64 / riscv32** call the helper with the WASM-internal
    calling convention (delta on the WASM stack, no cell arguments), so the
    memory index selects the *symbol*: memory 0 calls `__wasm_memory_grow`,
    memory `k` calls `__wasm_memory_grow_{k}`.
- `void __wasm_memory_init_copy[_k](u32 dest_off, u8* seg_base, u32 src_off, u32 len)` —
  `memory.init` on memory `k` (memory 0 keeps the unprefixed name). Copies
  `len` bytes from `seg_base + src_off` to `mem_k_base + dest_off`.
- `void __wasm_memory_copy[_dst][_src](u32 dest, u32 src, u32 len)` —
  `memory.copy` from memory `src` to memory `dst`; the 0→0 case keeps the
  unprefixed name (e.g. `__wasm_memory_copy_2_1` copies mem1 → mem2).
- `void __wasm_memory_fill[_k](u32 dest, u32 val, u32 len)` — `memory.fill` on
  memory `k` (memory 0 keeps the unprefixed name).

## Imports

- `{module}__{name}` (double underscore) — each wasm import, e.g.
  `wasi_snapshot_preview1__fd_write`. Called with the AAPCS64 marshalling the
  blitz import path emits (args in X0–X7 then stack for the megabinary's
  `is_import` path).

## Object-file note

With `AArch64Writer::set_object_mode(true)`, internal-label `ADRP`+`ADD`
pairs surface as `A64AdrpPage21`/`A64AddAbsLo12` relocations for the linker to
resolve against final vaddrs (required because a linked `__text` may be
non-page-aligned). The data symbols above surface as ordinary external relocs
the same way whether or not object mode is on.
