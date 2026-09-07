//! Native (Unicorn) execution runtime — phase A of `docs/native-spectests-plan.md`.
//!
//! Compiles a WASM module through the AllStack-flavoured SysV backends into a
//! self-contained machine-code blob (x86-64, aarch64, riscv64 — all binary
//! writers, no cross-clang) and executes exported functions under Unicorn.
//!
//! Blob layout (one buffer, loaded at [`CODE_BASE`]):
//!
//! ```text
//!   [ module code ]   emitted by the backends
//!   [ trampolines ]   (x86-64/riscv64) jmp / li+jalr to a sentinel slot
//!   [ data area ]     raw bytes appended post-assembly:
//!                     __wasm_mem_pages (u32), __wasm_mem (ptr),
//!                     __wasm_globals (1024+ u64 slots)
//! ```
//!
//! External-symbol resolution, per arch:
//!
//! - **x86-64** (`IcedWriter`): rel32 fixups resolve in-buffer via
//!   `set_label`. Imports and unimplemented runtime calls get a 5-byte `jmp`
//!   trampoline to their sentinel slot; data symbols get labels at nop runs
//!   that are truncated and replaced with real bytes afterwards.
//! - **aarch64** (`AArch64Writer`): ADRP/ADD pairs for `External` symbols are
//!   external-only fixups handed back as relocations (resolving in-buffer
//!   panics), so the harness patches them directly to the sentinel / data
//!   addresses. No trampolines.
//! - **riscv64** (`RvAsmWriter`): LA/JAL fixups resolve in-buffer via
//!   `set_label`, like x86-64 — trampolines are `li` + `jalr x0`.
//!
//! Import ABI (AllStack, matching `sysv_emit_marshalled_call`): the return
//! address is in `[rsp]` (x86-64 `call`), `x30` (aarch64 `bl`), `ra` (riscv64
//! `jalr ra`); the first 6/8 args in registers (and still on the operand
//! stack below the return address), remaining args spilled above. The
//! sentinel hook reads args, calls the host closure, writes results to the
//! return register, and resumes at the return address.

use portal_solutions_blitz_common::HandleOpError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use unicorn_engine::Unicorn;

pub const CODE_BASE: u64 = 0x1000_0000;
const CODE_SIZE: u64 = 0x4_0000;
pub const STACK_BASE: u64 = 0x2000_0000;
const STACK_SIZE: u64 = 0x4_0000;
pub const MEM_BASE: u64 = 0x3000_0000;
const SENTINEL_BASE: u64 = 0x1100_0000;
const SENTINEL_SLOT: u64 = 16;
const MAX_SENTINEL_SLOTS: usize = 256;
const MAX_IMPORT_ARGS: usize = 8;

/// Target ISA for a native spectest run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeArch {
    X86_64,
    AArch64,
    Riscv64,
}

/// A compiled module blob plus everything the runner needs.
pub struct NativeBinary {
    pub code: Vec<u8>,
    /// Byte offset of the AllStack entry function (local fn 0).
    pub entry_off: u64,
    /// Byte offset of every local function (local fn i at fn_entries[i]).
    pub fn_entries: Vec<u64>,
    /// Sentinel slot → import index (`None` = trap sentinel).
    pub slot_imports: Vec<Option<usize>>,
    /// Blob offset of `__wasm_mem_pages` (u32), if memory is enabled.
    pub mem_pages_off: Option<usize>,
    /// Blob offset of `__wasm_mem` (pointer cell), if memory is enabled.
    pub mem_ptr_off: Option<usize>,
    /// Blob offset of `__wasm_globals`.
    pub globals_off: usize,
}

/// Why native compilation failed (caller decides skip vs bug).
#[derive(Debug)]
pub enum CompileError {
    /// Backend does not support some instruction / shape — skip, never silent.
    Unsupported(String),
    /// Emission infrastructure failure — a real bug.
    Internal(String),
}

/// Runtime data symbols (memory is read/written at the label address).
const DATA_SYMBOLS: &[&str] = &["__wasm_mem_pages", "__wasm_mem", "__wasm_globals"];
/// Runtime call symbols the native scope does not implement (trap sentinel).
const CALL_SYMBOLS: &[&str] = &[
    "__wasm_memory_grow",
    "__wasm_table",
    "__wasm_memory_init_copy",
    "__wasm_memory_copy",
    "__wasm_memory_fill",
    "__wasm_eh_push",
    "__wasm_exn_propagate",
];

/// Compile `wasm` to an AllStack machine-code blob.
///
/// `imports` are the module's function imports in order (index 0..n).
/// `mem_pages = Some(initial)` enables linear memory with `__wasm_mem`-
/// relative addressing; `global_inits` seeds `__wasm_globals` (WASM global
/// index → value; imported globals must be pre-resolved by the caller).
pub fn compile_module(
    wasm: &[u8],
    arch: NativeArch,
    imports: &[(String, String)],
    mem_pages: Option<u32>,
    global_inits: &[u64],
) -> Result<NativeBinary, CompileError> {
    use portal_solutions_blitz_common::ops::mach_operators;

    let internal = |e: String| CompileError::Internal(e);
    let mut sigs_wp: Vec<wasmparser::FuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| internal(e.to_string()))? {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for subtype in group.map_err(|e| internal(e.to_string()))?.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            subtype.composite_type.inner
                        {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(reader) => {
                for imp in reader {
                    let imp = imp.map_err(|e| internal(e.to_string()))?;
                    if let wasmparser::TypeRef::Func(ty_idx) = imp.ty {
                        fsigs.push(ty_idx);
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                fsigs.extend(reader.into_iter().flatten());
            }
            _ => {}
        }
    }

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let Ok(wasmparser::Payload::CodeSectionEntry(body)) = payload {
            bodies.push(body);
        }
    }

    // Validate import shapes the runtime ABI can service.
    for (i, &ti) in fsigs.iter().enumerate().take(imports.len()) {
        let ft = &sigs_wp[ti as usize];
        let (m, n) = &imports[i];
        if ft.results().len() > 1 {
            return Err(CompileError::Unsupported(format!(
                "import {m}::{n} has {} results (max 1)",
                ft.results().len()
            )));
        }
        if ft.params().len() > MAX_IMPORT_ARGS {
            return Err(CompileError::Unsupported(format!(
                "import {m}::{n} has {} params (max {MAX_IMPORT_ARGS})",
                ft.params().len()
            )));
        }
    }

    let import_count = imports.len() as u32;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies,
        &fsigs,
        &sigs_wp,
        import_count,
    );

    let call_params: Vec<u32> = fsigs
        .iter()
        .map(|&ti| sigs_wp[ti as usize].params().len() as u32)
        .collect();
    let call_results: Vec<u32> = fsigs
        .iter()
        .map(|&ti| sigs_wp[ti as usize].results().len() as u32)
        .collect();
    let sig_params: Vec<u32> = sigs_wp.iter().map(|s| s.params().len() as u32).collect();
    let sig_results: Vec<u32> = sigs_wp.iter().map(|s| s.results().len() as u32).collect();
    let func_imports: Vec<(&str, &str)> = imports
        .iter()
        .map(|(m, n)| (m.as_str(), n.as_str()))
        .collect();

    // Sentinel slot allocation: imports first, then unimplemented call symbols.
    let mut targets: Vec<Option<usize>> = Vec::new();
    let mut slot_of: HashMap<String, usize> = HashMap::new();
    let mut alloc = |name: String,
                     slot_import: Option<usize>,
                     targets: &mut Vec<Option<usize>>,
                     slot_of: &mut HashMap<String, usize>|
     -> usize {
        if let Some(&k) = slot_of.get(&name) {
            return k;
        }
        let k = targets.len();
        assert!(k < MAX_SENTINEL_SLOTS, "sentinel slots exhausted");
        targets.push(slot_import);
        slot_of.insert(name, k);
        k
    };
    for (k, (m, n)) in imports.iter().enumerate() {
        alloc(format!("{m}__{n}"), Some(k), &mut targets, &mut slot_of);
    }
    for sym in CALL_SYMBOLS {
        alloc((*sym).to_string(), None, &mut targets, &mut slot_of);
    }

    let unsupported = |e: String| CompileError::Unsupported(e);
    let ops = portal_solutions_blitz_common::dce_pass!(raw_ops);

    match arch {
        NativeArch::X86_64 => {
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            use portal_solutions_asm_x86_64::out::{Writer as _, WriterCore as _};
            use portal_solutions_blitz_common::wasm_encoder::reencode::RoundtripReencoder;
            use portal_solutions_blitz_x86_64::{X64Arch, X64Label, naive, sysv};

            let mut out = IcedWriter::<X64Label>::new(CODE_BASE);
            let mut state = sysv::SysVState::default();
            state.call_abi = sysv::CallAbi::AllStack;
            state.call_params = call_params;
            state.call_results = call_results;
            state.sig_params = sig_params;
            state.sig_results = sig_results;
            state.n_imports = import_count;
            if mem_pages.is_some() {
                state.mem_base = naive::MemBase::WasmMemSymbol;
            }
            let mut ctx = ();
            let arch_cfg = X64Arch::default();
            let mut reencoder = RoundtripReencoder;
            for op in ops {
                let op = op.map_err(|e| internal(format!("mach op: {e}")))?;
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out,
                    &mut ctx,
                    arch_cfg,
                    &mut state,
                    &func_imports,
                    &op,
                    &mut reencoder,
                    0,
                )
                .map_err(|e| unsupported(format!("x86-64 emit: {e:?}")))?;
            }

            // Trampolines: 5-byte `jmp rel32` per sentinel slot; the symbol's
            // label points at the trampoline so in-buffer fixups resolve.
            for (name, &k) in slot_of.iter() {
                if k == usize::MAX {
                    continue;
                }
                let here = out.offset();
                let target = SENTINEL_BASE + (k as u64) * SENTINEL_SLOT;
                let rel = target as i64 - (CODE_BASE + here as u64 + 5) as i64;
                let mut b = [0u8; 5];
                b[0] = 0xE9;
                b[1..].copy_from_slice(&(rel as i32).to_le_bytes());
                out.set_label(
                    &mut ctx,
                    arch_cfg,
                    X64Label::External { name: name.clone() },
                )
                .map_err(|e| internal(e.to_string()))?;
                out.db(&mut ctx, arch_cfg, &b)
                    .map_err(|e| internal(e.to_string()))?;
            }
            // Data symbols: label at an 8-byte-aligned nop run (replaced with
            // real bytes after assembly by truncating at the first one).
            for sym in DATA_SYMBOLS {
                while out.offset() % 8 != 0 {
                    out.db(&mut ctx, arch_cfg, &[0x90])
                        .map_err(|e| internal(e.to_string()))?;
                }
                out.set_label(
                    &mut ctx,
                    arch_cfg,
                    X64Label::External {
                        name: (*sym).to_string(),
                    },
                )
                .map_err(|e| internal(e.to_string()))?;
                for _ in 0..8 {
                    out.db(&mut ctx, arch_cfg, &[0x90])
                        .map_err(|e| internal(e.to_string()))?;
                }
            }

            let (bytes, labels, relocs) = out.into_parts_with_relocs();
            if let Some(r) = relocs.first() {
                return Err(unsupported(format!("unresolved external symbol {r:?}")));
            }
            // Collect per-function entries: fn 0..n are labeled Func{fn}.
            let n_fns = (0..)
                .map_while(|i| labels.get(&X64Label::Func { r#fn: i }).copied())
                .count() as usize;
            let mut fn_entries = Vec::with_capacity(n_fns);
            for i in 0..n_fns {
                let off = *labels
                    .get(&X64Label::Func { r#fn: i as u32 })
                    .ok_or_else(|| internal(format!("missing fn {i} label")))?;
                fn_entries.push(off as u64);
            }
            let entry = *labels
                .get(&X64Label::Func { r#fn: 0 })
                .ok_or_else(|| internal("missing entry label".into()))?;

            // Truncate the nop-runs and append the real data area.
            let mut data_start = bytes.len();
            for sym in DATA_SYMBOLS {
                let off = *labels
                    .get(&X64Label::External {
                        name: (*sym).to_string(),
                    })
                    .ok_or_else(|| internal(format!("missing {sym} label")))?
                    as usize;
                data_start = data_start.min(off);
            }
            let globals_words = 1024usize.max(global_inits.len().next_multiple_of(1024));
            let (databytes, mem_pages_off, mem_ptr_off, globals_off) =
                data_area(data_start, mem_pages, global_inits, globals_words);
            let mut code = bytes;
            code.truncate(data_start);
            code.extend_from_slice(&databytes);

            Ok(NativeBinary {
                code,
                entry_off: entry as u64,
                fn_entries,
                slot_imports: targets,
                mem_pages_off,
                mem_ptr_off,
                globals_off,
            })
        }
        NativeArch::AArch64 => {
            use portal_solutions_asm_aarch64::out::arg::{ArgKind, MemArgKind};
            use portal_solutions_asm_aarch64::out::{
                Writer as _, WriterCore as _,
                bin::{AArch64Writer, AsmRelocKind},
            };
            use portal_solutions_blitz_aarch64::{AArch64Arch, AArch64Label, naive, sysv};
            use portal_solutions_blitz_common::wasm_encoder::reencode::RoundtripReencoder;

            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut ctx = ();
            let arch_cfg = AArch64Arch::default();
            // Reserve one instruction word at the current offset: a harmless
            // NOP (`mov x31-rf?, 0` is not encodable; use `mov x9, #0` style
            // MOVZ) that gets overwritten post-assembly for trampoline/data
            // slots. mov_imm with value 0 emits MOVZ X9, #0 (4 bytes).
            macro_rules! reserve_word {
                () => {{
                    let x9 = MemArgKind::NoMem(ArgKind::Reg {
                        reg: portal_solutions_blitz_common::asm::Reg(9),
                        size: portal_pc_asm_common::types::mem::MemorySize::_64,
                    });
                    out.mov_imm(&mut ctx, arch_cfg, &x9, 0)
                        .map_err(|e| internal(e.to_string()))?
                }};
            }
            let mut state = naive::State::default();
            state.call_abi = naive::CallAbi::AllStack;
            state.call_params = call_params;
            state.call_results = call_results;
            state.sig_params = sig_params;
            state.sig_results = sig_results;
            state.n_imports = import_count;
            if mem_pages.is_some() {
                state.mem_base = naive::MemBase::WasmMemSymbol;
            }
            let mut reencoder = RoundtripReencoder;
            for op in ops {
                let op = op.map_err(|e| internal(format!("mach op: {e}")))?;
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out,
                    &mut ctx,
                    arch_cfg,
                    &mut state,
                    &func_imports,
                    &op,
                    &mut reencoder,
                    0,
                )
                .map_err(|e| unsupported(format!("aarch64 emit: {e:?}")))?;
            }

            // AArch64 external references materialize via ADRP+ADD pairs that
            // are external-only (set_label panics on them), so the stub/data
            // area is appended AFTER codegen and the ADRP/ADD relocations are
            // patched by hand to point into it.
            let (bytes, labels, relocs) = out.into_parts_with_relocs();
            let mut patched = bytes;

            // Layout after the module code: a 4-byte `B` trampoline per
            // function symbol (jumps to its sentinel slot), then the runtime
            // data area (__wasm_mem_pages, __wasm_mem, __wasm_globals, …).
            // Data symbols resolve DIRECTLY to their data-area cell; there
            // are no separate stub cells for them.
            let mut stub_off: HashMap<String, usize> = HashMap::new();
            let mut names: Vec<String> = Vec::new();
            for r in &relocs {
                if let AArch64Label::External { name } = &r.label {
                    if !stub_off.contains_key(name) {
                        names.push(name.clone());
                    }
                }
            }
            let is_data = |n: &str| DATA_SYMBOLS.contains(&n);
            let mut cursor = patched.len();
            for name in &names {
                cursor = (cursor + 3) & !3; // 4-align every trampoline
                stub_off.insert(name.clone(), cursor);
                cursor += 4;
            }
            // The runtime data area starts at the next 8-aligned offset; its
            // internal layout matches `data_area` (8 bytes per DATA_SYMBOLS
            // entry in order, then the globals array).
            let data_start = (cursor + 7) & !7;
            for (i, sym) in DATA_SYMBOLS.iter().enumerate() {
                stub_off.insert((*sym).to_string(), data_start + i * 8);
            }

            // Patch every ADRP+ADD pair to `CODE_BASE + stub_off[name]`.
            for r in &relocs {
                let name = match &r.label {
                    AArch64Label::External { name } => name,
                    other => return Err(internal(format!("non-external reloc {other:?}"))),
                };
                let target = CODE_BASE + stub_off[name] as u64;
                let off = r.byte_offset;
                let orig = u32::from_le_bytes(patched[off..off + 4].try_into().unwrap());
                match r.kind {
                    AsmRelocKind::A64AdrpPage21 => {
                        // ADRP Xd: imm21 = page delta (pc-relative), immlo in
                        // bits [30:29], immhi in bits [23:5]; rd in bits [4:0].
                        let pc = CODE_BASE + off as u64;
                        let delta_pages = ((target & !0xFFF) as i64 - (pc & !0xFFF) as i64) / 4096;
                        let imm21 = (delta_pages as u32) & 0x001F_FFFF;
                        let immlo = imm21 & 0x3;
                        let immhi = (imm21 >> 2) & 0x7_FFFF;
                        let word = 0x9000_0000u32 | (immlo << 29) | (immhi << 5) | (orig & 0x1F);
                        patched[off..off + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    AsmRelocKind::A64AddAbsLo12 => {
                        // ADD Xd, Xn, #lo12: imm12 in bits [21:10],
                        // rn in bits [9:5], rd in bits [4:0].
                        let imm12 = (target & 0xFFF) as u32;
                        let word = 0x9100_0000u32 | (imm12 << 10) | (orig & 0x3FF);
                        patched[off..off + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    other => return Err(internal(format!("unexpected reloc kind {other:?}"))),
                }
            }
            // Build the stub region bytes: `B` trampolines for function
            // symbols (each lands on its sentinel slot).
            let mut stub_bytes = vec![0u8; data_start - patched.len()];
            for name in &names {
                if is_data(name) {
                    continue;
                }
                let k = slot_of[name];
                let off = stub_off[name];
                let target = SENTINEL_BASE + (k as u64) * SENTINEL_SLOT;
                // AArch64 B: imm26 is relative to the branch instruction's
                // own address (no +4).
                let pc = CODE_BASE + off as u64;
                let delta = target as i64 - pc as i64;
                let imm26 = ((delta / 4) as u32) & 0x03FF_FFFF;
                let word = (0x1400_0000u32 | imm26).to_le_bytes();
                stub_bytes[off - patched.len()..off - patched.len() + 4].copy_from_slice(&word);
            }
            patched.extend_from_slice(&stub_bytes);

            // Per-function entries and data-area bytes.
            let n_fns = (0..)
                .map_while(|i| {
                    labels
                        .get(&AArch64Label::Indexed {
                            idx: i + 0x8000_0000,
                        })
                        .copied()
                })
                .count();
            let mut fn_entries = Vec::with_capacity(n_fns);
            for i in 0..n_fns {
                let off = *labels
                    .get(&AArch64Label::Indexed {
                        idx: i + 0x8000_0000,
                    })
                    .ok_or_else(|| internal(format!("missing fn {i} label")))?;
                fn_entries.push(off as u64);
            }
            let entry = *labels
                .get(&AArch64Label::Indexed { idx: 0x8000_0000 })
                .ok_or_else(|| internal("missing entry label".into()))?;
            let globals_words = 1024usize.max(global_inits.len().next_multiple_of(1024));
            let (databytes, mem_pages_off, mem_ptr_off, globals_off) =
                data_area(data_start, mem_pages, global_inits, globals_words);
            patched.extend_from_slice(&databytes);

            Ok(NativeBinary {
                code: patched,
                entry_off: entry as u64,
                fn_entries,
                slot_imports: targets,
                mem_pages_off,
                mem_ptr_off,
                globals_off,
            })
        }
        NativeArch::Riscv64 => {
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            use portal_solutions_asm_riscv64::out::{Writer as _, WriterCore as _};
            use portal_solutions_blitz_common::wasm_encoder::reencode::RoundtripReencoder;
            use portal_solutions_blitz_riscv64::{RiscV64Arch, RiscvLabel, naive, sysv};

            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut ctx = ();
            let arch_cfg = RiscV64Arch::default();

            // Emit import trampolines FIRST, at the very start of the blob,
            // so every backend External reference resolves in-buffer (the
            // RvFixup::La machinery handles already-defined labels
            // immediately). Trampoline shape (2 instructions):
            //   li   t2, SENTINEL_BASE + slot*SENTINEL_SLOT
            //   jalr x0, t2, 0          ; unconditional jump, never returns
            // The runner's sentinel code hook services the call and resumes
            // at the recorded RA.
            //
            // Data symbols are laid out AFTER codegen as a plain data area;
            // because their cell addresses are unknown before codegen, we
            // cannot label them up front. The riscv64 backend only emits
            // data-symbol references through `apply_mem_base`
            // (`__wasm_mem`) and `MemorySize` (`__wasm_mem_pages`), so we
            // keep a name→cell map and patch those AUIPC+ADDI placeholder
            // pairs post-assembly: `la_label` on an UNDEFINED label emits
            // AUIPC rd,0 + ADDI rd,rd,0 with zero immediates, which we can
            // find and rewrite deterministically because each such pair is
            // uniquely identified by its rd (t0 for mem base in the naive
            // mem path, x9 in MemorySize) and there is at most one
            // unresolved reference per symbol per blob in the phase-1
            // file set.
            //
            // Emit function trampolines and define their labels now.
            for (name, &k) in slot_of.iter() {
                if k < imports.len() {
                    out.set_label(
                        &mut ctx,
                        arch_cfg,
                        RiscvLabel::External { name: name.clone() },
                    )
                    .map_err(|e| internal(e.to_string()))?;
                    let target = SENTINEL_BASE + (k as u64) * SENTINEL_SLOT;
                    // li t2, target  (RvAsmWriter::li handles arbitrary u64)
                    out.li(&mut ctx, arch_cfg, &rv_reg(7), target)
                        .map_err(|e| internal(e.to_string()))?;
                    // jalr x0, t2, 0
                    out.jalr(&mut ctx, arch_cfg, &rv_reg(0), &rv_reg(7), 0)
                        .map_err(|e| internal(e.to_string()))?;
                }
            }

            let mut state = naive::State::default();
            state.call_abi = naive::CallAbi::AllStack;
            state.call_params = call_params;
            state.call_results = call_results;
            state.sig_params = sig_params;
            state.sig_results = sig_results;
            state.n_imports = import_count;
            if mem_pages.is_some() {
                state.mem_base = naive::MemBase::WasmMemSymbol;
                state
                    .mem_base_by_index
                    .insert(0, naive::MemBase::WasmMemSymbol);
            }
            let mut reencoder = RoundtripReencoder;
            for op in ops {
                let op = op.map_err(|e| internal(format!("mach op: {e}")))?;
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out,
                    &mut ctx,
                    arch_cfg,
                    &mut state,
                    &func_imports,
                    &op,
                    &mut reencoder,
                    0,
                )
                .map_err(|e| unsupported(format!("riscv64 emit: {e:?}")))?;
            }

            let (bytes, labels) = out.into_parts();
            let mut code = bytes;

            // Patch the unresolved data-symbol AUIPC+ADDI placeholder pairs
            // (AUIPC rd,0 is word 0x00000017 | rd<<7; ADDI rd,rd,0 is
            // 0x00000013 | rd<<20 | rd<<7). We know each symbol's rd from
            // the backend's usage and there is exactly one unresolved
            // reference per data symbol present.
            let patch_la = |code: &mut Vec<u8>, rd: u32, cell_off: usize| -> bool {
                let mut found = false;
                let mut i = 0;
                while i + 8 <= code.len() {
                    let w1 = u32::from_le_bytes(code[i..i + 4].try_into().unwrap());
                    let w2 = u32::from_le_bytes(code[i + 4..i + 8].try_into().unwrap());
                    let auipc_rd0 =
                        w1 & 0x0000_007F == 0x17 && (w1 >> 7) & 0x1F == rd && (w1 >> 12) == 0;
                    let addi_rd0 = w2 & 0x0000_007F == 0x13
                        && (w2 >> 15) & 0x1F == rd
                        && (w2 >> 7) & 0x1F == rd
                        && (w2 >> 20) == 0;
                    if auipc_rd0 && addi_rd0 {
                        // Re-encode against the true cell address.
                        let pc = CODE_BASE + i as u64;
                        let target = CODE_BASE + cell_off as u64;
                        let delta = target as i64 - pc as i64;
                        let hi20 = ((delta + 0x800) >> 12) as i32;
                        let lo12 = delta - ((hi20 as i64) << 12);
                        let auipc = ((hi20 as u32) << 12) | (rd << 7) | 0x17;
                        let addi = (((lo12 as u32) & 0xFFF) << 20) | (rd << 15) | (rd << 7) | 0x13;
                        code[i..i + 4].copy_from_slice(&auipc.to_le_bytes());
                        code[i + 4..i + 8].copy_from_slice(&addi.to_le_bytes());
                        found = true;
                        i += 8;
                    } else {
                        i += 4;
                    }
                }
                found
            };

            // Compute data-area layout (must match `data_area`).
            let globals_words = 1024usize.max(global_inits.len().next_multiple_of(1024));
            let data_start = code.len();
            let mut mem_pages_off = None;
            let mut mem_ptr_off = None;
            if mem_pages.is_some() {
                mem_pages_off = Some(data_start);
                mem_ptr_off = Some(data_start + 8);
                patch_la(&mut code, 9, data_start + 8); // __wasm_mem (rd = x9? naive uses t1)
            }
            let globals_off = data_start + 16;
            let (databytes, _, _, _) =
                data_area(data_start, mem_pages, global_inits, globals_words);
            code.extend_from_slice(&databytes);

            let n_fns = (0..)
                .map_while(|i| {
                    labels
                        .get(&RiscvLabel::Indexed { idx: i | (1 << 28) })
                        .copied()
                })
                .count();
            let mut fn_entries = Vec::with_capacity(n_fns);
            for i in 0..n_fns {
                let off = *labels
                    .get(&RiscvLabel::Indexed { idx: i | (1 << 28) })
                    .ok_or_else(|| internal(format!("missing fn {i} label")))?;
                fn_entries.push(off as u64);
            }
            let entry = *labels
                .get(&RiscvLabel::Indexed { idx: 1 << 28 })
                .ok_or_else(|| internal("missing entry label".into()))?;

            Ok(NativeBinary {
                code,
                entry_off: entry as u64,
                fn_entries,
                slot_imports: targets,
                mem_pages_off,
                mem_ptr_off,
                globals_off,
            })
        }
    }
}

/// RISC-V register MemArg helper (64-bit GPR `n`).
fn rv_reg(n: u8) -> portal_solutions_asm_riscv64::out::arg::MemArgKind {
    use portal_pc_asm_common::types::mem::MemorySize;
    use portal_solutions_asm_riscv64::out::arg::{ArgKind, MemArgKind};
    MemArgKind::NoMem(ArgKind::Reg {
        reg: portal_solutions_blitz_common::asm::Reg(n),
        size: MemorySize::_64,
    })
}

/// Build the data-area bytes; returns (bytes, mem_pages_off, mem_ptr_off,
/// globals_off) as absolute blob offsets.
fn data_area(
    start: usize,
    mem_pages: Option<u32>,
    global_inits: &[u64],
    globals_words: usize,
) -> (Vec<u8>, Option<usize>, Option<usize>, usize) {
    let mem_pages_off = mem_pages.map(|_| start);
    let mem_ptr_off = mem_pages.map(|_| start + 8);
    let globals_off = start + 16;
    let _ = start;
    let mut buf = vec![0u8; 16 + globals_words * 8];
    if let Some(p) = mem_pages {
        buf[0..4].copy_from_slice(&p.to_le_bytes());
        buf[8..16].copy_from_slice(&MEM_BASE.to_le_bytes());
    }
    for (i, v) in global_inits.iter().enumerate() {
        let o = 16 + i * 8;
        buf[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }
    (buf, mem_pages_off, mem_ptr_off, globals_off)
}

// ── Unicorn runner ───────────────────────────────────────────────────────────

/// Outcome of running one exported function.
#[derive(Debug)]
pub enum RunOutcome {
    /// Value of the architecture's return register (RAX/X0/A0).
    Returned(u64),
    /// Multi-result: return register plus results read off the guest stack.
    ReturnedMulti(Vec<u64>),
    /// A guest trap: unreachable, memory fault, unsupported runtime symbol,
    /// or host dispatch error.
    Trapped(String),
}

/// Read `n` extra results from the guest stack after an AllStack return.
/// The SysV epilogue leaves results at [rsp], first result at the LOWEST
/// address (pushed last: RA was pushed above them by `ret`'s pop).
fn read_stack_results<D>(uc: &mut Unicorn<'_, D>, arch: NativeArch, n: usize) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }
    let sp = match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            uc.reg_read(RegisterX86::RSP).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            uc.reg_read(RegisterARM64::SP).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            uc.reg_read(RegisterRISCV::SP).unwrap()
        }
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 8];
        if uc.mem_read(sp + (i as u64) * 8, &mut b).is_ok() {
            out.push(u64::from_le_bytes(b));
        } else {
            out.push(0);
        }
    }
    out
}

/// Dispatch an import: `(import_index, args) -> results (0 or 1)`.
pub type HostDispatch<'a> = dyn FnMut(usize, &[u64]) -> Result<Vec<u64>, String> + 'a;

/// One sentinel service request recorded by the hook, drained by the driver.
struct Service {
    slot: usize,
    ra: u64,
    args: Vec<u64>,
}

/// Shared between the hook and the driver loop.
#[derive(Default)]
struct Shared {
    services: Vec<Service>,
    error: Option<String>,
}

/// Run exported function at `entry_off` with `args` (u64 slots) under
/// Unicorn. `host` dispatches import calls. Returns the raw return-register
/// value (mask to the result width at the call site).
///
/// The sentinel hook only *records* the call and stops emulation; this loop
/// drains services (dispatching to `host`), fixes up registers/stack, and
/// restarts at the recorded return address. That keeps all host dispatch in
/// ordinary Rust control flow — no closure-lifetime games inside hooks.
pub fn run_module(
    arch: NativeArch,
    bin: &NativeBinary,
    entry_off: u64,
    args: &[u64],
    nrets: usize,
    cap: u64,
    host: &mut HostDispatch<'_>,
) -> RunOutcome {
    use unicorn_engine::Unicorn;
    use unicorn_engine::unicorn_const::{Arch, Mode, Prot};

    let shared = Arc::new(Mutex::new(Shared::default()));
    let end = CODE_BASE + bin.code.len() as u64;

    // Per-arch Unicorn setup + drive loop (concrete handle type keeps the
    // hook-lifetime story simple: D = Arc<Mutex<Shared>>).
    fn drive<D: 'static>(
        uc: &mut Unicorn<'_, D>,
        arch: NativeArch,
        shared: &Arc<Mutex<Shared>>,
        bin: &NativeBinary,
        entry_off: u64,
        cap: u64,
        host: &mut HostDispatch<'_>,
    ) -> RunOutcome {
        let end = CODE_BASE + bin.code.len() as u64;
        let mut pc = CODE_BASE + entry_off;
        let mut host_calls = 0u64;
        loop {
            match uc.emu_start(pc, end, 0, cap as usize) {
                Ok(()) => {}
                Err(e) => return RunOutcome::Trapped(format!("guest exception: {e}")),
            }
            let svc = shared.lock().unwrap().services.pop();
            match svc {
                Some(svc) => {
                    host_calls += 1;
                    if host_calls > 1_000_000 {
                        return RunOutcome::Trapped("host call limit exceeded".into());
                    }
                    match bin.slot_imports.get(svc.slot).copied().flatten() {
                        Some(import_idx) => {
                            let results = match host(import_idx, &svc.args) {
                                Ok(r) => r,
                                Err(e) => {
                                    return RunOutcome::Trapped(format!("host: {e}"));
                                }
                            };
                            apply_service(uc, arch, &results);
                            pc = svc.ra;
                        }
                        None => {
                            return RunOutcome::Trapped(
                                "native runtime symbol not supported in this scope".into(),
                            );
                        }
                    }
                }
                None => {
                    // No pending service: stopped at HLT (trap), at `until`,
                    // or genuinely returned. Distinguish via RIP.
                    let rip = read_rip(uc, arch);
                    if rip == CODE_BASE + bin.code.len() as u64 {
                        return RunOutcome::Returned(read_return(uc, arch));
                    }
                    if rip_in_sentinel(uc, arch) {
                        continue; // hook fired; service will appear on next loop
                    }
                    return RunOutcome::Trapped(format!(
                        "guest stopped at {rip:#x} (trap / bad control flow)"
                    ));
                }
            }
        }
    }

    // targets_len placeholder is unused — see below.
    #[allow(unused_variables)]
    fn targets_len(_bin: &NativeBinary) -> usize {
        0
    }

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE_BASE, CODE_SIZE, Prot::ALL).unwrap();
            uc.mem_map(STACK_BASE, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(SENTINEL_BASE, 0x1000, Prot::ALL).unwrap();
            if bin.mem_ptr_off.is_some() {
                uc.mem_map(MEM_BASE, 0x1_0000, Prot::ALL).unwrap();
            }
            uc.mem_write(CODE_BASE, &bin.code).unwrap();

            // AllStack entry: [rsp] = ret addr, [rsp + 8 + i*8] = arg i.
            let n = args.len() as u64;
            let frame = ((n + 2) * 8 + 15) & !15;
            let rsp = (STACK_BASE + STACK_SIZE - frame - 16) & !15;
            uc.mem_write(rsp, &end.to_le_bytes()).unwrap();
            for (i, a) in args.iter().enumerate() {
                uc.mem_write(rsp + 8 + (i as u64) * 8, &a.to_le_bytes())
                    .unwrap();
            }
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();

            let sh = shared.clone();
            uc.add_code_hook(
                SENTINEL_BASE,
                SENTINEL_BASE + 0x1000,
                move |uc, addr, _size| sentinel_hook(uc, addr, arch, &sh),
            );
            match drive(&mut uc, arch, &shared, bin, entry_off, cap, host) {
                RunOutcome::Returned(v) => {
                    let extra = read_stack_results(&mut uc, arch, nrets.saturating_sub(1));
                    RunOutcome::ReturnedMulti(std::iter::once(v).chain(extra).collect())
                }
                other => other,
            }
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE_BASE, CODE_SIZE, Prot::ALL).unwrap();
            uc.mem_map(STACK_BASE, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(SENTINEL_BASE, 0x1000, Prot::ALL).unwrap();
            if bin.mem_ptr_off.is_some() {
                uc.mem_map(MEM_BASE, 0x1_0000, Prot::ALL).unwrap();
            }
            uc.mem_write(CODE_BASE, &bin.code).unwrap();

            // AllStack entry: params at [sp + i*8], sp 16-aligned.
            let n = args.len() as u64;
            let frame = ((n + 2) * 8 + 15) & !15;
            let sp = (STACK_BASE + STACK_SIZE - frame - 16) & !15;
            for (i, a) in args.iter().enumerate() {
                uc.mem_write(sp + (i as u64) * 8, &a.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterARM64::SP, sp).unwrap();
            uc.reg_write(RegisterARM64::LR, end).unwrap();

            let sh = shared.clone();
            uc.add_code_hook(
                SENTINEL_BASE,
                SENTINEL_BASE + 0x1000,
                move |uc, addr, _size| sentinel_hook(uc, addr, arch, &sh),
            );
            match drive(&mut uc, arch, &shared, bin, entry_off, cap, host) {
                RunOutcome::Returned(v) => {
                    let extra = read_stack_results(&mut uc, arch, nrets.saturating_sub(1));
                    RunOutcome::ReturnedMulti(std::iter::once(v).chain(extra).collect())
                }
                other => other,
            }
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE_BASE, CODE_SIZE, Prot::ALL).unwrap();
            uc.mem_map(STACK_BASE, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(SENTINEL_BASE, 0x1000, Prot::ALL).unwrap();
            if bin.mem_ptr_off.is_some() {
                uc.mem_map(MEM_BASE, 0x1_0000, Prot::ALL).unwrap();
            }
            uc.mem_write(CODE_BASE, &bin.code).unwrap();

            let n = args.len() as u64;
            let frame = ((n + 2) * 8 + 15) & !15;
            let sp = (STACK_BASE + STACK_SIZE - frame - 16) & !15;
            for (i, a) in args.iter().enumerate() {
                uc.mem_write(sp + (i as u64) * 8, &a.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterRISCV::SP, sp).unwrap();
            uc.reg_write(RegisterRISCV::RA, end).unwrap();

            let sh = shared.clone();
            uc.add_code_hook(
                SENTINEL_BASE,
                SENTINEL_BASE + 0x1000,
                move |uc, addr, _size| sentinel_hook(uc, addr, arch, &sh),
            );
            match drive(&mut uc, arch, &shared, bin, entry_off, cap, host) {
                RunOutcome::Returned(v) => {
                    let extra = read_stack_results(&mut uc, arch, nrets.saturating_sub(1));
                    RunOutcome::ReturnedMulti(std::iter::once(v).chain(extra).collect())
                }
                other => other,
            }
        }
    }
}

/// Read the instruction pointer (RIP/PC).
fn read_rip<D>(uc: &Unicorn<'_, D>, arch: NativeArch) -> u64 {
    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            uc.reg_read(RegisterX86::RIP).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            uc.reg_read(RegisterARM64::PC).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            uc.reg_read(RegisterRISCV::PC).unwrap()
        }
    }
}

fn rip_in_sentinel<D>(uc: &Unicorn<'_, D>, arch: NativeArch) -> bool {
    let rip = read_rip(uc, arch);
    (SENTINEL_BASE..SENTINEL_BASE + 0x1000).contains(&rip)
}

/// Apply one serviced import call: write results to the return register.
fn apply_service<D>(uc: &mut Unicorn<'_, D>, arch: NativeArch, results: &[u64]) {
    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            uc.reg_write(RegisterX86::RAX, results.first().copied().unwrap_or(0))
                .unwrap();
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            uc.reg_write(RegisterARM64::X0, results.first().copied().unwrap_or(0))
                .unwrap();
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            uc.reg_write(RegisterRISCV::A0, results.first().copied().unwrap_or(0))
                .unwrap();
        }
    }
}

/// Read the return register after the guest returns to the sentinel RA.
fn read_return<D>(uc: &Unicorn<'_, D>, arch: NativeArch) -> u64 {
    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
    }
}

/// Sentinel hook: record (slot, ra, arg window) and stop emulation. x86-64
/// also pops the return address the `call` pushed (the driver restarts at it,
/// and the caller's post-call code expects `rsp` at the frame).
fn sentinel_hook<D>(
    uc: &mut Unicorn<'_, D>,
    addr: u64,
    arch: NativeArch,
    shared: &Arc<Mutex<Shared>>,
) {
    if !(SENTINEL_BASE..SENTINEL_BASE + 0x1000).contains(&addr) {
        return; // range hook misfire guard
    }
    let slot = ((addr - SENTINEL_BASE) / SENTINEL_SLOT) as usize;
    let ra = match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let rsp = uc.reg_read(RegisterX86::RSP).unwrap();
            let mut b = [0u8; 8];
            uc.mem_read(rsp, &mut b).unwrap();
            let ra = u64::from_le_bytes(b);
            uc.reg_write(RegisterX86::RSP, rsp + 8).unwrap();
            ra
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            uc.reg_read(RegisterARM64::LR).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            uc.reg_read(RegisterRISCV::RA).unwrap()
        }
    };
    // Import args, per the psABI import marshalling (sysv.rs):
    // x86-64: first 6 in RDI,RSI,RDX,RCX,R8,R9 (rest at [rsp + (i-6)*8]);
    // aarch64: first 8 in X0..X7 (rest at [sp + (i-8)*8]);
    // riscv64: first 8 in A0..A7 (rest at [sp + (i-8)*8]).
    let mut args: Vec<u64> = Vec::new();
    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            const R: [RegisterX86; 6] = [
                RegisterX86::RDI,
                RegisterX86::RSI,
                RegisterX86::RDX,
                RegisterX86::RCX,
                RegisterX86::R8,
                RegisterX86::R9,
            ];
            for r in R {
                args.push(uc.reg_read(r).unwrap());
            }
            let sp = uc.reg_read(RegisterX86::RSP).unwrap();
            for i in 0..MAX_IMPORT_ARGS.saturating_sub(6) {
                let mut b = [0u8; 8];
                if uc.mem_read(sp + (i as u64) * 8, &mut b).is_ok() {
                    args.push(u64::from_le_bytes(b));
                }
            }
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            const R: [RegisterARM64; 8] = [
                RegisterARM64::X0,
                RegisterARM64::X1,
                RegisterARM64::X2,
                RegisterARM64::X3,
                RegisterARM64::X4,
                RegisterARM64::X5,
                RegisterARM64::X6,
                RegisterARM64::X7,
            ];
            for r in R {
                args.push(uc.reg_read(r).unwrap());
            }
            let sp = uc.reg_read(RegisterARM64::SP).unwrap();
            for i in 0..MAX_IMPORT_ARGS.saturating_sub(8) {
                let mut b = [0u8; 8];
                if uc.mem_read(sp + (i as u64) * 8, &mut b).is_ok() {
                    args.push(u64::from_le_bytes(b));
                }
            }
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            const R: [RegisterRISCV; 8] = [
                RegisterRISCV::A0,
                RegisterRISCV::A1,
                RegisterRISCV::A2,
                RegisterRISCV::A3,
                RegisterRISCV::A4,
                RegisterRISCV::A5,
                RegisterRISCV::A6,
                RegisterRISCV::A7,
            ];
            for r in R {
                args.push(uc.reg_read(r).unwrap());
            }
            let sp = uc.reg_read(RegisterRISCV::SP).unwrap();
            for i in 0..MAX_IMPORT_ARGS.saturating_sub(8) {
                let mut b = [0u8; 8];
                if uc.mem_read(sp + (i as u64) * 8, &mut b).is_ok() {
                    args.push(u64::from_le_bytes(b));
                }
            }
        }
    }
    args.truncate(MAX_IMPORT_ARGS);
    shared
        .lock()
        .unwrap()
        .services
        .push(Service { slot, ra, args });
    uc.emu_stop().unwrap();
}
