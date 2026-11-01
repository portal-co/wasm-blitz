//! Tests for `blitz-symmap`: WASM names/indices → platform symbol map files.
//!
//! Covers:
//! - name-section parsing and symbol collection (naming, sizing, aliases);
//! - byte-level round-trips of the ELF (Linux), Mach-O (macOS) and COFF
//!   (Windows) emitters for every supported architecture, parsed back by
//!   minimal independent readers;
//! - end-to-end: compile a named WASM module through the x86-64, AArch64 and
//!   RISC-V64 SysV backends, collect function-entry labels, and verify the
//!   emitted symbol maps resolve each function's name to its code offset.

use portal_solutions_blitz_common::wasm_encoder::{
    self, CodeSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType,
    reencode::RoundtripReencoder,
};
use portal_solutions_blitz_common::wasmparser::{self, FuncType as WpFuncType};
use portal_solutions_blitz_common::{HandleOpError, dce_pass, ops::mach_operators};
use portal_solutions_blitz_symmap::{
    Arch, Symbol, SymbolKind, WasmSymbol, WasmSymbolInfo, collect_symbols, emit_coff, emit_elf,
    emit_macho, emit_perf_map, sanitize_name, wasm_symbol_info,
};

// ---------------------------------------------------------------------------
// Minimal parse-back readers (independent of the emitters on purpose)
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
struct ParsedSym {
    name: String,
    value: u64,
    size: u64,
}

fn ru16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}
fn ru32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn ru64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn cstr(b: &[u8], off: usize) -> String {
    let end = b[off..].iter().position(|&c| c == 0).unwrap() + off;
    String::from_utf8(b[off..end].to_vec()).unwrap()
}

/// One ELF section header, normalized across ELF32/ELF64.
struct ElfShdr {
    ty: u32,
    offset: usize,
    size: usize,
    link: u32,
    entsize: usize,
}

fn parse_elf_symbols(b: &[u8]) -> (u16, Vec<ParsedSym>) {
    assert_eq!(&b[0..4], b"\x7fELF");
    assert_eq!(b[5], 1, "must be little-endian");
    let is64 = b[4] == 2;
    let machine = ru16(b, 18);
    let (shoff, shentsize, shnum) = if is64 {
        (
            ru64(b, 40) as usize,
            ru16(b, 58) as usize,
            ru16(b, 60) as usize,
        )
    } else {
        (
            ru32(b, 32) as usize,
            ru16(b, 46) as usize,
            ru16(b, 48) as usize,
        )
    };
    let shdr = |i: usize| -> ElfShdr {
        let o = shoff + i * shentsize;
        if is64 {
            ElfShdr {
                ty: ru32(b, o + 4),
                offset: ru64(b, o + 24) as usize,
                size: ru64(b, o + 32) as usize,
                link: ru32(b, o + 40),
                entsize: ru64(b, o + 56) as usize,
            }
        } else {
            ElfShdr {
                ty: ru32(b, o + 4),
                offset: ru32(b, o + 16) as usize,
                size: ru32(b, o + 20) as usize,
                link: ru32(b, o + 24),
                entsize: ru32(b, o + 36) as usize,
            }
        }
    };
    // Find .symtab (SHT_SYMTAB = 2) and its linked .strtab.
    let mut symtab = None;
    for i in 0..shnum {
        let s = shdr(i);
        if s.ty == 2 {
            symtab = Some(s);
        }
    }
    let symtab = symtab.expect("no .symtab section");
    let strtab = shdr(symtab.link as usize);
    let mut out = Vec::new();
    for i in 1..(symtab.size / symtab.entsize) {
        let o = symtab.offset + i * symtab.entsize;
        let (name_off, value, size, shndx_off) = if is64 {
            (ru32(b, o), ru64(b, o + 8), ru64(b, o + 16), o + 6)
        } else {
            (ru32(b, o), ru32(b, o + 4) as u64, ru32(b, o + 8) as u64, o + 14)
        };
        assert_eq!(ru16(b, shndx_off), 0xfff1, "st_shndx must be SHN_ABS");
        out.push(ParsedSym {
            name: cstr(b, strtab.offset + name_off as usize),
            value,
            size,
        });
    }
    (machine, out)
}

fn parse_macho_symbols(b: &[u8]) -> (u32, Vec<ParsedSym>) {
    let magic = ru32(b, 0);
    let is64 = match magic {
        0xfeed_facf => true,
        0xfeed_face => false,
        m => panic!("bad Mach-O magic {m:#x}"),
    };
    let cputype = ru32(b, 4);
    assert_eq!(ru32(b, 12), 1, "filetype must be MH_OBJECT");
    let header = if is64 { 32 } else { 28 };
    assert_eq!(ru32(b, header), 0x2, "first load command must be LC_SYMTAB");
    let symoff = ru32(b, header + 8) as usize;
    let nsyms = ru32(b, header + 12) as usize;
    let stroff = ru32(b, header + 16) as usize;
    let nl = if is64 { 16 } else { 12 };
    let mut out = Vec::new();
    for i in 0..nsyms {
        let o = symoff + i * nl;
        let strx = ru32(b, o) as usize;
        assert_eq!(b[o + 4], 0x03, "n_type must be N_EXT|N_ABS");
        assert_eq!(b[o + 5], 0, "n_sect must be NO_SECT");
        let value = if is64 {
            ru64(b, o + 8)
        } else {
            ru32(b, o + 8) as u64
        };
        out.push(ParsedSym {
            name: cstr(b, stroff + strx),
            value,
            size: 0, // Mach-O nlist carries no size
        });
    }
    (cputype, out)
}

fn parse_coff_symbols(b: &[u8]) -> (u16, Vec<ParsedSym>) {
    let machine = ru16(b, 0);
    assert_eq!(ru16(b, 2), 0, "no sections expected");
    let symptr = ru32(b, 8) as usize;
    let nsyms = ru32(b, 12) as usize;
    assert_eq!(symptr, 20);
    let strtab_off = symptr + nsyms * 18;
    let strtab_size = ru32(b, strtab_off) as usize;
    assert_eq!(strtab_off + strtab_size, b.len());
    let mut out = Vec::new();
    for i in 0..nsyms {
        let o = symptr + i * 18;
        let name = if ru32(b, o) == 0 {
            let off = ru32(b, o + 4) as usize;
            assert!(off >= 4, "string-table offsets start at 4");
            cstr(b, strtab_off + off)
        } else {
            let end = b[o..o + 8].iter().position(|&c| c == 0).unwrap_or(8);
            String::from_utf8(b[o..o + end].to_vec()).unwrap()
        };
        assert_eq!(ru16(b, o + 12), 0xffff, "SectionNumber must be ABSOLUTE");
        assert_eq!(b[o + 16], 2, "StorageClass must be EXTERNAL");
        out.push(ParsedSym {
            name,
            value: ru32(b, o + 8) as u64,
            size: 0, // COFF symbols carry no size
        });
    }
    (machine, out)
}

/// Compare a parsed-back symbol list against the emitter input. `check_size`
/// is false for formats that don't record symbol sizes (Mach-O, COFF).
fn assert_syms_match(got: &[ParsedSym], want: &[Symbol], check_size: bool) {
    assert_eq!(got.len(), want.len(), "symbol count");
    let mut got: Vec<(&str, u64, u64)> = got
        .iter()
        .map(|s| (s.name.as_str(), s.value, s.size))
        .collect();
    got.sort();
    let mut want: Vec<(&str, u64, u64)> = want
        .iter()
        .map(|s| (s.name.as_str(), s.value, s.size))
        .collect();
    want.sort();
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.0, w.0, "name");
        assert_eq!(g.1, w.1, "value of {}", w.0);
        if check_size {
            assert_eq!(g.2, w.2, "size of {}", w.0);
        }
    }
}

fn sample_symbols() -> Vec<Symbol> {
    vec![
        Symbol {
            name: "wasm_func_0".into(),
            value: 0x1000,
            size: 0x40,
            kind: SymbolKind::Func,
        },
        Symbol {
            // > 8 chars to exercise the COFF string table.
            name: "a_much_longer_function_name".into(),
            value: 0x1040,
            size: 0x28,
            kind: SymbolKind::Func,
        },
        Symbol {
            name: "embedded.table".into(),
            value: 0x1080,
            size: 0x10,
            kind: SymbolKind::Data,
        },
    ]
}

const ALL_ARCHES: [Arch; 6] = [
    Arch::X86_64,
    Arch::I686,
    Arch::AArch64,
    Arch::Arm,
    Arch::Riscv64,
    Arch::Riscv32,
];

// ---------------------------------------------------------------------------
// Pure unit tests (no backends involved)
// ---------------------------------------------------------------------------

#[test]
fn sanitize_name_replaces_unsafe_chars() {
    assert_eq!(sanitize_name("add_one"), "add_one");
    assert_eq!(sanitize_name("foo.bar$baz"), "foo.bar$baz");
    assert_eq!(sanitize_name("hello world"), "hello_world");
    assert_eq!(sanitize_name("λambda→x"), "_ambda_x");
    assert_eq!(sanitize_name(""), "_");
}

#[test]
fn wasm_symbol_info_reads_name_section_and_imports() {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], []); // type 0: () -> ()
    module.section(&types);
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import("env", "imp", wasm_encoder::EntityType::Function(0));
    module.section(&imports);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    funcs.function(0);
    module.section(&funcs);
    let mut code = CodeSection::new();
    for _ in 0..2 {
        let mut f = Function::new([]);
        f.instruction(&Instruction::End);
        code.function(&f);
    }
    module.section(&code);
    let mut names = wasm_encoder::NameSection::new();
    let mut fnames = wasm_encoder::NameMap::new();
    fnames.append(0, "imported_fn");
    fnames.append(1, "first defined");
    fnames.append(2, "second_defined");
    names.functions(&fnames);
    module.section(&names);
    let wasm = module.finish();

    let info = wasm_symbol_info(&wasm).unwrap();
    assert_eq!(info.imported_funcs, 1);
    assert_eq!(info.names.len(), 3);
    assert_eq!(info.names[&0], "imported_fn");
    assert_eq!(info.names[&1], "first defined");
    assert_eq!(info.names[&2], "second_defined");
}

#[test]
fn collect_symbols_names_sizes_and_aliases() {
    let mut info = WasmSymbolInfo::default();
    info.imported_funcs = 1;
    info.names.insert(1, "add one".to_string()); // absolute index 1 = defined f0; needs sanitizing
    let entries = vec![
        (0x30u64, WasmSymbol::Func(0)),          // named via name section
        (0x50, WasmSymbol::Func(1)),             // fallback: wasm_func_2
        (0x50, WasmSymbol::Data("__table".into())), // alias at same address
        (0x80, WasmSymbol::Func(2)),
    ];
    let syms = collect_symbols(entries, 0xa0, Some(&info));
    assert_eq!(syms.len(), 4);
    // Sorted by address, sizes = distance to next distinct address.
    assert_eq!(syms[0].name, "add_one");
    assert_eq!(syms[0].value, 0x30);
    assert_eq!(syms[0].size, 0x20);
    assert_eq!(syms[0].kind, SymbolKind::Func);
    assert_eq!(syms[1].value, 0x50);
    assert_eq!(syms[1].size, 0x30); // both aliases get the same size
    assert_eq!(syms[2].value, 0x50);
    assert_eq!(syms[2].size, 0x30);
    assert!(
        (syms[1].name == "wasm_func_2" && syms[2].name == "__table")
            || (syms[1].name == "__table" && syms[2].name == "wasm_func_2")
    );
    assert_eq!(syms[3].name, "wasm_func_3");
    assert_eq!(syms[3].size, 0x20); // extends to code_len
    // Without info, Func indices are used as-is.
    let bare = collect_symbols([(0u64, WasmSymbol::Func(7))], 0x10, None);
    assert_eq!(bare[0].name, "wasm_func_7");
}

#[test]
fn perf_map_format() {
    let out = emit_perf_map(&sample_symbols());
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], "1000 40 wasm_func_0");
    assert_eq!(lines[1], "1040 28 a_much_longer_function_name");
    assert_eq!(lines[2], "1080 10 embedded.table");
}

#[test]
fn elf_round_trip_all_arches() {
    let machines = [62u16, 3, 183, 40, 243, 243];
    for (arch, machine) in ALL_ARCHES.iter().zip(machines) {
        let bytes = emit_elf(&sample_symbols(), *arch).unwrap();
        let (m, syms) = parse_elf_symbols(&bytes);
        assert_eq!(m, machine, "e_machine for {arch:?}");
        assert_eq!(bytes[4], if arch.is_64() { 2 } else { 1 }, "class");
        assert_syms_match(&syms, &sample_symbols(), true);
    }
}

#[test]
fn macho_round_trip() {
    // (arch, expected cputype)
    let cases = [
        (Arch::X86_64, 0x0100_0007u32),
        (Arch::I686, 7),
        (Arch::AArch64, 0x0100_000c),
        (Arch::Arm, 12),
    ];
    for (arch, cputype) in cases {
        let bytes = emit_macho(&sample_symbols(), arch).unwrap();
        let (c, syms) = parse_macho_symbols(&bytes);
        assert_eq!(c, cputype, "cputype for {arch:?}");
        assert_syms_match(&syms, &sample_symbols(), false);
    }
}

#[test]
fn macho_rejects_riscv() {
    for arch in [Arch::Riscv64, Arch::Riscv32] {
        assert!(emit_macho(&sample_symbols(), arch).is_err());
    }
}

#[test]
fn coff_round_trip_all_arches() {
    let machines = [0x8664u16, 0x14c, 0xaa64, 0x1c4, 0x5064, 0x5032];
    for (arch, machine) in ALL_ARCHES.iter().zip(machines) {
        let bytes = emit_coff(&sample_symbols(), *arch).unwrap();
        let (m, syms) = parse_coff_symbols(&bytes);
        assert_eq!(m, machine, "Machine for {arch:?}");
        assert_syms_match(&syms, &sample_symbols(), false);
    }
}

/// Writes sample symbol maps to /tmp for inspection with native tools:
/// `SYMMAP_DUMP=1 cargo test -p portal-solutions-blitz-tests --test symmap -- --ignored`
/// then `nm /tmp/symmap.macho`, `nm /tmp/symmap.elf`, `llvm-nm /tmp/symmap.coff`.
#[test]
#[ignore]
fn dump_sample_files() {
    if std::env::var("SYMMAP_DUMP").is_err() {
        return;
    }
    let syms = sample_symbols();
    std::fs::write("/tmp/symmap.macho", emit_macho(&syms, Arch::AArch64).unwrap()).unwrap();
    std::fs::write("/tmp/symmap-x86_64.macho", emit_macho(&syms, Arch::X86_64).unwrap()).unwrap();
    std::fs::write("/tmp/symmap.elf", emit_elf(&syms, Arch::AArch64).unwrap()).unwrap();
    std::fs::write("/tmp/symmap-32.elf", emit_elf(&syms, Arch::Riscv32).unwrap()).unwrap();
    std::fs::write("/tmp/symmap.coff", emit_coff(&syms, Arch::X86_64).unwrap()).unwrap();
    std::fs::write("/tmp/symmap.map", emit_perf_map(&syms)).unwrap();
    eprintln!("wrote /tmp/symmap.{{macho,x86_64.macho,elf,32.elf,coff,map}}");
}

// ---------------------------------------------------------------------------
// End-to-end: compile a named WASM module natively, map labels → symbols
// ---------------------------------------------------------------------------

/// A 3-function module whose functions are named add_one / multiply_two /
/// main_entry via the name custom section.
fn named_module_wasm() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);
    module.section(&types);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    funcs.function(0);
    funcs.function(0);
    module.section(&funcs);
    let mut code = CodeSection::new();
    // f0: add
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::LocalGet(0));
    f0.instruction(&Instruction::LocalGet(1));
    f0.instruction(&Instruction::I64Add);
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    // f1: mul
    let mut f1 = Function::new([]);
    f1.instruction(&Instruction::LocalGet(0));
    f1.instruction(&Instruction::LocalGet(1));
    f1.instruction(&Instruction::I64Mul);
    f1.instruction(&Instruction::Return);
    f1.instruction(&Instruction::End);
    code.function(&f1);
    // f2: f0(a,b) - f1(a,b), exercising an internal call
    let mut f2 = Function::new([]);
    f2.instruction(&Instruction::LocalGet(0));
    f2.instruction(&Instruction::LocalGet(1));
    f2.instruction(&Instruction::Call(0));
    f2.instruction(&Instruction::LocalGet(0));
    f2.instruction(&Instruction::LocalGet(1));
    f2.instruction(&Instruction::Call(1));
    f2.instruction(&Instruction::I64Sub);
    f2.instruction(&Instruction::Return);
    f2.instruction(&Instruction::End);
    code.function(&f2);
    module.section(&code);
    let mut names = wasm_encoder::NameSection::new();
    let mut fnames = wasm_encoder::NameMap::new();
    fnames.append(0, "add_one");
    fnames.append(1, "multiply_two");
    fnames.append(2, "main_entry");
    names.functions(&fnames);
    module.section(&names);
    module.finish()
}

fn parse_sigs(wasm: &[u8]) -> (Vec<WpFuncType>, Vec<u32>) {
    let mut sigs_wp: Vec<WpFuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        match payload {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader.into_iter().flatten() {
                    for subtype in group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            subtype.composite_type.inner
                        {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(reader) => {
                for imp in reader.into_iter().flatten() {
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
    (sigs_wp, fsigs)
}

struct CompileParts {
    call_params: Vec<u32>,
    call_results: Vec<u32>,
    sig_params: Vec<u32>,
    sig_results: Vec<u32>,
    ops: Vec<portal_solutions_blitz_common::ops::MachOperator<'static, ()>>,
}

/// Parse the module into the operator stream and per-function/type arity
/// tables the SysV backends expect. Takes `'static` wasm bytes (the tests leak
/// the module, which is fine for test lifetimes) so the operator stream can
/// be collected.
fn compile_parts(wasm: &'static [u8]) -> CompileParts {
    let (sigs_wp, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    CompileParts {
        call_params: fsigs
            .iter()
            .map(|&ti| sigs_wp[ti as usize].params().len() as u32)
            .collect(),
        call_results: fsigs
            .iter()
            .map(|&ti| sigs_wp[ti as usize].results().len() as u32)
            .collect(),
        sig_params: sigs_wp.iter().map(|s| s.params().len() as u32).collect(),
        sig_results: sigs_wp.iter().map(|s| s.results().len() as u32).collect(),
        ops: ops.map(|r| r.unwrap()).collect(),
    }
}

/// Verify one backend's compiled output: the three named functions appear in
/// every platform symbol map, at offsets matching the backend's label map.
fn verify_symbols(
    bytes: &[u8],
    entries: Vec<(u64, WasmSymbol)>,
    info: &WasmSymbolInfo,
    backend: &str,
) {
    let syms = collect_symbols(entries, bytes.len() as u64, Some(info));
    assert_eq!(syms.len(), 3, "{backend}: expected 3 function symbols");
    for (i, s) in syms.iter().enumerate() {
        assert!(
            ["add_one", "multiply_two", "main_entry"].contains(&s.name.as_str()),
            "{backend}: unexpected symbol {}",
            s.name
        );
        assert!((s.value as usize) < bytes.len(), "{backend}: {}", s.name);
        if i + 1 < syms.len() {
            assert_eq!(
                s.size,
                syms[i + 1].value - s.value,
                "{backend}: {} size must reach the next symbol",
                s.name
            );
            assert!(s.size > 0, "{backend}: {} has zero size", s.name);
        }
    }

    // Every platform format must round-trip all three names at their offsets.
    let (m, elf) = parse_elf_symbols(&emit_elf(&syms, Arch::X86_64).unwrap());
    assert_eq!(m, 62);
    assert_syms_match(&elf, &syms, true);
    let (_, macho) = parse_macho_symbols(&emit_macho(&syms, Arch::AArch64).unwrap());
    assert_syms_match(&macho, &syms, false);
    let (_, coff) = parse_coff_symbols(&emit_coff(&syms, Arch::X86_64).unwrap());
    assert_syms_match(&coff, &syms, false);
    let perf = emit_perf_map(&syms);
    for name in ["add_one", "multiply_two", "main_entry"] {
        assert!(perf.contains(name), "{backend}: perf map missing {name}");
    }
}

#[test]
fn end_to_end_x86_64() {
    use portal_solutions_asm_x86_64::out::iced::IcedWriter;
    use portal_solutions_blitz_x86_64::{X64Arch, X64Label, sysv};

    let wasm: &'static [u8] = Box::leak(named_module_wasm().into_boxed_slice());
    let info = wasm_symbol_info(wasm).unwrap();
    let parts = compile_parts(wasm);
    let mut out = IcedWriter::<X64Label>::new(0x100000);
    let mut state = sysv::SysVState::default();
    state.call_abi = sysv::CallAbi::AllStack;
    state.call_params = parts.call_params.clone();
    state.call_results = parts.call_results.clone();
    state.sig_params = parts.sig_params.clone();
    state.sig_results = parts.sig_results.clone();
    let mut ctx = ();
    let mut reencoder = RoundtripReencoder;
    for op in &parts.ops {
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            X64Arch::default(),
            &mut state,
            &[],
            op,
            &mut reencoder,
            0,
        )
        .unwrap();
    }
    let (bytes, labels, relocs) = out.into_parts_with_relocs();
    assert!(relocs.is_empty(), "unexpected external relocs: {relocs:?}");
    let entries = labels
        .iter()
        .filter_map(|(l, off)| match l {
            X64Label::Func { r#fn } => Some((*off as u64, WasmSymbol::Func(*r#fn))),
            _ => None,
        })
        .collect();
    verify_symbols(&bytes, entries, &info, "x86-64");
}

#[test]
fn end_to_end_aarch64() {
    use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
    use portal_solutions_blitz_aarch64::{AArch64Arch, AArch64Label, naive, sysv};

    let wasm: &'static [u8] = Box::leak(named_module_wasm().into_boxed_slice());
    let info = wasm_symbol_info(wasm).unwrap();
    let parts = compile_parts(wasm);
    let mut out = AArch64Writer::<AArch64Label>::new();
    let mut state = naive::State::default();
    state.call_abi = naive::CallAbi::AllStack;
    state.call_params = parts.call_params.clone();
    state.call_results = parts.call_results.clone();
    state.sig_params = parts.sig_params.clone();
    state.sig_results = parts.sig_results.clone();
    let mut ctx = ();
    let mut reencoder = RoundtripReencoder;
    for op in &parts.ops {
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            AArch64Arch::default(),
            &mut state,
            &[],
            op,
            &mut reencoder,
            0,
        )
        .unwrap();
    }
    let (bytes, labels, relocs) = out.into_parts_with_relocs();
    assert!(relocs.is_empty(), "unexpected external relocs: {relocs:?}");
    // AArch64 SysV function entry labels are `Indexed { idx: 0x8000_0000 + f }`.
    let entries = labels
        .iter()
        .filter_map(|(l, off)| match l {
            AArch64Label::Indexed { idx } if *idx >= 0x8000_0000 => {
                Some((*off as u64, WasmSymbol::Func((*idx - 0x8000_0000) as u32)))
            }
            _ => None,
        })
        .collect();
    verify_symbols(&bytes, entries, &info, "aarch64");
}

#[test]
fn end_to_end_riscv64() {
    use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
    use portal_solutions_blitz_riscv64::{RiscvLabel, RiscV64Arch, sysv};

    let wasm: &'static [u8] = Box::leak(named_module_wasm().into_boxed_slice());
    let info = wasm_symbol_info(wasm).unwrap();
    let parts = compile_parts(wasm);
    let mut out = RvAsmWriter::<RiscvLabel>::new();
    let mut state = sysv::SysVState::default();
    state.call_abi = sysv::CallAbi::AllStack;
    state.n_imports = 0;
    state.call_params = parts.call_params.clone();
    state.call_results = parts.call_results.clone();
    state.sig_params = parts.sig_params.clone();
    state.sig_results = parts.sig_results.clone();
    let mut ctx = ();
    let mut reencoder = RoundtripReencoder;
    for op in &parts.ops {
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            RiscV64Arch::default(),
            &mut state,
            &[],
            op,
            &mut reencoder,
            0,
        )
        .unwrap();
    }
    let (bytes, labels) = out.into_parts();
    // RISC-V64 SysV function entry labels are `Indexed { idx: (1 << 28) + f }`.
    let entries = labels
        .iter()
        .filter_map(|(l, off)| match l {
            RiscvLabel::Indexed { idx } if *idx & (1 << 28) != 0 => {
                Some((*off as u64, WasmSymbol::Func((*idx - (1 << 28)) as u32)))
            }
            _ => None,
        })
        .collect();
    verify_symbols(&bytes, entries, &info, "riscv64");
}
