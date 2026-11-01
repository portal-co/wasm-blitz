//! Symbol map emission for blitz-compiled WASM code.
//!
//! Converts WASM function names (from the WASM *name* custom section) or plain
//! function indices into platform-native **symbol map files**, mapping native
//! code addresses back to the WASM functions they were compiled from:
//!
//! | Platform | Emitter                      | Format                       |
//! |----------|------------------------------|------------------------------|
//! | macOS    | [`emit_macho`]               | Mach-O with `LC_SYMTAB`      |
//! | Linux    | [`emit_elf`] / [`emit_perf_map`] | ELF `.symtab` / perf map |
//! | Windows  | [`emit_coff`]                | COFF symbol table            |
//!
//! All emitters take a `&[Symbol]` (name + absolute address + size) and return
//! the finished file bytes; the emitted objects carry an absolute-symbol table
//! only (no sections), which is what debuggers/profilers need to resolve
//! addresses in JIT-compiled or otherwise externally-linked code.
//!
//! The typical pipeline is:
//!
//! 1. Compile a module with any native backend; each backend's writer yields a
//!    label → offset map (e.g. `AArch64Label::Indexed { idx: 0x8000_0000 + f }`
//!    or `X64Label::Func { fn: i }` at every function entry).
//! 2. Turn the label map into `(offset, WasmSymbol)` pairs (a 3-line closure
//!    per backend — see `crates/blitz-tests/tests/symmap.rs`).
//! 3. [`collect_symbols`] fills in names (WASM name section, falling back to
//!    `wasm_func_{i}`) and sizes (distance to the next symbol).
//! 4. Emit with the platform emitter of choice.
//!
//! All supported targets are little-endian; the emitted files are always
//! little-endian regardless of host.
#![no_std]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A native target architecture supported by the blitz backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Arch {
    X86_64,
    I686,
    AArch64,
    /// 32-bit ARM (armv7, Thumb-2 capable).
    Arm,
    Riscv64,
    Riscv32,
}

impl Arch {
    /// Whether this is a 64-bit architecture (drives 64- vs 32-bit container
    /// layouts for ELF and Mach-O).
    pub fn is_64(self) -> bool {
        matches!(self, Arch::X86_64 | Arch::AArch64 | Arch::Riscv64)
    }

    /// ELF `e_machine` value.
    fn elf_machine(self) -> u16 {
        match self {
            Arch::X86_64 => 62,                   // EM_X86_64
            Arch::I686 => 3,                      // EM_386
            Arch::AArch64 => 183,                 // EM_AARCH64
            Arch::Arm => 40,                      // EM_ARM
            Arch::Riscv64 | Arch::Riscv32 => 243, // EM_RISCV
        }
    }

    /// ELF `e_flags` value.
    fn elf_flags(self) -> u32 {
        match self {
            // EF_ARM_EABI_VER5 — required for ARM objects to be well-formed.
            Arch::Arm => 0x0500_0000,
            _ => 0,
        }
    }

    /// Mach-O `(cputype, cpusubtype)`; `None` when the architecture has no
    /// Mach-O encoding (RISC-V).
    fn macho_cpu(self) -> Option<(u32, u32)> {
        Some(match self {
            Arch::X86_64 => (0x0100_0007, 3), // CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL
            Arch::I686 => (7, 3),             // CPU_TYPE_I386, CPU_SUBTYPE_I386_ALL
            Arch::AArch64 => (0x0100_000C, 0), // CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL
            Arch::Arm => (12, 9),             // CPU_TYPE_ARM, CPU_SUBTYPE_ARM_V7
            Arch::Riscv64 | Arch::Riscv32 => return None,
        })
    }

    /// COFF `Machine` value.
    fn coff_machine(self) -> u16 {
        match self {
            Arch::X86_64 => 0x8664,  // IMAGE_FILE_MACHINE_AMD64
            Arch::I686 => 0x014c,    // IMAGE_FILE_MACHINE_I386
            Arch::AArch64 => 0xaa64, // IMAGE_FILE_MACHINE_ARM64
            Arch::Arm => 0x01c4,     // IMAGE_FILE_MACHINE_ARMNT (Thumb-2 LE)
            Arch::Riscv32 => 0x5032, // IMAGE_FILE_MACHINE_RISCV32
            Arch::Riscv64 => 0x5064, // IMAGE_FILE_MACHINE_RISCV64
        }
    }
}

/// What kind of thing a [`Symbol`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// A compiled function entry point.
    Func,
    /// A data object embedded in the code image (e.g. an inlined table).
    Data,
}

/// One named address range in a compiled code image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// Sanitized symbol name (see [`sanitize_name`]).
    pub name: String,
    /// Absolute address (or offset) of the symbol in the loaded image.
    pub value: u64,
    /// Byte size; 0 is allowed (e.g. trailing/alias symbols).
    pub size: u64,
    pub kind: SymbolKind,
}

/// A raw label-to-symbol input for [`collect_symbols`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmSymbol {
    /// Entry of a *defined* WASM function, 0-based among defined functions
    /// (i.e. excluding imported functions). [`collect_symbols`] adds the
    /// import count to obtain the absolute WASM function index used by the
    /// name section.
    Func(u32),
    /// A named data object at the offset.
    Data(String),
}

/// Symbol map emission / parsing error.
#[derive(Debug, Clone)]
pub enum Error {
    /// The requested file format has no encoding for this architecture
    /// (e.g. Mach-O for RISC-V).
    UnsupportedCombination,
    /// The WASM input could not be parsed.
    Parse(wasmparser::BinaryReaderError),
    /// More than `u32::MAX` symbols were supplied.
    TooManySymbols,
}

impl From<wasmparser::BinaryReaderError> for Error {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        Error::Parse(e)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::UnsupportedCombination => {
                write!(
                    f,
                    "no symbol file encoding for this architecture/format combination"
                )
            }
            Error::Parse(e) => write!(f, "invalid WASM module: {e}"),
            Error::TooManySymbols => write!(f, "too many symbols"),
        }
    }
}

impl core::error::Error for Error {}

/// Make `name` safe to embed in a Mach-O/ELF/COFF symbol table or a perf map:
/// keep `[A-Za-z0-9_.$]`, replace everything else (including spaces, which
/// would break the perf-map line format) with `_`. Never returns an empty
/// string.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

/// Symbolic names extracted from a WASM module, keyed by *absolute* function
/// index (imported functions included, matching the name section's indexing).
#[derive(Debug, Clone, Default)]
pub struct WasmSymbolInfo {
    /// Function index → name, from the WASM `name` custom section (empty when
    /// the module has no name section).
    pub names: BTreeMap<u32, String>,
    /// Number of imported functions (defined-function `f` has absolute index
    /// `imported_funcs + f`).
    pub imported_funcs: u32,
}

/// Extract function names (name custom section) and the imported-function
/// count from a WASM module.
pub fn wasm_symbol_info(wasm: &[u8]) -> Result<WasmSymbolInfo, Error> {
    let mut info = WasmSymbolInfo::default();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload? {
            wasmparser::Payload::CustomSection(cs) if cs.name() == "name" => {
                let rdr = wasmparser::NameSectionReader::new(wasmparser::BinaryReader::new(
                    cs.data(),
                    cs.data_offset(),
                ));
                for sub in rdr {
                    if let wasmparser::Name::Function(map) = sub? {
                        for naming in map {
                            let n = naming?;
                            info.names.insert(n.index, String::from(n.name));
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(r) => {
                for imp in r {
                    if let wasmparser::TypeRef::Func(_) = imp?.ty {
                        info.imported_funcs += 1;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(info)
}

/// Build the final [`Symbol`] list from `(offset, WasmSymbol)` pairs.
///
/// - `WasmSymbol::Func(f)` is named from `info.names[info.imported_funcs + f]`
///   when present, else `wasm_func_{absolute_index}`.
/// - Symbols are sorted by address and their sizes are computed as the
///   distance to the next distinct address; the last symbol extends to
///   `code_len`.
/// - Aliases (several names at the same address) are all kept and receive the
///   same size.
pub fn collect_symbols(
    entries: impl IntoIterator<Item = (u64, WasmSymbol)>,
    code_len: u64,
    info: Option<&WasmSymbolInfo>,
) -> Vec<Symbol> {
    let mut syms: Vec<Symbol> = entries
        .into_iter()
        .map(|(value, ws)| {
            let (name, kind) = match ws {
                WasmSymbol::Func(local) => {
                    let abs = info.map_or(local, |i| i.imported_funcs + local);
                    let name = info
                        .and_then(|i| i.names.get(&abs))
                        .map(|n| sanitize_name(n))
                        .unwrap_or_else(|| format!("wasm_func_{abs}"));
                    (name, SymbolKind::Func)
                }
                WasmSymbol::Data(n) => (sanitize_name(&n), SymbolKind::Data),
            };
            Symbol {
                name,
                value,
                size: 0,
                kind,
            }
        })
        .collect();
    syms.sort_by(|a, b| a.value.cmp(&b.value).then_with(|| a.name.cmp(&b.name)));
    for i in 0..syms.len() {
        let mut j = i + 1;
        while j < syms.len() && syms[j].value == syms[i].value {
            j += 1;
        }
        let end = if j < syms.len() {
            syms[j].value
        } else {
            code_len
        };
        syms[i].size = end.saturating_sub(syms[i].value);
    }
    syms
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

fn w16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn w32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn w64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

fn pad_to(b: &mut Vec<u8>, len: usize) {
    b.resize(len, 0);
}

/// Build a string table starting with a NUL (Mach-O/ELF convention) and return
/// it plus one offset per symbol name.
fn strtab_with_nul(symbols: &[Symbol]) -> (Vec<u8>, Vec<u32>) {
    let mut tab = alloc::vec![0u8];
    let mut offs = Vec::with_capacity(symbols.len());
    for s in symbols {
        // A name longer than u32::MAX is absurd; offsets are u32 in all three
        // formats, so clamp rather than wrap.
        offs_push(&mut offs, tab.len());
        tab.extend_from_slice(s.name.as_bytes());
        tab.push(0);
    }
    (tab, offs)
}

fn offs_push(offs: &mut Vec<u32>, v: usize) {
    offs.push(u32::try_from(v).unwrap_or(u32::MAX));
}

// ---------------------------------------------------------------------------
// ELF (Linux)
// ---------------------------------------------------------------------------

/// Emit a little-endian ELF object (`ET_REL`) whose `.symtab` holds every
/// symbol as a global absolute (`SHN_ABS`) symbol. Readable by `nm`,
/// `readelf -sW`, `perf`, gdb and lldb.
pub fn emit_elf(symbols: &[Symbol], arch: Arch) -> Result<Vec<u8>, Error> {
    let is64 = arch.is_64();
    let nsyms = u32::try_from(symbols.len()).map_err(|_| Error::TooManySymbols)? + 1;
    let (strtab, stroffs) = strtab_with_nul(symbols);
    let shstrtab: &[u8] = b"\0.symtab\0.strtab\0.shstrtab\0";
    let (sh_name_symtab, sh_name_strtab, sh_name_shstrtab) = (1u32, 9u32, 17u32);

    let ehsize: usize = if is64 { 64 } else { 52 };
    let shentsize: usize = if is64 { 64 } else { 40 };
    let symentsize: usize = if is64 { 24 } else { 16 };
    let align: usize = if is64 { 8 } else { 4 };

    let strtab_off = ehsize;
    let shstrtab_off = strtab_off + strtab.len();
    let symtab_off = align_up(shstrtab_off + shstrtab.len(), align);
    let symtab_size = nsyms as usize * symentsize;
    let shoff = align_up(symtab_off + symtab_size, align);

    let mut b = Vec::with_capacity(shoff + 4 * shentsize);

    // ELF header.
    b.extend_from_slice(b"\x7fELF");
    b.push(if is64 { 2 } else { 1 }); // EI_CLASS
    b.push(1); // EI_DATA = LSB
    b.push(1); // EI_VERSION
    b.push(0); // EI_OSABI = System V
    b.extend_from_slice(&[0; 8]); // ABI version + padding
    w16(&mut b, 1); // e_type = ET_REL
    w16(&mut b, arch.elf_machine());
    w32(&mut b, 1); // e_version
    if is64 {
        w64(&mut b, 0); // e_entry
        w64(&mut b, 0); // e_phoff
        w64(&mut b, shoff as u64); // e_shoff
        w32(&mut b, arch.elf_flags());
        w16(&mut b, ehsize as u16);
        w16(&mut b, 56); // e_phentsize
        w16(&mut b, 0); // e_phnum
        w16(&mut b, shentsize as u16);
        w16(&mut b, 4); // e_shnum
        w16(&mut b, 3); // e_shstrndx
    } else {
        w32(&mut b, 0); // e_entry
        w32(&mut b, 0); // e_phoff
        w32(&mut b, shoff as u32); // e_shoff
        w32(&mut b, arch.elf_flags());
        w16(&mut b, ehsize as u16);
        w16(&mut b, 32); // e_phentsize
        w16(&mut b, 0); // e_phnum
        w16(&mut b, shentsize as u16);
        w16(&mut b, 4); // e_shnum
        w16(&mut b, 3); // e_shstrndx
    }
    debug_assert_eq!(b.len(), ehsize);

    // .strtab + .shstrtab contents, then .symtab.
    b.extend_from_slice(&strtab);
    debug_assert_eq!(b.len(), shstrtab_off);
    b.extend_from_slice(shstrtab);
    pad_to(&mut b, symtab_off);
    // Symbol 0: the required all-zero null symbol.
    pad_to(&mut b, symtab_off + symentsize);
    for (i, s) in symbols.iter().enumerate() {
        let st_info: u8 = (1 << 4)
            | match s.kind {
                SymbolKind::Func => 2, // STT_FUNC
                SymbolKind::Data => 1, // STT_OBJECT
            };
        if is64 {
            w32(&mut b, stroffs[i]);
            b.push(st_info);
            b.push(0); // st_other
            w16(&mut b, 0xfff1); // st_shndx = SHN_ABS
            w64(&mut b, s.value);
            w64(&mut b, s.size);
        } else {
            w32(&mut b, stroffs[i]);
            w32(&mut b, s.value as u32);
            w32(&mut b, s.size as u32);
            b.push(st_info);
            b.push(0); // st_other
            w16(&mut b, 0xfff1); // st_shndx = SHN_ABS
        }
    }
    pad_to(&mut b, shoff);

    // Section headers: [0] null, [1] .symtab, [2] .strtab, [3] .shstrtab.
    let shdr = |b: &mut Vec<u8>,
                name: u32,
                ty: u32,
                off: usize,
                size: usize,
                link: u32,
                info: u32,
                addralign: usize,
                entsize: usize| {
        if is64 {
            w32(b, name);
            w32(b, ty);
            w64(b, 0); // sh_flags
            w64(b, 0); // sh_addr
            w64(b, off as u64);
            w64(b, size as u64);
            w32(b, link);
            w32(b, info);
            w64(b, addralign as u64);
            w64(b, entsize as u64);
        } else {
            w32(b, name);
            w32(b, ty);
            w32(b, 0); // sh_flags
            w32(b, 0); // sh_addr
            w32(b, off as u32);
            w32(b, size as u32);
            w32(b, link);
            w32(b, info);
            w32(b, addralign as u32);
            w32(b, entsize as u32);
        }
    };
    pad_to(&mut b, shoff + shentsize); // [0] null section header
    // [1] .symtab: sh_link = 2 (.strtab), sh_info = 1 (first non-local symbol).
    shdr(
        &mut b,
        sh_name_symtab,
        2,
        symtab_off,
        symtab_size,
        2,
        1,
        align,
        symentsize,
    );
    // [2] .strtab, [3] .shstrtab (both SHT_STRTAB = 3).
    shdr(
        &mut b,
        sh_name_strtab,
        3,
        strtab_off,
        strtab.len(),
        0,
        0,
        1,
        0,
    );
    shdr(
        &mut b,
        sh_name_shstrtab,
        3,
        shstrtab_off,
        shstrtab.len(),
        0,
        0,
        1,
        0,
    );
    debug_assert_eq!(b.len(), shoff + 4 * shentsize);
    Ok(b)
}

// ---------------------------------------------------------------------------
// Mach-O (macOS)
// ---------------------------------------------------------------------------

/// Emit a little-endian Mach-O (`MH_OBJECT`) whose `LC_SYMTAB` holds every
/// symbol as an external absolute (`N_EXT | N_ABS`) symbol. Names are stored
/// as given — no leading-underscore mangling is applied. Readable by `nm` and
/// `lldb`. Returns [`Error::UnsupportedCombination`] for RISC-V (no Mach-O
/// CPU type exists for it).
pub fn emit_macho(symbols: &[Symbol], arch: Arch) -> Result<Vec<u8>, Error> {
    let (cputype, cpusubtype) = arch.macho_cpu().ok_or(Error::UnsupportedCombination)?;
    let is64 = arch.is_64();
    let nsyms = u32::try_from(symbols.len()).map_err(|_| Error::TooManySymbols)?;
    let (strtab, stroffs) = strtab_with_nul(symbols);

    let header_size: usize = if is64 { 32 } else { 28 };
    let cmd_size: usize = 24;
    let nl_size: usize = if is64 { 16 } else { 12 };
    let symoff = header_size + cmd_size;
    let stroff = symoff + symbols.len() * nl_size;

    let mut b = Vec::with_capacity(stroff + strtab.len());

    // Mach header.
    w32(&mut b, if is64 { 0xfeed_facf } else { 0xfeed_face }); // MH_MAGIC_64 / MH_MAGIC
    w32(&mut b, cputype);
    w32(&mut b, cpusubtype);
    w32(&mut b, 1); // filetype = MH_OBJECT
    w32(&mut b, 1); // ncmds
    w32(&mut b, cmd_size as u32); // sizeofcmds
    w32(&mut b, 0); // flags
    if is64 {
        w32(&mut b, 0); // reserved
    }
    // LC_SYMTAB.
    w32(&mut b, 0x2); // cmd = LC_SYMTAB
    w32(&mut b, cmd_size as u32);
    w32(&mut b, symoff as u32);
    w32(&mut b, nsyms);
    w32(&mut b, stroff as u32);
    w32(&mut b, strtab.len() as u32);
    debug_assert_eq!(b.len(), symoff);

    // nlist entries.
    for (i, s) in symbols.iter().enumerate() {
        w32(&mut b, stroffs[i]); // n_strx
        b.push(0x01 | 0x02); // n_type = N_EXT | N_ABS
        b.push(0); // n_sect = NO_SECT
        w16(&mut b, 0); // n_desc
        if is64 {
            w64(&mut b, s.value);
        } else {
            w32(&mut b, s.value as u32);
        }
    }
    debug_assert_eq!(b.len(), stroff);
    b.extend_from_slice(&strtab);
    Ok(b)
}

// ---------------------------------------------------------------------------
// COFF (Windows)
// ---------------------------------------------------------------------------

/// Emit a COFF object whose symbol table holds every symbol as an
/// `IMAGE_SYM_ABSOLUTE`, `IMAGE_SYM_CLASS_EXTERNAL` entry. Names longer than
/// 8 bytes go into the COFF string table automatically. Readable by
/// `dumpbin /SYMBOLS` and `llvm-nm`.
///
/// Note: the COFF symbol `Value` field is 32-bit by specification, so symbol
/// addresses are truncated to 32 bits. When full 64-bit addresses matter on
/// Windows, emit a [`emit_perf_map`] text map alongside.
pub fn emit_coff(symbols: &[Symbol], arch: Arch) -> Result<Vec<u8>, Error> {
    let nsyms = u32::try_from(symbols.len()).map_err(|_| Error::TooManySymbols)?;

    // COFF string table starts with its own 4-byte total size; string offsets
    // are relative to the start of the table, so the first string is at 4.
    let mut strtab = alloc::vec![0u8; 4];
    let mut long_offs: Vec<Option<u32>> = Vec::with_capacity(symbols.len());
    for s in symbols {
        if s.name.len() <= 8 {
            long_offs.push(None);
        } else {
            offs_push_strtab(&mut long_offs, strtab.len());
            strtab.extend_from_slice(s.name.as_bytes());
            strtab.push(0);
        }
    }
    let strtab_len = strtab.len() as u32;
    strtab[..4].copy_from_slice(&strtab_len.to_le_bytes());

    let header_size = 20usize;
    let symoff = header_size;

    let mut b = Vec::with_capacity(symoff + symbols.len() * 18 + strtab.len());

    // IMAGE_FILE_HEADER.
    w16(&mut b, arch.coff_machine());
    w16(&mut b, 0); // NumberOfSections
    w32(&mut b, 0); // TimeDateStamp
    w32(&mut b, symoff as u32); // PointerToSymbolTable
    w32(&mut b, nsyms); // NumberOfSymbols
    w16(&mut b, 0); // SizeOfOptionalHeader
    w16(&mut b, 0); // Characteristics

    // Symbol entries (18 bytes each).
    for (i, s) in symbols.iter().enumerate() {
        match long_offs[i] {
            None => {
                let mut name = [0u8; 8];
                name[..s.name.len()].copy_from_slice(s.name.as_bytes());
                b.extend_from_slice(&name);
            }
            Some(off) => {
                w32(&mut b, 0); // zeroes flag a string-table reference
                w32(&mut b, off);
            }
        }
        w32(&mut b, s.value as u32); // Value (32-bit per COFF spec)
        w16(&mut b, 0xffff); // SectionNumber = IMAGE_SYM_ABSOLUTE (-1)
        w16(
            &mut b,
            match s.kind {
                SymbolKind::Func => 0x20, // IMAGE_SYM_DTYPE_FUNCTION << 4
                SymbolKind::Data => 0,
            },
        );
        b.push(2); // StorageClass = IMAGE_SYM_CLASS_EXTERNAL
        b.push(0); // NumberOfAuxSymbols
    }
    b.extend_from_slice(&strtab);
    Ok(b)
}

fn offs_push_strtab(offs: &mut Vec<Option<u32>>, v: usize) {
    offs.push(Some(u32::try_from(v).unwrap_or(u32::MAX)));
}

// ---------------------------------------------------------------------------
// perf map (Linux JIT symbol map; also a handy plain-text map anywhere)
// ---------------------------------------------------------------------------

/// Emit a Linux [perf map](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/jit-interface.txt)
/// — one `<addr-hex> <size-hex> <name>` line per symbol. Written to
/// `/tmp/perf-<pid>.map`, it lets `perf report` attribute JIT-compiled frames
/// to WASM functions. The format is plain text, so it also doubles as a
/// portable symbol map on platforms without a COFF/Mach-O/ELF consumer.
pub fn emit_perf_map(symbols: &[Symbol]) -> String {
    let mut out = String::new();
    for s in symbols {
        out.push_str(&format!("{:x} {:x} {}\n", s.value, s.size, s.name));
    }
    out
}
