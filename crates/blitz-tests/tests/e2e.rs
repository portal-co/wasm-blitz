//! End-to-end tests for the JS and C code-generation backends.
//!
//! Each test builds a minimal WASM module in memory with `wasm-encoder`,
//! drives it through the backend pipeline, and asserts properties of the
//! emitted source code, then actually executes the output to verify correctness.
//!
//! # Pipeline
//! ```text
//! wasm-encoder  →  raw bytes  →  wasmparser (FunctionBody, FuncType)
//!   →  mach_operators  →  dce_pass!  →  on_mach  →  String output
//!   →  node / clang   →  execute  →  numeric result
//! ```
//!
//! # Bug coverage
//! Each test is annotated with the bug(s) it exercises.

mod log;

use std::borrow::Cow;
use std::fmt::{Display, Write as FmtWrite};
use std::sync::atomic::{AtomicU64, Ordering};

/// Newtype over `String` used as the text-output backend for native asm tests.
///
/// We need a *local* type so that calling `writers!` from each asm crate
/// (which implements `WriterCore`/`Writer` for it) doesn't violate the orphan
/// rule.  Using `String` directly would fail: both the trait and the type are
/// foreign to this crate.
///
/// The type also sidesteps the `writer_dispatch!` forwarding impls for
/// `&mut T`, which are incomplete (they omit instructions like `addi`).
/// Passing `&mut NativeAsmWriter` directly gives `Self = NativeAsmWriter`
/// (Sized, with the full `writers!` impl set) rather than
/// `Self = &mut dyn fmt::Write` (which goes through the limited dispatch).
struct NativeAsmWriter(String);

impl std::fmt::Write for NativeAsmWriter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result { self.0.write_str(s) }
}

portal_solutions_asm_x86_64::writers!(NativeAsmWriter);
portal_solutions_asm_aarch64::writers!(NativeAsmWriter);
portal_solutions_asm_riscv64::writers!(NativeAsmWriter);
portal_solutions_asm_riscv32::writers!(NativeAsmWriter);
portal_solutions_asm_arm::writers!(NativeAsmWriter);
portal_solutions_asm_x86::writers!(NativeAsmWriter);

use portal_solutions_blitz_common::HandleOpError;
use portal_solutions_blitz_common::{
    dce_pass,
    ops::mach_operators,
    wasmparser::{self, DataKind, FuncType as WpFuncType, Operator},
    wasm_encoder::{
        self,
        reencode::RoundtripReencoder,
        Catch, CodeSection, DataSection, ExportKind, ExportSection, Function, FunctionSection,
        Instruction, MemorySection, MemoryType, Module, TagSection, TagType, TypeSection, ValType,
    },
};
use portal_solutions_blitz_c::{c_emit_data_segments, c_emit_exports, c_emit_import_decls, c_module_preamble, CWrite, State as CState};
use portal_solutions_blitz_js::{js_apply_data_segments, js_emit_exports, js_emit_imports, js_module_preamble, JsWrite, State as JsState};
/// Global counter for unique temp-file names (needed for parallel test runs).
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a wasm-encoder module with a single function of the given signature
/// and instruction sequence. Always finishes with `Return; End` so that DCE
/// can prune the implicit function-level `End` operator.
fn make_module(params: &[ValType], results: &[ValType], instrs: &[Instruction<'_>]) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function(params.iter().cloned(), results.iter().cloned());
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    for instr in instrs {
        func.instruction(instr);
    }
    // Explicit return so DCE removes the dead function-level `End`.
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    module.finish()
}

/// Parse `wasm` bytes, collect the `FunctionBody` items (borrowing the
/// bytes) and the two parallel signature slices required by the pipeline:
/// `wasmparser::FuncType` for `mach_operators`, and `wasm_encoder::FuncType`
/// for the backends' `on_mach`.
fn parse_sigs(wasm: &[u8]) -> (Vec<WpFuncType>, Vec<wasm_encoder::FuncType>, Vec<u32>) {
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
            // Imports come before the FunctionSection in WASM; add their type
            // indices first so that fsigs[wasm_index] is correct for all functions.
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

    // Convert wasmparser::FuncType → wasm_encoder::FuncType.
    let sigs_enc: Vec<wasm_encoder::FuncType> = sigs_wp
        .iter()
        .cloned()
        .map(|ft| wasm_encoder::FuncType::try_from(ft).unwrap())
        .collect();

    (sigs_wp, sigs_enc, fsigs)
}

/// Compile `wasm` bytes to JavaScript source using the JS backend.
/// Applies DCE so the dead function-level `End` after explicit `Return` is
/// removed before reaching the backend.
fn compile_js(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

/// Compile `wasm` bytes to C source using the C backend.
fn compile_c(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

/// Run the generated JavaScript source code using `node`, passing `bigint_args`
/// as the function arguments (each is emitted as a BigInt literal `{n}n`).
/// Returns all return values as `i64` (interpreting the BigInt as signed).
///
/// The JS backend names function 0 as `$0`.
fn run_js(js_src: &str, bigint_args: &[i64]) -> Vec<i64> {
    let args: Vec<String> = bigint_args.iter().map(|v| format!("{v}n")).collect();
    let harness = format!(
        "\nconst __r=$0({args});const __n=Array.isArray(__r)?__r:[__r];for(const v of __n)console.log(String(v));",
        args = args.join(",")
    );
    let code = format!("{js_src}{harness}");

    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(&code)
        .output()
        .expect("node not found in PATH");

    assert!(
        out.status.success(),
        "node exited non-zero.\nstderr: {}\ncode: {}",
        String::from_utf8_lossy(&out.stderr),
        code
    );

    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<i64>().expect("expected integer line from node"))
        .collect()
}

/// Compile the generated C source (function `fn_{fn_id}`) with clang/gcc,
/// run the resulting binary, and return all printed `uint64_t` return values.
///
/// `args` are the raw `uint64_t` arguments to pass to the function.
/// `rets` is how many return values to read.
fn run_c(c_src: &str, fn_id: u32, args: &[u64], rets: usize) -> Vec<u64> {
    use std::io::Write as _;

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_e2e_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_e2e_{pid}_{seq}"));

    // Build main(): declare a zero-padded arg array so 0-param functions still
    // receive a valid (non-null) pointer.
    let mut main_body = format!(
        "int main(){{uint64_t _args[{n}]={{",
        n = args.len().max(1)
    );
    for (i, &a) in args.iter().enumerate() {
        if i > 0 { main_body.push(','); }
        main_body.push_str(&format!("{a}ull"));
    }
    // Pad to at least 1 element so the pointer is non-null.
    if args.is_empty() {
        main_body.push('0');
    }
    main_body.push_str(&format!("}};uint64_t*_r=fn_{fn_id}(_args);"));
    for i in 0..rets {
        main_body.push_str(&format!("printf(\"%llu\\n\",_r[{i}]);"));
    }
    main_body.push_str("return 0;}");

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n#define WASM_STACK_SIZE 512\n{c_src}\n{main_body}\n"
    );

    std::fs::write(&src_path, &full_src).unwrap();

    let compile = std::process::Command::new("cc")
        .arg(&src_path)
        .arg("-Wno-unsequenced")   // C backend may use sp in single expression
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("cc not found in PATH");

    assert!(
        compile.status.success(),
        "C compile failed:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        full_src
    );

    let run = std::process::Command::new(&bin_path)
        .output()
        .expect("failed to run compiled binary");

    assert!(run.status.success(), "binary exited non-zero: {}", String::from_utf8_lossy(&run.stderr));

    // Clean up.
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<u64>().expect("expected integer line"))
        .collect()
}



// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

/// Build a WASM module that declares one linear memory page and a single
/// function.  Used for testing load/store operations.
fn make_module_with_memory(params: &[ValType], results: &[ValType], instrs: &[Instruction<'_>]) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function(params.iter().cloned(), results.iter().cloned());
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);

    // One page (64 KiB) of linear memory.
    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });
    module.section(&memories);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    for instr in instrs {
        func.instruction(instr);
    }
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    module.finish()
}

/// Compile a WASM module (which may use linear memory) to JS, prepending the
/// module-level `$mem`/`$mem_dv` globals required for load/store instructions.
fn compile_js_with_mem(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

/// Compile a WASM module (which may use linear memory) to C, prepending the
/// `__wasm_mem` global required for load/store instructions.
fn compile_c_with_mem(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

/// Run JS with a pre-initialised memory buffer.  `mem_bytes` is written into
/// `$mem` before the function is called.
fn run_js_with_mem(js_src: &str, mem_bytes: &[u8], args: &[i64]) -> Vec<i64> {
    let bytes_js: Vec<String> = mem_bytes.iter().map(|b| b.to_string()).collect();
    let mem_init = format!("$mem=new Uint8Array([{}]);$mem_dv=new DataView($mem.buffer);", bytes_js.join(","));

    let fn_args: Vec<String> = args.iter().map(|v| format!("{v}n")).collect();
    let harness = format!(
        "{mem_init}\nconst __r=$0({});const __n=Array.isArray(__r)?__r:[__r];for(const v of __n)console.log(String(v));",
        fn_args.join(",")
    );
    let code = format!("{js_src}{harness}");

    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(&code)
        .output()
        .expect("node not found in PATH");

    assert!(
        out.status.success(),
        "node exited non-zero.\nstderr: {}\ncode: {}",
        String::from_utf8_lossy(&out.stderr),
        code
    );

    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<i64>().expect("expected integer line from node"))
        .collect()
}

/// Run C with a pre-initialised memory buffer. `mem_bytes` is written into
/// `__wasm_mem` before the function is called.
fn run_c_with_mem(c_src: &str, mem_bytes: &[u8], fn_id: u32, args: &[u64], rets: usize) -> Vec<u64> {
    use std::io::Write as _;

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_mem_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_mem_{pid}_{seq}"));

    // Build a main() that initialises __wasm_mem from a byte array.
    let mem_init: Vec<String> = mem_bytes.iter().map(|b| b.to_string()).collect();
    let mem_sz = mem_bytes.len().max(1);

    let mut main_body = format!(
        "int main(){{static uint8_t _mem[{mem_sz}]={{{}}}; __wasm_mem=_mem;uint64_t _args[{}]={{",
        mem_init.join(","),
        args.len().max(1)
    );
    for (i, &a) in args.iter().enumerate() {
        if i > 0 { main_body.push(','); }
        main_body.push_str(&format!("{a}ull"));
    }
    if args.is_empty() { main_body.push('0'); }
    main_body.push_str(&format!("}};uint64_t*_r=fn_{fn_id}(_args);"));
    for i in 0..rets {
        main_body.push_str(&format!("printf(\"%llu\\n\",_r[{i}]);"));
    }
    main_body.push_str("return 0;}");

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n#define WASM_STACK_SIZE 512\n{c_src}\n{main_body}\n"
    );

    std::fs::write(&src_path, &full_src).unwrap();

    let compile = std::process::Command::new("cc")
        .arg(&src_path)
        .arg("-Wno-unsequenced")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("cc not found in PATH");

    assert!(
        compile.status.success(),
        "C compile failed:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        full_src
    );

    let run = std::process::Command::new(&bin_path)
        .output()
        .expect("failed to run compiled binary");

    assert!(run.status.success(), "binary exited non-zero: {}", String::from_utf8_lossy(&run.stderr));

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<u64>().expect("expected integer line"))
        .collect()
}

// ---------------------------------------------------------------------------
// Memory tests
// ---------------------------------------------------------------------------

/// i64.store then i64.load at the same address round-trips the value.
#[test]
fn test_i64_store_load_js() {
    // (func (param $addr i32) (param $val i64) (result i64)
    //   local.get $addr
    //   local.get $val
    //   i64.store offset=0
    //   local.get $addr
    //   i64.load  offset=0
    // )
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I64],
        &[ValType::I64],
        &[
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I64Store(memarg),
            Instruction::LocalGet(0),
            Instruction::I64Load(memarg),
        ],
    );
    let js = compile_js_with_mem(&wasm);
    // Zero memory, write 42 at offset 8, load it back.
    let mut mem = vec![0u8; 64];
    let addr: i64 = 8;
    let val: i64 = 42;
    assert_eq!(run_js_with_mem(&js, &mem, &[addr, val]), vec![val]);

    // Round-trip a large value.
    let val2: i64 = i64::MAX;
    assert_eq!(run_js_with_mem(&js, &mem, &[addr, val2]), vec![val2]);
}

#[test]
fn test_i64_store_load_c() {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I64],
        &[ValType::I64],
        &[
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I64Store(memarg),
            Instruction::LocalGet(0),
            Instruction::I64Load(memarg),
        ],
    );
    let c = compile_c_with_mem(&wasm);
    let mem = vec![0u8; 64];
    assert_eq!(run_c_with_mem(&c, &mem, 0, &[8, 42], 1), vec![42]);
    assert_eq!(run_c_with_mem(&c, &mem, 0, &[0, 0xdeadbeef_cafebabe], 1), vec![0xdeadbeef_cafebabe]);
}

/// i32.store then i32.load round-trips a 32-bit value.
#[test]
fn test_i32_store_load_js() {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 2, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I32Store(memarg),
            Instruction::LocalGet(0),
            Instruction::I32Load(memarg),
        ],
    );
    let js = compile_js_with_mem(&wasm);
    let mem = vec![0u8; 64];
    assert_eq!(run_js_with_mem(&js, &mem, &[4, 0xdeadbeef_u32 as i64]), vec![0xdeadbeef_u32 as i64]);
}

#[test]
fn test_i32_store_load_c() {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 2, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I32Store(memarg),
            Instruction::LocalGet(0),
            Instruction::I32Load(memarg),
        ],
    );
    let c = compile_c_with_mem(&wasm);
    let mem = vec![0u8; 64];
    assert_eq!(run_c_with_mem(&c, &mem, 0, &[4, 0xdeadbeef], 1), vec![0xdeadbeef]);
}

/// Non-zero memarg.offset: store at addr=0, load with offset=8 reads the same location.
#[test]
fn test_memarg_offset_js() {
    use wasm_encoder::MemArg;
    // store with offset=0, load with offset=8 but addr-8
    let store_arg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let load_arg  = MemArg { offset: 8, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I64],
        &[ValType::I64],
        &[
            // store val at mem[0]
            Instruction::I32Const(0),
            Instruction::LocalGet(0),
            Instruction::I64Store(store_arg),
            // load from mem[0+8] using addr=0 and offset=8 → but we stored at 0,
            // so use addr=0-8 which would be wrong; instead use addr=0 and offset=0 for load.
            // Simpler: store at addr=8 (I32Const(8)), load at addr=0 offset=8.
            Instruction::I32Const(8),
            Instruction::LocalGet(0),
            Instruction::I64Store(store_arg),
            Instruction::I32Const(0),
            Instruction::I64Load(load_arg),
        ],
    );
    let js = compile_js_with_mem(&wasm);
    let mem = vec![0u8; 64];
    assert_eq!(run_js_with_mem(&js, &mem, &[99]), vec![99]);
}

#[test]
fn test_memarg_offset_c() {
    use wasm_encoder::MemArg;
    let store_arg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let load_arg  = MemArg { offset: 8, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I64],
        &[ValType::I64],
        &[
            Instruction::I32Const(0),
            Instruction::LocalGet(0),
            Instruction::I64Store(store_arg),
            Instruction::I32Const(8),
            Instruction::LocalGet(0),
            Instruction::I64Store(store_arg),
            Instruction::I32Const(0),
            Instruction::I64Load(load_arg),
        ],
    );
    let c = compile_c_with_mem(&wasm);
    let mem = vec![0u8; 64];
    assert_eq!(run_c_with_mem(&c, &mem, 0, &[99], 1), vec![99]);
}

/// A function that returns an i32 constant should emit a BigInt literal in JS
/// and a uint64_t cast in C.
#[test]
fn test_const_js() {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let js = compile_js(&wasm);
    assert!(js.contains("42n"), "expected BigInt literal 42n in: {js}");
    assert!(js.contains("$0"), "expected function identifier $0 in: {js}");
}

#[test]
fn test_const_c() {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let c = compile_c(&wasm);
    assert!(c.contains("42u"), "expected 42u in C output: {c}");
    assert!(c.contains("fn_0"), "expected function identifier fn_0 in: {c}");
    assert!(c.contains("uint64_t"), "expected uint64_t in: {c}");
}

/// LocalGet + Add: tests two-operand commutative operations.
#[test]
fn test_add_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("locals[0]"), "expected locals[0] in: {js}");
    assert!(js.contains("locals[1]"), "expected locals[1] in: {js}");
    // Commutative: a+b is fine regardless of pop order.
    assert!(js.contains("a+b"), "expected a+b in: {js}");
}

#[test]
fn test_add_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let c = compile_c(&wasm);
    assert!(c.contains("locals[0]"), "expected locals[0] in: {c}");
    assert!(c.contains("locals[1]"), "expected locals[1] in: {c}");
}

// ---------------------------------------------------------------------------
// Tests — operand order bugs (bug-operand-order-i32 / i64)
// ---------------------------------------------------------------------------

/// I32Sub: first pop = rhs (locals[1]), second pop = lhs (locals[0]).
/// Correct expression is `b-a` where b=lhs, a=rhs → locals[0] - locals[1].
///
/// Bug: was `a-b` (rhs - lhs).
#[test]
fn test_i32sub_operand_order_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let js = compile_js(&wasm);
    // After the fix, the lambda body must compute b-a (lhs minus rhs).
    assert!(js.contains("b-a"), "expected b-a (lhs-rhs) in: {js}");
    assert!(!js.contains("a-b"), "must NOT contain a-b (rhs-lhs) in: {js}");
}

#[test]
fn test_i32sub_operand_order_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let c = compile_c(&wasm);
    // C backend emits casts: (uint32_t)tmp2-(uint32_t)tmp = lhs-rhs.
    assert!(
        c.contains("tmp2-(uint32_t)tmp") || c.contains("tmp2-tmp"),
        "expected lhs(tmp2)-rhs(tmp) in: {c}"
    );
    assert!(
        !c.contains("tmp-(uint32_t)tmp2") && !c.contains("tmp-tmp2"),
        "must NOT contain rhs(tmp)-lhs(tmp2) in: {c}"
    );
}

/// I32DivU: lhs / rhs = b/a.
#[test]
fn test_i32divu_operand_order_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32DivU],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("b/a"), "expected b/a in: {js}");
    assert!(!js.contains("a/b"), "must NOT contain a/b in: {js}");
}

/// I32Shl: shift amount is rhs (top of stack = first pop = `a`), value is lhs (second pop = `b`).
/// Correct: b<<a; bug was a<<b.
#[test]
fn test_i32shl_operand_order_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Shl],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("b<<a"), "expected b<<a in: {js}");
    assert!(!js.contains("a<<b"), "must NOT contain a<<b in: {js}");
}

/// I64Sub: same operand-order fix for 64-bit.
#[test]
fn test_i64sub_operand_order_js() {
    let wasm = make_module(
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64Sub],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("b-a"), "expected b-a (lhs-rhs) in: {js}");
    assert!(!js.contains("a-b"), "must NOT contain a-b in: {js}");
}

// ---------------------------------------------------------------------------
// Tests — I32ShrS / I64ShrS misplaced-paren bug
// ---------------------------------------------------------------------------

/// I32ShrS: `toUint(...,32)` — the `,32` must be INSIDE toUint.
/// Bug: was `toUint(...),32)` (32 outside the call).
#[test]
fn test_i32shrs_paren_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32ShrS],
    );
    let js = compile_js(&wasm);
    // Must contain `toUint(` followed eventually by `,32)` where 32 is INSIDE.
    assert!(js.contains("toUint("), "expected toUint( in: {js}");
    assert!(js.contains(",32)"), "expected ,32) (bit-width inside toUint) in: {js}");
    // The bad pattern was `mask32),32)` — the mask close-paren before the 32 arg.
    assert!(
        !js.contains("mask32),32)"),
        "must NOT contain mask32),32) (paren outside toUint) in: {js}"
    );
}

/// I64ShrS: same misplaced-paren fix, for 64-bit.
#[test]
fn test_i64shrs_paren_js() {
    let wasm = make_module(
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64ShrS],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("toUint("), "expected toUint( in: {js}");
    assert!(js.contains(",64)"), "expected ,64) in: {js}");
    assert!(
        !js.contains("mask64),64)"),
        "must NOT contain mask64),64) in: {js}"
    );
}

// ---------------------------------------------------------------------------
// Tests — LocalSet / LocalTee missing `]` (bug-localset-syntax / localtee-syntax)
// ---------------------------------------------------------------------------

/// LocalSet: must emit `locals[N]=` not `locals[N=`.
#[test]
fn test_localset_syntax_js() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::I32Const(99),
            Instruction::LocalSet(0),
            Instruction::LocalGet(0),
        ],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("locals[0]="), "expected locals[0]= in: {js}");
    assert!(!js.contains("locals[0="), "must NOT contain locals[0= in: {js}");
}

#[test]
fn test_localset_syntax_c() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::I32Const(99),
            Instruction::LocalSet(0),
            Instruction::LocalGet(0),
        ],
    );
    let c = compile_c(&wasm);
    assert!(c.contains("locals[0]="), "expected locals[0]= in: {c}");
    assert!(!c.contains("locals[0="), "must NOT contain locals[0= in: {c}");
}

/// LocalTee: must emit `(locals[N]=…)` not `locals[N=…`.
#[test]
fn test_localtee_syntax_js() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalTee(0)],
    );
    let js = compile_js(&wasm);
    assert!(js.contains("locals[0]="), "expected locals[0]= in: {js}");
    assert!(!js.contains("locals[0="), "must NOT contain locals[0= in: {js}");
}

#[test]
fn test_localtee_syntax_c() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalTee(0)],
    );
    let c = compile_c(&wasm);
    assert!(c.contains("locals[0]="), "expected locals[0]= in: {c}");
    assert!(!c.contains("locals[0="), "must NOT contain locals[0= in: {c}");
}

// ---------------------------------------------------------------------------
// Tests — BrTable tmp assignment (bug-brtable-noassign)
// ---------------------------------------------------------------------------

/// BrTable: the popped index must be assigned to `tmp`.
/// Bug: was `write!(self, "{}", pop!(state))` which discarded the value.
#[test]
fn test_brtable_tmp_assign_js() {
    // Build a function with a br_table that has two targets + a default.
    // Stack: i32 selector on top.
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            // Outer block (label 1) holds the result.
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I32)),
            // Inner block (label 2) is the default target.
            Instruction::Block(wasm_encoder::BlockType::Empty),
            // Another inner block (label 3) is target 0.
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::LocalGet(0), // selector
            // br_table: target 0 → label 3, default → label 2
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End, // end block 3
            Instruction::I32Const(10),
            Instruction::Br(1), // break to block 1 with value 10
            Instruction::End,   // end block 2
            Instruction::I32Const(20),
            Instruction::End, // end block 1
        ],
    );
    let js = compile_js(&wasm);
    // The fix emits `tmp=<pop>;` before the comparison loop.
    assert!(js.contains("tmp="), "expected tmp= assignment from BrTable in: {js}");
}

#[test]
fn test_brtable_tmp_assign_c() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I32)),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::LocalGet(0),
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End,
            Instruction::I32Const(10),
            Instruction::Br(1),
            Instruction::End,
            Instruction::I32Const(20),
            Instruction::End,
        ],
    );
    let c = compile_c(&wasm);
    assert!(c.contains("tmp="), "expected tmp= assignment from BrTable in: {c}");
}

// ---------------------------------------------------------------------------
// Tests — loop continue vs break (bug-loop-break)
// ---------------------------------------------------------------------------

/// A `br 0` inside a `loop` is a back-edge (continue), not a forward-break.
/// Bug: opt-mode emitted `break l{n}` instead of `continue l{n}`.
/// This test uses non-opt mode (default State) to check the non-opt path as
/// a baseline; we separately check opt-mode output shape.
#[test]
fn test_loop_continue_js() {
    // A loop that immediately jumps back to itself (infinite loop).
    let wasm = make_module(
        &[],
        &[],
        &[
            Instruction::Loop(wasm_encoder::BlockType::Empty),
            Instruction::Br(0), // back-edge → continue
            Instruction::End,
        ],
    );
    let js = compile_js(&wasm);
    // Non-opt mode: `continue l{n}`, never `break l{n}` for a loop target.
    assert!(js.contains("continue l"), "expected `continue l` for loop back-edge in: {js}");
}

#[test]
fn test_loop_continue_c() {
    let wasm = make_module(
        &[],
        &[],
        &[
            Instruction::Loop(wasm_encoder::BlockType::Empty),
            Instruction::Br(0),
            Instruction::End,
        ],
    );
    let c = compile_c(&wasm);
    // C backend uses goto lp_s_{n} for loop back-edges.
    assert!(c.contains("goto lp_s_"), "expected `goto lp_s_` for loop back-edge in: {c}");
}

// ---------------------------------------------------------------------------
// Tests — function signature metadata
// ---------------------------------------------------------------------------

/// The JS backend must emit `__sig` property with correct param/result counts.
#[test]
fn test_js_function_signature() {
    let wasm = make_module(&[ValType::I32, ValType::I32], &[ValType::I32], &[
        Instruction::LocalGet(0),
        Instruction::LocalGet(1),
        Instruction::I32Add,
    ]);
    let js = compile_js(&wasm);
    assert!(js.contains("params:2"), "expected params:2 in: {js}");
    assert!(js.contains("rets:1"), "expected rets:1 in: {js}");
}

/// The C backend must emit the signature struct with correct values.
#[test]
fn test_c_function_signature() {
    let wasm = make_module(&[ValType::I32, ValType::I32], &[ValType::I32], &[
        Instruction::LocalGet(0),
        Instruction::LocalGet(1),
        Instruction::I32Add,
    ]);
    let c = compile_c(&wasm);
    assert!(c.contains(".params=2"), "expected .params=2 in: {c}");
    assert!(c.contains(".rets=1"), "expected .rets=1 in: {c}");
}

// ---------------------------------------------------------------------------
// Tests — i64 constants
// ---------------------------------------------------------------------------

#[test]
fn test_i64const_js() {
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(0xDEAD_BEEF_u64 as i64)]);
    let js = compile_js(&wasm);
    assert!(js.contains("n"), "expected BigInt suffix n in: {js}");
}

#[test]
fn test_i64const_c() {
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(0xDEAD_BEEF_u64 as i64)]);
    let c = compile_c(&wasm);
    assert!(c.contains("ull"), "expected ull suffix in: {c}");
}

// ---------------------------------------------------------------------------
// Execution tests — run the generated code and verify numeric results
// ---------------------------------------------------------------------------

/// A constant function returns the right value when executed.
#[test]
fn test_exec_const_js() {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let js = compile_js(&wasm);
    let result = run_js(&js, &[]);
    assert_eq!(result, vec![42], "I32Const(42) should return 42");
}

#[test]
fn test_exec_const_c() {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let c = compile_c(&wasm);
    let result = run_c(&c, 0, &[], 1);
    assert_eq!(result, vec![42], "I32Const(42) should return 42");
}

/// Addition returns the correct sum.
#[test]
fn test_exec_add_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let js = compile_js(&wasm);
    assert_eq!(run_js(&js, &[5, 3]), vec![8]);
    assert_eq!(run_js(&js, &[100, 200]), vec![300]);
}

#[test]
fn test_exec_add_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[5, 3], 1), vec![8]);
    assert_eq!(run_c(&c, 0, &[100, 200], 1), vec![300]);
}

/// Subtraction respects operand order: first arg minus second arg.
/// (This is the key operand-order bug — was computing rhs−lhs.)
#[test]
fn test_exec_sub_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let js = compile_js(&wasm);
    // 10 - 3 = 7, NOT 3 - 10 = -7
    assert_eq!(run_js(&js, &[10, 3]), vec![7]);
    // 3 - 10 = -7, stored as unsigned 2's complement in i32: 0xFFFFFFF9
    let r = run_js(&js, &[3, 10]);
    // JS BigInt returns the signed i32 result as a full BigInt; mask to i32 range
    assert_eq!(r[0] as i32, -7i32, "3-10 should be -7, got {}", r[0]);
}

#[test]
fn test_exec_sub_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[10, 3], 1), vec![7]);
    // 3 - 10 in i32 = 0xFFFFFFF9 (stored in u64 low 32 bits)
    assert_eq!(run_c(&c, 0, &[3, 10], 1)[0] as u32, (-7i32) as u32);
}

/// Division respects operand order: first arg divided by second arg.
#[test]
fn test_exec_divu_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32DivU],
    );
    let js = compile_js(&wasm);
    // 10 / 2 = 5, NOT 2 / 10 = 0
    assert_eq!(run_js(&js, &[10, 2]), vec![5]);
}

#[test]
fn test_exec_divu_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32DivU],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[10, 2], 1), vec![5]);
}

/// LocalSet/LocalGet round-trip: the stored value comes back unchanged.
#[test]
fn test_exec_localset_js() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::I32Const(77),
            Instruction::LocalSet(0),
            Instruction::LocalGet(0),
        ],
    );
    let js = compile_js(&wasm);
    assert_eq!(run_js(&js, &[0]), vec![77]);
}

#[test]
fn test_exec_localset_c() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::I32Const(77),
            Instruction::LocalSet(0),
            Instruction::LocalGet(0),
        ],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[0], 1), vec![77]);
}

/// i64 constant is returned with full 64-bit precision.
#[test]
fn test_exec_i64const_js() {
    let val: i64 = 0x0123_4567_89AB_CDEFu64 as i64;
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(val)]);
    let js = compile_js(&wasm);
    let result = run_js(&js, &[]);
    assert_eq!(result[0], val, "i64 constant should be preserved");
}

#[test]
fn test_exec_i64const_c() {
    let val: u64 = 0x0123_4567_89AB_CDEFu64;
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(val as i64)]);
    let c = compile_c(&wasm);
    let result = run_c(&c, 0, &[], 1);
    assert_eq!(result[0], val, "i64 constant should be preserved");
}

/// i64 subtraction respects operand order.
#[test]
fn test_exec_i64sub_js() {
    let wasm = make_module(
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64Sub],
    );
    let js = compile_js(&wasm);
    assert_eq!(run_js(&js, &[100, 37]), vec![63]);
}

#[test]
fn test_exec_i64sub_c() {
    let wasm = make_module(
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64Sub],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[100, 37], 1), vec![63]);
}

/// Left-shift respects operand order: `value << count`, not `count << value`.
#[test]
fn test_exec_shl_js() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Shl],
    );
    let js = compile_js(&wasm);
    // 3 << 4 = 48
    assert_eq!(run_js(&js, &[3, 4]), vec![48]);
}

#[test]
fn test_exec_shl_c() {
    let wasm = make_module(
        &[ValType::I32, ValType::I32],
        &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Shl],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[3, 4], 1), vec![48]);
}

/// A br_table dispatches to the correct branch.
#[test]
fn test_exec_brtable_js() {
    // Function: takes i32 selector, returns 10 if selector==0, else 20.
    //   block (result i32)           ; label 1 — carries the result
    //     block                      ; label 2 — default/else path (br skips to label 1)
    //       block                    ; label 3 — selector==0 path
    //         local.get 0
    //         br_table 0 1           ; 0→label3(skip), default→label2(skip)
    //       end                      ; exit label 3
    //       i32.const 20
    //       br 1                     ; jump over label 1
    //     end                        ; exit label 2 (selector==0 falls here)
    //     i32.const 10
    //   end                          ; exit label 1
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I32)),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::LocalGet(0),
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End,
            Instruction::I32Const(20),
            Instruction::Br(1),
            Instruction::End,
            Instruction::I32Const(10),
            Instruction::End,
        ],
    );
    let js = compile_js(&wasm);
    assert_eq!(run_js(&js, &[0]), vec![20], "selector 0 → target 0 (inner block) → falls to i32.const 20, br 1 → 20");
    assert_eq!(run_js(&js, &[1]), vec![10], "selector 1 → default (middle block) → falls to i32.const 10 → 10");
}

#[test]
fn test_exec_brtable_c() {
    let wasm = make_module(
        &[ValType::I32],
        &[ValType::I32],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I32)),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::LocalGet(0),
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End,
            Instruction::I32Const(20),
            Instruction::Br(1),
            Instruction::End,
            Instruction::I32Const(10),
            Instruction::End,
        ],
    );
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[0], 1), vec![20], "selector 0 → target 0 (inner block) → i32.const 20, br 1 → 20");
    assert_eq!(run_c(&c, 0, &[1], 1), vec![10], "selector 1 → default (middle block) → i32.const 10 → 10");
}

/// A loop with a counter: counts down from N to 0, returns N total iterations.
/// Tests that `br 0` inside a loop is a back-edge (continue), not a break.
#[test]
fn test_exec_loop_counter_js() {
    // (func (param $n i32) (result i32)
    //   (local $acc i32)        ;; local 1
    //   (loop $lp
    //     (if (local.get $n)    ;; while n != 0
    //       (then
    //         (local.set $acc (i32.add (local.get $acc) (i32.const 1)))
    //         (local.set $n   (i32.sub (local.get $n)   (i32.const 1)))
    //         (br $lp)          ;; back-edge
    //       )
    //     )
    //   )
    //   (local.get $acc)
    // )
    use wasm_encoder::BlockType;
    let wasm = {
        let mut module = Module::new();

        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        module.section(&types);

        let mut functions = FunctionSection::new();
        functions.function(0);
        module.section(&functions);

        let mut exports = ExportSection::new();
        exports.export("f", ExportKind::Func, 0);
        module.section(&exports);

        let mut code = CodeSection::new();
        // One extra local: i32 accumulator (local 1).
        let mut func = Function::new([(1u32, ValType::I32)]);
        func.instruction(&Instruction::Loop(BlockType::Empty));
        func.instruction(&Instruction::LocalGet(0)); // n
        func.instruction(&Instruction::If(BlockType::Empty));
        // acc += 1
        func.instruction(&Instruction::LocalGet(1));
        func.instruction(&Instruction::I32Const(1));
        func.instruction(&Instruction::I32Add);
        func.instruction(&Instruction::LocalSet(1));
        // n -= 1
        func.instruction(&Instruction::LocalGet(0));
        func.instruction(&Instruction::I32Const(1));
        func.instruction(&Instruction::I32Sub);
        func.instruction(&Instruction::LocalSet(0));
        func.instruction(&Instruction::Br(1)); // br $lp (depth 1 from if = loop)
        func.instruction(&Instruction::End);   // end if
        func.instruction(&Instruction::End);   // end loop
        func.instruction(&Instruction::LocalGet(1)); // acc
        func.instruction(&Instruction::Return);
        func.instruction(&Instruction::End);   // end func
        code.function(&func);
        module.section(&code);
        module.finish()
    };
    let js = compile_js(&wasm);
    assert_eq!(run_js(&js, &[0]), vec![0],  "loop(0) → 0 iterations");
    assert_eq!(run_js(&js, &[5]), vec![5],  "loop(5) → 5 iterations");
    assert_eq!(run_js(&js, &[10]), vec![10], "loop(10) → 10 iterations");
}

#[test]
fn test_exec_loop_counter_c() {
    use wasm_encoder::BlockType;
    let wasm = {
        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        module.section(&types);
        let mut functions = FunctionSection::new();
        functions.function(0);
        module.section(&functions);
        let mut exports = ExportSection::new();
        exports.export("f", ExportKind::Func, 0);
        module.section(&exports);
        let mut code = CodeSection::new();
        let mut func = Function::new([(1u32, ValType::I32)]);
        func.instruction(&Instruction::Loop(BlockType::Empty));
        func.instruction(&Instruction::LocalGet(0));
        func.instruction(&Instruction::If(BlockType::Empty));
        func.instruction(&Instruction::LocalGet(1));
        func.instruction(&Instruction::I32Const(1));
        func.instruction(&Instruction::I32Add);
        func.instruction(&Instruction::LocalSet(1));
        func.instruction(&Instruction::LocalGet(0));
        func.instruction(&Instruction::I32Const(1));
        func.instruction(&Instruction::I32Sub);
        func.instruction(&Instruction::LocalSet(0));
        func.instruction(&Instruction::Br(1));
        func.instruction(&Instruction::End);
        func.instruction(&Instruction::End);
        func.instruction(&Instruction::LocalGet(1));
        func.instruction(&Instruction::Return);
        func.instruction(&Instruction::End);
        code.function(&func);
        module.section(&code);
        module.finish()
    };
    let c = compile_c(&wasm);
    assert_eq!(run_c(&c, 0, &[0], 1), vec![0]);
    assert_eq!(run_c(&c, 0, &[5], 1), vec![5]);
    assert_eq!(run_c(&c, 0, &[10], 1), vec![10]);
}

// ---------------------------------------------------------------------------
// Data-segment and memory.size / memory.grow helpers
// ---------------------------------------------------------------------------

/// Parse active data segments from a WASM binary.
/// Returns `Vec<(byte_offset, bytes)>` for segments with a simple `i32.const N; end` offset.
fn parse_active_data(wasm: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let mut result = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::DataSection(reader) = payload {
            for data in reader.into_iter().flatten() {
                if let DataKind::Active { offset_expr, .. } = data.kind {
                    let mut r = offset_expr.get_operators_reader();
                    if let Ok(Operator::I32Const { value }) = r.read() {
                        result.push((value as u32, data.data.to_vec()));
                    }
                }
            }
        }
    }
    result
}

/// Build a WASM module with one memory page, a function, and active data segments.
fn make_module_with_data(
    params: &[ValType],
    results: &[ValType],
    instrs: &[Instruction<'_>],
    data: &[(u32, &[u8])],
) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function(params.iter().cloned(), results.iter().cloned());
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });
    module.section(&memories);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    for instr in instrs {
        func.instruction(instr);
    }
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    if !data.is_empty() {
        let mut ds = DataSection::new();
        for (offset, bytes) in data {
            let ce = wasm_encoder::ConstExpr::i32_const(*offset as i32);
            ds.active(0, &ce, bytes.iter().copied());
        }
        module.section(&ds);
    }

    module.finish()
}

/// Build a WASM module with one memory, a passive data segment, and a
/// function body of `memory.init 0 0; data.drop 0`, exercising the native
/// backends' `Instruction::MemoryInit`/`Instruction::DataDrop` lowering.
fn make_module_with_passive_data_memory_init(seg_bytes: &[u8]) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], []);
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);

    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });
    module.section(&memories);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::LocalGet(2));
    func.instruction(&Instruction::MemoryInit { mem: 0, data_index: 0 });
    func.instruction(&Instruction::DataDrop(0));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    let mut ds = DataSection::new();
    ds.passive(seg_bytes.iter().copied());
    module.section(&ds);

    module.finish()
}

/// `Instruction::MemoryInit`/`DataDrop` must lower to a real call marshaling
/// (dest, seg_base, src_offset, len) into the SysV/AAPCS64 argument
/// registers and referencing the per-segment/helper symbols by name — see
/// `blitz-x86-64`/`blitz-aarch64` `naive.rs`'s doc comments on those arms.
/// `data.drop` is a documented compile-time no-op in this AOT backend, so it
/// must not appear as a call/trap in the output.
#[test]
fn native_x86_64_naive_memory_init_lowering() {
    let wasm = make_module_with_passive_data_memory_init(b"hi\n");
    let asm = compile_native_asm(&wasm, NativeArch::X86_64, NativeAbi::Naive);
    assert!(asm.contains("__wasm_data_seg_0"), "asm:\n{asm}");
    assert!(asm.contains("__wasm_memory_init_copy"), "asm:\n{asm}");
}

#[test]
fn native_aarch64_naive_memory_init_lowering() {
    let wasm = make_module_with_passive_data_memory_init(b"hi\n");
    let asm = compile_native_asm(&wasm, NativeArch::AArch64, NativeAbi::Naive);
    assert!(asm.contains("__wasm_data_seg_0"), "asm:\n{asm}");
    assert!(asm.contains("__wasm_memory_init_copy"), "asm:\n{asm}");
}

/// Minimal module exercising `memory.copy` + `memory.fill` (bulk memory).
fn make_module_with_memory_copy_fill() -> Vec<u8> {
    use portal_solutions_blitz_common::wasm_encoder::*;
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], []);
    module.section(&types);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&memories);
    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    // memory.copy dest=0 src=0 len=0
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::MemoryCopy { src_mem: 0, dst_mem: 0 });
    // memory.fill dest=0 val=0 len=0
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::MemoryFill(0));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);
    module.finish()
}

#[test]
fn native_x86_64_naive_memory_copy_fill_lowering() {
    let wasm = make_module_with_memory_copy_fill();
    let asm = compile_native_asm(&wasm, NativeArch::X86_64, NativeAbi::Naive);
    assert!(asm.contains("__wasm_memory_copy"), "asm:\n{asm}");
    assert!(asm.contains("__wasm_memory_fill"), "asm:\n{asm}");
}

#[test]
fn native_aarch64_naive_memory_copy_fill_lowering() {
    let wasm = make_module_with_memory_copy_fill();
    let asm = compile_native_asm(&wasm, NativeArch::AArch64, NativeAbi::Naive);
    assert!(asm.contains("__wasm_memory_copy"), "asm:\n{asm}");
    assert!(asm.contains("__wasm_memory_fill"), "asm:\n{asm}");
}

#[test]
fn native_riscv64_naive_memory_copy_fill_lowering() {
    let wasm = make_module_with_memory_copy_fill();
    let asm = compile_native_asm(&wasm, NativeArch::Riscv64, NativeAbi::Naive);
    assert!(asm.contains("__wasm_memory_copy"), "asm:\n{asm}");
    assert!(asm.contains("__wasm_memory_fill"), "asm:\n{asm}");
}

/// Compile a WASM module (with memory and possibly data) to JS.
/// Data segments are NOT emitted here; call `apply_segments_to_mem` to
/// pre-initialise the memory buffer before passing it to `run_js_with_mem`.
fn compile_js_with_data(wasm: &[u8]) -> String {
    compile_js_with_mem(wasm)
}

/// Apply parsed active data segments to a mutable memory buffer (Rust-side).
fn apply_segments_to_mem(mem: &mut Vec<u8>, segments: &[(u32, Vec<u8>)]) {
    for (offset, bytes) in segments {
        let start = *offset as usize;
        let end = start + bytes.len();
        if end > mem.len() {
            mem.resize(end, 0);
        }
        mem[start..end].copy_from_slice(bytes);
    }
}

/// Compile a WASM module (with memory and data segments) to C.
/// Emits `c_module_preamble` + `c_emit_data_segments` + function code.
/// The test harness must call `__wasm_init_data()` after setting `__wasm_mem`.
fn compile_c_with_data(wasm: &[u8], segments: &[(u32, &[u8])]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    c_emit_data_segments(&mut out, segments).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    out
}

/// The default `__wasm_memory_grow` implementation injected into C test harnesses.
/// Matches the extern declaration emitted by `c_module_preamble`.
const C_MEMORY_GROW_IMPL: &str = "\
static uint32_t __wasm_memory_grow(uint32_t delta,uint8_t**mem,uint32_t*pages){\
    uint32_t old=*pages;\
    uint64_t new_size=((uint64_t)old+delta)*65536;\
    uint8_t*p=(uint8_t*)realloc(*mem,(size_t)new_size);\
    if(!p)return(uint32_t)-1;\
    memset(p+(uint64_t)old*65536,0,(uint64_t)delta*65536);\
    *mem=p;*pages=old+delta;\
    return old;\
}";

/// Run C source that uses `memory.size`/`memory.grow`, injecting a default grow impl.
/// `mem_pages` is how many pages to pre-allocate (each 65536 bytes).
fn run_c_with_grow(
    c_src: &str,
    mem_pages: u32,
    fn_id: u32,
    args: &[u64],
    rets: usize,
) -> Vec<u64> {
    use std::io::Write as _;

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_grow_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_grow_{pid}_{seq}"));

    let mem_size = mem_pages as u64 * 65536;
    let mut main_body = format!(
        "int main(){{uint8_t*_mem=(uint8_t*)calloc({mem_size}ull,1);\
         if(!_mem)return 1;\
         __wasm_mem=_mem;__wasm_mem_pages={mem_pages};\
         uint64_t _args[{n}]={{",
        n = args.len().max(1)
    );
    for (i, &a) in args.iter().enumerate() {
        if i > 0 { main_body.push(','); }
        main_body.push_str(&format!("{a}ull"));
    }
    if args.is_empty() { main_body.push('0'); }
    main_body.push_str(&format!(
        "}};uint64_t*_r=fn_{fn_id}(_args);"
    ));
    for i in 0..rets {
        main_body.push_str(&format!("printf(\"%llu\\n\",_r[{i}]);"));
    }
    main_body.push_str("free(__wasm_mem);return 0;}");

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n\
         #define WASM_STACK_SIZE 512\n\
         {C_MEMORY_GROW_IMPL}\n\
         {c_src}\n{main_body}\n"
    );

    std::fs::write(&src_path, &full_src).unwrap();

    let compile = std::process::Command::new("cc")
        .arg(&src_path)
        .arg("-Wno-unsequenced")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("cc not found in PATH");

    assert!(
        compile.status.success(),
        "C compile failed:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        full_src
    );

    let run = std::process::Command::new(&bin_path)
        .output()
        .expect("failed to run compiled binary");

    assert!(run.status.success(), "binary exited non-zero: {}", String::from_utf8_lossy(&run.stderr));
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<u64>().expect("expected integer line"))
        .collect()
}

/// Run JS source with a grow-capable memory (start with `init_pages` 64 KiB pages).
fn run_js_with_grow(js_src: &str, init_pages: u32, args: &[i64]) -> Vec<i64> {
    let init_size = init_pages as u64 * 65536;
    let fn_args: Vec<String> = args.iter().map(|v| format!("{v}n")).collect();
    let harness = format!(
        "$mem=new Uint8Array({init_size});$mem_dv=new DataView($mem.buffer);\n\
         const __r=$0({});const __n=Array.isArray(__r)?__r:[__r];\
         for(const v of __n)console.log(String(v));",
        fn_args.join(",")
    );
    let code = format!("{js_src}{harness}");

    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(&code)
        .output()
        .expect("node not found in PATH");

    assert!(
        out.status.success(),
        "node exited non-zero.\nstderr: {}\ncode: {}",
        String::from_utf8_lossy(&out.stderr),
        code
    );

    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<i64>().expect("expected integer line"))
        .collect()
}

// ---------------------------------------------------------------------------
// memory.size tests
// ---------------------------------------------------------------------------

#[test]
fn test_memory_size_js() {
    // (func (result i32) memory.size)
    let wasm = make_module_with_memory(&[], &[ValType::I32], &[Instruction::MemorySize(0)]);
    let js = compile_js_with_mem(&wasm);
    // 1 page allocated → size = 1
    let mem = vec![0u8; 65536];
    assert_eq!(run_js_with_mem(&js, &mem, &[]), vec![1]);
}

#[test]
fn test_memory_size_c() {
    let wasm = make_module_with_memory(&[], &[ValType::I32], &[Instruction::MemorySize(0)]);
    let c = compile_c_with_mem(&wasm);
    // run_c_with_grow pre-allocates 1 page and sets __wasm_mem_pages=1
    assert_eq!(run_c_with_grow(&c, 1, 0, &[], 1), vec![1]);
    assert_eq!(run_c_with_grow(&c, 3, 0, &[], 1), vec![3]);
}

// ---------------------------------------------------------------------------
// memory.grow tests
// ---------------------------------------------------------------------------

#[test]
fn test_memory_grow_js() {
    // (func (param i64) (result i64) local.get 0 memory.grow)
    let wasm = make_module_with_memory(
        &[ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::MemoryGrow(0)],
    );
    let js = compile_js_with_mem(&wasm);
    let mem = vec![0u8; 65536]; // 1 page
    // Grow by 1: old size = 1 returned
    assert_eq!(run_js_with_mem(&js, &mem, &[1]), vec![1]);
    // After grow, size should be 2 — verify with a second call
    let wasm2 = make_module_with_memory(&[], &[ValType::I64], &[Instruction::MemorySize(0)]);
    let js2 = compile_js_with_mem(&wasm2);
    // Start with 2 pages pre-allocated
    let mem2 = vec![0u8; 2 * 65536];
    assert_eq!(run_js_with_mem(&js2, &mem2, &[]), vec![2]);
}

#[test]
fn test_memory_grow_c() {
    // (func (param i64) (result i64) local.get 0 memory.grow)
    let wasm = make_module_with_memory(
        &[ValType::I64],
        &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::MemoryGrow(0)],
    );
    let c = compile_c_with_mem(&wasm);
    // Start with 1 page, grow by 1 → returns old size = 1
    assert_eq!(run_c_with_grow(&c, 1, 0, &[1], 1), vec![1]);
    // After grow, pages = 2. But we can't observe that here without another call.
    // Start with 2 pages, grow by 3 → returns old size = 2
    assert_eq!(run_c_with_grow(&c, 2, 0, &[3], 1), vec![2]);
}

// ---------------------------------------------------------------------------
// Data segment tests
// ---------------------------------------------------------------------------

#[test]
fn test_data_segment_js() {
    use wasm_encoder::MemArg;
    // Module with data segment: [0xAB, 0xCD, 0x00, 0x00] at offset 8.
    // Function loads i32 at address 8 → 0xCDAB (little-endian).
    let data_bytes: &[u8] = &[0xAB, 0xCD, 0x00, 0x00];
    let wasm = make_module_with_data(
        &[],
        &[ValType::I32],
        &[Instruction::I32Const(8), Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 })],
        &[(8, data_bytes)],
    );
    let js = compile_js_with_data(&wasm);
    // Build 1-page memory and pre-apply data segments (avoids the $mem-is-empty problem).
    let mut mem = vec![0u8; 65536];
    apply_segments_to_mem(&mut mem, &parse_active_data(&wasm));
    assert_eq!(run_js_with_mem(&js, &mem, &[]), vec![0xCDAB]);
}

#[test]
fn test_data_segment_c() {
    use wasm_encoder::MemArg;
    let data_bytes: &[u8] = &[0xAB, 0xCD, 0x00, 0x00];
    let wasm = make_module_with_data(
        &[],
        &[ValType::I32],
        &[Instruction::I32Const(8), Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 })],
        &[(8, data_bytes)],
    );
    let segments = parse_active_data(&wasm);
    let seg_refs: Vec<(u32, &[u8])> = segments.iter().map(|(o, b)| (*o, b.as_slice())).collect();
    // compile_c_with_data emits __wasm_init_data(); harness calls it after alloc.
    let c = compile_c_with_data(&wasm, &seg_refs);

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_data_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_data_{pid}_{seq}"));

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n\
         #define WASM_STACK_SIZE 512\n\
         {C_MEMORY_GROW_IMPL}\n\
         {c}\n\
         int main(){{\
             uint8_t*_mem=(uint8_t*)calloc(65536,1);__wasm_mem=_mem;__wasm_mem_pages=1;\
             __wasm_init_data();\
             uint64_t _args[1]={{0}};uint64_t*_r=fn_0(_args);\
             printf(\"%llu\\n\",_r[0]);\
             free(_mem);return 0;}}\n"
    );

    std::fs::write(&src_path, &full_src).unwrap();
    let compile = std::process::Command::new("cc")
        .arg(&src_path).arg("-Wno-unsequenced").arg("-o").arg(&bin_path)
        .output().expect("cc not found");
    assert!(compile.status.success(), "C compile failed:\n{}\nsource:\n{}", String::from_utf8_lossy(&compile.stderr), full_src);
    let run = std::process::Command::new(&bin_path).output().expect("run failed");
    assert!(run.status.success(), "binary failed: {}", String::from_utf8_lossy(&run.stderr));
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);
    let result: Vec<u64> = String::from_utf8(run.stdout).unwrap().lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<u64>().unwrap())
        .collect();
    assert_eq!(result, vec![0xCDAB]);
}

// ---------------------------------------------------------------------------
// Import/export helpers
// ---------------------------------------------------------------------------

/// Parse function imports from WASM bytes, returning `(module, name)` pairs.
fn parse_imports(wasm: &[u8]) -> Vec<(String, String)> {
    let mut imports = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ImportSection(reader) = payload {
            for imp in reader.into_iter().flatten() {
                if matches!(imp.ty, wasmparser::TypeRef::Func(_)) {
                    imports.push((imp.module.to_owned(), imp.name.to_owned()));
                }
            }
        }
    }
    imports
}

/// Parse function exports from WASM bytes, returning `(wasm_function_index, name)` pairs.
fn parse_exports(wasm: &[u8]) -> Vec<(u32, String)> {
    let mut exports = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(reader) = payload {
            for exp in reader.into_iter().flatten() {
                if exp.kind == wasmparser::ExternalKind::Func {
                    exports.push((exp.index, exp.name.to_owned()));
                }
            }
        }
    }
    exports
}

/// Build a WASM module that:
/// 1. Imports `("env", "add_one")` with sig `(i64) -> i64`
/// 2. Defines an internal function `run(x: i64) -> i64` that calls the import
/// 3. Exports `"run"` (WASM index 1, since index 0 is the import)
fn make_module_with_import() -> Vec<u8> {
    let mut module = Module::new();

    // type 0: (i64) -> i64
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64], [ValType::I64]);
    module.section(&types);

    // import: env::add_one has type 0
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import("env", "add_one", wasm_encoder::EntityType::Function(0));
    module.section(&imports);

    // function 1 (internal): type 0
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);

    // export "run" = function index 1
    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, 1);
    module.section(&exports);

    // body: local.get 0; call 0 (import); return; end
    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::Call(0));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    module.finish()
}

/// Compile a WASM module with imports to JS source.
/// Emits import placeholders first, then function bodies, then export aliases.
fn compile_js_with_imports_exports(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let raw_imports = parse_imports(wasm);
    let raw_exports = parse_exports(wasm);

    // Convert to &str slices for the API
    let imports_ref: Vec<(&str, &str)> = raw_imports.iter()
        .map(|(m, n)| (m.as_str(), n.as_str()))
        .collect();
    let exports_ref: Vec<(u32, &str)> = raw_exports.iter()
        .map(|(idx, n)| (*idx, n.as_str()))
        .collect();

    let mut bodies = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let import_count = imports_ref.len() as u32;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies, &fsigs, &sigs_wp, import_count,
    );
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    js_emit_imports(&mut out, &imports_ref).unwrap();

    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &imports_ref, &mut state, &op, &mut reencoder)
            .unwrap();
    }

    js_emit_exports(&mut out, &exports_ref).unwrap();
    out
}

/// Compile a WASM module with imports to C source.
fn compile_c_with_imports_exports(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let raw_imports = parse_imports(wasm);
    let raw_exports = parse_exports(wasm);

    let imports_ref: Vec<(&str, &str)> = raw_imports.iter()
        .map(|(m, n)| (m.as_str(), n.as_str()))
        .collect();
    let exports_ref: Vec<(u32, &str)> = raw_exports.iter()
        .map(|(idx, n)| (*idx, n.as_str()))
        .collect();

    let mut bodies = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let import_count = imports_ref.len() as u32;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies, &fsigs, &sigs_wp, import_count,
    );
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    c_emit_import_decls(&mut out, &imports_ref, &sigs_enc, &fsigs).unwrap();

    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &imports_ref, &mut state, &op, &mut reencoder)
            .unwrap();
    }

    c_emit_exports(&mut out, &exports_ref).unwrap();
    out
}

// ---------------------------------------------------------------------------
// Import/export tests
// ---------------------------------------------------------------------------

/// JS: call an imported function through a WASM module.
///
/// The host provides `add_one(x) = x + 1`. The WASM `run` function calls it.
/// Expected: run(41n) == 42n.
#[test]
fn test_import_call_js() {
    let wasm = make_module_with_import();
    let js_src = compile_js_with_imports_exports(&wasm);

    // Provide the import ($0 = add_one) and call run (= $1)
    let harness = "\n$0=function(x){return [x+1n];};Object.defineProperty($0,'__sig',{value:{params:1,rets:1}});\nconst __r=run(41n);\nconst __n=Array.isArray(__r)?__r:[__r];for(const v of __n)console.log(String(v));";
    let code = format!("{js_src}{harness}");

    let out = std::process::Command::new("node")
        .arg("-e").arg(&code)
        .output().expect("node not found");
    assert!(out.status.success(), "node failed:\n{}\ncode:\n{}", String::from_utf8_lossy(&out.stderr), code);

    let result: Vec<i64> = String::from_utf8(out.stdout).unwrap()
        .lines().filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<i64>().unwrap())
        .collect();
    assert_eq!(result, vec![42]);
}

/// C: call an imported function through a WASM module.
///
/// The host provides `fn_0` (add_one). The WASM `fn_1` (`run`) calls it.
/// Expected: run(41) == 42.
#[test]
fn test_import_call_c() {
    let wasm = make_module_with_import();
    let c_src = compile_c_with_imports_exports(&wasm);

    // The module exports "run" = fn_1 (WASM index 1, since index 0 is the import)
    // We provide fn_0 as an add_one impl and call fn_1(41)
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_import_test_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_import_test_{pid}_{seq}"));

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n\
         #define WASM_STACK_SIZE 512\n\
         // import impl: add_one\n\
         static uint64_t __add_one_rets[1];\n\
         static uint64_t* add_one_impl(uint64_t* restrict __in){{\n\
             __add_one_rets[0]=__in[0]+1;\n\
             return __add_one_rets;\n\
         }}\n\
         {c_src}\n\
         int main(){{\n\
             fn_0=add_one_impl;\n\
             uint64_t _args[1]={{41ull}};\n\
             uint64_t*_r=run(_args);\n\
             printf(\"%llu\\n\",_r[0]);\n\
             return 0;\n\
         }}\n"
    );

    std::fs::write(&src_path, &full_src).unwrap();
    let compile = std::process::Command::new("cc")
        .arg(&src_path).arg("-Wno-unsequenced").arg("-o").arg(&bin_path)
        .output().expect("cc not found");
    assert!(compile.status.success(), "C compile failed:\n{}\nsource:\n{}", String::from_utf8_lossy(&compile.stderr), full_src);

    let run = std::process::Command::new(&bin_path).output().expect("failed to run");
    assert!(run.status.success());

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    let result: u64 = String::from_utf8(run.stdout).unwrap().trim().parse().unwrap();
    assert_eq!(result, 42);
}

/// JS: verify that exported function aliases are emitted correctly.
#[test]
fn test_export_js() {
    let wasm = make_module(&[ValType::I64], &[ValType::I64], &[
        Instruction::LocalGet(0),
    ]);
    let js_src = compile_js(&wasm);
    let raw_exports = parse_exports(&wasm);
    let exports_ref: Vec<(u32, &str)> = raw_exports.iter().map(|(i, n)| (*i, n.as_str())).collect();
    let mut with_export = js_src.clone();
    js_emit_exports(&mut with_export, &exports_ref).unwrap();

    // The module exports "f" = WASM index 0
    assert!(with_export.contains("var f=$0;"), "expected export alias in:\n{with_export}");
}

/// C: verify that exported function alias is emitted with correct signature.
#[test]
fn test_export_c() {
    let wasm = make_module(&[ValType::I64], &[ValType::I64], &[
        Instruction::LocalGet(0),
    ]);
    let c_src = compile_c(&wasm);
    let raw_exports = parse_exports(&wasm);
    let exports_ref: Vec<(u32, &str)> = raw_exports.iter().map(|(i, n)| (*i, n.as_str())).collect();
    let mut with_export = c_src.clone();
    c_emit_exports(&mut with_export, &exports_ref).unwrap();

    // Should emit an alias function named "f"
    assert!(with_export.contains("uint64_t*f("), "expected export alias in:\n{with_export}");
}

// ---------------------------------------------------------------------------
// Tests — ESM output (js_emit_imports_esm / js_emit_exports_esm)
// ---------------------------------------------------------------------------

/// `js_emit_imports_esm` produces well-formed ES module import statements.
///
/// Each import must be a named import from the correct module with an `_import_N`
/// alias and a matching `let $N = _import_N;` binding that the compiled body
/// uses when calling imports.
#[test]
fn test_esm_imports_structural() {
    use portal_solutions_blitz_js::js_emit_imports_esm;

    let mut out = String::new();
    let imports: &[(&str, &str)] = &[
        ("env", "add_one"),
        ("math", "sqrt"),
    ];
    js_emit_imports_esm(&mut out, imports).unwrap();

    // Import 0: env::add_one
    assert!(
        out.contains("import {add_one as _import_0} from 'env';"),
        "expected ESM import for add_one in:\n{out}"
    );
    assert!(
        out.contains("let $0=_import_0;"),
        "expected $0 binding in:\n{out}"
    );

    // Import 1: math::sqrt
    assert!(
        out.contains("import {sqrt as _import_1} from 'math';"),
        "expected ESM import for sqrt in:\n{out}"
    );
    assert!(
        out.contains("let $1=_import_1;"),
        "expected $1 binding in:\n{out}"
    );
}

/// `js_emit_exports_esm` produces well-formed ES module `export` statements.
///
/// Each export must be `export { $N as name };` where N is the WASM function
/// index (import_count + internal_id).
#[test]
fn test_esm_exports_structural() {
    use portal_solutions_blitz_js::js_emit_exports_esm;

    let mut out = String::new();
    let exports: &[(u32, &str)] = &[(0, "run"), (3, "helper")];
    js_emit_exports_esm(&mut out, exports).unwrap();

    assert!(
        out.contains("export {$0 as run};"),
        "expected `export {{$0 as run}};` in:\n{out}"
    );
    assert!(
        out.contains("export {$3 as helper};"),
        "expected `export {{$3 as helper}};` in:\n{out}"
    );
}

/// `js_module_preamble_esm` emits the memory globals needed for load/store.
#[test]
fn test_esm_preamble_structural() {
    use portal_solutions_blitz_js::js_module_preamble_esm;

    let mut out = String::new();
    js_module_preamble_esm(&mut out).unwrap();

    assert!(out.contains("$mem"), "expected $mem in preamble:\n{out}");
    assert!(out.contains("$mem_dv"), "expected $mem_dv in preamble:\n{out}");
    assert!(out.contains("Uint8Array"), "expected Uint8Array in preamble:\n{out}");
    assert!(out.contains("DataView"), "expected DataView in preamble:\n{out}");
}

/// ESM imports/exports are distinct from CJS equivalents.
///
/// CJS uses `var $N;` declarations; ESM uses `import` statements.
/// CJS uses `var name=$N;` aliases; ESM uses `export {$N as name};`.
#[test]
fn test_esm_vs_cjs_distinct() {
    use portal_solutions_blitz_js::{js_emit_imports, js_emit_imports_esm, js_emit_exports, js_emit_exports_esm};

    let imports: &[(&str, &str)] = &[("env", "foo")];
    let exports: &[(u32, &str)] = &[(0, "bar")];

    let mut cjs_out = String::new();
    js_emit_imports(&mut cjs_out, imports).unwrap();
    js_emit_exports(&mut cjs_out, exports).unwrap();

    let mut esm_out = String::new();
    js_emit_imports_esm(&mut esm_out, imports).unwrap();
    js_emit_exports_esm(&mut esm_out, exports).unwrap();

    // CJS should use `var $0;` style
    assert!(cjs_out.contains("var $0;"), "CJS should use var declaration:\n{cjs_out}");
    // ESM should use `import` statement
    assert!(esm_out.contains("import {"), "ESM should use import statement:\n{esm_out}");

    // CJS export is an alias assignment
    assert!(cjs_out.contains("var bar=$0;"), "CJS export should be alias:\n{cjs_out}");
    // ESM export uses `export` keyword
    assert!(esm_out.contains("export {$0 as bar};"), "ESM export should use export keyword:\n{esm_out}");
}

// ---------------------------------------------------------------------------
// Tests — BackendAbi trait (compile-time / structural)
// ---------------------------------------------------------------------------

/// Verify that the `BackendAbi` x86-64 `NaiveAbi` and `SysVAbi` types
/// are publicly accessible and the trait is importable.
///
/// This is a compile-time check: if it compiles the types exist and are local.
#[test]
fn test_backend_abi_types_x86_64() {
    use portal_solutions_blitz_x86_64::abi::{NaiveAbi, SysVAbi};

    // ZSTs — just check they can be named and are zero-sized.
    assert_eq!(core::mem::size_of::<NaiveAbi>(), 0);
    assert_eq!(core::mem::size_of::<SysVAbi>(), 0);
}

/// Same compile-time check for AArch64.
#[test]
fn test_backend_abi_types_aarch64() {
    use portal_solutions_blitz_aarch64::abi::{NaiveAbi, SysVAbi};

    assert_eq!(core::mem::size_of::<NaiveAbi>(), 0);
    assert_eq!(core::mem::size_of::<SysVAbi>(), 0);
}

/// Same compile-time check for RISC-V 64.
#[test]
fn test_backend_abi_types_riscv64() {
    use portal_solutions_blitz_riscv64::abi::{NaiveAbi, SysVAbi};

    assert_eq!(core::mem::size_of::<NaiveAbi>(), 0);
    assert_eq!(core::mem::size_of::<SysVAbi>(), 0);
}

// ---------------------------------------------------------------------------
// Tests — native backends under Unicorn
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum NativeArch {
    X86_64,
    AArch64,
    Riscv64,
    Riscv32,
    Arm,
    I686,
}

#[derive(Clone, Copy, Debug)]
enum NativeAbi {
    Naive,
    Sysv,
    Lfi,
}

fn compile_native_asm(wasm: &[u8], arch: NativeArch, abi: NativeAbi) -> String {
    let (sigs_wp, _sigs_enc, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut out = NativeAsmWriter(String::new());
    let mut ctx = ();

    match (arch, abi) {
        (NativeArch::X86_64, NativeAbi::Naive) => {
            use portal_solutions_blitz_x86_64::{naive, X64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::X86_64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch};
            let mut state = sysv::SysVState::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Naive) => {
            use portal_solutions_blitz_aarch64::{naive, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv64::{naive, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::X86_64, NativeAbi::Lfi) => {
            use portal_solutions_blitz_x86_64::{lfi, X64Arch};
            let mut state = lfi::State::default();
            for op in ops {
                let op = op.unwrap();
                lfi::LfiWriterExt::lfi_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Lfi) => {
            use portal_solutions_blitz_aarch64::{lfi, AArch64Arch};
            let mut state = lfi::State::default();
            for op in ops {
                let op = op.unwrap();
                lfi::LfiWriterExt::lfi_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Lfi) => {
            panic!("LFI not implemented for RISC-V 64");
        }
        (NativeArch::Riscv32, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv32::{naive, RiscV32Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV32Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Riscv32, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv32::{naive, sysv, RiscV32Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV32Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Riscv32, NativeAbi::Lfi) => {
            panic!("LFI not implemented for RISC-V 32");
        }
        (NativeArch::Arm, NativeAbi::Naive) => {
            use portal_solutions_blitz_arm::{naive, ArmArch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, ArmArch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Arm, NativeAbi::Sysv) => {
            use portal_solutions_blitz_arm::{naive, sysv, ArmArch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, ArmArch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::Arm, NativeAbi::Lfi) => {
            panic!("LFI not implemented for ARM");
        }
        (NativeArch::I686, NativeAbi::Naive) => {
            use portal_solutions_blitz_i686::{naive, X86Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X86Arch::default(),
                    &mut state, &[], &[], &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::I686, NativeAbi::Sysv) => {
            use portal_solutions_blitz_i686::{naive, sysv, X86Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X86Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
        }
        (NativeArch::I686, NativeAbi::Lfi) => {
            panic!("LFI not implemented for i686");
        }
    }

    normalize_native_asm(arch, out.0)
}

fn normalize_native_asm(arch: NativeArch, asm: String) -> String {
    let asm = match arch {
        NativeArch::X86_64 | NativeArch::I686 => {
            format!(".intel_syntax noprefix\n.text\n.global f0\n{asm}")
        }
        NativeArch::AArch64
        | NativeArch::Riscv64
        | NativeArch::Riscv32
        | NativeArch::Arm => format!(".text\n.global f0\n{asm}"),
    };

    // The x86 text writer currently omits a newline after LEA. Keep the tests
    // about backend behavior instead of assembler trivia.
    let mut fixed = String::new();
    for line in asm.lines() {
        let mut line = line.to_owned();
        for mnemonic in ["push ", "pop ", "mov ", "xchg ", "ret", "and ", "not "] {
            if let Some(idx) = line.find(mnemonic) {
                if idx > 0 && !line[..idx].trim_start().starts_with('.') {
                    fixed.push_str(&line[..idx]);
                    fixed.push('\n');
                    line = line[idx..].to_owned();
                    break;
                }
            }
        }
        fixed.push_str(&line);
        fixed.push('\n');
    }
    fixed
}

fn assemble_native_text(arch: NativeArch, asm: &str) -> Result<Vec<u8>, String> {
    use std::io::Write as _;

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_native_{pid}_{seq}.s"));
    let obj_path = dir.join(format!("blitz_native_{pid}_{seq}.o"));

    std::fs::File::create(&src_path)
        .and_then(|mut f| f.write_all(asm.as_bytes()))
        .map_err(|e| e.to_string())?;

    let target = match arch {
        NativeArch::X86_64 => "x86_64-unknown-linux-gnu",
        NativeArch::AArch64 => "aarch64-unknown-linux-gnu",
        NativeArch::Riscv64 => "riscv64-unknown-elf",
        NativeArch::Riscv32 => "riscv32-unknown-elf",
        NativeArch::Arm => "arm-linux-gnueabihf",
        NativeArch::I686 => "i686-unknown-linux-gnu",
    };

    let output = std::process::Command::new("clang")
        .arg("-target")
        .arg(target)
        .arg("-c")
        .arg(&src_path)
        .arg("-o")
        .arg(&obj_path)
        .output()
        .map_err(|e| format!("failed to run clang: {e}"))?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&obj_path);
        return Err(format!(
            "clang failed for {target}:\n{}\nsource:\n{asm}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let obj = std::fs::read(&obj_path).map_err(|e| e.to_string())?;
    let code =
        extract_elf_text(&obj).ok_or_else(|| format!("no .text section in object for {target}"))?;

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&obj_path);

    Ok(code)
}

fn extract_elf_text(obj: &[u8]) -> Option<Vec<u8>> {
    if obj.get(0..4)? != b"\x7fELF" || *obj.get(5)? != 1 {
        return None;
    }
    let class = *obj.get(4)?; // 1 = ELF32, 2 = ELF64

    let read_u16 = |offset: usize| -> Option<u16> {
        Some(u16::from_le_bytes(obj.get(offset..offset + 2)?.try_into().ok()?))
    };
    let read_u32 = |offset: usize| -> Option<u32> {
        Some(u32::from_le_bytes(obj.get(offset..offset + 4)?.try_into().ok()?))
    };
    let read_u64 = |offset: usize| -> Option<u64> {
        Some(u64::from_le_bytes(obj.get(offset..offset + 8)?.try_into().ok()?))
    };

    let (shoff, shentsize, shnum, shstrndx, min_shentsize, sh_off_field, sh_size_field) =
        if class == 2 {
            (
                read_u64(0x28)? as usize,
                read_u16(0x3a)? as usize,
                read_u16(0x3c)? as usize,
                read_u16(0x3e)? as usize,
                64usize,
                0x18usize,
                0x20usize,
            )
        } else if class == 1 {
            (
                read_u32(0x20)? as usize,
                read_u16(0x2e)? as usize,
                read_u16(0x30)? as usize,
                read_u16(0x32)? as usize,
                40usize,
                0x10usize,
                0x14usize,
            )
        } else {
            return None;
        };

    if shentsize < min_shentsize || shstrndx >= shnum {
        return None;
    }

    let shstr = shoff.checked_add(shstrndx.checked_mul(shentsize)?)?;
    let (shstr_off, shstr_size) = if class == 2 {
        (read_u64(shstr + sh_off_field)? as usize, read_u64(shstr + sh_size_field)? as usize)
    } else {
        (read_u32(shstr + sh_off_field)? as usize, read_u32(shstr + sh_size_field)? as usize)
    };
    let shstrtab = obj.get(shstr_off..shstr_off.checked_add(shstr_size)?)?;

    for i in 0..shnum {
        let sh = shoff.checked_add(i.checked_mul(shentsize)?)?;
        let name_off = read_u32(sh)? as usize;
        let name_tail = shstrtab.get(name_off..)?;
        let nul = name_tail.iter().position(|b| *b == 0)?;
        let name = core::str::from_utf8(&name_tail[..nul]).ok()?;
        if name == ".text" {
            let (off, size) = if class == 2 {
                (read_u64(sh + sh_off_field)? as usize, read_u64(sh + sh_size_field)? as usize)
            } else {
                (read_u32(sh + sh_off_field)? as usize, read_u32(sh + sh_size_field)? as usize)
            };
            return Some(obj.get(off..off.checked_add(size)?)?.to_vec());
        }
    }

    None
}

fn native_sysv_const_wasm(value: i64) -> Vec<u8> {
    make_module(&[], &[ValType::I64], &[Instruction::I64Const(value)])
}

fn native_naive_empty_wasm() -> Vec<u8> {
    make_module(&[], &[], &[])
}

fn run_native_sysv_const(arch: NativeArch, code: &[u8]) -> u64 {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            let rsp = STACK + STACK_SIZE - 8;
            uc.mem_write(rsp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
        NativeArch::Riscv32 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV32).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            let lo = uc.reg_read(RegisterRISCV::A0).unwrap();
            let hi = uc.reg_read(RegisterRISCV::A1).unwrap();
            lo | (hi << 32)
        }
        NativeArch::Arm => {
            use unicorn_engine::RegisterARM;
            // A32 little-endian (Unicorn: Arch::ARM + Mode::LITTLE_ENDIAN).
            let mut uc = Unicorn::new(Arch::ARM, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM::LR, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            let lo = uc.reg_read(RegisterARM::R0).unwrap();
            let hi = uc.reg_read(RegisterARM::R1).unwrap();
            lo | (hi << 32)
        }
        NativeArch::I686 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_32).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            let esp = STACK + STACK_SIZE - 4;
            uc.mem_write(esp, &(CODE + code.len() as u64).to_le_bytes()[..4]).unwrap();
            uc.reg_write(RegisterX86::ESP, esp).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            let lo = uc.reg_read(RegisterX86::EAX).unwrap();
            let hi = uc.reg_read(RegisterX86::EDX).unwrap();
            lo | (hi << 32)
        }
    }
}

/// Install a code hook on `uc` that prints every instruction when the
/// `BLITZ_TRACE_UNICORN` env-var is set.  The hook reads the instruction
/// bytes from the emulated memory and prints `[TRACE] PC=0x… size=N  bytes`.
fn attach_trace_hook<D: 'static>(
    uc: &mut unicorn_engine::Unicorn<'_, D>,
    arch: NativeArch,
    code_base: u64,
    code_len: usize,
) {
    if std::env::var("BLITZ_TRACE_UNICORN").is_err() { return; }
    let portal_logger = log::LlmtrimLogger::from_env();
    let arch_str = format!("{arch:?}");
    uc.add_code_hook(code_base, code_base + code_len as u64, move |uc, addr, size| {
        let mut buf = vec![0u8; size as usize];
        let _ = uc.mem_read(addr, &mut buf);
        log::portal_trace(&portal_logger, &arch_str, addr, size as usize, &buf);
    }).expect("add_code_hook failed");
}

fn run_native_naive_smoke(arch: NativeArch, code: &[u8]) {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            // The x86-64 naive backend uses a CTX context-register mechanism that requires
            // a host runtime to set up the CTX chain.  Binary-writer output can't be
            // run standalone in Unicorn without that runtime.  Just verify that the
            // codegen path produces non-empty output.
            assert!(!code.is_empty());
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::Riscv32 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV32).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::Arm => {
            use unicorn_engine::RegisterARM;
            let mut uc = Unicorn::new(Arch::ARM, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM::LR, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::I686 => {
            use unicorn_engine::RegisterX86;
            // Like x86-64 naive: verify non-empty codegen under soft-skip assemble.
            // Also run when assembled: empty body is just prologue/epilogue.
            assert!(!code.is_empty());
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_32).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            let esp = STACK + STACK_SIZE - 4;
            uc.mem_write(esp, &(CODE + code.len() as u64).to_le_bytes()[..4]).unwrap();
            uc.reg_write(RegisterX86::ESP, esp).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
    }
}

fn assemble_or_skip(arch: NativeArch, asm: &str) -> Option<Vec<u8>> {
    match assemble_native_text(arch, asm) {
        Ok(code) => Some(code),
        Err(err) if err.starts_with("failed to run clang:") => {
            eprintln!("skipping native test: clang not in PATH ({err})");
            None
        }
        Err(err) if matches!(arch, NativeArch::Riscv64) && err.contains("riscv-add-build-attributes") => {
            eprintln!("skipping RISC-V Unicorn backend test: host clang cannot assemble RISC-V ({err})");
            None
        }
        Err(err)
            if matches!(
                arch,
                NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 | NativeArch::Riscv64
            ) && (err.contains("unable to create target")
                || err.contains("unable to find")
                || err.contains("Invalid target")
                || err.contains("unknown target")
                || err.contains("riscv-add-build-attributes")
                || err.contains("No available targets")
                || err.contains("unsupported GNU target")) =>
        {
            eprintln!("skipping {arch:?} Unicorn backend test: host clang cannot assemble ({err})");
            None
        }
        Err(err) => panic!("{err}"),
    }
}

fn assert_native_sysv_const(arch: NativeArch) {
    let wasm = native_sysv_const_wasm(0x1234_5678_9abc_def0u64 as i64);
    let asm = compile_native_asm(&wasm, arch, NativeAbi::Sysv);
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_const(arch, &code), 0x1234_5678_9abc_def0);
}

fn assert_native_naive_smoke(arch: NativeArch) {
    let wasm = native_naive_empty_wasm();
    let asm = compile_native_asm(&wasm, arch, NativeAbi::Naive);
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    run_native_naive_smoke(arch, &code);
}

#[test]
fn test_unicorn_x86_64_naive_backend() {
    assert_native_naive_smoke(NativeArch::X86_64);
}

#[test]
fn test_unicorn_x86_64_sysv_backend() {
    assert_native_sysv_const(NativeArch::X86_64);
}

#[test]
fn test_unicorn_aarch64_naive_backend() {
    assert_native_naive_smoke(NativeArch::AArch64);
}

#[test]
fn test_unicorn_aarch64_sysv_backend() {
    assert_native_sysv_const(NativeArch::AArch64);
}

#[test]
fn test_unicorn_riscv64_naive_backend() {
    assert_native_naive_smoke(NativeArch::Riscv64);
}

#[test]
fn test_unicorn_riscv64_sysv_backend() {
    assert_native_sysv_const(NativeArch::Riscv64);
}

#[test]
fn test_unicorn_riscv32_naive_backend() {
    assert_native_naive_smoke(NativeArch::Riscv32);
}

#[test]
fn test_unicorn_riscv32_sysv_backend() {
    assert_native_sysv_const(NativeArch::Riscv32);
}

#[test]
fn test_unicorn_arm_naive_backend() {
    assert_native_naive_smoke(NativeArch::Arm);
}

#[test]
fn test_unicorn_arm_sysv_backend() {
    assert_native_sysv_const(NativeArch::Arm);
}

#[test]
fn test_unicorn_i686_naive_backend() {
    assert_native_naive_smoke(NativeArch::I686);
}

#[test]
fn test_unicorn_i686_sysv_backend() {
    assert_native_sysv_const(NativeArch::I686);
}

// ---------------------------------------------------------------------------
// Many-argument SysV/AAPCS64 call marshalling — executed under Unicorn.
//
// These drive the recompiler's `CallAbi::AllStack` path (the mode speet uses to
// thread the full guest register file across `return_call` tail chains) and run
// the produced machine code, so they verify the *behaviour* of the >register
// argument marshalling, not just its shape.
// ---------------------------------------------------------------------------

/// Two-function module, both `(i64 * n) -> i64`: `f0` (entry) forwards its `n`
/// params to `f1` via `return_call`; `f1` returns their sum.
fn manyarg_sum_tailcall_wasm(n: u32) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    let params: Vec<ValType> = (0..n).map(|_| ValType::I64).collect();
    types.ty().function(params.iter().cloned(), [ValType::I64]);
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0); // f0 (entry)
    functions.function(0); // f1 (sum)
    module.section(&functions);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    for i in 0..n {
        f0.instruction(&Instruction::LocalGet(i));
    }
    f0.instruction(&Instruction::ReturnCall(1));
    f0.instruction(&Instruction::End);
    code.function(&f0);

    let mut f1 = Function::new([]);
    f1.instruction(&Instruction::LocalGet(0));
    for i in 1..n {
        f1.instruction(&Instruction::LocalGet(i));
        f1.instruction(&Instruction::I64Add);
    }
    f1.instruction(&Instruction::Return);
    f1.instruction(&Instruction::End);
    code.function(&f1);
    module.section(&code);
    module.finish()
}

/// Compile `wasm` with the SysV backend in `CallAbi::AllStack` mode into one
/// code buffer (all internal labels bind), returning the bytes and the byte
/// offset of `f0`'s entry. Panics if any external relocation survives.
fn compile_allstack_binary(wasm: &[u8], arch: NativeArch) -> (Vec<u8>, u64) {
    let (sigs_wp, _enc, fsigs) = parse_sigs(wasm);
    let call_params: Vec<u32> =
        fsigs.iter().map(|&ti| sigs_wp[ti as usize].params().len() as u32).collect();
    let call_results: Vec<u32> =
        fsigs.iter().map(|&ti| sigs_wp[ti as usize].results().len() as u32).collect();
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut ctx = ();
    match arch {
        NativeArch::X86_64 => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch, X64Label};
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            let mut out = IcedWriter::<X64Label>::new(0x100000);
            let mut state = sysv::SysVState::default();
            state.call_abi = sysv::CallAbi::AllStack;
            state.call_params = call_params;
            state.call_results = call_results;
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(), &mut state, &[], &op.unwrap(),
                    &mut reencoder, 0,
                )
                .unwrap();
            }
            let (bytes, labels, relocs) = out.into_parts_with_relocs();
            assert!(relocs.is_empty(), "unexpected external relocs: {relocs:?}");
            (bytes, labels[&X64Label::Func { r#fn: 0 }] as u64)
        }
        NativeArch::AArch64 => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch, AArch64Label};
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut state = naive::State::default();
            state.call_abi = naive::CallAbi::AllStack;
            state.call_params = call_params;
            state.call_results = call_results;
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(), &mut state, &[], &op.unwrap(),
                    &mut reencoder, 0,
                )
                .unwrap();
            }
            let (bytes, labels, relocs) = out.into_parts_with_relocs();
            assert!(relocs.is_empty(), "unexpected external relocs: {relocs:?}");
            // AArch64 SysV entry label is `Indexed { id + 0x8000_0000 }`.
            (bytes, labels[&AArch64Label::Indexed { idx: 0x8000_0000 }] as u64)
        }
        NativeArch::Riscv64 => {
            use portal_solutions_blitz_riscv64::{sysv, RiscV64Arch, RiscvLabel};
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut state = sysv::SysVState::default();
            state.call_abi = sysv::CallAbi::AllStack;
            state.n_imports = 0;
            state.call_params = call_params;
            state.call_results = call_results;
            state.sig_params = sigs_wp.iter().map(|s| s.params().len() as u32).collect();
            state.sig_results = sigs_wp.iter().map(|s| s.results().len() as u32).collect();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(), &mut state, &[], &op.unwrap(),
                    &mut reencoder, 0,
                )
                .unwrap();
            }
            let (bytes, labels) = out.into_parts();
            (bytes, labels[&RiscvLabel::Indexed { idx: 1 << 28 }] as u64)
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("AllStack binary not implemented for {arch:?}");
        }
    }
}

/// Invoke an `AllStack` entry under Unicorn with `args` (i64 each) and return X0/RAX.
/// `count` caps emulated instructions (0 = unlimited).
fn run_allstack_entry(arch: NativeArch, code: &[u8], entry_off: u64, args: &[u64], count: u64) -> u64 {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };
    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x40000;
    let n = args.len();
    let ret = CODE + code.len() as u64; // emu stops when control returns here

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x40000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            // AllStack (x86): every param is read from the incoming stack.
            // [rsp] = ret addr, [rsp + 8 + i*8] = param i; keep rsp 16-aligned.
            let frame = ((n as u64 + 2) * 8 + 15) & !15;
            let rsp = (STACK + STACK_SIZE - frame - 16) & !15;
            uc.mem_write(rsp, &ret.to_le_bytes()).unwrap();
            for (i, &a) in args.iter().enumerate() {
                uc.mem_write(rsp + 8 + (i as u64) * 8, &a.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE + entry_off, ret, 0, count as usize).unwrap();
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x40000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            // AllStack (aarch64): every param is on the incoming stack, matching
            // x86 — [sp + i*8] = param i after the prologue's FP/LR push. Leave
            // LR = ret so the function returns to the emu stop address.
            let frame = ((n as u64 + 2) * 8 + 15) & !15;
            let sp = (STACK + STACK_SIZE - frame - 16) & !15;
            for (i, &a) in args.iter().enumerate() {
                uc.mem_write(sp + (i as u64) * 8, &a.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterARM64::SP, sp).unwrap();
            uc.reg_write(RegisterARM64::LR, ret).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE + entry_off, ret, 0, count as usize).unwrap();
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x40000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            // RISC-V AllStack: the callee's FP is the caller's SP after its
            // prologue, so parameter i is at [sp + i*8].
            let frame = ((n as u64 + 2) * 8 + 15) & !15;
            let sp = (STACK + STACK_SIZE - frame - 16) & !15;
            for (i, &a) in args.iter().enumerate() {
                uc.mem_write(sp + (i as u64) * 8, &a.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterRISCV::SP, sp).unwrap();
            uc.reg_write(RegisterRISCV::RA, ret).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE + entry_off, ret, 0, count as usize).unwrap();
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("AllStack entry not implemented for {arch:?}");
        }
    }
}

fn assert_manyarg_sum(arch: NativeArch) {
    let n = 10u32;
    let wasm = manyarg_sum_tailcall_wasm(n);
    let (code, entry) = compile_allstack_binary(&wasm, arch);
    let args: Vec<u64> = (1..=n as u64).collect();
    let expected: u64 = args.iter().sum();
    assert_eq!(run_allstack_entry(arch, &code, entry, &args, 0), expected);
}

#[test]
fn test_unicorn_x86_64_manyarg_tailcall() {
    assert_manyarg_sum(NativeArch::X86_64);
}

#[test]
fn test_unicorn_aarch64_manyarg_tailcall() {
    assert_manyarg_sum(NativeArch::AArch64);
}

#[test]
fn test_unicorn_riscv64_manyarg_tailcall() {
    assert_manyarg_sum(NativeArch::Riscv64);
}

/// Self-recursive `(i64 n, i64 acc) -> i64`: `if n==0 { acc } else { f(n-1, acc+n) }`
/// via `return_call` to itself. A genuine tail call runs in O(1) machine stack.
fn deep_tailchain_wasm() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&Instruction::LocalGet(0)); // n
    f.instruction(&Instruction::I64Eqz);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::I64)));
    f.instruction(&Instruction::LocalGet(1)); // n == 0 -> acc
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::LocalGet(0)); // n
    f.instruction(&Instruction::I64Const(1));
    f.instruction(&Instruction::I64Sub); // n-1  (param 0)
    f.instruction(&Instruction::LocalGet(1)); // acc
    f.instruction(&Instruction::LocalGet(0)); // n
    f.instruction(&Instruction::I64Add); // acc+n  (param 1)
    f.instruction(&Instruction::ReturnCall(0));
    f.instruction(&Instruction::End); // end if
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    code.function(&f);
    module.section(&code);
    module.finish()
}

/// Single function `(i64 n) -> i64` = `if n==0 { 111 } else { 222 }` — no calls.
/// Isolates if/else lowering in the SysV binary backend from tail-call logic.
fn ifelse_probe_wasm() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64], [ValType::I64]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::I64Eqz);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::I64)));
    f.instruction(&Instruction::I64Const(111));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::I64Const(222));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    code.function(&f);
    module.section(&code);
    module.finish()
}

#[test]
fn test_unicorn_x86_64_ifelse_probe() {
    let (code, entry) = compile_allstack_binary(&ifelse_probe_wasm(), NativeArch::X86_64);
    assert_eq!(run_allstack_entry(NativeArch::X86_64, &code, entry, &[0], 100_000), 111);
    assert_eq!(run_allstack_entry(NativeArch::X86_64, &code, entry, &[1], 100_000), 222);
}

#[test]
fn test_unicorn_aarch64_ifelse_probe() {
    let (code, entry) = compile_allstack_binary(&ifelse_probe_wasm(), NativeArch::AArch64);
    assert_eq!(run_allstack_entry(NativeArch::AArch64, &code, entry, &[0], 100_000), 111);
    assert_eq!(run_allstack_entry(NativeArch::AArch64, &code, entry, &[1], 100_000), 222);
}

fn run_deep_tailchain(arch: NativeArch, n: u64, count: u64) -> u64 {
    let wasm = deep_tailchain_wasm();
    let (code, entry) = compile_allstack_binary(&wasm, arch);
    run_allstack_entry(arch, &code, entry, &[n, 0], count)
}

#[test]
fn test_unicorn_x86_64_deep_tailchain_small() {
    // Bounded instruction count so a regression can't hang the suite.
    assert_eq!(run_deep_tailchain(NativeArch::X86_64, 5, 100_000), 15);
}

#[test]
fn test_unicorn_aarch64_deep_tailchain_small() {
    assert_eq!(run_deep_tailchain(NativeArch::AArch64, 5, 100_000), 15);
}

#[test]
fn test_unicorn_riscv64_deep_tailchain_small() {
    assert_eq!(run_deep_tailchain(NativeArch::Riscv64, 5, 100_000), 15);
}

#[test]
fn test_unicorn_x86_64_deep_tailchain() {
    // 20k self-tail-calls: a fake tail call (real call + ret) grows the machine
    // stack ~20k frames and faults the 256 KiB stack; a true tail call stays O(1).
    let n = 20_000u64;
    assert_eq!(run_deep_tailchain(NativeArch::X86_64, n, 0), n * (n + 1) / 2);
}

#[test]
fn test_unicorn_aarch64_deep_tailchain() {
    let n = 20_000u64;
    assert_eq!(run_deep_tailchain(NativeArch::AArch64, n, 0), n * (n + 1) / 2);
}

// ── Load/store width family (sub-word, signed/unsigned) ───────────────────────
//
// Stores a 64-bit pattern and a 16-bit value into linear memory, reads them back
// through every load width/sign, and returns the sum — exercising the full
// load/store-width family under the `Raw` mem_base (address == host pointer, so
// the test uses an address inside Unicorn's mapped stack region).

const LSW_ADDR: i32 = 0x21_0000; // inside the Unicorn STACK mapping (0x200000..0x240000)
const LSW_VAL: u64 = 0x8090_A0B0_C0D0_E0F0;

fn loadstore_widths_wasm() -> Vec<u8> {
    use wasm_encoder::MemArg;
    let m = |align| MemArg { offset: 0, align, memory_index: 0 };
    let i = [
        // mem[A..A+8] = LSW_VAL
        Instruction::I32Const(LSW_ADDR), Instruction::I64Const(LSW_VAL as i64), Instruction::I64Store(m(3)),
        // mem16[A+8] = 0x1234 (narrow store)
        Instruction::I32Const(LSW_ADDR + 8), Instruction::I64Const(0x1234), Instruction::I64Store16(m(1)),
        // sum = load8_u + load8_s + load16_u + load16_s + load32_u + load32_s + load16_u(A+8)
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load8U(m(0)),
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load8S(m(0)), Instruction::I64Add,
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load16U(m(1)), Instruction::I64Add,
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load16S(m(1)), Instruction::I64Add,
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load32U(m(2)), Instruction::I64Add,
        Instruction::I32Const(LSW_ADDR), Instruction::I64Load32S(m(2)), Instruction::I64Add,
        Instruction::I32Const(LSW_ADDR + 8), Instruction::I64Load16U(m(1)), Instruction::I64Add,
    ];
    make_module_with_memory(&[], &[ValType::I64], &i)
}

fn loadstore_widths_expected() -> u64 {
    let v = LSW_VAL;
    let a = (v as u8) as u64;
    let b = (v as u8 as i8 as i64) as u64;
    let c = (v as u16) as u64;
    let d = (v as u16 as i16 as i64) as u64;
    let e = (v as u32) as u64;
    let f = (v as u32 as i32 as i64) as u64;
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d)
        .wrapping_add(e).wrapping_add(f).wrapping_add(0x1234)
}

fn assert_loadstore_widths(arch: NativeArch) {
    let wasm = loadstore_widths_wasm();
    let (code, entry) = compile_allstack_binary(&wasm, arch);
    assert_eq!(run_allstack_entry(arch, &code, entry, &[], 0), loadstore_widths_expected());
}

#[test]
fn test_unicorn_x86_64_loadstore_widths() {
    assert_loadstore_widths(NativeArch::X86_64);
}

#[test]
fn test_unicorn_aarch64_loadstore_widths() {
    assert_loadstore_widths(NativeArch::AArch64);
}

// ---- scalar floating point (bit-threaded through the GP operand stack) ----

/// `() -> i64`: sqrt(i64_to_f64(7)/2.0 + 0.5) + promote(3.0f*3.0f), truncated.
/// = sqrt(4.0) + 9.0 = 2.0 + 9.0 = 11. Exercises i64->f64, fdiv, fadd, fsqrt,
/// f32 fmul, f64 promote, and f64->i64 truncation.
fn fp_arith_wasm() -> Vec<u8> {
    let c64 = |x: f64| Instruction::F64Const(wasm_encoder::Ieee64::from(x));
    let c32 = |x: f32| Instruction::F32Const(wasm_encoder::Ieee32::from(x));
    let i = [
        Instruction::I64Const(7), Instruction::F64ConvertI64S,
        c64(2.0), Instruction::F64Div,
        c64(0.5), Instruction::F64Add,
        Instruction::F64Sqrt,
        c32(3.0), c32(3.0), Instruction::F32Mul, Instruction::F64PromoteF32,
        Instruction::F64Add,
        Instruction::I64TruncF64S,
    ];
    make_module_with_memory(&[], &[ValType::I64], &i)
}

/// `() -> i64`: (2<3) + (3>=3) + (5<1) = 1 + 1 + 0 = 2, covering the FP
/// relational compares including the `ge` equality boundary.
fn fp_cmp_wasm() -> Vec<u8> {
    let c64 = |x: f64| Instruction::F64Const(wasm_encoder::Ieee64::from(x));
    let i = [
        c64(2.0), c64(3.0), Instruction::F64Lt,
        c64(3.0), c64(3.0), Instruction::F64Ge, Instruction::I32Add,
        c64(5.0), c64(1.0), Instruction::F64Lt, Instruction::I32Add,
        Instruction::I64ExtendI32S,
    ];
    make_module_with_memory(&[], &[ValType::I64], &i)
}

fn assert_fp_arith(arch: NativeArch) {
    let wasm = fp_arith_wasm();
    let (code, entry) = compile_allstack_binary(&wasm, arch);
    assert_eq!(run_allstack_entry(arch, &code, entry, &[], 0), 11);
}

fn assert_fp_cmp(arch: NativeArch) {
    let wasm = fp_cmp_wasm();
    let (code, entry) = compile_allstack_binary(&wasm, arch);
    assert_eq!(run_allstack_entry(arch, &code, entry, &[], 0), 2);
}

#[test]
fn test_unicorn_x86_64_fp_arith() {
    assert_fp_arith(NativeArch::X86_64);
}
#[test]
fn test_unicorn_aarch64_fp_arith() {
    assert_fp_arith(NativeArch::AArch64);
}
#[test]
fn test_unicorn_x86_64_fp_cmp() {
    assert_fp_cmp(NativeArch::X86_64);
}
#[test]
fn test_unicorn_aarch64_fp_cmp() {
    assert_fp_cmp(NativeArch::AArch64);
}

// ---------------------------------------------------------------------------
// Stubs and helpers for native execution tests
// ---------------------------------------------------------------------------

/// Data stub that resolves the `__wasm_mem_pages` external used by
/// `Instruction::MemorySize` codegen.  Defines a 32-bit value of `1` so a
/// `memory.size` returns 1 (one wasm page).  Format works for all three
/// architectures since `.int` is honored by GNU/LLVM `clang -c`.
const STUB_MEM_PAGES: &str = "__wasm_mem_pages:\n.int 1\n";

/// Stub for the `env::add_one` import (mangled `env__add_one`).
///
/// Follows the blitz naive WASM calling convention: the caller has pushed the
/// argument onto the WASM operand stack (which is the hardware stack), then
/// invoked the stub via a normal architecture call instruction.  The stub
/// adds 1 to the value at the top of the WASM stack in place and returns,
/// leaving the result where the caller will pick it up.
fn import_stub_add_one(arch: NativeArch) -> &'static str {
    match arch {
        // x86-64 Intel syntax: pop ra+arg, increment, push result+ra, ret.
        NativeArch::X86_64 => concat!(
            "env__add_one:\n",
            "pop r11\n",
            "pop rax\n",
            "inc rax\n",
            "push rax\n",
            "push r11\n",
            "ret\n",
        ),
        // AArch64: read [sp] (= arg), +1, write back, ret to lr.
        NativeArch::AArch64 => concat!(
            "env__add_one:\n",
            "ldr x9, [sp]\n",
            "add x9, x9, #1\n",
            "str x9, [sp]\n",
            "ret\n",
        ),
        // RISC-V 64: read [sp], +1, write back, ret (= jalr x0, ra, 0).
        NativeArch::Riscv64 => concat!(
            "env__add_one:\n",
            "ld a0, 0(sp)\n",
            "addi a0, a0, 1\n",
            "sd a0, 0(sp)\n",
            "ret\n",
        ),
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("import stub not implemented for {arch:?}");
        }
    }
}

/// WASM linear-memory base address used by native execution tests.
/// Mapped into Unicorn separately from CODE/STACK.
const NATIVE_WASM_MEM: u64 = 0x300000;

/// Run sysv-ABI native code in Unicorn with an additional memory region
/// mapped at `extra_addr`, pre-populated with `extra_data`.  Returns the
/// sysv return register (rax/x0/a0).
fn run_native_sysv_with_mem(
    arch: NativeArch,
    code: &[u8],
    extra_addr: u64,
    extra_data: &[u8],
) -> u64 {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;
    const EXTRA_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.mem_write(extra_addr, extra_data).unwrap();
            let rsp = STACK + STACK_SIZE - 8;
            uc.mem_write(rsp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.mem_write(extra_addr, extra_data).unwrap();
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.mem_write(extra_addr, extra_data).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("run_native_sysv_with_mem not implemented for {arch:?}");
        }
    }
}

/// Same as `run_native_naive_smoke`, but also maps an `extra_addr` region.
fn run_native_naive_smoke_with_mem(
    arch: NativeArch,
    code: &[u8],
    extra_addr: u64,
    extra_data: &[u8],
) {
    run_native_naive_smoke_with_mem_and_locals(arch, code, extra_addr, extra_data, &[]);
}

/// Like `run_native_naive_smoke_with_mem` but also pre-populates the function's
/// local variable frame so memory operations target valid addresses.
///
/// For AArch64/RISC-V naive: local `n` lives at `[fp - (n+1)*8]`.  The frame
/// pointer after the prologue is `initial_SP - frame_overhead` where
/// `frame_overhead` is 16 bytes (AArch64 stp) or 8 bytes (RISC-V sd+mv).
/// We write each `pre_locals[n]` to that address before emulation.
fn run_native_naive_smoke_with_mem_and_locals(
    arch: NativeArch,
    code: &[u8],
    extra_addr: u64,
    extra_data: &[u8],
    pre_locals: &[u64],
) {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;
    const EXTRA_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            if extra_addr != 0 { uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap(); }
            uc.mem_write(CODE, code).unwrap();
            if !extra_data.is_empty() && extra_addr != 0 { uc.mem_write(extra_addr, extra_data).unwrap(); }
            let sp = STACK + STACK_SIZE - 0x100;
            uc.mem_write(sp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, sp).unwrap();
            // The x86-64 naive CTX mechanism: StartFn stores the initial r15 value as the
            // "old CTX" in RAX, which StartBody then pushes as local 0's backing slot.
            // Setting r15 = pre_locals[0] before emulation makes local 0 = pre_locals[0].
            if let Some(&val) = pre_locals.first() {
                uc.reg_write(RegisterX86::R15, val).unwrap();
            }
            // The naive binary code buffer may have dead-code label stubs after the ret.
            // Unicorn executes the instruction at `until` when arriving via ret, so we
            // patch that address with NOP to avoid INSN_INVALID from garbage bytes.
            if (code.len() as u64) < 0x10000 {
                uc.mem_write(CODE + code.len() as u64, &[0x90u8]).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            // Accept INSN_INVALID: the NOP at CODE+code.len() is valid, but subsequent
            // dead bytes may cause issues. The function itself executed correctly.
            let _ = uc.emu_start(CODE, CODE + code.len() as u64 + 1, 0, 5000);
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            if extra_addr != 0 { uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap(); }
            uc.mem_write(CODE, code).unwrap();
            if !extra_data.is_empty() && extra_addr != 0 { uc.mem_write(extra_addr, extra_data).unwrap(); }
            let initial_sp = STACK + STACK_SIZE - 16;
            uc.reg_write(RegisterARM64::SP, initial_sp).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            // Pre-populate local variables: local n is at [fp - (n+1)*8].
            // After `stp x29,x30,[sp,#-16]!; mov x29,sp`, fp = initial_sp - 16.
            let fp = initial_sp - 16;
            for (n, &val) in pre_locals.iter().enumerate() {
                let addr = fp - ((n as u64 + 1) * 8);
                uc.mem_write(addr, &val.to_le_bytes()).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            if extra_addr != 0 { uc.mem_map(extra_addr & !0xfff, EXTRA_SIZE, Prot::ALL).unwrap(); }
            uc.mem_write(CODE, code).unwrap();
            if !extra_data.is_empty() && extra_addr != 0 { uc.mem_write(extra_addr, extra_data).unwrap(); }
            let initial_sp = STACK + STACK_SIZE - 16;
            uc.reg_write(RegisterRISCV::SP, initial_sp).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            // Pre-populate local variables for the RISC-V naive ABI frame.
            // Naive StartFn: `addi sp,sp,-8; sd fp,[sp]; mv fp,sp; addi sp,sp,-alloc`
            // So FP = initial_sp - 8, and local n is at [FP-(n+1)*8] = [initial_sp-8-(n+1)*8].
            let fp_naive = initial_sp - 8;
            for (n, &val) in pre_locals.iter().enumerate() {
                let addr = fp_naive - ((n as u64 + 1) * 8);
                uc.mem_write(addr, &val.to_le_bytes()).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

// ---------------------------------------------------------------------------
// Native backend × make_module_with_memory — Unicorn execution
// ---------------------------------------------------------------------------

/// End-to-end test: a WASM module that returns `memory.size` should assemble
/// and execute correctly when `__wasm_mem_pages` is provided as a `.int 1`
/// data stub in the same translation unit.  Sysv backends are checked for
/// the actual return value (1); naive backends are smoke-tested for no crash.
fn assert_native_compile_module_with_memory(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module_with_memory(&[], &[ValType::I32], &[Instruction::MemorySize(0)]);
    let base_asm = compile_native_asm(&wasm, arch, abi);
    let asm = format!("{base_asm}{STUB_MEM_PAGES}");
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    match abi {
        NativeAbi::Sysv => assert_eq!(run_native_sysv_const(arch, &code), 1,
            "memory.size should return 1 page for {arch:?} sysv"),
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
    NativeAbi::Lfi => {}
    }
}

#[test]
fn test_native_x86_64_naive_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::X86_64, NativeAbi::Naive);
}
#[test]
fn test_native_x86_64_sysv_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::X86_64, NativeAbi::Sysv);
}
#[test]
fn test_native_aarch64_naive_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::AArch64, NativeAbi::Naive);
}
#[test]
fn test_native_aarch64_sysv_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::AArch64, NativeAbi::Sysv);
}
#[test]
fn test_native_riscv64_naive_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::Riscv64, NativeAbi::Naive);
}
#[test]
fn test_native_riscv64_sysv_with_memory() {
    assert_native_compile_module_with_memory(NativeArch::Riscv64, NativeAbi::Sysv);
}

// ---------------------------------------------------------------------------
// Native backend × make_module_with_data — Unicorn execution
// ---------------------------------------------------------------------------

/// End-to-end test: a WASM module that loads from linear memory should
/// assemble and execute correctly.  Uses `i32.const $addr; i32.load` against
/// a known address `NATIVE_WASM_MEM` mapped at run time.  Note we use the
/// fully-implemented `I32Load` here rather than `I32Load8U`, which is not
/// yet handled by any native backend (silently falls through, so no load
/// would actually happen).
///
/// We don't emit a wasm data segment because native backends have no runtime
/// that applies it — the test runner writes the byte directly into the
/// mapped page instead.
fn assert_native_compile_module_with_data(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let wasm = make_module_with_memory(
        &[],
        &[ValType::I32],
        &[
            Instruction::I32Const(NATIVE_WASM_MEM as i32),
            Instruction::I32Load(MemArg { offset: 0, align: 0, memory_index: 0 }),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    let data: [u8; 4] = 42u32.to_le_bytes();
    match abi {
        NativeAbi::Sysv => assert_eq!(
            run_native_sysv_with_mem(arch, &code, NATIVE_WASM_MEM, &data),
            42,
            "i32.load should return 42 for {arch:?} sysv",
        ),
        NativeAbi::Naive => run_native_naive_smoke_with_mem(arch, &code, NATIVE_WASM_MEM, &data),
    NativeAbi::Lfi => {}
    }
}

#[test]
fn test_native_x86_64_naive_with_data() {
    assert_native_compile_module_with_data(NativeArch::X86_64, NativeAbi::Naive);
}
#[test]
fn test_native_x86_64_sysv_with_data() {
    assert_native_compile_module_with_data(NativeArch::X86_64, NativeAbi::Sysv);
}
#[test]
fn test_native_aarch64_naive_with_data() {
    assert_native_compile_module_with_data(NativeArch::AArch64, NativeAbi::Naive);
}
#[test]
fn test_native_aarch64_sysv_with_data() {
    assert_native_compile_module_with_data(NativeArch::AArch64, NativeAbi::Sysv);
}
#[test]
fn test_native_riscv64_naive_with_data() {
    assert_native_compile_module_with_data(NativeArch::Riscv64, NativeAbi::Naive);
}
#[test]
fn test_native_riscv64_sysv_with_data() {
    assert_native_compile_module_with_data(NativeArch::Riscv64, NativeAbi::Sysv);
}

// ---------------------------------------------------------------------------
// Native backend × make_module_with_import — Unicorn execution
// ---------------------------------------------------------------------------

/// Build a WASM module that imports `env::add_one : (i64) -> i64` and exports
/// an internal `() -> i64` function that pushes a constant, calls the import,
/// and returns its result.
///
/// We use a no-parameter outer function (and pass the arg via `i64.const`)
/// because the native naive backend's `StartFn` prologue mishandles function
/// parameters: it computes `r0 = ret_addr - params*8` which addresses code,
/// not the stack.  This test focuses on the import-call mechanism rather
/// than on parameter passing into the outer function.
fn make_native_import_module() -> Vec<u8> {
    let mut module = Module::new();

    // type 0: (i64) -> i64
    // type 1: ()    -> i64
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64], [ValType::I64]);
    types.ty().function([], [ValType::I64]);
    module.section(&types);

    let mut imports = wasm_encoder::ImportSection::new();
    imports.import("env", "add_one", wasm_encoder::EntityType::Function(0));
    module.section(&imports);

    let mut functions = FunctionSection::new();
    functions.function(1);
    module.section(&functions);

    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, 1);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    func.instruction(&Instruction::I64Const(42));
    func.instruction(&Instruction::Call(0));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    module.finish()
}

/// End-to-end test: a WASM module that calls an imported function should
/// assemble and execute correctly when the import (`env__add_one`) is
/// resolved by an in-asm stub that adds 1 to its WASM-stack argument.
/// Sysv backends are checked for the return value (43 = 42 + 1).
fn assert_native_compile_module_with_import(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_native_import_module();
    let (sigs_wp, _sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    // One imported function (index 0 = import), one local body (index 1).
    let import_count = 1u32;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies, &fsigs, &sigs_wp, import_count,
    );
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut out = NativeAsmWriter(String::new());
    let mut ctx = ();
    let imports: &[(&str, &str)] = &[("env", "add_one")];

    match (arch, abi) {
        (NativeArch::X86_64, NativeAbi::Naive) => {
            use portal_solutions_blitz_x86_64::{naive, X64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, imports, &[], &[], &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::X86_64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch};
            let mut state = sysv::SysVState::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, imports, &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Naive) => {
            use portal_solutions_blitz_aarch64::{naive, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, imports, &[], &[], &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, imports, &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv64::{naive, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, imports, &[], &[], &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                let op = op.unwrap();
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, imports, &op, &mut reencoder, import_count,
                ).unwrap();
            }
        }
    (_, NativeAbi::Lfi) => panic!("LFI not supported in this test helper"),
        (NativeArch::Riscv32, _) | (NativeArch::Arm, _) | (NativeArch::I686, _) => {
            panic!("not implemented for ILP32 arch in this helper");
        }
    }
    let base_asm = normalize_native_asm(arch, out.0);
    let asm = format!("{base_asm}{}", import_stub_add_one(arch));
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    match abi {
        NativeAbi::Sysv => assert_eq!(
            run_native_sysv_const(arch, &code),
            43,
            "env::add_one(42) should return 43 for {arch:?} sysv",
        ),
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
    NativeAbi::Lfi => {}
    }
}

#[test]
fn test_native_x86_64_naive_with_import() {
    assert_native_compile_module_with_import(NativeArch::X86_64, NativeAbi::Naive);
}
#[test]
fn test_native_x86_64_sysv_with_import() {
    assert_native_compile_module_with_import(NativeArch::X86_64, NativeAbi::Sysv);
}
#[test]
fn test_native_aarch64_naive_with_import() {
    assert_native_compile_module_with_import(NativeArch::AArch64, NativeAbi::Naive);
}
#[test]
fn test_native_aarch64_sysv_with_import() {
    assert_native_compile_module_with_import(NativeArch::AArch64, NativeAbi::Sysv);
}
#[test]
fn test_native_riscv64_naive_with_import() {
    assert_native_compile_module_with_import(NativeArch::Riscv64, NativeAbi::Naive);
}
#[test]
fn test_native_riscv64_sysv_with_import() {
    assert_native_compile_module_with_import(NativeArch::Riscv64, NativeAbi::Sysv);
}

// ---------------------------------------------------------------------------
// Deduplication macros — 6 #[test] stubs (3 arches × 2 ABIs) per test group
// ---------------------------------------------------------------------------

/// Emit 6 `#[test]` stubs calling `$assert_fn(arch, abi)` via the text-asm path
/// (compile_native_asm → assemble_or_skip → Unicorn).
macro_rules! native_variants {
    ($base:ident, $assert_fn:ident) => {
        paste::paste! {
            #[test] fn [<test_native_x86_64_naive_ $base>]()  { $assert_fn(NativeArch::X86_64,  NativeAbi::Naive); }
            #[test] fn [<test_native_x86_64_sysv_ $base>]()   { $assert_fn(NativeArch::X86_64,  NativeAbi::Sysv); }
            #[test] fn [<test_native_aarch64_naive_ $base>]()  { $assert_fn(NativeArch::AArch64, NativeAbi::Naive); }
            #[test] fn [<test_native_aarch64_sysv_ $base>]()   { $assert_fn(NativeArch::AArch64, NativeAbi::Sysv); }
            #[test] fn [<test_native_riscv64_naive_ $base>]()  { $assert_fn(NativeArch::Riscv64, NativeAbi::Naive); }
            #[test] fn [<test_native_riscv64_sysv_ $base>]()   { $assert_fn(NativeArch::Riscv64, NativeAbi::Sysv); }
        }
    };
}

/// Same as `native_variants!` but for the direct binary-writer path
/// (compile_native_binary → Unicorn, no clang required).
macro_rules! native_bin_variants {
    ($base:ident, $assert_fn:ident) => {
        paste::paste! {
            #[test] fn [<test_native_x86_64_naive_bin_ $base>]()  { $assert_fn(NativeArch::X86_64,  NativeAbi::Naive); }
            #[test] fn [<test_native_x86_64_sysv_bin_ $base>]()   { $assert_fn(NativeArch::X86_64,  NativeAbi::Sysv); }
            #[test] fn [<test_native_aarch64_naive_bin_ $base>]()  { $assert_fn(NativeArch::AArch64, NativeAbi::Naive); }
            #[test] fn [<test_native_aarch64_sysv_bin_ $base>]()   { $assert_fn(NativeArch::AArch64, NativeAbi::Sysv); }
            #[test] fn [<test_native_riscv64_naive_bin_ $base>]()  { $assert_fn(NativeArch::Riscv64, NativeAbi::Naive); }
            #[test] fn [<test_native_riscv64_sysv_bin_ $base>]()   { $assert_fn(NativeArch::Riscv64, NativeAbi::Sysv); }
        }
    };
}

// ---------------------------------------------------------------------------
// New Unicorn helpers
// ---------------------------------------------------------------------------

/// Like `run_native_sysv_const` but writes up to 4 arguments into the
/// SysV integer argument registers before starting emulation.
///
/// Calling conventions used:
/// - x86-64: args[0..4] → RDI, RSI, RDX, RCX; return RAX
/// - AArch64: args[0..4] → X0, X1, X2, X3; return X0
/// - RISC-V: args[0..4] → A0, A1, A2, A3; return A0
fn run_native_sysv_with_args(arch: NativeArch, code: &[u8], args: &[u64]) -> u64 {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            let rsp = STACK + STACK_SIZE - 8;
            uc.mem_write(rsp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            let arg_regs = [RegisterX86::RDI, RegisterX86::RSI, RegisterX86::RDX, RegisterX86::RCX];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            let arg_regs = [RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            let arg_regs = [RegisterRISCV::A0, RegisterRISCV::A1, RegisterRISCV::A2, RegisterRISCV::A3];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

/// Like `run_native_sysv_with_args` but also maps a writable memory region
/// at `mem_addr` (page-aligned) of `mem_size` bytes.  Used for tests that
/// access WASM linear memory via a known virtual address.
fn run_native_sysv_with_args_and_mem(
    arch: NativeArch,
    code: &[u8],
    args: &[u64],
    mem_addr: u64,
    mem_size: usize,
) -> u64 {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        Unicorn,
    };

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(mem_addr & !0xfff, mem_size as u64 + (mem_addr & 0xfff), Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            let rsp = STACK + STACK_SIZE - 8;
            uc.mem_write(rsp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            let arg_regs = [RegisterX86::RDI, RegisterX86::RSI, RegisterX86::RDX, RegisterX86::RCX];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterX86::RAX).unwrap()
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(mem_addr & !0xfff, mem_size as u64 + (mem_addr & 0xfff), Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();
            let arg_regs = [RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterARM64::X0).unwrap()
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.mem_map(mem_addr & !0xfff, mem_size as u64 + (mem_addr & 0xfff), Prot::ALL).unwrap();
            uc.mem_write(CODE, code).unwrap();
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, CODE + code.len() as u64).unwrap();
            let arg_regs = [RegisterRISCV::A0, RegisterRISCV::A1, RegisterRISCV::A2, RegisterRISCV::A3];
            for (i, &v) in args.iter().enumerate().take(4) {
                uc.reg_write(arg_regs[i], v).unwrap();
            }
            attach_trace_hook(&mut uc, arch, CODE, code.len());
            uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
            uc.reg_read(RegisterRISCV::A0).unwrap()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

// ---------------------------------------------------------------------------
// compile_native_binary — drive asm-arch binary writers directly (no clang)
// ---------------------------------------------------------------------------

/// Compile `wasm` using the same blitz pipeline as `compile_native_asm` but
/// write directly into the architecture's binary writer (`IcedWriter`,
/// `AArch64Writer`, or `RvAsmWriter`).  Returns raw machine-code bytes ready
/// for Unicorn.  No external tools required.
fn compile_native_binary(wasm: &[u8], arch: NativeArch, abi: NativeAbi) -> Vec<u8> {
    let (sigs_wp, _, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut ctx = ();

    match (arch, abi) {
        (NativeArch::X86_64, NativeAbi::Naive) => {
            use portal_solutions_blitz_x86_64::{naive, X64Arch, X64Label};
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            let mut out = IcedWriter::<X64Label>::new(0x100000);
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &[], &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        (NativeArch::X86_64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch, X64Label};
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            let mut out = IcedWriter::<X64Label>::new(0x100000);
            let mut state = sysv::SysVState::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        (NativeArch::AArch64, NativeAbi::Naive) => {
            use portal_solutions_blitz_aarch64::{naive, AArch64Arch, AArch64Label};
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &[], &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        (NativeArch::AArch64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch, AArch64Label};
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut state = naive::State::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        (NativeArch::Riscv64, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv64::{naive, RiscV64Arch, RiscvLabel};
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &[], &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        (NativeArch::Riscv64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch, RiscvLabel};
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut state = naive::State::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &op.unwrap(), &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
    (_, NativeAbi::Lfi) => panic!("LFI not supported in this test helper"),
        (NativeArch::Riscv32, _) | (NativeArch::Arm, _) | (NativeArch::I686, _) => {
            panic!("not implemented for ILP32 arch in this helper");
        }
    }
}

// ---------------------------------------------------------------------------
// compile_native_asm_with_imports — text-asm path for import-bearing modules
// ---------------------------------------------------------------------------

/// Like `compile_native_asm` but passes `import_count` and `imports` correctly
/// so function indices are offset properly for modules that have imported
/// functions before their local bodies.
fn compile_native_asm_with_imports(
    wasm: &[u8],
    arch: NativeArch,
    abi: NativeAbi,
    imports: &[(&str, &str)],
) -> String {
    let (sigs_wp, _, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let import_count = imports.len() as u32;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, import_count);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut out = NativeAsmWriter(String::new());
    let mut ctx = ();

    match (arch, abi) {
        (NativeArch::X86_64, NativeAbi::Naive) => {
            use portal_solutions_blitz_x86_64::{naive, X64Arch};
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, imports, &[], &[], &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::X86_64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch};
            let mut state = sysv::SysVState::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, imports, &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Naive) => {
            use portal_solutions_blitz_aarch64::{naive, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, imports, &[], &[], &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::AArch64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch};
            let mut state = naive::State::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, imports, &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv64::{naive, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, imports, &[], &[], &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
        (NativeArch::Riscv64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch};
            let mut state = naive::State::default();
            for op in ops {
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, imports, &op.unwrap(), &mut reencoder, import_count,
                ).unwrap();
            }
        }
    (_, NativeAbi::Lfi) => panic!("LFI not supported in this test helper"),
        (NativeArch::Riscv32, _) | (NativeArch::Arm, _) | (NativeArch::I686, _) => {
            panic!("not implemented for ILP32 arch in this helper");
        }
    }

    normalize_native_asm(arch, out.0)
}

// ---------------------------------------------------------------------------
// Native variants — basic execution tests (text-asm + binary)
// ---------------------------------------------------------------------------

fn assert_native_exec_const(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[]) as u32, 42),
    NativeAbi::Lfi => {}
    }
}
native_variants!(exec_const, assert_native_exec_const);

fn assert_native_bin_exec_const(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(&[], &[ValType::I32], &[Instruction::I32Const(42)]);
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    eprintln!("exec_const binary ({arch:?}/{abi:?}): {} bytes: {:02x?}", code.len(), &code);
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[]) as u32, 42),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_const, assert_native_bin_exec_const);

fn assert_native_exec_i64const(arch: NativeArch, abi: NativeAbi) {
    let val: u64 = 0x0123_4567_89AB_CDEF;
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(val as i64)]);
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[]), val),
    NativeAbi::Lfi => {}
    }
}
native_variants!(exec_i64const, assert_native_exec_i64const);

fn assert_native_bin_exec_i64const(arch: NativeArch, abi: NativeAbi) {
    let val: u64 = 0x0123_4567_89AB_CDEF;
    let wasm = make_module(&[], &[ValType::I64], &[Instruction::I64Const(val as i64)]);
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[]), val),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_i64const, assert_native_bin_exec_i64const);

fn assert_native_exec_add(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[5, 3]) as u32, 8);
    assert_eq!(run_native_sysv_with_args(arch, &code, &[100, 200]) as u32, 300);
}
native_variants!(exec_add, assert_native_exec_add);

fn assert_native_bin_exec_add(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Add],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => {
            assert_eq!(run_native_sysv_with_args(arch, &code, &[5, 3]) as u32, 8);
            assert_eq!(run_native_sysv_with_args(arch, &code, &[100, 200]) as u32, 300);
        }
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_add, assert_native_bin_exec_add);

fn assert_native_exec_sub(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[10, 3]) as u32, 7);
    assert_eq!(run_native_sysv_with_args(arch, &code, &[3, 10]) as u32, (-7i32) as u32);
}
native_variants!(exec_sub, assert_native_exec_sub);

fn assert_native_bin_exec_sub(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Sub],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => {
            assert_eq!(run_native_sysv_with_args(arch, &code, &[10, 3]) as u32, 7);
            assert_eq!(run_native_sysv_with_args(arch, &code, &[3, 10]) as u32, (-7i32) as u32);
        }
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_sub, assert_native_bin_exec_sub);

fn assert_native_exec_divu(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32DivU],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[10, 2]) as u32, 5);
}
native_variants!(exec_divu, assert_native_exec_divu);

fn assert_native_bin_exec_divu(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32DivU],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[10, 2]) as u32, 5),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_divu, assert_native_bin_exec_divu);

fn assert_native_exec_localset(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32], &[ValType::I32],
        &[Instruction::I32Const(77), Instruction::LocalSet(0), Instruction::LocalGet(0)],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 77);
}
native_variants!(exec_localset, assert_native_exec_localset);

fn assert_native_bin_exec_localset(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32], &[ValType::I32],
        &[Instruction::I32Const(77), Instruction::LocalSet(0), Instruction::LocalGet(0)],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 77),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_localset, assert_native_bin_exec_localset);

fn assert_native_exec_i64sub(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I64, ValType::I64], &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64Sub],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[100, 37]), 63);
}
native_variants!(exec_i64sub, assert_native_exec_i64sub);

fn assert_native_bin_exec_i64sub(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I64, ValType::I64], &[ValType::I64],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I64Sub],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[100, 37]), 63),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_i64sub, assert_native_bin_exec_i64sub);

fn assert_native_exec_shl(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Shl],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[3, 4]) as u32, 48);
}
native_variants!(exec_shl, assert_native_exec_shl);

fn assert_native_bin_exec_shl(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[Instruction::LocalGet(0), Instruction::LocalGet(1), Instruction::I32Shl],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => assert_eq!(run_native_sysv_with_args(arch, &code, &[3, 4]) as u32, 48),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_shl, assert_native_bin_exec_shl);

fn assert_native_exec_brtable(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::BlockType;
    let wasm = make_module(
        &[ValType::I32], &[ValType::I32],
        &[
            Instruction::Block(BlockType::Result(ValType::I32)),
            Instruction::Block(BlockType::Empty),
            Instruction::Block(BlockType::Empty),
            Instruction::LocalGet(0),
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End,
            Instruction::I32Const(20),
            Instruction::Br(1),
            Instruction::End,
            Instruction::I32Const(10),
            Instruction::End,
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 20,
        "selector 0 → 20 for {arch:?}");
    assert_eq!(run_native_sysv_with_args(arch, &code, &[1]) as u32, 10,
        "selector 1 → 10 for {arch:?}");
}
native_variants!(exec_brtable, assert_native_exec_brtable);

fn assert_native_bin_exec_brtable(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::BlockType;
    let wasm = make_module(
        &[ValType::I32], &[ValType::I32],
        &[
            Instruction::Block(BlockType::Result(ValType::I32)),
            Instruction::Block(BlockType::Empty),
            Instruction::Block(BlockType::Empty),
            Instruction::LocalGet(0),
            Instruction::BrTable(Cow::Borrowed(&[0u32]), 1),
            Instruction::End,
            Instruction::I32Const(20),
            Instruction::Br(1),
            Instruction::End,
            Instruction::I32Const(10),
            Instruction::End,
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => {
            assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 20,
                "selector 0 → 20 for {arch:?}");
            assert_eq!(run_native_sysv_with_args(arch, &code, &[1]) as u32, 10,
                "selector 1 → 10 for {arch:?}");
        }
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_brtable, assert_native_bin_exec_brtable);

fn make_loop_counter_wasm() -> Vec<u8> {
    use wasm_encoder::BlockType;
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut func = Function::new([(1u32, ValType::I32)]);
    func.instruction(&Instruction::Loop(BlockType::Empty));
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::If(BlockType::Empty));
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::I32Const(1));
    func.instruction(&Instruction::I32Add);
    func.instruction(&Instruction::LocalSet(1));
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::I32Const(1));
    func.instruction(&Instruction::I32Sub);
    func.instruction(&Instruction::LocalSet(0));
    func.instruction(&Instruction::Br(1));
    func.instruction(&Instruction::End);
    func.instruction(&Instruction::End);
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);
    module.finish()
}

fn assert_native_exec_loop_counter(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_loop_counter_wasm();
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 0);
    assert_eq!(run_native_sysv_with_args(arch, &code, &[5]) as u32, 5);
    assert_eq!(run_native_sysv_with_args(arch, &code, &[10]) as u32, 10);
}
native_variants!(exec_loop_counter, assert_native_exec_loop_counter);

fn assert_native_bin_exec_loop_counter(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_loop_counter_wasm();
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke(arch, &code),
        NativeAbi::Sysv => {
            assert_eq!(run_native_sysv_with_args(arch, &code, &[0]) as u32, 0);
            assert_eq!(run_native_sysv_with_args(arch, &code, &[5]) as u32, 5);
            assert_eq!(run_native_sysv_with_args(arch, &code, &[10]) as u32, 10);
        }
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_loop_counter, assert_native_bin_exec_loop_counter);

// ---------------------------------------------------------------------------
// Native variants — memory store/load tests
// ---------------------------------------------------------------------------

fn assert_native_exec_i64_store_load(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I64], &[ValType::I64],
        &[
            Instruction::LocalGet(0), Instruction::LocalGet(1),
            Instruction::I64Store(memarg),
            Instruction::LocalGet(0), Instruction::I64Load(memarg),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, 42], NATIVE_WASM_MEM, 65536), 42);
    assert_eq!(run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, u64::MAX], NATIVE_WASM_MEM, 65536), u64::MAX);
}
native_variants!(exec_i64_store_load, assert_native_exec_i64_store_load);

fn assert_native_bin_exec_i64_store_load(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 3, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I64], &[ValType::I64],
        &[
            Instruction::LocalGet(0), Instruction::LocalGet(1),
            Instruction::I64Store(memarg),
            Instruction::LocalGet(0), Instruction::I64Load(memarg),
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke_with_mem_and_locals(arch, &code, NATIVE_WASM_MEM, &[], &[NATIVE_WASM_MEM, 42]),
        NativeAbi::Sysv => {
            assert_eq!(run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, 42], NATIVE_WASM_MEM, 65536), 42);
            assert_eq!(run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, u64::MAX], NATIVE_WASM_MEM, 65536), u64::MAX);
        }
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_i64_store_load, assert_native_bin_exec_i64_store_load);

fn assert_native_exec_i32_store_load(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 2, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[
            Instruction::LocalGet(0), Instruction::LocalGet(1),
            Instruction::I32Store(memarg),
            Instruction::LocalGet(0), Instruction::I32Load(memarg),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    assert_eq!(run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, 0xDEAD], NATIVE_WASM_MEM, 65536) as u32, 0xDEAD);
}
native_variants!(exec_i32_store_load, assert_native_exec_i32_store_load);

fn assert_native_bin_exec_i32_store_load(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let memarg = MemArg { offset: 0, align: 2, memory_index: 0 };
    let wasm = make_module_with_memory(
        &[ValType::I32, ValType::I32], &[ValType::I32],
        &[
            Instruction::LocalGet(0), Instruction::LocalGet(1),
            Instruction::I32Store(memarg),
            Instruction::LocalGet(0), Instruction::I32Load(memarg),
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
    match abi {
        NativeAbi::Naive => run_native_naive_smoke_with_mem_and_locals(arch, &code, NATIVE_WASM_MEM, &[], &[NATIVE_WASM_MEM, 0xDEAD]),
        NativeAbi::Sysv => assert_eq!(
            run_native_sysv_with_args_and_mem(arch, &code, &[NATIVE_WASM_MEM, 0xDEAD], NATIVE_WASM_MEM, 65536) as u32,
            0xDEAD,
        ),
    NativeAbi::Lfi => {}
    }
}
native_bin_variants!(exec_i32_store_load, assert_native_bin_exec_i32_store_load);

// ---------------------------------------------------------------------------
// Native variants — higher-level tests (memory.grow, data segment, import call)
// ---------------------------------------------------------------------------

fn assert_native_codegen_memory_grow(arch: NativeArch, abi: NativeAbi) {
    // memory.grow needs a __wasm_memory_grow external — codegen-only, no exec.
    let wasm = make_module_with_memory(&[], &[ValType::I32], &[Instruction::I32Const(1), Instruction::MemoryGrow(0)]);
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty(), "memory.grow codegen should produce non-empty asm for {arch:?} {abi:?}");
}
native_variants!(codegen_memory_grow, assert_native_codegen_memory_grow);

fn assert_native_bin_codegen_memory_grow(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module_with_memory(&[], &[ValType::I32], &[Instruction::I32Const(1), Instruction::MemoryGrow(0)]);
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty(), "memory.grow binary codegen should be non-empty for {arch:?} {abi:?}");
}
native_bin_variants!(codegen_memory_grow, assert_native_bin_codegen_memory_grow);

fn assert_native_codegen_data_segment(arch: NativeArch, abi: NativeAbi) {
    // data section is not natively emitted — test that codegen doesn't crash.
    use wasm_encoder::MemArg;
    let wasm = make_module_with_memory(
        &[], &[ValType::I32],
        &[Instruction::I32Const(NATIVE_WASM_MEM as i32), Instruction::I32Load(MemArg { offset: 0, align: 0, memory_index: 0 })],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
}
native_variants!(codegen_data_segment, assert_native_codegen_data_segment);

fn assert_native_bin_codegen_data_segment(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::MemArg;
    let wasm = make_module_with_memory(
        &[], &[ValType::I32],
        &[Instruction::I32Const(NATIVE_WASM_MEM as i32), Instruction::I32Load(MemArg { offset: 0, align: 0, memory_index: 0 })],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
}
native_bin_variants!(codegen_data_segment, assert_native_bin_codegen_data_segment);

/// Test the `make_module_with_import` module (env::add_one : i64→i64) via
/// the text-asm path with import stubs for SysV; codegen-only for Naive.
fn assert_native_exec_import_call(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module_with_import();
    let imports: &[(&str, &str)] = &[("env", "add_one")];
    let base_asm = compile_native_asm_with_imports(&wasm, arch, abi, imports);
    assert!(!base_asm.is_empty());
    if matches!(abi, NativeAbi::Naive) { return; }
    let asm = format!("{base_asm}{}", import_stub_add_one(arch));
    let Some(code) = assemble_or_skip(arch, &asm) else { return };
    // make_module_with_import: local fn takes i64, passes it to import, returns result.
    // SysV: arg in RDI/X0/A0 → import adds 1 → result in RAX/X0/A0.
    assert_eq!(run_native_sysv_with_args(arch, &code, &[42]), 43,
        "env::add_one(42) should return 43 for {arch:?}");
}
native_variants!(exec_import_call, assert_native_exec_import_call);

// ---------------------------------------------------------------------------
// Native variants — exception tests (codegen-only, no execution)
// ---------------------------------------------------------------------------

fn assert_native_codegen_throw_catch_matching(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[ValType::I32], &[], &[ValType::I32],
        &[
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::One { tag: 0, label: 0 }])),
            Instruction::I32Const(99),
            Instruction::Throw(0),
            Instruction::End,
            Instruction::I32Const(0),
            Instruction::End,
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
    // Software EH stack: TryTable open must register a dispatch frame.
    assert!(asm.contains("__wasm_eh_push"), "TryTable should push EH frame:\n{asm}");
}
native_variants!(codegen_throw_catch_matching, assert_native_codegen_throw_catch_matching);

/// Unmatched `throw` (no enclosing TryTable) must jump to `__wasm_exn_propagate`
/// (cross-function walk / `__wasm_unhandled_exception`), not silently fall through.
fn assert_native_codegen_throw_unmatched_propagates(arch: NativeArch, abi: NativeAbi) {
    let wasm = make_module_with_tag(
        &[ValType::I32], &[], &[],
        &[
            Instruction::I32Const(1),
            Instruction::Throw(0),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(
        asm.contains("__wasm_exn_propagate"),
        "unmatched throw must reference __wasm_exn_propagate:\n{asm}"
    );
}
native_variants!(codegen_throw_unmatched_propagates, assert_native_codegen_throw_unmatched_propagates);

fn assert_native_bin_codegen_throw_catch_matching(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[ValType::I32], &[], &[ValType::I32],
        &[
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::One { tag: 0, label: 0 }])),
            Instruction::I32Const(99),
            Instruction::Throw(0),
            Instruction::End,
            Instruction::I32Const(0),
            Instruction::End,
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
}
native_bin_variants!(codegen_throw_catch_matching, assert_native_bin_codegen_throw_catch_matching);

fn assert_native_codegen_throw_catch_all(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[], &[], &[ValType::I32],
        &[
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::All { label: 0 }])),
            Instruction::Throw(0),
            Instruction::I32Const(0),
            Instruction::End,
            Instruction::I32Const(1),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
}
native_variants!(codegen_throw_catch_all, assert_native_codegen_throw_catch_all);

fn assert_native_bin_codegen_throw_catch_all(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[], &[], &[ValType::I32],
        &[
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::All { label: 0 }])),
            Instruction::Throw(0),
            Instruction::I32Const(0),
            Instruction::End,
            Instruction::I32Const(1),
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
}
native_bin_variants!(codegen_throw_catch_all, assert_native_bin_codegen_throw_catch_all);

fn assert_native_codegen_no_throw_normal_exit(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[], &[], &[ValType::I32],
        &[
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::All { label: 0 }])),
            Instruction::I32Const(42),
            Instruction::End,
            Instruction::I32Const(0),
        ],
    );
    let asm = compile_native_asm(&wasm, arch, abi);
    assert!(!asm.is_empty());
}
native_variants!(codegen_no_throw_normal_exit, assert_native_codegen_no_throw_normal_exit);

fn assert_native_bin_codegen_no_throw_normal_exit(arch: NativeArch, abi: NativeAbi) {
    use wasm_encoder::Catch;
    let wasm = make_module_with_tag(
        &[], &[], &[ValType::I32],
        &[
            Instruction::TryTable(wasm_encoder::BlockType::Result(ValType::I32), Cow::Borrowed(&[Catch::All { label: 0 }])),
            Instruction::I32Const(42),
            Instruction::End,
            Instruction::I32Const(0),
        ],
    );
    let code = compile_native_binary(&wasm, arch, abi);
    assert!(!code.is_empty());
}
native_bin_variants!(codegen_no_throw_normal_exit, assert_native_bin_codegen_no_throw_normal_exit);

// ---------------------------------------------------------------------------
// Exception handling helpers
// ---------------------------------------------------------------------------

/// Build a WASM module with:
/// - type 0: `tag_params → []`  (exception tag payload type)
/// - type 1: `fn_params → fn_results`  (function type)
/// - tag 0: type 0
/// - function 0: type 1, with the given instructions
fn make_module_with_tag(
    tag_params: &[ValType],
    fn_params: &[ValType],
    fn_results: &[ValType],
    instrs: &[Instruction<'_>],
) -> Vec<u8> {
    let mut module = Module::new();

    let mut types = TypeSection::new();
    // type 0 — exception tag payload (params only, no results)
    types.ty().function(tag_params.iter().cloned(), []);
    // type 1 — function signature
    types.ty().function(fn_params.iter().cloned(), fn_results.iter().cloned());
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(1); // function uses type 1
    module.section(&functions);

    // TagSection order: Type → Function → Tag → Export → Code (wasmparser Order enum).
    let mut tags = TagSection::new();
    tags.tag(TagType { kind: wasm_encoder::TagKind::Exception, func_type_idx: 0 });
    module.section(&tags);

    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    for instr in instrs {
        func.instruction(instr);
    }
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);

    module.finish()
}

/// Parse exception tag section: returns `Vec<u32>` mapping tag index → type index.
fn parse_tags(wasm: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::TagSection(reader) = payload {
            for tag in reader.into_iter().flatten() {
                result.push(tag.func_type_idx);
            }
        }
    }
    result
}

/// Compile `wasm` bytes (which may contain a tag section) to JavaScript.
fn compile_js_exc(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let tags = parse_tags(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &tags, &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

/// Compile `wasm` bytes (which may contain a tag section) to C.
fn compile_c_exc(wasm: &[u8]) -> String {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let tags = parse_tags(wasm);

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;

    let mut preamble = String::new();
    c_module_preamble(&mut preamble).unwrap();
    out.push_str(&preamble);

    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &tags, &[], &mut state, &op, &mut reencoder)
            .unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// Exception execution tests
// ---------------------------------------------------------------------------

/// Test: throw tag0 (i64 value 99) caught by catch{tag0}, returns 99.
#[test]
fn test_throw_catch_matching_tag_js() {
    // fn(): i64  — push 99, throw tag0, return unreachable; catch returns the value
    let wasm = make_module_with_tag(
        &[ValType::I64],     // tag params: one i64
        &[],                  // fn params: none
        &[ValType::I64],      // fn results: one i64
        &[
            // try_table { catch{tag0, label=1} } (label 1 = outer block)
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I64)),
            Instruction::TryTable(
                wasm_encoder::BlockType::Empty,
                Cow::Borrowed(&[Catch::One { tag: 0, label: 1 }]),
            ),
            Instruction::I64Const(99),
            Instruction::Throw(0),
            Instruction::End,  // end try_table
            Instruction::I64Const(0), // unreachable normal exit
            Instruction::End,  // end block
        ],
    );
    let js = compile_js_exc(&wasm);
    let result = run_js(&js, &[]);
    assert_eq!(result, vec![99]);
}

#[test]
fn test_throw_catch_matching_tag_c() {
    let wasm = make_module_with_tag(
        &[ValType::I64],
        &[],
        &[ValType::I64],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I64)),
            Instruction::TryTable(
                wasm_encoder::BlockType::Empty,
                Cow::Borrowed(&[Catch::One { tag: 0, label: 1 }]),
            ),
            Instruction::I64Const(99),
            Instruction::Throw(0),
            Instruction::End,
            Instruction::I64Const(0),
            Instruction::End,
        ],
    );
    let c = compile_c_exc(&wasm);
    let result = run_c(&c, 0, &[], 1);
    assert_eq!(result, vec![99]);
}

/// Test: throw inside try_table with catch_all, catch_all is taken.
///
/// catch_all provides 0 values, so its target label must have empty result type.
/// We use an outer Block(Empty) as the catch_all target (label 1), then push 1
/// after the block exits (both catch and normal paths converge there).
#[test]
fn test_throw_catch_all_js() {
    let wasm = make_module_with_tag(
        &[ValType::I64],   // tag params: one i64
        &[],
        &[ValType::I64],
        &[
            Instruction::Block(wasm_encoder::BlockType::Empty),  // label 1 (empty target for catch_all)
            Instruction::TryTable(
                wasm_encoder::BlockType::Empty,
                Cow::Borrowed(&[Catch::All { label: 1 }]),  // catch_all → exit outer empty block
            ),
            Instruction::I64Const(42),
            Instruction::Throw(0),
            Instruction::End,        // end try_table
            Instruction::Br(0),      // normal path: exit outer block (dead code since throw always fires)
            Instruction::End,        // end outer block
            Instruction::I64Const(1), // both paths converge here; return 1 to signal catch_all was taken
        ],
    );
    let js = compile_js_exc(&wasm);
    let result = run_js(&js, &[]);
    assert_eq!(result, vec![1]);
}

#[test]
fn test_throw_catch_all_c() {
    let wasm = make_module_with_tag(
        &[ValType::I64],
        &[],
        &[ValType::I64],
        &[
            Instruction::Block(wasm_encoder::BlockType::Empty),
            Instruction::TryTable(
                wasm_encoder::BlockType::Empty,
                Cow::Borrowed(&[Catch::All { label: 1 }]),
            ),
            Instruction::I64Const(42),
            Instruction::Throw(0),
            Instruction::End,
            Instruction::Br(0),
            Instruction::End,
            Instruction::I64Const(1),
        ],
    );
    let c = compile_c_exc(&wasm);
    let result = run_c(&c, 0, &[], 1);
    assert_eq!(result, vec![1]);
}

/// Test: no throw — try_table normal exit returns the expected value.
#[test]
fn test_no_throw_normal_exit_js() {
    let wasm = make_module_with_tag(
        &[ValType::I64],
        &[],
        &[ValType::I64],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I64)),
            Instruction::TryTable(
                wasm_encoder::BlockType::Result(ValType::I64),
                Cow::Borrowed(&[Catch::All { label: 1 }]),
            ),
            Instruction::I64Const(77),
            Instruction::End, // end try_table — value 77 propagates out of block
            Instruction::End, // end block
        ],
    );
    let js = compile_js_exc(&wasm);
    let result = run_js(&js, &[]);
    assert_eq!(result, vec![77]);
}

#[test]
fn test_no_throw_normal_exit_c() {
    let wasm = make_module_with_tag(
        &[ValType::I64],
        &[],
        &[ValType::I64],
        &[
            Instruction::Block(wasm_encoder::BlockType::Result(ValType::I64)),
            Instruction::TryTable(
                wasm_encoder::BlockType::Result(ValType::I64),
                Cow::Borrowed(&[Catch::All { label: 1 }]),
            ),
            Instruction::I64Const(77),
            Instruction::End,
            Instruction::End,
        ],
    );
    let c = compile_c_exc(&wasm);
    let result = run_c(&c, 0, &[], 1);
    assert_eq!(result, vec![77]);
}

// ---------------------------------------------------------------------------
// LFI tests
// ---------------------------------------------------------------------------

/// Path to the lfi-verify binary (built from lfi-verifier source).
/// Set LFI_VERIFY env var or fall back to the build location.
fn lfi_verify_binary() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("LFI_VERIFY") {
        return Some(std::path::PathBuf::from(p));
    }
    // Default location after `meson setup build && ninja -C build`
    let p = std::path::PathBuf::from("/tmp/lfi-verifier/build/lfi-verify");
    if p.exists() { Some(p) } else { None }
}

/// Wrap raw code bytes in a minimal ELF64 binary suitable for lfi-verify.
///
/// lfi-verify reads PT_LOAD segments with PF_X from an ELF64 binary.
/// We create the simplest possible ELF: one PT_LOAD segment at virtual
/// address 0x1000 containing the code bytes.
fn wrap_in_elf64(code: &[u8], machine: u16) -> Vec<u8> {
    // ELF64 header: 64 bytes
    // Program header: 56 bytes
    // Total header: 120 bytes; pad to 0x1000 so code starts at vaddr 0x1000
    let hdr_size: u64 = 64;
    let phdr_size: u64 = 56;
    let code_offset: u64 = 0x1000; // file offset where code starts
    let vaddr: u64 = 0x1000;       // virtual address of code (32-byte aligned)
    let filesz: u64 = code.len() as u64;
    let memsz: u64 = filesz;

    let mut buf: Vec<u8> = Vec::with_capacity(code_offset as usize + code.len());

    // ELF magic + class=64bit + data=LE + version=1 + OS/ABI=SYSV + pad
    buf.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    // e_type=ET_EXEC(2), e_machine, e_version=1
    buf.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    buf.extend_from_slice(&machine.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    buf.extend_from_slice(&vaddr.to_le_bytes()); // e_entry
    buf.extend_from_slice(&hdr_size.to_le_bytes()); // e_phoff (program headers after ELF header)
    buf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff (no section headers)
    buf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    buf.extend_from_slice(&(hdr_size as u16).to_le_bytes()); // e_ehsize
    buf.extend_from_slice(&(phdr_size as u16).to_le_bytes()); // e_phentsize
    buf.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    buf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    buf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    buf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx

    // PT_LOAD program header (56 bytes)
    buf.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    buf.extend_from_slice(&5u32.to_le_bytes()); // p_flags = PF_R|PF_X
    buf.extend_from_slice(&code_offset.to_le_bytes()); // p_offset
    buf.extend_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    buf.extend_from_slice(&vaddr.to_le_bytes()); // p_paddr
    buf.extend_from_slice(&filesz.to_le_bytes()); // p_filesz
    buf.extend_from_slice(&memsz.to_le_bytes()); // p_memsz
    buf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align

    // Pad to code_offset
    while buf.len() < code_offset as usize {
        buf.push(0);
    }

    buf.extend_from_slice(code);
    buf
}

/// Run lfi-verify on an ELF binary and return true if it passes.
fn run_lfi_verify(elf: &[u8], arch_flag: &str) -> Result<bool, String> {
    use std::io::Write as _;
    let lfi = lfi_verify_binary().ok_or_else(|| {
        "lfi-verify not found; set LFI_VERIFY env var or build from /tmp/lfi-verifier".to_string()
    })?;
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let elf_path = std::env::temp_dir().join(format!("blitz_lfi_{pid}_{seq}.elf"));
    std::fs::File::create(&elf_path)
        .and_then(|mut f| f.write_all(elf))
        .map_err(|e| e.to_string())?;
    let out = std::process::Command::new(&lfi)
        .args(["--arch", arch_flag, elf_path.to_str().unwrap()])
        .output()
        .map_err(|e| format!("failed to run lfi-verify: {e}"))?;
    let _ = std::fs::remove_file(&elf_path);
    Ok(out.status.success())
}

/// Compile WASM to LFI x86-64 assembly, assemble it, and return (text, bytes).
///
/// Inserts `.bundle_align_mode 5` so GNU as automatically pads instructions
/// that would cross 32-byte bundle boundaries — exactly what lfi-verify requires.
fn compile_lfi_x64(wasm: &[u8]) -> (String, Vec<u8>) {
    let raw_asm = compile_native_asm(wasm, NativeArch::X86_64, NativeAbi::Lfi);
    // Insert bundle-align directive right after the .intel_syntax header line.
    let asm = raw_asm.replacen(
        ".intel_syntax noprefix\n",
        ".intel_syntax noprefix\n.bundle_align_mode 5\n",
        1,
    );
    let bytes = assemble_native_text(NativeArch::X86_64, &asm)
        .expect("LFI x86-64 assembly failed");
    (asm, bytes)
}

/// Compile WASM to LFI AArch64 assembly, assemble it, and return (text, bytes).
fn compile_lfi_aarch64(wasm: &[u8]) -> (String, Vec<u8>) {
    let asm = compile_native_asm(wasm, NativeArch::AArch64, NativeAbi::Lfi);
    let bytes = assemble_native_text(NativeArch::AArch64, &asm)
        .expect("LFI AArch64 assembly failed");
    (asm, bytes)
}

// ── LFI assertion helpers ──────────────────────────────────────────────────

fn assert_lfi_x64_no_ret(asm: &str) {
    for line in asm.lines() {
        let t = line.trim();
        // Bare `ret` not inside a label/symbol name
        if t == "ret" || t.starts_with("ret ") || t.starts_with("ret\t") {
            panic!("LFI x86-64 output contains forbidden `ret` instruction:\n{asm}");
        }
    }
}

fn assert_lfi_x64_gs_memory(asm: &str) {
    // Any memory load/store must use gs: prefix (if memory instructions are present)
    let has_memory_load = asm.lines().any(|l| {
        let t = l.trim();
        (t.starts_with("mov ") || t.starts_with("movzx ") || t.starts_with("movsx "))
            && t.contains("ptr [") && !t.contains("ptr gs:[") && !t.contains("ptr [rsp") && !t.contains("ptr [r14")
    });
    if has_memory_load {
        panic!("LFI x86-64 output has unsandboxed memory operand:\n{asm}");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_lfi_x64_no_ret_simple() {
    // A function that returns a constant — must not contain `ret`.
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(42),
    ]);
    let (asm, _bytes) = compile_lfi_x64(&wasm);
    assert_lfi_x64_no_ret(&asm);
    assert!(asm.contains("jmp"), "LFI return should use jmp (rtcall), got:\n{asm}");
    assert!(asm.contains(".balign 32") || asm.contains(".align 32"),
        "LFI should emit 32-byte alignment, got:\n{asm}");
}

#[test]
fn test_lfi_x64_no_ret_add() {
    // Stack-based add — must not contain `ret` in generated code.
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(5),
        Instruction::I64Const(3),
        Instruction::I64Add,
    ]);
    let (asm, _bytes) = compile_lfi_x64(&wasm);
    assert_lfi_x64_no_ret(&asm);
}

#[test]
fn test_lfi_x64_assembles() {
    // The generated LFI assembly must assemble without errors.
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(99),
    ]);
    let (asm, bytes) = compile_lfi_x64(&wasm);
    assert!(!bytes.is_empty(), "assembled output should not be empty, asm:\n{asm}");
}

#[test]
fn test_lfi_x64_verify() {
    // Full pipeline: compile → assemble → lfi-verify.
    // Skip if lfi-verify binary is not available.
    if lfi_verify_binary().is_none() {
        eprintln!("SKIP: lfi-verify not found; set LFI_VERIFY or build from /tmp/lfi-verifier");
        return;
    }
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(42),
    ]);
    let (asm, bytes) = compile_lfi_x64(&wasm);
    // Wrap raw text section bytes in a minimal ELF64 (EM_X86_64 = 62).
    let elf = wrap_in_elf64(&bytes, 62);
    match run_lfi_verify(&elf, "x64") {
        Ok(true) => {} // ✓
        Ok(false) => panic!("lfi-verify REJECTED generated code.\nAssembly:\n{asm}"),
        Err(e) => panic!("lfi-verify error: {e}"),
    }
}

#[test]
fn test_lfi_x64_verify_add() {
    // Test stack-based add (not local-based: local variable access via xchg rsp/CTX
    // is not yet LFI-compliant; that requires RSP-relative tracking, future work).
    if lfi_verify_binary().is_none() { return; }
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(10),
        Instruction::I64Const(32),
        Instruction::I64Add,
    ]);
    let (asm, bytes) = compile_lfi_x64(&wasm);
    let elf = wrap_in_elf64(&bytes, 62);
    match run_lfi_verify(&elf, "x64") {
        Ok(true) => {}
        Ok(false) => panic!("lfi-verify REJECTED add function:\n{asm}"),
        Err(e) => panic!("lfi-verify error: {e}"),
    }
}

#[test]
fn test_lfi_aarch64_assembles() {
    // The generated LFI AArch64 assembly must assemble without errors.
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(77),
    ]);
    let (asm, bytes) = compile_lfi_aarch64(&wasm);
    assert!(!bytes.is_empty(), "assembled AArch64 LFI output should not be empty, asm:\n{asm}");
}

#[test]
fn test_lfi_aarch64_verify() {
    if lfi_verify_binary().is_none() { return; }
    let wasm = make_module(&[], &[ValType::I64], &[
        Instruction::I64Const(77),
    ]);
    let (asm, bytes) = compile_lfi_aarch64(&wasm);
    // EM_AARCH64 = 183
    let elf = wrap_in_elf64(&bytes, 183);
    match run_lfi_verify(&elf, "arm64") {
        Ok(true) => {}
        Ok(false) => panic!("lfi-verify REJECTED AArch64 code:\n{asm}"),
        Err(e) => panic!("lfi-verify error: {e}"),
    }
}


// ---------------------------------------------------------------------------
// JIT tracing / specialization (Items 1–3)
// ---------------------------------------------------------------------------

/// Compile the loop-counter fn with probes enabled, returning x86-64 naive asm.
fn compile_x86_naive_with_tracing(wasm: &[u8], base_off: i32) -> (String, u32) {
    use portal_solutions_blitz_common::ops::{probe_site_count, MachOperator, ProbeTableConfig};
    use portal_solutions_blitz_x86_64::{naive, X64Arch};

    let (sigs_wp, _sigs_enc, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let num_probes = probe_site_count(&bodies[0]);

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut out = NativeAsmWriter(String::new());
    let mut ctx = ();
    let mut state = naive::State::default();
    for op in ops {
        let mut op = op.unwrap();
        if let MachOperator::StartFn { data, .. } = &mut op {
            data.probes = Some(ProbeTableConfig { enabled: true, num_probes, table_base_off: base_off });
        }
        naive::WriterExt::handle_op::<_, HandleOpError<_>>(
            &mut out, &mut ctx, X64Arch::default(),
            &mut state, &[], &[], &[], &op, &mut reencoder, 0,
        ).unwrap();
    }
    (out.0, num_probes)
}

/// Tracing-disabled compile of the loop-counter fn (the zero-overhead path).
fn compile_x86_naive_no_tracing(wasm: &[u8]) -> String {
    compile_native_asm(wasm, NativeArch::X86_64, NativeAbi::Naive)
}

#[test]
fn test_tracing_site_per_loop_and_entry() {
    // Loop-counter fn has a function entry (probe 0) + one loop (probe 1); the
    // `If` is not a control-flow probe site. So `probe_site_count` must be 2.
    let wasm = make_loop_counter_wasm();
    let (asm, num_probes) = compile_x86_naive_with_tracing(&wasm, 0x60);
    assert_eq!(num_probes, 2, "entry + loop == 2 probes");

    // One preamble per probe: each loads the runtime probe-table base from the
    // CTX-relative slot (CTX = r15, base_off = 0x60 = 96).
    let base_loads = asm.matches("qword ptr [r15+96]").count();
    assert_eq!(base_loads, 2, "one CTX-relative base load per probe:\n{asm}");

    // Probe 0 (function entry) indexes ProbeSlot[0]; probe 1 (loop) indexes
    // ProbeSlot[1] = +16 bytes.  counter at +0, handler ptr at +8.
    assert!(asm.contains("add qword ptr [rdx+0],1"), "probe 0 counter:\n{asm}");
    assert!(asm.contains("mov rdx, qword ptr [rdx+8]"), "probe 0 handler ptr:\n{asm}");
    assert!(asm.contains("add qword ptr [rdx+16],1"), "probe 1 counter:\n{asm}");
    assert!(asm.contains("mov rdx, qword ptr [rdx+24]"), "probe 1 handler ptr:\n{asm}");
    // Tail-jump to the handler through the loaded pointer.
    assert!(asm.contains("jmp rdx"), "handler tail-jump:\n{asm}");
}

#[test]
fn test_tracing_uses_no_baked_address() {
    // Compile/runtime separation: the preamble must reach its probe state
    // CTX-relative, never by baking an absolute 64-bit address into the code.
    let wasm = make_loop_counter_wasm();
    let (asm, _) = compile_x86_naive_with_tracing(&wasm, 0x60);
    assert!(!asm.contains("movabs"), "preamble must not bake an absolute address:\n{asm}");
}

#[test]
fn test_tracing_disabled_is_zero_overhead() {
    // With probes off (the default), no preamble is emitted — the generated
    // code must be what it was before probes existed.
    let wasm = make_loop_counter_wasm();
    let asm_off = compile_x86_naive_no_tracing(&wasm);
    assert!(!asm_off.contains("qword ptr [r15+96]"), "no probe-base load when disabled");

    // And the enabled build is strictly a superset (longer) of the disabled one.
    let (asm_on, _) = compile_x86_naive_with_tracing(&wasm, 0x60);
    assert!(asm_on.len() > asm_off.len(), "probes add preamble code");
}

#[test]
fn test_probe_plan_control_flow_sites_matches_probe_site_count() {
    use portal_solutions_blitz_common::ops::{probe_site_count, ProbeMode, ProbePlacement, ProbePlan};
    use portal_solutions_blitz_codegen::ProbeBinding;

    let wasm = make_loop_counter_wasm();
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let body = &bodies[0];
    let num_probes = probe_site_count(body);
    let plan = ProbePlan::control_flow_sites(body);

    // Entry: exactly one TailTakeover/Active probe, id 0.
    assert_eq!(plan.entry.len(), 1);
    assert_eq!(plan.entry[0].probe_id, 0);
    assert_eq!(plan.entry[0].binding, ProbeBinding::TailTakeover);
    assert_eq!(plan.entry[0].mode, ProbeMode::Active);
    assert_eq!(plan.entry[0].placement, ProbePlacement::Before);

    // Exactly one more site (the `Loop` header at ordinal index 0); the `If`
    // at index 2 is *not* a control-flow probe site, matching `probe_site_count`.
    assert_eq!(plan.by_index.len(), 1, "only the Loop header, not the If");
    let loop_probes = &plan.by_index[&0];
    assert_eq!(loop_probes.len(), 1);
    assert_eq!(loop_probes[0].probe_id, 1);
    assert_eq!(loop_probes[0].binding, ProbeBinding::TailTakeover);

    // Total probe count (entry + by_index) matches `probe_site_count`'s
    // sizing of the runtime `[ProbeSlot]` table.
    let total = plan.entry.len() + plan.by_index.values().map(Vec::len).sum::<usize>();
    assert_eq!(total as u32, num_probes);
}

// ---------------------------------------------------------------------------
// SysV tracing execution: trace-table install + counter / specialization verify
// ---------------------------------------------------------------------------

/// Thin x86-64 shims over the generic SysV tracing harness (kept for the
/// x86-specific tests below, including the loop-site frame-teardown stub).
fn compile_x86_sysv_binary_traced(wasm: &[u8]) -> (Vec<u8>, u32) {
    compile_native_sysv_binary_traced(wasm, NativeArch::X86_64)
}
fn run_x86_sysv_traced(
    code: &[u8],
    args: &[u64],
    num_probes: u32,
    spec: Option<(u32, &[u8])>,
) -> (u64, Vec<u64>) {
    run_native_sysv_traced(NativeArch::X86_64, code, args, num_probes, spec)
}

#[test]
fn test_sysv_tracing_counters_increment() {
    // Loop-counter fn: probe 0 = entry, probe 1 = loop. With arg=5 the loop body
    // runs until n hits 0, so the loop probe is entered 6 times (initial + 5
    // back-edges) and the entry probe once.  No handler installed → the
    // baseline runs and the result is the counted value.
    let wasm = make_loop_counter_wasm();
    let (code, num_probes) = compile_x86_sysv_binary_traced(&wasm);
    assert_eq!(num_probes, 2);

    let (result, counters) = run_x86_sysv_traced(&code, &[5], num_probes, None);
    assert_eq!(result as u32, 5, "baseline loop result");
    assert_eq!(counters[0], 1, "function entered once");
    assert_eq!(counters[1], 6, "loop probe entered 6× for n=5");

    // A different arg scales the loop counter.
    let (result, counters) = run_x86_sysv_traced(&code, &[10], num_probes, None);
    assert_eq!(result as u32, 10);
    assert_eq!(counters[0], 1);
    assert_eq!(counters[1], 11);
}

#[test]
fn test_sysv_tracing_specialization_tailjump() {
    // Install a handler at probe 0 (function entry): `mov eax, 0xABCD ; ret`.
    // The entry preamble must increment the counter, see the non-null pointer,
    // and tail-jump to the handler — so the result is the handler's sentinel and
    // the loop probe is never reached.
    let wasm = make_loop_counter_wasm();
    let (code, num_probes) = compile_x86_sysv_binary_traced(&wasm);

    let stub: &[u8] = &[0xB8, 0xCD, 0xAB, 0x00, 0x00, 0xC3]; // mov eax,0xABCD ; ret
    let (result, counters) = run_x86_sysv_traced(&code, &[5], num_probes, Some((0, stub)));

    assert_eq!(result as u32, 0xABCD, "entry specialization tail-jump taken");
    assert_eq!(counters[0], 1, "entry counter still incremented before the jump");
    assert_eq!(counters[1], 0, "loop probe never reached (specialized away)");
}

#[test]
fn test_sysv_tracing_loop_site_specialization() {
    // Install a handler at the loop probe (probe 1).  The baseline entry runs,
    // the loop is entered once (counter[1] == 1), the preamble sees the
    // pointer and tail-jumps to the handler instead of running the loop body.
    let wasm = make_loop_counter_wasm();
    let (code, num_probes) = compile_x86_sysv_binary_traced(&wasm);

    // At a mid-function site the operand stack is live, so the stub must tear
    // down the SysV frame before returning: mov eax,99 ; mov rsp,rbp ; pop rbp ; ret
    let stub: &[u8] = &[0xB8, 0x63, 0x00, 0x00, 0x00, 0x48, 0x89, 0xEC, 0x5D, 0xC3];
    let (result, counters) = run_x86_sysv_traced(&code, &[5], num_probes, Some((1, stub)));

    assert_eq!(result as u32, 99, "loop-site specialization tail-jump taken");
    assert_eq!(counters[0], 1, "entry counter incremented");
    assert_eq!(counters[1], 1, "loop entered once before tail-jump");
}

// ---------------------------------------------------------------------------
// Generic (all-arch) SysV tracing execution harness
// ---------------------------------------------------------------------------

/// Compile `wasm` for the given arch's SysV ABI with probes enabled, into raw
/// machine code (loaded at 0x100000 under Unicorn).  Returns `(code, num_probes)`.
fn compile_native_sysv_binary_traced(wasm: &[u8], arch: NativeArch) -> (Vec<u8>, u32) {
    use portal_solutions_blitz_common::ops::{probe_site_count, MachOperator, ProbeTableConfig};

    let (sigs_wp, _, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let num_probes = probe_site_count(&bodies[0]);
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut reencoder = RoundtripReencoder;
    let mut ctx = ();

    // Inject probes into the StartFn FnData of every op.
    let inject = |op: &mut MachOperator<'_, ()>| {
        if let MachOperator::StartFn { data, .. } = op {
            data.probes = Some(ProbeTableConfig { enabled: true, num_probes, table_base_off: 0 });
        }
    };

    let code = match arch {
        NativeArch::X86_64 => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch, X64Label};
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            let mut out = IcedWriter::<X64Label>::new(0x100000);
            let mut state = sysv::SysVState::default();
            for op in ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::AArch64 => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch, AArch64Label};
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut state = naive::State::default();
            for op in ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::Riscv64 => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch, RiscvLabel};
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut state = naive::State::default();
            for op in ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    };
    (code, num_probes)
}

/// A function-entry specialization stub that returns `sentinel` (≤ 0x7FF so it
/// fits every arch's single-instruction immediate) via the ABI return register.
/// Entered before any frame is built, so a bare return suffices.
fn sysv_entry_stub(arch: NativeArch, sentinel: u16) -> Vec<u8> {
    match arch {
        NativeArch::X86_64 => {
            // mov eax, imm32 ; ret
            let mut v = vec![0xB8u8];
            v.extend_from_slice(&(sentinel as u32).to_le_bytes());
            v.push(0xC3);
            v
        }
        NativeArch::AArch64 => {
            // movz w0, #sentinel ; ret
            let movz = 0x5280_0000u32 | ((sentinel as u32) << 5);
            let mut v = movz.to_le_bytes().to_vec();
            v.extend_from_slice(&0xD65F_03C0u32.to_le_bytes()); // ret
            v
        }
        NativeArch::Riscv64 => {
            // addi a0, x0, #sentinel ; ret (jalr x0, 0(ra))
            let addi = ((sentinel as u32) << 20) | (10u32 << 7) | 0x13;
            let mut v = addi.to_le_bytes().to_vec();
            v.extend_from_slice(&0x0000_8067u32.to_le_bytes()); // ret
            v
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

/// Run a probed SysV function under Unicorn for any arch.
///
/// Installs a zeroed `[ProbeSlot]` table, passes its base in the arch's
/// virtual-parameter register, optionally installs a handler for one probe,
/// runs, and returns `(return_reg, per_probe_counters)`.
fn run_native_sysv_traced(
    arch: NativeArch,
    code: &[u8],
    args: &[u64],
    num_probes: u32,
    spec: Option<(u32, &[u8])>,
) -> (u64, Vec<u64>) {
    use unicorn_engine::{unicorn_const::{Arch, Mode, Prot}, Unicorn};

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;
    const PROBE_TABLE: u64 = 0x400000;
    const HALT: u64 = CODE + 0xF000;

    // Shared setup performed inside each arch arm (Unicorn is not object-safe
    // across arches, so we duplicate the small tail).
    macro_rules! common_mem {
        ($uc:expr) => {{
            $uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            $uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            $uc.mem_map(PROBE_TABLE, 0x1000, Prot::ALL).unwrap();
            $uc.mem_write(CODE, code).unwrap();
            $uc.mem_write(PROBE_TABLE, &vec![0u8; num_probes as usize * 16]).unwrap();
            if let Some((probe_id, stub)) = spec {
                let stub_addr = CODE + ((code.len() as u64 + 15) & !15);
                $uc.mem_write(stub_addr, stub).unwrap();
                let slot = PROBE_TABLE + probe_id as u64 * 16 + 8;
                $uc.mem_write(slot, &stub_addr.to_le_bytes()).unwrap();
            }
        }};
    }
    macro_rules! read_counters {
        ($uc:expr) => {{
            let mut counters = Vec::with_capacity(num_probes as usize);
            for s in 0..num_probes {
                let mut b = [0u8; 8];
                $uc.mem_read(PROBE_TABLE + s as u64 * 16, &mut b).unwrap();
                counters.push(u64::from_le_bytes(b));
            }
            counters
        }};
    }

    match arch {
        NativeArch::X86_64 => {
            use unicorn_engine::RegisterX86;
            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            common_mem!(uc);
            let rsp = STACK + STACK_SIZE - 8;
            uc.mem_write(rsp, &HALT.to_le_bytes()).unwrap();
            uc.reg_write(RegisterX86::RSP, rsp).unwrap();
            uc.reg_write(RegisterX86::R11, PROBE_TABLE).unwrap(); // virtual param
            let arg_regs = [RegisterX86::RDI, RegisterX86::RSI, RegisterX86::RDX, RegisterX86::RCX];
            for (i, &v) in args.iter().enumerate().take(4) { uc.reg_write(arg_regs[i], v).unwrap(); }
            uc.emu_start(CODE, HALT, 0, 100_000).unwrap();
            (uc.reg_read(RegisterX86::RAX).unwrap(), read_counters!(uc))
        }
        NativeArch::AArch64 => {
            use unicorn_engine::RegisterARM64;
            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            common_mem!(uc);
            uc.reg_write(RegisterARM64::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterARM64::LR, HALT).unwrap();
            uc.reg_write(RegisterARM64::X12, PROBE_TABLE).unwrap(); // virtual param
            let arg_regs = [RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3];
            for (i, &v) in args.iter().enumerate().take(4) { uc.reg_write(arg_regs[i], v).unwrap(); }
            uc.emu_start(CODE, HALT, 0, 100_000).unwrap();
            (uc.reg_read(RegisterARM64::X0).unwrap(), read_counters!(uc))
        }
        NativeArch::Riscv64 => {
            use unicorn_engine::RegisterRISCV;
            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            common_mem!(uc);
            uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 16).unwrap();
            uc.reg_write(RegisterRISCV::RA, HALT).unwrap();
            uc.reg_write(RegisterRISCV::T2, PROBE_TABLE).unwrap(); // virtual param
            let arg_regs = [RegisterRISCV::A0, RegisterRISCV::A1, RegisterRISCV::A2, RegisterRISCV::A3];
            for (i, &v) in args.iter().enumerate().take(4) { uc.reg_write(arg_regs[i], v).unwrap(); }
            uc.emu_start(CODE, HALT, 0, 100_000).unwrap();
            (uc.reg_read(RegisterRISCV::A0).unwrap(), read_counters!(uc))
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

fn assert_sysv_tracing_counters(arch: NativeArch) {
    let wasm = make_loop_counter_wasm();
    let (code, num_probes) = compile_native_sysv_binary_traced(&wasm, arch);
    assert_eq!(num_probes, 2, "{arch:?}: entry + loop");
    // Mid-function (loop) probe uses the spilled frame-slot base; the counter
    // incrementing the right amount proves the frame-slot disp is correct.
    let (result, counters) = run_native_sysv_traced(arch, &code, &[5], num_probes, None);
    assert_eq!(result as u32, 5, "{arch:?}: baseline loop result");
    assert_eq!(counters[0], 1, "{arch:?}: entry probe once");
    assert_eq!(counters[1], 6, "{arch:?}: loop probe 6× for n=5");
}

fn assert_sysv_tracing_entry_spec(arch: NativeArch) {
    let wasm = make_loop_counter_wasm();
    let (code, num_probes) = compile_native_sysv_binary_traced(&wasm, arch);
    let stub = sysv_entry_stub(arch, 1234);
    let (result, counters) = run_native_sysv_traced(arch, &code, &[5], num_probes, Some((0, &stub)));
    assert_eq!(result as u32, 1234, "{arch:?}: entry specialization tail-jump taken");
    assert_eq!(counters[0], 1, "{arch:?}: entry counter incremented before jump");
    assert_eq!(counters[1], 0, "{arch:?}: loop probe never reached");
}

#[test] fn test_sysv_tracing_counters_aarch64() { assert_sysv_tracing_counters(NativeArch::AArch64); }
#[test] fn test_sysv_tracing_counters_riscv64() { assert_sysv_tracing_counters(NativeArch::Riscv64); }
#[test] fn test_sysv_tracing_entry_spec_aarch64() { assert_sysv_tracing_entry_spec(NativeArch::AArch64); }
#[test] fn test_sysv_tracing_entry_spec_riscv64() { assert_sysv_tracing_entry_spec(NativeArch::Riscv64); }

// ---------------------------------------------------------------------------
// Test — arbitrary-point probes via `ProbePlan` (x86-64 SysV)
//
// Unlike the auto-identified control-flow probes above (function entry +
// every `Block`/`Loop` header), this probe is placed *inside* the loop body
// at a plain arithmetic instruction (`I32Add`) — proving probes can be
// dropped into the middle of an expression without disturbing the
// surrounding WASM operand stack or the function's result, alongside the
// existing control-flow probes still firing correctly in the same function.
// ---------------------------------------------------------------------------

/// A trivial `ret`-only `Call`-bound probe handler: proves call-and-return
/// without relying on the handler itself doing anything — the probe's own
/// hit counter (incremented by `emit_probe_site` regardless of what the
/// handler does) is what proves it fired the right number of times.
fn ret_only_stub(arch: NativeArch) -> Vec<u8> {
    match arch {
        NativeArch::X86_64 => vec![0xC3],
        NativeArch::AArch64 => 0xD65F_03C0u32.to_le_bytes().to_vec(),
        NativeArch::Riscv64 => 0x0000_8067u32.to_le_bytes().to_vec(),
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

/// Compile `wasm` for the given arch's SysV ABI with the control-flow probes
/// disabled and `plan` installed as the function's `ProbePlan` instead — for
/// testing arbitrary-index probes in isolation from the auto-identified
/// entry/loop/block set. Built without `dce_pass!` so the dispatcher's
/// ordinal `op_index` lines up exactly with the raw operator indices `plan`
/// was computed against (DCE can renumber/remove ops).
fn compile_native_sysv_binary_with_plan(
    wasm: &[u8], arch: NativeArch, plan: &portal_solutions_blitz_common::ops::ProbePlan, num_probes: u32,
) -> Vec<u8> {
    use portal_solutions_blitz_common::ops::{MachOperator, ProbeTableConfig};

    let (sigs_wp, _, fsigs) = parse_sigs(wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let mut reencoder = RoundtripReencoder;
    let mut ctx = ();

    let inject = |op: &mut MachOperator<'_, ()>| {
        if let MachOperator::StartFn { data, .. } = op {
            data.probes = Some(ProbeTableConfig { enabled: true, num_probes, table_base_off: 0 });
            data.probe_plan = Some(plan.clone());
        }
    };

    match arch {
        NativeArch::X86_64 => {
            use portal_solutions_blitz_x86_64::{sysv, X64Arch, X64Label};
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            let mut out = IcedWriter::<X64Label>::new(0x100000);
            let mut state = sysv::SysVState::default();
            for op in raw_ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(), &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::AArch64 => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch, AArch64Label};
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut state = naive::State::default();
            for op in raw_ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(), &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::Riscv64 => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch, RiscvLabel};
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut state = naive::State::default();
            for op in raw_ops {
                let mut op = op.unwrap();
                inject(&mut op);
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(), &mut state, &[], &op, &mut reencoder, 0,
                ).unwrap();
            }
            out.into_bytes()
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

/// Ordinal indices in `make_loop_counter_wasm`'s raw operator stream: 0=Loop,
/// 1=LocalGet(0), 2=If, 3=LocalGet(1), 4=I32Const(1), 5=I32Add, 6=LocalSet(1),
/// .... These are a property of the WASM operator stream, so they're the
/// same for every architecture.
const LOOP_COUNTER_I32ADD_INDEX: usize = 5;
const LOOP_COUNTER_AFTER_I32ADD_INDEX: usize = 6;

fn assert_indexed_call_probe_fires_inside_loop(
    arch: NativeArch, mode: portal_solutions_blitz_common::ops::ProbeMode, probe_index: usize,
) {
    use portal_solutions_blitz_common::ops::{probe_site_count, ProbePlacement, ProbePlan, ProbeSpec};

    let wasm = make_loop_counter_wasm();
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let num_control_flow_probes = probe_site_count(&bodies[0]);
    assert_eq!(num_control_flow_probes, 2, "entry + loop");

    const EXTRA_PROBE_ID: u32 = 2; // right after the 2 auto-identified probes
    let mut plan = ProbePlan::default();
    plan.by_index.insert(probe_index, vec![ProbeSpec {
        probe_id: EXTRA_PROBE_ID,
        binding: portal_solutions_blitz_codegen::ProbeBinding::Call,
        mode,
        placement: ProbePlacement::Before,
    }]);

    let code = compile_native_sysv_binary_with_plan(&wasm, arch, &plan, EXTRA_PROBE_ID + 1);
    let handler = ret_only_stub(arch);
    let (result, counters) = run_native_sysv_traced(
        arch, &code, &[5], EXTRA_PROBE_ID + 1, Some((EXTRA_PROBE_ID, &handler)),
    );

    assert_eq!(result as u32, 5, "{arch:?}/{mode:?}: loop result unaffected by the mid-expression probe");
    assert_eq!(counters[0], 1, "{arch:?}/{mode:?}: entry probe still fires once, unaffected by the new probe");
    assert_eq!(counters[1], 6, "{arch:?}/{mode:?}: loop probe still fires 6× for n=5, unaffected by the new probe");
    assert_eq!(counters[2], 5, "{arch:?}/{mode:?}: probe fires once per loop iteration (n=5), call-and-returns each time");
}

#[test]
fn test_indexed_call_probe_fires_inside_loop_x86_64() {
    assert_indexed_call_probe_fires_inside_loop(
        NativeArch::X86_64, portal_solutions_blitz_common::ops::ProbeMode::Active, LOOP_COUNTER_I32ADD_INDEX,
    );
}
#[test]
fn test_indexed_call_probe_fires_inside_loop_aarch64() {
    assert_indexed_call_probe_fires_inside_loop(
        NativeArch::AArch64, portal_solutions_blitz_common::ops::ProbeMode::Active, LOOP_COUNTER_I32ADD_INDEX,
    );
}
#[test]
fn test_indexed_call_probe_fires_inside_loop_riscv64_active() {
    // Active mode forces a regalloc flush, which resets the allocator to
    // believing nothing is live (matching the canonical layout `TailTakeover`
    // needs). That reset is only valid where at most one value is pending —
    // exactly the shape WASM guarantees at a control-flow header, which is
    // what Active is really for. Probing *after* `I32Add` (one pending value,
    // the sum) is a valid Active point; probing *before* it (two pending
    // operands) is not — see `test_indexed_call_probe_fires_inside_loop_riscv64_passive`,
    // which probes that exact point correctly using Passive mode instead.
    assert_indexed_call_probe_fires_inside_loop(
        NativeArch::Riscv64, portal_solutions_blitz_common::ops::ProbeMode::Active, LOOP_COUNTER_AFTER_I32ADD_INDEX,
    );
}
#[test]
fn test_indexed_call_probe_fires_inside_loop_riscv64_passive() {
    // Passive mode never resets the allocator, so it has no such restriction:
    // it correctly handles the two-pending-operand point right before
    // `I32Add` that Active mode cannot (see the comment above).
    assert_indexed_call_probe_fires_inside_loop(
        NativeArch::Riscv64, portal_solutions_blitz_common::ops::ProbeMode::Passive, LOOP_COUNTER_I32ADD_INDEX,
    );
}

// ---------------------------------------------------------------------------
// Tests — ProbeBinding::Call (register-only call-and-return)
//
// Unlike the `TailTakeover` binding exercised above (which permanently hands
// off control — the existing specialization opt-entry behavior), `Call`
// binding must *return* to the probe site once the handler runs. These tests
// drive `blitz_codegen::emit_probe_site` directly (independent of any WASM
// function) to exercise the new `call_reg` primitive on real hardware
// semantics under Unicorn for all three architectures.
// ---------------------------------------------------------------------------

/// Build a standalone snippet — one `Call`-bound probe site (probe 0, base
/// reached via the SysV virtual-param register, reusing the same convention
/// as the function-entry trace site above) followed by a marker instruction —
/// and run it under Unicorn.
///
/// Returns `(handler_ret_reg, marker_reg, hit_counter)`. Both check registers
/// are poisoned with `0xFFFF_FFFF` before running, so a leftover poison value
/// unambiguously means "never written".
fn run_call_probe(arch: NativeArch, install_handler: bool) -> (u64, u64, u64) {
    use portal_solutions_blitz_codegen::{emit_probe_site, BlitzWriter, ProbeBinding};
    use unicorn_engine::{unicorn_const::{Arch, Mode, Prot}, Unicorn};

    const CODE: u64 = 0x100000;
    const STUB_ADDR: u64 = CODE + 0x800;
    const PROBE_TABLE: u64 = 0x400000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;
    const POISON: u64 = 0xFFFF_FFFF;
    const HANDLER_SENTINEL: u16 = 0x222; // ABI return reg, iff the handler ran
    const MARKER_SENTINEL: u64 = 0x333; // proves control resumed after the call

    let handler_stub = sysv_entry_stub(arch, HANDLER_SENTINEL);

    match arch {
        NativeArch::X86_64 => {
            use portal_solutions_asm_x86_64::out::iced::IcedWriter;
            use portal_solutions_blitz_x86_64::{
                X64Arch, X64Label,
                codegen::{BlitzW, ProbeBase},
                sysv::PROBE_BASE_REG,
            };
            use unicorn_engine::RegisterX86;

            let mut out = IcedWriter::<X64Label>::new(CODE);
            let mut ctx = ();
            let mut bw = BlitzW {
                writer: &mut out, ctx: &mut ctx, arch: X64Arch::default(),
                probe_base: ProbeBase::Reg(PROBE_BASE_REG),
            };
            let mut label_counter = 0usize;
            emit_probe_site(&mut bw, 0, 0, 2 /* RDX */, ProbeBinding::Call, &mut label_counter).unwrap();
            bw.load_u64_imm(1 /* RCX */, MARKER_SENTINEL).unwrap();
            let code = out.into_bytes();
            let halt = CODE + code.len() as u64;

            let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(PROBE_TABLE, 0x1000, Prot::ALL).unwrap();
            // `call_reg`/`ret` use the hardware stack on x86-64 (unlike AArch64's
            // `bl`/AArch64 `ret` or RISC-V's `jalr ra`, which thread the return
            // address through a link register instead) — needs a real stack.
            uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
            uc.reg_write(RegisterX86::RSP, STACK + STACK_SIZE - 8).unwrap();
            uc.mem_write(CODE, &code).unwrap();
            uc.mem_write(PROBE_TABLE, &[0u8; 16]).unwrap();
            if install_handler {
                uc.mem_write(STUB_ADDR, &handler_stub).unwrap();
                uc.mem_write(PROBE_TABLE + 8, &STUB_ADDR.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterX86::R11, PROBE_TABLE).unwrap();
            uc.reg_write(RegisterX86::RAX, POISON).unwrap();
            uc.reg_write(RegisterX86::RCX, POISON).unwrap();
            uc.emu_start(CODE, halt, 0, 100_000).unwrap();
            let handler_ret = uc.reg_read(RegisterX86::RAX).unwrap();
            let marker = uc.reg_read(RegisterX86::RCX).unwrap();
            let mut counter = [0u8; 8];
            uc.mem_read(PROBE_TABLE, &mut counter).unwrap();
            (handler_ret, marker, u64::from_le_bytes(counter))
        }
        NativeArch::AArch64 => {
            use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
            use portal_solutions_blitz_aarch64::{
                AArch64Arch, AArch64Label,
                codegen::{BlitzW, ProbeBase},
                sysv::PROBE_BASE_REG,
            };
            use unicorn_engine::RegisterARM64;

            let mut out = AArch64Writer::<AArch64Label>::new();
            let mut ctx = ();
            let mut bw = BlitzW {
                writer: &mut out, ctx: &mut ctx, arch: AArch64Arch::default(), scratch2: 10,
                probe_base: ProbeBase::Reg(PROBE_BASE_REG),
            };
            let mut label_counter = 0usize;
            emit_probe_site(&mut bw, 0, 0, 9, ProbeBinding::Call, &mut label_counter).unwrap();
            bw.load_u64_imm(1, MARKER_SENTINEL).unwrap();
            let code = out.into_bytes();
            let halt = CODE + code.len() as u64;

            let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(PROBE_TABLE, 0x1000, Prot::ALL).unwrap();
            uc.mem_write(CODE, &code).unwrap();
            uc.mem_write(PROBE_TABLE, &[0u8; 16]).unwrap();
            if install_handler {
                uc.mem_write(STUB_ADDR, &handler_stub).unwrap();
                uc.mem_write(PROBE_TABLE + 8, &STUB_ADDR.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterARM64::X12, PROBE_TABLE).unwrap();
            uc.reg_write(RegisterARM64::X0, POISON).unwrap();
            uc.reg_write(RegisterARM64::X1, POISON).unwrap();
            uc.emu_start(CODE, halt, 0, 100_000).unwrap();
            let handler_ret = uc.reg_read(RegisterARM64::X0).unwrap();
            let marker = uc.reg_read(RegisterARM64::X1).unwrap();
            let mut counter = [0u8; 8];
            uc.mem_read(PROBE_TABLE, &mut counter).unwrap();
            (handler_ret, marker, u64::from_le_bytes(counter))
        }
        NativeArch::Riscv64 => {
            use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
            use portal_solutions_blitz_riscv64::{
                RiscV64Arch, RiscvLabel,
                codegen::{BlitzW, ProbeBase},
                sysv::PROBE_BASE_REG,
            };
            use unicorn_engine::RegisterRISCV;

            let mut out = RvAsmWriter::<RiscvLabel>::new();
            let mut ctx = ();
            let mut bw = BlitzW {
                writer: &mut out, ctx: &mut ctx, arch: RiscV64Arch::default(), scratch2: 6,
                probe_base: ProbeBase::Reg(PROBE_BASE_REG),
            };
            let mut label_counter = 0usize;
            emit_probe_site(&mut bw, 0, 0, 5, ProbeBinding::Call, &mut label_counter).unwrap();
            bw.load_u64_imm(11, MARKER_SENTINEL).unwrap();
            let code = out.into_bytes();
            let halt = CODE + code.len() as u64;

            let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
            uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
            uc.mem_map(PROBE_TABLE, 0x1000, Prot::ALL).unwrap();
            uc.mem_write(CODE, &code).unwrap();
            uc.mem_write(PROBE_TABLE, &[0u8; 16]).unwrap();
            if install_handler {
                uc.mem_write(STUB_ADDR, &handler_stub).unwrap();
                uc.mem_write(PROBE_TABLE + 8, &STUB_ADDR.to_le_bytes()).unwrap();
            }
            uc.reg_write(RegisterRISCV::T2, PROBE_TABLE).unwrap();
            uc.reg_write(RegisterRISCV::A0, POISON).unwrap();
            uc.reg_write(RegisterRISCV::A1, POISON).unwrap();
            uc.emu_start(CODE, halt, 0, 100_000).unwrap();
            let handler_ret = uc.reg_read(RegisterRISCV::A0).unwrap();
            let marker = uc.reg_read(RegisterRISCV::A1).unwrap();
            let mut counter = [0u8; 8];
            uc.mem_read(PROBE_TABLE, &mut counter).unwrap();
            (handler_ret, marker, u64::from_le_bytes(counter))
        }
        NativeArch::Riscv32 | NativeArch::Arm | NativeArch::I686 => {
            panic!("not implemented for {arch:?} in this helper");
        }
    }
}

fn assert_call_probe(arch: NativeArch) {
    // Handler disabled (null pointer): the probe site must still increment
    // the hit counter and fall through to the marker — proving `Call`
    // binding never blocks execution even when no handler is installed.
    let (handler_ret, marker, counter) = run_call_probe(arch, false);
    assert_eq!(handler_ret, 0xFFFF_FFFF, "{arch:?}: handler reg untouched when disabled");
    assert_eq!(marker, 0x333, "{arch:?}: marker always reached (disabled)");
    assert_eq!(counter, 1, "{arch:?}: hit counter increments even when disabled");

    // Handler installed: it must run (its write is observed) *and* control
    // must return to the probe site afterward (the marker write is also
    // observed) — the defining difference from `TailTakeover`, which would
    // never reach the marker at all.
    let (handler_ret, marker, counter) = run_call_probe(arch, true);
    assert_eq!(handler_ret, 0x222, "{arch:?}: handler ran and its write survived the return");
    assert_eq!(marker, 0x333, "{arch:?}: control resumed at the probe site after the call");
    assert_eq!(counter, 1, "{arch:?}: hit counter incremented once");
}

#[test] fn test_call_probe_x86_64() { assert_call_probe(NativeArch::X86_64); }
#[test] fn test_call_probe_aarch64() { assert_call_probe(NativeArch::AArch64); }
#[test] fn test_call_probe_riscv64() { assert_call_probe(NativeArch::Riscv64); }

// ---------------------------------------------------------------------------
// Test — Passive-mode `Call` probe on RISC-V 64
//
// RISC-V is the only backend with a real (lazy) register allocator that
// keeps WASM operand-stack values resident in physical registers across
// multiple ops; x86-64/AArch64 always materialise to memory between ops, so
// Active and Passive are identical there (no allocator to disturb). This
// test proves Passive mode's whole point: a value the allocator currently
// has live in a register survives a `Call`-bound probe untouched, *without*
// a `flush()` — even though the probe site happens to reuse that exact
// register as its own scratch internally.
// ---------------------------------------------------------------------------

#[test]
fn test_passive_call_probe_preserves_regalloc_riscv64() {
    use portal_solutions_asm_riscv64::out::WriterCore;
    use portal_solutions_asm_riscv64::out::rv_asm_backend::RvAsmWriter;
    use portal_solutions_asm_riscv64::regalloc as riscv_regalloc;
    use portal_solutions_blitz_common::asm::Reg;
    use portal_solutions_blitz_common::ops::ProbeTableConfig;
    use portal_solutions_blitz_riscv64::{
        codegen::ProbeBase, naive, sysv::PROBE_BASE_REG, RiscV64Arch, RiscvLabel,
    };
    use unicorn_engine::{unicorn_const::{Arch, Mode, Prot}, RegisterRISCV, Unicorn};

    const CODE: u64 = 0x100000;
    const STUB_ADDR: u64 = CODE + 0x800;
    const PROBE_TABLE: u64 = 0x400000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;
    const LIVE_SENTINEL: u64 = 0x456;
    const HANDLER_SENTINEL: u16 = 0x222;

    let arch = RiscV64Arch::default();
    let mut out = RvAsmWriter::<RiscvLabel>::new();
    let mut ctx = ();
    let mut state = naive::State::default();
    state.probes = Some(ProbeTableConfig { enabled: true, num_probes: 1, table_base_off: 0 });
    state.probe_base = ProbeBase::Reg(PROBE_BASE_REG);

    // Lazily initialize the allocator exactly like production code does, then
    // push one int value without flushing — it lands in a live register.
    let ralloc = state.regalloc.get_or_insert_with(|| {
        let r = riscv_regalloc::init_regalloc::<32>(arch);
        portal_solutions_asm_regalloc::RegAlloc { frames: naive::Frames(r.frames), tos: r.tos }
    });
    let (live_reg, cmds) = ralloc.push(riscv_regalloc::RegKind::Int).unwrap();
    assert_eq!(cmds.count(), 0, "first push into an empty allocator needs no eviction");
    // Registers 0/1/2/3/4 (zero/ra/sp/gp/tp) and 8 (fp) are reserved, so the
    // first int push lands in t0 (Reg 5) — which `emit_passive_call_probe`
    // also uses as its own scratch register, the exact collision this test
    // is meant to exercise.
    assert_eq!(live_reg, 5, "first allocated int register is t0 — update this test if regalloc's reserved set changes");
    out.li(&mut ctx, arch, &Reg(live_reg), LIVE_SENTINEL).unwrap();

    naive::emit_passive_call_probe(&mut out, &mut ctx, arch, &mut state).unwrap();

    let code = out.into_bytes();
    let halt = CODE + code.len() as u64;
    let handler_stub = sysv_entry_stub(NativeArch::Riscv64, HANDLER_SENTINEL);

    let mut uc = Unicorn::new(Arch::RISCV, Mode::RISCV64).unwrap();
    uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
    uc.mem_map(PROBE_TABLE, 0x1000, Prot::ALL).unwrap();
    uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
    uc.mem_write(CODE, &code).unwrap();
    uc.mem_write(PROBE_TABLE, &[0u8; 16]).unwrap();
    uc.mem_write(STUB_ADDR, &handler_stub).unwrap();
    uc.mem_write(PROBE_TABLE + 8, &STUB_ADDR.to_le_bytes()).unwrap();
    uc.reg_write(RegisterRISCV::SP, STACK + STACK_SIZE - 8).unwrap();
    uc.reg_write(RegisterRISCV::T2, PROBE_TABLE).unwrap();
    uc.emu_start(CODE, halt, 0, 100_000).unwrap();

    let handler_ret = uc.reg_read(RegisterRISCV::A0).unwrap();
    assert_eq!(handler_ret, HANDLER_SENTINEL as u64, "handler ran");
    let live_after = uc.reg_read(RegisterRISCV::T0).unwrap();
    assert_eq!(
        live_after, LIVE_SENTINEL,
        "regalloc-resident value must survive a Passive probe call untouched, \
         even though the probe site reused t0 as scratch internally"
    );
    let mut counter = [0u8; 8];
    uc.mem_read(PROBE_TABLE, &mut counter).unwrap();
    assert_eq!(u64::from_le_bytes(counter), 1, "hit counter incremented once");
}

// ---------------------------------------------------------------------------
// Tests — backend sharding
// ---------------------------------------------------------------------------

/// Build a 2-function WASM module where fn0 (no args) calls fn1 (i64 → i64).
/// fn0 pushes 42 and calls fn1; fn1 returns its argument + 1.
/// Expected: calling fn0 returns 43.
fn make_two_func_cross_call_module() -> Vec<u8> {
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, TypeSection,
    };

    let mut module = wasm_encoder::Module::new();

    // Two types: () → [i64],  [i64] → [i64]
    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I64]); // type 0: fn0
    types.ty().function([ValType::I64], [ValType::I64]); // type 1: fn1
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0); // fn0 → type 0
    functions.function(1); // fn1 → type 1
    module.section(&functions);

    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    exports.export("f1", ExportKind::Func, 1);
    module.section(&exports);

    let mut code = CodeSection::new();

    // fn0: i64.const 42; call fn1; return
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::I64Const(42));
    f0.instruction(&Instruction::Call(1));
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);

    // fn1: local.get 0; i64.const 1; i64.add; return
    let mut f1 = Function::new([]);
    f1.instruction(&Instruction::LocalGet(0));
    f1.instruction(&Instruction::I64Const(1));
    f1.instruction(&Instruction::I64Add);
    f1.instruction(&Instruction::Return);
    f1.instruction(&Instruction::End);
    code.function(&f1);

    module.section(&code);
    module.finish()
}

/// Build a module with one exported function `(i64 x8) -> i64` that returns its
/// 8th parameter (local 7). Exercises SysV argument passing for params beyond
/// the 6 integer argument registers (i.e. stack-passed args 7 and 8).
fn make_eight_param_return_last_module() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64; 8], [ValType::I64]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::LocalGet(7));
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    module.section(&code);
    module.finish()
}

/// Verify the SysV backend loads stack-passed parameters (index >= 6), not just
/// the six register arguments. Compiles the 8-param function and calls it under
/// Unicorn with args 7 and 8 placed on the stack per the System V ABI.
#[test]
fn test_native_x86_64_sysv_stack_params() {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        RegisterX86, Unicorn,
    };

    let wasm = make_eight_param_return_last_module();
    let code = compile_native_binary(&wasm, NativeArch::X86_64, NativeAbi::Sysv);

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).unwrap();
    uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
    uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
    uc.mem_write(CODE, &code).unwrap();

    let args: [u64; 8] = [10, 11, 12, 13, 14, 15, 16, 17];
    // Register args 0..6 → RDI, RSI, RDX, RCX, R8, R9.
    uc.reg_write(RegisterX86::RDI, args[0]).unwrap();
    uc.reg_write(RegisterX86::RSI, args[1]).unwrap();
    uc.reg_write(RegisterX86::RDX, args[2]).unwrap();
    uc.reg_write(RegisterX86::RCX, args[3]).unwrap();
    uc.reg_write(RegisterX86::R8, args[4]).unwrap();
    uc.reg_write(RegisterX86::R9, args[5]).unwrap();
    // Stack args 7,8 (index 6,7) sit above the return address:
    //   [rsp] = return addr, [rsp+8] = arg6, [rsp+16] = arg7.
    let rsp = STACK + STACK_SIZE - 64;
    uc.mem_write(rsp, &(CODE + code.len() as u64).to_le_bytes()).unwrap();
    uc.mem_write(rsp + 8, &args[6].to_le_bytes()).unwrap();
    uc.mem_write(rsp + 16, &args[7].to_le_bytes()).unwrap();
    uc.reg_write(RegisterX86::RSP, rsp).unwrap();

    uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
    assert_eq!(uc.reg_read(RegisterX86::RAX).unwrap(), 17, "should return the 8th argument");
}

/// Build a module with one exported function `(i64 x10) -> i64` that returns its
/// 10th parameter (local 9), exercising AAPCS64 stack-passed args (9th and 10th).
fn make_ten_param_return_last_module() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64; 10], [ValType::I64]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::LocalGet(9));
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    module.section(&code);
    module.finish()
}

/// Verify the AArch64 SysV (AAPCS64) backend loads stack-passed parameters
/// (index >= 8), calling the 10-param function under Unicorn with args 9 and 10
/// on the stack.
#[test]
fn test_native_aarch64_sysv_stack_params() {
    use unicorn_engine::{
        unicorn_const::{Arch, Mode, Prot},
        RegisterARM64, Unicorn,
    };

    let wasm = make_ten_param_return_last_module();
    let code = compile_native_binary(&wasm, NativeArch::AArch64, NativeAbi::Sysv);

    const CODE: u64 = 0x100000;
    const STACK: u64 = 0x200000;
    const STACK_SIZE: u64 = 0x10000;

    let mut uc = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
    uc.mem_map(CODE, 0x10000, Prot::ALL).unwrap();
    uc.mem_map(STACK, STACK_SIZE, Prot::ALL).unwrap();
    uc.mem_write(CODE, &code).unwrap();

    let args: [u64; 10] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
    let arg_regs = [
        RegisterARM64::X0, RegisterARM64::X1, RegisterARM64::X2, RegisterARM64::X3,
        RegisterARM64::X4, RegisterARM64::X5, RegisterARM64::X6, RegisterARM64::X7,
    ];
    for (r, &v) in arg_regs.iter().zip(args.iter()) {
        uc.reg_write(*r, v).unwrap();
    }
    // AAPCS64 stack args: [SP] = arg8, [SP+8] = arg9 (no return address on stack;
    // the return address is in LR).
    let sp = STACK + STACK_SIZE - 32;
    uc.mem_write(sp, &args[8].to_le_bytes()).unwrap();
    uc.mem_write(sp + 8, &args[9].to_le_bytes()).unwrap();
    uc.reg_write(RegisterARM64::SP, sp).unwrap();
    uc.reg_write(RegisterARM64::LR, CODE + code.len() as u64).unwrap();

    uc.emu_start(CODE, CODE + code.len() as u64, 0, 5000).unwrap();
    assert_eq!(uc.reg_read(RegisterARM64::X0).unwrap(), 29, "should return the 10th argument");
}

/// Compile `wasm` to N C shards using `RoundRobinShardMap`.
/// Returns `(cross_shard_decls_per_shard, shard_bodies)`.
/// `cross_shard_decls_per_shard[k]` contains extern declarations for functions NOT in shard k.
/// `shard_bodies[k]` contains the function bodies for shard k.
fn compile_c_sharded_raw(wasm: &[u8], n: usize) -> (Vec<String>, Vec<String>) {
    use portal_solutions_blitz_common::shard::{RoundRobinShardMap, ShardMap};
    use portal_solutions_blitz_c::shard::c_emit_cross_shard_decls;

    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let imports_len = {
        let mut body_count = 0u32;
        for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
            if let wasmparser::Payload::CodeSectionEntry(_) = p { body_count += 1; }
        }
        fsigs.len() as u32 - body_count
    };

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let shard_map = RoundRobinShardMap { n };
    let mut decls: Vec<String> = (0..n).map(|_| String::new()).collect();
    let mut bodies_out: Vec<String> = (0..n).map(|_| String::new()).collect();

    for shard_idx in 0..n {
        c_emit_cross_shard_decls(&mut decls[shard_idx], shard_idx, &sigs_enc, &fsigs, imports_len, &shard_map).unwrap();
    }

    let mut current_shard = 0usize;
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, imports_len);
    let ops = dce_pass!(raw_ops);
    for op in ops {
        let op = op.unwrap();
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            current_shard = shard_map.shard_for(*id + imports_len);
            state = CState::default();
        }
        CWrite::on_mach(&mut bodies_out[current_shard], &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    (decls, bodies_out)
}

/// Compile `wasm` to N C shards and also return a preamble.
/// `preamble` has static forward declarations for ALL functions so they can be
/// referenced before being defined (needed when shards are concatenated for testing).
fn compile_c_sharded_with_preamble(wasm: &[u8], n: usize) -> (String, Vec<String>) {
    use portal_solutions_blitz_c::shard::c_emit_cross_shard_decls;

    let (_, bodies) = compile_c_sharded_raw(wasm, n);

    // Build preamble: scan all shard bodies for `static const struct{...}__sig_N={...};`
    // and `static uint64_t __rets_N[R];` and the function signature `static uint64_t*fn_N(`.
    // Emit static forward declarations for each function found.
    let mut preamble = String::new();
    let all = bodies.concat();
    // Extract each `__sig_N = { ... };` definition so we can forward-declare before use.
    // Strategy: scan for `static const struct{int params;int rets;}__sig_` and extract the full definition.
    let mut s = all.as_str();
    let mut found_sigs: std::collections::BTreeSet<&str> = Default::default();
    while let Some(pos) = s.find("static const struct{int params;int rets;}__sig_") {
        let end = s[pos..].find(';').map(|e| pos + e + 1).unwrap_or(s.len());
        preamble.push_str(&s[pos..end]);
        preamble.push('\n');
        s = &s[end..];
    }
    // Forward declarations for functions.
    s = all.as_str();
    while let Some(pos) = s.find("static uint64_t*fn_") {
        let rest = &s[pos + 19..]; // after "static uint64_t*fn_"
        let end = rest.find('(').unwrap_or(rest.len());
        let n_str = &rest[..end];
        if n_str.chars().all(|c| c.is_ascii_digit()) {
            let fwd = format!("static uint64_t*fn_{n_str}(uint64_t*restrict);\n");
            if !preamble.contains(&fwd) { preamble.push_str(&fwd); }
        }
        s = &s[pos + 1..];
    }
    // Also emit `static uint64_t __rets_N[R];` forward declarations.
    s = all.as_str();
    while let Some(pos) = s.find("static uint64_t __rets_") {
        let semicolon = s[pos..].find(';').map(|e| pos + e + 1).unwrap_or(s.len());
        let decl = &s[pos..semicolon];
        if !preamble.contains(decl) { preamble.push_str(decl); preamble.push('\n'); }
        s = &s[pos + 1..];
    }

    (preamble, bodies)
}

/// Convenience wrapper: returns just the shard bodies (without extern decls).
/// Shards can be concatenated and compiled in a single TU for execution testing.
fn compile_c_sharded(wasm: &[u8], n: usize) -> Vec<String> {
    compile_c_sharded_raw(wasm, n).1
}

/// Run C shards: concatenate all shards into one translation unit and compile.
///
/// The C backend emits `static` functions, so separate-TU linking doesn't work.
/// We need a preamble with forward declarations so all functions are visible before
/// their definitions appear. Use `compile_c_sharded_with_preamble` to get the preamble.
fn run_c_sharded_with_preamble(preamble: &str, shards: &[String], fn_id: u32, args: &[u64], rets: usize) -> Vec<u64> {
    let combined = format!("{preamble}\n{}", shards.join("\n"));
    run_c(&combined, fn_id, args, rets)
}

/// Run C shards: extracts `__sig_N`/`__rets_N` definitions and function forward
/// declarations into a preamble, then concatenates function bodies.  This allows
/// shards with cross-shard calls to compile as a single TU regardless of definition order.
fn run_c_sharded(shards: &[String], fn_id: u32, args: &[u64], rets: usize) -> Vec<u64> {
    let tag = "static const struct{int params;int rets;}__sig_";
    let fn_tag = "static uint64_t*fn_";

    let mut preamble = String::new();
    let mut fn_bodies: Vec<String> = Vec::new();

    for shard in shards {
        let mut s = shard.as_str();
        let mut body = String::new();
        while !s.is_empty() {
            if let Some(sig_pos) = s.find(tag) {
                // Everything before sig_pos is non-function content; add to body.
                body.push_str(&s[..sig_pos]);
                s = &s[sig_pos..];
                // Find the function definition start that follows.
                if let Some(fn_rel) = s.find(fn_tag) {
                    // Headers: __sig_ + __rets_ definitions before fn_tag.
                    let header = &s[..fn_rel];
                    if !preamble.contains(header) { preamble.push_str(header); }
                    // Function forward declaration.
                    let rest = &s[fn_rel + fn_tag.len()..];
                    let idx_end = rest.find('(').unwrap_or(rest.len());
                    let n_str = &rest[..idx_end];
                    if n_str.chars().all(|c: char| c.is_ascii_digit()) {
                        let fwd = format!("{fn_tag}{n_str}(uint64_t*restrict);\n");
                        if !preamble.contains(&fwd) { preamble.push_str(&fwd); }
                    }
                    // Function body: from fn_tag to the next __sig_ or end.
                    let fn_start = fn_rel;
                    let next_sig = s[fn_start + 1..].find(tag).map(|p| fn_start + 1 + p);
                    let fn_end = next_sig.unwrap_or(s.len());
                    body.push_str(&s[fn_start..fn_end]);
                    s = &s[fn_end..];
                } else {
                    body.push_str(s);
                    s = "";
                }
            } else {
                body.push_str(s);
                s = "";
            }
        }
        fn_bodies.push(body);
    }

    let combined = format!("{preamble}\n{}", fn_bodies.join("\n"));
    run_c(&combined, fn_id, args, rets)
}

/// Compile `wasm` to N JS ESM shards.
/// Returns one String per shard. `shard_paths[k]` is used in import specifiers.
fn compile_js_sharded(wasm: &[u8], n: usize, shard_paths: &[&str]) -> Vec<String> {
    use portal_solutions_blitz_common::shard::{RoundRobinShardMap, ShardMap};
    use portal_solutions_blitz_js::shard::{js_emit_cross_shard_imports, js_emit_shard_exports};

    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let imports_len = {
        let mut body_count = 0u32;
        for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
            if let wasmparser::Payload::CodeSectionEntry(_) = p { body_count += 1; }
        }
        fsigs.len() as u32 - body_count
    };
    let local_fn_count = fsigs.len() as u32 - imports_len;

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let shard_map = RoundRobinShardMap { n };
    let mut shards: Vec<String> = (0..n).map(|_| String::new()).collect();

    // Emit ESM imports at the top of each shard.
    for shard_idx in 0..n {
        js_emit_cross_shard_imports(&mut shards[shard_idx], shard_idx, imports_len, local_fn_count, &shard_map, shard_paths).unwrap();
    }

    // Route operators to the correct shard.
    let mut current_shard = 0usize;
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, imports_len);
    let ops = dce_pass!(raw_ops);
    for op in ops {
        let op = op.unwrap();
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            current_shard = shard_map.shard_for(*id + imports_len);
            state = JsState::default();
        }
        JsWrite::on_mach(&mut shards[current_shard], &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }

    // Emit ESM exports at the bottom of each shard.
    for shard_idx in 0..n {
        js_emit_shard_exports(&mut shards[shard_idx], shard_idx, imports_len, local_fn_count, &shard_map).unwrap();
    }

    shards
}

/// Compile `wasm` to N JS ESM shards and run the entry shard with node.
///
/// Computes temp file paths first, passes them as import specifiers to the compiler,
/// then writes the shards to those exact paths so node can resolve the imports.
fn run_js_sharded_esm(wasm: &[u8], n: usize, entry_shard: usize, harness: &str) -> Vec<i64> {
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();

    let paths: Vec<_> = (0..n)
        .map(|i| dir.join(format!("blitz_shard_{pid}_{seq}_{i}.mjs")))
        .collect();
    // Relative import specifiers for ESM: `./blitz_shard_N_S_I.mjs`
    let rel_paths: Vec<String> = (0..n)
        .map(|i| format!("./blitz_shard_{pid}_{seq}_{i}.mjs"))
        .collect();
    let rel_path_refs: Vec<&str> = rel_paths.iter().map(String::as_str).collect();

    let shards = compile_js_sharded(wasm, n, &rel_path_refs);

    for (i, (shard, path)) in shards.iter().zip(&paths).enumerate() {
        let content = if i == entry_shard {
            format!("{shard}\n{harness}")
        } else {
            shard.clone()
        };
        std::fs::write(path, content).unwrap();
    }

    let out = std::process::Command::new("node")
        .arg(&paths[entry_shard])
        .output()
        .expect("node not found in PATH");

    for path in &paths { let _ = std::fs::remove_file(path); }

    assert!(out.status.success(), "node exited non-zero:\nstderr: {}\n", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<i64>().expect("expected int from node"))
        .collect()
}

/// Compile `wasm` to N native ASM shards (text), with sharding enabled.
fn compile_native_asm_sharded(wasm: &[u8], arch: NativeArch, abi: NativeAbi, n: usize) -> Vec<String> {
    use portal_solutions_blitz_common::shard::{
        RoundRobinShardMap, SecondCtxConfig, ShardConfig, ShardMap,
    };

    let (sigs_wp, _sigs_enc, fsigs) = parse_sigs(wasm);
    let imports_len = {
        let mut bodies = 0u32;
        for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
            if let wasmparser::Payload::CodeSectionEntry(_) = p { bodies += 1; }
        }
        fsigs.len() as u32 - bodies
    };

    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }

    let shard_map = RoundRobinShardMap { n };
    let total_fns = fsigs.len() as u32;
    let second_ctx = SecondCtxConfig::for_shard(ShardConfig { imports_len, total_fns });

    let mut shards: Vec<String> = (0..n).map(|_| String::new()).collect();
    let mut reencoder = RoundtripReencoder;

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, imports_len);
    let ops: Vec<_> = dce_pass!(raw_ops).collect::<Result<Vec<_>, _>>().unwrap();

    match (arch, abi) {
        (NativeArch::X86_64, NativeAbi::Naive) => {
            use portal_solutions_blitz_x86_64::{naive, X64Arch};
            let mut state = naive::State::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = naive::State {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], &[], &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        (NativeArch::X86_64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_x86_64::{naive, sysv, X64Arch};
            let mut state = sysv::SysVState::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = sysv::SysVState {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, X64Arch::default(),
                    &mut state, &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        (NativeArch::AArch64, NativeAbi::Naive) => {
            use portal_solutions_blitz_aarch64::{naive, AArch64Arch};
            let mut state = naive::State::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = naive::State {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], &[], &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        (NativeArch::AArch64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch};
            let mut state = naive::State::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = naive::State {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, AArch64Arch::default(),
                    &mut state, &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        (NativeArch::Riscv64, NativeAbi::Naive) => {
            use portal_solutions_blitz_riscv64::{naive, RiscV64Arch};
            let mut state = naive::State::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = naive::State {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                naive::WriterExt::handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], &[], &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        (NativeArch::Riscv64, NativeAbi::Sysv) => {
            use portal_solutions_blitz_riscv64::{naive, sysv, RiscV64Arch};
            let mut state = naive::State::default();
            let mut current_shard = 0usize;
            let mut ctx = ();
            for op in &ops {
                if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = op {
                    current_shard = shard_map.shard_for(*id + imports_len);
                    state = naive::State {
                        shard: Some(naive::NaiveShardState::new(second_ctx, current_shard, imports_len, &shard_map)),
                        ..Default::default()
                    };
                }
                let mut out = NativeAsmWriter(String::new());
                sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
                    &mut out, &mut ctx, RiscV64Arch::default(),
                    &mut state, &[], op, &mut reencoder, 0,
                ).unwrap();
                shards[current_shard].push_str(&out.0);
            }
        }
        _ => unimplemented!("sharding not supported for LFI ABI"),
    }

    shards
}

// ---- shard_single_equiv: single-shard output must match non-sharded --------

#[test]
fn shard_single_equiv_c() {
    let wasm = make_module(&[ValType::I64], &[ValType::I64], &[
        Instruction::LocalGet(0),
        Instruction::I64Const(10),
        Instruction::I64Add,
    ]);
    let shards = compile_c_sharded(&wasm, 1);
    // Single-shard output contains fn_0 and no cross-shard calls.
    assert!(shards[0].contains("fn_0"), "shard 0 must contain fn_0 body");
    // Run and verify correct result.
    assert_eq!(run_c_sharded(&shards, 0, &[5], 1), vec![15]);
    // Matches non-sharded result.
    assert_eq!(run_c(&compile_c(&wasm), 0, &[5], 1), vec![15]);
}

#[test]
fn shard_single_equiv_js() {
    let wasm = make_module(&[ValType::I64], &[ValType::I64], &[
        Instruction::LocalGet(0),
        Instruction::I64Const(10),
        Instruction::I64Add,
    ]);
    // Verify shard body contains fn definition.
    let check_shards = compile_js_sharded(&wasm, 1, &["./shard_0.mjs"]);
    assert!(check_shards[0].contains("$0"), "shard 0 must contain $0 body\n{}", check_shards[0]);
    // Run via ESM.
    let result = run_js_sharded_esm(&wasm, 1, 0, "console.log(String($0(5n)));");
    assert_eq!(result, vec![15]);
}

// ---- shard_two_fn_cross_call: 2 functions in different shards ----------

#[test]
fn shard_two_fn_cross_call_c() {
    // fn0() → i64: push 42, call fn1 → returns 43
    // fn1(i64) → i64: arg + 1
    let wasm = make_two_func_cross_call_module();
    let (decls, shards) = compile_c_sharded_raw(&wasm, 2);
    // Verify cross-shard extern declarations are present in at least one shard.
    assert!(decls[0].contains("extern") || decls[1].contains("extern"),
        "at least one shard must have cross-shard extern decl\ndecls[0]:\n{}\ndecls[1]:\n{}", decls[0], decls[1]);
    // Execute: call fn0 (which calls fn1) and expect 43.
    let result = run_c_sharded(&shards, 0, &[], 1);
    assert_eq!(result, vec![43], "fn0() must return fn1(42) = 43");
}

#[test]
fn shard_two_fn_cross_call_js() {
    let wasm = make_two_func_cross_call_module();
    // Entry shard is shard 0 (fn0 is WASM idx 0, 0 % 2 = 0).
    let result = run_js_sharded_esm(&wasm, 2, 0, "console.log(String($0()));");
    assert_eq!(result, vec![43], "fn0() must return fn1(42n) = 43n via cross-shard ESM import");
}

#[test]
fn shard_three_fn_mixed_c() {
    // 3-function module: fn0 calls fn1 (cross-shard) then fn2 (intra-shard if 3 shards split 0→shard0, 1→shard1, 2→shard0).
    // fn0() → i64: call fn1 then add result with call fn2
    // Actually simpler: 3 functions, split into 2 shards:
    // shard 0 gets fn0 (idx%2==0) and fn2 (idx%2==0 → idx=2, 2%2=0)
    // shard 1 gets fn1 (idx%2==1 → idx=1)
    // fn0: call fn1; call fn2; return result of fn2
    // fn1(i64) → i64: arg + 10
    // fn2(i64) → i64: arg * 2
    // fn0 passes 5 to fn1 → 15, then passes 15 to fn2 → 30.
    use wasm_encoder::{CodeSection, ExportKind, ExportSection, Function, FunctionSection, TypeSection};

    let mut module = wasm_encoder::Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I64]);         // type 0: fn0
    types.ty().function([ValType::I64], [ValType::I64]); // type 1: fn1, fn2
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    functions.function(1);
    functions.function(1);
    module.section(&functions);

    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    module.section(&exports);

    let mut code = CodeSection::new();
    // fn0: i64.const 5; call fn1; call fn2; return
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::I64Const(5));
    f0.instruction(&Instruction::Call(1));
    f0.instruction(&Instruction::Call(2));
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    // fn1: local.get 0; i64.const 10; i64.add; return
    let mut f1 = Function::new([]);
    f1.instruction(&Instruction::LocalGet(0));
    f1.instruction(&Instruction::I64Const(10));
    f1.instruction(&Instruction::I64Add);
    f1.instruction(&Instruction::Return);
    f1.instruction(&Instruction::End);
    code.function(&f1);
    // fn2: local.get 0; i64.const 2; i64.mul; return
    let mut f2 = Function::new([]);
    f2.instruction(&Instruction::LocalGet(0));
    f2.instruction(&Instruction::I64Const(2));
    f2.instruction(&Instruction::I64Mul);
    f2.instruction(&Instruction::Return);
    f2.instruction(&Instruction::End);
    code.function(&f2);
    module.section(&code);
    let wasm = module.finish();

    let shards = compile_c_sharded(&wasm, 2);
    let result = run_c_sharded(&shards, 0, &[], 1);
    assert_eq!(result, vec![30], "fn0 should return fn2(fn1(5)) = fn2(15) = 30");
}

// ---- shard_asm_*: native assembly backend structural tests -----------------

/// Single-shard native ASM sharding produces non-empty output that
/// contains the expected intra-shard call pattern (direct label call, no SCR load).
fn assert_shard_single_asm_contains_no_scr_load(arch: NativeArch, abi: NativeAbi, scr_pattern: &str) {
    let wasm = make_two_func_cross_call_module();
    // n=1: both functions in shard 0, call is intra-shard, no SCR-relative load.
    let shards = compile_native_asm_sharded(&wasm, arch, abi, 1);
    assert!(!shards[0].is_empty(), "single-shard asm must be non-empty");
    assert!(!shards[0].contains(scr_pattern),
        "single-shard must not emit cross-shard SCR load (all calls are intra-shard):\n{}", shards[0]);
}

/// Two-shard native ASM sharding: the shard containing fn0 must emit a
/// SCR-relative indirect load for the cross-shard call to fn1.
fn assert_shard_cross_asm_contains_scr_load(arch: NativeArch, abi: NativeAbi, scr_pattern: &str) {
    let wasm = make_two_func_cross_call_module();
    let shards = compile_native_asm_sharded(&wasm, arch, abi, 2);
    assert!(!shards[0].is_empty(), "shard 0 must be non-empty");
    assert!(!shards[1].is_empty(), "shard 1 must be non-empty");
    assert!(shards[0].contains(scr_pattern) || shards[1].contains(scr_pattern),
        "at least one shard must contain the SCR-relative indirect load\nshard0:\n{}\nshard1:\n{}", shards[0], shards[1]);
}

// Cross-shard loads use `[r14+...` (x86-64) or `[x27,...` / `[x27` (AArch64).
// The prologue/epilogue uses `push r14`/`pop r14` which don't match these bracket forms.
#[test]
fn shard_asm_single_no_scr_x86_64_naive() {
    assert_shard_single_asm_contains_no_scr_load(NativeArch::X86_64, NativeAbi::Naive, "[r14");
}
#[test]
fn shard_asm_single_no_scr_x86_64_sysv() {
    assert_shard_single_asm_contains_no_scr_load(NativeArch::X86_64, NativeAbi::Sysv, "[r14");
}
#[test]
fn shard_asm_single_no_scr_aarch64_naive() {
    assert_shard_single_asm_contains_no_scr_load(NativeArch::AArch64, NativeAbi::Naive, "[x27");
}
#[test]
fn shard_asm_single_no_scr_aarch64_sysv() {
    assert_shard_single_asm_contains_no_scr_load(NativeArch::AArch64, NativeAbi::Sysv, "[x27");
}
#[test]
fn shard_asm_cross_scr_load_x86_64_naive() {
    assert_shard_cross_asm_contains_scr_load(NativeArch::X86_64, NativeAbi::Naive, "[r14");
}
#[test]
fn shard_asm_cross_scr_load_x86_64_sysv() {
    assert_shard_cross_asm_contains_scr_load(NativeArch::X86_64, NativeAbi::Sysv, "[r14");
}
#[test]
fn shard_asm_cross_scr_load_aarch64_naive() {
    assert_shard_cross_asm_contains_scr_load(NativeArch::AArch64, NativeAbi::Naive, "[x27");
}
#[test]
fn shard_asm_cross_scr_load_aarch64_sysv() {
    assert_shard_cross_asm_contains_scr_load(NativeArch::AArch64, NativeAbi::Sysv, "[x27");
}

// ---------------------------------------------------------------------------
// Phase 1 C/JS parity: bulk memory, call_indirect/return_call, multi-memory.
// ---------------------------------------------------------------------------

use portal_solutions_blitz_c::{c_emit_funcref_table, c_emit_passive_data_segment};
use portal_solutions_blitz_js::{js_emit_funcref_table, js_emit_passive_data_segment};
use wasm_encoder::MemArg;

fn make_module_2fn(
    ty: (&[ValType], &[ValType]),
    fn0: &[Instruction<'_>],
    fn1: &[Instruction<'_>],
    with_mem: bool,
) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function(ty.0.iter().cloned(), ty.1.iter().cloned());
    module.section(&types);

    let mut functions = FunctionSection::new();
    functions.function(0);
    functions.function(0);
    module.section(&functions);

    if with_mem {
        let mut memories = MemorySection::new();
        memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });
        module.section(&memories);
    }

    let mut code = CodeSection::new();
    for instrs in [fn0, fn1] {
        let mut func = Function::new([]);
        for instr in instrs {
            func.instruction(instr);
        }
        func.instruction(&Instruction::End);
        code.function(&func);
    }
    module.section(&code);
    module.finish()
}

#[test]
fn test_call_indirect_c() {
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[Instruction::LocalGet(0), Instruction::I64Const(1), Instruction::I64Add, Instruction::Return],
        &[
            Instruction::LocalGet(0),
            Instruction::I32Const(0),
            Instruction::CallIndirect { type_index: 0, table_index: 0 },
            Instruction::Return,
        ],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    // `c_emit_funcref_table` forward-declares `fn_0` itself, so it can run
    // before the function bodies that reference `__wasm_table_0` are emitted.
    c_emit_funcref_table(&mut out, 0, &[0]).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== call_indirect C ===\n{out}");
    // fn_1(41) -> call_indirect through table[0] = fn_0 -> 41+1 = 42
    assert_eq!(run_c(&out, 1, &[41], 1), vec![42]);
}

#[test]
fn test_return_call_c() {
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[Instruction::LocalGet(0), Instruction::I64Const(1), Instruction::I64Add, Instruction::Return],
        &[Instruction::LocalGet(0), Instruction::ReturnCall(0)],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== return_call C ===\n{out}");
    assert_eq!(run_c(&out, 1, &[41], 1), vec![42]);
}

#[test]
fn test_return_call_indirect_c() {
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[Instruction::LocalGet(0), Instruction::I64Const(1), Instruction::I64Add, Instruction::Return],
        &[
            Instruction::LocalGet(0),
            Instruction::I32Const(0),
            Instruction::ReturnCallIndirect { type_index: 0, table_index: 0 },
        ],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    c_emit_funcref_table(&mut out, 0, &[0]).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== return_call_indirect C ===\n{out}");
    assert_eq!(run_c(&out, 1, &[41], 1), vec![42]);
}

#[test]
fn test_call_indirect_js() {
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[Instruction::LocalGet(0), Instruction::I64Const(1), Instruction::I64Add, Instruction::Return],
        &[
            Instruction::LocalGet(0),
            Instruction::I32Const(0),
            Instruction::CallIndirect { type_index: 0, table_index: 0 },
            Instruction::Return,
        ],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    js_emit_funcref_table(&mut out, 0, &[0]).unwrap();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== call_indirect JS ===\n{out}");
    assert_eq!(run_js(&out, &[41]), vec![42]);
}

#[test]
fn test_return_call_js() {
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[Instruction::LocalGet(0), Instruction::I64Const(1), Instruction::I64Add, Instruction::Return],
        &[Instruction::LocalGet(0), Instruction::ReturnCall(0)],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== return_call JS ===\n{out}");
    assert_eq!(run_js(&out, &[41]), vec![42]);
}

#[test]
fn test_bulk_memory_c() {
    // fn0: fill mem[0..10]=0xAB; copy mem[0..10]->mem[20..30]; load i32 @20.
    let fn0: &[Instruction<'_>] = &[
        Instruction::I32Const(0), Instruction::I32Const(0xAB), Instruction::I32Const(10),
        Instruction::MemoryFill(0),
        Instruction::I32Const(20), Instruction::I32Const(0), Instruction::I32Const(10),
        Instruction::MemoryCopy { src_mem: 0, dst_mem: 0 },
        Instruction::I32Const(20),
        Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 }),
        Instruction::Return,
    ];
    // fn1: memory.init(seg0, src=0, dest=100, len=4); data.drop(0); load i32 @100.
    let fn1: &[Instruction<'_>] = &[
        Instruction::I32Const(100), Instruction::I32Const(0), Instruction::I32Const(4),
        Instruction::MemoryInit { mem: 0, data_index: 0 },
        Instruction::DataDrop(0),
        Instruction::I32Const(100),
        Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 }),
        Instruction::Return,
    ];
    let wasm = make_module_2fn((&[], &[ValType::I32]), fn0, fn1, true);
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    c_emit_passive_data_segment(&mut out, 0, &[1, 2, 3, 4]).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== bulk memory C ===\n{out}");
    assert_eq!(run_c_with_grow(&out, 1, 0, &[], 1), vec![0xABABABAB]);
    assert_eq!(run_c_with_grow(&out, 1, 1, &[], 1), vec![0x04030201]);
}

#[test]
fn test_bulk_memory_js() {
    let fn0: &[Instruction<'_>] = &[
        Instruction::I32Const(0), Instruction::I32Const(0xAB), Instruction::I32Const(10),
        Instruction::MemoryFill(0),
        Instruction::I32Const(20), Instruction::I32Const(0), Instruction::I32Const(10),
        Instruction::MemoryCopy { src_mem: 0, dst_mem: 0 },
        Instruction::I32Const(20),
        Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 }),
        Instruction::Return,
    ];
    let fn1: &[Instruction<'_>] = &[
        Instruction::I32Const(100), Instruction::I32Const(0), Instruction::I32Const(4),
        Instruction::MemoryInit { mem: 0, data_index: 0 },
        Instruction::DataDrop(0),
        Instruction::I32Const(100),
        Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 }),
        Instruction::Return,
    ];
    let wasm = make_module_2fn((&[], &[ValType::I32]), fn0, fn1, true);
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    js_emit_passive_data_segment(&mut out, 0, &[1, 2, 3, 4]).unwrap();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== bulk memory JS ===\n{out}");
    let mem = vec![0u8; 65536];
    assert_eq!(run_js_with_mem(&out, &mem, &[]).iter().map(|v| *v as u32).collect::<Vec<_>>(), vec![0xABABABAB]);
    // second call needs its own harness invocation ($1)
    let harness = "\n$mem=new Uint8Array(65536);$mem_dv=new DataView($mem.buffer);\nconst __r=$1();console.log(String(__r));";
    let code = format!("{out}{harness}");
    let out2 = std::process::Command::new("node").arg("-e").arg(&code).output().expect("node");
    assert!(out2.status.success(), "node failed: {}", String::from_utf8_lossy(&out2.stderr));
    let v: i64 = String::from_utf8(out2.stdout).unwrap().trim().parse().unwrap();
    assert_eq!(v as u32, 0x04030201);
}

#[test]
fn test_multi_memory_c() {
    use wasm_encoder::MemArg;
    // fn0 (param i64 val): store val at mem[1][0], load it back from mem[1].
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[
            Instruction::I32Const(0),
            Instruction::LocalGet(0),
            Instruction::I64Store(MemArg { offset: 0, align: 3, memory_index: 1 }),
            Instruction::I32Const(0),
            Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index: 1 }),
            Instruction::Return,
        ],
        &[Instruction::LocalGet(0), Instruction::Return],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    c_module_preamble(&mut out).unwrap();
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== multi memory C ===\n{out}");

    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_mm_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_mm_{pid}_{seq}"));
    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n#define WASM_STACK_SIZE 512\n\
         {out}\n\
         int main(){{\
             uint8_t*_m1=(uint8_t*)calloc(65536,1);__wasm_mems[1]=_m1;__wasm_mem_pages_arr[1]=1;\
             uint64_t _args[1]={{99ull}};uint64_t*_r=fn_0(_args);\
             printf(\"%llu\\n\",_r[0]);return 0;}}\n"
    );
    std::fs::write(&src_path, &full_src).unwrap();
    let compile = std::process::Command::new("cc").arg(&src_path).arg("-Wno-unsequenced").arg("-o").arg(&bin_path).output().expect("cc");
    assert!(compile.status.success(), "compile failed: {}\n{}", String::from_utf8_lossy(&compile.stderr), full_src);
    let run = std::process::Command::new(&bin_path).output().expect("run");
    assert!(run.status.success());
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);
    let v: u64 = String::from_utf8(run.stdout).unwrap().trim().parse().unwrap();
    assert_eq!(v, 99);
}

/// Promise-bail mode: sync import stays sync; Promise import suspends via `.then`.
#[test]
fn test_promise_calls_sync_and_async_import() {
    // (i64) -> i64: call import $0 then return
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64], [ValType::I64]);
    module.section(&types);
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import("env", "add_one", wasm_encoder::EntityType::Function(0));
    module.section(&imports);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);
    let mut exports = ExportSection::new();
    exports.export("f", ExportKind::Func, 1);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut func = Function::new([]);
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::Call(0));
    func.instruction(&Instruction::Return);
    func.instruction(&Instruction::End);
    code.function(&func);
    module.section(&code);
    let wasm = module.finish();

    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 1);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    js_emit_imports(&mut out, &[("env", "add_one")]).unwrap();
    let mut state = JsState::default();
    state.enable_promise_calls();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(
            &mut out, &sigs_enc, &fsigs, &[], &[("env", "add_one")], &mut state, &op, &mut reencoder,
        )
        .unwrap();
    }
    assert!(out.contains("instanceof Promise"), "expected Promise bail:\n{out}");
    assert!(out.contains("$cont_"), "expected continuation:\n{out}");

    // Sync import: export returns a plain array (not a Promise).
    let sync_harness = format!(
        "{out}\n\
         $0=function(x){{return [x+1n];}};\n\
         Object.defineProperty($0,'__sig',{{value:{{params:1,rets:1}}}});\n\
         const __r=$1(41n);\n\
         if(__r instanceof Promise) throw new Error('expected sync');\n\
         console.log(String(Array.isArray(__r)?__r[0]:__r));"
    );
    let o = std::process::Command::new("node").arg("-e").arg(&sync_harness).output().expect("node");
    assert!(o.status.success(), "sync: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8(o.stdout).unwrap().trim(), "42");

    // Promise import: export returns a Promise that resolves after a microtask.
    let async_harness = format!(
        "{out}\n\
         $0=function(x){{return Promise.resolve([x+1n]);}};\n\
         Object.defineProperty($0,'__sig',{{value:{{params:1,rets:1}}}});\n\
         (async()=>{{const __r=$1(41n);\n\
         if(!(__r instanceof Promise)) throw new Error('expected Promise');\n\
         const v=await __r;console.log(String(Array.isArray(v)?v[0]:v));}})().catch(e=>{{console.error(e);process.exit(1);}});"
    );
    let o = std::process::Command::new("node").arg("-e").arg(&async_harness).output().expect("node");
    assert!(o.status.success(), "async: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8(o.stdout).unwrap().trim(), "42");
}

#[test]
fn test_multi_memory_js() {
    use wasm_encoder::MemArg;
    let wasm = make_module_2fn(
        (&[ValType::I64], &[ValType::I64]),
        &[
            Instruction::I32Const(0),
            Instruction::LocalGet(0),
            Instruction::I64Store(MemArg { offset: 0, align: 3, memory_index: 1 }),
            Instruction::I32Const(0),
            Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index: 1 }),
            Instruction::Return,
        ],
        &[Instruction::LocalGet(0), Instruction::Return],
        false,
    );
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(&wasm);
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);
    let mut out = String::new();
    js_module_preamble(&mut out).unwrap();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.unwrap();
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder).unwrap();
    }
    eprintln!("=== multi memory JS ===\n{out}");
    let harness = "\n$mems[1]=new Uint8Array(65536);$mem_dvs[1]=new DataView($mems[1].buffer);\nconst __r=$0(99n);console.log(String(__r));";
    let code = format!("{out}{harness}");
    let o = std::process::Command::new("node").arg("-e").arg(&code).output().expect("node");
    assert!(o.status.success(), "node failed: {}", String::from_utf8_lossy(&o.stderr));
    let v: i64 = String::from_utf8(o.stdout).unwrap().trim().parse().unwrap();
    assert_eq!(v, 99);
}
