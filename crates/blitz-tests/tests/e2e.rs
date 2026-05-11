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

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use portal_solutions_blitz_common::{
    dce_pass,
    ops::mach_operators,
    wasmparser::{self, DataKind, FuncType as WpFuncType, Operator},
    wasm_encoder::{
        self,
        reencode::RoundtripReencoder,
        CodeSection, DataSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
        MemorySection, MemoryType, Module, TypeSection, ValType,
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
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &mut state, &op, &mut reencoder)
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
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &mut state, &op, &mut reencoder)
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
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &mut state, &op, &mut reencoder)
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
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &mut state, &op, &mut reencoder)
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
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &mut state, &op, &mut reencoder).unwrap();
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
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &imports_ref, &mut state, &op, &mut reencoder)
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
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &imports_ref, &mut state, &op, &mut reencoder)
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

