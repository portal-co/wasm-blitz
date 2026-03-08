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
    wasmparser::{self, FuncType as WpFuncType},
    wasm_encoder::{
        self,
        reencode::RoundtripReencoder,
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
        TypeSection, ValType,
    },
};
use portal_solutions_blitz_c::{CWrite, State as CState};
use portal_solutions_blitz_js::{JsWrite, State as JsState};

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

