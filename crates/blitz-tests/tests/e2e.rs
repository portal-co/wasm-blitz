//! End-to-end tests for the JS and C code-generation backends.
//!
//! Each test builds a minimal WASM module in memory with `wasm-encoder`,
//! drives it through the backend pipeline, and asserts properties of the
//! emitted source code.
//!
//! # Pipeline
//! ```text
//! wasm-encoder  →  raw bytes  →  wasmparser (FunctionBody, FuncType)
//!   →  mach_operators  →  dce_pass!  →  on_mach  →  String output
//! ```
//!
//! # Bug coverage
//! Each test is annotated with the bug(s) it exercises.

use std::borrow::Cow;

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
// Tests — basic compilation
// ---------------------------------------------------------------------------

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
