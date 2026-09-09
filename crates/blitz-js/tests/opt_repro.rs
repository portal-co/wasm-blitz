//! Opt-mode repro + execution validation for the blitz-js stack-depth fixes:
//! (1) values live on the stack across structured boundaries, (2) select pops
//! exactly 3, (3) conditional branches don't corrupt the fallthrough depth.
//! Each test also EXECUTES the emitted opt-mode ESM in Node to prove the
//! tracking produces correct code, not just non-crashing code.
use portal_solutions_blitz_common::wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    Module, TypeSection, ValType, reencode::RoundtripReencoder,
};
use portal_solutions_blitz_common::{dce_pass, ops::mach_operators, wasmparser};
use portal_solutions_blitz_js::{JsWrite, State as JsState, js_module_preamble_esm};

fn build(params: &[ValType], instrs: &[Instruction]) -> Vec<u8> {
    let mut m = Module::new();
    let mut t = TypeSection::new();
    t.ty().function(params.iter().cloned(), [ValType::I32]);
    m.section(&t);
    let mut fs = FunctionSection::new();
    fs.function(0);
    m.section(&fs);
    let mut e = ExportSection::new();
    e.export("f", ExportKind::Func, 0);
    m.section(&e);
    let mut c = CodeSection::new();
    let mut f = Function::new([]);
    for i in instrs {
        f.instruction(i);
    }
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);
    c.function(&f);
    m.section(&c);
    m.finish()
}

/// Compile `wasm` to opt-mode ESM JavaScript (exports included).
fn compile_opt(wasm: &[u8]) -> String {
    let mut sigs_wp = Vec::new();
    let mut fsigs = Vec::new();
    for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        match p {
            wasmparser::Payload::TypeSection(r) => {
                for g in r.into_iter().flatten() {
                    for st in g.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) = st.composite_type.inner {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::FunctionSection(r) => fsigs.extend(r.into_iter().flatten()),
            _ => {}
        }
    }
    let sigs_enc: Vec<_> = sigs_wp
        .iter()
        .cloned()
        .map(|ft| portal_solutions_blitz_common::wasm_encoder::FuncType::try_from(ft).unwrap())
        .collect();
    let mut exports = Vec::new();
    for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(r) = p {
            for e in r.into_iter().flatten() {
                if matches!(e.kind, wasmparser::ExternalKind::Func) {
                    exports.push((e.index, e.name.to_owned()));
                }
            }
        }
    }
    let mut bodies = Vec::new();
    for p in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(b) = p {
            bodies.push(b);
        }
    }
    let ops = dce_pass!(mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies, &fsigs, &sigs_wp, 0
    ));
    let mut out = String::new();
    js_module_preamble_esm(&mut out).unwrap();
    let mut state = JsState::default();
    state.enable_opt(portal_solutions_blitz_opt::OptState::default);
    let mut re = RoundtripReencoder;
    for op in ops {
        JsWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op.unwrap(), &mut re)
            .unwrap();
    }
    portal_solutions_blitz_js::js_emit_exports_esm(
        &mut out,
        &exports
            .iter()
            .map(|(i, n)| (*i, n.as_str()))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    out
}

/// Run the compiled opt-mode ESM in Node, calling export `f` with i64 args,
/// returning the numeric result. Fails closed if Node is unavailable.
fn run_f(js: &str, args: &[i64]) -> i128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "blitz_opt_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("m.mjs"), js).unwrap();
    let driver = "import { f } from './m.mjs';\n\
                  const r = f(...process.argv.slice(2).map(BigInt));\n\
                  const v = Array.isArray(r) ? r[0] : r;\n\
                  console.log(v.toString());\n";
    let drv = dir.join("d.mjs");
    std::fs::write(&drv, driver).unwrap();
    let out = std::process::Command::new("node")
        .arg(&drv)
        .args(args.iter().map(|a| a.to_string()))
        .output()
        .expect("node must be available to validate opt-mode output");
    assert!(
        out.status.success(),
        "node run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<i128>()
        .expect("numeric result")
}

// i32.const 1; block; i32.const 2; drop; end;  -> outer const live across the block
#[test]
fn repro_value_across_block() {
    let wasm = build(
        &[],
        &[
            Instruction::I32Const(1),
            Instruction::Block(BlockType::Empty),
            Instruction::I32Const(2),
            Instruction::Drop,
            Instruction::End,
        ],
    );
    assert_eq!(run_f(&compile_opt(&wasm), &[]), 1);
}

// Two outer values live across a block; a `br` inside carries a result out.
#[test]
fn repro_br_result_with_outer_values() {
    let wasm = build(
        &[],
        &[
            Instruction::I32Const(1),
            Instruction::I32Const(2),
            Instruction::Block(BlockType::Result(ValType::I32)),
            Instruction::I32Const(3),
            Instruction::Br(0),
            Instruction::End,
            Instruction::I32Add,
            Instruction::I32Add,
        ],
    );
    assert_eq!(run_f(&compile_opt(&wasm), &[]), 6);
}

// select pops exactly 3 (the old arm popped 5 and underflowed the tracker).
#[test]
fn repro_select_depth() {
    let wasm = build(
        &[],
        &[
            Instruction::I32Const(10),
            Instruction::I32Const(20),
            Instruction::I32Const(1),
            Instruction::Select, // -> 10
            Instruction::I32Const(40),
            Instruction::I32Const(50),
            Instruction::I32Const(0),
            Instruction::Select, // -> 50
            Instruction::I32Add, // -> 60
        ],
    );
    assert_eq!(run_f(&compile_opt(&wasm), &[]), 60);
}

// A conditional branch must not corrupt the fallthrough stack depth.
#[test]
fn repro_br_if_fallthrough_keeps_stack() {
    let wasm = build(
        &[ValType::I32],
        &[
            Instruction::I32Const(100),
            Instruction::LocalGet(0),
            Instruction::BrIf(0),
            Instruction::Drop,
            Instruction::I32Const(7),
        ],
    );
    let js = compile_opt(&wasm);
    assert_eq!(run_f(&js, &[1]), 100); // taken
    assert_eq!(run_f(&js, &[0]), 7); // fallthrough
}

// A loop with a back-edge: the continue branch and the loop-exit must agree
// on the tracked depth. Sum 1..=n.
#[test]
fn repro_loop_backedge() {
    let mut f = Function::new([(1, ValType::I32)]); // locals: [0]=n(param), [1]=acc
    f.instruction(&Instruction::Loop(BlockType::Empty));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(1));
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::I32Sub);
    f.instruction(&Instruction::LocalTee(0));
    f.instruction(&Instruction::BrIf(0)); // continue while n != 0
    f.instruction(&Instruction::End); // end loop
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Return);
    f.instruction(&Instruction::End);

    let mut m = Module::new();
    let mut t = TypeSection::new();
    t.ty().function([ValType::I32], [ValType::I32]);
    m.section(&t);
    let mut fs = FunctionSection::new();
    fs.function(0);
    m.section(&fs);
    let mut e = ExportSection::new();
    e.export("f", ExportKind::Func, 0);
    m.section(&e);
    let mut c = CodeSection::new();
    c.function(&f);
    m.section(&c);

    let js = compile_opt(&m.finish());
    assert_eq!(run_f(&js, &[4]), 10); // 4+3+2+1
    assert_eq!(run_f(&js, &[10]), 55);
}
