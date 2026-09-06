//! Spec-test suite entry point. One `#[test]` per (wast file × backend).
//!
//! Phase 1: JS backend only, hand-picked file set. See `docs/spectests-plan.md`.

mod spec;

use std::path::PathBuf;
use std::sync::OnceLock;

#[expect(dead_code)] // documented file list; tests use the macro expansion below
const PHASE1_FILES: &[&str] = &[
    "const",
    "local_get",
    "block",
    "loop",
    "if",
    "br",
    "br_if",
    "br_table",
    "call",
    "labels",
    "forward",
    "fac",
    "stack",
    "int_exprs",
    "int_literals",
    "left-to-right",
];

fn spec_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("BLITZ_SPEC_DIR") {
        if !d.is_empty() {
            return Some(PathBuf::from(d));
        }
    }
    if let Some(d) = std::env::var_os("SPECTESTS_DIR") {
        if !d.is_empty() {
            return Some(PathBuf::from(d));
        }
    }
    // Idempotent local fetch location.
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/spec");
    if target.join("test/core").is_dir() {
        return Some(target);
    }
    None
}

fn baseline() -> &'static spec::Baseline {
    static BASELINE: OnceLock<spec::Baseline> = OnceLock::new();
    BASELINE.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/spec/baseline.toml");
        spec::Baseline::load(&path)
    })
}

fn run_phase1(file: &str) {
    let log = spec::Logger::from_env();
    let Some(dir) = spec_dir() else {
        eprintln!(
            "skipping spectests: no spec suite found. Set BLITZ_SPEC_DIR or run \
             `git clone --depth 1 https://github.com/WebAssembly/spec target/spec`"
        );
        return;
    };
    let path = dir.join("test/core").join(format!("{file}.wast"));
    let result = spec::run_wast_file(&path, &log, baseline());
    log.begin_batch(&format!("spectests::js::{file}"));
    log.end_batch();
    assert!(
        result.fail_new.is_empty(),
        "{file}: {} new failing assertion(s) at directive indices {:?} \
         (pass={}, known-fail={}, skip={}). Re-run with PORTAL_LOG_JSON=1 for detail.",
        result.fail_new.len(),
        result.fail_new,
        result.pass,
        result.fail_known.len(),
        result.skip
    );
}

macro_rules! spectest_js {
    ($($name:ident => $file:expr),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_phase1($file);
            }
        )*
    };
}

spectest_js! {
    js_const => "const",
    js_local_get => "local_get",
    js_block => "block",
    js_loop => "loop",
    js_if => "if",
    js_br => "br",
    js_br_if => "br_if",
    js_br_table => "br_table",
    js_call => "call",
    js_labels => "labels",
    js_forward => "forward",
    js_fac => "fac",
    js_stack => "stack",
    js_int_exprs => "int_exprs",
    js_int_literals => "int_literals",
    js_left_to_right => "left-to-right",
}

// ---- C backend --------------------------------------------------------------

fn run_phase1_c(file: &str) {
    let log = spec::Logger::from_env();
    let Some(dir) = spec_dir() else {
        eprintln!("skipping spectests: no spec suite found (set BLITZ_SPEC_DIR)");
        return;
    };
    let path = dir.join("test/core").join(format!("{file}.wast"));
    let result = spec::run_wast_file_backend(&path, &log, baseline(), spec::Backend::C);
    assert!(
        result.fail_new.is_empty(),
        "[c] {file}: {} new failing assertion(s) at {:?} (pass={}, known={}, skip={})",
        result.fail_new.len(),
        result.fail_new,
        result.pass,
        result.fail_known.len(),
        result.skip
    );
}

macro_rules! spectest_c {
    ($($name:ident => $file:expr),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_phase1_c($file);
            }
        )*
    };
}

spectest_c! {
    c_const => "const",
    c_local_get => "local_get",
    c_block => "block",
    c_loop => "loop",
    c_if => "if",
    c_br => "br",
    c_br_if => "br_if",
    c_br_table => "br_table",
    c_call => "call",
    c_labels => "labels",
    c_forward => "forward",
    c_fac => "fac",
    c_stack => "stack",
    c_int_exprs => "int_exprs",
    c_int_literals => "int_literals",
    c_left_to_right => "left-to-right",
}

//#[test]
//fn _dump_js() {}

// ---- Native backends (phase A of docs/native-spectests-plan.md) ------------

/// Sentinel-page smoke (phase A): compile a trivial module with one import,
/// run it, and confirm the run completes (import not called by this module,
/// but the sentinel trampolines + data area must not break execution).
#[test]
fn native_smoke_import_service() {
    // (import "spectest" "print_i32" (func (param i32)))
    // (func (export "run") (param i64) (result i64) local.get 0)
    let mut module = wasm_encoder::Module::new();
    let mut types = wasm_encoder::TypeSection::new();
    types.ty().function([wasm_encoder::ValType::I32], []);
    types
        .ty()
        .function([wasm_encoder::ValType::I64], [wasm_encoder::ValType::I64]);
    module.section(&types);
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import(
        "spectest",
        "print_i32",
        wasm_encoder::EntityType::Function(0),
    );
    module.section(&imports);
    let mut funcs = wasm_encoder::FunctionSection::new();
    funcs.function(1);
    module.section(&funcs);
    let mut exports = wasm_encoder::ExportSection::new();
    exports.export("run", wasm_encoder::ExportKind::Func, 1);
    module.section(&exports);
    let mut code = wasm_encoder::CodeSection::new();
    let mut f = wasm_encoder::Function::new([]);
    f.instruction(&wasm_encoder::Instruction::LocalGet(0));
    f.instruction(&wasm_encoder::Instruction::End);
    code.function(&f);
    module.section(&code);
    let wasm = module.finish();

    for arch in [
        spec::native_exec::NativeArch::X86_64,
        spec::native_exec::NativeArch::AArch64,
    ] {
        let bin = spec::native_exec::compile_module(
            &wasm,
            arch,
            &[("spectest".into(), "print_i32".into())],
            None,
            &[],
        )
        .unwrap_or_else(|e| panic!("{arch:?} compile failed: {e:?}"));

        let mut calls: Vec<Vec<u64>> = Vec::new();
        {
            let mut host = |slot: usize, args: &[u64]| -> Result<Vec<u64>, String> {
                assert_eq!(slot, 0, "one import");
                // print_i32(i32): the arg slot is the last pushed; with the
                // operand window the host picks arg 0 of its own arity.
                calls.push(args[..1].to_vec());
                Ok(vec![])
            };
            let outcome = spec::native_exec::run_module(
                arch,
                &bin,
                bin.entry_off,
                &[0x1234_5678],
                1,
                10_000_000,
                &mut host,
            );
            match outcome {
                spec::native_exec::RunOutcome::ReturnedMulti(mut vs) => {
                    assert_eq!(vs.remove(0) as u32, 0x1234_5678);
                }
                spec::native_exec::RunOutcome::Returned(_) => {
                    panic!("expected multi-result path");
                }
                spec::native_exec::RunOutcome::Trapped(e) => {
                    panic!("{arch:?} trapped: {e}")
                }
            }
        }
        assert_eq!(calls.len(), 0, "print_i32 not called by this module");
    }
}

/// Sentinel-page smoke, import actually called: the module forwards its param
/// to `print_i32`, then returns a constant. The hook must service the import
/// and emulation must resume correctly afterward.
#[test]
fn native_smoke_import_called() {
    // (import "spectest" "print_i32" (func $p (param i32)))
    // (func (export "run") (param i64) (result i64)
    //   local.get 0  i32.wrap_i64  call $p
    //   i64.const 77)
    let mut module = wasm_encoder::Module::new();
    let mut types = wasm_encoder::TypeSection::new();
    types.ty().function([wasm_encoder::ValType::I32], []);
    types
        .ty()
        .function([wasm_encoder::ValType::I64], [wasm_encoder::ValType::I64]);
    module.section(&types);
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import(
        "spectest",
        "print_i32",
        wasm_encoder::EntityType::Function(0),
    );
    module.section(&imports);
    let mut funcs = wasm_encoder::FunctionSection::new();
    funcs.function(1);
    module.section(&funcs);
    let mut exports = wasm_encoder::ExportSection::new();
    exports.export("run", wasm_encoder::ExportKind::Func, 1);
    module.section(&exports);
    let mut code = wasm_encoder::CodeSection::new();
    let mut f = wasm_encoder::Function::new([]);
    f.instruction(&wasm_encoder::Instruction::LocalGet(0));
    f.instruction(&wasm_encoder::Instruction::I32WrapI64);
    f.instruction(&wasm_encoder::Instruction::Call(0));
    f.instruction(&wasm_encoder::Instruction::I64Const(77));
    f.instruction(&wasm_encoder::Instruction::End);
    code.function(&f);
    module.section(&code);
    let wasm = module.finish();

    for arch in [
        spec::native_exec::NativeArch::X86_64,
        spec::native_exec::NativeArch::AArch64,
    ] {
        let bin = spec::native_exec::compile_module(
            &wasm,
            arch,
            &[("spectest".into(), "print_i32".into())],
            None,
            &[],
        )
        .unwrap_or_else(|e| panic!("{arch:?} compile failed: {e:?}"));

        let mut seen: Option<u64> = None;
        {
            let mut host = |slot: usize, args: &[u64]| -> Result<Vec<u64>, String> {
                assert_eq!(slot, 0, "one import");
                // print_i32(i32): psABI arg 0 (RDI for x86-64).
                seen = Some(args[0]);
                Ok(vec![])
            };
            let outcome = spec::native_exec::run_module(
                arch,
                &bin,
                bin.entry_off,
                &[0x1234_5678],
                1,
                10_000_000,
                &mut host,
            );
            match outcome {
                spec::native_exec::RunOutcome::ReturnedMulti(mut vs) => {
                    assert_eq!(vs.remove(0), 77, "code after the import call must run");
                }
                spec::native_exec::RunOutcome::Returned(_) => {
                    panic!("expected multi-result path");
                }
                spec::native_exec::RunOutcome::Trapped(e) => {
                    panic!("{arch:?} trapped: {e}")
                }
            }
        }
        assert_eq!(
            seen,
            Some(0x1234_5678),
            "import not called with wrapped arg"
        );
    }
}

/// Sentinel trap: `unreachable` compiles to `hlt` and the driver reports
/// `RunOutcome::Trapped` rather than panicking.
#[test]
fn native_smoke_unreachable_trap() {
    // (func (export "run") (result i64) unreachable)
    let mut module = wasm_encoder::Module::new();
    let mut types = wasm_encoder::TypeSection::new();
    types.ty().function([], [wasm_encoder::ValType::I64]);
    module.section(&types);
    let mut funcs = wasm_encoder::FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);
    let mut exports = wasm_encoder::ExportSection::new();
    exports.export("run", wasm_encoder::ExportKind::Func, 0);
    module.section(&exports);
    let mut code = wasm_encoder::CodeSection::new();
    let mut f = wasm_encoder::Function::new([]);
    f.instruction(&wasm_encoder::Instruction::Unreachable);
    f.instruction(&wasm_encoder::Instruction::End);
    code.function(&f);
    module.section(&code);
    let wasm = module.finish();

    let arch = spec::native_exec::NativeArch::X86_64;
    let bin = spec::native_exec::compile_module(&wasm, arch, &[], None, &[])
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let mut host = |_slot: usize, _args: &[u64]| -> Result<Vec<u64>, String> { Ok(vec![]) };
    match spec::native_exec::run_module(arch, &bin, bin.entry_off, &[], 1, 10_000_000, &mut host) {
        spec::native_exec::RunOutcome::Trapped(_) => {}
        spec::native_exec::RunOutcome::ReturnedMulti(vs) => {
            panic!("unreachable must trap, returned {vs:?}");
        }
        spec::native_exec::RunOutcome::Returned(_) => {
            panic!("unreachable must trap (single-result path)");
        }
        spec::native_exec::RunOutcome::ReturnedMulti(_) => {
            panic!("unreachable returned values");
        }
    }
}

// ---- Native backend spectests (Unicorn x86-64; phase B) ---------------------

fn run_phase1_native(file: &str, backend: spec::Backend) {
    let log = spec::Logger::from_env();
    let Some(dir) = spec_dir() else {
        eprintln!("skipping spectests: no spec suite found (set BLITZ_SPEC_DIR)");
        return;
    };
    let path = dir.join("test/core").join(format!("{file}.wast"));
    let result = spec::run_wast_file_backend(&path, &log, baseline(), backend);
    assert!(
        result.fail_new.is_empty(),
        "[native] {file}: {} new failing assertion(s) at {:?} (pass={}, known={}, skip={})",
        result.fail_new.len(),
        result.fail_new,
        result.pass,
        result.fail_known.len(),
        result.skip
    );
}

macro_rules! spectest_native {
    ($($name:ident => $file:expr),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_phase1_native($file, spec::Backend::NativeX86);
            }
            paste::paste! {
                #[test]
                fn [<$name _aarch64>]() {
                    run_phase1_native($file, spec::Backend::NativeAArch64);
                }
            }
        )*
    };
}

spectest_native! {
    native_const => "const",
    native_local_get => "local_get",
    native_block => "block",
    native_loop => "loop",
    native_if => "if",
    native_br => "br",
    native_br_if => "br_if",
    native_br_table => "br_table",
    native_call => "call",
    native_labels => "labels",
    native_forward => "forward",
    native_fac => "fac",
    native_stack => "stack",
    native_int_exprs => "int_exprs",
    native_int_literals => "int_literals",
    native_left_to_right => "left-to-right",
}
