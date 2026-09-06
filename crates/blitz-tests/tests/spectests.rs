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

//#[test]
//fn _dump_js() {}
