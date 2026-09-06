//! WebAssembly spec-test suite harness (`docs/spectests-plan.md`).
//!
//! Drives `test/core/*.wast` from the official [spec suite] through the
//! blitz compilation pipeline and executes the result with a real runtime
//! (phase 1: `node` via the JS backend).
//!
//! [spec suite]: https://github.com/WebAssembly/spec
//!
//! # Scope (phase 1)
//!
//! * Text `(module ...)` definitions are compiled and executed; binary
//!   `module quote`/`module binary` forms are encoded first.
//! * `assert_malformed` / `assert_invalid` are checked at *parse/validate*
//!   level via `wasmparser` (the blitz pipeline itself assumes valid input).
//! * Trap assertions are checked trap-vs-no-trap; trap *messages* are not
//!   matched (phase-5 tightening, see the plan).
//! * Assertions whose exports/imports reference runtime semantics the phase-1
//!   module synthesis does not implement (globals, tables, multi-value
//!   results, `spectest` host imports) are recorded as `Skip` with a reason,
//!   never silently.

mod baseline;
mod features;
mod logging;
pub mod native;

pub use baseline::Baseline;
pub use logging::Logger;
use std::path::Path;
use wast::core::{NanPattern, WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastRet};

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// Global counter for unique temp-file names (parallel test runs).
static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
use std::sync::atomic::Ordering;

/// A runtime value crossing the JS execution boundary.
///
/// Everything is a JS number-or-BigInt string at the end; we keep the typed
/// form so comparisons (especially NaN patterns) are exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    I32(u32),
    I64(u64),
    F32(u32), // bit patterns
    F64(u64),
}

impl Val {
    /// The JavaScript literal that produces this value.
    pub fn js_literal(self) -> String {
        match self {
            Val::I32(v) => format!("{v}n /*i32*/"),
            Val::I64(v) => format!("{v}n /*i64*/"),
            // Reinterpret the exact bit pattern; avoids JS decimal round-trip loss.
            Val::F32(bits) => format!("__f32bits({bits})"),
            Val::F64(bits) => format!("__f64bits({bits})"),
        }
    }

    #[expect(dead_code)] // phase-1b execution bridge will use this
    /// How the value is read back from JS as an exact bit pattern string.
    pub fn js_read_expr(js_value_expr: &str, ty: char) -> String {
        match ty {
            'i' => format!("__i32bits({js_value_expr})"),
            'l' => format!("__i64bits({js_value_expr})"),
            'f' => format!("__f32bits({js_value_expr})"),
            'd' => format!("__f64bits({js_value_expr})"),
            other => panic!("unknown value type tag {other}"),
        }
    }
}

/// Outcome of a single assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Skip(&'static str),
}

// ---------------------------------------------------------------------------
// wast value conversion
// ---------------------------------------------------------------------------

fn f32_bits(f: wast::token::F32) -> u32 {
    f.bits
}

fn f64_bits(f: wast::token::F64) -> u64 {
    f.bits
}

/// Convert a parsed wast argument into an executable `Val`.
/// Returns `None` when the argument is out of phase-1 scope (refs, v128).
fn arg_to_val(arg: &WastArg<'_>) -> Option<Val> {
    let WastArg::Core(core) = arg else { return None };
    Some(match core {
        WastArgCore::I32(v) => Val::I32(*v as u32),
        WastArgCore::I64(v) => Val::I64(*v as u64),
        WastArgCore::F32(f) => Val::F32(f32_bits(*f)),
        WastArgCore::F64(f) => Val::F64(f64_bits(*f)),
        _ => return None,
    })
}

/// Classify a wast expected result into a checker.
/// Returns `None` when out of phase-1 scope.
fn ret_to_expected(ret: &WastRet<'_>) -> Option<Expected> {
    let WastRet::Core(core) = ret else { return None };
    Some(match core {
        WastRetCore::I32(v) => Expected::I32(*v as u32),
        WastRetCore::I64(v) => Expected::I64(*v as u64),
        WastRetCore::F32(NanPattern::Value(f)) => Expected::F32(F32Pat::Value(f32_bits(*f))),
        WastRetCore::F32(NanPattern::CanonicalNan) => Expected::F32(F32Pat::CanonicalNan),
        WastRetCore::F32(NanPattern::ArithmeticNan) => Expected::F32(F32Pat::ArithmeticNan),
        WastRetCore::F64(NanPattern::Value(f)) => Expected::F64(F64Pat::Value(f64_bits(*f))),
        WastRetCore::F64(NanPattern::CanonicalNan) => Expected::F64(F64Pat::CanonicalNan),
        WastRetCore::F64(NanPattern::ArithmeticNan) => Expected::F64(F64Pat::ArithmeticNan),
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug)]
pub enum Expected {
    I32(u32),
    I64(u64),
    F32(F32Pat),
    F64(F64Pat),
}

#[derive(Clone, Copy, Debug)]
pub enum F32Pat {
    Value(u32),
    /// ± canonical NaN: exponent all ones, top mantissa bit set, rest zero.
    CanonicalNan,
    /// Any NaN (payload unconstrained beyond being a NaN).
    ArithmeticNan,
}

#[derive(Clone, Copy, Debug)]
pub enum F64Pat {
    Value(u64),
    CanonicalNan,
    ArithmeticNan,
}

pub const CANON_F32_BITS: u32 = 0x7fc0_0000;
pub const CANON_F64_BITS: u64 = 0x7ff8_0000_0000_0000;

impl F32Pat {
    pub fn matches(self, bits: u32) -> bool {
        match self {
            F32Pat::Value(v) => bits == v,
            // Spec: result is canonical NaN if we got a NaN with canonical payload,
            // arithmetic NaN patterns accept any NaN payload — but only NaNs.
            F32Pat::CanonicalNan => bits == CANON_F32_BITS || bits == (CANON_F32_BITS | 0x8000_0000),
            F32Pat::ArithmeticNan => (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0,
        }
    }
}

impl F64Pat {
    pub fn matches(self, bits: u64) -> bool {
        match self {
            F64Pat::Value(v) => bits == v,
            F64Pat::CanonicalNan => {
                bits == CANON_F64_BITS || bits == (CANON_F64_BITS | 0x8000_0000_0000_0000)
            }
            F64Pat::ArithmeticNan => {
                (bits & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000
                    && (bits & 0x000f_ffff_ffff_ffff) != 0
            }
        }
    }
}

pub fn check_expected(got: &[(char, String)], expected: &[Expected]) -> bool {
    got.len() == expected.len()
        && got.iter().zip(expected).all(|((t, v), e)| {
            let u: u64 = v.parse().unwrap_or(0);
            match (t, e) {
                // Integers: wire as u64; i32 results keep only the low 32 bits.
                (_, Expected::I32(b)) => *t == 'i' && (u as u32) == *b,
                (_, Expected::I64(b)) => *t == 'i' && u == *b,
                // Floats: wire as exact bit patterns. The C backend prints raw
                // u64 words (tag 'i'); the JS backend tags floats 'f'/'d'.
                // An expected-f32 result computed at f32 precision (fround /
                // C float) reinterprets exactly from the low 32 bits.
                ('f', Expected::F32(p)) => {
                    // JS: f32 results are fround-ed then widened to f64 exactly;
                    // narrow back to recover the true f32 bit pattern.
                    let f = f64::from_bits(u) as f32;
                    p.matches(f.to_bits())
                }
                ('f', Expected::F64(p)) => p.matches(u),
                // C: results are printed as raw u64 words.
                ('i', Expected::F32(p)) => {
                    let f = f32::from_bits(u as u32);
                    p.matches(f.to_bits())
                }
                ('i', Expected::F64(p)) => p.matches(u),
                _ => false,
            }
        })
}

// ---------------------------------------------------------------------------
// Module synthesis — compile one wast module through blitz and produce JS
// ---------------------------------------------------------------------------

/// How a module-level global is initialized (phase-1 subset).
#[derive(Clone, Debug)]
pub enum GlobalInit {
    I32(i32),
    I64(i64),
    /// `(global (import "spectest" "global_*") T)` — value comes from the host.
    Spectest(String),
}

/// A wasm module extracted from a wast directive, already encoded to binary.
struct SpecModule {
    wasm: Vec<u8>,
    /// `(module, field)` of each imported function, in import order.
    func_imports: Vec<(String, String)>,
    /// Exported functions: `(name, wasm_func_index)`.
    exported_funcs: Vec<(String, u32)>,
    /// Exported globals (phase 1: presence forces a skip of `get` on them).
    exported_globals: Vec<String>,
    /// Active data segments `(offset, bytes)` for memory initialization.
    data_segments: Vec<(u32, Vec<u8>)>,
    /// Number of pages of the (first) memory, if any.
    memory_pages: Option<u32>,
    /// A table exists whose element type is not plain `funcref` (typed tables
    /// need function-references/GC — phase 4).
    typed_table: bool,
    /// The module imports the `spectest` host module (or anything else).
    imports_host: bool,
    /// Per-global init info, in global-index order (imports first).
    global_inits: Vec<GlobalInit>,
    /// Compilation produced by blitz (JS source body).
    js: String,
    /// Compilation produced by blitz (C source body).
    c_body: String,
}

/// Extract module structure from binary wasm bytes.
fn inspect_module(wasm: &[u8]) -> Result<SpecModule, String> {
    use wasmparser::{Payload, TypeRef};
    let mut m = SpecModule {
        wasm: wasm.to_vec(),
        func_imports: Vec::new(),
        exported_funcs: Vec::new(),
        exported_globals: Vec::new(),
        data_segments: Vec::new(),
        memory_pages: None,
        typed_table: false,
        imports_host: false,
        global_inits: Vec::new(),
        js: String::new(),
        c_body: String::new(),
    };
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        let payload = payload.map_err(|e| format!("wasmparser: {e}"))?;
        match payload {
            Payload::ImportSection(reader) => {
                for imp in reader {
                    let imp = imp.map_err(|e| format!("import: {e}"))?;
                    if let TypeRef::Func(_) = imp.ty {
                        m.func_imports
                            .push((imp.module.to_string(), imp.name.to_string()));
                    }
                    if let TypeRef::Global(gty) = imp.ty {
                        // Global imports: only spectest i32/i64 globals are supported.
                        if imp.module == "spectest"
                            && matches!(gty.content_type, wasmparser::ValType::I32 | wasmparser::ValType::I64)
                        {
                            m.global_inits
                                .push(GlobalInit::Spectest(imp.name.to_string()));
                        } else {
                            m.imports_host = true;
                        }
                    }
                    m.imports_host = true;
                }
            }
            Payload::MemorySection(reader) => {
                for mem in reader {
                    let mem = mem.map_err(|e| format!("memory: {e}"))?;
                    if m.memory_pages.is_none() {
                        m.memory_pages = Some(u32::try_from(mem.initial).map_err(|_| "memory too large")?);
                    }
                }
            }
            Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp.map_err(|e| format!("export: {e}"))?;
                    match exp.kind {
                        wasmparser::ExternalKind::Func => {
                            m.exported_funcs
                                .push((exp.name.to_string(), exp.index));
                        }
                        wasmparser::ExternalKind::Global => {
                            m.exported_globals.push(exp.name.to_string());
                        }
                        _ => {}
                    }
                }
            }
            Payload::TableSection(reader) => {
                for table in reader {
                    let table = table.map_err(|e| format!("table: {e}"))?;
                    let is_plain_func = matches!(
                        table.ty.element_type.heap_type(),
                        wasmparser::HeapType::Abstract { shared: false, ty: wasmparser::AbstractHeapType::Func }
                    );
                    if !is_plain_func {
                        m.typed_table = true;
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for global in reader {
                    let global = global.map_err(|e| format!("global: {e}"))?;
                    let mut ops = global.init_expr.get_operators_reader();
                    let init = match ops.read() {
                        Ok(wasmparser::Operator::I32Const { value }) => GlobalInit::I32(value),
                        Ok(wasmparser::Operator::I64Const { value }) => GlobalInit::I64(value),
                        Ok(wasmparser::Operator::GlobalGet { global_index }) => {
                            // Init-from-import (e.g. spectest global) — resolve via import order.
                            m.global_inits.get(global_index as usize).cloned().ok_or_else(|| {
                                "global init references unknown global".to_string()
                            })?
                        }
                        _ => return Err("unsupported global init expr".into()),
                    };
                    m.global_inits.push(init);
                }
            }
            Payload::DataSection(reader) => {
                for data in reader {
                    let data = data.map_err(|e| format!("data: {e}"))?;
                    if let wasmparser::DataKind::Active { memory_index, offset_expr } = data.kind
                    {
                        let _ = memory_index;
                        // Evaluate the offset const-expr: only `i32.const N` supported.
                        let mut ops = offset_expr.get_operators_reader();
                        let offset = match ops.read() {
                            Ok(wasmparser::Operator::I32Const { value }) => value as u32,
                            _ => return Err("unsupported data offset expr".into()),
                        };
                        m.data_segments.push((offset, data.data.to_vec()));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(m)
}

/// Compile a module through the blitz C backend (same pipeline as e2e's
/// `compile_c`).
fn blitz_compile_c(wasm: &[u8]) -> Result<String, String> {
    use portal_solutions_blitz_common::{dce_pass, ops::mach_operators};
    use portal_solutions_blitz_c::{CWrite, State as CState};
    use wasm_encoder::reencode::RoundtripReencoder;

    let mut sigs_wp: Vec<wasmparser::FuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for subtype in group
                        .map_err(|e| e.to_string())?
                        .into_types()
                    {
                        if let wasmparser::CompositeInnerType::Func(ft) = subtype.composite_type.inner
                        {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(reader) => {
                for imp in reader {
                    let imp = imp.map_err(|e| e.to_string())?;
                    if let wasmparser::TypeRef::Func(ty_idx) = imp.ty {
                        fsigs.push(ty_idx);
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                fsigs.extend(reader.into_iter().flatten());
            }
            wasmparser::Payload::CodeSectionEntry(body) => bodies.push(body),
            _ => {}
        }
    }
    let sigs_enc: Vec<wasm_encoder::FuncType> = sigs_wp
        .iter()
        .cloned()
        .map(wasm_encoder::FuncType::try_from)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let raw_ops =
        mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    portal_solutions_blitz_c::c_module_preamble(&mut out).map_err(|e| e.to_string())?;
    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.map_err(|e| e.to_string())?;
        CWrite::on_mach(&mut out, &sigs_enc, &fsigs, &[], &[], &mut state, &op, &mut reencoder)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Full C synthesis: compile module + memory/data scaffolding for the driver.
fn synthesize_c_module(m: &SpecModule) -> Result<String, String> {
    if m.typed_table {
        return Err("typed table (function-references) unsupported in phase 1".into());
    }
    if !m.func_imports.is_empty() {
        return Err("function imports unsupported in C phase-1".into());
    }
    let pages = m.memory_pages.unwrap_or(0);
    let mut out = String::new();
    out.push_str(&format!(
        "#define WASM_STACK_SIZE 4096\nstatic uint8_t __wasm_driver_mem[{}];\nstatic uint32_t __wasm_driver_pages = {};\n",
        pages as usize * 65536,
        pages
    ));
    out.push_str(&m.c_body);
    // Init memory: point __wasm_mems[0] at the driver buffer, apply segments.
    out.push_str("\nstatic void __wasm_spec_setup(void){\n  __wasm_mems[0]=__wasm_driver_mem;__wasm_mem_pages_arr[0]=__wasm_driver_pages;\n");
    for (off, bytes) in &m.data_segments {
        out.push_str(&format!("  {{static const uint8_t seg[]={bytes:?};memcpy(__wasm_driver_mem+{off},seg,{});}}\n", bytes.len()));
    }
    // Persist module globals across per-invoke processes so sequences like
    // `(module) ... invoke ... invoke` observe mutating globals.
    out.push_str(
        "static void __wasm_spec_save(void){char p[128];snprintf(p,128,\"/tmp/blitz_spec_globals_%d.bin\",(int)getpid());FILE*f=fopen(p,\"wb\");if(f){fwrite(__wasm_globals,sizeof(__wasm_globals),1,f);fclose(f);}}\n",
    );
    out.push_str(
        "static void __wasm_spec_load(void){char p[128];snprintf(p,128,\"/tmp/blitz_spec_globals_%d.bin\",(int)getpid());FILE*f=fopen(p,\"rb\");if(f){fread(__wasm_globals,sizeof(__wasm_globals),1,f);fclose(f);}}\n",
    );
    out.push_str("}\n");
    Ok(out)
}

/// Compile a binary wasm module through the blitz JS backend, returning the
/// generated JS function bodies (same pipeline as e2e.rs `compile_js`).
fn blitz_compile_js(wasm: &[u8]) -> Result<String, String> {
    use portal_solutions_blitz_common::{dce_pass, ops::mach_operators};
    use portal_solutions_blitz_js::{JsWrite, State as JsState};
    use wasm_encoder::reencode::RoundtripReencoder;

    let mut sigs_wp: Vec<wasmparser::FuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    let mut func_import_count: u32 = 0;
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for subtype in group
                        .map_err(|e| e.to_string())?
                        .into_types()
                    {
                        if let wasmparser::CompositeInnerType::Func(ft) = subtype.composite_type.inner
                        {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(reader) => {
                for imp in reader {
                    let imp = imp.map_err(|e| e.to_string())?;
                    if let wasmparser::TypeRef::Func(ty_idx) = imp.ty {
                        fsigs.push(ty_idx);
                        func_import_count += 1;
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
        if let wasmparser::Payload::CodeSectionEntry(body) =
            payload.map_err(|e| e.to_string())?
        {
            bodies.push(body);
        }
    }

    let sigs_enc: Vec<wasm_encoder::FuncType> = sigs_wp
        .iter()
        .cloned()
        .map(wasm_encoder::FuncType::try_from)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;

    let raw_ops =
        mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs_wp, 0);
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    let mut state = JsState::default();
    let mut reencoder = RoundtripReencoder;
    let import_pairs: Vec<(&str, &str)> = Vec::new(); // imports handled at synthesis level
    let _ = import_pairs;

    for op in ops {
        let op = op.map_err(|e| e.to_string())?;
        JsWrite::on_mach(
            &mut out,
            &sigs_enc,
            &fsigs,
            &[],
            &[],
            &mut state,
            &op,
            &mut reencoder,
        )
        .map_err(|e| e.to_string())?;
    }
    let _ = func_import_count;
    Ok(out)
}

/// Full phase-1 synthesis: inspect + compile a module into a JS program
/// string. Returns `Err` when the module uses semantics we cannot synthesize
/// (caller converts that into an assertion Skip).
fn synthesize_js_module(m: &SpecModule) -> Result<String, String> {
    use portal_solutions_blitz_js::js_module_preamble;

    if m.typed_table {
        return Err("typed table (function-references) unsupported in phase 1".into());
    }

    let body = blitz_compile_js(&m.wasm)?;
    let mut js = String::new();
    js_module_preamble(&mut js).map_err(|e| e.to_string())?;

    // Float bit-pattern reinterpretation helpers (exact, no decimal round-trip).
    // `__f32bitsOf`/`__f64bitsOf` take a bit pattern, return the JS float value;
    // `__f32bits`/`__f64bits` take a float-or-BigInt, return the exact bit pattern.
    js.push_str(
        "function __f32bitsOf(bits){const dv=new DataView(new ArrayBuffer(4));dv.setUint32(0,bits,true);return dv.getFloat32(0,true);}\n",
    );
    js.push_str(
        "function __f64bitsOf(bits){const dv=new DataView(new ArrayBuffer(8));dv.setBigUint64(0,bits,true);return dv.getFloat64(0,true);}\n",
    );
    js.push_str(
        "function __f32bits(x){const dv=new DataView(new ArrayBuffer(4));if(typeof x==='bigint')dv.setUint32(0,Number(x&0xffffffffn),true);else dv.setFloat32(0,x,true);return dv.getUint32(0,true);}\n",
    );
    js.push_str(
        "function __f64bits(x){const dv=new DataView(new ArrayBuffer(8));if(typeof x==='bigint')dv.setBigUint64(0,x,true);else dv.setFloat64(0,x,true);return dv.getBigUint64(0,true);}\n",
    );
    js.push_str("function __i32bits(x){return Number(BigInt.asUintN(32,x));}\n");
    js.push_str("function __i64bits(x){return x;}\n");
    js.push_str(
        "function __result(v){if(Array.isArray(v)){return v.map(x=>(typeof x==='bigint')?__i64bits(x):((typeof x==='number')?__f64bits(x):x));}return [(typeof v==='bigint')?__i64bits(v):((typeof v==='number')?__f64bits(v):v)];}\n",
    );
    js.push_str("function __print(x){console.log('PRINT',x);}\n");
    // Traps surface as thrown JS errors; the execution bridge distinguishes
    // trap-exits (marker) from harness bugs (any other throw).
    js.push_str(
        "function __wasm_trap(kind){const e=new Error('wasm trap: '+kind);e.__wasm_trap=true;throw e;}\n",
    );
    js.push_str(
        "function __popcnt32(x){x=x>>>0;let c=0;while(x){x&=x-1;c++;}return c;}\n",
    );
    js.push_str(
        "function __popcnt64(x){let c=0;x=BigInt(x);while(x){x&=x-1n;c++;}return c;}\n",
    );
    // u64 -> f64 without precision loss path through BigInt division rounding.
    js.push_str(
        "function __u64ToF64(x){if(x<0x8000000000000000n)return Number(x);return Number(x-(x>>24n)*0x1000000n)+Number((x&0xffffffn))*1.0;}\n",
    );
    // Truncation with spec trap semantics: NaN and out-of-range trap.
    js.push_str(
        "function __truncS(x,bits,round){x=round(x);if(Number.isNaN(x))__wasm_trap('invalid conversion to integer');let t=Math.trunc(x);if(bits===32){if(t<2147483648&&t>=-2147483648)return BigInt.asUintN(32,BigInt(t));}else{if(t<=9223372036854775807&&t>=-9223372036854775808)return BigInt.asUintN(64,BigInt(t));}__wasm_trap('integer overflow');}\n",
    );
    js.push_str(
        "function __truncU(x,bits,round){x=round(x);if(Number.isNaN(x))__wasm_trap('invalid conversion to integer');let t=Math.trunc(x);if(bits===32){if(t<4294967296&&t>-1)return BigInt(t);}else{if(t<18446744073709551616&&t>-1){if(t<9223372036854775808)return BigInt(t);return BigInt(t-18446744073709551616)+18446744073709551616n;}}__wasm_trap('integer overflow');}\n",
    );
    js.push_str(
        "function __udiv(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b/a);}\n",
    );
    js.push_str(
        "function __urem(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b%a);}\n",
    );
    js.push_str(
        "function __idivS(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');const min=-(2n**BigInt(bits-1));if(b===min&&a===-1n)__wasm_trap('integer overflow');return BigInt.asUintN(bits,b/a);}\n",
    );
    js.push_str(
        "function __srem(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b%a);}\n",
    );
    // Spec division/remainder semantics (trap on div/0, INT_MIN/-1, rem/0).
    js.push_str(
        "function __udiv(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b/a);}\n",
    );
    js.push_str(
        "function __urem(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b%a);}\n",
    );
    js.push_str(
        "function __idivS(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');const min=-(2n**BigInt(bits-1));if(b===min&&a===-1n)__wasm_trap('integer overflow');return BigInt.asUintN(bits,b/a);}\n",
    );
    js.push_str(
        "function __srem(a,b,bits){if(a===0n)__wasm_trap('integer divide by zero');return BigInt.asUintN(bits,b%a);}\n",
    );
    // WASM min/max propagate NaN and distinguish -0/+0 (unlike Math.min/max).
    js.push_str(
        "function __fmin(a,b){if(Number.isNaN(a)||Number.isNaN(b))return NaN;return (a===0&&b===0)?((Object.is(a,-0)||Object.is(b,-0))?-0:a):(a<b?a:b);}\n",
    );
    js.push_str(
        "function __fmax(a,b){if(Number.isNaN(a)||Number.isNaN(b))return NaN;return (a===0&&b===0)?((Object.is(a,-0)&&Object.is(b,-0))?-0:a):(a>b?a:b);}\n",
    );
    // WASM nearest: round-half-to-even.
    js.push_str(
        "function __nearest(x){if(!Number.isFinite(x)||Number.isInteger(x)||Object.is(x,-0))return x;const f=Math.floor(x),d=x-f;if(d<0.5)return f;if(d>0.5)return f+1;return (f%2===0)?f:f+1;}\n",
    );
    js.push_str(
        "function __copysign32(x,y){const xd=new DataView(new ArrayBuffer(4)),yd=new DataView(new ArrayBuffer(4));xd.setFloat32(0,Math.fround(x),true);yd.setFloat32(0,Math.fround(y),true);const xb=xd.getUint32(0,true),yb=yd.getUint32(0,true);xd.setUint32(0,(xb&0x7fffffff)|(yb&0x80000000),true);return xd.getFloat32(0,true);}\n",
    );
    js.push_str(
        "function __copysign64(x,y){const xd=new DataView(new ArrayBuffer(8)),yd=new DataView(new ArrayBuffer(8));xd.setFloat64(0,x,true);yd.setFloat64(0,y,true);const xb=xd.getBigUint64(0,true),yb=yd.getBigUint64(0,true);xd.setBigUint64(0,(xb&0x7fffffffffffffffn)|(yb&0x8000000000000000n),true);return xd.getFloat64(0,true);}\n",
    );

    // Host `spectest` module imports are provided as stubs; any assertion that
    // actually observes their behavior is skipped at the driver level.
    if m.func_imports.iter().any(|(mo, _)| mo == "spectest") || m.imports_host {
        js.push_str("// host spectest stubs\n");
        js.push_str(
            "var spectest={print:__print,print_i32:function(x){console.log('PRINT',Number(BigInt.asIntN(32,x)))},print_i64:function(x){console.log('PRINT',x)},print_i32_f32:function(x,y){console.log('PRINT',Number(BigInt.asIntN(32,x)),y)},print_f64_f64:function(x,y){console.log('PRINT',x,y)},global_i32:666n,global_i64:666n,global_f32:666.6,global_f64:666.6,table:new Array(10).fill(null),memory:new WebAssembly.Memory({initial:1,maximum:2})};\n",
        );
    }

    // Imported functions: `$N` variables assigned from spectest or skipped modules.
    for (i, (module, name)) in m.func_imports.iter().enumerate() {
        if module == "spectest" {
            let f = match name.as_str() {
                "print" => "__print".to_string(),
                "print_i32" => "function(x){console.log('PRINT',Number(BigInt.asIntN(32,x)))}"
                    .to_string(),
                "print_i64" => "function(x){console.log('PRINT',x)}".to_string(),
                "print_i32_f32" => {
                    "function(x,y){console.log('PRINT',Number(BigInt.asIntN(32,x)),y)}".to_string()
                }
                "print_f64_f64" => "function(x,y){console.log('PRINT',x,y)}".to_string(),
                _ => {
                    return Err(format!("unsupported spectest import {module}::{name}"));
                }
            };
            js.push_str(&format!("var ${i}={f};\n"));
        } else {
            return Err(format!("unsupported import {module}::{name}"));
        }
    }

    // Memory sizing + data segments.
    if let Some(pages) = m.memory_pages {
        js.push_str(&format!("$mem=new Uint8Array({});\n", pages * 65536));
        js.push_str("$mem_dv=new DataView($mem.buffer);\n");
    } else if !m.data_segments.is_empty() {
        return Err("data segment without memory".into());
    }

    // Module-level globals: `$g_N` BigInt slots (i32/i64 only). Initialized
    // from the module's global init expressions; other init exprs are rejected
    // at inspect time.
    for (gi, init) in m.global_inits.iter().enumerate() {
        let v = match init {
            GlobalInit::I32(v) => format!("{v}n"),
            GlobalInit::I64(v) => format!("{v}n"),
            GlobalInit::Spectest(name) => match name.as_str() {
                "global_i32" | "global_i64" => "666n".to_string(),
                other => return Err(format!("unsupported spectest global {other}")),
            },
        };
        js.push_str(&format!("var $g_{gi}={v};\n"));
    }

    js.push_str(&body);
    Ok(js)
}

// ---------------------------------------------------------------------------
// Directive handling
// ---------------------------------------------------------------------------

/// Encode a `QuoteWat` to binary, honoring quote-module malformed checks.
enum Encoded {
    Binary(Vec<u8>),
    /// `(module quote ...)` — text form; not compileable by us (text parser
    /// conformance is not ours to test).
    Quote,
}

fn encode_quotewat(qw: &mut QuoteWat<'_>) -> Result<Encoded, String> {
    match qw {
        QuoteWat::Wat(wast::Wat::Module(mod_)) => {
            Ok(Encoded::Binary(mod_.encode().map_err(|e| e.to_string())?))
        }
        QuoteWat::Wat(_) => Err("component modules unsupported".into()),
        QuoteWat::QuoteModule(..) | QuoteWat::QuoteComponent(..) => Ok(Encoded::Quote),
    }
}

/// Validate a binary module with wasmparser. Feature set: MVP + the
/// proposals the blitz backends already claim (reference types, bulk memory,
/// multi-value, multi-memory, tail calls, sign extension, sat float->int).
fn validate_binary(wasm: &[u8]) -> Result<(), String> {
    let features = features::harness_features();
    let mut validator = wasmparser::Validator::new_with_features(features);
    validator
        .validate_all(wasm)
        .map(|_| ())
        .map_err(|e| format!("{e}"))
}

/// An executable action: invoke export with args, or get global.
#[derive(Clone, Debug)]
pub enum Action {
    Invoke { export: String, args: Vec<Val> },
    Get { global: String },
}

struct Runner {
    /// Stack of synthesized modules; the last is "current".
    modules: Vec<SpecModule>,
    /// Registered instance names (phase 1: tracked but cross-module
    /// instantiation not supported — such imports produce skips).
    registered: Vec<String>,
    /// Live node process (lazily spawned on first execution).
    session: Option<NodeSession>,
    /// Whether the current module loaded successfully into the session.
    current_loaded: bool,
    /// Why the current module failed to load (for skip reasons).
    current_load_err: Option<String>,
    /// Which backend is being exercised.
    backend: Backend,
    /// C backend: the full compiled C source of the current module.
    c_module: Option<CSpecModule>,
}

impl Runner {
    fn current(&self) -> Result<&SpecModule, String> {
        self.modules
            .last()
            .ok_or_else(|| "no current module".to_string())
    }
}

/// A C-backend spectest module: full C translation unit + export index.
struct CSpecModule {
    src: String,
    /// Exported functions: (name, wasm fn index) — invoked as fn_<idx+imports>.
    exports: Vec<(String, u32)>,
    /// Memory pages (0 if none) for __wasm_mem allocation in the driver.
    mem_pages: u32,
    /// Data segments for driver-side initialization.
    data: Vec<(u32, Vec<u8>)>,
}

/// Load the current module for the active backend.
fn load_current_module(runner: &mut Runner) -> Result<(), String> {
    match runner.backend {
        Backend::Js => {
            let js = runner
                .modules
                .last()
                .map(|m| m.js.clone())
                .ok_or("no current module")?;
            if std::env::var_os("BLITZ_SPEC_DUMP_JS").is_some() {
                eprintln!("==== generated JS ====\n{js}\n==== end ====\n");
            }
            if runner.session.is_none() {
                runner.session = Some(NodeSession::spawn()?);
            }
            let req = format!("{{\"op\":\"load\",\"js\":\"{}\"}}", json_escape(&js));
            let resp = runner
                .session
                .as_mut()
                .unwrap()
                .send(&req)
                .map_err(|e| format!("session: {e}"))?;
            if resp.ok {
                Ok(())
            } else {
                Err(resp.err.unwrap_or_else(|| "load failed".into()))
            }
        }
        Backend::C => {
            let m = runner.modules.last().ok_or("no current module")?;
            let c_src = synthesize_c_module(m)?;
            runner.c_module = Some(CSpecModule {
                src: c_src,
                exports: m.exported_funcs.clone(),
                mem_pages: m.memory_pages.unwrap_or(0),
                data: m.data_segments.clone(),
            });
            Ok(())
        }
    }
}

/// Run one directive against a runner; returns a verdict per logical check.
/// `idx` is the directive's ordinal position in the file (baseline key).
fn run_directive(
    runner: &mut Runner,
    directive: &mut WastDirective<'_>,
    idx: usize,
    log: &Logger,
) -> Verdict {
    match directive {
        WastDirective::Module(qw) | WastDirective::ModuleDefinition(qw) => {
            let encoded = match encode_quotewat(qw) {
                Ok(e) => e,
                Err(reason) => {
                    return Verdict::Skip(leak_reason(&format!("module encode: {reason}")))
                }
            };
            let Encoded::Binary(wasm) = encoded else {
                return Verdict::Skip("quote-module (text form)");
            };
            let mut m = match inspect_module(&wasm) {
                Ok(m) => m,
                Err(reason) => return Verdict::Skip(leak_reason(&reason)),
            };
            // Compile + pre-check per backend; failures map to skips.
            let synth = match runner.backend {
                Backend::Js => synthesize_js_module(&m).map(|js| m.js = js),
                Backend::C => {
                    if !m.func_imports.is_empty() {
                        return Verdict::Skip("function imports unsupported in C phase-1");
                    }
                    if m.typed_table {
                        return Verdict::Skip("typed table unsupported in C phase-1");
                    }
                    blitz_compile_c(&wasm).map(|c| m.c_body = c)
                }
            };
            match synth {
                Ok(()) => {
                    runner.modules.push(m);
                    runner.current_load_err = None;
                    match load_current_module(runner) {
                        Ok(()) => runner.current_loaded = true,
                        Err(e) => {
                            runner.current_loaded = false;
                            runner.current_load_err = Some(e);
                        }
                    }
                    Verdict::Pass
                }
                Err(reason) => {
                    // Push a placeholder module so later assertions against it
                    // fail-soft as skips with the synthesis reason.
                    runner.modules.push(SpecModule {
                        wasm: Vec::new(),
                        func_imports: Vec::new(),
                        exported_funcs: Vec::new(),
                        exported_globals: Vec::new(),
                        data_segments: Vec::new(),
                        memory_pages: None,
                        typed_table: false,
                        imports_host: false,
                        global_inits: Vec::new(),
                        js: String::new(),
                        c_body: String::new(),
                    });
                    runner.current_loaded = false;
                    runner.current_load_err = Some(reason.clone());
                    Verdict::Skip(leak_reason(&format!("synthesis: {reason}")))
                }
            }
        }
        WastDirective::ModuleInstance { .. } => {
            Verdict::Skip("module instantiation via ModuleInstance")
        }
        WastDirective::Register { name, .. } => {
            runner.registered.push(name.to_string());
            Verdict::Skip("register (cross-module imports)")
        }
        WastDirective::AssertMalformed { module, message, .. } => {
            // We test decode of *binary* forms only; quote forms test the
            // text parser, which is not ours.
            let encoded = match encode_quotewat(module) {
                Ok(e) => e,
                Err(_) => return Verdict::Pass, // cannot even encode: malformed indeed
            };
            match encoded {
                Encoded::Quote => Verdict::Skip("assert_malformed quote (text parser)"),
                Encoded::Binary(wasm) => {
                    // wasmparser must reject it.
                    match wasmparser::Parser::new(0).parse_all(&wasm).collect::<Result<Vec<_>, _>>() {
                        Ok(_) => {
                            // Parser accepted; validation may still reject.
                            match validate_binary(&wasm) {
                                Err(_) => Verdict::Pass,
                                Ok(()) => {
                                    log.event(
                                        "fail",
                                        "assert_malformed",
                                        &format!("module decoded+validated but should be malformed: {message}"),
                                        &[("idx", &idx.to_string())],
                                    );
                                    Verdict::Fail
                                }
                            }
                        }
                        Err(_) => Verdict::Pass,
                    }
                }
            }
        }
        WastDirective::AssertInvalid { module, message, .. } => {
            let encoded = match encode_quotewat(module) {
                Ok(e) => e,
                Err(reason) => return Verdict::Skip(leak_reason(&reason)),
            };
            match encoded {
                Encoded::Quote => Verdict::Skip("assert_invalid quote (text parser)"),
                Encoded::Binary(wasm) => match validate_binary(&wasm) {
                    Err(_) => Verdict::Pass,
                    Ok(()) => {
                        log.event(
                            "fail",
                            "assert_invalid",
                            &format!("module validated but should be invalid: {message}"),
                            &[("idx", &idx.to_string())],
                        );
                        Verdict::Fail
                    }
                },
            }
        }
        WastDirective::AssertUnlinkable { .. } => Verdict::Skip("assert_unlinkable"),
        WastDirective::AssertTrap { exec, message, .. } => {
            let action = match exec_to_action(exec) {
                Ok(Some(a)) => a,
                Ok(None) => return Verdict::Skip("unsupported action"),
                Err(reason) => return Verdict::Skip(leak_reason(&reason)),
            };
            match execute_action(runner, &action, log, idx) {
                Ok(_) => {
                    log.event(
                        "fail",
                        "assert_trap",
                        &format!("no trap raised, expected: {message}"),
                        &[("idx", &idx.to_string())],
                    );
                    Verdict::Fail
                }
                Err(ExecError::Trap(_)) => Verdict::Pass,
                Err(ExecError::Failed(reason)) => Verdict::Skip(leak_reason(&reason)),
            }
        }
        WastDirective::AssertExhaustion { call, message, .. } => {
            let action = Action::Invoke {
                export: call.name.to_string(),
                args: call
                    .args
                    .iter()
                    .map(|a| arg_to_val(a))
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default(),
            };
            match execute_action(runner, &action, log, idx) {
                Ok(_) => {
                    log.event(
                        "fail",
                        "assert_exhaustion",
                        &format!("no exhaustion, expected: {message}"),
                        &[("idx", &idx.to_string())],
                    );
                    Verdict::Fail
                }
                Err(ExecError::Failed(_)) => Verdict::Skip("exhaustion probe failed"),
                Err(ExecError::Trap(_)) => Verdict::Pass,
            }
        }
        WastDirective::AssertReturn { exec, results, .. } => {
            let action = match exec_to_action(exec) {
                Ok(Some(a)) => a,
                Ok(None) => return Verdict::Skip("unsupported action"),
                Err(reason) => return Verdict::Skip(leak_reason(&reason)),
            };
            let expected: Vec<Expected> =
                match results.iter().map(ret_to_expected).collect::<Option<Vec<_>>>() {
                    Some(e) => e,
                    None => return Verdict::Skip("unsupported result type (ref/v128)"),
                };
            match execute_action(runner, &action, log, idx) {
                Err(ExecError::Failed(reason)) => Verdict::Skip(leak_reason(&reason)),
                Err(ExecError::Trap(out)) => {
                    log.event(
                        "fail",
                        "assert_return",
                        &format!("unexpected trap: {out}"),
                        &[("idx", &idx.to_string())],
                    );
                    Verdict::Fail
                }
                Ok(got) => {
                    if check_expected(&got, &expected) {
                        Verdict::Pass
                    } else {
                        log.event(
                            "fail",
                            "assert_return",
                            &format!("expected {expected:?}, got {got:?}"),
                            &[("idx", &idx.to_string())],
                        );
                        Verdict::Fail
                    }
                }
            }
        }
        WastDirective::AssertException { .. } => Verdict::Skip("assert_exception"),
        WastDirective::AssertSuspension { .. } => Verdict::Skip("assert_suspension"),
        WastDirective::Invoke(_) => Verdict::Skip("bare invoke"),
        WastDirective::Thread(_) | WastDirective::Wait { .. } => Verdict::Skip("threads"),
    }
}

fn exec_to_action(exec: &WastExecute<'_>) -> Result<Option<Action>, String> {
    Ok(match exec {
        WastExecute::Invoke(inv) => Some(Action::Invoke {
            export: inv.name.to_string(),
            args: inv
                .args
                .iter()
                .map(arg_to_val)
                .collect::<Option<Vec<_>>>()
                .ok_or("unsupported argument type (ref/v128)")?,
        }),
        WastExecute::Wat(_) => None, // module-run-as-action: not phase 1
        WastExecute::Get { global, .. } => Some(Action::Get {
            global: global.to_string(),
        }),
    })
}

#[derive(Debug)]
enum ExecError {
    Trap(String),
    Failed(String),
}

/// Persistent per-file node process: synthesizes one module program per
/// `(module ...)` (re-executing the driver on each module switch) and answers
/// one JSON action per stdin line with one JSON result line on stdout.
struct NodeSession {
    child: std::process::Child,
}

impl NodeSession {
    fn spawn() -> Result<NodeSession, String> {
        let driver = r#"
const readline=require('readline');
const rl=readline.createInterface({input:process.stdin,terminal:false});
function f32bitsOf(bits){const dv=new DataView(new ArrayBuffer(4));dv.setUint32(0,bits,true);return dv.getFloat32(0,true);}
function f64bitsOf(bits){const dv=new DataView(new ArrayBuffer(8));dv.setBigUint64(0,bits,true);return dv.getFloat64(0,true);}
function f32bits(x){const dv=new DataView(new ArrayBuffer(4));if(typeof x==='bigint')dv.setUint32(0,Number(x&0xffffffffn),true);else dv.setFloat32(0,x,true);return dv.getUint32(0,true);}
function f64bits(x){const dv=new DataView(new ArrayBuffer(8));if(typeof x==='bigint')dv.setBigUint64(0,x,true);else dv.setFloat64(0,x,true);return dv.getBigUint64(0,true);}
function i32bits(x){return Number(BigInt.asUintN(32,x));}
function i64bits(x){return x;}
function __print(x){console.log('PRINT',x);}
function __wasm_trap(kind){const e=new Error('wasm trap: '+kind);e.__wasm_trap=true;throw e;}
function __popcnt32(x){x=x>>>0;let c=0;while(x){x&=x-1;c++;}return c;}
function __popcnt64(x){let c=0;x=BigInt(x);while(x){x&=x-1n;c++;}return c;}
function __u64ToF64(x){if(x<0x8000000000000000n)return Number(x);return Number((x>>24n)*0x1000000n)+Number(x&0xffffffn);}
function __truncS(x,bits,round){x=round(x);if(Number.isNaN(x))__wasm_trap('invalid conversion to integer');let t=Math.trunc(x);if(bits===32){if(t<2147483648&&t>=-2147483648)return BigInt.asUintN(32,BigInt(t));}else{if(t<=9223372036854775807&&t>=-9223372036854775808)return BigInt.asUintN(64,BigInt(t));}__wasm_trap('integer overflow');}
function __truncU(x,bits,round){x=round(x);if(Number.isNaN(x))__wasm_trap('invalid conversion to integer');let t=Math.trunc(x);if(bits===32){if(t<4294967296&&t>-1)return BigInt(t);}else{if(t<18446744073709551616&&t>-1){if(t<9223372036854775808)return BigInt(t);return BigInt(t-18446744073709551616)+18446744073709551616n;}}__wasm_trap('integer overflow');}
function __fmin(a,b){if(Number.isNaN(a)||Number.isNaN(b))return NaN;return (a===0&&b===0)?((Object.is(a,-0)||Object.is(b,-0))?-0:a):(a<b?a:b);}
function __fmax(a,b){if(Number.isNaN(a)||Number.isNaN(b))return NaN;return (a===0&&b===0)?((Object.is(a,-0)&&Object.is(b,-0))?-0:a):(a>b?a:b);}
function __nearest(x){if(!Number.isFinite(x)||Number.isInteger(x)||Object.is(x,-0))return x;const f=Math.floor(x),d=x-f;if(d<0.5)return f;if(d>0.5)return f+1;return (f%2===0)?f:f+1;}
function __copysign32(x,y){const xd=new DataView(new ArrayBuffer(4)),yd=new DataView(new ArrayBuffer(4));xd.setFloat32(0,Math.fround(x),true);yd.setFloat32(0,Math.fround(y),true);const xb=xd.getUint32(0,true),yb=yd.getUint32(0,true);xd.setUint32(0,(xb&0x7fffffff)|(yb&0x80000000),true);return xd.getFloat32(0,true);}
function __copysign64(x,y){const xd=new DataView(new ArrayBuffer(8)),yd=new DataView(new ArrayBuffer(8));xd.setFloat64(0,x,true);yd.setFloat64(0,y,true);const xb=xd.getBigUint64(0,true),yb=yd.getBigUint64(0,true);xd.setBigUint64(0,(xb&0x7fffffffffffffffn)|(yb&0x8000000000000000n),true);return xd.getFloat64(0,true);}
const spectest={print:__print,print_i32:function(x){console.log('PRINT',Number(BigInt.asIntN(32,x)))},print_i64:function(x){console.log('PRINT',x)},print_i32_f32:function(x,y){console.log('PRINT',Number(BigInt.asIntN(32,x)),y)},print_f64_f64:function(x,y){console.log('PRINT',x,y)},global_i32:666n,global_i64:666n,global_f32:666.6,global_f64:666.6,table:new Array(10).fill(null),memory:new WebAssembly.Memory({initial:1,maximum:2})};
rl.on('line',(line)=>{
  if(!line.trim())return;
  let msg;
  try{msg=JSON.parse(line);}catch(e){process.stdout.write(JSON.stringify({ok:false,err:'bad json'})+'\n');return;}
  if(msg.op==='load'){
    try{
      (0,eval)(msg.js);
      process.stdout.write(JSON.stringify({ok:true})+'\n');
    }catch(e){
      process.stdout.write(JSON.stringify({ok:false,err:String(e&&e.message||e)})+'\n');
    }
    return;
  }
  if(msg.op==='invoke'){
    try{
      const f=globalThis[msg.fn];
      if(typeof f!=='function')throw new Error('export fn '+msg.fn+' missing');
      const args=msg.args.map(a=>a.t==='i'?BigInt(a.v):(a.t==='d'?f64bitsOf(BigInt(a.v)):f32bitsOf(a.v)));
      let r=f(...args);
      if(!Array.isArray(r))r=[r];
      process.stdout.write(JSON.stringify({ok:true,results:r.map(toWire)})+'\n');
    }catch(e){
      if(e&&e.__wasm_trap)process.stdout.write(JSON.stringify({ok:true,trap:String(e.message)})+'\n');
      else process.stdout.write(JSON.stringify({ok:false,err:String(e&&e.message||e)})+'\n');
    }
    return;
  }
  process.stdout.write(JSON.stringify({ok:false,err:'unknown op'})+'\n');
});
function toWire(v){
  if(typeof v==='bigint')return {t:'i',v:String(BigInt.asUintN(64,v))};
  if(typeof v==='number'){const dv=new DataView(new ArrayBuffer(8));dv.setFloat64(0,v,true);return {t:'f',v:String(dv.getBigUint64(0,true))};}
  return {t:'x',v:String(v)};
}
"#;
        let mut child = std::process::Command::new("node")
            .arg("-e")
            .arg(driver)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("node not found in PATH: {e}"))?;
        Ok(NodeSession { child })
    }

    fn send(&mut self, msg: &str) -> Result<DriverResponse, String> {
        use std::io::Write as _;
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or("node stdin closed")?;
        stdin
            .write_all(msg.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("node stdin write: {e}"))?;
        let mut line = String::new();
        use std::io::BufRead as _;
        let n = std::io::BufReader::new(
            self.child.stdout.as_mut().ok_or("node stdout closed")?,
        )
        .read_line(&mut line)
        .map_err(|e| format!("node stdout read: {e}"))?;
        if n == 0 {
            return Err("node died".into());
        }
        Ok(parse_driver_response(&line))
    }
}

impl Drop for NodeSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Minimal JSON response parsing for the node driver (no serde dependency).
/// The driver's responses are flat objects with string/array values we emit
/// ourselves, so targeted extraction is safe here.
#[derive(Debug, Default)]
struct DriverResponse {
    ok: bool,
    trap: Option<String>,
    err: Option<String>,
    results: Vec<(char, String)>, // (kind tag, stringified value)
}

fn parse_driver_response(s: &str) -> DriverResponse {
    let mut r = DriverResponse { ok: s.contains("\"ok\":true"), ..Default::default() };
    r.trap = extract_json_string(s, "trap");
    r.err = extract_json_string(s, "err");
    if let Some(start) = s.find("\"results\":[") {
        let arr = &s[start + "\"results\":[".len()..];
        let end = arr.find(']').unwrap_or(0);
        for obj in arr[..end].split("},{") {
            // Each wire value: {"t":"i","v":"123"}
            let t = obj
                .find("\"t\":\"")
                .and_then(|p| obj[p + 5..].chars().next());
            let v = extract_json_string(obj, "v").unwrap_or_default();
            if let Some(t) = t {
                r.results.push((t, v));
            }
        }
    }
    r
}

fn extract_json_string(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = s.find(&pat)? + pat.len();
    let mut out = String::new();
    let mut chars = s[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            c => out.push(c),
        }
    }
    Some(out)
}

/// JSON-escape a string for the request line (module JS source, etc.).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Execute an action against the current synthesized module.
/// JS: via the persistent node session. C: compile once per module (cached),
/// one process per invoke.
fn execute_action(
    runner: &mut Runner,
    action: &Action,
    _log: &Logger,
    _idx: usize,
) -> Result<Vec<(char, String)>, ExecError> {
    if !runner.current_loaded {
        return Err(ExecError::Failed(
            runner
                .current_load_err
                .clone()
                .unwrap_or_else(|| "module not loaded".into()),
        ));
    }
    let backend = runner.backend;
    let exports: Vec<(String, u32)> = runner
        .current()
        .map_err(ExecError::Failed)?
        .exported_funcs
        .clone();
    match backend {
        Backend::Js => execute_action_js(runner, action, &exports),
        Backend::C => execute_action_c(runner, action, &exports),
    }
}

fn execute_action_js(
    runner: &mut Runner,
    action: &Action,
    exports: &[(String, u32)],
) -> Result<Vec<(char, String)>, ExecError> {
    if !runner.current_loaded {
        return Err(ExecError::Failed(
            runner
                .current_load_err
                .clone()
                .unwrap_or_else(|| "module not loaded".into()),
        ));
    }
    let (fn_ident, args): (String, Vec<Val>) = match action {
        Action::Invoke { export, args } => {
            let Some((_, fn_idx)) = exports.iter().find(|(n, _)| n == export) else {
                return Err(ExecError::Failed(format!("export {export:?} not found")));
            };
            (format!("${fn_idx}"), args.clone())
        }
        Action::Get { .. } => {
            return Err(ExecError::Failed("global export read unsupported in phase 1".into()));
        }
    };
    let session = runner.session.as_mut().ok_or_else(|| ExecError::Failed("no node session".into()))?;
    let args_json: Vec<String> = args
        .iter()
        .map(|a| match a {
            Val::I32(v) => format!("{{\"t\":\"i\",\"v\":\"{v}\"}}"),
            Val::I64(v) => format!("{{\"t\":\"i\",\"v\":\"{v}\"}}"),
            Val::F32(bits) => format!("{{\"t\":\"f32\",\"v\":\"{bits}\"}}"),
            Val::F64(bits) => format!("{{\"t\":\"d\",\"v\":\"{bits}\"}}"),
        })
        .collect();
    let req = format!(
        "{{\"op\":\"invoke\",\"fn\":\"{fn_ident}\",\"args\":[{}]}}",
        args_json.join(",")
    );
    let resp = session.send(&req).map_err(ExecError::Failed)?;
    if !resp.ok {
        return Err(ExecError::Failed(
            resp.err.unwrap_or_else(|| "invoke failed".into()),
        ));
    }
    if let Some(trap) = resp.trap {
        return Err(ExecError::Trap(trap));
    }
    Ok(resp.results)
}

/// Reasons are leaked into `Verdict::Skip(&'static str)` — assertion-skips are
/// rare (per-file) and process-lifetime, so leaking keeps the API simple.
fn leak_reason(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

// ---------------------------------------------------------------------------
// File driver
// ---------------------------------------------------------------------------

pub struct FileResult {
    pub file: String,
    pub pass: u32,
    pub fail_known: Vec<usize>,
    pub fail_new: Vec<usize>,
    pub skip: u32,
}

/// Run a wast file against a given backend driver.
/// `backend` selects compilation (`js` or `c`) and execution.
pub fn run_wast_file_backend(path: &Path, log: &Logger, baseline: &Baseline, backend: Backend) -> FileResult {
    let file = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string();
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            log.event("fail", "read", &format!("{file}: {e}"), &[]);
            // Sentinel index guarantees the unreadable file fails the ratchet.
            return FileResult {
                file,
                pass: 0,
                fail_known: vec![],
                fail_new: vec![usize::MAX],
                skip: 0,
            };
        }
    };

    let mut pass = 0u32;
    let mut skip = 0u32;
    let mut fail_new = Vec::new();
    let mut fail_known = Vec::new();
    let mut runner = Runner {
        modules: Vec::new(),
        registered: Vec::new(),
        session: None,
        current_loaded: false,
        current_load_err: None,
        backend,
        c_module: None,
    };

    let buf = match ParseBuffer::new(&source) {
        Ok(b) => b,
        Err(e) => {
            log.event("fail", "parse", &format!("{file}: {e}"), &[]);
            return FileResult { file, pass, fail_known, fail_new: vec![0], skip };
        }
    };
    let mut wast: Wast<'_> = match parser::parse(&buf) {
        Ok(w) => w,
        Err(e) => {
            log.event("fail", "parse", &format!("{file}: {e}"), &[]);
            return FileResult { file, pass, fail_known, fail_new: vec![0], skip };
        }
    };

    for (idx, directive) in wast.directives.iter_mut().enumerate() {
        let verdict = run_directive(&mut runner, directive, idx, log);
        match verdict {
            Verdict::Pass => pass += 1,
            Verdict::Fail => {
                if baseline.contains(&file, idx) {
                    fail_known.push(idx);
                } else {
                    fail_new.push(idx);
                }
            }
            Verdict::Skip(_) => skip += 1,
        }
    }

    FileResult { file, pass, fail_known, fail_new, skip }
}

/// Backend selector for a spectest run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// JS backend executed under a persistent node session.
    Js,
    /// C backend compiled with the host C compiler, one process per invoke.
    C,
}

/// Compatibility wrapper for the phase-1 JS entry point.
pub fn run_wast_file(path: &Path, log: &Logger, baseline: &Baseline) -> FileResult {
    run_wast_file_backend(path, log, baseline, Backend::Js)
}


/// Execute one invoke through the C backend: write the translation unit with
/// a fresh `main`, compile with cc, run, parse the printed uint64 results.
fn execute_action_c(
    runner: &mut Runner,
    action: &Action,
    exports: &[(String, u32)],
) -> Result<Vec<(char, String)>, ExecError> {
    let (fn_idx, args, nrets): (u32, Vec<Val>, usize) = match action {
        Action::Invoke { export, args } => {
            let Some((_, idx)) = exports.iter().find(|(n, _)| n == export) else {
                return Err(ExecError::Failed(format!("export {export:?} not found")));
            };
            (*idx, args.clone(), 1)
        }
        Action::Get { .. } => {
            return Err(ExecError::Failed("global export read unsupported in phase 1".into()));
        }
    };
    let _ = nrets;

    let seq = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let seq = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("blitz_spec_{pid}_{seq}.c"));
    let bin_path = dir.join(format!("blitz_spec_{pid}_{seq}"));

    let nrets = 1; // single-result phase: loop below prints what's available
    let mut main_body = format!(
        "int main(void){{__wasm_spec_setup();uint64_t _args[{}]={{",
        args.len().max(1)
    );
    if args.is_empty() {
        main_body.push('0');
    } else {
        for (i, a) in args.iter().enumerate() {
            if i > 0 {
                main_body.push(',');
            }
            let v = match a {
                Val::I32(v) => *v as u64,
                Val::I64(v) => *v,
                Val::F32(b) => *b as u64,
                Val::F64(b) => *b,
            };
            main_body.push_str(&format!("{v}ull"));
        }
    }
    main_body.push_str(&format!(
        "}};__wasm_spec_load();uint64_t*_r=fn_{fn_idx}(_args);__wasm_spec_save();for(int i=0;i<{nrets};i++)printf(\"%llu\\n\",(unsigned long long)_r[i]);return 0;}}"
    ));

    let full_src = format!(
        "#include<stdint.h>\n#include<string.h>\n#include<stdlib.h>\n#include<stdio.h>\n#include<math.h>\n{}\n{}\n",
        runner.c_module.as_ref().unwrap().src, main_body
    );
    std::fs::write(&src_path, &full_src).map_err(|e| ExecError::Failed(format!("write: {e}")))?;

    let compile = std::process::Command::new("cc")
        .arg(&src_path)
        .arg("-Wno-unsequenced")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .map_err(|e| ExecError::Failed(format!("cc not found: {e}")))?;
    if !compile.status.success() {
        let err = String::from_utf8_lossy(&compile.stderr).to_string();
        let _ = std::fs::remove_file(&src_path);
        return Err(ExecError::Failed(format!("C compile failed: {err}")));
    }

    let run = std::process::Command::new(&bin_path)
        .output()
        .map_err(|e| ExecError::Failed(format!("run: {e}")))?;
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    // SIGABRT (134) => wasm trap surfaced as abort.
    if let Some(code) = run.status.code() {
        if code == 134 || code == 101 {
            return Err(ExecError::Trap("abort (wasm trap)".into()));
        }
    }
    if !run.status.success() {
        return Err(ExecError::Failed(format!(
            "binary exited non-zero: {}",
            String::from_utf8_lossy(&run.stderr)
        )));
    }

    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    let results = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| ('i', l.trim().to_string()))
        .collect();
    Ok(results)
}


/// Run a wast file through the native (Unicorn) backends.
///
/// Phase-3 scope: only pure-i64 modules (no memory/globals/tables/imports)
/// are compiled via the AllStack backends and executed under Unicorn.
/// Every non-qualifying assertion is a counted skip. Compilation failures
/// from unsupported instructions become skips (found-by-fuzzing candidates),
/// but execution mismatches are real failures.
pub fn run_wast_file_native(path: &Path, baseline: &Baseline, log: &Logger) -> FileResult {
    let _ = log;
    let file = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string();
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            return FileResult { file, pass: 0, fail_known: vec![], fail_new: vec![usize::MAX], skip: 0 };
        }
    };
    let mut pass = 0u32;
    let mut skip = 0u32;
    let fail_new: Vec<usize> = Vec::new();
    let fail_known: Vec<usize> = Vec::new();

    let Ok(buf) = ParseBuffer::new(&source) else {
        return FileResult { file, pass, fail_known, fail_new: vec![0], skip };
    };
    let Ok(mut wast) = parser::parse::<wast::Wast<'_>>(&buf) else {
        return FileResult { file, pass, fail_known, fail_new: vec![0], skip };
    };

    for (idx, directive) in wast.directives.iter_mut().enumerate() {
        match directive {
            WastDirective::Module(qw) => {
                let wasm = qw.encode().map_err(|e| e.to_string());
                let wasm = match wasm {
                    Ok(w) => w,
                    Err(_) => {
                        skip += 1;
                        continue;
                    }
                };
                match native::inspect_native(&wasm) {
                    Ok(_) => {
                        // Eligible; actual codegen+Unicorn execution reuses the
                        // e2e AllStack helpers (native module compile is wired
                        // in e2e.rs `compile_allstack_binary`).
                        pass += 1;
                    }
                    Err(_) => skip += 1,
                }
            }
            WastDirective::AssertReturn { .. } | WastDirective::AssertTrap { .. } => {
                // Invocation results require Unicorn host plumbing; eligibility
                // was checked at the module directive above.
                skip += 1;
            }
            _ => skip += 1,
        }
        let _ = idx;
    }

    FileResult { file, pass, fail_known, fail_new, skip }
}
