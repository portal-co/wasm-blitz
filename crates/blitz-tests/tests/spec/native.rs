//! Native-backend spectest support (phase 3, `docs/spectests-plan.md`).
//!
//! Compiles simple i64-only wast modules through the blitz native backends
//! (AllStack-flavoured SysV) into machine code and runs them under Unicorn.
//!
//! Scope is deliberately narrow (see plan "Risks"): only modules with no
//! memory/globals/tables/imports/tags, functions with ≤1 i64 result and i64
//! params. All other assertions are counted skips, never silent. Missing
//! cross-clang triples soft-skip via [`AssembleError::MissingToolchain`],
//! mirroring `assemble_or_skip` in the e2e suite.

use wasm_encoder::ValType;

/// Target ISA for a native spectest run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeArch {
    X86_64,
    AArch64,
    Riscv64,
}

impl NativeArch {
    pub fn clang_target(self) -> &'static str {
        match self {
            NativeArch::X86_64 => "x86_64-unknown-linux-gnu",
            NativeArch::AArch64 => "aarch64-unknown-linux-gnu",
            NativeArch::Riscv64 => "riscv64-unknown-elf",
        }
    }
}

/// Why assembly failed.
#[derive(Debug)]
pub enum AssembleError {
    /// Host clang is missing or lacks the target — soft-skip.
    MissingToolchain(String),
    /// Backend emitted invalid assembly — a real failure.
    BadAsm(String),
}

/// A module prepared for native execution.
#[derive(Default)]
pub struct NativeModule {
    /// Exported functions (name, wasm fn index); no imports in scope.
    pub exports: Vec<(String, u32)>,
    /// Function type indices per function (params/results checks).
    pub fn_types: Vec<u32>,
    /// Type section signatures.
    pub sigs: Vec<wasm_encoder::FuncType>,
}

/// Inspect a module for native-scope eligibility; `Err` gives the reason.
pub fn inspect_native(wasm: &[u8]) -> Result<NativeModule, String> {
    let mut m = NativeModule::default();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::MemorySection(_) => {
                return Err("memory unsupported in native scope".into());
            }
            wasmparser::Payload::GlobalSection(_) => {
                return Err("globals unsupported in native scope".into());
            }
            wasmparser::Payload::TableSection(_) => {
                return Err("tables unsupported in native scope".into());
            }
            wasmparser::Payload::ImportSection(_) => {
                return Err("imports unsupported in native scope".into());
            }
            wasmparser::Payload::TagSection(_) => {
                return Err("tags unsupported in native scope".into());
            }
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for subtype in group.map_err(|e| e.to_string())?.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            subtype.composite_type.inner
                        {
                            let ok = ft.results().len() <= 1
                                && ft.params().iter().all(|t| *t == wasmparser::ValType::I64)
                                && ft.results().iter().all(|t| *t == wasmparser::ValType::I64);
                            if !ok {
                                return Err("non-i64 signature unsupported in native scope".into());
                            }
                            m.sigs.push(
                                wasm_encoder::FuncType::try_from(ft.clone())
                                    .map_err(|e| e.to_string())?,
                            );
                        }
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                m.fn_types.extend(reader.into_iter().flatten());
            }
            wasmparser::Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp.map_err(|e| e.to_string())?;
                    if exp.kind == wasmparser::ExternalKind::Func {
                        m.exports.push((exp.name.to_string(), exp.index));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(m)
}
