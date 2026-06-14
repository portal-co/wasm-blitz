# Agent Guide — wasm-blitz WASM Compiler

wasm-blitz is a library-only WASM-to-native compiler. No CLI binary exists. AI agents interact with it through the `blitz-tests` integration test suite.

## Key architecture constraints

- `blitz-common` is `#![no_std]` — do not add std-dependent code there.
- Each backend crate (`blitz-x86-64`, `blitz-aarch64`, `blitz-riscv64`, etc.) is independently versioned.
- `blitz-tests` has full std access and exercises all backends.
- `BLITZ_TRACE_UNICORN` env var (existing) enables per-instruction Unicorn emulator cross-checking.

## Compression-aware logging

Token compression proxies can sit between this tool and an LLM provider. When a proxy is active, MORE verbose structured output is net-cheaper than terse plaintext.

Environment variables (set before running `cargo test -p blitz-tests`):

| Variable | Effect |
|---|---|
| `PORTAL_LOG_JSON=1` | Structured NDJSON trace events instead of ad-hoc `eprintln!` in test output. |
| `PORTAL_LOG_BATCH=1` | Group trace events by test function into single JSON arrays. |
| `BLITZ_TRACE_UNICORN` | (existing) Enable per-instruction Unicorn emulator trace hook. |

Logger implementation: `crates/blitz-tests/tests/log.rs` (test-only, zero cost when PORTAL_LOG_JSON is unset).

These variables have no effect when unset.
