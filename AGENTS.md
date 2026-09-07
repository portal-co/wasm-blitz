# Agent Guide — wasm-blitz WASM Compiler

wasm-blitz is a library-only WASM-to-native compiler. No CLI binary exists. AI agents interact with it through the `blitz-tests` integration test suite.

## Key architecture constraints

- `blitz-common` is `#![no_std]` — do not add std-dependent code there.
- Each backend crate (`blitz-x86-64`, `blitz-aarch64`, `blitz-riscv64`, `blitz-riscv32`, `blitz-arm`, `blitz-i686`, etc.) is independently versioned.
- `blitz-tests` has full std access and exercises all backends.
- `BLITZ_TRACE_UNICORN` env var (existing) enables per-instruction Unicorn emulator cross-checking.
- **ILP32 backends** (`blitz-riscv32`, `blitz-arm`, `blitz-i686`): WASM value slots stay **8 bytes**; host pointer / SCR / fn-ptr tables use **×4**. Unicorn/clang smokes soft-skip when the host lacks the clang triple (`riscv32-*`, `armv7-*` / `arm-linux-gnueabihf`, `i686-*`) — do not fail CI for missing cross toolchains; `assemble_or_skip` in `blitz-tests/tests/e2e.rs` is the pattern.

## Spec-test suite

- `cargo test -p portal-solutions-blitz-tests --test spectests` runs the official WASM spec suite (`BLITZ_SPEC_DIR` or `target/spec` clone).
- Known failures: `crates/blitz-tests/tests/spec/baseline.toml` (ratchet: new failures fail CI; stale entries fail CI too). Every entry requires a `reason`.
- Suite pin: `.github/workflows/spectests.yml` `SPEC_COMMIT`; bumping requires re-baselining.
- Do not fix baseline entries by editing test files; fix the backend, or move the entry with justification.

## Compression-aware logging

Token compression proxies can sit between this tool and an LLM provider. When a proxy is active, MORE verbose structured output is net-cheaper than terse plaintext.

Environment variables (set before running `cargo test -p blitz-tests`):

| Variable | Effect |
|---|---|
| `PORTAL_LOG_JSON=1` | Structured NDJSON trace events instead of ad-hoc `eprintln!` in test output. |
| `PORTAL_LOG_BATCH=1` | Group trace events by test function into single JSON arrays. |
| `BLITZ_TRACE_UNICORN` | (existing) Enable per-instruction Unicorn emulator trace hook. |
| `BLITZ_SPEC_DIR` | Directory of the WebAssembly/spec checkout for the spectest harness. |
| `BLITZ_SPEC_DUMP_JS` | Dump generated JS for each spectest module load (debugging). |

Logger implementation: `crates/blitz-tests/tests/log.rs` (test-only, zero cost when PORTAL_LOG_JSON is unset).

These variables have no effect when unset.
