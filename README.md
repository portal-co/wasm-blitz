# wasm-blitz

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

A fast WebAssembly (WASM) compiler that supports multiple target platforms including JavaScript and various native architectures.

## Overview

wasm-blitz is a `no_std` compatible WebAssembly compiler designed for flexibility and performance. It provides compilation backends for both JavaScript runtime environments and native CPU architectures.

## Features

- **Multi-target compilation**: Compile WASM to JavaScript or native machine code
- **Native architecture support**: x86-64, AArch64, PowerPC 64-bit, and RISC-V 64-bit
- **Dead Code Elimination (DCE)**: Automatic optimization to remove unused code
- **No standard library dependency**: Built with `#![no_std]` for embedded and constrained environments
- **Modular architecture**: Clean separation between frontend parsing, common utilities, and backend code generation

## Crates

This project is organized as a Cargo workspace with the following crates:

### Core Crates

- **`blitz-common`**: Common utilities and types used across all compilation targets
  - Machine operator abstractions
  - Dead code elimination passes
  - Assembly abstractions
  - Common type definitions

- **`blitz-reencode`**: Re-encodes blitz IR back to WebAssembly
  - Converts blitz machine operators back to WASM bytecode
  - Applies optimizations before final WASM encoding
  - Used for producing WASM output from blitz IR

### Target Backend Crates

- **`blitz-js`**: JavaScript code generation backend
  - Compiles WASM bytecode to JavaScript
  - Optimized stack management
  - Runtime type checking

- **`blitz-x86-64`**: x86-64 native code generator
  - Naive code generation strategy
  - Direct machine code emission

- **`blitz-aarch64`**: ARM AArch64 (ARM64) code generator *(Work in Progress)*
  - Support for 64-bit ARM architecture
  - ARMv8-A and later architectures

- **`blitz-ppc64`**: PowerPC 64-bit code generator *(Work in Progress)*
  - Support for PowerPC 64-bit architecture
  - Power ISA v2.07 and later

- **`blitz-riscv64`**: RISC-V 64-bit code generator *(Work in Progress)*
  - RV64IMAFD instruction set (I, M, A, F, D extensions)
  - Compatible with [rv-utils](https://github.com/portal-co/rv-utils)

- **`blitz-i4`**: Common backend logic for 4-byte instruction architectures
  - Shared infrastructure for AArch64 and RISC-V 64-bit
  - Common optimizations for fixed 4-byte instruction architectures

## Building

Build all crates in the workspace:

```bash
cargo build
```

Build in release mode for optimized output:

```bash
cargo build --release
```

Build a specific crate:

```bash
cargo build -p portal-solutions-blitz-js
```

## Documentation

Generate and view the documentation:

```bash
cargo doc --open
```

This will build documentation for all crates and open it in your browser.

## Dependencies

wasm-blitz uses the following major dependencies:

- **[wasmparser](https://crates.io/crates/wasmparser)**: WebAssembly binary format parser
- **[wasm-encoder](https://crates.io/crates/wasm-encoder)**: WebAssembly binary format encoder
- **[wax-core](https://github.com/portal-co/wax)**: Core utilities for WASM analysis and transformation

## Usage

Each backend crate can be used independently depending on your target platform:

```rust
// Example: Using the JavaScript backend
use portal_solutions_blitz_js::JsWrite;

// Example: Using the x86-64 backend
use portal_solutions_blitz_x86_64::WriterExt;
```

See individual crate documentation for detailed usage examples.

## Architecture

```
┌─────────────────┐
│   WASM Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  blitz-common   │  ◄─── Shared types, DCE, machine ops
└────────┬────────┘
         │
    ┌────┴────────────────────────┬─────────────┐
    ▼                             ▼             ▼
┌────────┐    ┌──────────────┬────────────┬────────────┐
│blitz-js│    │ blitz-x86-64 │blitz-aarch64│ blitz-... │
└────────┘    └──────────────┴────────────┴────────────┘
    │                       │
    ▼                       ▼
JavaScript              Native Code
```

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure your code:

- Builds without errors
- Follows the existing code style
- Is compatible with `#![no_std]` where applicable
- Includes appropriate documentation

## Project Status

This project is under active development. APIs may change between versions.
