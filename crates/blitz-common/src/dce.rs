//! Dead code elimination (DCE) utilities.
//!
//! This module provides functionality for removing unreachable code from WASM
//! functions. Dead code elimination is an important optimization that reduces
//! the size of generated code and can improve performance.
//!
//! The DCE implementation is based on the wax-core analysis framework.

use wasm_encoder::Instruction;

use crate::*;
pub use wax_core::analysis::dce::*;
