//! Assembly abstractions for code generation.
//!
//! This module provides common types and utilities for working with assembly
//! code across different target architectures. It re-exports types from the
//! `portal-pc-asm-common` crate.

use crate::*;

/// Common assembly types and utilities.
pub use portal_pc_asm_common::types as common;

/// Register abstraction for assembly code.
pub use portal_pc_asm_common::types::reg::Reg;
