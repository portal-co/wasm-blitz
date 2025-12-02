//! Optimization state management for stack-based code generation.
//!
//! This crate provides a generic optimization framework for tracking stack depth
//! during code generation. It uses a trait-based approach to abstract all code
//! generation patterns used by different backends.
//!
//! # Features
//!
//! - Generic stack depth tracking
//! - Trait-based code generation abstraction for different backends
//! - Thread-safe optimization state with interior mutability
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_opt::{OptState, OptCodegen};
//!
//! struct MyBackend;
//! impl OptCodegen for MyBackend {
//!     fn write_opt_push(w: &mut impl Write, value: &dyn Display, index: usize) -> core::fmt::Result {
//!         write!(w, "/* custom push code */")
//!     }
//!     // ... implement other methods
//! }
//!
//! let state = OptState::default();
//! ```

#![no_std]
use core::fmt::{Display, Write};
use spin::Mutex;

/// Trait that defines the code generation patterns for stack operations.
///
/// Different backends can implement this trait to specify their own
/// code generation patterns for optimized and non-optimized push/pop operations.
///
/// This trait is object-safe (dyn-compatible) to allow for runtime polymorphism
/// and better composability.
pub trait OptCodegen {
    /// Generates code for an optimized push operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `value` - The expression to push
    /// * `index` - The stack index where the value will be stored
    fn write_opt_push(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
        index: usize,
    ) -> core::fmt::Result;

    /// Generates code for a non-optimized push operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `value` - The expression to push
    fn write_non_opt_push(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
    ) -> core::fmt::Result;

    /// Generates code for an optimized pop operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `index` - The stack index to pop from
    fn write_opt_pop(&self, w: &mut (dyn Write + '_), index: usize) -> core::fmt::Result;

    /// Generates code for a non-optimized pop operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    fn write_non_opt_pop(&self, w: &mut (dyn Write + '_)) -> core::fmt::Result;
}

/// Optimization state for stack depth tracking.
///
/// When enabled, allows for more efficient code generation
/// by tracking stack depth statically.
#[derive(Default)]
#[non_exhaustive]
pub struct OptState {
    /// Current depth of the stack.
    pub depth: usize,
}

/// Pushes a value onto the execution stack.
///
/// Generates code to push the given expression onto the stack.
/// The behavior depends on whether optimized stack tracking is enabled.
///
/// # Arguments
///
/// * `codegen` - The code generator implementation
/// * `opt_state` - Optional reference to the optimization state
/// * `w` - The writer to output code to
/// * `a` - The expression to push onto the stack
pub fn push(
    codegen: &dyn OptCodegen,
    opt_state: Option<&Mutex<OptState>>,
    w: &mut (dyn Write + '_),
    a: &dyn Display,
) -> core::fmt::Result {
    if let Some(o) = opt_state {
        let index = {
            let mut o = o.lock();
            let index = o.depth + 1;
            o.depth += 1;
            index
        };
        codegen.write_opt_push(w, a, index)?;
    } else {
        codegen.write_non_opt_push(w, a)?;
    }
    Ok(())
}

/// Pops a value from the execution stack.
///
/// Generates code to pop a value from the stack.
/// The behavior depends on whether optimized stack tracking is enabled.
///
/// # Arguments
///
/// * `codegen` - The code generator implementation
/// * `opt_state` - Optional reference to the optimization state
/// * `w` - The writer to output code to
///
/// # Panics
///
/// In debug builds, this function will panic if depth is 0 (stack underflow).
/// The caller is responsible for ensuring pushes and pops are balanced.
pub fn pop(
    codegen: &dyn OptCodegen,
    opt_state: Option<&Mutex<OptState>>,
    w: &mut (dyn Write + '_),
) -> core::fmt::Result {
    if let Some(o) = opt_state {
        let index = {
            let mut o = o.lock();
            debug_assert!(o.depth > 0, "Stack underflow: attempting to pop from empty stack");
            o.depth -= 1;
            o.depth + 1
        };
        codegen.write_opt_pop(w, index)?;
    } else {
        codegen.write_non_opt_pop(w)?;
    }
    Ok(())
}

/// Macro to generate a pop operation as a Display-compatible closure.
///
/// This macro wraps the `pop` function to create a displayable value
/// that can be interpolated into format strings.
///
/// # Arguments
///
/// * `$codegen` - The code generator implementation
/// * `$opt_state` - The optimization state (optional reference)
///
/// # Example
///
/// ```ignore
/// use portal_solutions_blitz_opt::{pop_display, OptCodegen, OptState};
/// 
/// let codegen = MyCodegen;
/// let opt_state = Some(&mutex);
/// write!(w, "result = {}", pop_display!(codegen, opt_state));
/// ```
#[macro_export]
macro_rules! pop_display {
    ($codegen:expr, $opt_state:expr) => {
        &|f: &mut core::fmt::Formatter| $crate::pop($codegen, $opt_state, f)
    };
}
