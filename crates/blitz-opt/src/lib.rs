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
pub trait OptCodegen {
    /// Generates code for an optimized push operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `value` - The expression to push
    /// * `index` - The stack index where the value will be stored
    fn write_opt_push(
        w: &mut (impl Write + ?Sized),
        value: &(dyn Display + '_),
        index: usize,
    ) -> core::fmt::Result;

    /// Generates code for a non-optimized push operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `value` - The expression to push
    fn write_non_opt_push(
        w: &mut (impl Write + ?Sized),
        value: &(dyn Display + '_),
    ) -> core::fmt::Result;

    /// Generates code for an optimized pop operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    /// * `index` - The stack index to pop from
    fn write_opt_pop(w: &mut (impl Write + ?Sized), index: usize) -> core::fmt::Result;

    /// Generates code for a non-optimized pop operation.
    ///
    /// # Arguments
    ///
    /// * `w` - The writer to output code to
    fn write_non_opt_pop(w: &mut (impl Write + ?Sized)) -> core::fmt::Result;
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
/// * `opt_state` - Optional reference to the optimization state
/// * `w` - The writer to output code to
/// * `a` - The expression to push onto the stack
pub fn push<C: OptCodegen>(
    opt_state: Option<&Mutex<OptState>>,
    w: &mut (impl Write + ?Sized),
    a: &(dyn Display + '_),
) -> core::fmt::Result {
    if let Some(o) = opt_state {
        let mut o = o.lock();
        let index = o.depth + 1;
        C::write_opt_push(w, a, index)?;
        o.depth += 1;
    } else {
        C::write_non_opt_push(w, a)?;
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
/// * `opt_state` - Optional reference to the optimization state
/// * `w` - The writer to output code to
pub fn pop<C: OptCodegen>(
    opt_state: Option<&Mutex<OptState>>,
    w: &mut (impl Write + ?Sized),
) -> core::fmt::Result {
    if let Some(o) = opt_state {
        let mut o = o.lock();
        o.depth -= 1;
        let index = o.depth + 1;
        C::write_opt_pop(w, index)?;
    } else {
        C::write_non_opt_pop(w)?;
    }
    Ok(())
}
