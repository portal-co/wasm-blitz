//! JavaScript code generation backend for wasm-blitz.
//!
//! This crate provides functionality to compile WebAssembly bytecode into JavaScript.
//! The generated JavaScript code maintains WASM semantics while being executable in
//! JavaScript runtime environments.
//!
//! # Features
//!
//! - Stack-based execution model matching WASM semantics
//! - Optimized stack management with optional depth tracking
//! - Type checking for function signatures at runtime
//! - Support for all core WASM integer operations
//! - Control flow constructs (blocks, loops, if/else, branches)
//!
//! # Example
//!
//! ```ignore
//! use portal_solutions_blitz_js::{JsWrite, State};
//!
//! let mut state = State::default();
//! // Use the JsWrite trait to generate JavaScript code
//! ```
//!
//! # Stack Management
//!
//! The JavaScript backend uses a stack-based execution model. Two modes are available:
//!
//! - **Standard mode**: Uses JavaScript array operations for stack manipulation
//! - **Optimized mode**: Tracks stack depth statically for better performance

#![no_std]
use core::{
    cell::OnceCell,
    error::Error,
    fmt::{Display, Formatter, Write},
};

pub mod shard;
#[doc(hidden)]
pub mod __ {
    pub use portal_solutions_blitz_common::DisplayFn;
}
use alloc::vec::Vec;
use portal_solutions_blitz_common::{
    DisplayFn,
    ops::MachOperator,
    wasm_encoder::{BlockType, FuncType, Instruction, reencode::Reencode},
    wasmparser::{Operator, ValType},
};
use portal_solutions_blitz_opt::{self as blitz_opt, OptCodegen, OptState};
use spin::Mutex;
extern crate alloc;

/// JavaScript code for stack restoration using optional symbol iterator.
const STACK_WEAVE: &'static str = "(typeof $$stack_restore_symbol_iterator!=='undefined'?$$stack_restore_symbol_iterator:(a=>a))";

/// JavaScript implementation of the OptCodegen trait.
///
/// Provides JavaScript-specific code generation patterns for stack operations.
pub struct JsCodegen;

impl OptCodegen for JsCodegen {
    fn write_opt_push_start(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
    ) -> core::fmt::Result {
        write!(w, "(tmp={value}")
    }

    fn write_opt_push_end(&self, w: &mut (dyn Write + '_), index: usize) -> core::fmt::Result {
        write!(w, ",stack.length++,stack[{index}]=tmp,tmp)")
    }

    fn write_non_opt_push(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
    ) -> core::fmt::Result {
        write!(w, "(tmp={value},stack=[...{STACK_WEAVE}(stack),tmp],tmp)")
    }

    fn write_opt_pop(&self, w: &mut (dyn Write + '_), index: usize) -> core::fmt::Result {
        write!(w, "(tmp=stack[{index}],stack.length--,tmp)")
    }

    fn write_non_opt_pop(&self, w: &mut (dyn Write + '_)) -> core::fmt::Result {
        // Apply STACK_WEAVE into a fresh stack array, then pop the last element.
        // (`[...stack, tmp] = ...` is invalid JS — rest must be last in destructuring.)
        write!(w, "([...stack]={STACK_WEAVE}(stack),tmp=stack.pop(),tmp)")
    }
}

/// Pushes a value onto the JavaScript execution stack.
///
/// Generates JavaScript code to push the given expression onto the stack.
/// The behavior depends on whether optimized stack tracking is enabled in the state.
///
/// # Arguments
///
/// * `state` - The current compilation state
/// * `w` - The writer to output JavaScript code to
/// * `a` - The expression to push onto the stack
pub fn push(state: &State, w: &mut (dyn Write + '_), a: &dyn Display) -> core::fmt::Result {
    blitz_opt::push(&JsCodegen, state.opt(), w, a)
}

/// Pops a value from the JavaScript execution stack.
///
/// Generates JavaScript code to pop a value from the stack.
/// The behavior depends on whether optimized stack tracking is enabled in the state.
///
/// # Arguments
///
/// * `state` - The current compilation state
/// * `w` - The writer to output JavaScript code to
pub fn pop(state: &State, w: &mut (dyn Write + '_)) -> core::fmt::Result {
    blitz_opt::pop(&JsCodegen, state.opt(), w)
}
/// Macro to generate a pop operation as a DisplayFn.
///
/// This macro wraps the `pop` function to create a displayable value
/// that can be interpolated into format strings.
#[macro_export]
macro_rules! pop {
    ($state:ident) => {
        $crate::__::DisplayFn(&|f| match $state {
            ref state => $crate::pop(state, f),
        })
    };
}

// Re-export the generic pop_display macro from blitz-opt for consistency
#[doc(hidden)]
pub use portal_solutions_blitz_opt::pop_display;

/// State tracker for JavaScript code generation.
///
/// Maintains the current state of the compilation including control flow
/// stack and optional optimization state.
#[derive(Default)]
#[non_exhaustive]
pub struct State {
    stack: Vec<Frame>,
    opt_state: OnceCell<Mutex<OptState>>,
    /// When set, each `call`/`call_indirect` may bail with
    /// `$call.then($cont)` if the callee returns a `Promise`; the remainder
    /// of the function is compiled into `$cont`. Sync callees take the
    /// fall-through `return $cont($call)` path (still sync when `$cont`
    /// itself returns a plain array). See docs/abi.md "promise calls".
    pub promise_calls: bool,
    /// Open promise-mode continuations (call-site ids), innermost last.
    promise_close_stack: Vec<u32>,
    next_call_site: u32,
}

impl State {
    /// Enables optimization mode with the given initial state.
    ///
    /// This can be called once per State to enable optimized code generation.
    pub fn enable_opt(&self, opt: impl FnOnce() -> OptState) {
        self.opt_state.get_or_init(|| Mutex::new(opt()));
    }

    /// Enable Promise-bail call mode (JSPI-style). Incompatible with opt mode
    /// for now — opt stack depths are not tracked across `$cont` boundaries.
    pub fn enable_promise_calls(&mut self) {
        self.promise_calls = true;
    }

    fn opt(&self) -> Option<&Mutex<OptState>> {
        self.opt_state.get()
    }
}

/// Represents a control flow frame in the compilation state.
enum Frame {
    Block(BlockType),
    Loop(BlockType),
    /// `If` is like `Block` for branching purposes: `br N` that targets an `if`
    /// frame is a forward exit out of the if/else body.
    If(BlockType),
    /// `TryTable` acts like `Block` for branching (`br N` exits forward).
    /// Stores the catch clauses so they can be emitted at the matching `End`.
    TryTable(BlockType, alloc::vec::Vec<portal_solutions_blitz_common::wasm_encoder::Catch>),
    /// The implicit function-level label. `br N` with `N ==` stack depth
    /// targets this frame: it is a `return` carrying the function's results.
    /// Present only while a body is open (pushed at `StartBody` semantics,
    /// popped at `EndBody` — see `on_mach`).
    Function(BlockType),
}

/// Trait for writing JavaScript code for WASM operations.
///
/// This trait extends the `Write` trait with methods for generating JavaScript
/// code that implements WASM semantics. It handles function calls, branches,
/// and conversion of WASM operators to JavaScript.
pub trait JsWrite: Write {
    /// Generates JavaScript code for a function call.
    ///
    /// Emits code that validates the function signature at runtime and performs
    /// the call with proper argument handling and result management.
    ///
    /// # Arguments
    ///
    /// * `state` - The current compilation state
    /// * `sig` - The function signature (parameter and return types)
    /// * `function_index` - The function index or reference to call
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn call(
        &mut self,
        state: &mut State,
        sig: &FuncType,
        function_index: &(dyn Display + '_),
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        write!(
            self,
            "if({function_index}.__sig.params!={}||{function_index}.__sig.rets!={})throw new Error(`wasm sig mismatch`);",
            sig.params().len(),
            sig.results().len()
        )?;
        if state.promise_calls {
            // Promise-bail CPS: wrap the remainder of the function in `$cont_K`.
            // Opt mode is not supported together with promise_calls.
            let k = state.next_call_site;
            state.next_call_site += 1;
            state.promise_close_stack.push(k);
            let nrets = sig.results().len();
            write!(
                self,
                "args=[];for(let i = 0;i < {function_index}.__sig.params;i++)args=[...{STACK_WEAVE}(args),{}];\
                 let $call_{k}={function_index}(...args);\
                 const $cont_{k}=(val)=>{{tmp_locals=[...{STACK_WEAVE}(Array.isArray(val)?val:[val])];\
                 if(tmp_locals.length==={nrets}){{stack=[...{STACK_WEAVE}(stack),...{STACK_WEAVE}(tmp_locals)];}}\
                 else{{for(let i = 0;i < {nrets};i++)stack=[...{STACK_WEAVE}(stack),tmp_locals[i]];}}",
                pop!(state)
            )?;
            return Ok(());
        }
        if let Some(opt) = state.opt() {
            let mut o = opt.lock();
            let s = o.depth - sig.params().len();
            let od = o.depth;
            o.depth -= sig.params().len();
            let s2 = o.depth;
            o.depth += sig.results().len();
            // BUG FIX: stack indices are 1-based in opt mode (push writes to
            // stack[depth+1] then increments depth). Arguments live at
            // stack[s+1..=od]; results at stack[s2+1..=o.depth].
            write!(
                self,
                "args=[{}];stack.length -= {};
                tmp_locals=({function_index}(...args));
                stack.length += {};{}",
                DisplayFn(&|f| {
                    for n in (s + 1)..=od {
                        write!(f, "stack[{n}]")?;
                        if n != od {
                            write!(f, ",")?;
                        }
                    }
                    Ok(())
                }),
                sig.params().len(),
                sig.results().len(),
                DisplayFn(&|f| {
                    for (i, n) in ((s2 + 1)..=o.depth).enumerate() {
                        write!(f, "stack[{n}]=tmp_locals[{i}];")?;
                    }
                    Ok(())
                })
            )
        } else {
            write!(
                self,
                "args=[];
                for(let i = 0;i < {function_index}.__sig.params;i++)args=[...{STACK_WEAVE}(args),{}];
                tmp_locals=[...{STACK_WEAVE}({function_index}(...args))];
                if(tmp_locals.length==={function_index}.__sig.rets){{stack=[...{STACK_WEAVE}(stack),...{STACK_WEAVE}(tmp_locals)];}}else{{for(let i = 0;i < {function_index}.__sig.rets;i++)stack=[...{STACK_WEAVE}(stack),tmp_locals[i]];}};",
                pop!(state)
            )
        }
    }

    /// Generates JavaScript code for a true-tail `return_call`: packs the
    /// callee's arguments the same way [`JsWrite::call`] does, but instead of
    /// collecting results back onto this function's stack, directly
    /// `return`s the callee's call expression (a JS `return` is itself a
    /// tail call as far as this function's own stack/results are concerned —
    /// nothing runs afterward).
    fn return_call(
        &mut self,
        state: &State,
        sig: &FuncType,
        function_index: &(dyn Display + '_),
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        write!(
            self,
            "if({function_index}.__sig.params!={}||{function_index}.__sig.rets!={})throw new Error(`wasm sig mismatch`);",
            sig.params().len(),
            sig.results().len()
        )?;
        if let Some(opt) = state.opt() {
            let mut o = opt.lock();
            let s = o.depth - sig.params().len();
            let od = o.depth;
            o.depth -= sig.params().len();
            write!(
                self,
                "return {function_index}({});",
                DisplayFn(&|f| {
                    for n in (s + 1)..=od {
                        write!(f, "stack[{n}]")?;
                        if n != od {
                            write!(f, ",")?;
                        }
                    }
                    Ok(())
                })
            )
        } else {
            write!(
                self,
                "args=[];for(let i = 0;i < {function_index}.__sig.params;i++)args=[...{STACK_WEAVE}(args),{}];return {function_index}(...args);",
                pop!(state)
            )
        }
    }

    /// Generates JavaScript code for a branch (br) instruction.
    ///
    /// Creates a break or continue statement targeting the appropriate label
    /// based on the relative depth in the control flow stack.
    ///
    /// # Arguments
    ///
    /// * `sigs` - Array of function type signatures
    /// * `state` - The current compilation state
    /// * `idx` - The relative depth of the target label
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn br(&mut self, sigs: &[FuncType], state: &State, idx: u32) -> core::fmt::Result
    where
        Self: Sized,
    {
        let (idx, frame) = state
            .stack
            .iter()
            .enumerate()
            .rev()
            .nth(idx as usize)
            .unwrap();
        let idx = idx + 1;
        match frame {
            Frame::Block(blockty) | Frame::If(blockty) | Frame::TryTable(blockty, _) => {
                let result_count = match blockty {
                    portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                    portal_solutions_blitz_common::wasm_encoder::BlockType::Result(_) => 1,
                    portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                        sigs[*f as usize].results().len()
                    }
                };
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    let d = result_count;
                    let s = o.depth - d;
                    write!(
                        self,
                        "{{stack=[{}];break l{idx};}}",
                        DisplayFn(&|f| {
                            for n in s..o.depth {
                                write!(f, "stack[{n}]")?;
                                if n + 1 != o.depth {
                                    write!(f, ",")?;
                                }
                            }
                            Ok(())
                        })
                    )?;
                } else if result_count == 0 {
                    write!(self, "{{stack=[];break l{idx};}}")?;
                } else {
                    // Carry the top `result_count` values across the branch.
                    write!(self, "{{stack=stack.slice(-{result_count});break l{idx};}}")?;
                }
            }
            Frame::Loop(blockty) => {
                let param_count = match blockty {
                    portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                    portal_solutions_blitz_common::wasm_encoder::BlockType::Result(_) => 0,
                    portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                        sigs[*f as usize].params().len()
                    }
                };
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    let d = param_count;
                    let s = o.depth - d;
                    // BUG FIX: Loop branch is a back-edge; emit continue, not break.
                    write!(
                        self,
                        "{{stack=[{}];continue l{idx};}}",
                        DisplayFn(&|f| {
                            for n in s..o.depth {
                                write!(f, "stack[{n}]")?;
                                if n + 1 != o.depth {
                                    write!(f, ",")?;
                                }
                            }
                            Ok(())
                        })
                    )?;
                } else if param_count == 0 {
                    write!(self, "{{stack=[];continue l{idx};}}")?;
                } else {
                    write!(self, "{{stack=stack.slice(-{param_count});continue l{idx};}}")?;
                }
            }
            Frame::Function(_) => {
                // Branch to the function label = return the top `rets` values.
                return write!(self, "{{return stack.slice(-rets);}}");
            }
        };
        Ok(())
    }

    /// Generates JavaScript code for a single WASM instruction.
    ///
    /// Translates individual WASM operations into their JavaScript equivalents.
    /// Handles arithmetic, control flow, memory operations, and more.
    ///
    /// # Arguments
    ///
    /// * `sigs` - Array of function type signatures
    /// * `fsigs` - Function signature indices
    /// * `func_imports` - Information about imported functions
    /// * `state` - The current compilation state
    /// * `op` - The instruction to convert
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn on_op(
        &mut self,
        sigs: &[FuncType],
        fsigs: &[u32],
        tags: &[u32],
        func_imports: &[(&str, &str)],
        state: &mut State,
        op: &Instruction<'_>,
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        match op {
            Instruction::I64Const(value) => push(state, self, &format_args!("{}n", *value as u64)),
            Instruction::I32Const(value) => {
                push(state, self, &format_args!("{}n", *value as u32 as u64))
            }
            Instruction::I64Eqz | Instruction::I32Eqz => {
                push(state, self, &format_args!("({}===0n?1n:0n)", pop!(state)))
            }
            // Equality / comparison: sign-interpret for S variants, unsigned mask for U.
            Instruction::I32Eq | Instruction::I64Eq => push(
                state,
                self,
                &format_args!("({}==={}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32Ne | Instruction::I64Ne => push(
                state,
                self,
                &format_args!("({}!=={}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32LtS => push(
                state,
                self,
                // Pop order: rhs first; compare lhs OP rhs => first slot is lhs.
                &format_args!("(toInt({},32)>toInt({},32)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I64LtS => push(
                state,
                self,
                &format_args!("(toInt({},64)>toInt({},64)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32LtU | Instruction::I64LtU => push(
                state,
                self,
                &format_args!("((a={},b={})=>BigInt(b<a?1:0))()", pop!(state), pop!(state)),
            ),
            Instruction::I32GtS => push(
                state,
                self,
                &format_args!("(toInt({},32)>toInt({},32)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I64GtS => push(
                state,
                self,
                &format_args!("(toInt({},64)>toInt({},64)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32GtU | Instruction::I64GtU => push(
                state,
                self,
                &format_args!("((a={},b={})=>BigInt(b>a?1:0))()", pop!(state), pop!(state)),
            ),
            Instruction::I32LeS => push(
                state,
                self,
                &format_args!("(toInt({},32)>=toInt({},32)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I64LeS => push(
                state,
                self,
                &format_args!("(toInt({},64)>=toInt({},64)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32LeU | Instruction::I64LeU => push(
                state,
                self,
                &format_args!("((a={},b={})=>BigInt(b<=a?1:0))()", pop!(state), pop!(state)),
            ),
            Instruction::I32GeS => push(
                state,
                self,
                &format_args!("(toInt({},32)<=toInt({},32)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I64GeS => push(
                state,
                self,
                &format_args!("(toInt({},64)<=toInt({},64)?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::I32GeU | Instruction::I64GeU => push(
                state,
                self,
                &format_args!("((a={},b={})=>BigInt(b>=a?1:0))()", pop!(state), pop!(state)),
            ),
            // Bitwise ops work on the unsigned BigInt representations directly.
            Instruction::I32And => push(
                state,
                self,
                &format_args!("(({}&{})&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32Or => push(
                state,
                self,
                &format_args!("(({}|{})&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I32Xor => push(
                state,
                self,
                &format_args!("(({}^{})&mask32)()", pop!(state), pop!(state)),
            ),
            Instruction::I64And => push(
                state,
                self,
                &format_args!("(({}&{})&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64Or => push(
                state,
                self,
                &format_args!("(({}|{})&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I64Xor => push(
                state,
                self,
                &format_args!("(({}^{})&mask64)()", pop!(state), pop!(state)),
            ),
            Instruction::I32Clz => push(
                state,
                self,
                &format_args!("BigInt(32-Math.clz32(Number({})))", pop!(state)),
            ),
            Instruction::I32Ctz => push(
                state,
                self,
                &format_args!("BigInt(32-Math.clz32(Number({})&Number(-{})))", pop!(state), pop!(state)),
            ),
            Instruction::I32Popcnt => push(
                state,
                self,
                &format_args!("BigInt(__popcnt32(Number({})))", pop!(state)),
            ),
            Instruction::I64Clz => push(
                state,
                self,
                &format_args!("BigInt(64-Math.clz32(Number({})))", pop!(state)),
            ),
            Instruction::I64Ctz => push(
                state,
                self,
                &format_args!("BigInt(64-Math.clz32(Number({})&Number(-{})))", pop!(state), pop!(state)),
            ),
            Instruction::I64Popcnt => push(
                state,
                self,
                &format_args!("BigInt(__popcnt64({}))", pop!(state)),
            ),
            // Stack discipline: drop a value; nop is nothing.
            Instruction::Drop => {
                pop!(state);
                Ok(())
            }
            Instruction::Nop => Ok(()),
            Instruction::Unreachable => write!(self, "throw __wasm_trap('unreachable')"),
            Instruction::Select => push(
                state,
                self,
                &format_args!("(({}, {}, {}) => {} !== 0n ? {} : {})()", pop!(state), pop!(state), pop!(state), format_args!("{{let c={}}}", ""), pop!(state), pop!(state)),
            ),
            Instruction::I32Add => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a+b)&mask32)()", pop!(state), pop!(state)),
            ),
            // BUG FIX: a=first pop=rhs, b=second pop=lhs; use b-a (lhs-rhs).
            Instruction::I32Sub => push(
                state,
                self,
                &format_args!(
                    "((a={},b={})=>toUint((b-a)&mask32,32))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Mul => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a*b)&mask32)()", pop!(state), pop!(state)),
            ),
            // BUG FIX: swap operands — b/a = lhs/rhs.
            Instruction::I32DivU => push(
                state,
                self,
                &format_args!("__udiv({}, {}, 32)", pop!(state), pop!(state)),
            ),
            // BUG FIX: swap operands.
            Instruction::I32RemU => push(
                state,
                self,
                &format_args!("__urem({}, {}, 32)", pop!(state), pop!(state)),
            ),
            // BUG FIX: swap operands; also added missing ,32 to toUint.
            Instruction::I32DivS => push(
                state,
                self,
                &format_args!(
                    "__idivS(toInt({},32), toInt({},32), 32)",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: swap operands; also added missing ,32 to toUint.
            Instruction::I32RemS => push(
                state,
                self,
                &format_args!("__srem(toInt({},32), toInt({},32), 32)", pop!(state), pop!(state)),
            ),
            // BUG FIX: a=shift-count (rhs, first pop) modulo 32; b=value (lhs, second pop).
            Instruction::I32Shl => push(
                state,
                self,
                &format_args!(
                    "((a={}%32n,b={})=>(b<<a)&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: same — shift count is rhs (first pop).
            Instruction::I32ShrU => push(
                state,
                self,
                &format_args!(
                    "((a={}%32n,b={})=>(b>>a)&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: shift count (rhs=first pop) is unsigned; value (lhs=second pop) is sign-extended.
            // Also fixes misplaced ,32 — it was outside toUint in the original.
            Instruction::I32ShrS => push(
                state,
                self,
                &format_args!(
                    "((a={}%32n,b=toInt({},32))=>toUint((b>>a)&mask32,32))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: a=shift-count (rhs, first pop), b=value (lhs, second pop).
            Instruction::I32Rotl => push(
                state,
                self,
                &format_args!(
                    "((a={}%32n,b={})=>((b<<a)|(b>>(32n-a)))&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: same.
            Instruction::I32Rotr => push(
                state,
                self,
                &format_args!(
                    "((a={}%32n,b={})=>((b>>a)|(b<<(32n-a)))&mask32)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // 64 bit
            Instruction::I64Add => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a+b)&mask64)()", pop!(state), pop!(state)),
            ),
            // BUG FIX: a=rhs (first pop), b=lhs (second pop); use b-a.
            Instruction::I64Sub => push(
                state,
                self,
                &format_args!(
                    "((a={},b={})=>toUint((b-a)&mask64,64))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I64Mul => push(
                state,
                self,
                &format_args!("((a={},b={})=>(a*b)&mask64)()", pop!(state), pop!(state)),
            ),
            // BUG FIX: b/a = lhs/rhs.
            Instruction::I64DivU => push(
                state,
                self,
                &format_args!("__udiv({}, {}, 64)", pop!(state), pop!(state)),
            ),
            // BUG FIX: b%a = lhs%rhs.
            Instruction::I64RemU => push(
                state,
                self,
                &format_args!("__urem({}, {}, 64)", pop!(state), pop!(state)),
            ),
            // BUG FIX: swap operands; also add missing ,64 to toUint.
            Instruction::I64DivS => push(
                state,
                self,
                &format_args!("__idivS(toInt({},64), toInt({},64), 64)", pop!(state), pop!(state)),
            ),
            // BUG FIX: swap operands; also add missing ,64 to toUint.
            Instruction::I64RemS => push(
                state,
                self,
                &format_args!("__srem(toInt({},64), toInt({},64), 64)", pop!(state), pop!(state)),
            ),
            // BUG FIX: a=shift-count (rhs, first pop) modulo 64; b=value (lhs, second pop).
            Instruction::I64Shl => push(
                state,
                self,
                &format_args!(
                    "((a={}%64n,b={})=>(b<<a)&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: same — shift count is rhs (first pop).
            Instruction::I64ShrU => push(
                state,
                self,
                &format_args!(
                    "((a={}%64n,b={})=>(b>>a)&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: shift count unsigned (rhs=first pop), value sign-extended (lhs=second pop).
            // Also fixes misplaced ,64 — it was outside toUint in the original.
            Instruction::I64ShrS => push(
                state,
                self,
                &format_args!(
                    "((a={}%64n,b=toInt({},64))=>toUint((b>>a)&mask64,64))()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: a=shift-count (rhs, first pop), b=value (lhs, second pop).
            Instruction::I64Rotl => push(
                state,
                self,
                &format_args!(
                    "((a={}%64n,b={})=>((b<<a)|(b>>(64n-a)))&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // BUG FIX: same.
            Instruction::I64Rotr => push(
                state,
                self,
                &format_args!(
                    "((a={}%64n,b={})=>((b>>a)|(b<<(64n-a)))&mask64)()",
                    pop!(state),
                    pop!(state)
                ),
            ),
            // Wrap / extend / sign-extension ops.
            Instruction::I32WrapI64 => push(
                state,
                self,
                &format_args!("({}&mask32)", pop!(state)),
            ),
            Instruction::I64ExtendI32S => push(
                state,
                self,
                &format_args!("toInt({},32)&mask64", pop!(state)),
            ),
            Instruction::I64ExtendI32U => push(
                state,
                self,
                &format_args!("({}&mask32)", pop!(state)),
            ),
            Instruction::I32Extend8S => push(
                state,
                self,
                &format_args!("toInt({},8)&mask32", pop!(state)),
            ),
            Instruction::I32Extend16S => push(
                state,
                self,
                &format_args!("toInt({},16)&mask32", pop!(state)),
            ),
            Instruction::I64Extend8S => push(
                state,
                self,
                &format_args!("toInt({},8)&mask64", pop!(state)),
            ),
            Instruction::I64Extend16S => push(
                state,
                self,
                &format_args!("toInt({},16)&mask64", pop!(state)),
            ),
            Instruction::I64Extend32S => push(
                state,
                self,
                &format_args!("toInt({},32)&mask64", pop!(state)),
            ),
            // Float constants: reinterpret the exact bit pattern so the JS value
            // is bit-exact with the spec test data.
            // Float constants: reinterpret the exact bit pattern so the JS value
            // is bit-exact with the spec test data. Floats ride the wasm stack
            // as JS numbers; bit-exactness is restored via the DataView helpers
            // (`__f32bitsOf`/`__f64bitsOf` take a u32/u64 pattern, return a number).
            Instruction::F32Const(f) => {
                push(state, self, &format_args!("__f32bitsOf({})", f.bits()))
            }
            Instruction::F64Const(f) => {
                push(state, self, &format_args!("__f64bitsOf({}n)", f.bits()))
            }
            // Float ops. Floats ride the wasm stack as JS numbers (f32 ops are
            // computed in f64 then rounded to f32 via Math.fround).
            Instruction::F32Abs => push(
                state,
                self,
                &format_args!("Math.fround(Math.abs({}))", pop!(state)),
            ),
            Instruction::F32Neg => push(
                state,
                self,
                &format_args!("Math.fround(-{})", pop!(state)),
            ),
            Instruction::F32Ceil => push(
                state,
                self,
                &format_args!("Math.fround(Math.ceil({}))", pop!(state)),
            ),
            Instruction::F32Floor => push(
                state,
                self,
                &format_args!("Math.fround(Math.floor({}))", pop!(state)),
            ),
            Instruction::F32Trunc => push(
                state,
                self,
                &format_args!("Math.fround(Math.trunc({}))", pop!(state)),
            ),
            Instruction::F32Nearest => push(
                state,
                self,
                &format_args!("Math.fround(__nearest({}))", pop!(state)),
            ),
            Instruction::F32Sqrt => push(
                state,
                self,
                &format_args!("Math.fround(Math.sqrt({}))", pop!(state)),
            ),
            Instruction::F32Add => push(
                state,
                self,
                &format_args!("Math.fround({}+{})", pop!(state), pop!(state)),
            ),
            Instruction::F32Sub => push(
                state,
                self,
                &format_args!("Math.fround({}-{})", pop!(state), pop!(state)),
            ),
            Instruction::F32Mul => push(
                state,
                self,
                &format_args!("Math.fround({}*{})", pop!(state), pop!(state)),
            ),
            Instruction::F32Div => push(
                state,
                self,
                &format_args!("Math.fround({}/{})", pop!(state), pop!(state)),
            ),
            Instruction::F32Min => push(
                state,
                self,
                &format_args!("__fmin({},{});", pop!(state), pop!(state)),
            ),
            Instruction::F32Max => push(
                state,
                self,
                &format_args!("__fmax({},{});", pop!(state), pop!(state)),
            ),
            Instruction::F32Copysign => push(
                state,
                self,
                &format_args!("__copysign32({},{})", pop!(state), pop!(state)),
            ),
            Instruction::F64Abs => push(
                state,
                self,
                &format_args!("Math.abs({})", pop!(state)),
            ),
            Instruction::F64Neg => push(
                state,
                self,
                &format_args!("(-{})", pop!(state)),
            ),
            Instruction::F64Ceil => push(
                state,
                self,
                &format_args!("Math.ceil({})", pop!(state)),
            ),
            Instruction::F64Floor => push(
                state,
                self,
                &format_args!("Math.floor({})", pop!(state)),
            ),
            Instruction::F64Trunc => push(
                state,
                self,
                &format_args!("Math.trunc({})", pop!(state)),
            ),
            Instruction::F64Nearest => push(
                state,
                self,
                &format_args!("__nearest({})", pop!(state)),
            ),
            Instruction::F64Sqrt => push(
                state,
                self,
                &format_args!("Math.sqrt({})", pop!(state)),
            ),
            Instruction::F64Add => push(
                state,
                self,
                &format_args!("({}+{})", pop!(state), pop!(state)),
            ),
            Instruction::F64Sub => push(
                state,
                self,
                &format_args!("({}-{})", pop!(state), pop!(state)),
            ),
            Instruction::F64Mul => push(
                state,
                self,
                &format_args!("({}*{})", pop!(state), pop!(state)),
            ),
            Instruction::F64Div => push(
                state,
                self,
                &format_args!("({}/{})", pop!(state), pop!(state)),
            ),
            Instruction::F64Min => push(
                state,
                self,
                &format_args!("__fmin({},{});", pop!(state), pop!(state)),
            ),
            Instruction::F64Max => push(
                state,
                self,
                &format_args!("__fmax({},{});", pop!(state), pop!(state)),
            ),
            Instruction::F64Copysign => push(
                state,
                self,
                &format_args!("__copysign64({},{})", pop!(state), pop!(state)),
            ),
            // Float comparisons (result 1n/0n on the BigInt wasm stack).
            Instruction::F32Eq | Instruction::F64Eq => push(
                state,
                self,
                &format_args!("({}==={}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::F32Ne | Instruction::F64Ne => push(
                state,
                self,
                &format_args!("({}!=={}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::F32Lt | Instruction::F64Lt => push(
                state,
                self,
                &format_args!("({}<{}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::F32Gt | Instruction::F64Gt => push(
                state,
                self,
                &format_args!("({}>{}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::F32Le | Instruction::F64Le => push(
                state,
                self,
                &format_args!("({}<={}?1n:0n)", pop!(state), pop!(state)),
            ),
            Instruction::F32Ge | Instruction::F64Ge => push(
                state,
                self,
                &format_args!("({}>={}?1n:0n)", pop!(state), pop!(state)),
            ),
            // Conversions: int <-> float. Traps (trunc out-of-range/NaN) are
            // spec-critical; raise __wasm_trap for them.
            Instruction::F32ConvertI32S => push(
                state,
                self,
                &format_args!("Math.fround(Number(toInt({},32)))", pop!(state)),
            ),
            Instruction::F32ConvertI32U => push(
                state,
                self,
                &format_args!("Math.fround(Number({}&mask32))", pop!(state)),
            ),
            Instruction::F32ConvertI64S => push(
                state,
                self,
                &format_args!("Math.fround(Number(toInt({},64)))", pop!(state)),
            ),
            Instruction::F32ConvertI64U => push(
                state,
                self,
                &format_args!("Math.fround(Number(__u64ToF64({})))", pop!(state)),
            ),
            Instruction::F64ConvertI32S => push(
                state,
                self,
                &format_args!("Number(toInt({},32))", pop!(state)),
            ),
            Instruction::F64ConvertI32U => push(
                state,
                self,
                &format_args!("Number({}&mask32)", pop!(state)),
            ),
            Instruction::F64ConvertI64S => push(
                state,
                self,
                &format_args!("Number(toInt({},64))", pop!(state)),
            ),
            Instruction::F64ConvertI64U => push(
                state,
                self,
                &format_args!("__u64ToF64({})", pop!(state)),
            ),
            Instruction::F32DemoteF64 => push(
                state,
                self,
                &format_args!("Math.fround({})", pop!(state)),
            ),
            Instruction::F64PromoteF32 => push(
                state,
                self,
                &format_args!("({})", pop!(state)),
            ),
            Instruction::I32TruncF32S => push(
                state,
                self,
                &format_args!("__truncS({},32,Math.fround)", pop!(state)),
            ),
            Instruction::I32TruncF32U => push(
                state,
                self,
                &format_args!("__truncU({},32,Math.fround)", pop!(state)),
            ),
            Instruction::I32TruncF64S => push(
                state,
                self,
                &format_args!("__truncS({},32,x=>x)", pop!(state)),
            ),
            Instruction::I32TruncF64U => push(
                state,
                self,
                &format_args!("__truncU({},32,x=>x)", pop!(state)),
            ),
            Instruction::I64TruncF32S => push(
                state,
                self,
                &format_args!("__truncS({},64,Math.fround)", pop!(state)),
            ),
            Instruction::I64TruncF32U => push(
                state,
                self,
                &format_args!("__truncU({},64,Math.fround)", pop!(state)),
            ),
            Instruction::I64TruncF64S => push(
                state,
                self,
                &format_args!("__truncS({},64,x=>x)", pop!(state)),
            ),
            Instruction::I64TruncF64U => push(
                state,
                self,
                &format_args!("__truncU({},64,x=>x)", pop!(state)),
            ),
            Instruction::I32ReinterpretF32 => push(
                state,
                self,
                &format_args!("BigInt(__f32bits({}))", pop!(state)),
            ),
            Instruction::I64ReinterpretF64 => push(
                state,
                self,
                &format_args!("__f64bits({})", pop!(state)),
            ),
            Instruction::F32ReinterpretI32 => push(
                state,
                self,
                &format_args!("__f32bitsOf(Number({}))", pop!(state)),
            ),
            Instruction::F64ReinterpretI64 => push(
                state,
                self,
                &format_args!("__f64bitsOf({})", pop!(state)),
            ),
            //
            Instruction::Return => {
                write!(
                    self,
                    "if(stack.length===rets)return stack;tmp_locals=[];for(let i = 0; i < rets;i++)tmp_locals=[...{STACK_WEAVE}(tmp_locals),stack[stack.length-rets+i]];return tmp_locals;"
                )
            }
            Instruction::Call(function_index) => self.call(
                state,
                &sigs[fsigs[*function_index as usize] as usize],
                &format_args!("${function_index}"),
            ),

            // Pop order for call_indirect is args..., idx (idx on top). The
            // index is stashed in `var _idx` (not `tmp`, which `pop!`/`call`
            // itself uses internally as scratch and would clobber it) so it
            // survives until the callee reference is actually evaluated.
            // `$table_N[idx]` is the same function object `call()` uses for a
            // direct call, so `.__sig` gives a genuine per-element runtime
            // type check for free.
            Instruction::CallIndirect { type_index, table_index } => {
                write!(self, "var _idx={};", pop!(state))?;
                self.call(
                    state,
                    &sigs[*type_index as usize],
                    &format_args!("$table_{table_index}[Number(_idx)]"),
                )
            }

            Instruction::ReturnCall(function_index) => self.return_call(
                state,
                &sigs[fsigs[*function_index as usize] as usize],
                &format_args!("${function_index}"),
            ),

            Instruction::ReturnCallIndirect { type_index, table_index } => {
                write!(self, "var _idx={};", pop!(state))?;
                self.return_call(
                    state,
                    &sigs[*type_index as usize],
                    &format_args!("$table_{table_index}[Number(_idx)]"),
                )
            }
            Instruction::LocalGet(local_index) => {
                push(state, self, &format_args!("locals[{local_index}]"))
            }
            // Globals: module-scope `$g_N` BigInt slots (i32/i64). Float globals
            // are unsupported at synthesis level and rejected before codegen.
            Instruction::GlobalGet(global_index) => {
                push(state, self, &format_args!("$g_{global_index}"))
            }
            Instruction::GlobalSet(global_index) => {
                write!(self, "$g_{global_index}={}", pop!(state))
            }
            // BUG FIX: was `locals[{local_index}=` — missing `]` before `=`.
            Instruction::LocalSet(local_index) => {
                write!(self, "locals[{local_index}]={}", pop!(state))
            }
            // BUG FIX: same missing `]`; value must also be returned (tee leaves it on stack).
            Instruction::LocalTee(local_index) => push(
                state,
                self,
                &format_args!("(locals[{local_index}]={})", pop!(state)),
            ),
            Instruction::Block(blockty) => {
                state.stack.push(Frame::Block(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Result(
                            val_type,
                        ) => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                            sigs[*f as usize].params().len()
                        }
                    };
                }
                write!(self, "l{}: for(;;){{", state.stack.len())
            }
            Instruction::Loop(blockty) => {
                state.stack.push(Frame::Loop(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Result(
                            val_type,
                        ) => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                            sigs[*f as usize].params().len()
                        }
                    };
                }
                write!(self, "l{}: for(;;){{", state.stack.len())
            }
            Instruction::If(blockty) => {
                // Wrap in a labeled block so `br N` targeting this If frame can
                // use `break l{n}` to exit it (JavaScript allows labeled breaks
                // on any statement, not just loops).
                state.stack.push(Frame::If(blockty.clone()));
                write!(self, "l{}: {{if({}){{", state.stack.len(), pop!(state))
            }
            Instruction::Else => {
                write!(self, "}}else{{")
            }
            Instruction::End => {
                // Peek first: TryTable's catch dispatch uses `br` which needs the TryTable
                // frame still on the stack so label depths are computed correctly.
                // After the dispatch is emitted, the frame is popped.
                if let Some(Frame::TryTable(blockty, catches)) = state.stack.last() {
                    let blockty = blockty.clone();
                    let catches = catches.clone();
                    // Close try body and open catch block.
                    write!(self, "}}catch(__wasm_e){{")?;
                    for catch in &catches {
                        match catch {
                            portal_solutions_blitz_common::wasm_encoder::Catch::One { tag, label } => {
                                let arity = if (*tag as usize) < tags.len() {
                                    sigs[tags[*tag as usize] as usize].params().len()
                                } else { 0 };
                                write!(self, "if(__wasm_e?.__wasm_tag==={}n){{", tag)?;
                                for i in 0..arity {
                                    write!(self, "{};", DisplayFn(&|f| push(state, f, &format_args!("__wasm_e.__wasm_vals[{i}]"))))?;
                                }
                                write!(self, "{}}}", DisplayFn(&|f| f.br(sigs, state, *label)))?;
                            }
                            portal_solutions_blitz_common::wasm_encoder::Catch::All { label } => {
                                write!(self, "{{{}}}", DisplayFn(&|f| f.br(sigs, state, *label)))?;
                            }
                            portal_solutions_blitz_common::wasm_encoder::Catch::OneRef { .. }
                            | portal_solutions_blitz_common::wasm_encoder::Catch::AllRef { .. } => {
                                todo!("exnref catch deferred")
                            }
                        }
                    }
                    write!(self, "throw __wasm_e;}}")?;
                    if let Some(o) = state.opt() {
                        let mut o = o.lock();
                        o.depth = match blockty {
                            portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                            portal_solutions_blitz_common::wasm_encoder::BlockType::Result(_) => 1,
                            portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                        };
                    }
                    state.stack.pop();
                    return Ok(());
                }
                let s = match state.stack.pop() {
                    Some(s) => s,
                    // Function-level end (implicit outer block) — no frame to close.
                    None => return Ok(()),
                };
                match s {
                    Frame::Function(_) => {
                        // Function-level end: nothing to close; the function
                        // body's closing brace is emitted by EndBody.
                        return Ok(());
                    }
                    Frame::Block(blockty) => {
                        write!(self, "break;")?;
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty{
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Result(val_type) => 1,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                            };
                        }
                        write!(self, "}}")?;
                    }
                    Frame::Loop(blockty) => {
                        write!(self, "break;")?;
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty{
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Result(val_type) => 1,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                            };
                        }
                        write!(self, "}}")?;
                    }
                    Frame::If(blockty) => {
                        // Close if body, then close the labeled outer block wrapper.
                        write!(self, "}}}}")?;
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty {
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::Result(_) => 1,
                                portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                            };
                        }
                    }
                    Frame::TryTable(..) => unreachable!("TryTable handled before pop"),
                    Frame::Function(_) => {
                        // The function-level label frame: already consumed by
                        // the `stack.pop()` above; the function's closing brace
                        // is emitted by EndBody. Reset opt depth to zero.
                        if let Some(o) = state.opt() {
                            o.lock().depth = 0;
                        }
                    }
                }
                Ok(())
            }
            Instruction::Br(relative_depth) => self.br(sigs, state, *relative_depth),
            Instruction::BrIf(relative_depth) => write!(
                self,
                "if({}!==0n){}",
                pop!(state),
                DisplayFn(&|f| f.br(sigs, state, *relative_depth))
            ),
            Instruction::BrTable(targets, default) => {
                // BUG FIX: was `write!(self, "{}", pop!(state))` which discarded the
                // popped value — tmp was never assigned before the loop used it.
                write!(self, "tmp={};", pop!(state))?;
                for t in targets.iter().cloned() {
                    write!(
                        self,
                        "if(tmp===0n){{{}}};tmp--;",
                        DisplayFn(&|f| f.br(sigs, state, t))
                    )?;
                }
                self.br(sigs, state, *default)?;
                Ok(())
            }
            // ---- memory loads -----------------------------------------------
            // Loads use a one-pop arrow-function IIFE so the evaluated address
            // is captured in a local `_a` before passing to the DataView method.
            // `__wasm_dv(i)` resolves to `$mem_dv` for index 0 (harness-assigned)
            // or `$mem_dvs[i]` otherwise.
            Instruction::I64Load(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "((_a=Number({})+{off})=>BigInt.asUintN(64,BigInt(__wasm_dv({mem}).getBigUint64(_a,true))))()",
                    pop!(state)
                ))
            }
            Instruction::I32Load(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getUint32(Number({})+{off},true))",
                    pop!(state)
                ))
            }
            Instruction::I64Load8U(memarg) | Instruction::I32Load8U(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getUint8(Number({})+{off}))",
                    pop!(state)
                ))
            }
            Instruction::I64Load8S(memarg) | Instruction::I32Load8S(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getInt8(Number({})+{off}))",
                    pop!(state)
                ))
            }
            Instruction::I64Load16U(memarg) | Instruction::I32Load16U(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getUint16(Number({})+{off},true))",
                    pop!(state)
                ))
            }
            Instruction::I64Load16S(memarg) | Instruction::I32Load16S(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getInt16(Number({})+{off},true))",
                    pop!(state)
                ))
            }
            Instruction::I64Load32U(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getUint32(Number({})+{off},true))",
                    pop!(state)
                ))
            }
            Instruction::I64Load32S(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "BigInt(__wasm_dv({mem}).getInt32(Number({})+{off},true))",
                    pop!(state)
                ))
            }
            // ---- memory stores ----------------------------------------------
            // Use the arrow-function-with-defaults pattern (same as binary ops)
            // so that the two pop expressions don't clobber each other via `tmp`.
            // Pop order: value first (top of stack), then address.
            Instruction::I64Store(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setBigUint64(Number(a)+{off},v,true))()",
                    pop!(state), pop!(state))
            }
            Instruction::I32Store(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setUint32(Number(a)+{off},Number(v)&0xffffffff,true))()",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store8(memarg) | Instruction::I32Store8(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setUint8(Number(a)+{off},Number(v)&0xff))()",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store16(memarg) | Instruction::I32Store16(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setUint16(Number(a)+{off},Number(v)&0xffff,true))()",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store32(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setUint32(Number(a)+{off},Number(v)&0xffffffff,true))()",
                    pop!(state), pop!(state))
            }
            // Float stores / loads: bytes go through DataView float accessors so
            // the stored bits are exact.
            Instruction::F32Store(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setFloat32(Number(a)+{off},v,true))()",
                    pop!(state), pop!(state))
            }
            Instruction::F64Store(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                write!(self, "((v={},a={})=>__wasm_dv({mem}).setFloat64(Number(a)+{off},v,true))()",
                    pop!(state), pop!(state))
            }
            Instruction::F32Load(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "__wasm_dv({mem}).getFloat32(Number({})+{off},true)", pop!(state)
                ))
            }
            Instruction::F64Load(memarg) => {
                let off = memarg.offset;
                let mem = memarg.memory_index;
                push(state, self, &format_args!(
                    "__wasm_dv({mem}).getFloat64(Number({})+{off},true)", pop!(state)
                ))
            }
            // ---- memory.size / memory.grow ----------------------------------
            Instruction::MemorySize(mem) => {
                push(state, self, &format_args!("BigInt(__wasm_mb({mem}).byteLength/65536)"))
            }
            Instruction::MemoryGrow(mem) => {
                push(state, self, &format_args!("__wasm_grow({mem},{})", pop!(state)))
            }
            // ---- bulk memory --------------------------------------------------
            // Pop order for memory.copy/fill/init is len (top), then the middle
            // operand, then the bottom operand (same convention as the C backend
            // — mirrors native calling conventions for `memcpy`-shaped ops).
            Instruction::MemoryCopy { src_mem, dst_mem } => write!(
                self,
                "((n=Number({}),s=Number({}),d=Number({}))=>{{const mb=__wasm_mb({dst_mem});const sb=__wasm_mb({src_mem});if(d<=s)mb.set(sb.subarray(s,s+n),d);else for(let i=n-1;i>=0;i--)mb[d+i]=sb[s+i];}})()",
                pop!(state), pop!(state), pop!(state)
            ),
            Instruction::MemoryFill(mem) => write!(
                self,
                "((n=Number({}),v=Number({})&0xff,d=Number({}))=>__wasm_mb({mem}).fill(v,d,d+n))()",
                pop!(state), pop!(state), pop!(state)
            ),
            Instruction::MemoryInit { mem, data_index } => write!(
                self,
                "((n=Number({}),s=Number({}),d=Number({}))=>__wasm_mb({mem}).set(__wasm_data_seg_{data_index}.subarray(s,s+n),d))()",
                pop!(state), pop!(state), pop!(state)
            ),
            // No stack effect; segment liveness tracking is not modeled.
            Instruction::DataDrop(_) => write!(self, "0"),
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else {
                    0
                };
                write!(self, "throw{{__wasm_tag:{}n,__wasm_vals:[", tag_index)?;
                for i in 0..arity {
                    if i > 0 { write!(self, ",")?; }
                    write!(self, "{}", pop!(state))?;
                }
                write!(self, "]}}")
            }
            Instruction::TryTable(blockty, catches) => {
                state.stack.push(Frame::TryTable(
                    blockty.clone(),
                    catches.iter().cloned().collect(),
                ));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Empty => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::Result(_) => 0,
                        portal_solutions_blitz_common::wasm_encoder::BlockType::FunctionType(f) => {
                            sigs[*f as usize].params().len()
                        }
                    };
                }
                // Begin labeled try block; catch dispatch emitted at matching End.
                write!(self, "l{}:try{{", state.stack.len())
            }
            Instruction::ThrowRef => todo!("exnref deferred"),
            other => todo!("unimplemented WASM instruction in blitz-js: {other:?}"),
        }?;
        Ok(())
    }

    /// Generates JavaScript code for a machine operator.
    ///
    /// Handles high-level machine operations including function start/end markers,
    /// local variable declarations, and instruction execution.
    ///
    /// # Arguments
    ///
    /// * `sigs` - Array of function type signatures
    /// * `fsigs` - Function signature indices
    /// * `func_imports` - Information about imported functions
    /// * `state` - The current compilation state
    /// * `m` - The machine operator to process
    /// * `r` - Re-encoder for converting between instruction formats
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn on_mach<Annot>(
        &mut self,
        sigs: &[FuncType],
        fsigs: &[u32],
        tags: &[u32],
        func_imports: &[(&str, &str)],
        state: &mut State,
        m: &MachOperator<'_, Annot>,
        r: &mut impl Reencode,
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        match m {
            MachOperator::StartFn { id, data } => {
                let id = *id + func_imports.len() as u32;
                state.promise_close_stack.clear();
                // Push the implicit function-level branch label so `br` can
                // target the function itself (early return).
                state.stack.push(Frame::Function(BlockType::FunctionType(0)));
                write!(
                    self,
                    "
                    Object.defineProperty(${id},'__sig',{{
                        value:Object.freeze({{
                            params:{},
                            rets:{}
                        }}),
                        enumerable:false,
                        configurable:false,
                        writable:false
                    }});
                    function ${id}(...locals){{
                    let stack=[],tmp,mask32=0xffff_ffffn,mask64=(mask32<<32n)|mask32,{{params,rets}}=${id}.__sig,tmp_locals=[],args=[];
                    if(locals.length!==params){{
                        for(let i = 0; i < params;i++)tmp_locals=[...{STACK_WEAVE}(tmp_locals),locals[locals.length - params + i]];locals=tmp_locals;
                    }};
                    const toInt=(a,b)=>BigInt.asIntN(b,a);
                    const toUint=(a,b)=>BigInt.asUintN(b,a);
                    ",
                    data.num_params, data.num_returns
                )
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    write!(
                        self,
                        "locals=[...{STACK_WEAVE}(locals),{}];",
                        match ty {
                            ValType::F32 | ValType::F64 => "0",
                            _ => "0n",
                        }
                    )?
                }
                Ok(())
            }
            MachOperator::StartBody => Ok(()),
            MachOperator::Instruction { op, annot } => {
                self.on_op(sigs, fsigs, tags, func_imports, state, op)?;
                write!(self, ";")?;
                Ok(())
            }
            MachOperator::Operator { op, annot } => {
                let Some(op) = op.as_ref() else {
                    return Ok(());
                };
                let Ok(op) = r.instruction(op.clone()) else {
                    return Ok(());
                };
                self.on_op(sigs, fsigs, tags, func_imports, state, &op)?;
                write!(self, ";")?;
                Ok(())
            }
            MachOperator::EndBody => {
                // Pop the implicit function-level branch label. The function's
                // final `End` may already have popped it (DCE order / explicit
                // end handling) — accept both shapes.
                if let Some(Frame::Function(_)) = state.stack.last() {
                    state.stack.pop();
                }
                // Close promise-mode continuations innermost-first: each call
                // opened `const $cont_K=(val)=>{...` and left it open for the
                // remainder of the function.
                while let Some(k) = state.promise_close_stack.pop() {
                    write!(
                        self,
                        "}};if($call_{k} instanceof Promise)return $call_{k}.then($cont_{k});return $cont_{k}($call_{k});"
                    )?;
                }
                write!(self, "}}")
            }
            _ => todo!(),
        }
    }
}

/// Blanket implementation of JsWrite for all types that implement Write.
impl<T: Write + ?Sized> JsWrite for T {}

/// Multi-memory helpers shared by both preamble flavors (CommonJS and ESM).
///
/// `$mem`/`$mem_dv` remain index-0-only bindings (assigned directly by the
/// harness / `memory.grow`), so existing single-memory callers are unaffected.
/// `$mems`/`$mem_dvs` hold indices `1..` (sparse array — the harness or
/// `__wasm_grow` populates an entry the first time that memory index is used).
/// `__wasm_mb(i)`/`__wasm_dv(i)` are the accessors every load/store/bulk-memory
/// op goes through; `__wasm_grow(i, delta)` is `memory.grow`'s implementation,
/// keeping `$mem`/`$mem_dv` in sync for index 0 and `$mems[i]`/`$mem_dvs[i]`
/// otherwise.
const JS_MULTI_MEM_HELPERS: &str = "var $mems=[];var $mem_dvs=[];\
    function __wasm_mb(i){return i===0?$mem:$mems[i];}\
    function __wasm_dv(i){return i===0?$mem_dv:$mem_dvs[i];}\
    function __wasm_grow(i,d){var cur=__wasm_mb(i)||new Uint8Array(0);var o=BigInt(cur.byteLength/65536);\
    try{var n=new Uint8Array((Number(o)+Number(d))*65536);n.set(cur);\
    if(i===0){$mem=n;$mem_dv=new DataView(n.buffer);}else{$mems[i]=n;$mem_dvs[i]=new DataView(n.buffer);}\
    return o}catch(e){return -1n}}";

/// Emit module-level globals required for linear memory access (CommonJS / script).
///
/// Call this once before the first `on_mach` loop to declare `$mem` and
/// `$mem_dv` (plus the multi-memory helpers, see [`JS_MULTI_MEM_HELPERS`]) in
/// the generated JavaScript module scope.
pub fn js_module_preamble(w: &mut (dyn Write + '_)) -> core::fmt::Result {
    write!(w, "var $mem=new Uint8Array(0);var $mem_dv=new DataView($mem.buffer);")?;
    write!(w, "{JS_MULTI_MEM_HELPERS}")?;
    // Spec-exact integer division/remainder helpers (trap on div/0 and
    // INT_MIN/-1); used by the Div/Rem instruction arms.
    write!(w,
        "function __udiv(a,b,bits){{if(a===0n)throw {{__wasm_trap:true,message:'integer divide by zero'}};return BigInt.asUintN(bits,b/a);}}")?;
    write!(w,
        "function __urem(a,b,bits){{if(a===0n)throw {{__wasm_trap:true,message:'integer divide by zero'}};return BigInt.asUintN(bits,b%a);}}")?;
    write!(w,
        "function __idivS(a,b,bits){{if(a===0n)throw {{__wasm_trap:true,message:'integer divide by zero'}};const min=-(2n**BigInt(bits-1));if(b===min&&a===-1n)throw {{__wasm_trap:true,message:'integer overflow'}};return BigInt.asUintN(bits,b/a);}}")?;
    write!(w,
        "function __srem(a,b,bits){{if(a===0n)throw {{__wasm_trap:true,message:'integer divide by zero'}};return BigInt.asUintN(bits,b%a);}}")?;
    Ok(())
}

/// Emit module-level globals required for linear memory access (ES module).
///
/// Uses `let` instead of `var` for strict-mode ES module compatibility.
/// Call this once at the top of an ES module before the first `on_mach` loop.
pub fn js_module_preamble_esm(w: &mut (dyn Write + '_)) -> core::fmt::Result {
    write!(w, "let $mem=new Uint8Array(0);let $mem_dv=new DataView($mem.buffer);")?;
    write!(w, "{JS_MULTI_MEM_HELPERS}")
}


///
/// Each `(offset, bytes)` pair emits a `$mem.set([...], offset)` call.
/// This must appear **after** `$mem` has been resized to at least cover
/// `offset + bytes.len()`. Typically called after the test harness resizes
/// `$mem` but before invoking any compiled function.
pub fn js_apply_data_segments(
    w: &mut (dyn Write + '_),
    segments: &[(u32, &[u8])],
) -> core::fmt::Result {
    for (offset, bytes) in segments {
        write!(w, "$mem.set([")?;
        let mut first = true;
        for b in *bytes {
            if !first { write!(w, ",")?; }
            write!(w, "{b}")?;
            first = false;
        }
        write!(w, "],{offset});")?;
    }
    Ok(())
}

/// Emit one passive data segment, referenced by `memory.init { data_index, .. }`.
///
/// Unlike active segments (see [`js_apply_data_segments`]), passive segments
/// are not copied anywhere automatically — they just sit as a module-scope
/// `Uint8Array` under a name the `MemoryInit` codegen arm references directly
/// (`__wasm_data_seg_{data_index}`). Call once per passive segment, in any
/// order, before compiling a function that references it via `memory.init`.
pub fn js_emit_passive_data_segment(
    w: &mut (dyn Write + '_),
    data_index: u32,
    bytes: &[u8],
) -> core::fmt::Result {
    write!(w, "var __wasm_data_seg_{data_index}=new Uint8Array([")?;
    let mut first = true;
    for b in bytes {
        if !first { write!(w, ",")?; }
        write!(w, "{b}")?;
        first = false;
    }
    write!(w, "]);")
}

/// Emit a funcref table backing array for `call_indirect` / `return_call_indirect`.
///
/// `func_indices[i]` is the WASM function index (`$N`) stored at table slot
/// `i`; each entry is the same function object [`JsWrite::call`] uses for a
/// direct call, so indexing into it (`$table_N[idx]`) and reading `.__sig`
/// gives a genuine per-element runtime type check.
pub fn js_emit_funcref_table(
    w: &mut (dyn Write + '_),
    table_index: u32,
    func_indices: &[u32],
) -> core::fmt::Result {
    write!(w, "var $table_{table_index}=[")?;
    for (i, f) in func_indices.iter().enumerate() {
        if i > 0 { write!(w, ",")?; }
        write!(w, "${f}")?;
    }
    write!(w, "];")
}

/// Emit `var $N;` declarations for each imported function (CommonJS / script).
///
/// Import indices are 0-based WASM indices. The caller must assign these
/// variables before invoking any function that calls an import:
/// ```js
/// $0 = function(arg) { return [arg + 1n]; };
/// ```
pub fn js_emit_imports(w: &mut (dyn Write + '_), imports: &[(&str, &str)]) -> core::fmt::Result {
    for (i, (module, name)) in imports.iter().enumerate() {
        write!(w, "var ${i};// {module}::{name}\n")?;
    }
    Ok(())
}

/// Emit ES module import statements for each imported function.
///
/// Each import is rendered as:
/// ```js
/// import { name as _import_N } from 'module';
/// let $N = _import_N;
/// ```
///
/// The `_import_N` alias avoids identifier collisions when the same name is
/// imported from multiple modules.  `$N` is the identifier used by the
/// compiled WASM body.
pub fn js_emit_imports_esm(w: &mut (dyn Write + '_), imports: &[(&str, &str)]) -> core::fmt::Result {
    for (i, (module, name)) in imports.iter().enumerate() {
        write!(w, "import {{{name} as _import_{i}}} from '{module}';\nlet ${i}=_import_{i};\n")?;
    }
    Ok(())
}

/// Emit `var <name> = $N;` aliases for each exported function (CommonJS / script).
///
/// `wasm_idx` is the full WASM function index (import_count + internal_id).
pub fn js_emit_exports(w: &mut (dyn Write + '_), exports: &[(u32, &str)]) -> core::fmt::Result {
    for (wasm_idx, name) in exports {
        write!(w, "var {name}=${wasm_idx};\n")?;
    }
    Ok(())
}

/// Emit ES module `export` statements for each exported function.
///
/// Generates `export { $N as name };` for each `(wasm_idx, name)` pair.
/// The `$N` identifiers are the internal function variables emitted by
/// [`JsWrite::on_mach`]; they are valid JavaScript identifiers.
///
/// `wasm_idx` is the full WASM function index (import_count + internal_id).
pub fn js_emit_exports_esm(w: &mut (dyn Write + '_), exports: &[(u32, &str)]) -> core::fmt::Result {
    for (wasm_idx, name) in exports {
        write!(w, "export {{${wasm_idx} as {name}}};\n")?;
    }
    Ok(())
}
