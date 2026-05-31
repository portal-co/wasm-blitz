//! C code generation backend for wasm-blitz.
//!
//! This crate compiles WebAssembly bytecode into C source code. The generated
//! C maintains WASM semantics using a simple `uint64_t` stack and local-variable
//! array. All integer values are stored as `uint64_t` and cast to the appropriate
//! width at each operation site.
//!
//! # Generated code shape
//!
//! ```c
//! #include <stdint.h>
//! #include <string.h>
//! #include <stdlib.h>
//! #define WASM_STACK_SIZE 512
//!
//! static const struct { int params; int rets; } __sig_0 = { .params=1, .rets=1 };
//! static uint64_t __rets_0[1];
//! static uint64_t* fn_0(uint64_t* restrict locals_in) {
//!     uint64_t locals_buf[1 + 0];
//!     memcpy(locals_buf, locals_in, 1 * sizeof(uint64_t));
//!     memset(locals_buf + 1, 0, 0 * sizeof(uint64_t));
//!     uint64_t* locals = locals_buf;
//!     uint64_t stack[WASM_STACK_SIZE];
//!     uint64_t tmp = 0, tmp2 = 0, *tmp_locals = 0;
//!     int sp = 0;
//!     /* ... body ... */
//!     memcpy(__rets_0, stack + sp - 1, 1 * sizeof(uint64_t));
//!     return __rets_0;
//! }
//! ```
//!
//! # Stack management
//!
//! Two modes mirror the JS backend:
//!
//! - **Standard mode**: uses a runtime `sp` index (`stack[sp++]` / `stack[--sp]`)
//! - **Optimized mode**: tracks stack depth statically; generates direct indices
//!   (`stack[N]`) with no runtime counter
//!
//! # Bugs fixed relative to the JS backend
//!
//! - `LocalSet`/`LocalTee`: missing `]` before `=` in array index expression
//! - `I32ShrS`/`I64ShrS`: closing paren of `toUint` was placed outside the bit-width argument
//! - All non-commutative binary ops (`Sub`, `Div`, `Rem`, `Shl`, `Shr`, `Rotl`, `Rotr`
//!   for both i32 and i64): first `pop` yields the **rhs** (top of stack) and second
//!   yields the **lhs**; the JS backend computed `rhs op lhs` instead of `lhs op rhs`
//! - `Frame::Loop` in optimized branch mode emitted `break` instead of `continue`
//! - `BrTable`: the popped index was discarded instead of assigned to `tmp`
//! - `call()` optimized mode: argument and result index loops were off by one
//!   (`s..od` gives indices `D-N..D-1` but items live at `D-N+1..D`)

#![no_std]
use core::{
    cell::OnceCell,
    fmt::{Display, Write},
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
};
use portal_solutions_blitz_opt::{self as blitz_opt, OptCodegen, OptState};
use spin::Mutex;
extern crate alloc;

// ---------------------------------------------------------------------------
// OptCodegen implementation
// ---------------------------------------------------------------------------

/// C implementation of the `OptCodegen` trait.
///
/// Generates C expressions for stack push/pop in both optimised (static-index)
/// and non-optimised (runtime-`sp`) modes.
pub struct CCodegen;

impl OptCodegen for CCodegen {
    fn write_opt_push_start(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
    ) -> core::fmt::Result {
        write!(w, "(tmp=({value})")
    }

    fn write_opt_push_end(&self, w: &mut (dyn Write + '_), index: usize) -> core::fmt::Result {
        write!(w, ",stack[{index}]=tmp)")
    }

    fn write_non_opt_push(
        &self,
        w: &mut (dyn Write + '_),
        value: &dyn Display,
    ) -> core::fmt::Result {
        write!(w, "(stack[sp++]=({value}))")
    }

    fn write_opt_pop(&self, w: &mut (dyn Write + '_), index: usize) -> core::fmt::Result {
        write!(w, "stack[{index}]")
    }

    fn write_non_opt_pop(&self, w: &mut (dyn Write + '_)) -> core::fmt::Result {
        write!(w, "stack[--sp]")
    }
}

// ---------------------------------------------------------------------------
// push / pop helpers
// ---------------------------------------------------------------------------

/// Push `a` onto the C execution stack.
pub fn push(state: &State, w: &mut (dyn Write + '_), a: &dyn Display) -> core::fmt::Result {
    blitz_opt::push(&CCodegen, state.opt(), w, a)
}

/// Pop a value from the C execution stack.
pub fn pop(state: &State, w: &mut (dyn Write + '_)) -> core::fmt::Result {
    blitz_opt::pop(&CCodegen, state.opt(), w)
}

/// Wraps `pop` as a `DisplayFn` for use inside `format_args!`.
#[macro_export]
macro_rules! pop {
    ($state:ident) => {
        $crate::__::DisplayFn(&|f| match $state {
            ref state => $crate::pop(state, f),
        })
    };
}

#[doc(hidden)]
pub use portal_solutions_blitz_opt::pop_display;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Compilation state for C code generation.
///
/// Tracks the control-flow frame stack, optional optimisation state, and the
/// metadata needed to emit function headers and return statements.
#[derive(Default)]
#[non_exhaustive]
pub struct State {
    stack: Vec<Frame>,
    opt_state: OnceCell<Mutex<OptState>>,

    /// Number of parameters for the function currently being compiled.
    pub param_count: usize,
    /// Number of return values for the function currently being compiled.
    pub ret_count: usize,
    /// Global function index of the function currently being compiled.
    pub fn_id: u32,
    /// Running count of non-parameter local variables accumulated so far.
    pub local_count: usize,
}

impl State {
    /// Enable optimised stack tracking. May be called at most once per `State`.
    pub fn enable_opt(&self, opt: impl FnOnce() -> OptState) {
        self.opt_state.get_or_init(|| Mutex::new(opt()));
    }

    fn opt(&self) -> Option<&Mutex<OptState>> {
        self.opt_state.get()
    }
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

enum Frame {
    Block(BlockType),
    Loop(BlockType),
    /// `If` is like `Block` for branching purposes: `br N` that targets an `if`
    /// frame is a forward exit out of the if/else body.
    If(BlockType),
    /// `TryTable` acts like `Block` for forward branching.
    /// Stores catch clauses for emission at the matching `End`.
    TryTable(BlockType, alloc::vec::Vec<portal_solutions_blitz_common::wasm_encoder::Catch>),
}

// ---------------------------------------------------------------------------
// CWrite trait
// ---------------------------------------------------------------------------

/// Trait for writing C code for WASM operations.
///
/// Blanket-implemented for every `Write` type; call methods on any buffered
/// string writer.
pub trait CWrite: Write {
    // ------------------------------------------------------------------
    // call()
    // ------------------------------------------------------------------

    /// Emit a C function call, including runtime signature validation.
    ///
    /// **Bug fix vs JS backend**: the JS opt-mode argument/result loops used
    /// `s..od` which is off by one (items are 1-indexed at `s+1..=od`).
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn call(
        &mut self,
        state: &mut State,
        sig: &FuncType,
        function_index: u32,
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        write!(
            self,
            "if(__sig_{function_index}.params!={0}||__sig_{function_index}.rets!={1})abort();",
            sig.params().len(),
            sig.results().len()
        )?;

        if let Some(opt) = state.opt() {
            let mut o = opt.lock();
            // s = index of element just *below* the first argument (1-based stack).
            // Arguments live at stack[s+1 .. s+N] (inclusive).
            let s = o.depth - sig.params().len();
            o.depth -= sig.params().len();
            let s2 = o.depth; // = s, base for result placement
            o.depth += sig.results().len();

            // BUG FIX: pass stack+s+1 so callee's locals[0] == stack[s+1]
            write!(self, "tmp_locals=fn_{function_index}(stack+{});", s + 1)?;

            // BUG FIX: results start at s2+1, not s2
            for i in 0..sig.results().len() {
                write!(self, "stack[{}]=tmp_locals[{i}];", s2 + i + 1)?;
            }
        } else {
            let n = sig.params().len();
            let m = sig.results().len();
            write!(
                self,
                "{{uint64_t*_ca=stack+sp-{n};sp-={n};tmp_locals=fn_{function_index}(_ca);memcpy(stack+sp,tmp_locals,{m}*sizeof(uint64_t));sp+={m};}}"
            )?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // br()
    // ------------------------------------------------------------------

    /// Emit a C `goto` for a branch instruction targeting a Block or Loop frame.
    ///
    /// **Bug fix vs JS backend**: the JS opt-mode Loop path emitted `break l{n}`
    /// instead of `continue l{n}`. For C we use `goto` throughout, so there is
    /// no such ambiguity.
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn br(&mut self, _sigs: &[FuncType], state: &State, relative_depth: u32) -> core::fmt::Result
    where
        Self: Sized,
    {
        let (enum_idx, frame) = state
            .stack
            .iter()
            .enumerate()
            .rev()
            .nth(relative_depth as usize)
            .unwrap();

        // Label index: same convention as JS (`enumerate` is 0-based, labels are 1-based).
        let label = enum_idx + 1;

        match frame {
            // Branch to a Block or If = forward jump to its exit label.
            Frame::Block(_) | Frame::If(_) => write!(self, "goto blk_e_{label};"),
            // BUG FIX vs JS: Loop branch is a *back*-edge (continue), not a break.
            Frame::Loop(_) => write!(self, "goto lp_s_{label};"),
            // Exiting a try_table must clean up the setjmp depth counter.
            Frame::TryTable(_, _) => write!(self, "__wasm_exn_d--;goto blk_e_{label};"),
        }
    }

    // ------------------------------------------------------------------
    // on_op()
    // ------------------------------------------------------------------

    /// Translate a single WASM instruction into C.
    // TODO: Remove the Sized bound once push/pop can work with ?Sized types
    fn on_op(
        &mut self,
        sigs: &[FuncType],
        fsigs: &[u32],
        tags: &[u32],
        _func_imports: &[(&str, &str)],
        state: &mut State,
        op: &Instruction<'_>,
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        match op {
            // ---- constants ------------------------------------------------
            Instruction::I32Const(value) => {
                push(state, self, &format_args!("(uint64_t)(uint32_t){}u", *value as u32))
            }
            Instruction::I64Const(value) => {
                push(state, self, &format_args!("(uint64_t){}ull", *value as u64))
            }

            // ---- zero tests -----------------------------------------------
            Instruction::I32Eqz => push(
                state,
                self,
                &format_args!("(uint64_t)((uint32_t){}==0u?1u:0u)", pop!(state)),
            ),
            Instruction::I64Eqz => push(
                state,
                self,
                &format_args!("(uint64_t)({}==0ull?1ull:0ull)", pop!(state)),
            ),

            // ---- i32 commutative ops (order doesn't matter) ---------------
            Instruction::I32Add => push(
                state,
                self,
                &format_args!(
                    "(uint64_t)(uint32_t)((uint32_t){}+(uint32_t){})",
                    pop!(state),
                    pop!(state)
                ),
            ),
            Instruction::I32Mul => push(
                state,
                self,
                &format_args!(
                    "(uint64_t)(uint32_t)((uint32_t){}*(uint32_t){})",
                    pop!(state),
                    pop!(state)
                ),
            ),

            // ---- i32 non-commutative ops -----------------------------------
            // BUG FIX vs JS: pop order is rhs-first (top of stack), then lhs.
            // JS computed `a op b` with a=rhs, b=lhs → wrong for non-commutative.
            // Here we capture: tmp = rhs (first pop), tmp2 = lhs (second pop),
            // then compute tmp2 op tmp = lhs op rhs.
            Instruction::I32Sub => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)(uint32_t)((uint32_t)tmp2-(uint32_t)tmp)"),
                )
            }
            Instruction::I32DivU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)(uint32_t)((uint32_t)tmp2/(uint32_t)tmp)"),
                )
            }
            Instruction::I32RemU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)(uint32_t)((uint32_t)tmp2%(uint32_t)tmp)"),
                )
            }
            Instruction::I32DivS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)(uint32_t)((int32_t)tmp2/(int32_t)tmp)"),
                )
            }
            Instruction::I32RemS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)(uint32_t)((int32_t)tmp2%(int32_t)tmp)"),
                )
            }
            Instruction::I32Shl => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!(
                        "(uint64_t)(uint32_t)((uint32_t)tmp2<<((uint32_t)tmp%32u))"
                    ),
                )
            }
            Instruction::I32ShrU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!(
                        "(uint64_t)(uint32_t)((uint32_t)tmp2>>((uint32_t)tmp%32u))"
                    ),
                )
            }
            // BUG FIX vs JS: JS had `toUint((a>>b)&mask32),32)` — the bit-width
            // argument `32` was *outside* the toUint call due to a misplaced paren.
            Instruction::I32ShrS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!(
                        "(uint64_t)(uint32_t)((int32_t)tmp2>>((uint32_t)tmp%32u))"
                    ),
                )
            }
            Instruction::I32Rotl => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!(
                        "(uint64_t)(uint32_t)(((uint32_t)tmp2<<((uint32_t)tmp%32u))|((uint32_t)tmp2>>(32u-(uint32_t)tmp%32u)))"
                    ),
                )
            }
            Instruction::I32Rotr => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!(
                        "(uint64_t)(uint32_t)(((uint32_t)tmp2>>((uint32_t)tmp%32u))|((uint32_t)tmp2<<(32u-(uint32_t)tmp%32u)))"
                    ),
                )
            }

            // ---- i64 commutative ops --------------------------------------
            Instruction::I64Add => push(
                state,
                self,
                &format_args!("{}+{}", pop!(state), pop!(state)),
            ),
            Instruction::I64Mul => push(
                state,
                self,
                &format_args!("{}*{}", pop!(state), pop!(state)),
            ),

            // ---- i64 non-commutative ops (same BUG FIX as i32) -----------
            Instruction::I64Sub => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(state, self, &format_args!("tmp2-tmp"))
            }
            Instruction::I64DivU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(state, self, &format_args!("tmp2/tmp"))
            }
            Instruction::I64RemU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(state, self, &format_args!("tmp2%tmp"))
            }
            Instruction::I64DivS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)((int64_t)tmp2/(int64_t)tmp)"),
                )
            }
            Instruction::I64RemS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)((int64_t)tmp2%(int64_t)tmp)"),
                )
            }
            Instruction::I64Shl => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(state, self, &format_args!("tmp2<<(tmp%64ull)"))
            }
            Instruction::I64ShrU => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(state, self, &format_args!("tmp2>>(tmp%64ull)"))
            }
            // BUG FIX vs JS: same misplaced-paren issue as I32ShrS, now gone.
            Instruction::I64ShrS => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(uint64_t)((int64_t)tmp2>>(tmp%64ull))"),
                )
            }
            Instruction::I64Rotl => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(tmp2<<(tmp%64ull))|(tmp2>>(64ull-tmp%64ull))"),
                )
            }
            Instruction::I64Rotr => {
                write!(self, "tmp={};tmp2={};", pop!(state), pop!(state))?;
                push(
                    state,
                    self,
                    &format_args!("(tmp2>>(tmp%64ull))|(tmp2<<(64ull-tmp%64ull))"),
                )
            }

            // ---- control flow ---------------------------------------------
            Instruction::Return => {
                let id = state.fn_id;
                let rets = state.ret_count;
                if let Some(opt) = state.opt() {
                    // In opt mode the stack items are 1-indexed; top `rets` items
                    // start at stack[depth - rets + 1].
                    let depth = opt.lock().depth;
                    let start = depth.saturating_sub(rets) + 1;
                    write!(
                        self,
                        "memcpy(__rets_{id},stack+{start},{rets}*sizeof(uint64_t));return __rets_{id};"
                    )
                } else {
                    write!(
                        self,
                        "memcpy(__rets_{id},stack+sp-{rets},{rets}*sizeof(uint64_t));return __rets_{id};"
                    )
                }
            }

            Instruction::Call(function_index) => self.call(
                state,
                &sigs[fsigs[*function_index as usize] as usize],
                *function_index,
            ),

            Instruction::LocalGet(local_index) => {
                push(state, self, &format_args!("locals[{local_index}]"))
            }

            // BUG FIX vs JS: JS wrote `locals[{n}=<pop>` (missing `]`).
            Instruction::LocalSet(local_index) => {
                write!(self, "locals[{local_index}]={}", pop!(state))
            }

            // BUG FIX vs JS: same missing `]` bug; also the value must be returned
            // (LocalTee leaves it on the stack).
            Instruction::LocalTee(local_index) => push(
                state,
                self,
                &format_args!("(locals[{local_index}]={})", pop!(state)),
            ),

            // ---- blocks / loops / if -------------------------------------
            Instruction::Block(blockty) => {
                state.stack.push(Frame::Block(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        BlockType::Empty => 0,
                        BlockType::Result(_) => 0,
                        BlockType::FunctionType(f) => sigs[*f as usize].params().len(),
                    };
                }
                write!(self, "{{/*blk_s_{}*/", state.stack.len())
            }

            Instruction::Loop(blockty) => {
                state.stack.push(Frame::Loop(blockty.clone()));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        BlockType::Empty => 0,
                        BlockType::Result(_) => 0,
                        BlockType::FunctionType(f) => sigs[*f as usize].params().len(),
                    };
                }
                // Emit the back-edge label before the loop body.
                write!(self, "lp_s_{0}:;{{", state.stack.len())
            }

            Instruction::If(blockty) => {
                state.stack.push(Frame::If(blockty.clone()));
                write!(self, "if((uint64_t){}!=0ull){{", pop!(state))
            }

            Instruction::Else => write!(self, "}}else{{"),

            Instruction::End => {
                // TryTable: handle early (before pop) so `br` inside catches sees the
                // TryTable frame in the stack and computes label depths correctly.
                if let Some(Frame::TryTable(blockty, catches)) = state.stack.last() {
                    let blockty = blockty.clone();
                    let catches = catches.clone();
                    let label = state.stack.len();
                    write!(self, "__wasm_exn_d--;}}else{{__wasm_exn_d--;")?;
                    for catch in &catches {
                        match catch {
                            portal_solutions_blitz_common::wasm_encoder::Catch::One { tag, label: catch_label } => {
                                let arity = if (*tag as usize) < tags.len() {
                                    sigs[tags[*tag as usize] as usize].params().len()
                                } else { 0 };
                                write!(self, "if(__wasm_exn.tag=={tag}u){{")?;
                                for i in 0..arity {
                                    write!(self, "{};", DisplayFn(&|f| push(state, f, &format_args!("__wasm_exn.vals[{i}]"))))?;
                                }
                                write!(self, "{}}}", DisplayFn(&|f| f.br(sigs, state, *catch_label)))?;
                            }
                            portal_solutions_blitz_common::wasm_encoder::Catch::All { label: catch_label } => {
                                write!(self, "{{{}}}", DisplayFn(&|f| f.br(sigs, state, *catch_label)))?;
                            }
                            portal_solutions_blitz_common::wasm_encoder::Catch::OneRef { .. }
                            | portal_solutions_blitz_common::wasm_encoder::Catch::AllRef { .. } => {
                                todo!("exnref catch deferred")
                            }
                        }
                    }
                    write!(self, "if(__wasm_exn_d>=0)longjmp(__wasm_exn_jmp[__wasm_exn_d],1);abort();")?;
                    write!(self, "}}blk_e_{label}:;}}")?;
                    if let Some(o) = state.opt() {
                        let mut o = o.lock();
                        o.depth = match blockty {
                            BlockType::Empty => 0,
                            BlockType::Result(_) => 1,
                            BlockType::FunctionType(f) => sigs[f as usize].results().len(),
                        };
                    }
                    state.stack.pop();
                    return Ok(());
                }
                // Retrieve the label *before* popping so we can emit it.
                let label = state.stack.len();
                let frame = match state.stack.pop() {
                    Some(f) => f,
                    // Function-level end (implicit outer block) — no frame to close.
                    None => return Ok(()),
                };
                match frame {
                    Frame::Block(blockty) => {
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty {
                                BlockType::Empty => 0,
                                BlockType::Result(_) => 1,
                                BlockType::FunctionType(f) => {
                                    sigs[f as usize].results().len()
                                }
                            };
                        }
                        // Label as empty statement at the exit point of the block.
                        write!(self, "blk_e_{label}:;}}")
                    }
                    Frame::Loop(blockty) => {
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty {
                                BlockType::Empty => 0,
                                BlockType::Result(_) => 1,
                                BlockType::FunctionType(f) => {
                                    sigs[f as usize].results().len()
                                }
                            };
                        }
                        // Close loop scope; no explicit back-edge needed here because
                        // WASM's fall-through off a loop end exits the loop.
                        write!(self, "}}")
                    }
                    Frame::If(blockty) => {
                        if let Some(o) = state.opt() {
                            let mut o = o.lock();
                            o.depth = match blockty {
                                BlockType::Empty => 0,
                                BlockType::Result(_) => 1,
                                BlockType::FunctionType(f) => {
                                    sigs[f as usize].results().len()
                                }
                            };
                        }
                        // Emit exit label so `goto blk_e_{label}` can target it.
                        write!(self, "blk_e_{label}:;}}")
                    }
                    Frame::TryTable(..) => unreachable!("TryTable handled before pop"),
                }
            }

            Instruction::Br(relative_depth) => self.br(sigs, state, *relative_depth),

            Instruction::BrIf(relative_depth) => write!(
                self,
                "if((uint64_t){}!=0ull){{{}}}",
                pop!(state),
                DisplayFn(&|f| f.br(sigs, state, *relative_depth))
            ),

            // BUG FIX vs JS: JS wrote `write!(self, "{}", pop!(state))` which
            // evaluated the pop as a void expression — `tmp` was never set.
            Instruction::BrTable(targets, default_target) => {
                write!(self, "tmp={};", pop!(state))?;
                for t in targets.iter().cloned() {
                    write!(
                        self,
                        "if(tmp==0ull){{{}}}tmp--;",
                        DisplayFn(&|f| f.br(sigs, state, t))
                    )?;
                }
                self.br(sigs, state, *default_target)?;
                Ok(())
            }

            // ---- memory loads -----------------------------------------------
            Instruction::I64Load(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "*(uint64_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I32Load(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)*(uint32_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load8U(memarg) | Instruction::I32Load8U(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)*(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load8S(memarg) | Instruction::I32Load8S(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)(int64_t)*(int8_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load16U(memarg) | Instruction::I32Load16U(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)*(uint16_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load16S(memarg) | Instruction::I32Load16S(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)(int64_t)*(int16_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load32U(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)*(uint32_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            Instruction::I64Load32S(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)(int64_t)*(int32_t*)(__wasm_mem+(uint32_t)tmp+{off}ull)"
                ))
            }
            // ---- memory stores ----------------------------------------------
            // Pop order: value first (top of stack), then address.
            Instruction::I64Store(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};tmp2={};*(uint64_t*)(__wasm_mem+(uint32_t)tmp2+{off}ull)=tmp",
                    pop!(state), pop!(state))
            }
            Instruction::I32Store(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};tmp2={};*(uint32_t*)(__wasm_mem+(uint32_t)tmp2+{off}ull)=(uint32_t)tmp",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store8(memarg) | Instruction::I32Store8(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};tmp2={};*(__wasm_mem+(uint32_t)tmp2+{off}ull)=(uint8_t)tmp",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store16(memarg) | Instruction::I32Store16(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};tmp2={};*(uint16_t*)(__wasm_mem+(uint32_t)tmp2+{off}ull)=(uint16_t)tmp",
                    pop!(state), pop!(state))
            }
            Instruction::I64Store32(memarg) => {
                let off = memarg.offset;
                write!(self, "tmp={};tmp2={};*(uint32_t*)(__wasm_mem+(uint32_t)tmp2+{off}ull)=(uint32_t)tmp",
                    pop!(state), pop!(state))
            }
            // ---- memory.size / memory.grow ----------------------------------
            Instruction::MemorySize(_) => {
                push(state, self, &format_args!("(uint64_t)__wasm_mem_pages"))
            }
            Instruction::MemoryGrow(_) => {
                write!(self, "tmp=(uint32_t){};", pop!(state))?;
                push(state, self, &format_args!(
                    "(uint64_t)__wasm_memory_grow((uint32_t)tmp,&__wasm_mem,&__wasm_mem_pages)"
                ))
            }
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else {
                    0
                };
                write!(self, "__wasm_exn.tag={tag_index}u;__wasm_exn.nvals={arity};")?;
                for i in (0..arity).rev() {
                    write!(self, "__wasm_exn.vals[{i}]={};", pop!(state))?;
                }
                write!(self, "if(__wasm_exn_d>=0)longjmp(__wasm_exn_jmp[__wasm_exn_d],1);abort()")
            }
            Instruction::TryTable(blockty, catches) => {
                state.stack.push(Frame::TryTable(
                    blockty.clone(),
                    catches.iter().cloned().collect(),
                ));
                if let Some(o) = state.opt() {
                    let mut o = o.lock();
                    o.depth = match blockty {
                        BlockType::Empty => 0,
                        BlockType::Result(_) => 0,
                        BlockType::FunctionType(f) => sigs[*f as usize].params().len(),
                    };
                }
                // `!setjmp` → normal path (no exception) runs the body.
                // At End we emit the else{dispatch}blk_e_N:;} suffix.
                let _ = state.stack.len(); // label index for blk_e is computed at End
                write!(self, "{{__wasm_exn_d++;if(!setjmp(__wasm_exn_jmp[__wasm_exn_d])){{")
            }
            Instruction::ThrowRef => todo!("exnref deferred"),
            _ => todo!(),
        }?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // on_mach()
    // ------------------------------------------------------------------

    /// Handle a machine-level operator, including function start/end markers and
    /// local variable declarations.
    ///
    /// Unlike the JS backend, local variable counts are accumulated during
    /// `Local` processing and the full function header (including the locals
    /// buffer) is emitted during `StartBody` once all counts are known.
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
                // Offset by number of imports so function indices match Call sites.
                let id = *id + func_imports.len() as u32;
                state.fn_id = id;
                state.param_count = data.num_params;
                state.ret_count = data.num_returns;
                state.local_count = 0;

                // Emit the signature struct and result buffer.
                // The function body itself is emitted in StartBody once we know
                // the total number of locals.
                write!(
                    self,
                    "static const struct{{int params;int rets;}}__sig_{id}={{.params={params},.rets={rets}}};static uint64_t __rets_{id}[{rets_sz}];",
                    params = data.num_params,
                    rets   = data.num_returns,
                    rets_sz = data.num_returns.max(1),
                )
            }

            // Accumulate local variable counts; all WASM locals are zero-initialised
            // so no initialisation code is needed here — memset in StartBody handles it.
            MachOperator::Local { count, ty: _ } => {
                state.local_count += *count as usize;
                Ok(())
            }

            // Emit the full function signature and prologue now that we know
            // the total number of locals.
            MachOperator::StartBody => {
                let id = state.fn_id;
                let params = state.param_count;
                let locals = state.local_count;
                write!(
                    self,
                    "static uint64_t*fn_{id}(uint64_t*restrict locals_in){{uint64_t locals_buf[{buf_sz}];memcpy(locals_buf,locals_in,{params}*sizeof(uint64_t));memset(locals_buf+{params},0,{locals}*sizeof(uint64_t));uint64_t*locals=locals_buf;uint64_t stack[WASM_STACK_SIZE];uint64_t tmp=0,tmp2=0,*tmp_locals=0;int sp=0;",
                    buf_sz = (params + locals).max(1),
                )
            }

            MachOperator::Instruction { op, annot: _ } => {
                self.on_op(sigs, fsigs, tags, func_imports, state, op)?;
                write!(self, ";")?;
                Ok(())
            }

            MachOperator::Operator { op, annot: _ } => {
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
                let id = state.fn_id;
                let rets = state.ret_count;
                write!(
                    self,
                    "memcpy(__rets_{id},stack+sp-{rets},{rets}*sizeof(uint64_t));return __rets_{id};}}"
                )
            }

            _ => todo!(),
        }
    }
}

/// Blanket implementation of `CWrite` for all `Write` types.
impl<T: Write + ?Sized> CWrite for T {}

/// Emit module-level globals and extern declarations for linear memory access.
///
/// Call once before the first `on_mach` loop. Declares:
/// - `__wasm_mem`       – pointer to the byte buffer (set by the caller)
/// - `__wasm_mem_pages` – current page count (set/updated by the caller)
/// - `__wasm_memory_grow` – extern function that the caller must provide to
///   implement `memory.grow`; signature:
///   `uint32_t __wasm_memory_grow(uint32_t delta, uint8_t** mem, uint32_t* pages)`
///   It returns the old page count (or `UINT32_MAX` on failure) and updates
///   `*mem` and `*pages`.
pub fn c_module_preamble(w: &mut (dyn core::fmt::Write + '_)) -> core::fmt::Result {
    write!(
        w,
        "#include <setjmp.h>\n\
         typedef struct{{uint32_t tag;uint64_t vals[64];int nvals;}}__wasm_exn_t;\
         static __wasm_exn_t __wasm_exn;\
         static jmp_buf __wasm_exn_jmp[64];\
         static int __wasm_exn_d=-1;\
         static uint8_t*__wasm_mem=0;\
         static uint32_t __wasm_mem_pages=0;\
         extern uint32_t __wasm_memory_grow(uint32_t delta,uint8_t**mem,uint32_t*pages);"
    )
}

/// Emit a `__wasm_init_data()` function that copies active data segments into
/// `__wasm_mem`.
///
/// Each `(byte_offset, bytes)` pair becomes one `memcpy` call.  The caller
/// must invoke `__wasm_init_data()` after `__wasm_mem` has been initialised
/// and is large enough to hold all segments.
pub fn c_emit_data_segments(
    w: &mut (dyn core::fmt::Write + '_),
    segments: &[(u32, &[u8])],
) -> core::fmt::Result {
    write!(w, "static void __wasm_init_data(void){{")?;
    for (i, (offset, bytes)) in segments.iter().enumerate() {
        write!(w, "static const uint8_t __data_{i}[]={{")?;
        let mut first = true;
        for b in *bytes {
            if !first { write!(w, ",")?; }
            write!(w, "{b}")?;
            first = false;
        }
        write!(w, "}};memcpy(__wasm_mem+{offset},__data_{i},sizeof __data_{i});")?;
    }
    write!(w, "}}")
}

// ---------------------------------------------------------------------------
// C System V ABI
// ---------------------------------------------------------------------------

/// State tracker for the C System V ABI code generator.
///
/// Identical to `State` but the function is emitted with an individual-argument
/// signature (`fn_N_sysv`) that wraps the blitz C ABI `fn_N` internally.
#[derive(Default)]
pub struct SysVState {
    fn_id: u32,
    param_count: usize,
    ret_count: usize,
}

/// Trait for generating C code with the System V compatible ABI.
///
/// Emits wrapper functions of the form:
/// ```c
/// // 0 returns
/// void fn_N_sysv(uint64_t arg0, ..., uint64_t argN);
/// // 1 return
/// uint64_t fn_N_sysv(uint64_t arg0, ..., uint64_t argN);
/// // 2+ returns
/// uint64_t fn_N_sysv(uint64_t arg0, ..., uint64_t argN, uint64_t* extra);
/// ```
/// The wrapper simply packs its arguments into a local array, calls the blitz C ABI
/// `fn_N`, and returns the result(s).  This is intentionally slower than a hand-coded
/// System V function but is correct and avoids duplicating the full code generator.
pub trait CWriteSysV: Write {
    fn on_mach_sysv<Annot>(
        &mut self,
        sigs: &[FuncType],
        fsigs: &[u32],
        func_imports: &[(&str, &str)],
        state: &mut SysVState,
        m: &MachOperator<'_, Annot>,
        r: &mut impl Reencode,
    ) -> core::fmt::Result
    where
        Self: Sized,
    {
        match m {
            MachOperator::StartFn { id, data } => {
                let id = *id + func_imports.len() as u32;
                state.fn_id = id;
                state.param_count = data.num_params;
                state.ret_count = data.num_returns;
                Ok(())
            }
            MachOperator::Local { .. } => Ok(()),
            MachOperator::StartBody => {
                let id = state.fn_id;
                let params = state.param_count;
                let rets = state.ret_count;

                // Emit static signature struct (same as blitz C ABI)
                write!(
                    self,
                    "static const struct{{int params;int rets;}}__sysv_sig_{id}={{\
                     .params={params},.rets={rets}}};"
                )?;

                // Build parameter list
                write!(self, "static ")?;
                match rets {
                    0 => write!(self, "void")?,
                    _ => write!(self, "uint64_t")?,
                }
                write!(self, " fn_{id}_sysv(")?;
                for i in 0..params {
                    if i > 0 { write!(self, ",")?; }
                    write!(self, "uint64_t _a{i}")?;
                }
                // extra out-pointer for 2+ returns
                if rets >= 2 {
                    if params > 0 { write!(self, ",")?; }
                    write!(self, "uint64_t*_extra_rets")?;
                }
                if params == 0 && rets < 2 {
                    write!(self, "void")?;
                }
                write!(self, "){{")?;

                // Pack args into array and call fn_N
                write!(self, "uint64_t _args[{sz}]={{", sz = params.max(1))?;
                for i in 0..params {
                    if i > 0 { write!(self, ",")?; }
                    write!(self, "_a{i}")?;
                }
                if params == 0 { write!(self, "0")?; }
                write!(self, "}};uint64_t*_r=fn_{id}(_args);")?;

                // Return results
                match rets {
                    0 => {}
                    1 => write!(self, "return _r[0];")?,
                    _ => {
                        // First result as direct return, rest via out-pointer
                        write!(self, "if(_extra_rets){{")?;
                        for i in 1..rets {
                            write!(self, "_extra_rets[{j}]=_r[{i}];", j = i - 1)?;
                        }
                        write!(self, "}}return _r[0];")?;
                    }
                }
                write!(self, "}}")
            }
            MachOperator::Instruction { .. }
            | MachOperator::Operator { .. }
            | MachOperator::EndBody => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Write + ?Sized> CWriteSysV for T {}

/// Emit `__sig_N` and `fn_N` function-pointer declarations for each imported function.
///
/// The caller must set each pointer before invoking any function that calls an import:
/// ```c
/// fn_0 = my_host_add_one;
/// ```
/// The function pointer follows the blitz C ABI: `uint64_t* (*)(uint64_t* restrict)`.
pub fn c_emit_import_decls(
    w: &mut (dyn core::fmt::Write + '_),
    imports: &[(&str, &str)],
    sigs: &[FuncType],
    fsigs: &[u32],
) -> core::fmt::Result {
    for (i, (module, name)) in imports.iter().enumerate() {
        let sig = &sigs[fsigs[i] as usize];
        write!(w,
            "static const struct{{int params;int rets;}}__sig_{i}={{.params={0},.rets={1}}};",
            sig.params().len(), sig.results().len()
        )?;
        write!(w, "static uint64_t*(*fn_{i})(uint64_t*restrict)=0;// {module}::{name}\n")?;
    }
    Ok(())
}

/// Emit alias functions that expose internal functions under their export names.
///
/// `exports` is a list of `(wasm_function_index, export_name)` where
/// `wasm_function_index` is the full WASM index (import_count + internal_id).
pub fn c_emit_exports(
    w: &mut (dyn core::fmt::Write + '_),
    exports: &[(u32, &str)],
) -> core::fmt::Result {
    for (wasm_idx, name) in exports {
        write!(w, "uint64_t*{name}(uint64_t*restrict __in){{return fn_{wasm_idx}(__in);}}\n")?;
    }
    Ok(())
}
