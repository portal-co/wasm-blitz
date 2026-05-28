//! Naive AArch64 code generation — mirrors the RISC-V 64 backend structure.
//!
//! Calling convention: blitz WASM ABI (see docs/abi.md).
//! - SP  = Reg(31)/sp  — WASM operand stack
//! - FP  = Reg(29)/x29 — frame pointer
//! - LR  = Reg(30)/x30 — link register

use alloc::{collections::btree_map::BTreeMap, vec::Vec};
extern crate alloc;

use portal_solutions_asm_aarch64::{
    out::{
        arg::{AddressingMode, ArgKind, MemArgKind},
        Writer, WriterCore,
    },
    AArch64Arch, ConditionCode, RegisterClass,
};
use portal_pc_asm_common::types::mem::MemorySize;
#[allow(unused_imports)]
use portal_solutions_asm_aarch64::out::arg::MemArg;
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::{FnData, MachOperator, TracingConfig},
    wasm_encoder::{self, Catch, FuncType, Instruction, reencode::{self as reencode, Reencode}},
    wasmparser::Operator,
};

use crate::AArch64Label;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Code-generation state for an AArch64 function.
#[derive(Default)]
pub struct State {
    pub local_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub label_index: usize,
    /// Control-flow stack: each entry is the label pair (break_lbl, else_lbl_or_0)
    pub if_stack: Vec<Endable>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` so the tracing preamble is emitted
    /// after the function-entry label, ensuring every call-path is instrumented.
    pub tracing: Option<TracingConfig>,
    /// Next trace-site id to assign (function entry = site 0; each loop/block
    /// consumes the next).  See `emit_jit_preamble` / Item 1.
    pub next_site_id: u32,
}

/// Represents a control-flow frame.
#[derive(Clone)]
pub enum Endable {
    Block { end_lbl: AArch64Label },
    Loop  { head_lbl: AArch64Label },
    If    { else_lbl: AArch64Label, end_lbl: AArch64Label },
    TryTable {
        end_lbl: AArch64Label,
        dispatch_lbl: AArch64Label,
        catches: alloc::boxed::Box<[Catch]>,
    },
}

// ---------------------------------------------------------------------------
// Register helpers
// ---------------------------------------------------------------------------

/// WASM stack pointer = hardware SP.
const SP: Reg = Reg(31);
/// Frame pointer.
const FP: Reg = Reg(29);
/// Link register.
const LR: Reg = Reg(30);
/// Scratch registers (caller-saved / temporaries).
const T0: Reg = Reg(9);
const T1: Reg = Reg(10);
const T2: Reg = Reg(11);

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}
fn reg32(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_32 })
}
fn mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::Offset,
    }
}
fn mem_pre(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PreIndex,
    }
}
fn mem_post(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PostIndex,
    }
}

// ---------------------------------------------------------------------------
// Writer extension trait
// ---------------------------------------------------------------------------

/// Extension trait providing WASM code generation for AArch64 writers.
pub trait WriterExt<Context>: Writer<AArch64Label, Context> {
    /// Push `r` onto the WASM stack (pre-decrement SP).
    fn wasm_push(&mut self, ctx: &mut Context, arch: AArch64Arch, r: Reg)
        -> Result<(), Self::Error>
    {
        // str r, [sp, #-8]!
        self.str(ctx, arch, &reg(r), &mem_pre(SP, -8))
    }

    /// Pop top of WASM stack into `r`.
    fn wasm_pop(&mut self, ctx: &mut Context, arch: AArch64Arch, r: Reg)
        -> Result<(), Self::Error>
    {
        // ldr r, [sp], #8
        self.ldr(ctx, arch, &reg(r), &mem_post(SP, 8))
    }

    /// Load local variable N into `dest`.
    fn load_local(&mut self, ctx: &mut Context, arch: AArch64Arch, dest: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.ldr(ctx, arch, &reg(dest), &mem_base_disp(FP, disp))
    }

    /// Store `src` into local variable N.
    fn store_local(&mut self, ctx: &mut Context, arch: AArch64Arch, src: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.str(ctx, arch, &reg(src), &mem_base_disp(FP, disp))
    }

    // ---- binary op helpers ----
    fn pop2_push<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T1)?; // rhs / top
        self.wasm_pop(ctx, arch, T0)?; // lhs
        f(self, ctx, arch, T2, T0, T1)?;
        self.wasm_push(ctx, arch, T2)
    }

    // ---- compare helper ----
    fn cmp_push_bool(&mut self, ctx: &mut Context, arch: AArch64Arch, cc: ConditionCode)
        -> Result<(), Self::Error>
    {
        self.wasm_pop(ctx, arch, T1)?; // rhs
        self.wasm_pop(ctx, arch, T0)?; // lhs
        self.cmp(ctx, arch, &reg(T0), &reg(T1))?;
        self.mov_imm(ctx, arch, &reg(T0), 0)?;
        self.mov_imm(ctx, arch, &reg(T1), 1)?;
        self.csel(ctx, arch, cc, &reg(T2), &reg(T1), &reg(T0))?;
        self.wasm_push(ctx, arch, T2)
    }

    // ---- branch helper ----
    fn do_br(&mut self, ctx: &mut Context, arch: AArch64Arch, state: &State, depth: u32)
        -> Result<(), Self::Error>
    {
        let len = state.if_stack.len();
        let depth_usize = depth as usize;
        if depth_usize + 1 > len {
            // Branching out of the function (e.g., from an exception handler).
            // Emit a function exit: restore frame and return.
            self.mov(ctx, arch, &reg(SP), &reg(FP))?;
            self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
            return self.ret(ctx, arch);
        }
        let idx = len - depth_usize - 1;
        match state.if_stack[idx].clone() {
            Endable::TryTable { end_lbl, .. } => self.b_label(ctx, arch, end_lbl),
            Endable::Loop { head_lbl } => {
                self.b_label(ctx, arch, head_lbl)
            }
            Endable::Block { end_lbl } | Endable::If { end_lbl, .. } => {
                self.b_label(ctx, arch, end_lbl)
            }
        }
    }

    /// Emit a tracing/specialization preamble for a loop/block control-flow
    /// site, consuming the next `site_id`.  No-op when tracing is disabled.
    /// Uses T0 as scratch (T1 as the inner `inc_mem64` scratch).
    fn emit_trace_site(&mut self, ctx: &mut Context, arch: AArch64Arch, state: &mut State)
        -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.tracing.as_ref().copied().filter(|c| c.enabled) {
            let site_id = state.next_site_id;
            state.next_site_id += 1;
            let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: T1.0 };
            portal_solutions_blitz_codegen::emit_jit_preamble(
                &mut bw, cfg.table_base_off, site_id, T0.0, &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Handle a single WASM instruction (the inner match).
    fn handle_insn(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if target != state.body {
            // On the very first instruction of a function, `state.body` is
            // still the `Default::default()` value (0) and no body has been
            // entered yet.  Emitting a skip-jump here would reference a
            // `_idx_0` label that is never `set_label`'d (because we never
            // visited body 0), leaving an unresolved branch that on AArch64
            // assembles to `b .` (jump-to-self / infinite loop).  Detect
            // the uninitialized case via the empty body_labels map and just
            // adopt the new target body.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                let skip_lbl = *state.body_labels.entry(state.body).or_insert_with(|| {
                    state.label_index += 1;
                    state.label_index - 1
                });
                self.b_label(ctx, arch, AArch64Label::Indexed { idx: skip_lbl })?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, AArch64Label::Indexed { idx })?;
                }
            }
        }
        match op {
            // ---- constants ----
            Instruction::I64Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), *v as u64)?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::I32Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), *v as u32 as u64)?;
                self.wasm_push(ctx, arch, T0)
            }

            // ---- locals ----
            Instruction::LocalGet(idx) => {
                self.load_local(ctx, arch, T0, *idx as usize)?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::LocalSet(idx) => {
                self.wasm_pop(ctx, arch, T0)?;
                self.store_local(ctx, arch, T0, *idx as usize)
            }
            Instruction::LocalTee(idx) => {
                // peek (don't pop), store
                self.ldr(ctx, arch, &reg(T0), &mem_base_disp(SP, 0))?;
                self.store_local(ctx, arch, T0, *idx as usize)
            }

            // ---- i64 arithmetic ----
            Instruction::I64Add => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.add(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Sub => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.sub(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Mul => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.mul(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64DivU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.udiv(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64DivS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.sdiv(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64RemU => {
                // rem = a - (a / b) * b
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.udiv(ctx, arch, &reg(T2), &reg(T0), &reg(T1))?;
                self.mul(ctx, arch, &reg(T2), &reg(T2), &reg(T1))?;
                self.sub(ctx, arch, &reg(T2), &reg(T0), &reg(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64RemS => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sdiv(ctx, arch, &reg(T2), &reg(T0), &reg(T1))?;
                self.mul(ctx, arch, &reg(T2), &reg(T2), &reg(T1))?;
                self.sub(ctx, arch, &reg(T2), &reg(T0), &reg(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64And => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.and(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Or  => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.orr(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Xor => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.eor(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Shl => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.lsl(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64ShrU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.lsr(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64ShrS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.asr(c, a, &reg(d), &reg(x), &reg(y))),

            // ---- i32 arithmetic (zero-extend results to 64 bits) ----
            Instruction::I32Add => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.add(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32Sub => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sub(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32Mul => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.mul(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32DivU => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.udiv(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32DivS => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sdiv(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32And => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.and(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Or => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.orr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Xor => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.eor(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Shl => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.lsl(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32ShrU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.lsr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32ShrS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.asr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),

            // ---- comparisons ----
            Instruction::I64Eqz | Instruction::I32Eqz => {
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.mov_imm(ctx, arch, &reg(T0), 0)?;
                self.mov_imm(ctx, arch, &reg(T1), 1)?;
                self.csel(ctx, arch, ConditionCode::EQ, &reg(T2), &reg(T1), &reg(T0))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64Eq | Instruction::I32Eq => self.cmp_push_bool(ctx, arch, ConditionCode::EQ),
            Instruction::I64Ne | Instruction::I32Ne => self.cmp_push_bool(ctx, arch, ConditionCode::NE),
            Instruction::I64LtS | Instruction::I32LtS => self.cmp_push_bool(ctx, arch, ConditionCode::LT),
            Instruction::I64LtU | Instruction::I32LtU => self.cmp_push_bool(ctx, arch, ConditionCode::LO),
            Instruction::I64GtS | Instruction::I32GtS => self.cmp_push_bool(ctx, arch, ConditionCode::GT),
            Instruction::I64GtU | Instruction::I32GtU => self.cmp_push_bool(ctx, arch, ConditionCode::HI),
            Instruction::I64LeS | Instruction::I32LeS => self.cmp_push_bool(ctx, arch, ConditionCode::LE),
            Instruction::I64LeU | Instruction::I32LeU => self.cmp_push_bool(ctx, arch, ConditionCode::LS),
            Instruction::I64GeS | Instruction::I32GeS => self.cmp_push_bool(ctx, arch, ConditionCode::GE),
            Instruction::I64GeU | Instruction::I32GeU => self.cmp_push_bool(ctx, arch, ConditionCode::HS),

            // ---- memory loads (linear memory) ----
            Instruction::I64Load(m) => {
                self.wasm_pop(ctx, arch, T0)?; // address
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg(T1), &mem)?;
                self.wasm_push(ctx, arch, T1)
            }
            Instruction::I32Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg32(T1), &mem)?;
                self.uxt(ctx, arch, &reg(T1), &reg32(T1))?;
                self.wasm_push(ctx, arch, T1)
            }

            // ---- memory stores ----
            Instruction::I64Store(m) => {
                self.wasm_pop(ctx, arch, T1)?; // value
                self.wasm_pop(ctx, arch, T0)?; // address
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.str(ctx, arch, &reg(T1), &mem)
            }
            Instruction::I32Store(m) => {
                self.wasm_pop(ctx, arch, T1)?; // value
                self.wasm_pop(ctx, arch, T0)?; // address
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.str(ctx, arch, &reg32(T1), &mem)
            }

            // ---- memory.size / memory.grow ----
            Instruction::MemorySize(_) => {
                // Load address of __wasm_mem_pages, load 32-bit page count.
                self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_mem_pages".into() })?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg32(T0), &mem)?;
                self.uxt(ctx, arch, &reg(T0), &reg32(T0))?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::MemoryGrow(_) => {
                // delta is on WASM stack; call via blitz WASM convention.
                self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_memory_grow".into() })?;
                self.bl(ctx, arch, &reg(T0))
            }

            // ---- control flow ----
            Instruction::Block(_) => {
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                state.if_stack.push(Endable::Block { end_lbl });
                self.emit_trace_site(ctx, arch, state)?;
                Ok(())
            }
            Instruction::Loop(_) => {
                let head_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.set_label(ctx, arch, head_lbl.clone())?;
                state.if_stack.push(Endable::Loop { head_lbl });
                self.emit_trace_site(ctx, arch, state)?;
                Ok(())
            }
            Instruction::If(_) => {
                let else_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.bcond_label(ctx, arch, ConditionCode::EQ, else_lbl.clone())?;
                state.if_stack.push(Endable::If { else_lbl, end_lbl });
                Ok(())
            }
            Instruction::Else => {
                if let Some(Endable::If { else_lbl, end_lbl }) = state.if_stack.last().cloned() {
                    self.b_label(ctx, arch, end_lbl.clone())?;
                    self.set_label(ctx, arch, else_lbl)?;
                    *state.if_stack.last_mut().unwrap() = Endable::If {
                        else_lbl: AArch64Label::Indexed { idx: usize::MAX },
                        end_lbl,
                    };
                }
                Ok(())
            }
            Instruction::End => {
                if let Some(frame) = state.if_stack.pop() {
                    match frame {
                        Endable::Block { end_lbl } => self.set_label(ctx, arch, end_lbl)?,
                        Endable::If { end_lbl, .. } => self.set_label(ctx, arch, end_lbl)?,
                        Endable::Loop { .. } => {}
                        Endable::TryTable { end_lbl, dispatch_lbl, catches } => {
                            let after_lbl = AArch64Label::Indexed { idx: state.label_index };
                            state.label_index += 1;
                            // Normal path: jump past dispatch stub.
                            self.b_label(ctx, arch, after_lbl.clone())?;
                            // Dispatch stub.
                            self.set_label(ctx, arch, dispatch_lbl)?;
                            for catch in catches.iter() {
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else { 0 };
                                        let skip_lbl = AArch64Label::Indexed { idx: state.label_index };
                                        state.label_index += 1;
                                        self.mov_imm(ctx, arch, &reg(T1), *tag as u64)?;
                                        self.cmp(ctx, arch, &reg(T0), &reg(T1))?;
                                        self.bcond_label(ctx, arch, ConditionCode::NE, skip_lbl.clone())?;
                                        // Push exception values from x11, x12, x13 (arity 1, 2, 3)
                                        for i in (0..arity.min(3)).rev() {
                                            self.wasm_push(ctx, arch, Reg(11 + i as u8))?;
                                        }
                                        self.do_br(ctx, arch, state, *label)?;
                                        self.set_label(ctx, arch, skip_lbl)?;
                                    }
                                    Catch::All { label } => {
                                        self.do_br(ctx, arch, state, *label)?;
                                    }
                                    Catch::OneRef { .. } | Catch::AllRef { .. } => {}
                                }
                            }
                            self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_exn_propagate".into() })?;
                            self.bl(ctx, arch, &reg(T0))?;
                            self.set_label(ctx, arch, after_lbl)?;
                            self.set_label(ctx, arch, end_lbl)?;
                        }
                    }
                }
                Ok(())
            }
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else { 0 };
                self.mov_imm(ctx, arch, &reg(T0), *tag_index as u64)?; // tag in T0
                for i in 0..arity.min(3) {
                    self.wasm_pop(ctx, arch, Reg(11 + i as u8))?; // x11, x12, x13 for values
                }
                if let Some(dispatch_lbl) = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_lbl, .. } => Some(dispatch_lbl.clone()),
                    _ => None,
                }) {
                    self.b_label(ctx, arch, dispatch_lbl)
                } else {
                    self.adr_label(ctx, arch, &reg(T1), AArch64Label::External { name: "__wasm_exn_propagate".into() })?;
                    self.bl(ctx, arch, &reg(T1))
                }
            }
            Instruction::ThrowRef => todo!("exnref deferred"),
            Instruction::TryTable(_, catches) => {
                let dispatch_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                state.if_stack.push(Endable::TryTable {
                    end_lbl,
                    dispatch_lbl,
                    catches: catches.iter().cloned().collect::<alloc::vec::Vec<_>>().into_boxed_slice(),
                });
                Ok(())
            }
            Instruction::Br(depth) => self.do_br(ctx, arch, state, *depth),
            Instruction::BrIf(depth) => {
                let skip = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.bcond_label(ctx, arch, ConditionCode::EQ, skip.clone())?;
                self.do_br(ctx, arch, state, *depth)?;
                self.set_label(ctx, arch, skip)
            }
            Instruction::BrTable(targets, default) => {
                self.wasm_pop(ctx, arch, T0)?; // index
                let mut case_labels = Vec::new();
                for _ in targets.iter() {
                    case_labels.push(AArch64Label::Indexed { idx: state.label_index });
                    state.label_index += 1;
                }
                for (i, _) in targets.iter().enumerate() {
                    self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(i as u64)))?;
                    self.bcond_label(ctx, arch, ConditionCode::EQ, case_labels[i].clone())?;
                }
                self.do_br(ctx, arch, state, *default)?;
                for (i, target) in targets.iter().enumerate() {
                    self.set_label(ctx, arch, case_labels[i].clone())?;
                    self.do_br(ctx, arch, state, *target)?;
                }
                Ok(())
            }

            // ---- function calls ----
            Instruction::Call(fn_idx) => {
                match func_imports.get(*fn_idx as usize) {
                    Some((module, name)) => {
                        let sym = alloc::format!("{module}__{name}");
                        self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: sym })?;
                        self.bl(ctx, arch, &reg(T0))?;
                    }
                    None => {
                        let idx = *fn_idx - func_imports.len() as u32;
                        self.adr_label(ctx, arch, &reg(T0), AArch64Label::Func { r#fn: idx })?;
                        self.bl(ctx, arch, &reg(T0))?;
                    }
                }
                Ok(())
            }

            Instruction::Return => {
                // Restore SP from FP, reload FP+LR, return.
                self.mov(ctx, arch, &reg(SP), &reg(FP))?;
                self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
                self.ret(ctx, arch)
            }

            other => panic!("unimplemented WASM instruction in AArch64 naive handle_insn: {other:?}"),
        }
    }

    /// Emit the optional tracing preamble.
    ///
    /// - **NaiveAbi**: call from `StartBody` after the function-entry label.
    ///   Use `scratch = T0` (x9); FP and LR hold frame/return-addr.
    /// - **SysVAbi**: call from `StartFn` after `set_label`, before frame setup.
    ///   Use `scratch = T0` (x9); SysV arg regs (x0–x7) are untouched.
    /// Handle a `MachOperator` (the outer match, called by the pipeline).
    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;

                self.set_label(ctx, arch, AArch64Label::Func { r#fn: *id }).map_err(Err::from)?;

                state.tracing = data.tracing;
                state.next_site_id = 1;
                if let Some(cfg) = data.tracing.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: T1.0 };
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, cfg.table_base_off, 0,
                        T0.0, &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                self.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP)).map_err(Err::from)?;

                let locals_slots = state.local_count as i64 + state.control_depth as i64 * 2 + 2;
                if locals_slots > 0 {
                    let size = MemArgKind::NoMem(ArgKind::Lit((locals_slots * 8) as u64));
                    self.sub(ctx, arch, &reg(SP), &reg(SP), &size).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov_imm(ctx, arch, &reg(T0), 0).map_err(Err::from)?;
                for _ in 0..*count {
                    state.local_count += 1;
                    self.store_local(ctx, arch, T0, state.local_count - 1).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::StartBody => Ok(()),
            MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.handle_insn(ctx, arch, state, func_imports, sigs, tags, insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.handle_insn(ctx, arch, state, func_imports, sigs, tags, &insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()), // non-instruction operators (Local, StartBody, etc.) silently ignored
        }
    }
}

impl<T: Writer<AArch64Label, Context> + ?Sized, Context> WriterExt<Context> for T {}

/// Emit one-instruction jump stubs for each exported function.
///
/// Each stub emits an `External` label followed by a `b_label` to the
/// function's internal label.
pub fn emit_export_dispatchers<W, Ctx>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: AArch64Arch,
    exports: &[(u32, &str)],
) -> Result<(), W::Error>
where
    W: WriterExt<Ctx>,
{
    for (id, name) in exports {
        w.set_label(ctx, arch, AArch64Label::External { name: (*name).into() })?;
        w.b_label(ctx, arch, AArch64Label::Func { r#fn: *id })?;
    }
    Ok(())
}
