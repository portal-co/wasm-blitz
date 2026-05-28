//! LFI-compliant AArch64 ABI implementation.
//!
//! Generates code accepted by `lfi-verify --arch arm64`:
//! - `x27` is the sandbox base (`rbase`), never written.
//! - `x28` is the address register, set only via `add x28, x27, wN, uxtw`.
//! - All WASM linear-memory accesses go through `[x28, #disp]`.
//! - The standard `ret` instruction is used (x30 is set by `bl` calls, so the
//!   AArch64 calling convention is already LFI-compatible at the ret level).
//! - Functions are aligned to the platform default (4-byte, fixed-width instructions).

extern crate alloc;

use portal_solutions_asm_aarch64::{
    out::{
        arg::{AddressingMode, ArgKind, MemArgKind},
        Writer, WriterCore,
    },
    AArch64Arch, ConditionCode, RegisterClass,
};
use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_blitz_common::{
    abi::BackendAbi,
    asm::Reg,
    ops::{FnData, MachOperator},
    wasm_encoder::{Catch, FuncType, Instruction, reencode::{self as reencode, Reencode}},
};

use crate::AArch64Label;
use crate::naive::WriterExt as NaiveWriterExt;
pub use crate::naive::State;

// ── LFI reserved registers ────────────────────────────────────────────────────

/// x27 — rbase (sandbox base). NEVER written by generated code.
const RBASE: Reg = Reg(27);
/// x28 — address register. Only set via `add x28, x27, wN, uxtw`.
const ADDR: Reg = Reg(28);

// ── Local register aliases ────────────────────────────────────────────────────
const SP: Reg = Reg(31);
const FP: Reg = Reg(29);
const LR: Reg = Reg(30);
const T0: Reg = Reg(9);
const T1: Reg = Reg(10);

// ── ABI discriminant ─────────────────────────────────────────────────────────

/// LFI sandboxed AArch64 ABI.
pub struct LfiAbi;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}
fn reg32(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_32 })
}
fn mem_disp(base: Reg, disp: i32, size: MemorySize) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size,
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

/// Emit LFI-sandboxed memory address: `add x28, x27, wT0, uxtw`.
///
/// This is the only permitted way to set `x28` in LFI code, and it
/// zero-extends T0 (32-bit WASM address) to a 64-bit offset from rbase.
fn lfi_addr<W: WriterCore<Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: AArch64Arch,
) -> Result<(), W::Error> {
    w.add_uxtw(ctx, arch, &reg(ADDR), &reg(RBASE), &reg32(T0))
}

/// LFI-compliant SP decrement: `sub xT, sp, #N; add sp, x27, wT, uxtw`.
///
/// Replaces the forbidden `sub sp, sp, #N` with the two-instruction form the
/// LFI verifier accepts. Uses T0 as a scratch register.
fn lfi_sub_sp<W: WriterCore<Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: AArch64Arch,
    n: u64,
) -> Result<(), W::Error> {
    w.sub(ctx, arch, &reg(T0), &reg(SP), &MemArgKind::NoMem(ArgKind::Lit(n)))?;
    w.add_uxtw(ctx, arch, &reg(SP), &reg(RBASE), &reg32(T0))
}

/// LFI-compliant epilogue: restore SP from FP then ldp + ret.
///
/// `add sp, x27, w29, uxtw; ldp x29, x30, [sp], #16; ret`
fn lfi_epilogue<W: WriterCore<Context> + Writer<crate::AArch64Label, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: AArch64Arch,
) -> Result<(), W::Error> {
    // add sp, x27, w29, uxtw  — LFI-compliant restore of SP from FP
    w.add_uxtw(ctx, arch, &reg(SP), &reg(RBASE), &reg32(FP))?;
    // ldp x29, x30, [sp], #16  — restore callee-saved regs, post-increment SP
    w.ldp(ctx, arch, &reg(FP), &reg(LR), &MemArgKind::Mem {
        base: ArgKind::Reg { reg: SP, size: MemorySize::_64 },
        offset: None,
        disp: 16,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PostIndex,
    })?;
    // LFI requires x30 to be "guarded" before ret: add x30, x27, w30, uxtw
    // This re-sandboxes x30 after restoring it from the stack.
    w.add_uxtw(ctx, arch, &reg(LR), &reg(RBASE), &reg32(LR))?;
    w.ret(ctx, arch)
}

// ── BackendAbi impl ───────────────────────────────────────────────────────────

impl<W, Context> BackendAbi<W, Context> for LfiAbi
where
    W: LfiWriterExt<Context>,
{
    type Error = W::Error;
    type State = State;
    type Arch = AArch64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;
        state.tracing = data.tracing;
        state.next_site_id = 1;
        // No explicit alignment needed — AArch64 instructions are fixed 4-byte.
        w.set_label(ctx, arch, AArch64Label::Func { r#fn: id })?;
        if let Some(cfg) = data.tracing.as_ref().copied().filter(|c| c.enabled) {
            let mut bw = crate::codegen::BlitzW::new(w, ctx, arch, T1.0);
            portal_solutions_blitz_codegen::emit_jit_preamble(
                &mut bw, cfg.table_base_off, 0,
                T0.0, &mut state.label_index,
            )?;
        }
        // stp x29, x30, [sp, #-16]!  — save FP and LR (LFI allows pre-decrement stp)
        w.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16))?;
        // mov x29, sp  — reads SP (ok), writes FP (ok)
        w.mov(ctx, arch, &reg(FP), &reg(SP))?;
        let locals_slots = state.local_count as i64 + state.control_depth as i64 * 2 + 2;
        if locals_slots > 0 {
            // LFI: sub x9, sp, #N; add sp, x27, w9, uxtw (modsp pattern)
            lfi_sub_sp(w, ctx, arch, (locals_slots * 8) as u64)?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State,
    ) -> Result<(), W::Error> {
        w.mov_imm(ctx, arch, &reg(T0), 0)?;
        state.local_count += 1;
        w.str(ctx, arch, &reg(T0), &mem_disp(FP, -((state.local_count as i32) * 8), MemorySize::_64))
    }

    fn emit_start_body(
        _w: &mut W, _ctx: &mut Context, _arch: AArch64Arch, _state: &mut State,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State,
        n: u32,
    ) -> Result<(), W::Error> {
        let disp = -((n as i32 + 1) * 8);
        w.ldr(ctx, arch, &reg(T0), &mem_disp(FP, disp, MemorySize::_64))?;
        w.wasm_push(ctx, arch, T0)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State,
        n: u32,
    ) -> Result<(), W::Error> {
        w.wasm_pop(ctx, arch, T0)?;
        let disp = -((n as i32 + 1) * 8);
        w.str(ctx, arch, &reg(T0), &mem_disp(FP, disp, MemorySize::_64))
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State,
        n: u32,
    ) -> Result<(), W::Error> {
        w.wasm_pop(ctx, arch, T0)?;
        w.wasm_push(ctx, arch, T0)?;
        let disp = -((n as i32 + 1) * 8);
        w.str(ctx, arch, &reg(T0), &mem_disp(FP, disp, MemorySize::_64))
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &State,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        _sigs: &[FuncType],
        _fsigs: &[u32],
    ) -> Result<(), W::Error> {
        match func_imports.get(fn_idx as usize) {
            Some((module, name)) => {
                let sym = alloc::format!("{module}__{name}");
                w.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: sym })?;
                w.bl(ctx, arch, &reg(T0))
            }
            None => {
                let idx = fn_idx - func_imports.len() as u32;
                w.adr_label(ctx, arch, &reg(T0), AArch64Label::Func { r#fn: idx })?;
                w.bl(ctx, arch, &reg(T0))
            }
        }
    }

    fn emit_throw(
        _w: &mut W, _ctx: &mut Context, _arch: AArch64Arch, _state: &mut State,
        _tag_index: u32, _arity: u32,
    ) -> Result<(), W::Error> {
        todo!("LFI aarch64 emit_throw")
    }

    fn emit_try_table_start(
        _w: &mut W, _ctx: &mut Context, _arch: AArch64Arch, _state: &mut State,
        _catches: &[Catch], _sigs: &[FuncType], _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("LFI aarch64 emit_try_table_start")
    }

    fn emit_try_table_end(
        _w: &mut W, _ctx: &mut Context, _arch: AArch64Arch, _state: &mut State,
        _catches: &[Catch], _sigs: &[FuncType], _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("LFI aarch64 emit_try_table_end")
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &State,
    ) -> Result<(), W::Error> {
        // LFI-compliant epilogue:
        // add sp, x27, w29, uxtw   ; restore SP from FP (LFI modsp form)
        // ldp x29, x30, [sp], #16  ; restore callee-saved regs
        // ret                       ; return via x30 (LFI allows ret targeting x30)
        lfi_epilogue(w, ctx, arch)
    }
}

// ── LfiWriterExt trait ────────────────────────────────────────────────────────

/// Extension trait for LFI-compliant AArch64 code generation.
pub trait LfiWriterExt<Context>: NaiveWriterExt<Context> {
    /// Handle a single WASM `Instruction` with LFI memory constraints.
    fn lfi_handle_insn(
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
        Self: Sized;

    /// Handle a `MachOperator` with LFI constraints.
    fn lfi_handle_op<E, Err>(
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
        Self: Sized,
        Err: From<Self::Error> + From<reencode::Error<E>>;
}

impl<W, Context> LfiWriterExt<Context> for W
where
    W: NaiveWriterExt<Context> + Writer<AArch64Label, Context>,
{
    fn lfi_handle_insn(
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
        match op {
            // ── LFI sandboxed loads ───────────────────────────────────────────
            // Pattern: pop WASM address into T0, compute sandboxed x28, load from [x28, #disp].
            Instruction::I64Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                lfi_addr(self, ctx, arch)?; // add x28, x27, w9, uxtw
                self.ldr(ctx, arch, &reg(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_64))?;
                self.wasm_push(ctx, arch, T1)
            }
            Instruction::I32Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                lfi_addr(self, ctx, arch)?;
                self.ldr(ctx, arch, &reg32(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_32))?;
                self.uxt(ctx, arch, &reg(T1), &reg32(T1))?;
                self.wasm_push(ctx, arch, T1)
            }
            Instruction::F64Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                lfi_addr(self, ctx, arch)?;
                self.ldr(ctx, arch, &reg(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_64))?;
                self.wasm_push(ctx, arch, T1)
            }
            Instruction::F32Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                lfi_addr(self, ctx, arch)?;
                self.ldr(ctx, arch, &reg32(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_32))?;
                self.uxt(ctx, arch, &reg(T1), &reg32(T1))?;
                self.wasm_push(ctx, arch, T1)
            }

            // ── LFI sandboxed stores ──────────────────────────────────────────
            Instruction::I64Store(m) | Instruction::F64Store(m) => {
                self.wasm_pop(ctx, arch, T1)?; // value
                self.wasm_pop(ctx, arch, T0)?; // address
                lfi_addr(self, ctx, arch)?;
                self.str(ctx, arch, &reg(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_64))
            }
            Instruction::I32Store(m) | Instruction::F32Store(m) => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                lfi_addr(self, ctx, arch)?;
                self.str(ctx, arch, &reg32(T1), &mem_disp(ADDR, m.offset as i32, MemorySize::_32))
            }

            // ── LFI-compliant return ──────────────────────────────────────────
            Instruction::Return => lfi_epilogue(self, ctx, arch),

            // ── Everything else: naive (already LFI-compatible) ───────────────
            other => self.handle_insn(ctx, arch, state, func_imports, sigs, tags, other, target),
        }
    }

    fn lfi_handle_op<E, Err>(
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
        Self: Sized,
        Err: From<Self::Error> + From<reencode::Error<E>>,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.tracing = data.tracing;
                state.next_site_id = 1;
                self.set_label(ctx, arch, AArch64Label::Func { r#fn: *id }).map_err(Err::from)?;
                if let Some(cfg) = data.tracing.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW::new(self, ctx, arch, T1.0);
                    portal_solutions_blitz_codegen::emit_jit_preamble(
                        &mut bw, cfg.table_base_off, 0,
                        T0.0, &mut state.label_index,
                    ).map_err(Err::from)?;
                }
                self.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP)).map_err(Err::from)?;
                let locals = state.local_count as i64 + state.control_depth as i64 * 2 + 2;
                if locals > 0 {
                    lfi_sub_sp(self, ctx, arch, (locals * 8) as u64).map_err(Err::from)?;
                }
                Ok(())
            }
            MachOperator::Instruction { op: insn, .. } => {
                self.lfi_handle_insn(ctx, arch, state, func_imports, sigs, tags, insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.lfi_handle_insn(ctx, arch, state, func_imports, sigs, tags, &insn, target)
                    .map_err(Err::from)
            }
            // Remaining variants delegate to naive.
            other => self.handle_op::<E, Err>(ctx, arch, state, func_imports, sigs, tags, other, rewriter, target),
        }
    }
}
