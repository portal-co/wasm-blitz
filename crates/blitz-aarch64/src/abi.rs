//! [`BackendAbi`] implementations for AArch64.
//!
//! Provides `impl BackendAbi<W, Context> for NaiveAbi` (blitz WASM ABI) and
//! `impl BackendAbi<W, Context> for SysVAbi` (AAPCS64 / ARM64 System V).

extern crate alloc;

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_aarch64::{
    AArch64Arch, RegisterClass,
    out::arg::{AddressingMode, ArgKind, MemArgKind},
};
use portal_solutions_blitz_common::{
    abi::BackendAbi,
    asm::Reg,
    ops::FnData,
    wasm_encoder::{Catch, FuncType},
};

use crate::AArch64Label;
use crate::naive::{State as NaiveState, WriterExt as NaiveWriterExt};
use crate::sysv::SysVWriterExt;
use crate::{FP, LR, SP};

// ---------------------------------------------------------------------------
// Local ABI discriminant types
// ---------------------------------------------------------------------------

/// Blitz WASM calling convention for AArch64.
pub struct NaiveAbi;
/// AAPCS64 / ARM64 System V ABI.
pub struct SysVAbi;

// AAPCS64 argument registers: X0–X7
const ARG_REGS: [Reg; 8] = [
    Reg(0),
    Reg(1),
    Reg(2),
    Reg(3),
    Reg(4),
    Reg(5),
    Reg(6),
    Reg(7),
];
// Scratch register (caller-saved)
const T0: Reg = Reg(9);

// ---------------------------------------------------------------------------
// AArch64 MemArgKind helpers
// ---------------------------------------------------------------------------

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: r,
        size: MemorySize::_64,
    })
}

fn mem_pre(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PreIndex,
    }
}

fn mem_post(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PostIndex,
    }
}

fn mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::Offset,
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — blitz WASM ABI for AArch64
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for NaiveAbi
where
    W: NaiveWriterExt<Context>,
{
    type Error = W::Error;
    type State<'s>
        = NaiveState<'s>
    where
        Self: 's;
    type Arch = AArch64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut NaiveState<'_>,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;

        w.set_label(ctx, arch, AArch64Label::Func { r#fn: id })?;

        // Function-entry probe: after label, before frame setup.
        state.probes = data.probes;
        state.next_probe_id = 1;
        if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
            let mut bw = crate::codegen::BlitzW::new(w, ctx, arch, 10);
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                0,
                T0.0,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }

        // Standard AArch64 prologue: save FP+LR, set FP=SP
        w.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16))?;
        w.mov(ctx, arch, &reg(FP), &reg(SP))?;

        // Allocate frame for locals + control flow slots
        let locals_slots = state.local_count as i64 + state.control_depth as i64 * 2 + 2;
        if locals_slots > 0 {
            let size = MemArgKind::NoMem(ArgKind::Lit((locals_slots * 8) as u64));
            w.sub(ctx, arch, &reg(SP), &reg(SP), &size)?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut NaiveState<'_>,
    ) -> Result<(), W::Error> {
        w.mov_imm(ctx, arch, &reg(T0), 0)?;
        w.store_local(ctx, arch, T0, state.local_count)?;
        state.local_count += 1;
        Ok(())
    }

    fn emit_start_body(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState<'_>,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.load_local(ctx, arch, T0, n as usize)?;
        w.wasm_push(ctx, arch, T0)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.wasm_pop(ctx, arch, T0)?;
        w.store_local(ctx, arch, T0, n as usize)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        // Peek at stack top without popping, then store to local
        w.ldr(ctx, arch, &reg(T0), &mem_base_disp(SP, 0))?;
        w.store_local(ctx, arch, T0, n as usize)
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        _sigs: &[FuncType],
        _fsigs: &[u32],
    ) -> Result<(), W::Error> {
        match func_imports.get(fn_idx as usize) {
            Some((module, name)) => {
                let sym = alloc::format!("{module}__{name}");
                crate::load_label_addr(
                    w,
                    ctx,
                    arch,
                    &reg(T0),
                    AArch64Label::External { name: sym },
                )?;
                w.bl(ctx, arch, &reg(T0))
            }
            None => {
                let idx = fn_idx - func_imports.len() as u32;
                w.adr_label(ctx, arch, &reg(T0), AArch64Label::Func { r#fn: idx })?;
                w.bl(ctx, arch, &reg(T0))
            }
        }
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
    ) -> Result<(), W::Error> {
        // Restore SP from FP, reload FP+LR, return
        w.mov(ctx, arch, &reg(SP), &reg(FP))?;
        w.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
        w.ret(ctx, arch)
    }

    fn emit_throw(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _tag_index: u32,
        _arity: u32,
    ) -> Result<(), W::Error> {
        todo!("emit_throw via BackendAbi — use handle_insn directly")
    }
    fn emit_try_table_start(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _catches: &[Catch],
        _sigs: &[FuncType],
        _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("emit_try_table_start via BackendAbi — use handle_insn directly")
    }
    fn emit_try_table_end(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _catches: &[Catch],
        _sigs: &[FuncType],
        _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("emit_try_table_end via BackendAbi — use handle_insn directly")
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — AAPCS64 calling convention
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for SysVAbi
where
    W: SysVWriterExt<Context>,
{
    type Error = W::Error;
    type State<'s>
        = NaiveState<'s>
    where
        Self: 's;
    type Arch = AArch64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut NaiveState<'_>,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;

        // SysV function label (offset avoids collision with naive Func labels)
        w.set_label(
            ctx,
            arch,
            AArch64Label::Indexed {
                idx: id as usize + 0x80000000,
            },
        )?;

        // Function-entry probe: after label, before frame setup so AAPCS64
        // arg regs (X0–X7) are delivered intact to the outer-JIT specialisation.
        if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
            let mut bw = crate::codegen::BlitzW::new(w, ctx, arch, 10);
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                0,
                T0.0,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }

        // Standard AAPCS64 prologue
        w.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16))?;
        w.mov(ctx, arch, &reg(FP), &reg(SP))?;

        // Allocate frame for locals + control flow slots
        let locals_slots = data.num_params as i64 + state.control_depth as i64 * 2 + 2;
        if locals_slots > 0 {
            let size = MemArgKind::NoMem(ArgKind::Lit((locals_slots * 8) as u64));
            w.sub(ctx, arch, &reg(SP), &reg(SP), &size)?;
        }

        // Copy argument registers into FP-relative local frame
        for i in 0..data.num_params.min(8) {
            let disp = -((i as i32 + 1) * 8);
            w.str(ctx, arch, &reg(ARG_REGS[i]), &mem_base_disp(FP, disp))?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut NaiveState<'_>,
    ) -> Result<(), W::Error> {
        w.mov_imm(ctx, arch, &reg(T0), 0)?;
        w.store_local(ctx, arch, T0, state.local_count)?;
        state.local_count += 1;
        Ok(())
    }

    fn emit_start_body(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState<'_>,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.load_local(ctx, arch, T0, n as usize)?;
        w.wasm_push(ctx, arch, T0)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.wasm_pop(ctx, arch, T0)?;
        w.store_local(ctx, arch, T0, n as usize)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.ldr(ctx, arch, &reg(T0), &mem_base_disp(SP, 0))?;
        w.store_local(ctx, arch, T0, n as usize)
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        _state: &NaiveState<'_>,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        sigs: &[FuncType],
        fsigs: &[u32],
    ) -> Result<(), W::Error> {
        // Determine param/result counts from the type table
        let (n_params, n_results) = match fsigs.get(fn_idx as usize) {
            Some(&sig_idx) => {
                let sig = &sigs[sig_idx as usize];
                (sig.params().len(), sig.results().len())
            }
            None => (0, 0),
        };

        // Pop WASM stack args into AAPCS64 argument registers.
        // WASM stack top = last argument; pop highest-indexed reg first.
        for i in (0..n_params.min(8)).rev() {
            w.wasm_pop(ctx, arch, ARG_REGS[i])?;
        }

        // Call target
        if let Some((module, name)) = func_imports.get(fn_idx as usize) {
            let sym = alloc::format!("{module}__{name}");
            crate::load_label_addr(w, ctx, arch, &reg(T0), AArch64Label::External { name: sym })?;
            w.bl(ctx, arch, &reg(T0))?;
        } else {
            let local_id = fn_idx - func_imports.len() as u32;
            w.adr_label(
                ctx,
                arch,
                &reg(T0),
                AArch64Label::Indexed {
                    idx: local_id as usize + 0x80000000,
                },
            )?;
            w.bl(ctx, arch, &reg(T0))?;
        }

        // Push return values onto WASM stack.
        // result1 (X1) goes deeper, result0 (X0) on top.
        if n_results > 1 {
            w.wasm_push(ctx, arch, Reg(1))?;
        }
        if n_results > 0 {
            w.wasm_push(ctx, arch, Reg(0))?;
        }
        Ok(())
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &NaiveState,
    ) -> Result<(), W::Error> {
        if state.num_returns > 0 {
            w.wasm_pop(ctx, arch, Reg(0))?;
        }
        if state.num_returns > 1 {
            w.wasm_pop(ctx, arch, Reg(1))?;
        }
        // AAPCS64 epilogue: restore SP from FP, reload FP+LR, ret
        w.mov(ctx, arch, &reg(SP), &reg(FP))?;
        w.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
        w.ret(ctx, arch)
    }

    fn emit_throw(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _tag_index: u32,
        _arity: u32,
    ) -> Result<(), W::Error> {
        todo!("SysVAbi exception handling requires platform unwinder — deferred; see docs/abi.md")
    }
    fn emit_try_table_start(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _catches: &[Catch],
        _sigs: &[FuncType],
        _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("SysVAbi exception handling requires platform unwinder — deferred; see docs/abi.md")
    }
    fn emit_try_table_end(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: AArch64Arch,
        _state: &mut NaiveState,
        _catches: &[Catch],
        _sigs: &[FuncType],
        _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("SysVAbi exception handling requires platform unwinder — deferred; see docs/abi.md")
    }
}
