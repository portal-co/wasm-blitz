//! [`BackendAbi`] implementations for x86-64.
//!
//! Provides `impl BackendAbi<W, Context> for NaiveAbi` (delegates to the naive
//! WASM calling convention) and `impl BackendAbi<W, Context> for SysVAbi`
//! (System V AMD64 register marshalling).

extern crate alloc;

use portal_solutions_asm_x86_64::{RegisterClass, X64Arch, out::arg::MemArgKind};
use portal_solutions_blitz_common::{
    abi::BackendAbi,
    asm::Reg,
    asm::common::mem::MemorySize,
    ops::FnData,
    wasm_encoder::FuncType,
};

use crate::{X64Label, RSP};
use crate::naive::WriterExt as NaiveWriterExt;
use crate::sysv::{SysVWriterExt, SysVState};

// SysV AMD64 argument registers: rdi, rsi, rdx, rcx, r8, r9
const ARG_REGS: [Reg; 6] = [
    Reg(7), // rdi
    Reg(6), // rsi
    Reg(2), // rdx
    Reg(1), // rcx
    Reg(8), // r8
    Reg(9), // r9
];
const RAX: Reg = Reg(0);
const RDX: Reg = Reg(2);
const RBP: Reg = Reg(5);

// ---------------------------------------------------------------------------
// Local ABI discriminant types
// ---------------------------------------------------------------------------

/// Blitz WASM calling convention for x86-64.
pub struct NaiveAbi;
/// System V AMD64 ABI.
pub struct SysVAbi;

// ---------------------------------------------------------------------------
// NaiveAbi — x86-64 blitz WASM calling convention
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for NaiveAbi
where
    W: NaiveWriterExt<Context> + ?Sized,
{
    type Error = W::Error;
    type State = crate::naive::State;
    type Arch = X64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut Self::State,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;
        w.pop(ctx, arch, &Reg(1))?;
        w.lea(
            ctx,
            arch,
            &Reg(0),
            &MemArgKind::Mem {
                base: Reg(1),
                offset: None,
                disp: 0u32.wrapping_sub(data.num_params as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.set_label(ctx, arch, X64Label::Func { r#fn: id })
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut Self::State,
    ) -> Result<(), W::Error> {
        state.local_count += 1;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_start_body(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut Self::State,
    ) -> Result<(), W::Error> {
        w.push(ctx, arch, &Reg(1))?;
        w.push(ctx, arch, &Reg(0))?;
        w.lea(
            ctx,
            arch,
            &Reg(0),
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub(state.control_depth as u32 * 16),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.push(ctx, arch, &Reg(0))?;
        for _ in 0..state.control_depth {
            w.push(ctx, arch, &Reg(0))?;
            w.push(ctx, arch, &Reg(0))?;
        }
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &Self::State,
        n: u32,
    ) -> Result<(), W::Error> {
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub((n as isize * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.pop(ctx, arch, &Reg(0))?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub(((n as isize + 1) * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &Self::State,
        n: u32,
    ) -> Result<(), W::Error> {
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub((n as isize * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.push(ctx, arch, &Reg(0))?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub(((n as isize + 1) * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &Self::State,
        n: u32,
    ) -> Result<(), W::Error> {
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub((n as isize * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.push(ctx, arch, &Reg(0))?;
        w.lea(
            ctx,
            arch,
            &RSP,
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub(((n as isize + 1) * 8) as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &Self::State,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        _sigs: &[FuncType],
        _fsigs: &[u32],
    ) -> Result<(), W::Error> {
        match func_imports.get(fn_idx as usize) {
            Some((module, name)) => {
                let sym = alloc::format!("{module}__{name}");
                w.lea_label(ctx, arch, &Reg(0), X64Label::External { name: sym })?;
                w.call(ctx, arch, &Reg(0))
            }
            None => {
                let idx = fn_idx - func_imports.len() as u32;
                w.lea_label(ctx, arch, &Reg(0), X64Label::Func { r#fn: idx })?;
                w.call(ctx, arch, &Reg(0))
            }
        }
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &Self::State,
    ) -> Result<(), W::Error> {
        w.mov(ctx, arch, &Reg(1), &RSP)?;
        w.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.lea(
            ctx,
            arch,
            &Reg(0),
            &MemArgKind::Mem {
                base: Reg(0),
                offset: None,
                disp: (state.local_count + 3 * 8) as u32,
                size: MemorySize::_8,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        w.mov(ctx, arch, &RSP, &Reg(0))?;
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.pop(ctx, arch, &Reg(0))?;
        for _ in 0..state.num_returns {
            w.mov(ctx, arch, &Reg(2), &Reg(1))?;
            w.push(ctx, arch, &Reg(2))?;
        }
        w.push(ctx, arch, &Reg(0))?;
        w.ret(ctx, arch)
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — System V AMD64 calling convention
// ---------------------------------------------------------------------------

impl<W, Context> BackendAbi<W, Context> for SysVAbi
where
    W: SysVWriterExt<Context> + ?Sized,
{
    type Error = W::Error;
    type State = SysVState;
    type Arch = X64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.param_count = data.num_params;
        state.ret_count = data.num_returns;
        state.local_count = data.num_params;

        // Emit SysV function label (offset avoids collision with naive Func labels)
        w.set_label(ctx, arch, X64Label::Indexed {
            idx: id as usize | (1 << 28),
        })?;

        // Standard AMD64 prologue
        w.push(ctx, arch, &RBP)?;
        w.mov(ctx, arch, &RBP, &RSP)?;

        // Allocate frame: params + 16 slots, 16-byte aligned
        let frame_sz = ((data.num_params + 16) * 8 + 15) & !15;
        w.mov64(ctx, arch, &RAX, frame_sz as u64)?;
        w.sub(ctx, arch, &RSP, &RAX)?;

        // Copy argument registers into the rbp-relative local frame
        for i in 0..data.num_params.min(6) {
            w.sysv_store_local(ctx, arch, ARG_REGS[i], i)?;
        }
        Ok(())
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState,
    ) -> Result<(), W::Error> {
        w.mov64(ctx, arch, &RAX, 0)?;
        w.sysv_store_local(ctx, arch, RAX, state.local_count)?;
        state.local_count += 1;
        Ok(())
    }

    fn emit_start_body(
        _w: &mut W,
        _ctx: &mut Context,
        _arch: X64Arch,
        _state: &mut SysVState,
    ) -> Result<(), W::Error> {
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &SysVState,
        n: u32,
    ) -> Result<(), W::Error> {
        w.sysv_load_local(ctx, arch, RAX, n as usize)?;
        w.push(ctx, arch, &RAX)
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &SysVState,
        n: u32,
    ) -> Result<(), W::Error> {
        w.pop(ctx, arch, &RAX)?;
        w.sysv_store_local(ctx, arch, RAX, n as usize)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &SysVState,
        n: u32,
    ) -> Result<(), W::Error> {
        w.sysv_peek(ctx, arch, RAX)?;
        w.sysv_store_local(ctx, arch, RAX, n as usize)
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &SysVState,
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

        // Pop WASM stack args into SysV argument registers.
        // WASM stack top = last argument; pop highest-indexed reg first.
        for i in (0..n_params.min(6)).rev() {
            w.pop(ctx, arch, &ARG_REGS[i])?;
        }

        // Call target
        if let Some((module, name)) = func_imports.get(fn_idx as usize) {
            let sym = alloc::format!("{module}__{name}");
            w.lea_label(ctx, arch, &RAX, X64Label::External { name: sym })?;
            w.call(ctx, arch, &RAX)?;
        } else {
            let local_id = fn_idx - func_imports.len() as u32;
            w.lea_label(ctx, arch, &RAX, X64Label::Indexed {
                idx: local_id as usize | (1 << 28),
            })?;
            w.call(ctx, arch, &RAX)?;
        }

        // Push return values onto WASM stack.
        // result1 (rdx) goes deeper, result0 (rax) on top.
        if n_results > 1 {
            w.push(ctx, arch, &RDX)?;
        }
        if n_results > 0 {
            w.push(ctx, arch, &RAX)?;
        }
        Ok(())
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &SysVState,
    ) -> Result<(), W::Error> {
        if state.ret_count > 0 {
            w.pop(ctx, arch, &RAX)?;
        }
        if state.ret_count > 1 {
            w.pop(ctx, arch, &RDX)?;
        }
        // Standard AMD64 epilogue: rsp = rbp, pop rbp, ret
        w.mov(ctx, arch, &RSP, &RBP)?;
        w.pop(ctx, arch, &RBP)?;
        w.ret(ctx, arch)
    }
}
