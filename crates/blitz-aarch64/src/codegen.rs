//! [`portal_solutions_blitz_codegen::BlitzWriter`] implementation for AArch64.

use portal_solutions_asm_aarch64::{
    AArch64Arch, ConditionCode, RegisterClass,
    out::{Writer, WriterCore, arg::{AddressingMode, ArgKind, MemArgKind}},
};
use portal_solutions_blitz_common::asm::{Reg, common::mem::MemorySize};
use crate::AArch64Label;

fn aarch64_reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}

fn aarch64_mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        mode: AddressingMode::Offset,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// Where the runtime trace-table base pointer is found, for `load_trace_base`.
///
/// NaiveAbi uses the CTX frame pointer; the AAPCS64 SysV ABI passes the base as
/// a virtual function parameter (a reserved register) and spills it to an
/// FP-relative frame slot for mid-function sites.  See `docs/abi.md`.
#[derive(Clone, Copy, Default)]
pub enum TraceBase {
    /// Base pointer stored at `[CTX + base_off]` (NaiveAbi / LFI).
    #[default]
    CtxSlot,
    /// Base pointer held directly in this blitz register (SysV virtual param).
    Reg(u8),
    /// Base pointer stored at `[FP + disp]` (SysV mid-function frame slot).
    FrameSlot(i32),
}

/// Wrapper binding an AArch64 writer + ctx + arch for
/// [`portal_solutions_blitz_codegen::BlitzWriter`].
///
/// `scratch2` is used by `inc_mem64` (load→add→store since AArch64
/// has no memory-immediate arithmetic).
pub struct BlitzW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: AArch64Arch,
    pub scratch2: u8,
    /// How `load_trace_base` reaches the runtime trace-table base.
    pub trace_base: TraceBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative trace-base convention.
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: AArch64Arch, scratch2: u8) -> Self {
        BlitzW { writer, ctx, arch, scratch2, trace_base: TraceBase::CtxSlot }
    }
}

impl<'a, W, Context> portal_solutions_blitz_codegen::BlitzWriter for BlitzW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<AArch64Label, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.b_label(self.ctx, self.arch, AArch64Label::Indexed { idx: label_idx })
    }

    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.cmp(
            self.ctx, self.arch,
            &aarch64_reg(Reg(reg_n)),
            &MemArgKind::NoMem(ArgKind::Lit(0)),
        )?;
        self.writer.bcond_label(
            self.ctx, self.arch, ConditionCode::EQ,
            AArch64Label::Indexed { idx: label_idx },
        )
    }

    fn branch_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.br(self.ctx, self.arch, &aarch64_reg(Reg(reg_n)))
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, AArch64Label::Indexed { idx: label_idx })
    }

    // AArch64: sub reg, reg, 1
    fn reg_decrement(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.sub(
            self.ctx, self.arch,
            &aarch64_reg(Reg(reg_n)), &aarch64_reg(Reg(reg_n)),
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )
    }

    fn load_u64_imm(&mut self, dest: u8, imm: u64) -> Result<(), Self::Error> {
        self.writer.mov_imm(self.ctx, self.arch, &aarch64_reg(Reg(dest)), imm)
    }

    // AArch64: ldr s2,[ptr]; add s2,s2,1; str s2,[ptr]
    fn inc_mem64(&mut self, ptr_reg: u8) -> Result<(), Self::Error> {
        let s2 = Reg(self.scratch2);
        self.writer.ldr(
            self.ctx, self.arch,
            &aarch64_reg(s2),
            &aarch64_mem_base_disp(Reg(ptr_reg), 0),
        )?;
        self.writer.add(
            self.ctx, self.arch,
            &aarch64_reg(s2), &aarch64_reg(s2),
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )?;
        self.writer.str(
            self.ctx, self.arch,
            &aarch64_reg(s2),
            &aarch64_mem_base_disp(Reg(ptr_reg), 0),
        )
    }

    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error> {
        self.load_mem64_disp(dest, src, 0)
    }

    // Load the runtime trace-table base into `dest` from the configured source.
    fn load_trace_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.trace_base {
            TraceBase::CtxSlot => self.writer.ldr(
                self.ctx, self.arch,
                &aarch64_reg(Reg(dest)),
                &aarch64_mem_base_disp(Reg::CTX, base_off),
            ),
            TraceBase::Reg(r) => {
                if r != dest {
                    self.writer.mov(self.ctx, self.arch, &aarch64_reg(Reg(dest)), &aarch64_reg(Reg(r)))?;
                }
                Ok(())
            }
            // FP = crate::FP (x29); mid-function frame slot.
            TraceBase::FrameSlot(disp) => self.writer.ldr(
                self.ctx, self.arch,
                &aarch64_reg(Reg(dest)),
                &aarch64_mem_base_disp(crate::FP, disp),
            ),
        }
    }

    // AArch64: ldr s2,[ptr+disp]; add s2,s2,1; str s2,[ptr+disp]
    fn inc_mem64_disp(&mut self, ptr_reg: u8, disp: i32) -> Result<(), Self::Error> {
        let s2 = Reg(self.scratch2);
        self.writer.ldr(
            self.ctx, self.arch,
            &aarch64_reg(s2),
            &aarch64_mem_base_disp(Reg(ptr_reg), disp),
        )?;
        self.writer.add(
            self.ctx, self.arch,
            &aarch64_reg(s2), &aarch64_reg(s2),
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )?;
        self.writer.str(
            self.ctx, self.arch,
            &aarch64_reg(s2),
            &aarch64_mem_base_disp(Reg(ptr_reg), disp),
        )
    }

    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.ldr(
            self.ctx, self.arch,
            &aarch64_reg(Reg(dest)),
            &aarch64_mem_base_disp(Reg(src), disp),
        )
    }
}
