//! [`BlitzWriter`] implementation for x86-64.

use portal_solutions_asm_x86_64::{
    ConditionCode, RegisterClass, X64Arch,
    out::{Writer, WriterCore, arg::{ArgKind, MemArgKind}},
};
use portal_solutions_blitz_common::asm::common::mem::MemorySize;
use portal_solutions_blitz_common::asm::Reg;
use crate::X64Label;

/// Wrapper binding an x86-64 writer + ctx + arch for [`portal_solutions_blitz_codegen::BlitzWriter`].
pub struct BlitzW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: X64Arch,
}

impl<'a, W, Context> portal_solutions_blitz_codegen::BlitzWriter for BlitzW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.jmp_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    fn branch_zero_label(&mut self, reg: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.cmp0(self.ctx, self.arch, &Reg(reg))?;
        self.writer.jcc_label(self.ctx, self.arch, ConditionCode::E, X64Label::Indexed { idx: label_idx })
    }

    fn branch_reg(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.jmp(self.ctx, self.arch, &Reg(reg))
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    // x86-64: add reg, 0xFFFF... (-1 as unsigned) = decrement
    fn reg_decrement(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.add(
            self.ctx, self.arch,
            &Reg(reg),
            &MemArgKind::NoMem(ArgKind::Lit(u64::MAX)),
        )
    }

    fn load_u64_imm(&mut self, dest: u8, imm: u64) -> Result<(), Self::Error> {
        self.writer.mov64(self.ctx, self.arch, &Reg(dest), imm)
    }

    // x86-64: ADD [ptr_reg], 1 — single instruction, no scratch needed
    fn inc_mem64(&mut self, ptr_reg: u8) -> Result<(), Self::Error> {
        self.writer.add(
            self.ctx, self.arch,
            &MemArgKind::Mem {
                base: ArgKind::Reg { reg: Reg(ptr_reg), size: MemorySize::_64 },
                offset: None, disp: 0,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )
    }

    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error> {
        self.writer.mov(
            self.ctx, self.arch,
            &Reg(dest),
            &MemArgKind::Mem {
                base: ArgKind::Reg { reg: Reg(src), size: MemorySize::_64 },
                offset: None, disp: 0,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )
    }
}
