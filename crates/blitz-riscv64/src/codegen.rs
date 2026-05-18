//! [`portal_solutions_blitz_codegen::BlitzWriter`] implementation for RISC-V 64.

use portal_solutions_asm_riscv64::{
    ConditionCode, RegisterClass, RiscV64Arch,
    out::{Writer, WriterCore, arg::{ArgKind, MemArgKind}},
};
use portal_solutions_blitz_common::asm::{Reg, common::mem::MemorySize};
use crate::RiscvLabel;

fn riscv_reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}

fn riscv_mem_base(base: Reg) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None, disp: 0,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// Wrapper binding a RISC-V writer + ctx + arch for
/// [`portal_solutions_blitz_codegen::BlitzWriter`].
///
/// `scratch2` is used by `inc_mem64` (ld→addi→sd since RISC-V has no
/// memory-immediate arithmetic).
pub struct BlitzW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: RiscV64Arch,
    pub scratch2: u8,
}

impl<'a, W, Context> portal_solutions_blitz_codegen::BlitzWriter for BlitzW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<RiscvLabel, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        // JAL x0, label — unconditional jump, discard return addr into x0
        self.writer.jal_label(
            self.ctx, self.arch,
            &riscv_reg(Reg(0)),
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    // RISC-V: beq reg, x0, label — natural two-register form
    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.bcond_label(
            self.ctx, self.arch,
            ConditionCode::EQ,
            &riscv_reg(Reg(reg_n)),
            &riscv_reg(Reg(0)), // x0 = zero register
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    fn branch_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        // JALR x0, reg, 0 — indirect jump, discard return addr into x0
        self.writer.jalr(self.ctx, self.arch, &riscv_reg(Reg(0)), &riscv_reg(Reg(reg_n)), 0)
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, RiscvLabel::Indexed { idx: label_idx })
    }

    // RISC-V: addi reg, reg, -1
    fn reg_decrement(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.addi(self.ctx, self.arch, &riscv_reg(Reg(reg_n)), &riscv_reg(Reg(reg_n)), -1)
    }

    fn load_u64_imm(&mut self, dest: u8, imm: u64) -> Result<(), Self::Error> {
        self.writer.li(self.ctx, self.arch, &riscv_reg(Reg(dest)), imm)
    }

    // RISC-V: ld s2, 0(ptr); addi s2, s2, 1; sd s2, 0(ptr)
    fn inc_mem64(&mut self, ptr_reg: u8) -> Result<(), Self::Error> {
        let s2 = Reg(self.scratch2);
        let mem = riscv_mem_base(Reg(ptr_reg));
        self.writer.ld(self.ctx, self.arch, &riscv_reg(s2), &mem)?;
        self.writer.addi(self.ctx, self.arch, &riscv_reg(s2), &riscv_reg(s2), 1)?;
        self.writer.sd(self.ctx, self.arch, &riscv_reg(s2), &mem)
    }

    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error> {
        self.writer.ld(self.ctx, self.arch, &riscv_reg(Reg(dest)), &riscv_mem_base(Reg(src)))
    }
}
