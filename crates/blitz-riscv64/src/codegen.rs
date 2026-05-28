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
    riscv_mem_base_disp(base, 0)
}

fn riscv_mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None, disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// Where the runtime trace-table base pointer is found, for `load_trace_base`.
///
/// NaiveAbi uses the CTX frame pointer; the RISC-V SysV ABI passes the base as a
/// virtual function parameter (a reserved register) and spills it to an
/// fp-relative frame slot for mid-function sites.  See `docs/abi.md`.
#[derive(Clone, Copy, Default)]
pub enum TraceBase {
    /// Base pointer stored at `[CTX + base_off]` (NaiveAbi / LFI).
    #[default]
    CtxSlot,
    /// Base pointer held directly in this blitz register (SysV virtual param).
    Reg(u8),
    /// Base pointer stored at `[fp + disp]` (SysV mid-function frame slot).
    FrameSlot(i32),
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
    /// How `load_trace_base` reaches the runtime trace-table base.
    pub trace_base: TraceBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative trace-base convention.
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: RiscV64Arch, scratch2: u8) -> Self {
        BlitzW { writer, ctx, arch, scratch2, trace_base: TraceBase::CtxSlot }
    }
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
        self.load_mem64_disp(dest, src, 0)
    }

    // Load the runtime trace-table base into `dest` from the configured source.
    fn load_trace_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.trace_base {
            TraceBase::CtxSlot => self.writer.ld(
                self.ctx, self.arch,
                &riscv_reg(Reg(dest)),
                &riscv_mem_base_disp(Reg::CTX, base_off),
            ),
            // mv dest, r  (addi dest, r, 0)
            TraceBase::Reg(r) => {
                if r != dest {
                    self.writer.addi(self.ctx, self.arch, &riscv_reg(Reg(dest)), &riscv_reg(Reg(r)), 0)?;
                }
                Ok(())
            }
            // fp = Reg(8) (s0); mid-function frame slot.
            TraceBase::FrameSlot(disp) => self.writer.ld(
                self.ctx, self.arch,
                &riscv_reg(Reg(dest)),
                &riscv_mem_base_disp(Reg(8), disp),
            ),
        }
    }

    // RISC-V: ld s2, disp(ptr); addi s2, s2, 1; sd s2, disp(ptr)
    fn inc_mem64_disp(&mut self, ptr_reg: u8, disp: i32) -> Result<(), Self::Error> {
        let s2 = Reg(self.scratch2);
        let mem = riscv_mem_base_disp(Reg(ptr_reg), disp);
        self.writer.ld(self.ctx, self.arch, &riscv_reg(s2), &mem)?;
        self.writer.addi(self.ctx, self.arch, &riscv_reg(s2), &riscv_reg(s2), 1)?;
        self.writer.sd(self.ctx, self.arch, &riscv_reg(s2), &mem)
    }

    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.ld(self.ctx, self.arch, &riscv_reg(Reg(dest)), &riscv_mem_base_disp(Reg(src), disp))
    }
}
