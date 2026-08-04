//! [`portal_solutions_blitz_codegen::BlitzWriter`] implementation for RISC-V 32.

use portal_solutions_asm_riscv32::{
    ConditionCode, RegisterClass, RiscV32Arch,
    out::{Writer, WriterCore, arg::{ArgKind, MemArgKind}},
};
use portal_solutions_blitz_common::asm::{Reg, common::mem::MemorySize};
use crate::RiscvLabel;

fn riscv_reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_32 })
}

fn riscv_mem_base(base: Reg) -> MemArgKind {
    riscv_mem_base_disp(base, 0)
}

/// 8-byte (i64 / soft-expanded) memory operand; base register is ILP32-width.
fn riscv_mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_32 },
        offset: None, disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    }
}

/// 4-byte host pointer memory operand (ILP32).
fn riscv_ptr_mem_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_32 },
        offset: None, disp,
        size: MemorySize::_32,
        reg_class: RegisterClass::Gpr,
    }
}

/// Where the runtime probe-table base pointer is found, for `load_probe_base`.
///
/// NaiveAbi uses the CTX frame pointer; the RISC-V SysV ABI passes the base as a
/// virtual function parameter (a reserved register) and spills it to an
/// fp-relative frame slot for mid-function sites.  See `docs/abi.md`.
#[derive(Clone, Copy, Default)]
pub enum ProbeBase {
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
    pub arch: RiscV32Arch,
    pub scratch2: u8,
    /// How `load_probe_base` reaches the runtime probe-table base.
    pub probe_base: ProbeBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative probe-base convention.
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: RiscV32Arch, scratch2: u8) -> Self {
        BlitzW { writer, ctx, arch, scratch2, probe_base: ProbeBase::CtxSlot }
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

    // JALR ra, reg, 0 — indirect call, return addr into ra (Reg(1)); the
    // callee's `ret` (`jalr x0, ra, 0`) returns here.
    fn call_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.jalr(self.ctx, self.arch, &riscv_reg(Reg(1)), &riscv_reg(Reg(reg_n)), 0)
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

    // Load the runtime probe-table base into `dest` from the configured source.
    // Host pointer: ILP32 `lw` / 4-byte access.
    fn load_probe_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.probe_base {
            ProbeBase::CtxSlot => self.writer.lw(
                self.ctx, self.arch,
                &riscv_reg(Reg(dest)),
                &riscv_ptr_mem_disp(Reg::CTX, base_off),
            ),
            // mv dest, r  (addi dest, r, 0)
            ProbeBase::Reg(r) => {
                if r != dest {
                    self.writer.addi(self.ctx, self.arch, &riscv_reg(Reg(dest)), &riscv_reg(Reg(r)), 0)?;
                }
                Ok(())
            }
            // fp = Reg(8) (s0); mid-function frame slot.
            ProbeBase::FrameSlot(disp) => self.writer.lw(
                self.ctx, self.arch,
                &riscv_reg(Reg(dest)),
                &riscv_ptr_mem_disp(Reg(8), disp),
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

/// Wrapper binding a RISC-V writer + ctx + arch + regalloc state for
/// [`portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter`].
pub struct RegAllocW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: RiscV32Arch,
    pub regalloc: &'a mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_riscv32::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_riscv32::regalloc::RegKind,
                32,
            >,
        >,
    >,
}

impl<'a, W, Context> portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter<
    portal_solutions_asm_riscv32::regalloc::RegKind,
    32,
> for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<RiscvLabel, Context>,
{
    type Error = W::Error;

    fn regalloc_mut(
        &mut self,
    ) -> &mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_riscv32::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_riscv32::regalloc::RegKind,
                32,
            >,
        >,
    > {
        self.regalloc
    }

    fn init_regalloc(
        &self,
    ) -> portal_solutions_asm_regalloc::RegAlloc<
        portal_solutions_asm_riscv32::regalloc::RegKind,
        32,
        portal_solutions_blitz_codegen::regalloc_adapter::Frames<
            portal_solutions_asm_riscv32::regalloc::RegKind,
            32,
        >,
    > {
        let r = portal_solutions_asm_riscv32::regalloc::init_regalloc::<32>(self.arch);
        portal_solutions_asm_regalloc::RegAlloc {
            frames: portal_solutions_blitz_codegen::regalloc_adapter::Frames(r.frames),
            tos: r.tos,
        }
    }

    fn emit_regalloc_cmds(
        &mut self,
        cmds: alloc::vec::Vec<portal_solutions_asm_regalloc::Cmd<portal_solutions_asm_riscv32::regalloc::RegKind>>,
    ) -> Result<(), Self::Error> {
        crate::naive::emit_cmds(self.writer, self.ctx, self.arch, cmds.into_iter())
    }
}

/// Scratch register [`ControlFlowW::pop_cond`] reads the WASM condition
/// operand into — matches the `tmp = Reg(10)` convention the hand-written
/// `If`/`BrIf` arms used before this was extracted.
const COND_SCRATCH: Reg = Reg(10);

impl<'a, W, Context> portal_solutions_blitz_codegen::control_flow::ControlFlowWriter for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<RiscvLabel, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.jal_label(self.ctx, self.arch, &riscv_reg(Reg(0)), RiscvLabel::Indexed { idx: label_idx })
    }

    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.bcond_label(
            self.ctx, self.arch,
            ConditionCode::EQ,
            &riscv_reg(Reg(reg_n)),
            &riscv_reg(Reg(0)),
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, RiscvLabel::Indexed { idx: label_idx })
    }

    // Flush any register-held operand-stack values to memory and reset TOS,
    // so every path into a following label sees consistent state.
    fn flush(&mut self) -> Result<(), Self::Error> {
        if let Some(ralloc) = self.regalloc.as_mut() {
            let it = ralloc.flush();
            let cmds: alloc::vec::Vec<_> = it.collect();
            crate::naive::emit_cmds(self.writer, self.ctx, self.arch, cmds.into_iter())?;
            ralloc.tos = None;
        }
        Ok(())
    }

    // Read the WASM condition operand directly off the real stack (valid
    // immediately after `flush`, which guarantees it's there rather than
    // live in a register) via `ld tmp, [sp]; addi sp, sp, 8`.
    fn pop_cond(&mut self) -> Result<u8, Self::Error> {
        self.writer.ld(self.ctx, self.arch, &riscv_reg(COND_SCRATCH), &riscv_mem_base(Reg(2)))?;
        self.writer.addi(self.ctx, self.arch, &riscv_reg(Reg(2)), &riscv_reg(Reg(2)), 8)?;
        Ok(COND_SCRATCH.0)
    }
}
