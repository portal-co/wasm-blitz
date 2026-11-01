//! [`portal_solutions_blitz_codegen::BlitzWriter`] implementation for RISC-V 64.

use crate::RiscvLabel;
use portal_solutions_asm_riscv64::{
    ConditionCode, RegisterClass, RiscV64Arch,
    out::{
        Writer, WriterCore,
        arg::{ArgKind, MemArgKind},
    },
};
use portal_solutions_blitz_common::asm::{Reg, common::mem::MemorySize};

fn riscv_reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg {
        reg: r,
        size: MemorySize::_64,
    })
}

fn riscv_mem_base(base: Reg) -> MemArgKind {
    riscv_mem_base_disp(base, 0)
}

fn riscv_mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
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
    pub arch: RiscV64Arch,
    pub scratch2: u8,
    /// How `load_probe_base` reaches the runtime probe-table base.
    pub probe_base: ProbeBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative probe-base convention.
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: RiscV64Arch, scratch2: u8) -> Self {
        BlitzW {
            writer,
            ctx,
            arch,
            scratch2,
            probe_base: ProbeBase::CtxSlot,
        }
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
            self.ctx,
            self.arch,
            &riscv_reg(Reg(0)),
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    // RISC-V: beq reg, x0, label — natural two-register form
    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.bcond_label(
            self.ctx,
            self.arch,
            ConditionCode::EQ,
            &riscv_reg(Reg(reg_n)),
            &riscv_reg(Reg(0)), // x0 = zero register
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    fn branch_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        // JALR x0, reg, 0 — indirect jump, discard return addr into x0
        self.writer.jalr(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(0)),
            &riscv_reg(Reg(reg_n)),
            0,
        )
    }

    // JALR ra, reg, 0 — indirect call, return addr into ra (Reg(1)); the
    // callee's `ret` (`jalr x0, ra, 0`) returns here.
    fn call_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.jalr(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(1)),
            &riscv_reg(Reg(reg_n)),
            0,
        )
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .set_label(self.ctx, self.arch, RiscvLabel::Indexed { idx: label_idx })
    }

    // RISC-V: addi reg, reg, -1
    fn reg_decrement(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.addi(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(reg_n)),
            &riscv_reg(Reg(reg_n)),
            -1,
        )
    }

    fn load_u64_imm(&mut self, dest: u8, imm: u64) -> Result<(), Self::Error> {
        self.writer
            .li(self.ctx, self.arch, &riscv_reg(Reg(dest)), imm)
    }

    // RISC-V: ld s2, 0(ptr); addi s2, s2, 1; sd s2, 0(ptr)
    fn inc_mem64(&mut self, ptr_reg: u8) -> Result<(), Self::Error> {
        let s2 = Reg(self.scratch2);
        let mem = riscv_mem_base(Reg(ptr_reg));
        self.writer.ld(self.ctx, self.arch, &riscv_reg(s2), &mem)?;
        self.writer
            .addi(self.ctx, self.arch, &riscv_reg(s2), &riscv_reg(s2), 1)?;
        self.writer.sd(self.ctx, self.arch, &riscv_reg(s2), &mem)
    }

    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error> {
        self.load_mem64_disp(dest, src, 0)
    }

    // Load the runtime probe-table base into `dest` from the configured source.
    fn load_probe_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.probe_base {
            ProbeBase::CtxSlot => self.writer.ld(
                self.ctx,
                self.arch,
                &riscv_reg(Reg(dest)),
                &riscv_mem_base_disp(Reg::CTX, base_off),
            ),
            // mv dest, r  (addi dest, r, 0)
            ProbeBase::Reg(r) => {
                if r != dest {
                    self.writer.addi(
                        self.ctx,
                        self.arch,
                        &riscv_reg(Reg(dest)),
                        &riscv_reg(Reg(r)),
                        0,
                    )?;
                }
                Ok(())
            }
            // fp = Reg(8) (s0); mid-function frame slot.
            ProbeBase::FrameSlot(disp) => self.writer.ld(
                self.ctx,
                self.arch,
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
        self.writer
            .addi(self.ctx, self.arch, &riscv_reg(s2), &riscv_reg(s2), 1)?;
        self.writer.sd(self.ctx, self.arch, &riscv_reg(s2), &mem)
    }

    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.ld(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(dest)),
            &riscv_mem_base_disp(Reg(src), disp),
        )
    }
}

/// Wrapper binding a RISC-V writer + ctx + arch + regalloc state for
/// [`portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter`].
pub struct RegAllocW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: RiscV64Arch,
    pub regalloc: &'a mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_riscv64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_riscv64::regalloc::RegKind,
                32,
            >,
        >,
    >,
}

impl<'a, W, Context>
    portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter<
        portal_solutions_asm_riscv64::regalloc::RegKind,
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
            portal_solutions_asm_riscv64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_riscv64::regalloc::RegKind,
                32,
            >,
        >,
    > {
        self.regalloc
    }

    fn init_regalloc(
        &self,
    ) -> portal_solutions_asm_regalloc::RegAlloc<
        portal_solutions_asm_riscv64::regalloc::RegKind,
        32,
        portal_solutions_blitz_codegen::regalloc_adapter::Frames<
            portal_solutions_asm_riscv64::regalloc::RegKind,
            32,
        >,
    > {
        let r = portal_solutions_asm_riscv64::regalloc::init_regalloc::<32>(self.arch);
        portal_solutions_asm_regalloc::RegAlloc {
            frames: portal_solutions_blitz_codegen::regalloc_adapter::Frames(r.frames),
            tos: r.tos,
        }
    }

    fn emit_regalloc_cmds(
        &mut self,
        cmds: alloc::vec::Vec<
            portal_solutions_asm_regalloc::Cmd<portal_solutions_asm_riscv64::regalloc::RegKind>,
        >,
    ) -> Result<(), Self::Error> {
        crate::naive::emit_cmds(self.writer, self.ctx, self.arch, cmds.into_iter())
    }
}

/// Scratch register [`ControlFlowW::pop_cond`] reads the WASM condition
/// operand into — matches the `tmp = Reg(10)` convention the hand-written
/// `If`/`BrIf` arms used before this was extracted.
const COND_SCRATCH: Reg = Reg(10);

impl<'a, W, Context> portal_solutions_blitz_codegen::control_flow::ControlFlowWriter
    for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<RiscvLabel, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.jal_label(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(0)),
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.bcond_label(
            self.ctx,
            self.arch,
            ConditionCode::EQ,
            &riscv_reg(Reg(reg_n)),
            &riscv_reg(Reg(0)),
            RiscvLabel::Indexed { idx: label_idx },
        )
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .set_label(self.ctx, self.arch, RiscvLabel::Indexed { idx: label_idx })
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
        self.writer.ld(
            self.ctx,
            self.arch,
            &riscv_reg(COND_SCRATCH),
            &riscv_mem_base(Reg(2)),
        )?;
        self.writer.addi(
            self.ctx,
            self.arch,
            &riscv_reg(Reg(2)),
            &riscv_reg(Reg(2)),
            8,
        )?;
        Ok(COND_SCRATCH.0)
    }
}

// ---------------------------------------------------------------------------
// Code sharing (v1: SysV + rewindable binary writer) — parity with
// blitz-x86-64's and blitz-aarch64's codegen.rs;
// see docs/generic-code-sharing-plan.md.
// ---------------------------------------------------------------------------

/// Conservative RV64 SysV sharing contract for a canonical call boundary.
///
/// Value-only key: eligibility (empty WASM stack, fixed frame, no dynamic
/// context) must still be proven by the emitter before staging a candidate.
pub const RV64_SYSV_SHARE_CONTRACT: portal_solutions_blitz_codegen::code_sharing::ShareContract =
    portal_solutions_blitz_codegen::code_sharing::ShareContract::new(
        0x5A64,
        0x5359_5356,
        0x4652_5631,
        0x454d_5054,
        0x4459_4e30,
    );

/// Base of the RV64 helper-label namespace. SysV function entries use
/// `idx | (1 << 28)` and control-flow labels come from `label_index`
/// counting upward from 0; helper labels live at `1 << 29` and count upward,
/// so they cannot collide with either.
pub const RV64_HELPER_LABEL_BASE: usize = 1 << 29;

/// Measured RV64 layout costs: `jal ra` and `ret` (`jalr x0, ra, 0`) are each
/// one 4-byte instruction; label placement is free.
pub const RV64_SYSV_OUTLINE_COST: portal_solutions_blitz_codegen::code_sharing::OutlineCost =
    portal_solutions_blitz_codegen::code_sharing::OutlineCost {
        helper_entry_bytes: 0,
        helper_return_bytes: 4,
        call_bytes: 4,
        layout_bytes: 0,
    };

/// Adapter used at function-finalization time: forwards the rewindable-writer
/// transaction methods to the underlying writer and provides the
/// [`portal_solutions_blitz_codegen::code_sharing::SharingLayout`] emission
/// hooks with RV64 encodings (`jal ra` / `ret` / `set_label`).
pub struct SharingW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: RiscV64Arch,
    /// Next helper-label ordinal; combined with [`RV64_HELPER_LABEL_BASE`].
    pub next_helper: usize,
}

impl<'a, W, Context> portal_solutions_asm_writer::RewindableCodeWriter<crate::RiscvLabel>
    for SharingW<'a, W, Context>
where
    W: portal_solutions_asm_writer::RewindableCodeWriter<crate::RiscvLabel>,
{
    type Blob = W::Blob;
    type Error = W::Error;

    fn drain_after(&mut self, anchor: &crate::RiscvLabel) -> Result<Self::Blob, Self::Error> {
        self.writer.drain_after(anchor)
    }
    fn rewind_after(&mut self, anchor: &crate::RiscvLabel) -> Result<(), Self::Error> {
        self.writer.rewind_after(anchor)
    }
    fn append_blob(&mut self, blob: Self::Blob) -> Result<(), Self::Error> {
        self.writer.append_blob(blob)
    }
    fn blob_hash(&self, blob: &Self::Blob) -> portal_solutions_asm_writer::CodeHash {
        self.writer.blob_hash(blob)
    }
    fn blob_len(&self, blob: &Self::Blob) -> usize {
        self.writer.blob_len(blob)
    }
    fn blob_eq(&self, left: &Self::Blob, right: &Self::Blob) -> bool {
        self.writer.blob_eq(left, right)
    }
}

impl<'a, W, Context> portal_solutions_blitz_codegen::code_sharing::SharingLayout<crate::RiscvLabel>
    for SharingW<'a, W, Context>
where
    W: portal_solutions_asm_riscv64::out::WriterCore<Context>
        + portal_solutions_asm_riscv64::out::Writer<crate::RiscvLabel, Context>,
{
    type Error = W::Error;

    fn alloc_helper_label(&mut self) -> crate::RiscvLabel {
        let idx = RV64_HELPER_LABEL_BASE + self.next_helper;
        self.next_helper += 1;
        crate::RiscvLabel::Indexed { idx }
    }
    fn place_helper_label(&mut self, label: &crate::RiscvLabel) -> Result<(), Self::Error> {
        use portal_solutions_asm_riscv64::out::Writer as _;
        self.writer.set_label(self.ctx, self.arch, label.clone())
    }
    fn call_helper(&mut self, label: &crate::RiscvLabel) -> Result<(), Self::Error> {
        use portal_solutions_asm_riscv64::out::Writer as _;
        // `jal ra, helper`: link register RA (x1), as in SysV's call lowering.
        self.writer
            .jal_label(self.ctx, self.arch, &Reg(1), label.clone())
    }
    fn helper_return(&mut self) -> Result<(), Self::Error> {
        use portal_solutions_asm_riscv64::out::WriterCore as _;
        self.writer.ret(self.ctx, self.arch)
    }
}

/// Conservative RV64 naive sharing contract (distinct ABI key from SysV).
pub const RV64_NAIVE_SHARE_CONTRACT: portal_solutions_blitz_codegen::code_sharing::ShareContract =
    portal_solutions_blitz_codegen::code_sharing::ShareContract::new(
        0x5A64,
        0x4e41_4956, // "NAIV"
        0x4652_5631,
        0x454d_5054,
        0x4459_4e30,
    );

/// Naive helper layout costs equal the SysV ones on RV64 (`jal ra` + `ret`).
pub const RV64_NAIVE_OUTLINE_COST: portal_solutions_blitz_codegen::code_sharing::OutlineCost =
    RV64_SYSV_OUTLINE_COST;
