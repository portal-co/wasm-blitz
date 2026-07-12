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

/// Where the runtime probe-table base pointer is found, for `load_probe_base`.
///
/// NaiveAbi uses the CTX frame pointer; the AAPCS64 SysV ABI passes the base as
/// a virtual function parameter (a reserved register) and spills it to an
/// FP-relative frame slot for mid-function sites.  See `docs/abi.md`.
#[derive(Clone, Copy, Default)]
pub enum ProbeBase {
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
    /// How `load_probe_base` reaches the runtime probe-table base.
    pub probe_base: ProbeBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative probe-base convention.
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: AArch64Arch, scratch2: u8) -> Self {
        BlitzW { writer, ctx, arch, scratch2, probe_base: ProbeBase::CtxSlot }
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

    // AArch64: `bl` with a register operand is emitted as `blr` (branch with
    // link to register) by the writer — see asm-aarch64/src/out/asm.rs.
    fn call_reg(&mut self, reg_n: u8) -> Result<(), Self::Error> {
        self.writer.bl(self.ctx, self.arch, &aarch64_reg(Reg(reg_n)))
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

    // Load the runtime probe-table base into `dest` from the configured source.
    fn load_probe_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.probe_base {
            ProbeBase::CtxSlot => self.writer.ldr(
                self.ctx, self.arch,
                &aarch64_reg(Reg(dest)),
                &aarch64_mem_base_disp(Reg::CTX, base_off),
            ),
            ProbeBase::Reg(r) => {
                if r != dest {
                    self.writer.mov(self.ctx, self.arch, &aarch64_reg(Reg(dest)), &aarch64_reg(Reg(r)))?;
                }
                Ok(())
            }
            // FP = crate::FP (x29); mid-function frame slot.
            ProbeBase::FrameSlot(disp) => self.writer.ldr(
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

/// Wrapper binding an AArch64 writer + ctx + arch + regalloc state for
/// [`portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter`] and
/// [`portal_solutions_blitz_codegen::control_flow::ControlFlowWriter`].
///
/// `init_regalloc` reserves x29/x30/x31 (as `portal_solutions_asm_aarch64::regalloc`'s
/// own default does — FP/LR/SP) *plus* x9/x10/x11 (`naive.rs`'s `T0`/`T1`/`T2`
/// fixed scratch registers, used by every not-yet-regalloc-covered instruction
/// and notably by `emit_control_flow_probe`, which fires *without* a flush at
/// exactly the points — block/loop entry — where a WASM operand is most likely
/// sitting in a register) and x27 (`naive::SCR`, the Static Context Register).
/// Reserving all of these means the allocator never hands one to a WASM
/// operand, so the not-yet-migrated instructions' fixed-register code stays
/// correct without having to audit every call site for a missing `flush()`.
pub struct RegAllocW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: AArch64Arch,
    pub regalloc: &'a mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_aarch64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_aarch64::regalloc::RegKind,
                32,
            >,
        >,
    >,
}

/// Scratch register [`RegAllocW`]'s `ControlFlowWriter::pop_cond` reads the
/// WASM condition operand into. Reuses `naive.rs`'s `T0` (x9) — always
/// reserved from the allocator's pool (see the struct doc above), so it's
/// safe as a raw scratch regardless of what's currently regalloc-tracked.
const COND_SCRATCH: u8 = 9;

impl<'a, W, Context> portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter<
    portal_solutions_asm_aarch64::regalloc::RegKind,
    32,
> for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<AArch64Label, Context>,
{
    type Error = W::Error;

    fn regalloc_mut(
        &mut self,
    ) -> &mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_aarch64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_aarch64::regalloc::RegKind,
                32,
            >,
        >,
    > {
        self.regalloc
    }

    fn init_regalloc(
        &self,
    ) -> portal_solutions_asm_regalloc::RegAlloc<
        portal_solutions_asm_aarch64::regalloc::RegKind,
        32,
        portal_solutions_blitz_codegen::regalloc_adapter::Frames<
            portal_solutions_asm_aarch64::regalloc::RegKind,
            32,
        >,
    > {
        let r = portal_solutions_asm_aarch64::regalloc::init_regalloc::<32>(self.arch);
        let mut frames = portal_solutions_blitz_codegen::regalloc_adapter::Frames(r.frames);
        for reg in [9u8, 10, 11, 27] {
            frames.0[0][reg as usize] = portal_solutions_asm_regalloc::RegAllocFrame::Reserved;
        }
        portal_solutions_asm_regalloc::RegAlloc { frames, tos: r.tos }
    }

    fn emit_regalloc_cmds(
        &mut self,
        cmds: alloc::vec::Vec<portal_solutions_asm_regalloc::Cmd<portal_solutions_asm_aarch64::regalloc::RegKind>>,
    ) -> Result<(), Self::Error> {
        for cmd in cmds {
            portal_solutions_asm_aarch64::regalloc::process_cmd(self.writer, self.ctx, self.arch, &cmd)?;
        }
        Ok(())
    }
}

impl<'a, W, Context> portal_solutions_blitz_codegen::control_flow::ControlFlowWriter for RegAllocW<'a, W, Context>
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

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, AArch64Label::Indexed { idx: label_idx })
    }

    // Flush any register-held operand-stack values to memory and reset TOS,
    // so every path into a following label sees consistent state.
    fn flush(&mut self) -> Result<(), Self::Error> {
        if let Some(ralloc) = self.regalloc.as_mut() {
            let it = ralloc.flush();
            for cmd in it {
                portal_solutions_asm_aarch64::regalloc::process_cmd(self.writer, self.ctx, self.arch, &cmd)?;
            }
            ralloc.tos = None;
        }
        Ok(())
    }

    // Read the WASM condition operand directly off the real stack (valid
    // immediately after `flush`) via `naive.rs`'s wasm_pop convention
    // (`ldr r, [sp], #16`).
    fn pop_cond(&mut self) -> Result<u8, Self::Error> {
        self.writer.ldr(
            self.ctx, self.arch,
            &aarch64_reg(Reg(COND_SCRATCH)),
            &MemArgKind::Mem {
                base: ArgKind::Reg { reg: Reg(31), size: MemorySize::_64 },
                offset: None,
                mode: AddressingMode::PostIndex,
                disp: crate::naive::WASM_SLOT,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
            },
        )?;
        Ok(COND_SCRATCH)
    }
}
