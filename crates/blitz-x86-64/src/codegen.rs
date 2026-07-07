//! [`BlitzWriter`] implementation for x86-64.

use portal_solutions_asm_x86_64::{
    ConditionCode, RegisterClass, X64Arch,
    out::{Writer, WriterCore, arg::{ArgKind, MemArgKind}},
};
use portal_solutions_blitz_common::asm::common::mem::MemorySize;
use portal_solutions_blitz_common::asm::Reg;
use crate::X64Label;

/// Where the runtime probe-table base pointer is found, for `load_probe_base`.
///
/// The NaiveAbi/LFI backends use a frame-pointer (CTX) convention; the SysV ABI
/// has no CTX at the function-entry preamble, so it passes the base as a virtual
/// function parameter (a reserved register) and spills it to a frame slot for
/// mid-function sites.  See `docs/abi.md` (Probes).
#[derive(Clone, Copy)]
pub enum ProbeBase {
    /// Base pointer stored at `[CTX + base_off]` (NaiveAbi / LFI).
    CtxSlot,
    /// Base pointer held directly in this blitz register (SysV virtual param).
    Reg(u8),
    /// Base pointer stored at `[RBP + disp]` (SysV mid-function frame slot).
    FrameSlot(i32),
}

/// Wrapper binding an x86-64 writer + ctx + arch for [`portal_solutions_blitz_codegen::BlitzWriter`].
pub struct BlitzW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: X64Arch,
    /// How `load_probe_base` reaches the runtime probe-table base.
    pub probe_base: ProbeBase,
}

impl<'a, W, Context> BlitzW<'a, W, Context> {
    /// Construct a wrapper using the default CTX-relative probe-base convention
    /// (NaiveAbi / LFI).
    pub fn new(writer: &'a mut W, ctx: &'a mut Context, arch: X64Arch) -> Self {
        BlitzW { writer, ctx, arch, probe_base: ProbeBase::CtxSlot }
    }
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

    fn call_reg(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.call(self.ctx, self.arch, &Reg(reg))
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
                segment: Default::default(),
            },
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )
    }

    fn load_mem64(&mut self, dest: u8, src: u8) -> Result<(), Self::Error> {
        self.load_mem64_disp(dest, src, 0)
    }

    // Load the runtime probe-table base into `dest` from the configured source.
    fn load_probe_base(&mut self, dest: u8, base_off: i32) -> Result<(), Self::Error> {
        match self.probe_base {
            // NaiveAbi/LFI: base pointer stored at [CTX + base_off].
            ProbeBase::CtxSlot => self.writer.mov(
                self.ctx, self.arch,
                &Reg(dest),
                &MemArgKind::Mem {
                    base: ArgKind::Reg { reg: Reg::CTX, size: MemorySize::_64 },
                    offset: None, disp: base_off as u32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                },
            ),
            // SysV function-entry site: base is the virtual-param register.
            ProbeBase::Reg(r) => {
                if r != dest {
                    self.writer.mov(self.ctx, self.arch, &Reg(dest), &Reg(r))?;
                }
                Ok(())
            }
            // SysV mid-function site: base spilled to [RBP + disp]  (RBP = Reg(5)).
            ProbeBase::FrameSlot(disp) => self.writer.mov(
                self.ctx, self.arch,
                &Reg(dest),
                &MemArgKind::Mem {
                    base: ArgKind::Reg { reg: Reg(5), size: MemorySize::_64 },
                    offset: None, disp: disp as u32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Default::default(),
                },
            ),
        }
    }

    // x86-64: ADD [ptr_reg + disp], 1 — single instruction, no scratch needed
    fn inc_mem64_disp(&mut self, ptr_reg: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.add(
            self.ctx, self.arch,
            &MemArgKind::Mem {
                base: ArgKind::Reg { reg: Reg(ptr_reg), size: MemorySize::_64 },
                offset: None, disp: disp as u32,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )
    }

    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.mov(
            self.ctx, self.arch,
            &Reg(dest),
            &MemArgKind::Mem {
                base: ArgKind::Reg { reg: Reg(src), size: MemorySize::_64 },
                offset: None, disp: disp as u32,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
        )
    }
}
