//! [`BlitzWriter`] implementation for x86-64.

use crate::X64Label;
use portal_solutions_asm_x86_64::{
    ConditionCode, RegisterClass, X64Arch,
    out::{
        Writer, WriterCore,
        arg::{ArgKind, MemArgKind},
    },
};
use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::asm::common::mem::MemorySize;

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
        BlitzW {
            writer,
            ctx,
            arch,
            probe_base: ProbeBase::CtxSlot,
        }
    }
}

impl<'a, W, Context> portal_solutions_blitz_codegen::BlitzWriter for BlitzW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .jmp_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    fn branch_zero_label(&mut self, reg: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.cmp0(self.ctx, self.arch, &Reg(reg))?;
        self.writer.jcc_label(
            self.ctx,
            self.arch,
            ConditionCode::E,
            X64Label::Indexed { idx: label_idx },
        )
    }

    fn branch_reg(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.jmp(self.ctx, self.arch, &Reg(reg))
    }

    fn call_reg(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.call(self.ctx, self.arch, &Reg(reg))
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .set_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    // x86-64: add reg, 0xFFFF... (-1 as unsigned) = decrement
    fn reg_decrement(&mut self, reg: u8) -> Result<(), Self::Error> {
        self.writer.add(
            self.ctx,
            self.arch,
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
            self.ctx,
            self.arch,
            &MemArgKind::Mem {
                base: ArgKind::Reg {
                    reg: Reg(ptr_reg),
                    size: MemorySize::_64,
                },
                offset: None,
                disp: 0,
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
                self.ctx,
                self.arch,
                &Reg(dest),
                &MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg::CTX,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: base_off as u32,
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
                self.ctx,
                self.arch,
                &Reg(dest),
                &MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(5),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: disp as u32,
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
            self.ctx,
            self.arch,
            &MemArgKind::Mem {
                base: ArgKind::Reg {
                    reg: Reg(ptr_reg),
                    size: MemorySize::_64,
                },
                offset: None,
                disp: disp as u32,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
            &MemArgKind::NoMem(ArgKind::Lit(1)),
        )
    }

    fn load_mem64_disp(&mut self, dest: u8, src: u8, disp: i32) -> Result<(), Self::Error> {
        self.writer.mov(
            self.ctx,
            self.arch,
            &Reg(dest),
            &MemArgKind::Mem {
                base: ArgKind::Reg {
                    reg: Reg(src),
                    size: MemorySize::_64,
                },
                offset: None,
                disp: disp as u32,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
        )
    }
}

/// Wrapper binding an x86-64 writer + ctx + arch + regalloc state for
/// [`portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter`] and
/// [`portal_solutions_blitz_codegen::control_flow::ControlFlowWriter`].
///
/// `init_regalloc` reserves rsp/rbp (as `portal_solutions_asm_x86_64::regalloc`'s
/// own default does) *plus* RAX/r11/r14/r15 — RAX because `sysv.rs` uses it
/// pervasively as an assumed-free scratch register (locals, probes, epilogue,
/// `BrTable`'s selector, ...) in code that predates this regalloc integration
/// and doesn't all flush first (in particular `sysv_emit_control_flow_probe`/
/// `sysv_emit_indexed_probes`, which fire *without* a flush at exactly the
/// points — block/loop entry, arbitrary instruction boundaries — where a
/// WASM operand is most likely to be sitting in a register); r11/r14/r15 for
/// the SysV-specific probe-base virtual param, Static Context Register, and
/// call-argument marshalling (see `sysv.rs`'s `PROBE_BASE_REG`/`SCR`/
/// `sysv_emit_marshalled_call`). Reserving all four means the allocator never
/// hands one to a WASM operand, so SysV's own fixed-register code stays
/// correct without having to audit every call site for a missing `flush()`.
pub struct RegAllocW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: X64Arch,
    pub regalloc: &'a mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_x86_64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_x86_64::regalloc::RegKind,
                32,
            >,
        >,
    >,
}

/// Scratch register [`RegAllocW`]'s `ControlFlowWriter::pop_cond` reads the
/// WASM condition operand into. RAX (0) — free at any control-flow boundary,
/// since operands there are either regalloc-held (popped fresh here) or,
/// after `flush`, on the real RSP-based stack.
const COND_SCRATCH: u8 = 0;

impl<'a, W, Context>
    portal_solutions_blitz_codegen::regalloc_frontend::RegAllocWriter<
        portal_solutions_asm_x86_64::regalloc::RegKind,
        32,
    > for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn regalloc_mut(
        &mut self,
    ) -> &mut Option<
        portal_solutions_asm_regalloc::RegAlloc<
            portal_solutions_asm_x86_64::regalloc::RegKind,
            32,
            portal_solutions_blitz_codegen::regalloc_adapter::Frames<
                portal_solutions_asm_x86_64::regalloc::RegKind,
                32,
            >,
        >,
    > {
        self.regalloc
    }

    fn init_regalloc(
        &self,
    ) -> portal_solutions_asm_regalloc::RegAlloc<
        portal_solutions_asm_x86_64::regalloc::RegKind,
        32,
        portal_solutions_blitz_codegen::regalloc_adapter::Frames<
            portal_solutions_asm_x86_64::regalloc::RegKind,
            32,
        >,
    > {
        let r = portal_solutions_asm_x86_64::regalloc::init_regalloc::<32>(self.arch);
        let mut frames = portal_solutions_blitz_codegen::regalloc_adapter::Frames(r.frames);
        for reg in [0u8, 11, 14, 15] {
            frames.0[0][reg as usize] = portal_solutions_asm_regalloc::RegAllocFrame::Reserved;
        }
        // `RegAlloc`'s `N=32` slots per kind exist to accommodate a future
        // APX target (16 extra GPRs, R16-R31), which this backend's encoder
        // (`reg_to_iced`/`reg_to_iced32`/`xmm_to_iced`) does not implement —
        // they only ever emit RAX..R15/XMM0..XMM15. Without reserving 16..32,
        // the allocator hands out these non-existent indices under register
        // pressure, which the encoder then silently maps past R15 into
        // whatever discriminant happens to follow it in `iced_x86::Register`
        // (e.g. EIP), producing an "encoding error (backend bug)" panic
        // instead of a clean spill. Reserve both kinds' upper half until real
        // APX support (or AVX-512 for the float kind) lands end-to-end.
        for reg in 16u8..32 {
            frames.0[0][reg as usize] = portal_solutions_asm_regalloc::RegAllocFrame::Reserved;
            frames.0[1][reg as usize] = portal_solutions_asm_regalloc::RegAllocFrame::Reserved;
        }
        portal_solutions_asm_regalloc::RegAlloc { frames, tos: r.tos }
    }

    fn emit_regalloc_cmds(
        &mut self,
        cmds: alloc::vec::Vec<
            portal_solutions_asm_regalloc::Cmd<portal_solutions_asm_x86_64::regalloc::RegKind>,
        >,
    ) -> Result<(), Self::Error> {
        crate::regalloc_core::emit_cmds(self.writer, self.ctx, self.arch, cmds.into_iter())
    }
}

impl<'a, W, Context> portal_solutions_blitz_codegen::control_flow::ControlFlowWriter
    for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn branch_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .jmp_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    fn branch_zero_label(&mut self, reg_n: u8, label_idx: usize) -> Result<(), Self::Error> {
        self.writer.cmp0(self.ctx, self.arch, &Reg(reg_n))?;
        self.writer.jcc_label(
            self.ctx,
            self.arch,
            ConditionCode::E,
            X64Label::Indexed { idx: label_idx },
        )
    }

    fn place_label(&mut self, label_idx: usize) -> Result<(), Self::Error> {
        self.writer
            .set_label(self.ctx, self.arch, X64Label::Indexed { idx: label_idx })
    }

    // Flush any register-held operand-stack values to memory and reset TOS,
    // so every path into a following label sees consistent state.
    fn flush(&mut self) -> Result<(), Self::Error> {
        if let Some(ralloc) = self.regalloc.as_mut() {
            let it = ralloc.flush();
            let cmds: alloc::vec::Vec<_> = it.collect();
            crate::regalloc_core::emit_cmds(self.writer, self.ctx, self.arch, cmds.into_iter())?;
            ralloc.tos = None;
        }
        Ok(())
    }

    // Read the WASM condition operand directly off the real stack (valid
    // immediately after `flush`, which guarantees it's there) via `pop`.
    fn pop_cond(&mut self) -> Result<u8, Self::Error> {
        self.writer.pop(self.ctx, self.arch, &Reg(COND_SCRATCH))?;
        Ok(COND_SCRATCH)
    }
}
