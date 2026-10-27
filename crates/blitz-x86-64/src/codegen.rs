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

/// Conservative x86-64 SysV sharing contract for a canonical call boundary.
///
/// This is intentionally a value-only key. It does not claim that a candidate
/// is eligible; the SysV emitter must prove the empty-stack/flush/frame rules
/// before recording one with this contract.
pub const X86_64_SYSV_SHARE_CONTRACT: portal_solutions_blitz_codegen::code_sharing::ShareContract =
    portal_solutions_blitz_codegen::code_sharing::ShareContract::new(
        0x8664,
        0x5359_5356,
        0x4652_5631,
        0x454d_5054,
        0x4459_4e30,
    );

/// Backend-local seam for a future x86-64 SysV outlining pass.
///
/// The seam deliberately exposes only the operations needed around a generated
/// helper call. It does not make a candidate eligible: callers still need to
/// prove the empty-WASM-stack, fixed-frame, and no-dynamic-context contract
/// before draining a writer blob.
pub trait SharingSafepoint {
    type Error;

    /// Commit any register-held operand values to the canonical native stack.
    fn flush_for_sharing(&mut self) -> Result<(), Self::Error>;
    /// Forget cached register assignments after an inserted helper call.
    fn invalidate_after_sharing(&mut self);
    /// Emit a direct call to a generated helper label.
    fn emit_helper_call(&mut self, helper_label: usize) -> Result<(), Self::Error>;
    /// Emit the helper's ordinary return instruction.
    fn emit_helper_return(&mut self) -> Result<(), Self::Error>;
    /// Return the contract used for the canonical boundary represented by this
    /// adapter.
    fn sharing_contract(&self) -> portal_solutions_blitz_codegen::code_sharing::ShareContract;
}

/// Base of the x86-64 helper-label namespace. SysV function entries use
/// `idx | (1 << 28)` and control-flow labels come from `label_index` counting
/// upward from 0; helper labels live at `1 << 29` and count upward, so they
/// cannot collide with either. See the plan's separate-namespace rule.
pub const X86_HELPER_LABEL_BASE: usize = 1 << 29;

/// Conservative x86-64 naive (CTX-stack) sharing contract. Same value-key
/// discipline as the SysV contract; the ABI key differs so SysV and naive
/// blobs never share a helper.
pub const X86_64_NAIVE_SHARE_CONTRACT: portal_solutions_blitz_codegen::code_sharing::ShareContract =
    portal_solutions_blitz_codegen::code_sharing::ShareContract::new(
        0x8664,
        0x4e41_4956, // "NAIV"
        0x4652_5631,
        0x454d_5054,
        0x4459_4e30,
    );

/// Naive helper layout costs are the same encodings as SysV (`call rel32`,
/// `ret`); the helper area convention is identical.
pub const X86_64_NAIVE_OUTLINE_COST: portal_solutions_blitz_codegen::code_sharing::OutlineCost =
    X86_64_SYSV_OUTLINE_COST;

/// Measured x86-64 SysV layout costs for the sharing collector: a `call rel32`
/// is 5 bytes, the helper `ret` is 1 byte, and label placement is free.
pub const X86_64_SYSV_OUTLINE_COST: portal_solutions_blitz_codegen::code_sharing::OutlineCost =
    portal_solutions_blitz_codegen::code_sharing::OutlineCost {
        helper_entry_bytes: 0,
        helper_return_bytes: 1,
        call_bytes: 5,
        layout_bytes: 0,
    };

/// Adapter used at function-finalization time: forwards the rewindable-writer
/// transaction methods to the underlying writer and provides the
/// [`portal_solutions_blitz_codegen::code_sharing::SharingLayout`] emission
/// hooks with x86-64 encodings (`call rel32` / `ret` / `set_label`).
pub struct SharingW<'a, W, Context> {
    pub writer: &'a mut W,
    pub ctx: &'a mut Context,
    pub arch: X64Arch,
    /// Next helper-label ordinal; combined with [`X86_HELPER_LABEL_BASE`].
    pub next_helper: usize,
}

impl<'a, W, Context> portal_solutions_asm_writer::RewindableCodeWriter<X64Label>
    for SharingW<'a, W, Context>
where
    W: portal_solutions_asm_writer::RewindableCodeWriter<X64Label>,
{
    type Blob = W::Blob;
    type Error = W::Error;

    fn drain_after(&mut self, anchor: &X64Label) -> Result<Self::Blob, Self::Error> {
        self.writer.drain_after(anchor)
    }
    fn rewind_after(&mut self, anchor: &X64Label) -> Result<(), Self::Error> {
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

impl<'a, W, Context> portal_solutions_blitz_codegen::code_sharing::SharingLayout<X64Label>
    for SharingW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn alloc_helper_label(&mut self) -> X64Label {
        let idx = X86_HELPER_LABEL_BASE + self.next_helper;
        self.next_helper += 1;
        X64Label::Indexed { idx }
    }
    fn place_helper_label(&mut self, label: &X64Label) -> Result<(), Self::Error> {
        self.writer.set_label(self.ctx, self.arch, label.clone())
    }
    fn call_helper(&mut self, label: &X64Label) -> Result<(), Self::Error> {
        self.writer.call_label(self.ctx, self.arch, label.clone())
    }
    fn helper_return(&mut self) -> Result<(), Self::Error> {
        self.writer.ret(self.ctx, self.arch)
    }
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

impl<'a, W, Context> portal_solutions_blitz_codegen::code_sharing::SharingSafepoint
    for RegAllocW<'a, W, Context>
where
    W: WriterCore<Context> + Writer<X64Label, Context>,
{
    type Error = W::Error;

    fn flush_for_sharing(&mut self) -> Result<(), Self::Error> {
        portal_solutions_blitz_codegen::control_flow::ControlFlowWriter::flush(self)
    }

    fn invalidate_after_sharing(&mut self) {
        if let Some(regalloc) = self.regalloc.as_mut() {
            // `flush_for_sharing` has materialized every live value and reset
            // the allocator's frame entries. Clearing TOS makes the post-call
            // boundary explicit to future allocator users without inventing
            // a second register-state representation.
            regalloc.tos = None;
        }
    }

    fn emit_helper_call(&mut self, helper_label: usize) -> Result<(), Self::Error> {
        self.writer
            .call_label(self.ctx, self.arch, X64Label::Indexed { idx: helper_label })
    }

    fn emit_helper_return(&mut self) -> Result<(), Self::Error> {
        self.writer.ret(self.ctx, self.arch)
    }

    fn sharing_contract(&self) -> portal_solutions_blitz_codegen::code_sharing::ShareContract {
        X86_64_SYSV_SHARE_CONTRACT
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
