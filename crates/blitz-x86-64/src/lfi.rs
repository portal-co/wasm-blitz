//! LFI-compliant x86-64 ABI implementation.
//!
//! Generates code that passes the `lfi-verify` verifier:
//! - All linear-memory accesses use `gs:` segment with 32-bit address registers.
//! - `ret` is replaced by the rtcall macroinstruction (`lea r11, [rip+N]; jmp [r14+0]`).
//! - Functions begin at 32-byte-aligned addresses.
//! - `r14` is never written (sandbox base register).

extern crate alloc;

use portal_solutions_asm_x86_64::{RegisterClass, X64Arch};
use portal_solutions_asm_x86_64::out::arg::{ArgKind, MemArgKind, Segment};
use portal_solutions_asm_x86_64::out::{Writer, WriterCore};
use portal_solutions_blitz_common::{
    abi::BackendAbi,
    asm::Reg,
    asm::common::mem::MemorySize,
    ops::{FnData, MachOperator},
    wasm_encoder::{Catch, FuncType, Instruction, reencode::{self as reencode, Reencode}},
};

use crate::{X64Label, RSP};
use crate::naive::WriterExt as NaiveWriterExt;
pub use crate::naive::State;

/// Reserved register: r14 (sandbox base, never written).
const R14: Reg = Reg(14);
/// Scratch register for rtcall return-address: r11.
const R11: Reg = Reg(11);

// ── ABI discriminant ─────────────────────────────────────────────────────────

/// LFI sandboxed x86-64 ABI.
pub struct LfiAbi;

// ── BackendAbi impl ───────────────────────────────────────────────────────────

impl<W, Context> BackendAbi<W, Context> for LfiAbi
where
    W: LfiWriterExt<Context>,
{
    type Error = W::Error;
    type State<'s> = State<'s> where Self: 's;
    type Arch = X64Arch;

    fn emit_prologue(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        id: u32,
        data: &FnData,
    ) -> Result<(), W::Error> {
        state.local_count = data.num_params;
        state.num_returns = data.num_returns;
        state.control_depth = data.control_depth;
        state.tracing = data.tracing;
        w.align_to(ctx, arch, 32)?;
        w.pop(ctx, arch, &Reg(1))?;
        w.lea(
            ctx,
            arch,
            &Reg(0),
            &MemArgKind::Mem {
                base: Reg(1),
                offset: None,
                disp: 0u32.wrapping_sub(data.num_params as u32),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Segment::None,
            },
        )?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.set_label(ctx, arch, X64Label::Func { r#fn: id })
    }

    fn emit_new_local(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
    ) -> Result<(), W::Error> {
        state.local_count += 1;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_start_body(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
    ) -> Result<(), W::Error> {
        state.next_site_id = 1;
        if let Some(cfg) = state.tracing.as_ref().copied().filter(|c| c.enabled) {
            let mut bw = crate::codegen::BlitzW::new(w, ctx, arch);
            portal_solutions_blitz_codegen::emit_jit_preamble(
                &mut bw, cfg.table_base_off, 0,
                2, &mut state.label_index,
            )?;
        }
        w.push(ctx, arch, &Reg(1))?;
        w.push(ctx, arch, &Reg(0))?;
        w.lea(
            ctx,
            arch,
            &Reg(0),
            &MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 0u32.wrapping_sub(state.control_depth as u32 * 16),
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Segment::None,
            },
        )?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.push(ctx, arch, &Reg(0))?;
        for _ in 0..state.control_depth {
            w.push(ctx, arch, &Reg(0))?;
            w.push(ctx, arch, &Reg(0))?;
        }
        Ok(())
    }

    fn emit_local_get(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &State<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.lea(ctx, arch, &RSP, &MemArgKind::Mem {
            base: RSP, offset: None,
            disp: 0u32.wrapping_sub((n as isize * 8) as u32),
            size: MemorySize::_64, reg_class: RegisterClass::Gpr, segment: Segment::None,
        })?;
        w.pop(ctx, arch, &Reg(0))?;
        w.lea(ctx, arch, &RSP, &MemArgKind::Mem {
            base: RSP, offset: None,
            disp: 0u32.wrapping_sub(((n as isize + 1) * 8) as u32),
            size: MemorySize::_64, reg_class: RegisterClass::Gpr, segment: Segment::None,
        })?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_local_set(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &State<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)?;
        w.lea(ctx, arch, &RSP, &MemArgKind::Mem {
            base: RSP, offset: None,
            disp: 0u32.wrapping_sub(((n as isize + 1) * 8) as u32),
            size: MemorySize::_64, reg_class: RegisterClass::Gpr, segment: Segment::None,
        })?;
        w.push(ctx, arch, &Reg(0))?;
        w.lea(ctx, arch, &RSP, &MemArgKind::Mem {
            base: RSP, offset: None,
            disp: 0u32.wrapping_sub((n as isize * 8) as u32),
            size: MemorySize::_64, reg_class: RegisterClass::Gpr, segment: Segment::None,
        })?;
        w.xchg(ctx, arch, &RSP, &Reg::CTX)
    }

    fn emit_local_tee(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
        n: u32,
    ) -> Result<(), W::Error> {
        w.pop(ctx, arch, &Reg(0))?;
        w.push(ctx, arch, &Reg(0))?;
        Self::emit_local_set(w, ctx, arch, state, n)?;
        w.push(ctx, arch, &Reg(0))
    }

    fn emit_call(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        _state: &State<'_>,
        func_imports: &[(&str, &str)],
        fn_idx: u32,
        _sigs: &[FuncType],
        _fsigs: &[u32],
    ) -> Result<(), W::Error> {
        // Direct calls to known symbols are allowed by LFI without sandboxing.
        match func_imports.get(fn_idx as usize) {
            Some((module, name)) => {
                let sym = alloc::format!("{module}__{name}");
                w.lea_label(ctx, arch, &Reg(0), X64Label::External { name: sym })?;
                w.call(ctx, arch, &Reg(0))
            }
            None => {
                let idx = fn_idx - func_imports.len() as u32;
                w.lea_label(ctx, arch, &Reg(0), X64Label::Func { r#fn: idx })?;
                w.call(ctx, arch, &Reg(0))
            }
        }
    }

    fn emit_throw(
        _w: &mut W, _ctx: &mut Context, _arch: X64Arch, _state: &mut State<'_>,
        _tag_index: u32, _arity: u32,
    ) -> Result<(), W::Error> {
        todo!("LFI emit_throw")
    }

    fn emit_try_table_start(
        _w: &mut W, _ctx: &mut Context, _arch: X64Arch, _state: &mut State<'_>,
        _catches: &[Catch], _sigs: &[FuncType], _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("LFI emit_try_table_start")
    }

    fn emit_try_table_end(
        _w: &mut W, _ctx: &mut Context, _arch: X64Arch, _state: &mut State<'_>,
        _catches: &[Catch], _sigs: &[FuncType], _tags: &[u32],
    ) -> Result<(), W::Error> {
        todo!("LFI emit_try_table_end")
    }

    fn emit_return(
        w: &mut W,
        ctx: &mut Context,
        arch: X64Arch,
        state: &State<'_>,
    ) -> Result<(), W::Error> {
        // Restore RSP to frame base (same as NaiveAbi).
        w.mov(ctx, arch, &Reg(1), &RSP)?;
        w.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.lea(ctx, arch, &Reg(0), &MemArgKind::Mem {
            base: Reg(0), offset: None,
            disp: (state.local_count + 3 * 8) as u32,
            size: MemorySize::_8, reg_class: RegisterClass::Gpr, segment: Segment::None,
        })?;
        lfi_set_rsp(w, ctx, arch, Reg(0))?; // mov esp, eax; or rsp, r14
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.pop(ctx, arch, &Reg(0))?;
        w.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
        w.pop(ctx, arch, &Reg(0))?;
        for _ in 0..state.num_returns {
            w.mov(ctx, arch, &Reg(2), &Reg(1))?;
            w.push(ctx, arch, &Reg(2))?;
        }
        w.push(ctx, arch, &Reg(0))?;
        // LFI: rtcall replaces `ret`.
        w.lfi_rtcall(ctx, arch)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Emit the LFI-compliant RSP modification pattern (modsp macroinstruction).
///
/// Replaces a direct `mov rsp, rX` with the two-instruction sequence that
/// the LFI verifier accepts:
/// ```asm
/// mov esp, eX   ; 32-bit write clears upper bits (eX = lower 32 bits of rX)
/// or rsp, r14   ; restore sandbox base in upper bits
/// ```
///
/// Precondition: `src_reg` contains a value in sandbox form (r14 + 32-bit-offset).
/// The modsp pattern extracts the 32-bit offset and re-applies r14.
#[inline(always)]
fn lfi_set_rsp<W: WriterCore<Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: X64Arch,
    src_reg: Reg,
) -> Result<(), W::Error> {
    let esp = MemArgKind::NoMem(ArgKind::Reg { reg: RSP, size: MemorySize::_32 });
    let e_src = MemArgKind::NoMem(ArgKind::Reg { reg: src_reg, size: MemorySize::_32 });
    w.mov(ctx, arch, &esp, &e_src)?;  // mov esp, eSRC
    w.or(ctx, arch, &RSP, &R14)       // or rsp, r14
}

/// Build a GS-segment memory operand with a 32-bit base register.
///
/// The caller must have already zero-extended the address (via `mov eX, eX`)
/// before using this operand.
#[inline(always)]
fn gs_mem(base_reg: Reg, disp: u64, size: MemorySize) -> MemArgKind<ArgKind> {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base_reg, size: MemorySize::_32 },
        offset: None,
        disp: disp as u32,
        size,
        reg_class: RegisterClass::Gpr,
        segment: Segment::Gs,
    }
}

/// Emit `mov eX, eX` to zero-extend a register to a sandboxed 32-bit address.
#[inline(always)]
fn sandbox_addr<W: WriterCore<Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: X64Arch,
    reg: Reg,
) -> Result<(), W::Error> {
    let e_reg = MemArgKind::NoMem(ArgKind::Reg { reg, size: MemorySize::_32 });
    w.mov(ctx, arch, &e_reg, &e_reg)
}

// ── LfiWriterExt trait ────────────────────────────────────────────────────────

/// Extension trait for LFI-compliant x86-64 code generation.
pub trait LfiWriterExt<Context>: NaiveWriterExt<Context> {
    /// Emit the LFI runtime-call (rtcall) macroinstruction.
    ///
    /// Replaces `ret`:
    /// ```asm
    /// lea r11, [rip + return_label]
    /// jmp qword ptr [r14+0]
    /// return_label:
    /// ```
    fn lfi_rtcall(&mut self, ctx: &mut Context, arch: X64Arch) -> Result<(), Self::Error>;

    /// Handle a single WASM `Instruction` with LFI memory constraints.
    ///
    /// Memory ops use GS-segment sandboxed forms; all others delegate to
    /// the naive implementation.
    fn lfi_instr(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized;

    /// Handle a `MachOperator` with LFI constraints.
    ///
    /// This is the top-level entry point analogous to `NaiveWriterExt::handle_op`.
    fn lfi_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Self: Sized,
        Err: From<Self::Error> + From<reencode::Error<E>>;
}

impl<W, Context> LfiWriterExt<Context> for W
where
    W: NaiveWriterExt<Context> + Writer<X64Label, Context>,
{
    fn lfi_rtcall(&mut self, ctx: &mut Context, arch: X64Arch) -> Result<(), Self::Error> {
        // Use a per-call unique label so GAS resolves the forward reference.
        // The verifier validates: lea displacement == size-of-following-jmp.
        // GAS computes this automatically when the label is placed right after jmp.
        static LFI_RTCALL_CTR: core::sync::atomic::AtomicUsize =
            core::sync::atomic::AtomicUsize::new(0x4c465900); // "LFI\0" prefix
        let idx = LFI_RTCALL_CTR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let lbl = X64Label::Indexed { idx };

        self.lea_label(ctx, arch, &R11, lbl.clone())?;
        // jmp qword ptr [r14+0]
        self.jmp(ctx, arch, &MemArgKind::Mem {
            base: ArgKind::Reg { reg: R14, size: MemorySize::_64 },
            offset: None,
            disp: 0,
            size: MemorySize::_64,
            reg_class: RegisterClass::Gpr,
            segment: Segment::None,
        })?;
        self.set_label(ctx, arch, lbl)
    }

    fn lfi_instr(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        match op {
            // ── Sandboxed loads ───────────────────────────────────────────────
            Instruction::I64Load(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.mov(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_64))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I32Load(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let eax = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 });
                self.mov(ctx, arch, &eax, &gs_mem(Reg(0), m.offset, MemorySize::_32))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::F64Load(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.mov(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_64))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::F32Load(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let eax = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 });
                self.mov(ctx, arch, &eax, &gs_mem(Reg(0), m.offset, MemorySize::_32))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load8S(m) | Instruction::I32Load8S(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.movsx(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_8))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load8U(m) | Instruction::I32Load8U(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.movzx(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_8))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load16S(m) | Instruction::I32Load16S(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.movsx(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_16))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load16U(m) | Instruction::I32Load16U(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.movzx(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_16))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load32S(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.movsx(ctx, arch, &Reg(0), &gs_mem(Reg(0), m.offset, MemorySize::_32))?;
                self.push(ctx, arch, &Reg(0))
            }
            Instruction::I64Load32U(m) => {
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let eax = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(0), size: MemorySize::_32 });
                self.mov(ctx, arch, &eax, &gs_mem(Reg(0), m.offset, MemorySize::_32))?;
                self.push(ctx, arch, &Reg(0))
            }

            // ── Sandboxed stores ──────────────────────────────────────────────
            Instruction::I64Store(m) | Instruction::F64Store(m) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                self.mov(ctx, arch, &gs_mem(Reg(0), m.offset, MemorySize::_64), &Reg(2))
            }
            Instruction::I32Store(m) | Instruction::F32Store(m) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let edx = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_32 });
                self.mov(ctx, arch, &gs_mem(Reg(0), m.offset, MemorySize::_32), &edx)
            }
            Instruction::I64Store8(m) | Instruction::I32Store8(m) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let dl = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_8 });
                self.mov(ctx, arch, &gs_mem(Reg(0), m.offset, MemorySize::_8), &dl)
            }
            Instruction::I64Store16(m) | Instruction::I32Store16(m) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let dx = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_16 });
                self.mov(ctx, arch, &gs_mem(Reg(0), m.offset, MemorySize::_16), &dx)
            }
            Instruction::I64Store32(m) => {
                self.pop(ctx, arch, &Reg(2))?;
                self.pop(ctx, arch, &Reg(0))?;
                sandbox_addr(self, ctx, arch, Reg(0))?;
                let edx = MemArgKind::NoMem(ArgKind::Reg { reg: Reg(2), size: MemorySize::_32 });
                self.mov(ctx, arch, &gs_mem(Reg(0), m.offset, MemorySize::_32), &edx)
            }

            // ── LFI return: rtcall replaces `ret` ────────────────────────────
            // `Instruction::Return` in naive emits the restore sequence + `ret`.
            // We replicate the restore logic here and substitute rtcall.
            Instruction::Return => {
                self.mov(ctx, arch, &Reg(1), &RSP)?;
                self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.lea(ctx, arch, &Reg(0), &MemArgKind::Mem {
                    base: Reg(0), offset: None,
                    disp: 0u32.wrapping_sub(8),
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    segment: Segment::None,
                })?;
                lfi_set_rsp(self, ctx, arch, Reg(0))?; // mov esp, eax; or rsp, r14
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                self.pop(ctx, arch, &Reg(0))?;
                for _ in 0..state.num_returns {
                    self.mov(ctx, arch, &Reg(2), &Reg(1))?;
                    self.push(ctx, arch, &Reg(2))?;
                }
                self.push(ctx, arch, &Reg(0))?;
                self.lfi_rtcall(ctx, arch)
            }

            // ── Everything else: naive (already LFI-compatible) ───────────────
            other => self._handle_op(ctx, arch, state, func_imports, sigs, tags, other, target),
        }
    }

    fn lfi_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Self: Sized,
        Err: From<Self::Error> + From<reencode::Error<E>>,
    {
        // Replicate the MachOperator dispatch from naive::handle_op, routing
        // MachOperator::Instruction and ::Operator through lfi_instr instead
        // of _handle_op so we intercept sandboxed memory operations.
        if target != state.body {
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jmp_label(
                    ctx,
                    arch,
                    X64Label::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            state.label_index - 1
                        }),
                    },
                ).map_err(Err::from)?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, X64Label::Indexed { idx }).map_err(Err::from)?;
                }
            }
        }
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.tracing = data.tracing;
                self.align_to(ctx, arch, 32).map_err(Err::from)?;
                self.pop(ctx, arch, &Reg(1)).map_err(Err::from)?;
                self.lea(ctx, arch, &Reg(0), &MemArgKind::Mem {
                    base: Reg(1), offset: None,
                    disp: 0u32.wrapping_sub(data.num_params as u32),
                    size: MemorySize::_64, reg_class: RegisterClass::Gpr, segment: Segment::None,
                }).map_err(Err::from)?;
                self.xchg(ctx, arch, &Reg(0), &Reg::CTX).map_err(Err::from)?;
                self.set_label(ctx, arch, X64Label::Func { r#fn: *id }).map_err(Err::from)?;
            }
            MachOperator::Instruction { op, .. } => {
                self.lfi_instr(ctx, arch, state, func_imports, sigs, tags, op, target).map_err(Err::from)?;
            }
            MachOperator::Operator { op, .. } => match op.as_ref() {
                None => return Ok(()),
                Some(raw_op) => {
                    let instr = rewriter.instruction(raw_op.clone())?;
                    self.lfi_instr(ctx, arch, state, func_imports, sigs, tags, &instr, target).map_err(Err::from)?;
                }
            },
            // Remaining MachOperator variants (Local, StartBody, EndBody, etc.)
            // are handled identically to naive — delegate through handle_op.
            other => {
                self.handle_op::<E, Err>(ctx, arch, state, func_imports, sigs, tags, other, rewriter, target)?;
            }
        }
        Ok(())
    }
}
