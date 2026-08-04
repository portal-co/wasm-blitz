//! `wax-core` `InstructionSink` / `OperatorSink` impls for the RISC-V 32 backend.
//!
//! [`RiscV32WasmSink<Abi>`] is a newtype over [`WasmSink<naive::State, RiscV32Arch>`]
//! parameterised by an ABI marker type.  The default is [`NaiveAbi`]; use
//! `RiscV32WasmSink::<SysVAbi>::new(arch)` for the SysV calling convention.

use core::ops::{Deref, DerefMut};
use core::convert::Infallible;
use core::marker::PhantomData;
use portal_solutions_blitz_common::{
    HandleOpError,
    ops::MachOperator,
    sink::{WaxHandle, WasmSink},
};
use portal_solutions_blitz_common::asm::Reg;
use wax_core::build::{AmbientSink, InstructionSink, OperatorSink};
use wasm_encoder::{FuncType, Instruction, reencode::RoundtripReencoder};
use wasmparser::Operator;

use crate::{RiscV32Arch, RiscvLabel, naive};
use crate::abi::{NaiveAbi, SysVAbi};
use crate::sysv::SysVWriterExt;

/// RISC-V psABI argument registers: a0–a7 (x10–x17).
const RV32_ARG_REGS: [Reg; 8] = [Reg(10), Reg(11), Reg(12), Reg(13), Reg(14), Reg(15), Reg(16), Reg(17)];
/// First return value register (a0 = x10).
const A0: Reg = Reg(10);
/// Second return value register (a1 = x11).
const A1: Reg = Reg(11);
/// Scratch register for ambient label loads (t0 = x5, caller-saved, not an arg reg).
const T0: Reg = Reg(5);
/// Return address register (ra = x1).
const RA: Reg = Reg(1);

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// RISC-V 32 WASM sink with `InstructionSink` / `OperatorSink` support.
///
/// The `Abi` type parameter selects the calling convention:
/// - `NaiveAbi` (default) — blitz WASM stack convention
/// - `SysVAbi`            — RISC-V psABI ILP32
pub struct RiscV32WasmSink<Abi = NaiveAbi>(
    pub WasmSink<naive::State<'static>, RiscV32Arch>,
    PhantomData<Abi>,
);

impl<Abi> RiscV32WasmSink<Abi> {
    pub fn new(arch: RiscV32Arch) -> Self {
        Self(WasmSink::new(arch), PhantomData)
    }
}

impl<Abi> Deref for RiscV32WasmSink<Abi> {
    type Target = WasmSink<naive::State<'static>, RiscV32Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Abi> DerefMut for RiscV32WasmSink<Abi> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn operator(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        op: &Operator<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> =
            self.0.func_imports.iter().map(|(m, n)| (m.clone(), n.clone())).collect();
        let imports: alloc::vec::Vec<(&str, &str)> =
            imports_owned.iter().map(|(m, n)| (m.as_str(), n.as_str())).collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Operator { op: Some(op.clone()), annot: () };
        ctx.writer.handle_op(
            &mut ctx.asm_ctx,
            self.0.arch,
            &mut self.0.state,
            &imports,
            &sigs_owned,
            &tags_owned,
            &mach,
            &mut RoundtripReencoder,
            self.0.target,
        )
    }
}

impl<W, AsmCtx> InstructionSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn instruction(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        insn: &Instruction<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> =
            self.0.func_imports.iter().map(|(m, n)| (m.clone(), n.clone())).collect();
        let imports: alloc::vec::Vec<(&str, &str)> =
            imports_owned.iter().map(|(m, n)| (m.as_str(), n.as_str())).collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Instruction { op: insn.clone(), annot: () };
        ctx.writer.handle_op(
            &mut ctx.asm_ctx,
            self.0.arch,
            &mut self.0.state,
            &imports,
            &sigs_owned,
            &tags_owned,
            &mach,
            &mut RoundtripReencoder,
            self.0.target,
        )
    }
    fn as_ambient_sink(&mut self) -> Option<&mut (dyn AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>> + '_)> {
        Some(self)
    }
    fn has_ambient_sink(&self) -> bool { true }
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str, sig: &FuncType) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        let n_results = sig.results().len();
        for i in (0..n_params.min(8)).rev() {
            naive::pop(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, &RV32_ARG_REGS[i]).map_err(HandleOpError::from)?;
        }
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &RA, &T0, 0).map_err(HandleOpError::from)?;
        if n_results > 1 { naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, A1).map_err(HandleOpError::from)?; }
        if n_results > 0 { naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, A0).map_err(HandleOpError::from)?; }
        Ok(())
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str, sig: &FuncType) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        for i in (0..n_params.min(8)).rev() {
            naive::pop(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, &RV32_ARG_REGS[i]).map_err(HandleOpError::from)?;
        }
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &Reg(0), &T0, 0).map_err(HandleOpError::from)
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn operator(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        op: &Operator<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> =
            self.0.func_imports.iter().map(|(m, n)| (m.clone(), n.clone())).collect();
        let imports: alloc::vec::Vec<(&str, &str)> =
            imports_owned.iter().map(|(m, n)| (m.as_str(), n.as_str())).collect();
        let mach = MachOperator::Operator { op: Some(op.clone()), annot: () };
        ctx.writer.sysv_handle_op(
            &mut ctx.asm_ctx,
            self.0.arch,
            &mut self.0.state,
            &imports,
            &mach,
            &mut RoundtripReencoder,
            self.0.target,
        )
    }
}

impl<W, AsmCtx> InstructionSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn instruction(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        insn: &Instruction<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> =
            self.0.func_imports.iter().map(|(m, n)| (m.clone(), n.clone())).collect();
        let imports: alloc::vec::Vec<(&str, &str)> =
            imports_owned.iter().map(|(m, n)| (m.as_str(), n.as_str())).collect();
        let mach = MachOperator::Instruction { op: insn.clone(), annot: () };
        ctx.writer.sysv_handle_op(
            &mut ctx.asm_ctx,
            self.0.arch,
            &mut self.0.state,
            &imports,
            &mach,
            &mut RoundtripReencoder,
            self.0.target,
        )
    }
    fn as_ambient_sink(&mut self) -> Option<&mut (dyn AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>> + '_)> {
        Some(self)
    }
    fn has_ambient_sink(&self) -> bool { true }
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV32WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str, sig: &FuncType) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        let n_results = sig.results().len();
        for i in (0..n_params.min(8)).rev() {
            naive::pop(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, &RV32_ARG_REGS[i]).map_err(HandleOpError::from)?;
        }
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &RA, &T0, 0).map_err(HandleOpError::from)?;
        if n_results > 1 { naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, A1).map_err(HandleOpError::from)?; }
        if n_results > 0 { naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, A0).map_err(HandleOpError::from)?; }
        Ok(())
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str, sig: &FuncType) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        for i in (0..n_params.min(8)).rev() {
            naive::pop(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, &RV32_ARG_REGS[i]).map_err(HandleOpError::from)?;
        }
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &Reg(0), &T0, 0).map_err(HandleOpError::from)
    }
}
