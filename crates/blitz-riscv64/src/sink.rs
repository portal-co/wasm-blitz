//! `wax-core` `InstructionSink` / `OperatorSink` impls for the RISC-V 64 backend.
//!
//! [`RiscV64WasmSink<Abi>`] is a newtype over [`WasmSink<naive::State, RiscV64Arch>`]
//! parameterised by an ABI marker type.  The default is [`NaiveAbi`]; use
//! `RiscV64WasmSink::<SysVAbi>::new(arch)` for the SysV calling convention.

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
use wasm_encoder::{Instruction, reencode::RoundtripReencoder};
use wasmparser::Operator;

use crate::{RiscV64Arch, RiscvLabel, naive};
use crate::abi::{NaiveAbi, SysVAbi};
use crate::sysv::SysVWriterExt;

/// Scratch register for ambient label loads (t0 = x5).
const T0: Reg = Reg(5);
/// Return address register (ra = x1).
const RA: Reg = Reg(1);

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// RISC-V 64 WASM sink with `InstructionSink` / `OperatorSink` support.
///
/// The `Abi` type parameter selects the calling convention:
/// - `NaiveAbi` (default) — blitz WASM stack convention
/// - `SysVAbi`            — RISC-V psABI LP64
pub struct RiscV64WasmSink<Abi = NaiveAbi>(
    pub WasmSink<naive::State<'static>, RiscV64Arch>,
    PhantomData<Abi>,
);

impl<Abi> RiscV64WasmSink<Abi> {
    pub fn new(arch: RiscV64Arch) -> Self {
        Self(WasmSink::new(arch), PhantomData)
    }
}

impl<Abi> Deref for RiscV64WasmSink<Abi> {
    type Target = WasmSink<naive::State<'static>, RiscV64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Abi> DerefMut for RiscV64WasmSink<Abi> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink<NaiveAbi>
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
    for RiscV64WasmSink<NaiveAbi>
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
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &RA, &T0, 0).map_err(HandleOpError::from)
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &Reg(0), &T0, 0).map_err(HandleOpError::from)
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink<SysVAbi>
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
    for RiscV64WasmSink<SysVAbi>
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
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
    W::Error: From<core::fmt::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        naive::push(&mut ctx.writer, &mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &RA, &T0, 0).map_err(HandleOpError::from)
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.la_label(&mut ctx.asm_ctx, self.0.arch, &T0, RiscvLabel::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.jalr(&mut ctx.asm_ctx, self.0.arch, &Reg(0), &T0, 0).map_err(HandleOpError::from)
    }
}
