//! `wax-core` `InstructionSink` / `OperatorSink` impls for the AArch64 backend.
//!
//! [`AArch64WasmSink<Abi>`] is a newtype over [`WasmSink<naive::State, AArch64Arch>`]
//! parameterised by an ABI marker type.  The default is [`NaiveAbi`]; use
//! `AArch64WasmSink::<SysVAbi>::new(arch)` or `AArch64WasmSink::<LfiAbi>::new(arch)`
//! for other calling conventions.

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

use crate::{AArch64Arch, AArch64Label, naive};
use crate::abi::{NaiveAbi, SysVAbi};
use crate::lfi::{LfiAbi, LfiWriterExt};
use crate::sysv::SysVWriterExt;

/// Scratch register used for ambient label loads (T0 = x9).
const T0: Reg = Reg(9);

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// AArch64 WASM sink with `InstructionSink` / `OperatorSink` support.
///
/// The `Abi` type parameter selects the calling convention:
/// - `NaiveAbi` (default) — blitz WASM stack convention
/// - `SysVAbi`            — AAPCS64 / ARM64 System V
/// - `LfiAbi`             — LFI sandboxed ABI
pub struct AArch64WasmSink<Abi = NaiveAbi>(
    pub WasmSink<naive::State<'static>, AArch64Arch>,
    PhantomData<Abi>,
);

impl<Abi> AArch64WasmSink<Abi> {
    pub fn new(arch: AArch64Arch) -> Self {
        Self(WasmSink::new(arch), PhantomData)
    }
}

impl<Abi> Deref for AArch64WasmSink<Abi> {
    type Target = WasmSink<naive::State<'static>, AArch64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Abi> DerefMut for AArch64WasmSink<Abi> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
    for AArch64WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
    for AArch64WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.wasm_push(&mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.bl(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.br(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
}

// ---------------------------------------------------------------------------
// SysVAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
    for AArch64WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
    for AArch64WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.wasm_push(&mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.bl(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.br(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
}

// ---------------------------------------------------------------------------
// LfiAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<LfiAbi>
where
    W: LfiWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
        ctx.writer.lfi_handle_op(
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
    for AArch64WasmSink<LfiAbi>
where
    W: LfiWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
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
        ctx.writer.lfi_handle_op(
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
    for AArch64WasmSink<LfiAbi>
where
    W: LfiWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.wasm_push(&mut ctx.asm_ctx, self.0.arch, T0).map_err(HandleOpError::from)
    }
    fn call_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.bl(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
    fn jump_ambient(&mut self, ctx: &mut WaxHandle<W, AsmCtx>, name: &str) -> Result<(), HandleOpError<Infallible>> {
        ctx.writer.adr_label(&mut ctx.asm_ctx, self.0.arch, &T0, AArch64Label::Ambient { name: name.into() }).map_err(HandleOpError::from)?;
        ctx.writer.br(&mut ctx.asm_ctx, self.0.arch, &T0).map_err(HandleOpError::from)
    }
}
