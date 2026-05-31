//! `wax-core` `InstructionSink` / `OperatorSink` impls for the x86-64 backend.
//!
//! [`X64WasmSink<Abi, S>`] is a newtype over [`WasmSink<S, X64Arch>`] parameterised
//! by an ABI marker type and its associated state.  Convenience type aliases make
//! common combinations ergonomic:
//!
//! ```text
//! X64NaiveWasmSink  ≡  X64WasmSink<NaiveAbi, naive::State>   (default)
//! X64SysVWasmSink   ≡  X64WasmSink<SysVAbi,  sysv::SysVState>
//! X64LfiWasmSink    ≡  X64WasmSink<LfiAbi,   naive::State>
//! ```
//!
//! The newtype is required by Rust's orphan rule: both `InstructionSink` (from
//! `wax-core`) and `WasmSink` (from `blitz-common`) are foreign types, so the
//! impl must name a local type as `Self`.

use core::ops::{Deref, DerefMut};
use core::convert::Infallible;
use core::marker::PhantomData;
use portal_solutions_blitz_common::{
    HandleOpError,
    ops::MachOperator,
    sink::{WaxHandle, WasmSink},
};
use wax_core::build::{InstructionSink, OperatorSink};
use wasm_encoder::{Instruction, reencode::RoundtripReencoder};
use wasmparser::Operator;

use crate::{X64Arch, naive};
use crate::abi::{NaiveAbi, SysVAbi};
use crate::lfi::{LfiAbi, LfiWriterExt};
use crate::sysv::{SysVWriterExt, SysVState};

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// x86-64 WASM sink with `InstructionSink` / `OperatorSink` support.
///
/// `Abi` selects the calling convention; `S` is the associated state type
/// (inferred from the convenience aliases below).
pub struct X64WasmSink<Abi = NaiveAbi, S = naive::State<'static>>(
    pub WasmSink<S, X64Arch>,
    PhantomData<Abi>,
);

/// Convenience alias — blitz WASM stack convention.
pub type X64NaiveWasmSink = X64WasmSink<NaiveAbi, naive::State<'static>>;
/// Convenience alias — System V AMD64 ABI.
pub type X64SysVWasmSink  = X64WasmSink<SysVAbi, SysVState<'static>>;
/// Convenience alias — LFI sandboxed ABI.
pub type X64LfiWasmSink   = X64WasmSink<LfiAbi, naive::State<'static>>;

impl<Abi, S: Default> X64WasmSink<Abi, S> {
    pub fn new(arch: X64Arch) -> Self {
        Self(WasmSink::new(arch), PhantomData)
    }
}

impl<Abi, S> Deref for X64WasmSink<Abi, S> {
    type Target = WasmSink<S, X64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Abi, S> DerefMut for X64WasmSink<Abi, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// NaiveAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for X64WasmSink<NaiveAbi, naive::State<'static>>
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
    for X64WasmSink<NaiveAbi, naive::State<'static>>
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
}

// ---------------------------------------------------------------------------
// SysVAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for X64WasmSink<SysVAbi, SysVState<'static>>
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
    for X64WasmSink<SysVAbi, SysVState<'static>>
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
}

// ---------------------------------------------------------------------------
// LfiAbi — OperatorSink / InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for X64WasmSink<LfiAbi, naive::State<'static>>
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
    for X64WasmSink<LfiAbi, naive::State<'static>>
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
}
