//! `wax-core` `InstructionSink` / `OperatorSink` impls for the x86-64 backend.
//!
//! [`X64WasmSink`] is a newtype over [`WasmSink<naive::State, X64Arch>`].
//! The newtype is required by Rust's orphan rule: both `InstructionSink` (from
//! `wax-core`) and `WasmSink` (from `blitz-common`) are foreign types, so the
//! impl must name a local type as `Self`.

use core::ops::{Deref, DerefMut};
use core::convert::Infallible;
use portal_solutions_blitz_common::{
    HandleOpError,
    ops::MachOperator,
    sink::{WaxHandle, WasmSink},
};
use wax_core::build::{InstructionSink, OperatorSink};
use wasm_encoder::{Instruction, reencode::RoundtripReencoder};
use wasmparser::Operator;

use crate::{X64Arch, naive};

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// x86-64 WASM sink with `InstructionSink` / `OperatorSink` support.
///
/// Wraps [`WasmSink<naive::State, X64Arch>`] with a local type so that the
/// orphan rule permits implementing the foreign wax-core traits.
pub struct X64WasmSink(pub WasmSink<naive::State, X64Arch>);

impl X64WasmSink {
    pub fn new(arch: X64Arch) -> Self {
        Self(WasmSink::new(arch))
    }
}

impl Deref for X64WasmSink {
    type Target = WasmSink<naive::State, X64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for X64WasmSink {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// OperatorSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for X64WasmSink
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn operator(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        op: &Operator<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        // Clone into locals so there is no live borrow on self.0 when
        // handle_op needs &mut self.0.state alongside &imports / &sigs / &tags.
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

// ---------------------------------------------------------------------------
// InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> InstructionSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for X64WasmSink
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
