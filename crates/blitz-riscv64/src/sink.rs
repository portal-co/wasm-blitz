//! `wax-core` `InstructionSink` / `OperatorSink` impls for the RISC-V 64 backend.

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

use crate::{RiscV64Arch, naive};

// ---------------------------------------------------------------------------
// Newtype
// ---------------------------------------------------------------------------

/// RISC-V 64 WASM sink with `InstructionSink` / `OperatorSink` support.
pub struct RiscV64WasmSink(pub WasmSink<naive::State, RiscV64Arch>);

impl RiscV64WasmSink {
    pub fn new(arch: RiscV64Arch) -> Self {
        Self(WasmSink::new(arch))
    }
}

impl Deref for RiscV64WasmSink {
    type Target = WasmSink<naive::State, RiscV64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RiscV64WasmSink {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ---------------------------------------------------------------------------
// OperatorSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink
where
    W: naive::WriterExt<AsmCtx>,
    W::Error: Into<HandleOpError<Infallible>>,
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
        ctx.writer.handle_op(
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
// InstructionSink
// ---------------------------------------------------------------------------

impl<W, AsmCtx> InstructionSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for RiscV64WasmSink
where
    W: naive::WriterExt<AsmCtx>,
    W::Error: Into<HandleOpError<Infallible>>,
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
        ctx.writer.handle_op(
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
