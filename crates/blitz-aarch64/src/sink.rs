//! `wax-core` `InstructionSink` / `OperatorSink` impls for the AArch64 backend.
//!
//! [`AArch64WasmSink<Abi>`] is a newtype over [`WasmSink<naive::State, AArch64Arch>`]
//! parameterised by an ABI marker type.  The default is [`NaiveAbi`]; use
//! `AArch64WasmSink::<SysVAbi>::new(arch)` or `AArch64WasmSink::<LfiAbi>::new(arch)`
//! for other calling conventions.

use core::convert::Infallible;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::{
    HandleOpError,
    ops::MachOperator,
    sink::{WasmSink, WaxHandle},
};
use wasm_encoder::{FuncType, Instruction, reencode::RoundtripReencoder};
use wasmparser::Operator;
use wax_core::build::{AmbientSink, InstructionSink, OperatorSink};

use crate::abi::{NaiveAbi, SysVAbi};
use crate::lfi::{LfiAbi, LfiWriterExt};
use crate::sysv::SysVWriterExt;
use crate::{AArch64Arch, AArch64Label, naive};

/// AAPCS64 argument registers: X0–X7.
const AARCH64_ARG_REGS: [Reg; 8] = [
    Reg(0),
    Reg(1),
    Reg(2),
    Reg(3),
    Reg(4),
    Reg(5),
    Reg(6),
    Reg(7),
];
/// Scratch register for ambient label loads (T0 = X9, caller-saved, not an arg reg).
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Operator {
            op: Some(op.clone()),
            annot: (),
        };
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Instruction {
            op: insn.clone(),
            annot: (),
        };
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
    fn as_ambient_sink(
        &mut self,
    ) -> Option<&mut (dyn AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>> + '_)> {
        Some(self)
    }
    fn has_ambient_sink(&self) -> bool {
        true
    }
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<NaiveAbi>
where
    W: naive::WriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
    ) -> Result<(), HandleOpError<Infallible>> {
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .wasm_push(&mut ctx.asm_ctx, self.0.arch, T0)
            .map_err(HandleOpError::from)
    }
    fn call_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        let n_results = sig.results().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .bl(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)?;
        if n_results > 1 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(1))
                .map_err(HandleOpError::from)?;
        }
        if n_results > 0 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(0))
                .map_err(HandleOpError::from)?;
        }
        Ok(())
    }
    fn jump_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .br(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let mach = MachOperator::Operator {
            op: Some(op.clone()),
            annot: (),
        };
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let mach = MachOperator::Instruction {
            op: insn.clone(),
            annot: (),
        };
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
    fn as_ambient_sink(
        &mut self,
    ) -> Option<&mut (dyn AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>> + '_)> {
        Some(self)
    }
    fn has_ambient_sink(&self) -> bool {
        true
    }
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<SysVAbi>
where
    W: SysVWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
    ) -> Result<(), HandleOpError<Infallible>> {
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .wasm_push(&mut ctx.asm_ctx, self.0.arch, T0)
            .map_err(HandleOpError::from)
    }
    fn call_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        let n_results = sig.results().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .bl(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)?;
        if n_results > 1 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(1))
                .map_err(HandleOpError::from)?;
        }
        if n_results > 0 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(0))
                .map_err(HandleOpError::from)?;
        }
        Ok(())
    }
    fn jump_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .br(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Operator {
            op: Some(op.clone()),
            annot: (),
        };
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
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let sigs_owned: alloc::vec::Vec<_> = self.0.sigs.clone();
        let tags_owned: alloc::vec::Vec<u32> = self.0.tags.clone();
        let mach = MachOperator::Instruction {
            op: insn.clone(),
            annot: (),
        };
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
    fn as_ambient_sink(
        &mut self,
    ) -> Option<&mut (dyn AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>> + '_)> {
        Some(self)
    }
    fn has_ambient_sink(&self) -> bool {
        true
    }
}

impl<W, AsmCtx> AmbientSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64WasmSink<LfiAbi>
where
    W: LfiWriterExt<AsmCtx>,
    HandleOpError<Infallible>: From<W::Error>,
{
    fn push_ambient_addr(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
    ) -> Result<(), HandleOpError<Infallible>> {
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .wasm_push(&mut ctx.asm_ctx, self.0.arch, T0)
            .map_err(HandleOpError::from)
    }
    fn call_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        let n_results = sig.results().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .bl(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)?;
        if n_results > 1 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(1))
                .map_err(HandleOpError::from)?;
        }
        if n_results > 0 {
            ctx.writer
                .wasm_push(&mut ctx.asm_ctx, self.0.arch, Reg(0))
                .map_err(HandleOpError::from)?;
        }
        Ok(())
    }
    fn jump_ambient(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        name: &str,
        sig: &FuncType,
    ) -> Result<(), HandleOpError<Infallible>> {
        let n_params = sig.params().len();
        for i in (0..n_params.min(8)).rev() {
            ctx.writer
                .wasm_pop(&mut ctx.asm_ctx, self.0.arch, AARCH64_ARG_REGS[i])
                .map_err(HandleOpError::from)?;
        }
        crate::load_label_addr(
            &mut ctx.writer,
            &mut ctx.asm_ctx,
            self.0.arch,
            &T0,
            AArch64Label::Ambient { name: name.into() },
        )
        .map_err(HandleOpError::from)?;
        ctx.writer
            .br(&mut ctx.asm_ctx, self.0.arch, &T0)
            .map_err(HandleOpError::from)
    }
}

// ---------------------------------------------------------------------------
// Sharing sinks — opt-in code-sharing wrappers around the standard sinks.
//
// The wrapper owns the per-compilation sharing state and routes every
// operator/instruction through the backend's sharing driver. Lifecycle
// (`start_fn`, body boundaries) never arrives as a `MachOperator` on the sink
// path, so the wrapper handles it: `start_fn` resets staging (finalizing any
// leftover candidates from a malformed previous body) and `end_body`
// finalizes the just-finished function's helper area.
// ---------------------------------------------------------------------------

use crate::sysv::SysVSharingWriterExt;
use portal_solutions_asm_writer::RewindableCodeWriter;
use portal_solutions_blitz_codegen::code_sharing::CodeSharingConfig;
use portal_solutions_blitz_common::ops::FnData;

/// AArch64 SysV sink with code sharing.
pub struct AArch64SharingSysVWasmSink<B> {
    /// The underlying SysV sink (WASM state, imports, sigs, target).
    pub inner: AArch64WasmSink<SysVAbi>,
    /// Code-sharing collector state.
    pub sharing: crate::sysv::AArch64Sharing<B>,
}

impl<B> AArch64SharingSysVWasmSink<B> {
    pub fn new(arch: AArch64Arch, config: CodeSharingConfig) -> Self {
        Self {
            inner: AArch64WasmSink::new(arch),
            sharing: crate::sysv::AArch64Sharing::new(config),
        }
    }

    /// Start a function: reset the sharing collector (finalizing any leftover
    /// staging from a previous body), then emit the prologue.
    pub fn start_fn<W, AsmCtx>(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        id: u32,
        data: &FnData,
    ) -> Result<(), HandleOpError<Infallible>>
    where
        W: SysVSharingWriterExt<AsmCtx> + RewindableCodeWriter<AArch64Label, Blob = B>,
        HandleOpError<Infallible>: From<<W as portal_solutions_asm_aarch64::out::WriterCore<AsmCtx>>::Error>
            + From<<W as RewindableCodeWriter<AArch64Label>>::Error>,
    {
        self.sharing.depth = Some(0);
        self.sharing.open_anchor = None;
        if self.sharing.staging.staged_count() > 0 {
            let staging = core::mem::replace(
                &mut self.sharing.staging,
                portal_solutions_blitz_codegen::code_sharing::FunctionStaging::new(
                    self.sharing.config,
                    crate::codegen::AARCH64_SYSV_OUTLINE_COST,
                ),
            );
            let mut sw = crate::codegen::SharingW {
                writer: &mut ctx.writer,
                ctx: &mut ctx.asm_ctx,
                arch: self.inner.0.arch,
                next_helper: self.sharing.next_helper,
            };
            staging.finalize::<_, _, _, HandleOpError<Infallible>>(&mut sw)?;
            self.sharing.next_helper = sw.next_helper;
        }
        self.inner.0.target = id;
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .inner
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let mach = MachOperator::StartFn {
            id,
            data: data.clone(),
        };
        ctx.writer.sysv_handle_op_sharing(
            &mut ctx.asm_ctx,
            self.inner.0.arch,
            &mut self.inner.0.state,
            &mut self.sharing,
            &imports,
            &mach,
            &mut RoundtripReencoder,
            self.inner.0.target,
        )
    }

    /// End-of-body: feed `EndBody` through the driver so any still-open
    /// candidate closes and the helper area is emitted (for SysV the driver
    /// already finalized at the function-level `End`; `EndBody` is then a
    /// no-op).
    pub fn end_body<W, AsmCtx>(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
    ) -> Result<(), HandleOpError<Infallible>>
    where
        W: SysVSharingWriterExt<AsmCtx> + RewindableCodeWriter<AArch64Label, Blob = B>,
        HandleOpError<Infallible>: From<<W as portal_solutions_asm_aarch64::out::WriterCore<AsmCtx>>::Error>
            + From<<W as RewindableCodeWriter<AArch64Label>>::Error>,
    {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .inner
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        ctx.writer.sysv_handle_op_sharing(
            &mut ctx.asm_ctx,
            self.inner.0.arch,
            &mut self.inner.0.state,
            &mut self.sharing,
            &imports,
            &MachOperator::EndBody,
            &mut RoundtripReencoder,
            self.inner.0.target,
        )
    }
}

impl<B> Deref for AArch64SharingSysVWasmSink<B> {
    type Target = WasmSink<naive::State<'static>, AArch64Arch>;
    fn deref(&self) -> &Self::Target {
        &self.inner.0
    }
}

impl<B> DerefMut for AArch64SharingSysVWasmSink<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.0
    }
}

impl<W, AsmCtx, B> OperatorSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64SharingSysVWasmSink<B>
where
    W: SysVSharingWriterExt<AsmCtx> + RewindableCodeWriter<AArch64Label, Blob = B>,
    HandleOpError<Infallible>: From<<W as portal_solutions_asm_aarch64::out::WriterCore<AsmCtx>>::Error>
        + From<<W as RewindableCodeWriter<AArch64Label>>::Error>,
{
    fn operator(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        op: &Operator<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .inner
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let mach = MachOperator::Operator {
            op: Some(op.clone()),
            annot: (),
        };
        ctx.writer.sysv_handle_op_sharing(
            &mut ctx.asm_ctx,
            self.inner.0.arch,
            &mut self.inner.0.state,
            &mut self.sharing,
            &imports,
            &mach,
            &mut RoundtripReencoder,
            self.inner.0.target,
        )
    }
}

impl<W, AsmCtx, B> InstructionSink<WaxHandle<W, AsmCtx>, HandleOpError<Infallible>>
    for AArch64SharingSysVWasmSink<B>
where
    W: SysVSharingWriterExt<AsmCtx> + RewindableCodeWriter<AArch64Label, Blob = B>,
    HandleOpError<Infallible>: From<<W as portal_solutions_asm_aarch64::out::WriterCore<AsmCtx>>::Error>
        + From<<W as RewindableCodeWriter<AArch64Label>>::Error>,
{
    fn instruction(
        &mut self,
        ctx: &mut WaxHandle<W, AsmCtx>,
        insn: &Instruction<'_>,
    ) -> Result<(), HandleOpError<Infallible>> {
        let imports_owned: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = self
            .inner
            .0
            .func_imports
            .iter()
            .map(|(m, n)| (m.clone(), n.clone()))
            .collect();
        let imports: alloc::vec::Vec<(&str, &str)> = imports_owned
            .iter()
            .map(|(m, n)| (m.as_str(), n.as_str()))
            .collect();
        let mach = MachOperator::Instruction {
            op: insn.clone(),
            annot: (),
        };
        ctx.writer.sysv_handle_op_sharing(
            &mut ctx.asm_ctx,
            self.inner.0.arch,
            &mut self.inner.0.state,
            &mut self.sharing,
            &imports,
            &mach,
            &mut RoundtripReencoder,
            self.inner.0.target,
        )
    }
}

#[cfg(test)]
mod sharing_sink_tests {
    use super::*;
    use alloc::vec::Vec;
    use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
    use portal_solutions_blitz_common::ops::FnData;
    use portal_solutions_blitz_common::wasmparser;

    type Blob = <AArch64Writer<AArch64Label> as RewindableCodeWriter<AArch64Label>>::Blob;

    /// Drive the sharing SysV sink with the two-identical-regions corpus as
    /// raw wasmparser operators, exercising the Operator normalization path.
    fn compile_via_sink(enabled: bool) -> Vec<u8> {
        let w = AArch64Writer::<AArch64Label>::new();
        let mut handle = WaxHandle::new(w, ());
        let cfg = CodeSharingConfig {
            enabled,
            ..CodeSharingConfig::default()
        };
        let mut sink = AArch64SharingSysVWasmSink::<Blob>::new(AArch64Arch::default(), cfg);
        let mut data = FnData::default();
        data.num_params = 0;
        data.num_returns = 0;
        sink.start_fn(&mut handle, 0, &data).unwrap();

        let ops: [Operator<'static>; 12] = [
            Operator::I64Const { value: 40 },
            Operator::I64Const { value: 2 },
            Operator::I64Add,
            Operator::LocalSet { local_index: 0 },
            Operator::F32Const {
                value: wasmparser::Ieee32::from(0.0f32),
            },
            Operator::Drop,
            Operator::I64Const { value: 40 },
            Operator::I64Const { value: 2 },
            Operator::I64Add,
            Operator::LocalSet { local_index: 0 },
            Operator::Return,
            Operator::End,
        ];
        for op in &ops {
            sink.operator(&mut handle, op).unwrap();
        }
        sink.end_body(&mut handle).unwrap();
        handle.writer.into_bytes()
    }

    #[test]
    fn sink_path_outlines_repeated_region() {
        let plain = compile_via_sink(false);
        let shared = compile_via_sink(true);
        assert!(
            shared.len() < plain.len(),
            "shared {} must be smaller than plain {}",
            shared.len(),
            plain.len()
        );
    }
}
