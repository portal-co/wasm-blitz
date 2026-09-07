//! System V AMD64 ABI code generation for the x86-64 backend.
//!
//! This module provides a `SysVWriterExt` trait that generates x86-64 functions
//! following the System V AMD64 ABI, making them directly callable from C and
//! other System V–conforming code.
//!
//! # Calling Convention (System V AMD64)
//!
//! Integer arguments: RDI, RSI, RDX, RCX, R8, R9 (then stack, right-to-left)
//! Return value: RAX (single), RAX + RDX (pair)
//! Frame: `push rbp; mov rbp, rsp; sub rsp, N`
//!
//! Register index mapping in the blitz-x86-64 backend:
//!   Reg(0)=rax  Reg(1)=rcx  Reg(2)=rdx  Reg(3)=rbx
//!   Reg(4)=rsp  Reg(5)=rbp  Reg(6)=rsi  Reg(7)=rdi
//!   Reg(8)=r8   Reg(9)=r9
//!
//! # Design
//!
//! The SysV entry copies register arguments into a standard rbp-relative local
//! frame so that arithmetic, memory, and control-flow handlers (which use RSP as
//! the WASM operand stack) continue to work unchanged.  Only `StartFn`, local
//! variable access, and `Return` differ from the naive backend.  This is the
//! "full rewrite ABI" described in docs/abi.md.

use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
extern crate alloc;

use portal_solutions_asm_x86_64::{
    ConditionCode, RegisterClass, X64Arch,
    out::{
        Writer,
        arg::{ArgKind, MemArg, MemArgKind},
    },
};
use portal_solutions_blitz_common::{
    asm::Reg,
    asm::common::mem::MemorySize,
    ops::{FnData, MachOperator, ProbePlacement, ProbePlan, ProbeTableConfig},
    wasm_encoder::{
        self, FuncType, Instruction,
        reencode::{self as reencode, Reencode},
    },
};

use crate::codegen::ProbeBase;
use crate::naive::WriterExt as NaiveExt;
use crate::{RSP, X64Label};

/// Re-exported so callers don't need a separate `crate::naive::MemBase`
/// import alongside this module's own [`SysVState`]/[`CallAbi`] just to
/// configure memory addressing — see `docs/naive-abi-deprecation.md`.
pub use crate::naive::MemBase;

/// Blitz register number of the SysV **probe-base virtual parameter** (`r11`).
///
/// `r11` is caller-saved and never a SysV positional argument register, so the
/// runtime can pass the per-function probe-table base pointer in it without
/// disturbing the function's real arguments.  The function-entry preamble reads
/// it directly (before frame setup); `StartFn` then spills it to a frame slot so
/// mid-function (loop/block) sites can reload it after `r11` is clobbered.
pub const PROBE_BASE_REG: u8 = 11;

// ---- register shortcuts ----
const RAX: Reg = Reg(0);
const RBP: Reg = Reg(5);
use crate::naive::SCR; // r14 — Static Context Register

/// Integer argument registers in System V AMD64 ABI order.
const ARG_REGS: [Reg; 6] = [
    Reg(7), // rdi
    Reg(6), // rsi
    Reg(2), // rdx
    Reg(1), // rcx
    Reg(8), // r8
    Reg(9), // r9
];

fn mem64(base: Reg, disp: u32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: base,
            size: MemorySize::_64,
        },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        segment: Default::default(),
    }
}

/// One entry in the SysV control-flow stack (no CTX required).
#[derive(Clone, Copy)]
pub enum SysVCtrl {
    /// `loop`: label at the top (already set); `Br` jumps back.
    Loop(usize),
    /// `block`: label at the exit (not yet set); `Br` jumps forward, `End` sets it.
    Block(usize),
    /// `if`: labels `base`, `base+1` (else), `base+2` (after). `else_seen` tracks Else.
    If { base: usize, else_seen: bool },
}

impl SysVCtrl {
    /// The label that a `Br` targeting this block should jump to.
    fn br_target(self) -> usize {
        match self {
            SysVCtrl::Loop(top) => top,
            SysVCtrl::Block(exit) => exit,
            SysVCtrl::If { base, .. } => base + 2,
        }
    }
}

/// State tracker for System V x86-64 code generation.
///
/// The lifetime `'a` is the lifetime of the [`ShardMap`] reference in
/// [`shard`][SysVState::shard]; it has no constraints when `shard` is `None`.
///
/// [`ShardMap`]: portal_solutions_blitz_common::shard::ShardMap
#[derive(Default)]
pub struct SysVState<'a> {
    pub param_count: usize,
    pub ret_count: usize,
    pub local_count: usize,
    pub label_index: usize,
    pub if_stack: Vec<crate::naive::Endable>,
    pub ctrl_stack: Vec<SysVCtrl>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
    /// Probe-table config for this function (`None`/disabled → no probe code
    /// emitted).
    pub probes: Option<ProbeTableConfig>,
    /// Next probe id to assign (function entry = probe 0; each loop/block
    /// consumes the next, in source order).
    pub next_probe_id: u32,
    /// RBP-relative displacement of the spilled probe-base pointer slot, set by
    /// `StartFn` when probes are enabled.  Mid-function sites load the base from
    /// `[RBP + probe_base_disp]`.
    pub probe_base_disp: i32,
    /// Embedder-requested probes at arbitrary instruction indices, in addition
    /// to the function-entry/loop/block probes above.  `None` → zero overhead,
    /// identical codegen to today.
    pub probe_plan: Option<ProbePlan>,
    /// Ordinal index of the next dispatched instruction (0 = the first real
    /// WASM operator after locals), used to look up `probe_plan` entries.
    pub op_index: usize,
    /// Present when sharding is active. SCR (r14) is pushed in the prologue and
    /// popped in all return paths. Cross-shard calls load the target from SCR.
    pub shard: Option<crate::naive::NaiveShardState<'a>>,
    /// How linear-memory load/store addresses are translated. Propagated to the
    /// naive backend that lowers memory ops. Defaults to [`crate::naive::MemBase::Raw`].
    pub mem_base: crate::naive::MemBase,
    /// Per-memory-index base overrides (see [`crate::naive::State::mem_base_by_index`]).
    pub mem_base_by_index: alloc::collections::BTreeMap<u32, crate::naive::MemBase>,
    /// Calling convention for inter-function calls and the function prologue.
    /// Defaults to [`CallAbi::RegSysv`] (the legacy behavior, used by tests where
    /// each function is invoked directly with register arguments).
    pub call_abi: CallAbi,
    /// Number of imported functions (their WASM indices `0..n_imports` are calls
    /// to external `module__name` symbols using the C ABI). Only used in
    /// [`CallAbi::AllStack`].
    pub n_imports: u32,
    /// Param count per WASM function index (imports first, then internal
    /// functions). Only used in [`CallAbi::AllStack`] to marshal call arguments.
    pub call_params: Vec<u32>,
    /// Result count per WASM function index. Only used in [`CallAbi::AllStack`].
    pub call_results: Vec<u32>,
    /// Param count per WASM *type* index — the signature arity used by
    /// `call_indirect`/`return_call_indirect` (the callee is unknown at compile
    /// time, so the type determines the marshalling arity).
    pub sig_params: Vec<u32>,
    /// Result count per WASM type index. Companion to [`Self::sig_params`].
    pub sig_results: Vec<u32>,
    /// Regalloc-backed data-flow state (see `crate::regalloc_core`), shared
    /// with `crate::codegen::RegAllocW`'s `flush()`. `None` until the first
    /// regalloc-backed instruction runs. Every one of this file's own
    /// control-flow/branch/call arms must flush this *before* touching RSP or
    /// a fixed register directly — a label can be reached by more than one
    /// path, so it must always see the canonical (fully-spilled) state
    /// regardless of which branch got there, exactly as in
    /// `blitz-riscv64::naive`'s `emit_control_flow_probe`/`br`/`If`/etc.
    pub regalloc: Option<crate::regalloc_core::X64RegAlloc>,
}

/// Inter-function calling convention selected by the recompiler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallAbi {
    /// Legacy: each function reads its first 6 params from the SysV integer
    /// argument registers (and the rest from the stack); calls are *not*
    /// marshalled (the delegated naive path). Used by direct-invocation tests.
    #[default]
    RegSysv,
    /// Recompiler mode: internal functions pass *all* params on the stack
    /// (`param i` at `[RBP + 16 + i*8]`), so the per-instruction register-file
    /// threading round-trips. Import calls still use the C ABI (register args).
    /// The entry stays C-callable (its params are unused initial register values).
    AllStack,
}

/// Extension trait for generating System V AMD64-compatible functions.
///
/// All arithmetic, memory, and control-flow instructions delegate to the naive
/// backend's `_handle_op`.  Only the function boundary and local variable access
/// are different.
pub trait SysVWriterExt<Context>: Writer<X64Label, Context> + NaiveExt<Context> {
    /// Load local variable `n` from the rbp-relative frame into register `dest`.
    fn sysv_load_local(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        dest: Reg,
        n: usize,
    ) -> Result<(), Self::Error> {
        // Local n is at [rbp - (n+1)*8]
        let disp = 0u32.wrapping_sub(((n as isize + 1) * 8) as u32);
        self.mov(ctx, arch, &dest, &mem64(RBP, disp))
    }

    /// Store `src` into local variable `n` in the rbp-relative frame.
    fn sysv_store_local(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        src: Reg,
        n: usize,
    ) -> Result<(), Self::Error> {
        let disp = 0u32.wrapping_sub(((n as isize + 1) * 8) as u32);
        self.mov(ctx, arch, &mem64(RBP, disp), &src)
    }

    /// Returns the value at top of WASM operand stack (RSP) without popping.
    fn sysv_peek(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        dest: Reg,
    ) -> Result<(), Self::Error> {
        self.mov(ctx, arch, &dest, &mem64(RSP, 0))
    }

    /// Flush `state.regalloc` to the real RSP-based stack, if any values are
    /// currently register-held. Must be called before any control-flow,
    /// branch, call, or local-access code that touches RSP or a fixed
    /// register directly — see [`SysVState::regalloc`]'s doc comment.
    fn sysv_flush_regalloc(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let mut rw = crate::codegen::RegAllocW {
            writer: self,
            ctx,
            arch,
            regalloc: &mut state.regalloc,
        };
        portal_solutions_blitz_codegen::control_flow::ControlFlowWriter::flush(&mut rw)
    }

    /// Emit a control-flow probe (`TailTakeover` binding) for a mid-function
    /// (loop/block) site, consuming the next `probe_id`.  No-op when probes
    /// are disabled.
    ///
    /// The probe-table base is reloaded from the spilled frame slot
    /// (`[RBP + probe_base_disp]`) since the virtual-param register `r11` has
    /// been clobbered by the body.  Uses `RAX` as scratch (free at a block
    /// boundary — operands live on the WASM stack).
    fn sysv_emit_control_flow_probe(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
            let probe_id = state.next_probe_id;
            state.next_probe_id += 1;
            let mut bw = crate::codegen::BlitzW {
                writer: self,
                ctx,
                arch,
                probe_base: ProbeBase::FrameSlot(state.probe_base_disp),
            };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                probe_id,
                0,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Emit any embedder-requested probes (`state.probe_plan`) attached to
    /// the instruction at `state.op_index` with the given `placement`.
    ///
    /// Reuses the same mid-function probe-base addressing
    /// (`ProbeBase::FrameSlot`) and scratch register (`RAX`, free at any op
    /// boundary — operands live on the WASM stack between ops) that
    /// [`sysv_emit_control_flow_probe`](Self::sysv_emit_control_flow_probe)
    /// already establishes. `ProbeMode` (Active/Passive) makes no codegen
    /// difference here: x86-64 has no register allocator to disturb between
    /// ops, so both are already as non-disturbing as possible.
    fn sysv_emit_indexed_probes(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
        placement: ProbePlacement,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let (Some(plan), Some(cfg)) = (state.probe_plan.as_ref(), state.probes) else {
            return Ok(());
        };
        if !cfg.enabled {
            return Ok(());
        }
        let specs: Vec<_> = plan.at(state.op_index, placement).copied().collect();
        for spec in specs {
            let mut bw = crate::codegen::BlitzW {
                writer: self,
                ctx,
                arch,
                probe_base: ProbeBase::FrameSlot(state.probe_base_disp),
            };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw,
                cfg.table_base_off,
                spec.probe_id,
                0,
                spec.binding,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Marshal `arity` operand-stack arguments and call `target`.
    ///
    /// Operand stack on entry (RSP-based): `param_{arity-1}` at `[RSP]`, …,
    /// `param 0` at `[RSP + (arity-1)*8]`. `R15` holds the operand base across the
    /// call (saved in a scratch slot so nested calls can clobber it).
    /// - `is_import`: the target is a C-ABI host import — pass the first ≤6 params
    ///   in the SysV integer argument registers (these imports take ≤6 args).
    /// - otherwise (`CallAbi::AllStack`): write *all* params to `[SP_call + i*8]`
    ///   so the callee's prologue reads `param i` at `[RBP + 16 + i*8]`.
    /// `results` (0 or 1) values are pushed back onto the operand stack from RAX.
    fn sysv_emit_marshalled_call(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        target: X64Label,
        arity: u32,
        results: u32,
        is_import: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let r15 = Reg(15);
        let lit = |v: u64| MemArgKind::NoMem(ArgKind::Lit(v));
        self.mov(ctx, arch, &r15, &RSP)?; // R15 = operand base
        if is_import {
            // System V: first 6 integer args in registers; args 7+ on the stack
            // at [RSP + (i-6)*8]. Reserve a 16-aligned region big enough for the
            // stack args plus one slot to preserve the operand base across the call.
            let stack_args = arity.saturating_sub(6);
            let bytes = ((stack_args as u64) * 8 + 8 + 15) & !15;
            for i in 0..arity.min(6) {
                let disp = (arity - 1 - i) * 8;
                self.mov(ctx, arch, &ARG_REGS[i as usize], &mem64(r15, disp))?;
            }
            self.mov(ctx, arch, &RSP, &r15)?;
            self.mov64(ctx, arch, &RAX, bytes)?;
            self.sub(ctx, arch, &RSP, &RAX)?;
            self.and(ctx, arch, &RSP, &lit((-16i64) as u64))?;
            for i in 6..arity {
                self.mov(ctx, arch, &RAX, &mem64(r15, (arity - 1 - i) * 8))?;
                self.mov(ctx, arch, &mem64(RSP, (i - 6) * 8), &RAX)?;
            }
            self.mov(ctx, arch, &mem64(RSP, stack_args * 8), &r15)?; // save base above args
            self.lea_label(ctx, arch, &RAX, target)?;
            self.call(ctx, arch, &RAX)?;
            self.mov(ctx, arch, &r15, &mem64(RSP, stack_args * 8))?; // restore base
            self.lea(ctx, arch, &RSP, &mem64(r15, arity * 8))?; // pop args
        } else {
            self.mov(ctx, arch, &RSP, &r15)?;
            self.mov64(ctx, arch, &RAX, (arity as u64 + 1) * 8)?;
            self.sub(ctx, arch, &RSP, &RAX)?;
            self.and(ctx, arch, &RSP, &lit((-16i64) as u64))?;
            for i in 0..arity {
                self.mov(ctx, arch, &RAX, &mem64(r15, (arity - 1 - i) * 8))?;
                self.mov(ctx, arch, &mem64(RSP, i * 8), &RAX)?;
            }
            self.mov(ctx, arch, &mem64(RSP, arity * 8), &r15)?; // save base above args
            self.lea_label(ctx, arch, &RAX, target)?;
            self.call(ctx, arch, &RAX)?;
            self.mov(ctx, arch, &r15, &mem64(RSP, arity * 8))?; // restore base
            self.lea(ctx, arch, &RSP, &mem64(r15, arity * 8))?; // pop args
        }
        if results > 0 {
            self.push(ctx, arch, &RAX)?;
        }
        Ok(())
    }

    /// Emit a **true tail call** to an internal `target` with `arity` args.
    ///
    /// Unlike `sysv_emit_marshalled_call` (a real `call`), this reuses the current
    /// frame: the `arity` operand-stack args overwrite our own incoming-arg slots
    /// (`[RBP + 16 + scr_extra + i*8]`), the SysV epilogue restores the caller's
    /// frame leaving RSP at the return address, and a `jmp` transfers control so
    /// the callee returns straight to our caller. The machine stack therefore stays
    /// O(1) across a `return_call` chain. Requires `arity <= param_count` (checked
    /// by the caller) so the overwrite stays within our incoming-arg region.
    fn sysv_emit_tail_call(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        target: X64Label,
        arity: u32,
        shard: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let r15 = Reg(15);
        let scr_extra: u32 = if shard { 8 } else { 0 };
        self.mov(ctx, arch, &r15, &RSP)?; // operand base (args are below RBP)
        for i in 0..arity {
            self.mov(ctx, arch, &RAX, &mem64(r15, (arity - 1 - i) * 8))?;
            self.mov(ctx, arch, &mem64(RBP, 16 + scr_extra + i * 8), &RAX)?;
        }
        // Restore caller frame; RSP ends at the return address.
        self.mov(ctx, arch, &RSP, &RBP)?;
        if shard {
            self.pop(ctx, arch, &SCR)?;
        }
        self.pop(ctx, arch, &RBP)?;
        self.jmp_label(ctx, arch, target)
    }

    /// AllStack marshalled call to a **register** target (for `call_indirect`):
    /// all `arity` args go on the stack and the callee is `call target`. `target`
    /// must be a register the marshalling does not clobber (e.g. R10/R11).
    fn sysv_emit_marshalled_call_reg(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        target: Reg,
        arity: u32,
        results: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let r15 = Reg(15);
        let lit = |v: u64| MemArgKind::NoMem(ArgKind::Lit(v));
        self.mov(ctx, arch, &r15, &RSP)?; // operand base
        self.mov64(ctx, arch, &RAX, (arity as u64 + 1) * 8)?;
        self.sub(ctx, arch, &RSP, &RAX)?;
        self.and(ctx, arch, &RSP, &lit((-16i64) as u64))?;
        for i in 0..arity {
            self.mov(ctx, arch, &RAX, &mem64(r15, (arity - 1 - i) * 8))?;
            self.mov(ctx, arch, &mem64(RSP, i * 8), &RAX)?;
        }
        self.mov(ctx, arch, &mem64(RSP, arity * 8), &r15)?; // save base above args
        self.call(ctx, arch, &target)?;
        self.mov(ctx, arch, &r15, &mem64(RSP, arity * 8))?; // restore base
        self.lea(ctx, arch, &RSP, &mem64(r15, arity * 8))?; // pop args
        if results > 0 {
            self.push(ctx, arch, &RAX)?;
        }
        Ok(())
    }

    /// True tail call to a **register** target (`return_call_indirect`); mirrors
    /// `sysv_emit_tail_call` but jumps to a register. Requires `arity <= param_count`.
    fn sysv_emit_tail_call_reg(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        target: Reg,
        arity: u32,
        shard: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let r15 = Reg(15);
        let scr_extra: u32 = if shard { 8 } else { 0 };
        self.mov(ctx, arch, &r15, &RSP)?;
        for i in 0..arity {
            self.mov(ctx, arch, &RAX, &mem64(r15, (arity - 1 - i) * 8))?;
            self.mov(ctx, arch, &mem64(RBP, 16 + scr_extra + i * 8), &RAX)?;
        }
        self.mov(ctx, arch, &RSP, &RBP)?;
        if shard {
            self.pop(ctx, arch, &SCR)?;
        }
        self.pop(ctx, arch, &RBP)?;
        self.jmp(ctx, arch, &target)
    }

    /// Pop the `call_indirect` table index and load the function pointer from
    /// `__wasm_table[index]` into R10 (a register the marshalling preserves).
    fn sysv_load_indirect_target(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.pop(ctx, arch, &Reg(10))?; // table index → R10
        self.lea_label(
            ctx,
            arch,
            &Reg(11),
            X64Label::External {
                name: "__wasm_table".into(),
            },
        )?;
        // R10 := [R11 + R10*8]
        self.mov(
            ctx,
            arch,
            &Reg(10),
            &MemArgKind::Mem {
                base: Reg(11),
                offset: Some((Reg(10), 8)),
                disp: 0,
                size: MemorySize::_64,
                reg_class: RegisterClass::Gpr,
                segment: Default::default(),
            },
        )
    }

    /// Emit the SysV epilogue (restore frame, `ret`) — shared by `Return`,
    /// function-level `End`, and every `Br`/`BrIf` targeting the function's
    /// outermost block.
    ///
    /// WASM multi-value results are pushed on the operand stack in
    /// *declaration* order: result 0 is pushed first (deepest), result
    /// `n-1` last (on top) — see the WASM spec. The SysV ABI can carry at
    /// most two results back through registers (RAX, RDX), so this reads
    /// result 0 and result 1 directly by their known stack offset from
    /// `RSP` (`(n-1)*8` and `(n-2)*8` respectively) instead of just
    /// popping the top two values in stack order — popping in stack order
    /// would put the *last*-declared result in RAX instead of the first,
    /// which is wrong whenever a caller outside this WASM module (e.g. a C
    /// entry shim reading a recompiled guest's `main` return value) reads
    /// RAX expecting "result 0". Any results beyond index 1 are simply
    /// skipped (there is no native register for them); `mov rsp, rbp`
    /// below discards the whole operand stack unconditionally, so leaving
    /// them unread is not a leak.
    fn sysv_emit_epilogue(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // Every result this reads comes straight off [RSP]; flush first so a
        // regalloc-held result isn't silently skipped.
        self.sysv_flush_regalloc(ctx, arch, state)?;
        let n = state.ret_count;
        if n > 0 {
            self.mov(ctx, arch, &RAX, &mem64(RSP, ((n - 1) * 8) as u32))?;
        }
        if n > 1 {
            self.mov(ctx, arch, &Reg(2), &mem64(RSP, ((n - 2) * 8) as u32))?;
        }
        self.mov(ctx, arch, &RSP, &RBP)?;
        if state.shard.is_some() {
            self.pop(ctx, arch, &SCR)?;
        }
        self.pop(ctx, arch, &RBP)?;
        self.ret(ctx, arch)
    }

    /// Resolve a call/return_call target index to its label + (arity, results, is_import).
    fn sysv_call_target(
        state: &SysVState<'_>,
        func_imports: &[(&str, &str)],
        idx: u32,
    ) -> (X64Label, u32, u32, bool) {
        let widx = idx as usize;
        let is_import = idx < state.n_imports;
        let arity = state.call_params.get(widx).copied().unwrap_or(0);
        let results = state.call_results.get(widx).copied().unwrap_or(0);
        let label = if is_import {
            let (m, n) = func_imports[widx];
            X64Label::External {
                name: alloc::format!("{m}__{n}"),
            }
        } else {
            X64Label::Func {
                r#fn: idx - state.n_imports,
            }
        };
        (label, arity, results, is_import)
    }

    /// Handle an instruction using the SysV ABI (overrides local access and Return).
    fn sysv_handle_insn(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
        func_imports: &[(&str, &str)],
        op: &Instruction<'_>,
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // Redirect body-skip jump if needed (same as naive _handle_op preamble)
        if target != state.body {
            // First-instruction guard: see naive.rs.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                let skip_idx = *state.body_labels.entry(state.body).or_insert_with(|| {
                    state.label_index += 1;
                    state.label_index - 1
                });
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: skip_idx })?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, X64Label::Indexed { idx })?;
                }
            }
        }

        match op {
            // ---- local access (SysV rbp-relative frame) ----
            // LocalGet/LocalSet fall through to the "everything else" arm below,
            // which tries the regalloc-backed core first.
            Instruction::LocalTee(n) => {
                // Peek (don't pop), store. Flush first: the regalloc may be
                // holding the top-of-stack value in a register rather than
                // where `sysv_peek`'s raw `[RSP]` read expects it.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                self.sysv_peek(ctx, arch, RAX)?;
                self.sysv_store_local(ctx, arch, RAX, *n as usize)
            }

            // ---- Calls: marshal operand-stack args per the selected CallAbi ----
            Instruction::Call(idx) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                self.sysv_emit_marshalled_call(ctx, arch, label, arity, results, is_import)
            }
            Instruction::ReturnCall(idx) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let (label, arity, results, is_import) =
                    Self::sysv_call_target(state, func_imports, *idx);
                if is_import || arity as usize > state.param_count {
                    // Fake tail: a real call then the SysV epilogue. Required for C-ABI
                    // imports (proper alignment/return), and a safe fallback when the
                    // callee wants more args than our incoming-arg frame can hold.
                    self.sysv_emit_marshalled_call(ctx, arch, label, arity, results, is_import)?;
                    self.sysv_emit_epilogue(ctx, arch, state)
                } else {
                    // True tail call to an internal function: O(1) machine stack for
                    // long `return_call` chains (speet emits one cell per guest insn).
                    self.sysv_emit_tail_call(ctx, arch, label, arity, state.shard.is_some())
                }
            }

            // ---- Indirect calls: target = __wasm_table[index] ----
            Instruction::CallIndirect { type_index, .. } if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.sysv_load_indirect_target(ctx, arch)?; // index → R10 (fn ptr)
                self.sysv_emit_marshalled_call_reg(ctx, arch, Reg(10), arity, results)
            }
            Instruction::ReturnCallIndirect { type_index, .. }
                if state.call_abi == CallAbi::AllStack =>
            {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.sysv_load_indirect_target(ctx, arch)?; // index → R10 (fn ptr)
                if arity as usize > state.param_count {
                    self.sysv_emit_marshalled_call_reg(ctx, arch, Reg(10), arity, results)?;
                    self.sysv_emit_epilogue(ctx, arch, state)
                } else {
                    self.sysv_emit_tail_call_reg(ctx, arch, Reg(10), arity, state.shard.is_some())
                }
            }

            // ---- Function references: a "ref" on the operand stack is just the
            // raw native address of the target, exactly what `lea_label`
            // materializes for a direct `Call`/`ReturnCall` target — so
            // `ref.func` is `lea_label` + push, and `call_ref`/
            // `return_call_ref` are `call_indirect`/`return_call_indirect`
            // with the table-load step (`sysv_load_indirect_target`)
            // replaced by a plain pop, since the popped value already *is*
            // the target address rather than an index to look up. No WASM
            // table (typed or otherwise) exists at the machine-code level
            // in this backend — an embedder splices a compiled function's
            // entry address in directly (see os-jit-core's `JitArtifact::
            // Native`), so a typed funcref table's runtime type check has
            // no native equivalent to lower either; this backend accepts
            // any `(ref $t)`/`(ref null $t)` blindly, on the same trust
            // model already implied by `call_indirect`'s bare `Reg(10)`
            // load skipping any table bounds/type check. ----
            Instruction::RefFunc(idx) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let (label, ..) = Self::sysv_call_target(state, func_imports, *idx);
                self.lea_label(ctx, arch, &RAX, label)?;
                self.push(ctx, arch, &RAX)
            }
            Instruction::CallRef(type_index) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.pop(ctx, arch, &Reg(10))?; // funcref value (raw target address) → R10
                self.sysv_emit_marshalled_call_reg(ctx, arch, Reg(10), arity, results)
            }
            Instruction::ReturnCallRef(type_index) if state.call_abi == CallAbi::AllStack => {
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let ty = *type_index as usize;
                let arity = state.sig_params.get(ty).copied().unwrap_or(0);
                let results = state.sig_results.get(ty).copied().unwrap_or(0);
                self.pop(ctx, arch, &Reg(10))?;
                if arity as usize > state.param_count {
                    self.sysv_emit_marshalled_call_reg(ctx, arch, Reg(10), arity, results)?;
                    self.sysv_emit_epilogue(ctx, arch, state)
                } else {
                    self.sysv_emit_tail_call_reg(ctx, arch, Reg(10), arity, state.shard.is_some())
                }
            }

            // ---- Return: always emit SysV epilogue regardless of block depth ----
            Instruction::Return => self.sysv_emit_epilogue(ctx, arch, state),
            // ---- Function-level End (empty ctrl_stack) ----
            Instruction::End if state.if_stack.is_empty() => {
                self.sysv_emit_epilogue(ctx, arch, state)
            }

            // ---- Control flow — CTX-free (no naive delegation) ----
            Instruction::Loop(_) => {
                let i = state.label_index;
                state.label_index += 1;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                state.if_stack.push(crate::naive::Endable::Br);
                state.ctrl_stack.push(SysVCtrl::Loop(i));
                // Trace site after the loop-back label so it runs each iteration.
                self.sysv_emit_control_flow_probe(ctx, arch, state)
            }
            Instruction::Block(_) => {
                let i = state.label_index;
                state.label_index += 1;
                state.if_stack.push(crate::naive::Endable::Br);
                state.ctrl_stack.push(SysVCtrl::Block(i));
                self.sysv_emit_control_flow_probe(ctx, arch, state)
            }
            Instruction::If(_) => {
                let i = state.label_index;
                state.label_index += 3;
                // Flush first: the condition may be regalloc-held rather than at [RSP].
                self.sysv_flush_regalloc(ctx, arch, state)?;
                self.pop(ctx, arch, &RAX)?;
                self.cmp0(ctx, arch, &RAX)?;
                self.jcc_label(
                    ctx,
                    arch,
                    ConditionCode::E,
                    X64Label::Indexed { idx: i + 1 },
                )?;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i })?;
                state.if_stack.push(crate::naive::Endable::If { idx: i });
                state.ctrl_stack.push(SysVCtrl::If {
                    base: i,
                    else_seen: false,
                });
                Ok(())
            }
            Instruction::Else => {
                // Flush: the then-arm may have left values regalloc-held, but
                // the else-arm must start from the same (fully-spilled) state
                // the `if` itself started from, not inherit the then-arm's
                // now-irrelevant register assignments.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let Some(SysVCtrl::If { base: i, else_seen }) = state.ctrl_stack.last_mut() else {
                    return Ok(());
                };
                let i = *i;
                *else_seen = true;
                self.jmp_label(ctx, arch, X64Label::Indexed { idx: i + 2 })?;
                self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })
            }
            Instruction::End if !state.if_stack.is_empty() => {
                // Flush unconditionally before resolving any frame kind (matching
                // blitz-riscv64::naive's End arm) — a label may be reached by
                // more than one path and must always see canonical state.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                if state.ctrl_stack.is_empty() {
                    // TryTable's End: ctrl_stack has no entry for it (naive tracks it in
                    // if_stack only). Delegate to naive so TryTable can emit its dispatch stub.
                    let other = Instruction::End;
                    let mut naive_state = crate::naive::State {
                        local_count: state.local_count,
                        num_returns: state.ret_count,
                        control_depth: 0,
                        label_index: state.label_index,
                        if_stack: core::mem::take(&mut state.if_stack),
                        body: state.body,
                        body_labels: core::mem::take(&mut state.body_labels),
                        probes: None,
                        next_probe_id: 0,
                        shard: state.shard,
                        mem_base: state.mem_base,
                        mem_base_by_index: state.mem_base_by_index.clone(),
                    };
                    let result = self._handle_op(
                        ctx,
                        arch,
                        &mut naive_state,
                        func_imports,
                        &[],
                        &[],
                        &other,
                        target,
                    );
                    state.label_index = naive_state.label_index;
                    state.if_stack = naive_state.if_stack;
                    state.body = naive_state.body;
                    state.body_labels = naive_state.body_labels;
                    return result;
                }
                state.if_stack.pop();
                let ctrl = state.ctrl_stack.pop().unwrap();
                match ctrl {
                    SysVCtrl::Loop(_) => Ok(()),
                    SysVCtrl::Block(exit) => {
                        self.set_label(ctx, arch, X64Label::Indexed { idx: exit })
                    }
                    SysVCtrl::If { base: i, else_seen } => {
                        if !else_seen {
                            self.set_label(ctx, arch, X64Label::Indexed { idx: i + 1 })?;
                        }
                        self.set_label(ctx, arch, X64Label::Indexed { idx: i + 2 })
                    }
                }
            }
            Instruction::Br(n) => {
                // Flush: this label (or the function epilogue) may be reached
                // from other paths and must see canonical state.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let n = *n as usize;
                if let Some(ctrl) = state
                    .ctrl_stack
                    .len()
                    .checked_sub(n + 1)
                    .and_then(|idx| state.ctrl_stack.get(idx))
                    .copied()
                {
                    self.jmp_label(
                        ctx,
                        arch,
                        X64Label::Indexed {
                            idx: ctrl.br_target(),
                        },
                    )
                } else {
                    // Br targeting the function block = return
                    self.sysv_emit_epilogue(ctx, arch, state)
                }
            }
            Instruction::BrIf(n) => {
                // Flush first: the condition may be regalloc-held rather than at [RSP].
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let n = *n as usize;
                let skip = state.label_index;
                state.label_index += 1;
                self.pop(ctx, arch, &RAX)?;
                self.cmp0(ctx, arch, &RAX)?;
                if let Some(ctrl) = state
                    .ctrl_stack
                    .len()
                    .checked_sub(n + 1)
                    .and_then(|idx| state.ctrl_stack.get(idx))
                    .copied()
                {
                    self.jcc_label(
                        ctx,
                        arch,
                        ConditionCode::NE,
                        X64Label::Indexed {
                            idx: ctrl.br_target(),
                        },
                    )?;
                } else {
                    // BrIf targeting the function block = conditional return
                    self.jcc_label(ctx, arch, ConditionCode::E, X64Label::Indexed { idx: skip })?;
                    self.sysv_emit_epilogue(ctx, arch, state)?;
                }
                self.set_label(ctx, arch, X64Label::Indexed { idx: skip })
            }
            Instruction::BrTable(targets, default) => {
                // Flush first: the selector may be regalloc-held rather than at [RSP].
                self.sysv_flush_regalloc(ctx, arch, state)?;
                // Pop selector once; for each target: if selector==0 branch, else decrement.
                self.pop(ctx, arch, &RAX)?;
                let has_shard = state.shard.is_some();
                let ctrl = &state.ctrl_stack;
                macro_rules! do_br {
                    ($depth:expr) => {{
                        let n = $depth as usize;
                        let tgt = ctrl
                            .len()
                            .checked_sub(n + 1)
                            .and_then(|i| ctrl.get(i))
                            .map(|c| c.br_target());
                        if let Some(lbl) = tgt {
                            self.jmp_label(ctx, arch, X64Label::Indexed { idx: lbl })?;
                        } else {
                            self.mov(ctx, arch, &RSP, &RBP)?;
                            if has_shard {
                                self.pop(ctx, arch, &SCR)?;
                            }
                            self.pop(ctx, arch, &RBP)?;
                            self.ret(ctx, arch)?;
                        }
                    }};
                }
                for (arm_idx, &depth) in targets.iter().enumerate() {
                    let skip = state.label_index;
                    state.label_index += 1;
                    self.cmp0(ctx, arch, &RAX)?;
                    self.jcc_label(
                        ctx,
                        arch,
                        ConditionCode::NE,
                        X64Label::Indexed { idx: skip },
                    )?;
                    do_br!(depth);
                    self.set_label(ctx, arch, X64Label::Indexed { idx: skip })?;
                    if arm_idx + 1 < targets.len() {
                        self.lea(
                            ctx,
                            arch,
                            &RAX,
                            &MemArgKind::Mem {
                                base: RAX,
                                offset: None,
                                disp: 0u32.wrapping_sub(1),
                                size: MemorySize::_64,
                                reg_class: RegisterClass::Gpr,
                                segment: Default::default(),
                            },
                        )?;
                    }
                }
                do_br!(*default);
                Ok(())
            }

            // ---- Everything else: try the regalloc-backed core first, else
            // fall back to naive's pure-stack _handle_op (no CTX ops in these
            // paths). LocalGet/LocalSet land here and are handled by the core.
            other => {
                if crate::regalloc_core::handle_op(self, ctx, arch, &mut state.regalloc, other)? {
                    return Ok(());
                }
                // Not covered by the regalloc core: flush so naive's raw-stack
                // path sees canonical (fully-spilled) state, then delegate.
                self.sysv_flush_regalloc(ctx, arch, state)?;
                let mut naive_state = crate::naive::State {
                    local_count: state.local_count,
                    num_returns: state.ret_count,
                    control_depth: 0,
                    label_index: state.label_index,
                    if_stack: core::mem::take(&mut state.if_stack),
                    body: state.body,
                    body_labels: core::mem::take(&mut state.body_labels),
                    probes: None,
                    next_probe_id: 0,
                    // Propagate shard state so cross-shard Call dispatch works.
                    shard: state.shard,
                    mem_base: state.mem_base,
                    mem_base_by_index: state.mem_base_by_index.clone(),
                };
                let result = self._handle_op(
                    ctx,
                    arch,
                    &mut naive_state,
                    func_imports,
                    &[],
                    &[],
                    other,
                    target,
                );
                state.label_index = naive_state.label_index;
                state.if_stack = naive_state.if_stack;
                state.body = naive_state.body;
                state.body_labels = naive_state.body_labels;
                result
            }
        }
    }

    /// Handle a `MachOperator` using the SysV ABI.
    fn sysv_handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: X64Arch,
        state: &mut SysVState<'_>,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        use portal_solutions_blitz_common::wasm_encoder;

        match op {
            MachOperator::StartFn { id, data } => {
                state.param_count = data.num_params;
                state.ret_count = data.num_returns;
                state.local_count = data.num_params;
                state.probes = data.probes;
                state.next_probe_id = 1; // probe 0 is the function entry below
                state.probe_plan = data.probe_plan.clone();
                state.op_index = 0;
                // Register allocation is per-function state, unlike label_index
                // (which must stay monotonic across the whole compilation unit
                // for global label uniqueness) — reset it here or a multi-function
                // compilation exhausts the 32-register frame after enough functions.
                state.regalloc = None;

                self.set_label(
                    ctx,
                    arch,
                    X64Label::Indexed {
                        idx: *id as usize | (1 << 28),
                    },
                )
                .map_err(Err::from)?;
                // Also publish the `Func{id}` label at the entry so that
                // inter-function calls (which the delegated naive Call/ReturnCall
                // path emits as `Func{idx}`) resolve to this C-ABI entry.
                self.set_label(ctx, arch, X64Label::Func { r#fn: *id })
                    .map_err(Err::from)?;

                // Function-entry probe (probe 0): the probe-table base arrives in
                // the virtual-param register r11; read it directly so the
                // specialization tail-jump fires before any frame is built (SysV
                // arg registers RDI/RSI/… still intact).  Scratch = RAX.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW {
                        writer: self,
                        ctx,
                        arch,
                        probe_base: ProbeBase::Reg(PROBE_BASE_REG),
                    };
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw,
                        cfg.table_base_off,
                        0,
                        0,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    )
                    .map_err(Err::from)?;
                }

                self.push(ctx, arch, &RBP).map_err(Err::from)?;
                // SCR save: push r14 (callee-saved) if sharding is active so that
                // imported functions that follow SysV don't clobber the shard table.
                if state.shard.is_some() {
                    self.push(ctx, arch, &SCR).map_err(Err::from)?;
                }
                self.mov(ctx, arch, &RBP, &RSP).map_err(Err::from)?;

                // One extra slot (at the frame bottom) holds the spilled probe
                // base for mid-function sites; +16 headroom is the pre-existing
                // local budget.
                let slots = data.num_params + 16 + 1;
                let frame_sz = (slots * 8 + 15) & !15;
                self.mov64(ctx, arch, &RAX, frame_sz as u64)
                    .map_err(Err::from)?;
                self.sub(ctx, arch, &RSP, &RAX).map_err(Err::from)?;

                // Spill the virtual-param base (r11) to the bottom frame slot.
                if data.probes.as_ref().map(|c| c.enabled).unwrap_or(false) {
                    let disp = 0u32.wrapping_sub(frame_sz as u32);
                    state.probe_base_disp = -(frame_sz as i32);
                    self.mov(ctx, arch, &mem64(RBP, disp), &Reg(PROBE_BASE_REG))
                        .map_err(Err::from)?;
                }

                let scr_extra: u32 = if state.shard.is_some() { 8 } else { 0 };
                match state.call_abi {
                    CallAbi::RegSysv => {
                        for i in 0..data.num_params.min(6) {
                            self.sysv_store_local(ctx, arch, ARG_REGS[i], i)
                                .map_err(Err::from)?;
                        }
                        // Params 7+ (index >= 6) are passed by the caller on the
                        // stack, above the saved RBP (and saved SCR) and the
                        // return address: incoming arg `i` at [RBP+16+scr_extra+(i-6)*8].
                        for i in 6..data.num_params {
                            let src_disp = 16 + scr_extra + ((i - 6) as u32) * 8;
                            self.mov(ctx, arch, &RAX, &mem64(RBP, src_disp))
                                .map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, RAX, i)
                                .map_err(Err::from)?;
                        }
                    }
                    CallAbi::AllStack => {
                        // All params arrive on the stack; param `i` at
                        // [RBP + 16 + scr_extra + i*8].
                        for i in 0..data.num_params {
                            let src_disp = 16 + scr_extra + (i as u32) * 8;
                            self.mov(ctx, arch, &RAX, &mem64(RBP, src_disp))
                                .map_err(Err::from)?;
                            self.sysv_store_local(ctx, arch, RAX, i)
                                .map_err(Err::from)?;
                        }
                    }
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov64(ctx, arch, &RAX, 0).map_err(Err::from)?;
                for _ in 0..*count {
                    self.sysv_store_local(ctx, arch, RAX, state.local_count)
                        .map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }

            MachOperator::StartBody | MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before)
                    .map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, insn, target)
                    .map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After)
                    .map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator {
                op: Some(op_wasm), ..
            } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::Before)
                    .map_err(Err::from)?;
                self.sysv_handle_insn(ctx, arch, state, func_imports, &insn, target)
                    .map_err(Err::from)?;
                self.sysv_emit_indexed_probes(ctx, arch, state, ProbePlacement::After)
                    .map_err(Err::from)?;
                state.op_index += 1;
                Ok(())
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()),
        }
    }
}

impl<T: Writer<X64Label, Context> + NaiveExt<Context> + ?Sized, Context> SysVWriterExt<Context>
    for T
{
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sysv_manyarg_tests {
    use super::*;
    use portal_solutions_asm_x86_64::out::iced::IcedWriter;

    /// Drive `sysv_emit_marshalled_call` and return (code bytes, target relocs).
    fn marshal(
        arity: u32,
        results: u32,
        is_import: bool,
        target: X64Label,
    ) -> (Vec<u8>, Vec<X64Label>) {
        let mut w: IcedWriter<X64Label> = IcedWriter::new(0);
        let mut ctx = ();
        SysVWriterExt::sysv_emit_marshalled_call(
            &mut w,
            &mut ctx,
            X64Arch::default(),
            target,
            arity,
            results,
            is_import,
        )
        .unwrap();
        let (bytes, _labels, relocs) = w.into_parts_with_relocs();
        (bytes, relocs.into_iter().map(|r| r.label).collect())
    }

    #[test]
    fn internal_call_marshals_all_args_and_grows() {
        let (small, rel) = marshal(4, 1, false, X64Label::Func { r#fn: 0 });
        let (big, _) = marshal(20, 1, false, X64Label::Func { r#fn: 0 });
        // Exactly one relocation (the call target), pointing at Func{0}.
        assert_eq!(rel.len(), 1);
        assert!(matches!(&rel[0], X64Label::Func { r#fn } if *r#fn == 0));
        // AllStack marshals every arg to the stack, so 20 args emit more than 4.
        assert!(big.len() > small.len());
    }

    #[test]
    fn import_spills_args_beyond_six() {
        // 6-arg import: all in registers. 9-arg import: 3 spilled to the stack.
        let (six, _) = marshal(
            6,
            1,
            true,
            X64Label::External {
                name: "env__f".into(),
            },
        );
        let (nine, rel) = marshal(
            9,
            1,
            true,
            X64Label::External {
                name: "env__f".into(),
            },
        );
        assert!(
            nine.len() > six.len(),
            "args 7+ must add stack-store instructions"
        );
        assert!(
            rel.iter()
                .any(|l| matches!(l, X64Label::External { name } if name.as_str() == "env__f"))
        );
    }
}
