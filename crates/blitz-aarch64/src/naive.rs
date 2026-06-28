//! Naive AArch64 code generation — mirrors the RISC-V 64 backend structure.
//!
//! Calling convention: blitz WASM ABI (see docs/abi.md).
//! - SP  = Reg(31)/sp  — WASM operand stack
//! - FP  = Reg(29)/x29 — frame pointer
//! - LR  = Reg(30)/x30 — link register

use alloc::{collections::btree_map::BTreeMap, vec::Vec};
extern crate alloc;

use portal_solutions_asm_aarch64::{
    out::{
        arg::{AddressingMode, ArgKind, MemArgKind},
        Writer, WriterCore,
    },
    AArch64Arch, ConditionCode, RegisterClass,
};
use portal_pc_asm_common::types::mem::MemorySize;
#[allow(unused_imports)]
use portal_solutions_asm_aarch64::out::arg::MemArg;
use portal_solutions_blitz_common::{
    asm::Reg,
    ops::{FnData, MachOperator, ProbePlan, ProbeTableConfig},
    shard::{CallTarget, SecondCtxConfig},
    wasm_encoder::{self, Catch, FuncType, Instruction, reencode::{self as reencode, Reencode}},
    wasmparser::Operator,
};

use crate::AArch64Label;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Sharding state for AArch64 functions — same design as x86-64.
#[derive(Clone, Copy)]
pub struct NaiveShardState<'a> {
    pub config: SecondCtxConfig,
    pub current_shard: usize,
    pub imports_len: u32,
    pub map: &'a (dyn portal_solutions_blitz_common::shard::ShardMap + 'a),
}

impl<'a> NaiveShardState<'a> {
    pub fn new(
        config: SecondCtxConfig,
        current_shard: usize,
        imports_len: u32,
        map: &'a (dyn portal_solutions_blitz_common::shard::ShardMap + 'a),
    ) -> Self {
        Self { config, current_shard, imports_len, map }
    }

    pub fn call_target(&self, callee_fn: u32) -> CallTarget {
        if callee_fn < self.imports_len {
            return CallTarget::Import;
        }
        let callee_shard = self.map.shard_for(callee_fn);
        if callee_shard == self.current_shard {
            CallTarget::Local
        } else {
            CallTarget::CrossShard { table_slot: callee_fn }
        }
    }
}

/// Code-generation state for an AArch64 function.
///
/// The lifetime `'a` is the lifetime of the [`ShardMap`] reference in
/// [`shard`][State::shard]; it is unconstrained when `shard` is `None`.
///
/// [`ShardMap`]: portal_solutions_blitz_common::shard::ShardMap
/// How WASM linear-memory addresses are translated to host addresses by the
/// load/store lowering. See the x86-64 `naive::MemBase` for the rationale.
///
/// [`MemBase::Raw`] (default) uses the WASM address directly as a host pointer;
/// [`MemBase::WasmMemSymbol`] computes `__wasm_mem + (uint32_t)addr`, matching
/// the C backend, for ordinary OS processes where linear memory cannot be mapped
/// at a fixed virtual address. The full-binary recompiler selects symbol mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemBase {
    /// WASM address used directly as a host pointer (default; legacy behavior).
    #[default]
    Raw,
    /// Address as `__wasm_mem + (uint32_t)addr`, matching the C backend.
    WasmMemSymbol,
}

#[derive(Default)]
pub struct State<'a> {
    pub local_count: usize,
    /// Incoming parameter count (set by the SysV `StartFn`; never grown by
    /// `Local`). Used to bound true-tail-call argument overwrites.
    pub param_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub label_index: usize,
    /// Control-flow stack: each entry is the label pair (break_lbl, else_lbl_or_0)
    pub if_stack: Vec<Endable>,
    pub body: u32,
    pub body_labels: BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` so the function-entry probe is
    /// emitted after the function-entry label, ensuring every call-path is
    /// instrumented.
    pub probes: Option<ProbeTableConfig>,
    /// Next probe id to assign (function entry = probe 0; each loop/block
    /// consumes the next).  See `emit_probe_site`.
    pub next_probe_id: u32,
    /// How mid-function probe sites reach the runtime probe-table base.  The
    /// NaiveAbi keeps the default (CTX-relative); the SysV ABI sets this to a
    /// frame slot after spilling its virtual-param base register.
    pub probe_base: crate::codegen::ProbeBase,
    /// Embedder-requested probes at arbitrary instruction indices, in addition
    /// to the function-entry/loop/block probes above.  `None` → zero overhead,
    /// identical codegen to today.
    pub probe_plan: Option<ProbePlan>,
    /// Ordinal index of the next dispatched instruction (0 = the first real
    /// WASM operator after locals), used to look up `probe_plan` entries.
    pub op_index: usize,
    /// Present when sharding is active. SCR (X27) is pushed in the prologue
    /// and popped before return.
    pub shard: Option<NaiveShardState<'a>>,
    /// How linear-memory load/store addresses are translated. Defaults to
    /// [`MemBase::Raw`] (legacy raw-pointer behavior).
    pub mem_base: MemBase,
    /// Inter-function calling convention for the AAPCS64 (SysV) path. Defaults
    /// to [`CallAbi::RegSysv`] (legacy; calls delegate to the naive path). Set
    /// to [`CallAbi::AllStack`] by the recompiler to marshal *all* arguments per
    /// AAPCS64 (X0–X7 then stack) so the full guest register file round-trips.
    pub call_abi: CallAbi,
    /// Number of imported functions: WASM indices `0..n_imports` are calls to
    /// external `module__name` symbols. Only used in [`CallAbi::AllStack`].
    pub n_imports: u32,
    /// Param count per WASM function index (imports first, then internal
    /// functions). Only used in [`CallAbi::AllStack`] to marshal call arguments.
    pub call_params: Vec<u32>,
    /// Result count per WASM function index. Only used in [`CallAbi::AllStack`].
    pub call_results: Vec<u32>,
    /// Param count per WASM *type* index — the arity used by `call_indirect`.
    pub sig_params: Vec<u32>,
    /// Result count per WASM type index. Companion to [`Self::sig_params`].
    pub sig_results: Vec<u32>,
}

/// Inter-function calling convention selected by the recompiler for the AAPCS64
/// (SysV) backend. Mirrors `blitz-x86-64`'s `sysv::CallAbi`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallAbi {
    /// Legacy: `Call`/`ReturnCall` delegate to the naive path (register args,
    /// no stack spill). Used by direct-invocation tests.
    #[default]
    RegSysv,
    /// Recompiler mode: marshal *all* arguments per AAPCS64 (the first 8 in
    /// X0–X7, the rest on the outgoing stack), matching the SysV prologue which
    /// reads `param i` (i≥8) at `[FP + 16 + i_stack*8]`. Import calls use the
    /// same AAPCS64 convention (they are ordinary C functions).
    AllStack,
}

/// Represents a control-flow frame.
#[derive(Clone)]
pub enum Endable {
    Block { end_lbl: AArch64Label },
    Loop  { head_lbl: AArch64Label },
    If    { else_lbl: AArch64Label, end_lbl: AArch64Label },
    TryTable {
        end_lbl: AArch64Label,
        dispatch_lbl: AArch64Label,
        catches: alloc::boxed::Box<[Catch]>,
    },
}

// ---------------------------------------------------------------------------
// Register helpers
// ---------------------------------------------------------------------------

/// WASM stack pointer = hardware SP.
const SP: Reg = Reg(31);
/// Frame pointer.
const FP: Reg = Reg(29);
/// Link register.
const LR: Reg = Reg(30);
/// Static Context Register (SCR) — X27 on AArch64.
///
/// Callee-saved register used to hold the cross-shard function-pointer table
/// pointer when sharding is active. See `docs/second-context-register.md`.
pub const SCR: Reg = Reg(27);
/// Scratch registers (caller-saved / temporaries).
const T0: Reg = Reg(9);
const T1: Reg = Reg(10);
const T2: Reg = Reg(11);
/// FP scratch registers (V0–V2 / D0–D2 / S0–S2). FP values ride as raw bits on
/// the GP operand stack; these hold them only transiently inside a single FP op,
/// so they never collide with the AllStack register file (which lives in memory).
const FD0: Reg = Reg(0);
const FD1: Reg = Reg(1);
const FD2: Reg = Reg(2);

fn reg(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_64 })
}
fn reg32(r: Reg) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size: MemorySize::_32 })
}
fn reg_sz(r: Reg, size: MemorySize) -> MemArgKind {
    MemArgKind::NoMem(ArgKind::Reg { reg: r, size })
}
/// Access width in bits for a sub-word memory size (`_8`/`_16`/`_32`).
fn sz_bits(size: MemorySize) -> u64 {
    match size {
        MemorySize::_8 => 8,
        MemorySize::_16 => 16,
        MemorySize::_32 => 32,
        _ => 64,
    }
}
fn mem_base_disp(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::Offset,
    }
}
fn mem_pre(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PreIndex,
    }
}
fn mem_post(base: Reg, disp: i32) -> MemArgKind {
    MemArgKind::Mem {
        base: ArgKind::Reg { reg: base, size: MemorySize::_64 },
        offset: None,
        disp,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
        mode: AddressingMode::PostIndex,
    }
}

// ---------------------------------------------------------------------------
// Writer extension trait
// ---------------------------------------------------------------------------

/// Extension trait providing WASM code generation for AArch64 writers.
/// Bytes per WASM operand-stack slot. The operand stack lives on the hardware
/// SP, and AArch64 (notably macOS, which enforces SP alignment on every
/// SP-relative access) requires SP to be 16-byte aligned. Each operand value is
/// 8 bytes, but we consume a full 16-byte slot per push/pop so SP stays aligned.
/// All operand-stack offset math (marshalling, tail calls) uses this stride.
pub const WASM_SLOT: i32 = 16;

pub trait WriterExt<Context>: Writer<AArch64Label, Context> {
    /// Push `r` onto the WASM stack (pre-decrement SP by a 16-byte slot).
    fn wasm_push(&mut self, ctx: &mut Context, arch: AArch64Arch, r: Reg)
        -> Result<(), Self::Error>
    {
        // str r, [sp, #-16]!
        self.str(ctx, arch, &reg(r), &mem_pre(SP, -WASM_SLOT))
    }

    /// Pop top of WASM stack into `r`.
    fn wasm_pop(&mut self, ctx: &mut Context, arch: AArch64Arch, r: Reg)
        -> Result<(), Self::Error>
    {
        // ldr r, [sp], #16
        self.ldr(ctx, arch, &reg(r), &mem_post(SP, WASM_SLOT))
    }

    /// Load local variable N into `dest`.
    fn load_local(&mut self, ctx: &mut Context, arch: AArch64Arch, dest: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.ldr(ctx, arch, &reg(dest), &mem_base_disp(FP, disp))
    }

    /// Store `src` into local variable N.
    fn store_local(&mut self, ctx: &mut Context, arch: AArch64Arch, src: Reg, n: usize)
        -> Result<(), Self::Error>
    {
        let disp = -((n as i32 + 1) * 8);
        self.str(ctx, arch, &reg(src), &mem_base_disp(FP, disp))
    }

    // ---- binary op helpers ----
    fn pop2_push<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T1)?; // rhs / top
        self.wasm_pop(ctx, arch, T0)?; // lhs
        f(self, ctx, arch, T2, T0, T1)?;
        self.wasm_push(ctx, arch, T2)
    }

    // ---- compare helper ----
    fn cmp_push_bool(&mut self, ctx: &mut Context, arch: AArch64Arch, cc: ConditionCode)
        -> Result<(), Self::Error>
    {
        self.wasm_pop(ctx, arch, T1)?; // rhs
        self.wasm_pop(ctx, arch, T0)?; // lhs
        self.cmp(ctx, arch, &reg(T0), &reg(T1))?;
        self.mov_imm(ctx, arch, &reg(T0), 0)?;
        self.mov_imm(ctx, arch, &reg(T1), 1)?;
        self.csel(ctx, arch, cc, &reg(T2), &reg(T1), &reg(T0))?;
        self.wasm_push(ctx, arch, T2)
    }

    // ---- floating-point helpers (bit-threading via GP<->FP moves) ----
    /// Pop two F64 operands (as bits), move into D0/D1, run `f(dst=D2, a=D0, b=D1)`,
    /// then push the D2 result bits.
    fn fp_binop_d<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T1)?;
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_d(ctx, arch, &reg(FD0), &reg(T0))?;
        self.fmov_gp_to_d(ctx, arch, &reg(FD1), &reg(T1))?;
        f(self, ctx, arch, FD2, FD0, FD1)?;
        self.fmov_d_to_gp(ctx, arch, &reg(T2), &reg(FD2))?;
        self.wasm_push(ctx, arch, T2)
    }

    /// F32 counterpart of [`Self::fp_binop_d`] (S registers).
    fn fp_binop_s<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T1)?;
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_s(ctx, arch, &reg(FD0), &reg(T0))?;
        self.fmov_gp_to_s(ctx, arch, &reg(FD1), &reg(T1))?;
        f(self, ctx, arch, FD2, FD0, FD1)?;
        self.fmov_s_to_gp(ctx, arch, &reg(T2), &reg(FD2))?;
        self.wasm_push(ctx, arch, T2)
    }

    /// Pop one F64 operand, run `f(dst=D1, src=D0)`, push the result.
    fn fp_unop_d<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_d(ctx, arch, &reg(FD0), &reg(T0))?;
        f(self, ctx, arch, FD1, FD0)?;
        self.fmov_d_to_gp(ctx, arch, &reg(T1), &reg(FD1))?;
        self.wasm_push(ctx, arch, T1)
    }

    /// F32 counterpart of [`Self::fp_unop_d`].
    fn fp_unop_s<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch, Reg, Reg) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_s(ctx, arch, &reg(FD0), &reg(T0))?;
        f(self, ctx, arch, FD1, FD0)?;
        self.fmov_s_to_gp(ctx, arch, &reg(T1), &reg(FD1))?;
        self.wasm_push(ctx, arch, T1)
    }

    /// Pop two F64 operands, `fcmp` them, push the i32 boolean for `cc`.
    /// WASM compares are false on NaN (except `ne`); the caller selects `cc`
    /// accordingly (eq=EQ, ne=NE, lt=MI, gt=GT, le=LS, ge=GE).
    fn fp_cmp_d(&mut self, ctx: &mut Context, arch: AArch64Arch, cc: ConditionCode)
        -> Result<(), Self::Error>
    {
        self.wasm_pop(ctx, arch, T1)?;
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_d(ctx, arch, &reg(FD0), &reg(T0))?;
        self.fmov_gp_to_d(ctx, arch, &reg(FD1), &reg(T1))?;
        self.fcmp(ctx, arch, &reg(FD0), &reg(FD1))?;
        self.mov_imm(ctx, arch, &reg(T0), 0)?;
        self.mov_imm(ctx, arch, &reg(T1), 1)?;
        self.csel(ctx, arch, cc, &reg(T2), &reg(T1), &reg(T0))?;
        self.wasm_push(ctx, arch, T2)
    }

    /// F32 counterpart of [`Self::fp_cmp_d`].
    fn fp_cmp_s(&mut self, ctx: &mut Context, arch: AArch64Arch, cc: ConditionCode)
        -> Result<(), Self::Error>
    {
        self.wasm_pop(ctx, arch, T1)?;
        self.wasm_pop(ctx, arch, T0)?;
        self.fmov_gp_to_s(ctx, arch, &reg(FD0), &reg(T0))?;
        self.fmov_gp_to_s(ctx, arch, &reg(FD1), &reg(T1))?;
        self.fcmp_s(ctx, arch, &reg(FD0), &reg(FD1))?;
        self.mov_imm(ctx, arch, &reg(T0), 0)?;
        self.mov_imm(ctx, arch, &reg(T1), 1)?;
        self.csel(ctx, arch, cc, &reg(T2), &reg(T1), &reg(T0))?;
        self.wasm_push(ctx, arch, T2)
    }

    /// Pop one value into T0, run `f` (which produces the result in T1 using
    /// FD0/FD1 as FP scratch), and push T1. Used for all int<->fp and f32<->f64
    /// conversions.
    fn fp_convert<F>(&mut self, ctx: &mut Context, arch: AArch64Arch, f: F)
        -> Result<(), Self::Error>
    where
        F: FnOnce(&mut Self, &mut Context, AArch64Arch) -> Result<(), Self::Error>,
    {
        self.wasm_pop(ctx, arch, T0)?;
        f(self, ctx, arch)?;
        self.wasm_push(ctx, arch, T1)
    }

    // ---- branch helper ----
    fn do_br(&mut self, ctx: &mut Context, arch: AArch64Arch, state: &State<'_>, depth: u32)
        -> Result<(), Self::Error>
    {
        let len = state.if_stack.len();
        let depth_usize = depth as usize;
        if depth_usize + 1 > len {
            // Branching out of the function (e.g., from an exception handler).
            // Emit a function exit: restore frame and return.
            self.mov(ctx, arch, &reg(SP), &reg(FP))?;
            self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
            return self.ret(ctx, arch);
        }
        let idx = len - depth_usize - 1;
        match state.if_stack[idx].clone() {
            Endable::TryTable { end_lbl, .. } => self.b_label(ctx, arch, end_lbl),
            Endable::Loop { head_lbl } => {
                self.b_label(ctx, arch, head_lbl)
            }
            Endable::Block { end_lbl } | Endable::If { end_lbl, .. } => {
                self.b_label(ctx, arch, end_lbl)
            }
        }
    }

    /// Emit a control-flow probe site (`TailTakeover` binding) for a
    /// loop/block header, consuming the next `probe_id`.  No-op when probes
    /// are disabled.  Uses T0 as scratch (T1 as the inner `inc_mem64` scratch).
    fn emit_control_flow_probe(&mut self, ctx: &mut Context, arch: AArch64Arch, state: &mut State<'_>)
        -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
            let probe_id = state.next_probe_id;
            state.next_probe_id += 1;
            let probe_base = state.probe_base;
            let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: T1.0, probe_base };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw, cfg.table_base_off, probe_id, T0.0,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    /// Apply the [`MemBase::WasmMemSymbol`] transform to a load/store address in
    /// `addr`: wrap it to 32 bits and add the `__wasm_mem` base, leaving the host
    /// address in `addr`. `scratch` is clobbered. No-op for [`MemBase::Raw`]. The
    /// static `memarg.offset` is still added afterwards by the load/store's
    /// addressing-mode displacement, matching `__wasm_mem + (uint32_t)addr + off`.
    fn apply_mem_base(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State<'_>,
        addr: Reg,
        scratch: Reg,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if state.mem_base != MemBase::WasmMemSymbol {
            return Ok(());
        }
        // addr := (uint32_t)addr — zero-extend the low 32 bits.
        self.uxt(ctx, arch, &reg(addr), &reg32(addr))?;
        // scratch := __wasm_mem (load the base pointer value). External symbol →
        // ADRP+ADD (Mach-O can't relocate a plain ADR).
        crate::load_label_addr(self, ctx, arch, &reg(scratch), AArch64Label::External { name: "__wasm_mem".into() })?;
        self.ldr(ctx, arch, &reg(scratch), &mem_base_disp(scratch, 0))?;
        // addr := addr + scratch.
        self.add(ctx, arch, &reg(addr), &reg(addr), &reg(scratch))?;
        Ok(())
    }

    /// Width-generic memory load. `access` is the load width (`_8/_16/_32/_64`);
    /// `signed` sign-extends a sub-word load; `to64` selects an i64 vs i32 result
    /// (i32 results stay zero-extended in the high 32 bits, per the operand model).
    fn mem_load(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State<'_>,
        offset: i32,
        access: MemorySize,
        signed: bool,
        to64: bool,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.wasm_pop(ctx, arch, T0)?; // address
        self.apply_mem_base(ctx, arch, state, T0, T2)?;
        let mem = MemArgKind::Mem {
            base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
            offset: None,
            disp: offset,
            size: access,
            reg_class: RegisterClass::Gpr,
            mode: AddressingMode::Offset,
        };
        // LDRB/LDRH/LDR(W)/LDR(X): all zero-extend into the destination register.
        self.ldr(ctx, arch, &reg_sz(T1, access), &mem)?;
        if signed && !matches!(access, MemorySize::_64) {
            let w = sz_bits(access);
            if to64 {
                let sh = MemArgKind::NoMem(ArgKind::Lit(64 - w));
                self.lsl(ctx, arch, &reg(T1), &reg(T1), &sh)?;
                self.asr(ctx, arch, &reg(T1), &reg(T1), &sh)?;
            } else {
                let sh = MemArgKind::NoMem(ArgKind::Lit(32 - w));
                self.lsl(ctx, arch, &reg32(T1), &reg32(T1), &sh)?;
                self.asr(ctx, arch, &reg32(T1), &reg32(T1), &sh)?;
            }
        }
        self.wasm_push(ctx, arch, T1)
    }

    /// Width-generic memory store (writes the low `access` bits of the value).
    fn mem_store(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &State<'_>,
        offset: i32,
        access: MemorySize,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.wasm_pop(ctx, arch, T1)?; // value
        self.wasm_pop(ctx, arch, T0)?; // address
        self.apply_mem_base(ctx, arch, state, T0, T2)?;
        let mem = MemArgKind::Mem {
            base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
            offset: None,
            disp: offset,
            size: access,
            reg_class: RegisterClass::Gpr,
            mode: AddressingMode::Offset,
        };
        self.str(ctx, arch, &reg_sz(T1, access), &mem)
    }

    /// Handle a single WASM instruction (the inner match).
    fn handle_insn(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
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
        if target != state.body {
            // On the very first instruction of a function, `state.body` is
            // still the `Default::default()` value (0) and no body has been
            // entered yet.  Emitting a skip-jump here would reference a
            // `_idx_0` label that is never `set_label`'d (because we never
            // visited body 0), leaving an unresolved branch that on AArch64
            // assembles to `b .` (jump-to-self / infinite loop).  Detect
            // the uninitialized case via the empty body_labels map and just
            // adopt the new target body.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                let skip_lbl = *state.body_labels.entry(state.body).or_insert_with(|| {
                    state.label_index += 1;
                    state.label_index - 1
                });
                self.b_label(ctx, arch, AArch64Label::Indexed { idx: skip_lbl })?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, AArch64Label::Indexed { idx })?;
                }
            }
        }
        match op {
            // ---- constants ----
            Instruction::I64Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), *v as u64)?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::I32Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), *v as u32 as u64)?;
                self.wasm_push(ctx, arch, T0)
            }

            // ---- locals ----
            Instruction::LocalGet(idx) => {
                self.load_local(ctx, arch, T0, *idx as usize)?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::LocalSet(idx) => {
                self.wasm_pop(ctx, arch, T0)?;
                self.store_local(ctx, arch, T0, *idx as usize)
            }
            Instruction::LocalTee(idx) => {
                // peek (don't pop), store
                self.ldr(ctx, arch, &reg(T0), &mem_base_disp(SP, 0))?;
                self.store_local(ctx, arch, T0, *idx as usize)
            }

            // ---- i64 arithmetic ----
            Instruction::I64Add => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.add(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Sub => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.sub(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Mul => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.mul(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64DivU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.udiv(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64DivS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.sdiv(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64RemU => {
                // rem = a - (a / b) * b
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.udiv(ctx, arch, &reg(T2), &reg(T0), &reg(T1))?;
                self.mul(ctx, arch, &reg(T2), &reg(T2), &reg(T1))?;
                self.sub(ctx, arch, &reg(T2), &reg(T0), &reg(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64RemS => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sdiv(ctx, arch, &reg(T2), &reg(T0), &reg(T1))?;
                self.mul(ctx, arch, &reg(T2), &reg(T2), &reg(T1))?;
                self.sub(ctx, arch, &reg(T2), &reg(T0), &reg(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64And => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.and(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Or  => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.orr(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Xor => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.eor(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64Shl => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.lsl(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64ShrU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.lsr(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::I64ShrS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| w.asr(c, a, &reg(d), &reg(x), &reg(y))),

            // ---- i32 arithmetic (zero-extend results to 64 bits) ----
            Instruction::I32Add => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.add(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32Sub => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sub(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32Mul => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.mul(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32DivU => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.udiv(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32DivS => {
                self.wasm_pop(ctx, arch, T1)?;
                self.wasm_pop(ctx, arch, T0)?;
                self.sdiv(ctx, arch, &reg32(T2), &reg32(T0), &reg32(T1))?;
                self.uxt(ctx, arch, &reg(T2), &reg32(T2))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I32And => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.and(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Or => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.orr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Xor => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.eor(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32Shl => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.lsl(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32ShrU => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.lsr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),
            Instruction::I32ShrS => self.pop2_push(ctx, arch, |w, c, a, d, x, y| {
                w.asr(c, a, &reg32(d), &reg32(x), &reg32(y))?;
                w.uxt(c, a, &reg(d), &reg32(d))
            }),

            // ---- comparisons ----
            Instruction::I64Eqz | Instruction::I32Eqz => {
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.mov_imm(ctx, arch, &reg(T0), 0)?;
                self.mov_imm(ctx, arch, &reg(T1), 1)?;
                self.csel(ctx, arch, ConditionCode::EQ, &reg(T2), &reg(T1), &reg(T0))?;
                self.wasm_push(ctx, arch, T2)
            }
            Instruction::I64Eq | Instruction::I32Eq => self.cmp_push_bool(ctx, arch, ConditionCode::EQ),
            Instruction::I64Ne | Instruction::I32Ne => self.cmp_push_bool(ctx, arch, ConditionCode::NE),
            Instruction::I64LtS | Instruction::I32LtS => self.cmp_push_bool(ctx, arch, ConditionCode::LT),
            Instruction::I64LtU | Instruction::I32LtU => self.cmp_push_bool(ctx, arch, ConditionCode::LO),
            Instruction::I64GtS | Instruction::I32GtS => self.cmp_push_bool(ctx, arch, ConditionCode::GT),
            Instruction::I64GtU | Instruction::I32GtU => self.cmp_push_bool(ctx, arch, ConditionCode::HI),
            Instruction::I64LeS | Instruction::I32LeS => self.cmp_push_bool(ctx, arch, ConditionCode::LE),
            Instruction::I64LeU | Instruction::I32LeU => self.cmp_push_bool(ctx, arch, ConditionCode::LS),
            Instruction::I64GeS | Instruction::I32GeS => self.cmp_push_bool(ctx, arch, ConditionCode::GE),
            Instruction::I64GeU | Instruction::I32GeU => self.cmp_push_bool(ctx, arch, ConditionCode::HS),

            // ---- memory loads (linear memory) ----
            // F64 load/store reuse the i64 paths (FP values ride as raw bits);
            // F32 load/store reuse the i32 (low-32, zero-extended) paths.
            Instruction::I64Load(m) | Instruction::F64Load(m) => {
                self.wasm_pop(ctx, arch, T0)?; // address
                self.apply_mem_base(ctx, arch, state, T0, T2)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg(T1), &mem)?;
                self.wasm_push(ctx, arch, T1)
            }
            Instruction::I32Load(m) | Instruction::F32Load(m) => {
                self.wasm_pop(ctx, arch, T0)?;
                self.apply_mem_base(ctx, arch, state, T0, T2)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg32(T1), &mem)?;
                self.uxt(ctx, arch, &reg(T1), &reg32(T1))?;
                self.wasm_push(ctx, arch, T1)
            }

            // ---- memory stores ----
            Instruction::I64Store(m) | Instruction::F64Store(m) => {
                self.wasm_pop(ctx, arch, T1)?; // value
                self.wasm_pop(ctx, arch, T0)?; // address
                self.apply_mem_base(ctx, arch, state, T0, T2)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.str(ctx, arch, &reg(T1), &mem)
            }
            // i32.store, i64.store32 and f32.store all write the low 32 bits.
            Instruction::I32Store(m) | Instruction::I64Store32(m) | Instruction::F32Store(m) => {
                self.wasm_pop(ctx, arch, T1)?; // value
                self.wasm_pop(ctx, arch, T0)?; // address
                self.apply_mem_base(ctx, arch, state, T0, T2)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: m.offset as i32,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.str(ctx, arch, &reg32(T1), &mem)
            }

            // ---- sub-word loads (zero/sign-extended) ----
            Instruction::I32Load8U(m)  => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_8,  false, false),
            Instruction::I32Load8S(m)  => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_8,  true,  false),
            Instruction::I32Load16U(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_16, false, false),
            Instruction::I32Load16S(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_16, true,  false),
            Instruction::I64Load8U(m)  => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_8,  false, true),
            Instruction::I64Load8S(m)  => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_8,  true,  true),
            Instruction::I64Load16U(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_16, false, true),
            Instruction::I64Load16S(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_16, true,  true),
            Instruction::I64Load32U(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_32, false, true),
            Instruction::I64Load32S(m) => self.mem_load(ctx, arch, state, m.offset as i32, MemorySize::_32, true,  true),

            // ---- sub-word stores ----
            Instruction::I32Store8(m)  => self.mem_store(ctx, arch, state, m.offset as i32, MemorySize::_8),
            Instruction::I32Store16(m) => self.mem_store(ctx, arch, state, m.offset as i32, MemorySize::_16),
            Instruction::I64Store8(m)  => self.mem_store(ctx, arch, state, m.offset as i32, MemorySize::_8),
            Instruction::I64Store16(m) => self.mem_store(ctx, arch, state, m.offset as i32, MemorySize::_16),

            // ---- memory.size / memory.grow ----
            Instruction::MemorySize(_) => {
                // Load address of __wasm_mem_pages, load 32-bit page count.
                self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_mem_pages".into() })?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: T0, size: MemorySize::_64 },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                    mode: AddressingMode::Offset,
                };
                self.ldr(ctx, arch, &reg32(T0), &mem)?;
                self.uxt(ctx, arch, &reg(T0), &reg32(T0))?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::MemoryGrow(_) => {
                // delta is on WASM stack; call via blitz WASM convention.
                self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_memory_grow".into() })?;
                self.bl(ctx, arch, &reg(T0))
            }

            // ---- control flow ----
            Instruction::Block(_) => {
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                state.if_stack.push(Endable::Block { end_lbl });
                self.emit_control_flow_probe(ctx, arch, state)?;
                Ok(())
            }
            Instruction::Loop(_) => {
                let head_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.set_label(ctx, arch, head_lbl.clone())?;
                state.if_stack.push(Endable::Loop { head_lbl });
                self.emit_control_flow_probe(ctx, arch, state)?;
                Ok(())
            }
            Instruction::If(_) => {
                let else_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.bcond_label(ctx, arch, ConditionCode::EQ, else_lbl.clone())?;
                state.if_stack.push(Endable::If { else_lbl, end_lbl });
                Ok(())
            }
            Instruction::Else => {
                if let Some(Endable::If { else_lbl, end_lbl }) = state.if_stack.last().cloned() {
                    self.b_label(ctx, arch, end_lbl.clone())?;
                    self.set_label(ctx, arch, else_lbl)?;
                    *state.if_stack.last_mut().unwrap() = Endable::If {
                        else_lbl: AArch64Label::Indexed { idx: usize::MAX },
                        end_lbl,
                    };
                }
                Ok(())
            }
            Instruction::End => {
                if let Some(frame) = state.if_stack.pop() {
                    match frame {
                        Endable::Block { end_lbl } => self.set_label(ctx, arch, end_lbl)?,
                        Endable::If { end_lbl, .. } => self.set_label(ctx, arch, end_lbl)?,
                        Endable::Loop { .. } => {}
                        Endable::TryTable { end_lbl, dispatch_lbl, catches } => {
                            let after_lbl = AArch64Label::Indexed { idx: state.label_index };
                            state.label_index += 1;
                            // Normal path: jump past dispatch stub.
                            self.b_label(ctx, arch, after_lbl.clone())?;
                            // Dispatch stub.
                            self.set_label(ctx, arch, dispatch_lbl)?;
                            for catch in catches.iter() {
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else { 0 };
                                        let skip_lbl = AArch64Label::Indexed { idx: state.label_index };
                                        state.label_index += 1;
                                        self.mov_imm(ctx, arch, &reg(T1), *tag as u64)?;
                                        self.cmp(ctx, arch, &reg(T0), &reg(T1))?;
                                        self.bcond_label(ctx, arch, ConditionCode::NE, skip_lbl.clone())?;
                                        // Push exception values from x11, x12, x13 (arity 1, 2, 3)
                                        for i in (0..arity.min(3)).rev() {
                                            self.wasm_push(ctx, arch, Reg(11 + i as u8))?;
                                        }
                                        self.do_br(ctx, arch, state, *label)?;
                                        self.set_label(ctx, arch, skip_lbl)?;
                                    }
                                    Catch::All { label } => {
                                        self.do_br(ctx, arch, state, *label)?;
                                    }
                                    Catch::OneRef { .. } | Catch::AllRef { .. } => {}
                                }
                            }
                            self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: "__wasm_exn_propagate".into() })?;
                            self.bl(ctx, arch, &reg(T0))?;
                            self.set_label(ctx, arch, after_lbl)?;
                            self.set_label(ctx, arch, end_lbl)?;
                        }
                    }
                }
                Ok(())
            }
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else { 0 };
                self.mov_imm(ctx, arch, &reg(T0), *tag_index as u64)?; // tag in T0
                for i in 0..arity.min(3) {
                    self.wasm_pop(ctx, arch, Reg(11 + i as u8))?; // x11, x12, x13 for values
                }
                if let Some(dispatch_lbl) = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_lbl, .. } => Some(dispatch_lbl.clone()),
                    _ => None,
                }) {
                    self.b_label(ctx, arch, dispatch_lbl)
                } else {
                    self.adr_label(ctx, arch, &reg(T1), AArch64Label::External { name: "__wasm_exn_propagate".into() })?;
                    self.bl(ctx, arch, &reg(T1))
                }
            }
            Instruction::ThrowRef => todo!("exnref deferred"),
            Instruction::TryTable(_, catches) => {
                let dispatch_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                let end_lbl = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                state.if_stack.push(Endable::TryTable {
                    end_lbl,
                    dispatch_lbl,
                    catches: catches.iter().cloned().collect::<alloc::vec::Vec<_>>().into_boxed_slice(),
                });
                Ok(())
            }
            Instruction::Br(depth) => self.do_br(ctx, arch, state, *depth),
            Instruction::BrIf(depth) => {
                let skip = AArch64Label::Indexed { idx: state.label_index };
                state.label_index += 1;
                self.wasm_pop(ctx, arch, T0)?;
                self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.bcond_label(ctx, arch, ConditionCode::EQ, skip.clone())?;
                self.do_br(ctx, arch, state, *depth)?;
                self.set_label(ctx, arch, skip)
            }
            Instruction::BrTable(targets, default) => {
                self.wasm_pop(ctx, arch, T0)?; // index
                let mut case_labels = Vec::new();
                for _ in targets.iter() {
                    case_labels.push(AArch64Label::Indexed { idx: state.label_index });
                    state.label_index += 1;
                }
                for (i, _) in targets.iter().enumerate() {
                    self.cmp(ctx, arch, &reg(T0), &MemArgKind::NoMem(ArgKind::Lit(i as u64)))?;
                    self.bcond_label(ctx, arch, ConditionCode::EQ, case_labels[i].clone())?;
                }
                self.do_br(ctx, arch, state, *default)?;
                for (i, target) in targets.iter().enumerate() {
                    self.set_label(ctx, arch, case_labels[i].clone())?;
                    self.do_br(ctx, arch, state, *target)?;
                }
                Ok(())
            }

            // ---- function calls ----
            Instruction::Call(fn_idx) => {
                let fn_idx_val = *fn_idx;
                let target = state.shard.as_ref().map(|s| s.call_target(fn_idx_val));
                match target {
                    Some(CallTarget::CrossShard { table_slot }) => {
                        // Cross-shard: load fn ptr from [SCR + table_slot * 8].
                        self.ldr(ctx, arch, &reg(T0), &mem_base_disp(SCR, table_slot as i32 * 8))?;
                        self.bl(ctx, arch, &reg(T0))?;
                    }
                    _ => {
                        match func_imports.get(fn_idx_val as usize) {
                            Some((module, name)) => {
                                let sym = alloc::format!("{module}__{name}");
                                self.adr_label(ctx, arch, &reg(T0), AArch64Label::External { name: sym })?;
                                self.bl(ctx, arch, &reg(T0))?;
                            }
                            None => {
                                let idx = fn_idx_val - func_imports.len() as u32;
                                self.adr_label(ctx, arch, &reg(T0), AArch64Label::Func { r#fn: idx })?;
                                self.bl(ctx, arch, &reg(T0))?;
                            }
                        }
                    }
                }
                Ok(())
            }

            Instruction::Return => {
                // Restore SP from FP, reload FP+LR, return.
                self.mov(ctx, arch, &reg(SP), &reg(FP))?;
                self.ldp(ctx, arch, &reg(FP), &reg(LR), &mem_post(SP, 16))?;
                // Restore SCR if sharding active (T0 gets discarded garbage — OK).
                if state.shard.is_some() {
                    self.ldp(ctx, arch, &reg(SCR), &reg(T0), &mem_post(SP, 16))?;
                }
                self.ret(ctx, arch)
            }

            Instruction::Unreachable => {
                // Trap: BRK #0.
                self.brk(ctx, arch, 0)
            }
            Instruction::I64ExtendI32S | Instruction::I64Extend32S => {
                // Sign-extend the low 32 bits: lsl #32 then asr #32.
                self.wasm_pop(ctx, arch, T0)?;
                let sh = MemArgKind::NoMem(ArgKind::Lit(32));
                self.lsl(ctx, arch, &reg(T0), &reg(T0), &sh)?;
                self.asr(ctx, arch, &reg(T0), &reg(T0), &sh)?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::I32WrapI64 | Instruction::I64ExtendI32U => {
                // Both keep the low 32 bits and clear the upper word
                // (i32.wrap_i64 truncates; i64.extend_i32_u zero-extends).
                // The binary `and` only encodes the register form, so materialize
                // the mask in a scratch register first.
                self.wasm_pop(ctx, arch, T0)?;
                self.mov_imm(ctx, arch, &reg(Reg(10)), 0xFFFF_FFFF)?;
                self.and(ctx, arch, &reg(T0), &reg(T0), &reg(Reg(10)))?;
                self.wasm_push(ctx, arch, T0)
            }

            // Reinterprets are no-ops: FP values already ride as raw bits.
            Instruction::F32ReinterpretI32
            | Instruction::I32ReinterpretF32
            | Instruction::F64ReinterpretI64
            | Instruction::I64ReinterpretF64 => Ok(()),

            // ---- F64 arithmetic ----
            Instruction::F64Add => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fadd(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Sub => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fsub(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Mul => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fmul(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Div => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fdiv(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Min => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fmin(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Max => self.fp_binop_d(ctx, arch, |w, c, a, d, x, y| w.fmax(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F64Sqrt => self.fp_unop_d(ctx, arch, |w, c, a, d, x| w.fsqrt(c, a, &reg(d), &reg(x))),
            Instruction::F64Abs => self.fp_unop_d(ctx, arch, |w, c, a, d, x| w.fabs(c, a, &reg(d), &reg(x))),
            Instruction::F64Neg => self.fp_unop_d(ctx, arch, |w, c, a, d, x| w.fneg(c, a, &reg(d), &reg(x))),

            // ---- F32 arithmetic ----
            Instruction::F32Add => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fadd_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Sub => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fsub_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Mul => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fmul_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Div => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fdiv_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Min => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fmin_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Max => self.fp_binop_s(ctx, arch, |w, c, a, d, x, y| w.fmax_s(c, a, &reg(d), &reg(x), &reg(y))),
            Instruction::F32Sqrt => self.fp_unop_s(ctx, arch, |w, c, a, d, x| w.fsqrt_s(c, a, &reg(d), &reg(x))),
            Instruction::F32Abs => self.fp_unop_s(ctx, arch, |w, c, a, d, x| w.fabs_s(c, a, &reg(d), &reg(x))),
            Instruction::F32Neg => self.fp_unop_s(ctx, arch, |w, c, a, d, x| w.fneg_s(c, a, &reg(d), &reg(x))),

            // ---- FP compares (false on NaN except `ne`) ----
            Instruction::F64Eq => self.fp_cmp_d(ctx, arch, ConditionCode::EQ),
            Instruction::F64Ne => self.fp_cmp_d(ctx, arch, ConditionCode::NE),
            Instruction::F64Lt => self.fp_cmp_d(ctx, arch, ConditionCode::MI),
            Instruction::F64Gt => self.fp_cmp_d(ctx, arch, ConditionCode::GT),
            Instruction::F64Le => self.fp_cmp_d(ctx, arch, ConditionCode::LS),
            Instruction::F64Ge => self.fp_cmp_d(ctx, arch, ConditionCode::GE),
            Instruction::F32Eq => self.fp_cmp_s(ctx, arch, ConditionCode::EQ),
            Instruction::F32Ne => self.fp_cmp_s(ctx, arch, ConditionCode::NE),
            Instruction::F32Lt => self.fp_cmp_s(ctx, arch, ConditionCode::MI),
            Instruction::F32Gt => self.fp_cmp_s(ctx, arch, ConditionCode::GT),
            Instruction::F32Le => self.fp_cmp_s(ctx, arch, ConditionCode::LS),
            Instruction::F32Ge => self.fp_cmp_s(ctx, arch, ConditionCode::GE),

            // ---- conversions: int -> fp (source bits already in T0/GP) ----
            Instruction::F64ConvertI32S => self.fp_convert(ctx, arch, |w, c, a| { w.scvtf_d_w(c, a, &reg(FD0), &reg(T0))?; w.fmov_d_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F64ConvertI32U => self.fp_convert(ctx, arch, |w, c, a| { w.ucvtf_d_w(c, a, &reg(FD0), &reg(T0))?; w.fmov_d_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F64ConvertI64S => self.fp_convert(ctx, arch, |w, c, a| { w.scvtf_d_x(c, a, &reg(FD0), &reg(T0))?; w.fmov_d_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F64ConvertI64U => self.fp_convert(ctx, arch, |w, c, a| { w.ucvtf_d_x(c, a, &reg(FD0), &reg(T0))?; w.fmov_d_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F32ConvertI32S => self.fp_convert(ctx, arch, |w, c, a| { w.scvtf_s_w(c, a, &reg(FD0), &reg(T0))?; w.fmov_s_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F32ConvertI32U => self.fp_convert(ctx, arch, |w, c, a| { w.ucvtf_s_w(c, a, &reg(FD0), &reg(T0))?; w.fmov_s_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F32ConvertI64S => self.fp_convert(ctx, arch, |w, c, a| { w.scvtf_s_x(c, a, &reg(FD0), &reg(T0))?; w.fmov_s_to_gp(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::F32ConvertI64U => self.fp_convert(ctx, arch, |w, c, a| { w.ucvtf_s_x(c, a, &reg(FD0), &reg(T0))?; w.fmov_s_to_gp(c, a, &reg(T1), &reg(FD0)) }),

            // ---- conversions: fp -> int (truncating) ----
            Instruction::I32TruncF64S => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_d(c, a, &reg(FD0), &reg(T0))?; w.fcvtzs_w_d(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I32TruncF64U => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_d(c, a, &reg(FD0), &reg(T0))?; w.fcvtzu_w_d(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I64TruncF64S => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_d(c, a, &reg(FD0), &reg(T0))?; w.fcvtzs_x_d(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I64TruncF64U => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_d(c, a, &reg(FD0), &reg(T0))?; w.fcvtzu_x_d(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I32TruncF32S => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_s(c, a, &reg(FD0), &reg(T0))?; w.fcvtzs_w_s(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I32TruncF32U => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_s(c, a, &reg(FD0), &reg(T0))?; w.fcvtzu_w_s(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I64TruncF32S => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_s(c, a, &reg(FD0), &reg(T0))?; w.fcvtzs_x_s(c, a, &reg(T1), &reg(FD0)) }),
            Instruction::I64TruncF32U => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_s(c, a, &reg(FD0), &reg(T0))?; w.fcvtzu_x_s(c, a, &reg(T1), &reg(FD0)) }),

            // ---- conversions: f32 <-> f64 ----
            Instruction::F32DemoteF64 => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_d(c, a, &reg(FD0), &reg(T0))?; w.fcvt_s_d(c, a, &reg(FD1), &reg(FD0))?; w.fmov_s_to_gp(c, a, &reg(T1), &reg(FD1)) }),
            Instruction::F64PromoteF32 => self.fp_convert(ctx, arch, |w, c, a| { w.fmov_gp_to_s(c, a, &reg(FD0), &reg(T0))?; w.fcvt_d_s(c, a, &reg(FD1), &reg(FD0))?; w.fmov_d_to_gp(c, a, &reg(T1), &reg(FD1)) }),

            // FP constants ride as raw bits on the GP operand stack.
            Instruction::F64Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), v.bits())?;
                self.wasm_push(ctx, arch, T0)
            }
            Instruction::F32Const(v) => {
                self.mov_imm(ctx, arch, &reg(T0), v.bits() as u64)?;
                self.wasm_push(ctx, arch, T0)
            }

            // select: c ? a : b (pop c, b, a).
            Instruction::Select | Instruction::TypedSelect { .. } => {
                self.wasm_pop(ctx, arch, T2)?; // condition
                self.wasm_pop(ctx, arch, T1)?; // b (false value)
                self.wasm_pop(ctx, arch, T0)?; // a (true value)
                self.cmp(ctx, arch, &reg(T2), &MemArgKind::NoMem(ArgKind::Lit(0)))?;
                self.csel(ctx, arch, ConditionCode::NE, &reg(T0), &reg(T0), &reg(T1))?;
                self.wasm_push(ctx, arch, T0)
            }

            other => panic!("unimplemented WASM instruction in AArch64 naive handle_insn: {other:?}"),
        }
    }

    /// Emit the optional tracing preamble.
    ///
    /// - **NaiveAbi**: call from `StartBody` after the function-entry label.
    ///   Use `scratch = T0` (x9); FP and LR hold frame/return-addr.
    /// - **SysVAbi**: call from `StartFn` after `set_label`, before frame setup.
    ///   Use `scratch = T0` (x9); SysV arg regs (x0–x7) are untouched.
    /// Handle a `MachOperator` (the outer match, called by the pipeline).
    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: AArch64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;

                self.set_label(ctx, arch, AArch64Label::Func { r#fn: *id }).map_err(Err::from)?;

                state.probes = data.probes;
                state.next_probe_id = 1;
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW::new(self, ctx, arch, T1.0);
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw, cfg.table_base_off, 0, T0.0,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                // Save SCR (X27) in a 16-byte aligned pair before FP+LR.
                if state.shard.is_some() {
                    self.stp(ctx, arch, &reg(SCR), &reg(T0), &mem_pre(SP, -16)).map_err(Err::from)?;
                }
                self.stp(ctx, arch, &reg(FP), &reg(LR), &mem_pre(SP, -16)).map_err(Err::from)?;
                self.mov(ctx, arch, &reg(FP), &reg(SP)).map_err(Err::from)?;

                let locals_slots = state.local_count as i64 + state.control_depth as i64 * 2 + 2;
                if locals_slots > 0 {
                    // Round the frame to 16 bytes so SP stays 16-byte aligned.
                    let bytes = (locals_slots as u64 * 8 + 15) & !15;
                    let size = MemArgKind::NoMem(ArgKind::Lit(bytes));
                    self.sub(ctx, arch, &reg(SP), &reg(SP), &size).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::Local { count, .. } => {
                self.mov_imm(ctx, arch, &reg(T0), 0).map_err(Err::from)?;
                for _ in 0..*count {
                    state.local_count += 1;
                    self.store_local(ctx, arch, T0, state.local_count - 1).map_err(Err::from)?;
                }
                Ok(())
            }

            MachOperator::StartBody => Ok(()),
            MachOperator::EndBody => Ok(()),

            MachOperator::Instruction { op: insn, .. } => {
                self.handle_insn(ctx, arch, state, func_imports, sigs, tags, insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: Some(op_wasm), .. } => {
                let insn = rewriter.instruction(op_wasm.clone())?;
                self.handle_insn(ctx, arch, state, func_imports, sigs, tags, &insn, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op: None, .. } => Ok(()),
            _ => Ok(()), // non-instruction operators (Local, StartBody, etc.) silently ignored
        }
    }
}

impl<T: Writer<AArch64Label, Context> + ?Sized, Context> WriterExt<Context> for T {}

/// Emit one-instruction jump stubs for each exported function.
///
/// Each stub emits an `External` label followed by a `b_label` to the
/// function's internal label.
pub fn emit_export_dispatchers<W, Ctx>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: AArch64Arch,
    exports: &[(u32, &str)],
) -> Result<(), W::Error>
where
    W: WriterExt<Ctx>,
{
    for (id, name) in exports {
        w.set_label(ctx, arch, AArch64Label::External { name: (*name).into() })?;
        w.b_label(ctx, arch, AArch64Label::Func { r#fn: *id })?;
    }
    Ok(())
}

#[cfg(test)]
mod membase_tests {
    use super::*;
    use crate::{AArch64Arch, AArch64Label};
    use alloc::string::String;
    use alloc::vec::Vec;
    use portal_solutions_asm_aarch64::out::bin::AArch64Writer;
    use portal_solutions_blitz_common::wasm_encoder::MemArg;

    fn load_externals(mem_base: MemBase) -> Vec<String> {
        let mut out = AArch64Writer::<AArch64Label>::new();
        let mut ctx = ();
        let mut state = State { mem_base, ..State::default() };
        let op = Instruction::I64Load(MemArg { offset: 0, align: 3, memory_index: 0 });
        WriterExt::handle_insn(&mut out, &mut ctx, AArch64Arch::default(), &mut state, &[], &[], &[], &op, 0)
            .unwrap();
        let (_bytes, _labels, relocs) = out.into_parts_with_relocs();
        relocs
            .into_iter()
            .filter_map(|r| match r.label {
                AArch64Label::External { name } => Some(name),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn raw_mode_emits_no_wasm_mem_reference() {
        assert!(!load_externals(MemBase::Raw).iter().any(|n| n == "__wasm_mem"));
    }

    #[test]
    fn wasm_mem_symbol_mode_references_base() {
        let externs = load_externals(MemBase::WasmMemSymbol);
        assert_eq!(externs.iter().filter(|n| *n == "__wasm_mem").count(), 1);
    }
}
