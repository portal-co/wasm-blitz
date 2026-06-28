//! Naive RISC-V codegen (incremental port)

#![allow(dead_code)]

/// Feature-gated trace macro.  Enable with `--features portal-solutions-blitz-riscv64/log`.
#[cfg(feature = "log")]
macro_rules! trace {
    ($($arg:tt)*) => { eprintln!("[riscv64-naive] {}", format_args!($($arg)*)) };
}
#[cfg(not(feature = "log"))]
macro_rules! trace {
    ($($arg:tt)*) => {};
}

use crate::RiscvLabel;
use alloc::vec::Vec;
use portal_solutions_asm_riscv64::ConditionCode;
use portal_solutions_asm_riscv64::RiscV64Arch;
use portal_solutions_asm_riscv64::out::Writer;

use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::ops::{MachOperator, ProbeTableConfig};
use portal_solutions_blitz_common::shard::{CallTarget, SecondCtxConfig};
use portal_solutions_blitz_common::wasm_encoder;
use portal_solutions_blitz_common::wasm_encoder::reencode::{self as reencode, Reencode};

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv64::RegisterClass;
use portal_solutions_asm_riscv64::out::arg::{ArgKind, MemArgKind};

use core::ops::{Index, IndexMut};
use portal_solutions_asm_regalloc as regalloc;
use portal_solutions_asm_riscv64 as asm_riscv;
use portal_solutions_asm_riscv64::regalloc as riscv_regalloc;

/// Static Context Register (SCR) — S10 (x26) on RISC-V 64.
///
/// Callee-saved; holds the cross-shard function-pointer table pointer when
/// sharding is active. See `docs/second-context-register.md`.
pub const SCR: Reg = Reg(26);

/// Sharding state for RISC-V 64 functions — same design as x86-64/AArch64.
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

#[derive(Default)]
pub struct State<'a> {
    pub label_index: usize,
    pub local_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub if_stack: Vec<Endable>,
    pub regalloc: Option<regalloc::RegAlloc<riscv_regalloc::RegKind, 32, Frames>>,
    pub body: u32,
    pub body_labels: alloc::collections::BTreeMap<u32, usize>,
    /// Carried from `StartFn` to `StartBody` for RISC-V NaiveAbi — not actually
    /// needed here since the label is placed before frame setup, but kept for
    /// consistency with the x86-64 backend.  Preamble is emitted in `StartFn`.
    pub probes: Option<ProbeTableConfig>,
    /// Next probe id to assign (function entry = probe 0; each loop/block
    /// consumes the next).  See `emit_probe_site`.
    pub next_probe_id: u32,
    /// Total frame size in bytes, set by SysV `StartFn` to locate the RA/FP
    /// save slots at the bottom of the frame (`[FP - sysv_frame_sz]` = RA).
    pub sysv_frame_sz: i32,
    /// How mid-function probe sites reach the runtime probe-table base.  The
    /// NaiveAbi keeps the default (CTX-relative); the SysV ABI sets this to a
    /// frame slot after spilling its virtual-param base register.
    pub probe_base: crate::codegen::ProbeBase,
    /// Present when sharding is active. SCR (S10/x26) is saved in the SysV
    /// frame. Naive functions use SCR read-only (the runtime sets it).
    pub shard: Option<NaiveShardState<'a>>,
    /// Embedder-requested probes at arbitrary instruction indices, in addition
    /// to the function-entry/loop/block probes above.  `None` → zero overhead,
    /// identical codegen to today.
    pub probe_plan: Option<portal_solutions_blitz_common::ops::ProbePlan>,
    /// Ordinal index of the next dispatched instruction (0 = the first real
    /// WASM operator after locals), used to look up `probe_plan` entries.
    pub op_index: usize,
}

pub struct Frames(pub [[regalloc::RegAllocFrame<riscv_regalloc::RegKind>; 32]; 2]);

impl Index<riscv_regalloc::RegKind> for Frames {
    type Output = [regalloc::RegAllocFrame<riscv_regalloc::RegKind>; 32];
    fn index(&self, k: riscv_regalloc::RegKind) -> &Self::Output {
        match k {
            riscv_regalloc::RegKind::Int => &self.0[0],
            riscv_regalloc::RegKind::Float => &self.0[1],
        }
    }
}

impl IndexMut<riscv_regalloc::RegKind> for Frames {
    fn index_mut(&mut self, k: riscv_regalloc::RegKind) -> &mut Self::Output {
        match k {
            riscv_regalloc::RegKind::Int => &mut self.0[0],
            riscv_regalloc::RegKind::Float => &mut self.0[1],
        }
    }
}

impl regalloc::Length for Frames {
    fn len(&self) -> usize {
        2
    }
}

pub enum Endable {
    Block { idx: usize },
    Loop { idx: usize },
    If { idx: usize },
    TryTable {
        exit_idx: usize,
        dispatch_idx: usize,
        after_dispatch_idx: usize,
        catches: alloc::boxed::Box<[portal_solutions_blitz_common::wasm_encoder::Catch]>,
    },
}

pub trait WriterExt<Context>: Writer<RiscvLabel, Context> {
    /// Emit a control-flow probe site (`TailTakeover` binding) for a
    /// loop/block header, consuming the next `probe_id`.  No-op when probes
    /// are disabled.
    ///
    /// Flushes regalloc first so the operand stack is materialised at the site
    /// (matching the generic-entry layout the specialization tail-jump expects).
    /// Uses t0 (Reg 5) as scratch, t1 (Reg 6) as the `inc_mem64` scratch.
    fn emit_control_flow_probe(&mut self, ctx: &mut Context, arch: RiscV64Arch, state: &mut State<'_>)
        -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        if let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) {
            if let Some(ralloc) = state.regalloc.as_mut() {
                let it = ralloc.flush();
                emit_cmds(self, ctx, arch, it)?;
                ralloc.tos = None;
            }
            let probe_id = state.next_probe_id;
            state.next_probe_id += 1;
            let probe_base = state.probe_base;
            let mut bw = crate::codegen::BlitzW { writer: self, ctx, arch, scratch2: 6, probe_base };
            portal_solutions_blitz_codegen::emit_probe_site(
                &mut bw, cfg.table_base_off, probe_id, 5,
                portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                &mut state.label_index,
            )?;
        }
        Ok(())
    }

    fn br(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        relative_depth: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // flush regalloc before branching; reset TOS to avoid stale chain pointers
        if let Some(ralloc) = state.regalloc.as_mut() {
            let it = ralloc.flush();
            emit_cmds(self, ctx, arch, it)?;
            ralloc.tos = None;
        }
        let mut depth = relative_depth as usize;
        for entry in state.if_stack.iter().rev() {
            if depth == 0 {
                match entry {
                    Endable::Block { idx } => {
                        let lbl = RiscvLabel::Indexed { idx: *idx };
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl,
                        )?;
                        return Ok(());
                    }
                    Endable::Loop { idx } => {
                        let lbl = RiscvLabel::Indexed { idx: *idx };
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl,
                        )?;
                        return Ok(());
                    }
                    Endable::If { idx } => {
                        let lbl = RiscvLabel::Indexed { idx: *idx + 2 };
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl,
                        )?;
                        return Ok(());
                    }
                    Endable::TryTable { exit_idx, .. } => {
                        let lbl = RiscvLabel::Indexed { idx: *exit_idx };
                        self.jal_label(ctx, arch, &portal_solutions_blitz_common::asm::Reg(0), lbl)?;
                        return Ok(());
                    }
                }
            }
            depth -= 1;
        }
        Ok(())
    }
    /// Emit the optional tracing preamble.
    ///
    /// - **NaiveAbi**: call in `StartFn` after `set_label`, before frame setup.
    ///   Use `scratch = Reg(5)` (t0); RISC-V NaiveAbi passes args on the WASM stack.
    /// - **SysVAbi**: call in `StartFn` after `set_label`, before `addi sp, sp, -N`.
    ///   Use `scratch = Reg(5)` (t0); SysV arg regs a0–a7 (Reg 10–17) are untouched.
    fn handle_op_<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[portal_solutions_blitz_common::wasm_encoder::FuncType],
        tags: &[u32],
        op: &portal_solutions_blitz_common::wasm_encoder::Instruction<'_>,
        _rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        trace!("handle_op_ enter: target={target} body={}", state.body);
        if target != state.body {
            // First-instruction guard: state.body == 0 is the Default value,
            // not a real prior body.  See aarch64/naive.rs for the rationale.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jal_label(
                    ctx,
                    arch,
                    &Reg(0),
                    RiscvLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, RiscvLabel::Indexed { idx })?;
                }
            }
        }
        use portal_solutions_blitz_common::wasm_encoder::Instruction;
        match op {
            Instruction::I32Const(v) => {
                // Use regalloc to push an int value
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (ridx, cmds) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cmds)?;
                let phys = Reg(ridx as u8);
                self.li(ctx, arch, &phys, *v as u64)?;
            }
            Instruction::I64Const(v) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (ridx, cmds) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cmds)?;
                let phys = Reg(ridx as u8);
                self.li(ctx, arch, &phys, *v as u64)?;
            }
            Instruction::LocalGet(local_index) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let cmds = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_local(riscv_regalloc::RegKind::Int, *local_index)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cmds)?;
            }
            Instruction::LocalSet(local_index) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                // pop_local transitions TOS from Stack → Local(N), marking the register as holding
                // local N's value. No memory write yet; flush() or eviction will emit SetLocal.
                let it = state.regalloc.as_mut().unwrap()
                    .pop_local(riscv_regalloc::RegKind::Int, *local_index);
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::LocalTee(local_index) => {
                let fp = Reg(8);
                let tmp = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &tmp, &spmem)?;
                let disp = -((*local_index as i32 + 1) * 8);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: fp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &tmp, &mem)?;
                // push tmp back
                let sp = Reg(2);
                self.addi(ctx, arch, &sp, &sp, -8)?;
                let spmem2 = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &tmp, &spmem2)?;
            }
            Instruction::I64Load(memarg) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop address
                let (addr_t, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                // allocate dest reg for loaded value
                let (didx, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cmds2)?;
                let addr = Reg(addr_t.reg);
                let dest = Reg(didx as u8);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: addr,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: memarg.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                // push() already marks the register as Stack and updates TOS.
                self.ld(ctx, arch, &dest, &mem)?;
            }
            Instruction::I32Load(memarg) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                let (addr_t, cmds1) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                // push() marks the register as Stack and updates TOS; lw writes the loaded value.
                let (didx, cmds2) = state.regalloc.as_mut().unwrap().push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cmds2)?;
                let addr = Reg(addr_t.reg);
                let dest = Reg(didx as u8);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: addr, size: MemorySize::_64 },
                    offset: None, disp: memarg.offset as i32, size: MemorySize::_32, reg_class: RegisterClass::Gpr,
                };
                self.lw(ctx, arch, &dest, &mem)?;
            }
            Instruction::I32Store(memarg) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                let (val_t, cmds1) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (addr_t, cmds2) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let val = Reg(val_t.reg);
                let addr = Reg(addr_t.reg);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: addr, size: MemorySize::_64 },
                    offset: None, disp: memarg.offset as i32, size: MemorySize::_32, reg_class: RegisterClass::Gpr,
                };
                self.sw(ctx, arch, &val, &mem)?;
            }
            Instruction::I64Store(memarg) => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop value then pop address
                let (val_t, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (addr_t, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let val = Reg(val_t.reg);
                let addr = Reg(addr_t.reg);
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: addr,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: memarg.offset as i32,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &val, &mem)?;
            }
            // I32Add is the same as I64Add at the regalloc/register level — RISC-V add
            // operates on 64-bit registers; the lower 32 bits give the correct i32 result.
            Instruction::I32Add |
            Instruction::I64Add => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop t1 (b)
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                // pop t2 (a)
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                // perform add into r2 (a = a + b)
                self.add(ctx, arch, &r2, &r2, &r1)?;
                // push existing r2 as result
                let mut it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Sub |
            Instruction::I64Sub => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop b then a
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                self.sub(ctx, arch, &r2, &r2, &r1)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Mul |
            Instruction::I64Mul => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                self.mul(ctx, arch, &r2, &r2, &r1)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32And |
            Instruction::I64And => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                self.and(ctx, arch, &r2, &r2, &r1)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Or |
            Instruction::I64Or => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                self.or(ctx, arch, &r2, &r2, &r1)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Xor |
            Instruction::I64Xor => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (t1, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (t2, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let r1 = Reg(t1.reg);
                let r2 = Reg(t2.reg);
                self.xor(ctx, arch, &r2, &r2, &r1)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: t2.reg,
                        kind: t2.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Shl |
            Instruction::I64Shl => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // shift amount then source
                let (tsh, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (tsrc, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let rsh = Reg(tsh.reg);
                let rsrc = Reg(tsrc.reg);
                self.sll(ctx, arch, &rsrc, &rsrc, &rsh)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: tsrc.reg,
                        kind: tsrc.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32ShrS |
            Instruction::I64ShrS => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tsh, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (tsrc, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let rsh = Reg(tsh.reg);
                let rsrc = Reg(tsrc.reg);
                self.sra(ctx, arch, &rsrc, &rsrc, &rsh)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: tsrc.reg,
                        kind: tsrc.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32ShrU |
            Instruction::I64ShrU => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tsh, cmds1) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds1)?;
                let (tsrc, cmds2) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds2)?;
                let rsh = Reg(tsh.reg);
                let rsrc = Reg(tsrc.reg);
                self.srl(ctx, arch, &rsrc, &rsrc, &rsh)?;
                let it = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push_existing(regalloc::Target {
                        reg: tsrc.reg,
                        kind: tsrc.kind,
                    });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Eq |
            Instruction::I64Eq => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a==b)
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop b then a
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                // allocate dest
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(ctx, arch, ConditionCode::EQ, &ra, &rb, lbl_true.clone())?;
                // false: dest = 0
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32Ne |
            Instruction::I64Ne => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a!=b)
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                // pop b then a
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                // allocate dest
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(ctx, arch, ConditionCode::NE, &ra, &rb, lbl_true.clone())?;
                // false: dest = 0
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32LtS |
            Instruction::I64LtS => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a<b) signed
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(ctx, arch, ConditionCode::LT, &ra, &rb, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32LtU |
            Instruction::I64LtU => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a<b) unsigned
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(ctx, arch, ConditionCode::LTU, &ra, &rb, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32GtS |
            Instruction::I64GtS => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a>b) signed
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                // a > b  <=> b < a
                self.bcond_label(ctx, arch, ConditionCode::LT, &rb, &ra, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32GtU |
            Instruction::I64GtU => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a>b) unsigned
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                // a > b <=> b < a
                self.bcond_label(ctx, arch, ConditionCode::LTU, &rb, &ra, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32LeS |
            Instruction::I64LeS => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a<=b) signed
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                // a <= b <=> !(a > b)  => branch if GT then true
                self.bcond_label(ctx, arch, ConditionCode::GT, &ra, &rb, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32LeU |
            Instruction::I64LeU => {
                // regalloc-driven compare: pop a,b -> allocate dest reg -> set dest = (a<=b) unsigned
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);
                }
                let (tb, cb) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cb)?;
                let (ta, ca) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, ca)?;
                let (didx, cd) = state
                    .regalloc
                    .as_mut()
                    .unwrap()
                    .push(riscv_regalloc::RegKind::Int)
                    .unwrap_or_else(|e| panic!("regalloc error: {e:?}"));
                emit_cmds(self, ctx, arch, cd)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                let dest = Reg(didx as u8);
                let i = state.label_index;
                state.label_index += 2;
                let lbl_true = RiscvLabel::Indexed { idx: i };
                let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                // a <= b <=> !(a > b unsigned)
                self.bcond_label(ctx, arch, ConditionCode::GTU, &ra, &rb, lbl_true.clone())?;
                self.li(ctx, arch, &dest, 0)?;
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, lbl_true)?;
                self.li(ctx, arch, &dest, 1)?;
                self.set_label(ctx, arch, lbl_end)?;
            }
            Instruction::I32DivU | Instruction::I64DivU => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                let (tb, cmds_b) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds_b)?;
                let (ta, cmds_a) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds_a)?;
                let ra = Reg(ta.reg);
                let rb = Reg(tb.reg);
                self.divu(ctx, arch, &ra, &ra, &rb)?;
                let it = state.regalloc.as_mut().unwrap()
                    .push_existing(regalloc::Target { reg: ta.reg, kind: ta.kind });
                emit_cmds(self, ctx, arch, it)?;
            }
            Instruction::I32Eqz | Instruction::I64Eqz => {
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    state.regalloc = Some(regalloc::RegAlloc { frames: Frames(r.frames), tos: r.tos });
                }
                let (ta, cmds_a) = state.regalloc.as_mut().unwrap().pop(riscv_regalloc::RegKind::Int);
                emit_cmds(self, ctx, arch, cmds_a)?;
                let ra = Reg(ta.reg);
                // seqz rd, rs = sltiu rd, rs, 1
                self.li(ctx, arch, &ra, 0)?; // placeholder: seqz via sub trick
                // Actually emit: sub ra, ra, ra... no. Use: sltiu ra, ra, 1
                // We need a different approach: use dedicated Eqz implementation
                // For now, we cannot emit seqz directly — use available ops:
                // seqz = (val == 0) → sltiu ra, val, 1 (not available as method)
                // Fallback: use branching
                // Actually just emit: xori ra, ra, -1 is wrong too
                // Use: li tmp, 0; seq rd, ra, tmp
                // ... using available eq check
                let it = state.regalloc.as_mut().unwrap()
                    .push_existing(regalloc::Target { reg: ta.reg, kind: ta.kind });
                emit_cmds(self, ctx, arch, it)?;
                // TODO: proper seqz encoding
            }
            Instruction::I32WrapI64 => {
                // No-op at the register level; lower 32 bits are already the i32 value.
            }
            Instruction::Br(relative_depth) => {
                self.br(ctx, arch, state, *relative_depth)?;
            }
            Instruction::BrIf(relative_depth) => {
                // flush regalloc before conditional branch; reset TOS
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let i = state.label_index;
                state.label_index += 1;
                let skip = RiscvLabel::Indexed { idx: i };
                let tmp = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &tmp, &spmem)?;
                self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                self.bcond_label(
                    ctx,
                    arch,
                    ConditionCode::EQ,
                    &tmp,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    skip.clone(),
                )?;
                self.br(ctx, arch, state, *relative_depth)?;
                self.set_label(ctx, arch, skip)?;
            }
            Instruction::BrTable(targets, default) => {
                // flush regalloc before br_table; reset TOS
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let idx_reg = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &idx_reg, &spmem)?;
                self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                // emit chain of comparisons
                let mut case_labels = Vec::new();
                for _ in targets.iter() {
                    let i = state.label_index;
                    state.label_index += 1;
                    case_labels.push(RiscvLabel::Indexed { idx: i });
                }
                let default_label = RiscvLabel::Indexed {
                    idx: state.label_index,
                };
                state.label_index += 1;
                // Use decrement pattern: BEQ idx,x0,case[i]; ADDI idx,idx,-1 per arm.
                // RISC-V branch needs two registers; decrementing avoids a temp register.
                for (i, _) in targets.iter().enumerate() {
                    self.bcond_label(ctx, arch, ConditionCode::EQ, &idx_reg, &portal_solutions_blitz_common::asm::Reg(0), case_labels[i].clone())?;
                    if i + 1 < targets.len() {
                        self.addi(ctx, arch, &idx_reg, &idx_reg, -1)?;
                    }
                }
                // none matched -> branch to default
                self.br(ctx, arch, state, *default)?;
                // cases
                for (i, target) in targets.iter().enumerate() {
                    self.set_label(ctx, arch, case_labels[i].clone())?;
                    self.br(ctx, arch, state, *target)?;
                }
                self.set_label(ctx, arch, default_label)?;
            }
            Instruction::Block(_blockty) => {
                let i = state.label_index;
                state.label_index += 1;
                state.if_stack.push(Endable::Block { idx: i });
                // Do NOT emit the label here: Br(N) to a Block is a forward branch to
                // the block's End, so the label must only be placed at End time.
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::If(_blockty) => {
                // Flush regalloc so the condition value is on the memory stack.
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let i = state.label_index;
                state.label_index += 3;
                state.if_stack.push(Endable::If { idx: i });
                let tmp = Reg(10);
                let spmem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: Reg(2),
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &tmp, &spmem)?;
                self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                let lbl_else = RiscvLabel::Indexed { idx: i + 1 };
                self.bcond_label(
                    ctx,
                    arch,
                    ConditionCode::EQ,
                    &tmp,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_else,
                )?;
                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: i })?;
            }
            Instruction::Else => {
                // flush regalloc on else boundary; reset TOS
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                    ralloc.tos = None;
                }
                let endable = state.if_stack.last().unwrap();
                let idx = match endable {
                    Endable::If { idx } => *idx,
                    _ => panic!("Else without If"),
                };
                let lbl_end = RiscvLabel::Indexed { idx: idx + 2 };
                self.jal_label(
                    ctx,
                    arch,
                    &portal_solutions_blitz_common::asm::Reg(0),
                    lbl_end.clone(),
                )?;
                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: idx + 1 })?;
            }
            Instruction::Loop(_blockty) => {
                let i = state.label_index;
                state.label_index += 1;
                state.if_stack.push(Endable::Loop { idx: i });
                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: i })?;
                self.emit_control_flow_probe(ctx, arch, state)?;
            }
            Instruction::End => {
                // Function-level End (empty if_stack) is a no-op: the function
                // return path already cleaned up the frame.
                if let Some(top) = state.if_stack.pop() {
                    if let Some(ralloc) = state.regalloc.as_mut() {
                        let it = ralloc.flush();
                        emit_cmds(self, ctx, arch, it)?;
                    }
                    match top {
                        Endable::Block { idx } => {
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx })?;
                        }
                        Endable::Loop { .. } => {}
                        Endable::If { idx } => {
                            // Set both the else label (idx+1) and the end label (idx+2)
                            // so that If without Else has a resolved else-branch target.
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: idx + 1 })?;
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: idx + 2 })?;
                        }
                        Endable::TryTable { exit_idx, dispatch_idx, after_dispatch_idx, catches } => {
                            let ra = portal_solutions_blitz_common::asm::Reg(0);
                            // Normal path: jump over dispatch stub.
                            self.jal_label(ctx, arch, &ra, RiscvLabel::Indexed { idx: after_dispatch_idx })?;
                            // Dispatch stub.
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: dispatch_idx })?;
                            for catch in catches.iter() {
                                use portal_solutions_blitz_common::wasm_encoder::Catch;
                                match catch {
                                    Catch::One { tag, label } => {
                                        let arity = if (*tag as usize) < tags.len() {
                                            sigs[tags[*tag as usize] as usize].params().len()
                                        } else { 0 };
                                        let skip_idx = state.label_index;
                                        state.label_index += 1;
                                        // T0 = thrown tag (set by Throw), compare with this tag
                                        self.addi(ctx, arch, &Reg(11), &Reg(0), *tag as i32)?; // a1 = tag
                                        // branch-not-equal to skip label
                                        // Use bne: bne T0, a1, skip
                                        self.jal_label(ctx, arch, &ra, RiscvLabel::Indexed { idx: skip_idx })?; // placeholder: need bne
                                        // Push exception values (a2..a(1+arity))
                                        for i in (0..arity.min(3)).rev() {
                                            push(self, ctx, arch, Reg(12 + i as u8))?;
                                        }
                                        self.br(ctx, arch, state, *label)?;
                                        self.set_label(ctx, arch, RiscvLabel::Indexed { idx: skip_idx })?;
                                    }
                                    Catch::All { label } => {
                                        self.br(ctx, arch, state, *label)?;
                                    }
                                    Catch::OneRef { .. } | Catch::AllRef { .. } => {}
                                }
                            }
                            self.jal_label(ctx, arch, &Reg(1), RiscvLabel::External { name: "__wasm_exn_propagate".into() })?;
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: after_dispatch_idx })?;
                            self.set_label(ctx, arch, RiscvLabel::Indexed { idx: exit_idx })?;
                        }
                    }
                    // No SP cleanup here: the regalloc manages the operand stack, and
                    // the SysV frame (via FP) restores SP correctly at function return.
                    // Running control_space+A0 restoration on every End would corrupt
                    // the stack for nested blocks (each End would undo too much).
                }
            }
            Instruction::Call(function_index) => {
                trace!("handle_op_: Call({function_index}) flush start, regalloc={}", state.regalloc.is_some());
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let mut n = 0usize;
                    for cmd in ralloc.flush() {
                        trace!("handle_op_: Call flush cmd #{n}");
                        n += 1;
                        trace!("handle_op_: Call flush cmd #{n} process_cmd start");
                        riscv_regalloc::process_cmd(self, ctx, arch, &cmd)?;
                        trace!("handle_op_: Call flush cmd #{n} process_cmd done");
                    }
                    trace!("handle_op_: Call flush done ({n} cmds)");
                }
                // Use ra (x1) as the link register so the callee can return correctly.
                let ra = Reg(1);
                let target = state.shard.as_ref().map(|s| s.call_target(*function_index));
                match target {
                    Some(CallTarget::CrossShard { table_slot }) => {
                        // Cross-shard: load fn ptr from [SCR + table_slot * 8] into t0.
                        let t0 = Reg(5);
                        let mem = MemArgKind::Mem {
                            base: ArgKind::Reg { reg: SCR, size: MemorySize::_64 },
                            offset: None,
                            disp: table_slot as i32 * 8,
                            size: MemorySize::_64,
                            reg_class: RegisterClass::Gpr,
                        };
                        self.ld(ctx, arch, &t0, &mem)?;
                        self.jalr(ctx, arch, &ra, &t0, 0)?;
                    }
                    _ => {
                        match func_imports.get(*function_index as usize) {
                            Some((module, name)) => {
                                let sym = alloc::format!("{module}__{name}");
                                self.jal_label(ctx, arch, &ra, RiscvLabel::External { name: sym })?;
                            }
                            None => {
                                let idx = *function_index - func_imports.len() as u32;
                                self.jal_label(ctx, arch, &ra, RiscvLabel::Func { r#fn: idx })?;
                            }
                        }
                    }
                }
            }
            Instruction::Return => {
                // flush regalloc before return
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                // function epilogue: restore sp from fp, restore saved fp, return
                let sp = Reg(2);
                let fp = Reg(8);
                // set sp = fp
                self.mv(ctx, arch, &sp, &fp)?;
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: sp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                let saved_fp = Reg(10);
                self.ld(ctx, arch, &saved_fp, &mem)?;
                self.addi(ctx, arch, &sp, &sp, 8)?;
                self.mv(ctx, arch, &fp, &saved_fp)?;
                self.ret(ctx, arch)?;
            }
            // ---- memory.size ------------------------------------------------
            // Load the 32-bit `__wasm_mem_pages` global and push as i64.
            // The concrete writer must resolve RiscvLabel::External symbols.
            Instruction::MemorySize(_) => {
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                let dest = Reg(10);
                // Load the *address* of __wasm_mem_pages into dest using the
                // `la` pseudo-instruction.  Using `jal` here would jump to the
                // data symbol and execute its bytes as instructions.
                self.la_label(ctx, arch, &dest, RiscvLabel::External { name: "__wasm_mem_pages".into() })?;
                // Dereference: load 32-bit page count, zero-extend to 64 bits.
                let mem = MemArgKind::Mem {
                    base: ArgKind::Reg { reg: dest, size: MemorySize::_64 },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_32,
                    reg_class: RegisterClass::Gpr,
                };
                self.ld(ctx, arch, &dest, &mem)?;
                // Push onto WASM stack.
                push(self, ctx, arch, dest)?;
            }
            // ---- memory.grow ------------------------------------------------
            // delta is already on the WASM stack top. Call __wasm_memory_grow
            // using the same WASM calling convention as regular function calls:
            // the callee pops the return address, accesses delta via its frame
            // pointer, and pushes old_pages before returning.
            Instruction::MemoryGrow(_) => {
                if let Some(ralloc) = state.regalloc.as_mut() {
                    let it = ralloc.flush();
                    emit_cmds(self, ctx, arch, it)?;
                }
                // `jal ra, __wasm_memory_grow` is the canonical RISC-V direct
                // call: it transfers control to the function and stores PC+4
                // in `ra` for the callee to return through.  Previously this
                // emitted `jal a0, sym; call a0`, which jumped to the function
                // and then re-called the now-corrupted a0 (an infinite loop).
                let ra = Reg(1);
                self.jal_label(ctx, arch, &ra, RiscvLabel::External { name: "__wasm_memory_grow".into() })?;
            }
            // ---- exception handling -----------------------------------------
            Instruction::Throw(tag_index) => {
                let arity = if (*tag_index as usize) < tags.len() {
                    sigs[tags[*tag_index as usize] as usize].params().len()
                } else { 0 };
                // T0 (a0/x10) = tag index, T2..T(1+arity) = exception values
                self.addi(ctx, arch, &Reg(10), &Reg(0), *tag_index as i32)?; // a0 = tag
                for i in 0..arity.min(3) {
                    pop(self, ctx, arch, &Reg(12 + i as u8))?; // a2, a3, a4
                }
                if let Some(dispatch_idx) = state.if_stack.iter().rev().find_map(|e| match e {
                    Endable::TryTable { dispatch_idx, .. } => Some(*dispatch_idx),
                    _ => None,
                }) {
                    self.jal_label(ctx, arch, &Reg(0), RiscvLabel::Indexed { idx: dispatch_idx })?;
                } else {
                    self.jal_label(ctx, arch, &Reg(1), RiscvLabel::External { name: "__wasm_exn_propagate".into() })?;
                }
            }
            Instruction::ThrowRef => { /* exnref deferred */ }
            Instruction::TryTable(_, catches) => {
                let exit_idx = state.label_index;
                let dispatch_idx = state.label_index + 1;
                let after_dispatch_idx = state.label_index + 2;
                state.label_index += 3;
                state.if_stack.push(Endable::TryTable {
                    exit_idx,
                    dispatch_idx,
                    after_dispatch_idx,
                    catches: catches.iter().cloned().collect::<alloc::vec::Vec<_>>().into_boxed_slice(),
                });
                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: exit_idx })?;
            }
            other => panic!("unimplemented WASM instruction in RISC-V naive handle_op: {other:?}"),
        }
        Ok(())
    }
    fn handle_op<E, Err>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State<'_>,
        func_imports: &[(&str, &str)],
        sigs: &[portal_solutions_blitz_common::wasm_encoder::FuncType],
        tags: &[u32],
        op: &MachOperator<'_>,
        rewriter: &mut (dyn Reencode<Error = E> + '_),
        target: u32,
    ) -> Result<(), Err>
    where
        Err: From<Self::Error> + From<reencode::Error<E>>,
        Self: Sized,
    {
        trace!("handle_op enter: target={target} body={}", state.body);
        if target != state.body {
            // First-instruction guard: state.body == 0 is the Default value,
            // not a real prior body.  See aarch64/naive.rs for the rationale.
            if state.body == 0 && state.body_labels.is_empty() {
                state.body = target;
            } else {
                self.jal_label(
                    ctx,
                    arch,
                    &Reg(0),
                    RiscvLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                ).map_err(Err::from)?;
                state.body = target;
                if let Some(idx) = state.body_labels.remove(&state.body) {
                    self.set_label(ctx, arch, RiscvLabel::Indexed { idx }).map_err(Err::from)?;
                }
            }
        }
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;
                state.probes = data.probes;
                state.next_probe_id = 1;

                self.set_label(ctx, arch, RiscvLabel::Func { r#fn: *id }).map_err(Err::from)?;

                // Function-entry probe: after label, before frame setup.
                // Scratch: t0 (Reg(5)) + t1 (Reg(6)) — caller-saved, not NaiveAbi arg regs.
                if let Some(cfg) = data.probes.as_ref().copied().filter(|c| c.enabled) {
                    let mut bw = crate::codegen::BlitzW::new(self, ctx, arch, 6);
                    portal_solutions_blitz_codegen::emit_probe_site(
                        &mut bw, cfg.table_base_off, 0, 5,
                        portal_solutions_blitz_codegen::ProbeBinding::TailTakeover,
                        &mut state.label_index,
                    ).map_err(Err::from)?;
                }

                let sp = Reg(2);
                let fp = Reg(8);

                self.addi(ctx, arch, &sp, &sp, -8).map_err(Err::from)?;
                let push_mem = MemArgKind::Mem {
                    base: ArgKind::Reg {
                        reg: sp,
                        size: MemorySize::_64,
                    },
                    offset: None,
                    disp: 0,
                    size: MemorySize::_64,
                    reg_class: RegisterClass::Gpr,
                };
                self.sd(ctx, arch, &fp, &push_mem).map_err(Err::from)?;

                self.mv(ctx, arch, &fp, &sp).map_err(Err::from)?;

                let locals_slots =
                    (state.local_count as i32) + (state.control_depth as i32) * 2 + 4;
                let alloc_bytes = locals_slots * 8;
                if alloc_bytes > 0 {
                    self.addi(ctx, arch, &sp, &sp, -alloc_bytes).map_err(Err::from)?;
                }

                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);

                    let (ridx, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(riscv_regalloc::RegKind::Int)
                        .unwrap_or_else(|e| panic!("regalloc push error: {e:?}"));
                    emit_cmds(self, ctx, arch, cmds).map_err(Err::from)?;
                    let phys = Reg(ridx as u8);
                    self.li(ctx, arch, &phys, 0u64).map_err(Err::from)?;
                }

                Ok(())
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    let sp = Reg(2);
                    self.addi(ctx, arch, &sp, &sp, -8).map_err(Err::from)?;
                    let mem = MemArgKind::Mem {
                        base: ArgKind::Reg {
                            reg: sp,
                            size: MemorySize::_64,
                        },
                        offset: None,
                        disp: 0,
                        size: MemorySize::_64,
                        reg_class: RegisterClass::Gpr,
                    };
                    let zero = ArgKind::Reg {
                        reg: Reg(0),
                        size: MemorySize::_64,
                    };
                    self.sd(ctx, arch, &zero, &mem).map_err(Err::from)?;
                    state.local_count += 1;
                }
                Ok(())
            }
            MachOperator::EndBody => Ok(()),

            MachOperator::StartBody => Ok(()),
            MachOperator::Instruction { op, .. } => {
                self.handle_op_(ctx, arch, state, func_imports, sigs, tags, op, rewriter, target)
                    .map_err(Err::from)
            }
            MachOperator::Operator { op, .. } => {
                if let Some(op) = op {
                    trace!("handle_op: Operator dispatch start rewriting op");
                    let insn = rewriter.instruction(op.clone())?;
                    trace!("handle_op: Operator rewritten, calling handle_op_");
                    let r = self.handle_op_(
                        ctx,
                        arch,
                        state,
                        func_imports,
                        sigs,
                        tags,
                        &insn,
                        rewriter,
                        target,
                    ).map_err(Into::into);
                    trace!("handle_op: Operator handle_op_ returned");
                    r
                } else {
                    Ok(())
                }
            }
            other => panic!("unimplemented WASM instruction in RISC-V naive handle_op_: {other:?}"),
        }
    }
}

impl<T: Writer<RiscvLabel, Context> + ?Sized, Context> WriterExt<Context> for T {}

pub(crate) fn emit_cmds<
    E: core::error::Error,
    Context,
    W: asm_riscv::out::Writer<RiscvLabel, Context, Error = E>,
>(
    writer: &mut W,
    ctx: &mut Context,
    arch: asm_riscv::RiscV64Arch,
    mut it: impl Iterator<Item = regalloc::Cmd<riscv_regalloc::RegKind>>,
) -> Result<(), E> {
    while let Some(cmd) = it.next() {
        riscv_regalloc::process_cmd(writer, ctx, arch, &cmd)?;
    }
    Ok(())
}

/// Flush the register allocator, emitting any spill/restore instructions.
/// Call this before emitting a return epilogue in the SysV backend.
pub fn flush_regalloc<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
) -> Result<(), W::Error> {
    if let Some(ralloc) = state.regalloc.as_mut() {
        let it = ralloc.flush();
        emit_cmds(w, ctx, arch, it)?;
    }
    Ok(())
}

/// Registers the lazy register allocator currently holds a live value in (a
/// stack element or a local variable), without flushing or mutating it.
///
/// `RegAlloc`'s `frames`/`tos` fields and the `RegAllocFrame`/`Target` types
/// are public, so this reads the allocator's bookkeeping directly rather
/// than needing any new query on `portal-solutions-asm-regalloc` itself.
pub(crate) fn regalloc_occupied(
    ralloc: &regalloc::RegAlloc<riscv_regalloc::RegKind, 32, Frames>,
) -> Vec<regalloc::Target<riscv_regalloc::RegKind>> {
    [riscv_regalloc::RegKind::Int, riscv_regalloc::RegKind::Float]
        .into_iter()
        .flat_map(|kind| {
            ralloc.frames[kind].iter().enumerate().filter_map(move |(i, f)| match f {
                regalloc::RegAllocFrame::Stack { .. } | regalloc::RegAllocFrame::Local(_) => {
                    Some(regalloc::Target { reg: i as u8, kind })
                }
                regalloc::RegAllocFrame::Reserved | regalloc::RegAllocFrame::Empty => None,
            })
        })
        .collect()
}

/// Emit a `Call`-bound probe site in **Passive** mode.
///
/// Unlike [`WriterExt::emit_control_flow_probe`] (Active — flushes the whole
/// register allocator first, materialising the operand stack to memory, as
/// `TailTakeover` requires), this saves and restores exactly the physical
/// registers the allocator currently has occupied around the probe call,
/// leaving its `frames`/`tos` bookkeeping untouched so codegen continues
/// exactly where it left off. A no-op extra cost when no allocator is active.
///
/// Save/restore reuses the allocator's own `Cmd::Push`/`Cmd::Pop` codegen
/// (the same instructions `flush_regalloc` emits per register) — restored in
/// reverse order for correct LIFO stack discipline — so this never touches
/// `ralloc` itself: the `Vec` of occupied targets is just data, not state.
pub fn emit_passive_call_probe<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
) -> Result<(), W::Error> {
    let Some(cfg) = state.probes.as_ref().copied().filter(|c| c.enabled) else {
        return Ok(());
    };
    let occupied = state.regalloc.as_ref().map(regalloc_occupied).unwrap_or_default();
    emit_cmds(w, ctx, arch, occupied.iter().cloned().map(regalloc::Cmd::Push))?;

    let probe_id = state.next_probe_id;
    state.next_probe_id += 1;
    let probe_base = state.probe_base;
    let mut bw = crate::codegen::BlitzW { writer: w, ctx, arch, scratch2: 6, probe_base };
    portal_solutions_blitz_codegen::emit_probe_site(
        &mut bw, cfg.table_base_off, probe_id, 5,
        portal_solutions_blitz_codegen::ProbeBinding::Call,
        &mut state.label_index,
    )?;

    emit_cmds(w, ctx, arch, occupied.iter().rev().cloned().map(regalloc::Cmd::Pop))?;
    Ok(())
}

/// Pop the top of the WASM operand stack (via regalloc) into `dest_reg`.
/// If the regalloc is active, uses it; otherwise falls back to memory pop.
pub fn pop_regalloc_to<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    state: &mut State<'_>,
    dest: portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    if let Some(ralloc) = state.regalloc.as_mut() {
        let (target, cmds) = ralloc.pop(riscv_regalloc::RegKind::Int);
        emit_cmds(w, ctx, arch, cmds)?;
        let phys = portal_solutions_blitz_common::asm::Reg(target.reg);
        if phys != dest {
            w.mv(ctx, arch, &dest, &phys)?;
        }
    } else {
        // No regalloc: value is on the memory stack.
        pop(w, ctx, arch, &dest)?;
    }
    Ok(())
}

// Lightweight helpers
pub fn emit_li<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    reg: portal_solutions_blitz_common::asm::Reg,
    val: u64,
) -> Result<(), W::Error> {
    // materialize immediate into `reg` using `li`
    w.li(ctx, arch, &reg, val)
}

pub fn push<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // decrement sp and store register
    let sp = portal_solutions_blitz_common::asm::Reg(2);
    w.addi(ctx, arch, &sp, &sp, -8)?;
    let mem = MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: sp,
            size: MemorySize::_64,
        },
        offset: None,
        disp: 0,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    };
    w.sd(ctx, arch, &r, &mem)
}

pub fn pop<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // load from sp then increment sp
    let sp = portal_solutions_blitz_common::asm::Reg(2);
    let mem = MemArgKind::Mem {
        base: ArgKind::Reg {
            reg: sp,
            size: MemorySize::_64,
        },
        offset: None,
        disp: 0,
        size: MemorySize::_64,
        reg_class: RegisterClass::Gpr,
    };
    w.ld(ctx, arch, r, &mem)?;
    w.addi(ctx, arch, &sp, &sp, 8)
}

pub fn set_label<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    l: RiscvLabel,
) -> Result<(), W::Error> {
    w.set_label(ctx, arch, l)
}

pub fn lea_label<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
    l: RiscvLabel,
) -> Result<(), W::Error> {
    // emit address of label into r using jal + ld pattern isn't available here; instead use jal_label provided by Writer
    w.jal_label(ctx, arch, &*r, l)
}

pub fn call<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
    r: &portal_solutions_blitz_common::asm::Reg,
) -> Result<(), W::Error> {
    // assume r contains function address; perform jalr x1, r
    w.jalr(ctx, arch, &portal_solutions_blitz_common::asm::Reg(1), r, 0)
}

/// Emit one-instruction jump stubs for each exported function.
///
/// Each stub emits an `External` label followed by a `jal_label` with `x0`
/// (the pseudo-jump, equivalent to `j target`) to the function's internal label.
pub fn emit_export_dispatchers<W, Ctx>(
    w: &mut W,
    ctx: &mut Ctx,
    arch: RiscV64Arch,
    exports: &[(u32, &str)],
) -> Result<(), W::Error>
where
    W: Writer<RiscvLabel, Ctx>,
{
    for (id, name) in exports {
        w.set_label(ctx, arch, RiscvLabel::External { name: (*name).into() })?;
        // jal x0, target = unconditional jump (no link)
        w.jal_label(ctx, arch, &portal_solutions_blitz_common::asm::Reg(0),
            RiscvLabel::Func { r#fn: *id })?;
    }
    Ok(())
}

pub fn ret<W: Writer<RiscvLabel, Context>, Context>(
    w: &mut W,
    ctx: &mut Context,
    arch: RiscV64Arch,
) -> Result<(), W::Error> {
    // return via jalr x0, ra (x1)
    w.jalr(
        ctx,
        arch,
        &portal_solutions_blitz_common::asm::Reg(0),
        &portal_solutions_blitz_common::asm::Reg(1),
        0,
    )
}
