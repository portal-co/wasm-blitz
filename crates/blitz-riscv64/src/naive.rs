//! Naive RISC-V codegen (incremental port)

#![allow(dead_code)]

use crate::RiscvLabel;
use alloc::vec::Vec;
use portal_solutions_asm_riscv64::ConditionCode;
use portal_solutions_asm_riscv64::RiscV64Arch;
use portal_solutions_asm_riscv64::out::Writer;

use portal_solutions_blitz_common::asm::Reg;
use portal_solutions_blitz_common::ops::MachOperator;
use portal_solutions_blitz_common::wasm_encoder;
use portal_solutions_blitz_common::wasm_encoder::reencode::Reencode;

use portal_pc_asm_common::types::mem::MemorySize;
use portal_solutions_asm_riscv64::RegisterClass;
use portal_solutions_asm_riscv64::out::arg::{ArgKind, MemArgKind};

use core::ops::{Index, IndexMut};
use portal_solutions_asm_regalloc as regalloc;
use portal_solutions_asm_riscv64 as asm_riscv;
use portal_solutions_asm_riscv64::regalloc as riscv_regalloc;

#[derive(Default)]
pub struct State {
    pub label_index: usize,
    pub local_count: usize,
    pub num_returns: usize,
    pub control_depth: usize,
    pub if_stack: Vec<Endable>,
    pub regalloc: Option<regalloc::RegAlloc<riscv_regalloc::RegKind, 32, Frames>>,
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
}

pub trait WriterExt<Context>: Writer<RiscvLabel, Context> {
    fn br(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State,
        relative_depth: u32,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        // flush regalloc before branching
        if let Some(ralloc) = state.regalloc.as_mut() {
            let it = ralloc.flush();
            emit_cmds(self,ctx, arch, it)?;
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
                }
            }
            depth -= 1;
        }
        Ok(())
    }

    fn handle_op<E>(
        &mut self,
        ctx: &mut Context,
        arch: RiscV64Arch,
        state: &mut State,
        func_imports: &[(&str, &str)],
        op: &MachOperator<'_>,
        _rewriter: &mut (dyn Reencode<Error = E> + '_),
    ) -> Result<(), Self::Error>
    where
        wasm_encoder::reencode::Error<E>: Into<Self::Error>,
        Self::Error: From<core::fmt::Error>,
        Self: Sized,
    {
        match op {
            MachOperator::StartFn { id, data } => {
                state.local_count = data.num_params;
                state.num_returns = data.num_returns;
                state.control_depth = data.control_depth;

                self.set_label(ctx, arch, RiscvLabel::Func { r#fn: *id })?;

                let sp = Reg(2);
                let fp = Reg(8);

                // push fp
                self.addi(ctx, arch, &sp, &sp, -8)?;
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
                self.sd(ctx, arch, &fp, &push_mem)?;

                // set fp = sp
                self.mv(ctx, arch, &fp, &sp)?;

                // allocate locals
                let locals_slots =
                    (state.local_count as i32) + (state.control_depth as i32) * 2 + 4;
                let alloc_bytes = locals_slots * 8;
                if alloc_bytes > 0 {
                    self.addi(ctx, arch, &sp, &sp, -alloc_bytes)?;
                }

                // Ensure regalloc initialized (smoke test)
                if state.regalloc.is_none() {
                    let r = riscv_regalloc::init_regalloc::<32>(arch);
                    let new = regalloc::RegAlloc {
                        frames: Frames(r.frames),
                        tos: r.tos,
                    };
                    state.regalloc = Some(new);

                    // Small smoke: push an int register and emit its cmds
                    let (ridx, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(riscv_regalloc::RegKind::Int)
                        .map_err(|_| core::fmt::Error)?;
                    emit_cmds(self,ctx, arch, cmds)?;
                    // emit a simple li into the allocated phys reg
                    let phys = Reg(ridx as u8);
                    self.li(ctx, arch, &phys, 0u64)?;
                }

                Ok(())
            }
            MachOperator::Local { count, ty } => {
                for _ in 0..*count {
                    // push x0
                    let sp = Reg(2);
                    self.addi(ctx, arch, &sp, &sp, -8)?;
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
                    self.sd(ctx, arch, &zero, &mem)?;
                    state.local_count += 1;
                }
                Ok(())
            }
            MachOperator::StartBody => {
                // push sp marker and reserve control slots
                let sp = Reg(2);
                let tmp = Reg(10);
                self.addi(ctx, arch, &sp, &sp, -8)?;
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
                self.sd(ctx, arch, &tmp, &mem)?;

                let control_space = (state.control_depth as i32) * 16;
                if control_space > 0 {
                    self.addi(ctx, arch, &sp, &sp, -control_space)?;
                }
                Ok(())
            }
            MachOperator::Instruction { op, .. } => {
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
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cmds)?;
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
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cmds)?;
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
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cmds)?;
                    }
                    Instruction::LocalSet(local_index) => {
                        if state.regalloc.is_none() {
                            let r = riscv_regalloc::init_regalloc::<32>(arch);
                            let new = regalloc::RegAlloc {
                                frames: Frames(r.frames),
                                tos: r.tos,
                            };
                            state.regalloc = Some(new);
                        }
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds)?;
                        let it = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop_local(riscv_regalloc::RegKind::Int, *local_index);
                        emit_cmds(self,ctx, arch, it)?;
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        // allocate dest reg for loaded value
                        let (didx, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        self.ld(ctx, arch, &dest, &mem)?;
                        // push the dest register as the value
                        let mut it =
                            state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .push_existing(regalloc::Target {
                                    reg: didx as u8,
                                    kind: riscv_regalloc::RegKind::Int,
                                });
                        emit_cmds(self,ctx, arch, it)?;
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (addr_t, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        // pop t2 (a)
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
                        let r1 = Reg(t1.reg);
                        let r2 = Reg(t2.reg);
                        // perform add into r2 (a = a + b)
                        self.add(ctx, arch, &r2, &r2, &r1)?;
                        // push existing r2 as result
                        let mut it =
                            state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .push_existing(regalloc::Target {
                                    reg: t2.reg,
                                    kind: t2.kind,
                                });
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (t2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (tsrc, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (tsrc, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cmds1)?;
                        let (tsrc, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, cmds2)?;
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
                        emit_cmds(self,ctx, arch, it)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        // allocate dest
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        self.bcond_label(ctx, arch, ConditionCode::EQ, &ra, &rb, lbl_true)?;
                        // false: dest = 0
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        // allocate dest
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        self.bcond_label(ctx, arch, ConditionCode::NE, &ra, &rb, lbl_true)?;
                        // false: dest = 0
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        self.bcond_label(ctx, arch, ConditionCode::LT, &ra, &rb, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        self.bcond_label(ctx, arch, ConditionCode::LTU, &ra, &rb, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        // a > b  <=> b < a
                        self.bcond_label(ctx, arch, ConditionCode::LT, &rb, &ra, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        // a > b <=> b < a
                        self.bcond_label(ctx, arch, ConditionCode::LTU, &rb, &ra, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        // a <= b <=> !(a > b)  => branch if GT then true
                        self.bcond_label(ctx, arch, ConditionCode::GT, &ra, &rb, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
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
                        emit_cmds(self,ctx, arch, cb)?;
                        let (ta, ca) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(riscv_regalloc::RegKind::Int);
                        emit_cmds(self,ctx, arch, ca)?;
                        let (didx, cd) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(riscv_regalloc::RegKind::Int)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self,ctx, arch, cd)?;
                        let ra = Reg(ta.reg);
                        let rb = Reg(tb.reg);
                        let dest = Reg(didx as u8);
                        let i = state.label_index;
                        state.label_index += 2;
                        let lbl_true = RiscvLabel::Indexed { idx: i };
                        let lbl_end = RiscvLabel::Indexed { idx: i + 1 };
                        // a <= b <=> !(a > b unsigned)
                        self.bcond_label(ctx, arch, ConditionCode::GTU, &ra, &rb, lbl_true)?;
                        self.li(ctx, arch, &dest, 0)?;
                        self.jal_label(
                            ctx,
                            arch,
                            &portal_solutions_blitz_common::asm::Reg(0),
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, lbl_true)?;
                        self.li(ctx, arch, &dest, 1)?;
                        self.set_label(ctx, arch, lbl_end)?;
                    }
                    Instruction::Br(relative_depth) => {
                        self.br(ctx, arch, state, *relative_depth)?;
                    }
                    Instruction::BrIf(relative_depth) => {
                        // flush regalloc before conditional branch
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
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
                            skip,
                        )?;
                        self.br(ctx, arch, state, *relative_depth)?;
                        self.set_label(ctx, arch, skip)?;
                    }
                    Instruction::BrTable(targets, default) => {
                        // flush regalloc before br_table
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
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
                        for (i, target) in targets.iter().enumerate() {
                            let lit: u64 = i as u64;
                            self.bcond_label(
                                ctx,
                                arch,
                                ConditionCode::EQ,
                                &idx_reg,
                                &lit,
                                case_labels[i],
                            )?;
                        }
                        // none matched -> branch to default
                        self.br(ctx, arch, state, *default)?;
                        // cases
                        for (i, target) in targets.iter().enumerate() {
                            self.set_label(ctx, arch, case_labels[i])?;
                            self.br(ctx, arch, state, *target)?;
                        }
                        self.set_label(ctx, arch, default_label)?;
                    }
                    Instruction::Block(_blockty) => {
                        let i = state.label_index;
                        state.label_index += 1;
                        state.if_stack.push(Endable::Block { idx: i });
                        self.set_label(ctx, arch, RiscvLabel::Indexed { idx: i })?;
                    }
                    Instruction::If(_blockty) => {
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
                        // flush regalloc on else boundary
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
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
                            lbl_end,
                        )?;
                        self.set_label(ctx, arch, RiscvLabel::Indexed { idx: idx + 1 })?;
                    }
                    Instruction::Loop(_blockty) => {
                        let i = state.label_index;
                        state.label_index += 1;
                        state.if_stack.push(Endable::Loop { idx: i });
                        self.set_label(ctx, arch, RiscvLabel::Indexed { idx: i })?;
                    }
                    Instruction::End => {
                        // flush regalloc on end boundary
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
                        }
                        match state.if_stack.pop().unwrap() {
                            Endable::Block { idx } => {
                                self.set_label(ctx, arch, RiscvLabel::Indexed { idx })?;
                            }
                            Endable::Loop { idx } => {
                                // no-op; loop already has label at start
                                //  self.set_label(ctx,arch, RiscvLabel::Indexed { idx })?;
                            }
                            Endable::If { idx } => {
                                self.set_label(ctx, arch, RiscvLabel::Indexed { idx: idx + 2 })?;
                            }
                        }
                        // restore control stack space if reserved
                        let control_space = (state.control_depth as i32) * 16;
                        if control_space > 0 {
                            self.addi(ctx, arch, &Reg(2), &Reg(2), control_space)?;
                        }
                        // pop sp marker
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
                        let tmp = Reg(10);
                        self.ld(ctx, arch, &tmp, &spmem)?;
                        self.addi(ctx, arch, &Reg(2), &Reg(2), 8)?;
                    }
                    Instruction::Call(function_index) => {
                        // flush regalloc before call
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
                        }
                        match func_imports.get(*function_index as usize) {
                            Some(("blitz", h)) if h.starts_with("hypercall") => {
                                // not implemented: hypercall path
                            }
                            _ => {
                                let function_index = *function_index - func_imports.len() as u32;
                                self.jal_label(
                                    ctx,
                                    arch,
                                    &portal_solutions_blitz_common::asm::Reg(10),
                                    RiscvLabel::Func {
                                        r#fn: function_index,
                                    },
                                )?;
                                self.call(ctx, arch, &portal_solutions_blitz_common::asm::Reg(10))?;
                            }
                        }
                    }
                    Instruction::Return => {
                        // flush regalloc before return
                        if let Some(ralloc) = state.regalloc.as_mut() {
                            let it = ralloc.flush();
                            emit_cmds(self,ctx, arch, it)?;
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
                    _ => {}
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl<T: Writer<RiscvLabel, Context> + ?Sized, Context> WriterExt<Context> for T {}

fn emit_cmds<
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
    w.sd(ctx,arch, &r, &mem)
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
