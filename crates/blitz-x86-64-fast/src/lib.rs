//! Faster x86-64 backend for wasm-blitz.
//!
//! This crate provides a higher-performance x86-64 backend that uses
//! `portal-solutions-asm-regalloc` for register allocation and flushing.
#![no_std]
extern crate alloc;
use alloc::format;
use alloc::vec::Vec;
use core::fmt::Display;

pub use portal_solutions_asm_x86_64::*;

use portal_solutions_blitz_common::{asm::Reg, ops::MachOperator};

/// The stack pointer register (RSP).
const RSP: Reg = Reg(4);

/// Label types for the optimized x86-64 backend.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum X64FastLabel {
    Indexed { idx: usize },
    Func { r#fn: u32 },
}

impl Display for X64FastLabel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            X64FastLabel::Indexed { idx } => write!(f, "_fast_idx_{idx}"),
            X64FastLabel::Func { r#fn } => write!(f, "fast_f{}", r#fn),
        }
    }
}

/// Fast backend writer trait.
pub trait FastWriter: portal_solutions_blitz_common::Label<X64FastLabel> {}
impl<T: portal_solutions_blitz_common::Label<X64FastLabel> + ?Sized> FastWriter for T {}

pub mod fast {
    use super::*;
    use alloc::format;
    use portal_solutions_asm_regalloc as regalloc;
    use portal_solutions_asm_x86_64::regalloc as x86_regalloc;
    use portal_solutions_asm_x86_64::{self as asm_x86, stack::StackManager};
    use portal_solutions_blitz_common::asm::common::mem::MemorySize;

    /// Errors produced by the fast backend.
    #[derive(Debug)]
    pub enum FastError {
        WriterBoxed,
        RegallocError,
    }
    impl From<core::fmt::Error> for FastError {
        fn from(_: core::fmt::Error) -> Self {
            FastError::RegallocError
        }
    }
    impl From<FastError> for alloc::string::String {
        fn from(e: FastError) -> Self {
            format!("{:?}", e)
        }
    }

    use core::ops::{Index, IndexMut};

    pub struct Frames(pub [[regalloc::RegAllocFrame<x86_regalloc::RegKind>; 32]; 2]);

    impl Index<x86_regalloc::RegKind> for Frames {
        type Output = [regalloc::RegAllocFrame<x86_regalloc::RegKind>; 32];
        fn index(&self, k: x86_regalloc::RegKind) -> &Self::Output {
            match k {
                x86_regalloc::RegKind::Int => &self.0[0],
                x86_regalloc::RegKind::Float => &self.0[1],
            }
        }
    }
    impl IndexMut<x86_regalloc::RegKind> for Frames {
        fn index_mut(&mut self, k: x86_regalloc::RegKind) -> &mut Self::Output {
            match k {
                x86_regalloc::RegKind::Int => &mut self.0[0],
                x86_regalloc::RegKind::Float => &mut self.0[1],
            }
        }
    }
    impl regalloc::Length for Frames {
        fn len(&self) -> usize {
            2
        }
    }

    pub struct State {
        pub local_count: usize,
        pub num_returns: usize,
        pub control_depth: usize,
        pub label_index: usize,
        pub if_stack: alloc::vec::Vec<Endable>,
        pub regalloc: Option<regalloc::RegAlloc<x86_regalloc::RegKind, 32, Frames>>,
        pub stack_manager: StackManager,
        pub body: u32,
        pub body_labels: alloc::collections::BTreeMap<u32, usize>,
    }

    enum Endable {
        Br,
        If { idx: usize },
    }

    impl Default for State {
        fn default() -> Self {
            Self {
                local_count: 0,
                num_returns: 0,
                control_depth: 0,
                label_index: 0,
                if_stack: alloc::vec::Vec::new(),
                regalloc: None,
                stack_manager: StackManager::new(),
                body: 0,
                body_labels: alloc::collections::BTreeMap::new(),
            }
        }
    }

    fn emit_cmds<
        E: core::error::Error,
        Context,
        W: asm_x86::out::Writer<X64FastLabel, Context, Error = E>,
    >(
        writer: &mut W,
        ctx: &mut Context,
        arch: asm_x86::X64Arch,
        mut it: impl Iterator<Item = regalloc::Cmd<x86_regalloc::RegKind>>,
        stack: &mut StackManager,
    ) -> Result<(), E> {
        while let Some(cmd) = it.next() {
            x86_regalloc::process_cmd(writer, ctx, arch, &cmd, Some(stack))?;
        }
        Ok(())
    }

    pub trait WriterExt<Context>: asm_x86::out::Writer<X64FastLabel, Context> {
        fn br(
            &mut self,
            ctx: &mut Context,
            arch: asm_x86::X64Arch,
            _state: &mut State,
            relative_depth: u32,
        ) -> Result<(), Self::Error>
        where
            Self: Sized,
        {
            // match naive backend br sequence
            let rsp_mem = asm_x86::out::arg::MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 8u32,
                size: MemorySize::_64,
                reg_class: asm_x86::RegisterClass::Gpr,
            };
            self.xchg(ctx, arch, &rsp_mem, &Reg::CTX)?;
            for _ in 0..=relative_depth {
                self.pop(ctx, arch, &Reg(0))?;
                self.pop(ctx, arch, &Reg(1))?;
            }
            let rsp_mem = asm_x86::out::arg::MemArgKind::Mem {
                base: RSP,
                offset: None,
                disp: 8u32,
                size: MemorySize::_64,
                reg_class: asm_x86::RegisterClass::Gpr,
            };
            self.xchg(ctx, arch, &rsp_mem, &Reg::CTX)?;
            self.mov(ctx, arch, &RSP, &Reg(1))?;
            self.jmp(ctx, arch, &Reg(0))?;
            Ok(())
        }

        fn handle_op(
            &mut self,
            ctx: &mut Context,
            arch: asm_x86::X64Arch,
            state: &mut State,
            func_imports: &[(&str, &str)],
            op: &portal_solutions_blitz_common::wasm_encoder::Instruction<'_>,
            target: u32,
        ) -> Result<(), Self::Error>
        where
            Self: Sized,
            Self::Error: From<core::fmt::Error>,
        {
            if target != state.body {
                self.jmp_label(
                    ctx,
                    arch,
                    X64FastLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )?;
                state.body = target;
                self.set_label(
                    ctx,
                    arch,
                    X64FastLabel::Indexed {
                        idx: *state.body_labels.entry(state.body).or_insert_with(|| {
                            state.label_index += 1;
                            return state.label_index - 1;
                        }),
                    },
                )?;
            }
            // Ensure regalloc is initialized per function
            if state.regalloc.is_none() {
                let r = x86_regalloc::init_regalloc::<32>(arch);
                let new = regalloc::RegAlloc {
                    frames: Frames(r.frames),
                    tos: r.tos,
                };
                state.regalloc = Some(new);
            }
            use portal_solutions_blitz_common::wasm_encoder::Instruction;
            match op {
                Instruction::I32Const(value) => {
                    {
                        let (r, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push(x86_regalloc::RegKind::Int)
                            .map_err(|_| {
                                // convert error
                                core::fmt::Error
                            })?;
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        let phys = Reg(r as u8);
                        self.mov64(ctx, arch, &phys, *value as u32 as u64)?;
                    }
                }
                Instruction::I64Const(value) => {
                    let (r, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(x86_regalloc::RegKind::Int)
                        .map_err(|_| core::fmt::Error)?;
                    emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                    let phys = Reg(r as u8);
                    self.mov64(ctx, arch, &phys, *value as u64)?;
                }
                Instruction::F32Const(value) => {
                    let (r, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(x86_regalloc::RegKind::Float)
                        .map_err(|_| core::fmt::Error)?;
                    emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                    let phys = Reg(r as u8);
                    self.mov64(ctx, arch, &phys, value.bits() as u64)?;
                }
                Instruction::F64Const(value) => {
                    let (r, cmds) = state
                        .regalloc
                        .as_mut()
                        .unwrap()
                        .push(x86_regalloc::RegKind::Float)
                        .map_err(|_| core::fmt::Error)?;
                    emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                    let phys = Reg(r as u8);
                    self.mov64(ctx, arch, &phys, value.bits())?;
                }
                Instruction::I32Add | Instruction::I64Add => {
                    let t1;
                    let t2;
                    {
                        let (tt1, cmds1) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds1, &mut state.stack_manager)?;
                        t1 = tt1;
                    }
                    {
                        let (tt2, cmds2) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds2, &mut state.stack_manager)?;
                        t2 = tt2;
                    }
                    let r1 = Reg(t1.reg);
                    let r2 = Reg(t2.reg);
                    self.lea(
                        ctx,
                        arch,
                        &r1,
                        &asm_x86::out::arg::MemArgKind::Mem {
                            base: r1,
                            offset: Some((r2, 0)),
                            disp: 0,
                            size: MemorySize::_64,
                            reg_class: asm_x86::RegisterClass::Gpr,
                        },
                    )?;
                    // push existing
                    {
                        let iter =
                            state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .push_existing(regalloc::Target {
                                    reg: t1.reg,
                                    kind: t1.kind,
                                });
                        emit_cmds(self, ctx, arch, iter, &mut state.stack_manager)?;
                    }
                }
                Instruction::LocalGet(local_index) => {
                    // Use regalloc to push local as stack value
                    {
                        let cmds = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .push_local(x86_regalloc::RegKind::Int, *local_index)
                            .map_err(|_| core::fmt::Error)?;
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                    }
                }
                Instruction::LocalSet(local_index) => {
                    // pop into local
                    let t;
                    {
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        t = tt;
                    }
                    {
                        let it = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop_local(x86_regalloc::RegKind::Int, *local_index);
                        emit_cmds(self, ctx, arch, it, &mut state.stack_manager)?;
                    }
                }
                Instruction::I64Load(memarg) => {
                    // address calculation: pop addr, add offset, mov
                    let t;
                    {
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        t = tt;
                    }
                    let addr = Reg(t.reg);
                    let imm = Reg(0); // temporary immediate reg
                    self.mov64(ctx, arch, &imm, memarg.offset)?;
                    self.lea(
                        ctx,
                        arch,
                        &addr,
                        &asm_x86::out::arg::MemArgKind::Mem {
                            base: addr,
                            offset: Some((imm, 0)),
                            disp: 0,
                            size: MemorySize::_64,
                            reg_class: asm_x86::RegisterClass::Gpr,
                        },
                    )?;
                    self.mov(ctx, arch, &addr, &addr)?;
                    // push existing
                    {
                        let mut it =
                            state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .push_existing(regalloc::Target {
                                    reg: t.reg,
                                    kind: t.kind,
                                });
                        emit_cmds(self, ctx, arch, it, &mut state.stack_manager)?;
                    }
                }
                Instruction::I64Store(memarg) => {
                    // pop value then address
                    let val;
                    let addr;
                    {
                        let (vv, cmds_val) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds_val, &mut state.stack_manager)?;
                        val = vv;
                    }
                    {
                        let (aa, cmds_addr) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds_addr, &mut state.stack_manager)?;
                        addr = aa;
                    }
                    let imm = Reg(0);
                    self.mov64(ctx, arch, &imm, memarg.offset)?;
                    let base = Reg(addr.reg);
                    self.lea(
                        ctx,
                        arch,
                        &base,
                        &asm_x86::out::arg::MemArgKind::Mem {
                            base,
                            offset: Some((imm, 0)),
                            disp: 0,
                            size: MemorySize::_64,
                            reg_class: asm_x86::RegisterClass::Gpr,
                        },
                    )?;
                    self.xchg(ctx, arch, &Reg(val.reg), &base)?;
                }
                Instruction::Br(relative_depth) => {
                    // flush regalloc before control transfer
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.br(ctx, arch, state, *relative_depth)?;
                }
                Instruction::BrIf(relative_depth) => {
                    // create label
                    let i = state.label_index;
                    state.label_index += 1;
                    self.lea_label(ctx, arch, &Reg(1), X64FastLabel::Indexed { idx: i })?;
                    // pop cond
                    let t;
                    {
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        t = tt;
                    }
                    let cond = Reg(t.reg);
                    self.cmp0(ctx, arch, &cond)?;
                    self.jcc(
                        ctx,
                        arch,
                        portal_solutions_asm_x86_64::ConditionCode::E,
                        &Reg(1),
                    )?;
                    // flush and branch
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.br(ctx, arch, state, *relative_depth)?;
                    self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                }
                Instruction::BrTable(targets, default) => {
                    for relative_depth in targets.iter().cloned() {
                        let i = state.label_index;
                        state.label_index += 1;
                        self.lea_label(ctx, arch, &Reg(1), X64FastLabel::Indexed { idx: i })?;
                        let t;
                        {
                            let (tt, cmds) = state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .pop(x86_regalloc::RegKind::Int);
                            emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                            t = tt;
                        }
                        let cond = Reg(t.reg);
                        self.cmp0(ctx, arch, &cond)?;
                        self.jcc(
                            ctx,
                            arch,
                            portal_solutions_asm_x86_64::ConditionCode::E,
                            &Reg(1),
                        )?;
                        {
                            let flush = state.regalloc.as_mut().unwrap().flush();
                            emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                        }
                        self.br(ctx, arch, state, relative_depth)?;
                        self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                        let mut tmp = Reg(0);
                        self.lea(
                            ctx,
                            arch,
                            &tmp,
                            &asm_x86::out::arg::MemArgKind::Mem {
                                base: tmp,
                                offset: None,
                                disp: 0xffff_ffff,
                                size: MemorySize::_64,
                                reg_class: asm_x86::RegisterClass::Gpr,
                            },
                        )?;
                        {
                            emit_cmds(
                                self,
                                ctx,
                                arch,
                                state
                                    .regalloc
                                    .as_mut()
                                    .unwrap()
                                    .push_existing(regalloc::Target {
                                        reg: tmp.0,
                                        kind: x86_regalloc::RegKind::Int,
                                    }),
                                &mut state.stack_manager,
                            )?;
                        }
                    }
                    let t;
                    {
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        t = tt;
                    }
                    {
                        emit_cmds(
                            self,
                            ctx,
                            arch,
                            state
                                .regalloc
                                .as_mut()
                                .unwrap()
                                .push_existing(regalloc::Target {
                                    reg: t.reg,
                                    kind: t.kind,
                                }),
                            &mut state.stack_manager,
                        )?;
                    }
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.br(ctx, arch, state, *default)?;
                }
                Instruction::Block(_blockty) => {
                    state.if_stack.push(Endable::Br);
                    let i = state.label_index;
                    state.label_index += 1;
                    self.lea_label(ctx, arch, &Reg(0), X64FastLabel::Indexed { idx: i })?;
                    // flush to resume with current stack pointer saved
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.push(ctx, arch, &Reg(1))?;
                    self.push(ctx, arch, &Reg(0))?;
                    self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                }
                Instruction::If(_blockty) => {
                    let i = state.label_index;
                    state.label_index += 3;
                    state.if_stack.push(Endable::If { idx: i });
                    let t;
                    {
                        let (tt, cmds) = state
                            .regalloc
                            .as_mut()
                            .unwrap()
                            .pop(x86_regalloc::RegKind::Int);
                        emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                        t = tt;
                    }
                    let cond = Reg(t.reg);
                    self.lea_label(ctx, arch, &Reg(0), X64FastLabel::Indexed { idx: i })?;
                    self.lea_label(ctx, arch, &Reg(1), X64FastLabel::Indexed { idx: i + 1 })?;
                    self.cmp0(ctx, arch, &cond)?;
                    self.jcc(
                        ctx,
                        arch,
                        portal_solutions_asm_x86_64::ConditionCode::E,
                        &Reg(1),
                    )?;
                    self.jmp(ctx, arch, &Reg(0))?;
                    self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                }
                Instruction::Else => {
                    let Endable::If { idx: i } = state.if_stack.last().unwrap() else {
                        todo!()
                    };
                    self.lea_label(ctx, arch, &Reg(0), X64FastLabel::Indexed { idx: i + 2 })?;
                    self.jmp(ctx, arch, &Reg(0))?;
                    self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i + 1 })?;
                }
                Instruction::Loop(_blockty) => {
                    state.if_stack.push(Endable::Br);
                    let i = state.label_index;
                    state.label_index += 1;
                    self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                    self.lea_label(ctx, arch, &Reg(0), X64FastLabel::Indexed { idx: i })?;
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.push(ctx, arch, &Reg(1))?;
                    self.push(ctx, arch, &Reg(0))?;
                }
                Instruction::End => {
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    match state.if_stack.pop().unwrap() {
                        Endable::Br => {
                            self.pop(ctx, arch, &Reg(0))?;
                            self.pop(ctx, arch, &Reg(1))?;
                        }
                        Endable::If { idx: i } => {
                            self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i + 2 })?;
                        }
                    }
                }
                Instruction::Call(function_index) => {
                    // check for hypercall
                    if let Some(("blitz", h)) = func_imports.get(*function_index as usize) {
                        if h.starts_with("hypercall") {
                            // hcall: pop function pointer and setup return
                            let t;
                            {
                                let (tt, cmds) = state
                                    .regalloc
                                    .as_mut()
                                    .unwrap()
                                    .pop(x86_regalloc::RegKind::Int);
                                emit_cmds(self, ctx, arch, cmds, &mut state.stack_manager)?;
                                t = tt;
                            }
                            let fnptr = Reg(t.reg);
                            let i = state.label_index;
                            state.label_index += 1;
                            self.lea_label(ctx, arch, &Reg(0), X64FastLabel::Indexed { idx: i })?;
                            self.push(ctx, arch, &Reg(0))?;
                            self.push(ctx, arch, &fnptr)?;
                            self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                            self.xchg(ctx, arch, &Reg(0), &Reg(4))?;
                            self.ret(ctx, arch)?;
                            self.set_label(ctx, arch, X64FastLabel::Indexed { idx: i })?;
                        } else {
                            // normal call
                            let function_index = *function_index - func_imports.len() as u32;
                            self.lea_label(
                                ctx,
                                arch,
                                &Reg(0),
                                X64FastLabel::Func {
                                    r#fn: function_index,
                                },
                            )?;
                            self.call(ctx, arch, &Reg(0))?;
                        }
                    } else {
                        let function_index = *function_index - func_imports.len() as u32;
                        self.lea_label(
                            ctx,
                            arch,
                            &Reg(0),
                            X64FastLabel::Func {
                                r#fn: function_index,
                            },
                        )?;
                        self.call(ctx, arch, &Reg(0))?;
                    }
                }
                Instruction::Return => {
                    // flush regalloc then perform return sequence similar to naive
                    {
                        let flush = state.regalloc.as_mut().unwrap().flush();
                        emit_cmds(self, ctx, arch, flush, &mut state.stack_manager)?;
                    }
                    self.mov(ctx, arch, &Reg(1), &Reg(4))?;
                    self.mov(ctx, arch, &Reg(0), &Reg::CTX)?;
                    let mut tmp = Reg(0);
                    self.lea(
                        ctx,
                        arch,
                        &tmp,
                        &asm_x86::out::arg::MemArgKind::Mem {
                            base: tmp,
                            offset: None,
                            disp: (state.local_count + 3 * 8) as u32,
                            size: MemorySize::_8,
                            reg_class: asm_x86::RegisterClass::Gpr,
                        },
                    )?;
                    self.mov(ctx, arch, &Reg(4), &tmp)?;
                    self.pop(ctx, arch, &Reg(0))?;
                    self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                    self.pop(ctx, arch, &Reg(0))?;
                    self.xchg(ctx, arch, &Reg(0), &Reg::CTX)?;
                    self.pop(ctx, arch, &Reg(0))?;
                    for a in 0..state.num_returns {
                        self.mov(ctx, arch, &Reg(2), &Reg(1))?;
                        self.push(ctx, arch, &Reg(2))?;
                    }
                    self.push(ctx, arch, &Reg(0))?;
                    self.ret(ctx, arch)?;
                }
                _ => {}
            }
            Ok(())
        }
    }

    impl<T: asm_x86::out::Writer<X64FastLabel, Context> + ?Sized, Context> WriterExt<Context> for T {}
}
