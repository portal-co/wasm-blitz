//! Unit tests for slice analysis, specialization, and deopt-guard emission.

use super::*;
use alloc::vec;
use wasm_encoder::{Instruction, MemArg};

fn insn(i: Instruction<'static>) -> MachOperator<'static, ()> {
    MachOperator::Instruction { op: i, annot: () }
}

fn memarg(offset: u64) -> MemArg {
    MemArg { offset, align: 2, memory_index: 0 }
}

#[test]
fn branch_analysis_resolves_nested_targets() {
    // Block { Loop { Br 0 } Br 0 }
    //   idx0 Block   (site 1, ends at idx5)
    //   idx1 Loop    (site 2, back-edge target = idx1)
    //   idx2 Br 0    -> loop header idx1
    //   idx3 End     (loop end)
    //   idx4 Br 0    -> block end idx5
    //   idx5 End     (block end)
    let slice = FnSlice::from_ops([
        insn(Instruction::Block(wasm_encoder::BlockType::Empty)),
        insn(Instruction::Loop(wasm_encoder::BlockType::Empty)),
        insn(Instruction::Br(0)),
        insn(Instruction::End),
        insn(Instruction::Br(0)),
        insn(Instruction::End),
    ]);
    let a = BranchAnalysis::analyze(&slice);

    assert_eq!(a.header_to_end.get(&0), Some(&5));
    assert_eq!(a.header_to_end.get(&1), Some(&3));

    // Loop back-edge and block forward-exit.
    assert_eq!(a.branch_target(2), Some(&[1usize][..]));
    assert_eq!(a.branch_target(4), Some(&[5usize][..]));

    // Site numbering: function entry is 0, block=1, loop=2 (source order).
    assert_eq!(a.site_id_of(0), Some(1));
    assert_eq!(a.site_id_of(1), Some(2));

    // Depth before each op.
    assert_eq!(a.depth_at, vec![0, 1, 2, 2, 1, 1]);
}

#[test]
fn br_table_resolves_each_arm() {
    // Block { Block { BrTable [0,1] default 1 } End } End
    //   idx0 Block  (outer, end idx5)
    //   idx1 Block  (inner, end idx4)
    //   idx2 BrTable [0,1] 1
    //   idx3 ... (unreachable placeholder)
    //   idx4 End (inner)
    //   idx5 End (outer)
    let slice = FnSlice::from_ops([
        insn(Instruction::Block(wasm_encoder::BlockType::Empty)),
        insn(Instruction::Block(wasm_encoder::BlockType::Empty)),
        insn(Instruction::BrTable(vec![0, 1].into(), 1)),
        insn(Instruction::Nop),
        insn(Instruction::End),
        insn(Instruction::End),
    ]);
    let a = BranchAnalysis::analyze(&slice);
    // arm0 -> inner block end (idx4); arm1 -> outer block end (idx5); default 1 -> idx5
    assert_eq!(a.branch_target(2), Some(&[4usize, 5, 5][..]));
}

#[test]
fn specialize_value_folds_and_guards() {
    // local.get 0 ; i32.const 5 ; i32.add   with local 0 := 10
    let slice = FnSlice::from_ops([
        insn(Instruction::LocalGet(0)),
        insn(Instruction::I32Const(5)),
        insn(Instruction::I32Add),
    ]);
    let analysis = BranchAnalysis::analyze(&slice);
    let mut spec = SpecSpec::default();
    spec.locals.insert(0, ConstVal::I32(10));

    let r = specialize(&slice, &analysis, &spec);
    assert_eq!(r.guards, vec![Guard::Local { idx: 0, expected: ConstVal::I32(10) }]);
    // Folded to a single i32.const 15.
    assert_eq!(r.ops.len(), 1);
    match &r.ops[0] {
        MachOperator::Instruction { op: Instruction::I32Const(v), .. } => assert_eq!(*v, 15),
        other => panic!("expected i32.const 15, got {other:?}"),
    }
}

#[test]
fn specialize_memory_folds_const_region_load() {
    // i32.const 100 ; i32.load offset=0   with region @100 = [7,0,0,0]
    let slice = FnSlice::from_ops([
        insn(Instruction::I32Const(100)),
        insn(Instruction::I32Load(memarg(0))),
    ]);
    let analysis = BranchAnalysis::analyze(&slice);
    let mut spec = SpecSpec::default();
    spec.const_regions.push(ConstRegion { addr: 100, bytes: vec![7, 0, 0, 0] });

    let r = specialize(&slice, &analysis, &spec);
    assert_eq!(r.guards, vec![Guard::Mem { addr: 100, width: 4, value: 7 }]);
    assert_eq!(r.ops.len(), 1);
    match &r.ops[0] {
        MachOperator::Instruction { op: Instruction::I32Const(v), .. } => assert_eq!(*v, 7),
        other => panic!("expected i32.const 7, got {other:?}"),
    }
}

// --- emit_deopt_guard over a recording BlitzWriter mock --------------------

#[derive(Default)]
struct RecordWriter {
    events: Vec<Event>,
}

#[derive(Debug, PartialEq)]
enum Event {
    BranchZero(u8, usize),
    BranchLabel(usize),
    PlaceLabel(usize),
}

impl BlitzWriter for RecordWriter {
    type Error = core::convert::Infallible;
    fn branch_label(&mut self, l: usize) -> Result<(), Self::Error> {
        self.events.push(Event::BranchLabel(l));
        Ok(())
    }
    fn branch_zero_label(&mut self, r: u8, l: usize) -> Result<(), Self::Error> {
        self.events.push(Event::BranchZero(r, l));
        Ok(())
    }
    fn branch_reg(&mut self, _r: u8) -> Result<(), Self::Error> {
        Ok(())
    }
    fn place_label(&mut self, l: usize) -> Result<(), Self::Error> {
        self.events.push(Event::PlaceLabel(l));
        Ok(())
    }
    fn reg_decrement(&mut self, _r: u8) -> Result<(), Self::Error> {
        Ok(())
    }
    fn load_u64_imm(&mut self, _d: u8, _i: u64) -> Result<(), Self::Error> {
        Ok(())
    }
    fn inc_mem64(&mut self, _p: u8) -> Result<(), Self::Error> {
        Ok(())
    }
    fn load_mem64(&mut self, _d: u8, _s: u8) -> Result<(), Self::Error> {
        Ok(())
    }
    fn load_trace_base(&mut self, _d: u8, _o: i32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn inc_mem64_disp(&mut self, _p: u8, _o: i32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn load_mem64_disp(&mut self, _d: u8, _s: u8, _o: i32) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn deopt_guard_continues_on_hold_branches_on_fail() {
    let mut w = RecordWriter::default();
    let mut label = 7usize; // pretend label allocator is at 7
    let deopt = 3usize; // generic site entry label
    emit_deopt_guard(&mut w, 2 /*diff reg*/, deopt, &mut label).unwrap();
    assert_eq!(label, 8); // one continuation label allocated
    assert_eq!(
        w.events,
        vec![
            Event::BranchZero(2, 7), // diff==0 -> continue (label 7)
            Event::BranchLabel(3),   // else -> deopt target
            Event::PlaceLabel(7),    // continuation
        ]
    );
}
