use wasm_encoder::Instruction;

use crate::*;
#[derive(Default)]
pub struct DceStack(Vec<bool>);
impl DceStack {
    pub fn new() -> Self {
        return Default::default();
    }
}
pub fn dce(DceStack(stack): &mut DceStack, o: &Operator<'_>) -> bool {
    match o {
        Operator::Else => {
            if let Some(a) = stack.last_mut() {
                *a = false
            }
        }
        Operator::If { .. } | Operator::Block { .. } | Operator::Loop { .. } => {
            stack.push(false);
        }
        Operator::End => {
            stack.pop();
        }
        Operator::Br { .. }
        | Operator::BrTable { .. }
        | Operator::Return
        | Operator::ReturnCall { .. }
        | Operator::ReturnCallIndirect { .. }
        | Operator::ReturnCallRef { .. }
        | Operator::Unreachable => {
            if let Some(a) = stack.last_mut() {
                *a = true
            }
        }
        o => {
            if stack.iter().any(|a| *a) {
                return true;
            } else {
            }
        }
    };
    return false;
}
pub fn dce_instr(DceStack(stack): &mut DceStack, o: &Instruction<'_>) -> bool {
    match o {
        Instruction::Else => {
            if let Some(a) = stack.last_mut() {
                *a = false
            }
        }
        Instruction::If { .. } | Instruction::Block { .. } | Instruction::Loop { .. } => {
            stack.push(false);
        }
        Instruction::End => {
            stack.pop();
        }
        Instruction::Br { .. }
        | Instruction::BrTable { .. }
        | Instruction::Return
        | Instruction::ReturnCall { .. }
        | Instruction::ReturnCallIndirect { .. }
        | Instruction::ReturnCallRef { .. }
        | Instruction::Unreachable => {
            if let Some(a) = stack.last_mut() {
                *a = true
            }
        }
        o => {
            if stack.iter().any(|a| *a) {
                return true;
            } else {
            }
        }
    };
    return false;
}
