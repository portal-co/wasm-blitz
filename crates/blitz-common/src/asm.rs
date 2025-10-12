use crate::*;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Reg(pub u8);
impl Reg {
    pub const CTX: Reg = Reg(255);
  
}