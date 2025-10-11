use super::*;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub enum ArgKind{
    Reg(Reg),
    Lit(u64)
}
impl ArgKind{
    pub fn display(&self, opts: RegFormatOpts) -> ArgKindDisplay{
        match self{
            ArgKind::Reg(reg) => ArgKindDisplay::Reg(reg.display(opts)),
            ArgKind::Lit(i) => ArgKindDisplay::Lit(*i),
        }
    }
}
#[non_exhaustive]
pub enum ArgKindDisplay{
    Reg(RegDisplay),
    Lit(u64)
}
impl Display for ArgKindDisplay{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self{
            ArgKindDisplay::Reg(reg_display) => write!(f,"{reg_display}"),
            ArgKindDisplay::Lit(i) => write!(f,"{i}"),
        }
    }
}
impl Display for ArgKind{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"{}",self.display(Default::default()))
    }
}

pub trait Arg: Display {
    fn reg(&self) -> ArgKind;
    fn display(&self, opts: RegFormatOpts) -> ArgKindDisplay {
        return self.reg().display(opts);
    }
}
impl Arg for Reg {
    fn reg(&self) -> ArgKind {
        ArgKind::Reg(self.clone())
    }
    fn display(&self, opts: RegFormatOpts) -> ArgKindDisplay {
        ArgKindDisplay::Reg(self.display(opts))
    }
}
impl Arg for ArgKind{
    fn reg(&self) -> ArgKind {
        self.clone()
    }
}