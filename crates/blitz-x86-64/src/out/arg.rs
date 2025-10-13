use core::mem::transmute;

use portal_solutions_blitz_common::MemorySized;

use crate::reg::{RegDisplay, X64Reg};

use super::*;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub enum ArgKind {
    Reg { reg: Reg, size: MemorySize },
    Lit(u64),
}
impl ArgKind {
    pub fn display(&self, opts: X64Arch) -> ArgKindDisplay {
        match self {
            ArgKind::Reg { reg, size } => ArgKindDisplay::Reg(X64Reg::display(
                reg,
                RegFormatOpts::default_with_arch_and_size(opts, *size),
            )),
            ArgKind::Lit(i) => ArgKindDisplay::Lit(*i),
        }
    }
}
#[non_exhaustive]
pub enum ArgKindDisplay {
    Reg(RegDisplay),
    Lit(u64),
}
impl Display for ArgKindDisplay {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ArgKindDisplay::Reg(reg_display) => write!(f, "{reg_display}"),
            ArgKindDisplay::Lit(i) => write!(f, "{i}"),
        }
    }
}
impl Display for ArgKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.display(Default::default()))
    }
}

pub trait Arg {
    fn kind(&self) -> ArgKind;
    fn format(&self, f: &mut Formatter<'_>, opts: X64Arch) -> core::fmt::Result {
        write!(f, "{}", self.display(opts))
    }
    fn display(&self, opts: X64Arch) -> ArgKindDisplay {
        return self.kind().display(opts);
    }
}
impl Arg for Reg {
    fn kind(&self) -> ArgKind {
        ArgKind::Reg {
            reg: self.clone(),
            size: Default::default(),
        }
    }
    fn display(&self, opts: X64Arch) -> ArgKindDisplay {
        ArgKindDisplay::Reg(X64Reg::display(
            self,
            RegFormatOpts::default_with_arch(opts),
        ))
    }
    fn format(&self, f: &mut Formatter<'_>, opts: X64Arch) -> core::fmt::Result {
        X64Reg::format(self, f, &RegFormatOpts::default_with_arch(opts))
    }
}
impl<T: Arg> Arg for MemorySized<T> {
    fn kind(&self) -> ArgKind {
        let MemorySized { value, size } = self;
        if typeid::of::<T>() == typeid::of::<Reg>() {
            ArgKind::Reg {
                reg: unsafe { transmute::<&T, &Reg>(value) }.clone(),
                size: size.clone(),
            }
        } else {
            match value.kind() {
                ArgKind::Reg { reg, size: _ } => ArgKind::Reg { reg, size: *size },
                a => a,
            }
        }
    }
    fn display(&self, opts: X64Arch) -> ArgKindDisplay {
        let MemorySized { value, size } = self;
        if typeid::of::<T>() == typeid::of::<Reg>() {
            ArgKindDisplay::Reg(X64Reg::display(
                unsafe { transmute::<&T, &Reg>(value) },
                RegFormatOpts::default_with_arch_and_size(opts, *size),
            ))
        } else {
            self.kind().display(opts)
        }
    }
    fn format(&self, f: &mut Formatter<'_>, opts: X64Arch) -> core::fmt::Result {
        let MemorySized { value, size } = self;
        if typeid::of::<T>() == typeid::of::<Reg>() {
            X64Reg::format(
                unsafe { transmute::<&T, &Reg>(value) },
                f,
                &RegFormatOpts::default_with_arch_and_size(opts, *size),
            )
        } else {
            write!(f, "{}", self.display(opts))
        }
    }
}
impl Arg for ArgKind {
    fn kind(&self) -> ArgKind {
        self.clone()
    }
}
impl Arg for u64 {
    fn kind(&self) -> ArgKind {
        ArgKind::Lit(*self)
    }
    fn display(&self, opts: X64Arch) -> ArgKindDisplay {
        ArgKindDisplay::Lit(*self)
    }
    fn format(&self, f: &mut Formatter<'_>, opts: X64Arch) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
