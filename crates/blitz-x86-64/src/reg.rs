use crate::*;
pub trait X64Reg {
    fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result;
    fn display<'a>(&'a self, opts: RegFormatOpts) -> RegDisplay;
}
impl X64Reg for Reg {
    fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result {
        let idx = (self.0 as usize) % (if opts.arch.apx { 32 } else { 16 });
        if idx < 8 {
            write!(
                f,
                "{}",
                &(match &opts.size {
                    MemorySize::_8 => REG_NAMES_8,
                    MemorySize::_16 => REG_NAMES_16,
                    MemorySize::_32 => REG_NAMES_32,
                    MemorySize::_64 => REG_NAMES,
                })[idx]
            )
        } else {
            write!(
                f,
                "r{idx}{}",
                match &opts.size {
                    MemorySize::_8 => "b",
                    MemorySize::_16 => "w",
                    MemorySize::_32 => "d",
                    MemorySize::_64 => "",
                }
            )
        }
    }
    fn display<'a>(&'a self, opts: RegFormatOpts) -> RegDisplay {
        RegDisplay { reg: *self, opts }
    }
}
pub struct RegDisplay {
    reg: Reg,
    opts: RegFormatOpts,
}
impl Display for RegDisplay {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        X64Reg::format(&self.reg, f, &self.opts)
    }
}
