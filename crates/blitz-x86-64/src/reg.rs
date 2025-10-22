use crate::*;
pub trait X64Reg {
    fn format(&self, f: &mut Formatter<'_>, opts: &RegFormatOpts) -> core::fmt::Result;
    fn display<'a>(&'a self, opts: RegFormatOpts) -> RegDisplay;
    fn context_handle(&self, arch: &X64Arch) -> (Reg, u32, u32);
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
    fn context_handle(&self, arch: &X64Arch) -> (Reg, u32, u32) {
        (
            Reg(9),
            0x28,
            match (self.0) as u32 % (if arch.apx { 32 } else { 16 }) {
                a => match a {
                    0 => 0x78,
                    1 => 0x90,
                    2 => 0x80,
                    3 => 0x98,
                    4 => 0xa8,
                    5 => 0xa8,
                    6 => 0xb0,
                    7 => 0x88,
                    a => (a * 8) + 0xb8,
                },
            },
        )
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
