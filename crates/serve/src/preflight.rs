//! What a server checks before it fetches or loads anything: that the CPU
//! runs the release's host code ([`cpu_gate`], over [`runs_v3`] and
//! [`cpu_lacks`]) and that the NVIDIA driver runs its CUDA code
//! ([`driver`]). The verdicts are pure functions of what the caller read: the
//! CPU's feature words ([`CpuWords`]) and the driver's answer ([`Driver`]).
//!
//! The release's host code is built for x86-64-v3, so the compiler may use
//! any of its instructions anywhere. `std::is_x86_feature_detected!` cannot
//! see a CPU without them: under a target feature the build enables it is
//! `true` at compile time. The words are read with `cpuid` itself
//! ([`CpuWords::read`]).

/// The CUDA version the release's code needs of the driver, as
/// `cuDriverGetVersion` gives it (`1000 * major + 10 * minor`): 13.0, which
/// driver branch R580 is the first to carry.
pub const CUDA_MIN: i32 = 13000;

/// The CPU's `cpuid` words the x86-64-v3 check reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuWords {
    /// Leaf 1, ECX.
    pub leaf1_ecx: u32,
    /// Leaf 7 sub-leaf 0, EBX; 0 on a CPU whose highest leaf is below 7.
    pub leaf7_ebx: u32,
    /// Leaf `0x8000_0001`, ECX; 0 on a CPU without that leaf.
    pub ext1_ecx: u32,
}

/// Which of [`CpuWords`]' registers a feature's bit sits in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reg {
    Leaf1Ecx,
    Leaf7Ebx,
    Ext1Ecx,
}

/// One feature of x86-64-v3: its name as a refusal prints it, the register
/// and its bit.
struct Feature {
    name: &'static str,
    reg: Reg,
    bit: u32,
}

/// What x86-64-v3 needs of a CPU and its OS, in the order a refusal lists
/// them: the instruction sets, and XSAVE turned on by the OS (OSXSAVE), the
/// switch under which the OS saves the AVX registers — without it an AVX
/// instruction faults.
const V3: [Feature; 10] = [
    Feature {
        name: "AVX",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 28,
    },
    Feature {
        name: "AVX2",
        reg: Reg::Leaf7Ebx,
        bit: 1 << 5,
    },
    Feature {
        name: "BMI1",
        reg: Reg::Leaf7Ebx,
        bit: 1 << 3,
    },
    Feature {
        name: "BMI2",
        reg: Reg::Leaf7Ebx,
        bit: 1 << 8,
    },
    Feature {
        name: "F16C",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 29,
    },
    Feature {
        name: "FMA",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 12,
    },
    Feature {
        name: "LZCNT",
        reg: Reg::Ext1Ecx,
        bit: 1 << 5,
    },
    Feature {
        name: "MOVBE",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 22,
    },
    Feature {
        name: "XSAVE",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 26,
    },
    Feature {
        name: "XSAVE turned on by the OS (OSXSAVE)",
        reg: Reg::Leaf1Ecx,
        bit: 1 << 27,
    },
];

/// The bits every feature of [`V3`] in `reg` needs, OR-ed at compile time.
const fn need(reg: Reg) -> u32 {
    let mut bits = 0;
    let mut i = 0;
    while i < V3.len() {
        if V3[i].reg as u8 == reg as u8 {
            bits |= V3[i].bit;
        }
        i += 1;
    }
    bits
}

impl CpuWords {
    /// `reg`'s word.
    fn word(&self, reg: Reg) -> u32 {
        match reg {
            Reg::Leaf1Ecx => self.leaf1_ecx,
            Reg::Leaf7Ebx => self.leaf7_ebx,
            Reg::Ext1Ecx => self.ext1_ecx,
        }
    }

    /// This CPU's words, by `cpuid`: a leaf past the highest the CPU
    /// answers is read as 0.
    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    #[must_use]
    pub fn read() -> CpuWords {
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        let top = __cpuid(0).eax;
        let ext_top = __cpuid(0x8000_0000).eax;
        CpuWords {
            leaf1_ecx: if top >= 1 { __cpuid(1).ecx } else { 0 },
            leaf7_ebx: if top >= 7 { __cpuid_count(7, 0).ebx } else { 0 },
            ext1_ecx: if ext_top >= 0x8000_0001 {
                __cpuid(0x8000_0001).ecx
            } else {
                0
            },
        }
    }
}

/// Whether the CPU of `w` runs x86-64-v3 code: every register holds every
/// bit [`V3`] needs of it. Three masks and compares, nothing a compiler
/// vectorizes.
#[must_use]
pub fn runs_v3(w: &CpuWords) -> bool {
    const LEAF1: u32 = need(Reg::Leaf1Ecx);
    const LEAF7: u32 = need(Reg::Leaf7Ebx);
    const EXT1: u32 = need(Reg::Ext1Ecx);
    w.leaf1_ecx & LEAF1 == LEAF1 && w.leaf7_ebx & LEAF7 == LEAF7 && w.ext1_ecx & EXT1 == EXT1
}

/// The features of x86-64-v3 the CPU of `w` lacks, in [`V3`]'s order; empty
/// exactly when [`runs_v3`] holds.
#[must_use]
pub fn cpu_lacks(w: &CpuWords) -> Vec<&'static str> {
    V3.iter()
        .filter(|f| w.word(f.reg) & f.bit == 0)
        .map(|f| f.name)
        .collect()
}

/// The refusal's text before the missing features' names.
const CPU_HEAD: &str = ": this CPU lacks ";

/// The refusal's text after them.
const CPU_TAIL: &str = ", which the release's host code (built for x86-64-v3) runs; a virtual \
                        machine needs the host's CPU type passed through\n";

/// The refusal `cpu_gate` prints for the features `lacks`, as one string:
/// what a test reads of it.
#[must_use]
pub fn cpu_refusal(name: &str, lacks: &[&str]) -> String {
    format!("{name}{CPU_HEAD}{}{CPU_TAIL}", lacks.join(", "))
}

/// The process's first act on x86-64: on a CPU that does not run x86-64-v3
/// code ([`runs_v3`]), print [`cpu_refusal`]'s text on stderr and exit 1,
/// before any of that code runs. It is a call from `main` that is not
/// inlined; its passing path is `cpuid` and three scalar masks, and its
/// refusing path writes static pieces through std's own `write_all` — std is
/// built for baseline x86-64 — with no allocation or formatting of ours in
/// between.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn cpu_gate(name: &str) {
    use std::io::Write as _;
    let w = CpuWords::read();
    if runs_v3(&w) {
        return;
    }
    let mut err = std::io::stderr();
    let _ = err.write_all(name.as_bytes());
    let _ = err.write_all(CPU_HEAD.as_bytes());
    let mut first = true;
    for f in &V3 {
        if w.word(f.reg) & f.bit == 0 {
            if !first {
                let _ = err.write_all(b", ");
            }
            let _ = err.write_all(f.name.as_bytes());
            first = false;
        }
    }
    let _ = err.write_all(CPU_TAIL.as_bytes());
    std::process::exit(1);
}

/// What the driver answered a server's start, as the CUDA bindings' loader
/// read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Driver {
    /// The library loaded and its `cuDriverGetVersion` is the CUDA major the
    /// build needs, or it gave no version to hold the load to.
    Loaded,
    /// The driver library (`libcuda.so.1`) did not load: why.
    Missing(String),
    /// `cuDriverGetVersion`'s version, below the CUDA major the build needs.
    Version(i32),
}

/// `v` (`1000 * major + 10 * minor`) as `major.minor`.
fn cuda_version(v: i32) -> String {
    format!("{}.{}", v / 1000, v % 1000 / 10)
}

/// Whether the driver `d` runs the release's CUDA code: a version of at
/// least [`CUDA_MIN`]. Anything else is refused by name, saying what the
/// driver reported and what is needed.
pub fn driver(d: &Driver) -> Result<(), String> {
    const NEED: &str = "bloomery needs NVIDIA driver R580+ (CUDA 13)";
    match d {
        Driver::Loaded => Ok(()),
        Driver::Version(v) if *v >= CUDA_MIN => Ok(()),
        Driver::Version(v) if *v > 0 => Err(format!(
            "{NEED}; this driver reports CUDA {}",
            cuda_version(*v)
        )),
        Driver::Version(v) => Err(format!(
            "{NEED}; cuDriverGetVersion reported {v}, no CUDA version"
        )),
        Driver::Missing(why) => Err(format!(
            "{NEED}, and its library libcuda.so.1 did not load: {why}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{CUDA_MIN, CpuWords, Driver, cpu_lacks, cpu_refusal, driver, runs_v3};

    /// A Zen 3's words: every x86-64-v3 bit, and OSXSAVE.
    const ZEN3: CpuWords = CpuWords {
        leaf1_ecx: 0x7ed8_320b,
        leaf7_ebx: 0x219c_97a9,
        ext1_ecx: 0x7584_17ff,
    };

    #[test]
    fn a_v3_cpu_runs_and_each_missing_bit_is_named() {
        assert!(runs_v3(&ZEN3));
        assert!(cpu_lacks(&ZEN3).is_empty());
        let cases: [(&str, CpuWords); 10] = [
            (
                "AVX",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 28),
                    ..ZEN3
                },
            ),
            (
                "AVX2",
                CpuWords {
                    leaf7_ebx: ZEN3.leaf7_ebx & !(1 << 5),
                    ..ZEN3
                },
            ),
            (
                "BMI1",
                CpuWords {
                    leaf7_ebx: ZEN3.leaf7_ebx & !(1 << 3),
                    ..ZEN3
                },
            ),
            (
                "BMI2",
                CpuWords {
                    leaf7_ebx: ZEN3.leaf7_ebx & !(1 << 8),
                    ..ZEN3
                },
            ),
            (
                "F16C",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 29),
                    ..ZEN3
                },
            ),
            (
                "FMA",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 12),
                    ..ZEN3
                },
            ),
            (
                "LZCNT",
                CpuWords {
                    ext1_ecx: ZEN3.ext1_ecx & !(1 << 5),
                    ..ZEN3
                },
            ),
            (
                "MOVBE",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 22),
                    ..ZEN3
                },
            ),
            (
                "XSAVE",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 26),
                    ..ZEN3
                },
            ),
            (
                "XSAVE turned on by the OS (OSXSAVE)",
                CpuWords {
                    leaf1_ecx: ZEN3.leaf1_ecx & !(1 << 27),
                    ..ZEN3
                },
            ),
        ];
        for (name, w) in cases {
            assert!(!runs_v3(&w), "{name}");
            assert_eq!(cpu_lacks(&w), [name]);
        }
    }

    /// An Ivy Bridge (E5 v2) has AVX but none of AVX2, BMI and FMA; a
    /// `kvm64` guest has none of it and no OSXSAVE.
    #[test]
    fn an_older_cpu_and_a_bare_guest_are_refused_with_every_name() {
        let ivy = CpuWords {
            leaf1_ecx: 0x7fbe_e3ff & !(1 << 12) & !(1 << 22),
            leaf7_ebx: 0x0000_0281 & !(1 << 3) & !(1 << 5) & !(1 << 8),
            ext1_ecx: 0x0000_0001,
        };
        assert_eq!(
            cpu_lacks(&ivy),
            ["AVX2", "BMI1", "BMI2", "FMA", "LZCNT", "MOVBE"]
        );
        let kvm64 = CpuWords {
            leaf1_ecx: 0x8020_0001,
            leaf7_ebx: 0,
            ext1_ecx: 0x0000_0001,
        };
        let lacks = cpu_lacks(&kvm64);
        assert_eq!(lacks.len(), 10, "{lacks:?}");
        let said = cpu_refusal("bloomery-serve", &lacks[..2]);
        assert_eq!(
            said,
            "bloomery-serve: this CPU lacks AVX, AVX2, which the release's host code (built for \
             x86-64-v3) runs; a virtual machine needs the host's CPU type passed through\n"
        );
    }

    #[test]
    fn the_driver_runs_from_cuda_13_and_anything_else_is_named() {
        assert_eq!(driver(&Driver::Version(CUDA_MIN)), Ok(()));
        assert_eq!(driver(&Driver::Version(13020)), Ok(()));
        let e = driver(&Driver::Version(12080)).unwrap_err();
        assert!(
            e.contains("R580+ (CUDA 13)") && e.contains("reports CUDA 12.8"),
            "{e}"
        );
        assert!(
            driver(&Driver::Version(12020))
                .unwrap_err()
                .contains("CUDA 12.2")
        );
        let e = driver(&Driver::Version(0)).unwrap_err();
        assert!(e.contains("R580+") && e.contains("reported 0"), "{e}");
        assert_eq!(driver(&Driver::Loaded), Ok(()));
        let e = driver(&Driver::Missing(
            "libcuda.so.1: cannot open shared object file".into(),
        ))
        .unwrap_err();
        assert!(
            e.contains("R580+") && e.contains("libcuda.so.1 did not load: libcuda.so.1: cannot"),
            "{e}"
        );
    }
}
