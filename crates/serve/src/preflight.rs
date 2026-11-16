//! What a server checks before it fetches or loads anything: that the CPU
//! runs the release's host code ([`cpu_gate`], over [`runs_v3`] and
//! [`cpu_lacks`]; its refusal is the class `no-device`) and that the NVIDIA
//! driver runs its CUDA code ([`driver`]); and what it says when no card is in
//! view ([`no_card`], over [`NoCard`] and the cards `nvidia-smi` lists,
//! [`listed_cards`]). The verdicts are pure functions of what the caller read:
//! the CPU's feature words ([`CpuWords`]), the driver's answer ([`Driver`]) and
//! the card reading ([`NoCard`]).
//!
//! The release's host code is built for x86-64-v3, so the compiler may use
//! any of its instructions anywhere. `std::is_x86_feature_detected!` cannot
//! see a CPU without them: under a target feature the build enables it is
//! `true` at compile time. The words are read with `cpuid` itself
//! ([`CpuWords::read`]).

#[cfg(any(target_arch = "x86_64", test))]
use crate::exit::Code;

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
const CPU_REASON: &str = ", which the release's host code (built for x86-64-v3) runs; a virtual \
                          machine needs the host's CPU type passed through";

/// The `error` event's line up to its message, as [`crate::event::render`] writes it for the class
/// [`Code::NoDevice`]: a refused CPU writes the event from static pieces (a test holds them to that
/// rendering), since nothing of ours that formats may run on a CPU that lacks x86-64-v3.
#[cfg(any(target_arch = "x86_64", test))]
const EVENT_HEAD: &str =
    r#"{"schema":"bloomery.event/1","event":"error","exit":69,"code":"no-device","message":""#;

/// The `error` event's line from the end of its message: the hint's key and, after the hint, an empty `next`.
#[cfg(any(target_arch = "x86_64", test))]
const EVENT_HINT: &str = r#"","hint":""#;
#[cfg(any(target_arch = "x86_64", test))]
const EVENT_END: &str = "\",\"next\":[]}\n";

/// The refusal `cpu_gate` prints for the features `lacks`, as one string:
/// what a test reads of it.
#[must_use]
pub fn cpu_refusal(name: &str, lacks: &[&str]) -> String {
    format!("{name}{CPU_HEAD}{}{CPU_REASON}\n", lacks.join(", "))
}

/// The names of the features `w` lacks, written to `out` in [`V3`]'s order, joined by `, `.
#[cfg(any(target_arch = "x86_64", test))]
#[inline(never)]
fn write_lacks(w: &CpuWords, out: &mut dyn std::io::Write) {
    let mut first = true;
    for f in &V3 {
        if w.word(f.reg) & f.bit == 0 {
            if !first {
                let _ = out.write_all(b", ");
            }
            let _ = out.write_all(f.name.as_bytes());
            first = false;
        }
    }
}

/// What a CPU that does not run x86-64-v3 code ([`runs_v3`]) is told: the text of [`cpu_refusal`] on `err` and
/// the `error` event of the class [`Code::NoDevice`] on `out`, both written from static pieces and `write_all`
/// of std (built for baseline x86-64), with no allocation or formatting of ours in between. Returns the class the
/// process ends with.
#[cfg(any(target_arch = "x86_64", test))]
#[inline(never)]
fn refuse_cpu(
    name: &str,
    w: &CpuWords,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
) -> Code {
    let _ = err.write_all(name.as_bytes());
    let _ = err.write_all(CPU_HEAD.as_bytes());
    write_lacks(w, err);
    let _ = err.write_all(CPU_REASON.as_bytes());
    let _ = err.write_all(b"\n");
    let _ = out.write_all(EVENT_HEAD.as_bytes());
    let _ = out.write_all(name.as_bytes());
    let _ = out.write_all(CPU_HEAD.as_bytes());
    write_lacks(w, out);
    let _ = out.write_all(CPU_REASON.as_bytes());
    let _ = out.write_all(EVENT_HINT.as_bytes());
    let _ = out.write_all(Code::NoDevice.hint().as_bytes());
    let _ = out.write_all(EVENT_END.as_bytes());
    let _ = out.flush();
    Code::NoDevice
}

/// The process's first act on x86-64: on a CPU that does not run x86-64-v3
/// code ([`runs_v3`]), print [`cpu_refusal`]'s text on stderr and its
/// `error` event on stdout ([`refuse_cpu`]), and exit with the status of
/// the class [`Code::NoDevice`], before any of that code runs. It is a call
/// from `main` that is not inlined; its passing path is `cpuid` and three
/// scalar masks, and its refusing path writes static pieces through std's own
/// `write_all` — std is built for baseline x86-64 — with no allocation or
/// formatting of ours in between.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn cpu_gate(name: &str) {
    let w = CpuWords::read();
    if runs_v3(&w) {
        return;
    }
    let code = refuse_cpu(name, &w, &mut std::io::stdout(), &mut std::io::stderr());
    std::process::exit(i32::from(code.status()));
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

/// One card as `nvidia-smi` lists it, whatever `CUDA_VISIBLE_DEVICES` hides from the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedCard {
    /// The index `nvidia-smi` gives the card.
    pub index: String,
    /// The card's name.
    pub name: String,
    /// The card's UUID, `GPU-` and the sixteen bytes.
    pub uuid: String,
}

/// What a start read of the cards when none was in view.
#[derive(Clone, Copy, Debug)]
pub struct NoCard<'a> {
    /// Why the driver's census could not be read, when it could not.
    pub census_error: Option<&'a str>,
    /// `CUDA_VISIBLE_DEVICES` as the process holds it; `None` when unset.
    pub mask: Option<&'a str>,
    /// The cards the machine lists, whatever the mask; `None` when `nvidia-smi` did not answer.
    pub listed: Option<&'a [ListedCard]>,
}

/// The cards `text` lists, the answer of `nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader`: a
/// pure read of the tool's text. A line that is not an index, a name and a UUID gives `None`, never a guess.
#[must_use]
pub fn listed_cards(text: &str) -> Option<Vec<ListedCard>> {
    let mut cards = Vec::new();
    for line in text.lines() {
        let mut cells = line.split(',').map(str::trim);
        let (index, name, uuid) = (cells.next()?, cells.next()?, cells.next()?);
        if cells.next().is_some() || index.is_empty() || name.is_empty() || uuid.is_empty() {
            return None;
        }
        cards.push(ListedCard {
            index: index.to_owned(),
            name: name.to_owned(),
            uuid: uuid.to_owned(),
        });
    }
    Some(cards)
}

/// Whether one entry of a `CUDA_VISIBLE_DEVICES` list names `card`: its index, or a prefix of its UUID (the
/// driver takes a unique prefix, `GPU-8c12` for the whole).
fn names(entry: &str, card: &ListedCard) -> bool {
    entry == card.index || (entry.starts_with("GPU-") && card.uuid.starts_with(entry))
}

/// The cards as a refusal lists them: `<index> <name> (<uuid>)`, joined by `; `.
fn listing(cards: &[ListedCard]) -> String {
    cards
        .iter()
        .map(|c| format!("{} {} ({})", c.index, c.name, c.uuid))
        .collect::<Vec<_>>()
        .join("; ")
}

/// The sentence a start that needs a card ends with when none is in view: the cause first (what the
/// environment asked for and what the machine lists), then the step that fixes it.
#[must_use]
pub fn no_card(view: &NoCard<'_>) -> String {
    const NONE: &str = "no card is visible:";
    let lists = match view.listed {
        None => "nvidia-smi did not answer".to_owned(),
        Some([]) => "nvidia-smi lists no card".to_owned(),
        Some(cards) => format!("nvidia-smi lists: {}", listing(cards)),
    };
    if let Some(e) = view.census_error {
        let mask = view.mask.map_or_else(String::new, |m| match m {
            "" => " (CUDA_VISIBLE_DEVICES is set empty)".to_owned(),
            m => format!(" (CUDA_VISIBLE_DEVICES={m})"),
        });
        return format!(
            "{NONE} the driver's device census failed: {e}{mask}; {lists}; check the driver (R580+) and \
             that CUDA_VISIBLE_DEVICES, if set, names a listed card"
        );
    }
    match (view.mask, view.listed) {
        (Some(""), _) => format!(
            "{NONE} CUDA_VISIBLE_DEVICES is set empty, which hides every card ({lists}); unset it to see them"
        ),
        (Some(mask), None) => format!(
            "{NONE} CUDA_VISIBLE_DEVICES={mask} leaves no card in view and {lists}, so it cannot be \
             checked against the cards the machine has; unset CUDA_VISIBLE_DEVICES"
        ),
        (Some(mask), listed) => {
            let named = listed.is_some_and(|cards| {
                mask.split(',')
                    .map(str::trim)
                    .any(|entry| cards.iter().any(|c| names(entry, c)))
            });
            if named {
                format!(
                    "{NONE} CUDA_VISIBLE_DEVICES={mask} names a card nvidia-smi lists, yet the driver opened \
                     none ({lists}); check that the CUDA driver (R580+) runs it, or unset \
                     CUDA_VISIBLE_DEVICES"
                )
            } else {
                format!(
                    "{NONE} CUDA_VISIBLE_DEVICES={mask} names none of the cards the driver lists ({lists}); \
                     unset CUDA_VISIBLE_DEVICES or name one of them by index or UUID"
                )
            }
        }
        (None, Some([])) => {
            format!("{NONE} the machine lists none ({lists}); attach a card or run where one is")
        }
        (None, Some(_)) => format!(
            "{NONE} CUDA_VISIBLE_DEVICES is unset and the driver opened none of the cards nvidia-smi \
             lists ({}); check that the CUDA driver (R580+) runs them",
            lists.trim_start_matches("nvidia-smi lists: ")
        ),
        (None, None) => format!("{NONE} the driver opened none ({lists})"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CUDA_MIN, Code, CpuWords, Driver, ListedCard, NoCard, cpu_lacks, cpu_refusal, driver,
        listed_cards, no_card, refuse_cpu, runs_v3,
    };
    use crate::event::{self, Event};

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

    /// A CPU that lacks x86-64-v3 is told on stderr what it lacks and on stdout, in the one event of the
    /// start's schema, the class `no-device` (status 69): the pieces a refused CPU writes without formatting are
    /// the line `event::render` makes of the same message, for every feature name.
    ///
    /// PIN(2026-10-29): the refusal ended with status 1 and no event on stdout before the exit table.
    #[test]
    fn a_refused_cpu_prints_its_error_event_and_ends_as_no_device() {
        let kvm64 = CpuWords {
            leaf1_ecx: 0x8020_0001,
            leaf7_ebx: 0,
            ext1_ecx: 0x0000_0001,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = refuse_cpu("bloomery-serve", &kvm64, &mut out, &mut err);
        assert_eq!(code, Code::NoDevice);
        assert_eq!(code.status(), 69);
        let said = cpu_refusal("bloomery-serve", &cpu_lacks(&kvm64));
        assert_eq!(String::from_utf8(err).expect("text"), said);
        let line = event::render(&Event::Error {
            code,
            message: said.trim_end(),
            next: &[],
        });
        assert_eq!(String::from_utf8(out).expect("text"), format!("{line}\n"));
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
    /// The box's two cards as `nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader` prints them.
    const SMI: &str = "0, NVIDIA RTX A6000, GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4\n\
                       1, NVIDIA GeForce RTX 5090, GPU-5090aaaa-0000-0000-0000-000000000001\n";

    fn cards() -> Vec<ListedCard> {
        listed_cards(SMI).expect("the listing reads")
    }

    /// The listing reads as index, name and UUID; a line that is none of that gives no listing, and an empty
    /// answer lists no card.
    #[test]
    fn the_card_listing_reads_its_lines() {
        let cards = cards();
        assert_eq!(cards.len(), 2);
        assert_eq!(
            cards[1],
            ListedCard {
                index: "1".into(),
                name: "NVIDIA GeForce RTX 5090".into(),
                uuid: "GPU-5090aaaa-0000-0000-0000-000000000001".into(),
            }
        );
        assert_eq!(listed_cards(""), Some(Vec::new()));
        assert_eq!(listed_cards("No devices were found\n"), None);
        assert_eq!(listed_cards("0, only a name\n"), None);
    }

    /// The cause a start with no card in view is refused with names what it asked for (the variable's value)
    /// and what the machine lists, and says the step: here `CUDA_VISIBLE_DEVICES` names a card that is gone.
    #[test]
    fn a_mask_that_names_no_listed_card_is_the_cause() {
        let cards = cards();
        let said = no_card(&NoCard {
            census_error: None,
            mask: Some("GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1"),
            listed: Some(&cards),
        });
        assert!(
            said.starts_with(
                "no card is visible: CUDA_VISIBLE_DEVICES=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1 \
                 names none of the cards the driver lists"
            ),
            "{said}"
        );
        for listed in [
            "0 NVIDIA RTX A6000",
            "1 NVIDIA GeForce RTX 5090",
            "GPU-8c129fa6",
        ] {
            assert!(said.contains(listed), "{listed}: {said}");
        }
        assert!(said.contains("unset CUDA_VISIBLE_DEVICES"), "{said}");
    }

    /// A mask that does name a listed card, by index or by a UUID prefix, is not "names none": the driver
    /// opened none of them, and the sentence says that instead.
    #[test]
    fn a_mask_that_names_a_listed_card_is_not_called_unnamed() {
        let cards = cards();
        for mask in [
            "1",
            "0,1",
            " 1 ",
            "GPU-8c129fa6",
            "GPU-gone,GPU-8c129fa6-7382",
        ] {
            let said = no_card(&NoCard {
                census_error: None,
                mask: Some(mask),
                listed: Some(&cards),
            });
            assert!(
                said.starts_with("no card is visible: CUDA_VISIBLE_DEVICES=")
                    && said.contains("names a card nvidia-smi lists")
                    && !said.contains("names none"),
                "{mask}: {said}"
            );
        }
    }

    /// An empty mask hides every card on purpose; the sentence says so and the step is to unset it.
    #[test]
    fn an_empty_mask_hides_every_card() {
        let cards = cards();
        let said = no_card(&NoCard {
            census_error: None,
            mask: Some(""),
            listed: Some(&cards),
        });
        assert!(
            said.starts_with(
                "no card is visible: CUDA_VISIBLE_DEVICES is set empty, which hides every card"
            ) && said.contains("unset it")
                && said.contains("1 NVIDIA GeForce RTX 5090"),
            "{said}"
        );
    }

    /// With no mask the cause is the driver's: it opened none of the listed cards, or the machine lists none,
    /// or the tool that lists them did not answer.
    #[test]
    fn with_no_mask_the_cause_is_the_driver_or_the_machine() {
        let cards = cards();
        let said = no_card(&NoCard {
            census_error: None,
            mask: None,
            listed: Some(&cards),
        });
        assert!(
            said.starts_with(
                "no card is visible: CUDA_VISIBLE_DEVICES is unset and the driver opened none of the cards \
                 nvidia-smi lists"
            ) && said.contains("0 NVIDIA RTX A6000"),
            "{said}"
        );
        let said = no_card(&NoCard {
            census_error: None,
            mask: None,
            listed: Some(&[]),
        });
        assert!(
            said.starts_with(
                "no card is visible: the machine lists none (nvidia-smi lists no card)"
            ),
            "{said}"
        );
        let said = no_card(&NoCard {
            census_error: None,
            mask: None,
            listed: None,
        });
        assert!(
            said.starts_with(
                "no card is visible: the driver opened none (nvidia-smi did not answer)"
            ),
            "{said}"
        );
    }

    /// A census the driver would not answer is its own cause, beside the mask when one is set.
    #[test]
    fn a_census_that_fails_is_named_with_its_error() {
        let said = no_card(&NoCard {
            census_error: Some("cuInit: CUDA_ERROR_NO_DEVICE"),
            mask: Some("2"),
            listed: None,
        });
        assert!(
            said.starts_with(
                "no card is visible: the driver's device census failed: cuInit: CUDA_ERROR_NO_DEVICE"
            ) && said.contains("CUDA_VISIBLE_DEVICES=2")
                && said.contains("nvidia-smi did not answer"),
            "{said}"
        );
        let said = no_card(&NoCard {
            census_error: Some("cuInit: CUDA_ERROR_NO_DEVICE"),
            mask: Some(""),
            listed: None,
        });
        assert!(
            said.contains("(CUDA_VISIBLE_DEVICES is set empty)"),
            "{said}"
        );
    }
    /// A mask no listing can be held against (`nvidia-smi` silent) is not called "names none": the sentence
    /// says it could not check.
    #[test]
    fn a_mask_with_no_listing_says_it_could_not_check() {
        let said = no_card(&NoCard {
            census_error: None,
            mask: Some("GPU-gone"),
            listed: None,
        });
        assert!(
            said.starts_with(
                "no card is visible: CUDA_VISIBLE_DEVICES=GPU-gone leaves no card in view"
            ) && said.contains("nvidia-smi did not answer")
                && !said.contains("names none"),
            "{said}"
        );
    }
}
