//! The architecture registry: one [`ArchRow`] per group of `general.architecture` strings that one
//! reader, one draft rule and one gate fixture serve. The dispatch on a file's architecture reads
//! these rows: the reader ([`super::spec`]), a target's draft rule ([`super::draft::draft_usable`])
//! and the fixture of a family (the `fixture` binary and the gates' budget admission).
//! A family is added by one row here, and no `match` on a model.
//!
//! [`Arch::name`] and [`Arch::from_name`] stay the enum's own: a row holds function pointers, so a
//! caller of the registry reaches every reader, and the many that only name an architecture do not
//! read it. The registry's test holds the two to the same strings.
//!
//! [`print`] renders the rows; `tools/arch-registry.tsv` is its committed copy, which
//! `tools/hf-arch-check.py` reads instead of mirroring the strings.

use gguf::Split;

use super::coverage::Available;
use super::draft::Rule;
use super::{
    Arch, DEEPSEEK4, GEMMA_EMBEDDING, GEMMA_EMBEDDING2, GLM5_NEXT_LLAMA_CPP, QWEN35_BODY, Read,
    deepseek41, draft, flint, gemma_embedding, gemma_embedding2, glm5next, lfm3, llama, mimo2,
    minimaxm3, musespark, nemotron_h, nemotron_h_moe, qwen3moe, qwen35moe, raon1, solar2,
};
use crate::placement::PlacementError;

/// Every fixture family's spec, read by the fixture tool and the gates' budget admission.
pub const FIXTURE_SPECS: [fn() -> crate::fixture::FixtureSpec; 9] = [
    deepseek41::fixture::spec,
    qwen3moe::fixture::spec,
    qwen35moe::fixture::spec,
    qwen35moe::fixture::qwen35moe::spec,
    glm5next::fixture::spec,
    solar2::fixture::spec,
    nemotron_h::fixture::spec,
    nemotron_h_moe::fixture::spec,
    musespark::fixture::spec,
];

/// The fixture spec whose architecture string is `arch`, from [`FIXTURE_SPECS`].
#[must_use]
pub fn fixture_spec(arch: &str) -> Option<crate::fixture::FixtureSpec> {
    FIXTURE_SPECS
        .iter()
        .map(|spec| spec())
        .find(|spec| spec.arch == arch)
}

/// A family's reader: the description of one of its files and the role of each tensor.
pub type Reader = fn(&Split) -> Result<Read, PlacementError>;

/// Where a family's draft fixture comes from on the hub.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftSource {
    /// The family has no draft file.
    None,
    /// The draft is this file name in the target's repo.
    File(&'static str),
    /// No public file is the draft's source; the reason, said when `fetch` skips it.
    NotFetchable(&'static str),
}

/// A family's gate fixture, by how it is made: its directory under the fixture root is the row's
/// `family`, the one table `tools/fixture-dirs.tsv` holds for both kinds.
#[derive(Clone, Copy)]
pub enum Fixture {
    /// GGUF shards `fixture fetch` makes from a hub file's header ([`FixtureRow`]).
    Gguf(FixtureRow),
    /// A checkpoint folder `fixture cut` makes in its stored bits ([`CutRow`]).
    Cut(CutRow),
}

impl Fixture {
    /// The fixture's directory under the fixture root.
    pub fn family(&self) -> &'static str {
        match self {
            Fixture::Gguf(row) => row.family,
            Fixture::Cut(row) => row.family,
        }
    }
}

/// A family's GGUF gate fixture and the hub source it is made from: `fixture fetch <family>` reads
/// the row, and an explicit `<repo>:<quant>` overrides it. The row holds data only; the family's
/// fixture spec is its entry in [`FIXTURE_SPECS`].
#[derive(Clone, Copy)]
pub struct FixtureRow {
    /// The name `fetch` takes: the family's directory under the fixture root.
    pub family: &'static str,
    pub repo: &'static str,
    pub quant: &'static str,
    /// The repo revision the fixture's source header is the header of (`main` when `None`).
    pub rev: Option<&'static str>,
    pub draft: DraftSource,
}

/// A family's checkpoint-folder fixture: a hub folder's safetensors kept in their stored bits, the
/// chosen layers and widths only (`fixture cut`, verified by `fixture verify`). The row holds the
/// directory; the cut's rules, its source folder and revision are `tools/fixture-cuts/<family>.json`.
#[derive(Clone, Copy)]
pub struct CutRow {
    /// The family's directory under the fixture root, and its rules file's stem.
    pub family: &'static str,
}

/// A checkpoint-folder fixture of a gate profile that no row names: the profile's gates keep the
/// row's own fixture (or run on its real file) and only the clauses that name this cut read it. A
/// row holds one [`Fixture`], so a second meaning of one architecture is a profile cut, never the
/// row's. The directory is the profile's row in `tools/fixture-dirs.tsv`.
#[derive(Clone, Copy)]
pub struct ProfileCut {
    /// The `BLOOMERY_MODEL` profile whose row in `tools/fixture-dirs.tsv` names the cut's directory.
    /// No row of [`ROWS`] holds it as one of its strings.
    pub profile: &'static str,
    /// The canonical architecture string of the row whose reader reads the folder; the rules file
    /// names the same one.
    pub arch: &'static str,
    pub cut: CutRow,
}

/// One group of architecture strings and what serves them. A string is in one row.
pub struct ArchRow {
    /// The [`Arch`] the strings name; `None` for strings a reader takes that no `Arch` stands for.
    pub arch: Option<Arch>,
    /// The `general.architecture` strings, the canonical one first: the architecture of the row's
    /// fixture, and [`Arch::name`] when the row has an [`Arch`]. Never empty.
    pub names: &'static [&'static str],
    /// The family's reader; `None` for an architecture with no reader here.
    pub read: Option<Reader>,
    /// Whether a set of this architecture can run its draft file ([`draft::draft_usable`]); `None`
    /// for an architecture with no rule.
    pub draft: Option<Rule>,
    /// The coverage rows only this family's program runs, which [`super::coverage::rows`] adds to
    /// the common table for a file of this row; `None` for a family the common table covers whole.
    pub coverage: Option<&'static [Available]>,
    pub fixture: Option<Fixture>,
}

impl ArchRow {
    /// The canonical architecture string.
    pub fn canonical(&self) -> &'static str {
        self.names[0]
    }
}

/// Every row. The order is the fixture list's order; no lookup depends on it.
pub static ROWS: &[ArchRow] = &[
    ArchRow {
        arch: Some(Arch::Deepseek2),
        names: &["deepseek2"],
        read: None,
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::Deepseek41),
        names: &["deepseek41", DEEPSEEK4],
        read: Some(deepseek41::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "v41",
            repo: "vcruz305/DeepSeek-V4.1-Flash-GGUF",
            quant: "Q3_K_M",
            rev: None,
            draft: DraftSource::NotFetchable(
                "the draft fixture is made from our own DSpark copy, and no public file has its header",
            ),
        })),
    },
    ArchRow {
        arch: Some(Arch::Qwen3moe),
        names: &["qwen3moe"],
        read: Some(qwen3moe::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "qwen3moe",
            repo: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
            quant: "Q4_K_M",
            // The repo's head commit when the row was written (the hub's API): its Q4_K_M file
            // lists the box's file's size and sha256, so `fixture fetch qwen3moe` at it writes the
            // same bytes as `fixture generate` from the box's copy.
            rev: Some("eea7b2be5805a5f151f8847ede8e5f9a9284bf77"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: None,
        names: &["qwen4exp"],
        read: Some(qwen35moe::spec::read),
        draft: Some(draft::qwen4exp_usable),
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "qwen38",
            repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
            quant: "UD-Q4_K_XL",
            rev: None,
            draft: DraftSource::File("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
        })),
    },
    ArchRow {
        arch: Some(Arch::Qwen35moe),
        names: &["qwen35moe"],
        read: Some(qwen35moe::spec::read),
        draft: Some(draft::qwen35moe_usable),
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "qwen35moe",
            repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
            quant: "Q4_K_M",
            // The commit whose header the spec's map was chosen from, pinned so a rewritten upstream
            // file is a new fixture on purpose.
            rev: Some("68a34855558af61cbef0324d31f411be8a506b08"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: None,
        names: QWEN35_BODY,
        read: Some(qwen35moe::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        // No `Arch` variant: the Qwen3.5 body runs the file, as it runs `qwen35`.
        arch: None,
        names: &["llama"],
        read: Some(llama::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        // No `Arch` variant, and no family rows: the program that names the file is Qwen3.5's, whose
        // common rows have none for the normed NEOX head, so `coverage::check` lists it.
        arch: None,
        names: &["qwen3"],
        read: Some(llama::spec::read_qwen3),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        // No `Arch` variant: no program runs the file yet.
        arch: None,
        names: &["lfm3"],
        read: Some(lfm3::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::Glm5next),
        names: &["glm5next", GLM5_NEXT_LLAMA_CPP],
        read: Some(glm5next::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "glm5next",
            repo: "unsloth/GLM-5.3-Flash-GGUF",
            quant: "UD-Q4_K_XL",
            // unsloth's later rewrite of shard 1 renamed `glm5next` to `glm5-next`; this is the
            // commit before, whose header the fixture was made from.
            rev: Some("621d456e93"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: Some(Arch::Mimo2),
        names: &["mimo2"],
        read: Some(mimo2::spec::read),
        draft: Some(draft::mimo2_usable),
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::MiniMaxM3),
        names: &["minimax-m3"],
        read: Some(minimaxm3::spec::read),
        draft: None,
        coverage: Some(minimaxm3::spec::AVAILABLE),
        fixture: Some(Fixture::Cut(CutRow {
            family: "minimaxm31",
        })),
    },
    ArchRow {
        arch: Some(Arch::Raon1),
        names: &["raon1"],
        read: Some(raon1::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::SolarOpen2),
        names: &["solar-open2"],
        read: Some(solar2::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "solar2",
            repo: "llamaraspberryrabbit/Solar-Open2-250B-GGUF",
            quant: "Q4_K_M",
            // The repo's head commit when the row was written (the hub's API): its four Q4_K_M
            // shards list the byte sizes of the box's four files, and `fixture fetch solar2` at
            // it writes the same shards as `fixture generate` from the box's first shard.
            rev: Some("6e78cf22866264a9e3052c2f02ca056bf79c9ac9"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: Some(Arch::NemotronH),
        names: &["nemotron_h"],
        read: Some(nemotron_h::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "nemotron",
            repo: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            quant: "Q4_K_M",
            // The commit the 2,837,072,864-byte file's header was read at.
            rev: Some("ba223d14e45525f7fae81db77ea8cabeb2fc6c25"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: Some(Arch::NemotronHMoe),
        names: &["nemotron_h_moe"],
        read: Some(nemotron_h_moe::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "nemotron-bolt",
            repo: "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
            quant: "24GB",
            // The repo's head commit when the row was written (the hub's API); the box's copy of
            // the file was not compared with it.
            rev: Some("e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246"),
            draft: DraftSource::None,
        })),
    },
    ArchRow {
        arch: Some(Arch::GemmaEmbedding),
        names: &[GEMMA_EMBEDDING],
        read: Some(gemma_embedding::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::GemmaEmbedding2),
        names: &[GEMMA_EMBEDDING2],
        read: Some(gemma_embedding2::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::Flint),
        names: &["flint"],
        read: Some(flint::spec::read),
        draft: None,
        coverage: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::MuseSpark),
        names: &["muse-spark"],
        read: Some(musespark::spec::read),
        draft: None,
        coverage: None,
        fixture: Some(Fixture::Gguf(FixtureRow {
            family: "musespark",
            repo: "unsloth/Muse-Spark-Open-816B-A42B-GGUF",
            quant: "UD-IQ2_XXS",
            // The commit the specimen's header was read at (the design round's snapshot).
            rev: Some("c00670568b6c"),
            draft: DraftSource::None,
        })),
    },
];

/// Every profile's cut fixture, in table order.
pub static PROFILE_CUTS: &[ProfileCut] = &[
    ProfileCut {
        // The dense Qwen3.8 NVFP4 checkpoint: only the NVFP4 gate clauses read it, and `qwen35`
        // keeps its `self` row for the K-quant gates.
        profile: "qwen38d",
        arch: "qwen35",
        cut: CutRow { family: "qwen38nv" },
    },
    ProfileCut {
        // The Qwen3.8-Flash-Next NVFP4 checkpoint: only the NVFP4 gate reads it, and `qwen4exp`
        // keeps its K-quant GGUF fixture for the other gates.
        profile: "qwen4exp-nvfp4",
        arch: "qwen4exp",
        cut: CutRow {
            family: "qwen38nvfp4",
        },
    },
    ProfileCut {
        // The same layers of the FP4-Local checkpoint, whose routed stacks store `csf` block
        // scales: the NVFP4 gate's fixture tier loads it beside the NVFP4 cut.
        profile: "qwen4exp-fp4l",
        arch: "qwen4exp",
        cut: CutRow {
            family: "qwen38fp4l",
        },
    },
];

/// The row whose strings include `arch`.
pub fn row_of(arch: &str) -> Option<&'static ArchRow> {
    ROWS.iter().find(|row| row.names.contains(&arch))
}

/// The GGUF fixture row of the family whose canonical architecture string is `arch`; a family
/// whose fixture is a cut folder has none here ([`cuts`]).
pub fn fixture_of(arch: &str) -> Option<&'static FixtureRow> {
    ROWS.iter()
        .find(|row| row.canonical() == arch)
        .and_then(|row| match &row.fixture {
            Some(Fixture::Gguf(fx)) => Some(fx),
            _ => None,
        })
}

/// Every GGUF fixture row, in registry order.
pub fn fixtures() -> impl Iterator<Item = &'static FixtureRow> {
    ROWS.iter().filter_map(|row| match &row.fixture {
        Some(Fixture::Gguf(fx)) => Some(fx),
        _ => None,
    })
}

/// Every cut-folder fixture, in registry order: the rows' cuts, then [`PROFILE_CUTS`].
pub fn cuts() -> impl Iterator<Item = &'static CutRow> {
    ROWS.iter()
        .filter_map(|row| match &row.fixture {
            Some(Fixture::Cut(cut)) => Some(cut),
            _ => None,
        })
        .chain(PROFILE_CUTS.iter().map(|p| &p.cut))
}

/// The registry as `tools/arch-registry.tsv` holds it: a comment line, a header line, then one
/// tab-separated line per row: its strings (comma-separated, the canonical one first), the
/// [`Arch`] variant (`-` for none), whether the row has a reader and a draft rule, and the
/// fixture family (`-` for none).
pub fn print() -> String {
    let mut out = String::from(
        "# the rows of model::arch::registry::ROWS, as `fixture registry` prints them\n\
         names\tvariant\tread\tdraft\tfixture\n",
    );
    let yes = |has: bool| if has { "yes" } else { "no" };
    for row in ROWS {
        let variant = row
            .arch
            .map_or_else(|| "-".to_string(), |a| format!("{a:?}"));
        let fixture = row.fixture.map_or("-", |f| f.family());
        out.push_str(&format!(
            "{}\t{variant}\t{}\t{}\t{fixture}\n",
            row.names.join(","),
            yes(row.read.is_some()),
            yes(row.draft.is_some()),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each family's reader by its module path: the statement of which reader reads which
    /// architecture that the rows are held to, written apart from them.
    fn direct(arch: &str) -> Reader {
        match arch {
            "deepseek41" | DEEPSEEK4 => deepseek41::spec::read,
            "qwen3moe" => qwen3moe::spec::read,
            "qwen35moe" | "qwen4exp" => qwen35moe::spec::read,
            a if QWEN35_BODY.contains(&a) => qwen35moe::spec::read,
            "glm5next" | GLM5_NEXT_LLAMA_CPP => glm5next::spec::read,
            "llama" => llama::spec::read,
            "qwen3" => llama::spec::read_qwen3,
            "lfm3" => lfm3::spec::read,
            "mimo2" => mimo2::spec::read,
            "minimax-m3" => minimaxm3::spec::read,
            "raon1" => raon1::spec::read,
            "solar-open2" => solar2::spec::read,
            "nemotron_h" => nemotron_h::spec::read,
            "nemotron_h_moe" => nemotron_h_moe::spec::read,
            GEMMA_EMBEDDING => gemma_embedding::spec::read,
            GEMMA_EMBEDDING2 => gemma_embedding2::spec::read,
            "muse-spark" => musespark::spec::read,
            other => panic!("{other}: no reader is named for it"),
        }
    }

    /// Strings are in one row, a row has a canonical string, every [`Arch`] variant has exactly
    /// one row, and a string is the name of the variant its row holds.
    #[test]
    fn rows_partition_the_strings_and_the_variants() {
        let mut seen: Vec<&str> = Vec::new();
        for row in ROWS {
            assert!(!row.names.is_empty(), "a row with no string");
            for name in row.names {
                assert!(!seen.contains(name), "{name} is in two rows");
                seen.push(name);
            }
            assert_eq!(
                row_of(row.canonical()).map(|r| r.canonical()),
                Some(row.canonical())
            );
            if let Some(arch) = row.arch {
                assert_eq!(arch.name(), row.canonical());
                for name in row.names {
                    assert_eq!(Arch::from_name(name).ok(), Some(arch), "{name}");
                }
            } else {
                for name in row.names {
                    assert!(Arch::from_name(name).is_err(), "{name} names an Arch");
                }
            }
        }
        for arch in [
            Arch::Deepseek2,
            Arch::Deepseek41,
            Arch::Qwen3moe,
            Arch::Qwen35moe,
            Arch::Glm5next,
            Arch::Mimo2,
            Arch::SolarOpen2,
            Arch::Raon1,
            Arch::NemotronH,
            Arch::NemotronHMoe,
            Arch::GemmaEmbedding,
            Arch::GemmaEmbedding2,
            Arch::Flint,
            Arch::MuseSpark,
        ] {
            let rows = ROWS.iter().filter(|r| r.arch == Some(arch)).count();
            assert_eq!(rows, 1, "{arch:?} is in {rows} rows");
        }
    }

    /// A fixture row's family name is its directory under the fixture root (unique across both
    /// kinds), the row has a reader, and each kind's fixtures are listed in the order the fixture
    /// tool has always listed them.
    #[test]
    fn a_fixture_belongs_to_its_row() {
        let mut families: Vec<&str> = Vec::new();
        for row in ROWS {
            let Some(fx) = row.fixture else { continue };
            let family = fx.family();
            assert!(
                row.read.is_some(),
                "{family}: a fixture of a family with no reader"
            );
            assert!(!families.contains(&family), "{family} twice");
            families.push(family);
        }
        let archs_of = |gguf: bool| -> Vec<&str> {
            ROWS.iter()
                .filter(|r| matches!(r.fixture, Some(Fixture::Gguf(_))) == gguf)
                .filter(|r| r.fixture.is_some())
                .map(|r| r.canonical())
                .collect()
        };
        assert_eq!(
            archs_of(true),
            [
                "deepseek41",
                "qwen3moe",
                "qwen4exp",
                "qwen35moe",
                "glm5next",
                "solar-open2",
                "nemotron_h",
                "nemotron_h_moe",
                "muse-spark"
            ]
        );
        assert_eq!(archs_of(false), ["minimax-m3"]);
    }

    /// A row's fixture family is the directory `tools/fixture-dirs.tsv` gives its architecture (refset
    /// reads the table), and a row with no fixture has no directory there: the registry's literals are
    /// held to the one table.
    #[test]
    fn a_fixture_family_is_the_tables_directory() {
        for row in ROWS {
            let canonical = row.canonical();
            assert_eq!(
                row.fixture.map(|fx| fx.family()),
                refset::fixture::dir_of(canonical),
                "{canonical}: the row's fixture family and tools/fixture-dirs.tsv disagree"
            );
        }
    }

    /// A cut fixture's rules are `tools/fixture-cuts/<family>.json`, and a rules file there is a
    /// cut row's or a profile cut's: a cut with no file, or a file no cut names, cuts nothing.
    /// (The rules' own name and architecture are held to the cut by `fixture`'s tests, which
    /// parse them.)
    #[test]
    fn a_cut_fixture_is_its_rules_file() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/fixture-cuts");
        let mut files: Vec<String> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        let mut rows: Vec<String> = cuts().map(|c| format!("{}.json", c.family)).collect();
        rows.sort();
        assert_eq!(
            files, rows,
            "tools/fixture-cuts holds a rules file per cut row and nothing else"
        );
    }

    /// A profile cut is for a profile no row names (a row's own fixture is its `fixture`), the
    /// directory `tools/fixture-dirs.tsv` gives that profile, read by the reader of the row its
    /// architecture names, and a family no row's fixture already is.
    #[test]
    fn a_profile_cut_is_its_profiles_directory() {
        assert!(!PROFILE_CUTS.is_empty(), "no profile cut to hold");
        for p in PROFILE_CUTS {
            let (profile, family) = (p.profile, p.cut.family);
            assert!(
                row_of(profile).is_none(),
                "{profile}: a row names it, so its fixture is the row's, not a profile cut"
            );
            assert_eq!(
                refset::fixture::dir_of(profile),
                Some(family),
                "{profile}: tools/fixture-dirs.tsv gives the profile another directory than its cut"
            );
            let row = row_of(p.arch).unwrap_or_else(|| panic!("{family}: no row names {}", p.arch));
            assert_eq!(
                row.canonical(),
                p.arch,
                "{family}: not the row's canonical string"
            );
            assert!(
                row.read.is_some(),
                "{family}: a cut of a family with no reader"
            );
            assert!(
                ROWS.iter()
                    .all(|r| r.fixture.is_none_or(|fx| fx.family() != family)),
                "{family}: a row's fixture is already this directory"
            );
        }
    }

    /// The draft rules are the old table's three keys, `llama` has a reader row and no draft rule,
    /// and an architecture the rows do not name has no reader row.
    #[test]
    fn the_draft_rules_are_the_old_tables_keys() {
        let with: Vec<&str> = ROWS
            .iter()
            .filter(|r| r.draft.is_some())
            .map(|r| r.canonical())
            .collect();
        assert_eq!(with, ["qwen4exp", "qwen35moe", "mimo2"]);
        assert!(row_of("llama").is_some_and(|r| r.read.is_some() && r.draft.is_none()));
        assert!(row_of("llama4").is_none());
        assert!(row_of("deepseek2").is_some_and(|r| r.read.is_none()));
    }

    /// `tools/arch-registry.tsv` is the registry's print; regenerate it with
    /// `fixture registry > tools/arch-registry.tsv`.
    #[test]
    fn the_committed_print_is_the_registry() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/arch-registry.tsv");
        let committed = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(
            committed,
            print(),
            "tools/arch-registry.tsv is not the registry's print; run `fixture registry`"
        );
    }

    /// Every family with a fixture reads it through the registry exactly as its own reader does:
    /// the same `Read`, field by field, over the written fixture's first shard. A row that
    /// points at another family's reader, or an architecture the registry sends elsewhere, reads
    /// a fixture to an error or a different description. The registry's digest of each read is
    /// printed, for a base-against-head comparison of the table.
    #[test]
    #[ignore = "reads the written fixtures under the fixture root: gate-arch-registry"]
    fn hw_registry_reads_like_before() {
        use sha2::{Digest, Sha256};

        let mut read = Vec::new();
        let mut bad = Vec::new();
        for row in ROWS {
            // A cut folder is safetensors, not a GGUF this table's readers open; its own reader
            // reads it.
            let Some(Fixture::Gguf(fx)) = row.fixture else {
                continue;
            };
            let canonical = row.canonical();
            let first = match refset::fixture::first_shard(canonical) {
                Ok(first) => first,
                Err(e) => {
                    bad.push(format!("{}: {e}", fx.family));
                    continue;
                }
            };
            if !first.contains(&format!("/{}/", fx.family)) {
                bad.push(format!(
                    "{}: the fixture root holds it at {first}, whose directory is not the row's family",
                    fx.family
                ));
            }
            let split = match gguf::Split::open(&first) {
                Ok(split) => split,
                Err(e) => {
                    bad.push(format!("{first}: {e}"));
                    continue;
                }
            };
            let through = match crate::arch::spec(&split) {
                Ok(read) => format!("{read:#?}"),
                Err(e) => {
                    bad.push(format!(
                        "{first}: the registry cannot read the fixture: {e}"
                    ));
                    continue;
                }
            };
            let own = match direct(canonical)(&split) {
                Ok(read) => format!("{read:#?}"),
                Err(e) => format!("{}", crate::ModelError::from(e)),
            };
            if through != own {
                bad.push(format!(
                    "{}: the registry's read of {first} differs from {canonical}'s own reader",
                    fx.family
                ));
            }
            let digest: String = Sha256::digest(through.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            println!(
                "registry read {} sha256={digest} bytes={}",
                fx.family,
                through.len()
            );
            read.push(fx.family);
        }
        assert!(bad.is_empty(), "registry reads differ:\n{}", bad.join("\n"));
        assert_eq!(read.len(), fixtures().count(), "a fixture row was not read");
    }
}
