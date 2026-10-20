//! The architecture registry: one [`ArchRow`] per group of `general.architecture` strings that one
//! reader, one draft rule and one gate fixture serve. The dispatch on a file's architecture reads
//! these rows: the reader ([`super::spec`]), a target's draft rule ([`super::draft::draft_usable`])
//! and the fixture of a family (the `fixture` binary). A family is added by one row here, and no
//! `match` on a model.
//!
//! [`Arch::name`] and [`Arch::from_name`] stay the enum's own: a row holds function pointers, so a
//! caller of the registry reaches every reader, and the many that only name an architecture do not
//! read it. The registry's test holds the two to the same strings.
//!
//! [`print`] renders the rows; `tools/arch-registry.tsv` is its committed copy, which
//! `tools/hf-arch-check.py` reads instead of mirroring the strings.

use gguf::Split;

use super::draft::Rule;
use super::{
    Arch, DEEPSEEK4, GEMMA_EMBEDDING, GEMMA_EMBEDDING2, GLM5_NEXT_LLAMA_CPP, QWEN35_BODY, Read,
    deepseek41, draft, gemma_embedding, gemma_embedding2, glm5next, mimo2, minimaxm3, nemotron_h,
    nemotron_h_moe, qwen3moe, qwen35moe,
};
use crate::placement::PlacementError;

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

/// A family's gate fixture and the hub source it is made from: `fixture fetch <family>` reads the
/// row, and an explicit `<repo>:<quant>` overrides it. The row holds data only: the fixture
/// generators are the `fixture` binary's, so no gate that reads this registry reaches them.
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
    pub fixture: Option<FixtureRow>,
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
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::Deepseek41),
        names: &["deepseek41", DEEPSEEK4],
        read: Some(deepseek41::spec::read),
        draft: None,
        fixture: Some(FixtureRow {
            family: "v41",
            repo: "vcruz305/DeepSeek-V4.1-Flash-GGUF",
            quant: "Q3_K_M",
            rev: None,
            draft: DraftSource::NotFetchable(
                "the draft fixture is made from our own DSpark copy, and no public file has its header",
            ),
        }),
    },
    ArchRow {
        arch: Some(Arch::Qwen3moe),
        names: &["qwen3moe"],
        read: Some(qwen3moe::spec::read),
        draft: None,
        fixture: None,
    },
    ArchRow {
        arch: None,
        names: &["qwen4exp"],
        read: Some(qwen35moe::spec::read),
        draft: Some(draft::qwen4exp_usable),
        fixture: Some(FixtureRow {
            family: "qwen38",
            repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
            quant: "UD-Q4_K_XL",
            rev: None,
            draft: DraftSource::File("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
        }),
    },
    ArchRow {
        arch: Some(Arch::Qwen35moe),
        names: &["qwen35moe"],
        read: Some(qwen35moe::spec::read),
        draft: Some(draft::qwen35moe_usable),
        fixture: Some(FixtureRow {
            family: "qwen35moe",
            repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
            quant: "Q4_K_M",
            // The commit whose header the spec's map was chosen from, pinned so a rewritten upstream
            // file is a new fixture on purpose.
            rev: Some("68a34855558af61cbef0324d31f411be8a506b08"),
            draft: DraftSource::None,
        }),
    },
    ArchRow {
        arch: None,
        names: QWEN35_BODY,
        read: Some(qwen35moe::spec::read),
        draft: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::Glm5next),
        names: &["glm5next", GLM5_NEXT_LLAMA_CPP],
        read: Some(glm5next::spec::read),
        draft: None,
        fixture: Some(FixtureRow {
            family: "glm5next",
            repo: "unsloth/GLM-5.3-Flash-GGUF",
            quant: "UD-Q4_K_XL",
            // unsloth's later rewrite of shard 1 renamed `glm5next` to `glm5-next`; this is the
            // commit before, whose header the fixture was made from.
            rev: Some("621d456e93"),
            draft: DraftSource::None,
        }),
    },
    ArchRow {
        arch: Some(Arch::Mimo2),
        names: &["mimo2"],
        read: Some(mimo2::spec::read),
        draft: Some(draft::mimo2_usable),
        fixture: None,
    },
    ArchRow {
        // No `Arch` variant until a body runs the file: llama.cpp's spelling is the only string.
        arch: None,
        names: &["minimax-m3"],
        read: Some(minimaxm3::spec::read),
        draft: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::NemotronH),
        names: &["nemotron_h"],
        read: Some(nemotron_h::spec::read),
        draft: None,
        fixture: Some(FixtureRow {
            family: "nemotron",
            repo: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            quant: "Q4_K_M",
            // The commit the 2,837,072,864-byte file's header was read at.
            rev: Some("ba223d14e45525f7fae81db77ea8cabeb2fc6c25"),
            draft: DraftSource::None,
        }),
    },
    ArchRow {
        arch: Some(Arch::NemotronHMoe),
        names: &["nemotron_h_moe"],
        read: Some(nemotron_h_moe::spec::read),
        draft: None,
        fixture: Some(FixtureRow {
            family: "nemotron-bolt",
            repo: "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
            quant: "24GB",
            // The repo's head commit when the row was written (the hub's API); the box's copy of
            // the file was not compared with it.
            rev: Some("e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246"),
            draft: DraftSource::None,
        }),
    },
    ArchRow {
        arch: Some(Arch::GemmaEmbedding),
        names: &[GEMMA_EMBEDDING],
        read: Some(gemma_embedding::spec::read),
        draft: None,
        fixture: None,
    },
    ArchRow {
        arch: Some(Arch::GemmaEmbedding2),
        names: &[GEMMA_EMBEDDING2],
        read: Some(gemma_embedding2::spec::read),
        draft: None,
        fixture: None,
    },
];

/// The row whose strings include `arch`.
pub fn row_of(arch: &str) -> Option<&'static ArchRow> {
    ROWS.iter().find(|row| row.names.contains(&arch))
}

/// The fixture row of the family whose canonical architecture string is `arch`.
pub fn fixture_of(arch: &str) -> Option<&'static FixtureRow> {
    ROWS.iter()
        .find(|row| row.canonical() == arch)
        .and_then(|row| row.fixture.as_ref())
}

/// Every fixture row, in registry order.
pub fn fixtures() -> impl Iterator<Item = &'static FixtureRow> {
    ROWS.iter().filter_map(|row| row.fixture.as_ref())
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
        let fixture = row.fixture.map_or("-", |f| f.family);
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
            "mimo2" => mimo2::spec::read,
            "minimax-m3" => minimaxm3::spec::read,
            "nemotron_h" => nemotron_h::spec::read,
            "nemotron_h_moe" => nemotron_h_moe::spec::read,
            GEMMA_EMBEDDING => gemma_embedding::spec::read,
            GEMMA_EMBEDDING2 => gemma_embedding2::spec::read,
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
            Arch::NemotronH,
            Arch::NemotronHMoe,
            Arch::GemmaEmbedding,
            Arch::GemmaEmbedding2,
        ] {
            let rows = ROWS.iter().filter(|r| r.arch == Some(arch)).count();
            assert_eq!(rows, 1, "{arch:?} is in {rows} rows");
        }
    }

    /// A fixture row's family name is its directory under the fixture root (unique), the row has a
    /// reader, and the fixtures are listed in the order the fixture tool has always listed them.
    #[test]
    fn a_fixture_belongs_to_its_row() {
        let mut families: Vec<&str> = Vec::new();
        for row in ROWS {
            let Some(fx) = row.fixture else { continue };
            assert!(
                row.read.is_some(),
                "{}: a fixture of a family with no reader",
                fx.family
            );
            assert!(!families.contains(&fx.family), "{} twice", fx.family);
            families.push(fx.family);
        }
        let archs: Vec<&str> = ROWS
            .iter()
            .filter(|r| r.fixture.is_some())
            .map(|r| r.canonical())
            .collect();
        assert_eq!(
            archs,
            [
                "deepseek41",
                "qwen4exp",
                "qwen35moe",
                "glm5next",
                "nemotron_h",
                "nemotron_h_moe"
            ]
        );
    }

    /// A row's fixture family is the directory `tools/fixture-dirs.tsv` gives its architecture (refset
    /// reads the table), and a row with no fixture has no directory there: the registry's literals are
    /// held to the one table.
    #[test]
    fn a_fixture_family_is_the_tables_directory() {
        for row in ROWS {
            let canonical = row.canonical();
            assert_eq!(
                row.fixture.map(|fx| fx.family),
                refset::fixture::dir_of(canonical),
                "{canonical}: the row's fixture family and tools/fixture-dirs.tsv disagree"
            );
        }
    }

    /// The draft rules are the old table's three keys, and an architecture the rows do not name
    /// has no reader row.
    #[test]
    fn the_draft_rules_are_the_old_tables_keys() {
        let with: Vec<&str> = ROWS
            .iter()
            .filter(|r| r.draft.is_some())
            .map(|r| r.canonical())
            .collect();
        assert_eq!(with, ["qwen4exp", "qwen35moe", "mimo2"]);
        assert!(row_of("llama").is_none());
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
            let Some(fx) = row.fixture else { continue };
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
