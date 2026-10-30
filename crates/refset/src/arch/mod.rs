//! Each architecture's families, and the table of all of them. An
//! architecture's families live in its own module; shared code reaches them
//! by the architecture's name — the value its sets' `# arch` line carries,
//! which `model::arch::Arch::name` spells — through [`families`] and
//! [`node_dumps`], never by module path.

use crate::family::{Family, Identity};

pub use draft::{BesideDraft, DraftFrom, beside_draft, beside_drafts};

pub mod deepseek41;
pub mod deepseek41v;
pub mod dequant;
pub mod draft;
pub mod forced_logits;
pub mod gemma_embedding;
pub mod gemma_embedding2;
pub mod glm5next;
pub mod lfm3;
pub mod llama;
pub mod mimo2;
pub mod minicpmv4_7;
pub mod minimaxm3;
pub mod minimaxm31;
pub mod mistrallarge4;
pub mod musespark;
pub mod nemotron_h;
pub mod nemotron_h_moe;
pub mod prompt_tables;
pub mod quant;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3moe;
pub mod qwen4exp;
pub mod raon1;
pub mod solar2;
pub mod tokenizer;

/// The families of the architecture MiniMax-M3 and M3.1 share, the GGUF set's llama.cpp dumps first ([`node_dumps`] answers with
/// them) and the checkpoint folder's transformers dumps after: [`minimaxm3::FAMILIES`] then [`minimaxm31::FAMILIES`], which
/// `minimax_rows_are_both_modules_families` holds.
static MINIMAX_M3: &[&Family] = &[&minimaxm3::LCPP, &minimaxm31::TF, &minimaxm31::POOLED];

/// Each architecture's families, by the architecture's name.
static BY_ARCH: &[(&str, &[&Family])] = &[
    (deepseek41::ARCH, deepseek41::FAMILIES),
    (deepseek41v::ARCH, deepseek41v::FAMILIES),
    (qwen3moe::ARCH, qwen3moe::FAMILIES),
    (qwen35moe::ARCH, qwen35moe::FAMILIES),
    (qwen35::ARCH, qwen35::FAMILIES),
    (llama::ARCH, llama::FAMILIES),
    (glm5next::ARCH, glm5next::FAMILIES),
    (mimo2::ARCH, mimo2::FAMILIES),
    (minimaxm3::ARCH, MINIMAX_M3),
    (solar2::ARCH, solar2::FAMILIES),
    (minicpmv4_7::ARCH, minicpmv4_7::FAMILIES),
    (qwen4exp::ARCH, qwen4exp::FAMILIES),
    (raon1::ARCH, raon1::FAMILIES),
    (nemotron_h::ARCH, nemotron_h::FAMILIES),
    (nemotron_h_moe::ARCH, nemotron_h_moe::FAMILIES),
    (lfm3::ARCH, lfm3::FAMILIES),
    (musespark::ARCH, musespark::FAMILIES),
    (mistrallarge4::ARCH, mistrallarge4::FAMILIES),
    (tokenizer::ARCH, tokenizer::FAMILIES),
    (dequant::ARCH, dequant::FAMILIES),
    (quant::ARCH, quant::FAMILIES),
    (gemma_embedding::ARCH, gemma_embedding::FAMILIES),
    (gemma_embedding2::ARCH, gemma_embedding2::FAMILIES),
];

/// The families of architecture `arch`; none for one the table does not hold.
#[must_use]
pub fn families(arch: &str) -> &'static [&'static Family] {
    BY_ARCH
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|&(_, f)| f)
        .unwrap_or(&[])
}

/// The node-dump family of architecture `arch` ([`Identity::Manifest`]):
/// every set of it opens through that family's check.
#[must_use]
pub fn node_dumps(arch: &str) -> Option<&'static Family> {
    families(arch)
        .iter()
        .copied()
        .find(|f| f.identity == Identity::Manifest)
}

/// Every family of the table.
pub fn all() -> impl Iterator<Item = &'static Family> {
    BY_ARCH.iter().flat_map(|(_, f)| f.iter().copied())
}

/// The family named `name`.
#[must_use]
pub fn named(name: &str) -> Option<&'static Family> {
    all().find(|f| f.name == name)
}

/// A [`NEEDS_ENV`] row: the family, its variable, and the variable's reader.
type NeedsEnvRow = (&'static Family, &'static str, fn() -> bool);

/// The families whose sets exist only where a machine fact holds, each with the variable that states it: every consumer gate of
/// the family opens with `bash tools/ref/needs-env.sh <recipe> <VAR>`, so a machine without VAR defers the gate by name and has
/// no sets for it. `needs_env_rows_are_the_families_whose_consumers_open_with_needs_env` holds this table to the justfile.
static NEEDS_ENV: &[NeedsEnvRow] = &[(
    &qwen4exp::nvfp4::HF,
    "BLOOMERY_CARD_ALL_EXPERTS",
    card_all_experts,
)];

/// Whether `BLOOMERY_CARD_ALL_EXPERTS` is set and not empty here. A row carries its own reader because
/// `tools/check-levers.sh` accepts a direct read only by a literal name, which a lookup by the row's string is not.
fn card_all_experts() -> bool {
    std::env::var_os("BLOOMERY_CARD_ALL_EXPERTS").is_some_and(|v| !v.is_empty())
}

/// The variable family `f`'s sets wait on, when it is unset or empty here: the sets are not in place on this machine by design, and
/// a check of every family's sets in place names the family as deferred instead of failing on sets nothing here reads.
#[must_use]
pub fn deferred(f: &Family) -> Option<&'static str> {
    NEEDS_ENV
        .iter()
        .find(|(g, _, _)| std::ptr::eq(*g, f))
        .filter(|(_, _, set)| !set())
        .map(|&(_, var, _)| var)
}

#[cfg(test)]
mod node_dump_contract;
#[cfg(test)]
mod prompt_tables_tests;
#[cfg(test)]
mod qwen_vis_tests;

#[cfg(test)]
mod tests {
    use super::{
        BY_ARCH, NEEDS_ENV, all, deferred, families, minimaxm3, minimaxm31, named, node_dumps,
    };
    use crate::family::Identity;

    /// The fixture families whose real file's sets have no refset family: the card gates open
    /// those by name. Each is checked in its own module.
    const STANDALONE: [&str; 1] = ["fx-ik-qwen3moe"];

    /// One row an architecture: `families` answers with the first row of a name, so a second row would hide its families.
    #[test]
    fn each_architecture_has_one_row() {
        for (i, (arch, _)) in BY_ARCH.iter().enumerate() {
            assert!(
                BY_ARCH[..i].iter().all(|(a, _)| a != arch),
                "{arch} has two rows in BY_ARCH: merge them into one slice"
            );
        }
    }

    /// MiniMax-M3 and M3.1 share an architecture name, so one row holds both modules' families, M3's first: the gates' node dumps
    /// stay the llama.cpp ones, and a family a module gains is in the table or this is red.
    #[test]
    fn minimax_rows_are_both_modules_families() {
        let want: Vec<&str> = minimaxm3::FAMILIES
            .iter()
            .chain(minimaxm31::FAMILIES)
            .map(|f| f.name)
            .collect();
        let got: Vec<&str> = families(minimaxm3::ARCH).iter().map(|f| f.name).collect();
        assert_eq!(got, want);
        assert_eq!(
            got,
            ["lcpp-minimaxm3", "tf-minimaxm31", "tf-minimaxm31-pooled"]
        );
        assert_eq!(minimaxm31::ARCH, minimaxm3::ARCH);
        for set in minimaxm31::SETS {
            assert!(
                !minimaxm3::SETS.contains(set),
                "{set} is a minimaxm3 set: one data directory holds both families"
            );
        }
        assert_eq!(
            node_dumps(minimaxm3::ARCH).map(|f| f.name),
            Some("lcpp-minimaxm3")
        );
        assert!(
            named("tf-minimaxm31").is_some_and(|f| std::ptr::eq(f, &minimaxm31::TF)),
            "tf-minimaxm31 is not in the table"
        );
    }

    /// Every family's sets in place pass their family's check: each states
    /// the file the tree runs, names its family's ik build where the family
    /// pins one, and is complete. One line per set, then the verdict. A family
    /// whose consumer needs a machine fact this machine lacks ([`deferred`]) is
    /// named and left out: its sets are written where that fact holds.
    #[test]
    #[ignore = "hw: needs the reference sets in $BLOOMERY_DATA and BLOOMERY_DSPARK_MODEL on the box"]
    fn hw_every_set_in_place_is_its_familys() {
        let mut failed = 0usize;
        let mut sets = 0usize;
        let mut left = 0usize;
        for f in all() {
            if let Some(var) = deferred(f) {
                left += 1;
                println!(
                    "hw_refset: {}: deferred — its {} set(s) wait on {var}, unset here",
                    f.name,
                    f.sets.len()
                );
                continue;
            }
            for path in f.in_place() {
                sets += 1;
                match f.check_set(&path) {
                    Ok(p) => println!(
                        "hw_refset: {} {}: dumped from {}{} — PASS",
                        f.name,
                        path.display(),
                        p.dumped_from,
                        p.build.map_or_else(String::new, |b| format!(" build {b}"))
                    ),
                    Err(e) => {
                        failed += 1;
                        println!("hw_refset: {} {}: FAIL {e}", f.name, path.display());
                    }
                }
            }
        }
        println!("hw_refset: {sets} set(s) in place, {failed} failed, {left} deferred");
        assert_eq!(
            failed, 0,
            "hw_refset: {failed} set(s) in place fail their family's check"
        );
    }

    /// The variable `recipe` of `justfile` opens its box command with: the word after `needs-env.sh <recipe>` in the recipe's
    /// indented lines, as `tools/gate-batch.sh` reads the same call. None for a recipe that is not there or opens with no guard.
    fn needs_env_of(justfile: &str, recipe: &str) -> Option<String> {
        let mut lines = justfile.lines();
        lines.find(|l| {
            l.strip_prefix(recipe)
                .is_some_and(|rest| rest.starts_with(':') || rest.starts_with(' '))
        })?;
        let guard = format!("needs-env.sh {recipe} ");
        lines
            .take_while(|l| l.is_empty() || l.starts_with(char::is_whitespace))
            .find_map(|l| {
                let rest = &l[l.find(&guard)? + guard.len()..];
                let end = rest
                    .find(|c: char| c.is_whitespace() || c == '\'')
                    .unwrap_or(rest.len());
                Some(rest[..end].to_string())
            })
    }

    /// A family whose every consumer gate opens with `bash tools/ref/needs-env.sh <recipe> <VAR>`, one VAR for all, is in
    /// `NEEDS_ENV` with that VAR, and every row is such a family: a gate a machine defers by name leaves its sets out of the
    /// in-place check by name too, a new such family is not left to turn the check red on a machine that cannot hold its sets,
    /// and a row for a family whose consumer needs nothing would excuse missing sets.
    #[test]
    fn needs_env_rows_are_the_families_whose_consumers_open_with_needs_env() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../justfile");
        let justfile =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut derived: Vec<(&str, String)> = Vec::new();
        for f in all() {
            let vars: Vec<Option<String>> = f
                .consumers
                .iter()
                .map(|c| needs_env_of(&justfile, c))
                .collect();
            if let Some(Some(var)) = vars.first()
                && vars.iter().all(|v| v.as_deref() == Some(var.as_str()))
            {
                derived.push((f.name, var.clone()));
            }
        }
        let table: Vec<(&str, String)> = NEEDS_ENV
            .iter()
            .map(|&(f, var, _)| (f.name, var.to_string()))
            .collect();
        assert_eq!(
            derived, table,
            "the families whose consumers open with needs-env.sh (left) are not NEEDS_ENV's rows (right)"
        );
    }

    /// The fixture families are the table of eight, each the mirror of one real
    /// family: its name is `fx-` and the real family's, and it takes the real
    /// family's architecture, ik build and consumers, the fixture twin of its
    /// identity, a draft file when the real family has one and no per-file set
    /// name; each set is the real family's set under the `fx_` prefix
    /// (`ref-mtp/fx_` for an MTP set). The real family's node dumps stay the
    /// architecture's. [`STANDALONE`] are the fixture families whose real sets
    /// have no family, held by their own module's test.
    #[test]
    fn each_fixture_family_mirrors_a_real_one() {
        let sets_of = [
            ("fx-ik-glm5next", 4),
            ("fx-ik-glm5next-dsa", 2),
            ("fx-mtp-glm5next", 1),
            ("fx-ik-qwen4exp", 5),
            ("fx-mtp-qwen4exp", 1),
            ("fx-ik-qwen35moe", 3),
            ("fx-ik-deepseek41", 4),
            ("fx-ref-nemotron-moe-logits", 1),
        ];
        let mut found: Vec<&str> = all()
            .filter(|f| f.name.starts_with("fx-") && !STANDALONE.contains(&f.name))
            .map(|f| f.name)
            .collect();
        found.sort_unstable();
        let mut want: Vec<&str> = sets_of.iter().map(|(n, _)| *n).collect();
        want.sort_unstable();
        assert_eq!(found, want);

        for (name, count) in sets_of {
            let fx = named(name).unwrap_or_else(|| panic!("{name} is not in the table"));
            let real_name = &name["fx-".len()..];
            let real = named(real_name).unwrap_or_else(|| panic!("{name} mirrors no {real_name}"));
            assert_eq!(fx.arch, real.arch, "{name}: arch");
            assert_eq!(fx.build, real.build, "{name}: build");
            assert_eq!(fx.consumers, real.consumers, "{name}: consumers");
            assert_eq!(
                fx.draft_runs.is_some(),
                real.draft_runs.is_some(),
                "{name}: draft"
            );
            assert!(
                fx.resolve.is_none(),
                "{name}: one fixture file, no per-file suffix"
            );
            assert!(
                fx.recipe
                    .starts_with("BLOOMERY_TIER=fixture just dump-ref-"),
                "{name}: {}",
                fx.recipe
            );
            let identity = match real.identity {
                Identity::Manifest => Identity::FixtureManifest,
                Identity::MtpManifest => Identity::FixtureMtpManifest,
                Identity::NemotronLogits => Identity::FixtureNemotronLogits,
                other => panic!("{name}: {other:?} has no fixture twin"),
            };
            assert_eq!(fx.identity, identity, "{name}: identity");
            assert_eq!(fx.sets.len(), count, "{name}: sets");
            for set in fx.sets {
                let twin = match set.strip_prefix("ref-mtp/fx_") {
                    Some(rest) => format!("ref-mtp/{rest}"),
                    None => set
                        .strip_prefix("fx_")
                        .unwrap_or_else(|| panic!("{name}: {set} has no fx_ prefix"))
                        .to_string(),
                };
                assert!(
                    real.sets.contains(&twin.as_str()),
                    "{name}: {set} twins {twin}, which {real_name} does not hold"
                );
            }
        }
        for arch in ["glm5next", "qwen4exp", "deepseek41", "qwen35moe"] {
            let node = node_dumps(arch).map(|f| f.name);
            assert_eq!(node, Some(&*format!("ik-{arch}")), "{arch}: node dumps");
        }
    }
}
