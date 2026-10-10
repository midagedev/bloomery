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
pub mod glm5next;
pub mod mimo2;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen4exp;
pub mod tokenizer;

/// Each architecture's families, by the architecture's name.
static BY_ARCH: &[(&str, &[&Family])] = &[
    (deepseek41::ARCH, deepseek41::FAMILIES),
    (deepseek41v::ARCH, deepseek41v::FAMILIES),
    (qwen35moe::ARCH, qwen35moe::FAMILIES),
    (qwen35::ARCH, qwen35::FAMILIES),
    (glm5next::ARCH, glm5next::FAMILIES),
    (mimo2::ARCH, mimo2::FAMILIES),
    (qwen4exp::ARCH, qwen4exp::FAMILIES),
    (tokenizer::ARCH, tokenizer::FAMILIES),
    (dequant::ARCH, dequant::FAMILIES),
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

#[cfg(test)]
mod tests {
    use super::{all, named, node_dumps};
    use crate::family::Identity;

    /// Every family's sets in place pass their family's check: each states
    /// the file the tree runs, names its family's ik build where the family
    /// pins one, and is complete. One line per set, then the verdict.
    #[test]
    #[ignore = "hw: needs the reference sets in $BLOOMERY_DATA and BLOOMERY_DSPARK_MODEL on the box"]
    fn hw_every_set_in_place_is_its_familys() {
        let mut failed = 0usize;
        let mut sets = 0usize;
        for f in all() {
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
        println!("hw_refset: {sets} set(s) in place, {failed} failed");
        assert_eq!(
            failed, 0,
            "hw_refset: {failed} set(s) in place fail their family's check"
        );
    }

    /// The fixture families are the table of six, each the mirror of one real
    /// family: its name is `fx-` and the real family's, and it takes the real
    /// family's architecture, ik build and consumers, the fixture twin of its
    /// identity, a draft file when the real family has one and no per-file set
    /// name; each set is the real family's set under the `fx_` prefix
    /// (`ref-mtp/fx_` for an MTP set). The real family's node dumps stay the
    /// architecture's.
    #[test]
    fn each_fixture_family_mirrors_a_real_one() {
        let sets_of = [
            ("fx-ik-glm5next", 4),
            ("fx-ik-glm5next-dsa", 2),
            ("fx-mtp-glm5next", 1),
            ("fx-ik-qwen4exp", 5),
            ("fx-mtp-qwen4exp", 1),
            ("fx-ik-deepseek41", 4),
        ];
        let mut found: Vec<&str> = all()
            .filter(|f| f.name.starts_with("fx-"))
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
        for arch in ["glm5next", "qwen4exp", "deepseek41"] {
            let node = node_dumps(arch).map(|f| f.name);
            assert_eq!(node, Some(&*format!("ik-{arch}")), "{arch}: node dumps");
        }
    }
}
