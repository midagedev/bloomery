//! The Qwen3-30B-A3B-Instruct-2507 family of the fixture tier. The real file's oracle sets
//! (`ref_qwen3moe`, the CUDA oracle's and the decode-step sets, `tools/ref/models/qwen3moe.sh`)
//! are read by the card gates by name and have no family here; only the fixture's are
//! ([`fixture`]), so the architecture has no node-dump family ([`super::node_dumps`] answers
//! none) and the card gates keep opening their real sets as they do.

pub mod fixture;

/// The architecture every qwen3moe manifest names in its `# arch` line.
pub const ARCH: &str = "qwen3moe";

/// The ik tree the fixture's sets are dumped by (`tools/ref/models/qwen3moe.sh` names its `IK`).
pub const IK_BUILD: &str = "db517b69";

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&crate::family::Family] = &[&fixture::IK];

#[cfg(test)]
mod tests {
    use super::{ARCH, FAMILIES, IK_BUILD, fixture};
    use crate::arch::{families, node_dumps};
    use crate::family::{Build, Identity};

    /// The tools profile the real sets' names come from.
    const PROFILE: &str = include_str!("../../../../../tools/ref/models/qwen3moe.sh");

    /// The architecture's only family is the fixture's, so the architecture has no node-dump
    /// family and the card gates' real sets open as they did. The family is the fixture twin of
    /// the manifest identity at the architecture's ik build, and each of its sets is a set the
    /// profile dumps behind `fx_`.
    #[test]
    fn the_fixture_family_stands_without_a_real_twin() {
        let fx = &fixture::IK;
        assert_eq!(FAMILIES.len(), 1);
        assert!(std::ptr::eq(FAMILIES[0], fx));
        assert_eq!(
            families(ARCH).iter().map(|f| f.name).collect::<Vec<_>>(),
            ["fx-ik-qwen3moe"]
        );
        assert!(node_dumps(ARCH).is_none(), "a real node-dump family");
        assert_eq!(fx.identity, Identity::FixtureManifest);
        assert_eq!(fx.arch, Some(ARCH));
        assert_eq!(fx.build, Some(Build::Is(IK_BUILD)));
        assert!(fx.resolve.is_none(), "one fixture file, no per-file suffix");
        assert!(fx.draft_runs.is_none());
        assert!(
            fx.recipe
                .starts_with("BLOOMERY_TIER=fixture just dump-ref-fixture "),
            "{}",
            fx.recipe
        );
        for set in fx.sets {
            let twin = set.strip_prefix("fx_").unwrap_or(set);
            let real = twin.strip_suffix("_every_node").unwrap_or(twin);
            assert!(set.starts_with("fx_"), "{set}: no fx_ prefix");
            assert!(
                PROFILE.contains(&format!("={real}")),
                "{set}: the profile dumps no {real}"
            );
        }
    }
}
