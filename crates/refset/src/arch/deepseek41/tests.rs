use super::{ARCH, FAMILIES, IK_BUILD, fixture};
use crate::arch::{families, node_dumps};
use crate::family::{Build, Identity};

/// The tools profile the fixture sets' names come from.
const PROFILE: &str = include_str!("../../../../../tools/ref/models/deepseek41.sh");

/// The architecture's node dumps are the fixture family's alone: ik cannot read the real file, so
/// the architecture has no node-dump family of the real file's, and the gates that compare with
/// ik open the fixture sets. The fixture family is the manifest identity at the architecture's ik
/// build with its own consumers, and each of its sets is a set the profile dumps behind `fx_`.
#[test]
fn the_fixture_family_is_the_only_node_dump_family() {
    let fx = &fixture::IK;
    assert!(
        node_dumps(ARCH).is_none(),
        "a real node-dump family: ik cannot read the real file"
    );
    assert!(FAMILIES.iter().any(|f| std::ptr::eq(*f, fx)));
    assert_eq!(
        families(ARCH).iter().map(|f| f.name).collect::<Vec<_>>(),
        ["fx-ik-deepseek41"]
    );
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
    assert_eq!(fx.sets.len(), 4);
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
