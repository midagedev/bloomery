use super::{ARCH, MODEL, PAI, PAI_BUILD, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// Solar's row: the Q4_K_M file, the pinned build, this architecture; the sets it must refuse are the IQ4_XS split set (the same
/// vocabulary and architecture, so only the `# model` line tells it from the Q4_K_M), a shard that is not the first, the same shard
/// name in another directory, the unpatched commit (it has no such architecture) and the patch with nothing in it.
const ROW: Row = Row {
    family: &PAI,
    model: MODEL,
    build: PAI_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Solar-Open2-250B-IQ4_XS/Solar-Open2-250B-IQ4_XS-00001-of-00004.gguf",
        "/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00002-of-00004.gguf",
        "/models/elsewhere/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf",
    ],
    other_builds: &["6ea215d17", "6ea215d17+pai-000000000000", "f711c8f3"],
    other_arches: &["glm5next", "solar_open2"],
};

#[test]
fn the_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    contract::takes_a_set_of_its_file_build_and_arch(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_file(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_build(&ROW)
}

#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_arch(&ROW)
}

#[test]
fn the_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    contract::refuses_a_set_without_its_trailer(&ROW)
}

/// The fixture family mirrors the real one as every fixture family does (`arch::tests`): the same architecture, build and consumers,
/// the fixture twin of its identity, no per-file set name, and each set the real family's behind `fx_`.
#[test]
fn the_fixture_family_mirrors_the_real_one() {
    let fx = &fixture::PAI;
    assert_eq!(
        (fx.arch, fx.build, fx.consumers),
        (PAI.arch, PAI.build, PAI.consumers)
    );
    assert_eq!(
        (PAI.identity, fx.identity),
        (Identity::Manifest, Identity::FixtureManifest)
    );
    assert!(fx.resolve.is_none(), "one fixture file, no per-file suffix");
    assert!(
        fx.recipe
            .starts_with("BLOOMERY_TIER=fixture just dump-ref-"),
        "{}",
        fx.recipe
    );
    let twins: Vec<String> = fx
        .sets
        .iter()
        .map(|&s| s.strip_prefix("fx_").unwrap_or(s).to_string())
        .collect();
    assert_eq!(twins, SETS, "each fixture set is the real set behind fx_");
}
