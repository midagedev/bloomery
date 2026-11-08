use super::{ARCH, LCPP, LCPP_BUILD, MODEL, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// The family's row: the UD-IQ2_XXS file, the pinned build, this architecture; the sets it
/// must refuse are another quant of the same checkpoint (the same vocabulary and
/// architecture, so only the `# model` line tells it apart), a shard that is not the first,
/// the same shard name in another directory, and builds of another commit or library.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Muse-Spark-Open-816B-A42B-UD-Q4_K_XS/Muse-Spark-Open-816B-A42B-UD-Q4_K_XS-00001-of-00006.gguf",
        "/data/moved/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00002-of-00006.gguf",
        "/models/elsewhere/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf",
    ],
    other_builds: &[
        "ba3c05f71",
        "cd0a5b6a1+libllama-d846809bb9cd",
        "ba3c05f71+libllama-d846809bb9cd",
    ],
    other_arches: &["muse_spark", "glm5next"],
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

/// The fixture family mirrors the real one as every fixture family does (`arch::tests`): the
/// same architecture, build and consumers, the fixture twin of its identity, no per-file set
/// name, and each set the real family's behind `fx_`.
#[test]
fn the_fixture_family_mirrors_the_real_one() {
    let fx = &fixture::LCPP;
    assert_eq!(
        (fx.arch, fx.build, fx.consumers),
        (LCPP.arch, LCPP.build, LCPP.consumers)
    );
    assert_eq!(
        (LCPP.identity, fx.identity),
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
