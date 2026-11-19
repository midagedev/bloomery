use super::{ARCH, LCPP, LCPP_BUILD, MODEL, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// The family's row: the Q4_K_M file, the pinned build, this architecture; the sets it must refuse are another quantization of the
/// same checkpoint (the same vocabulary and architecture, so only the `# model` line tells it apart), a shard that is not the first,
/// the first shard's name in another directory, and the fixture's file. The builds it must refuse are the bare commit (no library
/// named), another library of the same commit and another commit with this library.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Hy4-flash-GGUF/Hy4-flash-Q8_0-00001-of-00003.gguf",
        "/models/Hy4-flash-GGUF/Hy4-flash-Q4_K_M-00002-of-00002.gguf",
        "/models/elsewhere/Hy4-flash-Q4_K_M-00001-of-00002.gguf",
        "/models/fixtures/hy4/hy4-fixture-00001-of-00001.gguf",
    ],
    other_builds: &[
        "0a054d0b0",
        "0a054d0b0+libllama-000000000000",
        "78feec062+libllama-5d0c8c15ad91",
    ],
    other_arches: &["kimi-k3", "deepseek41"],
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

/// The sets the dump recipe writes, in its order, under the names its script prints, and their fixture twins; the shared
/// fixture-family test (`arch::tests`) holds the rest of the mirror. The node dumps the gates read are the real family's, and the
/// architecture's families are the ones the registry lists, in `refset-check`'s order.
#[test]
fn the_family_holds_the_dump_recipes_sets() {
    assert_eq!(
        SETS,
        [
            "ref_hy4_a",
            "ref_hy4_b",
            "ref_hy4_c",
            "ref_hy4_d",
            "ref_hy4_e"
        ]
    );
    assert_eq!(LCPP.sets, SETS);
    assert_eq!(LCPP.identity, Identity::Manifest);
    assert_eq!(fixture::LCPP.identity, Identity::FixtureManifest);
    let twins: Vec<String> = fixture::LCPP
        .sets
        .iter()
        .map(|&s| s.strip_prefix("fx_").unwrap_or(s).to_string())
        .collect();
    assert_eq!(twins, SETS, "each fixture set is the real set behind fx_");
    let names: Vec<&str> = crate::arch::families(ARCH).iter().map(|f| f.name).collect();
    assert_eq!(
        names,
        ["lcpp-hy4", "fx-lcpp-hy4", "hy4-vis-preproc", "hy4-vis-taps"]
    );
    assert_eq!(
        crate::arch::node_dumps(ARCH).map(|f| f.name),
        Some("lcpp-hy4")
    );
}
