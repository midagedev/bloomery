use super::{ARCH, LCPP, LCPP_BUILD, MODEL, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// The family's row: the UD-IQ1_S file, the pinned build, this architecture; the sets it must refuse are another quantization of the
/// same checkpoint (the same vocabulary and architecture, so only the `# model` line tells it apart), a shard that is not the first,
/// the first shard's name in another directory, and the fixture's file. The builds it must refuse are the bare commit (no library
/// named), another library of the same commit (Mistral Large 4's tree, the sharpest: one commit, two builds) and another commit.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Kimi-K3-UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf",
        "/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00014.gguf",
        "/models/elsewhere/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf",
        "/models/fixtures/kimik3/kimi-k3-fixture-00001-of-00001.gguf",
    ],
    other_builds: &[
        "78feec062",
        "78feec062+libllama-22407f1ce644",
        "78feec062+libllama-000000000000",
        "ba3c05f71+libllama-ae99cf8db62c",
    ],
    other_arches: &["kimi-linear", "mistral-large4"],
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
            "ref_kimik3_a",
            "ref_kimik3_b",
            "ref_kimik3_c",
            "ref_kimik3_d"
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
    assert_eq!(names, ["lcpp-kimi-k3", "fx-lcpp-kimi-k3"]);
    assert_eq!(
        crate::arch::node_dumps(ARCH).map(|f| f.name),
        Some("lcpp-kimi-k3")
    );
}
