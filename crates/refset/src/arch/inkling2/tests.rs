use super::{ARCH, BUILD, COMMIT, LCPP, MODEL, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// The family's row: the Mini's UD-Q4_K_XL file, the pinned patched build, this architecture. The sets it must refuse are
/// another file of the same architecture (the Small's first shard — the same vocabulary, so only the `# model` line tells
/// it apart), the Mini's file under another path, and the fixture's file. The builds it must refuse are the bare commit
/// (no patch named: the unpatched tree, which refuses the files at load), another patch of the same commit, the unpatched
/// tree's library build shape, and another commit with this patch.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf",
        "/models/elsewhere/Inkling-2-Mini-UD-Q4_K_XL.gguf",
        "/models/fixtures/inkling2/inkling2-fixture-00001-of-00001.gguf",
    ],
    other_builds: &[
        COMMIT,
        "6196d9802+inkling2-scale-000000000000",
        "6196d9802+libllama-0ada36d74451",
        "999999999+inkling2-scale-ab1511fbc024",
    ],
    other_arches: &["inkling2_loop", "dflash"],
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
/// fixture-family test (`arch::tests`) holds the rest of the mirror. The node dumps the gates read are the real family's,
/// and the architecture's families are the ones the registry lists, in `refset-check`'s order.
#[test]
fn the_family_holds_the_dump_recipes_sets() {
    assert_eq!(
        SETS,
        [
            "ref_inkling2_a",
            "ref_inkling2_b",
            "ref_inkling2_c",
            "ref_inkling2_d",
            "ref_inkling2_e"
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
    assert_eq!(names, ["lcpp-inkling2", "fx-lcpp-inkling2"]);
    assert_eq!(
        crate::arch::node_dumps(ARCH).map(|f| f.name),
        Some("lcpp-inkling2")
    );
}
