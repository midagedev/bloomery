use super::{ARCH, BUILD, LCPP, MODEL, SETS};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;

/// The family's row: the fixture GGUF, the patched build, this architecture; the sets it must refuse are the cut folder the
/// GGUF was quantized from (a llama.cpp tree cannot read it, but the check is the `# model` line's, and a set of it must not
/// open as one of this file's), the vocab-only file (the same vocabulary and architecture, so only the `# model` line tells it
/// apart), a GGUF of the same name in another directory, and the real checkpoint folder's first shard. The builds it must
/// refuse are the bare commit (no patch named), the same commit with no patch at all (the unpatched library form — the
/// sharpest: one commit, two builds, and the unpatched one reads no sliding-window array), the same tag with another patch,
/// and another commit.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: BUILD,
    arch: ARCH,
    other_files: &[
        "/models/fixtures/exaoneh/model-00001-of-00005.safetensors",
        "/data/models/K-EXAONE-2.0-vocab/exaone-moe-vocab.gguf",
        "/models/elsewhere/exaoneh-gguf-fixture-00001-of-00001.gguf",
        "/models/fixtures/exaoneh-gguf/exaoneh-gguf-fixture-00002-of-00001.gguf",
    ],
    other_builds: &[
        "7bcbc04fb",
        "7bcbc04fb+libllama-f5b8c2413660",
        "7bcbc04fb+exaone-moe-000000000000",
        "78feec062+exaone-moe-c7a1bf15a85f",
    ],
    other_arches: &["flint", "kimi-k3"],
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

/// The sets the dump recipe writes, in its order, under the names its script prints. The node
/// dumps the gates read are the fixture GGUF's, and the architecture's families are the ones the
/// registry lists, in `refset-check`'s order.
#[test]
fn the_family_holds_the_dump_recipes_sets() {
    assert_eq!(
        SETS,
        [
            "ref_exaone_a",
            "ref_exaone_b",
            "ref_exaone_c",
            "ref_exaone_d"
        ]
    );
    assert_eq!(LCPP.sets, SETS);
    assert_eq!(LCPP.identity, Identity::Manifest);
    let names: Vec<&str> = crate::arch::families(ARCH).iter().map(|f| f.name).collect();
    assert_eq!(names, ["lcpp-exaone-moe"]);
    assert_eq!(
        crate::arch::node_dumps(ARCH).map(|f| f.name),
        Some("lcpp-exaone-moe")
    );
}
