use super::{ARCH, LCPP, LCPP_BUILD, MODEL, SETS};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};

/// The family's row: the UD-IQ1_S file, the pinned build, this architecture; the sets it
/// must refuse are another quant of the same checkpoint (the same vocabulary and
/// architecture, so only the `# model` line tells it apart), a shard that is not the first,
/// the same shard name in another directory, and builds of another commit or library.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/home/user/models/Mistral-Large-4-1T-A52B-UD-IQ1_M/Mistral-Large-4-1T-A52B-UD-IQ1_M-00001-of-00007.gguf",
        "/home/user/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00002-of-00006.gguf",
        "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf",
    ],
    other_builds: &[
        "78feec062",
        "ba3c05f71+libllama-22407f1ce644",
        "78feec062+libllama-d77207daa41e",
    ],
    other_arches: &["mistral_large4", "muse-spark"],
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

/// The two prompt sets the dump recipe writes, in its order, each under the name the family's
/// recipe prints.
#[test]
fn the_family_holds_the_dump_recipes_two_sets() {
    assert_eq!(SETS, ["ref_mistrallarge4_a", "ref_mistrallarge4_b"]);
    assert_eq!(LCPP.sets, SETS);
}
