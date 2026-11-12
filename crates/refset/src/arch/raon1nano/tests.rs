use super::{ARCH, LCPP, LCPP_BUILD, MODEL};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};

/// The nano family's row: the Q8_0 file, the vendor branch's build, this architecture; the sets it must refuse are
/// the MTP draft file beside the target (the same vocabulary and architecture, so only the `# model` line tells it),
/// a Q8_0 file of the other dialect, the file's name in another directory, the bare commit (no library named),
/// another library, and mainline llama.cpp (it has no Raon graph). Mutants, one a test (the module doc of
/// `node_dump_contract`): `runs` answering the MTP file, `runs`/`build`/`arch` unset (the pin each refusal leans
/// on), and the trailer the last test's manifest keeps.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/data/models/Raon-1-nano-gguf/raon-1-nano-Q8_0_MTP.gguf",
        "/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf",
        "/data/elsewhere/raon-1-nano-Q8_0.gguf",
    ],
    other_builds: &[
        "4d72ff0d4",
        "4d72ff0d4+libllama-000000000000",
        "fb895662d+libllama-fcd56f0bf834",
    ],
    other_arches: &["raon1", "minimax-m3"],
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
