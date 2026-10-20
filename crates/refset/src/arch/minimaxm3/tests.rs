use super::{ARCH, LCPP, LCPP_BUILD, MODEL};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};

/// MiniMax's row: the UD-IQ4_XS file, the mainline build, this architecture; the sets it must refuse are another quantization of
/// the same model (the same vocabulary and architecture, so only the `# model` line tells it), a shard that is not the first, the
/// first shard's name in another directory, the bare commit (no library named), another library, and the Solar PAI tree (it carries an
/// older copy of this graph).
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/MiniMax-M3-UD-Q3_K_XL/MiniMax-M3-UD-Q3_K_XL-00001-of-00005.gguf",
        "/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00002-of-00006.gguf",
        "/models/elsewhere/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf",
    ],
    other_builds: &[
        "cd0a5b6a1",
        "cd0a5b6a1+libllama-000000000000",
        "6ea215d17+pai-d2906be2c009",
    ],
    other_arches: &["solar-open2", "minimax_m3"],
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
