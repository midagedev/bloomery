use super::{ARCH, LCPP, LCPP_BUILD, MODEL, SETS, fixture};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;
use std::path::{Path, PathBuf};

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
        "/models/Mistral-Large-4-1T-A52B-UD-IQ1_M/Mistral-Large-4-1T-A52B-UD-IQ1_M-00001-of-00007.gguf",
        "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00002-of-00006.gguf",
        "/home/user/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf",
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
/// recipe prints, and their fixture twins; the shared fixture-family test (`arch::tests`) holds
/// the rest of the mirror.
#[test]
fn the_family_holds_the_dump_recipes_two_sets() {
    assert_eq!(SETS, ["ref_mistrallarge4_a", "ref_mistrallarge4_b"]);
    assert_eq!(LCPP.sets, SETS);
    assert_eq!(LCPP.identity, Identity::Manifest);
    assert_eq!(
        fixture::LCPP.sets,
        ["fx_ref_mistrallarge4_a", "fx_ref_mistrallarge4_b"]
    );
}

/// A complete set of the dumper's shape in a fresh directory: `model`, the family's build and
/// architecture, a `# fixture` line when given, one tensor row and the trailer.
fn set_of(what: &str, model: &str, fixture_line: Option<&str>) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-lcpp-mistrallarge4-{what}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let mut lines = vec![
        "# cb_dump — a llama.cpp tree's intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        format!("# build\t{LCPP_BUILD}"),
        format!("# arch\t{ARCH}"),
    ];
    lines.extend(fixture_line.map(|l| format!("# fixture\t{l}")));
    lines.extend([
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
        "# complete\t1\t0".to_string(),
    ]);
    std::fs::write(dir.join("MANIFEST.tsv"), format!("{}\n", lines.join("\n")))
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

/// `refused` is the stale refusal of a set that states `dumped_from` where the tree runs `runs`,
/// by name; the set's directory is removed.
fn stale_by_name(
    dir: &Path,
    refused: Result<crate::family::Provenance, RefError>,
    dumped_from: &str,
    runs: &str,
) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))?;
    match refused {
        Err(e @ RefError::Stale { .. }) => {
            let e = e.to_string();
            assert!(
                e.contains(&format!("dumped from {dumped_from}"))
                    && e.contains(&format!("the tree runs {runs}")),
                "{e}"
            );
            Ok(())
        }
        other => panic!("a set of {dumped_from}: {other:?}"),
    }
}

/// A set dumped from the fixture file is stale for the real family, by name: its `# model` line
/// is the fixture's first shard, and the refusal states that file beside the real one the tree
/// runs. The other way is `hw_a_real_set_is_stale_for_the_fixture_family`, which needs the
/// fixture file on the box.
#[test]
fn the_real_family_refuses_a_set_of_the_fixture() -> Result<(), RefError> {
    const FIXTURE: &str =
        "/models/fixtures/mistrallarge4/mistrallarge4-fixture-00001-of-00001.gguf";
    let dir = set_of("tier-fx", FIXTURE, Some("bloomery.fixture.version=1"))?;
    stale_by_name(&dir, LCPP.check_set(&dir), FIXTURE, MODEL)
}

/// A set dumped from the real file is stale for the fixture family, by name: the refusal states
/// the real file beside the fixture's first shard the tree runs.
#[test]
#[ignore = "hw: needs the fixture file under the fixture root (`fixture generate`, the `fixture-dirs.tsv` row)"]
fn hw_a_real_set_is_stale_for_the_fixture_family() -> Result<(), RefError> {
    let runs = crate::fixture::first_shard(ARCH)?;
    let dir = set_of("tier-real", MODEL, None)?;
    stale_by_name(&dir, fixture::LCPP.check_set(&dir), MODEL, &runs)
}
