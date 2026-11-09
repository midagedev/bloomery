use super::{
    ARCH, D3K, D3K_MIX256G, D3K_POSITION, D3K_SCORES, LCPP, LCPP_BUILD, LCPP_MIX256G, MODEL,
    MODEL_MIX256G, SETS, SETS_MIX256G, check_deep, fixture,
};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;
use crate::ik::RefManifest;
use std::path::{Path, PathBuf};

/// The family's row: the UD-IQ1_S file, the pinned build, this architecture; the sets it
/// must refuse are another quant of the same checkpoint (the same vocabulary and
/// architecture, so only the `# model` line tells it apart: the Mix-256G file among them), a
/// shard that is not the first, the same shard name in another directory, and builds of another
/// commit or library.
const ROW: Row = Row {
    family: &LCPP,
    model: MODEL,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        "/models/Mistral-Large-4-1T-A52B-UD-IQ1_M/Mistral-Large-4-1T-A52B-UD-IQ1_M-00001-of-00007.gguf",
        MODEL_MIX256G,
        "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00002-of-00006.gguf",
        "/models/elsewhere/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf",
    ],
    other_builds: &[
        "78feec062",
        "ba3c05f71+libllama-22407f1ce644",
        "78feec062+libllama-d77207daa41e",
    ],
    other_arches: &["mistral_large4", "muse-spark"],
};

/// The Mix-256G family's row, [`ROW`]'s mirror on the other real file: the build and architecture
/// are the same, the file is its own, and the sets it must refuse are the UD-IQ1_S file's (the
/// same checkpoint, only the `# model` line tells it apart), a shard that is not the first, the
/// first shard's name in another directory, and the other builds.
const ROW_MIX256G: Row = Row {
    family: &LCPP_MIX256G,
    model: MODEL_MIX256G,
    build: LCPP_BUILD,
    arch: ARCH,
    other_files: &[
        MODEL,
        "/data/moved/models/Mistral-Large-4-1T-A52B-Mix-256G/Mistral-Large-4-1T-A52B-Mix-256G-00002-of-00006.gguf",
        "/models/elsewhere/Mistral-Large-4-1T-A52B-Mix-256G-00001-of-00006.gguf",
    ],
    other_builds: ROW.other_builds,
    other_arches: ROW.other_arches,
};

#[test]
fn the_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    contract::takes_a_set_of_its_file_build_and_arch(&ROW)
}

#[test]
fn the_mix256g_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    contract::takes_a_set_of_its_file_build_and_arch(&ROW_MIX256G)
}

#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_file(&ROW)
}

#[test]
fn the_mix256g_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_file(&ROW_MIX256G)
}

#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_build(&ROW)
}

#[test]
fn the_mix256g_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_build(&ROW_MIX256G)
}

#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_arch(&ROW)
}

#[test]
fn the_mix256g_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    contract::refuses_a_set_of_another_arch(&ROW_MIX256G)
}

#[test]
fn the_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    contract::refuses_a_set_without_its_trailer(&ROW)
}

#[test]
fn the_mix256g_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    contract::refuses_a_set_without_its_trailer(&ROW_MIX256G)
}

/// The sets the dump recipe writes, in its order (the two prompts, then the `d3k` variant's deep
/// set and the `d3k-scores` variant's), each under the name the family's recipe prints, and the two prompts' fixture twins (the
/// real name with the fixture's cut, `e128`, in the family word; the fixture has no deep set); the
/// shared fixture-family test (`arch::tests`) holds the rest of the mirror.
#[test]
fn the_family_holds_the_dump_recipes_sets() {
    assert_eq!(
        SETS,
        [
            "ref_mistrallarge4_a",
            "ref_mistrallarge4_b",
            "ref_mistrallarge4_d3k",
            "ref_mistrallarge4_d3k_scores"
        ]
    );
    assert_eq!(LCPP.sets, SETS);
    assert_eq!(LCPP.identity, Identity::Manifest);
    assert_eq!(
        fixture::LCPP.sets,
        ["fx_ref_mistrallarge4e128_a", "fx_ref_mistrallarge4e128_b"]
    );
}

/// The Mix-256G file's prompt and deep sets are the UD-IQ1_S file's names behind `mix256g_`, the node
/// dumps the gates read stay the UD-IQ1_S family's, and the architecture's families are the ones
/// the registry lists, in `refset-check`'s order. A set name two families share would be one
/// directory with two identities.
#[test]
fn the_family_holds_the_mix256g_sets_apart() {
    assert_eq!(
        SETS_MIX256G,
        [
            "ref_mistrallarge4_mix256g_a",
            "ref_mistrallarge4_mix256g_b",
            "ref_mistrallarge4_mix256g_d3k"
        ]
    );
    assert_eq!(LCPP_MIX256G.sets, SETS_MIX256G);
    assert_eq!(LCPP_MIX256G.identity, Identity::Manifest);
    for set in SETS_MIX256G {
        assert!(!SETS.contains(set), "{set} is a UD-IQ1_S set");
    }
    let names: Vec<&str> = crate::arch::families(ARCH).iter().map(|f| f.name).collect();
    assert_eq!(
        names,
        [
            "lcpp-mistrallarge4",
            "lcpp-mistrallarge4-mix256g",
            "fx-lcpp-mistrallarge4",
            "lcpp-dspark-mistrallarge4",
            "mistrallarge4-vis-preproc",
            "mistrallarge4-vis-taps"
        ]
    );
    assert_eq!(
        crate::arch::node_dumps(ARCH).map(|f| f.name),
        Some("lcpp-mistrallarge4")
    );
}

/// The body's end-to-end gate reads the node dumps of whichever real file it runs on, so it is
/// the consumer of both real families.
#[test]
fn the_end_to_end_gate_is_the_consumer_of_both_real_families() {
    assert_eq!(LCPP.consumers, ["gate-gpu-mistrallarge4-e2e"]);
    assert_eq!(LCPP_MIX256G.consumers, LCPP.consumers);
}

/// Each real file's deep set is its own family's, under a name no other family holds: the deep
/// steps of the two files are two directories with two identities.
#[test]
fn each_real_file_holds_its_own_deep_set() {
    assert_eq!(D3K, "ref_mistrallarge4_d3k");
    assert_eq!(D3K_SCORES, "ref_mistrallarge4_d3k_scores");
    assert!(LCPP.sets.contains(&D3K_SCORES) && !LCPP_MIX256G.sets.contains(&D3K_SCORES));
    assert_eq!(D3K_MIX256G, "ref_mistrallarge4_mix256g_d3k");
    assert!(LCPP.sets.contains(&D3K) && !LCPP.sets.contains(&D3K_MIX256G));
    assert!(LCPP_MIX256G.sets.contains(&D3K_MIX256G) && !LCPP_MIX256G.sets.contains(&D3K));
    assert!(!fixture::LCPP.sets.iter().any(|s| s.ends_with("d3k")));
}

/// The deep set is a decode step at [`D3K_POSITION`]: a batch set and a step at another position
/// (the top-k's cut sits below it: a step at or under the cut selects every position) are each
/// refused, naming the set.
#[test]
fn the_deep_set_is_a_decode_step_at_its_position() -> Result<(), RefError> {
    let dir = set_of("deep-step", MODEL, None)?;
    let mut man = RefManifest::open(&dir, &LCPP)?;
    let at = D3K_POSITION as usize;
    man.header.tokens = Some(vec![1; at + 1]);
    man.header.decode_pos = Some(D3K_POSITION);
    man.header.prefill = Some(D3K_POSITION);
    check_deep(&man, D3K)?;
    for (count, pos) in [
        (18, None),
        (5, Some(4)),
        (2048, Some(2047)),
        (at, Some(D3K_POSITION - 1)),
        (at + 2, Some(D3K_POSITION + 1)),
    ] {
        man.header.tokens = Some(vec![1; count]);
        man.header.decode_pos = pos;
        man.header.prefill = pos;
        let err = check_deep(&man, D3K).expect_err("not the deep set's step");
        assert!(err.to_string().contains(D3K), "{err}");
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))
}

/// A deep step dumped from the UD-IQ1_S file is refused by the Mix-256G family and the other way
/// round, by name: each states its file, and the refusal states the file the tree runs.
#[test]
fn a_deep_set_of_the_other_file_is_refused_by_name() -> Result<(), RefError> {
    for (what, name, model, own, other, other_model) in [
        ("deep-ud", D3K, MODEL, &LCPP, &LCPP_MIX256G, MODEL_MIX256G),
        (
            "deep-mix",
            D3K_MIX256G,
            MODEL_MIX256G,
            &LCPP_MIX256G,
            &LCPP,
            MODEL,
        ),
    ] {
        let dir = set_of(what, model, None)?;
        let mut man = RefManifest::open(&dir, own)?;
        let at = D3K_POSITION as usize;
        man.header.tokens = Some(vec![1; at + 1]);
        man.header.decode_pos = Some(D3K_POSITION);
        man.header.prefill = Some(D3K_POSITION);
        check_deep(&man, name)?;
        stale_by_name(&dir, other.check_set(&dir), model, other_model)?;
    }
    Ok(())
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
        "/models/fixtures/mistrallarge4-e128/mistrallarge4-fixture-00001-of-00001.gguf";
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
