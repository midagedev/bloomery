use super::{
    A, ARCH, B, BUILD, C, D, E, E_PERHEAD, MODEL, PERHEAD_SETS, POOLED, SETS, TF, fixture,
};
use crate::RefError;
use crate::arch::node_dump_contract::{self as contract, Row};
use crate::family::Identity;
use std::path::Path;

/// The checkpoint row: the first shard of the vendor's folder, the writer's build, MiniMax-M3's architecture. The sets it must
/// refuse are another folder's first shard (a fixture cut, whose manifest differs only in this line), a shard that is not the
/// first, the first shard's name in another directory, and the M3 GGUF (the same architecture and tokenizer, another model); the
/// builds are another writer, another transformers or torch release, the local-tagged torch spelling, and llama.cpp's (none of
/// these carries the pinned writer's digits).
const ROW: Row = Row {
    family: &TF,
    model: MODEL,
    build: BUILD,
    arch: ARCH,
    other_files: &[
        "/models/fixtures/minimaxm31/model-00001-of-00002.safetensors",
        "/data/moved/models/MiniMax-M3.1-Flash/model-00002-of-00048.safetensors",
        "/models/elsewhere/model-00001-of-00048.safetensors",
        "/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf",
    ],
    other_builds: &[
        "transformers-5.12.1+torch-2.14.1+dump_tf-000000000000",
        "transformers-5.12.0+torch-2.14.1+dump_tf-000000000000",
        "transformers-5.12.1+torch-2.14.0+dump_tf-000000000000",
        "transformers-5.12.1+torch-2.14.1+cpu+dump_tf-000000000000",
        "cd0a5b6a1+libllama-d846809bb9cd",
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

/// The pin is the writer in the tree: an edit of `dump_tf.py` without new sets and a new pin is red.
#[test]
fn the_writer_pin_is_the_source_in_the_tree() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/mm31/dump_tf.py");
    let source = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let md5 = crate::md5::hex_of(&source);
    assert_eq!(
        BUILD,
        format!("transformers-5.12.1+torch-2.14.1+dump_tf-{}", &md5[..12]),
        "tools/ref/mm31/dump_tf.py changed: dump the sets again (just dump-ref-minimaxm31) and pin the new md5"
    );
}

/// The dump's set names are this family's and unique; the registry's `minimax_rows_are_both_modules_families` holds them apart
/// from the MiniMax-M3 sets, so one data directory holds both families.
#[test]
fn the_sets_are_this_familys_and_no_other_familys() {
    assert_eq!(SETS, [A, B, C, D, E, E_PERHEAD]);
    for (i, set) in SETS.iter().enumerate() {
        assert!(set.starts_with("ref_minimaxm31_"), "{set}");
        assert!(!SETS[..i].contains(set), "{set} twice");
    }
}

/// The two families split the dump by the selection rule, so a gate that opens every set of [`TF`] never meets the vendor's set e,
/// and between them they hold every set of the dump once: the writer, the file and the build are the same.
#[test]
fn the_families_split_the_dump_by_the_selection_rule() {
    assert_eq!(TF.sets, PERHEAD_SETS);
    assert_eq!(POOLED.sets, [E]);
    let mut held: Vec<&str> = TF.sets.iter().chain(POOLED.sets).copied().collect();
    held.sort_unstable();
    let mut dumped = SETS.to_vec();
    dumped.sort_unstable();
    assert_eq!(held, dumped);
    assert!(
        !TF.sets.contains(&E),
        "the card's rule is not the pooled set's"
    );
    assert_eq!(
        (TF.arch, TF.build, TF.identity),
        (POOLED.arch, POOLED.build, POOLED.identity)
    );
    assert_eq!(
        TF.runs.map(|f| f().unwrap()),
        POOLED.runs.map(|f| f().unwrap())
    );
    assert_eq!(TF.recipe, POOLED.recipe);
    assert_eq!(TF.consumers, ["gate-gpu-minimaxm3-e2e"]);
}

/// The fixture family mirrors the real one as every fixture family does (`arch::tests`): the same architecture, build and
/// consumers, the fixture twin of its identity, no per-file set name, and each set the real family's behind `fx_`.
#[test]
fn the_fixture_family_mirrors_the_real_one() {
    let fx = &fixture::TF;
    assert_eq!(
        (fx.arch, fx.build, fx.consumers),
        (TF.arch, TF.build, TF.consumers)
    );
    assert_eq!(
        (TF.identity, fx.identity),
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
    assert_eq!(
        twins, PERHEAD_SETS,
        "each fixture set is the real set behind fx_"
    );
}
