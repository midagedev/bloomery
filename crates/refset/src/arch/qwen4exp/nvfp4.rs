//! The Qwen3.8-Flash-Next NVFP4 oracle: the forward of `nvidia/Qwen3.8-Flash-Next-NVFP4` in
//! transformers' qwen4_exp, f32 on the CPU, with the weights a split `prepare` writes from that
//! folder holds — the routed experts by ModelOpt's NVFP4 rule, the PLE rows by their F8 rule, the
//! BF16 projections `prepare` rewrites to Q8_0 under `BLOOMERY_PREPARE_Q8=1` rounded by ggml's
//! reference Q8_0 rule (`tools/ref/qwen4exp-nvfp4-oracle.py`). Its sets are in ik's dump layout
//! and carry ik's tap names, so a gate reads them as it reads [`super::IK`]'s; the ids are the
//! batch set's and [`super::D3K`]'s. The two `_bf16` sets widen the checkpoint's BF16 projections
//! instead, as the split `prepare` writes by default holds them; between the two pairs lies the
//! Q8_0 rounding's own share. [`FX`] holds the same four sets of the oracle on the profile's cut
//! (`tools/fixture-cuts/qwen38nvfp4.json`), the fixture tier's.

use super::ARCH;
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The checkpoint folder every set states in its `# model` line, as a node's model store mounts
/// it: the repository at the revision the family pins.
pub const MODEL: &str =
    "/models/nvidia__Qwen3.8-Flash-Next-NVFP4/fc694b54fb0174e0913e6adf86691ef85a4ead47";

/// The md5 of `tools/ref/qwen4exp-nvfp4-oracle.py`, the sets' writer, which states it in each
/// set's `# build` line.
// PIN(2026-11-02): the writer at transformers 4c414256 that dequantizes one decoder layer at a
// time, writes the route ids' logical copy and exact twins as ik's dumper does, and reads a cut
// of the folder as the fixture tier's model only under the `# fixture` line of that cut's first
// shard (held to it by tools/bloomery/manifest.py's shared check), resuming only on the cut
// generation it paused on;
// `the_oracle_writer_pin_is_the_source_in_the_tree` holds it to the tree.
pub const HARNESS: &str = "a6612dbf9b703308d86474a5963e426d";

/// The 5-token batch, every position.
pub const BATCH: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4";
/// The 3,001 ids of [`super::D3K`] as a decode-step set: the step at position 3,000, every
/// position's taps kept.
pub const D3K: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k";
/// [`BATCH`] with the BF16 projections widened, as stored.
pub const BATCH_BF16: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_bf16";
/// [`D3K`] with the BF16 projections widened: the layer outputs and logits of the last position,
/// the routes of every position.
pub const D3K_BF16: &str = "hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k_bf16";

/// The model profile whose cut [`FX`]'s sets are dumped from (`tools/fixture-dirs.tsv`).
pub const PROFILE: &str = "qwen4exp-nvfp4";
/// [`BATCH`] on the cut.
pub const FX_BATCH: &str = "fx_hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4";
/// [`D3K`] on the cut.
pub const FX_D3K: &str = "fx_hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k";
/// [`BATCH_BF16`] on the cut.
pub const FX_BATCH_BF16: &str = "fx_hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_bf16";
/// [`D3K_BF16`] on the cut.
pub const FX_D3K_BF16: &str = "fx_hf-qwen4exp-nvfp4/hf_qwen4exp_nvfp4_d3k_bf16";

/// [`MODEL`], as the family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The cut's first shard, as [`FX`]'s `runs`.
fn cut() -> Result<String, RefError> {
    crate::fixture::first_shard(PROFILE)
}

/// The NVFP4 checkpoint's forward sets.
pub static HF: Family = Family {
    name: "hf-qwen4exp-nvfp4",
    sets: &[BATCH, D3K, BATCH_BF16, D3K_BF16],
    resolve: None,
    recipe: "just dump-ref-qwen4exp-nvfp4 IDS",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(HARNESS)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-nvfp4"],
};

/// [`HF`] on the profile's cut: the same writer, its `# model` the cut's first shard and its
/// `# fixture` line the cut's. Its consumer is the fixture-only twin of [`HF`]'s, which runs on
/// any card where [`HF`]'s needs one that holds the whole file's plan.
pub static FX: Family = Family {
    name: "fx-hf-qwen4exp-nvfp4",
    sets: &[FX_BATCH, FX_D3K, FX_BATCH_BF16, FX_D3K_BF16],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-qwen4exp-nvfp4 IDS",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(HARNESS)),
    runs: Some(cut),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-nvfp4-cut"],
};

#[cfg(test)]
mod tests {
    use super::super::{self as qwen, fixture};
    use super::{BATCH, HARNESS, HF, MODEL};
    use crate::RefError;
    use std::path::Path;

    /// The writer's pin is the source in the tree: an edit of the writer without new sets and a
    /// new pin is red.
    #[test]
    fn the_oracle_writer_pin_is_the_source_in_the_tree() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/qwen4exp-nvfp4-oracle.py");
        let source = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            crate::md5::hex_of(&source),
            HARNESS,
            "tools/ref/qwen4exp-nvfp4-oracle.py changed: write the sets again (just \
             dump-ref-qwen4exp-nvfp4) and pin the new md5"
        );
    }

    /// A complete set of the folder and the writer passes; one of the GGUF ik's sets are dumped
    /// from, or of another writer, is refused, naming what it states.
    #[test]
    fn the_family_takes_a_set_of_its_folder_and_writer() -> Result<(), RefError> {
        let dir =
            std::env::temp_dir().join(format!("bloomery-refset-nvfp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
        let check = |model: &str, build: &str| {
            let lines = [
                format!("# model\t{model}"),
                format!("# build\t{build}"),
                "# arch\tqwen4exp".to_string(),
                "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
                "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\thf".to_string(),
                "# complete\t1\t0".to_string(),
            ];
            let path = dir.join("MANIFEST.tsv");
            std::fs::write(&path, format!("{}\n", lines.join("\n")))
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            HF.check_set(&dir)
        };
        let p = check(MODEL, HARNESS)?;
        let mut man = crate::ik::RefManifest::open(&dir, &HF)?;
        assert_eq!(
            (p.dumped_from.as_str(), p.build.as_deref()),
            (MODEL, Some(HARNESS))
        );
        let gguf = check(super::super::MODEL, HARNESS).expect_err("a set of the GGUF");
        assert!(gguf.to_string().contains(super::super::MODEL), "{gguf}");
        let other = check(MODEL, "0123").expect_err("a set of another writer");
        assert!(other.to_string().contains("0123"), "{other}");
        assert!(HF.sets.contains(&BATCH));
        for (name, pos) in [
            (BATCH, 0),
            (super::BATCH_BF16, 0),
            (super::D3K, 3000),
            (super::D3K_BF16, 3000),
            (super::FX_BATCH, 0),
            (super::FX_BATCH_BF16, 0),
            (super::FX_D3K, 3000),
            (super::FX_D3K_BF16, 3000),
            (qwen::BATCH, 0),
            (fixture::BATCH, 0),
            (qwen::STEP4, 4),
            (fixture::STEP4, 4),
            (qwen::STEP4_EVERY_NODE, 4),
            (fixture::STEP4_EVERY_NODE, 4),
            (qwen::D1K, 1024),
            (fixture::D1K, 1024),
            (qwen::D3K, 3000),
            (fixture::D3K, 3000),
        ] {
            let deep = pos != 0;
            let count = if deep { pos as usize + 1 } else { 5 };
            man.header.tokens = Some(vec![1; count]);
            man.header.decode_pos = deep.then_some(pos);
            man.header.prefill = deep.then_some(pos);
            qwen::check_inputs(&man, name)?;
            for bad in [0, 2, 9, count - 1, count + 1] {
                man.header.tokens = Some(vec![1; bad]);
                man.header.decode_pos = deep.then_some(bad.saturating_sub(1) as u32);
                man.header.prefill = man.header.decode_pos;
                let err =
                    qwen::check_inputs(&man, name).expect_err("wrong oracle position or length");
                assert!(err.to_string().contains(name), "{err}");
            }
        }
        man.header.tokens = Some(vec![1; 5]);
        man.header.decode_pos = None;
        man.header.prefill = None;
        let err = qwen::check_inputs(&man, "other").expect_err("unknown oracle set");
        assert!(err.to_string().contains("other"), "{err}");
        std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))
    }
}
