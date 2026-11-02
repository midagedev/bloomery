//! mistrallarge4 DSpark draft gate: what the draft reader resolves from the header of
//! `dspark-Mistral-Large-4-1T-A52B-Q8_0.gguf` (unsloth/Mistral-Large-4-1T-A52B-GGUF), whose header
//! tables are checked in beside this test (`mistrallarge4_dspark_hdr.kv.json`,
//! `mistrallarge4_dspark_tensors.tsv`: every key and tensor of the file's header, the four
//! tokenizer arrays the fetch summarized and the chat template, which nothing reads, left out).
//! The tables are laid out as a header-only `Split` (`common/header_tables.rs`) and read through
//! the engine's own reader. One contract, one test: it prints the summary, then fails with every
//! pin that differs.

use gguf::Split;
use model::arch::mistrallarge4::dspark::{self, Hparams, Tensor};
use model::arch::mistrallarge4::hparams::{Hparams as Target, YarnKeys};

#[path = "common/header_tables.rs"]
mod header_tables;

const KV: &str = include_str!("mistrallarge4_dspark_hdr.kv.json");
const TENSORS: &str = include_str!("mistrallarge4_dspark_tensors.tsv");
const TARGET_KV: &str = include_str!("mistrallarge4_hdr.kv.json");
const TARGET_TENSORS: &str = include_str!("mistrallarge4_tensors.tsv");

/// The bytes the file's tensors take: the last tensor's offset and size, the offsets contiguous
/// at 32 bytes (the Q8_0 file's own, 2,245,605,984 bytes with the 14,235,488 of its header).
const TENSOR_BYTES: u64 = 2_231_370_496;

/// The hyperparameters the keys state.
fn pinned() -> Hparams {
    Hparams {
        n_layer: 3,
        n_embd: 6144,
        n_ff: 24_576,
        n_head: 64,
        kv_lora: 512,
        head_k: 128,
        head_v: 128,
        rope_dims: 64,
        rope_base: 10_000.0,
        rms_eps: 1e-5,
        yarn: YarnKeys {
            factor: 2048.0,
            orig_ctx: 512,
            beta_fast: 32.0,
            beta_slow: 1.0,
            log_mul: 1.0,
        },
        window: 2048,
        softcap: 30.0,
        n_ctx_train: 1_048_576,
        block_size: 3,
        target_layers: vec![2, 19, 37, 55, 72],
        sample_from_anchor: true,
        n_vocab: 204_800,
        markov_rank: 512,
    }
}

/// `read` over `split`, as text on a failure.
fn read(split: &Split) -> Result<Hparams, String> {
    Hparams::read(split).map_err(|e| e.to_string())
}

#[test]
fn mistrallarge4_dspark_header() {
    let split = header_tables::header_of("dspark", KV, TENSORS);
    let target_split = header_tables::header_of("dspark-target", TARGET_KV, TARGET_TENSORS);
    let mut bad: Vec<String> = Vec::new();
    let hp = match read(&split) {
        Ok(hp) => hp,
        Err(e) => panic!("hparams: {e}"),
    };
    let target = Target::read(&target_split).unwrap_or_else(|e| panic!("target hparams: {e}"));
    let unread = dspark::unread_keys(&split);
    let stated: u64 = dspark::tensors(&hp).iter().map(Tensor::bytes).sum();
    let filed: u64 = split.iter_tensors().map(|(_, t)| t.nbytes).sum();
    println!(
        "mistral-large4 dspark tensors={} stated_bytes={stated} file_bytes={filed} layers={} \
         taps={:?} block={} window={} rank={} unread={unread:?}",
        split.tensor_count(),
        hp.n_layer,
        hp.target_layers,
        hp.block_size,
        hp.window,
        hp.markov_rank,
    );
    let mut pin = |what: &str, ok: bool, detail: String| {
        if !ok {
            bad.push(format!("{what}: {detail}"));
        }
    };
    pin("is_draft", dspark::is_draft(&split), "false".to_string());
    pin(
        "hparams",
        hp == pinned(),
        format!("{hp:?} is not {:?}", pinned()),
    );
    pin(
        "check",
        dspark::check(&split, &hp).is_ok(),
        format!("{:?}", dspark::check(&split, &hp)),
    );
    pin(
        "fit",
        dspark::fit(&hp, &target).is_ok(),
        format!("{:?}", dspark::fit(&hp, &target)),
    );
    pin(
        "tensor bytes",
        stated == TENSOR_BYTES && filed == TENSOR_BYTES,
        format!("stated {stated}, file {filed}, pinned {TENSOR_BYTES}"),
    );
    let mut sorted = unread.clone();
    sorted.sort();
    pin(
        "unread keys",
        sorted == ["general.sampling.temp", "general.sampling.top_p"],
        format!("{unread:?}"),
    );
    // The two other drafts under the same architecture string each refuse this file by name.
    let v41 = model::arch::dspark::DraftHparams::read(&split)
        .err()
        .map(|e| e.to_string());
    pin(
        "V4.1's reader refuses the file",
        v41.is_some(),
        "it read an MLA draft as V4.1's".to_string(),
    );
    let dflash2 = model::arch::dflash2::Hparams::read(&split)
        .err()
        .map(|e| e.to_string());
    pin(
        "DFlash2's reader refuses the file",
        dflash2
            .as_deref()
            .is_some_and(|e| e.contains("selector_hidden")),
        format!("{dflash2:?}"),
    );
    // A target of another width, vocabulary or depth is refused by name.
    let mut narrow = target.clone();
    narrow.n_embd -= 1;
    pin(
        "fit refuses another stream width",
        dspark::fit(&hp, &narrow)
            .err()
            .is_some_and(|e| e.to_string().contains("embedding_length")),
        "accepted".to_string(),
    );
    let mut shallow = target.clone();
    shallow.n_layer = 71;
    pin(
        "fit refuses a tap past the target",
        dspark::fit(&hp, &shallow)
            .err()
            .is_some_and(|e| e.to_string().contains("target_layers")),
        "accepted".to_string(),
    );
    let mut capped = target.clone();
    capped.softcap = 29.0;
    pin(
        "fit refuses another soft cap",
        dspark::fit(&hp, &capped)
            .err()
            .is_some_and(|e| e.to_string().contains("final_logit_softcapping")),
        "accepted".to_string(),
    );
    let mut vocab = target;
    vocab.n_vocab += 1;
    pin(
        "fit refuses another vocabulary",
        dspark::fit(&hp, &vocab)
            .err()
            .is_some_and(|e| e.to_string().contains("vocab_size")),
        "accepted".to_string(),
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
