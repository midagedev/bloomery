//! The DeepSeek-V4.1-Flash families. Every one is dumped from the V4.1 file
//! the tree runs ([`gguf::v41::model`]); the node dumps, the draft set and the
//! KLD bases by the sink-fixed ik tree, [`IK_BUILD`].

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The ik tree every V4.1 oracle family is dumped from, and the one place a
/// re-take of the oracle changes it.
// PIN(2026-09-23): the sink-fixed tree (/home/user/ik-idxkey); its 5-token
// batch set keeps the bytes 49ef19d0 dumped, since the fixed branch does not
// run in a 5-token prefill.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every V4.1 manifest names in its `# arch` line.
pub const ARCH: &str = "deepseek41";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_deepseek41";
/// Step 4, at an even position, where no csa group completes.
pub const STEP4: &str = "ref_deepseek41_step4_every_node";
/// Step 301 with the file's top-k, where neither stream builds an indexer.
pub const D1N: &str = "ref_deepseek41_d1n_every_node";
/// Step 301, where a csa group completes; the window mask 512 cells wide.
pub const D1: &str = "ref_deepseek41_d1_every_node";
/// [`D1`] with the indexer's scores and top-k as nodes of their own.
pub const D1_UNFUSED: &str = "ref_deepseek41_d1_unfused_every_node";
/// Step 1,025, where a csa group completes; the window mask 1,280 cells wide.
pub const D2: &str = "ref_deepseek41_d2_every_node";
/// [`D2`] with the indexer's scores and top-k as nodes of their own.
pub const D2_UNFUSED: &str = "ref_deepseek41_d2_unfused_every_node";

/// The decode-step sets, each one token after a prefill run node by node
/// under the dumped schedule (the model profile's `ref_step_variant`,
/// `tools/ref/models/deepseek41.sh`), by position. The order is part of what
/// a gate prints: where sets tie — d1n, d1 and d1_unfused run their layers
/// before the first compressed stream alike — a line that names the first of
/// equal results names the set read first.
pub const STEP_SETS: &[&str] = &[STEP4, D1N, D1, D1_UNFUSED, D2, D2_UNFUSED];

/// ik's node dumps: the batch set and the decode-step sets. Each name
/// resolves to the set dumped from the file the tree runs
/// ([`gguf::v41::set`]).
pub static IK: Family = Family {
    name: "ik-deepseek41",
    sets: &[BATCH, STEP4, D1N, D1, D1_UNFUSED, D2, D2_UNFUSED],
    resolve: Some(gguf::v41::set),
    recipe: "just dump-ref-v41 [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(gguf::v41::model),
    draft_runs: None,
    consumers: &[
        "gate-ds41-oracle",
        "gate-ds41-plan",
        "gate-ds41-meta",
        "gate-ds41-host",
        "gate-union",
        "gate-ds41-load",
        "gate-gpu-mcol",
        "gate-gpu-ds41-attn",
        "gate-gpu-ds41-comp",
        "gate-gpu-ds41-engram",
        "gate-gpu-ds41-hc",
        "gate-gpu-ds41-index",
        "gate-gpu-ds41-moe",
        "gate-gpu-ds41-rope",
        "gate-gpu-ds41-woa",
        "gate-gpu-ds41-chain-attn",
        "gate-gpu-ds41-chain-ffn",
        "gate-gpu-ds41-chain-glue",
        "gate-gpu-ds41-step",
        "gate-gpu-ds41-skew",
    ],
};

/// The DSpark draft set: every node ik's draft computes while the target
/// decodes the first 64 ids of the code corpus, 32 positions at block width 3.
pub const DSREF_SET: &str = "ref-draft/code64_n32_w3";

/// ik's DSpark draft sets, from the target file and the draft file the tree
/// runs, by the draft tree (the pin plus its patch).
pub static DSREF: Family = Family {
    name: "dsref-deepseek41",
    sets: &[DSREF_SET],
    resolve: None,
    recipe: "just dump-ref-draft",
    identity: Identity::ManifestAndDraft,
    arch: Some(ARCH),
    build: Some(Build::Patched(IK_BUILD)),
    runs: Some(gguf::v41::model),
    draft_runs: Some(dspark_model),
    consumers: &[
        "gate-gpu-dspark-graph",
        "gate-gpu-dspark-kv",
        "gate-gpu-dspark-hc",
        "gate-gpu-dspark-experts",
    ],
};

/// The draft file the tree runs: `$BLOOMERY_DSPARK_MODEL`, which the dspark
/// recipes export from the V4.1 profile's `DSPARK_MODEL`.
pub fn dspark_model() -> Result<String, RefError> {
    match std::env::var("BLOOMERY_DSPARK_MODEL") {
        Ok(p) if !p.is_empty() => Ok(p),
        _ => Err(RefError::Missing {
            path: std::path::PathBuf::new(),
            what: "BLOOMERY_DSPARK_MODEL unset — the dspark recipes export it from the V4.1 \
                   profile's DSPARK_MODEL"
                .to_string(),
        }),
    }
}

/// ik's greedy continuation of prompt row 7, 64 steps: the long gate's
/// `--free` reference.
pub const GREEDY_P7: &str = "greedy-ds41/greedy-ik-cpu-64-p7.tsv";
/// ik's greedy continuation of prompt row 0, which ends at the model's EOS.
pub const GREEDY_P0: &str = "greedy-ds41/greedy-ik-cpu-64-p0.tsv";

/// ik's greedy continuations on the CPU, one token per decode.
pub static GREEDY: Family = Family {
    name: "greedy-deepseek41",
    sets: &[GREEDY_P7, GREEDY_P0],
    resolve: None,
    recipe: "just ik-greedy-ds41 PROMPT",
    identity: Identity::ArgmaxHeader,
    arch: None,
    build: None,
    runs: Some(gguf::v41::model),
    draft_runs: None,
    consumers: &["gate-gpu-ds41-long"],
};

/// ik's KL-divergence base of wikitext-2 at ctx 2048 over 4 chunks.
pub const KLD_C2048X4: &str = "ikppl/kldbase-c2048x4";
/// ik's KL-divergence base of wikitext-2 at ctx 512 over 16 chunks.
pub const KLD_C512X16: &str = "ikppl/kldbase-c512x16";

/// ik's KL-divergence bases, each `<tag>.kld` with the log of the run that
/// wrote it, `<tag>.log`.
pub static KLD: Family = Family {
    name: "kld-deepseek41",
    sets: &[KLD_C2048X4, KLD_C512X16],
    resolve: None,
    recipe: "just ik-ppl /home/user/ik-idxkey TAG --kld-base [--ctx N --chunks N]",
    identity: Identity::RunLog,
    arch: None,
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(gguf::v41::model),
    draft_runs: None,
    consumers: &["gate-ds41-kld", "run-ds41-ppl"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&IK, &DSREF, &GREEDY, &KLD];
