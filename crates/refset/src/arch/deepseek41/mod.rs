//! The DeepSeek-V4.1-Flash families: the draft set and the KLD bases, dumped from the V4.1 file
//! the tree runs ([`gguf::v41::model`]) by the sink-fixed ik tree, [`IK_BUILD`], and the node dumps
//! of the fixture file in [`fixture`].
//!
//! PIN(2026-11-18): the real-file ik node dumps, the candidate-mask set and the greedy sets retired; ik cannot read bq3km's engram schema; the fixture node dumps carry V4.1's ik comparison, so the architecture has no real node-dump family

use crate::RefError;
use crate::family::{Build, Family, Identity};

pub mod fixture;

/// The ik tree every V4.1 oracle family is dumped from, and the one place a
/// re-take of the oracle changes it.
// PIN(2026-09-23): the sink-fixed tree (/home/user/ik-idxkey); its 5-token
// batch set keeps the bytes 49ef19d0 dumped, since the fixed branch does not
// run in a 5-token prefill.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every V4.1 manifest names in its `# arch` line.
pub const ARCH: &str = "deepseek41";

/// The V4.1 file the tree runs ([`gguf::v41::try_model`]), as a family's
/// `runs`: the refusal of variables that name two files is the family's
/// error, not a panic.
pub fn model() -> Result<String, RefError> {
    gguf::v41::try_model().map_err(|e| RefError::missing(std::path::PathBuf::new(), e.to_string()))
}

/// The deepest context our numbers are held to ik's at: [`fixture::D2`]'s decode step
/// at position 1,025, which reads 1,026 positions. Past 16,384 positions,
/// where the file's mask of 2,048 blocks selects, no ik set holds ours; there
/// the mask is held to the reference's rule (`gate_deepseek41_chain_attn`'s
/// candidate clause, `weekly-gpu-ds41-cand`). A load prints it beside the context it
/// serves; it does not bound that context.
pub const VERIFIED_POSITIONS: u64 = 1026;

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
    runs: Some(model),
    draft_runs: Some(dspark_model),
    consumers: &[
        "gate-gpu-dspark-graph",
        "gate-gpu-dspark-kv",
        "gate-gpu-dspark-hc",
        "gate-gpu-dspark-experts",
    ],
};

/// The draft file the tree runs: `$BLOOMERY_DSPARK_MODEL`
/// ([`super::draft::dspark_lever`]), which the dspark recipes export from the
/// V4.1 profile's `DSPARK_MODEL`.
pub fn dspark_model() -> Result<String, RefError> {
    super::draft::dspark_lever().ok_or_else(|| RefError::Missing {
        path: std::path::PathBuf::new(),
        what: "BLOOMERY_DSPARK_MODEL unset — the dspark recipes export it from the V4.1 \
               profile's DSPARK_MODEL"
            .to_string(),
    })
}

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
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-ds41-kld", "run-ds41-ppl"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&DSREF, &KLD, &fixture::IK];

#[cfg(test)]
mod tests;
