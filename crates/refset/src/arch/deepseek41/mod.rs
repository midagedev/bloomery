//! The DeepSeek-V4.1-Flash families: the node dumps of the fixture file in [`fixture`], by the
//! sink-fixed ik tree, [`IK_BUILD`].
//!
//! PIN(2026-11-18): the real-file ik node dumps, the candidate-mask set and the greedy sets retired; ik cannot read bq3km's engram schema; the fixture node dumps carry V4.1's ik comparison, so the architecture has no real node-dump family
//! PIN(2026-11-18): the DSpark draft set (`ref-draft/code64_n32_w3`) and the KLD bases (`ikppl/kldbase-c2048x4`, `kldbase-c512x16`) retired with them; they were dumped from the Q3_K_M file and ik cannot read bq3km

use crate::RefError;
use crate::family::Family;

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

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&fixture::IK];

#[cfg(test)]
mod tests;
