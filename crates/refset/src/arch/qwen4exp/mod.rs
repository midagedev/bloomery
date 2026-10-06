//! The Qwen3.8-Flash-Next families. The node dumps are ik's over one of the
//! two split sets the tree runs — unsloth's UD-Q4_K_XL, [`MODEL`], by the
//! qwen4exp profile (`tools/ref/models/qwen4exp.sh`), and its UD-Q3_K_XL,
//! [`MODEL_Q3`], by the qwen4exp-q3 profile (`tools/ref/models/qwen4exp-q3.sh`)
//! — both by the ik tree [`IK_BUILD`], a family a file, so a set of one is
//! refused by name where the other is read ([`DUMPS`] holds a row a file;
//! [`dumps_of`] picks one by the first shard a program loads); the MTP draft
//! set from the Q4 file and the shared draft file by the ik tree that carries
//! the MTP graph ([`mtp`]).

use std::path::Path;

use crate::RefError;
use crate::family::{Build, Family, Identity};

pub mod mtp;

/// The first shard of the split set every qwen4exp set is dumped from and
/// the tree runs: the identity each set's `# model` line states. The family
/// owns the path until the model crate reads this architecture; a set of
/// the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

/// The first shard of unsloth's UD-Q3_K_XL split set (three shards): the
/// file the Q3 family's sets state and a load under the qwen4exp-q3 profile
/// runs, as [`MODEL`] is the Q4 file's.
pub const MODEL_Q3: &str =
    "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf";

/// The ik tree every qwen4exp oracle family is dumped from, and the one
/// place a re-take of the oracle changes it.
// PIN(2026-09-27): /home/user/ik-idxkey at its HEAD, whose libllama.so builds
// qwen4exp and which the installed dump_ref links against: the build the
// deepseek41, qwen35moe and glm5next families pin.
pub const IK_BUILD: &str = "db517b69";

/// The architecture every qwen4exp manifest names in its `# arch` line.
pub const ARCH: &str = "qwen4exp";

/// The 5-token prefill, ik on the CPU.
pub const BATCH: &str = "ref_qwen4exp";
/// Step 4 after a fused prefill of the batch set's first four ids.
pub const STEP4: &str = "ref_qwen4exp_step4";
/// [`STEP4`] after a prefill run node by node under the dumped schedule.
pub const STEP4_EVERY_NODE: &str = "ref_qwen4exp_step4_every_node";
/// Step 1,024 of the prose after a fused prefill.
pub const D1K: &str = "ref_qwen4exp_d1k";
/// Step 3,000 of the prose after a fused prefill: the QSA layers attend to
/// the cells their indexer selects, not to every cell.
pub const D3K: &str = "ref_qwen4exp_d3k";

/// The deepest context our numbers are held to ik's at: [`D3K`]'s decode
/// step at position 3,000, which reads 3,001 positions. A load prints it
/// beside the context it serves; it does not bound that context.
pub const VERIFIED_POSITIONS: u64 = 3001;

/// The decode-step sets (the model profile's `ref_step_variant`), by
/// position.
pub const STEP_SETS: &[&str] = &[STEP4, STEP4_EVERY_NODE, D1K, D3K];

/// [`MODEL`], as a family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// ik's node dumps: the batch set and the decode-step sets.
pub static IK: Family = Family {
    name: "ik-qwen4exp",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K, D3K],
    resolve: None,
    recipe: "just dump-ref-qwen4exp [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-e2e"],
};

/// The UD-Q3_K_XL file's set names, the Q4 family's one for one.
pub const BATCH_Q3: &str = "ref_qwen4exp_q3";
/// Step 4 after a fused prefill of the batch set's first four ids, of the
/// UD-Q3_K_XL file.
pub const STEP4_Q3: &str = "ref_qwen4exp_q3_step4";
/// [`STEP4_Q3`] after a prefill run node by node under the dumped schedule.
pub const STEP4_EVERY_NODE_Q3: &str = "ref_qwen4exp_q3_step4_every_node";
/// Step 1,024 of the prose after a fused prefill, of the UD-Q3_K_XL file.
pub const D1K_Q3: &str = "ref_qwen4exp_q3_d1k";
/// Step 3,000 of the prose after a fused prefill, of the UD-Q3_K_XL file.
pub const D3K_Q3: &str = "ref_qwen4exp_q3_d3k";

/// The UD-Q3_K_XL file's decode-step sets (the qwen4exp-q3 profile's
/// `ref_step_variant`), by position.
pub const STEP_SETS_Q3: &[&str] = &[STEP4_Q3, STEP4_EVERY_NODE_Q3, D1K_Q3, D3K_Q3];

/// [`MODEL_Q3`], as the Q3 family's `runs`.
fn model_q3() -> String {
    MODEL_Q3.to_string()
}

/// ik's node dumps of the UD-Q3_K_XL split set: the Q4 family's batch and
/// decode-step sets over the Q3 file. The sets are not in place yet: they
/// land with `just dump-ref-qwen4exp-q3` (the dump round's, under its lease
/// card), and `sets` fills with them as they land — a name listed before
/// its set lands reads as a red `refset-check` on the box.
pub static IK_Q3: Family = Family {
    name: "ik-qwen4exp-q3",
    sets: &[],
    resolve: None,
    recipe: "just dump-ref-qwen4exp-q3 [VARIANT]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(model_q3),
    draft_runs: None,
    consumers: &["gate-gpu-qwen4exp-e2e"],
};

/// One of the two split sets the tree runs, with everything a program that
/// loads the file is judged against: its node-dump family and the family's
/// set names, one for one with the Q4 file's.
#[derive(Debug)]
pub struct Dumps {
    /// The first shard, the identity every set of the file states.
    pub model: &'static str,
    /// The family every set of the file opens through.
    pub family: &'static Family,
    /// The 5-token prefill, ik on the CPU.
    pub batch: &'static str,
    /// Step 4 after a fused prefill of the batch set's first four ids.
    pub step4: &'static str,
    /// [`Dumps::step4`] after a prefill run node by node under the dumped
    /// schedule.
    pub step4_every_node: &'static str,
    /// Step 1,024 of the prose after a fused prefill.
    pub d1k: &'static str,
    /// Step 3,000 of the prose after a fused prefill: the QSA layers attend
    /// to the cells their indexer selects, not to every cell.
    pub d3k: &'static str,
}

/// The UD-Q4_K_XL split set's row.
pub static Q4: Dumps = Dumps {
    model: MODEL,
    family: &IK,
    batch: BATCH,
    step4: STEP4,
    step4_every_node: STEP4_EVERY_NODE,
    d1k: D1K,
    d3k: D3K,
};

/// The UD-Q3_K_XL split set's row.
pub static Q3: Dumps = Dumps {
    model: MODEL_Q3,
    family: &IK_Q3,
    batch: BATCH_Q3,
    step4: STEP4_Q3,
    step4_every_node: STEP4_EVERY_NODE_Q3,
    d1k: D1K_Q3,
    d3k: D3K_Q3,
};

/// The two split sets' rows, in the order [`FAMILIES`] lists their families.
static DUMPS: &[&Dumps] = &[&Q4, &Q3];

/// The row of the first shard `model`: the Q4 file's or the Q3 file's, by
/// path. Any other file is an error naming it and the two — a program of
/// this architecture runs one of the two split sets, and no set of either
/// family opens for another file.
pub fn dumps_of(model: &Path) -> Result<&'static Dumps, RefError> {
    DUMPS
        .iter()
        .copied()
        .find(|d| Path::new(d.model) == model)
        .ok_or_else(|| {
            RefError::missing(
                model,
                format!(
                    "{} is neither of the two qwen4exp split sets the tree runs, {} (the Q4 file, \
                     the qwen4exp profile) and {} (the Q3 file, the qwen4exp-q3 profile)",
                    model.display(),
                    MODEL,
                    MODEL_Q3
                ),
            )
        })
}

/// The architecture's families, in the order `refset-check` lists them.
/// [`IK`] first: `arch::node_dumps` returns the first node-dump family, and
/// the Q4 file's is the architecture's.
pub static FAMILIES: &[&Family] = &[&IK, &IK_Q3, &mtp::MTP];

#[cfg(test)]
mod tests;
