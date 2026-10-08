//! What a V4.1 whole-model gate reads of the fixture tier beyond `bloomery_gpu_gates::tier`: the
//! model file (one owner, `gguf::v41`, and in the fixture tier a whole fixture), the card the gate
//! plan is made on, the levers a gate parses, and the facts of the V4.1 file that a literal used to
//! stand for — the layer the CED triangle is anchored at and the length it reaches from there.
//! Every number here is read from the header (`Hparams`) or from a plan, and the gate that used its
//! literal prints the two side by side (`tier::witness`), equal on the real file.

use std::path::Path;

use bloomery_gpu_deepseek41::body::{self, CHUNK, CedLayer};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::oracle::deepseek41::{D1, D2, STEP4};
use bloomery_gpu_gates::tier::{self, Tier};
use gguf::Gguf;
use model::arch::deepseek41::hparams::Hparams;
use model::placement::Machine;
use model::placement::workstation::CardSpec;

/// The decode-step sets of the real file's oracle (`step4`, `d1`, `d2`), each with the position of
/// its step and the indexer `top_k` ik ran it with (`None`: the file's): `step4` at 4, where no csa
/// group completes; `d1` at 301 with `top_k` 64, so every indexer layer selects, and one group
/// completes with the window ring wrapped; `d2` at 1025. The gates that read a set for its position,
/// its ids or its `top_k` read this table in the fixture tier (the corpus ids stand in for the
/// set's), and in the real tier print each set's own beside it ([`witness_set`]).
pub const DECODE_SETS: [(&str, u32, Option<usize>); 3] =
    [(STEP4, 4, None), (D1, 301, Some(64)), (D2, 1025, None)];

/// The position and `top_k` override [`DECODE_SETS`] names for the set `name`.
///
/// # Errors
/// A set the table does not hold.
pub fn decode_set(name: &str) -> Result<(u32, Option<usize>), GateError> {
    DECODE_SETS
        .iter()
        .find(|(n, ..)| *n == name)
        .map(|&(_, at, k)| (at, k))
        .ok_or_else(|| format!("{name} is not a decode-step set of DECODE_SETS").into())
}

/// The real tier's move proof of [`DECODE_SETS`]: the set `name`'s own step position `pos` and
/// `top_k` (the file's `file_top_k` where it has no override) beside the table's, which must be
/// equal.
///
/// # Errors
/// A set the table does not hold, or one whose position or `top_k` differs from it.
pub fn witness_set(name: &str, pos: u32, top_k: usize, file_top_k: usize) -> Result<(), GateError> {
    let (at, over) = decode_set(name)?;
    let ok = tier::witness(&format!("{name} step position"), pos, at)
        & tier::witness(&format!("{name} top_k"), top_k, over.unwrap_or(file_top_k));
    if ok {
        Ok(())
    } else {
        Err(
            format!("{name}: the set's position {pos} and top_k {top_k} are not DECODE_SETS'")
                .into(),
        )
    }
}

/// The first `n` ids of `$BLOOMERY_DATA/engram/corpus-prose.ids` (text ids of the real
/// tokenizer, which a fixture's embedding table holds the rows of).
///
/// # Errors
/// A file that is missing or holds fewer ids.
pub fn prose_ids(n: usize) -> Result<Vec<u32>, GateError> {
    tier::prose_ids("engram", n)
}

/// The V4.1 file a gate opens: the one `gguf::v41::model` owner's answer (a
/// `BLOOMERY_V41_MODEL` that disagrees with `BLOOMERY_REF_MODEL` is its named error), and in the
/// fixture tier a file proven to be a whole fixture, so a gate never runs on the real file under
/// the fixture's name.
///
/// # Errors
/// The owner's refusal, a file that cannot be opened, or one the fixture tier refuses.
pub fn model_path() -> Result<String, GateError> {
    let path = gguf::v41::try_model()?;
    let tier = Tier::from_env()?;
    if tier == Tier::Fixture {
        let g = Gguf::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        tier.check_file(Path::new(&path), g.iter_kv().map(|(k, _)| k))?;
    }
    Ok(path)
}

/// Fixes the card [`plan_gate`] plans on and prints its move proof: the real tier keeps the gate
/// plan of the card the runner put in view (`gate_card`: the 3090's bytes under that card's name),
/// the fixture tier plans on `a`, the largest visible card as the device reports itself
/// (`Place::A`).
///
/// # Errors
/// A census no card is found in, or a card spec in the real tier that is not the 3090's bytes.
pub fn init() -> Result<CardSpec, GateError> {
    tier::init_card(crate::gate_card::init)
}

/// The gate placement on the card [`init`] fixed: every layer and the head on it, beside the host.
///
/// # Panics
/// Before [`init`].
#[must_use]
pub fn plan_gate(layers: usize) -> Machine {
    tier::plan_gate(layers)
}

/// The levers a gate parses: `real`, and in the fixture tier the card budget too — the fixture's
/// runner exports the header's budget, which [`tier::plan_levers`] then holds equal to the header's.
///
/// # Errors
/// A `BLOOMERY_TIER` that is neither tier.
pub fn acts_on(real: &[&'static str]) -> Result<Vec<&'static str>, GateError> {
    tier::acts_on(real)
}

/// A self-consistency clause: it runs in both tiers, so a tier that deferred it is refused by name
/// here, not skipped.
///
/// # Errors
/// As [`tier::run_clause`], or a tier that deferred a self-consistency clause.
pub fn sc(name: &str) -> Result<(), GateError> {
    tier::sc(name)
}

/// The last layer that owns a compressor or index keys: the layers above it run their block at the
/// positions a later reader needs and no more (the CED triangle, `body::Need`), the layers up to it
/// at every position.
///
/// # Errors
/// A file with no such layer.
pub fn last_owner(hp: &Hparams) -> Result<usize, GateError> {
    CedLayer::table(hp)
        .iter()
        .rposition(|l| l.compressor || l.index_keys)
        .ok_or_else(|| "the file has no layer that owns a compressor or index keys".into())
}

/// How many positions back from an aligned call end the triangle reaches at [`last_owner`]: the
/// block of layer `n_layer − 1 − j` starts `CHUNK + window · j` positions before the end, so the
/// owner's `CHUNK + window · (n_layer − 1 − owner)` (`body::ced` walks it back from the head).
///
/// # Errors
/// As [`last_owner`].
pub fn ced_reach(hp: &Hparams) -> Result<usize, GateError> {
    Ok(CHUNK + hp.window * (hp.n_layer - 1 - last_owner(hp)?))
}

/// Where layer `l`'s block starts in a single call of `p` positions from 0 that ends aligned:
/// every layer up to the owner runs from 0, the layer `j` below the last from `p − (CHUNK + window
/// · j)`.
#[must_use]
pub fn block_start(hp: &Hparams, owner: usize, l: usize, p: usize) -> usize {
    if l < owner {
        return 0;
    }
    p.saturating_sub(CHUNK + hp.window * (hp.n_layer - 1 - l))
}

/// The prompt length of the group-fault clause: the largest `p` of at most `cap` positions (steps
/// of one chunk) whose call is cut into at least three batches (`body::batches`) and at which the
/// triangle starts the block of some layer that has card experts (`n_l` of the plan) inside the
/// second batch — so that layer first runs after the call's first batch and a third batch follows
/// it.
///
/// # Errors
/// A file whose triangle puts no layer's block in the second batch at any length.
pub fn group_fault_p(hp: &Hparams, n_l: &[u64], cap: usize) -> Result<usize, GateError> {
    let owner = last_owner(hp)?;
    let mut p = cap - cap % CHUNK;
    while p >= CHUNK {
        let runs = body::batches(0, p);
        if let (Some(second), Some(_third)) = (runs.get(1), runs.get(2)) {
            let hit = (0..hp.n_layer).any(|l| {
                n_l.get(l).is_some_and(|&n| n > 0) && second.contains(&block_start(hp, owner, l, p))
            });
            if hit {
                return Ok(p);
            }
        }
        p -= CHUNK;
    }
    Err(format!(
        "no length of at most {cap} positions starts a card layer's block inside the second of          three batches: owner layer {owner}, window {}, {} layers",
        hp.window, hp.n_layer
    )
    .into())
}
