//! A glm5next file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `deepseek41::place` is for
//! V4.1. The hyperparameters, every tensor's role, the typed description and
//! each layer's KV bytes ([`KvLayout`]) are read once
//! ([`PlanInputs::read`]); a file with a feature the engine does not run is
//! refused there by the coverage check; then the placement
//! ([`PlanInputs::plan`]) runs the expert rule on the routed layers whose
//! stacks the card experts read ([`card_routed`]), the rest of every layer's
//! experts on the host, and refuses a context past the deepest one a
//! reference set checks the token-pool selector at ([`ORACLE_POSITIONS`]).

use gguf::{GgmlType, Split};
use models::ModelSpec;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::{
    self, CardFormat, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers,
    Unimplemented, Violation,
};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// The widest call a later call may roll back into, in positions: the conv
/// ring keeps `conv − 1 + PASS_ROWS` inputs. The card body binds it to its
/// kernels' own constant.
pub const PASS_ROWS: usize = 8;

/// The routed stacks the program's card experts run, in their file bytes
/// ([`CardFormat::KQuant`]): Q4_K (the gate·up) and Q5_K (a gate·up or the
/// down `_sel`). A layer with a stack of any other type — the Q6_K downs —
/// keeps its experts on the host.
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    match ty {
        GgmlType::Q4_K | GgmlType::Q5_K => CardFormat::of(ty),
        _ => None,
    }
}

/// What a plan of a glm5next file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's recurrent and cache bytes.
    pub kv: KvLayout,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// The context asks for more positions than a reference set checks the
    /// token-pool selector at: past them no oracle defines what it keeps.
    #[error(
        "ctx_max {ctx_max}: the deepest context a reference set checks the token-pool selector \
         at is {served} positions; past it nothing defines what the latent layers keep"
    )]
    PastOracle { ctx_max: u64, served: u64 },
}

/// The violations, `; `-separated.
fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
}

/// The deepest context the plan serves: ik's `--dsa` step set at position
/// 16,382 of a 16,384-position context (`tools/ref/models/glm5next.sh`,
/// variant `d16kdsa`), the deepest reference the selector is checked at.
pub const ORACLE_POSITIONS: u64 = 16_384;

/// The most visible positions at which the token-pool indexer keeps every
/// one: every whole pool while there are at most `top_k / kpool` of them,
/// plus the tail of fewer than `kpool` positions no pool holds yet. Up to
/// this count a row's list is every position.
#[must_use]
pub fn dense_positions(hp: &Hparams) -> u64 {
    let (top_k, kpool) = (hp.indexer.top_k as u64, hp.indexer.kpool as u64);
    (top_k / kpool) * kpool + kpool - 1
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then the file is
    /// refused if it has a feature the engine does not run
    /// ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::ThinkSpan))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
        })
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`: the expert rule on the layers whose routed
    /// stacks [`card_routed`] runs, each layer's id prefix;
    /// refused past [`ORACLE_POSITIONS`], when it cannot be built, or when it
    /// breaks an invariant.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max > ORACLE_POSITIONS {
            return Err(PlaceError::PastOracle {
                ctx_max,
                served: ORACLE_POSITIONS,
            });
        }
        let plan =
            placement::plan_routed(&self.model, machine, ctx_max, &self.kv, levers, card_routed)?;
        let broken = plan.violations();
        if broken.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::Broken(broken))
        }
    }
}

/// One file's per-layer bytes beside the weights: a KDA layer's recurrent
/// state (`n_head` heads of `d × d` f32) and conv ring (`conv − 1 +
/// PASS_ROWS` rows of the q, k and v channels in f32), both fixed by the
/// file; a latent layer's cache, a latent row and an index row
/// (`[key; gate]`, twice the indexer's key width) in f16 a position, and a
/// pool key (the indexer's key width in f16) every `kpool` positions, the
/// last pool whole.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer: its kind.
    kinds: Vec<Kind>,
    /// A KDA layer's state and ring, in bytes.
    recurrent: u64,
    /// A latent layer's cache bytes a position.
    row: u64,
    /// A latent layer's pool key bytes, and the positions a pool holds.
    pool_row: u64,
    kpool: u64,
}

impl KvLayout {
    /// The layout `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout {
            kinds: hp.kinds[..hp.n_trunk].to_vec(),
            recurrent: recurrent_bytes(hp.n_head, hp.kda_head_dim, hp.conv),
            row: row_bytes(hp.kv_lora, hp.indexer.head_dim),
            pool_row: hp.indexer.head_dim as u64 * F16_BYTES,
            kpool: hp.indexer.kpool as u64,
        }
    }
}

/// A KDA layer's state and conv ring over `heads` heads `d` wide with a
/// `conv`-tap conv, in bytes.
fn recurrent_bytes(heads: usize, d: usize, conv: usize) -> u64 {
    let (heads, d) = (heads as u64, d as u64);
    let ring_rows = (conv as u64).saturating_sub(1) + PASS_ROWS as u64;
    heads * d * d * F32_BYTES + ring_rows * 3 * heads * d * F32_BYTES
}

/// A latent layer's cache bytes a position: the `latent`-wide row and the
/// index row of an `index_d`-wide key and its gate.
fn row_bytes(latent: usize, index_d: usize) -> u64 {
    (latent as u64 + 2 * index_d as u64) * F16_BYTES
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.kinds.get(layer) {
            Some(Kind::Kda) => self.recurrent,
            Some(Kind::Latent) => ctx_max * self.row + ctx_max.div_ceil(self.kpool) * self.pool_row,
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind, KvBytes, KvLayout, recurrent_bytes, row_bytes};

    /// GLM-5.3-Flash's sizes: a KDA layer holds 64 heads of 128 × 128 f32 and
    /// eleven conv rows of 24,576 f32 channels; a latent layer 512 + 256 f16 a
    /// position and 128 f16 a pool of four, the last one whole; a layer past
    /// the trunk nothing.
    #[test]
    fn glm_layer_bytes() {
        assert_eq!(recurrent_bytes(64, 128, 4), 4_194_304 + 11 * 24_576 * 4);
        assert_eq!(row_bytes(512, 128), 1536);
        let kv = KvLayout {
            kinds: vec![Kind::Kda, Kind::Kda, Kind::Kda, Kind::Latent],
            recurrent: recurrent_bytes(64, 128, 4),
            row: row_bytes(512, 128),
            pool_row: 256,
            kpool: 4,
        };
        assert_eq!(kv.layer_bytes(0, 2051), 5_275_648);
        assert_eq!(kv.layer_bytes(3, 2051), 2051 * 1536 + 513 * 256);
        assert_eq!(kv.layer_bytes(3, 16_384), 16_384 * (1536 + 64));
        assert_eq!(kv.layer_bytes(4, 2051), 0);
    }
}
