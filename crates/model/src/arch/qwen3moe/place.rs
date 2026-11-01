//! The qwen3moe program's plans: what a placed load of a Qwen3-30B-A3B
//! (`qwen3moe`) or a Qwen3.6-35B-A3B (`qwen35moe`) file runs on one card and
//! the host tier, or on a stage card with one expert tier beside that host.
//! The program (`gpu::arch::qwen3moe`'s `Body` and `Body35`)
//! puts every tensor but the routed stacks on the card — attention or delta
//! rule, router, shared expert, embedding and head — and the routed experts
//! by the expert rule over the layers whose three stacks its card experts
//! read in their parts ([`routed_rule`] of [`PROGRAM`]), the rest on the host.
//!
//! A qwen35moe file plans through `qwen35moe::place::PlanInputs::plan_rule`
//! with [`routed_rule`] of [`Program::Qwen35Body`] (its headers, roles and caches are that module's);
//! a qwen3moe file through [`PlanInputs`] here. One card is the machine
//! `workstation::placed_with_arena` lays out. A stage with one expert tier is
//! [`workstation::plan_tiers`], then [`stage_placed`]: the tier card holds no
//! layer, and the stage keeps the token embedding and the scratch the placed
//! load holds.

use gguf::Split;

use super::hparams::Hparams;
use super::roles;
use crate::placement::accepts::{Program, leg_down_forms, routed_rule};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::workstation::{self, TierBatchBytes};
use crate::placement::{
    self, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Violation, checked,
    joined,
};
use runtime::stores::{F32_BYTES, kv_plane_bytes};

/// The qwen3moe program, whose card experts read the routed stacks
/// ([`routed_rule`] of it): by part, a gate and an up of one type the placed
/// card leg's gate·up launches read and a down its down launches read
/// ([`crate::placement::accepts::prompt_leg_gate_up`],
/// [`crate::placement::accepts::prompt_leg_down`]: the same arms as Qwen3.6's
/// program, whose body shares the placed side). A layer with a stack of another
/// type (a Q3_K, a Q2_K), or of one of these in a part no launch runs it in (a
/// Q6_K gate over a Q3_K down), keeps its experts on the host.
pub const PROGRAM: Program = Program::Qwen3moeBody;

/// Tokens one chunk of the tier's prompt-batch service runs: the stage gemv
/// pass. GLM and Solar name the same chunk.
pub const TIER_CHUNK: u64 = 8;

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for blocks of
/// the host union's columns ([`crate::moe::UNION_MAX_COLS`]): the tier card's
/// staging, one chunk of the card leg's scratch and the host's rows.
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let hp = &inputs.hp;
    workstation::try_tier_leg_batch_bytes_of(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        (TIER_CHUNK, crate::moe::UNION_MAX_COLS as u64),
        leg_down_forms(PROGRAM, &inputs.model),
        false,
    )
}

/// The stage card [`workstation::plan_tiers`] lays, held as a placed load
/// holds it: the token embedding on the card, and scratch of the m = 1
/// scratch plus `arena`. [`workstation::plan_tiers`] leaves the embedding off
/// the card and the scratch at the m = 1 figure.
///
/// # Panics
///
/// When the machine has no stage card.
pub fn stage_placed(machine: &mut Machine, arena: u64) {
    let card = machine.cards.first_mut().expect("a plan has a stage card");
    card.token_embedding = true;
    card.scratch_bytes = workstation::SCRATCH + arena;
}

/// A qwen3moe file's per-layer cache: every layer's K and V planes, a
/// position's `2 · kv_heads · head_dim` values — f16, or q8_0's two-plane
/// layout when the load runs its cache in it ([`KvLayout::in_q8`], the
/// cache lever's `q8_0`).
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    q8: bool,
}

impl KvLayout {
    /// The layout `hp` describes, in f16.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout {
            layers: hp.n_layer,
            kv_heads: hp.n_head_kv,
            head_dim: hp.head_dim,
            q8: false,
        }
    }

    /// The same layout in q8_0's two-plane planes.
    #[must_use]
    pub const fn in_q8(mut self) -> KvLayout {
        self.q8 = true;
        self
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer < self.layers {
            kv_plane_bytes(self.kv_heads, self.head_dim, self.q8, ctx_max)
        } else {
            0
        }
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.head_dim as u64 * F32_BYTES * ctx_max
    }
}

/// Why a plan of a qwen3moe file was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The plan could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position.
    #[error("ctx_max 0: a plan holds at least one position")]
    Positions,
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            _ => None,
        }
    }
}

/// What a plan of a qwen3moe file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// Each layer's cache bytes.
    pub kv: KvLayout,
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, the first that
    /// fails being the error.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let kv = KvLayout::of(&hp);
        Ok(PlanInputs { hp, model, kv })
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// `levers`: the routed experts by the expert rule over the layers whose
    /// stacks [`routed_rule`] of [`PROGRAM`] loads, the rest on the host, a
    /// stage card's id prefix first and each expert tier's next ids after it.
    /// Refused when it cannot be built or breaks an invariant (every violation
    /// listed), and at a context of 0.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::Positions);
        }
        let plan = placement::plan_routed(
            &self.model,
            machine,
            ctx_max,
            &self.kv,
            levers,
            routed_rule(PROGRAM),
        )?;
        checked(plan).map_err(PlaceError::Broken)
    }
}

#[cfg(test)]
mod tests {
    use super::super::hparams::{Experts, Hparams, Rope, RopeMode, Score};
    use super::{KvBytes, KvLayout};

    #[test]
    fn the_stage_rope_row_is_the_width_the_body_copies() {
        let kv = KvLayout::of(&Hparams {
            n_layer: 1,
            n_embd: 128,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 128,
            rope: Rope {
                mode: RopeMode::Neox,
                base: 1e6,
                dims: 128,
            },
            rms_eps: 1e-6,
            n_vocab: 2,
            n_ctx_train: 128,
            experts: Experts {
                n_expert: 1,
                n_used: 1,
                ff: 1,
                score: Score::Softmax,
                weights_norm: true,
            },
        });
        assert_eq!(kv.stage_position_bytes(10), 128 * 4 * 10);
        assert_eq!(kv.in_q8().stage_position_bytes(10), 128 * 4 * 10);
    }
}
