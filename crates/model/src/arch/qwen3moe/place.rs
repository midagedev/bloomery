//! The qwen3moe program's plans: what a placed load of a Qwen3-30B-A3B
//! (`qwen3moe`) or a Qwen3.6-35B-A3B (`qwen35moe`) file runs on one card and
//! the host tier. The program (`gpu::arch::qwen3moe`'s `Body` and `Body35`)
//! puts every tensor but the routed stacks on the card — attention or delta
//! rule, router, shared expert, embedding and head — and the routed experts
//! by the expert rule over the layers whose three stacks its card experts
//! read in their parts ([`routed_rule`] of [`PROGRAM`]), the rest on the host.
//!
//! A qwen35moe file plans through `qwen35moe::place::PlanInputs::plan_rule`
//! with [`routed_rule`] of [`Program::Qwen35Body`] (its headers, roles and caches are that module's);
//! a qwen3moe file through [`PlanInputs`] here. Both run on the machine
//! [`machine`] lays out: one card with the workstation's context and margin
//! and a scratch the caller sizes from the program's own arenas.

use gguf::Split;

use super::hparams::Hparams;
use super::roles;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::workstation::{self, CardSpec, SCRATCH};
use crate::placement::{
    self, Card, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Violation,
    checked, joined,
};
use runtime::stores::{kv_q8_row_bytes, kv_row_bytes};

/// The qwen3moe program, whose card experts read the routed stacks
/// ([`routed_rule`] of it): by part, a gate and an up
/// of one type the program's routed gate·up launches read (a Q4_K or Q5_K)
/// and a down its down launches read (a Q4_K, Q5_K or Q6_K). A layer with a
/// stack of another type, or of one of these in a part no entry runs it in
/// (a Q6_K gate), keeps its experts on the host.
pub const PROGRAM: Program = Program::Qwen3moeBody;

/// The machine a placed load of the program runs on: `card` runs every one
/// of `layers`, the head and the token embedding whole, with the
/// workstation's context and margin, the m = 1 scratch plus `arena` bytes —
/// what the load's prompt arenas hold beyond it, which the program sizes
/// and its load refuses to pass ([`workstation::placed_on`]).
#[must_use]
pub fn machine(card: CardSpec, layers: usize, arena: u64) -> Machine {
    workstation::placed_on(card, layers, SCRATCH + arena)
}

/// The prompt-arena bytes a plan made on `card` by [`machine`] counts: its
/// scratch past the m = 1 scratch. A load whose arenas hold more is refused
/// by name.
#[must_use]
pub const fn counted_arena_bytes(card: &Card) -> u64 {
    card.scratch_bytes.saturating_sub(SCRATCH)
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
            let row = if self.q8 {
                kv_q8_row_bytes(self.kv_heads, self.head_dim)
            } else {
                kv_row_bytes(self.kv_heads, self.head_dim)
            };
            ctx_max * row
        } else {
            0
        }
    }
}

/// Why a plan of a qwen3moe file was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The plan could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position.
    #[error("ctx_max 0: a plan holds at least one position")]
    Positions,
    /// A machine with an expert tier card: the program hangs none.
    #[error(
        "the machine has the expert tier card {tier}; the qwen3moe program hangs no expert tier"
    )]
    Tier { tier: String },
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
    /// stacks [`routed_rule`] of [`PROGRAM`] loads, the rest on the host. Refused when it
    /// cannot be built or breaks an invariant (every violation listed), at a
    /// context of 0, and on a machine with an expert tier card.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::Positions);
        }
        if let Some(tier) = machine.tiers.first() {
            return Err(PlaceError::Tier {
                tier: tier.name.clone(),
            });
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
