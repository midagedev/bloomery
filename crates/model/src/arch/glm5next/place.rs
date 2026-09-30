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
use runtime::stores;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::workstation::GRANULE;
use crate::placement::{
    self, Card, CardFormat, CardTotals, Host, KvBytes, Machine, ModelTensor, ModelTensors,
    PlacementError, Plan, PlanLevers, Unimplemented, Violation,
};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// The widest call a later call may roll back into, in positions: the conv
/// ring keeps `conv − 1 + PASS_ROWS` inputs. The card body binds it to its
/// kernels' own constant.
pub const PASS_ROWS: usize = 8;

/// Lanes of a KDA layer's recurrent state: a verify of up to this many rows
/// keeps the state after each row in a lane of its own, each lane stamped
/// with its position. The card body binds it to its own constant.
pub const KDA_LANES: usize = 2;

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
    /// A NextN load on a machine of other than one card: the draft layer
    /// runs beside the target's chain on its card.
    #[error(
        "the NextN load runs the next-token layer on the target's one card; the machine has \
         {cards} cards"
    )]
    NextnCards { cards: usize },
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

/// Card bytes the NextN walk's arena may hold beside the plan's tensors
/// [derived, an upper bound: the one-row step buffers the full row runs on
/// (`RowScratch` at the shared expert's width: the streams 32,768 f32, the
/// latent mixer's rows 101,376 f32 and its attention's partials over a
/// list's 2,051 positions, 33 segments of 64 rows of 512 f32 and their
/// maxima, about 1.09 M f32, the KDA rows 90,112 f32 the full row never
/// writes, the selector's scores 4,096 f32 at 16,384 positions); the walk's
/// rows for eight positions (the embedding, hidden and packed rows, the
/// projection's output, its norm and the joined projection's rows, 77,824
/// f32); the head's input, logits (154,880 f32) and argmax words; the host
/// sums' row and the verify's row-0 streams (4,096 and 16,384 f32) — about
/// 1.45 M values, 5.8 MB, rounded up to 16 MiB]. A load whose arena passes it
/// is refused by name.
pub const NEXTN_ARENA_BYTES: u64 = 16 << 20;

/// What a NextN load is planned from ([`PlanInputs::plan_nextn`]): the
/// file's next-token layer as a model of one layer — its tensors in their
/// [`roles::nextn_role`], layer 0 of one — on a card of its own and a host of
/// its own, both of no bound (the bounds are the target machine's, checked on
/// the sums), and its store: a latent layer's, one latent row and one index
/// row a position and a pool key a pool.
#[derive(Debug)]
pub struct NextnInputs {
    /// The layer's index in the file, the first past the trunk: its tensors
    /// are `blk.{index}.*`.
    pub index: usize,
    /// The layer's tensors as layer 0 of one.
    pub model: ModelTensors,
    /// The layer's own machine: one card and a host, no bound on either.
    pub machine: Machine,
    kv: NextnKv,
}

/// The stems a next-token layer's load needs: its own (the head's norm
/// among them — the load runs the shared head through it), every latent
/// mixer's, a routed block's and a shared expert's the file must carry.
fn nextn_required() -> impl Iterator<Item = &'static str> {
    roles::NEXTN
        .iter()
        .map(|s| s.name)
        .chain(roles::required(roles::LATENT))
        .chain(roles::required(roles::MOE))
        .chain(roles::required(roles::SHARED))
}

impl NextnInputs {
    /// The next-token layer of the file `inputs` describes. Refused by name:
    /// a file of other than one next-token layer, a layer that is not a
    /// latent one, a stem of it no next-token layer carries, and a stem the
    /// load needs absent.
    pub fn read(inputs: &PlanInputs) -> Result<NextnInputs, PlacementError> {
        let hp = &inputs.hp;
        let refuse = |detail: String| PlacementError::Metadata {
            key: "nextn_predict_layers".to_string(),
            detail,
        };
        if hp.n_layer != hp.n_trunk + 1 {
            return Err(refuse(format!(
                "{} next-token layers; the NextN load runs one",
                hp.n_layer - hp.n_trunk
            )));
        }
        let index = hp.n_trunk;
        if hp.kinds.get(index) != Some(&Kind::Latent) {
            return Err(refuse(format!(
                "layer {index} is {:?}; the NextN load runs a latent layer",
                hp.kinds.get(index)
            )));
        }
        let prefix = format!("blk.{index}.");
        let mut tensors = Vec::new();
        for t in &inputs.model.tensors {
            let Some(stem) = t.name.strip_prefix(&prefix) else {
                continue;
            };
            let role = roles::nextn_role(stem).ok_or_else(|| PlacementError::Tensor {
                name: t.name.clone(),
                detail: "a stem no next-token layer carries".to_string(),
            })?;
            tensors.push(ModelTensor {
                layer: Some(0),
                role,
                ..t.clone()
            });
        }
        if let Some(missing) =
            nextn_required().find(|stem| !tensors.iter().any(|t| t.name == prefix.clone() + stem))
        {
            return Err(PlacementError::Tensor {
                name: format!("{prefix}{missing}"),
                detail: "is not in the file; the NextN load needs it".to_string(),
            });
        }
        Ok(NextnInputs {
            index,
            model: ModelTensors {
                tensors,
                layers: 1,
                experts: inputs.model.experts,
                experts_used: inputs.model.experts_used,
            },
            machine: nextn_machine(),
            kv: NextnKv {
                row: row_bytes(hp.kv_lora, hp.indexer.head_dim),
                pool_row: hp.indexer.head_dim as u64 * F16_BYTES,
                kpool: hp.indexer.kpool as u64,
            },
        })
    }
}

/// The NextN layer's own machine: one card of no bound with the
/// workstation's granule, running its one layer, and a host of no bound.
fn nextn_machine() -> Machine {
    Machine {
        cards: vec![Card {
            name: "nextn".to_string(),
            usable_bytes: u64::MAX,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: GRANULE,
            layers: 0..1,
            head: true,
            token_embedding: false,
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        host: Host {
            usable_bytes: u64::MAX,
            reserves: Vec::new(),
        },
    }
}

/// The NextN layer's store: a latent layer's ([`KvLayout`]'s rule).
#[derive(Clone, Copy, Debug)]
struct NextnKv {
    row: u64,
    pool_row: u64,
    kpool: u64,
}

impl KvBytes for NextnKv {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer != 0 {
            return 0;
        }
        ctx_max * self.row + ctx_max.div_ceil(self.kpool) * self.pool_row
    }
}

/// A glm5next plan with its NextN layer ([`PlanInputs::plan_nextn`]).
#[derive(Debug)]
pub struct NextnPlan<'a> {
    /// The target's plan: [`PlanInputs::plan`]'s expert rule within the
    /// card's budget less the NextN layer's card bytes and arena, so the sum
    /// fits; its totals are the target's alone.
    pub plan: Plan<'a>,
    /// The NextN layer's plan on [`NextnInputs::machine`]: its tensors but
    /// the routed stacks on its card, the routed stacks every expert on the
    /// host, and its store (`cards[0].kv_bytes`).
    pub nextn: Plan<'a>,
    /// [`NEXTN_ARENA_BYTES`], counted beside the NextN layer's card bytes.
    pub arena_bytes: u64,
    /// The card's usable bytes (capped by the card budget) less the target's
    /// and the NextN layer's card terms; the margin is inside it.
    pub headroom_bytes: i128,
    /// The host's headroom less the NextN layer's routed experts.
    pub host_headroom_bytes: i128,
}

impl NextnPlan<'_> {
    /// Bytes the NextN layer adds to the card: its granules and its store.
    #[must_use]
    pub fn nextn_card_bytes(&self) -> u64 {
        nextn_card_bytes(&self.nextn.cards[0])
    }

    /// Device bytes the NextN layer's tensors hold, the allocator's
    /// rounding excluded: what its load's uploads sum to.
    #[must_use]
    pub fn nextn_resident_bytes(&self) -> u64 {
        self.nextn.cards[0].dense_bytes + self.nextn.cards[0].expert_bytes
    }
}

/// A NextN plan's card terms: its granules and its store.
fn nextn_card_bytes(d: &CardTotals) -> u64 {
    d.dense_bytes + d.expert_bytes + d.rounding_bytes + d.kv_bytes
}

impl PlanInputs {
    /// [`PlanInputs::plan`] with the next-token layer `nextn` loaded on the
    /// machine's one card: the NextN layer's own plan (its routed stacks on
    /// the host), the target's expert rule within the card's budget less the
    /// NextN layer's card bytes and arena, the card's bound checked on the
    /// sum and the host's on the target's host set and the NextN layer's
    /// routed experts. Refused as [`PlanInputs::plan`] refuses and on a
    /// machine of other than one card; every violation of either plan is
    /// listed, the card's and the host's bound once.
    pub fn plan_nextn<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        nextn: &'a NextnInputs,
    ) -> Result<NextnPlan<'a>, PlaceError> {
        if ctx_max > ORACLE_POSITIONS {
            return Err(PlaceError::PastOracle {
                ctx_max,
                served: ORACLE_POSITIONS,
            });
        }
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::NextnCards {
                cards: machine.cards.len(),
            });
        };
        let draft = placement::plan_host_routed(
            &nextn.model,
            &nextn.machine,
            ctx_max,
            &nextn.kv,
            &PlanLevers::default(),
        )?;
        let arena = NEXTN_ARENA_BYTES;
        let reserve = nextn_card_bytes(&draft.cards[0]) + arena;
        let plan = placement::plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &self.kv,
            levers,
            card_routed,
            reserve,
        )?;
        let t = &plan.cards[0];
        let total = [
            t.dense_bytes,
            t.expert_bytes,
            t.rounding_bytes,
            t.kv_bytes,
            t.scratch_bytes,
            t.context_bytes,
            t.reserve_bytes,
            nextn_card_bytes(&draft.cards[0]),
            arena,
        ]
        .iter()
        .sum::<u64>();
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        let host_headroom = plan.host.headroom_bytes - i128::from(draft.host.expert_bytes);
        let mut broken: Vec<Violation> = plan
            .violations()
            .into_iter()
            .filter(|v| !matches!(v, Violation::CardOver { .. } | Violation::HostOver { .. }))
            .chain(
                draft
                    .violations()
                    .into_iter()
                    .filter(|v| !matches!(v, Violation::HostOver { .. })),
            )
            .collect();
        if total > limit {
            broken.push(Violation::CardOver {
                card: card.name.clone(),
                total,
                limit,
            });
        }
        if host_headroom < 0 {
            let usable = machine.host.usable_bytes;
            broken.push(Violation::HostOver {
                total: u64::try_from(i128::from(usable) - host_headroom).unwrap_or(u64::MAX),
                usable,
            });
        }
        if !broken.is_empty() {
            return Err(PlaceError::Broken(broken));
        }
        Ok(NextnPlan {
            headroom_bytes: i128::from(usable) - i128::from(total),
            host_headroom_bytes: host_headroom,
            plan,
            nextn: draft,
            arena_bytes: arena,
        })
    }
}

/// One file's per-layer bytes beside the weights: a KDA layer's recurrent
/// state ([`KDA_LANES`] lanes of `n_head` heads of `d × d` f32, a u32 stamp
/// a lane) and conv ring (`conv − 1 + PASS_ROWS` rows of the q, k and v
/// channels in f32), both fixed by the file; a latent layer's cache, a latent row and an index row
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

/// A KDA layer's state of [`KDA_LANES`] lanes with their stamps and its conv
/// ring over `heads` heads `d` wide with a `conv`-tap conv, in bytes: the
/// lanes past the first counted by `runtime::stores`, the rule the delta
/// stores are allocated by.
fn recurrent_bytes(heads: usize, d: usize, conv: usize) -> u64 {
    let lanes = stores::delta_lane_bytes(heads, d, KDA_LANES);
    let (heads, d) = (heads as u64, d as u64);
    let ring_rows = (conv as u64).saturating_sub(1) + PASS_ROWS as u64;
    heads * d * d * F32_BYTES + lanes + ring_rows * 3 * heads * d * F32_BYTES
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

    /// GLM-5.3-Flash's sizes: a KDA layer holds two lanes of 64 heads of 128
    /// × 128 f32, a u32 stamp a lane, and eleven conv rows of 24,576 f32
    /// channels; a latent layer 512 + 256 f16 a position and 128 f16 a pool
    /// of four, the last one whole; a layer past the trunk nothing.
    #[test]
    fn glm_layer_bytes() {
        assert_eq!(
            recurrent_bytes(64, 128, 4),
            2 * 4_194_304 + 2 * 4 + 11 * 24_576 * 4
        );
        assert_eq!(row_bytes(512, 128), 1536);
        let kv = KvLayout {
            kinds: vec![Kind::Kda, Kind::Kda, Kind::Kda, Kind::Latent],
            recurrent: recurrent_bytes(64, 128, 4),
            row: row_bytes(512, 128),
            pool_row: 256,
            kpool: 4,
        };
        assert_eq!(kv.layer_bytes(0, 2051), 9_469_960);
        assert_eq!(kv.layer_bytes(3, 2051), 2051 * 1536 + 513 * 256);
        assert_eq!(kv.layer_bytes(3, 16_384), 16_384 * (1536 + 64));
        assert_eq!(kv.layer_bytes(4, 2051), 0);
    }
}
