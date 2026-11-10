//! A DeepSeek-V4.1 or V4 file planned onto a machine: the one path from the
//! file's headers to a plan that keeps its invariants. The hyperparameters
//! ([`Hparams`]), every tensor's role ([`roles::classify`]), the typed model
//! description ([`ModelSpec`]) and each layer's KV bytes ([`KvLayout`]) are
//! read once ([`PlanInputs::read`]), and a file with a feature the engine does
//! not run yet is refused there, every such feature listed by the coverage
//! check ([`PlanInputs::unimplemented`]); the caller lays its machine
//! out for the model's layers; then the placement and its invariants
//! ([`PlanInputs::plan`]). The engine's entry and the load gates all take this
//! path, so they refuse the same file for the same reason, in the same order.
//! [`PlanInputs::describe`] is the same read without that refusal, for what
//! only describes a file (its inventory gate).

use std::num::NonZeroUsize;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::Split;
use models::ModelSpec;

use super::batch::{BatchShape, prompt_parts};
use super::hparams::Hparams;
use super::kv::KvLayout;
use super::{roles, spec};
use crate::arch::coverage;
use crate::arch::place_hybrid;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::heads::{HeadLayout, model_row_heads};
use crate::placement::workstation::{self, HostRead, TierBatchBytes};
use bloomery_levers::KvHost;
use runtime::stores::Class;

use crate::placement::{
    self, Card, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented,
    Violation, checked_beside, joined,
};

/// The rows of the body's widest pass: the pair pass, a row a slot of two slots (the card
/// body's `PAIR_ROWS`).
pub const PASS_ROWS: usize = 2;

/// The heads a load of `slots` resident sequences of `model` makes past the load's own on stage
/// card `card` ([`model_row_heads`]: the body writes a head a row, under its RMS final norm), in
/// the card's allocator's bytes: on the card that carries the head, a pass of two slots' rows
/// ([`PASS_ROWS`]) writes row 2's head; one slot makes none, and another card none. The plan
/// holds them in the card's prompt term beside the prompt batch.
pub fn pass_heads_bytes(
    model: &ModelTensors,
    card: &Card,
    slots: usize,
) -> Result<u64, PlacementError> {
    if !card.head || slots < 2 {
        return Ok(0);
    }
    let parts = model_row_heads(model, true, HeadLayout::PerRow, 1..=PASS_ROWS)?;
    Ok(parts.card_bytes(card.granule_bytes))
}

/// What a plan of a V4.1 or V4 file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's cache and compressor bytes.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the
    /// routed experts the room cannot hold follow it to the NVMe tier
    /// ([`placement::expert_nvme_tier`]). The engram row tables are on the
    /// NVMe tier by their role at any room, so no row-table tier runs here. A
    /// gate that plans an arm by a room of its own sets it, read
    /// [`HostRead::Given`].
    ///
    /// The effective room is [`Plan::plan_at`]'s: the reading's plan room,
    /// capped by the machine's host room and unified pool.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it
    /// as its `given`. A gate that plans an arm by an arena of its own sets it.
    pub arena: Option<u64>,
    /// The page a host-capable room takes on the stage card (`bloomery_gpu::host_page`), which
    /// a plan under `BLOOMERY_KV_HOST` `spill` or `all` counts the selected rows' host bytes
    /// at; `None` where no binary read it, a plan under either word refused by name.
    pub kv_host_page: Option<u64>,
}

/// The expert tier's prompt-batch bytes plan (b′) reserves for a file of
/// hyperparameters `hp` (`workstation::plan_bp`): blocks of the host union's
/// [`crate::moe::UNION_MAX_COLS`] columns of rows of `n_embd`, routed to
/// `n_used` experts of `ff` rows — the sizes the tier's staging and tile
/// scratch are allocated at ([`workstation::tier_batch_bytes`]).
#[must_use]
pub fn tier_batch(hp: &Hparams) -> TierBatchBytes {
    workstation::tier_batch_bytes(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        crate::moe::UNION_MAX_COLS as u64,
    )
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A stage card whose plan leaves less beside it than its prompt batch
    /// takes ([`PlanInputs::prompt_reserve_bytes`]): the dense granules, the
    /// cache, context, scratch, reserves and margin already pass what the
    /// card holds less it, so no expert placement makes room.
    #[error(
        "card {card}: the prompt batch takes {batch} B, and the plan leaves {left} B beside its \
         dense granules, cache, context, scratch, reserves and margin ({short} B short)"
    )]
    BatchOver {
        card: String,
        batch: u64,
        left: u64,
        short: u64,
    },
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            Self::Broken(_) | Self::BatchOver { .. } => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            Self::Placement(_) | Self::BatchOver { .. } => None,
        }
    }
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, then its layers' KV bytes, the first that fails being the
    /// error; then the file is refused if it has a feature the engine does
    /// not run yet ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features: the
    /// file as it is, for a reader that only describes it, and the host's
    /// room ([`workstation::host_available_read`]) and the arena's given
    /// bytes, a room that cannot be read refused by name. The room is read
    /// after the file's description, through the same steps as
    /// [`PlanInputs::describe_in`].
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || {
            workstation::host_available_read().map_err(PlacementError::HostRoom)
        })
    }

    /// The file's description with `room` as the host's room, no host read
    /// for it: another machine's plan checked from this one, `room` that
    /// machine's bytes and how they were read. The arena is still this
    /// process's `BLOOMERY_NVTIER_BYTES`, as in [`PlanInputs::describe`].
    pub fn describe_in(split: &Split, room: (u64, HostRead)) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || Ok(room))
    }

    /// The file's hyperparameters, roles, spec and KV layout, then the room
    /// `room` gives, then the arena, the first that fails being the error.
    fn describe_with(
        split: &Split,
        room: impl FnOnce() -> Result<(u64, HostRead), PlacementError>,
    ) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let spec = spec::spec_of(&hp, &model, spec::chat(split)?)?;
        let kv = KvLayout::from_file(split, &hp)?;
        let room = room()?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
            kv_host_page: None,
        })
    }

    /// Every feature of the file the engine does not run yet, layer by layer
    /// in layer order, then the tensor formats, then the model-wide ones
    /// ([`coverage::check`]). Empty for a file the chain runs whole.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, refused when it cannot be built or breaks an
    /// invariant: the plan of one resident sequence
    /// ([`PlanInputs::plan_with_slots`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of
    /// `ctx_max` positions each: every layer's cache counted `slots` times on
    /// its card and every ring shadow `slots` times on the host
    /// ([`SeqTerms::slots_of`] over [`PlanInputs::seq_terms`]), refused as
    /// [`PlanInputs::plan`] refuses. A sequence holds nothing beside its
    /// stores, so the card's kv class is `slots` sequences' stores.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots_nodes(machine, ctx_max, levers, slots, None)
    }

    /// Lower explicit node-owned expert ranges through the common placement owner.
    pub fn plan_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        owned: &[u64],
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots_nodes(machine, ctx_max, levers, NonZeroUsize::MIN, Some(owned))
    }

    fn plan_slots_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
        owned: Option<&[u64]>,
    ) -> Result<Plan<'a>, PlaceError> {
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let rule = routed_rule(Program::Deepseek41Chain);
        let tiered = machine.expert_tiers().next().is_some();
        let fronts = machine
            .cards
            .iter()
            .map(|c| {
                Ok::<_, PlacementError>(
                    self.prompt_reserve_bytes(c, ctx_max, tiered)?
                        + pass_heads_bytes(&self.model, c, slots.get())?,
                )
            })
            .collect::<Result<Vec<u64>, _>>()?;
        let reserve = fronts.iter().copied().max().unwrap_or(0);
        let plan = placement::plan_in_pool(machine, levers, |levers| {
            let mut plan = placement::nodes::loaded_routed(
                &self.model,
                machine,
                ctx_max,
                &kv,
                levers,
                rule,
                reserve,
                owned,
            )?;
            // Each stage card's prompt batch is reserved out of its expert
            // budget above and is a term of its count, so the first prompt's
            // reading holds it.
            for (t, &front) in plan.cards.iter_mut().zip(&fronts) {
                t.grow_prompt(front);
            }
            // The selected rows' pages that sit on the host come out of the host room the
            // plan was made at, before the NVMe tier's split reads it.
            let host =
                self.kv_host_bytes(ctx_max, slots.get() as u64, levers.kv_host, plan.kv_floor)?;
            if host > 0 {
                plan.kv_host_planes(host)?;
            }
            let room = plan.plan_at(&self.room);
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            Ok(plan)
        })?;
        // The batches stay beside the plan's terms (`refuse_reserve` holds each
        // to its card), so a unified pool counts them beside them.
        let plan = checked_beside(plan, fronts.iter().sum()).map_err(PlaceError::Broken)?;
        for ((card, t), &front) in machine.cards.iter().zip(&plan.cards).zip(&fronts) {
            let limit = plan.usable_bytes(card).saturating_sub(card.margin_bytes);
            place_hybrid::refuse_reserve(
                card,
                t.bytes_beside_prompt(),
                limit,
                front,
                |card, batch, left, short| PlaceError::BatchOver {
                    card,
                    batch,
                    left,
                    short,
                },
            )?;
        }
        Ok(plan)
    }

    /// The card bytes the plan counts on stage card `card` for a load of
    /// the file at `ctx_max` positions, with an expert tier or not
    /// (`tiered`): its prompt batch's parts over the card's layers at the
    /// plan's levers ([`prompt_parts`], [`BatchShape::planned`]), rounded
    /// per buffer as the card's allocator takes them at its granule
    /// ([`Parts::card_bytes`](crate::placement::parts::Parts::card_bytes)).
    /// The plan reserves it out of the card's expert budget and counts it as
    /// the card's prompt term
    /// ([`CardTotals::grow_prompt`](crate::placement::CardTotals::grow_prompt));
    /// the card body holds the buffers it makes to the same parts.
    pub fn prompt_reserve_bytes(
        &self,
        card: &Card,
        ctx_max: u64,
        tiered: bool,
    ) -> Result<u64, PlacementError> {
        let ctx = usize::try_from(ctx_max).map_err(|_| PlacementError::Metadata {
            key: "ctx_max".to_owned(),
            detail: format!("{ctx_max} positions pass usize"),
        })?;
        let shape = BatchShape::planned(&self.hp, card.layers.clone(), ctx, tiered);
        Ok(prompt_parts(&shape)?.card_bytes(card.granule_bytes))
    }

    /// The host bytes the selected rows ([`Class::Host`] buffers of the layout's rooms) take
    /// over `slots` sequences under the lever word `host`, each run rounded to whole pages of
    /// the host-capable room's page ([`PlanInputs::kv_host_page`]): none under `off`; every
    /// row at the cap under `all`; under `spill` the pages past the floor the plan spreads
    /// its experts at (`floor`, none when the cache is planned whole), which a session that
    /// grows to the cap may spill, so no growth meets a host refusal mid-session.
    pub fn kv_host_bytes(
        &self,
        ctx_max: u64,
        slots: u64,
        host: KvHost,
        floor: Option<u64>,
    ) -> Result<u64, PlacementError> {
        if host == KvHost::Off {
            return Ok(0);
        }
        let page = self.kv_host_page.ok_or_else(|| PlacementError::Metadata {
            key: "BLOOMERY_KV_HOST".to_owned(),
            detail: "the page a host-capable room takes was not read; a binary that plans under \
                     the lever reads it once (gpu's host_page) and sets PlanInputs::kv_host_page"
                .to_owned(),
        })?;
        let rooms = self.kv.rooms().ok_or_else(|| PlacementError::Metadata {
            key: "the stores' rooms".to_owned(),
            detail: "this layout declares none".to_owned(),
        })?;
        let on_host = |class: Class| (class == Class::Host).then_some(page);
        let past = || PlacementError::Metadata {
            key: "the K/V rows on the host".to_owned(),
            detail: "the selected rows' bytes over the resident slots pass u64".to_owned(),
        };
        let paged = |at: u64| -> Result<u64, PlacementError> {
            (0..rooms.layers())
                .try_fold(0u64, |sum, l| sum.checked_add(rooms.paged(l, at, on_host)))
                .ok_or_else(past)
        };
        let held = match (host, floor) {
            (KvHost::Off, _) | (KvHost::Spill, None) => 0,
            (KvHost::Spill, Some(floor)) => paged(ctx_max)?.saturating_sub(paged(floor)?),
            (KvHost::All, _) => paged(ctx_max)?,
        };
        held.checked_mul(slots).ok_or_else(past)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the file's layers ([`KvLayout`]: each
    /// layer's window ring, compressed rows, index keys and compressor
    /// state), no draft (the DSpark draft's window is its own card's), and
    /// nothing beside them — the body's sequence is its layers' caches alone.
    /// Its ring shadows are host bytes, which [`SeqTerms::slots_of`] counts
    /// on the host ([`crate::placement::KvBytes::shadow_bytes`]).
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.model.layers,
            },
            draft: None,
            beside: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use gguf::GgmlType;

    use super::{PASS_ROWS, pass_heads_bytes};
    use crate::placement::{Card, ModelTensor, ModelTensors, PlacementError, Role};

    fn card(head: bool) -> Card {
        Card {
            name: "card".to_string(),
            tier_leg: None,
            device: None,
            usable_bytes: 1 << 40,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: NonZeroU64::new(2 << 20).expect("2 MiB"),
            free_bytes: None,
            held_by: None,
            layers: 0..1,
            head,
            token_embedding: false,
            reserve_only: false,
            reserves: Vec::new(),
        }
    }

    fn model(head: bool) -> ModelTensors {
        let tensors = head
            .then(|| ModelTensor {
                name: "output.weight".to_string(),
                shard: 0,
                layer: None,
                role: Role::Head,
                ty: GgmlType::Q8_0,
                dims: vec![4096, 129_280],
                file_bytes: 4096 / 32 * 34 * 129_280,
                gathered_rows: None,
            })
            .into_iter()
            .collect();
        ModelTensors {
            tensors,
            layers: 1,
            experts: 0,
            experts_used: 0,
            bf16_words: |_| false,
        }
    }

    /// The passes' heads a plan adds to the head card's prompt term: a load of two or more
    /// slots runs the pair pass of a row a slot, whose row-2 head of 129,280 rows over 4,096
    /// values under the RMS norm (549,900 B: input and normed rows, logits, readback words) the
    /// card's allocator packs into one shared 2 MiB granule. One slot adds none and another
    /// card none, so a one-slot plan's fronts, and with them its experts and every term, are
    /// the plan's before the heads; a file with no head reads none there, and two slots of it
    /// are refused by name. FAIL-first: the plan counted no row head past the load's own.
    #[test]
    fn two_slots_count_the_pair_passes_row_head() {
        assert_eq!(PASS_ROWS, 2);
        let (with, headless) = (model(true), model(false));
        let (head_card, other) = (card(true), card(false));
        let heads =
            |m: &ModelTensors, c: &Card, slots| pass_heads_bytes(m, c, slots).expect("the heads");
        assert_eq!(heads(&with, &head_card, 2), 2 << 20);
        assert_eq!(heads(&with, &head_card, 4), 2 << 20);
        assert_eq!(heads(&with, &head_card, 1), 0);
        assert_eq!(heads(&with, &other, 2), 0);
        assert_eq!(heads(&headless, &head_card, 1), 0);
        assert!(matches!(
            pass_heads_bytes(&headless, &head_card, 2),
            Err(PlacementError::PassHeads)
        ));
        let row = crate::placement::heads::model_row_heads(
            &with,
            true,
            crate::placement::heads::HeadLayout::PerRow,
            1..=2,
        )
        .expect("a head");
        assert_eq!(row.bytes(), 549_900);
    }
}
