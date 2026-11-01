//! A placed load of qwen3moe (Qwen3-30B-A3B), qwen35moe
//! (Qwen3.6-35B-A3B), or dense qwen35, as `generate_qwen3moe` and the qwen3 serve seat
//! open one under `--place`: the placement word's cards
//! (`generate::Place`; one expert tier is planned, and more than one is
//! refused by name), the machine [`laid`] lays out — one card as
//! `workstation::placed_with_arena`, a stage and one tier as
//! `workstation::plan_tiers` with this model's tier batch and the stage held
//! as the placed load holds it — with the scratch
//! the body's placed load holds past the m = 1 scratch, the plan by the
//! expert rule over the stacks `routed_rule` loads — the trunk on the card,
//! each layer's routed id prefix as the budget holds, the rest on the host
//! tier — and the `plan` record of it. A plan that holds no host expert
//! opens the whole model on the plan's card, as the unplaced load does, when
//! the whole-fit verdict takes the load ([`PlaceQ3::whole_card_qwen3`]); any
//! other opens the body's placed load, which holds no host expert as it
//! holds some ([`open_qwen3`]).
//!
//! The same two binaries' `--place`-unset default lives here too
//! ([`unplaced_qwen3`]), on `a`'s card, the largest visible card
//! ([`PlaceQ3::unplaced_on`]): the whole model while that fits what the card
//! had — the whole-fit verdict (`placement::whole_need`) over what the whole
//! load asks for: its weights' granules, its cache planes' granules for
//! every resident slot the load serves ([`kv_bytes`]), and the program's own
//! arena and the reserve its load keeps free past it — and when it does
//! not, the placed plan on the same card. The serve seat chooses its
//! context and load through [`serve_choice`]. Its resident slots open
//! through [`open_qwen3_slots`] and [`open_qwen35_slots`]; [`Seqs`]
//! counts their stores and checkpoints. [`census_reads`] counts census
//! readings; [`serve_choice`] returns each choice's probe count.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};

use bloomery_gpu::arch::qwen3moe::{
    Body, Body35, Open35, OpenOpts, Qwen35moeModel, checkpoint_price35,
};
use bloomery_gpu::host::SERVED_TIERS;
use bloomery_gpu::kv_format::KvQ8;
use bloomery_gpu::linear::{KHeadMap, LinearShape};
use bloomery_gpu::model::{GpuModel, Slots};
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::generate::{Place, PlaceWhy, SEAT_STEPS, TierRule, choose_ctx, short_in};
use bloomery_gpu_gates::image_plan;
use bloomery_gpu_gates::kv_planes::layer_planes;
use bloomery_gpu_gates::qwen_seat::{Family, decode_mma};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{checkpoints_pinned, reserve_checkpoints};
use bloomery_gpu_gates::ubatch_pick::{self, Ask, Rule, Step};
use bloomery_levers::{HostCfg, Levers};
use gguf::Split;
use model::arch::models::{Ffn, Mixer, ModelSpec};
use model::arch::qwen3moe::place as q3;
use model::arch::qwen35moe::place as q35;
use model::placement::accepts::{Program, routed_rule};
use model::placement::ctx_ask::{Chosen, Probe};
use model::placement::workstation::{self, CardSpec, DeviceInfo};
use model::placement::{
    self, Device, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Role,
    WholeLoad, WholeNeed,
};

#[allow(
    clippy::duplicate_mod,
    reason = "three seats of one bin include this file"
)]
#[path = "ubatch_rule.rs"]
mod ubatch_rule;
use ubatch_rule::ubatch_rule;

/// The census readings this module took since the process began: a caller
/// reads them around a choice of context ([`census_reads`]), so a choice
/// that reads the census per probe shows in its counts.
static CENSUSES: AtomicUsize = AtomicUsize::new(0);

/// A choice of context's probes and the census readings it took
/// ([`serve_choice`]).
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reads {
    pub censuses: usize,
    pub probes: usize,
}

/// The census readings so far.
#[allow(
    dead_code,
    reason = "the serve choice and the routed legs read it; the CLI and the e2e gates include the planner without it"
)]
pub fn census_reads() -> usize {
    CENSUSES.load(Ordering::Relaxed)
}

/// This process's census ([`bloomery_gpu_gates::gpu_census::census`]),
/// counted: a choice that probes several contexts reads it once and hands
/// the reading to each probe.
pub fn census() -> Result<Vec<DeviceInfo>, GateError> {
    CENSUSES.fetch_add(1, Ordering::Relaxed);
    bloomery_gpu_gates::gpu_census::census()
}

/// The levers a placed load acts on: the plan's card budget and the host
/// set's load (`Levers::host`). A binary lists them as acted on only when
/// `--place` is given, so one set beside an unplaced load is refused by
/// name.
pub const PLACED_LEVERS: [&str; 4] = [
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
];

/// The tier cards this body serves, and no break-even: no sitting has shown
/// the tier not slower, so two cards run `a` and the plan is not asked.
const TIER_RULE: TierRule = TierRule {
    tiers: SERVED_TIERS,
    break_even: None,
    basis: "",
};

/// The placement a qwen3moe or qwen35moe binary's `--place` names, and why,
/// on `census` (`Place::choose` by [`TIER_RULE`]): `flag` as given when set,
/// else `a`, the plan never asked. An unset load still takes
/// [`PlaceQ3::unplaced_on`]'s pick, the same card; the caller prints
/// [`PlaceWhy::record`].
#[allow(
    dead_code,
    reason = "the CLI and the qwen3 seat choose by the rule; the e2e gates include the planner without it"
)]
pub fn choose(flag: Option<Place>, census: &[DeviceInfo]) -> Result<PlaceWhy, GateError> {
    Place::choose(flag, census, TIER_RULE, |offer| {
        Err(format!(
            "--place unset: no sitting has shown the tier not slower, so the plan is not asked for {}",
            offer.name()
        )
        .into())
    })
}

/// The plan inputs of one of the two architectures; the qwen35moe ones,
/// which carry the typed model description too, boxed.
enum Inputs {
    Qwen3(q3::PlanInputs),
    Qwen35(Box<q35::PlanInputs>),
}

/// The plan of `inputs` over `machine` at `ctx` positions (a total) under
/// `levers`, every refusal by name with its terms.
fn plan_inputs<'a>(
    inputs: &'a Inputs,
    machine: &'a Machine,
    ctx: usize,
    levers: &PlanLevers,
) -> Result<Plan<'a>, GateError> {
    let ctx = u64::try_from(ctx)?;
    Ok(match inputs {
        Inputs::Qwen3(i) => i.plan(machine, ctx, levers)?,
        Inputs::Qwen35(i) if attention_only(&i.spec) => {
            return Err(
                "a trunk of attention layers alone has only the whole load on a card: it has no \
                 routed expert to place, so no plan"
                    .into(),
            );
        }
        Inputs::Qwen35(i) => i.plan_rule(
            machine,
            (ctx, ctx.min(q35::UBATCH_PLANNED)),
            levers,
            routed_rule(Program::Qwen35Body),
        )?,
    })
}

/// What a placed load plans from: the placement on this process's devices,
/// the file's plan inputs, the sequences the load holds and the machine its
/// stage card lays out.
pub struct PlaceQ3 {
    place: Place,
    arch: &'static str,
    inputs: Inputs,
    /// The cache format this load counts in — the same choice `inputs.kv`
    /// carries, kept beside it for the plane list [`kv_bytes`] splits.
    kv: KvQ8,
    /// The delta lanes this load's whole-fit verdict prices a qwen35 family
    /// file's delta layers at ([`delta_planes`]): one on a plain load, the
    /// load's verify lanes on a load that verifies drafted rows
    /// (`GpuModel::open_verifying`, `GpuModel::open_mtp`) — the serve seat's,
    /// from its draft word. A qwen3moe file has no delta layer, and the
    /// CLI's and the gates' loads are plain.
    lanes: usize,
    /// The sequences the load holds, whose rows every machine of it carries
    /// ([`laid`]).
    seqs: Seqs,
    /// What a media load adds to every machine and verdict of this load.
    media: Media,
    machine: Machine,
}

/// The sequences a load holds and the price of the checkpoints they take
/// (the lib's declaration, `residency38::Seqs`): `slots` resident
/// sequences, the context split across them, each taking checkpoints when
/// `checkpoints` carries a price (a qwen35moe file's seat load: its prompt
/// calls take the points a kept prefix comes back to), reserved at the
/// context a slot is laid for ([`laid`]).
pub use bloomery_gpu_gates::residency38::Seqs;

/// A Body35 file's `slots` resident sequences, each taking checkpoints
/// priced by the body's own derivation (`checkpoint_price35`) at the ubatch
/// `ubatch` the load is asked for, which sets their spacing: the seat's. A
/// trunk with no recurrent store has no point to copy, so its sequences take
/// none.
#[allow(
    dead_code,
    reason = "the serve seat's qwen35moe loads take checkpoints; the CLI and the e2e gates take none"
)]
pub fn seqs_qwen35(file: &Split, ubatch: usize, slots: usize) -> Result<Seqs, GateError> {
    let spec = model::arch::spec(file)?.spec;
    let price = checkpoint_price35(&spec, ubatch)?;
    Ok(if price.point_bytes == 0 {
        Seqs::none(slots)
    } else {
        Seqs::priced(slots, price)
    })
}

/// The card reserve of a load's resident sequences past the live one
/// ([`laid`]): each one's stores that no position moves.
pub const SLOTS_RESERVE: &str = "resident slots";

/// The card reserve of a media load's rotation tables, one for every resident sequence ([`laid`]):
/// a placed plan's KV term counts a text load's bytes a position, not the table each sequence's
/// image positions rewrite (`gpu::media::SeqRope`).
pub use bloomery_gpu_gates::image_plan::TABLES_RESERVE;

/// The layers the model loads: a qwen35moe file's trunk (its next-token
/// layers are never loaded).
fn layers_of(inputs: &Inputs) -> usize {
    match inputs {
        Inputs::Qwen3(i) => i.hp.n_layer,
        Inputs::Qwen35(i) => i.spec.layers.len(),
    }
}

/// The bytes one sequence of `inputs`'s file holds whatever its length: each
/// layer's cache bytes at no position, its plan's own [`KvBytes`] — a
/// qwen35moe file's delta layers' recurrent states and conv rings, nothing on
/// a qwen3moe file, whose cache is per-position.
fn fixed_seq_bytes(inputs: &Inputs) -> u64 {
    let kv: &dyn KvBytes = match inputs {
        Inputs::Qwen3(i) => &i.kv,
        Inputs::Qwen35(i) => &i.kv,
    };
    (0..layers_of(inputs)).map(|l| kv.layer_bytes(l, 0)).sum()
}

/// The machine a placed load of `inputs`'s file lays out on `specs`' cards
/// over `arena`: one card as [`workstation::placed_with_arena`], or a stage
/// and one expert tier as [`workstation::plan_tiers`] with this model's tier
/// batch and the stage held as the placed load holds it
/// ([`q3::stage_placed`]). Then the rows of what `seqs` holds past the
/// plan's one-sequence terms: on the stage card, a [`SLOTS_RESERVE`] of every
/// resident sequence's fixed stores ([`fixed_seq_bytes`]) but the live one's
/// — the plan's KV term at the total counts the per-position rows of every
/// slot's share and the fixed stores once — none where they are 0; on the
/// host, every sequence's checkpoints when they take them, through the lib's
/// one owner ([`reserve_checkpoints`]). The one owner of a placed
/// load's machine, so a relaid machine keeps the
/// rows. The checkpoints are priced at `slot_ctx` positions a slot, the
/// context the machine is laid for; on a media load each slot's rotation
/// table holds as many positions.
fn laid(
    specs: &[CardSpec],
    inputs: &Inputs,
    arena: u64,
    seqs: Seqs,
    slot_ctx: u64,
    media: Media,
) -> Result<Machine, GateError> {
    let layers = layers_of(inputs);
    let mut machine = match specs {
        [stage] => workstation::placed_with_arena(*stage, layers, arena),
        [stage, tier] => {
            let batch = match inputs {
                Inputs::Qwen3(i) => q3::tier_batch(i),
                Inputs::Qwen35(i) => q35::leg_tier_batch(i),
            }?;
            let mut machine = workstation::plan_tiers(layers, *stage, &[*tier], None, batch)?;
            q3::stage_placed(&mut machine, arena);
            machine
        }
        other => {
            return Err(format!(
                "a qwen plan runs one stage card and at most one expert tier; {} cards were named",
                other.len()
            )
            .into());
        }
    };
    let past = u64::try_from(seqs.slots.saturating_sub(1))
        .map_or(u64::MAX, |n| n.saturating_mul(fixed_seq_bytes(inputs)));
    if past > 0 {
        machine.cards[0]
            .reserves
            .push((SLOTS_RESERVE.to_owned(), past));
    }
    if let Some(width) = media.tables {
        let positions = u64::try_from(seqs.slots).map_or(u64::MAX, |n| n.saturating_mul(slot_ctx));
        machine.cards[0].reserves.push((
            TABLES_RESERVE.to_owned(),
            positions.saturating_mul(image_plan::table_bytes_per_position(width)),
        ));
    }
    reserve_checkpoints(&mut machine, seqs, slot_ctx)?;
    beside::reserve_on(&mut machine, media.beside);
    Ok(machine)
}

/// The rows each of `slots` slots loads with when they split `ctx`
/// positions: `ctx / slots`, the split's floor.
fn slot_ctx(ctx: usize, slots: usize) -> usize {
    ctx / slots.max(1)
}

/// What a load adds to its machines and verdicts beside its own rows ([`laid`], [`whole_on`]):
/// the reserves of the models the process serves beside it that sit on the stage card
/// (`vision_seat::EncoderSeat::reserve`, entered by [`beside::reserve_on`]), and the width in
/// f32 of the rotation table every sequence holds on a media load (`Open35::media`;
/// [`image_plan::table_bytes_per_position`] a position). A text load with nothing beside it adds
/// neither.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Media {
    beside: Beside,
    tables: Option<usize>,
}

/// `o` for the load each of `slots` slots opens when they split `ctx`
/// positions: a slot's share ([`slot_ctx`]).
#[must_use]
pub fn share(o: &Open35, ctx: usize, slots: usize) -> Open35 {
    Open35 {
        ctx: slot_ctx(ctx, slots),
        ..*o
    }
}

impl PlaceQ3 {
    /// `place` resolved against `census`. More than one expert tier card is
    /// refused by name: `arch`'s body serves [`SERVED_TIERS`].
    fn resolve(place: Place, census: &[DeviceInfo], arch: &str) -> Result<Place, GateError> {
        let place = place.on(census)?;
        place.serves(arch, SERVED_TIERS)?;
        Ok(place)
    }

    /// A qwen3moe file's placed load at `ctx` positions on `place`'s card,
    /// its cache planes in the `kv` format ([`KvQ8`]), resolved against one
    /// census of this process's devices.
    #[allow(
        dead_code,
        reason = "the e2e gates' load; the serve seat and the routed legs plan on one census (qwen3_on)"
    )]
    pub fn qwen3(file: &Split, place: Place, ctx: usize, kv: KvQ8) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen3_on(file, place, &census()?, ctx, kv, Beside::default())
    }

    /// [`PlaceQ3::qwen3`] resolved against `census`, its machine carrying `beside`'s reserves.
    pub fn qwen3_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        ctx: usize,
        kv: KvQ8,
        beside: Beside,
    ) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census, "qwen3moe")?;
        let mut inputs = q3::PlanInputs::read(file)?;
        if kv == KvQ8::Q8 {
            inputs.kv = inputs.kv.in_q8();
        }
        let inputs = Inputs::Qwen3(inputs);
        let arena = Qwen3moeModel::placed_arena_bytes(file, ctx)?;
        let media = Media {
            beside,
            tables: None,
        };
        let machine = laid(
            &place.card_specs()?,
            &inputs,
            arena,
            Seqs::ONE,
            u64::try_from(ctx)?,
            media,
        )?;
        Ok(PlaceQ3 {
            place,
            arch: "qwen3moe",
            inputs,
            kv,
            lanes: 1,
            seqs: Seqs::ONE,
            media,
            machine,
        })
    }

    /// A qwen35moe file's placed load under `o` on `place`'s card, its
    /// attention planes in `o.kv`'s format, resolved against one census of
    /// this process's devices: one sequence taking no checkpoints, the CLI's.
    #[allow(
        dead_code,
        reason = "the CLI's and the e2e gates' load; the serve seat's holds its slots (qwen35_on)"
    )]
    pub fn qwen35(file: &Split, place: Place, o: Open35) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen35_on(file, place, &census()?, o, Seqs::ONE, Beside::default())
    }

    /// [`PlaceQ3::qwen35`] of a load that holds `seqs`, each of its slots
    /// opened under `o` (a slot's share of the total, [`share`]), its machine
    /// carrying their rows ([`laid`]) and, on a media load (`o.media`), the
    /// tables of its sequences, and `beside`'s reserves (the image encoder's
    /// bytes when the tower sits on this card), resolved against `census`: the serve
    /// seat's, and one sequence's under [`PlaceQ3::qwen35`].
    pub fn qwen35_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        o: Open35,
        seqs: Seqs,
        beside: Beside,
    ) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen35_lanes(file, place, census, o, seqs, beside, 1)
    }

    /// [`PlaceQ3::qwen35_on`] with the delta lanes (`lanes`, at least one)
    /// its whole-fit verdict prices a delta layer at ([`delta_planes`]): the
    /// serve seat's, which opens loads that verify drafted rows.
    fn qwen35_lanes(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        o: Open35,
        seqs: Seqs,
        beside: Beside,
        lanes: usize,
    ) -> Result<PlaceQ3, GateError> {
        let mut inputs = q35::PlanInputs::describe(file)?;
        let arch = Family::of_spec(&inputs.spec).map_or(Program::Qwen35Body.name(), Family::name);
        let place = PlaceQ3::resolve(place, census, arch)?;
        inputs.kv = inputs.kv.in_q8_if(o.kv == KvQ8::Q8);
        let tables = if o.media {
            let head = inputs.spec.gqa().ok_or(
                "the description holds no attention layer to rotate an image's positions by",
            )?;
            Some(head.rope.dims as usize)
        } else {
            None
        };
        let media = Media { beside, tables };
        let arena = if inputs.spec.routed().is_none() {
            match Qwen35moeModel::whole_load(file, o)? {
                WholeLoad::Checked { arena, .. } => arena,
                WholeLoad::Unchecked => 0,
            }
        } else {
            Qwen35moeModel::placed_arena_bytes(file, o)?
        };
        let inputs = Inputs::Qwen35(Box::new(inputs));
        let machine = laid(
            &place.card_specs()?,
            &inputs,
            arena,
            seqs,
            u64::try_from(o.ctx)?,
            media,
        )?;
        Ok(PlaceQ3 {
            place,
            arch,
            inputs,
            kv: o.kv,
            lanes,
            seqs,
            media,
            machine,
        })
    }

    /// The load as one whose prompt calls stream host experts: its machine reserves
    /// the lane's staging on the host (`workstation::reserve_stream_staging`) at a qwen35moe
    /// file's ring slot ([`q35::PlanInputs::stream_slot`]); a qwen3moe file streams nothing.
    #[allow(
        dead_code,
        reason = "the e2e gate's residency loads; the CLI and the seat stream none"
    )]
    pub fn streaming(mut self) -> Result<PlaceQ3, GateError> {
        let slot = match &self.inputs {
            Inputs::Qwen35(inputs) => inputs.stream_slot(q35::Experts::Card)?,
            Inputs::Qwen3(_) => 0,
        };
        workstation::reserve_stream_staging(&mut self.machine, slot);
        Ok(self)
    }

    /// The plan at `ctx` positions under `levers` (the card budget), every
    /// refusal by name with its terms.
    pub fn plan(&self, ctx: usize, levers: &PlanLevers) -> Result<Plan<'_>, GateError> {
        plan_inputs(&self.inputs, &self.machine, ctx, levers)
    }

    /// The smallest card budget, in bytes, under which the plan at `ctx`
    /// positions keeps at least `want` routed experts on the card, searched up
    /// to `ceiling`, which must itself plan that many. The expert rule fills
    /// the layers in ascending order, one expert at a time, so a plan of
    /// fewer experts than the file has layers leaves the later layers none.
    /// Card experts only grow with the budget and the planner is host
    /// arithmetic, so the search is a bisection from a budget the planner
    /// refuses as below the card's floor; a refusal other than the floor's
    /// ends the search by name.
    #[allow(
        dead_code,
        reason = "the e2e gate's reference plan; the CLI and the seat plan at no budget"
    )]
    pub fn budget_for_card_experts(
        &self,
        ctx: usize,
        want: u64,
        ceiling: u64,
    ) -> Result<u64, GateError> {
        let card_experts = |budget: u64| -> Result<Option<u64>, GateError> {
            let levers = PlanLevers {
                card_budget_bytes: Some(budget).into(),
            };
            match self.plan(ctx, &levers) {
                Ok(plan) => Ok(Some(plan.cards.first().ok_or("a plan of no card")?.experts)),
                Err(e)
                    if e.to_string()
                        .contains(&format!("budget {budget} B is below its floor ")) =>
                {
                    Ok(None)
                }
                Err(e) => Err(e),
            }
        };
        let (mut lo, mut hi) = (0u64, ceiling);
        if card_experts(lo)?.is_some() {
            return Err(
                format!("a card budget of {lo} B plans: the card's floor is not read").into(),
            );
        }
        match card_experts(hi)? {
            Some(top) if top >= want => {}
            Some(top) => {
                return Err(format!(
                    "the search's upper bound, a card budget of {hi} B, plans {top} card \
                     experts, under the {want} the caller needs"
                )
                .into());
            }
            None => {
                return Err(format!(
                    "the search's upper bound, a card budget of {hi} B, is below the card's floor"
                )
                .into());
            }
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if card_experts(mid)?.is_some_and(|n| n >= want) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(hi)
    }

    /// The layers the model loads ([`layers_of`]).
    fn layers(&self) -> usize {
        layers_of(&self.inputs)
    }

    /// The whole-fit verdict of this file on the card's device at `ctx`
    /// positions for `slots` resident sequences ([`whole_on`]), each taking
    /// checkpoints when this load's do, the cache in the planes' format this
    /// load counts, `load` what the program holds past it or its refusal.
    fn whole_at(
        &self,
        ctx: u64,
        slots: usize,
        load: Result<WholeLoad, String>,
    ) -> Result<Whole, GateError> {
        let layers = 0..self.layers();
        let card = self.place.card_specs()?[0];
        let kv = kv_bytes(
            &layers,
            &self.inputs,
            (self.kv, self.media),
            (self.lanes, card.granule_bytes),
            slots,
            ctx,
        );
        whole_on(
            card,
            inputs_model(&self.inputs),
            layers.end,
            kv,
            (
                Seqs { slots, ..self.seqs },
                u64::try_from(slot_ctx(usize::try_from(ctx)?, slots))?,
                self.media.beside,
            ),
            load,
        )
    }

    /// The placement's word, as the `plan` record prints it.
    #[allow(
        dead_code,
        reason = "the serve seat names its load's card in a crash report; the CLI prints the plan record"
    )]
    pub fn name(&self) -> &'static str {
        self.place.name()
    }

    /// The `plan` record of `plan`: the card, the architecture, the expert
    /// rule, the routed experts on the card and the host, the fewest and
    /// the most one layer keeps on the card, the expert tier card and the
    /// experts it holds when the plan has one, the card's free bytes when the
    /// census read them, the host room the plan was made at when it had
    /// one, and its device. `why` names what made a run with `--place`
    /// unset take this placed plan.
    pub fn records<'a>(
        &self,
        plan: &Plan<'a>,
        why: Option<&str>,
    ) -> Result<impl Iterator<Item = Record> + 'a, GateError> {
        record::require(
            record::groups::qwen3moe_place::PLAN,
            "qwen3moe_place::PlaceQ3::records",
        );
        let card = &plan.cards[0];
        let r = Record::new(&record::PLAN38)
            .w("place", self.place.name())
            .w("card", &self.machine.cards[0].name)
            .w("arch", self.arch)
            .w("experts", "card")
            .u("ctx_max", plan.ctx_max)
            .u("host_experts", plan.host.experts)
            .u("card_experts", card.experts)
            .w(
                "n_l",
                format!(
                    "{}-{}",
                    plan.n_l.iter().min().copied().unwrap_or(0),
                    plan.n_l.iter().max().copied().unwrap_or(0)
                ),
            );
        let r = match (self.machine.tiers.first(), plan.tier_n_l.first()) {
            (Some(t), Some(n)) => r
                .w("tier", &t.name)
                .u("tier_experts", n.iter().sum::<u64>()),
            _ => r,
        };
        let r = match self.machine.cards.first().and_then(|c| c.free_bytes) {
            Some(free) => r.u("card_free", free),
            None => r,
        };
        let r = record::with_room(r, plan.host.room.as_ref());
        let r = match why {
            Some(w) => r.w("why", w),
            None => r,
        };
        // A qwen35moe-family file's PLE table: its tier and the room's
        // reading that chose it; a plan with none, and a qwen3moe file,
        // print neither. A tier no reader takes prints as refused.
        let r = match (&self.inputs, plan.row_tier()) {
            (Inputs::Qwen35(i), Ok(Some(Device::Nvme))) => {
                r.w("rows", "nvme").w("read", i.room.1.word())
            }
            (Inputs::Qwen35(i), Ok(Some(_))) => r.w("rows", "host").w("read", i.room.1.word()),
            (Inputs::Qwen35(_), Err(_)) => r.w("rows", "refused"),
            (Inputs::Qwen35(_), Ok(None)) | (Inputs::Qwen3(_), _) => r,
        };
        let r = r
            .csv("devices", record::plan_devices(plan.machine))
            .w("cuda_order", record::cuda_order());
        let r = record::with_requantized(r, plan.model);
        let program = match self.inputs {
            Inputs::Qwen3(_) => Program::Qwen3moeBody,
            Inputs::Qwen35(_) => Program::Qwen35Body,
        };
        let need = bloomery_gpu_gates::need::staged(plan, 0, 0)?;
        Ok(std::iter::once(r)
            .chain(std::iter::once(need))
            .chain(record::host_only(plan.model, Some(routed_rule(program)))?))
    }
}

/// The allocator's bytes of the cache the body of `inputs`'s model allocates
/// for `slots` resident sequences of a load asked for `ctx` positions: the
/// seat splits the total across its slots, each `ctx / slots` rows (never
/// past it: a cache row is the granularity, the floor exact), and every
/// slot, layer by layer, holds the planes this function lists through the
/// card's allocation granule ([`placement::allocator_bytes`]) — the engine allocates
/// each plane its own buffer per layer (`KvPlanes::new`, mirrored by
/// [`layer_planes`], and a delta layer's `RecStore`, mirrored by
/// [`delta_planes`] at the load's `lanes`) and each added slot repeats the set
/// ([`Slots::new_seq`]), so the verdict counts the granules those
/// allocations round to, not the raw sum a `KvBytes` layer gives. One slot
/// is the one-sequence load, plus that rounding. `slots` is at least one;
/// the raw plane bytes of the list sum to what `layer_bytes` gives over the
/// same layers.
fn kv_bytes(
    layers: &std::ops::Range<usize>,
    inputs: &Inputs,
    (kv, media): (KvQ8, Media),
    (lanes, granule): (usize, NonZeroU64),
    slots: usize,
    ctx: u64,
) -> u64 {
    let rows = ctx / slots as u64;
    let mut planes = Vec::new();
    for _ in 0..slots {
        for l in layers.clone() {
            match inputs {
                Inputs::Qwen3(i) => {
                    layer_planes(
                        &mut planes,
                        i.hp.n_head_kv,
                        i.hp.head_dim,
                        kv == KvQ8::Q8,
                        rows,
                    );
                }
                Inputs::Qwen35(i) => match i.spec.layers.get(l).map(|layer| &layer.mixer) {
                    Some(Mixer::Gqa(g)) => {
                        layer_planes(
                            &mut planes,
                            g.kv_heads as usize,
                            g.head_dim as usize,
                            kv == KvQ8::Q8,
                            rows,
                        );
                    }
                    // The shape of the file's own delta layers; the head map
                    // enters none of its lengths, and a qwen3.6 delta layer is
                    // the tiled one (`delta_shape` of the gpu side's plan).
                    Some(Mixer::DeltaRule(d)) => {
                        delta_planes(
                            &mut planes,
                            LinearShape {
                                n_k: d.k_heads as usize,
                                n_v: d.v_heads as usize,
                                map: KHeadMap::Tiled,
                            },
                            lanes,
                        );
                    }
                    Some(other) => panic!(
                        "kv_bytes: layer {l} mixes with {}, which the plan inputs refuse",
                        other.what()
                    ),
                    None => {}
                },
            }
        }
        // A media load's sequence holds its rotation table after its stores,
        // one buffer of its own (`SeqRope::new`).
        if let Some(width) = media.tables {
            planes.push(rows * image_plan::table_bytes_per_position(width));
        }
    }
    placement::allocator_bytes(granule, planes)
}

/// The plane bytes a delta layer of `shape` holds at `lanes` lanes onto
/// `planes`: the two buffers `RecStore::new` zeroes — the state, a lane, and
/// the conv ring once (`LinearShape::lens`) — and, on a load that keeps a
/// lane a verify row, the `lanes` u32 stamps its stamped store holds
/// (`RecStore::stamped`); the engine's layout prices the same count
/// (`KvLayout::of_spec_lanes`, `runtime::stores`'s terms).
///
/// # Panics
///
/// On no lane, by name.
fn delta_planes(planes: &mut Vec<u64>, shape: LinearShape, lanes: usize) {
    assert!(lanes >= 1, "delta_planes: a state of at least one lane");
    planes.push(4 * lanes as u64 * shape.state_len() as u64);
    planes.push(4 * shape.ring_len() as u64);
    if lanes > 1 {
        planes.push(4 * lanes as u64);
    }
}

/// The model of `inputs`'s file as it holds it.
fn inputs_model(inputs: &Inputs) -> &ModelTensors {
    match inputs {
        Inputs::Qwen3(i) => &i.model,
        Inputs::Qwen35(i) => &i.model,
    }
}

/// A qwen3moe file's whole load (`Qwen3moeModel::whole_load`): the load's
/// own per-layer routed-type check, read from the file's header, its refusal
/// by name; past it `WholeLoad::Unchecked` — the body allocates its ubatch
/// arena with no fit check, and the arena's bytes have no formula outside
/// the allocation, so the card keeps the expert rule's margin past the step
/// arenas in its place, as the placed machine does.
fn qwen3_whole(file: &Split) -> Result<WholeLoad, String> {
    Qwen3moeModel::whole_load(file).map_err(|e| e.to_string())
}

/// A qwen35moe file's whole load under `o`: its ubatch arena at `o.ubatch`
/// and the reserve its fit check keeps free past it
/// (`GpuModel::whole_load`); the load's own refusal of the file or of `o`,
/// by name.
fn qwen35_whole(file: &Split, o: Open35) -> Result<WholeLoad, String> {
    Qwen35moeModel::whole_load(file, o).map_err(|e| e.to_string())
}

/// The whole-fit verdict ([`whole_on`]).
enum Whole {
    /// The whole load fits what the card had.
    Fits(WholeNeed),
    /// It does not.
    Short(WholeNeed),
    /// The whole load refuses the file, the card and why: a tensor no card
    /// format loads, or the program's own read of the file.
    Refused(String),
}

impl Whole {
    /// Whether the whole load is the load.
    fn fits(&self) -> bool {
        matches!(self, Whole::Fits(_))
    }

    /// The need and what the card had, when the verdict has terms.
    fn terms(&self) -> Option<(u64, u64)> {
        match self {
            Whole::Fits(n) | Whole::Short(n) => Some((n.need, n.budget)),
            Whole::Refused(_) => None,
        }
    }

    /// The verdict at `ctx` positions in one line: the card, the answer,
    /// the need term by term and what the card had.
    fn line(&self, ctx: usize) -> String {
        match self {
            Whole::Fits(n) | Whole::Short(n) => {
                format!("whole fit at ctx {ctx}: {n}{}", beside::note(&n.card))
            }
            Whole::Refused(why) => format!("whole fit at ctx {ctx}: {why}"),
        }
    }

    /// The refusal of a whole load this verdict does not take, of a model
    /// with no routed expert to move to the host tier.
    /// Keep an unrepresentable reserve a named overflow, not a retryable shortage.
    fn dense_refusal(&self, ctx: usize) -> GateError {
        if let Whole::Short(n) = self {
            let Some(reserves) = n.card.reserve_bytes() else {
                return PlacementError::CardBytesPastU64 {
                    card: n.card.name.clone(),
                }
                .into();
            };
            let (margin, beside) = match n.load {
                WholeLoad::Checked { arena, reserve } => (reserve, arena),
                WholeLoad::Unchecked => (n.card.margin_bytes, 0),
            };
            return PlacementError::CardBudgetFloor {
                card: n.card.name.clone(),
                budget: n.budget,
                floor: n.need,
                dense: n.weights,
                kv: n.kv,
                context: n.card.context_bytes,
                scratch: n.card.scratch_bytes,
                reserves,
                margin,
                beside,
            }
            .into();
        }
        format!(
            "{}: a dense backbone has no routed experts to move to the host tier",
            self.line(ctx)
        )
        .into()
    }
}

/// The refusal of a ubatch flag on a load whose ubatch no verdict settles ([`Rule`]): `what`
/// takes its size from `BLOOMERY_QWEN3_UBATCH`, and the flag sizes the whole load of a file with no
/// routed expert.
fn refuse_flag(flag: Option<usize>, what: &str) -> Result<(), GateError> {
    match flag {
        Some(size) => Err(format!(
            "--ubatch-size {size}: {what} takes its ubatch from BLOOMERY_QWEN3_UBATCH, and the flag \
             sizes the whole load of a file with no routed expert"
        )
        .into()),
        None => Ok(()),
    }
}

/// A dense whole load's verdict at the ubatch `rule` settles on ([`ubatch_pick::settle`]), with what
/// the caller kept of that size's load. `verdict(u)` takes the whole-fit verdict at the size `u`; the
/// walk ends at the first that fits, at a refusal of the file (no size changes one) or at the last
/// size, whose verdict is then the one the caller refuses with.
fn settle<T>(
    rule: Rule,
    key: (usize, usize, usize),
    mut verdict: impl FnMut(usize) -> Result<(Whole, T), GateError>,
) -> Result<(usize, Whole, T), GateError> {
    let (u, step) = ubatch_pick::settle(rule, key, |u| {
        let (whole, kept) = verdict(u)?;
        Ok::<_, GateError>(Step {
            terms: whole.terms(),
            fits: whole.fits(),
            refused: matches!(whole, Whole::Refused(_)),
            kept: (whole, kept),
        })
    })?;
    let (whole, kept) = step.kept;
    Ok((u, whole, kept))
}

/// Whether no layer of `spec` has a routed FFN: the file's only load is the whole one.
fn dense_ffn(spec: &ModelSpec) -> bool {
    spec.layers
        .iter()
        .all(|l| matches!(l.ffn, Ffn::Dense { .. }))
}

/// The whole-fit verdict of a load of the whole of `model` — `layers`
/// layers, `kv` bytes of cache — on `spec`'s card: the card the plan lays
/// out for the program ([`workstation::placed_with_arena`], its step arenas' scratch), on a
/// unified machine with the checkpoints `seqs` pin at `slot_ctx` positions a
/// slot ([`reserve_checkpoints`], `Machine::whole_card`) and `beside`'s
/// reserves ([`beside::reserve_on`], the reserves a placed plan carries), and
/// what the program holds past it
/// (`load`, or
/// the program's refusal), sized by `placement::whole_need`, the verdict's
/// one owner. A tensor no card format loads is a refusal; any other planner
/// error is the error.
fn whole_on(
    spec: CardSpec,
    model: &ModelTensors,
    layers: usize,
    kv: u64,
    (seqs, slot_ctx, beside): (Seqs, u64, Beside),
    load: Result<WholeLoad, String>,
) -> Result<Whole, GateError> {
    let refused = |why: &dyn std::fmt::Display| {
        Whole::Refused(format!("the whole load on {}: refused: {why}", spec.name))
    };
    let load = match load {
        Ok(l) => l,
        Err(why) => return Ok(refused(&why)),
    };
    let mut machine = workstation::placed_with_arena(spec, layers, 0);
    reserve_checkpoints(&mut machine, seqs, slot_ctx)?;
    beside::reserve_on(&mut machine, beside);
    let card = machine
        .whole_card()
        .ok_or_else(|| format!("the whole load on {}: a machine with no card", spec.name))?;
    Ok(match placement::whole_need(model, &card, kv, load) {
        Ok(need) if need.fits() => Whole::Fits(need),
        Ok(need) => Whole::Short(need),
        Err(e @ PlacementError::NoCardFormat { .. }) => refused(&e),
        Err(e) => return Err(e.into()),
    })
}

impl PlaceQ3 {
    /// The placement a run with `--place` unset probes and loads on,
    /// resolved against `census`: `a` ([`Place::A`]), the largest visible
    /// card by usable bytes, ties to the lower ordinal — whatever the CUDA
    /// order puts at device 0. Its card takes the whole model when the
    /// whole-fit verdict on it does, else this file's placed plan, so both
    /// loads sit on one card. With no visible device, `a`'s refusal by name.
    pub fn unplaced_on(census: &[DeviceInfo]) -> Result<Place, GateError> {
        Place::A.on(census)
    }

    /// The load a run with `--place` unset takes by `whole`, this probe's
    /// verdict at `ctx`, its line on stderr; the probe is on
    /// [`PlaceQ3::unplaced_on`]'s placement. The whole load on the probe's
    /// card when the verdict takes it, else the probe itself, this file's
    /// placed plan on that card. A model with no routed expert that the
    /// verdict does not take is refused by name with its terms.
    fn unplaced(self, ctx: usize, whole: &Whole) -> Result<Unplaced, GateError> {
        eprintln!("{}", whole.line(ctx));
        if whole.fits() {
            return Ok(Unplaced::Whole(self.place.card_specs()?[0]));
        }
        let routed = inputs_model(&self.inputs)
            .tensors
            .iter()
            .any(|t| t.role == Role::RoutedExperts);
        if !routed {
            return Err(whole.dense_refusal(ctx));
        }
        Ok(Unplaced::Placed(Box::new(self)))
    }

    /// The card a run with `--place` set opens the whole model on, for the
    /// qwen3moe file `file` planned as `plan` at `ctx` positions for `slots`
    /// resident sequences: this placement's card when the plan holds no
    /// host expert and the whole-fit verdict at `ctx` takes the load
    /// ([`qwen3_whole`]); `None` when the plan opens placed. A plan that
    /// keeps every expert on the card but whose whole load does not fit
    /// opens placed ([`open_qwen3`]).
    pub fn whole_card_qwen3(
        &self,
        file: &Split,
        plan: &Plan<'_>,
        (ctx, slots): (usize, usize),
    ) -> Result<Option<CardSpec>, GateError> {
        self.whole_card(plan, (ctx, slots), qwen3_whole(file))
    }

    /// [`Self::whole_card_qwen3`] of a qwen35moe file, `at` the [`Open35`] the
    /// plan was made by (a slot's share of `ctx`, [`share`]).
    pub fn whole_card_qwen35(
        &self,
        file: &Split,
        plan: &Plan<'_>,
        (ctx, slots): (usize, usize),
        at: Open35,
    ) -> Result<Option<CardSpec>, GateError> {
        self.whole_card(plan, (ctx, slots), qwen35_whole(file, at))
    }

    fn whole_card(
        &self,
        plan: &Plan<'_>,
        (ctx, slots): (usize, usize),
        load: Result<WholeLoad, String>,
    ) -> Result<Option<CardSpec>, GateError> {
        if plan.host.experts > 0 || !self.whole_at(u64::try_from(ctx)?, slots, load)?.fits() {
            return Ok(None);
        }
        Ok(Some(self.place.card_specs()?[0]))
    }
}

/// What a run with `--place` unset loads for one sequence
/// ([`unplaced_qwen3_slots`]), as the CLI's own runs do.
pub fn unplaced_qwen3(file: &Split, ctx: usize, kv: KvQ8) -> Result<Unplaced, GateError> {
    unplaced_qwen3_slots(file, &census()?, ctx, 1, (kv, Beside::default()))
}

/// What a run with `--place` unset loads, decided before any load on one
/// census reading (`census`), on `a`'s card ([`PlaceQ3::unplaced_on`]): the whole
/// model there (`Whole` — no plan, no record) while the whole-fit verdict at
/// `ctx` for `slots` resident sequences takes it ([`whole_on`]: the qwen3moe
/// body's whole load ([`qwen3_whole`]), the cache [`kv_bytes`] counts at the
/// slots' shares); else the placed plan there (`Placed`). The verdict's line
/// goes to stderr; a file with no routed experts the verdict does not take
/// is refused by name. `beside`'s reserves are in the verdict and in the plan.
#[allow(
    dead_code,
    reason = "the serve seat decides its load at the total for its --parallel slots; the CLI and the e2e gates load one sequence"
)]
pub fn unplaced_qwen3_slots(
    file: &Split,
    census: &[DeviceInfo],
    ctx: usize,
    slots: usize,
    (kv, beside): (KvQ8, Beside),
) -> Result<Unplaced, GateError> {
    let probe = PlaceQ3::qwen3_on(file, PlaceQ3::unplaced_on(census)?, census, ctx, kv, beside)?;
    let whole = probe.whole_at(u64::try_from(ctx)?, slots, qwen3_whole(file))?;
    probe.unplaced(ctx, &whole)
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3`] decides it,
/// of a qwen35moe file under `o`: one sequence taking no checkpoints
/// ([`unplaced_qwen35_slots`]), as the CLI's own runs do.
pub fn unplaced_qwen35(file: &Split, o: &Open35) -> Result<Unplaced, GateError> {
    unplaced_qwen35_slots(file, &census()?, o, Seqs::ONE, Beside::default())
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3_slots`]
/// decides it, of a qwen35moe file under `o` at `o.ctx` positions — the
/// total its `seqs` split: the whole-fit verdict of every slot's stores at
/// its share over the program's own arena and
/// reserve as a slot's load gives them ([`qwen35_whole`] at [`share`]); the
/// placed plan on `a`'s card, carrying the sequences' rows ([`laid`]), when
/// it does not take it; `beside`'s reserves are in the verdict and in the
/// plan.
pub fn unplaced_qwen35_slots(
    file: &Split,
    census: &[DeviceInfo],
    o: &Open35,
    seqs: Seqs,
    beside: Beside,
) -> Result<Unplaced, GateError> {
    unplaced_qwen35_lanes(file, census, o, seqs, beside, 1)
}

/// [`unplaced_qwen35_slots`] with the delta lanes (`lanes`, at least one)
/// its whole-fit verdict prices a delta layer at ([`delta_planes`]): the
/// serve seat's, which opens loads that verify drafted rows.
fn unplaced_qwen35_lanes(
    file: &Split,
    census: &[DeviceInfo],
    o: &Open35,
    seqs: Seqs,
    beside: Beside,
    lanes: usize,
) -> Result<Unplaced, GateError> {
    if attention_only(&model::arch::spec(file)?.spec) {
        return whole_dense(
            (file, census),
            (o, seqs.slots, beside),
            lanes,
            None,
            Rule::Open,
        )
        .map(|(card, _)| Unplaced::Whole(card));
    }
    let at = share(o, o.ctx, seqs.slots);
    let probe = PlaceQ3::qwen35_lanes(
        file,
        PlaceQ3::unplaced_on(census)?,
        census,
        at,
        seqs,
        beside,
        lanes,
    )?;
    let whole = probe.whole_at(u64::try_from(o.ctx)?, seqs.slots, qwen35_whole(file, at))?;
    probe.unplaced(o.ctx, &whole)
}

/// What shapes the qwen3 serve seat's load beside its context and slots: the
/// file's family, `--place`, the plan's levers, the cache planes, and under
/// `--mmproj` the media flag (every sequence holds the rotation table an
/// image's positions rewrite), and the reserves of what the process serves
/// beside the model on the stage card (the encoder's, a side seat's embedding
/// model's), and the prompt ubatch the user's flag asks for (`ubatch`).
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
#[derive(Clone, Copy)]
pub struct ServeOpts<'a> {
    pub place: Option<Place>,
    pub levers: &'a PlanLevers,
    pub kv: KvQ8,
    pub media: bool,
    pub beside: Beside,
    /// `-ub` / `--ubatch-size`: the whole load of a file with no routed expert takes exactly this
    /// size or is refused by name ([`serve_load`]); any other load refuses the flag.
    pub ubatch: Option<usize>,
}

/// What the serve seat's file loads as, decided once before the open at the
/// total context — the context its whole-fit verdict and its plan count — so
/// the open makes no fit call of its own ([`serve_load`]).
#[allow(
    dead_code,
    reason = "the serve seat loads it; the routed legs and the types gate read its context"
)]
pub enum ServeLoad {
    /// The whole model on this card, `a`'s: `--place` unset, and the
    /// whole-fit verdict at the total on it takes the whole load.
    Whole(WholeOn),
    /// The placed plan made at `total` positions on `q`'s card: `--place`'s
    /// (no `why`), or `a`'s card's when the whole load does not fit at the
    /// total (`why` names that, [`WHY_NOT_WHOLE`]). Under `--place` whose
    /// plan holds no host expert, `whole` is the card the whole model opens
    /// on instead, when the whole-fit verdict takes it.
    Placed {
        q: Box<PlaceQ3>,
        total: usize,
        why: Option<&'static str>,
        whole: Option<CardSpec>,
    },
}

/// A whole-card load decided before the open: the card, and for a file with no routed expert,
/// whose verdict settles the ubatch ([`Rule`]), the size the open takes.
#[allow(
    dead_code,
    reason = "the serve seat opens through it; the routed legs and the types gate read its card"
)]
#[derive(Clone, Copy)]
pub struct WholeOn {
    pub card: CardSpec,
    pub ubatch: Option<usize>,
}

impl ServeLoad {
    /// `o`, a Body35 serve open, with the ubatch this load's verdict settled when it settled one.
    #[allow(
        dead_code,
        reason = "the serve seat opens through it; the routed legs and the types gate only decide"
    )]
    pub fn open35(&self, o: Open35) -> Open35 {
        match self {
            ServeLoad::Whole(WholeOn {
                ubatch: Some(ubatch),
                ..
            }) => Open35 {
                ubatch: *ubatch,
                ..o
            },
            _ => o,
        }
    }

    /// The ubatch this load's verdict settled, when it settled one.
    #[allow(
        dead_code,
        reason = "the serve seat prices a sequence's checkpoints by it; the routed legs and the types gate only decide"
    )]
    pub fn ubatch(&self) -> Option<usize> {
        match self {
            ServeLoad::Whole(w) => w.ubatch,
            ServeLoad::Placed { .. } => None,
        }
    }
}

/// What a Body35 file's serve load opens at with a `ctx`-row cache: the
/// decode flash selected from its head shape, the levers' ubatch and the `kv` planes, as
/// `generate_qwen3moe` opens it, with `media`'s rotation tables.
///
/// # Errors
/// The file's head shape or the ubatch lever's refusal.
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
pub fn serve_open35(split: &Split, ctx: usize, kv: KvQ8, media: bool) -> Result<Open35, GateError> {
    let spec = model::arch::spec(split)?.spec;
    // Every attention layer of a Body35 file shares one head shape; the body holds the file to it.
    let (heads, kv_heads, head) = spec
        .layers
        .iter()
        .find_map(|l| match &l.mixer {
            Mixer::Gqa(g) => Some((g.heads as usize, g.kv_heads as usize, g.head_dim as usize)),
            _ => None,
        })
        .ok_or("the description holds no attention layer to choose the decode flash by")?;
    Ok(Open35 {
        ctx,
        mma: decode_mma(head, heads / kv_heads)?,
        ubatch: bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size()?,
        kv,
        media,
    })
}

/// Whether every layer of `spec` is a GQA layer: a trunk whose stores are K/V planes alone, with
/// no recurrent state to checkpoint.
fn attention_only(spec: &ModelSpec) -> bool {
    spec.layers.iter().all(|l| matches!(l.mixer, Mixer::Gqa(_)))
}

/// The card a dense trunk (no routed expert to place on the host tier) loads whole on, with the
/// ubatch the load opens at: `place`'s card, else `a`'s ([`PlaceQ3::unplaced_on`]), while the
/// whole-fit verdict at `o.ctx` positions for `slots` resident sequences takes it at the size
/// `rule` settles ([`settle`]), each size priced with the checkpoints its spacing sets (a trunk of
/// attention layers alone has no recurrent store to copy, so its sequences take none:
/// [`seqs_qwen35`]) and its delta layers at `lanes` lanes ([`delta_planes`]), else the verdict's
/// refusal with its terms. The verdict's line goes to stderr, as [`PlaceQ3::unplaced`] prints it.
///
/// # Errors
/// An image input on a trunk of attention layers alone (it has no rotation table for one), the
/// verdict's refusal, or the plan's or the load's own refusal of the file.
fn whole_dense(
    (file, census): (&Split, &[DeviceInfo]),
    (o, slots, beside): (&Open35, usize, Beside),
    lanes: usize,
    place: Option<Place>,
    rule: Rule,
) -> Result<(CardSpec, usize), GateError> {
    if o.media && attention_only(&model::arch::spec(file)?.spec) {
        return Err("a trunk of attention layers alone takes no image input".into());
    }
    let place = match place {
        Some(p) => p,
        None => PlaceQ3::unplaced_on(census)?,
    };
    let (ubatch, whole, probe) = settle(rule, (o.ubatch, o.ctx, slots), |u| {
        let at = share(&Open35 { ubatch: u, ..*o }, o.ctx, slots);
        let probe = PlaceQ3::qwen35_lanes(
            file,
            place,
            census,
            at,
            seqs_qwen35(file, u, slots)?,
            beside,
            lanes,
        )?;
        let whole = probe.whole_at(u64::try_from(o.ctx)?, slots, qwen35_whole(file, at))?;
        Ok((whole, probe))
    })?;
    match probe.unplaced(o.ctx, &whole)? {
        Unplaced::Whole(card) => Ok((card, ubatch)),
        Unplaced::Placed(_) => Err("a trunk with no routed expert was given a placed plan".into()),
    }
}

/// `o`, a one-sequence Body35 open of the CLI, with the ubatch a file with no routed expert settles
/// for its whole load ([`Rule`]): `BLOOMERY_QWEN3_UBATCH` as asked, else the largest of
/// [`ubatch_pick::LADDER`] the card holds at `o.ctx`; `o` as it stands for any other file. `place` is
/// `--place`'s word, `None` for the unset default.
///
/// # Errors
/// The verdict's refusal, as [`whole_dense`]'s.
#[allow(
    dead_code,
    reason = "the CLI opens through it; the serve seat settles in serve_load and the e2e gates name their size"
)]
pub fn dense_open35(file: &Split, o: Open35, place: Option<Place>) -> Result<Open35, GateError> {
    let read = model::arch::spec(file)?;
    let attention = attention_only(&read.spec);
    if !dense_ffn(&read.spec) || !(attention || place.is_none()) {
        return Ok(o);
    }
    let census = census()?;
    let rule = ubatch_rule(None)?;
    let (_, ubatch) = whole_dense((file, &census), (&o, 1, Beside::default()), 1, place, rule)?;
    Ok(Open35 { ubatch, ..o })
}

/// The card a one-sequence load of `file` under `--place`'s `place` takes whole, when `file` is an
/// attention-only trunk ([`whole_dense`]); `None` for a file whose placement is the plan's
/// ([`PlaceQ3::qwen35`]).
///
/// # Errors
/// As [`whole_dense`].
#[allow(
    dead_code,
    reason = "the CLI's `--place` load; the serve seat and the e2e gates plan on one census"
)]
pub fn whole_card_attention(
    file: &Split,
    place: Place,
    o: &Open35,
) -> Result<Option<CardSpec>, GateError> {
    let read = model::arch::spec(file)?;
    if !attention_only(&read.spec) {
        return Ok(None);
    }
    let census = census()?;
    whole_dense(
        (file, &census),
        (o, 1, Beside::default()),
        1,
        Some(place),
        Rule::Open,
    )
    .map(|(card, _)| Some(card))
}

/// The serve seat's load of `slots` resident sequences at `total` positions,
/// decided on one census reading: under `--place`, the placed plan there;
/// unset, the whole model while the whole-fit verdict takes it, else the
/// placed plan ([`unplaced_qwen3_slots`], [`unplaced_qwen35_lanes`]). The
/// file's family is its description's program ([`Family::of`]). The
/// verdict prices a qwen35 family file's delta layers at `lanes` lanes
/// (at least one): one on a plain load, the load's verify lanes on a load
/// that verifies drafted rows — the seat's, from its draft word. A file
/// with no routed expert has only the whole load, and its verdict settles
/// the ubatch the open takes: the user's (`o.ubatch`, else
/// `BLOOMERY_QWEN3_UBATCH`) or the largest of [`ubatch_pick::LADDER`] the card holds
/// ([`Rule`]); any other load takes the levers' ubatch and refuses
/// `o.ubatch` by name.
///
/// # Errors
/// The plan's or the verdict's refusal.
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
pub fn serve_load(
    (split, census): (&Split, &[DeviceInfo]),
    (total, slots): (usize, usize),
    o: ServeOpts<'_>,
    lanes: usize,
) -> Result<ServeLoad, GateError> {
    let unplaced = match (Family::of(split)?, o.place) {
        (Family::Qwen35moe | Family::Qwen35, Some(p)) => {
            let open = serve_open35(split, total, o.kv, o.media)?;
            let seqs = seqs_qwen35(split, open.ubatch, slots)?;
            let read = model::arch::spec(split)?;
            if attention_only(&read.spec) {
                let rule = ubatch_rule(o.ubatch)?;
                let (card, ubatch) = whole_dense(
                    (split, census),
                    (&open, slots, o.beside),
                    lanes,
                    Some(p),
                    rule,
                )?;
                return Ok(ServeLoad::Whole(WholeOn {
                    card,
                    ubatch: Some(ubatch),
                }));
            }
            refuse_flag(o.ubatch, "a file placed by the plan")?;
            let at = share(&open, total, slots);
            let q = PlaceQ3::qwen35_lanes(split, p, census, at, seqs, o.beside, lanes)?;
            let plan = q.plan(total, o.levers)?;
            let whole = q.whole_card_qwen35(split, &plan, (total, slots), at)?;
            return Ok(ServeLoad::Placed {
                q: Box::new(q),
                total,
                why: None,
                whole,
            });
        }
        (Family::Qwen35moe | Family::Qwen35, None) => {
            let open = serve_open35(split, total, o.kv, o.media)?;
            let seqs = seqs_qwen35(split, open.ubatch, slots)?;
            let read = model::arch::spec(split)?;
            if dense_ffn(&read.spec) {
                let rule = ubatch_rule(o.ubatch)?;
                let (card, ubatch) =
                    whole_dense((split, census), (&open, slots, o.beside), lanes, None, rule)?;
                return Ok(ServeLoad::Whole(WholeOn {
                    card,
                    ubatch: Some(ubatch),
                }));
            }
            refuse_flag(o.ubatch, "a routed file")?;
            unplaced_qwen35_lanes(split, census, &open, seqs, o.beside, lanes)?
        }
        (Family::Qwen3moe, Some(p)) => {
            refuse_flag(o.ubatch, "a qwen3moe file")?;
            let q = PlaceQ3::qwen3_on(split, p, census, total, o.kv, o.beside)?;
            let plan = q.plan(total, o.levers)?;
            let whole = q.whole_card_qwen3(split, &plan, (total, slots))?;
            return Ok(ServeLoad::Placed {
                q: Box::new(q),
                total,
                why: None,
                whole,
            });
        }
        (Family::Qwen3moe, None) => {
            refuse_flag(o.ubatch, "a qwen3moe file")?;
            unplaced_qwen3_slots(split, census, total, slots, (o.kv, o.beside))?
        }
    };
    Ok(match unplaced {
        Unplaced::Whole(card) => ServeLoad::Whole(WholeOn { card, ubatch: None }),
        Unplaced::Placed(q) => ServeLoad::Placed {
            q,
            total,
            why: Some(WHY_NOT_WHOLE),
            whole: None,
        },
    })
}

/// The serve seat's context and the load at it, the seats' common choice
/// ([`choose_ctx`]): a set `--ctx` (`set`) is the total the `slots` split,
/// planned once at its share; unset, a slot asks the target and halves only
/// while the plan is short. Each candidate is the load [`serve_load`] makes
/// at its total, a placed one planned there too, so a short card shows here
/// and not at the open, all on one census reading; the probes it ran and
/// the readings it took come back beside the choice.
///
/// # Errors
/// As [`choose_ctx`].
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
pub fn serve_choice(
    split: &Split,
    (set, slots): (Option<usize>, usize),
    o: ServeOpts<'_>,
) -> Result<(Chosen<ServeLoad>, Reads), GateError> {
    serve_choice_lanes(split, (set, slots), o, 1)
}

/// [`serve_choice`] with the delta lanes (`lanes`, at least one) the
/// whole-fit verdict prices a qwen35 family file's delta layers at
/// ([`delta_planes`]): the serve seat's, which opens loads that verify
/// drafted rows.
#[allow(
    dead_code,
    reason = "the serve seat reads it; the routed legs and the types gate read the plain count, and the CLI and the e2e gates include the planner without it"
)]
pub fn serve_choice_lanes(
    split: &Split,
    (set, slots): (Option<usize>, usize),
    o: ServeOpts<'_>,
    lanes: usize,
) -> Result<(Chosen<ServeLoad>, Reads), GateError> {
    let before = census_reads();
    let census = census()?;
    let mut probes = 0;
    let asked = if dense_ffn(&model::arch::spec(split)?.spec) {
        ubatch_rule(o.ubatch)?.asked()
    } else {
        None
    };
    let plan_at = |c: u64| -> Result<Probe<ServeLoad>, GateError> {
        probes += 1;
        let total = usize::try_from(c)?.saturating_mul(slots);
        let load = match serve_load((split, &census), (total, slots), o, lanes) {
            Ok(load) => load,
            Err(e) => {
                // The load asked at a ladder size: it fits, or is short of the card.
                let fits = |u: usize| {
                    let at = ServeOpts {
                        ubatch: Some(u),
                        ..o
                    };
                    match serve_load((split, &census), (total, slots), at, lanes) {
                        Ok(_) => Ok(true),
                        Err(e) => short_or(e).map(|_| false),
                    }
                };
                return short_or(e).and_then(|p| named_size(p, asked, (c, slots), fits));
            }
        };
        if let ServeLoad::Placed { q, total, .. } = &load
            && let Err(e) = q.plan(*total, o.levers)
        {
            return short_or(e);
        }
        Ok(Probe::Holds(load))
    };
    let chosen = choose_ctx(split, (set, slots), (None, None), plan_at)?;
    let censuses = census_reads() - before;
    Ok((chosen, Reads { censuses, probes }))
}

/// A short probe of a load whose ubatch the user named (`ask`), at the least context the choice tries
/// (`c` positions a slot over `slots`): no smaller context helps, so the refusal is that size's
/// ([`ubatch_pick::asked_short`]), where the context choice's own words ask for fewer slots or a
/// smaller file, with the largest size of the ladder that load holds at this context and slots
/// (`fits(u)`: whether the load asked at the size `u` fits, [`ubatch_pick::largest_fit`]). A probe
/// above the floor stays short, for the choice to search below.
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
fn named_size(
    probe: Probe<ServeLoad>,
    ask: Option<Ask>,
    (c, slots): (u64, usize),
    fits: impl FnMut(usize) -> Result<bool, GateError>,
) -> Result<Probe<ServeLoad>, GateError> {
    match (probe, ask) {
        (Probe::Short(s), Some(a)) if c <= SEAT_STEPS.floor => {
            let largest = ubatch_pick::largest_fit(fits)?;
            Err(
                ubatch_pick::asked_short(a, (usize::try_from(c)?, slots), s.need, s.room, largest)
                    .into(),
            )
        }
        (probe, _) => Ok(probe),
    }
}

/// A plan's refusal as a short memory ([`Probe::Short`]), else the refusal
/// as it stands.
#[allow(
    dead_code,
    reason = "the serve seat, the routed legs and the types gate read it; the CLI and the e2e gates include the planner without it"
)]
fn short_or(e: GateError) -> Result<Probe<ServeLoad>, GateError> {
    let short = short_in::<q3::PlaceError>(&e).or_else(|| short_in::<q35::PlaceError>(&e));
    short.map(Probe::Short).ok_or(e)
}

/// The load a run with `--place` unset takes ([`unplaced_qwen3`]), on `a`'s
/// card ([`PlaceQ3::unplaced_on`]): the whole model, or the placed plan.
pub enum Unplaced {
    /// The whole-card load on this card, the one the verdict counted.
    Whole(CardSpec),
    /// The placed plan on `a`'s card: its `plan` record prints with
    /// `why=whole_does_not_fit` before the open.
    Placed(Box<PlaceQ3>),
}

/// The card of a whole-card load with no plan, opened by [`Gpu::open_card`]:
/// such a load prints no `need` record, so the need hook has nothing to be
/// asked about and is not (a load that does print one opens through
/// [`Gpu::open_card_after_hook`]). A hook that is set says so on stderr.
#[allow(
    dead_code,
    reason = "generate_qwen3moe and the serve seat open their unset whole loads through it; the e2e gates include the planner without it"
)]
pub fn open_unplanned_card(card: CardSpec) -> Result<Gpu, GateError> {
    if let Some(hook) = bloomery_levers::need_levers()?.hook {
        eprintln!(
            "need hook {} is not asked: a whole model on one card with no plan states no need",
            hook.display()
        );
    }
    Ok(Gpu::open_card(card.name, card.device)?)
}

/// What a run with `--place` unset loads: the whole model on `a`'s card —
/// no plan, no record — while the whole-fit verdict takes it
/// ([`unplaced_qwen3`]); else the placed plan on that card, its `plan`
/// record with `why=whole_does_not_fit` handed to `emit` before the open,
/// the host set as `levers` asks.
#[allow(
    dead_code,
    reason = "generate_qwen3moe opens its unset load through it; the serve seat decides its load at the total before the open"
)]
pub fn open_unplaced_qwen3(
    file: Split,
    ctx: usize,
    opts: OpenOpts,
    levers: &Levers,
    emit: fn(Record),
) -> Result<Qwen3moeModel, GateError> {
    match unplaced_qwen3(&file, ctx, opts.kv)? {
        Unplaced::Whole(card) => Ok(Qwen3moeModel::open(open_unplanned_card(card)?, file, opts)?),
        Unplaced::Placed(q) => {
            let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
            for line in q.records(&plan, Some(WHY_NOT_WHOLE))? {
                emit(line);
            }
            open_qwen3(file, &plan, opts, levers.host())
        }
    }
}

/// What a run with `--place` unset loads, as [`open_unplaced_qwen3`] loads
/// it, of a qwen35moe file: one sequence taking no checkpoints.
#[allow(
    dead_code,
    reason = "generate_qwen3moe opens its unset load through it; the serve seat decides its load at the total for its slots"
)]
pub fn open_unplaced_qwen35(
    file: Split,
    o: Open35,
    levers: &Levers,
    emit: fn(Record),
) -> Result<Qwen35moeModel, GateError> {
    match unplaced_qwen35(&file, &o)? {
        Unplaced::Whole(card) => Ok(Qwen35moeModel::open(open_unplanned_card(card)?, file, o)?),
        Unplaced::Placed(q) => {
            let plan = q.plan(o.ctx, &PlanLevers::from_levers(levers)?)?;
            for line in q.records(&plan, Some(WHY_NOT_WHOLE))? {
                emit(line);
            }
            open_qwen35(file, &plan, o, levers.host())
        }
    }
}

/// The `why` of a placed plan a run with `--place` unset took because its
/// whole file did not fit the card's free bytes.
pub const WHY_NOT_WHOLE: &str = "whole_does_not_fit";

/// The Qwen3-30B model of `file` by `plan`: the body's placed load, its host
/// set as `host` asks — whatever the plan holds on the host, none included.
/// Whether the whole model loads in its place is not read off the plan: the
/// caller decides it ([`unplaced_qwen3`], [`PlaceQ3::whole_card_qwen3`]) and
/// opens the whole model itself.
pub fn open_qwen3(
    file: Split,
    plan: &Plan<'_>,
    opts: OpenOpts,
    host: HostCfg,
) -> Result<Qwen3moeModel, GateError> {
    Ok(GpuModel::<Body>::open_placed(file, plan, opts, host)?)
}

/// The Qwen3-30B model of `file` by `plan` ([`open_qwen3`]) for `slots`
/// resident sequences of `opts.ctx` rows each ([`open_slots`]). One slot is
/// [`open_qwen3`] itself.
#[allow(
    dead_code,
    reason = "the serve seat's resident slots; the CLI and the e2e gates load one sequence"
)]
pub fn open_qwen3_slots(
    file: Split,
    plan: &Plan<'_>,
    slots: usize,
    opts: OpenOpts,
    host: HostCfg,
) -> Result<Qwen3moeModel, GateError> {
    if slots <= 1 {
        return open_qwen3(file, plan, opts, host);
    }
    open_slots(plan, slots, opts.ctx, |at| open_qwen3(file, at, opts, host))
}

/// The Qwen3.6 model of `file` by `plan` ([`open_qwen35`]) for `slots`
/// resident sequences of `o.ctx` rows each ([`open_slots`]), `plan` carrying
/// their rows ([`PlaceQ3::qwen35_on`]). One slot is [`open_qwen35`]
/// itself.
#[allow(
    dead_code,
    reason = "the serve seat's resident slots; the CLI and the e2e gates load one sequence"
)]
pub fn open_qwen35_slots(
    file: Split,
    plan: &Plan<'_>,
    slots: usize,
    o: Open35,
    host: HostCfg,
) -> Result<Qwen35moeModel, GateError> {
    if slots <= 1 {
        return open_qwen35(file, plan, o, host);
    }
    open_slots(plan, slots, o.ctx, |at| open_qwen35(file, at, o, host))
}

/// The model `open` makes by `plan` for `slots` resident sequences, the
/// seat's split for both bodies: `plan` made at the total context the slots
/// split, the model holding its one sequence at `ctx` rows — a slot's share
/// — so the sequences the caller adds after it sit inside the cache the plan
/// counts. `slots · ctx` past the plan's `ctx_max` is refused by name before
/// anything loads; `open` is handed the plan with its `ctx_max` at `ctx`
/// (the rows the caches hold, which the open checks and the model steps
/// against), every other term the plan's own at the total; once open, the
/// `slots` sequences' stores — the body's own sequence bytes
/// ([`Slots::seq_bytes`]), `slots` times — past what the plan counts for
/// them — its KV term and its card's [`SLOTS_RESERVE`] and, on a media load,
/// [`TABLES_RESERVE`] rows ([`laid`]) — are refused by name, and both stand
/// on one stderr line.
fn open_slots<B: Slots>(
    plan: &Plan<'_>,
    slots: usize,
    ctx: usize,
    open: impl FnOnce(&Plan<'_>) -> Result<GpuModel<B>, GateError>,
) -> Result<GpuModel<B>, GateError> {
    const WHAT: &str = "placed slots";
    let rows = u64::try_from(slots.saturating_mul(ctx))?;
    if rows > plan.ctx_max {
        return Err(format!(
            "{WHAT}: {slots} sequences of {ctx} rows hold {rows} positions, past the {} the plan \
             counts",
            plan.ctx_max
        )
        .into());
    }
    let at = Plan {
        ctx_max: u64::try_from(ctx)?,
        ..plan.clone()
    };
    let m = open(&at)?;
    let held = u64::try_from(m.body(WHAT)?.seq_bytes())?.saturating_mul(u64::try_from(slots)?);
    let (Some(totals), Some(card)) = (plan.cards.first(), plan.machine.cards.first()) else {
        return Err(format!("{WHAT}: a plan of no card").into());
    };
    let kv = totals.kv_bytes;
    let reserved = |what: &str| -> u64 {
        card.reserves
            .iter()
            .filter(|(name, _)| name == what)
            .map(|&(_, b)| b)
            .sum()
    };
    let (row, tables) = (reserved(SLOTS_RESERVE), reserved(TABLES_RESERVE));
    let counted = kv.saturating_add(row).saturating_add(tables);
    let terms = match (row > 0, tables > 0) {
        (false, false) => String::new(),
        (true, false) => format!(" (its KV term {kv} B and its {SLOTS_RESERVE} reserve {row} B)"),
        (_, true) => format!(
            " (its KV term {kv} B, its {SLOTS_RESERVE} reserve {row} B and its {TABLES_RESERVE} \
             reserve {tables} B)"
        ),
    };
    let line = format!(
        "{WHAT}: {slots} sequences of {ctx} rows hold {held} B of cache; the plan at {} positions \
         counts {counted} B{terms}",
        plan.ctx_max
    );
    if held > counted {
        return Err(format!("{line}: past its count").into());
    }
    eprintln!("{line}");
    Ok(m)
}

/// The checkpoints `seqs` pin on the host in a load of `slot_ctx` positions a
/// slot ([`checkpoints_pinned`]) held against what the load counts them in,
/// one stderr line either way: a placed load's plan carries them as its
/// host's checkpoint reserve ([`laid`]), which its own check holds against
/// the host's usable bytes and refuses by name; a whole-card load has no host
/// plan, so the host's available bytes now ([`workstation::host_available`])
/// must hold them, else they are refused by name before anything loads (on a
/// unified machine the whole fit counts them with the card's too,
/// [`whole_on`]). Nothing for `seqs` that take none.
#[allow(
    dead_code,
    reason = "the serve seat's qwen35moe loads take checkpoints; the CLI and the e2e gates take none"
)]
pub fn checkpoints_fit(seqs: Seqs, slot_ctx: usize, placed: bool) -> Result<(), GateError> {
    let Some(price) = seqs.checkpoints else {
        return Ok(());
    };
    let slot_ctx = u64::try_from(slot_ctx)?;
    let pin = checkpoints_pinned(seqs, slot_ctx)?;
    let each = price
        .budget(slot_ctx)
        .map_err(|e| format!("the checkpoint budget at {slot_ctx} positions a slot: {e}"))?;
    let n = seqs.slots;
    let what = if n == 1 {
        "sequence pins"
    } else {
        "sequences pin"
    };
    let head = format!(
        "checkpoints: {n} {what} at most {pin} B of host ({each} B each at {slot_ctx} positions)"
    );
    if placed {
        eprintln!("{head}, counted in the placed plan's host reserves");
        return Ok(());
    }
    let (available, read) = workstation::host_available_read()?;
    if pin > available {
        return Err(format!(
            "{head}, past the host room of {available} B ({read}): give a smaller --parallel"
        )
        .into());
    }
    eprintln!("{head}, of the host room of {available} B ({read})");
    Ok(())
}

/// The Qwen3.6 model of `file` by `plan`, as [`open_qwen3`].
pub fn open_qwen35(
    file: Split,
    plan: &Plan<'_>,
    o: Open35,
    host: HostCfg,
) -> Result<Qwen35moeModel, GateError> {
    Ok(GpuModel::<Body35>::open_placed(file, plan, o, host)?)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use bloomery_gpu_gates::generate::choose_ctx_of;
    use model::placement::ModelTensor;

    /// A drafted verdict's delta bytes are the engine's lane formula
    /// ([`delta_planes`]): at Qwen3.6's shape (32 value heads of 128, 16 key
    /// heads, a 4-tap conv) one lane is the state in place with the ring and
    /// no stamp, and `Body35::VERIFY_LANES` lanes hold the other lanes and a
    /// u32 stamp a lane — `runtime::stores`'s own terms, the ones
    /// `KvLayout::of_spec_lanes` sums and the load's stamped stores hold
    /// (`RecStore::stamped`).
    #[allow(
        dead_code,
        reason = "`generate_qwen3moe`'s test calls it; every other bin that includes this file leaves it unused"
    )]
    pub(crate) fn a_drafted_verdict_prices_the_engines_lanes() {
        use runtime::stores::{U32_BYTES, delta_lane_bytes, recurrent_bytes};
        const V: usize = 32;
        const K: usize = 16;
        const D: usize = 128;
        const CONV: usize = 4;
        let shape = LinearShape {
            n_k: K,
            n_v: V,
            map: KHeadMap::Tiled,
        };
        let sum = |lanes: usize| {
            let mut planes = Vec::new();
            delta_planes(&mut planes, shape, lanes);
            planes.iter().sum::<u64>()
        };
        // The engine's bytes at the same count: `gdn_bytes`'s own terms, the
        // ones `KvLayout::of_spec_lanes` puts in a delta layer's `recurrent`.
        let engine = |lanes: usize| {
            recurrent_bytes(V, K, D, CONV)
                + if lanes > 1 {
                    delta_lane_bytes(V, D, lanes)
                } else {
                    0
                }
        };
        let lanes = Body35::VERIFY_LANES;
        assert_eq!(sum(1), engine(1), "one lane: the state in place, no stamp");
        assert_eq!(
            sum(lanes),
            engine(lanes),
            "the verdict prices every lane a drafted load's store holds"
        );
        assert_eq!(
            sum(lanes) - sum(1),
            (lanes as u64 - 1) * 4 * V as u64 * D as u64 * D as u64 + U32_BYTES * lanes as u64
        );
    }

    #[allow(
        dead_code,
        reason = "`generate_qwen3moe`'s test calls it; every other bin that includes this file leaves it unused"
    )]
    pub(crate) fn dense_shortage_fits_the_default_context() {
        const TARGET: u64 = 16_384;
        let model = ModelTensors {
            tensors: vec![ModelTensor {
                name: "blk.0.ffn_up.weight".into(),
                shard: 0,
                layer: Some(0),
                role: Role::DenseFfn,
                ty: gguf::GgmlType::Q8_0,
                dims: vec![256, 16],
                file_bytes: 0,
                gathered_rows: None,
            }],
            layers: 1,
            experts: 0,
            experts_used: 0,
            bf16_words: |_| false,
        };
        let machine = workstation::placed_with_arena(workstation::A6000, 1, 0);
        let mut card = machine.whole_card().expect("one card");
        for load in [
            WholeLoad::Checked {
                arena: 8192,
                reserve: 4096,
            },
            WholeLoad::Unchecked,
        ] {
            card.free_bytes = None;
            let need = |ctx| placement::whole_need(&model, &card, ctx * 1024, load).unwrap();
            let lower = need(TARGET / 2);
            let upper = need(TARGET);
            assert!(lower.fits() && upper.fits());
            assert!(lower.need < upper.need);
            card.free_bytes = Some(lower.need);
            let mut probes = Vec::new();
            let chosen = choose_ctx_of(TARGET, (None, 1), (None, None), |ctx| {
                probes.push(ctx);
                let need = placement::whole_need(&model, &card, ctx * 1024, load)?;
                if need.fits() {
                    return Ok(Probe::Holds(ServeLoad::Whole(WholeOn {
                        card: workstation::A6000,
                        ubatch: None,
                    })));
                }
                let error = Whole::Short(need.clone()).dense_refusal(ctx as usize);
                let Some(PlacementError::CardBudgetFloor {
                    budget,
                    floor,
                    dense,
                    kv,
                    context,
                    scratch,
                    reserves,
                    margin,
                    beside,
                    ..
                }) = error.downcast_ref::<PlacementError>()
                else {
                    panic!("expected typed CardBudgetFloor, got {error}");
                };
                let past = match load {
                    WholeLoad::Checked { arena, reserve } => (reserve, arena),
                    WholeLoad::Unchecked => (card.margin_bytes, 0),
                };
                assert_eq!(
                    (
                        *budget, *floor, *dense, *kv, *context, *scratch, *reserves, *margin,
                        *beside
                    ),
                    (
                        lower.need,
                        need.need,
                        need.weights,
                        need.kv,
                        card.context_bytes,
                        card.scratch_bytes,
                        card.reserve_bytes().expect("bounded reserves"),
                        past.0,
                        past.1
                    )
                );
                short_or(error)
            })
            .unwrap_or_else(|e| panic!("dense default context must fit: {e}"));
            assert_eq!(chosen.total, TARGET / 2);
            // The target, the midpoint of the steps 4 to 15 under it, the step under that, and the
            // fit their shortfalls' line names.
            assert_eq!(probes, [TARGET, 10_240, 9216, 8192]);
        }
        let refused = Whole::Refused("unsupported tensor".into()).dense_refusal(TARGET as usize);
        assert!(short_or(refused).is_err());
    }

    /// A dense qwen35 header of the body's smallest shapes, written to `path` over a sparse file
    /// of its full length: hidden 256, three gated-delta layers (16 key and 32 value heads of
    /// 128) and a gated attention layer (16 query and 2 key heads of 256), a SwiGLU FFN of 256 in
    /// each and no expert key; every projection Q4_K, the head Q6_K, the norms, β, α and the
    /// conv F32.
    fn dense_header(path: &std::path::Path) -> Split {
        use gguf::Value as V;
        const H: u64 = 256;
        const FF: u64 = 256;
        const VOCAB: u64 = 256;
        const CHANNELS: u64 = (2 * 16 + 32) * 128;
        const ATTN: u64 = 32 * 128;
        let u = |k: &str, v: u32| (format!("qwen35.{k}"), V::U32(v));
        let kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                V::String("qwen35".to_owned()),
            ),
            u("block_count", 4),
            u("embedding_length", H as u32),
            u("context_length", 4096),
            u("feed_forward_length", FF as u32),
            u("attention.head_count", 16),
            u("attention.head_count_kv", 2),
            u("attention.key_length", 256),
            u("attention.value_length", 256),
            (
                "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
                V::F32(1e-6),
            ),
            u("rope.dimension_count", 64),
            (
                "qwen35.rope.dimension_sections".to_owned(),
                V::Array([11, 11, 10, 0].map(V::I32).to_vec()),
            ),
            ("qwen35.rope.freq_base".to_owned(), V::F32(1e7)),
            u("ssm.conv_kernel", 4),
            u("ssm.state_size", 128),
            u("ssm.group_count", 16),
            u("ssm.time_step_rank", 32),
            u("ssm.inner_size", ATTN as u32),
            u("full_attention_interval", 4),
            (
                "tokenizer.ggml.model".to_owned(),
                V::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre".to_owned(),
                V::String("qwen35".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                V::Array((0..VOCAB).map(|i| V::String(i.to_string())).collect()),
            ),
        ];
        let (q4, q6, f32) = (
            gguf::GgmlType::Q4_K,
            gguf::GgmlType::Q6_K,
            gguf::GgmlType::F32,
        );
        let mut tensors = vec![
            ("token_embd.weight".to_owned(), vec![H, VOCAB], q4),
            ("output_norm.weight".to_owned(), vec![H], f32),
            ("output.weight".to_owned(), vec![H, VOCAB], q6),
        ];
        for l in 0..4 {
            let mixer: &[(&str, &[u64], gguf::GgmlType)] = if l < 3 {
                &[
                    ("attn_qkv.weight", &[H, CHANNELS], q4),
                    ("attn_gate.weight", &[H, ATTN], q4),
                    ("ssm_conv1d.weight", &[4, CHANNELS], f32),
                    ("ssm_dt.bias", &[32], f32),
                    ("ssm_a", &[32], f32),
                    ("ssm_alpha.weight", &[H, 32], f32),
                    ("ssm_beta.weight", &[H, 32], f32),
                    ("ssm_norm.weight", &[128], f32),
                    ("ssm_out.weight", &[ATTN, H], q4),
                ]
            } else {
                &[
                    ("attn_q.weight", &[H, 2 * ATTN], q4),
                    ("attn_k.weight", &[H, 512], q4),
                    ("attn_v.weight", &[H, 512], q4),
                    ("attn_q_norm.weight", &[256], f32),
                    ("attn_k_norm.weight", &[256], f32),
                    ("attn_output.weight", &[ATTN, H], q4),
                ]
            };
            let block: [(&str, &[u64], gguf::GgmlType); 5] = [
                ("attn_norm.weight", &[H], f32),
                ("post_attention_norm.weight", &[H], f32),
                ("ffn_gate.weight", &[H, FF], q4),
                ("ffn_up.weight", &[H, FF], q4),
                ("ffn_down.weight", &[FF, H], q4),
            ];
            for &(stem, dims, ty) in mixer.iter().chain(&block) {
                tensors.push((format!("blk.{l}.{stem}"), dims.to_vec(), ty));
            }
        }
        let decls = tensors
            .into_iter()
            .map(|(name, dims, ty)| {
                let (Some(blck), Some(tsz)) = (ty.blck_size(), ty.type_size()) else {
                    panic!("{name}: {ty:?} has no block info");
                };
                let nbytes = gguf::tensor_bytes(blck, tsz, &dims).expect("small tensors");
                gguf::write::TensorDecl {
                    name,
                    dims,
                    type_id: ty.as_u32(),
                    nbytes,
                }
            })
            .collect();
        let layout = gguf::write::Layout::new(&kvs, decls).expect("the dense layout");
        let len = layout.file_len();
        let mut file = std::fs::File::create(path).expect("the header file");
        drop(gguf::write::Writer::new(&mut file, layout).expect("the header"));
        file.set_len(len).expect("the file's length");
        Split::open(path).expect("the dense header opens")
    }

    /// A dense qwen35 file (no router) plans through [`PlaceQ3::qwen35_on`]: its machine counts
    /// the whole load's ubatch arena, as the load sizes it, not a placed arena, which a file with
    /// no router refuses; an expert tier is refused by name through the same consumer.
    #[allow(
        dead_code,
        reason = "`generate_qwen3moe`'s test calls it; every other bin that includes this file leaves it unused"
    )]
    pub(crate) fn a_dense_header_plans_through_qwen35_on() {
        let path =
            std::env::temp_dir().join(format!("bloomery-q35-dense-{}.gguf", std::process::id()));
        let split = dense_header(&path);
        let o = Open35 {
            ctx: 4096,
            mma: false,
            ubatch: 512,
            kv: KvQ8::F16,
            media: false,
        };
        let census = [DeviceInfo {
            ordinal: 0,
            name: "NVIDIA RTX A6000".to_owned(),
            total_bytes: 50_952_536_064,
            free_bytes: 50_000_000_000,
            uuid: [1; 16],
            pci_bus: "0000:41:00.0".to_owned(),
            held_by: None,
            compute_capability: Some((8, 6)),
            granule_bytes: None,
            integrated: false,
        }];
        let place = Place::parse("cuda0").expect("a card word");
        let q = PlaceQ3::qwen35_on(&split, place, &census, o, Seqs::ONE, Beside::default())
            .unwrap_or_else(|e| panic!("a dense file plans: {e}"));
        let WholeLoad::Checked { arena, .. } =
            Qwen35moeModel::whole_load(&split, o).expect("the whole load reads the file")
        else {
            panic!("a qwen35 whole load checks its arena");
        };
        assert_eq!(q.arch, "qwen35");
        assert_eq!(workstation::counted_arena_bytes(&q.machine.cards[0]), arena);
        q.plan(o.ctx, &PlanLevers::default())
            .unwrap_or_else(|e| panic!("the dense plan at {}: {e}", o.ctx));
        // The whole load counts the cache in the census card's granule.
        let cache = |granule| {
            let mut fine = census.clone();
            fine[0].granule_bytes = granule;
            let place = Place::parse("cuda0").expect("a card word");
            let q = PlaceQ3::qwen35_on(&split, place, &fine, o, Seqs::ONE, Beside::default())
                .unwrap_or_else(|e| panic!("a dense file plans: {e}"));
            let load = Qwen35moeModel::whole_load(&split, o).map_err(|e| e.to_string());
            match q.whole_at(4096, 1, load).expect("a whole verdict") {
                Whole::Fits(need) | Whole::Short(need) => need.kv,
                Whole::Refused(why) => panic!("the whole load refused: {why}"),
            }
        };
        let (fine, coarse) = (cache(NonZeroU64::new(64 << 10)), cache(None));
        assert!(fine < coarse, "{fine} B at 64 KiB, {coarse} B at 2 MiB");
        let tier = DeviceInfo {
            ordinal: 1,
            name: "NVIDIA RTX 3090".to_owned(),
            total_bytes: 25_769_803_776,
            free_bytes: 25_000_000_000,
            uuid: [2; 16],
            pci_bus: "0000:81:00.0".to_owned(),
            held_by: None,
            compute_capability: Some((8, 6)),
            granule_bytes: None,
            integrated: false,
        };
        let census = [census.into_iter().next().expect("the stage card"), tier];
        let two = Place::parse("cuda0+1").expect("two card words");
        let Err(refusal) =
            PlaceQ3::qwen35_on(&split, two, &census, o, Seqs::ONE, Beside::default())
        else {
            panic!("a dense model cannot use an expert tier")
        };
        assert!(matches!(
            refusal.downcast_ref::<PlacementError>(),
            Some(PlacementError::Metadata { key, .. }) if key == "Qwen3.6 expert tier"
        ));
        std::fs::remove_file(&path).expect("the header file");
    }
}
