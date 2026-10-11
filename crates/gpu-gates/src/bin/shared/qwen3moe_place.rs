//! A placed load of a qwen3moe (Qwen3-30B-A3B) or qwen35moe
//! (Qwen3.6-35B-A3B) file, as `generate_qwen3moe` and the qwen3 serve seat
//! open one under `--place`: the placement word's stage card
//! (`generate::Place`, an alias or a card list of one card; a tier card is
//! refused by name, the program hangs none), the machine
//! `model::arch::qwen3moe::place::machine` lays out on it with the scratch
//! the body's placed load holds past the m = 1 scratch, the plan by the
//! expert rule over the stacks `card_routed` loads — the trunk on the card,
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
//! not, the placed plan on the same card. The serve seat's unset
//! `--ctx` default lives here too ([`default_ctx_qwen3`], [`default_ctx_qwen35`]:
//! the common rule over the whole load or a placed one), and its resident slots open a
//! placed plan through [`open_qwen3_slots`] and [`open_qwen35_slots`]. A
//! seat's placed plan counts what its resident sequences hold past the plan's
//! one-sequence terms ([`Seqs`]): on the card the stores no position moves,
//! once a sequence past the live one; on the host the checkpoints a
//! qwen35moe sequence's prompt calls pin. Every census reading the module
//! takes and every probe its searches run is counted ([`reads`]).

use std::sync::atomic::{AtomicUsize, Ordering};

use bloomery_gpu::arch::qwen3moe::{
    Body, Body35, KvQ8, Open35, OpenOpts, Qwen35moeModel, checkpoint_price35,
};
use bloomery_gpu::linear::{KHeadMap, LinearShape};
use bloomery_gpu::model::{GpuModel, Slots};
use bloomery_gpu::rope_neox::q8_plane_lens;
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, PlaceWhy};
use bloomery_gpu_gates::image_plan;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{checkpoints_pinned, reserve_checkpoints};
use bloomery_levers::{HostCfg, Levers};
use gguf::Split;
use model::arch::qwen3moe::place::{self as q3, card_routed};
use model::arch::qwen35moe::hparams::Kind;
use model::arch::qwen35moe::place as q35;
use model::placement::ctx::{ExpertBytes, SlotCtx};
use model::placement::workstation::{self, CardSpec, DeviceInfo, GRANULE};
use model::placement::{
    self, Device, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Role,
    WholeLoad, WholeNeed,
};

/// The census readings this module took and the probes its context searches
/// ran, since the process began: a caller reads them around a search
/// ([`reads`], [`Reads::since`]), so a search that reads the census per
/// probe shows in its counts.
static CENSUSES: AtomicUsize = AtomicUsize::new(0);
static PROBES: AtomicUsize = AtomicUsize::new(0);

/// The module's census readings and search probes at one moment ([`reads`]).
#[allow(
    dead_code,
    reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reads {
    pub censuses: usize,
    pub probes: usize,
}

impl Reads {
    /// The readings and probes taken since `before`.
    #[allow(
        dead_code,
        reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
    )]
    #[must_use]
    pub fn since(self, before: Reads) -> Reads {
        Reads {
            censuses: self.censuses - before.censuses,
            probes: self.probes - before.probes,
        }
    }
}

/// The census readings and search probes so far.
#[allow(
    dead_code,
    reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
)]
#[must_use]
pub fn reads() -> Reads {
    Reads {
        censuses: CENSUSES.load(Ordering::Relaxed),
        probes: PROBES.load(Ordering::Relaxed),
    }
}

/// This process's census ([`bloomery_gpu_gates::gpu_census::census`]),
/// counted.
fn census() -> Result<Vec<DeviceInfo>, GateError> {
    CENSUSES.fetch_add(1, Ordering::Relaxed);
    bloomery_gpu_gates::gpu_census::census()
}

/// One probe of a context search, counted.
fn probed() {
    PROBES.fetch_add(1, Ordering::Relaxed);
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

/// The placement a qwen3moe or qwen35moe binary's `--place` names, and why,
/// on `census` (`Place::choose_untiered`: the body serves no tier card):
/// `flag` as given when set,
/// else `a`, the plan never asked. An unset load still takes
/// [`PlaceQ3::unplaced_on`]'s pick, the same card; the caller prints
/// [`PlaceWhy::record`].
#[allow(
    dead_code,
    reason = "the CLI and the qwen3 seat choose by the rule; the e2e gates include the planner without it"
)]
pub fn choose(flag: Option<Place>, census: &[DeviceInfo]) -> Result<PlaceWhy, GateError> {
    Place::choose_untiered(flag, census, "qwen3")
}

/// The plan inputs of one of the two architectures; the qwen35moe ones,
/// which carry the typed model description too, boxed.
enum Inputs {
    Qwen3(q3::PlanInputs),
    Qwen35(Box<q35::PlanInputs>),
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

/// A qwen35moe file's `slots` resident sequences, each taking checkpoints
/// priced by the body's own derivation (`checkpoint_price35`) at the ubatch
/// `ubatch` the load is asked for, which sets their spacing: the seat's.
#[allow(
    dead_code,
    reason = "the serve seat's qwen35moe loads take checkpoints; the CLI and the e2e gates take none"
)]
pub fn seqs_qwen35(file: &Split, ubatch: usize, slots: usize) -> Result<Seqs, GateError> {
    let spec = model::arch::qwen35moe::spec::read(file)?.spec;
    Ok(Seqs::priced(slots, checkpoint_price35(&spec, ubatch)?))
}

/// The card reserve of a load's resident sequences past the live one
/// ([`laid`]): each one's stores that no position moves.
pub const SLOTS_RESERVE: &str = "resident slots";

/// The card reserve of a media load's rotation tables, one for every resident sequence ([`laid`]):
/// a placed plan's KV term counts a text load's bytes a position, not the table each sequence's
/// image positions rewrite (`gpu::media::SeqRope`).
pub const TABLES_RESERVE: &str = "sequence rotation tables";

/// The layers the model loads: a qwen35moe file's trunk (its next-token
/// layers are never loaded).
fn layers_of(inputs: &Inputs) -> usize {
    match inputs {
        Inputs::Qwen3(i) => i.hp.n_layer,
        Inputs::Qwen35(i) => i.hp.n_trunk,
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

/// The machine a placed load of `inputs`'s file lays out on `spec`'s card
/// over `arena` ([`q3::machine`]), with the rows of what `seqs` holds past the
/// plan's one-sequence terms: on the card, a [`SLOTS_RESERVE`] of every
/// resident sequence's fixed stores ([`fixed_seq_bytes`]) but the live one's
/// — the plan's KV term at the total counts the per-position rows of every
/// slot's share and the fixed stores once — none where they are 0; on the
/// host, every sequence's checkpoints when they take them, through the lib's
/// one owner ([`reserve_checkpoints`]). The one owner of a placed
/// load's machine, so a relaid machine ([`PlaceQ3::machine_at`]) keeps the
/// rows. The checkpoints are priced at `slot_ctx` positions a slot, the
/// context the machine is laid for; on a media load each slot's rotation
/// table holds as many positions.
fn laid(
    spec: CardSpec,
    inputs: &Inputs,
    arena: u64,
    seqs: Seqs,
    slot_ctx: u64,
    media: Media,
) -> Result<Machine, GateError> {
    let mut machine = q3::machine(spec, layers_of(inputs), arena);
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
    image_plan::reserve_on(&mut machine, media.encoder);
    Ok(machine)
}

/// The rows each of `slots` slots loads with when they split `ctx`
/// positions: `ctx / slots`, the split's floor.
fn slot_ctx(ctx: usize, slots: usize) -> usize {
    ctx / slots.max(1)
}

/// What a media load adds to the machines and verdicts of a load ([`laid`], [`whole_on`]): the
/// image encoder's bytes when its tower sits on the stage card
/// (`vision_seat::EncoderSeat::reserve`, entered by [`image_plan::reserve_on`]), and the width
/// in f32 of the rotation table every sequence holds on a media load (`Open35::media`;
/// [`image_plan::table_bytes_per_position`] a position). A text load adds neither.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Media {
    encoder: Option<u64>,
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
    /// `place` resolved against `census`, refused by name when it names an
    /// expert tier card.
    fn resolve(place: Place, census: &[DeviceInfo]) -> Result<Place, GateError> {
        let place = place.on(census)?;
        place.serves("qwen3moe", 0)?;
        Ok(place)
    }

    /// A qwen3moe file's placed load at `ctx` positions on `place`'s card,
    /// its cache planes in the `kv` format ([`KvQ8`]), resolved against one
    /// census of this process's devices.
    pub fn qwen3(file: &Split, place: Place, ctx: usize, kv: KvQ8) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen3_on(file, place, &census()?, ctx, kv)
    }

    /// [`PlaceQ3::qwen3`] resolved against `census`.
    fn qwen3_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        ctx: usize,
        kv: KvQ8,
    ) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census)?;
        let mut inputs = q3::PlanInputs::read(file)?;
        if kv == KvQ8::Q8 {
            inputs.kv = inputs.kv.in_q8();
        }
        let inputs = Inputs::Qwen3(inputs);
        let arena = Qwen3moeModel::placed_arena_bytes(file, ctx)?;
        let media = Media::default();
        let machine = laid(
            place.card_specs()?[0],
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
        reason = "the CLI's and the e2e gates' load; the serve seat's holds its slots (qwen35_seqs)"
    )]
    pub fn qwen35(file: &Split, place: Place, o: Open35) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen35_on(file, place, &census()?, o, Seqs::ONE, None)
    }

    /// [`PlaceQ3::qwen35`] of a load that holds `seqs`, each of its slots
    /// opened under `o` (a slot's share of the total, [`share`]), its machine
    /// carrying their rows ([`laid`]) and, on a media load (`o.media`), the
    /// tables of its sequences and the image encoder's `encoder` bytes when
    /// the tower sits on this card: the serve seat's.
    #[allow(
        dead_code,
        reason = "the serve seat's resident slots; the CLI and the e2e gates load one sequence"
    )]
    pub fn qwen35_seqs(
        file: &Split,
        place: Place,
        o: Open35,
        seqs: Seqs,
        encoder: Option<u64>,
    ) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen35_on(file, place, &census()?, o, seqs, encoder)
    }

    /// [`PlaceQ3::qwen35_seqs`] resolved against `census`.
    fn qwen35_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        o: Open35,
        seqs: Seqs,
        encoder: Option<u64>,
    ) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census)?;
        let mut inputs = q35::PlanInputs::describe(file)?;
        if o.kv == KvQ8::Q8 {
            inputs.kv = inputs.kv.in_q8();
        }
        let media = Media {
            encoder,
            tables: o.media.then_some(inputs.hp.rope_dims),
        };
        let inputs = Inputs::Qwen35(Box::new(inputs));
        let arena = Qwen35moeModel::placed_arena_bytes(file, o)?;
        let machine = laid(
            place.card_specs()?[0],
            &inputs,
            arena,
            seqs,
            u64::try_from(o.ctx)?,
            media,
        )?;
        Ok(PlaceQ3 {
            place,
            arch: "qwen35moe",
            inputs,
            kv: o.kv,
            seqs,
            media,
            machine,
        })
    }

    /// The plan at `ctx` positions under `levers` (the card budget), every
    /// refusal by name with its terms.
    pub fn plan(&self, ctx: usize, levers: &PlanLevers) -> Result<Plan<'_>, GateError> {
        self.plan_over(&self.machine, ctx, levers)
    }

    /// [`Self::plan`] over a machine [`Self::machine_at`] relayed at another
    /// context.
    fn plan_over<'a>(
        &'a self,
        machine: &'a Machine,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, GateError> {
        let ctx = u64::try_from(ctx)?;
        Ok(match &self.inputs {
            Inputs::Qwen3(i) => i.plan(machine, ctx, levers)?,
            Inputs::Qwen35(i) => i.plan_rule(machine, ctx, levers, card_routed)?,
        })
    }

    /// The same placed load's machine at a probe's context: the place, the
    /// inputs and the sequences this one resolved and read stay — the census
    /// and the file's headers a context search's every probe retook before —
    /// and the machine is relaid over the arena the context sets, the
    /// checkpoints a slot of `slot_ctx` positions pins and, on a media load,
    /// a slot's rotation table, its ctx-dependent terms.
    fn machine_at(&self, arena: u64, slot_ctx: u64) -> Result<Machine, GateError> {
        laid(
            self.place.card_specs()?[0],
            &self.inputs,
            arena,
            self.seqs,
            slot_ctx,
            self.media,
        )
    }

    /// The routed-expert bytes the plan over `machine` at `ctx` (a total) holds
    /// on the card and on the NVMe tier: one probe of a placed `--ctx` search.
    /// The plan's own refusal is the refusal: a plan that keeps every routed
    /// expert on the card opens placed ([`open_qwen3`]) like any other, its
    /// arena the placed machine's, so the plan stands wherever it is made and
    /// the search is monotone.
    fn expert_bytes(
        &self,
        machine: &Machine,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<ExpertBytes, GateError> {
        let plan = self.plan_over(machine, ctx, levers)?;
        Ok(ExpertBytes {
            card: plan
                .cards
                .first()
                .ok_or("a plan with no card")?
                .expert_bytes,
            nvme: plan.host.nvme_expert_bytes,
        })
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
        let kv = kv_bytes(&layers, &self.inputs, (self.kv, self.media), slots, ctx);
        whole_on(
            self.place.card_specs()?[0],
            inputs_model(&self.inputs),
            layers.end,
            kv,
            (
                Seqs { slots, ..self.seqs },
                u64::try_from(slot_ctx(usize::try_from(ctx)?, slots))?,
                self.media.encoder,
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
    /// the most one layer keeps on the card, the card's free bytes when the
    /// census read them, and its device. `why` names what made a run with
    /// `--place` unset take this placed plan.
    pub fn record(&self, plan: &Plan<'_>, why: Option<&str>) -> Record {
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
        let r = match self.machine.cards.first().and_then(|c| c.free_bytes) {
            Some(free) => r.u("card_free", free),
            None => r,
        };
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
        r.csv("devices", record::plan_devices(plan.machine))
            .w("cuda_order", record::cuda_order())
    }
}

/// The allocator's bytes of the cache the body of `inputs`'s model allocates
/// for `slots` resident sequences of a load asked for `ctx` positions: the
/// seat splits the total across its slots, each `ctx / slots` rows (never
/// past it: a cache row is the granularity, the floor exact), and every
/// slot, layer by layer, holds the planes this function lists through the
/// card's allocation granule ([`placement::allocator_bytes`]) — the engine allocates
/// each plane its own buffer per layer (`KvPlanes::new`, mirrored by
/// [`gqa_planes`]) and each added slot repeats the set
/// ([`Slots::new_seq`]), so the verdict counts the granules those
/// allocations round to, not the raw sum a `KvBytes` layer gives. One slot
/// is the one-sequence load, plus that rounding. `slots` is at least one;
/// the raw plane bytes of the list sum to what `layer_bytes` gives over the
/// same layers.
fn kv_bytes(
    layers: &std::ops::Range<usize>,
    inputs: &Inputs,
    (kv, media): (KvQ8, Media),
    slots: usize,
    ctx: u64,
) -> u64 {
    let rows = ctx / slots as u64;
    let mut planes = Vec::new();
    for _ in 0..slots {
        for l in layers.clone() {
            match inputs {
                Inputs::Qwen3(i) => {
                    gqa_planes(&mut planes, i.hp.n_head_kv, i.hp.head_dim, kv, rows);
                }
                Inputs::Qwen35(i) => match i.hp.kinds.get(l) {
                    Some(Kind::Attention) => {
                        gqa_planes(&mut planes, i.hp.n_head_kv, i.hp.head_dim, kv, rows);
                    }
                    // The shape of the file's own delta layers, its lengths
                    // the two buffers `RecStore::new` zeroes; the head map
                    // enters neither length, and a qwen3.6 delta layer is
                    // the tiled one (`delta_shape` of the gpu side's plan).
                    Some(Kind::DeltaRule) => {
                        let shape = LinearShape {
                            n_k: i.hp.k_heads,
                            n_v: i.hp.v_heads,
                            map: KHeadMap::Tiled,
                        };
                        // One state lane, `Body35`'s `LANES`.
                        planes.push(4 * shape.state_len() as u64);
                        planes.push(4 * shape.ring_len() as u64);
                    }
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
    placement::allocator_bytes(GRANULE, planes)
}

/// The GQA cache planes' bytes at `rows` positions onto `planes`, in the
/// order the body allocates them: f16's `k` and `v`, q8_0's `kq,kd,vq,vd` —
/// the buffers `KvPlanes::new` zeroes, their lengths and widths
/// (`q8_plane_lens`'s u32 words and u16 scales) that owner's own.
fn gqa_planes(planes: &mut Vec<u64>, n_kv: usize, head: usize, kv: KvQ8, rows: u64) {
    match kv {
        KvQ8::F16 => {
            let bytes = n_kv as u64 * rows * head as u64 * 2;
            planes.push(bytes);
            planes.push(bytes);
        }
        KvQ8::Q8 => {
            let rows = usize::try_from(rows).expect("a cache of usize rows");
            let (words, scales) = q8_plane_lens(head, n_kv, rows);
            planes.push(4 * words as u64);
            planes.push(2 * scales as u64);
            planes.push(4 * words as u64);
            planes.push(2 * scales as u64);
        }
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

    /// The verdict at `ctx` positions in one line: the card, the answer,
    /// the need term by term and what the card had.
    fn line(&self, ctx: usize) -> String {
        match self {
            Whole::Fits(n) | Whole::Short(n) => {
                format!("whole fit at ctx {ctx}: {n}{}", image_plan::note(&n.card))
            }
            Whole::Refused(why) => format!("whole fit at ctx {ctx}: {why}"),
        }
    }

    /// The refusal of a whole load this verdict does not take, of a model
    /// with no routed expert to move to the host tier.
    fn dense_refusal(&self, ctx: usize) -> GateError {
        format!(
            "{}: a dense backbone has no routed experts to move to the host tier",
            self.line(ctx)
        )
        .into()
    }
}

/// The whole-fit verdict of a load of the whole of `model` — `layers`
/// layers, `kv` bytes of cache — on `spec`'s card: the card the plan lays
/// out for the program ([`q3::machine`], its step arenas' scratch), on a
/// unified machine with the checkpoints `seqs` pin at `slot_ctx` positions a
/// slot ([`reserve_checkpoints`], `Machine::whole_card`) and the image
/// `encoder`'s bytes when its tower sits on the card ([`image_plan::reserve_on`],
/// the reserve a placed plan carries), and what the program holds past it
/// (`load`, or
/// the program's refusal), sized by `placement::whole_need`, the verdict's
/// one owner. A tensor no card format loads is a refusal; any other planner
/// error is the error.
fn whole_on(
    spec: CardSpec,
    model: &ModelTensors,
    layers: usize,
    kv: u64,
    (seqs, slot_ctx, encoder): (Seqs, u64, Option<u64>),
    load: Result<WholeLoad, String>,
) -> Result<Whole, GateError> {
    let refused = |why: &dyn std::fmt::Display| {
        Whole::Refused(format!("the whole load on {}: refused: {why}", spec.name))
    };
    let load = match load {
        Ok(l) => l,
        Err(why) => return Ok(refused(&why)),
    };
    let mut machine = q3::machine(spec, layers, 0);
    reserve_checkpoints(&mut machine, seqs, slot_ctx)?;
    image_plan::reserve_on(&mut machine, encoder);
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
    unplaced_qwen3_slots(file, ctx, 1, kv)
}

/// What a run with `--place` unset loads, decided before any load on one
/// census reading, on `a`'s card ([`PlaceQ3::unplaced_on`]): the whole
/// model there (`Whole` — no plan, no record) while the whole-fit verdict at
/// `ctx` for `slots` resident sequences takes it ([`whole_on`]: the qwen3moe
/// body's whole load ([`qwen3_whole`]), the cache [`kv_bytes`] counts at the
/// slots' shares); else the placed plan there (`Placed`). The verdict's line
/// goes to stderr; a file with no routed experts the verdict does not take
/// is refused by name.
#[allow(
    dead_code,
    reason = "the serve seat decides its load at the total for its --parallel slots; the CLI and the e2e gates load one sequence"
)]
pub fn unplaced_qwen3_slots(
    file: &Split,
    ctx: usize,
    slots: usize,
    kv: KvQ8,
) -> Result<Unplaced, GateError> {
    let census = census()?;
    let probe = PlaceQ3::qwen3_on(file, PlaceQ3::unplaced_on(&census)?, &census, ctx, kv)?;
    let whole = probe.whole_at(u64::try_from(ctx)?, slots, qwen3_whole(file))?;
    probe.unplaced(ctx, &whole)
}

pub use bloomery_gpu_gates::generate::trained_ctx;

/// The constants of the unset `--ctx` rule a seat supplies
/// (`placement::ctx::default_slot_ctx`): the base context a slot takes, the
/// step the answer moves in and the card expert bytes it may trade for
/// positions.
#[derive(Clone, Copy, Debug)]
pub struct SlotRule {
    pub base: usize,
    pub step: usize,
    pub margin: u64,
}

/// A default context and the load it was searched over.
#[allow(
    dead_code,
    reason = "the serve seat reads the answer; the CLI and the e2e gates include the planner without it"
)]
#[derive(Clone, Copy, Debug)]
pub struct Defaulted {
    pub slot: SlotCtx,
    /// `whole` or `placed`: the load the probes planned.
    pub basis: &'static str,
}

/// The routed-expert bytes of every stack a whole load holds on the card: the
/// file's own bytes of the layers the model loads.
fn routed_bytes(model: &ModelTensors) -> u64 {
    model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts && t.layer.is_none_or(|l| l < model.layers))
        .map(|t| t.file_bytes)
        .sum()
}

/// What a context search sizes the program by at a probe's total: the arena
/// a placed machine carries past the m = 1 scratch, and what a whole load
/// holds past its cache.
struct Sizing<'a> {
    arena: &'a dyn Fn(usize) -> Result<u64, GateError>,
    load: &'a dyn Fn(usize) -> Result<WholeLoad, String>,
}

/// The slot context a run with `--ctx` unset takes
/// (`placement::ctx::default_slot_ctx` under `rule`, the file's `cap`): the
/// probes plan `slots` resident sequences of a slot's context each, the load
/// counted at the total they hold, over one resolved `probe` — its place,
/// inputs and sequences taken once, the machine relaid per probe, one census
/// reading for the whole search. The load is the whole model on `a`'s card
/// while `whole` — no `--place`, and the whole-fit verdict at the base takes
/// it ([`PlaceQ3::unplaced_on`]) — its every routed expert on the card at
/// every context (the verdict bounds the fit alone); else the placed plan on
/// the probe's card, whose card and NVMe expert bytes the margin counts.
fn default_over(
    probe: &PlaceQ3,
    (cap, rule): (usize, SlotRule),
    (slots, levers): (usize, &PlanLevers),
    whole: bool,
    sizing: &Sizing<'_>,
) -> Result<Defaulted, GateError> {
    let held = |ctx: usize| -> Result<ExpertBytes, GateError> {
        probed();
        let total = ctx.saturating_mul(slots);
        if whole {
            let verdict = probe.whole_at(u64::try_from(total)?, slots, (sizing.load)(total))?;
            if !verdict.fits() {
                return Err(verdict.line(total).into());
            }
            return Ok(ExpertBytes {
                card: routed_bytes(inputs_model(&probe.inputs)),
                nvme: 0,
            });
        }
        let machine = probe.machine_at((sizing.arena)(total)?, u64::try_from(ctx)?)?;
        probe.expert_bytes(&machine, total, levers)
    };
    let slot = placement::ctx::default_slot_ctx(cap, rule.base, rule.step, rule.margin, &held)?;
    Ok(Defaulted {
        slot,
        basis: if whole { "whole" } else { "placed" },
    })
}

/// Whether the whole load takes `rule`'s base at `slots` resident sequences:
/// the verdict on `probe`'s card, one probe of the search it decides.
fn whole_holds_base(
    probe: &PlaceQ3,
    rule: SlotRule,
    slots: usize,
    sizing: &Sizing<'_>,
) -> Result<bool, GateError> {
    probed();
    let total = rule.base.saturating_mul(slots);
    Ok(probe
        .whole_at(u64::try_from(total)?, slots, (sizing.load)(total))?
        .fits())
}

/// The slot context a run of a qwen3moe file with `--ctx` unset takes for
/// `slots` resident sequences ([`default_over`]): on `a`'s card
/// ([`PlaceQ3::unplaced_on`]) the whole load while its whole-fit verdict
/// takes the base, else the placed plan on `place`'s card (`a`'s with no
/// `--place`) under `levers`. The cap is the file's trained context, the
/// rule's base when it states none.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn default_ctx_qwen3(
    file: &Split,
    place: Option<Place>,
    rule: SlotRule,
    (slots, levers): (usize, &PlanLevers),
    kv: KvQ8,
) -> Result<Defaulted, GateError> {
    let census = census()?;
    let at_base = rule.base.saturating_mul(slots);
    let on = PlaceQ3::unplaced_on(&census)?;
    let probe = PlaceQ3::qwen3_on(file, place.unwrap_or(on), &census, at_base, kv)?;
    let sizing = Sizing {
        arena: &|total| Ok(Qwen3moeModel::placed_arena_bytes(file, total)?),
        load: &|_| qwen3_whole(file),
    };
    let whole = place.is_none() && whole_holds_base(&probe, rule, slots, &sizing)?;
    let cap = trained_ctx(file).unwrap_or(rule.base);
    default_over(&probe, (cap, rule), (slots, levers), whole, &sizing)
}

/// [`default_ctx_qwen3`] of a qwen35moe file, `o` giving its non-context
/// terms, for the load's `seqs`: every plan carries their rows ([`laid`]),
/// its arena the load each slot's share opens at ([`share`]), and the whole
/// verdict counts every slot's stores at its share of the total
/// ([`kv_bytes`]: the attention planes at a slot's rows and the delta layers'
/// fixed stores, once a slot, and on a media load a slot's rotation table);
/// the image `encoder`'s bytes are a reserve of each plan.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn default_ctx_qwen35(
    file: &Split,
    o: &Open35,
    place: Option<Place>,
    rule: SlotRule,
    (seqs, levers, encoder): (Seqs, &PlanLevers, Option<u64>),
) -> Result<Defaulted, GateError> {
    let census = census()?;
    let slots = seqs.slots;
    let at_base = rule.base.saturating_mul(slots);
    let on = PlaceQ3::unplaced_on(&census)?;
    let probe = PlaceQ3::qwen35_on(
        file,
        place.unwrap_or(on),
        &census,
        share(o, at_base, slots),
        seqs,
        encoder,
    )?;
    let sizing = Sizing {
        arena: &|total| {
            Ok(Qwen35moeModel::placed_arena_bytes(
                file,
                share(o, total, slots),
            )?)
        },
        load: &|total| qwen35_whole(file, share(o, total, slots)),
    };
    let whole = place.is_none() && whole_holds_base(&probe, rule, slots, &sizing)?;
    let cap = trained_ctx(file).unwrap_or(rule.base);
    default_over(&probe, (cap, rule), (slots, levers), whole, &sizing)
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3`] decides it,
/// of a qwen35moe file under `o`: one sequence taking no checkpoints
/// ([`unplaced_qwen35_slots`]), as the CLI's own runs do.
pub fn unplaced_qwen35(file: &Split, o: &Open35) -> Result<Unplaced, GateError> {
    unplaced_qwen35_slots(file, o, Seqs::ONE, None)
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3_slots`]
/// decides it, of a qwen35moe file under `o` at `o.ctx` positions — the
/// total its `seqs` split: the whole-fit verdict of every slot's stores at
/// its share ([`default_ctx_qwen35`]'s count) over the program's own arena and
/// reserve as a slot's load gives them ([`qwen35_whole`] at [`share`]); the
/// placed plan on `a`'s card, carrying the sequences' rows ([`laid`]), when
/// it does not take it; the image `encoder`'s bytes are in the verdict and in
/// the plan.
pub fn unplaced_qwen35_slots(
    file: &Split,
    o: &Open35,
    seqs: Seqs,
    encoder: Option<u64>,
) -> Result<Unplaced, GateError> {
    let census = census()?;
    let at = share(o, o.ctx, seqs.slots);
    let probe = PlaceQ3::qwen35_on(
        file,
        PlaceQ3::unplaced_on(&census)?,
        &census,
        at,
        seqs,
        encoder,
    )?;
    let whole = probe.whole_at(u64::try_from(o.ctx)?, seqs.slots, qwen35_whole(file, at))?;
    probe.unplaced(o.ctx, &whole)
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
        Unplaced::Whole(card) => Ok(Qwen3moeModel::open(
            Gpu::open_card(card.name, card.device)?,
            file,
            opts,
        )?),
        Unplaced::Placed(q) => {
            let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
            emit(q.record(&plan, Some(WHY_NOT_WHOLE)));
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
        Unplaced::Whole(card) => Ok(Qwen35moeModel::open(
            Gpu::open_card(card.name, card.device)?,
            file,
            o,
        )?),
        Unplaced::Placed(q) => {
            let plan = q.plan(o.ctx, &PlanLevers::from_levers(levers)?)?;
            emit(q.record(&plan, Some(WHY_NOT_WHOLE)));
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
/// their rows ([`PlaceQ3::qwen35_seqs`]). One slot is [`open_qwen35`]
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
    let available = workstation::host_available()?;
    if pin > available {
        return Err(format!(
            "{head}, past the {available} B the host has available (MemAvailable): give a smaller \
             --parallel"
        )
        .into());
    }
    eprintln!("{head}, of {available} B the host has available");
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
