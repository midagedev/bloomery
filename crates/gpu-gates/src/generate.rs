//! [`Generator`] — the load-and-step loop a decode CLI drives: open the file
//! by a placement, load it, capture the step graph before the prompt, feed
//! the prompt one real step per token, then one step per chosen token. It
//! owns the [`GpuModel`] and nothing else: no sampling, no text.
//!
//! Generic over the chain body. The body's own entry (`body::open` of the
//! V4.1 device crate, with the levers the binary parsed) and whatever the
//! caller checks on the loaded body come in as arguments, because this
//! library's source does not name a device
//! crate: named here, it would be linked, device bundle and all, into every
//! gate built with the feature, the ones that launch none of its kernels
//! too (the reason [`crate::ds41_meta::RopeMeta::read`] takes its ropes as
//! functions). The plan line is the caller's for the same reason: the plan's
//! inputs are the architecture's type.
//!
//! The diagnosis flags every generate binary shares ([`Diag`]) live here
//! too, generic over the session (`runtime::Target`): the serve's pick and
//! feed ([`NoEog`], [`ServeFeed`]), the rows behind the first tokens
//! ([`TopRows`]), runs chained by the serve seat's reset ([`repeat_runs`]),
//! and the stage card's copy of the slot map — read back against the host
//! map and the residency ledger ([`Residence`], [`slot_table`]), dumped to a
//! file ([`write_table`], [`dump_table`]) and loaded as a fixed placement
//! ([`place_table`], [`card_table`]). A binary hands in what only its body
//! names: the card's copy of the map, its host tier, its plan.

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bloomery_gpu::host::node::node_cards;
use bloomery_gpu::host::slots::{HOST, MAX_TIERS, SlotMap};
use bloomery_gpu::host::swap::{CardTable, SwapMachine, TableCheck};
use bloomery_gpu::host::tier::TierCard;
use bloomery_gpu::host::{HostExperts, HostTier};
use bloomery_gpu::lane;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::placement::workstation::{
    self, CardSpec, DeviceId, DeviceInfo, Pick, PickError, PickWhy, TierBatchBytes,
};
use model::placement::{self, Device, ExpertList, Machine, Plan, Reserve, Role, Row, card_budget};
use runtime::{Advance, Committed, Out, Target, Want};
use tokenizer::Tokenizer;

use crate::record::{self, Record};
use crate::{GateError, ref_model_path};

/// A load's lane line: the stage card's name and the rate [`lane_of`] took.
pub type Lane = (&'static str, lane::LaneRate);

/// The lane's rate on `stage`, taken before a plan reads the room it plans
/// by: [`lane::probe_card`] drops its extents and its handle first. A load
/// prints it beside its `load` record ([`record::lane`]).
pub fn lane_of(stage: CardSpec) -> Result<Lane, GateError> {
    Ok((
        stage.name,
        lane::probe_card(stage.name, stage.device, lane::PROBE_BYTES)?,
    ))
}

/// The lane's rate on `place`'s stage card ([`lane_of`]).
pub fn lane_at(place: Place) -> Result<Lane, GateError> {
    lane_of(place.card_specs()?[0])
}

/// The VMM page of `stage`'s device ([`bloomery_gpu::allocation_granularity`],
/// the page the room allocator backs stores by), read without opening a
/// context or a kernel bundle: the page a plan reserves the stores' rounding
/// by (`PlanInputs::reserve_room`).
pub fn stage_page(stage: CardSpec) -> Result<u64, GateError> {
    let ordinal = Gpu::resolve_card(stage.name, stage.device)?;
    Ok(u64::try_from(bloomery_gpu::allocation_granularity(
        ordinal,
    )?)?)
}

/// Which placement the engine loads by: a stage card that runs every layer
/// and the head, then the expert tier cards beside the host tier. The flag
/// word is an alias — `a` (the stage on the largest visible card), `bp`
/// (that stage and the next-largest as its tier), `gate` (the bytes of a
/// 3090, alone) — or the list `<stage>[+<tier>…]` of card names and device
/// ordinals (`workstation::word_picks`); a list of an alias's cards on this
/// workstation by name prints as that alias.
///
/// A parsed placement is this workstation's by name, with no census: its
/// cards are `workstation::CARDS` ([`workstation::workstation_spec`]), each
/// opened by its name — the gates' machines. A binary resolves it against
/// this process's devices ([`Place::on_host`]) before it plans: then each
/// card is the spec of the device it picked and opens on that device.
///
/// `tp` is tensor parallel: this process's one card is rank 0 and
/// `BLOOMERY_TP_NODES` names the others. It is no card list and resolves
/// against no census here ([`Place::on`] refuses it): a family that plans
/// it does so before [`Place::choose`] ([`Place::tp`]).
#[derive(Debug, Clone, Copy)]
pub struct Place {
    /// The flag value as the records print it.
    word: &'static str,
    /// How each card is found, the stage then the tiers in tier order; the
    /// first `n` are the placement's.
    picks: [Pick; 1 + MAX_TIERS],
    n: u8,
    /// Each card's spec: this workstation's by name before
    /// [`Place::on`] (`None` for an ordinal, which names no card without a
    /// census), the device's after.
    specs: [Option<CardSpec>; 1 + MAX_TIERS],
    resolved: bool,
    /// Each tier card is on a host of its own, so the plan holds it in its
    /// node's pool and not this process's ([`Place::on_nodes`]).
    own_pools: bool,
}

/// Two placements are one when they print the same word: the word is what
/// the records and the binaries' checks read (`place == Place::Gate`), the
/// same before and after [`Place::on`].
impl PartialEq for Place {
    fn eq(&self, other: &Place) -> bool {
        self.word == other.word
    }
}

impl Eq for Place {}

/// The alias `ALIASES[i]` as a placement, this workstation's by name.
const fn alias(i: usize) -> Place {
    let (word, list) = workstation::ALIASES[i];
    let mut picks = [Pick::Rank(0); 1 + MAX_TIERS];
    let mut specs = [None; 1 + MAX_TIERS];
    let mut k = 0;
    while k < list.len() {
        picks[k] = list[k];
        specs[k] = workstation::workstation_spec(list[k]);
        k += 1;
    }
    Place {
        word,
        picks,
        n: list.len() as u8,
        specs,
        resolved: false,
        own_pools: false,
    }
}

/// The step every context an unset `--ctx` searches rounds to: a round power
/// of two a cache row count stays readable in.
pub const CTX_GRAN: usize = 1024;

/// The file's trained context (`<arch>.context_length`), `None` when the
/// file states none.
pub fn trained_ctx(file: &Split) -> Option<usize> {
    file.arch_get_u64("context_length")
        .and_then(|v| usize::try_from(v).ok())
}

/// The bounds a seat names beside the file's trained context
/// ([`placement::ctx::CtxLimit`]): what its kernels index (`kernel`) and what
/// its references were checked to (`verified`), each a position count and
/// its name; `None` where the seat names none.
pub type SeatBounds = (Option<(u64, &'static str)>, Option<(u64, &'static str)>);

/// The steps every serving seat's context moves in
/// ([`placement::ctx_ask::choose`]): the base a slot, the least context an
/// unset `--ctx` searches and the least share searched after a set one is
/// refused, and [`CTX_GRAN`], the rounding of every context searched.
pub const SEAT_STEPS: placement::ctx_ask::Steps = placement::ctx_ask::Steps {
    floor: 4096,
    step: CTX_GRAN as u64,
};

/// The context a serving seat's `slots` serve and the plan made at it, the
/// one owner every seat calls ([`placement::ctx_ask::choose`] over
/// [`SEAT_STEPS`]); `plan_at` is the seat's plan at a slot's context, read by
/// [`placement::ctx_ask::probe_of`]. The model's limit is the least of the
/// file's trained context and the bounds the seat names (`kernel`,
/// `verified`; [`placement::ctx::CtxLimit`]): a set `--ctx` (`set`, the total
/// the slots split) whose share passes it stops by name before any plan
/// ([`placement::ctx::PastLimit`]), and an unset one asks the limit and fits
/// under it. The caller prints the chosen line ([`placement::ctx_ask::Chosen::line`]).
///
/// # Errors
/// The file's missing `context_length`, [`placement::ctx::PastLimit`], the
/// choice's refusal ([`placement::ctx_ask::AskError`]) in its words, or the
/// seat's own plan error.
pub fn choose_ctx<P, E: Into<GateError> + std::fmt::Display>(
    file: &Split,
    asked: (Option<usize>, usize),
    bounds: SeatBounds,
    plan_at: impl FnMut(u64) -> Result<placement::ctx_ask::Probe<P>, E>,
) -> Result<placement::ctx_ask::Chosen<P>, GateError> {
    choose_ctx_of(context_length(file)?, asked, bounds, plan_at)
}

/// [`choose_ctx`] for a seat that holds the file's trained context
/// already (`trained`, its `context_length`).
///
/// # Errors
/// As [`choose_ctx`].
pub fn choose_ctx_of<P, E: Into<GateError> + std::fmt::Display>(
    trained: u64,
    (set, slots): (Option<usize>, usize),
    bounds: SeatBounds,
    plan_at: impl FnMut(u64) -> Result<placement::ctx_ask::Probe<P>, E>,
) -> Result<placement::ctx_ask::Chosen<P>, GateError> {
    use placement::ctx_ask::{Ask, AskError, Limit, choose};
    let share = set.and_then(|total| total.checked_div(slots));
    let bounds = check_share(trained, share, bounds)?;
    let ask = match set {
        Some(total) => Ask::Set {
            total: u64::try_from(total)?,
        },
        None => Ask::Default,
    };
    let (positions, source) = bounds.limit();
    let chosen = choose(ask, slots, Limit { positions, source }, SEAT_STEPS, plan_at).map_err(
        |e| -> GateError {
            match e {
                AskError::Plan(e) => e.into(),
                AskError::FloorShort { floor, short, .. } => Box::new(FloorShort {
                    floor,
                    short,
                    text: e.to_string(),
                }),
                AskError::DoesNotFit {
                    slots,
                    short,
                    largest,
                    ..
                } => Box::new(CtxDoesNotFit {
                    largest_total: largest.and_then(|share| share.checked_mul(slots as u64)),
                    largest_slot: largest,
                    short,
                    text: e.to_string(),
                }),
                other => other.to_string().into(),
            }
        },
    )?;
    crate::start::note_ctx(
        chosen.slot_ctx,
        chosen.slots as u64,
        chosen.rule.word(),
        chosen.limit.positions,
    );
    Ok(chosen)
}

/// A default context whose floor the machine cannot hold
/// ([`placement::ctx_ask::AskError::FloorShort`]'s words), kept as its own
/// type so a caller can read the floor and plan there for the refusal itself.
#[derive(Debug)]
pub struct FloorShort {
    /// The least context the default asked.
    pub floor: u64,
    /// The plan at `floor` and the memory it is short of.
    pub short: placement::ctx_ask::Short,
    text: String,
}

impl std::fmt::Display for FloorShort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::error::Error for FloorShort {}

/// Whether a seat's context search refused `e` because the machine holds no
/// context it may take: a default short at its floor ([`FloorShort`]), a set
/// context it cannot hold ([`CtxDoesNotFit`]), a load short of host room
/// ([`placement::workstation::HostShort`], including a discrete host's
/// folder read-ahead beside the draft's reserves), or a host room under a
/// floor the draft raises and no smaller context lowers
/// ([`placement::PlacementError::host_floor_short`]: the NVMe expert tier's
/// floor and arena, and the draft's own host experts), read through the
/// family's refusal `E`, anywhere in the source chain. Every other refusal is
/// the plan's own error.
#[must_use]
pub fn holds_no_ctx<E>(e: &GateError) -> bool
where
    E: placement::ctx_ask::PlanRefusal + std::error::Error + 'static,
{
    use placement::ctx_ask::PlanRefusal;
    std::iter::successors(Some(&**e as &dyn std::error::Error), |e| e.source()).any(|e| {
        e.is::<FloorShort>()
            || e.is::<CtxDoesNotFit>()
            || e.is::<placement::workstation::HostShort>()
            || e.downcast_ref::<E>()
                .and_then(PlanRefusal::placement)
                .or_else(|| e.downcast_ref::<placement::PlacementError>())
                .and_then(placement::PlacementError::host_floor_short)
                .is_some()
    })
}

/// What an unset draft's yield to the context chose ([`draft_yield`]).
#[derive(Debug)]
pub enum DraftYield<D, P> {
    /// The drafted plan serves.
    Keep(D),
    /// The draft goes off: the plain plan serves, and the cost names both
    /// contexts.
    Yield(P, bloomery_levers::RungCost),
}

/// The serve seats' one rule for an unset draft's yield to the context: the
/// context comes first. Where the draft is unset (`asks`) and the drafted
/// plan fitted its context under the target (`fitted` gives its positions)
/// or holds no context it may take, a set one included ([`holds_no_ctx`],
/// zero positions), `plain`
/// plans without the draft and gives its plan, its positions and the draft's
/// card bytes there; when the plain plan serves twice the positions or more
/// ([`bloomery_levers::RungCost::of`]) the draft yields. Otherwise the drafted
/// plan, or its refusal, stands: a refusal of any other kind is never read
/// as a yield. A drafted plan that stands keeps its own context note
/// ([`crate::start::note_ctx`]), which `plain` overwrote.
///
/// # Errors
/// The drafted plan's refusal where the draft does not yield, and the plain
/// plan's.
pub fn draft_yield<E, D, P>(
    asks: bool,
    drafted: Result<D, GateError>,
    fitted: impl FnOnce(&D) -> Option<usize>,
    plain: impl FnOnce() -> Result<(P, usize, u64), GateError>,
) -> Result<DraftYield<D, P>, GateError>
where
    E: placement::ctx_ask::PlanRefusal + std::error::Error + 'static,
{
    let with = match &drafted {
        _ if !asks => None,
        Ok(d) => fitted(d),
        Err(e) if holds_no_ctx::<E>(e) => Some(0),
        Err(_) => None,
    };
    let Some(with) = with else {
        return drafted.map(DraftYield::Keep);
    };
    let noted = crate::start::noted_ctx();
    let (plain, without, bytes) = plain()?;
    match bloomery_levers::RungCost::of(
        bloomery_levers::Reserve::Draft,
        u64::try_from(with)?,
        u64::try_from(without)?,
        bytes,
    ) {
        Some(y) => Ok(DraftYield::Yield(plain, y)),
        None => {
            crate::start::restore_ctx(noted);
            drafted.map(DraftYield::Keep)
        }
    }
}

/// A set context the machine cannot hold ([`placement::ctx_ask::AskError::DoesNotFit`]'s words),
/// kept as its own type so a start can offer the largest context that does.
#[derive(Debug)]
pub struct CtxDoesNotFit {
    /// The `--ctx` total the largest share a slot holds gives over the slots; `None` when not
    /// even the floor holds.
    pub largest_total: Option<u64>,
    /// The largest share one slot holds, for a retry of `--slot-ctx`.
    pub largest_slot: Option<u64>,
    /// The plan at the asked context and the memory it is short of.
    pub short: placement::ctx_ask::Short,
    text: String,
}

impl std::fmt::Display for CtxDoesNotFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::error::Error for CtxDoesNotFit {}

/// A seat's boxed plan refusal read as a short memory
/// ([`placement::ctx_ask::PlanRefusal::short`]): the family's error `E` when
/// the box holds one, else a bare placement error's card reading, else a host
/// need's on a unified pool ([`placement::workstation::HostShort::pool_short`]),
/// else a load's refusal of a plan that pages experts, anywhere in its source
/// chain ([`placement::ctx_ask::Unpaged`]); `None` when it is none of them, or
/// not short.
#[must_use]
pub fn short_in<E>(e: &GateError) -> Option<placement::ctx_ask::Short>
where
    E: placement::ctx_ask::PlanRefusal + std::error::Error + 'static,
{
    match e.downcast_ref::<E>() {
        Some(family) => placement::ctx_ask::PlanRefusal::short(family),
        None => e
            .downcast_ref::<placement::PlacementError>()
            .and_then(placement::PlacementError::card_short)
            .or_else(|| {
                e.downcast_ref::<placement::workstation::HostShort>()
                    .and_then(placement::workstation::HostShort::pool_short)
            })
            .or_else(|| {
                std::iter::successors(Some(&**e as &dyn std::error::Error), |e| e.source())
                    .find_map(|e| e.downcast_ref::<placement::ctx_ask::Unpaged>())
                    .map(placement::ctx_ask::Unpaged::short)
            }),
    }
}

/// The trained context and optional slot share checked against the seat's
/// bounds: the model's limit ([`placement::ctx::CtxLimit`]), and a set
/// `--ctx`'s share of one slot past it stopped by name before any plan
/// ([`placement::ctx::check_slot_ctx`]).
///
/// # Errors
/// [`placement::ctx::PastLimit`].
pub fn check_share(
    trained: u64,
    share: Option<usize>,
    (kernel, verified): SeatBounds,
) -> Result<placement::ctx::CtxLimit, GateError> {
    let bounds = placement::ctx::CtxLimit {
        trained,
        kernel,
        verified,
    };
    if let Some(share) = share {
        placement::ctx::check_slot_ctx(u64::try_from(share)?, &bounds)?;
    }
    Ok(bounds)
}

/// The file's trained context (`<arch>.context_length`), refused by name when
/// the file states none: no context can be held to it.
fn context_length(file: &Split) -> Result<u64, GateError> {
    Ok(file.arch_get_u64("context_length").ok_or(
        "the file states no context_length, so the context cannot be held to the model's limit",
    )?)
}

/// A family's input to the common unset-`--place` rule ([`Place::choose`]):
/// the expert tier cards its body serves, the tier count that pays once a
/// sitting has shown the tier not slower (`None` until one has), and the card
/// file that decides it (empty: the family names none).
#[derive(Clone, Copy)]
pub struct TierRule {
    /// The tier cards the family's body serves.
    pub tiers: usize,
    /// The least tier experts that pay, where a sitting has shown they do.
    pub break_even: Option<u64>,
    /// The card file that decided [`Self::break_even`].
    pub basis: &'static str,
}

impl TierRule {
    /// A body that serves no tier card: an unset `--place` runs `a` on the
    /// largest card and no card file decides anything.
    pub const UNTIERED: TierRule = TierRule {
        tiers: 0,
        break_even: None,
        basis: "",
    };
}

/// [`Place::choose`]'s answer: the placement to run, why in the words a
/// `place unset` record prints, the tier count where the rule asked the plan
/// for it, and the rule that chose.
pub struct PlaceWhy {
    /// The placement, resolved against the census.
    pub place: Place,
    pub why: &'static str,
    /// The experts the offered placement's tier cards hold in its plan, where
    /// the rule asked ([`Place::choose`]'s `tier_of`); `None` where it did
    /// not.
    pub tier_experts: Option<u64>,
    /// The rule that chose.
    pub rule: TierRule,
}

impl PlaceWhy {
    /// The `place unset` record: the word and why always; the tier count only
    /// where the rule asked the plan for it; the rule's break-even and basis
    /// whenever it has them.
    pub fn record(&self) -> Record {
        crate::start::note_place(self.place.name(), self.why);
        let r = Record::new(&record::PLACE_UNSET)
            .w("place", self.place.name())
            .w("why", self.why);
        let r = match self.tier_experts {
            Some(n) => r.u("tier_experts", n),
            None => r,
        };
        let r = match self.rule.break_even {
            Some(e) => r.u("break_even", e),
            None => r,
        };
        match self.rule.basis {
            "" => r,
            b => r.w("basis", b),
        }
    }
}

/// A drafted seat's break-even: `per_token` is the ids a reset re-prefills in
/// the time the draft saves one reply token, so a kept prefix of fewer than
/// `⌈reply · per_token⌉` positions costs less to prefill again than keeping
/// it costs the reply's `reply` tokens.
#[derive(Clone, Copy, Debug)]
pub struct BreakEven {
    pub per_token: f64,
}

impl BreakEven {
    /// The draft's gain a token (the plain step's seconds less the drafted
    /// decode's, from their positions per second) times the ids per second a
    /// reset re-prefills at.
    pub fn new(plain_tps: f64, drafted_tps: f64, prompt_ids_per_s: f64) -> BreakEven {
        BreakEven {
            per_token: (1.0 / plain_tps - 1.0 / drafted_tps) * prompt_ids_per_s,
        }
    }

    /// The kept positions that break even with a reply of `reply` tokens
    /// through passes.
    pub fn at(self, reply: usize) -> usize {
        let r = f64::from(u32::try_from(reply).unwrap_or(u32::MAX));
        (r * self.per_token).ceil() as usize
    }
}

#[allow(
    non_upper_case_globals,
    reason = "the aliases keep the names callers match on: Place::A, Place::Gate, Place::Bp"
)]
impl Place {
    /// The serving plan: the stage on the largest visible card
    /// (`workstation::plan_a` on this workstation, the A6000).
    pub const A: Place = alias(0);
    /// The step gate's plan (`workstation::plan_gate`): the 3090's bytes, on
    /// this workstation the 3090; resolved, on the smallest visible card with
    /// the room for them.
    pub const Gate: Place = alias(1);
    /// Plan (b′): plan (a)'s stage, the next-largest card an expert tier
    /// under the host tier (`workstation::plan_bp` on this workstation, the
    /// 3090 under the A6000), holding the DSpark draft's reserve when a
    /// draft is served there.
    pub const Bp: Place = alias(2);

    /// The flag value's placement: an alias, or a list word whose parts are
    /// refused by name when there are more tiers than a slot map names
    /// (`MAX_TIERS`), a part that is no card name and no ordinal, or a name
    /// or ordinal given twice. Any card may be the stage.
    pub fn parse(v: &str) -> Result<Place, GateError> {
        match v {
            "a" => return Ok(Place::A),
            "gate" => return Ok(Place::Gate),
            "bp" => return Ok(Place::Bp),
            "tp" => {
                return Ok(Place {
                    word: "tp",
                    ..alias(0)
                });
            }
            _ => {}
        }
        let list = workstation::word_picks(v, MAX_TIERS).map_err(|e| {
            format!(
                "--place is a, gate, bp (plan (b′): the next-largest card as the largest's \
                 expert tier), tp (tensor parallel over BLOOMERY_TP_NODES) or a card list \
                 <stage>[+<tier>…] of card names and device ordinals: {e}"
            )
        })?;
        let mut picks = [Pick::Rank(0); 1 + MAX_TIERS];
        let mut specs = [None; 1 + MAX_TIERS];
        for (i, &p) in list.iter().enumerate() {
            picks[i] = p;
            specs[i] = workstation::workstation_spec(p);
        }
        let named = list.iter().all(|p| matches!(p, Pick::Name(_)));
        let alias = [Place::A, Place::Gate, Place::Bp]
            .into_iter()
            .find(|a| named && a.specs[..usize::from(a.n)] == specs[..list.len()]);
        // A list word that is no alias is spelled once per parse and kept for
        // the process: the records and the open take the name as `'static`.
        let word: &'static str = match alias {
            Some(a) => a.word,
            None => Box::leak(v.to_ascii_lowercase().into_boxed_str()),
        };
        Ok(Place {
            word,
            picks,
            n: u8::try_from(list.len())?,
            specs,
            resolved: false,
            own_pools: false,
        })
    }

    /// The placement on this process's devices ([`crate::gpu_census::census`],
    /// the driver's census with nvidia-smi's holders): [`Place::on`] of the
    /// census.
    pub fn on_host(self) -> Result<Place, GateError> {
        self.on(&crate::gpu_census::census()?)
    }

    /// The placement resolved against `census`: each card the spec of the
    /// device its pick finds ([`workstation::resolve`]), refused by name with
    /// every visible device when a pick finds none, several, or a device an
    /// earlier card is. A list of names spelled as an alias that lands on
    /// other devices than the alias does here prints as its names. With
    /// `BLOOMERY_TIER_NODES` set, the tier cards are its nodes'
    /// ([`Place::on_nodes`]). A per-card `BLOOMERY_CARD_BUDGET` or
    /// `BLOOMERY_CARD_MARGIN` naming a device `census` does not list is
    /// refused by name ([`refuse_card_lists`]), and so is one beside tier
    /// nodes, whose cards no ordinal of this process names. The margin the
    /// lever gives a card is on the resolved card's spec
    /// ([`Place::keeping_free`]). The levers are read here; [`Place::on_with`]
    /// takes them parsed.
    pub fn on(self, census: &[DeviceInfo]) -> Result<Place, GateError> {
        if self.tp() {
            return Err(
                "--place tp: a tensor-parallel run is planned by its family's own path; this one \
                 has none"
                    .into(),
            );
        }
        self.on_with(
            census,
            &bloomery_levers::card_budget()?,
            &bloomery_levers::card_margin()?,
            &bloomery_levers::tier_nodes()?,
        )
    }

    /// [`Place::on`] with the levers it reads given parsed: `budget`
    /// (`BLOOMERY_CARD_BUDGET`) and `margin` (`BLOOMERY_CARD_MARGIN`) refused
    /// against the census and the tier nodes before any card resolves, the
    /// tier cards on `nodes` when it is not empty, and `margin` on each
    /// resolved card's spec at the end of either path, so a margin reaches the
    /// machine's cards the same way whatever the cards came from.
    pub(crate) fn on_with(
        self,
        census: &[DeviceInfo],
        budget: &card_budget::CardBudget,
        margin: &card_budget::CardBudget,
        nodes: &[SocketAddr],
    ) -> Result<Place, GateError> {
        refuse_card_lists(&self.spelled(), budget, margin, census, nodes.len())?;
        let placed = if nodes.is_empty() {
            self.on_census(census)?
        } else {
            self.on_nodes(census, nodes)?
        };
        placed.keeping_free(margin)
    }

    /// The placement on `census` alone ([`Place::on`]'s path with no tier
    /// nodes): each pick resolved, the word printed by its names where the
    /// devices are not the alias's.
    fn on_census(self, census: &[DeviceInfo]) -> Result<Place, GateError> {
        let resolved = workstation::resolve(self.picks(), census)
            .map_err(|e| format!("--place {}: {e}", self.spelled()))?;
        let mut specs = [None; 1 + MAX_TIERS];
        for (i, s) in resolved.iter().enumerate() {
            specs[i] = Some(*s);
        }
        let devices = |s: &[CardSpec]| s.iter().map(|c| c.device).collect::<Vec<_>>();
        let mut word = self.word;
        if let Some((_, alias)) = workstation::ALIASES.iter().find(|(w, _)| *w == self.word)
            && *alias != self.picks()
            && workstation::resolve(alias, census)
                .ok()
                .is_none_or(|theirs| devices(&theirs) != devices(&resolved))
        {
            let names: Vec<String> = resolved
                .iter()
                .map(|s| s.name.to_ascii_lowercase())
                .collect();
            word = Box::leak(names.join("+").into_boxed_str());
        }
        Ok(Place {
            word,
            specs,
            resolved: true,
            ..self
        })
    }

    /// The placement with the margin `margin` on each resolved card's spec
    /// ([`card_budget::keep_free`]), so the machine the placement lays out
    /// gives each card its own `margin_bytes`; a card the margin names none
    /// keeps its own.
    fn keeping_free(self, margin: &card_budget::CardBudget) -> Result<Place, GateError> {
        let mut specs = self.specs;
        for spec in specs.iter_mut().flatten() {
            *spec = card_budget::keep_free(margin, *spec)
                .map_err(|e| format!("--place {}: {e}", self.spelled()))?;
        }
        Ok(Place { specs, ..self })
    }

    /// [`Place::on`] with the tiers on the nodes at `nodes`, one a tier
    /// card in tier order: the stage card resolved against this process's
    /// `census`, each tier card the one device its node's census lists
    /// ([`node_cards`]). Another count of nodes than the placement's tier
    /// cards is refused by name. The tiers plan in their own nodes' pools
    /// when every node is off this host ([`off_host`]); else every card
    /// plans in this process's pool.
    fn on_nodes(self, census: &[DeviceInfo], nodes: &[SocketAddr]) -> Result<Place, GateError> {
        let tiers = usize::from(self.n) - 1;
        if nodes.len() != tiers {
            return Err(format!(
                "--place {}: {} names {} tier nodes for its {tiers} expert tier cards",
                self.spelled(),
                bloomery_levers::TIER_NODES,
                nodes.len()
            )
            .into());
        }
        let cards = node_cards(nodes)?;
        self.at_nodes(
            census,
            nodes
                .iter()
                .zip(cards)
                .map(|(&addr, (_, spec))| (addr, spec)),
        )
    }

    /// [`Place::on_nodes`] once each tier node's card is read: `tiers` holds
    /// each node's address and card, in tier order.
    fn at_nodes(
        self,
        census: &[DeviceInfo],
        tiers: impl Iterator<Item = (SocketAddr, CardSpec)>,
    ) -> Result<Place, GateError> {
        let stage = workstation::resolve(&self.picks()[..1], census)
            .map_err(|e| format!("--place {}: {e}", self.spelled()))?;
        let mut specs = [None; 1 + MAX_TIERS];
        specs[0] = stage.first().copied();
        let mut own_pools = true;
        for (s, (addr, spec)) in specs[1..].iter_mut().zip(tiers) {
            own_pools &= off_host(census, addr, &spec);
            *s = Some(spec);
        }
        Ok(Place {
            specs,
            resolved: true,
            own_pools,
            ..self
        })
    }

    /// The placement the cards in `census` offer, resolved against it
    /// ([`Place::on`]): [`Place::Bp`] when its picks find two devices (the
    /// largest the stage, the next-largest its tier), else [`Place::A`]. A
    /// census with no device is `a`'s refusal by name; a refusal of `bp`'s
    /// other than too few devices is `bp`'s.
    pub fn by_cards(census: &[DeviceInfo]) -> Result<Place, GateError> {
        match workstation::resolve(Place::Bp.picks(), census) {
            Ok(_) => Place::Bp.on(census),
            Err(PickError {
                why: PickWhy::Few(_),
                ..
            }) => Place::A.on(census),
            Err(e) => Err(format!("--place bp: {e}").into()),
        }
    }

    /// [`Place::choose`] for a `body` that serves no tier card
    /// ([`TierRule::UNTIERED`]): `flag` as given when set, else `a`, the plan
    /// never asked; the refusal of a tier count names `body`.
    pub fn choose_untiered(
        flag: Option<Place>,
        census: &[DeviceInfo],
        body: &str,
    ) -> Result<PlaceWhy, GateError> {
        Place::choose(flag, census, TierRule::UNTIERED, |p| {
            Err(format!(
                "--place unset: the {body} body serves no tier card, so no tier count for {}",
                p.name()
            )
            .into())
        })
    }

    /// The placement a serving seat runs by, set or unset, and why in the
    /// words a `place unset` record prints: `flag` set runs it as given
    /// (`set`), the plan never asked; unset, the cards' offer
    /// ([`Place::by_cards`]) — itself on one card (`one card`) — kept by
    /// `rule` on two: the body serves no tier card, or no sitting has shown
    /// the tier not slower, and [`Place::A`] runs, the plan never asked;
    /// else `tier_of` names the experts the offer's plan holds on its tier
    /// card, and the offer keeps the card at or past `rule`'s break-even with
    /// more than no expert — under it, for no expert, or for a plan that
    /// refuses (the refusal on stderr), [`Place::A`] runs. An unset flag
    /// never refuses a load `--place a` serves.
    pub fn choose(
        flag: Option<Place>,
        census: &[DeviceInfo],
        rule: TierRule,
        tier_of: impl FnOnce(Place) -> Result<u64, GateError>,
    ) -> Result<PlaceWhy, GateError> {
        if let Some(p) = flag {
            return Ok(PlaceWhy {
                place: p.on(census)?,
                why: "set",
                tier_experts: None,
                rule,
            });
        }
        let offer = Place::by_cards(census)?;
        if offer.tier_cards().is_empty() {
            return Ok(PlaceWhy {
                place: offer,
                why: "one card",
                tier_experts: None,
                rule,
            });
        }
        let a = |why: &'static str, n: Option<u64>| -> Result<PlaceWhy, GateError> {
            Ok(PlaceWhy {
                place: Place::A.on(census)?,
                why,
                tier_experts: n,
                rule,
            })
        };
        if rule.tiers == 0 {
            return a("two cards, the body serves no tier card", None);
        }
        let Some(e) = rule.break_even else {
            return a("two cards, no sitting has shown the tier not slower", None);
        };
        let (place, why, n) = match tier_of(offer) {
            Err(x) => {
                eprintln!(
                    "--place unset: the tier's plan refused ({x}); the largest card alone (a)"
                );
                (
                    Place::A.on(census)?,
                    "two cards, the tier's plan refused",
                    None,
                )
            }
            Ok(0) => (
                Place::A.on(census)?,
                "two cards, the tier holds no expert",
                Some(0),
            ),
            Ok(n) if n >= e => (offer, "two cards, tier at or past the break-even", Some(n)),
            Ok(n) => (
                Place::A.on(census)?,
                "two cards, tier under the break-even",
                Some(n),
            ),
        };
        Ok(PlaceWhy {
            place,
            why,
            tier_experts: n,
            rule,
        })
    }

    /// The word as typed: an alias's own word, or the picks of a list
    /// spelled as an alias (`a6000` for `a`).
    fn spelled(&self) -> String {
        let alias = workstation::ALIASES.iter().find(|(w, _)| *w == self.word);
        match alias {
            Some((_, picks)) if *picks != self.picks() => self
                .picks()
                .iter()
                .map(|p| p.to_string().to_ascii_lowercase())
                .collect::<Vec<_>>()
                .join("+"),
            _ => self.word.to_string(),
        }
    }

    /// How the placement's cards are found.
    fn picks(&self) -> &[Pick] {
        &self.picks[..usize::from(self.n)]
    }

    /// The placement's card specs, the stage first; a card an ordinal names
    /// before [`Place::on`] is refused by name.
    pub fn card_specs(self) -> Result<Vec<CardSpec>, GateError> {
        self.specs[..usize::from(self.n)]
            .iter()
            .zip(self.picks())
            .map(|(s, p)| {
                s.ok_or_else(|| {
                    format!(
                        "--place {}: {p} is a device of this process's census, and the \
                         placement was not resolved against it (Place::on_host)",
                        self.word
                    )
                    .into()
                })
            })
            .collect()
    }

    /// Refuse, before any plan, a placement of more tier cards than `body`
    /// serves — each body declares its count; `tiers` is it.
    pub fn serves(self, body: &str, tiers: usize) -> Result<(), GateError> {
        let k = usize::from(self.n) - 1;
        if k <= tiers {
            return Ok(());
        }
        Err(format!(
            "--place {}: {k} expert tier cards, and the {body} body serves {tiers} at most",
            self.name()
        )
        .into())
    }

    /// [`Place::machine`] for a body that serves no tier card, with no draft:
    /// a placement that names a tier card is planned with no batch bytes, so
    /// the plan's own refusal of the tier reaches the seat.
    pub fn machine_untiered(self) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
        self.machine(
            None,
            (!self.tier_cards().is_empty()).then_some(workstation::NO_TIER_BATCH),
        )
    }

    /// The machine a placed load plans over, by layer count: the placement's
    /// one card with the token embedding on it and the plan's own `SCRATCH`
    /// ([`workstation::placed_on`]), no tier card. A placement that names a
    /// tier card is refused by name: the plan has no expert tier to give it.
    pub fn machine_placed(self) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
        if !self.tier_cards().is_empty() {
            return Err(format!(
                "--place {}: a placed load runs one card with the host tier beside it, and the \
                 placement names {} tier cards",
                self.name(),
                self.tier_cards().len()
            )
            .into());
        }
        let stage = self.card_specs()?[0];
        Ok(move |layers| workstation::placed_on(stage, layers, workstation::SCRATCH))
    }

    /// The machine the placement plans over, by layer count. `draft` is the
    /// DSpark draft's seat ([`Place::draft_on`]): its resident bytes reserved
    /// on the card it sits on, by that card's place in the plan's device list
    /// (the stage is 0, then the tiers, then the draft's own card where it
    /// is none of the placement's and joins the plan as a reserve-only
    /// member), so the plan sets them aside on that card before its experts;
    /// `batch` is the expert tier's prompt-batch bytes the plan reserves on
    /// each tier card and the host (the model's own figure, from its
    /// hyperparameters). A batch handed to a placement with no tier card is
    /// refused by name, a placement with tier cards without `batch` too, a
    /// draft on a card past the plan's device list likewise (the card is
    /// outside the plan), a draft's own card beside tiers that plan on nodes
    /// of their own too, and every refusal of `workstation::plan_holding`
    /// before the machine is planned.
    pub fn machine(
        self,
        draft: Option<DraftSeat>,
        batch: Option<TierBatchBytes>,
    ) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
        let specs = self.card_specs()?;
        let (stage, tiers) = (specs[0], &specs[1..]);
        let tiered = self.n > 1;
        let reserve = draft.map(|d| d.reserve);
        let holder = draft.and_then(|d| d.holder);
        if holder.is_some() && self.own_pools {
            return Err(format!(
                "--place {}: the DSpark draft's card joins the plan as a reserve-only member of \
                 this process's node, and the placement plans its tier cards on nodes of their \
                 own",
                self.name()
            )
            .into());
        }
        let holders: &[CardSpec] = holder.as_slice();
        match batch {
            Some(_) if !tiered => {
                return Err(format!(
                    "--place {}: no tier card serves a prompt batch; the tier's batch reserve is \
                     outside the plan",
                    self.name()
                )
                .into());
            }
            Some(b) => {
                workstation::check_holding(stage, tiers, holders, reserve, b)?;
            }
            None if tiered => {
                return Err(format!(
                    "--place {}: the plan reserves the expert tier's prompt-batch bytes, and none \
                     were given",
                    self.name()
                )
                .into());
            }
            None => {
                workstation::check_holding(
                    stage,
                    tiers,
                    holders,
                    reserve,
                    workstation::NO_TIER_BATCH,
                )?;
            }
        }
        Ok(move |layers| {
            self.machine_with(|stage, tiers| {
                Ok(workstation::plan_holding(
                    layers,
                    stage,
                    tiers,
                    holder.as_slice(),
                    reserve,
                    batch.unwrap_or(workstation::NO_TIER_BATCH),
                )?)
            })
            .expect("Place::machine checked the placement's cards before planning")
        })
    }

    /// The machine `layout` lays out over this placement's stage card and
    /// tier cards, holding the nodes the placement plans the tiers on: tier
    /// card `t` on node `t + 1` ([`workstation::node_of`]) when the tiers plan
    /// in their own nodes' pools ([`Place::on_nodes`]), else none, every card
    /// then in this process's pool. [`Place::machine`] lays out the common
    /// plan with it, and a family with a layout of its own lays out that.
    pub fn machine_with(
        self,
        layout: impl FnOnce(CardSpec, &[CardSpec]) -> Result<Machine, GateError>,
    ) -> Result<Machine, GateError> {
        let specs = self.card_specs()?;
        let mut machine = layout(specs[0], &specs[1..])?;
        if self.own_pools {
            machine
                .nodes
                .extend(specs[1..].iter().map(workstation::node_of));
        }
        Ok(machine)
    }

    /// Whether the placement is `tp`, tensor parallel over `BLOOMERY_TP_NODES`.
    pub fn tp(self) -> bool {
        self.word == "tp"
    }

    /// The flag value, as the `plan` and `load` lines print it.
    pub fn name(self) -> &'static str {
        self.word
    }

    /// The cards the placement loads, the stage card then the tiers, by the
    /// names the plan gives them (an ordinal before [`Place::on`] as
    /// `cuda<N>`).
    pub fn cards(self) -> Vec<&'static str> {
        self.specs[..usize::from(self.n)]
            .iter()
            .zip(self.picks())
            .map(|(s, p)| match (s, p) {
                (Some(s), _) => s.name,
                (None, Pick::Ordinal(o)) => ordinal_word(*o),
                (None, _) => "?",
            })
            .collect()
    }

    /// The placement's expert tier cards in tier order, by name: the 3090
    /// under `bp` on this workstation, none under `a` and `gate`.
    pub fn tier_cards(self) -> Vec<&'static str> {
        self.cards()[1..].to_vec()
    }

    /// The device of the first tier card, where a DSpark draft sits when
    /// `BLOOMERY_DSPARK_CARD` names none ([`Place::draft_on`]), once resolved;
    /// `None` under a placement with no tier card.
    pub fn draft_device(self) -> Option<DeviceId> {
        (self.n > 1)
            .then_some(self.specs[1])
            .flatten()
            .and_then(|s| s.device)
    }
}

/// The flags a launcher reads of its own: those that take no value and those that take one,
/// and whether the shared diagnosis flags ([`Diag`]) are among them.
#[derive(Clone, Copy)]
pub struct Known<'a> {
    pub bare: &'a [&'a str],
    pub valued: &'a [&'a str],
    pub diag: bool,
}

/// Refuse the first argument of `args` (argv[0] first, as [`crate::llama_launch::args`] gives
/// them) that is no flag of `known`, by name with the flags it has: a launcher that looks its
/// flags up by name would otherwise let a typo run the default. A valued flag takes the next
/// argument whatever it looks like, and one with none is refused too.
pub fn refuse_unknown(bin: &str, args: &[String], known: Known<'_>) -> Result<(), GateError> {
    let diag = known.diag;
    let bare = |a: &str| known.bare.contains(&a) || (diag && Diag::BARE.contains(&a));
    let valued = |a: &str| known.valued.contains(&a) || (diag && Diag::VALUED.contains(&a));
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        if bare(arg.as_str()) {
            continue;
        }
        if !valued(arg.as_str()) {
            let mut flags: Vec<String> = known.bare.iter().map(|f| (*f).to_owned()).collect();
            flags.extend(known.valued.iter().map(|f| format!("{f} <value>")));
            if diag {
                flags.push(Diag::USAGE.to_owned());
            }
            return Err(format!(
                "{bin}: unknown argument {arg:?}; its flags are {}",
                flags.join(" ")
            )
            .into());
        }
        if it.next().is_none() {
            return Err(format!("{bin}: {arg} needs a value").into());
        }
    }
    Ok(())
}

/// `--place {word}`'s refusals against the per-card levers, before any card is
/// resolved: a `BLOOMERY_CARD_BUDGET` or `BLOOMERY_CARD_MARGIN` list naming a
/// device `census` does not list ([`card_budget::check_visible`]), shares as a
/// margin, and a list of either beside `tier_nodes` tier nodes, whose cards
/// are another process's, no ordinal of this one's.
fn refuse_card_lists(
    word: &str,
    budget: &card_budget::CardBudget,
    margin: &card_budget::CardBudget,
    census: &[DeviceInfo],
    tier_nodes: usize,
) -> Result<(), GateError> {
    card_budget::check_visible(budget, census).map_err(|e| format!("--place {word}: {e}"))?;
    card_budget::check_margin_visible(margin, census)
        .map_err(|e| format!("--place {word}: {e}"))?;
    if let card_budget::CardBudget::Shares(_) = margin {
        return Err(format!(
            "--place {word}: {} takes bytes, not shares: {margin}",
            bloomery_levers::CARD_MARGIN
        )
        .into());
    }
    if tier_nodes > 0 {
        for (lever, value) in [
            (bloomery_levers::CARD_BUDGET, budget),
            (bloomery_levers::CARD_MARGIN, margin),
        ] {
            if value.listed().is_some() {
                return Err(format!(
                    "--place {word}: {lever}={value} names cards by this process's CUDA \
                     ordinals, and {} puts the tier cards on nodes of their own; give one value \
                     for every card",
                    bloomery_levers::TIER_NODES
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Whether the tier node at `addr`, whose census gave `spec`, is another
/// host: an address other than loopback, and a device that is none of this
/// process's `census`. Such a node's card takes its bytes from its own pool.
fn off_host(census: &[DeviceInfo], addr: SocketAddr, spec: &CardSpec) -> bool {
    !addr.ip().is_loopback()
        && spec
            .device
            .is_some_and(|d| census.iter().all(|c| c.uuid != d.uuid))
}

/// `cuda<o>` as a `'static` word, spelled once per ordinal.
fn ordinal_word(o: u32) -> &'static str {
    static WORDS: std::sync::Mutex<Vec<(u32, &'static str)>> = std::sync::Mutex::new(Vec::new());
    let mut w = WORDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, s)) = w.iter().find(|(k, _)| *k == o) {
        return s;
    }
    let s: &'static str = Box::leak(format!("cuda{o}").into_boxed_str());
    w.push((o, s));
    s
}

/// The devices a model runs on, the stage card `stage` then the expert tier
/// cards `tiers` (a tier on a node, the device of its node's census it opened
/// on), as a `load` line's `cards` field names them: by the names
/// their drivers report, each space written `_`, since the field is one
/// word. `planned` is the placement `--place <word>`'s cards by the names
/// the plan gives them, and `specs` the devices it resolved to on this
/// process's census (`None` for a census-free placement). Another count of
/// devices than the placement's cards, and a device that is not its card —
/// of a resolved placement, another device; of a census-free one, a name
/// that does not hold the card's — are refused by name. Every body's load
/// line names its cards here.
pub fn card_words(
    word: &str,
    planned: &[&str],
    specs: Option<&[CardSpec]>,
    stage: &Gpu,
    tiers: &[TierCard],
) -> Result<Vec<String>, GateError> {
    let mut devices = vec![stage.device_name()?];
    let mut ids = vec![stage.device_id()?];
    for t in tiers {
        match t.node_device() {
            Some(d) => {
                devices.push(d.name.clone());
                ids.push(d.device());
            }
            None => {
                let g = t.gpu()?;
                devices.push(g.device_name()?);
                ids.push(g.device_id()?);
            }
        }
    }
    let named = devices.len() == planned.len()
        && match specs {
            Some(specs) => ids
                .iter()
                .zip(specs)
                .all(|(id, s)| s.device.is_some_and(|d| d.uuid == id.uuid)),
            None => devices.iter().zip(planned).all(|(d, p)| d.contains(p)),
        };
    if !named {
        return Err(format!(
            "--place {word}: the model runs on {devices:?} ({}), the placement's cards are \
             {planned:?}",
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into());
    }
    Ok(devices.iter().map(|d| d.replace(' ', "_")).collect())
}

/// `r`, a `load` record ([`record::LOAD_GENERATOR`]) written up to its
/// `unified_addressing` field, with the devices the model runs on (`cards`,
/// [`card_words`] of the placement's cards) and, when the placement has an
/// expert tier card, the tier's experts and resident bytes. A device that is
/// not the placement's card ([`card_words`]), tiers that do not match the
/// placement's — another count, or a tier on another card — and more than
/// the one tier the record's fields hold are refused by name.
pub fn with_cards(
    r: Record,
    place: Place,
    stage: &Gpu,
    tiers: &[TierCard],
) -> Result<Record, GateError> {
    let specs = if place.resolved {
        Some(place.card_specs()?)
    } else {
        None
    };
    let words = card_words(place.name(), &place.cards(), specs.as_deref(), stage, tiers)?;
    let r = r.csv("cards", words);
    let want = place.tier_cards();
    let got: Vec<&str> = tiers.iter().map(TierCard::name).collect();
    match (tiers, want.as_slice()) {
        ([], []) => Ok(r),
        ([t], [name]) if t.name() == *name => Ok(r
            .u("tier_experts", t.set().experts())
            .u("tier_bytes", t.resident_bytes())),
        ([_, _, ..], _) if got == want => Err(format!(
            "--place {}: the load record holds one expert tier's fields, and the model loaded \
             {} tiers",
            place.name(),
            got.len()
        )
        .into()),
        _ => Err(format!(
            "--place {}: the loaded model's expert tiers are {got:?}, the placement's {want:?}",
            place.name()
        )
        .into()),
    }
}

/// How [`Generator::open`] loads: the placement and the expert tier's
/// prompt-batch bytes its plan reserves ([`Place::machine`]'s `batch`), the
/// context the caches are sized for, the step mode, and whether the calling
/// thread pins itself to the dispatcher's cpu slot (`BLOOMERY_PIN_MAIN` is
/// the caller's to read).
#[derive(Debug, Clone, Copy)]
pub struct OpenArgs {
    pub place: Place,
    pub tier_batch: Option<TierBatchBytes>,
    pub ctx: usize,
    pub mode: StepMode,
    pub pin_main: bool,
}

/// A loaded model standing at a position, and the context it may reach.
pub struct Generator<B: ChainBody> {
    model: GpuModel<B>,
    ctx: usize,
}

impl<B: ChainBody> Generator<B> {
    /// Open `$BLOOMERY_REF_MODEL` by `args.place` through `open`, the body's
    /// loader (the file, the placement's machine and the context), run `check`
    /// on the loaded model (it refuses a body that cannot run this file, and
    /// adds the body's fields to the `load` record it is handed), then write
    /// the `load` record ([`record::LOAD_GENERATOR`]), the load's phases when
    /// the load timed them ([`record::LOAD_PHASES`]), the host set's records
    /// of a placed load, and in graph mode capture the step before any token
    /// (the `capture` record) — all to `log`.
    pub fn open<O, C>(
        args: OpenArgs,
        open: O,
        check: C,
        log: &mut dyn Write,
    ) -> Result<Generator<B>, GateError>
    where
        O: FnOnce(Split, &dyn Fn(usize) -> Machine, usize) -> Result<GpuModel<B>, GpuError>,
        C: FnOnce(&GpuModel<B>, Record) -> Result<Record, GateError>,
    {
        let pinned = args.pin_main && threads::pool().pin_caller();
        let path = ref_model_path()?;
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // The lane's rate, taken before the plan reads the room (`lane_of`).
        let lane = lane_at(args.place)?;
        let machine = args.place.machine(None, args.tier_batch)?;
        let headers = t.elapsed();
        let mut model = open(file, &machine, args.ctx)?;
        model.set_mode(args.mode);
        let load = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", model.resident_bytes())
            .u("ctx", args.ctx);
        let load = check(&model, load)?;
        let total = t.elapsed();
        writeln!(log, "{}", record::lane(lane.0, &lane.1).line())?;
        let load = load
            .w("mode", mode_name(args.mode))
            .w("place", args.place.name())
            .w("pin_main", if args.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", total.as_secs_f64());
        writeln!(log, "{}", load.line())?;
        if let Some(times) = model.load_times() {
            let phases = record::load_phases(
                total.as_secs_f64(),
                headers.as_secs_f64(),
                times.plan.map(|d| d.as_secs_f64()),
                times.context.as_secs_f64(),
                times.upload.as_secs_f64(),
                times.derive.as_secs_f64(),
                times.host_set.as_secs_f64(),
                times.body.as_secs_f64(),
                times.head.as_secs_f64(),
            );
            writeln!(log, "{}", phases.line())?;
        }
        if let Some(h) = model.host_residency() {
            for r in record::host_residency(h) {
                writeln!(log, "{}", r.line())?;
            }
        }
        if args.mode == StepMode::Graph {
            // Captured before the prompt, so no token's step pays for it.
            let nodes = model.capture_step()?;
            writeln!(
                log,
                "{}",
                Record::new(&record::CAPTURE).u("graph_nodes", nodes).line()
            )?;
        }
        Ok(Generator {
            model,
            ctx: args.ctx,
        })
    }

    /// Feed `ids` from the current position, one real step per id, and
    /// return the argmax after the last. Refused before any step when the
    /// ids do not fit the context ([`Generator::check_feed`]).
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.check_feed(ids.len())?;
        Ok(self.model.step(ids)?)
    }

    /// Whether a feed of `n` ids may start at the current position: at
    /// least one, and all inside the context. The one owner of that check
    /// for every feed of this generator, [`Generator::prefill`] and a
    /// body's own prompt feed alike.
    pub fn check_feed(&self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Err("prefill: no ids to feed".into());
        }
        if self.pos() + n > self.ctx {
            return Err(format!(
                "prefill: position {} + {n} ids exceed the context {}",
                self.pos(),
                self.ctx
            ));
        }
        Ok(())
    }

    /// One step on `tok` at the current position; the argmax after it.
    pub fn step(&mut self, tok: u32) -> Result<u32, GateError> {
        if usize::try_from(self.model.pos())? >= self.ctx {
            return Err(format!("step: the context {} is full", self.ctx).into());
        }
        Ok(self.model.step(&[tok])?)
    }

    /// The head's logits after the last step (`n_vocab` f32). A blocking
    /// read of the whole row, on top of the step's own argmax readback.
    pub fn logits(&self) -> Result<Vec<f32>, GateError> {
        Ok(self.model.logits()?)
    }

    /// The position the next fed token lands in.
    pub fn pos(&self) -> usize {
        // u32 widens into usize on the 64-bit hosts this runs on.
        self.model.pos() as usize
    }

    /// The context the caches were sized for: `pos` never passes it.
    pub fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The model, for what the loop does not own (the pair pass, probes).
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
    }

    /// See [`Generator::model`].
    pub fn model_mut(&mut self) -> &mut GpuModel<B> {
        &mut self.model
    }
}

/// The step mode as the `load` line prints it.
pub fn mode_name(mode: StepMode) -> &'static str {
    if mode == StepMode::Graph {
        "graph"
    } else {
        "eager"
    }
}

// ---------------------------------------------------------------- diagnosis

/// The diagnosis flags a generate binary takes beside its own, each acting
/// on the plain runs of the prompt flags:
///
/// - `--top2 K`: a `top2` record for each of the first K generated tokens,
///   after the loop: the logits row behind that token (`i` 0 is the prompt
///   call's last row), read back after its step, by its two largest entries
///   — ids and values, the larger first, ties to the lower id as the argmax
///   takes them — and their margin. A row whose largest entry is not the
///   token the run took, and a NaN in a row, are named errors ([`TopRows`]).
/// - `--rows FILE`: the K rows `--top2` reads, whole, appended to FILE as
///   little-endian f32, run after run (FILE emptied before the first), each
///   run's followed by a `rows` record.
/// - `--repeat R`: the run R times on one load, each after the first from
///   `runtime::Target::reset` — the serve seat's reset on a request that
///   misses its cache, which leaves the adaptive residency where the run
///   before it took it — each opened by an `arm` record (`feed=repeat`)
///   ([`repeat_runs`]).
/// - `--table`: the stage card's copy of the slot map read back after the
///   runs, a `slot table` record: its entries against the host map, the
///   residency ledger (a load that runs the machine) and the map before the
///   first run, and the card slots two ids of one layer both name
///   ([`Residence`], [`slot_table`]).
/// - `--ignore-eos`: each token taken as the serve takes it under a
///   request's `ignore_eos` ([`NoEog`]); `--top2` ranks the rows without the
///   end-of-generation ids, which an `ignore eos` record names, and `--rows`
///   writes the rows as the model made them.
/// - `--serve-feed`: the prompt fed as the serve feeds it ([`ServeFeed`]):
///   every id but the last in one call, then the last id as a pass of the
///   run's pick, whose row is generated token 0's.
/// - `--dump-table FILE` (with `--repeat` 2 or more): the stage card's copy
///   of the map as the second run's token 0 ran on it, read after
///   `Target::reset` (FILE.before) and after that run's token-0 step (FILE),
///   and before the first run (FILE.seed) ([`write_table`]), and a `table
///   dump` record ([`dump_table`]).
/// - `--card-table FILE`: the load placed by that table, residency off:
///   each routed layer's card experts the ids FILE puts on the card, in id
///   order, the rest on the host, every other row the planner's
///   ([`place_table`]); a `card table` record after the load holds the
///   card's copy to FILE's sets ([`card_table`]).
#[derive(Clone, Debug)]
pub struct Diag {
    pub top2: usize,
    pub rows: Option<PathBuf>,
    pub repeat: usize,
    pub table: bool,
    /// The vocabulary's end-of-generation ids under `--ignore-eos`
    /// ([`Diag::finish`] reads them); empty without it.
    pub ignore_eos: Vec<u32>,
    pub serve_feed: bool,
    pub dump_table: Option<PathBuf>,
    pub card_table: Option<PathBuf>,
    /// `--ignore-eos` was given.
    no_eog: bool,
}

impl Default for Diag {
    /// No flag: one run, nothing read back.
    fn default() -> Diag {
        Diag {
            top2: 0,
            rows: None,
            repeat: 1,
            table: false,
            ignore_eos: Vec::new(),
            serve_feed: false,
            dump_table: None,
            card_table: None,
            no_eog: false,
        }
    }
}

impl Diag {
    /// The flags that take no value ([`Diag::take`] takes each).
    pub const BARE: [&'static str; 3] = ["--table", "--ignore-eos", "--serve-feed"];

    /// The flags that take a value ([`Diag::take`] takes each).
    pub const VALUED: [&'static str; 5] = [
        "--top2",
        "--repeat",
        "--rows",
        "--dump-table",
        "--card-table",
    ];

    /// The flags, as a usage line shows them.
    pub const USAGE: &'static str = "[--top2 K [--rows FILE]] [--repeat R] [--table] \
                                     [--ignore-eos] [--serve-feed] [--dump-table FILE] \
                                     [--card-table FILE]";

    /// Take `flag` when it is one of these, with its value from `it` when it
    /// takes one: `true` when taken, `false` for a flag of the binary's own.
    /// A flag given twice takes its last value.
    pub fn take(
        &mut self,
        flag: &str,
        it: &mut impl Iterator<Item = String>,
    ) -> Result<bool, GateError> {
        match flag {
            "--table" => self.table = true,
            "--ignore-eos" => self.no_eog = true,
            "--serve-feed" => self.serve_feed = true,
            "--top2" | "--repeat" | "--rows" | "--dump-table" | "--card-table" => {
                let v = it
                    .next()
                    .ok_or_else(|| format!("{flag} needs a value: {}", Diag::USAGE))?;
                match flag {
                    "--top2" => self.top2 = v.parse()?,
                    "--repeat" => self.repeat = v.parse()?,
                    "--rows" => self.rows = Some(PathBuf::from(v)),
                    "--dump-table" => self.dump_table = Some(PathBuf::from(v)),
                    _ => self.card_table = Some(PathBuf::from(v)),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The flags against each other and against the run's `-n` (`n_gen`),
    /// each refusal by name; under `--ignore-eos` the end-of-generation ids
    /// of `model`'s vocabulary, one at least.
    pub fn finish(&mut self, n_gen: usize, model: &Path) -> Result<(), GateError> {
        if self.repeat == 0 {
            return Err("--repeat 0 runs nothing: give 1 or more".into());
        }
        if self.dump_table.is_some() && self.repeat < 2 {
            return Err(
                "--dump-table reads the second run's table: give --repeat 2 or more".into(),
            );
        }
        if self.dump_table.is_some() && self.card_table.is_some() {
            return Err("--dump-table reads a residency load; --card-table loads none".into());
        }
        if self.rows.is_some() && self.top2 == 0 {
            return Err("--rows writes the rows --top2 K reads: give --top2".into());
        }
        if self.top2 > n_gen {
            return Err(format!(
                "--top2 {} asks for more rows than -n {n_gen} tokens",
                self.top2
            )
            .into());
        }
        if self.no_eog {
            let tok = Tokenizer::from_gguf(model)
                .map_err(|e| format!("--ignore-eos: the vocabulary of {}: {e}", model.display()))?;
            self.ignore_eos = tok.eog().to_vec();
            if self.ignore_eos.is_empty() {
                return Err(format!(
                    "--ignore-eos: {} names no end-of-generation id",
                    model.display()
                )
                .into());
            }
        }
        Ok(())
    }

    /// The flags set, by name: a binary refuses them beside a run they do
    /// not act on (a draft, a probe, several slots).
    #[must_use]
    pub fn set(&self) -> Vec<&'static str> {
        [
            (self.top2 > 0, "--top2"),
            (self.rows.is_some(), "--rows"),
            (self.repeat > 1, "--repeat"),
            (self.table, "--table"),
            (self.no_eog, "--ignore-eos"),
            (self.serve_feed, "--serve-feed"),
            (self.dump_table.is_some(), "--dump-table"),
            (self.card_table.is_some(), "--card-table"),
        ]
        .into_iter()
        .filter_map(|(set, name)| set.then_some(name))
        .collect()
    }

    /// `--serve-feed` against a run of `ids` fed ids: the call runs every id
    /// but the last, so two at least.
    pub fn check_feed(&self, ids: usize) -> Result<(), GateError> {
        if self.serve_feed && ids < 2 {
            return Err(format!(
                "--serve-feed calls every id but the last: give 2 or more ids, not {ids}"
            )
            .into());
        }
        Ok(())
    }

    /// Before the first run: `--rows`' file emptied, and under
    /// `--ignore-eos` the `ignore eos` record.
    pub fn begin(&self) -> Result<(), GateError> {
        if let Some(p) = &self.rows {
            std::fs::File::create(p).map_err(|e| format!("--rows {}: {e}", p.display()))?;
        }
        if !self.ignore_eos.is_empty() {
            Record::new(&record::IGNORE_EOS)
                .csv("ids", &self.ignore_eos)
                .print();
        }
        Ok(())
    }
}

/// `--ignore-eos`'s pick, the serve's under `ignore_eos`
/// (`serve::genloop::choose`): the engine's argmax unless it is one of the
/// end-of-generation ids, else the largest logit of the row with those ids
/// at -inf, first of equals. One step a pass, every row read back.
#[derive(Clone, Debug)]
pub struct NoEog {
    ids: Vec<u32>,
}

impl NoEog {
    /// The pick passing over `ids`.
    #[must_use]
    pub fn new(ids: Vec<u32>) -> NoEog {
        NoEog { ids }
    }

    /// The token taken from `out`, a read back of [`Want::Logits`].
    #[must_use]
    pub fn pick(&self, out: &Out<'_>) -> u32 {
        let greedy = out.argmax();
        let Out::Logits { row, .. } = *out else {
            return greedy;
        };
        if !self.ids.contains(&greedy) {
            return greedy;
        }
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in row.iter().enumerate() {
            let v = match u32::try_from(i) {
                Ok(id) if self.ids.contains(&id) => f32::NEG_INFINITY,
                _ => v,
            };
            if best.is_none_or(|(_, b)| v.total_cmp(&b).is_gt()) {
                best = Some((i, v));
            }
        }
        best.and_then(|(i, _)| u32::try_from(i).ok())
            .unwrap_or(greedy)
    }
}

impl<T: Target> Advance<T> for NoEog {
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        let out = t.prompt(ids, Want::Logits)?;
        Ok(self.pick(&out))
    }

    fn begin(&mut self, _t: &T, _prompt: &[u32], _first: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        let pos = t.pos();
        let o = t.step(last, Want::Logits)?;
        out.push(self.pick(&o));
        Ok(Committed {
            pos,
            kept: 1,
            rows: 1,
            proposed: false,
        })
    }
}

/// `--serve-feed`'s prompt, the serve's (`serve::genloop`:
/// `prefill_marked(ids, 0, n - 1)`, then `next(ids[n - 1])`): every id but
/// the last in one call, then the last id as one of `inner`'s passes, whose
/// kept token is generated token 0. The passes after it are `inner`'s own.
pub struct ServeFeed<'a, A> {
    pub inner: &'a mut A,
}

impl<T, A> Advance<T> for ServeFeed<'_, A>
where
    T: Target,
    T::Error: From<GpuError>,
    A: Advance<T>,
{
    const ROWS: usize = A::ROWS;

    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        let refused = |detail: &str| GpuError::Shape {
            what: "ServeFeed",
            detail: detail.to_string(),
        };
        let Some((&last, head)) = ids.split_last() else {
            return Err(refused("an empty prompt: the serve feeds one id or more").into());
        };
        if !head.is_empty() {
            t.prompt(head, Want::Argmax)?;
        }
        let mut out = Vec::with_capacity(A::ROWS);
        self.inner.pass(t, last, &mut out)?;
        out.first()
            .copied()
            .ok_or_else(|| refused("the last id's pass kept no token").into())
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.inner.begin(t, prompt, first)
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        self.inner.pass(t, last, out)
    }
}

/// A logits row's two largest entries, the larger first, ties to the lower
/// id as the argmax takes them.
struct Top2 {
    ids: [usize; 2],
    values: [f32; 2],
}

impl Top2 {
    /// The logits row behind `token`, the token the run took from it,
    /// ranked without the ids in `skip`; a row whose largest such entry is
    /// another, and a NaN, are named errors.
    fn of(row: &[f32], token: u32, skip: &[u32]) -> Result<Top2, GateError> {
        let mut t = Top2 {
            ids: [usize::MAX; 2],
            values: [f32::NEG_INFINITY; 2],
        };
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return Err(format!("top2: the logits row is NaN at id {i}").into());
            }
            if u32::try_from(i).is_ok_and(|i| skip.contains(&i)) {
                continue;
            }
            if v > t.values[0] {
                t = Top2 {
                    ids: [i, t.ids[0]],
                    values: [v, t.values[0]],
                };
            } else if v > t.values[1] {
                t.ids[1] = i;
                t.values[1] = v;
            }
        }
        if u32::try_from(t.ids[0]).ok() != Some(token) {
            return Err(format!(
                "top2: the row's largest entry is id {}, and the run took {token} from it",
                t.ids[0]
            )
            .into());
        }
        Ok(t)
    }

    fn print(&self, i: usize) {
        let [a, b] = self.values.map(f64::from);
        Record::new(&record::TOP2)
            .u("i", i)
            .u("top1", self.ids[0])
            .f("top1_logit", a)
            .u("top2", self.ids[1])
            .f("top2_logit", b)
            .f("margin", a - b)
            .print();
    }
}

/// A run's `--top2` rows ([`Diag`]): the first `top2` logits rows the run's
/// tokens came from, each ranked as it is read, and under `--rows` kept
/// whole.
pub struct TopRows {
    top2: usize,
    skip: Vec<u32>,
    tops: Vec<Top2>,
    kept: Option<Vec<f32>>,
}

impl TopRows {
    /// The rows `d` asks a run for.
    #[must_use]
    pub fn new(d: &Diag) -> TopRows {
        TopRows {
            top2: d.top2,
            skip: d.ignore_eos.clone(),
            tops: Vec::with_capacity(d.top2),
            kept: d.rows.as_ref().map(|_| Vec::new()),
        }
    }

    /// Whether the run still wants a row: fewer than `--top2` read.
    #[must_use]
    pub fn wants(&self) -> bool {
        self.tops.len() < self.top2
    }

    /// The logits `row` behind generated token `token`, the next one read.
    pub fn read(&mut self, row: &[f32], token: u32) -> Result<(), GateError> {
        self.tops.push(Top2::of(row, token, &self.skip)?);
        if let Some(k) = self.kept.as_mut() {
            k.extend_from_slice(row);
        }
        Ok(())
    }

    /// After the run's loop: a `top2` record a row and, under `--rows`
    /// (`rows`), the rows appended whole to that file and the `rows` record.
    pub fn finish(&self, rows: Option<&Path>) -> Result<(), GateError> {
        for (i, t) in self.tops.iter().enumerate() {
            t.print(i);
        }
        let (Some(path), Some(kept)) = (rows, &self.kept) else {
            return Ok(());
        };
        let bytes = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut f| crate::logit_rows::write_f32s(&mut f, kept))
            .map_err(|e| format!("--rows {}: {e}", path.display()))?;
        Record::new(&record::ROWS)
            .u("rows", self.tops.len())
            .u("n", kept.len() / self.tops.len().max(1))
            .u("bytes", bytes)
            .w("path", path.display())
            .print();
        Ok(())
    }
}

/// `--repeat`'s `runs` runs on one load: `each(t, i)` in order, each run
/// after the first from `Target::reset` (the adaptive residency stays where
/// the run before it took it), each opened by its `arm` record (`ids` fed
/// ids, `n_gen` generated tokens). The first failure ends them, naming its
/// run and `bin`.
pub fn repeat_runs<T: Target>(
    t: &mut T,
    runs: usize,
    ids: usize,
    n_gen: usize,
    bin: &str,
    mut each: impl FnMut(&mut T, usize) -> Result<(), GateError>,
) -> Result<(), GateError> {
    for i in 0..runs {
        if i > 0 {
            t.reset()?;
        }
        Record::new(&record::ARM)
            .u("i", i)
            .u("arms", runs)
            .w("feed", "repeat")
            .u("ids", ids)
            .u("n", n_gen)
            .print();
        each(t, i).map_err(|e| format!("{bin}: repeat {i} of {runs}: {e}"))?;
    }
    Ok(())
}

/// What the residency diagnostics read of a loaded body: the stage card's
/// copy of the slot map, which the body holds and hands the machine at its
/// start; the host tier's map and residency machine (`HostTier::slots`,
/// `HostTier::swap`); and the engine stream, which writes the copy. A
/// binary supplies them from its body; the readback is the residency's own
/// ([`CardTable::read`], [`CardTable::check`]).
pub struct Residence<'a> {
    pub card: &'a DeviceBuffer<u32>,
    pub map: &'a SlotMap,
    pub machine: Option<&'a SwapMachine>,
    pub stream: &'a CudaStream,
}

impl<'a> Residence<'a> {
    /// A body's views: `card` its stage card's copy of the map, `tier` its
    /// host tier, on `gpu`'s engine stream.
    #[must_use]
    pub fn of<H: HostExperts>(
        gpu: &'a Gpu,
        card: &'a DeviceTensor<u32>,
        tier: &'a HostTier<H>,
    ) -> Residence<'a> {
        Residence {
            card: card.buf(),
            map: tier.slots(),
            machine: tier.swap(),
            stream: gpu.stream(),
        }
    }

    /// The stage card's copy, read back.
    pub fn table(&self) -> Result<CardTable, GateError> {
        Ok(CardTable::read(
            self.card,
            self.map,
            self.machine,
            self.stream,
        )?)
    }

    /// `t` against the host map and, when the load runs a machine, its
    /// ledger.
    pub fn check(&self, t: &CardTable) -> Result<TableCheck, GateError> {
        Ok(t.check(self.map, self.machine.map(SwapMachine::ledger))?)
    }
}

/// The `slot table` record of `t`, checked as `c`, `vs_load` of its entries
/// off the map before the first run.
pub fn slot_table(t: &CardTable, c: &TableCheck, vs_load: usize) -> Record {
    let show = |v: u32| match v {
        HOST => "h".to_string(),
        v => v.to_string(),
    };
    let first: Vec<String> = c
        .first
        .iter()
        .map(|o| {
            format!(
                "l{}/e{}:card={},map={},ledger={}",
                o.layer,
                o.id,
                show(o.card),
                show(o.map),
                show(o.ledger)
            )
        })
        .collect();
    Record::new(&record::SLOT_TABLE)
        .u("layers", t.layers().len())
        .u("entries", t.entries().len())
        .u("on_card", t.on_card())
        .w("machine", c.vs_ledger.is_some())
        .u("vs_map", c.vs_map)
        .u("vs_ledger", c.vs_ledger.unwrap_or(0))
        .u("vs_load", vs_load)
        .u("doubled", c.doubled)
        .w(
            "first",
            if first.is_empty() {
                "none".to_string()
            } else {
                first.join(";")
            },
        )
}

/// `FILE.before`: the `--dump-table` read after `Target::reset`.
#[must_use]
pub fn before_path(p: &Path) -> PathBuf {
    suffixed(p, ".before")
}

/// `FILE.seed`: the `--dump-table` read before the first run.
#[must_use]
pub fn seed_path(p: &Path) -> PathBuf {
    suffixed(p, ".seed")
}

fn suffixed(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// `t` into `path`: the first layer, the layer count, the expert count,
/// then the entries, little-endian `u32`s.
pub fn write_table(path: &Path, t: &CardTable) -> Result<(), GateError> {
    let layers = t.layers();
    let head = [layers.start, layers.len(), t.n_expert()]
        .map(u32::try_from)
        .into_iter()
        .collect::<Result<Vec<u32>, _>>()?;
    let bytes: Vec<u8> = head
        .iter()
        .chain(t.entries())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// [`write_table`]'s file back; a length that is not its header's is
/// refused by name.
pub fn read_table(path: &Path) -> Result<CardTable, GateError> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (words, rest) = bytes.as_chunks::<4>();
    let words: Vec<u32> = words.iter().map(|&b| u32::from_le_bytes(b)).collect();
    let [start, len, n, ..] = words[..] else {
        return Err(format!("{}: {} bytes hold no header", path.display(), bytes.len()).into());
    };
    let (start, len, n) = (start as usize, len as usize, n as usize);
    if !rest.is_empty() || words.len() != 3 + len * n {
        return Err(format!(
            "{}: {} bytes; its header names {len} layers of {n} experts",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(CardTable::new(start..start + len, n, words[3..].to_vec())?)
}

/// `--dump-table`: `r`'s card copy into `path`, held to its read after
/// `Target::reset` (FILE.before, written before), and the `table dump`
/// record: the entries, those on the card, those off the host map, and
/// whether the two reads are one table.
pub fn dump_table(r: &Residence<'_>, path: &Path) -> Result<Record, GateError> {
    let t = r.table()?;
    write_table(path, &t)?;
    let before = read_table(&before_path(path))?;
    let c = t.check(r.map, None)?;
    Ok(Record::new(&record::TABLE_DUMP)
        .u("entries", t.entries().len())
        .u("on_card", t.on_card())
        .u("vs_map", c.vs_map)
        .w("same_before", before == t)
        .w("path", path.display()))
}

/// `--card-table`: each routed layer of `plan` holds on its stage card the
/// ids `t` puts on the card, in id order (`placement::routed_row`, the one
/// owner of a stack's split), the rest on the host; the layer's `n_l`
/// follows. A stack the planner left wholly on the host stays there when `t`
/// puts none of its layer on the card. Refused by name: another expert
/// count, a routed layer `t` does not cover, a stack of other than one card
/// segment that `t` puts experts of on the card, and a card format other
/// than the planner's.
pub fn place_table(plan: &mut Plan<'_>, t: &CardTable) -> Result<(), GateError> {
    let model = plan.model;
    let n = t.n_expert();
    let layers = t.layers();
    if u64::try_from(n)? != model.experts {
        return Err(format!(
            "--card-table: {n} experts a layer, the model's {}",
            model.experts
        )
        .into());
    }
    for row in &mut plan.rows {
        let tensor = &model.tensors[row.tensor];
        if tensor.role != Role::RoutedExperts {
            continue;
        }
        let l = tensor
            .layer
            .ok_or_else(|| format!("{}: a routed stack without a layer", tensor.name))?;
        if !layers.contains(&l) {
            return Err(format!("--card-table covers layers {layers:?}, not {l}").into());
        }
        let at = (l - layers.start) * n;
        let ids: Vec<u32> = (0..n)
            .filter(|&e| t.entries()[at + e] != HOST)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let cards: Vec<&placement::Segment> = row
            .segments
            .iter()
            .filter(|s| matches!(s.device, Device::Card(_)))
            .collect();
        if cards.is_empty() && ids.is_empty() {
            continue;
        }
        let [card] = cards[..] else {
            return Err(format!(
                "{}: {} card segments; --card-table places one card",
                tensor.name,
                cards.len()
            )
            .into());
        };
        let Device::Card(c) = card.device else {
            return Err(format!("{}: a card segment off the card", tensor.name).into());
        };
        let len = ids.len();
        let new = placement::routed_row(
            row.tensor,
            tensor,
            c,
            ExpertList::new(ids, model.experts)?,
            model,
        )?;
        let fmt = new
            .segments
            .iter()
            .find(|s| s.device == Device::Card(c))
            .map(|s| s.format);
        if fmt != Some(card.format) {
            return Err(format!(
                "{}: routed_row's card format {fmt:?}, the planner's {:?}",
                tensor.name, card.format
            )
            .into());
        }
        *row = Row {
            segments: new.segments,
            ..row.clone()
        };
        if let Some(nl) = plan.n_l.get_mut(l) {
            *nl = u64::try_from(len)?;
        }
    }
    Ok(())
}

/// The `card table` record of a `--card-table` load: its card's copy
/// `loaded` against `file`'s table, read from `path`: the experts on one
/// side only, and those on the card in both at another slot.
pub fn card_table(loaded: &CardTable, file: &CardTable, path: &Path) -> Result<Record, GateError> {
    let d = loaded.sets_vs(file)?;
    Ok(Record::new(&record::CARD_TABLE)
        .u("layers", loaded.layers().len())
        .u("on_card", loaded.on_card())
        .u("set_diff", d.set)
        .u("slot_diff", d.slot)
        .w("path", path.display()))
}

#[cfg(test)]
mod tests {
    use model::placement::devices::census_usable;
    use model::placement::workstation::{
        self, A6000, CardSpec, DeviceInfo, RTX_3090, TierBatchBytes,
    };
    use model::placement::{Device, Machine};

    use super::{
        BreakEven, Diag, DraftSeat, GateError, Known, Place, TierRule, off_host, refuse_card_lists,
        refuse_unknown,
    };
    use std::net::SocketAddr;

    const BATCH: TierBatchBytes = workstation::tier_batch_bytes(5120, 2304, 6, 512);

    /// Every alias and its list spelling is one placement, printed by the
    /// alias, and plans the machine its plan function makes: `a` and `a6000`
    /// plan (a), `gate` and `3090` the gate plan, `bp` and `a6000+3090` plan
    /// (b′), with and without a draft, for every layer count the plans see.
    #[test]
    fn aliases_and_their_lists_plan_as_before() {
        for (words, want) in [
            (["a", "a6000", "A6000"], Place::A),
            (["gate", "3090", "3090"], Place::Gate),
            (["bp", "a6000+3090", "A6000+3090"], Place::Bp),
        ] {
            for w in words {
                let p = Place::parse(w).unwrap_or_else(|e| panic!("{w}: {e}"));
                assert_eq!(p, want, "{w}");
                assert_eq!(p.name(), want.name(), "{w}");
            }
        }
        assert_eq!(
            (Place::A.name(), Place::Gate.name(), Place::Bp.name()),
            ("a", "gate", "bp")
        );
        assert_eq!(Place::A.cards(), vec![A6000.name]);
        assert_eq!(Place::Gate.cards(), vec![RTX_3090.name]);
        assert_eq!(Place::Bp.cards(), vec![A6000.name, RTX_3090.name]);
        assert_eq!(Place::Bp.tier_cards(), vec![RTX_3090.name]);
        assert!(Place::A.tier_cards().is_empty() && Place::Gate.tier_cards().is_empty());
        // PIN(2026-10-24): a draft's card is a census fact (`draft_on`), so a placement not yet
        // resolved against one names no device, the tier's included; was the tier card's name.
        assert_eq!(
            [Place::A, Place::Gate, Place::Bp].map(|p| p.draft_device().is_some()),
            [false, false, false]
        );
        let a = Place::A.machine(None, None).expect("a");
        let gate = Place::Gate.machine(None, None).expect("gate");
        for layers in [0, 1, 43, 47, 48, 78] {
            assert_eq!(a(layers), workstation::plan_a(layers));
            assert_eq!(gate(layers), workstation::plan_gate(layers));
            for draft in [None, Some(1_843_200_000)] {
                let seat = |at: usize, holder: Option<CardSpec>| {
                    draft.map(|bytes| DraftSeat {
                        reserve: workstation::draft_reserve(Device::Card(at), bytes),
                        holder,
                    })
                };
                let bp = Place::Bp.machine(seat(1, None), Some(BATCH));
                assert_eq!(
                    bp.expect("bp")(layers),
                    workstation::plan_bp(layers, draft, BATCH)
                );
                let on_a = Place::A.machine(seat(0, None), None).expect("a");
                assert_eq!(
                    on_a(layers),
                    workstation::plan_a_draft(layers, draft).expect("the stage holds the draft")
                );
                // PIN(2026-10-24): the holder is the draft seat's, so no draft plans no holder card;
                // was the holder planned beside no draft.
                let held = Place::A.machine(seat(1, Some(RTX_3090)), None).expect("a");
                let holders: &[CardSpec] = if draft.is_some() { &[RTX_3090] } else { &[] };
                assert_eq!(
                    held(layers),
                    workstation::plan_holding(
                        layers,
                        A6000,
                        &[],
                        holders,
                        seat(1, None).map(|s| s.reserve),
                        workstation::NO_TIER_BATCH
                    )
                    .expect("the 3090 holds the draft")
                );
                assert_eq!(held(layers).tiers.len(), usize::from(draft.is_some()));
            }
        }
    }

    /// `tp` is its own word: every other word parses as before and is not `tp`, and `tp`
    /// resolves against no census.
    #[test]
    fn tp_is_a_word_of_its_own() {
        for w in ["a", "gate", "bp", "a6000", "3090+a6000", "cuda1+0"] {
            assert!(!Place::parse(w).unwrap().tp(), "{w}");
        }
        let tp = Place::parse("tp").unwrap();
        assert!(tp.tp() && tp.name() == "tp" && tp.tier_cards().is_empty());
        assert_ne!(tp, Place::A);
        let e = tp.on(&census(&["A6000"])).unwrap_err().to_string();
        assert!(e.starts_with("--place tp: "), "{e}");
    }

    /// Any card may be the stage: a list word of this workstation's names
    /// plans its stage on its first card, and a list of ordinals plans
    /// only once resolved.
    #[test]
    fn any_card_is_the_stage() {
        for w in ["3090+a6000", "3090+A6000"] {
            let p = Place::parse(w).unwrap_or_else(|e| panic!("{w}: {e}"));
            assert_eq!(p.name(), "3090+a6000");
            let m = p.machine(None, Some(BATCH)).expect("3090+a6000");
            assert_eq!(
                m(43),
                workstation::plan_tiers(43, RTX_3090, &[A6000], None, BATCH).expect("plan")
            );
        }
        let ordinals = Place::parse("cuda1+0").expect("ordinals");
        assert_eq!(ordinals.cards(), vec!["cuda1", "cuda0"]);
        let unresolved = match ordinals.machine(None, Some(BATCH)) {
            Ok(_) => panic!("an unresolved ordinal was planned"),
            Err(e) => e.to_string(),
        };
        assert!(
            unresolved.contains("cuda1 is a device of this process's census"),
            "{unresolved}"
        );
    }

    /// On this workstation's census, in both of its enumeration orders, the
    /// aliases resolve to the cards they name with no census, each now
    /// naming its device; a list of ordinals plans the cards it numbers.
    /// Names that spell an alias but land elsewhere print as their names.
    #[test]
    fn resolved_on_the_box_plans_as_before() {
        // The plan's shape is what this test holds; the census readings the
        // resolution fills in (the device binding, the free bytes and their
        // holders) ride whatever card state the sitting meets.
        let strip = |mut m: Machine| {
            for c in m.cards.iter_mut().chain(m.tiers.iter_mut()) {
                c.device = None;
                c.free_bytes = None;
                c.held_by = None;
            }
            m
        };
        for order in [["3090", "A6000"], ["A6000", "3090"]] {
            let census = census(&order);
            let a6000 = u32::try_from(order.iter().position(|n| *n == "A6000").expect("A6000"))
                .expect("small");
            let a = Place::A.on(&census).expect("a");
            assert_eq!((a, a.name(), a.cards()), (Place::A, "a", vec![A6000.name]));
            let m = a.machine(None, None).expect("a")(43);
            assert_eq!(m.cards[0].device.map(|d| d.ordinal), Some(a6000));
            assert_eq!(strip(m), workstation::plan_a(43));
            let bp = Place::Bp.on(&census).expect("bp");
            assert_eq!(bp.draft_device().map(|d| d.ordinal), Some(1 - a6000));
            let m = bp.machine(None, Some(BATCH)).expect("bp")(43);
            assert_eq!(strip(m), workstation::plan_bp(43, None, BATCH));
            let gate = Place::Gate.on(&census).expect("gate");
            assert_eq!(
                strip(gate.machine(None, None).expect("gate")(43)),
                workstation::plan_gate(43)
            );
            assert_eq!((a.draft_device(), gate.draft_device()), (None, None));
            let w = format!("{a6000}+{}", 1 - a6000);
            let list = Place::parse(&w)
                .expect("ordinals")
                .on(&census)
                .expect("resolved");
            assert_eq!(
                (list.name(), list.cards()),
                (w.as_str(), vec!["A6000", "3090"])
            );
            let m = list.machine(None, Some(BATCH)).expect("list")(43);
            assert_eq!(strip(m), workstation::plan_bp(43, None, BATCH));
        }
        let ada = census(&["NVIDIA RTX 6000 Ada Generation", "A6000"]);
        let named = Place::parse("a6000").expect("a6000");
        assert_eq!(named.name(), "a");
        let on = named.on(&ada).expect("the A6000");
        assert_eq!((on.name(), on.cards()), ("a6000", vec!["A6000"]));
        let e = named
            .on(&census(&["3090"]))
            .expect_err("no A6000")
            .to_string();
        assert!(
            e.starts_with("--place a6000: no visible device is A6000 (visible: cuda0"),
            "{e}"
        );
        let largest = Place::A.on(&ada).expect("a");
        assert_eq!(largest.cards(), vec!["RTX_6000_Ada_Generation"]);
        let two = census(&["3090", "3090"]);
        let gate = Place::Gate.on(&two).expect("two 3090s");
        assert_eq!(
            gate.card_specs().expect("resolved")[0].device,
            Some(two[0].device())
        );
        let bp = Place::Bp.on(&two).expect("two 3090s");
        assert_eq!(bp.cards(), vec!["3090", "3090"]);
        let m = bp.machine(None, Some(BATCH)).expect("stage + tier")(43);
        assert_eq!(m.tiers[0].device.map(|d| d.ordinal), Some(1));
    }

    /// A tier node's census is read before the plan; a loopback node shares this
    /// host's pool, and a node on another host plans its card in its own.
    #[test]
    fn loopback_census_is_required_without_adding_a_pool() {
        use bloomery_gpu::host::node::wire::{Out, Tag};
        use std::io::{Read, Write};
        use std::time::{Duration, Instant};

        let mut own = census(&["NVIDIA GB10"]);
        own[0].integrated = true;
        own[0].free_bytes = 90 << 30;
        own[0].total_bytes = 128 << 30;
        let mut remote = own[0].clone();
        remote.uuid = [7; 16];
        remote.free_bytes = 60 << 30;
        let remote_spec = workstation::spec_of_device(&remote);
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("an address");
        let addr = listener.local_addr().expect("the address");
        drop(listener);
        assert!(
            Place::parse("0+1")
                .expect("a place")
                .on_nodes(&own, &[addr])
                .is_err()
        );
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("a census server");
        let addr = listener.local_addr().expect("the server address");
        listener.set_nonblocking(true).expect("bounded accept");
        let (served, exchange) = std::sync::mpsc::channel();
        let (server, _) = threads::helper::spawn_helper(
            "census-fixture",
            threads::helper::Placement::Float,
            move || {
                let _ = served.send((|| -> std::io::Result<()> {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    let mut stream = loop {
                        match listener.accept() {
                            Ok((stream, _)) => break stream,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                if Instant::now() >= deadline {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::TimedOut,
                                        "no remote census requested",
                                    ));
                                }
                                std::thread::sleep(Duration::from_millis(10));
                            }
                            Err(e) => return Err(e),
                        }
                    };
                    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                    let mut payload = Out::default();
                    bloomery_gpu::host::node::NodeCensus {
                        devices: vec![remote],
                        host_room: (u64::MAX, workstation::HostRead::Given),
                        arena_given: None,
                    }
                    .write(&mut payload);
                    stream.write_all(
                        &u32::try_from(1 + payload.done().len())
                            .expect("a frame")
                            .to_le_bytes(),
                    )?;
                    stream.write_all(&[Tag::Census as u8])?;
                    stream.write_all(payload.done())?;
                    let mut close = [0; 5];
                    stream.read_exact(&mut close)?;
                    assert_eq!(close, [1, 0, 0, 0, Tag::Close as u8]);
                    stream.write_all(&[1, 0, 0, 0, Tag::Ok as u8])
                })());
            },
        )
        .expect("a census server thread");
        let place = Place::parse("0+1")
            .expect("a place")
            .on_nodes(&own, &[addr]);
        server.join().expect("the census server");
        exchange
            .recv()
            .expect("the census server")
            .expect("the census exchange");
        let place = place.expect("resolved against the remote census");
        let specs = place.card_specs().expect("the census specs");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0], workstation::spec_of_device(&own[0]));
        assert_eq!(specs[1], remote_spec);
        let machine = place.machine(None, Some(BATCH)).expect("loopback tiers")(3);
        assert_eq!(machine.nodes.len(), 1, "loopback shares this host's pool");
        assert_eq!(
            machine.unified(),
            workstation::unified_of(&workstation::spec_of_device(&own[0]))
        );
        assert_eq!(machine.tiers[0].device, remote_spec.device);
        assert_eq!(machine.tiers[0].free_bytes, remote_spec.free_bytes);
        // A node on another host plans its card in its own pool; a loopback
        // address or a device of this process's census is this host. Every
        // machine over the placement holds that topology, the common plan's
        // and a family's own layout, whose cards and node 0 stay its own.
        let lan: SocketAddr = "198.18.4.1:19882".parse().expect("an address");
        assert!(off_host(&own, lan, &remote_spec));
        assert!(!off_host(&own, addr, &remote_spec));
        assert!(!off_host(&own, lan, &workstation::spec_of_device(&own[0])));
        let layout = |stage: CardSpec, tiers: &[CardSpec]| -> Result<Machine, GateError> {
            let mut m = workstation::plan_tiers(3, stage, tiers, None, BATCH)?;
            m.cards[0].scratch_bytes += 1;
            m.host_mut().reserves.push(("family".to_string(), 1));
            Ok(m)
        };
        let laid = layout(specs[0], &specs[1..]).expect("a layout");
        let apart = vec![workstation::node_of(&remote_spec)];
        for (at, tier_nodes) in [(addr, Vec::new()), (lan, apart)] {
            let place = Place::parse("0+1")
                .expect("a place")
                .at_nodes(&own, std::iter::once((at, remote_spec)))
                .expect("resolved");
            let common = place.machine(None, Some(BATCH)).expect("the tier plan")(3);
            assert_eq!(common.nodes[0], machine.nodes[0], "{at}");
            assert_eq!(common.nodes[1..], tier_nodes, "{at}");
            let own_layout = place.machine_with(layout).expect("the family's layout");
            assert_eq!(own_layout.nodes[0], laid.nodes[0], "{at}");
            assert_eq!(own_layout.nodes[1..], tier_nodes, "{at}");
            assert_eq!(
                (&own_layout.cards, &own_layout.tiers),
                (&laid.cards, &laid.tiers),
                "{at}"
            );
        }
        let single = Place::parse("0")
            .expect("a place")
            .on(&own)
            .expect("resolved");
        assert_eq!(
            single.machine(None, None).expect("one node")(3).nodes.len(),
            1
        );
    }

    /// The unset rule ([`Place::choose`]) at a GLM-like [`TierRule`]: one
    /// card is `a` on it, whichever card it is, whatever the rule and the
    /// count; a body that serves no tier card and a rule with no break-even
    /// run `a` without asking the plan; with both, the offer keeps its tier
    /// card at and past the break-even — in both enumeration orders, on two
    /// of one name, and of three the two largest — and is `a` on the larger
    /// under it, for no expert, and for a plan that refuses; the flag set
    /// names its word, the plan never asked; of four equal cards, the two
    /// lowest ordinals; no device is `a`'s refusal by name. Where the rule
    /// must not ask the plan, the closure it would ask panics.
    #[test]
    fn the_unset_rule() {
        const BREAK_EVEN: u64 = 526;
        const PAYS: &str = "two cards, tier at or past the break-even";
        const SHORT: &str = "two cards, tier under the break-even";
        const NONE_YET: &str = "two cards, no sitting has shown the tier not slower";
        const NO_TIER: &str = "two cards, the body serves no tier card";
        const REFUSED: &str = "two cards, the tier's plan refused";
        const EMPTY: &str = "two cards, the tier holds no expert";
        const RULE: TierRule = TierRule {
            tiers: 1,
            break_even: Some(BREAK_EVEN),
            basis: "docs/cards/glmbp-ab.card",
        };
        /// What the rule chose, as the test holds it: the word, the cards,
        /// the draft's device, the why and the tier count.
        type Chosen = (
            &'static str,
            Vec<&'static str>,
            Option<u32>,
            &'static str,
            Option<u64>,
        );
        let at = |flag: Option<Place>,
                  names: &[&str],
                  rule: TierRule,
                  tier_of: &dyn Fn(Place) -> Result<u64, GateError>|
         -> Result<Chosen, String> {
            Place::choose(flag, &census(names), rule, tier_of)
                .map(|w| {
                    (
                        w.place.name(),
                        w.place.cards(),
                        w.place.draft_device().map(|d| d.ordinal),
                        w.why,
                        w.tier_experts,
                    )
                })
                .map_err(|e| e.to_string())
        };
        fn count(n: u64) -> impl Fn(Place) -> Result<u64, GateError> {
            move |_| Ok(n)
        }
        fn refused(_: Place) -> Result<u64, GateError> {
            Err("the plan is refused: the tier card would stay idle (IdleTier)".into())
        }
        fn never_asked(_: Place) -> Result<u64, GateError> {
            panic!("the rule asked the plan for its tier count where it must not")
        }
        // One card is the offer itself, whatever the card, the rule and the
        // count.
        for (names, card) in [
            (&["A6000"][..], "A6000"),
            (&["3090"][..], "3090"),
            (
                &["Blackwell"][..],
                "RTX_PRO_6000_Blackwell_Workstation_Edition",
            ),
        ] {
            for rule in [
                RULE,
                TierRule {
                    tiers: 0,
                    break_even: None,
                    basis: "",
                },
            ] {
                assert_eq!(
                    at(None, names, rule, &never_asked),
                    Ok(("a", vec![card], None, "one card", None)),
                    "{names:?}"
                );
            }
        }
        // Two cards the plan is never asked about.
        for (rule, why) in [
            (
                TierRule {
                    tiers: 0,
                    break_even: Some(BREAK_EVEN),
                    basis: "",
                },
                NO_TIER,
            ),
            (
                TierRule {
                    tiers: 1,
                    break_even: None,
                    basis: "",
                },
                NONE_YET,
            ),
        ] {
            assert_eq!(
                at(None, &["3090", "A6000"], rule, &never_asked),
                Ok(("a", vec!["A6000"], None, why, None))
            );
        }
        // Two cards by the plan's count, in both enumeration orders; n 0 is
        // the no-expert arm below, whatever the break-even is.
        for (order, tier) in [(["3090", "A6000"], 0), (["A6000", "3090"], 1)] {
            for n in [1, BREAK_EVEN - 1] {
                assert_eq!(
                    at(None, &order, RULE, &count(n)),
                    Ok(("a", vec!["A6000"], None, SHORT, Some(n))),
                    "{order:?} {n}"
                );
            }
            assert_eq!(
                at(None, &order, RULE, &count(BREAK_EVEN)),
                Ok((
                    "bp",
                    vec!["A6000", "3090"],
                    Some(tier),
                    PAYS,
                    Some(BREAK_EVEN)
                )),
                "{order:?}"
            );
            assert_eq!(
                at(None, &order, RULE, &count(BREAK_EVEN + 1)),
                Ok((
                    "bp",
                    vec!["A6000", "3090"],
                    Some(tier),
                    PAYS,
                    Some(BREAK_EVEN + 1)
                )),
                "{order:?}"
            );
            assert_eq!(
                at(None, &order, RULE, &refused),
                Ok(("a", vec!["A6000"], None, REFUSED, None)),
                "{order:?}"
            );
        }
        // No expert is `a` whatever the break-even is.
        for e in [0, BREAK_EVEN] {
            assert_eq!(
                at(
                    None,
                    &["A6000", "3090"],
                    TierRule {
                        tiers: 1,
                        break_even: Some(e),
                        basis: ""
                    },
                    &count(0)
                ),
                Ok(("a", vec!["A6000"], None, EMPTY, Some(0))),
                "e {e}"
            );
        }
        assert_eq!(
            at(
                None,
                &["A6000", "3090"],
                TierRule {
                    tiers: 1,
                    break_even: Some(0),
                    basis: ""
                },
                &count(1)
            ),
            Ok(("bp", vec!["A6000", "3090"], Some(1), PAYS, Some(1)))
        );
        assert_eq!(
            at(None, &["3090", "3090"], RULE, &count(BREAK_EVEN)),
            Ok(("bp", vec!["3090", "3090"], Some(1), PAYS, Some(BREAK_EVEN)))
        );
        assert_eq!(
            at(
                None,
                &["3090", "NVIDIA RTX 6000 Ada Generation", "A6000"],
                RULE,
                &count(BREAK_EVEN)
            ),
            Ok((
                "bp",
                vec!["RTX_6000_Ada_Generation", "A6000"],
                Some(2),
                PAYS,
                Some(BREAK_EVEN)
            ))
        );
        // The flag set names its word, the plan never asked.
        for (word, p) in [("a", Place::A), ("bp", Place::Bp)] {
            assert_eq!(
                at(Some(p), &["3090", "A6000"], RULE, &never_asked),
                Ok((
                    word,
                    if word == "a" {
                        vec!["A6000"]
                    } else {
                        vec!["A6000", "3090"]
                    },
                    if word == "a" { None } else { Some(0) },
                    "set",
                    None
                ))
            );
        }
        // Four equal cards: `bp` takes the two lowest ordinals, in either
        // enumeration order.
        for (order, stage, tier) in [
            (["a", "b", "c", "d"], "fake_card_a", "fake_card_b"),
            (["d", "c", "b", "a"], "fake_card_d", "fake_card_c"),
        ] {
            let names: Vec<String> = order
                .map(|w| format!("fake card {w}"))
                .into_iter()
                .collect();
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            assert_eq!(
                at(None, &names, RULE, &count(BREAK_EVEN)),
                Ok(("bp", vec![stage, tier], Some(1), PAYS, Some(BREAK_EVEN))),
                "{order:?}"
            );
        }
        let e = at(None, &[], RULE, &never_asked).expect_err("no device");
        assert!(
            e.starts_with(
                "--place a: the placement takes the largest visible card, and 0 devices are visible"
            ),
            "{e}"
        );
    }

    /// The Qwen3.8 family's break-even as its rule states it
    /// (`shared/qwen38_place.rs`'s `Q38_RULE`, mirrored here — the bins'
    /// module is out of the lib's reach; the card file that decides it holds
    /// the sitting and the derivation): a tier count at or past it keeps the
    /// offer `bp`, one below it and no expert run `a`, the count named.
    #[test]
    fn the_q38_break_even_keeps_bp() {
        const BREAK_EVEN: u64 = 1_152;
        const RULE: TierRule = TierRule {
            tiers: 1,
            break_even: Some(BREAK_EVEN),
            basis: "docs/cards/q38bpbug-ab.card",
        };
        let at = |tier: u64| {
            let w = Place::choose(None, &census(&["3090", "A6000"]), RULE, move |_| Ok(tier))
                .unwrap_or_else(|e| panic!("{e}"));
            (w.place.name(), w.why, w.tier_experts)
        };
        for n in [BREAK_EVEN, BREAK_EVEN + 1] {
            assert_eq!(
                at(n),
                ("bp", "two cards, tier at or past the break-even", Some(n)),
                "{n}"
            );
        }
        assert_eq!(
            at(BREAK_EVEN - 1),
            (
                "a",
                "two cards, tier under the break-even",
                Some(BREAK_EVEN - 1)
            )
        );
        assert_eq!(at(0), ("a", "two cards, the tier holds no expert", Some(0)));
    }

    /// The Qwen3.8 seat's break-even at its nominal reply of 277 tokens, on
    /// the seat's rates (`serve_seats/qwen38.rs`: `PLAIN_TPS`, `DRAFTED_TPS`,
    /// `PROMPT_IDS_PER_S`, and `BP_PROMPT_RATIO` for `bp`): the ubatch walk's
    /// rate under `a`, that rate times the ratio under `bp`.
    #[test]
    fn the_break_even_at_the_nominal_reply() {
        let a = BreakEven::new(57.88, 79.33, 1224.1);
        let bp = BreakEven::new(57.88, 79.33, 1224.1 * 1.379);
        assert_eq!((a.at(277), bp.at(277)), (1585, 2185));
    }

    /// A fake census of devices by short name, each its measured total
    /// (the A6000's, the 3090's, a 96 GB Blackwell's) or twice the A6000's,
    /// and its full usable bytes free (`census_usable`'s figure, a quiet
    /// card). Shared with `draft_card_tests` below.
    pub(super) fn census(names: &[&str]) -> Vec<DeviceInfo> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let (name, total_bytes, compute_capability) = match *n {
                    "3090" => (
                        "NVIDIA GeForce RTX 3090".to_string(),
                        25_351_356_416,
                        Some((8, 6)),
                    ),
                    "A6000" => ("NVIDIA RTX A6000".to_string(), 50_952_536_064, Some((8, 6))),
                    "Blackwell" => (
                        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".to_string(),
                        96 * 1024 * 1024 * 1024,
                        None,
                    ),
                    other => (other.to_string(), 2 * 50_952_536_064, None),
                };
                DeviceInfo {
                    ordinal: u32::try_from(i).expect("small"),
                    name,
                    total_bytes,
                    free_bytes: census_usable(total_bytes),
                    uuid: [u8::try_from(i).expect("small") + 1; 16],
                    pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                    held_by: None,
                    integrated: false,
                    compute_capability,
                    granule_bytes: None,
                }
            })
            .collect()
    }

    /// A launcher's table names the flags it reads: a typo, a stray word and a valued flag with no
    /// value are refused by name, a value that looks like a flag is a value, and the diagnosis
    /// flags count only where the table says so.
    #[test]
    fn a_launcher_refuses_a_flag_its_table_does_not_name() {
        let args = |line: &str| -> Vec<String> {
            std::iter::once("bin")
                .chain(line.split_whitespace())
                .map(str::to_owned)
                .collect()
        };
        let own = Known {
            bare: &["--time"],
            valued: &["-n", "--tokens"],
            diag: false,
        };
        let with_diag = Known { diag: true, ..own };
        let refused = |line: &str, known: Known<'_>| {
            refuse_unknown("bin", &args(line), known)
                .expect_err(line)
                .to_string()
        };
        assert!(refuse_unknown("bin", &args(""), own).is_ok());
        assert!(refuse_unknown("bin", &args("--time -n 4 --tokens 1,2"), own).is_ok());
        assert!(refuse_unknown("bin", &args("--tokens --time"), own).is_ok());
        assert!(refuse_unknown("bin", &args("--top2 3 --table --rows f"), with_diag).is_ok());
        let e = refused("--tiem", own);
        assert!(
            e.starts_with(
                "bin: unknown argument \"--tiem\"; its flags are --time -n <value> --tokens <value>"
            ),
            "{e}"
        );
        assert!(refused("stray", own).contains("unknown argument \"stray\""));
        assert!(refused("--table", own).contains("unknown argument \"--table\""));
        assert!(refused("--nope", with_diag).contains(Diag::USAGE));
        assert_eq!(refused("--time -n", own), "bin: -n needs a value");
        assert_eq!(refused("--top2", with_diag), "bin: --top2 needs a value");
    }

    /// The diagnosis flags' two name lists are the flags [`Diag::take`] takes, with and without a
    /// value, and no other.
    #[test]
    fn the_diagnosis_flag_tables_are_what_take_takes() {
        for flag in Diag::BARE {
            let mut none = std::iter::empty();
            assert!(Diag::default().take(flag, &mut none).expect(flag), "{flag}");
        }
        for flag in Diag::VALUED {
            let mut one = std::iter::once("1".to_owned());
            assert!(Diag::default().take(flag, &mut one).expect(flag), "{flag}");
            assert!(one.next().is_none(), "{flag} took its value");
        }
        assert!(
            !Diag::default()
                .take("--tokens", &mut std::iter::empty())
                .expect("not the diagnosis flags'")
        );
    }

    /// The levers `Place::on` reads reach the machine's cards through
    /// `Place::on_with`: a margin is the one value on every card, a list entry
    /// on the card of that ordinal alone, a card the margin names none at the
    /// plan's own margin, and a margin that leaves a card no usable byte is
    /// refused naming the card.
    #[test]
    fn a_margin_reaches_the_machines_cards() {
        use model::placement::card_budget::CardBudget;

        let two = census(&["A6000", "3090"]);
        let on = |m: &CardBudget| Place::Bp.on_with(&two, &CardBudget::Unset, m, &[]);
        let margins = |m: &CardBudget| {
            let kept = on(m).expect("a margin");
            let machine = kept.machine(None, Some(BATCH)).expect("stage + tier")(43);
            (machine.cards[0].margin_bytes, machine.tiers[0].margin_bytes)
        };
        assert_eq!(
            margins(&CardBudget::Unset),
            (workstation::MARGIN, workstation::MARGIN)
        );
        assert_eq!(margins(&CardBudget::Every(2 << 30)), (2 << 30, 2 << 30));
        assert_eq!(
            margins(&CardBudget::Per(vec![(1, 3 << 30)])),
            (workstation::MARGIN, 3 << 30)
        );
        assert_eq!(
            margins(&CardBudget::Per(vec![(0, 3 << 30)])),
            (3 << 30, workstation::MARGIN)
        );
        let e = on(&CardBudget::Per(vec![(1, RTX_3090.usable_bytes())]))
            .expect_err("no usable byte")
            .to_string();
        assert!(
            e.starts_with("--place bp: card 3090: a margin of") && e.contains("leaves none"),
            "{e}"
        );
    }

    /// The per-card levers are refused against the census and the tier nodes
    /// before any card resolves, each by name: a device the census lacks in
    /// either list, shares as a margin, and a list beside tier nodes; one value
    /// for every card passes everywhere.
    #[test]
    fn card_lists_are_refused_against_the_census_and_the_tier_nodes() {
        use model::placement::card_budget::CardBudget::{Every, Per, Shares, Unset};

        let two = census(&["A6000", "3090"]);
        let check = |budget, margin, nodes| refuse_card_lists("bp", &budget, &margin, &two, nodes);
        for ok in [
            check(Unset, Unset, 0),
            check(Per(vec![(1, 1)]), Per(vec![(0, 1)]), 0),
            check(Shares(vec![(0, 3), (1, 1)]), Every(1), 0),
            check(Every(1), Every(1), 2),
        ] {
            assert!(ok.is_ok(), "{ok:?}");
        }
        let refused = |r: Result<(), GateError>| r.expect_err("refused").to_string();
        let e = refused(check(Per(vec![(5, 1)]), Unset, 0));
        assert!(
            e.starts_with("--place bp: the per-card budget CUDA5=1 names CUDA5"),
            "{e}"
        );
        let e = refused(check(Unset, Per(vec![(5, 1)]), 0));
        assert!(
            e.starts_with("--place bp: the per-card margin CUDA5=1 names CUDA5"),
            "{e}"
        );
        let e = refused(check(Unset, Shares(vec![(0, 1)]), 0));
        assert!(
            e.contains("BLOOMERY_CARD_MARGIN takes bytes, not shares: CUDA0=1x"),
            "{e}"
        );
        for (budget, margin, lever) in [
            (Per(vec![(0, 1)]), Unset, "BLOOMERY_CARD_BUDGET=CUDA0=1"),
            (Shares(vec![(0, 1)]), Unset, "BLOOMERY_CARD_BUDGET=CUDA0=1x"),
            (Unset, Per(vec![(1, 2)]), "BLOOMERY_CARD_MARGIN=CUDA1=2"),
        ] {
            let e = refused(check(budget, margin, 1));
            assert!(
                e.contains(lever) && e.contains("BLOOMERY_TIER_NODES puts the tier cards on nodes"),
                "{e}"
            );
        }
        // `Place::on_with` runs the same refusals before it resolves a card or reads a node, each
        // lever under its own name and the tier nodes counted.
        let node: SocketAddr = "127.0.0.1:9".parse().expect("an address");
        for (budget, margin, nodes, want) in [
            (
                Per(vec![(5, 1)]),
                Unset,
                vec![],
                "the per-card budget CUDA5=1 names CUDA5",
            ),
            (
                Unset,
                Per(vec![(5, 1)]),
                vec![],
                "the per-card margin CUDA5=1 names CUDA5",
            ),
            (
                Unset,
                Shares(vec![(0, 1)]),
                vec![],
                "BLOOMERY_CARD_MARGIN takes bytes, not shares",
            ),
            (
                Per(vec![(0, 1)]),
                Unset,
                vec![node],
                "BLOOMERY_TIER_NODES puts the tier cards on nodes",
            ),
        ] {
            let e = Place::Bp
                .on_with(&two, &budget, &margin, &nodes)
                .expect_err(want)
                .to_string();
            assert!(e.starts_with("--place bp: ") && e.contains(want), "{e}");
        }
    }

    /// The word's refusals, each by name before any plan: more tiers than a
    /// slot map names, a name no card has, a card named twice; then the
    /// placement's: more tier cards than the body serves, and a draft or a
    /// batch beside no tier, or no batch beside one, in today's words.
    #[test]
    fn refusals_are_named_before_any_plan() {
        let refused = |w: &str| Place::parse(w).expect_err(w).to_string();
        let many = ["a6000"; 10].join("+");
        assert!(
            refused(&many).contains("names 9 tier cards, past the 8"),
            "{}",
            refused(&many)
        );
        assert!(
            refused("b").contains("names the card \"b\""),
            "{}",
            refused("b")
        );
        assert!(
            refused("a6000+4090").contains("names the card \"4090\""),
            "{}",
            refused("a6000+4090")
        );
        assert!(
            refused("3090+3090").contains("names the card 3090 twice"),
            "{}",
            refused("3090+3090")
        );
        assert_eq!(
            Place::Bp
                .serves("deepseek41", 0)
                .expect_err("k 1 > 0")
                .to_string(),
            "--place bp: 1 expert tier cards, and the deepseek41 body serves 0 at most"
        );
        assert!(Place::Bp.serves("deepseek41", 1).is_ok() && Place::A.serves("x", 0).is_ok());
        let machine_err =
            |p: Place, d: Option<DraftSeat>, b: Option<TierBatchBytes>| match p.machine(d, b) {
                Ok(_) => panic!("{} {d:?} {b:?} was planned", p.name()),
                Err(e) => e.to_string(),
            };
        let draft = |at: usize, holder: Option<CardSpec>| {
            Some(DraftSeat {
                reserve: workstation::draft_reserve(Device::Card(at), 1),
                holder,
            })
        };
        assert!(Place::A.machine(draft(0, None), None).is_ok());
        assert!(Place::A.machine(draft(1, Some(RTX_3090)), None).is_ok());
        assert_eq!(
            machine_err(Place::A, draft(1, None), None),
            "the DSpark draft reserve is on Card(1), outside the plan: a reserve sits on one of \
             the plan's 1 cards, stage first, or on the host"
        );
        assert_eq!(
            machine_err(Place::A, draft(2, Some(RTX_3090)), None),
            "the DSpark draft reserve is on Card(2), outside the plan: a reserve sits on one of \
             the plan's 2 cards, stage first, or on the host"
        );
        assert!(
            machine_err(Place::A, draft(1, Some(A6000)), None)
                .contains("2 plan cards are this one device"),
            "a holder that is the stage's own device is the plan's refusal"
        );
        let own_pools = Place {
            own_pools: true,
            ..Place::Bp
        };
        assert!(
            machine_err(own_pools, draft(2, Some(RTX_3090)), Some(BATCH))
                .contains("a reserve-only member of this process's node"),
            "a holder beside tiers that plan in nodes of their own is refused by name"
        );
        assert_eq!(
            machine_err(Place::Gate, None, Some(BATCH)),
            "--place gate: no tier card serves a prompt batch; the tier's batch reserve is \
             outside the plan"
        );
        assert_eq!(
            machine_err(Place::Bp, None, None),
            "--place bp: the plan reserves the expert tier's prompt-batch bytes, and none were \
             given"
        );
    }

    /// A family's plan refusal for the shared seat rules, carrying a placement
    /// error or none, so the tests read no model's own error type.
    #[derive(Debug)]
    struct FamilyRefusal(Option<model::placement::PlacementError>);

    impl std::fmt::Display for FamilyRefusal {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the family's plan refusal")
        }
    }

    impl std::error::Error for FamilyRefusal {}

    impl model::placement::ctx_ask::PlanRefusal for FamilyRefusal {
        fn placement(&self) -> Option<&model::placement::PlacementError> {
            self.0.as_ref()
        }
        fn violations(&self) -> Option<&[model::placement::Violation]> {
            None
        }
    }

    /// A load's refusal of a plan that pages experts is short on the host wherever the seat's
    /// error carries it in its source chain (the gpu crate's plan refusal wraps it), so a
    /// context or slot search fits below it. Mutant: only the error itself read, so the wrapped
    /// refusal stays the seat's error.
    #[test]
    fn a_wrapped_unpaged_refusal_is_short_on_the_host() {
        use model::placement::ctx_ask::{Short, Side, Unpaged};

        #[derive(Debug)]
        struct Wrapped(Unpaged);
        impl std::fmt::Display for Wrapped {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("the load")
            }
        }
        impl std::error::Error for Wrapped {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let refusal = Unpaged {
            paged: 3,
            need: 10,
            room: 7,
        };
        let e: GateError = Box::new(Wrapped(refusal));
        assert_eq!(
            super::short_in::<FamilyRefusal>(&e),
            Some(Short {
                on: Side::Host,
                need: 10,
                room: 7
            })
        );
        let other: GateError = "a refusal of another kind".into();
        assert_eq!(super::short_in::<FamilyRefusal>(&other), None);
    }

    /// An unset draft yields only where the plain plan serves twice the
    /// positions or more and the drafted plan fitted its context or holds none
    /// (a floor short, a set context short, a host room the NVMe tier's arena
    /// does not fit, also under a named reserve); a refusal of any other kind
    /// passes up with its name, and a set draft never yields.
    /// Mutant: every drafted refusal read as no context (the draft goes off on
    /// a read error).
    #[test]
    fn only_a_context_no_plan_holds_yields_the_draft() {
        use model::placement::PlacementError;
        use model::placement::ctx_ask::{Short, Side};

        use super::{CtxDoesNotFit, DraftYield, FloorShort, draft_yield};

        // A kept draft gives back the context note, which is process-wide.
        let _note = crate::start::slot_note_tests();
        let short = Short {
            on: Side::Host,
            need: 9,
            room: 8,
        };
        let floor_short = || -> GateError {
            Box::new(FloorShort {
                floor: 4096,
                short,
                text: "short at the floor".into(),
            })
        };
        let host_room_floor = || PlacementError::HostRoomFloor {
            room: 8,
            floor: 9,
            base: 1,
            layer: 1,
            need: 9,
            arena: None,
        };
        let host_floor = || -> GateError { Box::new(FamilyRefusal(Some(host_room_floor()))) };
        let bare_host_floor = || -> GateError { Box::new(host_room_floor()) };
        let arena_past = || -> GateError {
            Box::new(FamilyRefusal(Some(PlacementError::NvTierArena {
                arena: 9,
                room: 16,
                floor: 8,
            })))
        };
        let arena_under = || -> GateError {
            Box::new(FamilyRefusal(Some(PlacementError::NvTierArenaUnder {
                arena: 1,
                least: model::placement::ArenaFloor::new(8, 5, 0, 4).expect("a small arena floor"),
            })))
        };
        let named_host_floor = || -> GateError {
            crate::beside::refusal(
                crate::beside::Beside {
                    image: Some(1),
                    embedding: None,
                },
                FamilyRefusal(Some(host_room_floor())),
            )
        };
        let named_draft_host = || -> GateError {
            crate::beside::refusal(
                crate::beside::Beside {
                    image: Some(1),
                    embedding: None,
                },
                FamilyRefusal(Some(PlacementError::DraftHostRoom { room: 8, draft: 9 })),
            )
        };
        let family_other = || -> GateError { Box::new(FamilyRefusal(None)) };
        let host_short = || -> GateError {
            use model::placement::workstation::{HostNeed, HostRead};
            let need = HostNeed {
                experts: 0,
                tables: 0,
                shadows: 0,
                reserves: 9,
                extra: 0,
                read_ahead: 0,
                arena: 0,
                cards: 0,
                os: 0,
                card_budget: false,
            };
            let short = need
                .check(&(8, HostRead::Given))
                .expect_err("a discrete host shortage");
            assert!(short.pool_short().is_none());
            Box::new(short)
        };
        let set_short = || -> GateError {
            Box::new(CtxDoesNotFit {
                largest_total: None,
                largest_slot: None,
                short,
                text: "the set context does not fit".into(),
            })
        };
        let other = || -> GateError { "a read of the folder failed".into() };
        let fitted = |&(c, fit): &(usize, bool)| fit.then_some(c);
        let rule = |asks: bool, drafted: Result<(usize, bool), GateError>, without: usize| {
            draft_yield::<FamilyRefusal, _, _>(asks, drafted, fitted, || Ok(("plain", without, 7)))
                .map(|y| match y {
                    DraftYield::Keep((c, _)) => format!("keep {c}"),
                    DraftYield::Yield(p, y) => format!("{p} {}", y.with),
                })
                .map_err(|e| e.to_string())
        };
        assert_eq!(rule(true, Err(floor_short()), 16_384), Ok("plain 0".into()));
        assert_eq!(rule(true, Err(host_short()), 16_384), Ok("plain 0".into()));
        assert_eq!(rule(true, Err(host_floor()), 16_384), Ok("plain 0".into()));
        assert_eq!(
            rule(true, Err(bare_host_floor()), 16_384),
            Ok("plain 0".into())
        );
        assert_eq!(rule(true, Err(arena_past()), 16_384), Ok("plain 0".into()));
        assert_eq!(rule(true, Err(arena_under()), 16_384), Ok("plain 0".into()));
        assert_eq!(
            rule(true, Err(named_host_floor()), 16_384),
            Ok("plain 0".into())
        );
        assert_eq!(
            rule(true, Err(named_draft_host()), 16_384),
            Ok("plain 0".into())
        );
        assert_eq!(
            rule(true, Err(family_other()), 16_384),
            Err("the family's plan refusal".into())
        );
        assert_eq!(rule(true, Err(set_short()), 4096), Ok("plain 0".into()));
        assert_eq!(
            rule(true, Err(other()), 16_384),
            Err("a read of the folder failed".into())
        );
        assert_eq!(rule(true, Ok((4096, true)), 8192), Ok("plain 4096".into()));
        assert_eq!(rule(true, Ok((4096, true)), 8191), Ok("keep 4096".into()));
        assert_eq!(
            rule(true, Ok((4096, false)), 16_384),
            Ok("keep 4096".into())
        );
        assert_eq!(
            rule(false, Err(floor_short()), 16_384),
            Err("short at the floor".into())
        );
        assert_eq!(rule(false, Ok((4096, true)), 8192), Ok("keep 4096".into()));
    }
}

/// Where a DSpark draft sits in a placement ([`Place::draft_on`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftCard {
    /// The card's place in the plan's device list: the placement's cards, the
    /// stage first, then — when the card is none of them — the one more it
    /// makes.
    pub at: Device,
    pub spec: CardSpec,
    /// Whether the card is none of the placement's and joins the plan as a
    /// reserve-only member to hold the draft.
    pub holder: bool,
}

/// A DSpark draft's seat in a plan: the reserve its plan takes on the card it
/// sits on and, where that card joins the plan as a reserve-only member, the
/// card ([`Place::machine`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftSeat {
    pub reserve: Reserve,
    pub holder: Option<CardSpec>,
}

impl DraftCard {
    /// The seat of a draft of `bytes` on this card.
    #[must_use]
    pub fn seat(self, bytes: u64) -> DraftSeat {
        DraftSeat {
            reserve: workstation::draft_reserve(self.at, bytes),
            holder: self.holder.then_some(self.spec),
        }
    }
}

/// The DSpark draft's card, by device.
impl Place {
    /// The card a DSpark draft loads on under this placement, on `census`, by
    /// its place in the plan's device list (the placement's cards, the stage
    /// first), with its device. `set` (`BLOOMERY_DSPARK_CARD`) is one card as
    /// a `--place` list word names one — a CUDA ordinal (`1`, `cuda1`) or a
    /// card name exactly one visible device carries ([`workstation::resolve`]).
    /// Unset, the draft sits on the first tier card where the placement has
    /// one (its plan holds the rest of that card for experts, the stage's
    /// prefix unmoved), else on the visible device of the fewest usable bytes,
    /// ties to the higher ordinal — the 3090 on this workstation wherever it is
    /// in view, the stage itself when it is the only card or the placement
    /// is `gate`. A card of the placement is its place in the list; any other
    /// visible device is the plan's one more card, a reserve-only member that
    /// holds the draft and no expert ([`DraftCard::holder`]), so the draft's
    /// bytes leave that card's budget in the plan and no card that holds
    /// experts loses one. Each refusal by name.
    pub fn draft_on(
        self,
        set: Option<&str>,
        census: &[DeviceInfo],
    ) -> Result<DraftCard, GateError> {
        let margin = bloomery_levers::card_margin()?;
        let placed = if self.resolved {
            self
        } else {
            self.on(census)?
        };
        placed.draft_card(set, census, &margin)
    }

    /// [`Place::draft_on`] on a resolved placement (`self`), the margin lever's value
    /// `margin` given parsed: a card that joins the plan to hold the draft keeps the margin
    /// the lever gives it ([`card_budget::keep_free`]), as the placement's own cards do.
    fn draft_card(
        self,
        set: Option<&str>,
        census: &[DeviceInfo],
        margin: &card_budget::CardBudget,
    ) -> Result<DraftCard, GateError> {
        let lever = |v: &str, e: &dyn std::fmt::Display| -> GateError {
            format!("BLOOMERY_DSPARK_CARD={v:?}: {e}").into()
        };
        let cards = self.card_specs()?;
        let spec = match set {
            Some(v) => {
                let picks = workstation::word_picks(v, 0).map_err(|e| match e {
                    workstation::WordError::TooManyTiers { .. } => lever(
                        v,
                        &"the lever names one card (a CUDA ordinal or a card name), not a list",
                    ),
                    e => lever(v, &e),
                })?;
                workstation::resolve(&picks, census).map_err(|e| lever(v, &e))?[0]
            }
            None if self.n > 1 => cards[1],
            None => {
                let last = census
                    .len()
                    .checked_sub(1)
                    .and_then(|k| u8::try_from(k).ok())
                    .ok_or_else(|| {
                        format!(
                            "the DSpark draft's card: {} visible devices (visible: {})",
                            census.len(),
                            workstation::visible(census)
                        )
                    })?;
                workstation::resolve(&[Pick::Rank(last)], census)
                    .map_err(|e| format!("the DSpark draft's card: {e}"))?[0]
            }
        };
        Ok(match cards.iter().position(|c| c.same_device(&spec)) {
            Some(i) => DraftCard {
                at: Device::Card(i),
                spec: cards[i],
                holder: false,
            },
            None => DraftCard {
                at: Device::Card(cards.len()),
                spec: card_budget::keep_free(margin, spec)
                    .map_err(|e| format!("the DSpark draft's card: {e}"))?,
                holder: true,
            },
        })
    }

    /// Refuse a draft that opens on `opens` when its plan reserved the bytes on
    /// another card, or on a card past the plan's device list: the device the
    /// load opens is the device the reserve left, by name and before any
    /// allocation.
    pub fn check_draft_card(self, seat: DraftSeat, opens: &DraftCard) -> Result<(), GateError> {
        let reserve = seat.reserve;
        let cards = usize::from(self.n) + usize::from(seat.holder.is_some());
        match reserve.on {
            Device::Card(i) if i < cards => {}
            on => {
                return Err(format!(
                    "--place {}: the draft's reserve is on {on:?} and the plan has {cards} cards: \
                     the draft's card is outside the plan",
                    self.name()
                )
                .into());
            }
        }
        if reserve.on != opens.at || seat.holder.is_some() != opens.holder {
            return Err(format!(
                "--place {}: the DSpark draft opens on {:?} and its plan reserved its {} B on \
                 {:?}: the load's device is the reserve's",
                self.name(),
                opens.at,
                reserve.bytes,
                reserve.on
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod draft_card_tests {
    use model::placement::workstation::{A6000, DeviceInfo, RTX_3090};
    use model::placement::{Device, Reserve};

    use super::tests::census;
    use super::{DraftCard, DraftSeat, Place};

    /// `BLOOMERY_DSPARK_CARD`'s card on a census ([`Place::draft_on`]), by its
    /// place in the plan's device list: unset, the tier under `bp`, the stage
    /// under `gate` and on one card, and under `a` beside a second card that
    /// card as the plan's one more, reserve-only member; set, an ordinal or a
    /// name one device carries, a card of the placement or any other visible
    /// device as that member.
    #[test]
    fn the_draft_card_is_a_card_of_the_plan() {
        let at = |p: Place, set: Option<&str>, c: &[DeviceInfo]| {
            p.draft_on(set, c)
                .map(|d| {
                    (
                        d.at,
                        d.spec.name,
                        d.spec.device.map(|d| d.ordinal),
                        d.holder,
                    )
                })
                .map_err(|e| e.to_string())
        };
        for order in [["3090", "A6000"], ["A6000", "3090"]] {
            let c = census(&order);
            let o = |n: &str| {
                Some(u32::try_from(order.iter().position(|m| *m == n).expect(n)).expect("small"))
            };
            let tier = (Device::Card(1), RTX_3090.name, o("3090"), false);
            let stage = (Device::Card(0), A6000.name, o("A6000"), false);
            let held = (Device::Card(1), RTX_3090.name, o("3090"), true);
            assert_eq!(at(Place::Bp, None, &c), Ok(tier));
            assert_eq!(
                at(Place::A, None, &c),
                Ok(held),
                "the 3090 beside plan (a) joins the plan, the A6000 keeps every expert"
            );
            assert_eq!(
                at(Place::Gate, None, &c),
                Ok((Device::Card(0), RTX_3090.name, o("3090"), false))
            );
            assert_eq!(at(Place::A, Some("A6000"), &c), Ok(stage));
            let ordinal = format!("cuda{}", o("A6000").expect("A6000"));
            assert_eq!(at(Place::A, Some(&ordinal), &c), Ok(stage));
            let tier_word = o("3090").expect("3090").to_string();
            assert_eq!(at(Place::Bp, Some(&tier_word), &c), Ok(tier));
            assert_eq!(
                at(Place::Bp.on(&c).expect("bp"), Some("3090"), &c),
                Ok(tier)
            );
            assert_eq!(
                at(Place::Bp, Some("A6000"), &c),
                Ok(stage),
                "the stage is a card of bp"
            );
            assert_eq!(at(Place::A, Some("3090"), &c), Ok(held));
            assert_eq!(
                at(Place::Gate, Some("a6000"), &c),
                Ok((Device::Card(1), A6000.name, o("A6000"), true))
            );
        }
        let one = census(&["3090"]);
        assert_eq!(
            at(Place::Gate, None, &one),
            Ok((Device::Card(0), RTX_3090.name, Some(0), false))
        );
        let one = census(&["A6000"]);
        assert_eq!(
            at(Place::A, None, &one),
            Ok((Device::Card(0), A6000.name, Some(0), false)),
            "one card: the stage is the only room"
        );
        let e = at(Place::A, Some("3090"), &one).expect_err("no 3090");
        assert!(
            e.starts_with("BLOOMERY_DSPARK_CARD=\"3090\": no visible device is 3090"),
            "{e}"
        );
        let two = census(&["3090", "3090"]);
        let stage_zero = Place::parse("cuda0").expect("cuda0");
        assert_eq!(
            at(stage_zero, None, &two),
            Ok((Device::Card(1), RTX_3090.name, Some(1), true))
        );
        assert_eq!(
            at(stage_zero, Some("0"), &two),
            Ok((Device::Card(0), RTX_3090.name, Some(0), false))
        );
        assert_eq!(
            at(stage_zero, Some("cuda1"), &two),
            Ok((Device::Card(1), RTX_3090.name, Some(1), true))
        );
        let e = at(stage_zero, Some("3090"), &two).expect_err("two of one name");
        assert!(e.contains("2 visible devices carry the name 3090"), "{e}");
        for (v, want) in [
            ("a6000+3090", "the lever names one card"),
            ("4090", "names the card \"4090\""),
            ("cuda7", "no visible device is cuda7"),
        ] {
            let e = at(Place::A, Some(v), &census(&["A6000", "3090"])).expect_err(v);
            assert!(e.contains(want), "{v}: {e}");
        }
        // PIN(2026-10-24): the draft's card is asked of the resolved placement, so an empty census is
        // the placement's refusal, not a lack of the A6000 by name.
        let e = at(Place::A, None, &[]).expect_err("no device");
        assert!(
            e.starts_with(
                "--place a: the placement takes the largest visible card, and 0 devices are visible"
            ),
            "{e}"
        );
    }

    /// The card that joins a plan to hold the draft keeps the margin the lever gives it: the one
    /// value, the list's entry for its ordinal, its own where the list names it none; the
    /// placement's own cards are not this card.
    #[test]
    fn a_draft_holder_keeps_the_margin_the_lever_gives_it() {
        use model::placement::card_budget::CardBudget::{Every, Per, Unset};
        use model::placement::workstation::MARGIN;

        let c = census(&["A6000", "3090"]);
        let placed = Place::A.on_with(&c, &Unset, &Unset, &[]).expect("a");
        let holder = |margin| {
            let d = placed.draft_card(None, &c, &margin).expect("a card");
            assert!(d.holder, "the 3090 beside plan (a) joins the plan");
            d.spec.margin()
        };
        assert_eq!(holder(Unset), MARGIN);
        assert_eq!(holder(Every(2 << 30)), 2 << 30);
        assert_eq!(holder(Per(vec![(1, 3 << 30)])), 3 << 30);
        assert_eq!(holder(Per(vec![(0, 3 << 30)])), MARGIN);
    }

    /// A draft opens on the card its plan reserved on: the same place in the
    /// plan's device list is accepted, another card or a card past the list is
    /// refused by name before any allocation.
    #[test]
    fn a_draft_opens_on_the_card_its_reserve_is_on() {
        let c = census(&["A6000", "3090"]);
        let draft = |p: Place, set| p.draft_on(set, &c).expect("a card");
        let seat = |on: usize, holder: bool| DraftSeat {
            reserve: Reserve {
                on: Device::Card(on),
                name: "DSpark draft",
                bytes: 8,
            },
            holder: holder.then(|| draft(Place::A, None).spec),
        };
        let tier: DraftCard = draft(Place::Bp, None);
        let stage: DraftCard = draft(Place::Bp, Some("A6000"));
        let held: DraftCard = draft(Place::A, None);
        assert!(Place::Bp.check_draft_card(seat(1, false), &tier).is_ok());
        assert!(Place::Bp.check_draft_card(seat(0, false), &stage).is_ok());
        assert!(Place::A.check_draft_card(seat(1, true), &held).is_ok());
        assert_eq!(
            tier.seat(8),
            seat(1, false),
            "the seat of a tier card holds no spec"
        );
        assert_eq!(held.seat(8), seat(1, true));
        assert_eq!(
            Place::Bp
                .check_draft_card(seat(1, false), &stage)
                .expect_err("opens on the stage, reserved on the tier")
                .to_string(),
            "--place bp: the DSpark draft opens on Card(0) and its plan reserved its 8 B on \
             Card(1): the load's device is the reserve's"
        );
        assert!(
            Place::A
                .check_draft_card(seat(1, false), &held)
                .expect_err("reserved on a card the plan does not have")
                .to_string()
                .contains("outside the plan"),
        );
        assert!(
            Place::A
                .check_draft_card(seat(0, false), &held)
                .expect_err("opens on the member, reserved on the stage")
                .to_string()
                .contains("the load's device is the reserve's"),
        );
    }
}
