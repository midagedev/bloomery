//! A qwen4exp file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `glm5next::place` is for
//! GLM-5.3-Flash. The hyperparameters, every tensor's role, the typed
//! description and each layer's recurrent and cache bytes ([`KvLayout`]) are
//! read once ([`PlanInputs::read`]); a file with a feature the engine does not
//! run is refused there by the coverage check; then the placement puts every
//! tensor but the routed stacks where its role says — the trunk, the
//! hyper-connections, the PLE site's projections, the router, the shared
//! expert and the head on the card, the PLE table where the host's room says
//! ([`placement::row_table_tier`]: on the host in its file bytes when the
//! room holds it, else on the NVMe tier with its row-read room charged to
//! the host), its rows gathered by the host — and the routed
//! experts where [`Experts`] says: every one on the host
//! ([`PlanInputs::plan`], the plan the program loads), or by the expert rule
//! ([`PlanInputs::plan_with`] under [`Experts::Card`]): each eligible layer's
//! id prefix `[0, n_l)`, spread evenly, on the layers whose three routed stacks the card experts read
//! in their parts ([`routed_rule`] of [`PROGRAM`]: a Q4_K, Q5_K, IQ3_XXS or
//! IQ4_XS gate and up of one type, a Q5_1, Q8_0 or IQ4_NL down, each in the
//! file's blocks), the rest on the host. The same routed rule names every
//! stack whose kernel contract keeps its layer on the host
//! ([`placement::RoutedRule::host_only`]).
//!
//! The attention layers select their positions by the mean-pool indexer at
//! every position the program runs, so no count of positions is refused for
//! the attention's sake; the one bound is the kernels' position argument
//! ([`KERNEL_POSITIONS`]). A card that cannot hold the cache at a context is
//! the plan's own violation ([`Violation::CardOver`]).
//!
//! A plan with an MTP draft ([`PlanInputs::plan_mtp`]) is that plan, bit for
//! bit, beside the draft layer's own plan ([`MtpInputs`]): its file's tensors
//! as the card holds them ([`mtp_tensors`]), its routed experts every one on
//! the card and its dense store, counted in the draft's own heap, and the
//! head's row map; the target card's bound is checked on the sum. The draft
//! borrows the target's `token_embd` and `output`, which the target's plan
//! already holds on the card, so they add nothing: a reduced head reads its
//! rows of `output` in place, and its map is planned at one word a
//! vocabulary id whatever the head, so a plan with a row list is the plan
//! with the full head, byte for byte. Under
//! [`Experts::Card`] ([`PlanInputs::plan_mtp_with`]) the target's expert rule
//! spreads within its card's budget less the draft's card bytes, so the sum
//! fits.

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::{GgmlType, Split};
use models::headpick::{HeadRowsError, HeadSet, vocab_sha256};
use models::{
    DeltaKind, DeltaRule, DraftSpec, EngramSpec, Extra, Ffn, HcKind, Mixer, ModelSpec, MtpDraft,
    NgramRule, OutGate, Role, Selector,
};

use super::hparams::Kind;
use super::{mtp, names};
use crate::arch::names::routed_stacks;
use crate::arch::{architecture_is, coverage, place_hybrid};
use crate::placement::accepts::{
    Program, leg_card_format, leg_down_forms, program_of, qwen38_down, qwen38_gate_up, routed_rule,
};
use crate::placement::ctx::{CtxLimit, PastLimit, check_slot_ctx};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::workstation::{
    self, A6000, CardSpec, GRANULE, HostRead, RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE,
    TierBatchBytes, tier_batch_host_bytes, tier_batch_staging_bytes,
};
use crate::placement::{
    self, Card, CardFormat, Device, ExpertList, Format, Host, KvBytes, Machine, ModelTensor,
    ModelTensors, Node, PlacementError, Plan, PlanLevers, RoutedRule, Unimplemented, Violation,
    checked, joined, kernels,
};
use gguf::digest::hex;

use runtime::stores::{
    DELTA_VERIFY_ROWS, F32_BYTES, delta_carry_bytes, delta_lane_bytes, delta_record_bytes,
    index_ring_bytes, kv_plane_bytes, ple_ring_bytes, recurrent_bytes, selecting_rows_bytes,
};
pub use runtime::stores::{PASS_ROWS, conv_ring_rows, ple_ring_rows};

/// The positions the selector's kernels take: the QSA passes take the
/// context as a `u32` launch argument and the selected flash reads `u32`
/// cache rows, so a context past this is one those launches cannot name.
pub const KERNEL_POSITIONS: u64 = u32::MAX as u64;

/// The most positions a load of the file `spec` describes serves: its
/// `ctx_train`, the context it was trained at (the rope runs unscaled;
/// YaRN scaling is not built), within [`KERNEL_POSITIONS`]. `ctx` past it is
/// refused by name ([`check_slot_ctx`]); else the cap. `generate_qwen3moe`
/// and the Qwen3.8 seat take the rule here.
pub fn serve_ctx(ctx: u64, spec: &ModelSpec) -> Result<u64, PlaceError> {
    let bounds = CtxLimit {
        trained: u64::from(spec.ctx_train),
        kernel: Some((KERNEL_POSITIONS, "KERNEL_POSITIONS")),
        verified: None,
    };
    check_slot_ctx(ctx, &bounds).map_err(|p: PastLimit| PlaceError::PastTrained {
        ctx: p.ctx,
        trained: p.trained,
        cap: p.limit,
    })
}

/// Where a plan puts the file's routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Experts {
    /// Every routed expert on the host: the plan [`PlanInputs::plan`] makes.
    Host,
    /// The expert rule on the layers whose routed stacks [`routed_rule`] of
    /// [`PROGRAM`] loads: each such layer's id prefix `[0, n_l)` on the card, as many as
    /// the card's budget holds, spread evenly over them; the rest on the host.
    Card,
}

/// The Qwen3.8 program, whose card experts read the routed stacks
/// ([`routed_rule`] of it) by part, each in its card format
/// ([`leg_card_format`]: NVFP4 in its planes, every other type as the
/// file stores it): a gate and an up of one type, a Q4_K, Q5_K, IQ3_XXS,
/// IQ4_XS or NVFP4 ([`qwen38_gate_up`]), and a Q5_1, Q8_0, IQ4_NL or NVFP4
/// down ([`qwen38_down`]). A layer with a stack of another type, or of one of
/// these in a part the card leg does not run it in (a Q8_0 gate, a Q4_K
/// down), keeps its experts on the host.
pub const PROGRAM: Program = Program::Qwen38Body;

/// A type [`PROGRAM`]'s card leg reads in some part, in its card format.
/// Coverage reads this type union; the plan and its host-only diagnosis use
/// [`routed_rule`] by part.
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    (qwen38_gate_up(ty).is_some() || qwen38_down(ty).is_some()).then(|| leg_card_format(ty))
}

/// What a target plan sets aside beside its own terms: `reserve` bytes the
/// expert rule spreads within on each card (kept beside the plan under the
/// all-host rule too, so a unified pool counts it), and the draft `mtp` the
/// load carries, whose saved states the prompt cache's floor counts
/// ([`PlanInputs::seq_state_bytes`]).
#[derive(Clone, Copy)]
struct Beside<'a> {
    reserve: u64,
    mtp: Option<&'a MtpInputs>,
    /// The resident sequences the load steps as one pass; the plan's NVMe
    /// tier arena is floored for their columns
    /// ([`placement::HostTotals::step_slots`]).
    slots: usize,
}

impl<'a> Beside<'a> {
    /// A plan with no draft and no reserve.
    const fn none() -> Self {
        Beside::reserve(0)
    }

    /// A plan with no draft, `reserve` bytes set aside.
    const fn reserve(reserve: u64) -> Self {
        Beside {
            reserve,
            mtp: None,
            slots: 1,
        }
    }

    /// A plan of a load that carries `mtp`, `reserve` bytes set aside.
    const fn draft(reserve: u64, mtp: &'a MtpInputs) -> Self {
        Beside {
            reserve,
            mtp: Some(mtp),
            slots: 1,
        }
    }

    /// The same plan for a load that serves `slots` resident sequences.
    const fn slots(self, slots: usize) -> Self {
        Beside { slots, ..self }
    }
}

/// What a plan of a qwen4exp file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description: the one owner of the file's shape. The
    /// coverage check, the layout of each layer's bytes and every reader of
    /// a width, a count or a context read it.
    pub spec: ModelSpec,
    /// Each layer's recurrent and cache bytes.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the PLE
    /// split table's tier follows it ([`placement::row_table_tier`]): on the host
    /// while the room holds the plan with the table there, else on the NVMe
    /// tier; the routed experts the room still cannot hold follow it there
    /// ([`placement::expert_nvme_tier`]). A gate that plans an arm by a room
    /// of its own sets it, read [`HostRead::Given`]. Part-source tables stay
    /// on NVMe regardless of this room ([`PlanInputs::row_source`]).
    ///
    /// The effective room is [`Plan::plan_at`]'s: the reading's plan room,
    /// capped by the machine's host room and unified pool.
    pub room: (u64, HostRead),
    /// A part source's rows stay on NVMe, with room for each stored plane:
    /// the PLE table of a split that does not hold it as one file run (a
    /// header alone, or a checkpoint folder whose part source makes it),
    /// which the host cannot read where the split stores it.
    pub row_source: placement::RowTableSource,
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it
    /// as its `given` ([`placement::expert_nvme_tier`]). A gate that plans an
    /// arm by an arena of its own sets it.
    pub arena: Option<u64>,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// A load whose ubatch is outside its context or supported width.
    #[error("ubatch {ubatch}: expected 1..=min({ctx_max}, {UBATCH_PLANNED})")]
    Ubatch { ctx_max: u64, ubatch: u64 },
    /// A planning callback returned without handing on its plan.
    #[error("widen_card_route: the planning callback returned no plan")]
    MissingPlan,
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position, or of more than the kernels take.
    #[error(
        "ctx_max {ctx_max}: a context of 1 to {KERNEL_POSITIONS} positions (the QSA passes \
         take the context as a u32 launch argument, the selected flash reads u32 cache rows)"
    )]
    Positions { ctx_max: u64 },
    /// A context past the one the file was trained at ([`serve_ctx`]).
    #[error(
        "a context of {ctx} positions: the file was trained at {trained} (context_length), and a \
         load serves at most {cap}; YaRN scaling past the trained context is not implemented"
    )]
    PastTrained { ctx: u64, trained: u64, cap: u64 },
    /// A draft planned beside a machine of other than one card.
    #[error("the MTP draft is planned on the target's one card; the machine has {cards} cards")]
    DraftCards { cards: usize },
    /// The draft's plan put some of its routed experts off the card.
    #[error(
        "the MTP layer's card holds {held} of its {experts} routed experts; its program runs \
         every one on the card"
    )]
    DraftExperts { held: u64, experts: u64 },
    /// The reduced head's row list, refused.
    #[error(transparent)]
    HeadRows(#[from] HeadRowsError),
    /// A machine with an expert tier card, planned with every routed expert
    /// on the host: the tier would hold none. The wrapped refusal is the
    /// planner's own; this text carries what it lacks — the lever that made
    /// the all-host plan (`BLOOMERY_QWEN38_EXPERTS=host`) and the plan that
    /// runs the card experts (plan (b′)).
    #[error(
        "the machine has the expert tier card {tier}, and the plan puts every routed expert on \
         the host (BLOOMERY_QWEN38_EXPERTS=host): the tier would hold none; plan (b′) runs the \
         card experts"
    )]
    TierHost {
        tier: String,
        /// The planner's own refusal of the tier card.
        origin: Box<PlacementError>,
    },
    /// A machine with an expert tier card. [`PlanInputs::plan_rule`] does not
    /// return it: a tier machine is planned. Callers that match every refusal
    /// still name it.
    #[error(
        "the machine has the expert tier card {tier}; the program this plan is for hangs no \
         expert tier"
    )]
    Tier { tier: String },
    /// The stage card's MTP draft reserve ([`MTP_RESERVE`]) is not the one
    /// the plan needs: none for a plan without the draft or on a machine with
    /// no tier card, the draft's card bytes for a draft beside a tier.
    #[error(
        "card {card} reserves {got:?} B as \"{MTP_RESERVE}\"; this plan needs {}",
        want.map_or("no such row".to_string(), |b| format!("one row of {b} B"))
    )]
    DraftReserve {
        card: String,
        got: Vec<u64>,
        want: Option<u64>,
    },
    /// The stage card's reserve for the card route's run of the whole
    /// ubatch ([`ROUTE_UNIT_RESERVE`]) is not the one a load of ubatches of
    /// up to `u` positions reads ([`card_route_run`]).
    #[error(
        "card {card} reserves {got:?} B as \"{ROUTE_UNIT_RESERVE}\"; a load of ubatches of up to \
         {u} positions reads no such row or one of {want} B"
    )]
    RouteUnitReserve {
        card: String,
        got: Vec<u64>,
        u: u64,
        want: u64,
    },
    /// The stage card's reserve for the activations' MMA forms
    /// ([`ACT_FORMS_RESERVE`]) is not the one a load of ubatches of up to `u`
    /// positions reads ([`act_forms_bytes`]).
    #[error(
        "card {card} reserves {got:?} B as \"{ACT_FORMS_RESERVE}\"; a load of ubatches of up to \
         {u} positions reads no such row or one of at least {want} B"
    )]
    ActFormsReserve {
        card: String,
        got: Vec<u64>,
        u: u64,
        want: u64,
    },
    /// A plan of no resident sequence slot.
    #[error("{slots} resident sequence slots: a plan serves at least one")]
    Slots { slots: usize },
    /// A target matrix an MTP draft file borrows ([`PlanInputs::mtp_borrows`])
    /// absent, or in another format than the one the draft's program reads.
    #[error(
        "the target's {name} is {}; the MTP draft reads the target's token_embd as Q8_0 or BF16 \
         and its output as Q8_0, Q6_K or BF16",
        ty.map_or("absent".to_string(), |t| t.to_string())
    )]
    DraftBorrow { name: String, ty: Option<GgmlType> },
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

/// The planner's refusal of a tier card on the all-host plan
/// (`PlacementError::HostRoutedTier`), wrapped as the program's own
/// ([`PlaceError::TierHost`]); every other refusal passes through.
fn tier_host(e: PlacementError) -> PlaceError {
    let PlacementError::HostRoutedTier { card, .. } = &e else {
        return PlaceError::Placement(e);
    };
    PlaceError::TierHost {
        tier: card.clone(),
        origin: Box::new(e),
    }
}

/// A description run by the Qwen3.5 body or the Qwen3.8 body, which these
/// inputs plan; any other program, or none, is refused naming it.
fn refuse_other_program(spec: &ModelSpec) -> Result<(), PlacementError> {
    let program = program_of(spec.arch);
    if matches!(program, Some(Program::Qwen35Body | Program::Qwen38Body)) {
        return Ok(());
    }
    Err(architecture_is(&format!(
        "{}, run by {}; this plan is for the programs {} and {}",
        spec.arch.name(),
        program.map_or("no program", Program::name),
        Program::Qwen35Body.name(),
        Program::Qwen38Body.name()
    )))
}

impl PlanInputs {
    /// Bytes beyond the positional rows that room-backed stores can hold:
    /// every buffer rounded independently on every resident slot, and the
    /// device's one shared spare page. `page` is read when the stage card opens.
    pub fn room_padding(
        &self,
        ctx: u64,
        slots: u64,
        drafts: bool,
        page: u64,
    ) -> Result<u64, PlacementError> {
        assert!(page > 0, "the device allocation granularity is positive");
        let padding = |n: u64| n.div_ceil(page).checked_mul(page)?.checked_sub(n);
        let planes = |g: &models::Gqa| -> Option<u64> {
            let values = u64::from(g.kv_heads)
                .checked_mul(ctx)?
                .checked_mul(u64::from(g.head_dim))?;
            let one = if self.kv.is_q8() {
                padding(values)?.checked_add(padding(values / 16)?)?
            } else {
                padding(values.checked_mul(2)?)?
            };
            one.checked_mul(2)
        };
        let count = || -> Option<u64> {
            let mut extra = 0u64;
            for layer in &self.spec.layers {
                if let Mixer::Gqa(g) = &layer.mixer {
                    extra = extra.checked_add(planes(g)?)?;
                    if let Some(Selector::TokenPool { d, pool, .. }) = &g.select {
                        let bytes = ctx
                            .div_ceil(u64::from(*pool))
                            .checked_mul(u64::from(*d))?
                            .checked_mul(2)?;
                        extra = extra.checked_add(padding(bytes)?)?;
                    }
                }
            }
            if drafts {
                extra = extra.checked_add(planes(self.spec.gqa()?)?)?;
            }
            extra.checked_mul(slots)?.checked_add(page)
        };
        count().ok_or_else(|| PlacementError::CardBytesPastU64 {
            card: "room-backed stores".to_owned(),
        })
    }

    /// `machine`'s stage card with the page rounding of room-backed stores
    /// ([`PlanInputs::room_padding`]) reserved: what every plan of a load
    /// that backs its stores by room counts before admission.
    pub fn reserve_room(
        &self,
        machine: &mut Machine,
        (ctx, slots, drafts): (u64, u64, bool),
        page: u64,
    ) -> Result<(), PlacementError> {
        let padding = self.room_padding(ctx, slots, drafts, page)?;
        machine.cards[0]
            .reserves
            .push(("store page rounding".to_owned(), padding));
        Ok(())
    }

    /// `split`'s description and its tensors' roles, then the host's room
    /// ([`PlanInputs::describe`]), the first that fails being the error; then
    /// the file is refused if it has a feature the engine does not run
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

    /// The target's matrices an MTP draft file borrows, `token_embd` and
    /// `output`, each in a format the draft's program reads: `token_embd`
    /// Q8_0 or BF16 (the embedding rows read either), `output` one of
    /// [`mtp::head_kind`]'s forms (Q8_0, Q6_K or BF16, one owner with it in
    /// [`mtp::output_form`]); else the first that is absent or another
    /// format, by name ([`PlaceError::DraftBorrow`]). [`MtpInputs::read`]
    /// refuses a draft by it; the Qwen3.8 seat's unset `BLOOMERY_DRAFT`
    /// drafts nothing by it.
    pub fn mtp_borrows(&self) -> Result<(), PlaceError> {
        for (name, borrowed) in [(names::token_embd(), false), (names::output(), true)] {
            let ty = self
                .model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.ty);
            let ok = if borrowed {
                mtp::output_form(ty.unwrap_or(GgmlType::F32)).is_some()
            } else {
                matches!(ty, Some(GgmlType::Q8_0 | GgmlType::BF16))
            };
            if !ok {
                return Err(PlaceError::DraftBorrow { name, ty });
            }
        }
        Ok(())
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features:
    /// the file's description and this host's room
    /// ([`workstation::host_available_read`]), a room that cannot be read
    /// refused by name — the PLE table's tier would follow a room nobody
    /// read. The room is read after the file's description, through the
    /// same steps as [`PlanInputs::describe_in`].
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

    /// The file's description and roles (`arch::spec_placed`, the one read of
    /// its headers) and the KV layout they give, then the room `room` gives,
    /// then the arena, the first that fails being the error. A file whose
    /// description runs on neither the Qwen3.5 nor the Qwen3.8 body is
    /// refused by name.
    fn describe_with(
        split: &Split,
        room: impl FnOnce() -> Result<(u64, HostRead), PlacementError>,
    ) -> Result<PlanInputs, PlacementError> {
        let read = crate::arch::spec_placed(split)?;
        let (spec, model) = (read.spec, read.tensors);
        refuse_other_program(&spec)?;
        let kv = KvLayout::of_spec(&spec)?;
        let room = room()?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            model,
            spec,
            kv,
            room,
            row_source: match split.place(&names::per_layer_token_embd()) {
                Some(gguf::Place::File { .. }) | None => placement::RowTableSource::Split,
                Some(_) => placement::RowTableSource::Parts,
            },
            arena,
        })
    }

    /// The inputs of a load whose GDN layers keep `lanes` stamped lanes of
    /// state ([`KvLayout::of_spec_lanes`]): a Qwen3.6 load that verifies drafted
    /// rows asks for `MAX_PASS_ROWS` of them, and the plan, the whole-fit
    /// verdict and the sequences' stores all price the one count. The layout's
    /// q8_0 form ([`KvLayout::in_q8`]) is kept.
    ///
    /// # Errors
    ///
    /// The layout's refusal of the description ([`KvLayout::of_spec_lanes`]).
    ///
    /// # Panics
    ///
    /// On no lane, by name.
    pub fn with_lanes(mut self, lanes: usize) -> Result<PlanInputs, PlacementError> {
        let q8 = self.kv.q8;
        self.kv = KvLayout::of_spec_lanes(&self.spec, lanes)?;
        self.kv.q8 = q8;
        Ok(self)
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The file's layers and the card bytes a position of the ubatch walk
    /// holds in the quantized forms its wide dense sites read: for each width
    /// the walk's rows hold, the forms the layers' dense matrices of that row
    /// width read ([`kernels::wide_act_bytes`]), the one rule the load
    /// allocates them by. A machine counts them beside the walk's other rows
    /// ([`machine_for_route`]).
    #[must_use]
    pub fn stack(&self) -> Stack {
        let dense = |t: &&ModelTensor| {
            t.layer.is_some_and(|l| l < self.model.layers)
                && t.dims.len() == 2
                && matches!(
                    t.role,
                    Role::Attention | Role::SharedExpert | Role::DenseFfn | Role::EngramDense
                )
        };
        let mut widths: Vec<u64> = self
            .model
            .tensors
            .iter()
            .filter(dense)
            .map(|t| t.dims[0])
            .collect();
        widths.sort_unstable();
        widths.dedup();
        let act_token_bytes = widths
            .into_iter()
            .map(|k| {
                let tys = self
                    .model
                    .tensors
                    .iter()
                    .filter(dense)
                    .filter(|t| t.dims[0] == k)
                    .map(|t| t.ty);
                kernels::wide_act_bytes(tys, 1, k)
            })
            .sum();
        Stack {
            layers: self.spec.layers.len(),
            act_token_bytes,
        }
    }

    /// The form a load of the file holds its card route's runs in, on a
    /// build that has the native FP4 products or not (`native_fp4`):
    /// [`RouteForm::Native`] where the card reads every routed stack of the
    /// file's layers in NVFP4's planes ([`card_routed`]: a GGUF's NVFP4 or a
    /// checkpoint folder's NVFP4 rows), which the card's experts of any
    /// layers then read in one native launch; [`RouteForm::Rows`] otherwise.
    #[must_use]
    pub fn route_form(&self, native_fp4: bool) -> RouteForm {
        let nvfp4 = self
            .model
            .tensors
            .iter()
            .filter(|t| {
                t.role == Role::RoutedExperts && t.layer.is_some_and(|l| l < self.model.layers)
            })
            .all(|t| card_routed(t.ty) == Some(CardFormat::Nvfp4Planes));
        if native_fp4 && nvfp4 {
            RouteForm::Native
        } else {
            RouteForm::Rows
        }
    }

    /// One slot of the expert stream a load of the file under `experts` runs: the widest layer's
    /// routed expert ([`place_hybrid::stream_slot_bytes`]); 0 where the load streams nothing:
    /// under [`Experts::Host`], whose plan holds no card expert, so its residency is `off` and no
    /// stream starts, and where a routed stack reaches the card route as NVFP4 planes, which the
    /// ring pass does not read. A load that streams reserves the lane's staging of it before its
    /// plan (`workstation::reserve_stream_staging`).
    pub fn stream_slot(&self, experts: Experts) -> Result<u64, PlacementError> {
        let planes = self.model.tensors.iter().any(|t| {
            t.role == Role::RoutedExperts && card_routed(t.ty) == Some(CardFormat::Nvfp4Planes)
        });
        if experts == Experts::Host || planes {
            return Ok(0);
        }
        place_hybrid::stream_slot_bytes(&self.model, routed_stacks, self.model.experts)
    }

    /// The placement of the file on `machine` at `ctx_max` positions with
    /// prompt calls of up to `ubatch` positions under the placement's `levers`, every routed expert on the host and the PLE
    /// table where [`PlanInputs::room`] says ([`PlanInputs::plan_with`] under
    /// [`Experts::Host`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with(machine, (ctx_max, ubatch), levers, Experts::Host)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, the routed experts where `experts` says and
    /// the PLE table where [`PlanInputs::room`] says
    /// ([`placement::row_table_tier`]); refused past [`KERNEL_POSITIONS`],
    /// when it cannot be built, or when it breaks an invariant; a machine
    /// whose stage card reserves bytes for an MTP draft ([`MTP_RESERVE`]) is
    /// refused by name, as is a tier machine under [`Experts::Host`]. A card
    /// plan whose budget leaves no expert on the card is the [`Experts::Host`]
    /// plan of the same levers.
    pub fn plan_with<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        experts: Experts,
    ) -> Result<Plan<'a>, PlaceError> {
        check_shape(ctx_max, ubatch)?;
        check_draft_reserve(machine, None)?;
        let plan = self.target(machine, (ctx_max, ubatch), levers, experts, Beside::none())?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// A node assignment lowered into the loaded map. Counts are global id
    /// ranges in node order; node zero's card keeps only its assigned prefix.
    /// Each tier opens its file-backed range against its own host room.
    pub fn plan_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        owned: &[u64],
    ) -> Result<Plan<'a>, PlaceError> {
        check_shape(ctx_max, ubatch)?;
        check_draft_reserve(machine, None)?;
        let plan = placement::plan_in_pool(machine, levers, |levers| {
            let mut plan = placement::nodes::loaded_routed(
                &self.model,
                machine,
                ctx_max,
                &self.kv,
                levers,
                routed_rule(PROGRAM),
                0,
                Some(owned),
            )?;
            let room = plan.plan_at(&self.room);
            let floor = 2 * self.seq_state_bytes(ctx_max, None);
            let page = os::page_bytes().map_err(|e| PlacementError::Host(e.to_string()))?;
            self.place_rows(&mut plan, room, floor, ubatch, page)?;
            plan.host.step_slots(1, placement::STEP_PORT_COLUMNS)?;
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            Ok(plan)
        })
        .map_err(tier_host)?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// [`PlanInputs::plan_with`] for a load that serves `slots` resident
    /// sequences ([`GpuModel::add_slots`](bloomery_gpu::model::Slots)): every
    /// per-load sequence term ([`PlanInputs::seq_terms`]) — the positional
    /// rows at the load's context and the fixed recurrent, conv and PLE terms
    /// ([`KvLayout::layer_bytes`]), and the bytes a sequence holds beside its
    /// stores ([`slot_resident_bytes`]) — counts them all, the bytes beside the
    /// stores reserved out of the expert budget before it fills so a card the
    /// plan saturates holds them too, so a card that cannot hold them is
    /// refused as [`PlanInputs::plan_with`] refuses it. The live sequence's
    /// bytes beside its stores are the load's own (its arenas' rows and its
    /// lane word), so the card's kv class is [`SeqTerms::plan_kv`]. One slot
    /// is [`PlanInputs::plan_with`] itself; zero is refused by name.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        experts: Experts,
        slots: usize,
    ) -> Result<Plan<'a>, PlaceError> {
        if slots == 1 {
            return self.plan_with(machine, (ctx_max, ubatch), levers, experts);
        }
        check_shape(ctx_max, ubatch)?;
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        check_draft_reserve(machine, None)?;
        let terms = self.seq_terms(None);
        let rows = terms.plan_beside(slots as u64);
        let kv = terms.slots_of(slots as u64);
        let mut plan = self.target_of(
            machine,
            (ctx_max, ubatch),
            levers,
            experts,
            Beside::reserve(rows).slots(slots),
            &kv,
        )?;
        // The bytes a sequence holds beside its stores ride the card's kv
        // class, out of the budget `rows` reserved above, so
        // [`Plan::violations`] holds the card's bound against them.
        plan.cards[0].grow_kv(rows);
        checked(plan).map_err(PlaceError::Broken)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the file's layers ([`KvLayout`]), the
    /// MTP draft's store on a load that carries `mtp`, and the bytes it holds
    /// beside them ([`slot_resident_bytes`]) and its delta replay table.
    #[must_use]
    pub fn seq_terms<'a>(&'a self, mtp: Option<&'a MtpInputs>) -> SeqTerms<'a> {
        let replay = if program_of(self.spec.arch) == Some(PROGRAM) {
            self.kv
                .kinds
                .iter()
                .filter(|&&k| k == Kind::DeltaRule)
                .count() as u64
                * runtime::stores::DELTA_REPLAY_BYTES
        } else {
            0
        };
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.model.layers,
            },
            draft: mtp.map(MtpInputs::stores),
            beside: slot_resident_bytes(&self.spec) + replay,
        }
    }

    /// The most host bytes one saved sequence state of `positions` positions
    /// holds — the prompt cache's entry: every attention layer's positional
    /// bytes and a budget-bounded set of checkpoints ([`KvLayout::state_bytes`]) and, on
    /// a load that carries `mtp`, the draft's store below the same positions,
    /// the step's and the pass's arena rows and one hidden row per saved
    /// inner checkpoint. The engine's own owner is `Body38`'s
    /// `seq38_bytes`; the gate holds the two equal.
    #[must_use]
    pub fn seq_state_bytes(&self, positions: u64, mtp: Option<&MtpInputs>) -> u64 {
        let points = self.kv.checkpoint_price().saved_points(positions);
        let draft = mtp.map_or(0, |m| {
            let point_row = streams(&self.spec) * hidden(&self.spec) * std::mem::size_of::<f32>();
            m.kv.layer_bytes(0, positions) + slot_resident_bytes(&self.spec)
                - runtime::stores::U32_BYTES
                + points * point_row as u64
        });
        self.kv.state_bytes(positions) + draft
    }

    /// The target's plan, unchecked: every routed expert on the host, or the
    /// expert rule over [`routed_rule`]'s layers within each card's budget
    /// less `beside`'s reserve ([`placement::plan_routed_reserving`]) — on a machine
    /// with expert tier cards, the stage cards first and each tier the next
    /// ids after them (plan (b′), [`machine_bp`]); then the PLE table's tier
    /// by the host's room ([`placement::row_table_tier`]), beside the prompt
    /// cache's floor ([`PlanInputs::target_rule_of`]). A tier machine
    /// under [`Experts::Host`] is refused by name ([`PlaceError::TierHost`]):
    /// its tier would hold nothing.
    fn target<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        experts: Experts,
        beside: Beside<'_>,
    ) -> Result<Plan<'a>, PlaceError> {
        let rule = match experts {
            Experts::Host => None,
            Experts::Card => Some(routed_rule(PROGRAM)),
        };
        self.target_rule(machine, (ctx_max, ubatch), levers, rule, beside)
    }

    fn place_rows(
        &self,
        plan: &mut Plan<'_>,
        room: u64,
        floor: u64,
        ubatch: u64,
        page: u64,
    ) -> Result<(), PlacementError> {
        placement::row_table_tier(
            plan,
            room,
            floor,
            runtime::stores::reader_columns(ubatch),
            page,
            self.row_source,
        )
    }

    /// [`PlanInputs::target`] over `kv`'s per-layer bytes instead of the
    /// file's own — the multiplying view a multi-slot plan counts by.
    fn target_of<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        experts: Experts,
        beside: Beside<'_>,
        kv: &dyn KvBytes,
    ) -> Result<Plan<'a>, PlaceError> {
        let rule = match experts {
            Experts::Host => None,
            Experts::Card => Some(routed_rule(PROGRAM)),
        };
        self.target_rule_of(machine, (ctx_max, ubatch), levers, rule, beside, kv)
    }

    /// The target's plan, unchecked, under the card rule `rule` over `kv`'s
    /// per-layer bytes: every routed expert on the host for `None`, else the
    /// expert rule over the layers whose routed stacks `rule` loads within each
    /// card's budget less `reserve` (kept beside the plan under either, so a
    /// unified pool counts it, [`placement::plan_host_routed_reserving`]); then
    /// the PLE table's tier by the host's room ([`placement::row_table_tier`])
    /// beside the prompt cache's floor: two saved sequence states at
    /// `ctx_max` ([`PlanInputs::seq_state_bytes`], the draft's side under
    /// `beside`'s draft) — a function of the context being planned and the file's
    /// layers alone, so it needs no plan to be read first.
    /// On a unified machine the card rule's plan is planned in the pool
    /// ([`placement::plan_in_pool`]): a card share that leaves the host's room
    /// under the NVMe tier's floor is lowered until the floor holds. The
    /// all-host rule puts no routed expert on the card, so a lower card
    /// budget leaves its floor where it is, and it is planned once.
    /// The all-host rule's own refusal of a tier card
    /// ([`PlaceError::TierHost`]) is wrapped to carry the program's lever and
    /// plan.
    fn target_rule_of<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        rule: Option<RoutedRule>,
        beside: Beside<'_>,
        kv: &dyn KvBytes,
    ) -> Result<Plan<'a>, PlaceError> {
        let make = |levers: &PlanLevers| -> Result<Plan<'a>, PlacementError> {
            let mut plan = match rule {
                None => placement::plan_host_routed_reserving(
                    &self.model,
                    machine,
                    ctx_max,
                    kv,
                    levers,
                    beside.reserve,
                )?,
                Some(rule) => placement::plan_routed_reserving(
                    &self.model,
                    machine,
                    ctx_max,
                    kv,
                    levers,
                    rule,
                    beside.reserve,
                )?,
            };
            let floor = 2 * self.seq_state_bytes(ctx_max, beside.mtp);
            let room = plan.plan_at(&self.room);
            let page = os::page_bytes().map_err(|e| PlacementError::Host(e.to_string()))?;
            self.place_rows(&mut plan, room, floor, ubatch, page)?;
            plan.host.step_slots(
                if beside.mtp.is_some() {
                    beside.slots.saturating_mul(DELTA_VERIFY_ROWS)
                } else {
                    beside.slots
                },
                placement::STEP_PORT_COLUMNS,
            )?;
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            Ok(plan)
        };
        let plan = match rule {
            None => make(levers),
            Some(_) => placement::plan_in_pool(machine, levers, make),
        };
        plan.map_err(tier_host)
    }

    /// The target's plan, unchecked, under the card rule `rule`: every
    /// routed expert on the host for `None`, else the expert rule over the
    /// layers whose routed stacks `rule` loads within each card's budget
    /// less `reserve`; then the PLE table's tier by the host's room.
    fn target_rule<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        rule: Option<RoutedRule>,
        beside: Beside<'_>,
    ) -> Result<Plan<'a>, PlaceError> {
        self.target_rule_of(machine, (ctx_max, ubatch), levers, rule, beside, &self.kv)
    }
    /// `levers`, the routed experts by the expert rule over the layers whose
    /// three routed stacks `rule` loads — the card rule of the program that
    /// runs the plan ([`routed_rule`]: [`PROGRAM`]'s, or the qwen35moe body's
    /// for a qwen35moe file) — the rest on the host, a stage card's id prefix
    /// first and each expert tier's next ids after it. Refused as
    /// [`PlanInputs::plan_with`] refuses. A plan with no draft wants no
    /// [`MTP_RESERVE`] row ([`check_draft_reserve`]).
    pub fn plan_rule<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        rule: RoutedRule,
    ) -> Result<Plan<'a>, PlaceError> {
        check_shape(ctx_max, ubatch)?;
        check_draft_reserve(machine, None)?;
        let plan = self.target_rule(
            machine,
            (ctx_max, ubatch),
            levers,
            Some(rule),
            Beside::none(),
        )?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// [`PlanInputs::plan`] with the MTP draft `mtp` on the machine's one
    /// card: the target's plan as [`PlanInputs::plan`] makes it, the draft's
    /// on its own unbounded card ([`MtpInputs`]), and the card's bound
    /// checked on their sum — the target's granules, cache, scratch and
    /// context, the draft's granules and store, and the head's row map —
    /// against its usable bytes (capped by the card budget) less its margin.
    /// Refused as [`PlanInputs::plan`] refuses, on a machine of other than
    /// one card, and when the draft's plan keeps a routed expert off the
    /// card; every violation of either plan is listed, the card's bound once.
    pub fn plan_mtp<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        self.plan_mtp_with(machine, (ctx_max, ubatch), levers, mtp, Experts::Host)
    }

    /// [`PlanInputs::plan_mtp`] with the target's routed experts where
    /// `experts` says: under [`Experts::Card`] the target's expert rule
    /// spreads within its card's budget less the draft's card bytes
    /// ([`MtpPlan::draft_card_bytes`]), so the sum keeps the card's bound;
    /// the draft's plan is the same under either. On a machine with an
    /// expert tier card (plan (b′), [`machine_bp`]) the draft's card bytes
    /// are the stage card's named reserve [`MTP_RESERVE`] instead, which the
    /// tier's budget does not see: the stage card's row must be the draft's
    /// [`MtpInputs::card_bytes`], else the plan is refused by name; the
    /// stage card's own bound and headroom then hold the sum, and the tier
    /// plans as without the draft.
    pub fn plan_mtp_with<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
        experts: Experts,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        check_shape(ctx_max, ubatch)?;
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::DraftCards {
                cards: machine.cards.len(),
            });
        };
        let (draft, reserve) = mtp.draft_plan(ctx_max)?;
        let arena = mtp.arena_bytes();
        if !machine.tiers.is_empty() {
            check_draft_reserve(machine, Some(reserve))?;
            let plan = self.target(
                machine,
                (ctx_max, ubatch),
                levers,
                experts,
                Beside::draft(0, mtp),
            )?;
            let broken: Vec<Violation> = plan
                .violations()
                .into_iter()
                .chain(draft.violations())
                .collect();
            if !broken.is_empty() {
                return Err(PlaceError::Broken(broken));
            }
            return Ok(MtpPlan {
                headroom_bytes: plan.cards[0].headroom_bytes,
                plan,
                draft,
                map_bytes: mtp.map_bytes,
                arena_bytes: arena,
            });
        }
        check_draft_reserve(machine, None)?;
        let plan = self.target(
            machine,
            (ctx_max, ubatch),
            levers,
            experts,
            Beside::draft(reserve, mtp),
        )?;
        let t = &plan.cards[0];
        // The draft's term is the reserve the expert rule spread within, so
        // the bound and the budget count one number. The stage rope rows are
        // the target card's, beside the cache.
        let total =
            t.bytes()
                .checked_add(reserve)
                .ok_or_else(|| PlacementError::CardBytesPastU64 {
                    card: card.name.clone(),
                })?;
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        let mut broken: Vec<Violation> = plan
            .violations_beside(reserve)
            .into_iter()
            .filter(|v| !matches!(v, Violation::CardOver { .. }))
            .chain(draft.violations())
            .collect();
        if total > limit {
            broken.push(Violation::CardOver {
                card: card.name.clone(),
                total,
                limit,
            });
        }
        if !broken.is_empty() {
            return Err(PlaceError::Broken(broken));
        }
        Ok(MtpPlan {
            headroom_bytes: i128::from(usable) - i128::from(total),
            plan,
            draft,
            map_bytes: mtp.map_bytes,
            arena_bytes: arena,
        })
    }

    /// [`PlanInputs::plan_mtp_with`] for a load that serves `slots` resident
    /// sequences ([`PlanInputs::plan_with_slots`]): the target's and the
    /// draft's per-load sequence terms ([`PlanInputs::seq_terms`]) — their
    /// stores, and the bytes a sequence holds beside them
    /// ([`slot_resident_bytes`]) — count them all, the bytes beside the
    /// stores reserved out of the expert budget before it fills, the sum
    /// keeping the card's bound. The target's kv class and the draft's sum to
    /// [`SeqTerms::plan_kv`]. One slot is [`PlanInputs::plan_mtp_with`]
    /// itself; zero is refused by name.
    pub fn plan_mtp_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
        experts: Experts,
        slots: usize,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        if slots == 1 {
            return self.plan_mtp_with(machine, (ctx_max, ubatch), levers, mtp, experts);
        }
        check_shape(ctx_max, ubatch)?;
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::DraftCards {
                cards: machine.cards.len(),
            });
        };
        let (draft, reserve) = mtp.draft_plan_of(ctx_max, slots)?;
        let arena = mtp.arena_bytes();
        let terms = self.seq_terms(Some(mtp));
        let rows = terms.plan_beside(slots as u64);
        let kv = terms.slots_of(slots as u64);
        if !machine.tiers.is_empty() {
            check_draft_reserve(machine, Some(reserve))?;
            let mut plan = self.target_of(
                machine,
                (ctx_max, ubatch),
                levers,
                experts,
                Beside::draft(rows, mtp).slots(slots),
                &kv,
            )?;
            plan.cards[0].grow_kv(rows);
            let broken: Vec<Violation> = plan
                .violations()
                .into_iter()
                .chain(draft.violations())
                .collect();
            if !broken.is_empty() {
                return Err(PlaceError::Broken(broken));
            }
            return Ok(MtpPlan {
                headroom_bytes: plan.cards[0].headroom_bytes,
                plan,
                draft,
                map_bytes: mtp.map_bytes,
                arena_bytes: arena,
            });
        }
        check_draft_reserve(machine, None)?;
        let mut plan = self.target_of(
            machine,
            (ctx_max, ubatch),
            levers,
            experts,
            Beside::draft(reserve + rows, mtp).slots(slots),
            &kv,
        )?;
        plan.cards[0].grow_kv(rows);
        let t = &plan.cards[0];
        // As in [`PlanInputs::plan_mtp_with`]: the draft's term is the
        // reserve the expert rule spread within, and the stage rope rows are
        // the target card's.
        let total =
            t.bytes()
                .checked_add(reserve)
                .ok_or_else(|| PlacementError::CardBytesPastU64 {
                    card: card.name.clone(),
                })?;
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        let mut broken: Vec<Violation> = plan
            .violations_beside(reserve)
            .into_iter()
            .filter(|v| !matches!(v, Violation::CardOver { .. }))
            .chain(draft.violations())
            .collect();
        if total > limit {
            broken.push(Violation::CardOver {
                card: card.name.clone(),
                total,
                limit,
            });
        }
        if !broken.is_empty() {
            return Err(PlaceError::Broken(broken));
        }
        Ok(MtpPlan {
            headroom_bytes: i128::from(usable) - i128::from(total),
            plan,
            draft,
            map_bytes: mtp.map_bytes,
            arena_bytes: arena,
        })
    }

    /// The target's plan of `slots` resident sequences on `machine` at
    /// `ctx_max` positions under `levers` and `experts`:
    /// [`PlanInputs::plan_with_slots`], or with the MTP draft `mtp` beside it
    /// [`PlanInputs::plan_mtp_with_slots`]'s target plan. Refused as they
    /// refuse.
    pub fn plan_target<'a>(
        &'a self,
        machine: &'a Machine,
        (ctx_max, ubatch): (u64, u64),
        levers: &PlanLevers,
        experts: Experts,
        mtp: Option<&'a MtpInputs>,
        slots: usize,
    ) -> Result<Plan<'a>, PlaceError> {
        match mtp {
            None => self.plan_with_slots(machine, (ctx_max, ubatch), levers, experts, slots),
            Some(mi) => Ok(self
                .plan_mtp_with_slots(machine, (ctx_max, ubatch), levers, mi, experts, slots)?
                .plan),
        }
    }
}

fn check_shape(ctx_max: u64, ubatch: u64) -> Result<(), PlaceError> {
    if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
        return Err(PlaceError::Positions { ctx_max });
    }
    if ubatch == 0 || ubatch > ctx_max.min(UBATCH_PLANNED) {
        return Err(PlaceError::Ubatch { ctx_max, ubatch });
    }
    Ok(())
}

/// A draft plan's card terms: its granules (dense, experts, rounding), its
/// store, and the head's row map.
fn draft_card_bytes(d: &placement::CardTotals, map_bytes: u64) -> u64 {
    d.dense_bytes + d.expert_bytes + d.rounding_bytes + d.kv_bytes + map_bytes
}

/// Card bytes the MTP draft's program arena holds at a vocabulary of `vocab`
/// [derived: the arena `MtpArena::new` allocates
/// for eight rows — its input record 65 u32 (a store walk's 64 ids); the hidden
/// rows in and the last row's own copy 2 · (81,920 + 10,240) f32; the
/// embedding, the mix and sub-layer outputs and the shared expert's rows, 7 ·
/// 20,480 f32; the input pack 163,840 f32; the streams 81,920 f32; the mix
/// scratch 94,880 f32 (the normed streams 81,920, the down's dots 10,240, the
/// inject's 128, the bottleneck 2,560, the weights 32); the query rows with
/// their gates 98,304 f32; the keys and values 2 · 4,096 f32; the attention
/// rows 2 · 49,152 f32; the dense flash's partials 49,536 f32 a segment, 80 of
/// them a row's head — `MTP_FLASH_SEGMENTS` below, `flash_gqa::SEGMENTS`
/// restated, the gpu crate the model crate does not read;
/// the router's buffers 8,288 f32 and 89 u32; the routed slots 80 u32 and the
/// identity map 512 u32; the routed rows 3 · 51,200 and their down 204,800 f32;
/// the head's input and logits' rows 20,480 f32 each; the head's logits 8 ·
/// `vocab` f32; the readback 18 u32, its ticket count, the stand-in map word
/// and the chain's 10 words; the taps none (the engine's walk arms none) —
/// 1,212,160 + 8 · `vocab` f32 and 793 u32; then the store walk's arena for 64
/// rows, 5,049,938 values: the hidden rows 655,360, the embedding 163,840 and
/// its positions and key counts 128, the pack 1,310,720 and its 32-value q8
/// blocks over 256 columns of 5,120 (codes, scales and sums, 20 words a
/// 64-value step) 409,600, the streams 655,360, the wide mix's scratch
/// 1,573,696 (the normed streams and the up's rows 655,360 each, the q8 of the
/// normed streams 204,800, the down's dots and the bottleneck 20,480 each, its
/// q8 6,400, its BF16 pairs 10,240 and refusals 64, the inject's dots and
/// the weights 256 each), the mix 163,840 and
/// its q8 51,200, the keys and values 65,536, and the two one-expert tables
/// of 256 and 64 slots, 524 and 134 words — four bytes a value].
///
/// [`MtpArena::new`]: bloomery_gpu::arch::qwen3moe::Mtp38
#[must_use]
pub fn mtp_arena_bytes(vocab: u64) -> u64 {
    4 * (1_212_160 + 8 * vocab + 49_536 * MTP_FLASH_SEGMENTS + 793 + 5_049_938)
}

/// The decode flash's segments a row of the MTP program's arena holds
/// partials for: `flash_gqa::SEGMENTS` restated, the model crate not reading
/// the gpu crate's. `gate_qwen4exp_mtp`'s arena clause holds the two equal.
const MTP_FLASH_SEGMENTS: u64 = 80;

/// The most rows one walk of the draft takes, the head's activation's
/// columns: `MTP_ROWS` restated, the model crate not reading the gpu crate's.
const MTP_HEAD_ROWS: u64 = 8;

/// Card bytes a Q6_K borrowed head's walk adds to the program's arena beside
/// [`mtp_arena_bytes`]: the head's input rows quantized to q8_1, 0 for a
/// Q8_0 or BF16 head whose projection reads the f32 rows. [derived: `Q8Act::with_k`
/// at `MTP_HEAD_ROWS` columns of `hidden` values — per column, at n_sb =
/// hidden/256 super-blocks, q3 `64 · ⌈n_sb/2⌉` u64, q4 `256 · ⌈n_sb/4⌉`
/// u32, q6 `128 · ⌈n_sb/2⌉` u32, s8 `8 · n_sb` i32 and d8 `2 · n_sb` f32,
/// `tensor::Q8Act`'s layout — at the file's hidden 2,560 (n_sb 10): 2,560 +
/// 3,072 + 2,560 + 320 + 80 = 8,592 B a column, 68,736 B the eight.]
///
/// [`MtpArena::new`]: bloomery_gpu::arch::qwen3moe::Mtp38
#[must_use]
pub fn mtp_head_act_bytes(head: mtp::BorrowedHead, hidden: u64) -> u64 {
    match head {
        mtp::BorrowedHead::Q8_0 | mtp::BorrowedHead::Bf16 => 0,
        mtp::BorrowedHead::Q6K => crate::placement::kernels::q8_act_bytes(MTP_HEAD_ROWS, hidden),
    }
}

/// A qwen4exp plan with its MTP draft ([`PlanInputs::plan_mtp`]).
#[derive(Debug)]
pub struct MtpPlan<'a> {
    /// The target's plan — under [`Experts::Host`] [`PlanInputs::plan`]'s
    /// bit for bit, under [`Experts::Card`] its expert rule within the card's
    /// budget less the draft's card bytes: its card totals and headroom are
    /// the target's alone — on a plan (b′) machine the stage card's
    /// [`MTP_RESERVE`] row is the draft's bytes, so that card's reserve and
    /// headroom count them.
    pub plan: Plan<'a>,
    /// The draft's plan on [`MtpInputs::machine`]: its tensors' rows, its
    /// routed experts (every one on the card), its granules in its own heap
    /// and its store (`cards[0].kv_bytes`).
    pub draft: Plan<'a>,
    /// The head's row → id map, one `u32` a vocabulary id whatever the head.
    pub map_bytes: u64,
    /// The draft program's arena ([`MtpInputs::arena_bytes`]: at the
    /// draft's vocabulary, with its borrowed head's activation), counted
    /// beside the draft's card bytes; the load's `Mtp38::arm` holds the arena
    /// it allocates to it.
    pub arena_bytes: u64,
    /// The card's usable bytes (capped by the card budget) less the target's
    /// and the draft's card terms; the margin is inside it.
    pub headroom_bytes: i128,
}

impl MtpPlan<'_> {
    /// Bytes the draft adds to the card: its granules, its store and the
    /// head's row map.
    #[must_use]
    pub fn draft_card_bytes(&self) -> u64 {
        draft_card_bytes(&self.draft.cards[0], self.map_bytes)
    }

    /// Device bytes the draft's tensors hold, the allocator's rounding
    /// excluded: what its load's uploads sum to (the borrowed matrices are
    /// the target's).
    #[must_use]
    pub fn draft_resident_bytes(&self) -> u64 {
        let d = &self.draft.cards[0];
        d.dense_bytes + d.expert_bytes
    }
}

/// The most positions a qwen4exp ubatch walk takes, and the size a load runs
/// with its ubatch lever unset: the default [`machine`]'s `ubatch`.
pub const UBATCH_PLANNED: u64 = 4096;

/// Card bytes a qwen4exp ubatch walk's arena holds a position: its
/// token-major rows, 113,860 f32, 10,240 BF16, 12 u32 and 1,440 u8 — the embedding, the
/// F32 and the BF16 stream buffers, the mix and sub-layer outputs, the attention rows, the delta
/// layer's, the selecting layer's and the PLE site's intermediates (with the
/// table's rows as read, 9 bytes a 16 values), the shared
/// expert's; no selected-flash queries, partials or output. The
/// arena's own allocation is held to it
/// (`crates/gpu/src/arch/qwen3moe/scratch38.rs`'s `Arena38::wide`).
// PIN(2026-10-29): 497,888 (was 496,448): the PLE site's NVFP4 planes, 1,440 B a position.
// PIN(2026-10-30): 477,408 (was 497,888): the stream buffer the walk carries in BF16 holds a
// halfword a value, 4 · 2,560 · 2 = 20,480 B a position less.
pub const UBATCH_ARENA_TOKEN_BYTES: u64 = 477_408;

/// Card bytes a qwen4exp ubatch walk holds a position [derived: the ubatch
/// arena's, [`UBATCH_ARENA_TOKEN_BYTES`]; the walk's own
/// buffers, 98,964.125 B — the wide
/// mixes' scratch (98,356 B: the normed streams and the up's rows at 40,960 B
/// each, their activations, the down's and inject's dots, the bottleneck and
/// its Q8 activations, BF16 pairs and refusal, the weights), the indexer
/// keys, the slots and the one-expert table; the record's id word and the
/// host sums (10,244 B);
/// rounded up].
// PIN(2026-10-24): the HC bottleneck adds 320 BF16 values and one refusal word per column, 644 B.
// PIN(2026-10-25): 644,345 (was 668,921): the prefill flash stores the selecting layer's rows
// under their gate, so the arena holds no flash rows, 6,144 f32 a position.
// PIN(2026-10-28): 617,337 (was 644,345): the arena holds no query rows (6,144 f32), no
// separate β and α rows (96 f32) and no one-row indexer queries (512 f32) a position.
// PIN(2026-11-01): 109,209 (was 120,889): the activations' quantized forms are the file's
// ([`Stack::act_token_bytes`]), counted beside this.
pub const UBATCH_TOKEN_BYTES: u64 = UBATCH_ARENA_TOKEN_BYTES + 109_209;

/// Card bytes a qwen4exp ubatch walk's selection holds for one run of its
/// selecting rows: the walk takes as many rows a run as these bytes hold at
/// the selection's bytes a row (its query heads, scores, list and length:
/// 4 B each of 512, `⌈ctx / 4⌉`, 2,051 and 1), in whole groups of eight
/// rows and at least one group (`crates/gpu/src/arch/qwen3moe/scratch38.rs`'s
/// `select_run`).
pub const SELECT_RUN_BYTES: u64 = 22_000_000;

/// Card bytes a qwen4exp ubatch walk holds whatever its size, at a cache of
/// up to 2,000,000 positions [derived, an upper bound: the router's buffers
/// for 2,048 tokens, 8,577,028 B; the eight-column mix scratch, 379,520 B;
/// the selection's run, at most [`SELECT_RUN_BYTES`] (the least run, eight
/// rows, holds 16,082,048 B at 2,000,000); the delta layers' chunk records
/// past the q·k·v rows of a walk of up to five positions, at most 133,124 B;
/// 31,089,672 B, rounded up to 32 MiB]. A load whose ubatch arena passes it refuses the arena by name.
pub const UBATCH_FIXED_BYTES: u64 = 32 << 20;

/// The card bytes a position of a qwen4exp ubatch walk holds in the q8
/// blocks of 32 of its model-width, attention-width and SwiGLU rows
/// [derived: 2,560, 6,144 and 640 values at 1.25 B, 3,200, 7,680 and 800 B]:
/// what a machine built from a layer count alone counts for the quantized
/// forms its dense sites read ([`Stack`]), every row's q8 blocks of 32 and no
/// q8_1 form.
pub const Q8_BLOCKS32_TOKEN_BYTES: u64 = 3_200 + 7_680 + 800;

/// The facts a machine counts a qwen4exp load's ubatch walk by: the file's
/// layers and the card bytes a position of the walk holds in the quantized
/// forms its wide dense sites read ([`PlanInputs::stack`] for a file,
/// [`Stack::q8_blocks32`] for a synthetic spec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stack {
    pub layers: usize,
    pub act_token_bytes: u64,
}

impl Stack {
    /// `layers` layers of a stack whose dense sites read the q8 blocks of 32
    /// of every row ([`Q8_BLOCKS32_TOKEN_BYTES`]): a synthetic spec's, which
    /// has no file types for [`PlanInputs::stack`] to read.
    #[must_use]
    pub const fn q8_blocks32(layers: usize) -> Stack {
        Stack {
            layers,
            act_token_bytes: Q8_BLOCKS32_TOKEN_BYTES,
        }
    }
}

/// The card bytes a qwen4exp ubatch walk of up to `u` positions holds: what
/// its load's allocations may not pass.
#[must_use]
pub const fn ubatch_scratch_bytes(u: u64) -> u64 {
    u * UBATCH_TOKEN_BYTES + UBATCH_FIXED_BYTES
}

/// The card scratch of a qwen4exp plan whose load runs ubatches of up to
/// `ubatch` positions: the m = 1 scratch and the ubatch walk's
/// ([`ubatch_scratch_bytes`]).
#[must_use]
pub const fn card_scratch_bytes(ubatch: u64) -> u64 {
    SCRATCH + ubatch_scratch_bytes(ubatch)
}

/// The most tokens one routed run of a ubatch walk's card route takes, and
/// so the run buffers' width: the walk routes in runs of this many tokens
/// (`crates/gpu/src/arch/qwen3moe/scratch38.rs`'s `ROUTE_ROWS`, restated
/// here — the model crate does not read the gpu crate), unless the plan
/// widened the run to the whole ubatch ([`widen_card_route`]).
pub const CARD_ROUTE_ROWS: u64 = 2_048;

/// Card bytes a ubatch walk's card route holds a run token, an upper bound
/// [derived: the compressed ids and the places 80 B (ten u32 each); the
/// gate·up input `GemmAct` at K = 2,560, 8,592 B a column (q3 2,560 + q4
/// 3,072 + q6 2,560 + s8 320 + d8 80), which also holds a load whose card
/// gate·up the 32-value GEMM reads in its one input form, a `GemmAct32` of
/// 3,200 B a column (40 steps of 80 B); the gate and up f32 rows
/// 2 · 10 · 640 · 4 = 51,200 B; the SwiGLU `GemmAct32` at K = 640, 800 B a
/// column, 8,000 B; the down f32 rows 10 · 2,560 · 4 = 102,400 B (a load
/// whose every layer runs without the host leg keeps them in BF16, half);
/// two route tables — the cols list 2 · 40 B, their tile words at the most
/// experts 2 · 8 · (320 + 512) / 2,048 = 3.25 B, the counts; the identity
/// map of the ids compression 512 words = 1 B: 170,272 + 87.6 + 1, rounded
/// up].
pub const CARD_ROUTE_RUN_TOKEN_BYTES: u64 = 170_360;

/// Card bytes a ubatch walk's card route holds a run token in
/// [`RouteForm::Native`], an upper bound [derived: the compressed ids, the
/// places and a landing batch's places 120 B (ten u32 each); the gate·up
/// input's native FP4 form at K = 2,560, 1,444 B a column (codes 1,280,
/// block scales 160, the column's scale 4), and no 32-value or q8_1 form;
/// no gate or up rows; the SwiGLU `GemmAct32` at K = 640 in its native
/// form alone, with the frames the gate and up launch writes it through,
/// 408 B a column (codes 320, block scales 40, the column's scale 4, a
/// frame a 64-value step 40, a tile's arrival word 4), 4,080 B; the down
/// f32 rows 10 · 2,560 · 4 = 102,400 B (a load whose every layer runs
/// without the host leg keeps them in BF16, half); two route tables — the
/// cols list 2 · 40 B, the zero ids of a table over one expert 40 B, their
/// tile words at the most experts 2 · 8 · (320 + 512) / 2,048 = 6.5 B, the
/// counts; the identity map of the ids compression 512 words = 1 B:
/// 108,164 + 6.5 + 1, rounded up]. The tables' expert tiles and the
/// identity map are spread over a run of [`CARD_ROUTE_ROWS`] tokens, as in
/// [`CARD_ROUTE_RUN_TOKEN_BYTES`].
pub const NATIVE_ROUTE_RUN_TOKEN_BYTES: u64 = 108_172;

pub use crate::placement::kernels::ffn::RouteForm;

/// The card bytes a run token of the route holds in `form`
/// ([`CARD_ROUTE_RUN_TOKEN_BYTES`], [`NATIVE_ROUTE_RUN_TOKEN_BYTES`]), which
/// the plan counts a run token by ([`card_route_scratch_bytes`]); a load
/// reads its file's form ([`PlanInputs::route_form`]).
const fn route_token_bytes(form: RouteForm) -> u64 {
    match form {
        RouteForm::Rows => CARD_ROUTE_RUN_TOKEN_BYTES,
        RouteForm::Native => NATIVE_ROUTE_RUN_TOKEN_BYTES,
    }
}

/// Card bytes a ubatch walk's card route holds a token of the whole unit
/// [derived: the card slots' weighted sums, `HIDDEN` f32 a row].
pub const CARD_ROUTE_ACC_TOKEN_BYTES: u64 = 10_240;

/// The card bytes a qwen4exp ubatch walk's card route in `form` holds at
/// ubatches of up to `u` positions: the run buffers and route tables once
/// `min(u, CARD_ROUTE_ROWS)` run tokens, the sums a token of the whole
/// unit. Counted under [`Experts::Card`] only
/// ([`machine_for_route`]): a host-routed plan's walk holds none of it.
#[must_use]
pub const fn card_route_scratch_bytes(u: u64, form: RouteForm) -> u64 {
    let run = if u < CARD_ROUTE_ROWS {
        u
    } else {
        CARD_ROUTE_ROWS
    };
    run * route_token_bytes(form) + u * CARD_ROUTE_ACC_TOKEN_BYTES
}

/// Card bytes the ubatch walk's card route holds past
/// [`card_route_scratch_bytes`] on a load with an expert tier, at ubatches
/// of up to `u` positions [derived: a tier layer's join sums the card's and
/// the tier's slots in slot order after the host's wait, so the route keeps
/// its card slots' down rows and places for the whole unit, not one run —
/// (u − min(u, CARD_ROUTE_ROWS)) tokens more of [`JOIN_TOKEN_BYTES`] — and
/// the unit's tier places and their ranks among the tier slots,
/// 2 · 10 · 4 = 80 B a token].
#[must_use]
pub const fn card_tier_join_bytes(u: u64) -> u64 {
    let run = if u < CARD_ROUTE_ROWS {
        u
    } else {
        CARD_ROUTE_ROWS
    };
    (u - run) * JOIN_TOKEN_BYTES + u * 80
}

/// Card bytes a tier layer's join keeps a token of the unit past the card
/// route's run: its card slots' down rows and places [derived:
/// 10 · 2,560 · 4 + 10 · 4], which a run holds a run token too
/// ([`route_token_bytes`]).
const JOIN_TOKEN_BYTES: u64 = 102_440;

/// What a stage card's reserve for its card route's run of the whole
/// ubatch is called ([`widen_card_route`]): the bytes that run holds past
/// the runs of [`CARD_ROUTE_ROWS`] tokens the card's scratch counts
/// ([`route_unit_bytes`]). A load reads its run from it
/// ([`card_route_run`]).
const ROUTE_UNIT_RESERVE: &str = "card route's whole-ubatch run";

/// The card bytes the card route's run in `form` of a whole ubatch of up
/// to `u` positions holds past its runs of [`CARD_ROUTE_ROWS`] tokens: the
/// `u − min(u, CARD_ROUTE_ROWS)` run tokens more
/// ([`route_token_bytes`]), less on a load with an expert tier
/// (`tiered`) the down rows and places the join already keeps for them
/// ([`card_tier_join_bytes`]). 0 for a ubatch of one such run; saturated
/// past u64 bytes, which no card holds.
#[must_use]
const fn route_unit_bytes(u: u64, tiered: bool, form: RouteForm) -> u64 {
    let past = u.saturating_sub(CARD_ROUTE_ROWS);
    let token = if tiered {
        route_token_bytes(form) - JOIN_TOKEN_BYTES
    } else {
        route_token_bytes(form)
    };
    past.saturating_mul(token)
}

/// Where a plan puts each tensor: each row's pieces as their device,
/// format and experts ([`placement::Segment`]), in the plan's row order.
pub type TensorPlaces = Vec<Vec<(Device, Format, Option<ExpertList>)>>;

/// [`TensorPlaces`] of `plan`.
#[must_use]
pub fn tensor_places(plan: &Plan<'_>) -> TensorPlaces {
    plan.rows
        .iter()
        .map(|r| {
            r.segments
                .iter()
                .map(|s| (s.device, s.format, s.experts.clone()))
                .collect()
        })
        .collect()
}

/// `machine`, a qwen4exp machine for ubatches of up to `ubatch` positions
/// ([`machine_for_route`], [`machine_bp_on`]) whose card route in `form`
/// runs in runs of [`CARD_ROUTE_ROWS`] tokens, with that run widened to the
/// whole ubatch when nothing the plan places moves for it: its stage card
/// then reserves [`route_unit_bytes`] as [`ROUTE_UNIT_RESERVE`], and a
/// ubatch walk reads each card expert's weights once instead of once a run.
/// `plan` is the caller's plan of a machine ([`PlanInputs::plan_target`]):
/// the run is widened when the stage card holds a routed expert and the plan
/// of the widened machine places every tensor where the plan of `machine`
/// does ([`tensor_places`]), so the
/// wider run takes only room no expert, table or tier would use (on a
/// unified pool the host's room shrinks with the card's reserves). The
/// reserve is the one difference between the two machines, so a plan of the
/// widened one that fails is the reserve not fitting, and keeps the runs.
/// `plan` hands each plan it makes to its second argument, which reads what
/// the caller needs of it (`read`), so the caller does not plan the machine
/// returned again: the plan of `machine` is read, and the plan of the widened
/// one only when its places equal those, the one case that machine is
/// returned. Returns the machine and the read of its plan. Refused as `plan`
/// refuses `machine`, or as [`PlaceError::MissingPlan`] when a successful
/// callback returns without handing on a plan.
pub fn widen_card_route<T>(
    mut machine: Machine,
    ubatch: u64,
    form: RouteForm,
    plan: impl Fn(&Machine, &mut dyn FnMut(&Plan<'_>)) -> Result<(), PlaceError>,
    read: impl Fn(&Plan<'_>) -> T,
) -> Result<(Machine, T), PlaceError> {
    let mut first = None;
    plan(&machine, &mut |p| first = Some((tensor_places(p), read(p))))?;
    let (held, held_read) = first.ok_or(PlaceError::MissingPlan)?;
    let bytes = route_unit_bytes(ubatch, !machine.tiers.is_empty(), form);
    let stage_expert = held.iter().flatten().any(|(device, _, experts)| {
        *device == Device::Card(0) && experts.as_ref().is_some_and(|e| !e.is_empty())
    });
    if bytes == 0 || !stage_expert {
        return Ok((machine, held_read));
    }
    let Some(stage) = machine.cards.first_mut() else {
        return Ok((machine, held_read));
    };
    stage.reserves.push((ROUTE_UNIT_RESERVE.to_string(), bytes));
    let mut wide = None;
    let mut seen = false;
    let widened = plan(&machine, &mut |p| {
        seen = true;
        if tensor_places(p) == held {
            wide = Some(read(p));
        }
    });
    if widened.is_ok() && !seen {
        return Err(PlaceError::MissingPlan);
    }
    match (widened, wide) {
        (Ok(()), Some(wide_read)) => Ok((machine, wide_read)),
        _ => {
            machine.cards[0].reserves.pop();
            Ok((machine, held_read))
        }
    }
}

/// The tokens one run of a load's card route in `form` takes at ubatches of
/// up to `u` positions on `card`, the stage card of its plan's machine
/// (`tiered`, a machine with an expert tier), and the card bytes its ubatch
/// walk may hold past the m = 1 scratch: the whole ubatch with the run's
/// reserve beside [`workstation::counted_arena_bytes`] on a card that holds
/// [`ROUTE_UNIT_RESERVE`] ([`widen_card_route`]), else at most
/// [`CARD_ROUTE_ROWS`] and the counted bytes alone. Refused by name: more
/// than one such reserve, or one of other than [`route_unit_bytes`]'s
/// bytes (none for a ubatch of one run).
pub fn card_route_run(
    card: &Card,
    u: u64,
    tiered: bool,
    form: RouteForm,
) -> Result<(u64, u64), PlaceError> {
    let counted = workstation::counted_arena_bytes(card);
    let got = reserve_rows(card, ROUTE_UNIT_RESERVE);
    let want = route_unit_bytes(u, tiered, form);
    match *got.as_slice() {
        [] => return Ok((u.min(CARD_ROUTE_ROWS), counted)),
        [b] if b == want && want > 0 => return Ok((u, counted.saturating_add(b))),
        _ => {}
    }
    Err(PlaceError::RouteUnitReserve {
        card: card.name.clone(),
        got,
        u,
        want,
    })
}

/// What a stage card's reserve for the ubatch walk's activations in the
/// forms the dense MMA tile copies is called ([`machine_for_experts`]):
/// the model-width rows rounded once to BF16 for their BF16 sites, and the
/// model-width, attention-width and SwiGLU rows in MXFP8 for their MXFP8
/// sites' W8A8 products. A load reads it from the row ([`act_forms_bytes`]).
const ACT_FORMS_RESERVE: &str = "activations' MMA forms";

/// Card bytes a position of the ubatch walk holds in [`ACT_FORMS_RESERVE`]
/// [derived: the model-width row's BF16 form, 2,560 values at 2 B and a
/// refusal word, 5,124 B; the MXFP8 forms of the model-width, attention-width
/// and SwiGLU rows, 2,560, 6,144 and 640 values at a byte, a scale byte per
/// 32 values and a refusal word each, 2,644, 6,340 and 664 B].
const ACT_FORMS_TOKEN_BYTES: u64 = 5_124 + 2_644 + 6_340 + 664;

/// The bytes `card`, the stage card of a load's plan, sets aside for the
/// activations' MMA forms of ubatches of up to `u` positions: its
/// [`ACT_FORMS_RESERVE`] row, or 0 on a card without one, whose load
/// converts the f32 rows in each tile instead. Refused by name: more than
/// one such row, or one of fewer than `u` · [`ACT_FORMS_TOKEN_BYTES`].
pub fn act_forms_bytes(card: &Card, u: u64) -> Result<u64, PlaceError> {
    let got = reserve_rows(card, ACT_FORMS_RESERVE);
    let want = u * ACT_FORMS_TOKEN_BYTES;
    match *got.as_slice() {
        [] => Ok(0),
        [b] if b >= want => Ok(b),
        _ => Err(PlaceError::ActFormsReserve {
            card: card.name.clone(),
            got,
            u,
            want,
        }),
    }
}

/// Tier-card bytes a run token of the tier's block route holds
/// (`tier38`'s `BlockRoute38`) [derived: the route's ids 10 · 4 = 40 B; the
/// run's rows' `GemmAct` at K = 2,560, 8,592 B; the gate and up f32 rows
/// 2 · 10 · 640 · 4 = 51,200 B; the SwiGLU `GemmAct32` at K = 640, 800 B a
/// slot, 8,000 B; the down GEMM's f32 rows the run's pack reads, 10 · 2,560 ·
/// 4 = 102,400 B].
pub const TIER_ROUTE_RUN_TOKEN_BYTES: u64 = 170_232;

/// The most experts a layer has: the bound a tier route table is counted at
/// before the plan sets the tier's counts.
const ROUTE_EXPERTS_BOUND: u64 = 512;

/// The tier-card bytes of the tier's block route for blocks of up to `cols`
/// columns [derived: [`TIER_ROUTE_RUN_TOKEN_BYTES`] a token of one run of
/// `min(cols, CARD_ROUTE_ROWS)`; two route tables of that run's `S` slots
/// counted at [`ROUTE_EXPERTS_BOUND`] experts, each its slot list 4 · S, its
/// tile words 8 · (S / 64 + min(512, S)) and its counts 8; two identity maps
/// of 513 words; the block's slots' ranks, 10 · 4 = 40 B a column]. The
/// tier holds the tables of its own counts and the rest of this bound beside
/// them, so its block bytes are this reserve.
#[must_use]
pub const fn tier_route_scratch_bytes(cols: u64) -> u64 {
    let run = if cols < CARD_ROUTE_ROWS {
        cols
    } else {
        CARD_ROUTE_ROWS
    };
    let slots = run * 10;
    let tiles = slots / 64
        + if slots < ROUTE_EXPERTS_BOUND {
            slots
        } else {
            ROUTE_EXPERTS_BOUND
        };
    run * TIER_ROUTE_RUN_TOKEN_BYTES
        + cols * 40
        + 2 * (4 * slots + 8 * tiles + 8)
        + 2 * 4 * (ROUTE_EXPERTS_BOUND + 1)
}

/// The machine a qwen4exp plan runs on: `card` runs every one of `layers`,
/// the head and the token embedding table whole (the file's q8_0 rows, which
/// the card gathers), with this workstation's context and margin and the host
/// its card's machine has ([`workstation::placed_on`]: the pool on a unified
/// machine)
/// and the scratch of a load that runs ubatches of up to `ubatch` positions
/// ([`card_scratch_bytes`]): the size the load itself takes, read once by
/// the caller that builds this machine and opens the load. A host-routed
/// plan's load ([`Experts::Host`]); a card one's
/// ([`machine_for_experts`]) beside the ubatch walk's card route.
#[must_use]
pub fn machine(card: CardSpec, stack: Stack, ubatch: u64) -> Machine {
    machine_for_experts(card, stack, ubatch, Experts::Host)
}

/// [`machine_for_route`] with the card route counted in [`RouteForm::Rows`],
/// the larger form.
#[must_use]
pub fn machine_for_experts(card: CardSpec, stack: Stack, ubatch: u64, experts: Experts) -> Machine {
    machine_for_route(card, stack, ubatch, experts, RouteForm::Rows)
}

/// [`machine`] for a plan whose routed experts go where `experts` says: under
/// [`Experts::Card`] the ubatch walk's card route scratch in `form`
/// ([`card_route_scratch_bytes`]) is counted beside the ubatch's — the walk
/// holds those buffers when the plan puts experts on the card. A unified
/// card's stage reserves the activations' MMA forms
/// ([`ACT_FORMS_RESERVE`]): the dense MMA tile reads them, and the plan
/// sees the pool, not the compute capability or the sites' formats, so
/// another unified card's load, or a load with no site of a form, leaves
/// those bytes unread. A discrete card reserves none, and its dense MMA
/// tile, where its device runs it, converts the f32 rows in each tile.
#[must_use]
pub fn machine_for_route(
    card: CardSpec,
    stack: Stack,
    ubatch: u64,
    experts: Experts,
    form: RouteForm,
) -> Machine {
    let route = match experts {
        Experts::Host => 0,
        Experts::Card => card_route_scratch_bytes(ubatch, form),
    };
    let scratch = card_scratch_bytes(ubatch) + ubatch * stack.act_token_bytes + route;
    let mut machine = workstation::placed_on(card, stack.layers, scratch);
    if card.integrated {
        machine.cards[0].reserves.push((
            ACT_FORMS_RESERVE.to_string(),
            ubatch * ACT_FORMS_TOKEN_BYTES,
        ));
    }
    machine
}

/// What the MTP draft's reserve on the stage card of a plan (b′) machine is
/// called ([`machine_bp`]): its card bytes and arena
/// ([`MtpInputs::card_bytes`]).
pub const MTP_RESERVE: &str = "MTP draft";

/// Tokens one chunk of the Qwen3.6 card leg's prompt-batch service runs.
/// [`tier_batch`] is the qwen4exp plan's figure, not this one.
pub const LEG_TIER_CHUNK: u64 = 8;

/// The Qwen3.6 card leg's prompt-batch bytes ([`TierBatchBytes`]): the tier
/// card's staging, one chunk of the leg's scratch and the host's rows, for
/// blocks of the host union's columns. [`tier_batch`] is the qwen4exp plan's
/// figure.
pub fn leg_tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let spec = &inputs.spec;
    let moe = spec.routed().ok_or_else(|| PlacementError::Metadata {
        key: "Qwen3.6 expert tier".into(),
        detail: "a dense model has no routed experts for an expert tier".into(),
    })?;
    let (expert_ff, n_used) = (u64::from(moe.expert_ff), u64::from(moe.top_k));
    workstation::try_tier_leg_batch_bytes_of(
        u64::from(spec.hidden),
        expert_ff,
        n_used,
        (LEG_TIER_CHUNK, crate::moe::UNION_MAX_COLS as u64),
        leg_down_forms(Program::Qwen35Body, &inputs.model),
        false,
    )
}

/// The expert tier's prompt-batch bytes a qwen4exp plan (b′) reserves
/// ([`machine_bp`]) for a load whose ubatches take up to `ubatch` positions.
/// The host tier's batch port is made for blocks as wide as the ubatch
/// (`Body38`'s host columns, at least [`PASS_ROWS`]), and its tier side is
/// the common one: on the tier card the staging of a block — the
/// activations, the tier places, their q8_1 form (the same 8,592 bytes a
/// column at the model width as the card route's gate·up input), the down
/// outputs by slot ([`tier_batch_staging_bytes`]) — and on the host, per
/// exchange set, the rows the tier hands back and the places it reads
/// ([`tier_batch_host_bytes`]). The tier's block scratch is its block
/// route's at those columns ([`tier_route_scratch_bytes`]): the tier runs
/// its slots of a block through the route the stage card runs its own
/// through, into the staging's down outputs, and writes its slots' rows
/// into the host's rows itself.
#[must_use]
pub fn tier_batch(spec: &ModelSpec, ubatch: u64) -> TierBatchBytes {
    let n_used = spec.routed().map_or(0, |m| u64::from(m.top_k));
    tier_batch_of(u64::from(spec.hidden), n_used, ubatch)
}

/// [`tier_batch`] for rows of `n_embd` routed to `n_used` experts.
const fn tier_batch_of(n_embd: u64, n_used: u64, ubatch: u64) -> TierBatchBytes {
    let cols = if ubatch > PASS_ROWS as u64 {
        ubatch
    } else {
        PASS_ROWS as u64
    };
    TierBatchBytes {
        staging: tier_batch_staging_bytes(n_embd, n_used, cols),
        scratch: tier_route_scratch_bytes(cols),
        host: tier_batch_host_bytes(n_embd, n_used, cols),
        leg: None,
    }
}

/// Plan (b′) for a qwen4exp file (`--place bp`): the A6000 runs every one
/// of `layers`, the head and the token embedding as
/// [`machine_for_experts`] lays it out under [`Experts::Card`] for
/// ubatches of up to `ubatch` positions, its scratch with the ubatch walk's
/// tier join beside it ([`card_tier_join_bytes`]), with `draft` — the MTP draft's
/// card bytes ([`MtpInputs::card_bytes`]) when the draft runs beside the
/// target — as its named reserve [`MTP_RESERVE`]; the 3090 is an expert
/// tier beside the host, with no stage, its prompt-batch service `batch`
/// ([`tier_batch`]) a named reserve on it and on the host.
///
/// **The split.** The planner fills the A6000 first and then the tier with
/// each layer's next ids, each card to its own budget, so the A6000 keeps
/// the counts plan (a) gives it less what the tier join's scratch takes,
/// and the tier takes the next ids it holds after them. That is the rule this plan keeps, from
/// the per-slot costs:
/// - the residency machine holds the tier's experts away for the load's
///   life (never admitted, never a victim), so the tier's share of a
///   pass's routed slots is the routing mass of its fixed ids — about
///   its count over the 512 experts under the id prefix, whatever the
///   A6000 keeps — and the A6000's residency serves the hottest of the
///   other ids;
/// - an expert the A6000 gives up under its budget goes to the host, not
///   the tier (the tier is at its own budget), and a host slot costs four
///   to five A6000 slots, so a smaller A6000 share trades one card slot for
///   four or five card slots' time on the host leg, which wins only where
///   the host leg sits well under the A6000's; how far under is a hit-rate
///   question, which a plan made before any routing cannot answer;
/// - so the tier takes from the A6000 in routed mass, not in count: its
///   fixed ids carry the share of the routing that plan (a)'s residency
///   served on the A6000 (and the rest from the host), and the A6000's leg
///   shortens by that share while the tier's runs beside it.
///
/// The per-card counts at 4,096 positions and their derivation are pinned in
/// `crates/model/tests/qwen4exp_meta.rs` (plain) and
/// `crates/model/tests/qwen4exp_mtp_meta.rs` (with the draft). The machine is
/// the named reproduction of the maintainer's two cards; a load lays out
/// [`machine_bp_on`] over the cards its placement resolved on the census.
#[must_use]
pub fn machine_bp(stack: Stack, ubatch: u64, draft: Option<u64>, batch: TierBatchBytes) -> Machine {
    machine_bp_on((A6000, RTX_3090), stack, ubatch, draft, batch)
}

/// [`machine_bp`] with `stage` the stage card and `tier` its expert tier
/// card: the cards a placement resolved on this process's devices.
#[must_use]
pub fn machine_bp_on(
    (stage, tier): (CardSpec, CardSpec),
    stack: Stack,
    ubatch: u64,
    draft: Option<u64>,
    batch: TierBatchBytes,
) -> Machine {
    let mut m = machine_for_experts(stage, stack, ubatch, Experts::Card);
    m.cards[0].scratch_bytes += card_tier_join_bytes(ubatch);
    m.cards[0]
        .reserves
        .extend(draft.map(|b| (MTP_RESERVE.to_string(), b)));
    m.tiers.push(workstation::tier_card(tier, batch));
    m.host_mut()
        .reserves
        .push((TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
    m
}

/// The bytes of each of `card`'s reserves named `name`, in order.
fn reserve_rows(card: &Card, name: &str) -> Vec<u64> {
    card.reserves
        .iter()
        .filter(|(n, _)| n == name)
        .map(|&(_, b)| b)
        .collect()
}

/// The stage cards' [`MTP_RESERVE`] rows against `want`: none for `None`,
/// exactly one of `want` bytes on the one stage card for `Some`; anything
/// else is refused by name ([`PlaceError::DraftReserve`]).
fn check_draft_reserve(machine: &Machine, want: Option<u64>) -> Result<(), PlaceError> {
    for (i, card) in machine.cards.iter().enumerate() {
        let got = reserve_rows(card, MTP_RESERVE);
        let ok = match want {
            Some(b) if i == 0 => got == [b],
            _ => got.is_empty(),
        };
        if !ok {
            return Err(PlaceError::DraftReserve {
                card: card.name.clone(),
                got,
                want,
            });
        }
    }
    Ok(())
}

/// One file's per-layer bytes beside the weights, all on the card:
/// - a GDN layer: its recurrent state (`v_heads` heads of `state × state`
///   f32) and its conv ring (`conv − 1 + PASS_ROWS` rows of the conv's
///   channels, `2·k_heads·state + v_heads·state`, in f32), both fixed by the
///   file; on a qwen4exp file its one state lane is stamped
///   (`runtime::stores::delta_lane_bytes`) and beside it are a carry
///   (`runtime::stores::delta_carry_bytes`) and a verify record of
///   `runtime::stores::DELTA_VERIFY_ROWS` rows
///   (`runtime::stores::delta_record_bytes`); a qwen35moe load that verifies
///   drafted rows asks for its own lane count ([`KvLayout::of_spec_lanes`]);
/// - the PLE site's layer, beside that: the PLE conv ring
///   (`(taps − 1)·dilation + PASS_ROWS` rows of `streams · n_embd` f32);
/// - an attention layer: a position's K and V (`2 · kv_heads · head_dim`
///   values — f16, or q8_0's two-plane layout when the load runs its cache
///   in it ([`KvLayout::in_q8`], the cache lever's `q8_0`), a selecting
///   layer's planes as well); with the selector, also a pooled key (`idx_dim`
///   f16, in either layout) per pool of `ratio` positions, the last pool
///   counted whole, and a raw indexer key ring of `ratio − 1 + PASS_ROWS`
///   rows of `idx_dim` f16 whatever the context
///   (`runtime::stores::selecting_bytes`).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per layer: its kind.
    kinds: Vec<Kind>,
    /// A GDN layer's state and conv ring, in bytes.
    recurrent: u64,
    /// What one checkpoint copies of a GDN layer: the committed state lane
    /// and the conv ring ([`recurrent_bytes`], without the other lanes), and
    /// the carry on a qwen4exp file.
    copy: u64,
    /// The PLE site's layer and its conv ring's bytes.
    ple: Option<(usize, u64)>,
    /// An attention layer's K and V heads and their width.
    kv_heads: usize,
    head_dim: usize,
    /// The selector's key width and each layer's pool (0 on a GDN layer);
    /// `None` on a qwen35moe file.
    select: Option<(usize, Vec<usize>)>,
    /// An attention layer's K/V planes in q8_0's layout.
    q8: bool,
    /// The stage rope table, in bytes a position: `rope.dimension_count`
    /// f32. The body copies one table at load, not one per sequence. A media
    /// sequence's own table is outside this count.
    rope_row: u64,
}

impl KvLayout {
    /// The layout `spec` describes, its GDN layers keeping one lane: under
    /// [`PROGRAM`] a verify records its rows beside the lane, and under the
    /// Qwen3.5 body a load that verifies drafted rows asks for more lanes
    /// ([`KvLayout::of_spec_lanes`]).
    ///
    /// # Errors
    ///
    /// The layout's refusal of the description ([`KvLayout::of_spec_lanes`]).
    pub fn of_spec(spec: &ModelSpec) -> Result<KvLayout, PlacementError> {
        KvLayout::of_spec_lanes(spec, 1)
    }

    /// The layout `spec` describes with every GDN layer keeping `lanes`
    /// stamped lanes of state (`runtime::stores::delta_lane_bytes`): one lane
    /// is the state in place with no stamps, more are a verify's rows, one
    /// lane a row. The lane count is an input of the layout, whichever model.
    /// A layer is a GQA layer (a plane of every position, or the token-pool
    /// selector's pooled keys) or a gated-delta-rule layer; the layers of one
    /// kind share one shape.
    ///
    /// # Errors
    ///
    /// A layer whose stores this layout does not size, or whose shape is not
    /// its kind's first, by layer ([`PlacementError::Unimplemented`]).
    ///
    /// # Panics
    ///
    /// On no lane, by name.
    pub fn of_spec_lanes(spec: &ModelSpec, lanes: usize) -> Result<KvLayout, PlacementError> {
        assert!(
            lanes >= 1,
            "KvLayout::of_spec_lanes: a state of at least one lane"
        );
        let mut kinds = Vec::with_capacity(spec.layers.len());
        let mut pools = Vec::with_capacity(spec.layers.len());
        let mut gdn: Option<&DeltaRule> = None;
        let mut attn = None;
        for (layer, l) in spec.layers.iter().enumerate() {
            let refuse = |what: &str| unsized_layer(layer, what);
            match &l.mixer {
                Mixer::Gqa(g) => {
                    if g.window.is_some() || g.value_dim != g.head_dim {
                        return Err(refuse(
                            "a windowed GQA layer or one with its own value width",
                        ));
                    }
                    let (pool, idx_dim) = match &g.select {
                        None => (0, None),
                        Some(Selector::TokenPool { pool: 0, .. }) => {
                            return Err(refuse("a token-pool selector of no positions a pool"));
                        }
                        Some(Selector::TokenPool { d, pool, .. }) => (*pool as usize, Some(*d)),
                        Some(_) => return Err(refuse("a GQA layer selecting by another rule")),
                    };
                    let shape = (g.kv_heads, g.head_dim, idx_dim);
                    if *attn.get_or_insert(shape) != shape {
                        return Err(refuse("a GQA layer of another shape than the first's"));
                    }
                    kinds.push(Kind::Attention);
                    pools.push(pool);
                }
                Mixer::DeltaRule(d) if matches!(d.kind, DeltaKind::Gdn { .. }) => {
                    if *gdn.get_or_insert(d) != d {
                        return Err(refuse(
                            "a delta-rule layer of another shape than the first's",
                        ));
                    }
                    kinds.push(Kind::DeltaRule);
                    pools.push(0);
                }
                other => return Err(refuse(other.what())),
            }
        }
        let (recurrent, copy) = gdn.map_or((0, 0), |d| {
            gdn_bytes(
                d.v_heads as usize,
                d.k_heads as usize,
                d.d as usize,
                d.conv as usize,
                lanes,
            )
        });
        // The Qwen3.8 body's chunked prompt rule keeps a chunk's rows beside
        // each GDN layer's lane; a checkpoint copies them with the state. Its
        // verify record and the lane's stamp stay on the card.
        let program = gdn.filter(|_| program_of(spec.arch) == Some(PROGRAM));
        let carry = program.map_or(0, |d| {
            delta_carry_bytes(d.v_heads as usize, d.k_heads as usize, d.d as usize)
        });
        let record = program.map_or(0, |d| {
            let (v, s) = (d.v_heads as usize, d.d as usize);
            delta_record_bytes(v, s, DELTA_VERIFY_ROWS)
                + if lanes == 1 {
                    delta_lane_bytes(v, s, 1)
                } else {
                    0
                }
        });
        let (kv_heads, head_dim, idx_dim) = attn.unwrap_or((0, 0, None));
        Ok(KvLayout {
            kinds,
            recurrent: recurrent + carry + record,
            copy: copy + carry,
            ple: ple_site(spec)?,
            kv_heads: kv_heads as usize,
            head_dim: head_dim as usize,
            select: idx_dim.map(|d| (d as usize, pools)),
            q8: false,
            rope_row: spec.gqa().map_or(0, |g| u64::from(g.rope.dims)) * F32_BYTES,
        })
    }

    /// The same layout with an attention layer's K/V planes in q8_0's
    /// two-plane layout.
    #[must_use]
    pub const fn in_q8(mut self) -> KvLayout {
        self.q8 = true;
        self
    }

    /// [`KvLayout::in_q8`] when `q8`, else the same layout: the one place a
    /// chosen cache format reaches the plan's layout.
    #[must_use]
    pub const fn in_q8_if(self, q8: bool) -> KvLayout {
        if q8 { self.in_q8() } else { self }
    }

    /// Whether an attention layer's K/V planes are in q8_0's layout
    /// ([`KvLayout::in_q8`]): the format the load's stores allocate.
    #[must_use]
    pub const fn is_q8(&self) -> bool {
        self.q8
    }

    /// The bytes the whole trunk `spec` describes holds at `ctx_max` positions: every layer's
    /// ([`KvLayout::of_spec`]), its plain attention layers' planes in q8_0's two-plane layout
    /// when `q8` ([`KvLayout::in_q8`]). A GDN layer's stores are the same either way, so one
    /// sum serves a trunk of any layers.
    ///
    /// # Errors
    ///
    /// The layout's refusal of the description ([`KvLayout::of_spec_lanes`]).
    pub fn trunk_bytes(spec: &ModelSpec, ctx_max: u64, q8: bool) -> Result<u64, PlacementError> {
        let layout = KvLayout::of_spec(spec)?.in_q8_if(q8);
        Ok((0..spec.layers.len())
            .map(|layer| layout.layer_bytes(layer, ctx_max))
            .sum())
    }
}

/// A GDN layer's resident bytes and what one checkpoint of it copies, for
/// `lanes` lanes of state: `(recurrent, copy)`. One lane is the state in
/// place ([`recurrent_bytes`], no stamps); more add
/// [`delta_lane_bytes`] — the other lanes and a stamp a lane.
///
/// # Panics
///
/// On no lane, by name.
fn gdn_bytes(
    v_heads: usize,
    k_heads: usize,
    state: usize,
    conv: usize,
    lanes: usize,
) -> (u64, u64) {
    assert!(lanes >= 1, "gdn_bytes: a state of at least one lane");
    let copy = recurrent_bytes(v_heads, k_heads, state, conv);
    let lane_bytes = if lanes > 1 {
        delta_lane_bytes(v_heads, state, lanes)
    } else {
        0
    };
    (copy + lane_bytes, copy)
}

/// The refusal of layer `layer`, whose `what` this layout does not size.
fn unsized_layer(layer: usize, what: &str) -> PlacementError {
    PlacementError::Unimplemented(vec![Unimplemented {
        layer: Some(layer),
        feature: format!(
            "{what}: this plan sizes the stores of the GQA layers of one shape (a plane of every \
             position, or the token-pool selector's pooled keys) and the gated-delta-rule layers \
             of one shape"
        ),
    }])
}

/// The PLE site's layer and its conv ring's bytes: the one layer that carries
/// [`Extra::Ple`], with the ring the engram dimensions give it; `None` for a
/// file with no site. A second site, or a site without its dimensions, is
/// refused by layer.
fn ple_site(spec: &ModelSpec) -> Result<Option<(usize, u64)>, PlacementError> {
    let mut sites = spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.extras.contains(&Extra::Ple))
        .map(|(layer, _)| layer);
    let Some(layer) = sites.next() else {
        return Ok(None);
    };
    if let Some(second) = sites.next() {
        return Err(unsized_layer(second, "a second PLE site"));
    }
    let Some(EngramSpec {
        max_ngram,
        rule: NgramRule::Ple { conv, .. },
        ..
    }) = spec.engram
    else {
        return Err(unsized_layer(
            layer,
            "a PLE site without the PLE dimensions",
        ));
    };
    let ring = ple_ring_bytes(
        conv as usize,
        max_ngram as usize,
        streams(spec),
        hidden(spec),
    );
    Ok(Some((layer, ring)))
}

/// The hyper-connection streams of a residual row (1 without).
fn streams(spec: &ModelSpec) -> usize {
    spec.hc.map_or(1, |hc| hc.streams as usize)
}

/// The values of a residual row of one stream.
fn hidden(spec: &ModelSpec) -> usize {
    spec.hidden as usize
}

/// Card bytes one resident sequence of a qwen4exp load holds beside its
/// per-layer stores, apart from its replay table ([`PlanInputs::seq_terms`]):
/// the step's one row and the pass's [`PASS_ROWS`] rows of the four streams
/// — the arena rows a sequence's own draft walks read, held in its own
/// buffers while the arenas address whichever sequence is live — and the
/// lane word every delta launch of the sequence reads [derived: `(1 +
/// PASS_ROWS) · streams · n_embd` f32, one u32 device word]. The engine's own
/// owner of the rows term is `Body38`'s `seq38_bytes`; the gate holds the two
/// equal.
#[must_use]
pub fn slot_resident_bytes(spec: &ModelSpec) -> u64 {
    ((1 + PASS_ROWS) * streams(spec) * hidden(spec) * std::mem::size_of::<f32>() + 4) as u64
}

impl KvLayout {
    /// Attention layer `layer`'s bytes at `ctx_max` positions: its
    /// positional rows, and a selecting layer's raw key ring, which a
    /// checkpoint copies.
    fn attention_bytes(&self, layer: usize, ctx_max: u64) -> (u64, u64) {
        match &self.select {
            Some((idx_dim, pools)) => {
                let pool =
                    pools.get(layer).copied().filter(|&p| p > 0).expect(
                        "KvLayout: an attention layer's pool (of_spec_lanes refuses 0 there)",
                    );
                let ctx = usize::try_from(ctx_max).expect("KvLayout: a context of usize");
                (
                    selecting_rows_bytes(
                        self.kv_heads,
                        self.head_dim,
                        self.q8,
                        *idx_dim,
                        pool,
                        ctx,
                    ),
                    index_ring_bytes(*idx_dim, pool),
                )
            }
            None => (
                kv_plane_bytes(self.kv_heads, self.head_dim, self.q8, ctx_max),
                0,
            ),
        }
    }

    /// The host bytes one saved sequence state of `positions` positions holds
    /// of the file's own layers: every attention layer's positional rows
    /// and the current recurrent state plus saved points bounded by the
    /// context and checkpoint budget: each GDN layer's committed lane and
    /// conv ring, the PLE ring and each selecting layer's raw key ring.
    fn state_bytes(&self, positions: u64) -> u64 {
        let rows: u64 = (0..self.kinds.len())
            .filter(|&l| self.kinds[l] == Kind::Attention)
            .map(|l| self.attention_bytes(l, positions).0)
            .sum();
        let price = self.checkpoint_price();
        rows + (1 + price.saved_points(positions)) * price.point_bytes
    }

    /// One checkpoint's bytes and spacing: each GDN layer's copy, the PLE
    /// ring and each selecting layer's raw key ring, taken every
    /// [`UBATCH_PLANNED`] positions, the Qwen3.8 body's spacing. A Qwen3.5
    /// or Qwen3.6 body spaces its points by the load's ubatch instead; their
    /// files carry no PLE table, and this price reaches only the row table's
    /// host floor, which a load without one never reads.
    fn checkpoint_price(&self) -> runtime::seqstate::Price {
        let gdn = self.kinds.iter().filter(|&&k| k == Kind::DeltaRule).count() as u64;
        let rings = (0..self.kinds.len())
            .filter(|&l| self.kinds[l] == Kind::Attention)
            .map(|l| self.attention_bytes(l, 0).1)
            .sum::<u64>();
        runtime::seqstate::Price {
            point_bytes: gdn * self.copy + self.ple.map_or(0, |(_, bytes)| bytes) + rings,
            every: UBATCH_PLANNED as u32,
        }
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = match self.ple {
            Some((site, bytes)) if site == layer => bytes,
            _ => 0,
        };
        let own = match self.kinds.get(layer) {
            Some(Kind::DeltaRule) => self.recurrent,
            Some(Kind::Attention) => {
                let (rows, ring) = self.attention_bytes(layer, ctx_max);
                rows + ring
            }
            None => 0,
        };
        own + ple
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.rope_row * ctx_max
    }
}

// ------------------------------------------------------------- the MTP draft

/// How the MTP layer's program holds one of its file's tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpLoad {
    /// On the card in its file type's card format, for `role`.
    Card(Role),
    /// Decoded to f32 at load (exactly: every Q8_0 value is its f16 scale
    /// times an i8) and held as the derived weight [`mtp_widened`] names, for
    /// `role`; the file's tensor itself is not uploaded.
    Widened(Role),
    /// Read only by a selecting layer's indexer: never loaded.
    Unread,
}

/// One tensor of the MTP layer's file: its name under `blk.{index}.`, its
/// file type and shape in ggml's `ne` order (`None` for an unread tensor,
/// taken in any form), and how the program holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpTensor {
    pub stem: &'static str,
    pub form: Option<(GgmlType, Vec<u64>)>,
    pub load: MtpLoad,
}

impl MtpTensor {
    /// Its name in the draft file of layer `index`.
    #[must_use]
    pub fn name(&self, index: u32) -> String {
        format!("blk.{index}.{}", self.stem)
    }
}

/// The derived weight a [`MtpLoad::Widened`] tensor `stem` of layer `index`
/// is held as.
#[must_use]
pub fn mtp_widened(index: u32, stem: &str) -> String {
    format!("derived.blk.{index}.{stem}")
}

/// Every tensor of the dense MTP layer `d` describes, in the form its program
/// reads: attention (the query projection writing each head's gate beside it
/// when the layer has the output gate), its two hyper-connection sites (the
/// inject widened to f32, as the target's is f32), the router and the shared
/// expert, the routed stacks, the input join (`nextn.eh_proj`, `enorm`,
/// `hnorm`) and the gated-residual head (`nextn.hc_head_*`), each Q8_0 or
/// F32; the indexer's four, unread. Refused by name: a layer that is not a
/// routed GQA layer with a gated shared expert in gated hyper-connections.
pub fn mtp_tensors(d: &MtpDraft) -> Result<Vec<MtpTensor>, PlacementError> {
    let refuse = |what: &str| PlacementError::Metadata {
        key: "mtp layer".to_string(),
        detail: format!(
            "{what}; the MTP program loads a routed GQA layer with a gated shared expert in \
             gated hyper-connections"
        ),
    };
    let Mixer::Gqa(g) = &d.layer.mixer else {
        return Err(refuse("its mixer is not GQA"));
    };
    let Ffn::Moe(m) = &d.layer.ffn else {
        return Err(refuse("its FFN is dense"));
    };
    let shared = m
        .shared
        .filter(|s| s.sigmoid_gate)
        .ok_or_else(|| refuse("it has no gated shared expert"))?;
    let Some((streams, HcKind::Gated { rank })) = d.hc.map(|h| (h.streams, h.kind)) else {
        return Err(refuse("it has no gated hyper-connections"));
    };
    let u = u64::from;
    let (h, hd) = (u(d.hidden), u(g.head_dim));
    let q = u(g.heads) * hd * if g.out_gate == OutGate::Packed { 2 } else { 1 };
    let (kv, attn, wide) = (
        u(g.kv_heads) * hd,
        u(g.heads) * hd,
        u(streams) * u(d.hidden),
    );
    let (ff, e, sff, r) = (u(m.expert_ff), u(m.experts), u(shared.ff), u(rank));
    use GgmlType::{F32, Q8_0};
    use MtpLoad::{Card, Unread, Widened};
    use Role::{Attention, Head, HyperConnection, RoutedExperts, Router, SharedExpert};
    let t = |stem: &'static str, ty: GgmlType, dims: &[u64], load: MtpLoad| MtpTensor {
        stem,
        form: Some((ty, dims.to_vec())),
        load,
    };
    let unread = |stem: &'static str| MtpTensor {
        stem,
        form: None,
        load: Unread,
    };
    Ok(vec![
        t("attn_q.weight", Q8_0, &[h, q], Card(Attention)),
        t("attn_k.weight", Q8_0, &[h, kv], Card(Attention)),
        t("attn_v.weight", Q8_0, &[h, kv], Card(Attention)),
        t("attn_output.weight", Q8_0, &[attn, h], Card(Attention)),
        t("attn_q_norm.weight", F32, &[hd], Card(Attention)),
        t("attn_k_norm.weight", F32, &[hd], Card(Attention)),
        t("hc_attn_norm.weight", F32, &[wide], Card(HyperConnection)),
        t(
            "hc_attn_down.weight",
            Q8_0,
            &[wide, r],
            Card(HyperConnection),
        ),
        t("hc_attn_up.weight", Q8_0, &[r, wide], Card(HyperConnection)),
        t(
            "hc_attn_inject.weight",
            Q8_0,
            &[wide, u(streams)],
            Widened(HyperConnection),
        ),
        t("hc_ffn_norm.weight", F32, &[wide], Card(HyperConnection)),
        t(
            "hc_ffn_down.weight",
            Q8_0,
            &[wide, r],
            Card(HyperConnection),
        ),
        t("hc_ffn_up.weight", Q8_0, &[r, wide], Card(HyperConnection)),
        t(
            "hc_ffn_inject.weight",
            Q8_0,
            &[wide, u(streams)],
            Widened(HyperConnection),
        ),
        t("ffn_gate_inp.weight", F32, &[h, e], Card(Router)),
        t("ffn_gate_inp_shexp.weight", F32, &[h], Card(SharedExpert)),
        t("ffn_gate_shexp.weight", Q8_0, &[h, sff], Card(SharedExpert)),
        t("ffn_up_shexp.weight", Q8_0, &[h, sff], Card(SharedExpert)),
        t("ffn_down_shexp.weight", Q8_0, &[sff, h], Card(SharedExpert)),
        t(
            "ffn_gate_exps.weight",
            Q8_0,
            &[h, ff, e],
            Card(RoutedExperts),
        ),
        t("ffn_up_exps.weight", Q8_0, &[h, ff, e], Card(RoutedExperts)),
        t(
            "ffn_down_exps.weight",
            Q8_0,
            &[ff, h, e],
            Card(RoutedExperts),
        ),
        t(
            "nextn.eh_proj.weight",
            Q8_0,
            &[2 * h, h],
            Card(HyperConnection),
        ),
        t("nextn.enorm.weight", F32, &[h], Card(HyperConnection)),
        t("nextn.hnorm.weight", F32, &[wide], Card(HyperConnection)),
        t("nextn.hc_head_norm.weight", F32, &[wide], Card(Head)),
        t("nextn.hc_head_down.weight", Q8_0, &[wide, r], Card(Head)),
        t("nextn.hc_head_up.weight", Q8_0, &[r, wide], Card(Head)),
        unread("indexer.q_proj.weight"),
        unread("indexer.k_proj.weight"),
        unread("indexer.q_norm.weight"),
        unread("indexer.k_norm.weight"),
    ])
}

/// One tensor of the draft file as its header states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileTensor {
    pub name: String,
    pub shard: usize,
    pub ty: GgmlType,
    pub dims: Vec<u64>,
    pub nbytes: u64,
}

/// What a plan with an MTP draft is made from ([`PlanInputs::plan_mtp`]):
/// the draft's description, its file's tensors as a one-layer model on a
/// card of its own ([`MtpInputs::machine`]), and its store.
#[derive(Debug)]
pub struct MtpInputs {
    /// The layer, its head rows the list the head reads or the full head.
    pub draft: MtpDraft,
    /// [`mtp_tensors`] of `draft`.
    pub tensors: Vec<MtpTensor>,
    /// The draft's tensors in its file's order, as layer 0 of one — each
    /// [`MtpLoad::Card`] tensor in its role, every other one
    /// [`Role::Unused`] — then the widened tensors ([`mtp_widened`], f32).
    /// A row list adds none: its head reads the target's `output` in place.
    pub model: ModelTensors,
    /// One card with no bound and the target cards' granule: the draft's
    /// plan places everything on it and counts its uploads in its own heap.
    /// The bound is the target's card's, on the sum.
    pub machine: Machine,
    kv: MtpKv,
    /// The head's row → id map, one `u32` a vocabulary id whatever the head,
    /// so the plan with a row list is the plan with the full head.
    pub map_bytes: u64,
    /// The borrowed `output`'s form as [`MtpInputs::read`] read it from the
    /// target ([`mtp::head_kind`]): a Q6_K head's walk quantizes its input
    /// rows, which the plan counts beside the arena
    /// ([`mtp_head_act_bytes`]). [`MtpInputs::from_parts`] reads no target,
    /// so it records the Q8_0 form.
    pub borrowed_head: mtp::BorrowedHead,
}

impl MtpInputs {
    /// Card bytes the draft's program arena holds: [`mtp_arena_bytes`] at
    /// the draft's vocabulary and its borrowed head's activation
    /// ([`mtp_head_act_bytes`]). The one owner of the arena term: the
    /// reserve the target's expert rule spreads within
    /// ([`MtpInputs::draft_plan`]), the drafted plan's card bound and its
    /// [`MtpPlan::arena_bytes`] all take it from here.
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        mtp_arena_bytes(u64::from(self.draft.vocab))
            + mtp_head_act_bytes(self.borrowed_head, u64::from(self.draft.hidden))
    }

    /// The draft's plan at `ctx_max` positions on its own card
    /// ([`MtpInputs::machine`]), and the bytes it adds to the target's card:
    /// its granules, its store, the head's row map and its program's arena
    /// ([`MtpInputs::arena_bytes`]). Refused by name when the plan keeps a
    /// routed expert off the card.
    fn draft_plan(&self, ctx_max: u64) -> Result<(Plan<'_>, u64), PlaceError> {
        self.draft_plan_of(ctx_max, 1)
    }

    /// [`MtpInputs::draft_plan`] for a load that serves `slots` resident
    /// sequences: the draft's store counts them all.
    fn draft_plan_of(&self, ctx_max: u64, slots: usize) -> Result<(Plan<'_>, u64), PlaceError> {
        // The draft's own plan is a plan of its one layer, whose sequence is
        // its store; the rows its walks read are the target's `beside`.
        let terms = SeqTerms {
            layers: self.stores(),
            draft: None,
            beside: 0,
        };
        let kv = terms.slots_of(slots as u64);
        let draft = placement::plan_routed(
            &self.model,
            &self.machine,
            ctx_max,
            &kv,
            &PlanLevers::default(),
            RoutedRule::PerType(CardFormat::of_routed),
        )?;
        let held: u64 = draft.n_l.iter().sum();
        if held != self.model.experts {
            return Err(PlaceError::DraftExperts {
                held,
                experts: self.model.experts,
            });
        }
        let bytes = draft_card_bytes(&draft.cards[0], self.map_bytes) + self.arena_bytes();
        Ok((draft, bytes))
    }

    /// The draft layer's store of one sequence ([`MtpKv`]), over its one
    /// layer.
    fn stores(&self) -> Stores<'_> {
        Stores {
            kv: &self.kv,
            count: self.model.layers,
        }
    }

    /// The bytes the draft adds to the target's card at `ctx_max` positions
    /// ([`MtpPlan::draft_card_bytes`] and its arena): the stage card's
    /// [`MTP_RESERVE`] row of a plan (b′) machine that runs the draft
    /// ([`machine_bp`]) for one sequence. Refused as [`PlanInputs::plan_mtp`]
    /// refuses the context and the draft's plan.
    pub fn card_bytes(&self, ctx_max: u64) -> Result<u64, PlaceError> {
        self.card_bytes_of(ctx_max, 1)
    }

    /// [`MtpInputs::card_bytes`] for a load that serves `slots` resident
    /// sequences: the reserve [`PlanInputs::plan_mtp_with_slots`] checks a
    /// plan (b′) machine for, from the one owner of both,
    /// [`MtpInputs::draft_plan_of`].
    pub fn card_bytes_of(&self, ctx_max: u64, slots: usize) -> Result<u64, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        self.draft_plan_of(ctx_max, slots).map(|(_, bytes)| bytes)
    }

    /// The MTP draft `draft` read against the target `target` that `inputs`
    /// describes (`mtp::mtp_of`), its head scoring `rows`, its store in the
    /// target's cache layout ([`KvLayout::is_q8`]). Refused by name:
    /// a target whose borrowed matrices are not the draft's format
    /// ([`PlanInputs::mtp_borrows`]); a draft file that carries its own
    /// `token_embd` or `output` (this load borrows the target's), before its
    /// layer is read, so a self-contained file of any revision names that
    /// reason; what `mtp_of` refuses; a row list of another tokenizer than
    /// the target's; and what [`MtpInputs::from_parts`] refuses.
    pub fn read(
        draft: &Split,
        target: &Split,
        inputs: &PlanInputs,
        rows: HeadSet,
    ) -> Result<MtpInputs, PlaceError> {
        inputs.mtp_borrows()?;
        let borrows = crate::arch::draft::borrows(draft)?;
        let carried: Vec<&str> = [
            ("token_embd.weight", borrows.embedding),
            ("output.weight", borrows.head),
        ]
        .into_iter()
        .filter(|(_, b)| !b)
        .map(|(n, _)| n)
        .collect();
        if let Some(first) = carried.first() {
            return Err(PlacementError::Tensor {
                name: (*first).to_string(),
                detail: format!(
                    "is in the draft file ({}); this load borrows the target's token_embd and \
                     output, which its card holds: a shared draft file \
                     (MTP/{})",
                    carried.join(", "),
                    super::fixture::DRAFT_FILE,
                ),
            }
            .into());
        }
        let d = match mtp::mtp_of(draft, target, &inputs.spec)? {
            DraftSpec::Mtp(d) => *d,
            DraftSpec::Block(_) => {
                return Err(PlacementError::Metadata {
                    key: "mtp draft".to_string(),
                    detail: "reads as a block draft, not an MTP layer".to_string(),
                }
                .into());
            }
        };
        if let HeadSet::List { digest, .. } = &rows {
            let want = vocab_sha256(target)?;
            if *digest != want {
                return Err(HeadRowsError::Digest {
                    what: "the list given".to_string(),
                    file: hex(digest),
                    target: hex(&want),
                }
                .into());
            }
        }
        let file: Vec<FileTensor> = draft
            .iter_tensors()
            .map(|(shard, t)| FileTensor {
                name: t.name.clone(),
                shard,
                ty: t.ty,
                dims: t.dims.clone(),
                nbytes: t.nbytes,
            })
            .collect();
        Ok(MtpInputs::from_parts(
            MtpDraft {
                head_rows: rows,
                ..d
            },
            &file,
        )?
        .with_head(mtp::head_kind(
            target,
            inputs.spec.hidden,
            inputs.spec.vocab,
        )?)
        .with_kv_q8(inputs.kv.is_q8()))
    }

    /// The inputs of `draft` over `file`, the draft file's tensors in its
    /// order. Refused by name: a tensor of [`mtp_tensors`] the program reads
    /// absent (an unread one may be), one of another type or shape, and a
    /// tensor the file holds that is none of them.
    pub fn from_parts(draft: MtpDraft, file: &[FileTensor]) -> Result<MtpInputs, PlacementError> {
        let mut tensors = mtp_tensors(&draft)?;
        let (Mixer::Gqa(g), Ffn::Moe(m)) = (&draft.layer.mixer, &draft.layer.ffn) else {
            return Err(PlacementError::Metadata {
                key: "mtp layer".to_string(),
                detail: "is not a routed GQA layer".to_string(),
            });
        };
        let index = draft.index;
        let mut scales = Vec::new();
        for t in &mut tensors {
            let Some(f) = file.iter().find(|f| f.name == t.name(index)) else {
                continue;
            };
            let Some((ty, _)) = &mut t.form else {
                continue;
            };
            match (*ty, f.ty, t.load) {
                (GgmlType::Q8_0, GgmlType::BF16, MtpLoad::Card(role))
                    if role != Role::RoutedExperts =>
                {
                    *ty = f.ty
                }
                (GgmlType::F32, GgmlType::BF16, MtpLoad::Card(Role::Router))
                | (GgmlType::F32, GgmlType::BF16, MtpLoad::Card(Role::SharedExpert)) => *ty = f.ty,
                (GgmlType::Q8_0, GgmlType::FP8_B128, MtpLoad::Card(Role::RoutedExperts)) => {
                    *ty = f.ty
                }
                (GgmlType::Q8_0, GgmlType::F32, MtpLoad::Widened(role)) => {
                    *ty = f.ty;
                    t.load = MtpLoad::Card(role);
                }
                (GgmlType::Q8_0, GgmlType::NVFP4, MtpLoad::Card(Role::RoutedExperts)) => {
                    *ty = f.ty;
                    let stem = match t.stem {
                        "ffn_gate_exps.weight" => "ffn_gate_exps.scale",
                        "ffn_up_exps.weight" => "ffn_up_exps.scale",
                        "ffn_down_exps.weight" => "ffn_down_exps.scale",
                        _ => {
                            return Err(PlacementError::Tensor {
                                name: f.name.clone(),
                                detail: "is not a routed MTP stack".to_string(),
                            });
                        }
                    };
                    scales.push(MtpTensor {
                        stem,
                        form: Some((GgmlType::F32, vec![u64::from(m.experts)])),
                        load: MtpLoad::Card(Role::Unread),
                    });
                }
                _ => {}
            }
        }
        tensors.extend(scales);
        let layer = Some(0);
        let mut model = Vec::with_capacity(file.len() + 3);
        for f in file {
            let Some(t) = tensors.iter().find(|t| t.name(index) == f.name) else {
                return Err(PlacementError::Tensor {
                    name: f.name.clone(),
                    detail: "is in the draft file and is none of the MTP layer's".to_string(),
                });
            };
            if let Some((ty, dims)) = &t.form
                && (f.ty != *ty || f.dims != *dims)
            {
                return Err(PlacementError::Tensor {
                    name: f.name.clone(),
                    detail: format!(
                        "is {} {:?}; the MTP layer's program reads {ty} {dims:?}",
                        f.ty, f.dims
                    ),
                });
            }
            model.push(ModelTensor {
                name: f.name.clone(),
                shard: f.shard,
                layer,
                role: match t.load {
                    MtpLoad::Card(role) => role,
                    MtpLoad::Widened(_) | MtpLoad::Unread => Role::Unused,
                },
                ty: f.ty,
                dims: f.dims.clone(),
                file_bytes: f.nbytes,
                gathered_rows: None,
            });
        }
        if let Some(t) = tensors
            .iter()
            .filter(|t| t.load != MtpLoad::Unread)
            .find(|t| !file.iter().any(|f| f.name == t.name(index)))
        {
            return Err(PlacementError::Tensor {
                name: t.name(index),
                detail: "is not in the draft file".to_string(),
            });
        }
        for t in &tensors {
            if let (MtpLoad::Widened(role), Some((_, dims))) = (t.load, &t.form) {
                let values: u64 = dims.iter().product();
                model.push(ModelTensor {
                    name: mtp_widened(index, t.stem),
                    shard: 0,
                    layer,
                    role,
                    ty: GgmlType::F32,
                    dims: dims.clone(),
                    file_bytes: values * 4,
                    gathered_rows: None,
                });
            }
        }
        let map_bytes = u64::from(draft.vocab) * runtime::stores::U32_BYTES;
        let (experts, experts_used) = (u64::from(m.experts), u64::from(m.top_k));
        let kv = MtpKv {
            kv_heads: g.kv_heads as usize,
            head_dim: g.head_dim as usize,
            q8: false,
        };
        Ok(MtpInputs {
            draft,
            tensors,
            model: ModelTensors {
                tensors: model,
                layers: 1,
                experts,
                experts_used,
                bf16_words: if file
                    .iter()
                    .filter(|t| {
                        t.name.ends_with(".ffn_gate_inp.weight")
                            || t.name.ends_with(".ffn_gate_inp_shexp.weight")
                    })
                    .all(|t| t.ty == GgmlType::BF16)
                {
                    |_| true
                } else {
                    |t| t.role != Role::Router && !t.name.ends_with(".ffn_gate_inp_shexp.weight")
                },
            },
            machine: draft_machine(),
            kv,
            map_bytes,
            borrowed_head: mtp::BorrowedHead::Q8_0,
        })
    }

    /// [`MtpInputs`] with its borrowed head's form `head`: the one record
    /// [`MtpInputs::read`] adds over [`MtpInputs::from_parts`], which reads
    /// no target.
    fn with_head(mut self, head: mtp::BorrowedHead) -> MtpInputs {
        self.borrowed_head = head;
        self
    }

    /// The same inputs with the draft's store in q8_0's two-plane layout
    /// when `q8` (the target's cache layout, [`KvLayout::is_q8`]).
    fn with_kv_q8(mut self, q8: bool) -> MtpInputs {
        self.kv.q8 = q8;
        self
    }

    /// Whether the draft's store is in q8_0's layout, the target's cache
    /// layout ([`KvLayout::is_q8`]).
    #[must_use]
    pub fn kv_q8(&self) -> bool {
        self.kv.q8
    }
}

/// The draft's own machine: one card of no bound with the workstation's
/// granule, running its one layer and its head; no host tier.
fn draft_machine() -> Machine {
    Machine {
        cards: vec![Card {
            name: "mtp draft".to_string(),
            tier_leg: None,
            device: None,
            usable_bytes: u64::MAX,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: GRANULE,
            free_bytes: None,
            held_by: None,
            layers: 0..1,
            head: true,
            token_embedding: false,
            reserve_only: false,
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: 0,
                reserves: Vec::new(),
            },
            unified: None,
        }],
    }
}

/// The MTP layer's store: every position's K and V
/// (`runtime::stores::kv_plane_bytes`), in the target's cache layout (`q8`,
/// [`KvLayout::is_q8`]). The draft program reads the target's rope table
/// and allocates none of its own, so the stage rows stay the trait's
/// default.
#[derive(Clone, Copy, Debug)]
struct MtpKv {
    kv_heads: usize,
    head_dim: usize,
    q8: bool,
}

impl KvBytes for MtpKv {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer != 0 {
            return 0;
        }
        kv_plane_bytes(self.kv_heads, self.head_dim, self.q8, ctx_max)
    }
}

#[cfg(test)]
mod tests {
    use models::{
        Act, Arch, ChatSpec, DeltaKind, DeltaRule, GdnGate, Gqa, KHeadMap, LayerSpec, Residual,
        Rope, RopeMode,
    };

    use super::{
        DELTA_VERIFY_ROWS, Ffn, Kind, KvBytes, KvLayout, Mixer, ModelSpec, OutGate, PlacementError,
        delta_carry_bytes, delta_lane_bytes, delta_record_bytes, gdn_bytes, ple_ring_bytes,
        recurrent_bytes, refuse_other_program,
    };

    /// A GQA mixer of `kv_heads` key heads of 256 values, rotating `rope_dims` values of
    /// each, a plane of every position.
    fn gqa(kv_heads: u32, rope_dims: u32) -> Mixer {
        Mixer::Gqa(Gqa {
            heads: kv_heads,
            kv_heads,
            head_dim: 256,
            value_dim: 256,
            rope: Rope {
                mode: RopeMode::Imrope {
                    sections: [0, 0, 0, 0],
                },
                dims: rope_dims,
                base: 1e7,
                yarn: None,
                factors: false,
            },
            qk_norm: false,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        })
    }

    /// A GDN mixer of 2 key heads and 4 value heads of 128 values, a 4-tap conv.
    fn gdn() -> Mixer {
        Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 2,
            v_heads: 4,
            d: 128,
            conv: 4,
        })
    }

    /// The description of `arch`'s file with one dense layer for each of `mixers`, on a plain
    /// residual.
    fn plain(arch: Arch, mixers: Vec<Mixer>) -> ModelSpec {
        ModelSpec {
            arch,
            hidden: 256,
            vocab: 2,
            ctx_train: 8,
            rms_eps: 1e-6,
            layers: mixers
                .into_iter()
                .map(|mixer| LayerSpec {
                    mixer,
                    ffn: Ffn::Dense {
                        ff: 1,
                        act: Act::SwiGlu { limit: None },
                    },
                    residual: Residual::Plain,
                    extras: Vec::new(),
                })
                .collect(),
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat: ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        }
    }

    /// Qwen3.8's layers by kind (the sizes themselves are
    /// `runtime::stores`'s, pinned there): a GDN layer's state lane and its
    /// stamp, verify record, conv ring and carry, the PLE ring beside it on its layer, an attention layer's K/V and
    /// pooled keys at a context of whole pools and one past and its raw key ring.
    #[test]
    fn qwen38_layer_bytes() {
        let kv = KvLayout {
            kinds: vec![Kind::DeltaRule, Kind::DeltaRule, Kind::Attention],
            recurrent: recurrent_bytes(48, 16, 128, 4)
                + delta_lane_bytes(48, 128, 1)
                + delta_record_bytes(48, 128, DELTA_VERIFY_ROWS)
                + delta_carry_bytes(48, 16, 128),
            copy: recurrent_bytes(48, 16, 128, 4) + delta_carry_bytes(48, 16, 128),
            ple: Some((1, ple_ring_bytes(4, 3, 4, 2560))),
            kv_heads: 2,
            head_dim: 256,
            select: Some((128, vec![0, 0, 4])),
            q8: false,
            rope_row: 64 * 4,
        };
        assert_eq!(kv.layer_bytes(0, 4096), 3_596_288 + 294_916 + 530_448);
        assert_eq!(
            kv.layer_bytes(1, 4096),
            3_596_288 + 294_916 + 530_448 + 696_320
        );
        assert_eq!(kv.layer_bytes(2, 4096), 4096 * 2048 + 1024 * 256 + 2816);
        assert_eq!(kv.layer_bytes(2, 4097), 4097 * 2048 + 1025 * 256 + 2816);
        assert_eq!(kv.layer_bytes(3, 4096), 0);
    }

    /// A qwen4exp file whose PLE table is F8 rows and whose layer 2 routed
    /// stacks are NVFP4 with an expert scale each describes with every scale
    /// beside its tensor (the shared walk's rule, no row of the family's
    /// table): the table's on the PLE site, each stack's on its layer, each
    /// left in the file for the load to read by name. The file's table is
    /// read where it lies; a header's, and a checkpoint folder's whose part
    /// source makes it, from the NVMe tier at every room. FAIL-first: the
    /// located folder read its made table as the file's, on the host.
    #[test]
    fn an_f8_ple_table_and_its_scale_describe() {
        use crate::arch::qwen35moe::hparams::tests::{keys, tensors};
        use crate::arch::synthetic::{Ty, V, header_typed};
        use crate::placement::workstation::{self, HostRead};
        use crate::placement::{self, PlanLevers, Role};
        use gguf::Split;

        let table = "per_layer_token_embd.weight";
        let scale = "per_layer_token_embd.scale";
        let stack_scales = [
            "blk.2.ffn_gate_exps.scale",
            "blk.2.ffn_up_exps.scale",
            "blk.2.ffn_down_exps.scale",
        ];
        let typed: Vec<_> = tensors()
            .into_iter()
            .map(|(n, d)| {
                if n == table {
                    (n, d, Ty::F8Rows)
                } else if n.starts_with("blk.2.ffn_") && n.ends_with("_exps.weight") {
                    (n, vec![64, 1, 8], Ty::Nvfp4)
                } else if n.contains(".ffn_") && n.ends_with("_exps.weight") {
                    (n, vec![64, 1, 8], Ty::Q8_0)
                } else if n.ends_with(".attn_q.weight") {
                    // The query projection writes each of the 4 heads' 16
                    // values and its gate beside them.
                    (n, vec![64, 128], Ty::F32)
                } else {
                    (n, d, Ty::F32)
                }
            })
            .chain([(scale.to_string(), vec![1], Ty::F32)])
            .chain(
                stack_scales
                    .iter()
                    .map(|n| (n.to_string(), vec![8], Ty::F32)),
            )
            .collect();
        let pre = [("tokenizer.ggml.pre", V::Str("qwen35"))];
        let path = header_typed("q4x-f8-ple", "qwen4exp", &keys(), &pre, &typed);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let inputs = super::PlanInputs::describe_in(&split, (u64::MAX, HostRead::Given))
            .expect("the file describes");
        let layout = gguf::write::Layout::new(
            split.keys(),
            split
                .iter_tensors()
                .map(|(_, t)| gguf::write::TensorDecl {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    type_id: t.ty.as_u32(),
                    nbytes: t.nbytes,
                })
                .collect(),
        )
        .unwrap();
        let header = Split::header_only(path.clone(), &layout).unwrap();
        assert!(!split.is_header_only());
        assert!(header.is_header_only());
        let source = super::PlanInputs::describe_in(&header, (u64::MAX, HostRead::Given))
            .expect("the folder header describes");
        let made = (0..split.tensor_count())
            .map(|_| gguf::Place::Made)
            .collect();
        let located = Split::located(path.clone(), &layout, std::slice::from_ref(&path), made)
            .expect("a folder whose part source makes every tensor");
        assert!(!located.is_header_only());
        let located = super::PlanInputs::describe_in(&located, (u64::MAX, HostRead::Given))
            .expect("the located folder describes");
        assert_eq!(located.row_source, placement::RowTableSource::Parts);
        assert_eq!(inputs.row_source, placement::RowTableSource::Split);
        let machine = super::machine(
            super::A6000,
            crate::arch::qwen35moe::place::Stack::q8_blocks32(inputs.model.layers),
            8,
        );
        let levers = PlanLevers {
            card_budget_bytes: bloomery_levers::CardBudget::default(),
        };
        for ubatch in [1, 8, 512, 4096] {
            for (input, tier) in [
                (&inputs, placement::Device::Host),
                (&source, placement::Device::Nvme),
            ] {
                for plan in [
                    input.plan_with(&machine, (4096, ubatch), &levers, super::Experts::Card),
                    input.plan_nodes(&machine, (4096, ubatch), &levers, &[8]),
                    input.plan_with_slots(
                        &machine,
                        (4096, ubatch),
                        &levers,
                        super::Experts::Card,
                        2,
                    ),
                ] {
                    let plan = plan.unwrap();
                    assert_eq!(plan.row_tier().unwrap(), Some(tier));
                    if tier == placement::Device::Nvme {
                        assert_eq!(plan.host.table_bytes, 0);
                        assert_eq!(
                            plan.host.row_reserve_bytes,
                            placement::row_room(
                                &input.model,
                                runtime::stores::reader_columns(ubatch),
                                os::page_bytes().unwrap(),
                                input.row_source,
                            )
                            .unwrap()
                        );
                    }
                }
            }
        }
        // A folder's reserved part reads stay among the plan's host reserves where the table's
        // tier sets its row room, so the need is the plan's.
        let mut folder = machine;
        workstation::reserve_part_reads(&mut folder, 17);
        for input in [&inputs, &source] {
            let plan = input
                .plan_with(&folder, (4096, 512), &levers, super::Experts::Card)
                .unwrap();
            let need = workstation::HostNeed::of(&plan, 0);
            assert_eq!(need.read_ahead, 17);
            let how = workstation::Hosting {
                lock: false,
                arena_pinned: false,
            };
            need.classes(&plan, how).expect("the folder plan's need");
        }
        let _ = std::fs::remove_file(&path);
        let role = |name: &str| {
            inputs
                .model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| (t.role, t.layer))
        };
        assert_eq!(role(table), Some((Role::EngramTable, Some(1))));
        assert_eq!(role(scale), Some((Role::Unread, Some(1))));
        for s in stack_scales {
            assert_eq!(role(s), Some((Role::Unread, Some(2))), "{s}");
        }
    }

    /// A qwen4exp file's stream slot is its widest routed expert, the gate, up and down at their
    /// file bytes: three Q8_0 rows of 64 values, 68 B each. A file whose layer 2 stacks are NVFP4
    /// streams nothing (slot 0), nor does a load of every expert on the host, whose plan keeps
    /// the need of the plan without staging. The lane's staging a plan reserves for the slot
    /// (`workstation::reserve_stream_staging`) is the need's, pinned: 64 slots more than the plan
    /// without it. FAIL-first: the qwen38 seat and the CLI planned the Qwen3.8 stream's staging
    /// nowhere, then reserved it for a load of every expert on the host, which streams nothing.
    #[test]
    fn a_stream_slot_is_the_widest_routed_expert() {
        use crate::arch::qwen35moe::hparams::tests::{keys, tensors};
        use crate::arch::synthetic::{Ty, V, header_typed};
        use crate::placement::PlanLevers;
        use crate::placement::workstation::{self, HostNeed, HostRead, Hosting};

        let described = |name: &str, nvfp4: bool| {
            let typed: Vec<_> = tensors()
                .into_iter()
                .map(|(n, d)| {
                    if n.contains(".ffn_") && n.ends_with("_exps.weight") {
                        let ty = if nvfp4 && n.starts_with("blk.2.") {
                            Ty::Nvfp4
                        } else {
                            Ty::Q8_0
                        };
                        (n, vec![64, 1, 8], ty)
                    } else if n.ends_with(".attn_q.weight") {
                        (n, vec![64, 128], Ty::F32)
                    } else {
                        (n, d, Ty::F32)
                    }
                })
                .collect();
            let pre = [("tokenizer.ggml.pre", V::Str("qwen35"))];
            let path = header_typed(name, "qwen4exp", &keys(), &pre, &typed);
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let inputs = super::PlanInputs::describe_in(&split, (u64::MAX, HostRead::Given))
                .expect("the file describes");
            let _ = std::fs::remove_file(&path);
            inputs
        };
        let (card, host) = (super::Experts::Card, super::Experts::Host);
        let words = described("q4x-stream-slot", false);
        assert_eq!(words.stream_slot(card).unwrap(), 3 * 68);
        assert_eq!(
            described("q4x-stream-planes", true)
                .stream_slot(card)
                .unwrap(),
            0
        );
        let machine = super::machine(
            super::A6000,
            crate::arch::qwen35moe::place::Stack::q8_blocks32(words.model.layers),
            8,
        );
        let mut staged = machine.clone();
        workstation::reserve_stream_staging(&mut staged, words.stream_slot(card).unwrap());
        let mut host_only = machine.clone();
        workstation::reserve_stream_staging(&mut host_only, words.stream_slot(host).unwrap());
        let levers = PlanLevers {
            card_budget_bytes: bloomery_levers::CardBudget::default(),
        };
        let how = Hosting {
            lock: false,
            arena_pinned: false,
        };
        let pinned = |m, experts| {
            let plan = words.plan_with(m, (4096, 512), &levers, experts).unwrap();
            let need = HostNeed::of(&plan, 0);
            need.classes(&plan, how).expect("the plan's need").pinned
        };
        assert_eq!(pinned(&staged, card), pinned(&machine, card) + 64 * 3 * 68);
        assert_eq!(pinned(&host_only, host), pinned(&machine, host));
    }

    /// One saved sequence state of Qwen3.8's 48 layers (36 GDN, every fourth
    /// one attention, the PLE ring on layer 1) at a context of whole pools
    /// [derived]: twelve attention layers' 2,048 B a position and 256 B a
    /// pool of four, and checkpoint copies of 36 · 4,126,736 B (the committed
    /// lane, conv ring and carry, not the record a layer holds), the
    /// PLE ring's 696,320 B and twelve raw key rings of 2,816 B.
    #[test]
    fn qwen38_state_bytes() {
        let kv = KvLayout {
            kinds: (0..48)
                .map(|l| {
                    if l % 4 == 3 {
                        Kind::Attention
                    } else {
                        Kind::DeltaRule
                    }
                })
                .collect(),
            recurrent: recurrent_bytes(48, 16, 128, 4)
                + delta_lane_bytes(48, 128, 1)
                + delta_record_bytes(48, 128, DELTA_VERIFY_ROWS)
                + delta_carry_bytes(48, 16, 128),
            copy: recurrent_bytes(48, 16, 128, 4) + delta_carry_bytes(48, 16, 128),
            ple: Some((1, ple_ring_bytes(4, 3, 4, 2560))),
            kv_heads: 2,
            head_dim: 256,
            select: Some((
                128,
                (0..48).map(|l| if l % 4 == 3 { 4 } else { 0 }).collect(),
            )),
            q8: false,
            rope_row: 64 * 4,
        };
        let checkpoint = 36 * 4_126_736 + 696_320 + 12 * 2816;
        assert_eq!(kv.state_bytes(0), checkpoint);
        assert_eq!(
            kv.state_bytes(262_144),
            12 * (262_144 * 2048 + 65_536 * 256) + 9 * checkpoint
        );
        assert_eq!(kv.state_bytes(262_144), 7_987_411_008);
        assert_eq!(
            kv.state_bytes(4097),
            12 * (4097 * 2048 + 1025 * 256) + 4 * checkpoint
        );
        let rows = 12 * (16_384 * 2048 + 4096 * 256);
        // PIN(2026-10-30): the Qwen3.5 spacing clause is removed; its files have no PLE table.
        assert_eq!(kv.state_bytes(16_384), rows + 5 * checkpoint);
    }

    /// The lane term ([`gdn_bytes`], the layout's one rule): Qwen3.6's 30 GDN
    /// layers (32 value heads of 128, 16 key heads, a 4-tap conv) at one
    /// lane are the state in place with no stamp; at eight each holds seven
    /// more lanes of 2,097,152 B and a stamp a lane, so 30 layers add 420 MiB
    /// and 960 B [derived]. A checkpoint copies one lane whatever the count.
    /// The layer's bytes follow the term.
    #[test]
    fn the_lane_count_is_the_layouts_input() {
        let one = gdn_bytes(32, 16, 128, 4, 1);
        let eight = gdn_bytes(32, 16, 128, 4, 8);
        assert_eq!(
            one,
            (
                recurrent_bytes(32, 16, 128, 4),
                recurrent_bytes(32, 16, 128, 4)
            )
        );
        assert_eq!(one.0, 2_097_152 + 360_448);
        assert_eq!(eight.1, one.1, "a checkpoint copies one lane");
        assert_eq!(eight.0 - one.0, 7 * 2_097_152 + 8 * 4);
        assert_eq!(30 * (eight.0 - one.0), 420 * (1 << 20) + 960);
        let kinds = vec![Kind::DeltaRule, Kind::Attention];
        let layout = |lanes| KvLayout {
            kinds: kinds.clone(),
            recurrent: gdn_bytes(32, 16, 128, 4, lanes).0,
            copy: gdn_bytes(32, 16, 128, 4, lanes).1,
            ple: None,
            kv_heads: 2,
            head_dim: 256,
            select: None,
            q8: false,
            rope_row: 64 * 4,
        };
        assert_eq!(layout(1).layer_bytes(0, 4096), one.0);
        assert_eq!(layout(8).layer_bytes(0, 4096), eight.0);
        assert_eq!(
            layout(8).layer_bytes(1, 4096),
            layout(1).layer_bytes(1, 4096)
        );
    }

    /// The cache lever's q8_0 term ([`KvLayout::in_q8`]): an attention
    /// layer's K/V at `17/16` bytes a value, plain or selecting, a selecting
    /// layer's index keys untouched (f16 in either layout).
    #[test]
    fn q8_layer_bytes() {
        let plain = KvLayout {
            kinds: vec![Kind::Attention],
            recurrent: 0,
            copy: 0,
            ple: None,
            kv_heads: 4,
            head_dim: 128,
            select: None,
            q8: false,
            rope_row: 128 * 4,
        };
        let f16 = plain.layer_bytes(0, 4096);
        let q8 = plain.in_q8().layer_bytes(0, 4096);
        assert_eq!(f16, 4096 * 4 * 128 * 2 * 2);
        assert_eq!(q8, 4096 * 4 * 128 * 2 * 17 / 16);
        assert_eq!(q8 * 32, f16 * 17);
        let selecting = KvLayout {
            kinds: vec![Kind::Attention],
            recurrent: 0,
            copy: 0,
            ple: None,
            kv_heads: 2,
            head_dim: 256,
            select: Some((128, vec![4])),
            q8: false,
            rope_row: 64 * 4,
        };
        let sel = selecting.layer_bytes(0, 4096);
        let idx = 1024 * 128 * 2 + 11 * 128 * 2;
        assert_eq!(sel, 4096 * 2 * 256 * 2 * 2 + idx);
        assert_eq!(
            selecting.in_q8().layer_bytes(0, 4096),
            4096 * 2 * 256 * 2 * 17 / 16 + idx
        );
    }

    #[test]
    fn the_stage_rope_row_is_rope_dims_not_the_head() {
        let layout = |rope_dims| {
            KvLayout::of_spec(&plain(Arch::Qwen35Moe, vec![gqa(1, rope_dims)]))
                .expect("a GQA layer is sized")
        };
        let narrow = layout(64);
        assert_eq!(narrow.stage_position_bytes(8), 64 * 4 * 8);
        assert_eq!(narrow.in_q8().stage_position_bytes(8), 64 * 4 * 8);
        assert_eq!(layout(128).stage_position_bytes(8), 128 * 4 * 8);
    }

    /// A trunk of GQA layers alone (a dense file's) holds a plane of every position a layer and
    /// nothing past its layers: K and V of 2 heads of 256 values in f16 [derived: `4096 · 2 · 2 ·
    /// 2 · 256` B].
    #[test]
    fn an_attention_trunk_holds_planes_alone() {
        let kv = KvLayout::of_spec(&plain(Arch::Llama, vec![gqa(2, 64), gqa(2, 64)]))
            .expect("a GQA layer is sized");
        assert_eq!(kv.layer_bytes(0, 4096), 4096 * 2 * 2 * 2 * 256);
        assert_eq!(kv.layer_bytes(1, 4097), 4097 * 2 * 2 * 2 * 256);
        assert_eq!(kv.layer_bytes(2, 4096), 0);
    }

    /// The cache of a trunk of one GDN layer and one GQA layer is the sum of its layers' bytes, the
    /// attention planes in q8_0's layout when the cache runs in it and the GDN layer's state the
    /// same in both [derived: the layer's `recurrent_bytes(4, 2, 128, 4)`; the attention layer's `4096
    /// · 2 · 2 · 256` values, 2 B each in f16, 17/16 B each in q8_0].
    #[test]
    fn a_hybrid_trunk_prices_its_attention_planes_in_the_cache_format() {
        let spec = plain(Arch::Qwen35, vec![gdn(), gqa(2, 64)]);
        let ctx = 4096;
        let state = recurrent_bytes(4, 2, 128, 4);
        let f16 = KvLayout::trunk_bytes(&spec, ctx, false).expect("a hybrid trunk is sized");
        let q8 = KvLayout::trunk_bytes(&spec, ctx, true).expect("a hybrid trunk is sized");
        assert_eq!(f16, state + ctx * 2 * 2 * 256 * 2);
        assert_eq!(q8, state + ctx * 2 * 2 * 256 * 17 / 16);
        let layout = KvLayout::of_spec(&spec).expect("a hybrid trunk is sized");
        let sum = |kv: &KvLayout| (0..2).map(|l| kv.layer_bytes(l, ctx)).sum::<u64>();
        assert_eq!(f16, sum(&layout));
        assert_eq!(q8, sum(&layout.in_q8()));
    }

    /// A layer whose stores the layout does not size, and a layer of another shape than the first of
    /// its kind, are refused by their index.
    #[test]
    fn the_layout_refuses_a_layer_it_does_not_size() {
        let refused = |mixers| match KvLayout::of_spec(&plain(Arch::Qwen35Moe, mixers)) {
            Err(PlacementError::Unimplemented(list)) => list,
            other => panic!("want the layout's refusal, got {other:?}"),
        };
        let none = refused(vec![gqa(2, 64), Mixer::None]);
        assert_eq!(none.iter().map(|u| u.layer).collect::<Vec<_>>(), [Some(1)]);
        let wider = refused(vec![gqa(2, 64), gqa(2, 64), gqa(4, 64)]);
        assert_eq!(wider.iter().map(|u| u.layer).collect::<Vec<_>>(), [Some(2)]);
    }

    /// The plan is for the Qwen3.5 and Qwen3.8 bodies' files; another program's description is
    /// refused naming its architecture and program.
    #[test]
    fn a_description_of_another_program_is_refused_by_name() {
        for arch in [Arch::Qwen35Moe, Arch::Qwen35, Arch::Qwen4Exp, Arch::Llama] {
            let spec = plain(arch, vec![gqa(2, 64)]);
            assert!(refuse_other_program(&spec).is_ok(), "{arch:?}");
        }
        let err = refuse_other_program(&plain(Arch::Qwen3Moe, vec![gqa(2, 64)]))
            .expect_err("the qwen3moe body's file is not this plan's");
        let text = err.to_string();
        assert!(text.contains("is qwen3moe, run by qwen3moe"), "{text}");
        assert!(text.contains("the programs qwen35moe and qwen38"), "{text}");
    }

    /// The card rule over Qwen3.8's routed stacks ([`super::routed_rule`]) on
    /// six synthetic layers of the file's per-expert shapes (512 experts; a
    /// Q4_K gate and up of 640 rows of 2560 values, a Q5_1 down of 2560 rows
    /// of 640): layers 0, 1, 3 and 5 as most of the file's layers, layer 2 as
    /// its layer 2 (a q5_K gate and up, a q8_0 down), layer 4 as its layers 4,
    /// 30, 46 and 47 (a q8_0 down); each layer a 2560-value F32 norm first.
    mod card {
        use std::num::NonZeroU64;

        use gguf::GgmlType;

        use super::super::{PROGRAM, card_routed, routed_rule};
        use crate::placement::{
            self, Card, CardFormat, Device, Format, Host, KvBytes, Machine, ModelTensor,
            ModelTensors, Node, Plan, PlanLevers, Role,
        };

        const MIB: u64 = 1 << 20;
        const EXPERTS: u64 = 512;
        /// A Q4_K gate (or up) expert, 640 rows of ten 144-byte super-blocks.
        const GATE: u64 = 640 * 10 * 144;
        /// A Q5_1 down expert, 2560 rows of twenty 24-byte blocks.
        const DOWN: u64 = 2560 * 20 * 24;
        /// A Q5_K gate (or up) expert, 640 rows of ten 176-byte super-blocks.
        const GATE_Q5K: u64 = 640 * 10 * 176;
        /// A Q8_0 down expert, 2560 rows of twenty 34-byte blocks.
        const DOWN_Q80: u64 = 2560 * 20 * 34;
        /// An IQ4_NL down expert, 2560 rows of twenty 18-byte blocks.
        const DOWN_NL: u64 = 2560 * 20 * 18;

        /// Layer `l`'s card bytes an expert: its gate, up and down.
        fn expert_bytes(l: usize) -> u64 {
            match l {
                2 => 2 * GATE_Q5K + DOWN_Q80,
                4 => 2 * GATE + DOWN_Q80,
                _ => 2 * GATE + DOWN,
            }
        }

        struct NoKv;

        impl KvBytes for NoKv {
            fn layer_bytes(&self, _: usize, _: u64) -> u64 {
                0
            }
        }

        fn tensor(layer: usize, name: &str, role: Role, ty: GgmlType, dims: &[u64]) -> ModelTensor {
            let values: u64 = dims.iter().product();
            let (blck, size) = (
                ty.blck_size().expect("a block"),
                ty.type_size().expect("a size"),
            );
            ModelTensor {
                name: format!("blk.{layer}.{name}"),
                shard: 0,
                layer: Some(layer),
                role,
                ty,
                dims: dims.to_vec(),
                file_bytes: values / blck * size,
                gathered_rows: None,
            }
        }

        fn model() -> ModelTensors {
            let mut tensors = Vec::new();
            for l in 0..6 {
                let (gu, down) = match l {
                    2 => (GgmlType::Q5_K, GgmlType::Q8_0),
                    4 => (GgmlType::Q4_K, GgmlType::Q8_0),
                    _ => (GgmlType::Q4_K, GgmlType::Q5_1),
                };
                tensors.push(tensor(
                    l,
                    "ffn_norm.weight",
                    Role::Attention,
                    GgmlType::F32,
                    &[2560],
                ));
                for stem in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                    tensors.push(tensor(
                        l,
                        stem,
                        Role::RoutedExperts,
                        gu,
                        &[2560, 640, EXPERTS],
                    ));
                }
                tensors.push(tensor(
                    l,
                    "ffn_down_exps.weight",
                    Role::RoutedExperts,
                    down,
                    &[640, 2560, EXPERTS],
                ));
            }
            ModelTensors {
                tensors,
                layers: 6,
                experts: EXPERTS,
                experts_used: 10,
                bf16_words: |_| false,
            }
        }

        fn machine(usable: u64) -> Machine {
            Machine {
                cards: vec![Card {
                    name: "card".to_string(),
                    tier_leg: None,
                    device: None,
                    usable_bytes: usable,
                    context_bytes: 0,
                    scratch_bytes: 0,
                    margin_bytes: 0,
                    granule_bytes: NonZeroU64::new(2 * MIB).expect("2 MiB"),
                    free_bytes: None,
                    held_by: None,
                    layers: 0..6,
                    head: true,
                    token_embedding: false,
                    reserve_only: false,
                    reserves: Vec::new(),
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: u64::MAX,
                        reserves: Vec::new(),
                    },
                    unified: None,
                }],
            }
        }

        fn card_plan<'a>(
            model: &'a ModelTensors,
            machine: &'a Machine,
            levers: &PlanLevers,
            reserve: u64,
        ) -> Plan<'a> {
            placement::plan_routed_reserving(
                model,
                machine,
                4096,
                &NoKv,
                levers,
                routed_rule(PROGRAM),
                reserve,
            )
            .expect("the card plan")
        }

        /// The card format of each routed type; the Q5_1, Q8_0 and IQ4_NL downs'
        /// card bytes are their file bytes, a whole Q5_1 tensor keeps its packed
        /// format and a whole Q8_0 tensor its planes.
        #[test]
        fn routed_types_and_formats() {
            for ty in [
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::IQ2_XS,
                GgmlType::IQ2_S,
                GgmlType::IQ3_XXS,
                GgmlType::IQ1_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M,
                GgmlType::IQ4_NL,
            ] {
                assert_eq!(card_routed(ty), Some(CardFormat::KQuant), "{ty}");
            }
            // PIN(2026-10-25): iq2_s stood in this list until it had card kernels (the gate·up
            // and prompt GEMM `qwen38_gate_up` reads); tq2_0 has none.
            for ty in [
                GgmlType::Q6_K,
                GgmlType::Q5_0,
                GgmlType::Q3_K,
                GgmlType::TQ2_0,
            ] {
                assert_eq!(card_routed(ty), None, "{ty}");
            }
            for n in [1, 3, 323] {
                assert_eq!(
                    CardFormat::KQuant.resident_bytes(GgmlType::Q5_1, 640, 2560 * n),
                    Some(DOWN * n),
                    "{n} experts"
                );
                assert_eq!(
                    CardFormat::KQuant.resident_bytes(GgmlType::Q8_0, 640, 2560 * n),
                    Some(DOWN_Q80 * n),
                    "{n} experts"
                );
                assert_eq!(
                    CardFormat::KQuant.resident_bytes(GgmlType::IQ4_NL, 640, 2560 * n),
                    Some(DOWN_NL * n),
                    "{n} experts"
                );
            }
            // Rows of an odd count of blocks: the stream in whole words, a
            // whole number a row (35 blocks of 34 bytes over 3 rows: 3,570 B,
            // 893 words, 298 a row).
            assert_eq!(
                CardFormat::KQuant.buffer_bytes(GgmlType::Q8_0, 1120, 3),
                Some(vec![298 * 3 * 4])
            );
            assert_eq!(
                CardFormat::KQuant.resident_bytes(GgmlType::Q8_0, 48, 3),
                None
            );
            assert_eq!(CardFormat::of(GgmlType::Q5_1), Some(CardFormat::Q5_1));
            assert_eq!(CardFormat::of(GgmlType::Q8_0), Some(CardFormat::Q8_0Planes));
        }

        // PIN(2026-10-01): the card rule's counts on the synthetic model, every layer eligible
        // once the card reads a q5_K gate and up and a q8_0 down [derived: at six experts a layer
        // a q4_K gate and an up of 5,529,600 B take three 2 MiB granules each and a q5_1 down of
        // 7,372,800 B four, 20 MiB a layer; layer 2's q5_K gate and up of 6,758,400 B four each and
        // its q8_0 down of 10,444,800 B five, 26 MiB; layer 4's 6 + 6 + 10, 22 MiB; the six norms
        // share one granule: 130 MiB, and a seventh expert on layer 0 takes 6 MiB more, past 131
        // MiB. Under a reserve of 3 MiB the budget is 128 MiB: at five experts layer 5 takes 18
        // MiB, 128 MiB in all, and its sixth passes it. A replica of the rule reproduced the
        // four-layer counts these replaced (10 on layers 0, 1, 3, 5; 9 on layer 5 under the
        // reserve) before it gave these].
        const N_L: [u64; 6] = [6, 6, 6, 6, 6, 6];
        const N_L_RESERVED: [u64; 6] = [6, 6, 6, 6, 6, 5];
        const HEAP: u64 = 130 * MIB;
        const HEAP_RESERVED: u64 = 128 * MIB;

        /// The id prefix on every layer, the card experts reading each of its
        /// stacks; each card segment in the file's bytes as words,
        /// its bytes the prefix's file bytes; the rest on the host; deepseek2's
        /// `routed_row` still takes the packed q5_1.
        #[test]
        fn card_plan_holds_the_prefix_of_eligible_layers() {
            let (model, machine) = (model(), machine(131 * MIB));
            let plan = card_plan(&model, &machine, &PlanLevers::default(), 0);
            assert_eq!(plan.n_l, N_L);
            assert!(plan.violations().is_empty(), "{:?}", plan.violations());
            for r in &plan.rows {
                let t = &model.tensors[r.tensor];
                if t.role != Role::RoutedExperts {
                    continue;
                }
                let n = N_L[t.layer.expect("a layer")];
                let per = t.file_bytes / EXPERTS;
                let host = r.segments.last().expect("a host segment");
                assert_eq!(host.device, Device::Host, "{}", t.name);
                assert_eq!(host.resident_bytes, (EXPERTS - n) * per, "{}", t.name);
                match r.segments.as_slice() {
                    [_] => assert_eq!(n, 0, "{}", t.name),
                    [c, _] => {
                        assert_eq!(c.device, Device::Card(0), "{}", t.name);
                        assert_eq!(c.format, Format::Card(CardFormat::KQuant), "{}", t.name);
                        assert_eq!(c.resident_bytes, n * per, "{}", t.name);
                        let ids = c.experts.as_ref().map(|e| e.as_prefix());
                        assert_eq!(ids, Some(Some(n)), "{}", t.name);
                    }
                    s => panic!("{}: {} segments", t.name, s.len()),
                }
            }
            let c = &plan.cards[0];
            let expert_bytes: u64 = (0..6).map(|l| N_L[l] * expert_bytes(l)).sum();
            assert_eq!(c.expert_bytes, expert_bytes);
            assert_eq!(c.dense_bytes, 6 * 2560 * 4);
            assert_eq!(c.dense_bytes + c.expert_bytes + c.rounding_bytes, HEAP);
            let down = model
                .tensors
                .iter()
                .position(|t| t.name == "blk.0.ffn_down_exps.weight")
                .expect("layer 0's down");
            let prefix = placement::ExpertList::prefix(10).expect("a prefix");
            let row = placement::routed_row(down, &model.tensors[down], 0, prefix, &model)
                .expect("routed_row");
            assert_eq!(row.segments[0].format, Format::Card(CardFormat::Q5_1));
        }

        /// A reserve narrows the rule's budget and nothing else: the plan's
        /// card budget stays unset and its headroom is the card's usable bytes
        /// less its own heap.
        #[test]
        fn a_reserve_narrows_the_rule_alone() {
            let (model, machine) = (model(), machine(131 * MIB));
            let plan = card_plan(&model, &machine, &PlanLevers::default(), 3 * MIB);
            assert_eq!(plan.n_l, N_L_RESERVED);
            assert!(!plan.card_budget.is_set());
            let c = &plan.cards[0];
            assert_eq!(
                c.dense_bytes + c.expert_bytes + c.rounding_bytes,
                HEAP_RESERVED
            );
            assert_eq!(c.headroom_bytes, i128::from(131 * MIB - HEAP_RESERVED));
            assert!(plan.violations().is_empty(), "{:?}", plan.violations());
        }

        /// A card plan whose budget leaves no expert on the card is the
        /// host-routed plan of the same levers, field for field.
        #[test]
        fn a_card_plan_of_no_expert_is_the_host_plan() {
            let (model, machine) = (model(), machine(131 * MIB));
            let levers = PlanLevers {
                card_budget_bytes: Some(2 * MIB).into(),
            };
            let card = card_plan(&model, &machine, &levers, 0);
            let host = placement::plan_host_routed(&model, &machine, 4096, &NoKv, &levers)
                .expect("the host plan");
            let view = |p: &Plan<'_>| {
                format!(
                    "{:?}",
                    (
                        &p.rows,
                        &p.cards,
                        &p.host,
                        p.nvme_bytes,
                        &p.n_l,
                        p.ctx_max,
                        &p.card_budget
                    )
                )
            };
            assert_eq!(card.n_l, [0; 6]);
            assert_eq!(view(&card), view(&host));
        }
    }

    mod mtp {
        use gguf::GgmlType;
        use models::headpick::{HeadSet, vocab_sha256};
        use models::{
            Act, Borrows, Ffn, Gqa, HcKind, HcSpec, LayerSpec, Mixer, Moe, MtpDraft, MtpHeadNorm,
            MtpInput, MtpSource, OutGate, Residual, Rope, RopeMode, Router, Score, Shared,
        };

        use super::super::{
            FileTensor, MtpInputs, MtpLoad, mtp_arena_bytes, mtp_tensors, mtp_widened,
        };
        use crate::arch::synthetic::header;
        use crate::placement::{self, CardFormat, Device, PlanLevers, Role, RoutedRule};
        use gguf::digest::hex;

        /// The draft program's arena formula, at the file's vocabulary:
        /// the derived value the doc states, the dense flash partials' term
        /// the fixed 80 segments of a row's head.
        // PIN(2026-10-04): re-pinned for the ctx-free formula — the decode
        // flash's segment count is `flash_gqa::SEGMENTS` (80) at every cache
        // height [derived: the old pins' 47,186,092 B at ctx 3,072 counted
        // ceil(3,072/64) = 48 segments; the formula now holds 80, so the
        // arena at the file's 248,320-row head is 4 · (1,212,160 + 8·248,320
        // + 49,536·80 + 793 + 6,219,282) = 53,526,700 B, and only the
        // vocabulary moves it — one row of the head's logits, 4 · 8 B].
        // PIN(2026-10-24): the store walk holds no query rows or turned
        // queries, 4,718,592 B fewer [derived: 64 rows · (12,288 + 6,144) f32
        // · 4 B; 53,526,700 - 4,718,592 = 48,808,108 B].
        // PIN(2026-10-24): the HC bottleneck's BF16 pairs and refusal add 64 · 644 = 41,216 B.
        #[test]
        fn the_mtp_arena_formula() {
            assert_eq!(mtp_arena_bytes(248_320), 48_849_324);
            assert_eq!(
                mtp_arena_bytes(248_320),
                4 * (1_212_160 + 8 * 248_320 + 49_536 * 80 + 793 + 5_049_938)
            );
            assert_eq!(mtp_arena_bytes(248_321) - mtp_arena_bytes(248_320), 4 * 8);
        }

        /// Qwen3.8's MTP layer as `mtp_of` reads the shared file.
        pub(super) fn draft(head_rows: HeadSet) -> MtpDraft {
            MtpDraft {
                source: MtpSource::File {
                    first_shard: "/m/mtp.gguf".into(),
                    bytes: 2_775_621_632,
                    borrows: Borrows {
                        embedding: true,
                        head: true,
                    },
                },
                layer: LayerSpec {
                    mixer: Mixer::Gqa(Gqa {
                        heads: 24,
                        kv_heads: 2,
                        head_dim: 256,
                        value_dim: 256,
                        rope: Rope {
                            mode: RopeMode::Imrope {
                                sections: [11, 11, 10, 0],
                            },
                            dims: 64,
                            base: 1e7,
                            yarn: None,
                            factors: false,
                        },
                        qk_norm: true,
                        o_norm: false,
                        out_gate: OutGate::Packed,
                        select: None,
                        window: None,
                        sinks: false,
                        value_scale: None,
                    }),
                    ffn: Ffn::Moe(Moe {
                        experts: 512,
                        top_k: 10,
                        expert_ff: 640,
                        latent: None,
                        pre_exp_norm: false,
                        post_exp_norm: None,
                        act: Act::SwiGlu { limit: None },
                        router: Router {
                            score: Score::Softmax,
                            bias: false,
                            norm: models::RouterNorm::SelectedSum,
                            scale: 1.0,
                            hash: false,
                        },
                        shared: Some(Shared {
                            ff: 640,
                            act: Act::SwiGlu { limit: None },
                            sigmoid_gate: true,
                        }),
                    }),
                    residual: Residual::Hc,
                    extras: Vec::new(),
                },
                index: 48,
                hidden: 2560,
                vocab: 248_320,
                rms_eps: 1e-6,
                hc: Some(HcSpec {
                    streams: 4,
                    kind: HcKind::Gated { rank: 320 },
                }),
                input: MtpInput::Streams,
                head_norm: MtpHeadNorm::HcHead,
                head_rows,
            }
        }

        /// The shared file's 32 tensors as its header states them, in its
        /// (name) order: the statement's forms, the indexer's projections
        /// BF16 and its norms F32 [the header dump].
        pub(super) fn file() -> Vec<FileTensor> {
            let d = draft(HeadSet::Full);
            let mut out: Vec<FileTensor> = mtp_tensors(&d)
                .expect("the layer's statement")
                .into_iter()
                .map(|t| {
                    let (ty, dims) = t.form.clone().unwrap_or(match t.stem {
                        "indexer.q_proj.weight" => (GgmlType::BF16, vec![2560, 512]),
                        "indexer.k_proj.weight" => (GgmlType::BF16, vec![2560, 128]),
                        _ => (GgmlType::F32, vec![128]),
                    });
                    let values: u64 = dims.iter().product();
                    let nbytes = match ty {
                        GgmlType::Q8_0 => values / 32 * 34,
                        GgmlType::BF16 => values * 2,
                        _ => values * 4,
                    };
                    FileTensor {
                        name: t.name(d.index),
                        shard: 0,
                        ty,
                        dims,
                        nbytes,
                    }
                })
                .collect();
            out.sort_by(|a, b| a.name.cmp(&b.name));
            out
        }

        // PIN(2026-10-01): the MTP layer's card bytes [derived from the header dump's table: every
        // Q8_0 tensor but the injects in two planes of its file bytes, 2,766,827,520 B, of which
        // the routed stacks 3 x 891,289,600; the F32 tensors but the indexer's norms, 5,429,248 B;
        // the two injects widened to F32 [10240, 4], 2 x 163,840; so 98,715,648 dense and
        // 2,673,868,800 of experts, 2,772,584,448 resident; the uploads in the file's order, then
        // the injects, through the 2 MiB-granule heap: 2,785,017,856 B, rounding 12,433,408. The
        // store: 2 x 2 x 256 f16 a position. The head's map 248,320 x 4 B with the full head and
        // with a list of 40,960 rows alike: the list's rows are the target's `output`, read in
        // place].
        const DENSE: u64 = 98_715_648;
        const EXPERTS: u64 = 2_673_868_800;
        const ROUNDING: u64 = 12_433_408;
        const KV_4096: u64 = 8_388_608;
        const LIST_ROWS: usize = 40_960;
        const MAP: u64 = 993_280;

        #[test]
        fn mtp_stored_bf16_and_nvfp4_keep_their_types_and_scales() {
            let d = draft(HeadSet::Full);
            let mut stored = file();
            let mut scales = Vec::new();
            for f in &mut stored {
                if f.ty == GgmlType::Q8_0 {
                    f.ty = if f.name.contains("_exps.weight") {
                        scales.push(FileTensor {
                            name: f.name.replace(".weight", ".scale"),
                            shard: 0,
                            ty: GgmlType::F32,
                            dims: vec![512],
                            nbytes: 512 * 4,
                        });
                        GgmlType::NVFP4
                    } else if f.name.contains("_inject.weight") {
                        GgmlType::F32
                    } else {
                        GgmlType::BF16
                    };
                    f.nbytes = f.dims.iter().product::<u64>() / f.ty.blck_size().unwrap()
                        * f.ty.type_size().unwrap();
                }
            }
            stored.extend(scales);
            let m = MtpInputs::from_parts(d.clone(), &stored).unwrap();
            assert!(
                m.tensors
                    .iter()
                    .all(|t| !matches!(t.load, MtpLoad::Widened(_)))
            );
            let plan = m.draft_plan(4096).unwrap().0;
            assert!(plan.violations().is_empty(), "{:?}", plan.violations());
            assert_eq!(plan.n_l, [512]);
            for t in &m.model.tensors {
                if !t.name.contains("indexer.") {
                    assert_ne!(t.role, Role::Unused, "{}", t.name);
                }
                if t.ty == GgmlType::BF16 {
                    assert!((m.model.bf16_words)(t));
                }
            }
            let scale = stored.pop().unwrap();
            assert!(
                MtpInputs::from_parts(d.clone(), &stored)
                    .unwrap_err()
                    .to_string()
                    .contains(&scale.name)
            );
            stored.push(scale);
            let f = stored
                .iter_mut()
                .find(|f| f.name.ends_with("attn_q.weight"))
                .unwrap();
            f.ty = GgmlType::F16;
            assert!(
                MtpInputs::from_parts(d, &stored)
                    .unwrap_err()
                    .to_string()
                    .contains("attn_q.weight")
            );
        }

        #[test]
        fn mtp_router_parts_use_one_resident_type() {
            let mut mismatches = Vec::new();
            for (router, shared, raw) in [
                (GgmlType::BF16, GgmlType::BF16, true),
                (GgmlType::BF16, GgmlType::F32, false),
                (GgmlType::F32, GgmlType::BF16, false),
                (GgmlType::F32, GgmlType::F32, false),
            ] {
                let mut stored = file();
                let parts = [
                    ("blk.48.ffn_gate_inp.weight", router),
                    ("blk.48.ffn_gate_inp_shexp.weight", shared),
                    ("blk.48.attn_q.weight", GgmlType::BF16),
                ];
                for (name, ty) in parts {
                    let f = stored.iter_mut().find(|f| f.name == name).unwrap();
                    f.ty = ty;
                    f.nbytes = f.dims.iter().product::<u64>() * ty.type_size().unwrap();
                }
                let m = MtpInputs::from_parts(draft(HeadSet::Full), &stored).unwrap();
                let plan = m.draft_plan(4096).unwrap().0;
                assert!(plan.violations().is_empty(), "{:?}", plan.violations());
                for (name, ty) in parts {
                    let row = plan
                        .rows
                        .iter()
                        .find(|r| m.model.tensors[r.tensor].name == name)
                        .unwrap();
                    let t = &m.model.tensors[row.tensor];
                    assert_eq!(t.ty, ty);
                    let words = raw || name == "blk.48.attn_q.weight";
                    let (format, bytes) = match (ty, words) {
                        (GgmlType::BF16, true) => (CardFormat::Bf16Raw, 2),
                        (GgmlType::BF16, false) => (CardFormat::Bf16AsF32, 4),
                        _ => (CardFormat::F32, 4),
                    };
                    let want_bytes = t.dims.iter().product::<u64>() * bytes;
                    assert_eq!(row.segments.len(), 1);
                    let s = &row.segments[0];
                    if s.device != Device::Card(0)
                        || s.format != placement::Format::Card(format)
                        || s.resident_bytes != want_bytes
                    {
                        mismatches.push(format!(
                            "router={router:?} shared={shared:?} {name}: {s:?}, \
                             want {format:?} {want_bytes} bytes on card 0"
                        ));
                    }
                }
            }
            assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
        }

        fn list(n: usize) -> HeadSet {
            HeadSet::List {
                ids: (0..n as u32).map(|i| i * 6).collect::<Vec<_>>().into(),
                digest: [7; 32],
            }
        }

        /// The draft's plan on its own card at 4,096 positions: every tensor
        /// once, the loaded ones on the card, the unread and the widened
        /// files' tensors nowhere, all 512 experts on the card; its bytes as
        /// derived, the same with the full head and with a list of 40,960
        /// rows.
        #[test]
        fn mtp_card_bytes() {
            for rows in [HeadSet::Full, list(LIST_ROWS)] {
                let with_list = matches!(rows, HeadSet::List { .. });
                let m = MtpInputs::from_parts(draft(rows), &file()).expect("the draft's inputs");
                let plan = placement::plan_routed(
                    &m.model,
                    &m.machine,
                    4096,
                    &m.kv,
                    &PlanLevers::default(),
                    RoutedRule::PerType(CardFormat::of_routed),
                )
                .expect("the draft's plan");
                assert!(plan.violations().is_empty(), "{:?}", plan.violations());
                assert_eq!(plan.n_l, [512]);
                let c = &plan.cards[0];
                assert_eq!(
                    (c.dense_bytes, c.expert_bytes, c.rounding_bytes, c.kv_bytes),
                    (DENSE, EXPERTS, ROUNDING, KV_4096),
                    "list {with_list}"
                );
                assert_eq!(m.map_bytes, MAP, "list {with_list}");
                assert_eq!(plan.rows.len(), 32 + 2, "list {with_list}");
                for r in &plan.rows {
                    let t = &m.model.tensors[r.tensor];
                    let want = if t.role == Role::Unused {
                        Device::Unused
                    } else {
                        Device::Card(0)
                    };
                    assert!(
                        r.segments.iter().all(|s| s.device == want),
                        "{} {:?}",
                        t.name,
                        r.segments
                    );
                }
                let unused: Vec<&str> = m
                    .model
                    .tensors
                    .iter()
                    .filter(|t| t.role == Role::Unused)
                    .map(|t| t.name.as_str())
                    .collect();
                assert_eq!(unused.len(), 6, "{unused:?}");
                assert!(
                    m.model
                        .tensors
                        .iter()
                        .any(|t| t.name == mtp_widened(48, "hc_ffn_inject.weight")
                            && t.ty == GgmlType::F32)
                );
            }
        }

        /// The statement's loads: 26 on the card, the two injects widened,
        /// the indexer's four unread.
        #[test]
        fn mtp_statement_loads() {
            let t = mtp_tensors(&draft(HeadSet::Full)).expect("the statement");
            let count = |f: fn(&MtpLoad) -> bool| t.iter().filter(|x| f(&x.load)).count();
            assert_eq!(t.len(), 32);
            assert_eq!(count(|l| matches!(l, MtpLoad::Card(_))), 26);
            assert_eq!(count(|l| matches!(l, MtpLoad::Widened(_))), 2);
            assert_eq!(count(|l| matches!(l, MtpLoad::Unread)), 4);
        }

        // PIN(2026-10-22): F32 injects load directly; BF16 injects require normalization.
        #[test]
        fn a_draft_tensor_of_another_form_absent_or_foreign_is_refused() {
            let err = |f: Vec<FileTensor>| {
                MtpInputs::from_parts(draft(HeadSet::Full), &f)
                    .expect_err("refused")
                    .to_string()
            };
            let mut f = file();
            f.iter_mut()
                .find(|t| t.name == "blk.48.hc_attn_inject.weight")
                .expect("the inject")
                .ty = GgmlType::BF16;
            assert!(
                err(f).contains("tensor blk.48.hc_attn_inject.weight: is bf16 [10240, 4]"),
                "type"
            );
            let mut f = file();
            f.retain(|t| t.name != "blk.48.nextn.eh_proj.weight");
            assert!(
                err(f).contains("tensor blk.48.nextn.eh_proj.weight: is not in the draft file"),
                "absent"
            );
            let mut f = file();
            f.retain(|t| !t.name.contains("indexer."));
            assert!(
                MtpInputs::from_parts(draft(HeadSet::Full), &f).is_ok(),
                "a file without the unread indexer reads"
            );
            let mut f = file();
            f.push(FileTensor {
                name: "output.weight".to_string(),
                shard: 0,
                ty: GgmlType::Q8_0,
                dims: vec![2560, 248_320],
                nbytes: 675_430_400,
            });
            assert!(
                err(f).contains("tensor output.weight: is in the draft file and is none"),
                "foreign"
            );
        }

        /// The vocabulary digest: the synthetic header's two tokens, "a" and
        /// "b", each as its length (u64, LE) and bytes. The same constant
        /// is `tools/ref/draft-vocab.py --self-test`'s.
        #[test]
        fn vocab_digest_of_two_tokens() {
            let path = header("vocab-digest", "qwen4exp", &[], &[]);
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let got = vocab_sha256(&split).expect("the digest");
            let _ = std::fs::remove_file(&path);
            assert_eq!(
                hex(&got),
                "cf6ab613e3942391f88ed698557e1680f160bd10e88c6b668c50360c10930e2b"
            );
        }
    }

    /// The multi-slot plans ([`super::PlanInputs::plan_with_slots`],
    /// [`super::PlanInputs::plan_mtp_with_slots`]) on a synthetic file of
    /// four layers — three delta layers, the PLE site on the second, and a
    /// selecting attention layer — every routed expert on the host, beside
    /// the MTP draft of the shared file.
    mod slots {
        use std::num::NonZeroU64;

        use gguf::GgmlType;
        use models::headpick::HeadSet;
        use models::{
            Act, Arch, ChatSpec, DeltaKind, DeltaRule, EngramSpec, Extra, Ffn, GdnGate, Gqa,
            HcKind, HcSpec, KHeadMap, LayerSpec, Mixer, ModelSpec, Moe, NgramRule, OutGate,
            PoolRule, Residual, Rope, RopeMode, Router, Score, Selector, Shared,
        };

        use super::super::{
            Beside, Experts, KvLayout, MtpInputs, PlaceError, PlanInputs, ROUTE_UNIT_RESERVE,
            RouteForm, UBATCH_PLANNED, card_route_run, draft_card_bytes, machine_bp,
            route_unit_bytes, slot_resident_bytes, tier_batch_of, widen_card_route,
        };
        use super::mtp::{draft, file};
        use crate::placement::workstation::HostRead;
        use crate::placement::{
            self, Card, CardFormat, Host, KvBytes, Machine, ModelTensor, ModelTensors, Node,
            PlacementError, Plan, PlanLevers, Role, RoutedRule, UnifiedPool,
        };

        const CTX: u64 = 4096;
        const LAYERS: usize = 4;

        /// The file's description: the stores' terms of the qwen38 layer
        /// test's layout (three GDN layers and a selecting attention layer,
        /// the PLE site on layer 1, four hyper-connection streams); the rest
        /// Qwen3.8's, which no plan reads.
        fn description() -> ModelSpec {
            let rope = Rope {
                mode: RopeMode::Imrope {
                    sections: [11, 11, 10, 0],
                },
                dims: 64,
                base: 1e7,
                yarn: None,
                factors: false,
            };
            let gdn = Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::Gdn {
                    khead_map: KHeadMap::Tiled,
                    gate: GdnGate::Sigmoid,
                },
                k_heads: 16,
                v_heads: 48,
                d: 128,
                conv: 4,
            });
            let attention = Mixer::Gqa(Gqa {
                heads: 16,
                kv_heads: 2,
                head_dim: 256,
                value_dim: 256,
                rope,
                qk_norm: true,
                o_norm: false,
                out_gate: OutGate::Packed,
                select: Some(Selector::TokenPool {
                    heads: 4,
                    d: 128,
                    top_k: 2048,
                    pool: 4,
                    rule: PoolRule::Mean { rope },
                }),
                window: None,
                sinks: false,
                value_scale: None,
            });
            let layer = |l: usize, mixer: Mixer| LayerSpec {
                mixer,
                ffn: Ffn::Moe(Moe {
                    experts: 512,
                    top_k: 10,
                    expert_ff: 640,
                    latent: None,
                    pre_exp_norm: false,
                    post_exp_norm: None,
                    act: Act::SwiGlu { limit: None },
                    router: Router {
                        score: Score::Softmax,
                        bias: false,
                        norm: models::RouterNorm::SelectedSum,
                        scale: 1.0,
                        hash: false,
                    },
                    shared: Some(Shared {
                        ff: 640,
                        act: Act::SwiGlu { limit: None },
                        sigmoid_gate: true,
                    }),
                }),
                residual: Residual::Hc,
                extras: if l == 1 { vec![Extra::Ple] } else { Vec::new() },
            };
            ModelSpec {
                arch: Arch::Qwen4Exp,
                hidden: 2560,
                vocab: 248_320,
                ctx_train: 262_144,
                rms_eps: 1e-6,
                layers: vec![
                    layer(0, gdn.clone()),
                    layer(1, gdn.clone()),
                    layer(2, gdn),
                    layer(3, attention),
                ],
                mtp: Vec::new(),
                hc: Some(HcSpec {
                    streams: 4,
                    kind: HcKind::Gated { rank: 320 },
                }),
                engram: Some(EngramSpec {
                    heads: 4,
                    max_ngram: 3,
                    key_length: 256,
                    rule: NgramRule::Ple {
                        eos: 0,
                        image: None,
                        conv: 4,
                        dilation: 3,
                    },
                }),
                encoder: None,
                logit_epilogue: None,
                chat: ChatSpec {
                    pre: String::new(),
                    template: None,
                    tools: None,
                    reasoning: None,
                },
            }
        }

        /// The plan's inputs: a norm a layer for the tensors, and the file's
        /// description ([`description`]).
        fn inputs() -> PlanInputs {
            let spec = description();
            let kv = KvLayout::of_spec(&spec).expect("the description's layout");
            let tensors = (0..LAYERS)
                .map(|l| ModelTensor {
                    name: format!("blk.{l}.attn_norm.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::Attention,
                    ty: GgmlType::F32,
                    dims: vec![2560],
                    file_bytes: 2560 * 4,
                    gathered_rows: None,
                })
                .collect();
            PlanInputs {
                model: ModelTensors {
                    tensors,
                    layers: LAYERS,
                    experts: 512,
                    experts_used: 10,
                    bf16_words: |_| false,
                },
                spec,
                kv,
                room: (u64::MAX, HostRead::Given),
                row_source: placement::RowTableSource::Split,
                arena: None,
            }
        }

        /// Every positional buffer rounds independently on every resident
        /// slot, including the draft's planes; the shared spare counts once.
        /// The reserve is part of admission even with no expert stream.
        #[test]
        fn room_padding_counts_each_buffer_each_slot_and_one_spare() {
            for q8 in [false, true] {
                let mut inputs = inputs();
                inputs.kv = inputs.kv.in_q8_if(q8);
                for ctx in [16u64, 17, 4096, 16896] {
                    let values = 2 * ctx * 256;
                    let planes = if q8 {
                        vec![values, values / 16, values, values / 16]
                    } else {
                        vec![2 * values; 2]
                    };
                    let pooled = ctx.div_ceil(4) * 128 * 2;
                    for page in [65536u64, 2 << 20] {
                        for slots in [1u64, 16] {
                            for drafts in [false, true] {
                                let mut buffers = planes.clone();
                                buffers.push(pooled);
                                if drafts {
                                    buffers.extend(&planes);
                                }
                                let raw: u64 = buffers.iter().sum();
                                let whole: u64 = buffers
                                    .iter()
                                    .map(|&n| {
                                        if n.is_multiple_of(page) {
                                            n
                                        } else {
                                            n + page - n % page
                                        }
                                    })
                                    .sum();
                                let padding = inputs
                                    .room_padding(ctx, slots, drafts, page)
                                    .expect("room padding");
                                assert_eq!(
                                    raw * slots + padding,
                                    whole * slots + page,
                                    "q8 {q8}, ctx {ctx}, page {page}, slots {slots}, drafts {drafts}"
                                );
                            }
                        }
                    }
                }
            }
            let mut inputs = inputs();
            inputs.kv = inputs.kv.in_q8_if(true);
            let (ctx, slots, page) = (16, 16, 2 << 20);
            let drafts = false;
            let padding = inputs
                .room_padding(ctx, slots, drafts, page)
                .expect("padding");
            let mut machine = machine();
            inputs
                .reserve_room(&mut machine, (ctx, slots, drafts), page)
                .expect("reserve");
            let levers = PlanLevers::default();
            let plan = inputs
                .plan_with_slots(&machine, (ctx, 1), &levers, Experts::Host, slots as usize)
                .expect("admitted plan");
            assert_eq!(plan.cards[0].reserve_bytes, padding);
            let budget = plan.cards[0].bytes();
            drop(plan);
            machine.cards[0].usable_bytes = budget - 1;
            assert!(
                inputs
                    .plan_with_slots(&machine, (ctx, 1), &levers, Experts::Host, slots as usize)
                    .is_err(),
                "one byte below the whole load refuses admission"
            );
            machine.cards[0].usable_bytes = budget;
            assert!(
                inputs
                    .plan_with_slots(&machine, (ctx, 1), &levers, Experts::Host, slots as usize)
                    .is_ok(),
                "the exact whole-load budget admits"
            );
            let drafts = true;
            for (ctx, slots, page) in [
                (u64::MAX, 1, 4096),
                (16, u64::MAX, 2 << 20),
                (16, 1, u64::MAX),
            ] {
                let e = inputs
                    .room_padding(ctx, slots, drafts, page)
                    .expect_err("overflow refuses");
                assert!(
                    matches!(e, PlacementError::CardBytesPastU64 { ref card } if card == "room-backed stores")
                );
            }
        }

        /// The layout the description gives is the one `qwen38_layer_bytes` pins layer by layer: a
        /// GDN layer's state lane and stamp, verify record, conv ring and carry, the PLE ring on
        /// layer 1, the selecting layer's K/V and pooled keys at a context of whole pools and one
        /// past and its raw key ring.
        #[test]
        fn the_description_gives_the_qwen38_layer_bytes() {
            let kv = KvLayout::of_spec(&description()).expect("the description's layout");
            assert_eq!(kv.layer_bytes(0, 4096), 3_596_288 + 294_916 + 530_448);
            assert_eq!(
                kv.layer_bytes(1, 4096),
                3_596_288 + 294_916 + 530_448 + 696_320
            );
            assert_eq!(kv.layer_bytes(2, 4096), 3_596_288 + 294_916 + 530_448);
            assert_eq!(kv.layer_bytes(3, 4096), 4096 * 2048 + 1024 * 256 + 2816);
            assert_eq!(kv.layer_bytes(3, 4097), 4097 * 2048 + 1025 * 256 + 2816);
            assert_eq!(kv.layer_bytes(4, 4096), 0);
            assert_eq!(kv.stage_position_bytes(4096), 64 * 4 * 4096);
        }

        /// One card running every layer, with room for every plan here; a
        /// host with no bound.
        fn machine() -> Machine {
            Machine {
                cards: vec![Card {
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
                    layers: 0..LAYERS,
                    head: true,
                    token_embedding: false,
                    reserve_only: false,
                    reserves: Vec::new(),
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: u64::MAX,
                        reserves: Vec::new(),
                    },
                    unified: None,
                }],
            }
        }

        /// `kv`'s per-layer bytes and shadows times `slots`, each its own
        /// product: the sum the plans are held to, written apart from the
        /// placement's view.
        struct Times<'a> {
            kv: &'a dyn KvBytes,
            slots: u64,
        }

        impl KvBytes for Times<'_> {
            fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
                self.kv.layer_bytes(layer, ctx_max) * self.slots
            }

            fn shadow_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
                self.kv.shadow_bytes(layer, ctx_max) * self.slots
            }

            /// One copy for every slot: the slots read one stage's rows.
            fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
                self.kv.stage_position_bytes(ctx_max)
            }
        }

        /// The target's plan of `slots` slots: each layer's bytes times the
        /// slots, and the bytes beside the stores of every slot past the
        /// first reserved with `reserve` and in the card's kv class.
        fn target<'a>(
            inputs: &'a PlanInputs,
            machine: &'a Machine,
            slots: u64,
            reserve: u64,
        ) -> Plan<'a> {
            let rows = (slot_resident_bytes(&inputs.spec) + 120) * (slots - 1);
            let kv = Times {
                kv: &inputs.kv,
                slots,
            };
            let levers = PlanLevers::default();
            // The want steps its slots as one pass, as the plan under test
            // does: the host totals count them as the arena floor's columns.
            let mut plan = inputs
                .target_of(
                    machine,
                    (CTX, UBATCH_PLANNED),
                    &levers,
                    Experts::Host,
                    Beside::reserve(reserve + rows).slots(slots as usize),
                    &kv,
                )
                .expect("the target's plan");
            plan.cards[0].grow_kv(rows);
            plan
        }

        /// The draft's plan of `slots` slots, its store times the slots, and
        /// the bytes it adds to the target's card.
        fn draft_of(mtp: &MtpInputs, slots: u64) -> (Plan<'_>, u64) {
            let kv = Times { kv: &mtp.kv, slots };
            let plan = placement::plan_routed(
                &mtp.model,
                &mtp.machine,
                CTX,
                &kv,
                &PlanLevers::default(),
                RoutedRule::PerType(CardFormat::of_routed),
            )
            .expect("the draft's plan");
            let bytes = draft_card_bytes(&plan.cards[0], mtp.map_bytes) + mtp.arena_bytes();
            (plan, bytes)
        }

        fn view(p: &Plan<'_>) -> String {
            format!(
                "{:?}",
                (
                    &p.rows,
                    &p.cards,
                    &p.host,
                    p.nvme_bytes,
                    &p.n_l,
                    &p.tier_n_l,
                    p.ctx_max,
                    &p.card_budget
                )
            )
        }

        /// At 1, 2 and 4 slots, the plain plan and the drafted plan's target
        /// and draft are the plans of each layer's bytes times the slots and
        /// the bytes beside the stores times the slots past the first, the
        /// target plan without and with the draft
        /// ([`PlanInputs::plan_target`]) being the plain plan and the
        /// drafted plan's target; and each plan's kv classes are
        /// `SeqTerms::plan_kv` of the slots.
        #[test]
        fn slot_plans_count_every_term_per_slot() {
            let (inputs, machine) = (inputs(), machine());
            assert_eq!(
                inputs.seq_terms(None).beside,
                slot_resident_bytes(&inputs.spec) + 120
            );
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            for (n, columns) in [(1, 4), (2, 8), (4, 8)] {
                let slots = n as u64;
                let plain = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Host,
                        n,
                    )
                    .expect("the plan");
                let want = target(&inputs, &machine, slots, 0);
                assert_eq!(view(&plain), view(&want), "{n} slots");
                let got = inputs
                    .plan_target(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Host,
                        None,
                        n,
                    )
                    .expect("the target plan");
                assert_eq!(view(&got), view(&want), "{n} slots: the target plan");
                assert_eq!(
                    plain.cards[0].kv_bytes,
                    inputs.seq_terms(None).plan_kv(CTX, slots),
                    "{n} slots"
                );
                let drafted = inputs
                    .plan_mtp_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        Experts::Host,
                        n,
                    )
                    .expect("the drafted plan");
                let (draft_want, reserve) = draft_of(&mtp, slots);
                assert_eq!(view(&drafted.draft), view(&draft_want), "{n} slots");
                let mut want = target(&inputs, &machine, slots, reserve);
                want.host.arena_columns = columns;
                assert_eq!(view(&drafted.plan), view(&want), "{n} slots");
                let got = inputs
                    .plan_target(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Host,
                        Some(&mtp),
                        n,
                    )
                    .expect("the drafted target plan");
                assert_eq!(
                    view(&got),
                    view(&want),
                    "{n} slots: the drafted target plan"
                );
                assert_eq!(
                    drafted.plan.cards[0].kv_bytes + drafted.draft.cards[0].kv_bytes,
                    inputs.seq_terms(Some(&mtp)).plan_kv(CTX, slots),
                    "{n} slots"
                );
            }
        }

        /// A card's headroom and its terms: the plan's `headroom_bytes`, and
        /// the card's usable bytes (capped as the plan capped them) less
        /// every term the plan counts on it, written from the fields.
        fn headroom(p: &Plan<'_>) -> (i128, i128) {
            let t = &p.cards[0];
            let terms = t.dense_bytes
                + t.expert_bytes
                + t.rounding_bytes
                + t.kv_bytes
                + t.stage_rows_bytes
                + t.scratch_bytes
                + t.context_bytes
                + t.reserve_bytes;
            (
                t.headroom_bytes,
                i128::from(p.usable_bytes(&p.machine.cards[0])) - i128::from(terms),
            )
        }

        /// `inputs` with a routed Q4_K stack on every layer, 512 experts of
        /// 24,576 rows of 2,560 values (35,389,440 B an expert, 72.5 GB in
        /// all): more than plan (b′)'s two cards hold, so the stage card,
        /// the tier card and the host each keep some.
        fn routed(mut inputs: PlanInputs) -> PlanInputs {
            for l in 0..LAYERS {
                inputs.model.tensors.push(ModelTensor {
                    name: format!("blk.{l}.ffn_gate_exps.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty: GgmlType::Q4_K,
                    dims: vec![2560, 24_576, 512],
                    file_bytes: 512 * 24_576 * 10 * 144,
                    gathered_rows: None,
                });
            }
            inputs
        }

        #[test]
        fn drafted_slot_plans_floor_the_arena_for_every_verify_row() {
            let mut machine = machine();
            machine.cards[0].usable_bytes = 8 << 30;
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            for experts in [Experts::Host, Experts::Card] {
                for (n, columns) in [(2, 8), (1, 4), (4, 8)] {
                    let mut inputs = routed(inputs());
                    let (room, slot_set, lane_slots) = {
                        let drafted = inputs
                            .plan_mtp_with_slots(
                                &machine,
                                (CTX, UBATCH_PLANNED.min(CTX)),
                                &levers,
                                &mtp,
                                experts,
                                n,
                            )
                            .expect("the resident plan");
                        let plan = &drafted.plan;
                        let floor = placement::ArenaFloor::of(plan).expect("the arena floor");
                        assert_eq!(floor.columns, columns, "{experts:?}, {n} slots");
                        assert!(floor.slot_set > 0);
                        let lane_slots = if experts == Experts::Card { 5 } else { 0 };
                        assert_eq!(floor.lane_slots, lane_slots);
                        (
                            placement::workstation::HostNeed::of(plan, 0).bytes()
                                - plan.host.expert_bytes / 32,
                            floor.slot_set,
                            lane_slots,
                        )
                    };
                    inputs.room = (room, HostRead::Given);
                    let old_arena = 37 * slot_set;
                    let slots = 10 * columns + lane_slots;
                    let arena = slots * slot_set;
                    inputs.arena = Some(old_arena);
                    match inputs.plan_mtp_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        experts,
                        n,
                    ) {
                        Err(PlaceError::Placement(PlacementError::NvTierArenaUnder {
                            arena: got,
                            least,
                        })) => {
                            assert_eq!(got, old_arena);
                            assert_eq!(
                                (least.columns, least.slots, least.bytes),
                                (columns, slots, arena)
                            );
                        }
                        other => panic!(
                            "{experts:?}, {n} slots: not the arena's refusal: {:?}",
                            other.err()
                        ),
                    }
                    inputs.arena = Some(arena);
                    let drafted = inputs
                        .plan_mtp_with_slots(
                            &machine,
                            (CTX, UBATCH_PLANNED.min(CTX)),
                            &levers,
                            &mtp,
                            experts,
                            n,
                        )
                        .expect("the exact arena floor fits");
                    assert_eq!(drafted.plan.host.nvme_arena_bytes, arena);
                    assert!(drafted.plan.host.nvme_expert_bytes > 0);
                }
            }
        }

        /// A stage card with a named reserve (an integrated card's
        /// activation forms): the drafted plan's headroom is the card's usable
        /// bytes less every term the plan counts, the reserve and the draft's
        /// bytes included, for one slot and for several.
        #[test]
        fn drafted_headroom_counts_the_stage_cards_reserves() {
            let (inputs, mut machine) = (inputs(), machine());
            machine.cards[0].reserves.push((
                crate::arch::qwen35moe::place::ACT_FORMS_RESERVE.to_string(),
                64 << 20,
            ));
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            for n in [1usize, 2] {
                let drafted = inputs
                    .plan_mtp_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        Experts::Host,
                        n,
                    )
                    .expect("the drafted plan");
                let (_, reserve) = draft_of(&mtp, n as u64);
                let t = &drafted.plan.cards[0];
                assert_eq!(t.reserve_bytes, 64 << 20, "{n} slots");
                assert_eq!(
                    drafted.headroom_bytes,
                    i128::from(drafted.plan.usable_bytes(&machine.cards[0]))
                        - i128::from(t.bytes())
                        - i128::from(reserve),
                    "{n} slots"
                );
            }
        }

        /// A named reserve that makes the target fit a card of `u64::MAX`
        /// bytes exactly leaves no room for the draft's reserve: the sum past
        /// the bound is refused by name, for one slot and for several, and not
        /// saturated into a plan with no headroom.
        #[test]
        fn a_draft_reserve_past_u64_is_refused_by_name() {
            let (inputs, mut machine) = (inputs(), machine());
            machine.cards[0].usable_bytes = u64::MAX;
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            for n in [1usize, 2] {
                let reserve = crate::arch::qwen35moe::place::ACT_FORMS_RESERVE.to_string();
                machine.cards[0].reserves = vec![(reserve.clone(), 0)];
                let others = inputs
                    .plan_mtp_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        Experts::Host,
                        n,
                    )
                    .expect("the plan")
                    .plan
                    .cards[0]
                    .bytes();
                machine.cards[0].reserves = vec![(reserve, u64::MAX - others)];
                assert!(
                    matches!(
                        inputs.plan_mtp_with_slots(
                            &machine,
                            (CTX, UBATCH_PLANNED.min(CTX)),
                            &levers,
                            &mtp,
                            Experts::Host,
                            n
                        ),
                        Err(PlaceError::Placement(
                            PlacementError::CardBytesPastU64 { .. }
                        ))
                    ),
                    "{n} slots"
                );
            }
        }

        /// The stage card's headroom past the bytes a plan of several slots
        /// adds to its kv class after the totals (the bytes beside every
        /// slot's stores) is the card's usable bytes less every term: on the
        /// plain plan, the drafted plan on one card, and the drafted plan
        /// (b′) whose `MtpPlan::headroom_bytes` is that card's.
        #[test]
        fn slot_plans_keep_the_headroom_usable_less_every_term() {
            let (inputs, machine) = (inputs(), machine());
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            let tiered = routed(self::inputs());
            let batch = tier_batch_of(2560, 10, 4096);
            for n in [2, 4] {
                let plain = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Host,
                        n,
                    )
                    .expect("the plan");
                let (got, want) = headroom(&plain);
                assert_eq!(got, want, "{n} slots: the plain plan");
                let drafted = inputs
                    .plan_mtp_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        Experts::Host,
                        n,
                    )
                    .expect("the drafted plan");
                let (got, want) = headroom(&drafted.plan);
                assert_eq!(got, want, "{n} slots: the drafted plan on one card");
                let reserve = mtp.card_bytes_of(CTX, n).expect("the draft's card bytes");
                let bp = machine_bp(
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(LAYERS),
                    4096,
                    Some(reserve),
                    batch,
                );
                let drafted = tiered
                    .plan_mtp_with_slots(
                        &bp,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &mtp,
                        Experts::Card,
                        n,
                    )
                    .expect("the drafted plan (b′)");
                assert!(
                    drafted.plan.tier_n_l[0].iter().sum::<u64>() > 0
                        && drafted.plan.host.experts > 0,
                    "{n} slots: the tier card and the host both hold experts"
                );
                let (got, want) = headroom(&drafted.plan);
                assert_eq!(
                    (got, drafted.headroom_bytes),
                    (want, want),
                    "{n} slots: the drafted plan (b′)"
                );
            }
        }

        /// On one pool of 64 GiB, which the card's free reading and the
        /// host's figure both read, the card rule's plan of one and of two
        /// slots under the card's whole bytes leaves the host's room under
        /// the NVMe tier's floor and is refused by name; with no budget given,
        /// the card budget is lowered until the floor holds: the card keeps
        /// experts, the rest page through the tier, and the budget is marked
        /// as the pool's. The all-host rule, under its floor on a room of
        /// 32 GiB, keeps the floor's refusal: a lower card budget does not
        /// move its floor, so it is planned once.
        #[test]
        fn a_unified_pool_lowers_the_card_rules_budget_to_the_tiers_floor() {
            let pool: u64 = 64 << 30;
            let mut machine = machine();
            machine.cards[0].free_bytes = Some(pool);
            *machine.host_mut() = Host {
                usable_bytes: pool,
                reserves: vec![("pinned".to_string(), 1 << 30)],
            };
            *machine.unified_mut() = Some(UnifiedPool { bytes: pool });
            let mut inputs = routed(inputs());
            inputs.room = (pool, HostRead::Given);
            let whole = PlanLevers {
                card_budget_bytes: Some(u64::MAX).into(),
            };
            for n in [1, 2] {
                match inputs.plan_with_slots(
                    &machine,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &whole,
                    Experts::Card,
                    n,
                ) {
                    Err(PlaceError::Placement(PlacementError::HostRoomFloor {
                        room,
                        floor,
                        ..
                    })) => assert!(floor > room, "{n} slot(s): {floor} B over {room} B"),
                    other => panic!("{n} slot(s): not the floor's refusal: {:?}", other.err()),
                }
                let plan = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &PlanLevers::default(),
                        Experts::Card,
                        n,
                    )
                    .unwrap_or_else(|e| panic!("{n} slot(s): {e}"));
                let budget = plan.card_budget.every().expect("a card budget");
                assert!(
                    plan.budget_from_pool && budget < pool,
                    "{n} slot(s): the pool's budget, {budget} B"
                );
                assert!(
                    plan.host.nvme_expert_bytes > 0 && plan.n_l.iter().any(|&l| l > 0),
                    "{n} slot(s): card experts and paged experts"
                );
            }
            inputs.room = (32 << 30, HostRead::Given);
            match inputs.plan_with_slots(
                &machine,
                (CTX, UBATCH_PLANNED.min(CTX)),
                &PlanLevers::default(),
                Experts::Host,
                1,
            ) {
                Err(PlaceError::Placement(PlacementError::HostRoomFloor {
                    room, floor, ..
                })) => {
                    assert!(floor > room, "the all-host rule: {floor} B over {room} B")
                }
                other => panic!(
                    "the all-host rule: not the floor's refusal: {:?}",
                    other.err()
                ),
            }
        }

        #[test]
        fn nvme_room_is_capped_by_the_unified_pool() {
            let levers = PlanLevers {
                card_budget_bytes: bloomery_levers::CardBudget::Every(1 << 30),
            };
            let mut inputs = routed(inputs());
            for slots in [1, 2, 4] {
                let mut machine = machine();
                let ample = 1 << 40;
                machine.host_mut().usable_bytes = ample;
                *machine.unified_mut() = Some(UnifiedPool { bytes: ample });
                inputs.room = (ample, HostRead::Given);
                let whole = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Card,
                        slots,
                    )
                    .unwrap();
                let pool = placement::workstation::HostNeed::of(&whole, 0).bytes() - 1;
                machine.cards[0].free_bytes = Some(pool);
                machine.host_mut().usable_bytes = pool;
                *machine.unified_mut() = Some(UnifiedPool { bytes: pool });
                inputs.room = (pool, HostRead::Given);
                let expected = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Card,
                        slots,
                    )
                    .expect("the pool fits with NVMe");
                let expected_rows = format!("{:?}", expected.rows);
                let expected_counts = expected.n_l.clone();
                let expected_headroom = expected.host.headroom_bytes;
                inputs.room = (ample, HostRead::Given);
                let plan = inputs
                    .plan_with_slots(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        Experts::Card,
                        slots,
                    )
                    .expect("a larger host reading keeps the same pool budget");
                assert!(plan.host.nvme_expert_bytes > 0);
                assert_eq!(plan.n_l, expected_counts);
                assert_eq!(format!("{:?}", plan.rows), expected_rows);
                assert_eq!(plan.host.headroom_bytes, expected_headroom);
            }
        }

        #[test]
        fn node_nvme_room_is_capped_by_the_unified_pool() {
            let levers = PlanLevers {
                card_budget_bytes: bloomery_levers::CardBudget::Every(1 << 30),
            };
            let mut inputs = routed(inputs());
            for ranks in [1, 2, 4] {
                let owned = vec![inputs.model.experts / ranks; ranks as usize];
                let ample = 1 << 40;
                let mut m = machine();
                m.host_mut().usable_bytes = ample;
                *m.unified_mut() = Some(UnifiedPool { bytes: ample });
                for rank in 1..ranks {
                    let mut tier = m.cards[0].clone();
                    tier.name = format!("tier-{rank}");
                    tier.head = false;
                    tier.layers = 0..0;
                    m.tiers.push(tier);
                    m.nodes.push(m.nodes[0].clone());
                }
                inputs.room = (ample, HostRead::Given);
                let whole = inputs
                    .plan_nodes(&m, (CTX, UBATCH_PLANNED.min(CTX)), &levers, &owned)
                    .unwrap();
                let pool = placement::workstation::HostNeed::of(&whole, 0).bytes() - 1;
                m.cards[0].free_bytes = Some(pool);
                m.host_mut().usable_bytes = pool;
                *m.unified_mut() = Some(UnifiedPool { bytes: pool });
                inputs.room = (pool, HostRead::Given);
                let expected = inputs
                    .plan_nodes(&m, (CTX, UBATCH_PLANNED.min(CTX)), &levers, &owned)
                    .expect("the node-owned experts fit with NVMe");
                let expected_rows = format!("{:?}", expected.rows);
                inputs.room = (ample, HostRead::Given);
                let plan = inputs
                    .plan_nodes(&m, (CTX, UBATCH_PLANNED.min(CTX)), &levers, &owned)
                    .expect("each rank keeps the same local pool budget");
                assert!(plan.host.nvme_expert_bytes > 0, "{ranks} ranks");
                assert_eq!(plan.tier_n_l.len(), ranks as usize - 1);
                assert_eq!(format!("{:?}", plan.rows), expected_rows);
            }
        }

        #[test]
        fn node_assignments_precede_resident_and_shared_room_checks() {
            let mut inputs = routed(inputs());
            inputs.model.tensors.push(ModelTensor {
                name: "blk.1.ple.table.weight".into(),
                shard: 0,
                layer: Some(1),
                role: Role::EngramTable,
                ty: GgmlType::F32,
                dims: vec![256, 1 << 30],
                file_bytes: 1 << 40,
                gathered_rows: Some(3),
            });
            let mut machine = machine();
            let mut tier = machine.cards[0].clone();
            tier.name = "remote".into();
            tier.head = false;
            tier.layers = 0..0;
            tier.scratch_bytes = 1 << 40;
            machine.tiers.push(tier);
            let pool = 64 << 30;
            machine.cards[0].usable_bytes = pool;
            machine.cards[0].free_bytes = Some(pool);
            machine.nodes[0].unified = Some(UnifiedPool { bytes: pool });
            inputs.room = (pool, HostRead::Given);
            let mut roomy = machine.clone();
            roomy.cards[0].usable_bytes = 1 << 40;
            roomy.cards[0].free_bytes = None;
            roomy.nodes[0].unified = None;
            let plan = inputs
                .plan_nodes(
                    &roomy,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &PlanLevers::default(),
                    &[256, 256],
                )
                .unwrap();
            assert_eq!(plan.n_l, vec![256; LAYERS]);
            assert_eq!(plan.tier_n_l, vec![vec![256; LAYERS]]);
            // PIN(2026-10-30): 64 MiB -> 32 MiB. The delta layers' one lane and record put the
            // floor at 25,661,116 B, and 64 MiB now holds an expert on layer 0 ([1, 0, 0, 0]);
            // 32 MiB is past the floor and short of an expert, as 64 MiB was.
            for budget in [None, Some(32 << 20), Some(u64::MAX)] {
                let levers = PlanLevers {
                    card_budget_bytes: budget.into(),
                };
                for owned in [[0, 512], [256, 256], [511, 1]] {
                    let result = inputs.plan_nodes(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &levers,
                        &owned,
                    );
                    if budget == Some(u64::MAX) && owned[0] == 511 {
                        assert!(matches!(
                            result,
                            Err(PlaceError::Placement(PlacementError::HostRoomFloor { .. }))
                        ));
                        continue;
                    }
                    let plan = result.unwrap_or_else(|e| panic!("{budget:?}, {owned:?}: {e}"));
                    assert!(plan.violations().is_empty());
                    assert_eq!(plan.row_tier().unwrap(), Some(placement::Device::Nvme));
                    assert_eq!(
                        plan.host.row_reserve_bytes,
                        placement::row_room(
                            &inputs.model,
                            UBATCH_PLANNED,
                            os::page_bytes().expect("the base page reads"),
                            inputs.row_source,
                        )
                        .unwrap()
                    );
                    assert_eq!(plan.tier_n_l, vec![vec![owned[1]; LAYERS]]);
                    assert!(plan.n_l.iter().all(|&n| n <= owned[0]));
                    if budget == Some(32 << 20) {
                        assert_eq!(plan.n_l, vec![0; LAYERS]);
                    }
                    assert_eq!(plan.cards[1].kv_bytes, 0);
                    assert_eq!(plan.cards[1].expert_bytes, 0);
                    let stage = &plan.cards[0];
                    let need = stage.dense_bytes
                        + stage.expert_bytes
                        + stage.rounding_bytes
                        + stage.kv_bytes
                        + stage.stage_rows_bytes
                        + stage.scratch_bytes
                        + stage.context_bytes
                        + stage.reserve_bytes;
                    assert_eq!(plan.pool_need().unwrap().cards, need);
                    assert_eq!(plan.host.pool_card_bytes, need);
                }
            }
            assert!(
                inputs
                    .plan_nodes(
                        &machine,
                        (CTX, UBATCH_PLANNED.min(CTX)),
                        &PlanLevers {
                            card_budget_bytes: Some(0).into(),
                        },
                        &[256, 256],
                    )
                    .is_err()
            );
            let levers = PlanLevers::default();
            for owned in [&[512, 0][..], &[256, 257], &[512], &[u64::MAX, 513]] {
                assert!(
                    inputs
                        .plan_nodes(&machine, (CTX, UBATCH_PLANNED.min(CTX)), &levers, owned)
                        .is_err()
                );
            }
            for ctx in [0, super::super::KERNEL_POSITIONS + 1] {
                assert!(matches!(
                    inputs.plan_nodes(
                        &machine,
                        (ctx, UBATCH_PLANNED.min(ctx)),
                        &levers,
                        &[256, 256]
                    ),
                    Err(PlaceError::Positions { .. })
                ));
            }
            machine.tiers[0].head = true;
            assert!(matches!(
                inputs.plan_nodes(
                    &machine,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &levers,
                    &[256, 256]
                ),
                Err(PlaceError::Placement(PlacementError::Tier { .. }))
            ));
            machine.tiers[0].head = false;
            machine.tiers[0].name = machine.cards[0].name.clone();
            assert!(matches!(
                inputs.plan_nodes(
                    &machine,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &levers,
                    &[256, 256]
                ),
                Err(PlaceError::Placement(PlacementError::DeviceTwice { .. }))
            ));
            machine.tiers[0].name = "remote".into();
            inputs.room = (1, HostRead::Given);
            assert!(matches!(
                inputs.plan_nodes(
                    &machine,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &PlanLevers {
                        card_budget_bytes: Some(u64::MAX).into(),
                    },
                    &[0, 512],
                ),
                Err(PlaceError::Placement(PlacementError::HostRoomFloor { .. }))
            ));
        }

        /// The target's `token_embd` and `output`, which an MTP draft file
        /// borrows, pass as Q8_0 or BF16, and `output` besides as the Q6_K
        /// form its gemv also reads (the UD-Q3_K_XL file's `output`); the
        /// first absent or of another format is named.
        // PIN(2026-10-06): `output` widened to `mtp::head_kind`'s set (Q8_0
        // or Q6_K, one owner `mtp::output_form`) so the UD-Q3_K_XL target
        // lends its Q6_K head to the draft; `token_embd` stays Q8_0 alone —
        // the old test refused a Q6_K `output` by name.
        // PIN(2026-10-20): `token_embd` and `output` also pass as BF16, the
        // words the bf16 embedding rows and head gemvs read.
        #[test]
        fn mtp_borrows_name_a_matrix_the_head_does_not_read() {
            let matrix = |name: &str, ty: GgmlType| ModelTensor {
                name: name.to_string(),
                shard: 0,
                layer: None,
                role: Role::Head,
                ty,
                dims: vec![2560, 248_320],
                file_bytes: 0,
                gathered_rows: None,
            };
            let with = |embd: Option<GgmlType>, out: Option<GgmlType>| {
                let mut i = inputs();
                i.model.tensors.extend(
                    [("token_embd.weight", embd), ("output.weight", out)]
                        .into_iter()
                        .filter_map(|(n, ty)| ty.map(|ty| matrix(n, ty))),
                );
                i.mtp_borrows()
            };
            for out in [GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::BF16] {
                for embd in [GgmlType::Q8_0, GgmlType::BF16] {
                    assert!(with(Some(embd), Some(out)).is_ok(), "{embd} {out}");
                }
            }
            for (embd, out, name, ty) in [
                (
                    Some(GgmlType::Q8_0),
                    Some(GgmlType::Q4_K),
                    "output.weight",
                    Some(GgmlType::Q4_K),
                ),
                (
                    Some(GgmlType::Q6_K),
                    Some(GgmlType::Q8_0),
                    "token_embd.weight",
                    Some(GgmlType::Q6_K),
                ),
                (Some(GgmlType::Q8_0), None, "output.weight", None),
            ] {
                match with(embd, out) {
                    Err(PlaceError::DraftBorrow { name: n, ty: t }) => {
                        assert_eq!((n.as_str(), t), (name, ty));
                    }
                    other => panic!("{embd:?} {out:?}: {other:?}"),
                }
            }
            match with(Some(GgmlType::Q8_0), Some(GgmlType::Q4_K)) {
                Err(e) => assert_eq!(
                    e.to_string(),
                    "the target's output.weight is q4_K; the MTP draft reads the target's \
                     token_embd as Q8_0 or BF16 and its output as Q8_0, Q6_K or BF16"
                ),
                Ok(()) => panic!("a Q4_K output passed"),
            }
        }

        /// A plan (b′) machine that reserves the draft's card bytes for the
        /// slots it plans (`MtpInputs::card_bytes_of`, the serve seat's
        /// reserve) passes the drafted plan's reserve check at 1, 2 and 4
        /// slots; the one-sequence reserve (`MtpInputs::card_bytes`) is
        /// refused by name at 2, the draft's store counting both.
        #[test]
        fn bp_reserves_the_drafts_slots() {
            let inputs = inputs();
            let mtp =
                MtpInputs::from_parts(draft(HeadSet::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            let batch = tier_batch_of(2560, 10, 4096);
            for n in [1, 2, 4] {
                let reserve = mtp.card_bytes_of(CTX, n).expect("the draft's card bytes");
                let machine = machine_bp(
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(LAYERS),
                    4096,
                    Some(reserve),
                    batch,
                );
                let r = inputs.plan_mtp_with_slots(
                    &machine,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &levers,
                    &mtp,
                    Experts::Card,
                    n,
                );
                assert!(
                    !matches!(r, Err(PlaceError::DraftReserve { .. })),
                    "{n} slots: {:?}",
                    r.err()
                );
            }
            let one = mtp.card_bytes(CTX).expect("one sequence's card bytes");
            let two = mtp
                .card_bytes_of(CTX, 2)
                .expect("two sequences' card bytes");
            assert!(two > one, "{two} B for two sequences, {one} B for one");
            let machine = machine_bp(
                crate::arch::qwen35moe::place::Stack::q8_blocks32(LAYERS),
                4096,
                Some(one),
                batch,
            );
            match inputs.plan_mtp_with_slots(
                &machine,
                (CTX, UBATCH_PLANNED.min(CTX)),
                &levers,
                &mtp,
                Experts::Card,
                2,
            ) {
                Err(PlaceError::DraftReserve { got, want, .. }) => {
                    assert_eq!((got, want), (vec![one], Some(two)));
                }
                other => panic!("want the draft reserve refused: {:?}", other.err()),
            }
        }

        /// The card route's run over the synthetic file's routed stacks
        /// ([`routed`]: 35,389,440 B an expert): on a card that holds every
        /// expert the plan widens it to a ubatch of 4,096 positions,
        /// reserving 2,048 run tokens more of 170,360 B, which a load reads
        /// back as its run and arena bytes; on a 40 GiB card the experts
        /// fill, that reserve would take experts, so the card keeps runs of
        /// 2,048 tokens, as do a host-routed plan and a ubatch of one run. On
        /// one pool the card's reserves take the host's room too: at the
        /// least room that keeps a 1 GiB PLE table on the host with every
        /// expert on the card, the reserve would page the table from the
        /// NVMe tier, so the card keeps its runs; a run's bytes more room
        /// widens it. A load refuses by name a reserve of other bytes (a
        /// tiered load's run holds 2,048 · (170,360 − 102,440) B more, the
        /// join keeping the down rows and places), two of them, or one beside
        /// a ubatch of one run, even of 0 B. In the native form the reserve
        /// is 2,048 run tokens of 108,172 B (a tiered load's 108,172 −
        /// 102,440), which a load reads back in that form and refuses in the
        /// other. What the caller reads of a plan is read of the plan of the
        /// machine returned, widened or not.
        #[test]
        fn the_card_route_runs_the_whole_ubatch_where_nothing_moves_for_it() {
            let inputs = routed(inputs());
            let levers = PlanLevers::default();
            assert!(matches!(
                widen_card_route(machine(), 4096, RouteForm::Rows, |_, _| Ok(()), |_| ()),
                Err(super::super::PlaceError::MissingPlan)
            ));
            let calls = std::cell::Cell::new(0);
            assert!(matches!(
                widen_card_route(
                    machine(),
                    4096,
                    RouteForm::Rows,
                    |m, on| {
                        calls.set(calls.get() + 1);
                        if calls.get() == 1 {
                            inputs
                                .plan_with(m, (CTX, UBATCH_PLANNED), &levers, Experts::Card)
                                .map(|p| on(&p))
                        } else {
                            Ok(())
                        }
                    },
                    |_| ()
                ),
                Err(super::super::PlaceError::MissingPlan)
            ));
            for (ctx, ubatch) in [(CTX, 0), (CTX, 4097), (8, 9)] {
                assert!(matches!(
                    inputs.plan_with(&machine(), (ctx, ubatch), &levers, Experts::Card),
                    Err(super::super::PlaceError::Ubatch { .. })
                ));
            }
            let widen_in = |inputs: &PlanInputs,
                            machine: Machine,
                            ub: u64,
                            experts: Experts,
                            form: RouteForm| {
                let (machine, read) = widen_card_route(
                    machine,
                    ub,
                    form,
                    |m, on| {
                        inputs
                            .plan_with(m, (CTX, UBATCH_PLANNED.min(CTX)), &levers, experts)
                            .map(|p| on(&p))
                    },
                    |p| p.cards[0].reserve_bytes,
                )
                .expect("the plan");
                let returned = inputs
                    .plan_with(&machine, (CTX, UBATCH_PLANNED.min(CTX)), &levers, experts)
                    .expect("the plan")
                    .cards[0]
                    .reserve_bytes;
                assert_eq!(read, returned, "the read is of the machine returned");
                machine
            };
            let widen_by = |inputs: &PlanInputs, machine: Machine, ub: u64, experts: Experts| {
                widen_in(inputs, machine, ub, experts, RouteForm::Rows)
            };
            let widen = |machine: Machine, ub: u64, experts: Experts| {
                widen_by(&inputs, machine, ub, experts)
            };
            let unit = route_unit_bytes(4096, false, RouteForm::Rows);
            assert_eq!(
                (unit, route_unit_bytes(4096, true, RouteForm::Rows)),
                (348_897_280, 139_100_160)
            );
            let native = route_unit_bytes(4096, false, RouteForm::Native);
            assert_eq!(
                (native, route_unit_bytes(4096, true, RouteForm::Native)),
                (221_536_256, 11_739_136)
            );
            let whole_native = widen_in(&inputs, machine(), 4096, Experts::Card, RouteForm::Native);
            assert_eq!(
                whole_native.cards[0].reserves,
                vec![(ROUTE_UNIT_RESERVE.to_string(), native)]
            );
            assert_eq!(
                card_route_run(&whole_native.cards[0], 4096, false, RouteForm::Native)
                    .expect("the native run"),
                (4096, native)
            );
            let whole = widen(machine(), 4096, Experts::Card);
            assert_eq!(
                whole.cards[0].reserves,
                vec![(ROUTE_UNIT_RESERVE.to_string(), unit)]
            );
            let plan = inputs
                .plan_with(
                    &whole,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &levers,
                    Experts::Card,
                )
                .expect("the widened plan");
            assert_eq!(plan.n_l, vec![512; LAYERS]);
            assert_eq!(
                card_route_run(&whole.cards[0], 4096, false, RouteForm::Rows).expect("the run"),
                (4096, unit)
            );
            let mut filled = machine();
            filled.cards[0].usable_bytes = 40 << 30;
            let n_l = inputs
                .plan_with(
                    &filled,
                    (CTX, UBATCH_PLANNED.min(CTX)),
                    &levers,
                    Experts::Card,
                )
                .expect("the plan")
                .n_l;
            assert!(n_l.iter().all(|&n| 0 < n && n < 512), "{n_l:?}");
            for (m, ub, experts) in [
                (filled, 4096, Experts::Card),
                (machine(), 4096, Experts::Host),
                (machine(), 2048, Experts::Card),
            ] {
                let kept = widen(m, ub, experts);
                assert!(kept.cards[0].reserves.is_empty(), "{ub} {experts:?}");
                assert_eq!(
                    card_route_run(&kept.cards[0], ub, false, RouteForm::Rows).expect("the run"),
                    (2048, 0)
                );
            }
            let mut tabled = routed(self::inputs());
            tabled.model.tensors.push(ModelTensor {
                name: "blk.1.ple.table.weight".into(),
                shard: 0,
                layer: Some(1),
                role: Role::EngramTable,
                ty: GgmlType::F32,
                dims: vec![256, 1 << 20],
                file_bytes: 1 << 30,
                gathered_rows: Some(3),
            });
            let pool: u64 = 1 << 40;
            let mut pooled = machine();
            pooled.cards[0].free_bytes = Some(pool);
            *pooled.host_mut() = Host {
                usable_bytes: pool,
                reserves: Vec::new(),
            };
            *pooled.unified_mut() = Some(UnifiedPool { bytes: pool });
            let table_on_host = |inputs: &PlanInputs, m: &Machine| {
                inputs
                    .plan_with(m, (CTX, UBATCH_PLANNED.min(CTX)), &levers, Experts::Card)
                    .is_ok_and(|p| {
                        p.n_l == vec![512; LAYERS]
                            && p.row_tier().ok() == Some(Some(placement::Device::Host))
                    })
            };
            let (mut lo, mut hi) = (0, pool);
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                tabled.room = (mid, HostRead::Given);
                if table_on_host(&tabled, &pooled) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            tabled.room = (hi, HostRead::Given);
            assert!(table_on_host(&tabled, &pooled), "the least room: {hi} B");
            let kept = widen_by(&tabled, pooled.clone(), 4096, Experts::Card);
            assert!(kept.cards[0].reserves.is_empty(), "{hi} B of room");
            tabled.room = (hi + unit, HostRead::Given);
            let wide = widen_by(&tabled, pooled, 4096, Experts::Card);
            assert_eq!(
                wide.cards[0].reserves,
                vec![(ROUTE_UNIT_RESERVE.to_string(), unit)]
            );
            assert!(table_on_host(&tabled, &wide), "{} B of room", hi + unit);
            let refused_in = |rows: Vec<u64>, ub: u64, tiered: bool, form: RouteForm| {
                let mut card = whole.cards[0].clone();
                card.reserves = rows
                    .iter()
                    .map(|&b| (ROUTE_UNIT_RESERVE.to_string(), b))
                    .collect();
                match card_route_run(&card, ub, tiered, form) {
                    Err(PlaceError::RouteUnitReserve { got, u, want, .. }) => {
                        assert_eq!(
                            (got, u, want),
                            (rows, ub, route_unit_bytes(ub, tiered, form))
                        );
                    }
                    other => panic!("{rows:?} at {ub}: not refused: {other:?}"),
                }
            };
            let refused = |rows: Vec<u64>, ub: u64, tiered: bool| {
                refused_in(rows, ub, tiered, RouteForm::Rows)
            };
            refused(vec![unit], 4096, true);
            refused(vec![unit + 1], 4096, false);
            refused(vec![unit, unit], 4096, false);
            refused(vec![unit], 2048, false);
            refused(vec![0], 2048, false);
            refused(vec![native], 4096, false);
            refused_in(vec![unit], 4096, false, RouteForm::Native);
        }

        /// A file's card route form ([`PlanInputs::route_form`]): native on a
        /// native FP4 build over a file whose layers' routed stacks are all
        /// NVFP4, a GGUF's or a folder's rows (a stack past its layers, a
        /// draft's, not read), the rows' form on another build or with one
        /// stack of another type; and a
        /// card-experts machine counts its route in the form it is given,
        /// 2,048 run tokens of 170,360 − 108,172 B fewer at U 4,096 in the
        /// native one [derived: 127,361,024 B], a host-routed machine none.
        #[test]
        fn a_card_route_is_counted_in_its_files_form() {
            use crate::placement::workstation::RTX_3090;
            let mut inputs = routed(inputs());
            assert_eq!(inputs.route_form(true), RouteForm::Rows);
            for t in &mut inputs.model.tensors {
                if t.role == Role::RoutedExperts {
                    t.ty = GgmlType::NVFP4;
                }
            }
            let mut past = inputs.model.tensors[inputs.model.tensors.len() - 1].clone();
            past.layer = Some(LAYERS);
            past.ty = GgmlType::Q8_0;
            inputs.model.tensors.push(past);
            assert_eq!(
                (inputs.route_form(true), inputs.route_form(false)),
                (RouteForm::Native, RouteForm::Rows)
            );
            let last = inputs.model.tensors.len() - 2;
            inputs.model.tensors[last].ty = GgmlType::NVFP4_ROWS;
            assert_eq!(inputs.route_form(true), RouteForm::Native);
            inputs.model.tensors[last].ty = GgmlType::Q4_K;
            assert_eq!(inputs.route_form(true), RouteForm::Rows);
            let scratch = |experts: Experts, form: RouteForm| {
                super::super::machine_for_route(
                    RTX_3090,
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(LAYERS),
                    4096,
                    experts,
                    form,
                )
                .cards[0]
                    .scratch_bytes
            };
            assert_eq!(
                scratch(Experts::Card, RouteForm::Rows) - scratch(Experts::Card, RouteForm::Native),
                127_361_024
            );
            assert_eq!(
                scratch(Experts::Card, RouteForm::Rows),
                super::super::machine_for_experts(
                    RTX_3090,
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(LAYERS),
                    4096,
                    Experts::Card
                )
                .cards[0]
                    .scratch_bytes
            );
            assert_eq!(
                scratch(Experts::Host, RouteForm::Rows),
                scratch(Experts::Host, RouteForm::Native)
            );
        }
    }

    /// Plan (b′)'s machine ([`super::machine_bp`]): the A6000 as
    /// [`super::machine_for_experts`] lays it out under the card experts, its
    /// scratch with the tier join's beside it, the draft's bytes its one
    /// named reserve when given; the 3090 a tier with no stage and the tier's
    /// prompt batch as its one reserve; the host's reserves the
    /// workstation's and the tier's rows. The tier's prompt batch at U 4,096
    /// and 512 [derived: cols = U, 10 slots a column of 2,560 values; staging
    /// 4·cols·2560 + 4·slots + 8,592·cols + 4·slots·2560, the block route's
    /// scratch (run = min(cols, 2,048) tokens of 170,232 B, 40 B a column of
    /// ranks, two tables of S = 10·run slots at 4·S + 8·(S/64 + 512) + 8 B,
    /// two maps of 513 words: 348,980,248 at 4,096, 87,233,816 at 512), host
    /// 2·(4·slots·2560 + 4·slots)]; the tier join at U 4,096: 2,048 ·
    /// 102,440 + 4,096 · 80 = 210,124,800 B. The stage card's draft reserve is checked against what
    /// the plan needs.
    mod bp {
        use super::super::{
            Experts, MTP_RESERVE, PlaceError, card_tier_join_bytes, check_draft_reserve,
            machine_bp, machine_for_experts, tier_batch_of,
        };
        use crate::placement::workstation::{
            A6000, CONTEXT, MARGIN, OS_RESERVE, RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE,
            TIER_BATCH_RESERVE, TierBatchBytes,
        };

        // PIN(2026-10-02): the tier's block scratch is its block route's
        // with the run's down rows and the block's ranks (q38tier2b: the tier
        // packs its slots' rows by rank for the copy engine), and the A6000's
        // scratch gains the tier join's unit-wide card rows, places and
        // ranks; derivation in the doc.
        #[test]
        fn tier_batch_at_two_ubatches() {
            assert_eq!(card_tier_join_bytes(4096), 210_124_800);
            assert_eq!(card_tier_join_bytes(512), 40_960);
            assert_eq!(
                tier_batch_of(2560, 10, 4096),
                TierBatchBytes {
                    staging: 496_730_112,
                    scratch: 348_980_248,
                    host: 839_188_480,
                    leg: None,
                }
            );
            assert_eq!(
                tier_batch_of(2560, 10, 512),
                TierBatchBytes {
                    staging: 62_091_264,
                    scratch: 87_233_816,
                    host: 104_898_560,
                    leg: None,
                }
            );
        }

        #[test]
        fn the_machine_of_plan_bp() {
            let batch = tier_batch_of(2560, 10, 4096);
            for draft in [None, Some(2_843_762_444u64)] {
                let m = machine_bp(
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                    4096,
                    draft,
                    batch,
                );
                let a = machine_for_experts(
                    A6000,
                    crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                    4096,
                    Experts::Card,
                );
                let [stage] = m.cards.as_slice() else {
                    panic!("{} stage cards", m.cards.len())
                };
                let want = draft.map(|b| (MTP_RESERVE.to_string(), b));
                assert_eq!(stage.reserves, want.into_iter().collect::<Vec<_>>());
                let s = &a.cards[0];
                assert_eq!(
                    (
                        &stage.name,
                        stage.usable_bytes,
                        stage.context_bytes,
                        stage.scratch_bytes - card_tier_join_bytes(4096),
                        stage.margin_bytes,
                        stage.granule_bytes,
                        &stage.layers,
                        stage.head,
                        stage.token_embedding
                    ),
                    (
                        &s.name,
                        s.usable_bytes,
                        s.context_bytes,
                        s.scratch_bytes,
                        s.margin_bytes,
                        s.granule_bytes,
                        &s.layers,
                        s.head,
                        s.token_embedding
                    )
                );
                let [tier] = m.tiers.as_slice() else {
                    panic!("{} tiers", m.tiers.len())
                };
                assert_eq!(tier.name, RTX_3090.name);
                assert_eq!(tier.usable_bytes, RTX_3090.usable_bytes());
                assert_eq!(
                    (
                        tier.context_bytes,
                        tier.scratch_bytes,
                        tier.margin_bytes,
                        tier.layers.clone(),
                        tier.head,
                        tier.token_embedding
                    ),
                    (CONTEXT, SCRATCH, MARGIN, 0..0, false, false)
                );
                assert_eq!(
                    tier.reserves,
                    vec![(TIER_BATCH_RESERVE.to_string(), 845_710_360)]
                );
                let names: Vec<&str> = m.host().reserves.iter().map(|(n, _)| n.as_str()).collect();
                // PIN(2026-10-07): the OS reserve and the tier batch's rows;
                // was [ROW_CACHE_RESERVE, OS_RESERVE, TIER_BATCH_HOST_RESERVE].
                // workstation::host() carries the OS's reserve alone: the row
                // reserve is a plan's, set aside while its row-gathered tables
                // lie on the NVMe tier (HostTotals::row_reserve_bytes), so the
                // machine's list is its host's plus the tiers' batch rows.
                assert_eq!(names, [OS_RESERVE, TIER_BATCH_HOST_RESERVE]);
                assert_eq!(m.host().reserves[1].1, 839_188_480);
                assert_eq!(m.host().usable_bytes, a.host().usable_bytes);
            }
        }

        #[test]
        fn the_draft_reserve_is_the_plans_own() {
            let batch = tier_batch_of(2560, 10, 4096);
            let plain = machine_bp(
                crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                4096,
                None,
                batch,
            );
            let drafted = machine_bp(
                crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                4096,
                Some(100),
                batch,
            );
            assert!(check_draft_reserve(&plain, None).is_ok());
            assert!(check_draft_reserve(&drafted, Some(100)).is_ok());
            let mut twice = machine_bp(
                crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                4096,
                Some(100),
                batch,
            );
            twice.cards[0].reserves.push((MTP_RESERVE.to_string(), 100));
            for (m, want, got) in [
                (&plain, Some(100), vec![]),
                (&drafted, None, vec![100]),
                (&drafted, Some(101), vec![100]),
                (&twice, Some(100), vec![100, 100]),
            ] {
                match check_draft_reserve(m, want) {
                    Err(PlaceError::DraftReserve {
                        card,
                        got: g,
                        want: w,
                    }) => {
                        assert_eq!((card.as_str(), &g, w), (A6000.name, &got, want));
                    }
                    other => panic!("want {want:?} got {got:?}: {other:?}"),
                }
            }
        }
    }

    /// A unified card's stage reserves the activations' MMA forms for the
    /// ubatch its machine is made for, 14,772 B a position, which a load
    /// reads at that ubatch or a smaller one; a discrete card reserves none,
    /// and its load reads 0. A short row and two rows are refused by name.
    // PIN(2026-10-24): replaces a_unified_stage_reserves_the_bf16_form; the
    // reserve now holds the MXFP8 forms beside the BF16 one.
    #[test]
    fn a_unified_stage_reserves_the_act_forms() {
        use super::{ACT_FORMS_RESERVE, Experts, PlaceError, act_forms_bytes, machine_for_experts};
        use crate::placement::workstation::{A6000, CardSpec};

        let unified = CardSpec {
            integrated: true,
            free_bytes: Some(A6000.usable_bytes()),
            ..A6000
        };
        for ub in [512, 4096] {
            let stage = machine_for_experts(
                unified,
                crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                ub,
                Experts::Card,
            )
            .cards[0]
                .clone();
            assert_eq!(
                stage.reserves,
                vec![(ACT_FORMS_RESERVE.to_string(), ub * 14_772)]
            );
            for at in [ub, ub / 2] {
                assert_eq!(act_forms_bytes(&stage, at).expect("the form"), ub * 14_772);
            }
            let discrete = machine_for_experts(
                A6000,
                crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
                ub,
                Experts::Card,
            )
            .cards[0]
                .clone();
            assert!(discrete.reserves.is_empty(), "{:?}", discrete.reserves);
            assert_eq!(act_forms_bytes(&discrete, ub).expect("no form"), 0);
        }
        let mut card = machine_for_experts(
            unified,
            crate::arch::qwen35moe::place::Stack::q8_blocks32(48),
            512,
            Experts::Card,
        )
        .cards[0]
            .clone();
        for rows in [vec![512 * 14_772 - 1], vec![512 * 14_772; 2]] {
            card.reserves = rows
                .iter()
                .map(|&b| (ACT_FORMS_RESERVE.to_string(), b))
                .collect();
            match act_forms_bytes(&card, 512) {
                Err(PlaceError::ActFormsReserve { got, u, want, .. }) => {
                    assert_eq!((got, u, want), (rows, 512, 512 * 14_772));
                }
                other => panic!("{rows:?}: not refused: {other:?}"),
            }
        }
    }
}
