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
//! ([`card_routed`]: a Q4_K, Q5_K, IQ3_XXS or IQ4_XS gate and up, a Q5_1,
//! Q8_0 or IQ4_NL down, each in the file's blocks), the rest on the host. A
//! layer with a stack they do not
//! read keeps every expert on the host, each such stack named with the reason
//! ([`PlanInputs::host_only`]).
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

use std::path::Path;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::{GgmlType, Split, Value};
use models::{DraftSpec, Ffn, HcKind, HeadRows, Mixer, ModelSpec, MtpDraft, MtpSource, Role};
use sha2::{Digest, Sha256};

use super::hparams::{Hparams, Kind};
use super::{mtp, names, roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::fileio::hex;
use crate::placement::workstation::{
    self, A6000, CONTEXT, CardSpec, GRANULE, HostRead, MARGIN, RTX_3090, SCRATCH,
    TIER_BATCH_HOST_RESERVE, TIER_BATCH_RESERVE, TierBatchBytes, host_of, tier_batch_host_bytes,
    tier_batch_staging_bytes, unified_of,
};
use crate::placement::{
    self, Card, CardFormat, Host, KvBytes, Machine, ModelTensor, ModelTensors, PlacementError,
    Plan, PlanLevers, RoutedFormat, Unimplemented, Violation, checked, joined,
};

use runtime::stores::{
    DELTA_LANES, delta_lane_bytes, dense_kv_bytes, kv_q8_row_bytes, kv_row_bytes, ple_ring_bytes,
    recurrent_bytes, selecting_bytes,
};
pub use runtime::stores::{PASS_ROWS, conv_ring_rows, ple_ring_rows};

/// The positions the selector's kernels take: the QSA passes take the
/// context as a `u32` launch argument and the selected flash reads `u32`
/// cache rows, so a context past this is one those launches cannot name.
pub const KERNEL_POSITIONS: u64 = u32::MAX as u64;

/// The most positions a load of the file `hp` describes serves: its
/// `context_length`, the context it was trained at (the rope runs unscaled;
/// YaRN scaling is not built), within [`KERNEL_POSITIONS`]. `ctx` past it is
/// refused by name; else the cap. The one owner of the rule for
/// `generate_qwen3moe` and the Qwen3.8 seat.
pub fn serve_ctx(ctx: u64, hp: &Hparams) -> Result<u64, PlaceError> {
    let trained = u64::try_from(hp.n_ctx_train).unwrap_or(u64::MAX);
    let cap = trained.min(KERNEL_POSITIONS);
    if ctx > cap {
        return Err(PlaceError::PastTrained { ctx, trained, cap });
    }
    Ok(cap)
}

/// Where a plan puts the file's routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Experts {
    /// Every routed expert on the host: the plan [`PlanInputs::plan`] makes.
    Host,
    /// The expert rule on the layers whose routed stacks [`card_routed`]
    /// loads: each such layer's id prefix `[0, n_l)` on the card, as many as
    /// the card's budget holds, spread evenly over them; the rest on the host.
    Card,
}

/// The routed stacks the card experts read, in their card format, each as the
/// file stores it ([`CardFormat::KQuant`]): a Q4_K, Q5_K, IQ3_XXS or IQ4_XS
/// gate and up (`kq_gate_up_act_q4k`, `kq_gate_up_act_q5k`,
/// `iq3_xxs_gate_up_sel`, `iq4_xs_gate_up_sel`) and a Q5_1, Q8_0 or IQ4_NL
/// down (`q5_1_gemv_sel`, `q8_0_gemv_sel32`, `iq4_nl_gemv_sel32`: the down's
/// `block_q5_1`s, 24 bytes a 32 values, its `block_q8_0`s, 34 bytes, or its
/// `block_iq4_nl`s, 18 bytes, unpacked). The rule reads a stack's
/// type alone: a stack of one of these types in a place the card leg does not
/// run it in (a Q5_1, Q8_0 or IQ4_NL gate or up, a Q4_K, Q5_K, IQ3_XXS or
/// IQ4_XS down) is refused by
/// name at load (`Card38::new`, `swap38::stacks`). A stack of any other type
/// keeps its layer's experts on the host ([`host_only_reason`]).
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    match ty {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::IQ3_XXS
        | GgmlType::IQ4_XS
        | GgmlType::IQ4_NL => Some(CardFormat::KQuant),
        _ => None,
    }
}

/// Why a routed stack of type `ty` keeps its layer's experts on the host;
/// `None` for a stack [`card_routed`] loads.
#[must_use]
pub fn host_only_reason(ty: GgmlType) -> Option<String> {
    if card_routed(ty).is_some() {
        return None;
    }
    Some(format!(
        "{ty}: no card expert kernel of this program reads it (the gate and up Q4_K, Q5_K, \
         IQ3_XXS or IQ4_XS, the down Q5_1, Q8_0 or IQ4_NL)"
    ))
}

/// A routed stack whose layer keeps every expert on the host under
/// [`Experts::Card`], and why ([`host_only_reason`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostOnly {
    pub layer: usize,
    pub tensor: String,
    pub why: String,
}

/// What a plan of a qwen4exp file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's recurrent and cache bytes.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the PLE
    /// table's tier follows it ([`placement::row_table_tier`]) — on the host
    /// while the room holds the plan with the table there, else on the NVMe
    /// tier; the routed experts the room still cannot hold follow it there
    /// ([`placement::expert_nvme_tier`]). A gate that plans an arm by a room
    /// of its own sets it, read [`HostRead::Given`].
    pub room: (u64, HostRead),
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
    /// A machine with an expert tier card, under a card rule whose program
    /// hangs no tier ([`PlanInputs::plan_rule`]).
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
    /// A plan of no resident sequence slot.
    #[error("{slots} resident sequence slots: a plan serves at least one")]
    Slots { slots: usize },
    /// A target matrix an MTP draft file borrows ([`PlanInputs::mtp_borrows`])
    /// absent, or in another format than the one the draft's program reads.
    #[error(
        "the target's {name} is {}; the MTP draft reads the target's token_embd as Q8_0 and its \
         output as Q8_0 or Q6_K",
        ty.map_or("absent".to_string(), |t| t.to_string())
    )]
    DraftBorrow { name: String, ty: Option<GgmlType> },
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

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, then the host's room ([`PlanInputs::describe`]), the
    /// first that fails being the error; then the file is refused if it has
    /// a feature the engine does not run ([`PlacementError::Unimplemented`],
    /// every one listed).
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
    /// Q8_0 (the embedding-row reads its q8_0 rows), `output` one of
    /// [`mtp::head_kind`]'s forms (Q8_0 or Q6_K, one owner with it in
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
                ty == Some(GgmlType::Q8_0)
            };
            if !ok {
                return Err(PlaceError::DraftBorrow { name, ty });
            }
        }
        Ok(())
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features:
    /// the file's description and the host's room
    /// ([`workstation::host_available_read`]), a room that cannot be read
    /// refused by name — the PLE table's tier would follow a room nobody
    /// read.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, None, None)?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        let room = workstation::host_available_read().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
        })
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, every routed expert on the host and the PLE
    /// table where [`PlanInputs::room`] says ([`PlanInputs::plan_with`] under
    /// [`Experts::Host`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with(machine, ctx_max, levers, Experts::Host)
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
        ctx_max: u64,
        levers: &PlanLevers,
        experts: Experts,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        check_draft_reserve(machine, None)?;
        let plan = self.target(machine, ctx_max, levers, experts, 0)?;
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
        ctx_max: u64,
        levers: &PlanLevers,
        experts: Experts,
        slots: usize,
    ) -> Result<Plan<'a>, PlaceError> {
        if slots == 1 {
            return self.plan_with(machine, ctx_max, levers, experts);
        }
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        check_draft_reserve(machine, None)?;
        let terms = self.seq_terms(None);
        let rows = terms.plan_beside(slots as u64);
        let kv = terms.slots_of(slots as u64);
        let mut plan = self.target_of(machine, ctx_max, levers, experts, rows, &kv)?;
        // The bytes a sequence holds beside its stores ride the card's kv
        // class, out of the budget `rows` reserved above, so
        // [`Plan::violations`] holds the card's bound against them.
        plan.cards[0].grow_kv(rows);
        checked(plan).map_err(PlaceError::Broken)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the file's layers ([`KvLayout`]), the
    /// MTP draft's store on a load that carries `mtp`, and the bytes it holds
    /// beside them ([`slot_resident_bytes`]).
    fn seq_terms<'a>(&'a self, mtp: Option<&'a MtpInputs>) -> SeqTerms<'a> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.model.layers,
            },
            draft: mtp.map(MtpInputs::stores),
            beside: slot_resident_bytes(&self.hp),
        }
    }

    /// The target's plan, unchecked: every routed expert on the host, or the
    /// expert rule over [`card_routed`]'s layers within each card's budget
    /// less `reserve` ([`placement::plan_routed_reserving`]) — on a machine
    /// with expert tier cards, the stage cards first and each tier the next
    /// ids after them (plan (b′), [`machine_bp`]); then the PLE table's tier
    /// by the host's room ([`placement::row_table_tier`]). A tier machine
    /// under [`Experts::Host`] is refused by name ([`PlaceError::TierHost`]):
    /// its tier would hold nothing.
    fn target<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        experts: Experts,
        reserve: u64,
    ) -> Result<Plan<'a>, PlaceError> {
        let rule = match experts {
            Experts::Host => None,
            Experts::Card => Some(card_routed as RoutedFormat),
        };
        self.target_rule(machine, ctx_max, levers, rule, reserve)
    }

    /// [`PlanInputs::target`] over `kv`'s per-layer bytes instead of the
    /// file's own — the multiplying view a multi-slot plan counts by.
    fn target_of<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        experts: Experts,
        reserve: u64,
        kv: &dyn KvBytes,
    ) -> Result<Plan<'a>, PlaceError> {
        let rule = match experts {
            Experts::Host => None,
            Experts::Card => Some(card_routed as RoutedFormat),
        };
        self.target_rule_of(machine, ctx_max, levers, rule, reserve, kv)
    }

    /// The target's plan, unchecked, under the card rule `rule` over `kv`'s
    /// per-layer bytes: every routed expert on the host for `None`, else the
    /// expert rule over the layers whose routed stacks `rule` loads within
    /// each card's budget less `reserve`; then the PLE table's tier by the
    /// host's room ([`placement::row_table_tier`]). The all-host rule's own
    /// refusal of a tier card ([`PlaceError::TierHost`]) is wrapped to carry
    /// the program's lever and plan.
    fn target_rule_of<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        rule: Option<RoutedFormat>,
        reserve: u64,
        kv: &dyn KvBytes,
    ) -> Result<Plan<'a>, PlaceError> {
        let mut plan = match rule {
            None => placement::plan_host_routed(&self.model, machine, ctx_max, kv, levers)
                .map_err(tier_host)?,
            Some(rule) => placement::plan_routed_reserving(
                &self.model,
                machine,
                ctx_max,
                kv,
                levers,
                rule,
                reserve,
            )?,
        };
        placement::row_table_tier(&mut plan, self.room.0, UBATCH_PLANNED)?;
        placement::expert_nvme_tier(&mut plan, self.room.0)?;
        Ok(plan)
    }

    /// The target's plan, unchecked, under the card rule `rule`: every
    /// routed expert on the host for `None`, else the expert rule over the
    /// layers whose routed stacks `rule` loads within each card's budget
    /// less `reserve`; then the PLE table's tier by the host's room.
    fn target_rule<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        rule: Option<RoutedFormat>,
        reserve: u64,
    ) -> Result<Plan<'a>, PlaceError> {
        self.target_rule_of(machine, ctx_max, levers, rule, reserve, &self.kv)
    }
    /// `levers`, the routed experts by the expert rule over the layers whose
    /// three routed stacks `rule` loads — the card rule of the program that
    /// runs the plan, as [`card_routed`] is Qwen3.8's and
    /// `qwen3moe::place::card_routed` the qwen3moe program's for a qwen35moe
    /// file — the rest on the host. Refused as [`PlanInputs::plan_with`]
    /// refuses; a machine with an expert tier card is refused by name.
    pub fn plan_rule<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        rule: RoutedFormat,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        if let Some(tier) = machine.tiers.first() {
            return Err(PlaceError::Tier {
                tier: tier.name.clone(),
            });
        }
        check_draft_reserve(machine, None)?;
        let plan = self.target_rule(machine, ctx_max, levers, Some(rule), 0)?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// Every routed stack whose layer keeps all its experts on the host
    /// under [`Experts::Card`] — a stack of a type [`card_routed`] does not
    /// load — with the reason, in the file's tensor order.
    #[must_use]
    pub fn host_only(&self) -> Vec<HostOnly> {
        self.model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
            .filter_map(|t| {
                let why = host_only_reason(t.ty)?;
                Some(HostOnly {
                    layer: t.layer?,
                    tensor: t.name.clone(),
                    why,
                })
            })
            .collect()
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
        ctx_max: u64,
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        self.plan_mtp_with(machine, ctx_max, levers, mtp, Experts::Host)
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
        ctx_max: u64,
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
        experts: Experts,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::DraftCards {
                cards: machine.cards.len(),
            });
        };
        let (draft, reserve) = mtp.draft_plan(ctx_max)?;
        let arena = mtp.arena_bytes();
        if !machine.tiers.is_empty() {
            check_draft_reserve(machine, Some(reserve))?;
            let plan = self.target(machine, ctx_max, levers, experts, 0)?;
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
        let plan = self.target(machine, ctx_max, levers, experts, reserve)?;
        let t = &plan.cards[0];
        // The draft's term is the reserve the expert rule spread within, so
        // the bound and the budget count one number.
        let total = [
            t.dense_bytes,
            t.expert_bytes,
            t.rounding_bytes,
            t.kv_bytes,
            t.scratch_bytes,
            t.context_bytes,
            reserve,
        ]
        .iter()
        .sum::<u64>();
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
        ctx_max: u64,
        levers: &PlanLevers,
        mtp: &'a MtpInputs,
        experts: Experts,
        slots: usize,
    ) -> Result<MtpPlan<'a>, PlaceError> {
        if slots == 1 {
            return self.plan_mtp_with(machine, ctx_max, levers, mtp, experts);
        }
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
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
            let mut plan = self.target_of(machine, ctx_max, levers, experts, rows, &kv)?;
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
        let mut plan = self.target_of(machine, ctx_max, levers, experts, reserve + rows, &kv)?;
        plan.cards[0].grow_kv(rows);
        let t = &plan.cards[0];
        // As in [`PlanInputs::plan_mtp_with`]: the draft's term is the
        // reserve the expert rule spread within.
        let total = [
            t.dense_bytes,
            t.expert_bytes,
            t.rounding_bytes,
            t.kv_bytes,
            t.scratch_bytes,
            t.context_bytes,
            reserve,
        ]
        .iter()
        .sum::<u64>();
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
/// rows, 6,219,282 values: the hidden rows 655,360, the embedding 163,840 and
/// its positions and key counts 128, the pack 1,310,720 and its 32-value q8
/// blocks over 256 columns of 5,120 (codes, scales and sums, 20 words a
/// 64-value step) 409,600, the streams 655,360, the wide mix's scratch
/// 1,563,392 (the normed streams and the up's rows 655,360 each, the q8 of the
/// normed streams 204,800, the down's dots and the bottleneck 20,480 each, its
/// q8 6,400, the inject's dots and the weights 256 each), the mix 163,840 and
/// its q8 51,200, the query rows with their gates 786,432, the turned queries
/// 393,216, the keys and values 65,536, and the two one-expert tables of 256
/// and 64 slots, 524 and 134 words — four bytes a value].
///
/// [`MtpArena::new`]: bloomery_gpu::arch::qwen3moe::Mtp38
#[must_use]
pub fn mtp_arena_bytes(vocab: u64) -> u64 {
    4 * (1_212_160 + 8 * vocab + 49_536 * MTP_FLASH_SEGMENTS + 793 + 6_219_282)
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
/// Q8_0 head whose projection reads the f32 rows. [derived: `Q8Act::with_k`
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
        mtp::BorrowedHead::Q8_0 => 0,
        mtp::BorrowedHead::Q6K => {
            let n_sb = hidden / 256;
            let per = 8 * 64 * n_sb.div_ceil(2)
                + 4 * 256 * n_sb.div_ceil(4)
                + 4 * 128 * n_sb.div_ceil(2)
                + 4 * 8 * n_sb
                + 4 * 2 * n_sb;
            MTP_HEAD_ROWS * per
        }
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

/// Card bytes a qwen4exp ubatch walk holds a position [derived: the ubatch
/// arena's token-major rows, 136,996 f32 and 12 u32 a position — the
/// embedding, the two stream buffers, the mix and sub-layer outputs, the
/// attention and flash rows, the delta layer's, the selecting layer's and the
/// PLE site's intermediates, the shared expert's — 548,032 B; the walk's own
/// buffers, 110,000.125 B — the 32-value activations of the model-width, the
/// attention-width and the SwiGLU rows (3,200, 7,680 and 800 B), the wide
/// mixes' scratch (97,712 B: the normed streams and the up's rows at 40,960 B
/// each, their activations, the down's and inject's dots, the bottleneck and
/// its activations, the weights), the indexer keys, the slots and the
/// one-expert table; the record's id word and the host sums (10,244 B);
/// rounded up].
pub const UBATCH_TOKEN_BYTES: u64 = 668_277;

/// Card bytes a qwen4exp ubatch walk holds whatever its size, at a cache of
/// up to 2,000,000 positions [derived, an upper bound: the router's buffers
/// for 2,048 tokens, 8,577,028 B; the eight-column mix scratch, 379,520 B;
/// the selected flash's partials for eight rows of 24 heads over 33 segments
/// of 64 keys, 6,538,752 B; the selection's queries and lists, 82,048 B; its
/// scores, 8 B a cache position, 16,000,000 B at 2,000,000; rounded up to
/// 32 MiB]. A load past that cache refuses its ubatch arena by name.
pub const UBATCH_FIXED_BYTES: u64 = 32 << 20;

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
/// here — the model crate does not read the gpu crate).
pub const CARD_ROUTE_ROWS: u64 = 2_048;

/// Card bytes a ubatch walk's card route holds a run token, an upper bound
/// [derived: the compressed ids and the places 80 B (ten u32 each); the
/// gate·up input `GemmAct` at K = 2,560, 8,592 B a column (q3 2,560 + q4
/// 3,072 + q6 2,560 + s8 320 + d8 80); the gate and up f32 rows
/// 2 · 10 · 640 · 4 = 51,200 B; the SwiGLU `GemmAct32` at K = 640, 800 B a
/// column, 8,000 B; the down f32 rows 10 · 2,560 · 4 = 102,400 B; two route
/// tables — the cols list 2 · 40 B, their tile words at the most experts
/// 2 · 8 · (320 + 512) / 2,048 = 3.25 B, the counts; the identity map of
/// the ids compression 512 words = 1 B: 170,272 + 87.6 + 1, rounded up].
pub const CARD_ROUTE_RUN_TOKEN_BYTES: u64 = 170_360;

/// Card bytes a ubatch walk's card route holds a token of the whole unit
/// [derived: the card slots' weighted sums, `HIDDEN` f32 a row].
pub const CARD_ROUTE_ACC_TOKEN_BYTES: u64 = 10_240;

/// The card bytes a qwen4exp ubatch walk's card route holds at ubatches of
/// up to `u` positions: the run buffers and route tables once
/// `min(u, CARD_ROUTE_ROWS)` run tokens, the sums a token of the whole
/// unit. Counted under [`Experts::Card`] only
/// ([`machine_for_experts`]): a host-routed plan's walk holds none of it.
#[must_use]
pub const fn card_route_scratch_bytes(u: u64) -> u64 {
    let run = if u < CARD_ROUTE_ROWS {
        u
    } else {
        CARD_ROUTE_ROWS
    };
    run * CARD_ROUTE_RUN_TOKEN_BYTES + u * CARD_ROUTE_ACC_TOKEN_BYTES
}

/// Card bytes the ubatch walk's card route holds past
/// [`card_route_scratch_bytes`] on a load with an expert tier, at ubatches
/// of up to `u` positions [derived: a tier layer's join sums the card's and
/// the tier's slots in slot order after the host's wait, so the route keeps
/// its card slots' down rows and places for the whole unit, not one run —
/// (u − min(u, CARD_ROUTE_ROWS)) tokens more of 10 · 2,560 · 4 + 10 · 4 =
/// 102,440 B — and the unit's tier places and their ranks among the tier
/// slots, 2 · 10 · 4 = 80 B a token].
#[must_use]
pub const fn card_tier_join_bytes(u: u64) -> u64 {
    let run = if u < CARD_ROUTE_ROWS {
        u
    } else {
        CARD_ROUTE_ROWS
    };
    (u - run) * 102_440 + u * 80
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

/// The ubatch arena bytes a plan made on `card` counts: its scratch past the
/// m = 1 scratch. The load of a ubatch whose arena holds more is refused by
/// name.
#[must_use]
pub const fn counted_ubatch_bytes(card: &Card) -> u64 {
    card.scratch_bytes.saturating_sub(SCRATCH)
}

/// The machine a qwen4exp plan runs on: `card` runs every one of `layers`,
/// the head and the token embedding table whole (the file's q8_0 rows, which
/// the card gathers), with this workstation's context and margin, the host
/// its card's machine has ([`host_of`]: the pool on a unified machine)
/// and the scratch of a load that runs ubatches of up to `ubatch` positions
/// ([`card_scratch_bytes`]): the size the load itself takes, read once by
/// the caller that builds this machine and opens the load. A host-routed
/// plan's load ([`Experts::Host`]); a card one's
/// ([`machine_for_experts`]) beside the ubatch walk's card route.
#[must_use]
pub fn machine(card: CardSpec, layers: usize, ubatch: u64) -> Machine {
    machine_for_experts(card, layers, ubatch, Experts::Host)
}

/// [`machine`] for a plan whose routed experts go where `experts` says: under
/// [`Experts::Card`] the ubatch walk's card route scratch
/// ([`card_route_scratch_bytes`]) is counted beside the ubatch's — the walk
/// holds those buffers when the plan puts experts on the card.
#[must_use]
pub fn machine_for_experts(
    card: CardSpec,
    layers: usize,
    ubatch: u64,
    experts: Experts,
) -> Machine {
    let route = match experts {
        Experts::Host => 0,
        Experts::Card => card_route_scratch_bytes(ubatch),
    };
    Machine {
        cards: vec![Card {
            name: card.name.to_string(),
            device: card.device,
            usable_bytes: card.usable_bytes(),
            context_bytes: CONTEXT,
            scratch_bytes: card_scratch_bytes(ubatch) + route,
            margin_bytes: MARGIN,
            granule_bytes: GRANULE,
            free_bytes: card.free_bytes,
            held_by: card.held_by,
            layers: 0..layers,
            head: true,
            token_embedding: true,
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        host: host_of(&card),
        unified: unified_of(&card),
    }
}

/// What the MTP draft's reserve on the stage card of a plan (b′) machine is
/// called ([`machine_bp`]): its card bytes and arena
/// ([`MtpInputs::card_bytes`]).
pub const MTP_RESERVE: &str = "MTP draft";

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
pub fn tier_batch(hp: &Hparams, ubatch: u64) -> TierBatchBytes {
    tier_batch_of(hp.n_embd as u64, hp.n_used as u64, ubatch)
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
/// `crates/model/tests/qwen4exp_mtp_meta.rs` (with the draft).
#[must_use]
pub fn machine_bp(
    layers: usize,
    ubatch: u64,
    draft: Option<u64>,
    batch: TierBatchBytes,
) -> Machine {
    machine_bp_on((A6000, RTX_3090), layers, ubatch, draft, batch)
}

/// [`machine_bp`] with `stage` the stage card and `tier` its expert tier
/// card: the cards a placement resolved on this process's devices.
#[must_use]
pub fn machine_bp_on(
    (stage, tier): (CardSpec, CardSpec),
    layers: usize,
    ubatch: u64,
    draft: Option<u64>,
    batch: TierBatchBytes,
) -> Machine {
    let mut m = machine_for_experts(stage, layers, ubatch, Experts::Card);
    m.cards[0].scratch_bytes += card_tier_join_bytes(ubatch);
    m.cards[0]
        .reserves
        .extend(draft.map(|b| (MTP_RESERVE.to_string(), b)));
    m.tiers.push(Card {
        name: tier.name.to_string(),
        device: tier.device,
        usable_bytes: tier.usable_bytes(),
        context_bytes: CONTEXT,
        scratch_bytes: SCRATCH,
        margin_bytes: MARGIN,
        granule_bytes: GRANULE,
        free_bytes: tier.free_bytes,
        held_by: tier.held_by,
        layers: 0..0,
        head: false,
        token_embedding: false,
        reserves: vec![(TIER_BATCH_RESERVE.to_string(), batch.card())],
    });
    m.host
        .reserves
        .push((TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
    m
}

/// The stage cards' [`MTP_RESERVE`] rows against `want`: none for `None`,
/// exactly one of `want` bytes on the one stage card for `Some`; anything
/// else is refused by name ([`PlaceError::DraftReserve`]).
fn check_draft_reserve(machine: &Machine, want: Option<u64>) -> Result<(), PlaceError> {
    for (i, card) in machine.cards.iter().enumerate() {
        let got: Vec<u64> = card
            .reserves
            .iter()
            .filter(|(name, _)| name == MTP_RESERVE)
            .map(|&(_, b)| b)
            .collect();
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
///   file; on a qwen4exp file its state keeps
///   `runtime::stores::DELTA_LANES` lanes, each stamped
///   (`runtime::stores::delta_lane_bytes`);
/// - the PLE site's layer, beside that: the PLE conv ring
///   (`(taps − 1)·dilation + PASS_ROWS` rows of `streams · n_embd` f32);
/// - an attention layer: a position's K and V (`2 · kv_heads · head_dim`
///   values — f16, or q8_0's two-plane layout when the load runs its cache
///   in it ([`KvLayout::in_q8`], the cache lever's `q8_0`); a selecting
///   layer's planes stay f16: the qwen38 family's stores carry no q8_0
///   form); with the selector, also its raw indexer key (`idx_dim` f16) and
///   a pooled key (`idx_dim` f16) per pool of `ratio` positions, the last
///   pool counted whole (`runtime::stores::selecting_bytes`).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per layer: its kind.
    kinds: Vec<Kind>,
    /// A GDN layer's state and conv ring, in bytes.
    recurrent: u64,
    /// The PLE site's layer and its conv ring's bytes.
    ple: Option<(usize, u64)>,
    /// An attention layer's K and V heads and their width.
    kv_heads: usize,
    head_dim: usize,
    /// The selector's key width and each layer's pool (0 on a GDN layer);
    /// `None` on a qwen35moe file.
    select: Option<(usize, Vec<usize>)>,
    /// A plain attention layer's planes in q8_0's layout.
    q8: bool,
}

impl KvLayout {
    /// The layout `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        let exp = hp.exp.as_ref();
        KvLayout {
            kinds: hp.kinds.clone(),
            recurrent: recurrent_bytes(hp.v_heads, hp.k_heads, hp.state, hp.conv)
                + exp.map_or(0, |_| delta_lane_bytes(hp.v_heads, hp.state, DELTA_LANES)),
            ple: exp.and_then(|e| e.ple).map(|p| {
                (
                    p.layer,
                    ple_ring_bytes(p.conv, p.ngram, streams(hp), hp.n_embd),
                )
            }),
            kv_heads: hp.n_head_kv,
            head_dim: hp.head_dim,
            select: exp.map(|e| (e.idx_dim, e.ratios.clone())),
            q8: false,
        }
    }

    /// The same layout with a plain attention layer's planes in q8_0's
    /// two-plane layout.
    #[must_use]
    pub const fn in_q8(mut self) -> KvLayout {
        self.q8 = true;
        self
    }
}

/// The hyper-connection streams of a qwen4exp file (1 without).
fn streams(hp: &Hparams) -> usize {
    hp.exp.as_ref().map_or(1, |e| e.hc_streams)
}

/// Card bytes one resident sequence of a qwen4exp load holds beside its
/// per-layer stores, its [`SeqTerms::beside`] ([`PlanInputs::seq_terms`]):
/// the step's one row and the pass's [`PASS_ROWS`] rows of the four streams
/// — the arena rows a sequence's own draft walks read, held in its own
/// buffers while the arenas address whichever sequence is live — and the
/// lane word every delta launch of the sequence reads [derived: `(1 +
/// PASS_ROWS) · streams · n_embd` f32, one u32 device word]. The engine's own
/// owner of the rows term is `Body38`'s `seq38_bytes`; the gate holds the two
/// equal.
#[must_use]
pub fn slot_resident_bytes(hp: &Hparams) -> u64 {
    ((1 + PASS_ROWS) * streams(hp) * hp.n_embd * std::mem::size_of::<f32>() + 4) as u64
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = match self.ple {
            Some((site, bytes)) if site == layer => bytes,
            _ => 0,
        };
        let own = match self.kinds.get(layer) {
            Some(Kind::DeltaRule) => self.recurrent,
            Some(Kind::Attention) => match &self.select {
                Some((idx_dim, pools)) => {
                    let pool = pools.get(layer).copied().filter(|&p| p > 0).expect(
                        "KvLayout: an attention layer's pool (Hparams::read refuses 0 there)",
                    );
                    let ctx = usize::try_from(ctx_max).expect("KvLayout: a context of usize");
                    selecting_bytes(self.kv_heads, self.head_dim, *idx_dim, pool, ctx)
                }
                None => {
                    let row = if self.q8 {
                        kv_q8_row_bytes(self.kv_heads, self.head_dim)
                    } else {
                        kv_row_bytes(self.kv_heads, self.head_dim)
                    };
                    ctx_max * row
                }
            },
            None => 0,
        };
        own + ple
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
    let (streams, rank) = match d.hc.map(|h| (h.streams, h.kind)) {
        Some((s, HcKind::Gated { rank })) => (s, rank),
        _ => return Err(refuse("it has no gated hyper-connections")),
    };
    let u = u64::from;
    let (h, hd) = (u(d.hidden), u(g.head_dim));
    let q = u(g.heads) * hd * if g.out_gate { 2 } else { 1 };
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
            CardFormat::of_routed,
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
    /// describes (`mtp::mtp_of`), its head scoring `rows`. Refused by name:
    /// a target whose borrowed matrices are not the draft's format
    /// ([`PlanInputs::mtp_borrows`]); what `mtp_of` refuses; a draft file that carries its own `token_embd`
    /// or `output` (this load borrows the target's); a row list of another
    /// tokenizer than the target's; and what [`MtpInputs::from_parts`]
    /// refuses.
    pub fn read(
        draft: &Split,
        target: &Split,
        inputs: &PlanInputs,
        rows: HeadRows,
    ) -> Result<MtpInputs, PlaceError> {
        inputs.mtp_borrows()?;
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
        let MtpSource::File { borrows, .. } = &d.source else {
            return Err(PlacementError::Metadata {
                key: "mtp source".to_string(),
                detail: "the layer is in the target's file; this load reads a draft file".into(),
            }
            .into());
        };
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
                     output, which its card holds: a shared draft file",
                    carried.join(", ")
                ),
            }
            .into());
        }
        if let HeadRows::List { digest, .. } = &rows {
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
        )?))
    }

    /// The inputs of `draft` over `file`, the draft file's tensors in its
    /// order. Refused by name: a tensor of [`mtp_tensors`] the program reads
    /// absent (an unread one may be), one of another type or shape, and a
    /// tensor the file holds that is none of them.
    pub fn from_parts(draft: MtpDraft, file: &[FileTensor]) -> Result<MtpInputs, PlacementError> {
        let tensors = mtp_tensors(&draft)?;
        let index = draft.index;
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
        // `mtp_tensors` refused any other layer.
        let (Mixer::Gqa(g), Ffn::Moe(m)) = (&draft.layer.mixer, &draft.layer.ffn) else {
            return Err(PlacementError::Metadata {
                key: "mtp layer".to_string(),
                detail: "is not a routed GQA layer".to_string(),
            });
        };
        let (experts, experts_used) = (u64::from(m.experts), u64::from(m.top_k));
        let kv = MtpKv {
            kv_heads: g.kv_heads as usize,
            head_dim: g.head_dim as usize,
        };
        Ok(MtpInputs {
            draft,
            tensors,
            model: ModelTensors {
                tensors: model,
                layers: 1,
                experts,
                experts_used,
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
}

/// The draft's own machine: one card of no bound with the workstation's
/// granule, running its one layer and its head; no host tier.
fn draft_machine() -> Machine {
    Machine {
        cards: vec![Card {
            name: "mtp draft".to_string(),
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
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        host: Host {
            usable_bytes: 0,
            reserves: Vec::new(),
        },
        unified: None,
    }
}

/// The MTP layer's store: every position's K and V
/// (`runtime::stores::dense_kv_bytes`).
#[derive(Clone, Copy, Debug)]
struct MtpKv {
    kv_heads: usize,
    head_dim: usize,
}

impl KvBytes for MtpKv {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer != 0 {
            return 0;
        }
        let ctx = usize::try_from(ctx_max).expect("MtpKv: a context of usize");
        dense_kv_bytes(self.kv_heads, self.head_dim, ctx)
    }
}

// ------------------------------------------------------- the head's row list

/// The row list file's first line, before its fields.
pub const HEAD_ROWS_FORMAT: &str = "# bloomery mtp-head-rows 1";

/// The fields of the row list's first line, in order.
const HEAD_ROWS_FIELDS: [&str; 3] = ["vocab", "rows", "vocab_sha256"];

/// Why a row list is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HeadRowsError {
    #[error("head rows {what}: cannot read: {detail}")]
    Read { what: String, detail: String },
    #[error(
        "head rows {what}: line 1 is not `{HEAD_ROWS_FORMAT} vocab=<n> rows=<n> \
         vocab_sha256=<64 hex digits>`: {line:?}"
    )]
    Header { what: String, line: String },
    #[error("head rows {what}: the first line's {field}: {detail}")]
    Field {
        what: String,
        field: &'static str,
        detail: String,
    },
    #[error("head rows {what}: a list over a vocabulary of {file}; the target's has {target}")]
    Vocab {
        what: String,
        file: u64,
        target: u32,
    },
    #[error("head rows {what}: vocab_sha256 {file}; the target's tokenizer is {target}")]
    Digest {
        what: String,
        file: String,
        target: String,
    },
    #[error("head rows {what}: line {line}: {text:?} is not a token id")]
    NotAnId {
        what: String,
        line: usize,
        text: String,
    },
    #[error("head rows {what}: line {line}: id {id} is past the vocabulary of {vocab}")]
    PastVocab {
        what: String,
        line: usize,
        id: u32,
        vocab: u32,
    },
    #[error("head rows {what}: line {line}: id {id} again")]
    Duplicate { what: String, line: usize, id: u32 },
    #[error("head rows {what}: line {line}: id {id} after {prev}; the ids ascend")]
    Unsorted {
        what: String,
        line: usize,
        id: u32,
        prev: u32,
    },
    #[error("head rows {what}: the first line says {said} rows, the file lists {listed}")]
    Count {
        what: String,
        said: u64,
        listed: usize,
    },
    #[error(
        "head rows {what}: {rows} rows; a list holds 1 to {max} (the whole vocabulary is the \
         full head)"
    )]
    Size { what: String, rows: usize, max: u32 },
}

/// The row list in `text` (`what` names it in a refusal), for a target of
/// `vocab` tokens whose tokenizer digests to `digest` ([`vocab_sha256`]):
/// the first line [`HEAD_ROWS_FORMAT`] and its three fields, then one
/// decimal id a line. Refused by name: a first line of another form, a field
/// absent, repeated, unknown or malformed; a vocabulary or a digest other
/// than the target's; a line that is not an id (a blank one too); an id at
/// or past `vocab`, repeated, or below the one before; a count other than
/// the first line's, or outside `1..vocab`.
pub fn parse_head_rows(
    what: &str,
    text: &str,
    vocab: u32,
    digest: &[u8; 32],
) -> Result<HeadRows, HeadRowsError> {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let header = || HeadRowsError::Header {
        what: what.to_string(),
        line: first.to_string(),
    };
    let rest = first
        .strip_prefix(HEAD_ROWS_FORMAT)
        .filter(|r| r.is_empty() || r.starts_with(' '))
        .ok_or_else(header)?;
    let field = |field: &'static str, detail: String| HeadRowsError::Field {
        what: what.to_string(),
        field,
        detail,
    };
    let mut values: [Option<&str>; 3] = [None; 3];
    for pair in rest.split_whitespace() {
        let (key, value) = pair.split_once('=').ok_or_else(header)?;
        let Some(i) = HEAD_ROWS_FIELDS.iter().position(|k| *k == key) else {
            return Err(header());
        };
        if values[i].replace(value).is_some() {
            return Err(field(HEAD_ROWS_FIELDS[i], "is given twice".to_string()));
        }
    }
    let get = |i: usize| values[i].ok_or_else(|| field(HEAD_ROWS_FIELDS[i], "is absent".into()));
    let number = |i: usize| -> Result<u64, HeadRowsError> {
        let v = get(i)?;
        v.parse::<u64>()
            .map_err(|_| field(HEAD_ROWS_FIELDS[i], format!("{v:?} is not a count")))
    };
    let file_vocab = number(0)?;
    let said = number(1)?;
    let file_digest = get(2)?;
    if file_digest.len() != 64 || !file_digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(field(
            "vocab_sha256",
            format!("{file_digest:?} is not 64 hex digits"),
        ));
    }
    if file_vocab != u64::from(vocab) {
        return Err(HeadRowsError::Vocab {
            what: what.to_string(),
            file: file_vocab,
            target: vocab,
        });
    }
    let want = hex(digest);
    if !file_digest.eq_ignore_ascii_case(&want) {
        return Err(HeadRowsError::Digest {
            what: what.to_string(),
            file: file_digest.to_string(),
            target: want,
        });
    }
    let mut ids: Vec<u32> = Vec::new();
    for (i, text) in lines.enumerate() {
        let line = i + 2;
        let at = |text: &str| HeadRowsError::NotAnId {
            what: what.to_string(),
            line,
            text: text.to_string(),
        };
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return Err(at(text));
        }
        let id: u32 = text.parse().map_err(|_| at(text))?;
        if id >= vocab {
            return Err(HeadRowsError::PastVocab {
                what: what.to_string(),
                line,
                id,
                vocab,
            });
        }
        match ids.last() {
            Some(&prev) if prev == id => {
                return Err(HeadRowsError::Duplicate {
                    what: what.to_string(),
                    line,
                    id,
                });
            }
            Some(&prev) if prev > id => {
                return Err(HeadRowsError::Unsorted {
                    what: what.to_string(),
                    line,
                    id,
                    prev,
                });
            }
            _ => {}
        }
        ids.push(id);
    }
    if u64::try_from(ids.len()).ok() != Some(said) {
        return Err(HeadRowsError::Count {
            what: what.to_string(),
            said,
            listed: ids.len(),
        });
    }
    if ids.is_empty() || ids.len() >= vocab as usize {
        return Err(HeadRowsError::Size {
            what: what.to_string(),
            rows: ids.len(),
            max: vocab.saturating_sub(1),
        });
    }
    Ok(HeadRows::List {
        ids: ids.into(),
        digest: *digest,
    })
}

/// The row list at `path` ([`parse_head_rows`]) for the target file
/// `target`, of `vocab` tokens: its tokenizer digested ([`vocab_sha256`]).
pub fn read_head_rows(path: &Path, target: &Split, vocab: u32) -> Result<HeadRows, PlaceError> {
    let what = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| HeadRowsError::Read {
        what: what.clone(),
        detail: e.to_string(),
    })?;
    let digest = vocab_sha256(target)?;
    Ok(parse_head_rows(&what, &text, vocab, &digest)?)
}

/// The SHA-256 of `split`'s vocabulary: each entry of
/// `tokenizer.ggml.tokens`, in id order, as its UTF-8 bytes' count (u64,
/// little-endian) and the bytes — the array's payload as the file stores it
/// (`tools/ref/draft-vocab.py` reads the same bytes). Refused by name: no
/// array, an entry that is not a string, and one holding U+FFFD, which is
/// what the reader makes of bytes that are not UTF-8, so its bytes are not
/// the file's.
pub fn vocab_sha256(split: &Split) -> Result<[u8; 32], PlacementError> {
    let key = crate::arch::TOKENS;
    let refuse = |detail: String| PlacementError::Metadata {
        key: key.to_string(),
        detail,
    };
    let Some(Value::Array(tokens)) = split.value(key) else {
        return Err(refuse("is absent or not an array".to_string()));
    };
    let mut h = Sha256::new();
    for (i, t) in tokens.iter().enumerate() {
        let Value::String(s) = t else {
            return Err(refuse(format!("entry {i} is {t:?}, not a string")));
        };
        if s.contains('\u{FFFD}') {
            return Err(refuse(format!(
                "entry {i} holds U+FFFD, the reader's stand-in for bytes that are not UTF-8: \
                 its file bytes cannot be digested"
            )));
        }
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    Ok(h.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::{
        DELTA_LANES, Kind, KvBytes, KvLayout, delta_lane_bytes, ple_ring_bytes, recurrent_bytes,
    };

    /// Qwen3.8's layers by kind (the sizes themselves are
    /// `runtime::stores`'s, pinned there): a GDN layer's state lanes and conv ring,
    /// the PLE ring beside it on its layer, an attention layer's K/V, raw and
    /// pooled keys at a context of whole pools and one past.
    #[test]
    fn qwen38_layer_bytes() {
        let kv = KvLayout {
            kinds: vec![Kind::DeltaRule, Kind::DeltaRule, Kind::Attention],
            recurrent: recurrent_bytes(48, 16, 128, 4) + delta_lane_bytes(48, 128, DELTA_LANES),
            ple: Some((1, ple_ring_bytes(4, 3, 4, 2560))),
            kv_heads: 2,
            head_dim: 256,
            select: Some((128, vec![0, 0, 4])),
            q8: false,
        };
        assert_eq!(kv.layer_bytes(0, 4096), 3_596_288 + 9_437_200);
        assert_eq!(kv.layer_bytes(1, 4096), 3_596_288 + 9_437_200 + 696_320);
        assert_eq!(kv.layer_bytes(2, 4096), 4096 * 2304 + 1024 * 256);
        assert_eq!(kv.layer_bytes(2, 4097), 4097 * 2304 + 1025 * 256);
        assert_eq!(kv.layer_bytes(3, 4096), 0);
    }

    /// The cache lever's q8_0 term ([`KvLayout::in_q8`]): a plain attention
    /// layer's K/V at `17/16` bytes a value, a selecting layer's and a GDN
    /// layer's stores untouched (the qwen38 family carries no q8_0 form).
    #[test]
    fn q8_layer_bytes() {
        let plain = KvLayout {
            kinds: vec![Kind::Attention],
            recurrent: 0,
            ple: None,
            kv_heads: 4,
            head_dim: 128,
            select: None,
            q8: false,
        };
        let f16 = plain.layer_bytes(0, 4096);
        let q8 = plain.in_q8().layer_bytes(0, 4096);
        assert_eq!(f16, 4096 * 4 * 128 * 2 * 2);
        assert_eq!(q8, 4096 * 4 * 128 * 2 * 17 / 16);
        assert_eq!(q8 * 32, f16 * 17);
        let selecting = KvLayout {
            kinds: vec![Kind::Attention],
            recurrent: 0,
            ple: None,
            kv_heads: 2,
            head_dim: 256,
            select: Some((128, vec![4])),
            q8: false,
        };
        let sel = selecting.layer_bytes(0, 4096);
        assert_eq!(selecting.in_q8().layer_bytes(0, 4096), sel);
    }

    /// The card rule over Qwen3.8's routed stacks ([`super::card_routed`]) on
    /// six synthetic layers of the file's per-expert shapes (512 experts; a
    /// Q4_K gate and up of 640 rows of 2560 values, a Q5_1 down of 2560 rows
    /// of 640): layers 0, 1, 3 and 5 as most of the file's layers, layer 2 as
    /// its layer 2 (a q5_K gate and up, a q8_0 down), layer 4 as its layers 4,
    /// 30, 46 and 47 (a q8_0 down); each layer a 2560-value F32 norm first.
    mod card {
        use std::num::NonZeroU64;

        use gguf::GgmlType;

        use super::super::{card_routed, host_only_reason};
        use crate::placement::{
            self, Card, CardFormat, Device, Format, Host, KvBytes, Machine, ModelTensor,
            ModelTensors, Plan, PlanLevers, Role,
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
            }
        }

        fn machine(usable: u64) -> Machine {
            Machine {
                cards: vec![Card {
                    name: "card".to_string(),
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
                    reserves: Vec::new(),
                }],
                tiers: Vec::new(),
                host: Host {
                    usable_bytes: u64::MAX,
                    reserves: Vec::new(),
                },
                unified: None,
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
                card_routed,
                reserve,
            )
            .expect("the card plan")
        }

        /// The card format of each routed type, and the named reason of each
        /// type it leaves on the host; the Q5_1, Q8_0 and IQ4_NL downs' card
        /// bytes are their file bytes, a whole Q5_1 tensor keeps its packed
        /// format and a whole Q8_0 tensor its planes.
        #[test]
        fn routed_types_and_their_reasons() {
            for ty in [
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::IQ3_XXS,
                GgmlType::IQ4_XS,
                GgmlType::IQ4_NL,
            ] {
                assert_eq!(card_routed(ty), Some(CardFormat::KQuant), "{ty}");
                assert_eq!(host_only_reason(ty), None, "{ty}");
            }
            for ty in [
                GgmlType::Q6_K,
                GgmlType::Q5_0,
                GgmlType::Q3_K,
                GgmlType::IQ2_S,
            ] {
                assert_eq!(card_routed(ty), None, "{ty}");
                let why = host_only_reason(ty).expect("a stack the card does not read is named");
                assert!(why.starts_with(&ty.to_string()), "{why}");
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
            assert_eq!(plan.card_budget, None);
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
                card_budget_bytes: Some(2 * MIB),
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
                        p.card_budget
                    )
                )
            };
            assert_eq!(card.n_l, [0; 6]);
            assert_eq!(view(&card), view(&host));
        }
    }

    mod mtp {
        use std::sync::Arc;

        use gguf::GgmlType;
        use models::{
            Act, Borrows, Ffn, Gqa, HcKind, HcSpec, HeadRows, LayerSpec, Mixer, Moe, MtpDraft,
            MtpHeadNorm, MtpInput, MtpSource, Residual, Rope, RopeMode, Router, Score, Shared,
        };

        use super::super::{
            FileTensor, HEAD_ROWS_FORMAT, HeadRowsError, MtpInputs, MtpLoad, mtp_arena_bytes,
            mtp_tensors, mtp_widened, parse_head_rows, vocab_sha256,
        };
        use crate::arch::synthetic::header;
        use crate::fileio::hex;
        use crate::placement::{self, CardFormat, Device, PlanLevers, Role};

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
        #[test]
        fn the_mtp_arena_formula() {
            assert_eq!(mtp_arena_bytes(248_320), 53_526_700);
            assert_eq!(
                mtp_arena_bytes(248_320),
                4 * (1_212_160 + 8 * 248_320 + 49_536 * 80 + 793 + 6_219_282)
            );
            assert_eq!(mtp_arena_bytes(248_321) - mtp_arena_bytes(248_320), 4 * 8);
        }

        /// Qwen3.8's MTP layer as `mtp_of` reads the shared file.
        pub(super) fn draft(head_rows: HeadRows) -> MtpDraft {
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
                        },
                        qk_norm: true,
                        out_gate: true,
                        select: None,
                        window: None,
                        sinks: false,
                        value_scale: None,
                    }),
                    ffn: Ffn::Moe(Moe {
                        experts: 512,
                        top_k: 10,
                        expert_ff: 640,
                        act: Act::SwiGlu { limit: None },
                        router: Router {
                            score: Score::Softmax,
                            bias: false,
                            norm: true,
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
            let d = draft(HeadRows::Full);
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

        fn list(n: usize) -> HeadRows {
            HeadRows::List {
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
            for rows in [HeadRows::Full, list(LIST_ROWS)] {
                let with_list = matches!(rows, HeadRows::List { .. });
                let m = MtpInputs::from_parts(draft(rows), &file()).expect("the draft's inputs");
                let plan = placement::plan_routed(
                    &m.model,
                    &m.machine,
                    4096,
                    &m.kv,
                    &PlanLevers::default(),
                    CardFormat::of_routed,
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
            let t = mtp_tensors(&draft(HeadRows::Full)).expect("the statement");
            let count = |f: fn(&MtpLoad) -> bool| t.iter().filter(|x| f(&x.load)).count();
            assert_eq!(t.len(), 32);
            assert_eq!(count(|l| matches!(l, MtpLoad::Card(_))), 26);
            assert_eq!(count(|l| matches!(l, MtpLoad::Widened(_))), 2);
            assert_eq!(count(|l| matches!(l, MtpLoad::Unread)), 4);
        }

        #[test]
        fn a_draft_tensor_of_another_form_absent_or_foreign_is_refused() {
            let err = |f: Vec<FileTensor>| {
                MtpInputs::from_parts(draft(HeadRows::Full), &f)
                    .expect_err("refused")
                    .to_string()
            };
            let mut f = file();
            f.iter_mut()
                .find(|t| t.name == "blk.48.hc_attn_inject.weight")
                .expect("the inject")
                .ty = GgmlType::F32;
            assert!(
                err(f).contains("tensor blk.48.hc_attn_inject.weight: is f32 [10240, 4]"),
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
                MtpInputs::from_parts(draft(HeadRows::Full), &f).is_ok(),
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

        const DIGEST: [u8; 32] = [0xab; 32];

        fn rows_file(vocab: u32, ids: &[u32], said: usize) -> String {
            let mut s = format!(
                "{HEAD_ROWS_FORMAT} vocab={vocab} rows={said} vocab_sha256={}\n",
                hex(&DIGEST)
            );
            for id in ids {
                s.push_str(&format!("{id}\n"));
            }
            s
        }

        #[test]
        fn a_row_list_reads() {
            let got = parse_head_rows("t", &rows_file(10, &[0, 3, 9], 3), 10, &DIGEST);
            assert_eq!(
                got,
                Ok(HeadRows::List {
                    ids: Arc::from(vec![0, 3, 9]),
                    digest: DIGEST
                })
            );
        }

        /// A refusal case: what it breaks, the file, the refusal it wants.
        type Case = (&'static str, String, fn(&HeadRowsError) -> bool);

        /// Every refusal, each by its name.
        #[test]
        fn a_row_list_is_refused_by_name() {
            let digest_line = format!("vocab_sha256={}", hex(&DIGEST));
            let cases: Vec<Case> = vec![
                ("no header", "0\n1\n".to_string(), |e| {
                    matches!(e, HeadRowsError::Header { .. })
                }),
                ("empty", String::new(), |e| {
                    matches!(e, HeadRowsError::Header { .. })
                }),
                (
                    "digest absent",
                    format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1\n0\n"),
                    |e| {
                        matches!(
                            e,
                            HeadRowsError::Field {
                                field: "vocab_sha256",
                                ..
                            }
                        )
                    },
                ),
                (
                    "field twice",
                    format!("{HEAD_ROWS_FORMAT} vocab=10 vocab=10 rows=1 {digest_line}\n0\n"),
                    |e| matches!(e, HeadRowsError::Field { field: "vocab", .. }),
                ),
                (
                    "unknown field",
                    format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1 size=3 {digest_line}\n0\n"),
                    |e| matches!(e, HeadRowsError::Header { .. }),
                ),
                (
                    "short digest",
                    format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1 vocab_sha256=abcd\n0\n"),
                    |e| {
                        matches!(
                            e,
                            HeadRowsError::Field {
                                field: "vocab_sha256",
                                ..
                            }
                        )
                    },
                ),
                (
                    "other digest",
                    format!(
                        "{HEAD_ROWS_FORMAT} vocab=10 rows=1 vocab_sha256={}\n0\n",
                        hex(&[0xcd; 32])
                    ),
                    |e| matches!(e, HeadRowsError::Digest { .. }),
                ),
                ("other vocab", rows_file(11, &[0], 1), |e| {
                    matches!(
                        e,
                        HeadRowsError::Vocab {
                            file: 11,
                            target: 10,
                            ..
                        }
                    )
                }),
                ("not an id", rows_file(10, &[0], 1) + "x\n", |e| {
                    matches!(e, HeadRowsError::NotAnId { line: 3, .. })
                }),
                ("blank line", rows_file(10, &[0], 1) + "\n", |e| {
                    matches!(e, HeadRowsError::NotAnId { line: 3, .. })
                }),
                ("signed", rows_file(10, &[], 0) + "+1\n", |e| {
                    matches!(e, HeadRowsError::NotAnId { line: 2, .. })
                }),
                ("past the vocabulary", rows_file(10, &[0, 10], 2), |e| {
                    matches!(
                        e,
                        HeadRowsError::PastVocab {
                            line: 3,
                            id: 10,
                            ..
                        }
                    )
                }),
                ("duplicate", rows_file(10, &[0, 4, 4], 3), |e| {
                    matches!(e, HeadRowsError::Duplicate { line: 4, id: 4, .. })
                }),
                ("unsorted", rows_file(10, &[0, 5, 4], 3), |e| {
                    matches!(
                        e,
                        HeadRowsError::Unsorted {
                            line: 4,
                            id: 4,
                            prev: 5,
                            ..
                        }
                    )
                }),
                ("count", rows_file(10, &[0, 1], 3), |e| {
                    matches!(
                        e,
                        HeadRowsError::Count {
                            said: 3,
                            listed: 2,
                            ..
                        }
                    )
                }),
                ("no rows", rows_file(10, &[], 0), |e| {
                    matches!(e, HeadRowsError::Size { rows: 0, .. })
                }),
                (
                    "the whole vocabulary",
                    rows_file(4, &[0, 1, 2, 3], 4),
                    |e| {
                        matches!(
                            e,
                            HeadRowsError::Size {
                                rows: 4,
                                max: 3,
                                ..
                            }
                        )
                    },
                ),
            ];
            for (what, text, want) in cases {
                let vocab = if what == "the whole vocabulary" {
                    4
                } else {
                    10
                };
                let got = parse_head_rows("t", &text, vocab, &DIGEST);
                match got {
                    Err(e) if want(&e) => {}
                    other => panic!("{what}: {other:?}"),
                }
            }
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
        use models::{Arch, ChatSpec, HeadRows, ModelSpec};

        use super::super::{
            Experts, KvLayout, MtpInputs, PlaceError, PlanInputs, draft_card_bytes, machine_bp,
            slot_resident_bytes, tier_batch_of,
        };
        use super::mtp::{draft, file};
        use crate::arch::qwen35moe::hparams::{Exp, FfnKind, Hparams, Kind, Ple, Variant};
        use crate::placement::workstation::HostRead;
        use crate::placement::{
            self, Card, CardFormat, Host, KvBytes, Machine, ModelTensor, ModelTensors, Plan,
            PlanLevers, Role,
        };

        const CTX: u64 = 4096;
        const LAYERS: usize = 4;

        /// The file's hyperparameters: the stores' terms of the qwen38 layer
        /// test's layout; the rest Qwen3.8's, which no plan reads.
        fn hparams() -> Hparams {
            Hparams {
                variant: Variant::Qwen4Exp,
                n_layer: LAYERS,
                n_trunk: LAYERS,
                n_embd: 2560,
                n_head: 16,
                n_head_kv: 2,
                head_dim: 256,
                rope_dims: 64,
                rope_sections: [11, 11, 10, 0],
                rope_base: 1e7,
                rms_eps: 1e-6,
                n_vocab: 248_320,
                n_ctx_train: 262_144,
                n_expert: 512,
                n_used: 10,
                expert_ff: 640,
                shared_ff: Some(640),
                ff: None,
                conv: 4,
                state: 128,
                v_heads: 48,
                k_heads: 16,
                interval: LAYERS,
                kinds: vec![
                    Kind::DeltaRule,
                    Kind::DeltaRule,
                    Kind::DeltaRule,
                    Kind::Attention,
                ],
                ffns: vec![FfnKind::Routed; LAYERS],
                exp: Some(Exp {
                    hc_streams: 4,
                    hc_rank: 320,
                    idx_heads: 4,
                    idx_dim: 128,
                    idx_top_k: 2048,
                    ratios: vec![0, 0, 0, 4],
                    ple: Some(Ple {
                        layer: 1,
                        ngram: 3,
                        heads_per_ngram: 4,
                        conv: 4,
                        eos: 0,
                        image: None,
                        row: 256,
                    }),
                }),
                defaults: Vec::new(),
            }
        }

        /// The plan's inputs: a norm a layer for the tensors, and a
        /// description of no layer, which no plan reads.
        fn inputs() -> PlanInputs {
            let hp = hparams();
            let kv = KvLayout::of(&hp);
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
            let spec = ModelSpec {
                arch: Arch::Qwen4Exp,
                hidden: 2560,
                vocab: 248_320,
                ctx_train: 262_144,
                rms_eps: 1e-6,
                layers: Vec::new(),
                mtp: Vec::new(),
                hc: None,
                engram: None,
                chat: ChatSpec {
                    pre: String::new(),
                    template: None,
                    tools: None,
                    reasoning: None,
                },
            };
            PlanInputs {
                hp,
                model: ModelTensors {
                    tensors,
                    layers: LAYERS,
                    experts: 512,
                    experts_used: 10,
                },
                spec,
                kv,
                room: (u64::MAX, HostRead::Given),
            }
        }

        /// One card running every layer, with room for every plan here; a
        /// host with no bound.
        fn machine() -> Machine {
            Machine {
                cards: vec![Card {
                    name: "card".to_string(),
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
                    reserves: Vec::new(),
                }],
                tiers: Vec::new(),
                host: Host {
                    usable_bytes: u64::MAX,
                    reserves: Vec::new(),
                },
                unified: None,
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
            let rows = slot_resident_bytes(&inputs.hp) * (slots - 1);
            let kv = Times {
                kv: &inputs.kv,
                slots,
            };
            let levers = PlanLevers::default();
            let mut plan = inputs
                .target_of(machine, CTX, &levers, Experts::Host, reserve + rows, &kv)
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
                CardFormat::of_routed,
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
                    p.card_budget
                )
            )
        }

        /// At 1, 2 and 4 slots, the plain plan and the drafted plan's target
        /// and draft are the plans of each layer's bytes times the slots and
        /// the bytes beside the stores times the slots past the first; and
        /// each plan's kv classes are `SeqTerms::plan_kv` of the slots.
        #[test]
        fn slot_plans_count_every_term_per_slot() {
            let (inputs, machine) = (inputs(), machine());
            let mtp =
                MtpInputs::from_parts(draft(HeadRows::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            for n in [1, 2, 4] {
                let slots = n as u64;
                let plain = inputs
                    .plan_with_slots(&machine, CTX, &levers, Experts::Host, n)
                    .expect("the plan");
                let want = target(&inputs, &machine, slots, 0);
                assert_eq!(view(&plain), view(&want), "{n} slots");
                assert_eq!(
                    plain.cards[0].kv_bytes,
                    inputs.seq_terms(None).plan_kv(CTX, slots),
                    "{n} slots"
                );
                let drafted = inputs
                    .plan_mtp_with_slots(&machine, CTX, &levers, &mtp, Experts::Host, n)
                    .expect("the drafted plan");
                let (draft_want, reserve) = draft_of(&mtp, slots);
                assert_eq!(view(&drafted.draft), view(&draft_want), "{n} slots");
                let want = target(&inputs, &machine, slots, reserve);
                assert_eq!(view(&drafted.plan), view(&want), "{n} slots");
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

        /// The stage card's headroom past the bytes a plan of several slots
        /// adds to its kv class after the totals (the bytes beside every
        /// slot's stores) is the card's usable bytes less every term: on the
        /// plain plan, the drafted plan on one card, and the drafted plan
        /// (b′) whose `MtpPlan::headroom_bytes` is that card's.
        #[test]
        fn slot_plans_keep_the_headroom_usable_less_every_term() {
            let (inputs, machine) = (inputs(), machine());
            let mtp =
                MtpInputs::from_parts(draft(HeadRows::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            let tiered = routed(self::inputs());
            let batch = tier_batch_of(2560, 10, 4096);
            for n in [2, 4] {
                let plain = inputs
                    .plan_with_slots(&machine, CTX, &levers, Experts::Host, n)
                    .expect("the plan");
                let (got, want) = headroom(&plain);
                assert_eq!(got, want, "{n} slots: the plain plan");
                let drafted = inputs
                    .plan_mtp_with_slots(&machine, CTX, &levers, &mtp, Experts::Host, n)
                    .expect("the drafted plan");
                let (got, want) = headroom(&drafted.plan);
                assert_eq!(got, want, "{n} slots: the drafted plan on one card");
                let reserve = mtp.card_bytes_of(CTX, n).expect("the draft's card bytes");
                let bp = machine_bp(LAYERS, 4096, Some(reserve), batch);
                let drafted = tiered
                    .plan_mtp_with_slots(&bp, CTX, &levers, &mtp, Experts::Card, n)
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

        /// The target's `token_embd` and `output`, which an MTP draft file
        /// borrows, pass as Q8_0, and `output` besides as the Q6_K form its
        /// gemv also reads (the UD-Q3_K_XL file's `output`); the first
        /// absent or of another format is named.
        // PIN(2026-10-06): `output` widened to `mtp::head_kind`'s set (Q8_0
        // or Q6_K, one owner `mtp::output_form`) so the UD-Q3_K_XL target
        // lends its Q6_K head to the draft; `token_embd` stays Q8_0 alone —
        // the old test refused a Q6_K `output` by name.
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
            for out in [GgmlType::Q8_0, GgmlType::Q6_K] {
                assert!(with(Some(GgmlType::Q8_0), Some(out)).is_ok(), "{out}");
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
                     token_embd as Q8_0 and its output as Q8_0 or Q6_K"
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
                MtpInputs::from_parts(draft(HeadRows::Full), &file()).expect("the draft's inputs");
            let levers = PlanLevers::default();
            let batch = tier_batch_of(2560, 10, 4096);
            for n in [1, 2, 4] {
                let reserve = mtp.card_bytes_of(CTX, n).expect("the draft's card bytes");
                let machine = machine_bp(LAYERS, 4096, Some(reserve), batch);
                let r = inputs.plan_mtp_with_slots(&machine, CTX, &levers, &mtp, Experts::Card, n);
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
            let machine = machine_bp(LAYERS, 4096, Some(one), batch);
            match inputs.plan_mtp_with_slots(&machine, CTX, &levers, &mtp, Experts::Card, 2) {
                Err(PlaceError::DraftReserve { got, want, .. }) => {
                    assert_eq!((got, want), (vec![one], Some(two)));
                }
                other => panic!("want the draft reserve refused: {:?}", other.err()),
            }
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
                }
            );
            assert_eq!(
                tier_batch_of(2560, 10, 512),
                TierBatchBytes {
                    staging: 62_091_264,
                    scratch: 87_233_816,
                    host: 104_898_560,
                }
            );
        }

        #[test]
        fn the_machine_of_plan_bp() {
            let batch = tier_batch_of(2560, 10, 4096);
            for draft in [None, Some(2_843_762_444u64)] {
                let m = machine_bp(48, 4096, draft, batch);
                let a = machine_for_experts(A6000, 48, 4096, Experts::Card);
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
                let names: Vec<&str> = m.host.reserves.iter().map(|(n, _)| n.as_str()).collect();
                // PIN(2026-10-07): the OS reserve and the tier batch's rows;
                // was [ROW_CACHE_RESERVE, OS_RESERVE, TIER_BATCH_HOST_RESERVE].
                // workstation::host() carries the OS's reserve alone: the row
                // reserve is a plan's, set aside while its row-gathered tables
                // lie on the NVMe tier (HostTotals::row_reserve_bytes), so the
                // machine's list is its host's plus the tiers' batch rows.
                assert_eq!(names, [OS_RESERVE, TIER_BATCH_HOST_RESERVE]);
                assert_eq!(m.host.reserves[1].1, 839_188_480);
                assert_eq!(m.host.usable_bytes, a.host.usable_bytes);
            }
        }

        #[test]
        fn the_draft_reserve_is_the_plans_own() {
            let batch = tier_batch_of(2560, 10, 4096);
            let plain = machine_bp(48, 4096, None, batch);
            let drafted = machine_bp(48, 4096, Some(100), batch);
            assert!(check_draft_reserve(&plain, None).is_ok());
            assert!(check_draft_reserve(&drafted, Some(100)).is_ok());
            let mut twice = machine_bp(48, 4096, Some(100), batch);
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
}
