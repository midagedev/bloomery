//! A mimo2 file planned onto a machine: the one path from the file's headers
//! to a plan that keeps its invariants, as `glm5next::place` is for GLM. The
//! hyperparameters, every tensor's role, the typed description and each
//! layer's KV bytes ([`KvLayout`]) are read once ([`PlanInputs::read`], which
//! refuses a file whose next-token layers carry tensors — the draft that
//! reads them is a later round's — and one with a feature no program runs,
//! every one listed); then the placement ([`PlanInputs::plan`]) runs the
//! expert rule on the routed layers whose stacks the card experts read
//! ([`card_routed`]) and keeps the rest of every layer's experts on the host
//! ([`placement::plan_routed`], the path the other hybrid families plan by),
//! a window layer's KV a ring of [`ring_rows`] under [`Rows::Ring`] and a
//! full plane under [`Rows::Plane`], a full layer a plane either way, and
//! refuses by name a
//! machine that hangs an expert tier card, a context of no position and a
//! host room that cannot hold the host experts: the body serves no NVMe
//! expert tier, so a plan that pages one ([`PlaceError::NoNvmeTier`]) would
//! load experts nothing serves.

use std::num::NonZeroUsize;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::{GgmlType, Split};
use models::ModelSpec;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::workstation::{self, HostNeed, HostRead};
use crate::placement::{
    self, CardFormat, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers,
    Unimplemented, Violation, checked, joined,
};

const F16_BYTES: u64 = 2;

/// The routed stacks the card experts read, each as the file stores it
/// ([`CardFormat::KQuant`]): the types the common card leg runs
/// ([`placement::kernels::card_runs`]). The rule reads a stack's type alone; a
/// layer with a stack of any other type keeps its experts on the host.
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    placement::kernels::card_runs(ty).then_some(CardFormat::KQuant)
}

/// What a plan of a mimo2 file is made from, read from its headers.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's KV bytes.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the
    /// routed experts the room cannot hold follow it to the NVMe tier
    /// ([`placement::expert_nvme_tier`]), which the body does not serve, so
    /// such a plan is refused ([`PlaceError::NoNvmeTier`]). A gate that
    /// plans an arm by a room of its own sets it, read [`HostRead::Given`].
    ///
    /// The plan is made at [`workstation::plan_room`]: a read room less
    /// [`workstation::HOST_DRIFT`], a given room as given.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it
    /// as its `given`. A gate that plans an arm by an arena of its own sets it.
    pub arena: Option<u64>,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// A machine with an expert tier card: this program hangs no expert tier
    /// on its card experts.
    #[error(
        "the machine has the expert tier card {card} (tier {tier}); the program this plan is \
         for hangs no expert tier"
    )]
    Tier {
        /// The tier card's name.
        card: String,
        /// Its index among the machine's tiers.
        tier: usize,
    },
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
    /// The host room cannot hold the host experts, so the plan pages some to
    /// the NVMe expert tier. The body builds its host run without that tier
    /// (`HostRun::build`, not the common `host_run` that attaches it), so the
    /// paged experts would load into nothing that serves them.
    #[error(
        "the host need {need} B passes the room, so the plan pages {paged} B of routed experts \
         to the NVMe expert tier{room}, and the MiMo engine serves no NVMe expert tier; free \
         host memory (the room is the host's available bytes, or BLOOMERY_HOST_ROOM where set)"
    )]
    NoNvmeTier {
        /// The routed-expert bytes the plan leaves on the NVMe tier.
        paged: u64,
        /// The host need before any expert is paged ([`HostNeed::bytes`]).
        need: u64,
        /// The room the plan was made at, as the plan names it
        /// ([`placement::HostTotals::room_clause`]).
        room: String,
    },
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file whose
    /// next-token layers carry tensors is refused by the next-token count's
    /// name (the draft that reads them is a later round's; this plan loads
    /// the trunk without them), and one with a feature the engine does not
    /// run ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        if inputs.hp.n_layer > inputs.hp.n_trunk {
            return Err(PlacementError::Metadata {
                key: "nextn_predict_layers".to_string(),
                detail: format!(
                    "is {}; the draft that reads the next-token layers is a later round's, and \
                     this plan loads the trunk without them",
                    inputs.hp.n_layer - inputs.hp.n_trunk
                ),
            });
        }
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a
    /// gate that plans or describes a file whatever the coverage list holds,
    /// and the host's room ([`workstation::host_available_read`]) and the
    /// arena's given bytes, a room that cannot be read refused by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::ThinkSpan))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        let room = workstation::host_available_read().map_err(PlacementError::HostRoom)?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
        })
    }

    /// The same inputs at an unbounded host room (`HostRead::Given` of every
    /// byte, no arena): the plan they make pages no expert to the NVMe tier,
    /// so what it holds on the card is the card's alone. The card probe of
    /// the default context's search; a machine whose card shares the host's
    /// memory has no such plan, and the ladder
    /// ([`placement::ctx::Veto::Ladder`]) does not ask this one.
    #[must_use]
    pub fn room_free(&self) -> PlanInputs {
        PlanInputs {
            room: (u64::MAX, HostRead::Given),
            arena: None,
            ..self.clone()
        }
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`: the expert rule on the layers whose routed
    /// stacks [`card_routed`] runs, each such layer's id prefix on the card
    /// as far as its budget holds them, the rest on the host; refused by name
    /// a context of no position, a machine that hangs an expert tier card,
    /// a context whose KV planes do not fit the card and a host room under the
    /// host experts' need ([`PlanInputs::plan_with_slots`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of
    /// `ctx_max` positions each: every layer's KV planes counted `slots`
    /// times on the card ([`SeqTerms::slots_of`] over
    /// [`PlanInputs::seq_terms`]), refused as [`PlanInputs::plan`] refuses.
    /// A sequence holds nothing beside its stores, so the card's kv class is
    /// `slots` sequences' stores.
    ///
    /// The host experts are held to the room the inputs name
    /// ([`PlanInputs::room`]): a room under the host need runs the common
    /// NVMe tier split ([`placement::expert_nvme_tier`], which refuses a room
    /// under the tier's floor by name), and a plan that comes out paging any
    /// routed expert is refused ([`PlaceError::NoNvmeTier`]), for the body
    /// serves no such tier.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        if let Some((tier, card)) = machine.tiers.iter().enumerate().next() {
            return Err(PlaceError::Tier {
                card: card.name.clone(),
                tier,
            });
        }
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let mut plan = placement::plan_routed(
            &self.model,
            machine,
            ctx_max,
            &kv,
            levers,
            routed_rule(Program::Mimo2Body),
        )?;
        let need = HostNeed::of(&plan, 0).bytes();
        let room = plan.plan_at(&self.room);
        placement::expert_nvme_tier(&mut plan, room, self.arena)?;
        if plan.host.nvme_expert_bytes > 0 {
            return Err(PlaceError::NoNvmeTier {
                paged: plan.host.nvme_expert_bytes,
                need,
                room: plan.host.room_clause(),
            });
        }
        checked(plan).map_err(PlaceError::Broken)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the trunk's layers ([`KvLayout`]), no
    /// draft and nothing beside them — the body's sequence is its layers'
    /// KV planes alone.
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.hp.n_trunk,
            },
            draft: None,
            beside: 0,
        }
    }
}

/// How many rows a window layer holds. A full layer holds `ctx_max` under
/// either choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The engine's default: [`ring_rows`] rows.
    Ring,
    /// Every position. The default-context search and the oracle sequence
    /// use this term.
    Plane,
}

impl Rows {
    /// Rows held under this choice. No window means a full layer.
    #[must_use]
    pub fn count(self, window: Option<u64>, ctx_max: u64) -> u64 {
        match (self, window) {
            (Self::Ring, Some(w)) => ring_rows(w, ctx_max),
            _ => ctx_max,
        }
    }
}

/// Rows a window of `window` positions holds at `ctx_max`: the window plus
/// one prompt batch, padded up to a multiple of 256, and never past
/// `ctx_max`. The batch is [`crate::moe::UNION_MAX_COLS`].
#[must_use]
pub fn ring_rows(window: u64, ctx_max: u64) -> u64 {
    let span = window.saturating_add(crate::moe::UNION_MAX_COLS as u64 - 1);
    ctx_max.min(span.div_ceil(256) * 256)
}

/// Whether a cut of `depth` positions (`pos - n`) still leaves the keys the
/// pass at `n` reads. A plane (`rows == ctx`) does. A ring holds
/// `[pos - rows, pos)` and that pass appends row `n` and reads
/// `[n - window + 1, n]`, so the cut is free when
/// `depth <= rows - window + 1`. The ring is at least the window plus one
/// prompt batch minus one, which is why a cut inside that slack writes
/// nothing.
#[must_use]
pub fn cut_depth_keeps(depth: u64, rows: u64, ctx: u64, window: u64) -> bool {
    if rows == ctx {
        return true;
    }
    depth <= rows.saturating_sub(window).saturating_add(1)
}

/// One file's per-layer KV bytes. A full layer holds a plane of every
/// position. A window layer holds that plane under [`Rows::Plane`], and
/// [`ring_rows`] rows under [`Rows::Ring`]. Each row is the layer's KV-head
/// count's key and value rows in f16.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer: its KV-head count (`attention.head_count_kv`).
    kv_heads: Vec<usize>,
    /// Per trunk layer: the sliding window, or none for a full layer.
    windows: Vec<Option<u64>>,
    /// One KV head's key and value rows a position, in bytes: the key and
    /// value head widths (`attention.key_length`,
    /// `attention.value_length`) in f16.
    row: u64,
    /// Which row count a window layer uses.
    rows: Rows,
}

impl KvLayout {
    /// The layout `hp` describes, window layers on the ring.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout::with_rows(hp, Rows::Ring)
    }

    /// [`KvLayout::of`] with the window layers on `rows`.
    #[must_use]
    pub fn with_rows(hp: &Hparams, rows: Rows) -> KvLayout {
        let n = hp.n_trunk;
        KvLayout {
            kv_heads: hp.kv_heads[..n].to_vec(),
            windows: hp.kinds[..n]
                .iter()
                .map(|k| match k {
                    Kind::Swa => Some(hp.window as u64),
                    Kind::Full => None,
                })
                .collect(),
            row: (hp.head_k + hp.head_v) as u64 * F16_BYTES,
            rows,
        }
    }

    /// The bytes of layers `layers` at `ctx_max` positions: what a card that
    /// runs them holds beside its weights ([`KvBytes::layer_bytes`] summed).
    #[must_use]
    pub fn bytes(&self, layers: std::ops::Range<usize>, ctx_max: u64) -> u64 {
        layers.map(|l| self.layer_bytes(l, ctx_max)).sum()
    }

    /// The rows layer `layer` holds at `ctx_max`. A layer past the trunk
    /// holds none.
    #[must_use]
    pub fn rows_at(&self, layer: usize, ctx_max: u64) -> u64 {
        if self.kv_heads.get(layer).is_none() {
            return 0;
        }
        self.rows
            .count(self.windows.get(layer).copied().flatten(), ctx_max)
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.kv_heads
            .get(layer)
            .map_or(0, |&kv| self.rows_at(layer, ctx_max) * kv as u64 * self.row)
    }
}

#[cfg(test)]
mod tests {
    use gguf::{GgmlType, Split};

    use super::super::hparams::tests::{keys, tensors};
    use super::{
        CardFormat, F16_BYTES, KvBytes, KvLayout, PlaceError, PlanInputs, PlanLevers, Rows,
        SeqTerms, Stores, card_routed,
    };
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::workstation::{self, HostRead};

    /// A layer's rows: its own KV-head count's key and value rows in f16 a
    /// position. Layers 1 and 2 are a window of 8; 0 and 3 are full. At 9
    /// positions the ring still holds every position, because
    /// [`super::ring_rows`] pads the window plus one prompt batch past 9.
    /// Past that pad the ring stays at the pad and the plane grows with the
    /// context. A layer past the trunk holds nothing. The widths are the
    /// `hparams` tests' small header's: key 16 and value 8 values, 2 KV heads
    /// on a full layer and 1 on a window one.
    fn layout(rows: Rows) -> KvLayout {
        KvLayout {
            kv_heads: vec![2, 1, 1, 2],
            windows: vec![None, Some(8), Some(8), None],
            row: (16 + 8) as u64 * F16_BYTES,
            rows,
        }
    }

    #[test]
    fn layer_bytes() {
        let pad = super::ring_rows(8, u64::MAX);
        assert!(pad > 9, "the ring pad is past the 9-position case");
        for rows in [Rows::Ring, Rows::Plane] {
            let kv = layout(rows);
            assert_eq!(kv.row, 48);
            assert_eq!(kv.layer_bytes(0, 9), 9 * 2 * 48);
            assert_eq!(kv.layer_bytes(1, 9), 9 * 48);
            assert_eq!(kv.layer_bytes(4, 9), 0);
            assert_eq!(kv.bytes(0..4, 1), 6 * 48);
            assert_eq!(kv.bytes(0..4, 9), 9 * 6 * 48);
        }
        let ring = layout(Rows::Ring);
        let plane = layout(Rows::Plane);
        let past = pad + 32;
        assert_eq!(ring.rows_at(1, past), pad);
        assert_eq!(plane.rows_at(1, past), past);
        assert_eq!(ring.layer_bytes(1, past), pad * 48);
        assert_eq!(plane.layer_bytes(1, past), past * 48);
        assert_eq!(ring.layer_bytes(0, past), past * 2 * 48);
        assert_eq!(plane.layer_bytes(0, past), past * 2 * 48);
    }

    /// A ring cut keeps `n` through `rows - window + 1` positions and refuses
    /// the next one. A plane (`rows == ctx`) keeps a cut of any depth. At a
    /// context the ring still fills, the two are the same term.
    #[test]
    fn a_ring_cut_past_the_slack_is_refused() {
        let window = 128;
        let ctx = 4_160;
        let rows = super::ring_rows(window, ctx);
        let span = window + crate::moe::UNION_MAX_COLS as u64 - 1;
        assert_eq!(rows, span.div_ceil(256) * 256);
        assert!(rows < ctx, "the ring is shorter than this context");
        let slack = rows - window + 1;
        assert!(super::cut_depth_keeps(slack, rows, ctx, window));
        assert!(!super::cut_depth_keeps(slack + 1, rows, ctx, window));
        assert!(super::cut_depth_keeps(slack + 1, ctx, ctx, window));
        let filled = super::ring_rows(8, 9);
        assert_eq!(filled, 9);
        assert!(super::cut_depth_keeps(1_000, filled, 9, 8));
    }

    /// A feed of 1,600 leaves the written end at 1,600. The cut to 960 is
    /// depth 640, inside the slack. The end stays 1,600, so the cut to 950
    /// is depth 650 and is refused. Depth 10, measured from 960, is the
    /// overwritten-key reading.
    #[test]
    fn a_second_cut_is_measured_from_the_written_end() {
        let window = 128;
        let ctx = 4_160;
        let rows = super::ring_rows(window, ctx);
        let written = 1_600;
        let first = 960;
        let second = 950;
        assert!(super::cut_depth_keeps(written - first, rows, ctx, window));
        assert!(!super::cut_depth_keeps(
            written - second,
            rows,
            ctx,
            window
        ));
        assert!(super::cut_depth_keeps(first - second, rows, ctx, window));
    }

    /// The published MiMo trunk: nine full layers of 4 KV heads and thirty-nine
    /// window layers of 8, each value 320 f16 bytes a head. A position is
    /// 222,720 B on either term, because the ring is the context. At the gate
    /// context the plane is that figure times 4,160 and the ring is the full
    /// layers' plane plus 768 window rows.
    #[test]
    fn the_ring_bytes_at_the_gate_context() {
        let full = [0usize, 5, 11, 17, 23, 29, 35, 41, 47];
        let mut kv_heads = vec![8usize; 48];
        let mut windows = vec![Some(128u64); 48];
        for &layer in &full {
            kv_heads[layer] = 4;
            windows[layer] = None;
        }
        let row = (192 + 128) as u64 * super::F16_BYTES;
        let layout = |rows: Rows| KvLayout {
            kv_heads: kv_heads.clone(),
            windows: windows.clone(),
            row,
            rows,
        };
        let plane = layout(Rows::Plane);
        let ring = layout(Rows::Ring);
        assert_eq!(plane.bytes(0..48, 1), 222_720);
        assert_eq!(ring.bytes(0..48, 1), 222_720);
        assert_eq!(plane.bytes(0..48, 4_160), 4_160 * 222_720);
        assert_eq!(ring.bytes(0..48, 4_160), 249_200_640);
        assert_eq!(super::ring_rows(128, 4_160), 768);
    }

    /// One sequence's terms are the layout's stores over the trunk with
    /// nothing beside them, and `slots` sequences count each layer `slots`
    /// times — the card's kv class of a load that serves them.
    #[test]
    fn seq_terms_count_every_sequence() {
        let kv = layout(Rows::Ring);
        let terms = SeqTerms {
            layers: Stores { kv: &kv, count: 4 },
            draft: None,
            beside: 0,
        };
        assert_eq!(terms.bytes(9, 1), kv.bytes(0..4, 9));
        assert_eq!(terms.bytes(9, 3), 3 * kv.bytes(0..4, 9));
        assert_eq!(terms.plan_kv(9, 3), 3 * kv.bytes(0..4, 9));
        let three = terms.slots_of(3);
        for l in 0..4 {
            assert_eq!(three.layer_bytes(l, 9), 3 * kv.layer_bytes(l, 9));
        }
        assert_eq!(three.layer_bytes(4, 9), 0);
    }

    /// A file whose next-token layers carry tensors is refused at
    /// [`PlanInputs::read`] by the next-token count's name — the draft that
    /// reads them is a later round's, and this plan loads the trunk without
    /// them — before the coverage list; the same file without its next-token
    /// layer is refused by that list alone (the synthetic head width is one no
    /// row of the mimo2 body runs), so the count is the named refusal.
    #[test]
    fn next_token_layers_are_refused_at_read() {
        let err = read_inputs("mimo2-place-nextn", &keys(), &tensors())
            .expect_err("a file with a next-token layer is refused");
        assert!(
            err.contains("metadata nextn_predict_layers: is 1")
                && err.contains("the draft that reads the next-token layers is a later round's"),
            "{err}"
        );
        let tensors: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| !n.starts_with("blk.4."))
            .collect();
        let err = read_inputs("mimo2-place-trunk", &trunk_keys(), &tensors)
            .expect_err("a trunk-only file is refused by the coverage list");
        assert!(
            err.contains("feature(s) of this file are not implemented"),
            "{err}"
        );
    }

    /// A stack is card-routed when the common kernel table gives its type the
    /// entries this program's leg launches, and not otherwise: the types the
    /// table runs on the card for other families (Q8_0 and IQ4_NL) and the
    /// ones it has no row for (Q2_K, the floats) keep their layer's experts on
    /// the host.
    // PIN(2026-10-11): card_runs now runs Q4_K/Q5_K through the common leg.
    // PIN(2026-10-11): the leg also runs IQ2_XS, IQ3_XXS and IQ4_XS (the
    // raw-block entries of `iq_sel.rs`), so those three are card-routed too.
    // PIN(2026-10-12): the leg also runs IQ3_S (`iq3_s_gate_up_sel`,
    // `iq3_s_gemv_sel`), so it is card-routed too.
    #[test]
    fn only_the_stacks_the_leg_runs_are_card_routed() {
        for ty in [
            GgmlType::MXFP4,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            assert_eq!(card_routed(ty), Some(CardFormat::KQuant), "{ty}");
        }
        for ty in [
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::F32,
            GgmlType::BF16,
        ] {
            assert_eq!(card_routed(ty), None, "{ty}");
        }
    }

    /// A machine that hangs an expert tier card is refused by name, before
    /// any plan is made: the program's card experts have no tier to join,
    /// and a context of no position is refused first.
    #[test]
    fn a_tier_card_is_refused_by_name() {
        let path = header_shaped(
            "mimo2-place-tier",
            "mimo2",
            &keys(),
            &[
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        let mut machine = workstation::plan_gate(inputs.hp.n_trunk);
        let mut tier = machine.cards[0].clone();
        tier.layers = 0..0;
        tier.head = false;
        tier.token_embedding = false;
        machine.tiers.push(tier);
        let levers = PlanLevers::default();
        match inputs.plan(&machine, 4, &levers) {
            Err(e @ PlaceError::Tier { tier: 0, .. }) => assert_eq!(
                e.to_string(),
                "the machine has the expert tier card 3090 (tier 0); the program this plan is \
                 for hangs no expert tier"
            ),
            other => panic!("{other:?}, not the tier refusal"),
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
    }

    /// A host room under the host experts' need is refused by name: the need,
    /// the bytes the plan pages, the room the plan was made at, and that the
    /// engine serves no NVMe expert tier. The serve gate holds the fixed words
    /// (`gate_mimo2_serve`'s `NO_TIER_REFUSAL`).
    #[test]
    fn a_room_under_the_host_need_is_refused_by_name() {
        let err = PlaceError::NoNvmeTier {
            paged: 1_073_741_824,
            need: 150_000_000_000,
            room: "; the plan was made at a reading of 149000000000 B (plan room 149000000000 B)"
                .to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "the host need 150000000000 B passes the room, so the plan pages 1073741824 B of \
             routed experts to the NVMe expert tier; the plan was made at a reading of \
             149000000000 B (plan room 149000000000 B), and the MiMo engine serves no NVMe \
             expert tier; free host memory (the room is the host's available bytes, or \
             BLOOMERY_HOST_ROOM where set)"
        );
    }

    /// The card probe's inputs are the file's at an unbounded given room and
    /// no arena, and nothing else changes.
    #[test]
    fn room_free_unbounds_the_room_alone() {
        let path = header_shaped(
            "mimo2-place-room-free",
            "mimo2",
            &keys(),
            &[
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let mut inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs.room = (7, HostRead::MemAvailable);
        inputs.arena = Some(3);
        let free = inputs.room_free();
        assert_eq!(free.room, (u64::MAX, HostRead::Given));
        assert_eq!(free.arena, None);
        assert_eq!(free.hp, inputs.hp);
        assert_eq!(free.model.layers, inputs.model.layers);
        assert_eq!(
            inputs.room,
            (7, HostRead::MemAvailable),
            "the original stays"
        );
    }

    /// The small header of the `hparams` tests' keys with its next-token
    /// layer taken out: four trunk blocks, the per-layer arrays four long.
    fn trunk_keys() -> Vec<(&'static str, V)> {
        keys()
            .into_iter()
            .map(|(k, v)| match k {
                "block_count" => ("block_count", V::U32(4)),
                "nextn_predict_layers" => ("nextn_predict_layers", V::U32(0)),
                "attention.head_count_kv" => ("attention.head_count_kv", V::I32s(vec![2, 1, 1, 2])),
                "attention.sliding_window_pattern" => (
                    "attention.sliding_window_pattern",
                    V::I32s(vec![0, 1, 1, 0]),
                ),
                _ => (k, v),
            })
            .collect()
    }

    /// [`PlanInputs::read`] of a synthetic header, or its error's text.
    fn read_inputs(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<PlanInputs, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "mimo2", kv, &global, tensors);
        let split = Split::open(&path).expect("the synthetic header opens");
        let read = PlanInputs::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }
}
