//! A qwen4exp file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `glm5next::place` is for
//! GLM-5.3-Flash. The hyperparameters, every tensor's role, the typed
//! description and each layer's recurrent and cache bytes ([`KvLayout`]) are
//! read once ([`PlanInputs::read`]); a file with a feature the engine does not
//! run is refused there by the coverage check; then the placement
//! ([`PlanInputs::plan`]) keeps every routed expert on the host
//! ([`placement::plan_host_routed`]: no card expert kernel serves the file's
//! routed stacks yet) and the rest where its role says — the trunk, the
//! hyper-connections, the PLE site's projections, the router, the shared
//! expert and the head on the card, the PLE table on the host in its file
//! bytes ([`ple_on_host`]), its rows gathered by the host.
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
//! the card, its dense store and the reduced head's gathered rows, counted in
//! the draft's own heap; the target card's bound is checked on the sum. The
//! draft borrows the target's `token_embd` and `output`, which the target's
//! plan already holds on the card, so they add nothing.

use std::path::Path;

use gguf::{GgmlType, Split, Value};
use models::{DraftSpec, Ffn, HcKind, HeadRows, Mixer, ModelSpec, MtpDraft, MtpSource, Role};
use sha2::{Digest, Sha256};

use super::hparams::{Hparams, Kind};
use super::{mtp, roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::fileio::hex;
use crate::placement::workstation::{CONTEXT, CardSpec, GRANULE, MARGIN, SCRATCH, host};
use crate::placement::{
    self, Card, CardFormat, Device, Format, Host, KvBytes, Machine, ModelTensor, ModelTensors,
    PlacementError, Plan, PlanLevers, Unimplemented, Violation,
};

use runtime::stores::{
    DELTA_LANES, delta_lane_bytes, dense_kv_bytes, kv_row_bytes, ple_ring_bytes, recurrent_bytes,
    selecting_bytes,
};
pub use runtime::stores::{PASS_ROWS, conv_ring_rows, ple_ring_rows};

/// The positions the selector's kernels take: the QSA passes take the
/// context as a `u32` launch argument and the selected flash reads `u32`
/// cache rows, so a context past this is one those launches cannot name.
pub const KERNEL_POSITIONS: u64 = u32::MAX as u64;

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
}

/// The violations, `; `-separated.
fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
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
        let chat = chat_of(split, None, None)?;
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
    /// the placement's `levers`, every routed expert and the PLE table on the
    /// host; refused past [`KERNEL_POSITIONS`], when it cannot be built, or
    /// when it breaks an invariant.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        let plan = self.target(machine, ctx_max, levers)?;
        let broken = plan.violations();
        if broken.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::Broken(broken))
        }
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
        if ctx_max == 0 || ctx_max > KERNEL_POSITIONS {
            return Err(PlaceError::Positions { ctx_max });
        }
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::DraftCards {
                cards: machine.cards.len(),
            });
        };
        let plan = self.target(machine, ctx_max, levers)?;
        let draft = placement::plan_routed(
            &mtp.model,
            &mtp.machine,
            ctx_max,
            &mtp.kv,
            &PlanLevers::default(),
            CardFormat::of_routed,
        )?;
        let held: u64 = draft.n_l.iter().sum();
        if held != mtp.model.experts {
            return Err(PlaceError::DraftExperts {
                held,
                experts: mtp.model.experts,
            });
        }
        let (t, d) = (&plan.cards[0], &draft.cards[0]);
        let total = [
            t.dense_bytes,
            t.expert_bytes,
            t.rounding_bytes,
            t.kv_bytes,
            t.scratch_bytes,
            t.context_bytes,
            draft_card_bytes(d, mtp.map_bytes),
        ]
        .iter()
        .sum::<u64>();
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        let mut broken: Vec<Violation> = plan
            .violations()
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
        })
    }

    /// The target's placement, before its invariants are checked: every
    /// routed expert on the host ([`placement::plan_host_routed`]), then the
    /// PLE table moved to the host ([`ple_on_host`]).
    fn target<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlacementError> {
        let mut plan =
            placement::plan_host_routed(&self.model, machine, ctx_max, &self.kv, levers)?;
        ple_on_host(&mut plan)?;
        Ok(plan)
    }
}

/// `plan` with the PLE table's one segment moved from the NVMe tier, where
/// the placement's role rule puts a row-gathered table, to the host in its
/// file bytes, and its bytes from `nvme_bytes` to the host's tables (and out
/// of the host's headroom). Every step reads the table's hashed rows from
/// host RAM before its launch, so its pages belong to the host set a load
/// reads in and may lock ([`placement::host_lock::HostSet`]): a first touch
/// in a step is a read from the drive. In this architecture the one
/// [`Role::EngramTable`] is the PLE table; a file without one is left as it
/// is. A PLE row of other than one NVMe file segment is refused by name.
fn ple_on_host(plan: &mut Plan<'_>) -> Result<(), PlacementError> {
    let model = plan.model;
    for r in &mut plan.rows {
        let Some(t) = model
            .tensors
            .get(r.tensor)
            .filter(|t| t.role == Role::EngramTable)
        else {
            continue;
        };
        let refuse = |detail: String| PlacementError::Tensor {
            name: t.name.clone(),
            detail,
        };
        let n = r.segments.len();
        let [s] = r.segments.as_mut_slice() else {
            return Err(refuse(format!("the PLE table in {n} segments, not one")));
        };
        if (s.device, s.format) != (Device::Nvme, Format::NvmeFile) {
            return Err(refuse(format!(
                "the PLE table on {:?} as {}, not the NVMe tier's file bytes",
                s.device, s.format
            )));
        }
        let b = s.resident_bytes;
        let nvme = plan.nvme_bytes.checked_sub(b).ok_or_else(|| {
            refuse(format!(
                "the PLE table's {b} bytes pass the NVMe tier's {}",
                plan.nvme_bytes
            ))
        })?;
        let tables = plan.host.table_bytes.checked_add(b).ok_or_else(|| {
            refuse(format!(
                "the host's {} table bytes and the PLE table's {b} pass u64",
                plan.host.table_bytes
            ))
        })?;
        plan.nvme_bytes = nvme;
        plan.host.table_bytes = tables;
        plan.host.headroom_bytes -= i128::from(b);
        (s.device, s.format) = (Device::Host, Format::HostFile);
    }
    Ok(())
}

/// A draft plan's card terms: its granules (dense, experts, rounding), its
/// store, and the head's row map.
fn draft_card_bytes(d: &placement::CardTotals, map_bytes: u64) -> u64 {
    d.dense_bytes + d.expert_bytes + d.rounding_bytes + d.kv_bytes + map_bytes
}

/// A qwen4exp plan with its MTP draft ([`PlanInputs::plan_mtp`]).
#[derive(Debug)]
pub struct MtpPlan<'a> {
    /// The target's plan, [`PlanInputs::plan`]'s bit for bit: its card
    /// totals and headroom are the target's alone.
    pub plan: Plan<'a>,
    /// The draft's plan on [`MtpInputs::machine`]: its tensors' rows, its
    /// routed experts (every one on the card), its granules in its own heap
    /// and its store (`cards[0].kv_bytes`).
    pub draft: Plan<'a>,
    /// The head's row → id map, one `u32` a row; 0 for the full head.
    pub map_bytes: u64,
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

/// The most positions a qwen4exp ubatch walk takes, the load's size at most:
/// the card scratch the plan counts is the walk's at this size.
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

/// The machine a qwen4exp plan runs on: `card` runs every one of `layers`,
/// the head and the token embedding table whole (the file's q8_0 rows, which
/// the card gathers), with this workstation's context, scratch, margin and
/// host tier — the scratch with the ubatch walk's at [`UBATCH_PLANNED`]
/// positions.
#[must_use]
pub fn machine(card: CardSpec, layers: usize) -> Machine {
    Machine {
        cards: vec![Card {
            name: card.name.to_string(),
            usable_bytes: card.usable_bytes(),
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH + ubatch_scratch_bytes(UBATCH_PLANNED),
            margin_bytes: MARGIN,
            granule_bytes: GRANULE,
            layers: 0..layers,
            head: true,
            token_embedding: true,
        }],
        host: host(),
    }
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
/// - an attention layer: a position's K and V (`2 · kv_heads · head_dim` f16);
///   with the selector, also its raw indexer key (`idx_dim` f16) and a
///   pooled key (`idx_dim` f16) per pool of `ratio` positions, the last pool
///   counted whole (`runtime::stores::selecting_bytes`).
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
        }
    }
}

/// The hyper-connection streams of a qwen4exp file (1 without).
fn streams(hp: &Hparams) -> usize {
    hp.exp.as_ref().map_or(1, |e| e.hc_streams)
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
                None => ctx_max * kv_row_bytes(self.kv_heads, self.head_dim),
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

/// The reduced head's rows of the target's `output.weight`, gathered at load
/// into one Q8_0 matrix, row `i` the list's `i`th id.
pub const MTP_HEAD_ROWS: &str = "derived.mtp.head_rows";

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
    /// The layer, its head rows the list the load gathers or the full head.
    pub draft: MtpDraft,
    /// [`mtp_tensors`] of `draft`.
    pub tensors: Vec<MtpTensor>,
    /// The draft's tensors in its file's order, as layer 0 of one — each
    /// [`MtpLoad::Card`] tensor in its role, every other one
    /// [`Role::Unused`] — then the widened tensors ([`mtp_widened`], f32)
    /// and, for a row list, the head's gathered rows ([`MTP_HEAD_ROWS`],
    /// Q8_0 `[hidden, rows]`, [`Role::Head`]).
    pub model: ModelTensors,
    /// One card with no bound and the target cards' granule: the draft's
    /// plan places everything on it and counts its uploads in its own heap.
    /// The bound is the target's card's, on the sum.
    pub machine: Machine,
    kv: MtpKv,
    /// The head's row → id map, one `u32` a row; 0 for the full head.
    pub map_bytes: u64,
}

impl MtpInputs {
    /// The MTP draft `draft` read against the target `target` that `inputs`
    /// describes (`mtp::mtp_of`), its head scoring `rows`. Refused by name:
    /// what `mtp_of` refuses; a draft file that carries its own `token_embd`
    /// or `output` (this load borrows the target's); a row list of another
    /// tokenizer than the target's; and what [`MtpInputs::from_parts`]
    /// refuses.
    pub fn read(
        draft: &Split,
        target: &Split,
        inputs: &PlanInputs,
        rows: HeadRows,
    ) -> Result<MtpInputs, PlaceError> {
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
        )?)
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
        let map_bytes = match &draft.head_rows {
            HeadRows::Full => 0,
            HeadRows::List { ids, .. } => {
                let n = ids.len() as u64;
                let hidden = u64::from(draft.hidden);
                let row = GgmlType::Q8_0
                    .type_size()
                    .zip(GgmlType::Q8_0.blck_size())
                    .map(|(size, blck)| hidden / blck * size)
                    .ok_or_else(|| PlacementError::Metadata {
                        key: "q8_0".to_string(),
                        detail: "has no block size".to_string(),
                    })?;
                model.push(ModelTensor {
                    name: MTP_HEAD_ROWS.to_string(),
                    shard: 0,
                    layer: None,
                    role: Role::Head,
                    ty: GgmlType::Q8_0,
                    dims: vec![hidden, n],
                    file_bytes: n * row,
                    gathered_rows: None,
                });
                n * runtime::stores::U32_BYTES
            }
        };
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
        })
    }
}

/// The draft's own machine: one card of no bound with the workstation's
/// granule, running its one layer and its head; no host tier.
fn draft_machine() -> Machine {
    Machine {
        cards: vec![Card {
            name: "mtp draft".to_string(),
            usable_bytes: u64::MAX,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: GRANULE,
            layers: 0..1,
            head: true,
            token_embedding: false,
        }],
        host: Host {
            usable_bytes: 0,
            reserves: Vec::new(),
        },
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
        };
        assert_eq!(kv.layer_bytes(0, 4096), 3_596_288 + 9_437_200);
        assert_eq!(kv.layer_bytes(1, 4096), 3_596_288 + 9_437_200 + 696_320);
        assert_eq!(kv.layer_bytes(2, 4096), 4096 * 2304 + 1024 * 256);
        assert_eq!(kv.layer_bytes(2, 4097), 4097 * 2304 + 1025 * 256);
        assert_eq!(kv.layer_bytes(3, 4096), 0);
    }

    mod mtp {
        use std::sync::Arc;

        use gguf::GgmlType;
        use models::{
            Act, Borrows, Ffn, Gqa, HcKind, HcSpec, HeadRows, LayerSpec, Mixer, Moe, MtpDraft,
            MtpHeadNorm, MtpInput, MtpSource, Residual, Rope, RopeMode, Router, Score, Shared,
        };

        use super::super::{
            FileTensor, HEAD_ROWS_FORMAT, HeadRowsError, MTP_HEAD_ROWS, MtpInputs, MtpLoad,
            mtp_tensors, mtp_widened, parse_head_rows, vocab_sha256,
        };
        use crate::arch::synthetic::header;
        use crate::fileio::hex;
        use crate::placement::{self, CardFormat, Device, PlanLevers, Role};

        /// Qwen3.8's MTP layer as `mtp_of` reads the shared file.
        fn draft(head_rows: HeadRows) -> MtpDraft {
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
        fn file() -> Vec<FileTensor> {
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

        // PIN(2026-09-28): the MTP layer's card bytes [derived from the header dump's table: every
        // Q8_0 tensor but the injects in two planes of its file bytes, 2,766,827,520 B, of which
        // the routed stacks 3 x 891,289,600; the F32 tensors but the indexer's norms, 5,429,248 B;
        // the two injects widened to F32 [10240, 4], 2 x 163,840; so 98,715,648 dense and
        // 2,673,868,800 of experts, 2,772,584,448 resident; the uploads in the file's order, then
        // the injects, through the 2 MiB-granule heap: 2,785,017,856 B, rounding 12,433,408. The
        // store: 2 x 2 x 256 f16 a position. A list of 40,960 rows: 40,960 x 2,720 = 111,411,200
        // B of rows, the heap 2,898,264,064, and 40,960 x 4 B of map].
        const DENSE: u64 = 98_715_648;
        const EXPERTS: u64 = 2_673_868_800;
        const ROUNDING: u64 = 12_433_408;
        const KV_4096: u64 = 8_388_608;
        const LIST_ROWS: usize = 40_960;
        const LIST_HEAD: u64 = 111_411_200;
        const LIST_ROUNDING: u64 = 14_268_416;
        const LIST_MAP: u64 = 163_840;

        fn list(n: usize) -> HeadRows {
            HeadRows::List {
                ids: (0..n as u32).map(|i| i * 6).collect::<Vec<_>>().into(),
                digest: [7; 32],
            }
        }

        /// The draft's plan on its own card at 4,096 positions: every tensor
        /// once, the loaded ones on the card, the unread and the widened
        /// files' tensors nowhere, all 512 experts on the card; its bytes as
        /// derived, with the full head and with a list of 40,960 rows.
        #[test]
        fn mtp_card_bytes() {
            for (rows, head, rounding, map) in [
                (HeadRows::Full, 0, ROUNDING, 0),
                (list(LIST_ROWS), LIST_HEAD, LIST_ROUNDING, LIST_MAP),
            ] {
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
                    (DENSE + head, EXPERTS, rounding, KV_4096),
                    "list {with_list}"
                );
                assert_eq!(m.map_bytes, map);
                assert_eq!(plan.rows.len(), 32 + 2 + usize::from(with_list));
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
                assert_eq!(
                    m.model.tensors.iter().any(|t| t.name == MTP_HEAD_ROWS),
                    with_list
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
}
