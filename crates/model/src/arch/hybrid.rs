//! The Nemotron-H hybrid trunk's common pieces, which `nemotron_h` and
//! `nemotron_h_moe` read their files through: the Mamba-2 and no-rope
//! attention layers' tensor names, role stems, widths and dims, the layer
//! kinds from the two per-layer arrays, the tensors each kind carries, the
//! two mixers' descriptions, and the whole-card plan of a file of either
//! family ([`PlanInputs`]). A family's own feed-forward layer (the dense MLP,
//! the latent MoE) stays in the family. The dialect's authority is
//! llama.cpp's `src/models/nemotron-h.cpp`, which loads both.

use gguf::Split;
use models::{Act, Ffn, Gqa, Mamba2, ModelSpec, OutGate, Rope, RopeMode, ToolFormat};
use runtime::stores::{dense_kv_bytes, mamba2_bytes};

use crate::arch::coverage;
use crate::arch::names::{output, output_norm, token_embd};
use crate::arch::stems::{Stem, opt, req, required};
use crate::arch::{meta_usize, metadata, mul, optional_usize, positive, spec_u32};
use crate::placement::accepts::{program_of, routed_rule};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::{
    self, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Role, Unimplemented,
    Violation, checked, joined,
};

// ------------------------------------------------------------------ names

/// The block RMS gain. Every layer carries it, whatever the layer mixes.
pub const ATTN_NORM: &str = "attn_norm.weight";

/// Mamba-2 `A`, stored without a `.weight` suffix.
pub const SSM_A: &str = "ssm_a";
/// Mamba-2 `D`.
pub const SSM_D: &str = "ssm_d";
/// The conv's bias. The loader marks it not required.
pub const SSM_CONV_BIAS: &str = "ssm_conv1d.bias";
/// The conv's weight.
pub const SSM_CONV: &str = "ssm_conv1d.weight";
/// The time-step bias.
pub const SSM_DT_BIAS: &str = "ssm_dt.bias";
/// The input projection. Its presence marks a Mamba-2 layer.
pub const SSM_IN: &str = "ssm_in.weight";
/// The grouped norm.
pub const SSM_NORM: &str = "ssm_norm.weight";
/// The output projection.
pub const SSM_OUT: &str = "ssm_out.weight";

/// The query projection. Its presence marks an attention layer.
pub const ATTN_Q: &str = "attn_q.weight";
/// The key projection.
pub const ATTN_K: &str = "attn_k.weight";
/// The value projection.
pub const ATTN_V: &str = "attn_v.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";

// ------------------------------------------------------------------ stems

/// A Mamba-2 layer. The conv bias is the one stem the loader does not require.
pub(in crate::arch) const MAMBA: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(SSM_A, Role::Attention),
    req(SSM_D, Role::Attention),
    req(SSM_DT_BIAS, Role::Attention),
    req(SSM_CONV, Role::Attention),
    opt(SSM_CONV_BIAS, Role::Attention),
    req(SSM_IN, Role::Attention),
    req(SSM_NORM, Role::Attention),
    req(SSM_OUT, Role::Attention),
];

/// An attention layer. No rope table is among the stems.
pub(in crate::arch) const ATTENTION: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(ATTN_Q, Role::Attention),
    req(ATTN_K, Role::Attention),
    req(ATTN_V, Role::Attention),
    req(ATTN_OUTPUT, Role::Attention),
];

// ---------------------------------------------------------------- widths

/// The Mamba-2 widths, from the `ssm.*` keys and the products the projections
/// are built at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ssm {
    /// `ssm.conv_kernel`.
    pub d_conv: usize,
    /// `ssm.inner_size`.
    pub d_inner: usize,
    /// `ssm.state_size`. 128: a scan written for another width is refused.
    pub d_state: usize,
    /// `ssm.group_count`.
    pub n_groups: usize,
    /// `ssm.time_step_rank`: the scan's heads.
    pub n_heads: usize,
    /// `d_inner / n_heads`.
    pub head_dim: usize,
    /// Input-projection columns: `2 * d_inner + 2 * n_groups * d_state + n_heads`.
    pub d_in_proj: usize,
    /// Conv columns: `d_inner + 2 * n_groups * d_state`.
    pub conv_dim: usize,
}

/// The attention widths every attention layer of a Nemotron-H file shares,
/// from the `attention.*` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::arch) struct Attn {
    /// `attention.head_count`.
    pub(in crate::arch) n_head: usize,
    /// `attention.key_length`.
    pub(in crate::arch) key_len: usize,
    /// `attention.value_length`.
    pub(in crate::arch) value_len: usize,
}

impl Attn {
    /// Attention layer `l` with `heads` key and value heads: a count above 0
    /// that divides the query heads, and projections whose widths fit. Each
    /// refusal names its key.
    pub(in crate::arch) fn check_layer(
        &self,
        split: &Split,
        l: usize,
        heads: usize,
    ) -> Result<(), PlacementError> {
        super::check_kv(split, self.n_head, l, heads)?;
        mul(split, "attention.key_length", self.n_head, self.key_len)?;
        mul(split, "attention.value_length", self.n_head, self.value_len)?;
        Ok(())
    }

    /// The projections' dims of a layer of `kv` key and value heads on an
    /// `e`-value row, ggml order, the norm aside.
    pub(in crate::arch) fn dims(&self, kv: usize, e: u64) -> [(&'static str, Vec<u64>); 4] {
        let (h, kv) = (self.n_head as u64, kv as u64);
        let (k, v) = (self.key_len as u64, self.value_len as u64);
        [
            (ATTN_Q, vec![e, h * k]),
            (ATTN_K, vec![e, kv * k]),
            (ATTN_V, vec![e, kv * v]),
            (ATTN_OUTPUT, vec![h * v, e]),
        ]
    }
}

/// `attention.head_count`, `.key_length` and `.value_length`, and
/// `rope.dimension_count`: loaded, and not a reason to refuse the file —
/// the token forward does not rotate — but a missing key, or a count that
/// is no rope of the key head, is still named.
pub(in crate::arch) fn attn(split: &Split) -> Result<Attn, PlacementError> {
    let n_head = positive(split, "attention.head_count")?;
    let key_len = positive(split, "attention.key_length")?;
    let value_len = positive(split, "attention.value_length")?;
    let rope_dims = meta_usize(split, "rope.dimension_count")?;
    if rope_dims == 0 || !rope_dims.is_multiple_of(2) || rope_dims > key_len {
        return Err(metadata(
            split,
            "rope.dimension_count",
            format!("is {rope_dims}, for a key head of {key_len}"),
        ));
    }
    Ok(Attn {
        n_head,
        key_len,
        value_len,
    })
}

/// A Mamba-2 layer's dims on an `e`-value row, ggml order, the norm aside.
pub(in crate::arch) fn mamba_dims(s: &Ssm, e: u64) -> [(&'static str, Vec<u64>); 8] {
    let (heads, groups) = (s.n_heads as u64, s.n_groups as u64);
    [
        (SSM_A, vec![1, heads]),
        (SSM_D, vec![1, heads]),
        (SSM_DT_BIAS, vec![heads]),
        (SSM_CONV, vec![s.d_conv as u64, s.conv_dim as u64]),
        (SSM_CONV_BIAS, vec![s.conv_dim as u64]),
        (SSM_IN, vec![e, s.d_in_proj as u64]),
        (SSM_NORM, vec![(s.d_inner / s.n_groups) as u64, groups]),
        (SSM_OUT, vec![s.d_inner as u64, e]),
    ]
}

/// `ssm.*`. The state width is checked before any tensor is walked.
pub(in crate::arch) fn ssm(split: &Split) -> Result<Ssm, PlacementError> {
    let d_conv = positive(split, "ssm.conv_kernel")?;
    let d_inner = positive(split, "ssm.inner_size")?;
    let d_state = meta_usize(split, "ssm.state_size")?;
    if d_state != 128 {
        return Err(metadata(
            split,
            "ssm.state_size",
            format!("is {d_state}, not 128"),
        ));
    }
    let n_groups = positive(split, "ssm.group_count")?;
    let n_heads = positive(split, "ssm.time_step_rank")?;
    if !d_inner.is_multiple_of(n_heads) {
        return Err(metadata(
            split,
            "ssm.inner_size",
            format!("is {d_inner}, not divisible by ssm.time_step_rank {n_heads}"),
        ));
    }
    if !d_inner.is_multiple_of(n_groups) {
        return Err(metadata(
            split,
            "ssm.inner_size",
            format!("is {d_inner}, not divisible by ssm.group_count {n_groups}"),
        ));
    }
    let groups_state = mul(split, "ssm.group_count", n_groups, d_state)?;
    let twice_groups = mul(split, "ssm.group_count", 2, groups_state)?;
    let twice_inner = mul(split, "ssm.inner_size", 2, d_inner)?;
    let conv_dim = add(split, "ssm.inner_size", d_inner, twice_groups)?;
    let d_in_proj = add(
        split,
        "ssm.inner_size",
        add(split, "ssm.inner_size", twice_inner, twice_groups)?,
        n_heads,
    )?;
    Ok(Ssm {
        d_conv,
        d_inner,
        d_state,
        n_groups,
        n_heads,
        head_dim: d_inner / n_heads,
        d_in_proj,
        conv_dim,
    })
}

/// `a + b`, or the error that names `suffix`.
fn add(split: &Split, suffix: &str, a: usize, b: usize) -> Result<usize, PlacementError> {
    a.checked_add(b)
        .ok_or_else(|| metadata(split, suffix, format!("{a} + {b} does not fit")))
}

/// `nextn_predict_layers`: absent or 0. A count is refused by name. This
/// reader runs no next-token layer.
pub(in crate::arch) fn no_nextn(split: &Split) -> Result<(), PlacementError> {
    let key = "nextn_predict_layers";
    let Some(n) = optional_usize(split, key)? else {
        return Ok(());
    };
    if n != 0 {
        return Err(metadata(
            split,
            key,
            format!("is {n}; this reader runs no next-token layer"),
        ));
    }
    Ok(())
}

/// Every layer's kind from the two per-layer arrays, as the loader decides
/// it (nemotron-h.cpp:12-16, :102): both 0 is the first of `markers`'
/// kinds (Mamba-2), else a zero feed-forward length the second
/// (attention), else the third (the family's feed-forward layer). Each
/// marker's stem is the tensor whose presence marks its kind, and a layer
/// carries exactly one of the three, the one the arrays say; a layer whose
/// tensors say another kind is refused naming both kinds.
pub(in crate::arch) fn layer_kinds<K: Copy + PartialEq + std::fmt::Debug>(
    split: &Split,
    kv: &[usize],
    ff: &[usize],
    markers: [(&'static str, K); 3],
) -> Result<Vec<K>, PlacementError> {
    let [(m0, k0), (m1, k1), (m2, k2)] = markers;
    kv.iter()
        .zip(ff)
        .enumerate()
        .map(|(l, (&kv, &ff))| {
            let said = if kv == 0 && ff == 0 {
                k0
            } else if ff == 0 {
                k1
            } else {
                k2
            };
            let present = |stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
            let (a, b, c) = (present(m0), present(m1), present(m2));
            let kind = match (a, b, c) {
                (true, false, false) => k0,
                (false, true, false) => k1,
                (false, false, true) => k2,
                _ => {
                    return Err(PlacementError::Tensor {
                        name: format!("blk.{l}.{m0}"),
                        detail: format!(
                            "layer {l} needs exactly one of {m0}, {m1} and {m2} ({a} / {b} / {c})"
                        ),
                    });
                }
            };
            if kind != said {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!(
                        "with feed_forward_length makes layer {l} {said:?}, and its tensors make \
                         it {kind:?}"
                    ),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// Every layer carries the required stems of its kind's groups (`stems_of`)
/// and none of the other kinds' (`all` lists every kind) that its own do not
/// hold — `attn_norm`, which every kind holds, among those. The model-level
/// three are required. The first missing or misplaced name, with how many
/// more there are.
pub(in crate::arch) fn layers_agree<K: Copy + PartialEq>(
    split: &Split,
    kinds: &[K],
    stems_of: impl Fn(K) -> Vec<&'static [Stem]>,
    all: &[K],
) -> Result<(), PlacementError> {
    let has = |name: &str| split.find(name).is_some();
    let mut missing = Vec::new();
    let mut misplaced = Vec::new();
    for name in [token_embd(), output_norm(), output()] {
        if !has(&name) {
            missing.push(name);
        }
    }
    for (l, &kind) in kinds.iter().enumerate() {
        let layer_has = |stem: &str| has(&format!("blk.{l}.{stem}"));
        let own = stems_of(kind);
        let owns = |stem: &str| own.iter().flat_map(|g| g.iter()).any(|s| s.name == stem);
        missing.extend(
            own.iter()
                .flat_map(|g| required(g))
                .filter(|s| !layer_has(s))
                .map(|s| format!("blk.{l}.{s}")),
        );
        let mut banned: Vec<&str> = Vec::new();
        for s in all
            .iter()
            .filter(|&&k| k != kind)
            .flat_map(|&k| stems_of(k))
            .flat_map(|g| g.iter())
        {
            if !owns(s.name) && !banned.contains(&s.name) {
                banned.push(s.name);
            }
        }
        misplaced.extend(
            banned
                .iter()
                .filter(|s| layer_has(s))
                .map(|s| format!("blk.{l}.{s}")),
        );
    }
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is not in the file ({} tensor(s) the layers need are missing: {})",
                missing.len(),
                missing.join(", ")
            ),
        });
    }
    if let Some(name) = misplaced.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is in the file, and its layer's kind carries none ({} such: {})",
                misplaced.len(),
                misplaced.join(", ")
            ),
        });
    }
    Ok(())
}

// ------------------------------------------------------------ description

/// The tool-call markup the family's template teaches: `<tool_call>` with
/// `<function=…>` and `<parameter=…>`.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::QwenXml);

/// A Mamba-2 or attention layer's feed-forward block: none.
pub(in crate::arch) const NO_BLOCK: Ffn = Ffn::Dense {
    ff: 0,
    act: Act::ReluSqr,
};

/// The Mamba-2 mixer of the widths `s`.
pub(in crate::arch) fn mamba2(s: &Ssm) -> Result<Mamba2, PlacementError> {
    Ok(Mamba2 {
        d_inner: spec_u32("ssm.inner_size", s.d_inner)?,
        d_state: spec_u32("ssm.state_size", s.d_state)?,
        n_heads: spec_u32("ssm.time_step_rank", s.n_heads)?,
        head_dim: spec_u32("ssm.inner_size / ssm.time_step_rank", s.head_dim)?,
        n_groups: spec_u32("ssm.group_count", s.n_groups)?,
        d_conv: spec_u32("ssm.conv_kernel", s.d_conv)?,
        d_in_proj: spec_u32("ssm input projection", s.d_in_proj)?,
    })
}

/// A grouped-query attention mixer of `a`'s widths and `kv` key and value
/// heads that rotates no value: `rope.dimension_count` is read and is not
/// the layer's.
pub(in crate::arch) fn nope_gqa(a: &Attn, kv: usize) -> Result<Gqa, PlacementError> {
    Ok(Gqa {
        heads: spec_u32("attention.head_count", a.n_head)?,
        kv_heads: spec_u32("attention.head_count_kv", kv)?,
        head_dim: spec_u32("attention.key_length", a.key_len)?,
        value_dim: spec_u32("attention.value_length", a.value_len)?,
        rope: Rope {
            mode: RopeMode::None,
            dims: 0,
            base: 0.0,
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

// ------------------------------------------------------------- placement

/// What a plan of a Nemotron-H file of either family is made from, read from
/// its headers by the family (`nemotron_h::place::describe`,
/// `nemotron_h_moe::place::describe`): every tensor with its role, the typed
/// description and each layer's store bytes. The placement
/// ([`PlanInputs::plan`]) puts the whole model on one card: the body runs
/// every layer there, its routed experts too, so no tier and no host leg has
/// anything to hold; it refuses by name a machine of more than one card or
/// with a tier card, a context of no position, a context whose stores the
/// card cannot hold, and a card too small for every routed expert, which
/// would leave some on the host. [`PlanInputs::plan_with`] takes the other
/// two places a plan may put the routed experts ([`Experts`]): the card's
/// share with the rest on the host tier, or all on the host. The dense
/// layers, the stores and the refusals above stay the card's under every
/// arm.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    /// Every tensor with its role, and the model's layer count.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's store bytes.
    pub kv: KvLayout,
    /// The card bytes the load makes beside the plan's terms, its prompt pass's buffers: the
    /// card keeps them out of the expert budget ([`placement::plan_routed_reserving`]) and the
    /// body holds its pass to them. 0 for a family whose body sizes its pass apart from the plan.
    pub prompt_reserve: u64,
}

/// Where a plan puts the routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Experts {
    /// Every routed expert on the one card; a card that cannot hold them all
    /// is refused.
    Card,
    /// Each routed layer's id prefix `[0, n_l)` on the card as its budget
    /// holds, the rest on the host tier.
    Placed,
    /// Every routed expert on the host tier: `n_l = 0` on every layer.
    Host,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// A machine with an expert tier card: the body runs every layer on its
    /// one card, so the tier would hold nothing.
    #[error(
        "the machine has the expert tier card {card} (tier {tier}); the {arch} body runs every layer on one card, so a tier would hold nothing"
    )]
    Tier {
        /// The tier card's name.
        card: String,
        /// Its index among the machine's tiers.
        tier: usize,
        /// The file's architecture.
        arch: &'static str,
    },
    /// A machine of other than one card: the body runs its chain on one.
    #[error("the placement puts the layers on {0} cards; the {1} chain runs on one")]
    Cards(usize, &'static str),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
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

impl PlanInputs {
    /// `self`, or — for a file with a feature the engine does not run —
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<PlanInputs, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`: the whole model on the machine's one card,
    /// each routed stack in the format its program's rule reads
    /// ([`Experts::Card`]). A file no program runs is refused with its
    /// coverage list.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with(machine, ctx_max, levers, Experts::Card)
    }

    /// [`PlanInputs::plan`] with the routed experts where `experts` says:
    /// the context, tier, card-count and program refusals hold under every
    /// arm. A file with no routed stacks plans the same under every arm.
    pub fn plan_with<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        experts: Experts,
    ) -> Result<Plan<'a>, PlaceError> {
        let arch = self.spec.arch.name();
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        if let Some((tier, card)) = machine.expert_tiers().enumerate().next() {
            return Err(PlaceError::Tier {
                card: card.name.clone(),
                tier,
                arch,
            });
        }
        if machine.cards.len() != 1 {
            return Err(PlaceError::Cards(machine.cards.len(), arch));
        }
        let program = program_of(self.spec.arch)
            .ok_or_else(|| PlacementError::Unimplemented(self.unimplemented()))?;
        let plan = match experts {
            Experts::Card | Experts::Placed => placement::plan_routed_reserving(
                &self.model,
                machine,
                ctx_max,
                &self.kv,
                levers,
                routed_rule(program),
                self.prompt_reserve,
            )?,
            Experts::Host => placement::plan_host_routed_reserving(
                &self.model,
                machine,
                ctx_max,
                &self.kv,
                levers,
                self.prompt_reserve,
            )?,
        };
        // Under `Card` the body uploads every expert stack, so the host leg
        // must hold none.
        if experts == Experts::Card && plan.host.experts != 0 {
            return Err(PlacementError::Experts(format!(
                "{arch} requires every expert on one card; {} experts remain on the host",
                plan.host.experts
            ))
            .into());
        }
        checked(plan).map_err(PlaceError::Broken)
    }
}

/// What a layer's mixer holds per sequence, whatever the family calls the
/// layer: the family maps its own kinds to these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holds {
    /// A Mamba-2 layer: its scan state and conv ring.
    Scan,
    /// An attention layer of `kv_heads` key and value heads.
    Attention { kv_heads: usize },
    /// A feed-forward layer (the dense MLP, the latent MoE): nothing.
    Nothing,
}

/// What one layer holds per sequence, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Store {
    /// An attention layer: K and V of this many bytes a position.
    Rows(u64),
    /// A Mamba-2 layer: this many bytes of scan state and conv ring, whatever
    /// the context.
    State(u64),
    /// A feed-forward layer.
    Nothing,
}

/// Each layer's store bytes of a Nemotron-H file: the bytes the body's
/// allocations hold ([`runtime::stores`] is their owner).
#[derive(Clone, Debug)]
pub struct KvLayout {
    layers: Vec<Store>,
}

impl KvLayout {
    /// The layout of `layers` (in layer order) of a file of Mamba-2 widths
    /// `ssm` and attention keys of `key_len` values.
    #[must_use]
    pub fn of(ssm: &Ssm, key_len: usize, layers: impl IntoIterator<Item = Holds>) -> KvLayout {
        let state = mamba2_bytes(ssm.d_inner, ssm.d_state, ssm.d_conv, ssm.conv_dim);
        let layers = layers
            .into_iter()
            .map(|h| match h {
                Holds::Scan => Store::State(state),
                Holds::Attention { kv_heads } => Store::Rows(dense_kv_bytes(kv_heads, key_len, 1)),
                Holds::Nothing => Store::Nothing,
            })
            .collect();
        KvLayout { layers }
    }
}

impl KvBytes for KvLayout {
    /// An attention layer's K and V of `ctx_max` positions, a Mamba-2
    /// layer's state, a feed-forward layer's nothing; 0 past the last layer.
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.layers.get(layer) {
            Some(Store::Rows(row)) => row.saturating_mul(ctx_max),
            Some(Store::State(bytes)) => *bytes,
            Some(Store::Nothing) | None => 0,
        }
    }
}
