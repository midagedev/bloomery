//! The Nemotron-H hybrid trunk's common pieces, which `nemotron_h` and
//! `nemotron_h_moe` read their files through: the Mamba-2 and no-rope
//! attention layers' tensor names, role stems, widths and dims, the layer
//! kinds from the two per-layer arrays, the tensors each kind carries, and
//! the two mixers' descriptions. A family's own feed-forward layer (the
//! dense MLP, the latent MoE) stays in the family. The dialect's authority
//! is llama.cpp's `src/models/nemotron-h.cpp`, which loads both.

use gguf::Split;
use models::{Act, Ffn, Gqa, Mamba2, Rope, RopeMode, ToolFormat};

use crate::arch::names::{output, output_norm, token_embd};
use crate::arch::stems::{Stem, opt, req, required};
use crate::arch::{meta_f32, meta_usize, metadata, mul, optional_usize, positive, spec_u32};
use crate::placement::{PlacementError, Role};

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
        if heads == 0 || !self.n_head.is_multiple_of(heads) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!("is {heads} at layer {l} for {} query heads", self.n_head),
            ));
        }
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

/// `attention.layer_norm_rms_epsilon`: every block norm's, a finite positive
/// value.
pub(in crate::arch) fn rms_eps(split: &Split) -> Result<f32, PlacementError> {
    let key = "attention.layer_norm_rms_epsilon";
    let eps = meta_f32(split, key)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(metadata(
            split,
            key,
            format!("is {eps}, not a finite positive epsilon"),
        ));
    }
    Ok(eps)
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
        },
        qk_norm: false,
        out_gate: false,
        select: None,
        window: None,
        sinks: false,
        value_scale: None,
    })
}
