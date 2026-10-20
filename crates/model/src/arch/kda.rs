//! Kimi delta attention (KDA): the layer's model-side facts, common to every
//! family whose layers mix with it — the tensor stems a KDA layer carries,
//! the dims each stem holds and the keys that fix them. The dialect's
//! authority is ik_llama.cpp's `src/llama-kda.cpp`, which loads it.
//!
//! What stays with a family: what a KDA layer decays by (the gate's own
//! parameter), which of its layers is a KDA one, and the plan that places
//! them. Those are the family's facts; the mixer's shape is the
//! architecture's.

use gguf::Split;

use crate::arch::stems::{Stem, req};
use crate::arch::{meta_usize, metadata, optional_usize, positive};
use crate::placement::{PlacementError, Role};

/// A KDA delta-rule layer's mixer; `ssm_f_b` and `ssm_g_b` are required
/// because without them ik takes another decay rule (`src/llama-kda.cpp`,
/// `build_kda_beta_gate`).
pub const STEMS: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
    req("ssm_conv1d_q.weight", Role::Attention),
    req("ssm_conv1d_k.weight", Role::Attention),
    req("ssm_conv1d_v.weight", Role::Attention),
    req("ssm_f_a.weight", Role::Attention),
    req("ssm_f_b.weight", Role::Attention),
    req("ssm_g_a.weight", Role::Attention),
    req("ssm_g_b.weight", Role::Attention),
    req("ssm_beta.weight", Role::Attention),
    req("ssm_a", Role::Attention),
    req("ssm_dt.bias", Role::Attention),
    req("ssm_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// The KDA numbers a file's keys give: the conv's taps and the head width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keys {
    /// `ssm.conv_kernel`: the depthwise conv's taps.
    pub conv: usize,
    /// `kda.head_dim`: the values per key and value head.
    pub head_dim: usize,
}

/// The KDA keys of `split`, read under the file's own architecture prefix:
/// `ssm.conv_kernel`, `kda.head_dim` (with `ssm.state_size` as its
/// fallback) and the `ssm.group_count` check against the head count. A
/// default taken is recorded in `defaults` with the line that sets it.
pub(in crate::arch) fn read(
    split: &Split,
    n_head: usize,
    defaults: &mut Vec<String>,
) -> Result<Keys, PlacementError> {
    let conv = positive(split, "ssm.conv_kernel")?;
    let head_dim = match optional_usize(split, "kda.head_dim")? {
        Some(v) => v,
        None => {
            let v = meta_usize(split, "ssm.state_size")?;
            defaults.push(format!(
                "kda.head_dim = {v} (ssm.state_size, src/llama-hparams.cpp:2370-2374)"
            ));
            v
        }
    };
    if head_dim == 0 {
        return Err(metadata(split, "kda.head_dim", "is 0"));
    }
    if let Some(g) = optional_usize(split, "ssm.group_count")?
        && g != n_head
    {
        return Err(metadata(
            split,
            "ssm.group_count",
            format!("is {g}; ik's KDA layer takes attention.head_count {n_head} key heads"),
        ));
    }
    Ok(Keys { conv, head_dim })
}

/// A KDA layer's tensors by stem and the dims they hold, ggml order:
/// `d_state` is the head dim, `d_inner` that times the heads.
pub(in crate::arch) fn dims(
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    conv: usize,
) -> [(&'static str, Vec<u64>); 16] {
    let (e, h, st) = (n_embd as u64, n_head as u64, head_dim as u64);
    let inner = st * h;
    let conv = conv as u64;
    [
        ("attn_norm.weight", vec![e]),
        ("attn_q.weight", vec![e, inner]),
        ("attn_k.weight", vec![e, inner]),
        ("attn_v.weight", vec![e, inner]),
        ("ssm_conv1d_q.weight", vec![conv, 1, inner]),
        ("ssm_conv1d_k.weight", vec![conv, 1, inner]),
        ("ssm_conv1d_v.weight", vec![conv, 1, inner]),
        ("ssm_f_a.weight", vec![e, st]),
        ("ssm_f_b.weight", vec![st, inner]),
        ("ssm_g_a.weight", vec![e, st]),
        ("ssm_g_b.weight", vec![st, inner]),
        ("ssm_beta.weight", vec![e, h]),
        ("ssm_a", vec![h]),
        ("ssm_dt.bias", vec![inner]),
        ("ssm_norm.weight", vec![st]),
        ("attn_output.weight", vec![inner, e]),
    ]
}

/// Every KDA layer of `layers` holds exactly the dims `want` gives, as ik
/// requires: the kernels check their buffers' lengths only from below, so a
/// short tensor would leave the shared scratch's tail to the last layer and
/// a long one would be read in part. The first tensor of other dims, by
/// name, with how many more there are.
pub(in crate::arch) fn dims_agree(
    split: &Split,
    layers: &[usize],
    want: &[(&'static str, Vec<u64>)],
) -> Result<(), PlacementError> {
    let pad = |d: &[u64]| {
        let mut d = d.to_vec();
        while d.last() == Some(&1) {
            d.pop();
        }
        d
    };
    let mut off = Vec::new();
    for &l in layers {
        for (stem, dims) in want {
            let name = format!("blk.{l}.{stem}");
            if let Some((_, info)) = split.find(&name)
                && pad(&info.dims) != pad(dims)
            {
                off.push((name, format!("{:?} (want {dims:?})", info.dims)));
            }
        }
    }
    match off.first() {
        None => Ok(()),
        Some((name, _)) => Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "has other dims than ik creates the KDA tensor with ({} such: {})",
                off.len(),
                off.iter()
                    .map(|(n, d)| format!("{n} {d}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}
