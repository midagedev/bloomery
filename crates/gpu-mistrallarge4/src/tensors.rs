//! Each layer's tensors, built at load from the file's own arrays: the kind
//! every array gives the layer (its rope, its indexer's, its cache's owner,
//! the dense lead), then the projections as [`Site`]s (the gemv machinery's
//! own type check), the f32 vectors as resident names and the two vectors the
//! load-time row move rewrites as the moved device buffers themselves.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::hybrid::Site;
use bloomery_gpu::site::{SiteTy, vector};
use bloomery_gpu::weights::{DevWeight, Weights};
use cuda_core::CudaStream;
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::mistrallarge4::hparams::Hparams;
use model::arch::mistrallarge4::names as stems;
use model::arch::names::blk;

/// The site types the K-quant projections read: Q6_K through the q8_1 kgemv.
const KQ_TYPES: &[SiteTy] = &[SiteTy::Q6K];
/// The site types the q8_0 projections read: the f32 rows gemv.
const Q8_TYPES: &[SiteTy] = &[SiteTy::Q8_0];
/// The site types the f32 projections read.
const F32_TYPES: &[SiteTy] = &[SiteTy::F32];

/// What the file's own arrays make a layer: the kind the whole step keys on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Kind {
    /// The layer ropes its queries and its keys (`attention.rope_pattern`).
    pub rope: bool,
    /// The layer runs its own indexer (`attention.indexer.types`); a shared
    /// layer reads the previous full one's keys and list.
    pub full: bool,
    /// The layer whose cache this layer reads (`attention.kv_source_map`;
    /// itself where it owns one).
    pub cache: usize,
    /// The layer whose index keys and list this layer reads (itself on a
    /// full layer).
    pub indexer: usize,
    /// The layer is the dense lead (`leading_dense_block_count`).
    pub dense: bool,
}

impl Kind {
    /// Layer `l`'s kind from `hp`'s arrays.
    pub(crate) fn of(hp: &Hparams, l: usize) -> Kind {
        let full = hp.indexer_full[l];
        Kind {
            rope: hp.rope_layer[l],
            full,
            cache: if hp.owns_cache(l) { l } else { hp.kv_source[l] },
            indexer: if full { l } else { hp.prev_full_indexer(l) },
            dense: l < hp.dense_lead,
        }
    }
}

/// A gated feed-forward block's three projections.
pub(crate) struct GateUpDown {
    pub gate: Site,
    pub up: Site,
    pub down: Site,
}

/// The feed-forward block: the dense lead's, or the routed stacks with the
/// router and the shared expert.
pub(crate) enum Ffn {
    Dense(GateUpDown),
    Moe {
        /// `ffn_gate_inp`, the file's F32 rows.
        router: String,
        /// The routed stacks' names, read by the `_sel` entries, the
        /// gate·up's one i-quant type and the down's own (a file may mix
        /// them between the stacks).
        gate: String,
        up: String,
        down: String,
        ty: GgmlType,
        ty_down: GgmlType,
        shared: GateUpDown,
    },
}

/// A full indexer layer's own tensors and stores.
pub(crate) struct Idx {
    /// `indexer.q_norm`, the LayerNorm's gain and bias over the hidden row.
    pub q_norm_w: String,
    pub q_norm_b: String,
    /// `indexer.attn_q_b`, the queries' projection.
    pub q_b: Site,
    /// `indexer.attn_k`, the key's projection.
    pub k: Site,
    /// `indexer.k_norm`, the LayerNorm's gain and bias over the key, its
    /// head's leading rope rows moved to the tail at load
    /// (`rows::rope_rows_to_tail`) so the key kernel turns the tail.
    pub k_norm_w: DeviceBuffer<f32>,
    pub k_norm_b: DeviceBuffer<f32>,
    /// `indexer.proj`, the head weights' projection.
    pub proj: Site,
    /// `indexer.ssmax`, the index queries' own SSMax pair.
    pub ssmax_w: String,
    pub ssmax_b: String,
}

/// One layer's tensors, by name and site; the stores (the caches, the lists,
/// the scratch) live in the body, which sizes them.
pub(crate) struct Layer {
    pub kind: Kind,
    /// `attn_norm`.
    pub attn_norm: String,
    /// `attn_ssmax`, the main queries' SSMax pair.
    pub ssmax_w: String,
    pub ssmax_b: String,
    /// `attn_q`, the whole head row a query.
    pub q: Site,
    /// `attn_kv_a_mqa` and its norm, on a layer that owns its cache.
    pub kv_a: Option<Site>,
    pub kv_a_norm: Option<String>,
    /// `attn_k_b` and `attn_v_b`, the per-head up-projections, q8_0 planes.
    pub k_b: String,
    pub v_b: String,
    /// `attn_output`.
    pub o: Site,
    pub idx: Option<Idx>,
    /// `ffn_norm`.
    pub ffn_norm: String,
    pub ffn: Ffn,
}

/// The sites of layer `l`'s projections, for the load's activation forms.
pub(crate) fn sites(layer: &Layer) -> Vec<&Site> {
    let mut out = vec![&layer.q, &layer.o];
    if let Some(kv_a) = &layer.kv_a {
        out.push(kv_a);
    }
    if let Some(idx) = &layer.idx {
        out.push(&idx.q_b);
        out.push(&idx.k);
        out.push(&idx.proj);
    }
    match &layer.ffn {
        Ffn::Dense(d) | Ffn::Moe { shared: d, .. } => {
            out.push(&d.gate);
            out.push(&d.up);
            out.push(&d.down);
        }
    }
    out
}

/// The f32 values of the file tensor `name`, `want` of them.
pub(crate) fn file_f32(file: &Split, name: &str, want: usize) -> Result<Vec<f32>, GpuError> {
    let (s, t) = file.find(name).ok_or_else(|| {
        GpuError::shape(
            "mistrallarge4 tensors",
            format!("{name} is not in the model file"),
        )
    })?;
    if t.ty != GgmlType::F32 || t.dims.iter().product::<u64>() != want as u64 {
        return Err(GpuError::shape(
            "mistrallarge4 tensors",
            format!("{name} is {:?} {:?}, want F32 x {want}", t.ty, t.dims),
        ));
    }
    let g = file.shard(s).ok_or_else(|| {
        GpuError::shape(
            "mistrallarge4 tensors",
            format!("{name}'s shard {s} is out of range"),
        )
    })?;
    Ok(g.data(t)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// The k-norm pair of layer `l`, its head's leading rope rows moved to the
/// tail (`model::arch::mistrallarge4::rows::rope_rows_to_tail` — the one
/// model-only fact of the family, run here at load): the file lays each
/// index head out `[rope | nope]` while the key kernel turns the tail.
/// A NoPE layer's index head has no rope rows, so its pair moves none.
fn moved_k_norm(
    stream: &CudaStream,
    file: &Split,
    hp: &Hparams,
    l: usize,
) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), GpuError> {
    let (mut w, mut b) = (
        file_f32(file, &blk(l, stems::INDEXER_K_NORM_W), hp.indexer_d)?,
        file_f32(file, &blk(l, stems::INDEXER_K_NORM_B), hp.indexer_d)?,
    );
    let rope = if hp.rope_layer[l] { hp.rope_dims } else { 0 };
    model::arch::mistrallarge4::rows::rope_rows_to_tail(&mut w, hp.indexer_d, rope);
    model::arch::mistrallarge4::rows::rope_rows_to_tail(&mut b, hp.indexer_d, rope);
    Ok((
        DeviceBuffer::from_host(stream, &w)?,
        DeviceBuffer::from_host(stream, &b)?,
    ))
}

/// Layer `l`'s tensors: its kind from `hp`'s arrays, every projection a site
/// of the type the file holds (refused by name where it holds another), the
/// f32 vectors resident names and the k-norm pair moved.
#[allow(clippy::too_many_lines, reason = "one branch a tensor, nothing shared")]
pub(crate) fn layer(
    (file, w): (&Split, &Weights),
    hp: &Hparams,
    stream: &CudaStream,
    l: usize,
) -> Result<Layer, GpuError> {
    const WHAT: &str = "mistrallarge4 tensors";
    let kind = Kind::of(hp, l);
    let (hidden, heads, head_k, head_v, kv_lora) =
        (hp.n_embd, hp.n_head, hp.head_k, hp.head_v, hp.kv_lora);
    let pe = hp.pe(l);
    let site = |stem: &str, rows: usize, k: usize, allowed: &[SiteTy]| {
        Site::of_file((file, w), WHAT, blk(l, stem), (rows, k), allowed)
    };
    let vec_name = |stem: &str, len: usize| vector(w, WHAT, blk(l, stem), len);
    let q8_planes = |stem: &str, rows: usize, k: usize| -> Result<String, GpuError> {
        let name = blk(l, stem);
        match w.resident(WHAT, &name)? {
            DevWeight::Q8_0 { d, .. } if d.rows() == rows && d.cols() * 32 == k => Ok(name),
            _ => Err(GpuError::shape(
                WHAT,
                format!("{name} is not a q8_0 weight of {rows} rows of {k} values"),
            )),
        }
    };
    let idx = if kind.full {
        let (k_norm_w, k_norm_b) = moved_k_norm(stream, file, hp, l)?;
        Some(Idx {
            q_norm_w: vec_name(stems::INDEXER_Q_NORM_W, hidden)?,
            q_norm_b: vec_name(stems::INDEXER_Q_NORM_B, hidden)?,
            q_b: site(
                stems::INDEXER_ATTN_Q_B,
                hp.indexer_heads * hp.indexer_d,
                hidden,
                KQ_TYPES,
            )?,
            k: site(stems::INDEXER_ATTN_K, hp.indexer_d, hidden, Q8_TYPES)?,
            k_norm_w,
            k_norm_b,
            proj: site(stems::INDEXER_PROJ, hp.indexer_heads, hidden, F32_TYPES)?,
            ssmax_w: vec_name(stems::INDEXER_SSMAX_W, hp.indexer_heads)?,
            ssmax_b: vec_name(stems::INDEXER_SSMAX_B, hp.indexer_heads)?,
        })
    } else {
        None
    };
    let ffn = if kind.dense {
        Ffn::Dense(GateUpDown {
            gate: site(stems::FFN_GATE, hp.dense_ff, hidden, Q8_TYPES)?,
            up: site(stems::FFN_UP, hp.dense_ff, hidden, Q8_TYPES)?,
            down: site(stems::FFN_DOWN, hidden, hp.dense_ff, Q8_TYPES)?,
        })
    } else {
        let stack = |stem: &str, rows: usize, k: usize| -> Result<(String, GgmlType), GpuError> {
            let name = blk(l, stem);
            match w.resident(WHAT, &name)? {
                DevWeight::KQuant { ty, w: t, k: got } if t.rows() == rows && *got == k => {
                    Ok((name, *ty))
                }
                _ => Err(GpuError::shape(
                    WHAT,
                    format!("{name} is not an i-quant weight of {rows} rows of {k} values"),
                )),
            }
        };
        let (gate, gate_ty) = stack(stems::FFN_GATE_EXPS, hp.n_expert * hp.expert_ff, hidden)?;
        let (up, up_ty) = stack(stems::FFN_UP_EXPS, hp.n_expert * hp.expert_ff, hidden)?;
        let (down, down_ty) = stack(stems::FFN_DOWN_EXPS, hp.n_expert * hidden, hp.expert_ff)?;
        if gate_ty != up_ty {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "blk.{l}'s gate stack is {gate_ty} and its up stack {up_ty}: the gate·up entry \
                     takes one type"
                ),
            ));
        }
        let shared_ff = hp.n_shared * hp.expert_ff;
        Ffn::Moe {
            router: blk(l, stems::FFN_GATE_INP),
            gate,
            up,
            down,
            ty: gate_ty,
            ty_down: down_ty,
            shared: GateUpDown {
                gate: site(stems::FFN_GATE_SHEXP, shared_ff, hidden, Q8_TYPES)?,
                up: site(stems::FFN_UP_SHEXP, shared_ff, hidden, Q8_TYPES)?,
                down: site(stems::FFN_DOWN_SHEXP, hidden, shared_ff, Q8_TYPES)?,
            },
        }
    };
    Ok(Layer {
        kind,
        attn_norm: vec_name(stems::ATTN_NORM, hidden)?,
        ssmax_w: vec_name(stems::ATTN_SSMAX_W, heads)?,
        ssmax_b: vec_name(stems::ATTN_SSMAX_B, heads)?,
        q: site(stems::ATTN_Q, heads * head_k, hidden, KQ_TYPES)?,
        kv_a: if hp.owns_cache(l) {
            Some(site(stems::ATTN_KV_A_MQA, kv_lora + pe, hidden, Q8_TYPES)?)
        } else {
            None
        },
        kv_a_norm: if hp.owns_cache(l) {
            Some(vec_name(stems::ATTN_KV_A_NORM, kv_lora)?)
        } else {
            None
        },
        k_b: q8_planes(stems::ATTN_K_B, heads * kv_lora, head_k - pe)?,
        v_b: q8_planes(stems::ATTN_V_B, heads * head_v, kv_lora)?,
        o: site(stems::ATTN_OUTPUT, hidden, heads * head_v, KQ_TYPES)?,
        idx,
        ffn_norm: vec_name(stems::FFN_NORM, hidden)?,
        ffn,
    })
}
