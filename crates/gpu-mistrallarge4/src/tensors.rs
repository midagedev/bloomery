//! Each layer's attention tensors, built at load from the file's own arrays:
//! the kind every array gives the layer (its rope, its indexer's, its cache's
//! owner), then the projections as [`Site`]s (the gemv machinery's own type
//! check), the f32 vectors as resident names and the two vectors the
//! load-time row move rewrites as the moved device buffers themselves. Every
//! tensor's type is the file's own, read per tensor and per layer from the
//! common tables; a type with no entry is refused by name at load. The
//! feed-forward half's tensors are the common routed block's
//! ([`crate::ffn::Routed`]).

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
use model::arch::mistrallarge4::sites::{
    SelUp, UpProj, k_b as k_b_entry, launch, v_b as v_b_entry,
};
use model::arch::names::blk;

const WHAT: &str = "mistrallarge4 tensors";

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
        }
    }
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

/// How `attn_v_b` up-projects each head's latent output, by the file's type
/// ([`UpProj`]).
pub(crate) enum VUp {
    /// The file's Q8_0 planes, `heads · head_v` rows of the latent's width,
    /// through the per-head q8_0 gemv.
    Planes(String),
    /// A stack of `heads · head_v` rows, the heads its experts, through the
    /// down `_sel` launch of `family` (`ty` the file's type, which the
    /// i-quant launch takes).
    Sel {
        name: String,
        ty: GgmlType,
        family: SelUp,
    },
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
    /// `attn_k_b`, the per-head absorption into the latent: q8_0 planes.
    pub k_b: String,
    /// `attn_v_b`, the per-head up-projection of the latent to the values.
    pub v_b: VUp,
    /// `attn_output`.
    pub o: Site,
    pub idx: Option<Idx>,
}

/// The sites of layer `l`'s attention projections, for the load's activation forms.
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

/// The model-level tensors' launches, refused by name at load: the embedding
/// rows and the head of the file's own types.
pub(crate) fn globals(file: &Split) -> Result<(), GpuError> {
    for name in [stems::token_embd(), stems::output()] {
        let (_, t) = file
            .find(&name)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{name} is not in the model file")))?;
        launch(&name, t.ty).map_err(|e| GpuError::shape(WHAT, e))?;
    }
    Ok(())
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

/// Layer `l`'s attention tensors: its kind from `hp`'s arrays, every projection a site
/// of the type the file holds (refused by name where the common site table has
/// no entry for it), the per-head pair through its own entries, the f32
/// vectors resident names and the k-norm pair moved.
#[allow(clippy::too_many_lines, reason = "one branch a tensor, nothing shared")]
pub(crate) fn layer(
    (file, w): (&Split, &Weights),
    hp: &Hparams,
    stream: &CudaStream,
    l: usize,
) -> Result<Layer, GpuError> {
    let kind = Kind::of(hp, l);
    let (hidden, heads, head_k, head_v, kv_lora) =
        (hp.n_embd, hp.n_head, hp.head_k, hp.head_v, hp.kv_lora);
    let pe = hp.pe(l);
    // A site takes every family the card's dense table names, so the file's
    // type alone picks the launch.
    let site = |stem: &str, rows: usize, k: usize| {
        Site::of_file((file, w), WHAT, blk(l, stem), (rows, k), SiteTy::ALL)
    };
    let vec_name = |stem: &str, len: usize| vector(w, WHAT, blk(l, stem), len);
    let file_ty = |name: &str| -> Result<GgmlType, GpuError> {
        file.find(name)
            .map(|(_, t)| t.ty)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{name} is not in the model file")))
    };
    let q8_planes = |name: &str, rows: usize, k: usize| -> Result<String, GpuError> {
        match w.resident(WHAT, name)? {
            DevWeight::Q8_0 { d, .. } if d.rows() == rows && d.cols() * 32 == k => {
                Ok(name.to_string())
            }
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
            )?,
            k: site(stems::INDEXER_ATTN_K, hp.indexer_d, hidden)?,
            k_norm_w,
            k_norm_b,
            proj: site(stems::INDEXER_PROJ, hp.indexer_heads, hidden)?,
            ssmax_w: vec_name(stems::INDEXER_SSMAX_W, hp.indexer_heads)?,
            ssmax_b: vec_name(stems::INDEXER_SSMAX_B, hp.indexer_heads)?,
        })
    } else {
        None
    };
    // The per-head pair: k_b reads Q8_0 planes alone, v_b planes or a down
    // `_sel` stack of the heads.
    let k_b = {
        let name = blk(l, stems::ATTN_K_B);
        k_b_entry(&name, file_ty(&name)?).map_err(|e| GpuError::shape(WHAT, e))?;
        q8_planes(&name, heads * kv_lora, head_k - pe)?
    };
    let v_b = {
        let name = blk(l, stems::ATTN_V_B);
        let (rows, k) = (heads * head_v, kv_lora);
        match v_b_entry(&name, file_ty(&name)?).map_err(|e| GpuError::shape(WHAT, e))? {
            UpProj::Planes => VUp::Planes(q8_planes(&name, rows, k)?),
            UpProj::Sel(family) => match w.resident(WHAT, &name)? {
                DevWeight::KQuant { ty, w: t, k: got } if t.rows() == rows && *got == k => {
                    VUp::Sel {
                        ty: *ty,
                        family,
                        name,
                    }
                }
                _ => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{name} is not a down `_sel` weight of {rows} rows of {k} values"),
                    ));
                }
            },
        }
    };
    Ok(Layer {
        kind,
        attn_norm: vec_name(stems::ATTN_NORM, hidden)?,
        ssmax_w: vec_name(stems::ATTN_SSMAX_W, heads)?,
        ssmax_b: vec_name(stems::ATTN_SSMAX_B, heads)?,
        q: site(stems::ATTN_Q, heads * head_k, hidden)?,
        kv_a: if hp.owns_cache(l) {
            Some(site(stems::ATTN_KV_A_MQA, kv_lora + pe, hidden)?)
        } else {
            None
        },
        kv_a_norm: if hp.owns_cache(l) {
            Some(vec_name(stems::ATTN_KV_A_NORM, kv_lora)?)
        } else {
            None
        },
        k_b,
        v_b,
        o: site(stems::ATTN_OUTPUT, hidden, heads * head_v)?,
        idx,
    })
}
