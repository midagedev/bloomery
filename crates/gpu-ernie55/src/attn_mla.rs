//! The latent attention of ERNIE 5.5's MLA-DSA layers (llama.cpp PR #31925: `build_mla_qk`,
//! `build_mla_out`, `build_mla_top_k`) over the common latent attention and indexer kernels. One
//! chain serves a step's token and a prompt chunk's units of columns:
//!
//! 1. the query's low-rank path: `q_a`, its norm, `q_b`; every head's tail of [`ROPE`] values
//!    turned by the token's rope row (the rope launch turns each 64-value run of a head, the
//!    gather keeps the tail's);
//! 2. the key's: `kv_a` (`[latent | k_pe]`), the latent's norm, the tail's rope and the
//!    `[k_rope | latent]` f16 row appended at the token's position, one launch;
//! 3. the selection list `[top-k of the candidates | the window]` of every token: on a selecting
//!    state ([`MlaState::selects`]) the lightning indexer scores the candidates (the key's
//!    LayerNorm and the NEOX turn of its first 64 values into the f32 key cache, the query's
//!    projection of the normed query latent and the head weights' of the input, the f32 score
//!    pass); the top-k pass and the forced window then write the list, which below the cut is the
//!    identity over the candidates, so a token's list is every key it sees;
//! 4. the absorbed query `[q_pe | k_bᵀ q_nope]` a head (`k_b` in the Q8_0 planes [`split_kv_b`]
//!    derives at load, [`derive`]), the flash with a sink a head over the list, the heads' values through
//!    `v_b`;
//! 5. the head mix (VHA) through the [`Vha`] hook, the sigmoid output gate of the normed query
//!    latent, `o_proj`.
//!
//! Every position-dependent fact is a device word or a rope row of the caller's [`SpanWords`]
//! (a unit's positions, visible counts and candidate counts, `top_k`, the tokens' rope rows), so
//! a step's chain is the same launches at every position and one capture replays it. A state of
//! `ctx` rows selects iff its last position has more candidates than `top_k` (`ctx > top_k +
//! window`); below that no score launch runs, the key cache is not written, and every token's
//! list is the identity over its keys. One list form serves every state and every unit width, so
//! a prompt chunk's rows are the step's bits. The input `x` is the layer's normed row
//! (`input_layernorm`), the output `o_proj`'s; the hyper-connection mixes stay with the trunk.

use bloomery_gpu::flash::{FlashGeom, FlashSelArgs, partials_ms_len, partials_v_len};
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::index_key::{IndexKeyKernels, IndexKeyLnNeoxArgs};
use bloomery_gpu::indexer::{
    HEAD_DIM, HEADS_F32, HIST_BINS, IndexerF32Kernels, ScoreF32Args, TopkArgs, WindowArgs,
};
use bloomery_gpu::latent::LATENT;
use bloomery_gpu::linear::HEAD;
use bloomery_gpu::linear::out_gate::{OutGateArgs, OutGateKernels};
use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
use bloomery_gpu::q8f32::Q8_0GemvHeadsMcolArgs;
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::site::{BY_NAME_COLS, ByName, ByNameKernels, Out};
use bloomery_gpu::weights::{DevWeight, TensorPart, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Window, WindowMut, capturing, copy_row};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::quant::{GgmlType, dequant_row, quantize_q8_0};
use gguf::{Place, Split, TensorInfo};
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::names;
use model::arch::models::OutGate;
use model::arch::names::{attn_k_b, attn_v_b};

use std::mem::size_of;

/// What this file's refusals name.
const WHAT: &str = "ernie55 attn_mla";
/// The values a rope turns: a head's tail, the key's beside the latent, and the span the
/// indexer's NEOX turn covers. One table row of [`ROPE`] f32 serves both.
const ROPE: usize = 64;
const _: () = assert!(model::arch::ernie55::ops::INDEX_ROPE as usize == ROPE);
/// The most cells one token's forced window holds: the window pass's block.
const MAX_WINDOW: usize = 128;
/// A quiet f16 NaN, and an f32 one: what [`MlaState::reset`] fills the stores with.
const NAN_F16: u16 = 0x7e00;
const NAN_CHUNK: usize = 1 << 20;

/// The widths the chain runs at, from the file's header, and the cache's rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    /// Every norm's ε, the indexer key's LayerNorm included.
    pub eps: f32,
    pub q_lora: usize,
    pub kv_lora: usize,
    /// A query head's values before the absorption.
    pub nope: usize,
    /// The rotated values at a head's tail, the key's beside the latent.
    pub pe: usize,
    pub heads: usize,
    pub head_v: usize,
    pub idx_heads: usize,
    pub idx_d: usize,
    pub top_k: usize,
    pub window: usize,
    pub rope_base: f32,
    /// The most rows a state holds: the scratch's widest cache.
    pub ctx: usize,
}

impl Shape {
    /// The shape of `hp` over a cache of `ctx` rows; a width no common kernel runs is refused by
    /// name.
    pub fn of(hp: &Hparams, ctx: usize) -> Result<Shape, GpuError> {
        let m = &hp.mla;
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        if ctx == 0 {
            return Err(refuse("a cache of no rows".into()));
        }
        if m.kv_lora != LATENT {
            return Err(refuse(format!(
                "kv_lora {}: the flash's latent tail is {LATENT}",
                m.kv_lora
            )));
        }
        if m.pe != ROPE {
            return Err(refuse(format!(
                "pe {}: the rope turns {ROPE} values, the rope row's width the indexer's turn shares",
                m.pe
            )));
        }
        if m.nope() == 0 || !m.nope().is_multiple_of(32) {
            return Err(refuse(format!(
                "nope {}: k_b's Q8_0 blocks run along it, a positive multiple of 32",
                m.nope()
            )));
        }
        if m.head_v == 0 || !m.head_v.is_multiple_of(HEAD) {
            return Err(refuse(format!(
                "v {}: the output gate runs whole {HEAD}-value heads",
                m.head_v
            )));
        }
        if hp.heads == 0 || hp.n_embd == 0 || !hp.n_embd.is_multiple_of(128) {
            return Err(refuse(format!(
                "heads {} and hidden {}: at least one head, and a hidden width the K-quant \
                 sites' q8_1 forms take (a multiple of 128)",
                hp.heads, hp.n_embd
            )));
        }
        if m.q_lora == 0 || !m.q_lora.is_multiple_of(128) {
            return Err(refuse(format!(
                "q_lora {}: a positive multiple of 128, the K-quant sites' q8_1 forms",
                m.q_lora
            )));
        }
        if hp.indexer.heads != HEADS_F32 || hp.indexer.d != HEAD_DIM {
            return Err(refuse(format!(
                "an indexer of {} heads of {}: the f32 score pass runs {HEADS_F32} of {HEAD_DIM}",
                hp.indexer.heads, hp.indexer.d
            )));
        }
        if hp.indexer.top_k == 0 {
            return Err(refuse("top_k of no row".into()));
        }
        if hp.hca.window == 0 || hp.hca.window > MAX_WINDOW {
            return Err(refuse(format!(
                "window {}: the forced window's pass runs 1..={MAX_WINDOW} cells a token",
                hp.hca.window
            )));
        }
        Ok(Shape {
            hidden: hp.n_embd,
            eps: hp.rms_eps,
            q_lora: m.q_lora,
            kv_lora: m.kv_lora,
            nope: m.nope(),
            pe: m.pe,
            heads: hp.heads,
            head_v: m.head_v,
            idx_heads: hp.indexer.heads,
            idx_d: hp.indexer.d,
            top_k: hp.indexer.top_k,
            window: hp.hca.window,
            rope_base: hp.rope.base,
            ctx,
        })
    }

    /// A query head's values: the no-position ones, then the rotated tail.
    #[must_use]
    pub fn head_k(&self) -> usize {
        self.nope + self.pe
    }

    /// A cache row: the rotated tail, then the latent.
    #[must_use]
    pub fn row(&self) -> usize {
        ROPE + self.kv_lora
    }

    /// The score scale `1/√(nope + pe)`.
    #[must_use]
    pub fn kq_scale(&self) -> f32 {
        1.0 / (self.head_k() as f32).sqrt()
    }

    /// The selection list's width: the top-k and the forced window.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.top_k + self.window
    }

    /// The bytes one layer's stores take for one sequence ([`MlaState`]): `ctx` cache rows in f16
    /// and `ctx` indexer keys of [`HEAD_DIM`] values in f32 — what the plan's latent hold counts.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.ctx * (self.row() * size_of::<u16>() + HEAD_DIM * size_of::<f32>())
    }

    /// The bytes of the working buffers one sequence's latent layers share, for units of `unit`
    /// columns ([`Scratch::new`]'s lengths and [`SpanWords`] of the same span): the chain's
    /// intermediates, the flash partials over a list's width, the indexer's scores over the
    /// context, histogram, list and counts. The layers run one after another on one stream, so one
    /// set serves all of them.
    #[must_use]
    pub fn scratch_bytes(&self, unit: usize) -> usize {
        let u = unit;
        let f32s = 2 * u * self.q_lora
            + 2 * u * self.heads * self.head_k()
            + u * self.heads * self.row()
            + 3 * u * self.row()
            + u * self.idx_heads * self.idx_d
            + u * self.idx_d
            + u * self.idx_heads
            + partials_v_len(u * self.heads, self.stride())
            + partials_ms_len(u * self.heads, self.stride())
            + u * self.heads * self.kv_lora
            + 4 * u * self.heads * self.head_v
            + u * self.ctx
            + u * ROPE;
        let u32s = u * HIST_BINS + u * self.stride() + u + 3 * u + 1;
        size_of::<f32>() * f32s + size_of::<u32>() * u32s
    }
}

/// One latent layer's tensor names: the eight by-name projections of the file, the fused
/// per-head up projection the load splits, its two derived halves, and the f32 vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sites {
    pub q_a: String,
    pub q_b: String,
    pub kv_a: String,
    /// The file's fused `[kv_lora, heads · (nope + v)]`: the key and value halves, a head's rows
    /// together, read on the host at load only.
    pub kv_b: String,
    /// The derived key-up half, `heads · kv_lora` Q8_0 rows of `nope`.
    pub k_b: String,
    /// The derived value half, `heads · v` Q8_0 rows of `kv_lora`.
    pub v_b: String,
    pub gate: String,
    pub o: String,
    pub q_a_norm: String,
    pub kv_a_norm: String,
    pub sinks: String,
    /// The indexer's query projection, of the normed query latent.
    pub idx_q: String,
    /// The indexer key's projection, of the layer input.
    pub idx_k: String,
    /// The indexer head weights' projection, of the layer input.
    pub idx_w: String,
    pub idx_k_norm: String,
    pub idx_k_bias: String,
}

impl Sites {
    /// Layer `l`'s names.
    #[must_use]
    pub fn of(l: usize) -> Sites {
        Sites {
            q_a: names::attn_q_a(l),
            q_b: names::attn_q_b(l),
            kv_a: names::attn_kv_a_mqa(l),
            kv_b: names::attn_kv_b(l),
            k_b: attn_k_b(l),
            v_b: attn_v_b(l),
            gate: names::attn_gate(l),
            o: names::attn_output(l),
            q_a_norm: names::attn_q_a_norm(l),
            kv_a_norm: names::attn_kv_a_norm(l),
            sinks: names::attn_sinks(l),
            idx_q: names::indexer_attn_q_b(l),
            idx_k: names::indexer_attn_k(l),
            idx_w: names::indexer_proj(l),
            idx_k_norm: names::indexer_k_norm(l),
            idx_k_bias: names::indexer_k_norm_bias(l),
        }
    }

    /// The by-name projections with the file dims each must have: `(name, k, rows)`.
    fn by_name(&self, sh: &Shape) -> [(&str, usize, usize); 8] {
        [
            (&self.q_a, sh.hidden, sh.q_lora),
            (&self.q_b, sh.q_lora, sh.heads * sh.head_k()),
            (&self.kv_a, sh.hidden, sh.row()),
            (&self.gate, sh.q_lora, sh.heads * sh.head_v),
            (&self.o, sh.heads * sh.head_v, sh.hidden),
            (&self.idx_q, sh.q_lora, sh.idx_heads * sh.idx_d),
            (&self.idx_k, sh.hidden, sh.idx_d),
            (&self.idx_w, sh.hidden, sh.idx_heads),
        ]
    }

    /// The f32 vectors with the least values each must hold: `(name, len)`.
    fn vectors(&self, sh: &Shape) -> [(&str, usize); 5] {
        [
            (&self.q_a_norm, sh.q_lora),
            (&self.kv_a_norm, sh.kv_lora),
            (&self.sinks, sh.heads),
            (&self.idx_k_norm, sh.idx_d),
            (&self.idx_k_bias, sh.idx_d),
        ]
    }
}

/// One latent layer the component runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    /// The layer's number in the file: its fault sink's layer, and the [`Vha`] hook's.
    pub layer: usize,
    pub sites: Sites,
}

/// The derived halves of one layer's `attn_kv_b`: each its header entry as
/// [`Weights::add_parts`] reads it, and its rows' Q8_0 bytes.
pub struct KvHalves {
    pub parts: [(TensorInfo, Vec<u8>); 2],
}

/// `sh`'s split of layer `sites`' fused `attn_kv_b` (`[kv_lora, heads · (nope + v)]`, a head's
/// key-up rows then its value rows) into the Q8_0 planes the absorbed projections read:
/// - `k_b`: `heads · kv_lora` rows of `nope`, row `h · kv_lora + j` the head's key-up column `j`
///   over its `nope` values, the blocks along that axis (the layout V3's derived weight has, so
///   `enqueue_q8_0_gemv_heads` gives `k_bᵀ q_nope` a head);
/// - `v_b`: `heads · v` rows of `kv_lora`, a file row each.
///
/// Every file row is dequantized whole, by the file's own type, before a half takes it: a block
/// grid that runs along the rows (a head's `nope + v` rows are three and a half 128-row blocks
/// at ERNIE's widths, and the halves cut at `nope`) is never sliced — only values are. A tensor
/// of another shape is refused by name.
pub fn split_kv_b(sh: &Shape, file: &Split, sites: &Sites) -> Result<KvHalves, GpuError> {
    let info = kv_b_info(sh, file, sites)?;
    split_rows(sh, sites, info.ty, file.data(&sites.kv_b)?)
}

/// The header entry of layer `sites`' fused `attn_kv_b` in `file`, `[kv_lora, heads · (nope + v)]`
/// at `sh`'s widths; another shape, or a tensor the file lacks, is refused by name.
fn kv_b_info<'a>(sh: &Shape, file: &'a Split, sites: &Sites) -> Result<&'a TensorInfo, GpuError> {
    let (k, rows) = (sh.kv_lora, sh.heads * (sh.nope + sh.head_v));
    let Some((_, info)) = file.find(&sites.kv_b) else {
        return Err(GpuError::tensor(WHAT, &sites.kv_b, "a tensor of the file"));
    };
    if info.dims.len() != 2 || info.dims[0] != k as u64 || info.dims[1] != rows as u64 {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{} has dims {:?}, want [{k}, {rows}]: kv_lora by heads · (nope + v)",
                sites.kv_b, info.dims
            ),
        ));
    }
    Ok(info)
}

/// The file blocks of the resident weight `name` and their type, read back from its two planes
/// and regrouped as a file stores them: a Q8_0 block's f16 scale then its 32 codes, an FP8_B128
/// block's f32 scale then its 128 E4M3 bytes. These are the bytes of a tensor a step makes
/// ([`Place::Made`]), which the load uploaded and no file run holds. Another resident format,
/// and planes of another length than whole blocks, are refused by name.
fn resident_blocks(
    stream: &CudaStream,
    w: &Weights,
    name: &str,
) -> Result<(GgmlType, Vec<u8>), GpuError> {
    let refuse = |detail: String| GpuError::shape(WHAT, format!("{name}: {detail}"));
    let (ty, words, scales, per) = match w.resident(WHAT, name)? {
        DevWeight::Q8_0 { qs, d, .. } => {
            let d = d.buf().to_host_vec(stream)?;
            let scales: Vec<u8> = d.iter().flat_map(|s| s.to_le_bytes()).collect();
            (
                GgmlType::Q8_0,
                qs.buf().to_host_vec(stream)?,
                scales,
                (2, 8),
            )
        }
        DevWeight::Fp8B128 { qs, e, .. } => {
            let e = e.buf().to_host_vec(stream)?;
            let scales: Vec<u8> = e.iter().flat_map(|s| s.to_le_bytes()).collect();
            (
                GgmlType::FP8_B128,
                qs.buf().to_host_vec(stream)?,
                scales,
                (4, 32),
            )
        }
        _ => {
            return Err(refuse(
                "resident in neither Q8_0 nor FP8_B128 planes".into(),
            ));
        }
    };
    let (scale_bytes, block_words) = per;
    let blocks = scales.len() / scale_bytes;
    if words.len() != blocks * block_words {
        return Err(refuse(format!(
            "{} code words for {blocks} block scales, not {block_words} a block",
            words.len()
        )));
    }
    let mut out = Vec::with_capacity(blocks * (scale_bytes + 4 * block_words));
    for (scale, codes) in scales
        .chunks_exact(scale_bytes)
        .zip(words.chunks_exact(block_words))
    {
        out.extend_from_slice(scale);
        for word in codes {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    Ok((ty, out))
}

/// [`split_kv_b`] over the fused tensor's rows `bytes`, stored as `ty`: every row the same length.
fn split_rows(sh: &Shape, sites: &Sites, ty: GgmlType, bytes: &[u8]) -> Result<KvHalves, GpuError> {
    let k = sh.kv_lora;
    let per_head = sh.nope + sh.head_v;
    let rows = sh.heads * per_head;
    let row_bytes = bytes.len() / rows;
    if row_bytes == 0 || row_bytes * rows != bytes.len() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{}: {} bytes are not {rows} rows of one length",
                sites.kv_b,
                bytes.len()
            ),
        ));
    }
    let dequant = |at: usize, out: &mut [f32]| -> Result<(), GpuError> {
        let off = at * row_bytes;
        dequant_row(ty, &bytes[off..off + row_bytes], out)
            .map_err(|e| GpuError::shape(WHAT, format!("{} row {at}: {e}", sites.kv_b)))
    };
    let nblocks = sh.nope / 32;
    let mut k_bytes = Vec::with_capacity(sh.heads * k * nblocks * 34);
    let mut v_bytes = Vec::with_capacity(sh.heads * sh.head_v * (k / 32) * 34);
    let mut row = vec![0.0f32; k];
    let mut head_rows = vec![0.0f32; sh.nope * k];
    let mut blk = [0.0f32; 32];
    for h in 0..sh.heads {
        // The head's key-up rows, dequantized once: rows[d][j] = file row h · (nope + v) + d.
        for d in 0..sh.nope {
            dequant(h * per_head + d, &mut row)?;
            head_rows[d * k..(d + 1) * k].copy_from_slice(&row);
        }
        // Block (j, b) = the 32 d-values `32b ..` of column j.
        for j in 0..k {
            for b in 0..nblocks {
                for (l, v) in blk.iter_mut().enumerate() {
                    *v = head_rows[(32 * b + l) * k + j];
                }
                k_bytes.extend_from_slice(&quantize_q8_0(&blk).to_bytes());
            }
        }
        for v in 0..sh.head_v {
            dequant(h * per_head + sh.nope + v, &mut row)?;
            for blk in row.as_chunks::<32>().0 {
                v_bytes.extend_from_slice(&quantize_q8_0(blk).to_bytes());
            }
        }
    }
    let info = |name: &str, k: usize, rows: usize, nbytes: usize| TensorInfo {
        name: name.to_string(),
        dims: vec![k as u64, rows as u64],
        ty: GgmlType::Q8_0,
        offset: 0,
        nbytes: nbytes as u64,
    };
    Ok(KvHalves {
        parts: [
            (
                info(&sites.k_b, sh.nope, sh.heads * k, k_bytes.len()),
                k_bytes,
            ),
            (
                info(&sites.v_b, k, sh.heads * sh.head_v, v_bytes.len()),
                v_bytes,
            ),
        ],
    })
}

/// The derived halves of every latent layer of `layers` of `hp`'s `file` ([`split_kv_b`]), filed
/// under `attn_k_b`/`attn_v_b` beside the file tensors `w` holds ([`Weights::add_parts`]): the
/// load's derive hook, before [`Mla::load`] reads them. A fused tensor a step makes (a checkpoint
/// folder's, [`Place::Made`]) is read from the planes `w` holds of it, the bytes the load made. An HCA layer among `layers` is passed over. Refused by name: a width no kernel runs, a
/// fused tensor missing or of another shape. Load-time only.
pub fn derive(
    stream: &CudaStream,
    file: &Split,
    hp: &Hparams,
    layers: impl IntoIterator<Item = usize>,
    w: &mut Weights,
) -> Result<(), GpuError> {
    let sh = Shape::of(hp, 1)?;
    for l in layers.into_iter().filter(|&l| hp.latent_layer(l)) {
        let sites = Sites::of(l);
        let halves = match file.place(&sites.kv_b) {
            Some(Place::Made) => {
                let want = kv_b_info(&sh, file, &sites)?.ty;
                let (ty, bytes) = resident_blocks(stream, w, &sites.kv_b)?;
                if ty != want {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "{}: the header says {want}, the card holds {ty}",
                            sites.kv_b
                        ),
                    ));
                }
                split_rows(&sh, &sites, ty, &bytes)?
            }
            _ => split_kv_b(&sh, file, &sites)?,
        };
        let parts: Vec<TensorPart<'_>> = halves
            .parts
            .iter()
            .map(|(info, bytes)| TensorPart {
                info: info.clone(),
                spans: vec![bytes.as_slice()],
                scale: None,
            })
            .collect();
        w.add_parts(stream, &parts)?;
    }
    Ok(())
}

/// Every site of `sites` resident after the load, with the file dims it must have.
fn check_sites(sh: &Shape, file: &Split, w: &Weights, sites: &Sites) -> Result<(), GpuError> {
    for (name, k, rows) in sites.by_name(sh) {
        let dims = file
            .find(name)
            .map(|(_, t)| t.dims.clone())
            .ok_or_else(|| GpuError::tensor(WHAT, name, "a tensor of the file"))?;
        if dims.len() != 2 || dims[0] != k as u64 || dims[1] != rows as u64 {
            return Err(GpuError::shape(
                WHAT,
                format!("{name} has dims {dims:?}, want [{k}, {rows}]"),
            ));
        }
        let have = w.resident(WHAT, name)?.rows();
        if have != rows {
            return Err(GpuError::shape(
                WHAT,
                format!("{name} is resident with {have} rows, want {rows}"),
            ));
        }
    }
    for (name, len) in sites.vectors(sh) {
        let have = w.f32_buf(WHAT, name)?.len();
        if have < len {
            return Err(GpuError::shape(
                WHAT,
                format!("{name} holds {have} values, want at least {len}"),
            ));
        }
    }
    Ok(())
}

/// The kernels the chain launches beyond the `Gpu`'s own (`elem`, `flash`, `q8f32`): module
/// handles, shared once loaded.
struct Kernels {
    by_name: ByNameKernels,
    g32: Gemm32Kernels,
    step: StepKernels,
    fused: FusedKernels,
    out_gate: OutGateKernels,
    index_key: IndexKeyKernels,
    indexer: IndexerF32Kernels,
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        Ok(Kernels {
            by_name: ByNameKernels::load(gpu)?,
            g32: Gemm32Kernels::load(gpu.context())?,
            step: StepKernels::load(gpu.context())?,
            fused: FusedKernels::load(gpu.context())?,
            out_gate: OutGateKernels::load(gpu.context())?,
            index_key: IndexKeyKernels::load(gpu.context())?,
            indexer: IndexerF32Kernels::load(gpu.context())?,
        })
    }
}

/// The gather's pair tables for every column a unit holds: pair `c · heads · ROPE + h · ROPE + i`
/// copies the rotated tail's value `i` of head `h` of column `c` (`q_rope`, `heads · head_k` a
/// column) to the absorbed query's front (`q_cur`, `heads · row` a column). A unit of `u` columns
/// takes the first `u · heads · ROPE` pairs, column 0's first.
fn gather_tables(sh: &Shape) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    let n = BY_NAME_COLS * sh.heads * ROPE;
    let mut src = Vec::with_capacity(n);
    let mut dst = Vec::with_capacity(n);
    let word =
        |v: usize| u32::try_from(v).map_err(|_| GpuError::shape(WHAT, "a gather index past u32"));
    for c in 0..BY_NAME_COLS {
        for h in 0..sh.heads {
            for i in 0..ROPE {
                src.push(word(
                    c * sh.heads * sh.head_k() + h * sh.head_k() + sh.nope + i,
                )?);
                dst.push(word(c * sh.heads * sh.row() + h * sh.row() + i)?);
            }
        }
    }
    Ok((src, dst))
}

/// The latent attention of every MLA-DSA layer the trunk runs: the shape, the kernels, the
/// gather's tables, and the layers' sites. Its state
/// is a [`MlaState`] a layer ([`Mla::state`]), its working buffers a [`Scratch`] a stream
/// ([`Mla::scratch`]), its by-name forms [`Mla::forms`] and its position words a [`SpanWords`] a
/// sequence ([`Mla::words`]).
pub struct Mla {
    sh: Shape,
    k: Kernels,
    rope_src: DeviceBuffer<u32>,
    rope_dst: DeviceBuffer<u32>,
    layers: Vec<Layer>,
}

impl Mla {
    /// The latent layers `layers` of `hp` over a cache of `ctx` rows: every site checked against
    /// the file's header and the derived halves ([`derive`]) resident, then the kernels and the
    /// gather's tables loaded. Refused by name, before anything is uploaded: a width no kernel
    /// runs, no layer, a layer past the file's, twice, or an HCA layer, a site missing or of
    /// another shape, a half not derived. Load-time only.
    pub fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: &Hparams,
        ctx: usize,
        layers: &[usize],
    ) -> Result<Mla, GpuError> {
        let sh = Shape::of(hp, ctx)?;
        if layers.is_empty() {
            return Err(GpuError::shape(WHAT, "no layer to run"));
        }
        let mut ls: Vec<Layer> = Vec::with_capacity(layers.len());
        for &l in layers {
            if l >= hp.n_layer {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {l} past the file's {}", hp.n_layer),
                ));
            }
            if !hp.latent_layer(l) {
                return Err(GpuError::shape(WHAT, format!("layer {l} is an HCA layer")));
            }
            if ls.iter().any(|x| x.layer == l) {
                return Err(GpuError::shape(WHAT, format!("layer {l} twice")));
            }
            ls.push(Layer {
                layer: l,
                sites: Sites::of(l),
            });
        }
        for layer in &ls {
            check_sites(&sh, file, w, &layer.sites)?;
            for half in [&layer.sites.k_b, &layer.sites.v_b] {
                w.q8_planes(WHAT, half)?;
            }
        }
        let k = Kernels::load(gpu)?;
        let (src, dst) = gather_tables(&sh)?;
        let rope_src = DeviceBuffer::from_host(gpu.stream(), &src)?;
        let rope_dst = DeviceBuffer::from_host(gpu.stream(), &dst)?;
        Ok(Mla {
            sh,
            k,
            rope_src,
            rope_dst,
            layers: ls,
        })
    }

    /// The widths the chain runs at.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.sh
    }

    /// Device bytes of the gather's tables.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rope_src.num_bytes() + self.rope_dst.num_bytes()
    }

    /// The layers, in the order they were loaded.
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The index of layer `l` of the file among [`Mla::layers`], `None` when it is not one.
    #[must_use]
    pub fn index_of(&self, l: usize) -> Option<usize> {
        self.layers.iter().position(|x| x.layer == l)
    }

    /// A fresh state of `ctx` rows (at most [`Shape::ctx`]) for the layer at `at` of
    /// [`Mla::layers`]: the absorbed cache and the indexer's keys, zero. Load-time only.
    pub fn state(&self, stream: &CudaStream, at: usize, ctx: usize) -> Result<MlaState, GpuError> {
        let Some(layer) = self.layers.get(at) else {
            return Err(GpuError::shape(
                WHAT,
                format!("state {at}: {} layers were loaded", self.layers.len()),
            ));
        };
        if ctx == 0 || ctx > self.sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("a state of {ctx} rows: 1..={}", self.sh.ctx),
            ));
        }
        MlaState::new(self.sh, stream, at, layer.layer, ctx)
    }

    /// The by-name forms of the layers' sites for units of `cols` columns
    /// (`1 ..= `[`BY_NAME_COLS`]). Load-time only.
    pub fn forms(&self, gpu: &Gpu, w: &Weights, cols: usize) -> Result<ByName, GpuError> {
        ByName::for_sites(
            gpu,
            w,
            self.layers
                .iter()
                .flat_map(|l| l.sites.by_name(&self.sh).map(|(n, _, _)| n)),
            cols,
        )
    }

    /// The working buffers of a stream, for units of `unit` columns (`1 ..= `[`BY_NAME_COLS`]`).
    /// Load-time only.
    pub fn scratch(&self, stream: &CudaStream, unit: usize) -> Result<Scratch, GpuError> {
        Scratch::new(&self.sh, stream, unit)
    }

    /// The position words and rope rows of a sequence whose prompt chunks hold at most
    /// `max_span` tokens. Load-time only.
    pub fn words(&self, stream: &CudaStream, max_span: usize) -> Result<SpanWords, GpuError> {
        SpanWords::new(
            stream,
            (self.sh.top_k, self.sh.window, self.sh.rope_base),
            max_span,
        )
    }

    /// The chain of one token at `pos` over the normed row `x` (`hidden` values) into `out`
    /// (`hidden`): [`SpanWords`] set to the span `(pos, 1)` first, by the caller, since that
    /// write synchronizes and this does not: launches only, capturable, replayed at any position
    /// once the words are set again. Refused by name, before any launch: a position at or past
    /// the cache's rows, an eager step over a span of another shape (a capture reads no span,
    /// its replays the words set before each), short `x` or `out`.
    pub fn enqueue_step(
        &self,
        c: &mut Call<'_>,
        st: &mut MlaState,
        pos: u32,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if pos as usize >= st.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("a step at position {pos}: the state holds {} rows", st.ctx),
            ));
        }
        if c.words.span() != (pos, 1) && !capturing(c.gpu.stream())? {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a step at position {pos} over the span {:?}: set the span to (pos, 1) first",
                    c.words.span()
                ),
            ));
        }
        if x.len() < self.sh.hidden || out.len() < self.sh.hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "x holds {} and out {} values, want {} each",
                    x.len(),
                    out.len(),
                    self.sh.hidden
                ),
            ));
        }
        self.enqueue_unit(c, st, (0, 1), x, out)
    }

    /// The chain of a prompt chunk, the `t` tokens at `pos0 .. pos0 + t` of `x` (`[t][hidden]`)
    /// into `out` (`[t][hidden]`), unit by unit of [`Scratch::unit`] columns: [`SpanWords`] set
    /// to the span `(pos0, t)` first, by the caller. Each token's rows are the step's at its
    /// position. Refused by name, before any launch: no token, a span of another shape, rows past
    /// the cache, short `x` or `out`.
    pub fn enqueue_prompt(
        &self,
        c: &mut Call<'_>,
        st: &mut MlaState,
        pos0: u32,
        t: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if t == 0 {
            return Err(GpuError::shape(WHAT, "a prompt of no token"));
        }
        if c.words.span() != (pos0, t) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a prompt of {t} tokens at {pos0} over the span {:?}: set the span first",
                    c.words.span()
                ),
            ));
        }
        if pos0 as usize + t > st.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "rows {pos0}..{} land past the state's {} rows",
                    pos0 as usize + t,
                    st.ctx
                ),
            ));
        }
        if x.len() < t * self.sh.hidden || out.len() < t * self.sh.hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "x holds {} and out {} values, want {} each",
                    x.len(),
                    out.len(),
                    t * self.sh.hidden
                ),
            ));
        }
        let mut at = 0;
        while at < t {
            let u = c.s.unit.min(t - at);
            self.enqueue_unit(c, st, (at, u), x, out)?;
            at += u;
        }
        Ok(())
    }

    /// The chain of one unit (module doc): `u` tokens at span offset `at`, `u` at most the
    /// scratch's columns, columns `at .. at + u` of `x` into `out`'s. The caller states `u`: a
    /// step's capture comes before any span is set, so the unit's width is not read off the span.
    fn enqueue_unit(
        &self,
        c: &mut Call<'_>,
        st: &mut MlaState,
        (at, u): (usize, usize),
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let gpu = c.gpu;
        let w = c.w;
        let vha = c.vha;
        let words = c.words;
        let bn = &mut *c.bn;
        let s = &mut *c.s;
        let sh = self.sh;
        let layer = &self.layers[st.at];
        let l = layer.layer;
        let fault = gpu.layer_sink(l)?;
        let stream = gpu.stream();
        let f = size_of::<f32>();
        let x_at = Window::<f32>::of(x, at * sh.hidden * f, u * sh.hidden)?;
        let mut out_at = WindowMut::<f32>::of_mut(out, at * sh.hidden * f, u * sh.hidden)?;
        let unit = words.unit(at, u)?;
        let Scratch {
            qa,
            qr,
            q,
            q_rope,
            q_cur,
            kv_a,
            kv_s,
            kvr,
            iq,
            ik,
            wi,
            part_v,
            part_ms,
            att,
            av,
            avh,
            g,
            gated,
            scores,
            hist,
            list,
            counts,
            ..
        } = s;
        let kernels = (Some(&self.k.by_name), Some(&self.k.g32));
        let mut project = |site: &str, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>| {
            bn.gemv(gpu, kernels, (w, site), l, x, u, Out::whole(y))
        };
        // 1. The query's low-rank path: q_a, its norm, q_b, the tail of every head turned.
        project(&layer.sites.q_a, &x_at, qa)?;
        gpu.elem().enqueue_rms_norm(
            stream,
            qa,
            w.f32_buf(WHAT, &layer.sites.q_a_norm)?,
            sh.eps,
            sh.q_lora,
            u,
            qr,
        )?;
        project(&layer.sites.q_b, qr, q)?;
        let runs = u32::try_from(sh.heads * sh.head_k() / ROPE)
            .map_err(|_| GpuError::shape(WHAT, "the rope's runs past u32"))?;
        gpu.elem()
            .enqueue_rope(stream, q, &unit.cs, ROPE, runs, u, q_rope)?;
        // 2. The key: the latent normed, the tail turned, the row appended.
        project(&layer.sites.kv_a, &x_at, kv_a)?;
        self.k.fused.enqueue_kv_norm_rope_append_rows(
            stream,
            kv_a,
            w.f32_buf(WHAT, &layer.sites.kv_a_norm)?,
            &unit.cs,
            &unit.pos,
            sh.eps,
            sh.kv_lora,
            ROPE,
            u,
            kv_s,
            kvr,
            &mut st.cache,
            fault,
        )?;
        // 3. The selection list. A selecting state scores the candidates first: the indexer's key,
        //    query and weights, then the f32 score pass; the top-k pass (the identity while a
        //    token has at most `top_k` candidates) and the forced window write the list.
        let ints = &words.words;
        let (n_vis_at, top_k_at) = (words.n_vis_at(at), words.top_k_at());
        if st.selects() {
            project(&layer.sites.idx_k, &x_at, ik)?;
            self.k.index_key.enqueue_index_key_ln_neox(
                stream,
                IndexKeyLnNeoxArgs {
                    x: ik,
                    gain: w.f32_buf(WHAT, &layer.sites.idx_k_norm)?,
                    bias: w.f32_buf(WHAT, &layer.sites.idx_k_bias)?,
                    pos: &unit.pos,
                    cs: &unit.cs,
                    eps: sh.eps,
                    n_dims: ROPE,
                    m: u,
                    cache: &mut st.keys,
                    fault,
                },
            )?;
            project(&layer.sites.idx_q, qr, iq)?;
            project(&layer.sites.idx_w, &x_at, wi)?;
            self.k.indexer.enqueue_score(
                stream,
                ScoreF32Args {
                    q: iq,
                    w: wi,
                    ints,
                    n_vis_at,
                    top_k_at,
                    tables: &words.rope,
                    rope_at: at * ROPE,
                    rope_stride: ROPE,
                    keys: &st.keys,
                    tokens: u,
                    scores,
                    hist,
                    fault,
                },
            )?;
        }
        self.k.indexer.enqueue_topk(
            stream,
            TopkArgs {
                ints,
                n_vis_at,
                top_k_at,
                scores,
                tokens: u,
                rows: st.ctx,
                stride: sh.stride(),
                hist,
                list,
            },
        )?;
        self.k.indexer.enqueue_window(
            stream,
            WindowArgs {
                ints,
                n_vis_at,
                n_win_at: words.n_win_at(at),
                top_k_at,
                tokens: u,
                stride: sh.stride(),
                fault,
                list,
                counts,
            },
        )?;
        // 4. The absorbed query a head: k_bᵀ q_nope, then the rotated tail gathered to its front.
        let (qs, d) = w.q8_planes(WHAT, &layer.sites.k_b)?;
        self.heads_gemv(
            gpu,
            (qs, d),
            u,
            HeadGemv {
                x: q,
                rows_per_head: sh.kv_lora,
                x_head_stride: sh.head_k(),
                x_col_stride: sh.heads * sh.head_k(),
                y: q_cur,
                y_head_stride: sh.row(),
                y_col_stride: sh.heads * sh.row(),
                y_off: ROPE,
            },
        )?;
        self.k.step.enqueue_gather(
            stream,
            q_rope,
            &self.rope_src,
            &self.rope_dst,
            u * sh.heads * ROPE,
            q_cur,
        )?;
        // The flash with a sink a head over the token's list.
        let geom = FlashGeom {
            tokens: u,
            heads: sh.heads,
            rope_dims: ROPE,
            latent_dims: sh.kv_lora,
        };
        let sinks = w.f32_buf(WHAT, &layer.sites.sinks)?;
        gpu.flash().enqueue_flash_latent_sel_sink(
            stream,
            FlashSelArgs {
                q: q_cur,
                kv: &st.cache,
                list,
                counts,
                stride: sh.stride(),
                kq_scale: sh.kq_scale(),
                geom,
                part_v,
                part_ms,
                fault,
            },
            sinks,
            att,
        )?;
        // The heads' values.
        let (qs, d) = w.q8_planes(WHAT, &layer.sites.v_b)?;
        self.heads_gemv(
            gpu,
            (qs, d),
            u,
            HeadGemv {
                x: att,
                rows_per_head: sh.head_v,
                x_head_stride: sh.kv_lora,
                x_col_stride: sh.heads * sh.kv_lora,
                y: av,
                y_head_stride: sh.head_v,
                y_col_stride: sh.heads * sh.head_v,
                y_off: 0,
            },
        )?;
        // 5. The head mix through the hook, the sigmoid gate of the normed query latent, o_proj.
        vha.mix(stream, l, u, av, avh)?;
        project(&layer.sites.gate, qr, g)?;
        self.k.out_gate.enqueue_out_gate_head(
            stream,
            OutGateArgs {
                kind: OutGate::Proj,
                attn: avh,
                gate: g,
                n_head: sh.heads,
                m: u,
                fault,
                y: gated,
            },
            sh.head_v,
        )?;
        project(&layer.sites.o, gated, &mut out_at)
    }

    /// `y = W · x` per head over `m` columns for the Q8_0 planes `(qs, d)`: the step's per-head
    /// launch at one column, the `m`-column form above.
    fn heads_gemv(
        &self,
        gpu: &Gpu,
        planes: (&DeviceTensor<u32>, &DeviceTensor<u16>),
        m: usize,
        a: HeadGemv<'_>,
    ) -> Result<(), GpuError> {
        let (qs, d) = planes;
        if m == 1 {
            return self.k.step.enqueue_q8_0_gemv_heads(
                gpu.stream(),
                Q8_0GemvHeadsArgs {
                    qs,
                    d,
                    x: a.x,
                    rows_per_head: a.rows_per_head,
                    x_head_stride: a.x_head_stride,
                    y_head_stride: a.y_head_stride,
                    y_off: a.y_off,
                    y: a.y,
                },
            );
        }
        gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
            gpu.stream(),
            Q8_0GemvHeadsMcolArgs {
                qs,
                d,
                x: a.x,
                rows_per_head: a.rows_per_head,
                x_head_stride: a.x_head_stride,
                y_head_stride: a.y_head_stride,
                y_off: a.y_off,
                m,
                x_col_stride: a.x_col_stride,
                y_col_stride: a.y_col_stride,
                y: a.y,
            },
        )
    }
}

/// [`Mla::heads_gemv`]'s geometry: the activation and the output with their head and column
/// strides (f32 values), the rows a head, and where in a head's output the rows land.
struct HeadGemv<'a> {
    x: &'a DeviceBuffer<f32>,
    y: &'a mut DeviceBuffer<f32>,
    rows_per_head: usize,
    x_head_stride: usize,
    y_head_stride: usize,
    y_off: usize,
    x_col_stride: usize,
    y_col_stride: usize,
}

/// The head mix of a latent layer (VHA: `o[h] ← Σ_g (I + V·Uᵀ)[h, g] · o[g]` over the heads),
/// the hook the chain hands its heads' values: `x` holds `t` tokens of `heads` heads of `width`
/// values (`[t][heads][width]`), `y` receives the mixed rows in the same layout. This crate's
/// attention carries no mix; the track that writes it supplies it, and a gate the identity
/// ([`NoVha`]).
pub trait Vha {
    /// Mix the heads' values of layer `layer`'s `t` tokens from `x` into `y`.
    fn mix(
        &self,
        stream: &CudaStream,
        layer: usize,
        t: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;
}

/// The identity head mix: `y = x`, every layer's.
pub struct NoVha;

impl Vha for NoVha {
    fn mix(
        &self,
        stream: &CudaStream,
        _layer: usize,
        _t: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        copy_row(stream, x, y, x.len())
    }
}

/// What one chain call reads besides the state: the card, the resident weights, the head mix,
/// the sequence's position words, the by-name forms and the working buffers.
pub struct Call<'a> {
    pub gpu: &'a Gpu,
    pub w: &'a Weights,
    pub vha: &'a dyn Vha,
    pub words: &'a SpanWords,
    pub bn: &'a mut ByName,
    pub s: &'a mut Scratch,
}

/// One layer's state, sized from the cache's rows at [`Mla::state`]: the absorbed cache (`ctx`
/// rows of `[k_rope | latent]` f16, a position a row) and the indexer's keys (`ctx` rows of 128
/// f32). A position's row is written by its own token before any launch reads it, and every read
/// is bounded by a count word, so a fresh sequence needs no reset; [`MlaState::reset`] fills
/// both stores with NaN so a stray read surfaces.
pub struct MlaState {
    sh: Shape,
    /// The layer's index among the component's ([`Mla::layers`]): the sites the chain runs.
    at: usize,
    /// The layer in the file: the fault sink's layer.
    pub layer: usize,
    ctx: usize,
    cache: DeviceTensor<u16>,
    keys: DeviceTensor<f32>,
}

impl MlaState {
    fn new(
        sh: Shape,
        stream: &CudaStream,
        at: usize,
        layer: usize,
        ctx: usize,
    ) -> Result<MlaState, GpuError> {
        Ok(MlaState {
            sh,
            at,
            layer,
            ctx,
            cache: DeviceTensor::zeroed(stream, ctx, sh.row())?,
            keys: DeviceTensor::zeroed(stream, ctx, HEAD_DIM)?,
        })
    }

    /// The cache's rows: the most positions a sequence holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.ctx
    }

    /// Whether a sequence over this state ever selects: its last position's candidates older
    /// than the window are more than `top_k`. Only then the key cache is written and the score
    /// pass runs.
    #[must_use]
    pub fn selects(&self) -> bool {
        self.ctx > self.sh.stride()
    }

    /// Device bytes of the two stores.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cache.buf().num_bytes() + self.keys.buf().num_bytes()
    }

    /// The absorbed cache, `[rows][k_rope | latent]` f16.
    #[must_use]
    pub fn cache(&self) -> &DeviceTensor<u16> {
        &self.cache
    }

    /// The indexer's keys, `[rows][128]` f32.
    #[must_use]
    pub fn keys(&self) -> &DeviceTensor<f32> {
        &self.keys
    }

    /// `h` folded with both stores' bits, read back (synchronizes `stream`).
    pub fn digest(&self, h: u64, stream: &CudaStream) -> Result<u64, GpuError> {
        let cache = self.cache.buf().to_host_vec(stream)?;
        let h = crate::body::fold(h, cache.into_iter().map(u32::from));
        let keys = self.keys.buf().to_host_vec(stream)?;
        Ok(crate::body::fold(h, keys.into_iter().map(f32::to_bits)))
    }

    /// Fill both stores with NaN. Synchronizes `stream` per chunk; never inside a capture.
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        fill_nan::<u16>(stream, self.cache.buf_mut(), NAN_F16)?;
        fill_nan::<f32>(stream, self.keys.buf_mut(), f32::NAN)
    }
}

/// `buf` filled with `v`, a chunk of [`NAN_CHUNK`] values at a time.
fn fill_nan<T: cuda_core::DeviceCopy + Copy>(
    stream: &CudaStream,
    buf: &mut DeviceBuffer<T>,
    v: T,
) -> Result<(), GpuError> {
    let mut at = 0;
    while at < buf.len() {
        let n = NAN_CHUNK.min(buf.len() - at);
        let mut win = WindowMut::<T>::of_mut(buf, at * size_of::<T>(), n)?;
        win.copy_from_host(stream, &vec![v; n])?;
        at += n;
    }
    Ok(())
}

/// The position words and rope rows of one sequence, shared by its layers: `[pos: max_span][n_win:
/// max_span][cands: max_span][top_k]`, token `r` of the current span at `pos[r] = pos0 + r` (the
/// cache row its key lands on), `n_win[r] = pos0 + r + 1` (the keys it sees, its own the last)
/// and `cands[r] = max(0, pos0 + r + 1 − window)` (the candidates the indexer scores: those
/// older than its window), and `top_k` once; and `max_span` rows of [`ROPE`] f32, row `r` the
/// rope table's row of position `pos0 + r` (plain rope at the layer's base). [`SpanWords::set`]
/// writes them for a span; the chain reads them on the device through fixed addresses, so a
/// captured step replays at any position once they are set again.
pub struct SpanWords {
    words: DeviceBuffer<u32>,
    rope: DeviceBuffer<f32>,
    table: RopeTable,
    max_span: usize,
    window: usize,
    top_k: u32,
    span: (u32, usize),
}

impl SpanWords {
    /// The words for spans of at most `max_span` tokens, the layer's `(top_k, window,
    /// rope base)`. Load-time only.
    pub fn new(
        stream: &CudaStream,
        (top_k, window, rope_base): (usize, usize, f32),
        max_span: usize,
    ) -> Result<SpanWords, GpuError> {
        if max_span == 0 || top_k == 0 || window == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("max_span {max_span}, top_k {top_k} and window {window}: at least 1 each"),
            ));
        }
        Ok(SpanWords {
            words: DeviceBuffer::zeroed(stream, 3 * max_span + 1)?,
            rope: DeviceBuffer::zeroed(stream, max_span * ROPE)?,
            table: RopeTable::new(&RopeSpec::window(rope_base, ROPE))?,
            max_span,
            window,
            top_k: u32::try_from(top_k).map_err(|_| GpuError::shape(WHAT, "top_k past u32"))?,
            span: (0, 0),
        })
    }

    /// The words and rope rows of the span `(pos0, t)`: `t` consecutive tokens from `pos0`.
    /// Synchronizes `stream`; never inside a capture. Refused by name, before the write: no
    /// token, more than `max_span`, a position past `u32`.
    pub fn set(&mut self, stream: &CudaStream, pos0: u32, t: usize) -> Result<(), GpuError> {
        if t == 0 || t > self.max_span {
            return Err(GpuError::shape(
                WHAT,
                format!("a span of {t} tokens: 1..={}", self.max_span),
            ));
        }
        let word = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(WHAT, format!("position {v} past u32")))
        };
        let mut host = Vec::with_capacity(3 * self.max_span + 1);
        for r in 0..self.max_span {
            host.push(word(pos0 as usize + r)?);
        }
        for r in 0..self.max_span {
            host.push(word(pos0 as usize + r + 1)?);
        }
        for r in 0..self.max_span {
            let p = pos0 as usize + r + 1;
            host.push(word(p.saturating_sub(self.window))?);
        }
        host.push(self.top_k);
        let mut rows = Vec::with_capacity(t * ROPE);
        for r in 0..t {
            self.table
                .push(word(pos0 as usize + r)?, Direction::Forward, &mut rows);
        }
        self.words.copy_from_host(stream, &host)?;
        WindowMut::<f32>::of_mut(&mut self.rope, 0, t * ROPE)?.copy_from_host(stream, &rows)?;
        self.span = (pos0, t);
        Ok(())
    }

    /// The span last set: `(pos0, t)`, `(0, 0)` before any.
    #[must_use]
    pub fn span(&self) -> (u32, usize) {
        self.span
    }

    /// The most tokens a span holds.
    #[must_use]
    pub fn max_span(&self) -> usize {
        self.max_span
    }

    /// Device bytes of the words and the rope rows.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.words.num_bytes() + self.rope.num_bytes()
    }

    /// One unit's words and rope rows: `u` tokens at span offset `at`.
    fn unit(&self, at: usize, u: usize) -> Result<Unit<'_>, GpuError> {
        let w = size_of::<u32>();
        Ok(Unit {
            pos: Window::of(&self.words, at * w, u)?,
            cs: Window::of(&self.rope, at * ROPE * size_of::<f32>(), u * ROPE)?,
        })
    }

    fn n_vis_at(&self, at: usize) -> usize {
        2 * self.max_span + at
    }

    fn n_win_at(&self, at: usize) -> usize {
        self.max_span + at
    }

    fn top_k_at(&self) -> usize {
        3 * self.max_span
    }
}

/// One unit's device words: its tokens' positions (the rows their keys land on) and its tokens'
/// rope rows.
struct Unit<'a> {
    pos: Window<'a, u32>,
    cs: Window<'a, f32>,
}

/// The chain's working buffers for one stream, sized for units of `unit` columns: every
/// intermediate of the module doc's steps, the flash partials over a list's width, and the
/// indexer's scores, histogram, list and counts for a unit's tokens. A buffer holds the current
/// unit only (column `c` of the unit at `c` times its width); the histogram is zeroed here and
/// kept zero by the two passes that use it.
pub struct Scratch {
    /// The columns a unit holds.
    pub unit: usize,
    qa: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    q_cur: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    iq: DeviceBuffer<f32>,
    ik: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    avh: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    hist: DeviceBuffer<u32>,
    list: DeviceBuffer<u32>,
    counts: DeviceBuffer<u32>,
}

impl Scratch {
    /// The buffers for `sh` over a cache of `sh.ctx` rows, units of `unit` columns
    /// (`1 ..= `[`BY_NAME_COLS`]). Load-time only.
    pub fn new(sh: &Shape, stream: &CudaStream, unit: usize) -> Result<Scratch, GpuError> {
        if !(1..=BY_NAME_COLS).contains(&unit) {
            return Err(GpuError::shape(
                WHAT,
                format!("{unit} columns a unit: 1..={BY_NAME_COLS}"),
            ));
        }
        let u = unit;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let heads_row = u * sh.heads * sh.row();
        Ok(Scratch {
            unit,
            qa: f(u * sh.q_lora)?,
            qr: f(u * sh.q_lora)?,
            q: f(u * sh.heads * sh.head_k())?,
            q_rope: f(u * sh.heads * sh.head_k())?,
            q_cur: f(heads_row)?,
            kv_a: f(u * sh.row())?,
            kv_s: f(u * sh.row())?,
            kvr: f(u * sh.row())?,
            iq: f(u * sh.idx_heads * sh.idx_d)?,
            ik: f(u * sh.idx_d)?,
            wi: f(u * sh.idx_heads)?,
            part_v: f(partials_v_len(u * sh.heads, sh.stride()))?,
            part_ms: f(partials_ms_len(u * sh.heads, sh.stride()))?,
            att: f(u * sh.heads * sh.kv_lora)?,
            av: f(u * sh.heads * sh.head_v)?,
            avh: f(u * sh.heads * sh.head_v)?,
            g: f(u * sh.heads * sh.head_v)?,
            gated: f(u * sh.heads * sh.head_v)?,
            scores: f(u * sh.ctx)?,
            hist: DeviceBuffer::zeroed(stream, u * HIST_BINS)?,
            list: DeviceBuffer::zeroed(stream, u * sh.stride())?,
            counts: DeviceBuffer::zeroed(stream, u)?,
        })
    }

    /// Device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        [
            self.qa.num_bytes(),
            self.qr.num_bytes(),
            self.q.num_bytes(),
            self.q_rope.num_bytes(),
            self.q_cur.num_bytes(),
            self.kv_a.num_bytes(),
            self.kv_s.num_bytes(),
            self.kvr.num_bytes(),
            self.iq.num_bytes(),
            self.ik.num_bytes(),
            self.wi.num_bytes(),
            self.part_v.num_bytes(),
            self.part_ms.num_bytes(),
            self.att.num_bytes(),
            self.av.num_bytes(),
            self.avh.num_bytes(),
            self.g.num_bytes(),
            self.gated.num_bytes(),
            self.scores.num_bytes(),
            self.hist.num_bytes(),
            self.list.num_bytes(),
            self.counts.num_bytes(),
        ]
        .iter()
        .sum()
    }

    /// The indexer's list of the unit's token `t` (`stride` rows) and its live count, as the
    /// chain left them: for a gate's reading.
    pub fn selection(&self, stream: &CudaStream, t: usize) -> Result<(Vec<u32>, u32), GpuError> {
        if t >= self.unit {
            return Err(GpuError::shape(
                WHAT,
                format!("token {t} of a unit of {}", self.unit),
            ));
        }
        let stride = self.list.len() / self.unit;
        stream.synchronize()?;
        let list = Window::<u32>::of(&self.list, t * stride * size_of::<u32>(), stride)?
            .to_host_vec(stream)?;
        let counts = self.counts.to_host_vec(stream)?;
        Ok((list, counts[t]))
    }
}
