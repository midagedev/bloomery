//! The dense Qwen3-shaped draft layer: GQA attention with q/k norms, a NEOX rotation over the whole
//! head and an optional sliding window, then a SiLU MLP, stacked `layers` deep ([`DenseLayers`]).
//!
//! # What a pass is
//!
//! The draft's context rows are the target's tapped rows after fc and the hidden norm (the owner's).
//! Every layer projects them to its own K and V with no norm of their own before the projection, turns
//! K by its position and appends both to its cache ([`DenseLayers::append_kv`]). A block pass
//! ([`DenseLayers::enqueue_block`]) runs the embedded block rows through the layers
//! (`x + attn(norm(x))`, then `x + mlp(norm(x))`); each row attends to the visible context rows and to
//! the block's rows under the file's mask ([`DenseSpec::block_causal`]), and the block's own K and V
//! sit in scratch rows behind the context. The final norm is the head's.
//!
//! # Kernels
//!
//! No kernel of its own. The projections are the Q8_0 row kernel over f32 activations
//! (`q8f32::enqueue_q8_0_gemv_mcol`, at most [`BLOCK_MAX`] columns), the per-head norms `elem::rms_norm`
//! over `head_dim`-wide vectors, the rotation `elem::rope`, the append `flash::kv_append_pos_buf`, and
//! the attention `flash_gqa::enqueue_pass_256` (a group of 8 query heads per key head, else the
//! pack-of-four entry; a DFlash2 draft's heads take `enqueue_pass_p4_ring`), and a DFlash2 draft's
//! convolution `super::conv`'s. Qwen3.6's own rotation kernel (`rope_neox::head_norm_neox_append_256`) turns 64
//! dims of a q+gate row and cannot serve a draft that turns all 256 of an ungated one.
//!
//! The draft rotates NEOX pairs `(i, i + head_dim/2)`; `elem::rope` turns adjacent pairs
//! `(2i, 2i + 1)`. The loader permutes each query and key head's rows (and the two norm gains) so that
//! pair `i` of the file is adjacent pair `i`: new row `2i` is the file's `i`, new row `2i + 1` the
//! file's `i + head_dim/2`. A head's RMS norm sums the same squares in another order, and `q · k` sums
//! the same products, because q and k carry the same permutation inside every head; V and the output
//! projection are untouched. NEOX frequency `i` is then adjacent pair `i` of the host table
//! (`RopeTable`, `[cos_0, sin_0, cos_1, …]`). A Q8_0 block runs along the input axis, so permuting
//! output rows moves whole rows and changes no block.
//!
//! # Where the context lives
//!
//! Each layer holds, per key head, a plane of `cap + BLOCK_MAX` rows of `head_dim` f16 values, `cap`
//! being the window (or the context capacity of a draft with no window). A context row at position `p`
//! sits at slot `(p - base) % cap`, K stored after its rotation at its absolute position. Attention is
//! a softmax over a set of keys that are all stored rotated; it does not depend on the order the keys
//! are walked in, so the ring needs no unrolling: the flash kernel takes the first `n_keys` rows of
//! the plane, and with the visible rows at the plane's head (`min(count, cap)` of them, which is all
//! `cap` once the ring is full) the block's K and V go to the slots right behind them, rewritten by
//! every pass. A row's key count is the visible rows plus the block rows it may see (its own and the
//! earlier ones under a causal mask, all `m` otherwise). A layer's cache is preceded by [`GUARD`] rows
//! no head reads, which let an append that wraps past the ring's end be two fixed-length launches
//! ([`DenseLayers::stage_kv`]).
//!
//! # Staging
//!
//! Every position-dependent word a replayed pass reads is on the card in an image the stage calls write
//! (never inside a capture): the cos/sin rows of the staged positions, the cache rows the K and V
//! appends start at, and the rows' key counts. The graphs hold launches only.
//!
//! # A DFlash2 draft's layer
//!
//! [`DenseLayers::load_dflash2`] loads the same layer from a DFlash2 draft's GGUF file, with five
//! parameters of this one stack set:
//!
//! - **Q8_0 rows as the file stores them.** The file's matrices are Q8_0; their blocks go to the card
//!   unchanged ([`Q8Rows::from_gguf`]), permuted by whole rows as above.
//! - **Heads of [`HEAD`] in groups of a multiple of [`PACK_4`]**, attended by
//!   `flash_gqa::enqueue_pass_p4_ring`.
//! - **The window per block row.** llama.cpp's standard sliding window masks a key at `p_k` for a
//!   query at `p_q` when `p_q − p_k >= window`, so block row `t`, at `pos + t`, reads the last
//!   `c_t = min(count, window − 1 − t)` context rows and every row of the (non-causal) block. The ring
//!   holds `min(window − 1, ctx) + BLOCK_MAX` rows, context row `p` at slot `(p − base) % ring`, and the
//!   block's K and V go to its positions' slots, which no row any block row reads holds: row `t`'s
//!   keys are one run of the ring, `c_t + m` rows from the slot of its first visible row, wrapping once
//!   ([`per_row_image`]). The block's append is two fixed-length launches, as a context append is.
//! - **The dynamic convolution around both sub-layers** (`super::conv`): the sub-layer's normed rows
//!   give their coefficients (`*_conv_proj`); side 0 mixes those rows before the projections, side 1
//!   the sub-layer's output before the residual add.
//! - **The target's embedding** ([`super::EmbedSource::Target`]): the load writes the mask's row into
//!   the block's rows 1.., each stage the anchor's into row 0, and the pass copies them into its input
//!   first.

use std::sync::{Mutex, PoisonError};

use cuda_core::DeviceBuffer;
use gguf::quant::Q8Block;
use gguf::{GgmlType, Split};
use model::arch::dflash2::{self, Conv, Hparams};
use model::arch::dspark::DraftHeld;
use model::embed::EmbedTable;
use model::quant::quantize_q8_0;

use super::conv::{ConvArgs, ConvKernels, Side};
use super::{DraftLayers, OwnHeld, Q8Rows};
use crate::flash::PlaneAppend;
use crate::flash_gqa::{
    FlashGqaKernels, GROUP, GqaArgs, GqaRingArgs, HEAD, HEAD_256, PACK_4, partials_ms_len,
    partials_v_len, partials_v_len_256,
};
use crate::graph::capturing;
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::rope_table::{Direction, RopeSpec, RopeTable};
use crate::weights::q8_0_planes;
use crate::{DeviceTensor, Gpu, GpuError, Window, WindowMut};

const WHAT: &str = "dspark::dense";

/// Rows of one block pass and of one append: the Q8_0 row kernel's column count.
pub const BLOCK_MAX: usize = 8;
/// Rows one [`DenseLayers::stage_kv`] stages.
pub const APPEND_MAX: usize = BLOCK_MAX;
/// Rows before the first key head's plane in a layer's cache (see the module doc).
const GUARD: usize = BLOCK_MAX;
/// Whether the attention takes the tensor-core segment pass.
const MMA: bool = false;

/// A draft's weights as bf16 bits by tensor name, the form its safetensors file stores them in. The
/// owner of a file format implements it; this crate holds no file reader.
pub trait Bf16Source {
    /// Tensor `name` as its bits and shape, or a named error.
    fn bf16(&self, name: &str) -> Result<(&[u16], &[usize]), GpuError>;
}

/// What a draft's config states about its dense layers; the widths come from the tensors' shapes.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseSpec {
    /// Decoder layers to load.
    pub layers: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The rotary base.
    pub rope_theta: f32,
    /// How many context rows back each layer reads; `None` is the whole context.
    pub window: Option<usize>,
    /// Whether a block row attends to the block's earlier rows only; `false` is every row of the block.
    pub block_causal: bool,
    /// The most positions a context holds: a draft with no window keeps every row, a windowed one
    /// never needs more rows than this either.
    pub ctx: usize,
}

/// The layers' widths, read from the first layer's tensors and held equal by every other layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseDims {
    /// The draft's hidden width.
    pub hidden: usize,
    /// Query heads.
    pub n_q: usize,
    /// Key/value heads.
    pub n_kv: usize,
    /// Values per head.
    pub head_dim: usize,
    /// The MLP's inner width.
    pub inter: usize,
}

fn name(layer: usize, tail: &str) -> String {
    format!("layers.{layer}.{tail}")
}

fn shape_err(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

fn matrix<'a>(
    src: &'a dyn Bf16Source,
    layer: usize,
    tail: &str,
    rows: usize,
    k: usize,
) -> Result<&'a [u16], GpuError> {
    let n = name(layer, tail);
    let (bits, shape) = src.bf16(&n)?;
    if shape != [rows, k] || bits.len() != rows * k {
        return Err(shape_err(format!(
            "{n} is {shape:?} ({} values); the layer wants [{rows}, {k}]",
            bits.len()
        )));
    }
    Ok(bits)
}

fn vector<'a>(
    src: &'a dyn Bf16Source,
    layer: usize,
    tail: &str,
    len: usize,
) -> Result<&'a [u16], GpuError> {
    let n = name(layer, tail);
    let (bits, shape) = src.bf16(&n)?;
    if shape != [len] || bits.len() != len {
        return Err(shape_err(format!(
            "{n} is {shape:?}; the layer wants [{len}]"
        )));
    }
    Ok(bits)
}

impl DenseDims {
    /// The widths of `src`'s layer 0, refused by name when the head is not one this stack runs: a head of
    /// 256 values, and a query group the attention entries serve.
    pub fn read(src: &dyn Bf16Source) -> Result<DenseDims, GpuError> {
        let dim = |tail: &str, axis: usize| -> Result<usize, GpuError> {
            let n = name(0, tail);
            let (_, shape) = src.bf16(&n)?;
            match shape {
                [_, _] | [_] => shape
                    .get(axis)
                    .copied()
                    .ok_or_else(|| shape_err(format!("{n} is {shape:?}: no axis {axis}"))),
                other => Err(shape_err(format!(
                    "{n} is {other:?}: not a matrix or a vector"
                ))),
            }
        };
        let head_dim = dim("self_attn.q_norm.weight", 0)?;
        let hidden = dim("input_layernorm.weight", 0)?;
        let q_rows = dim("self_attn.q_proj.weight", 0)?;
        let k_rows = dim("self_attn.k_proj.weight", 0)?;
        let inter = dim("mlp.gate_proj.weight", 0)?;
        if head_dim != HEAD_256 {
            return Err(shape_err(format!(
                "heads of {head_dim} values; the attention entries serve {HEAD_256}"
            )));
        }
        if q_rows % head_dim != 0 || k_rows % head_dim != 0 || k_rows == 0 {
            return Err(shape_err(format!(
                "q_proj has {q_rows} rows and k_proj {k_rows}: not whole heads of {head_dim}"
            )));
        }
        let (n_q, n_kv) = (q_rows / head_dim, k_rows / head_dim);
        if n_q % n_kv != 0 || !(n_q / n_kv == GROUP || (n_q / n_kv) % PACK_4 == 0) {
            return Err(shape_err(format!(
                "{n_q} query heads over {n_kv} key heads: a group of {GROUP} or a multiple of {PACK_4}"
            )));
        }
        for (what, v) in [("hidden", hidden), ("inter", inter), ("q width", q_rows)] {
            if v == 0 || v % 32 != 0 {
                return Err(shape_err(format!(
                    "{what} {v}: the Q8_0 rows run in blocks of 32 values"
                )));
            }
        }
        Ok(DenseDims {
            hidden,
            n_q,
            n_kv,
            head_dim,
            inter,
        })
    }

    /// Context rows each key head's plane holds in a ring (the block's scratch rows are behind them).
    #[must_use]
    pub fn ring(spec: &DenseSpec) -> usize {
        spec.window.map_or(spec.ctx, |w| w.min(spec.ctx)).max(1)
    }

    /// Card bytes of `spec.layers` layers at `spec.ctx` positions: the Q8_0 weights, the norm gains, every
    /// layer's K and V planes, the pass's scratch and the images.
    #[must_use]
    pub fn card_bytes(&self, spec: &DenseSpec) -> u64 {
        self.bytes(spec, Features::NONE)
    }

    /// [`DenseDims::card_bytes`] of layers that embed the block from the target's table
    /// ([`DenseLayers::load_target_embed`]): the plain layer plus the staged anchor and mask rows.
    pub fn card_bytes_target_embed(&self, spec: &DenseSpec) -> u64 {
        self.bytes(
            spec,
            Features {
                target_embed: true,
                ..Features::NONE
            },
        )
    }

    /// [`DenseDims::card_bytes`] of layers with `f` set: the convolution's weights and scratch, the
    /// target embedding's rows, the per-row window's ring and image.
    fn bytes(&self, spec: &DenseSpec, f: Features) -> u64 {
        let DenseDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = *self;
        let q8 = |rows: usize, k: usize| (rows * (k + k / 16)) as u64;
        let conv_weights = f.conv.map_or(0, |c| {
            2 * q8(c.coeffs(h), h) + 2 * 4 * (2 * c.kernel * h) as u64
        });
        let weights = q8(n_q * hd, h)
            + 2 * q8(n_kv * hd, h)
            + q8(h, n_q * hd)
            + 2 * q8(inter, h)
            + q8(h, inter)
            + 4 * (2 * h + 2 * hd) as u64
            + conv_weights;
        let planes = ring_planes(ring_cap(spec, f));
        let kv = 2 * 2 * ((GUARD + n_kv * planes) * hd) as u64;
        let m = BLOCK_MAX;
        let conv_scratch = f.conv.map_or(0, |c| 4 * (m * (c.coeffs(h) + 2 * h)) as u64);
        let embed = if f.target_embed {
            4 * (m * h) as u64
        } else {
            0
        };
        let scratch = 4 * (m * (5 * h + 3 * inter + 4 * n_q * hd + 4 * n_kv * hd)) as u64
            + 4 * (partials_v(m, n_q, hd) + partials_ms_len(m, n_q)) as u64
            + 4 * (2 * m * hd + block_words(n_kv, f) + 2 * n_kv) as u64
            + conv_scratch
            + embed;
        spec.layers as u64 * (weights + kv) + scratch
    }
}

fn ring_planes(cap: usize) -> usize {
    cap + BLOCK_MAX
}

/// Context rows a key head's ring holds: the window or the context (the module doc), and under the
/// per-row window the last block row's `window − 1` rows behind the block's own.
fn ring_cap(spec: &DenseSpec, f: Features) -> usize {
    match f.per_row {
        None => DenseDims::ring(spec),
        Some(w) => (w - 1).min(spec.ctx) + BLOCK_MAX,
    }
}

/// Words of the block's image: the block append's first rows and the rows' key counts, and under the
/// per-row window the append's second rows and the rows' first ring rows ([`per_row_image`]).
fn block_words(n_kv: usize, f: Features) -> usize {
    let one = n_kv + BLOCK_MAX;
    if f.per_row.is_some() { 2 * one } else { one }
}

/// The attention's `Σ exp·V` partials for `m` rows of `n_q` heads of `hd` values.
fn partials_v(m: usize, n_q: usize, hd: usize) -> usize {
    if hd == HEAD_256 {
        partials_v_len_256(m, n_q)
    } else {
        partials_v_len(m, n_q)
    }
}

/// What a load sets beyond the plain layer ([`DenseLayers::load`] sets none; the module doc's DFlash2
/// section names each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Features {
    /// The window cut per block row, of this many positions; `None` is the ring's rule.
    per_row: Option<usize>,
    /// The convolution around both sub-layers.
    conv: Option<Conv>,
    /// Whether the layers embed the block from the target's table.
    target_embed: bool,
}

impl Features {
    const NONE: Features = Features {
        per_row: None,
        conv: None,
        target_embed: false,
    };
}

/// Block row `t`'s run of the ring under the per-row window: the last `min(count, window − 1 − t)`
/// context rows, then the block's own rows (its earlier ones and itself when `causal`, all `m`
/// otherwise). Its first context row is the ring's slot `(count − c_t) % cap`.
#[derive(Clone, Copy, Debug)]
struct PerRow {
    window: usize,
    cap: usize,
    count: usize,
    m: usize,
    causal: bool,
}

impl PerRow {
    /// Context rows row `t` reads.
    fn context(&self, t: usize) -> usize {
        self.count.min(self.window - 1 - t)
    }

    /// Row `t`'s key count.
    fn keys(&self, t: usize) -> usize {
        self.context(t) + if self.causal { t + 1 } else { self.m }
    }

    /// The ring slot of row `t`'s first key.
    fn first_slot(&self, t: usize) -> usize {
        (self.count - self.context(t)) % self.cap
    }
}

/// The block's image under the per-row window, `row(h, slot)` the cache row of key head `h`'s `slot`:
/// the block append's first launch's row per key head (the slot of the block's first position), the
/// [`BLOCK_MAX`] rows' key counts, the second launch's row per key head (the first launch's again, or
/// `wrap` rows before slot 0 when the block runs past the ring's end), and the rows' first ring
/// slots. Rows past `m` are padding no launch reads.
fn per_row_image(p: PerRow, n_kv: usize, row: impl Fn(usize, usize) -> u32) -> Vec<u32> {
    let word = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    let slot0 = p.count % p.cap;
    let wrap = if slot0 + p.m > p.cap {
        p.cap - slot0
    } else {
        0
    };
    let mut words: Vec<u32> = (0..n_kv).map(|h| row(h, slot0)).collect();
    words.extend((0..BLOCK_MAX).map(|t| word(p.keys(t))));
    words.extend((0..n_kv).map(|h| {
        if wrap == 0 {
            row(h, slot0)
        } else {
            row(h, 0) - word(wrap)
        }
    }));
    words.extend((0..BLOCK_MAX).map(|t| word(p.first_slot(t))));
    words
}

/// `bits` widened to f32.
fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

/// The file's index of new row `j` of a head of `hd` values after the NEOX-to-adjacent permutation
/// (the module doc): new `2i` is `i`, new `2i + 1` is `i + hd/2`.
fn adjacent_pairs(hd: usize, j: usize) -> usize {
    if j.is_multiple_of(2) {
        j / 2
    } else {
        hd / 2 + j / 2
    }
}

impl Q8Rows {
    /// `rows` rows of `k` values quantized from bf16 on the host, row `r` of the result being row `pick(r)`
    /// of `bits` (`k` a multiple of 32). The blocks are `quantize_q8_0`'s, the engine's one Q8_0 rule; the
    /// card gets them in the two-plane layout of `q8_0_planes`. The bf16 is kept nowhere.
    pub fn from_bf16(
        gpu: &Gpu,
        bits: &[u16],
        rows: usize,
        k: usize,
        pick: &(dyn Fn(usize) -> usize + Sync),
    ) -> Result<Q8Rows, GpuError> {
        if k == 0 || !k.is_multiple_of(32) || bits.len() != rows * k {
            return Err(shape_err(format!(
                "{} values for {rows} rows of {k}: a Q8_0 row runs in blocks of 32 values",
                bits.len()
            )));
        }
        // Each chunk of rows is quantized on the resident pool and handed back once, tagged by its first row.
        let parts: Mutex<Vec<(usize, Vec<Q8Block>)>> = Mutex::new(Vec::new());
        threads::pool().for_each_chunk(rows, |range| {
            let mut blocks = Vec::with_capacity(range.len() * (k / 32));
            let mut vals = [0.0f32; 32];
            for r in range.clone() {
                let row = &bits[pick(r) * k..][..k];
                for blk in row.as_chunks::<32>().0 {
                    for (v, &b) in vals.iter_mut().zip(blk) {
                        *v = f32::from_bits(u32::from(b) << 16);
                    }
                    blocks.push(quantize_q8_0(&vals));
                }
            }
            parts
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((range.start, blocks));
        });
        let mut parts = parts.into_inner().unwrap_or_else(PoisonError::into_inner);
        parts.sort_by_key(|(first, _)| *first);
        let blocks: Vec<Q8Block> = parts.into_iter().flat_map(|(_, b)| b).collect();
        upload_blocks(gpu, &blocks, rows, k)
    }

    /// Tensor `name` of `split`, a Q8_0 matrix of `rows` rows of `k` values (ggml `[k, rows]`), on the
    /// card in the two-plane layout of `q8_0_planes` with the file's blocks unchanged. A missing
    /// tensor, another type and another shape are refused by name.
    pub fn from_gguf(
        gpu: &Gpu,
        split: &Split,
        name: &str,
        rows: usize,
        k: usize,
    ) -> Result<Q8Rows, GpuError> {
        Self::gguf_rows(gpu, split, name, &Take::all(rows, k, &|r| r))
    }

    /// [`Q8Rows::from_gguf`] of the rows `take` names.
    fn gguf_rows(
        gpu: &Gpu,
        split: &Split,
        name: &str,
        take: &Take<'_>,
    ) -> Result<Q8Rows, GpuError> {
        let k = take.k;
        if k == 0 || !k.is_multiple_of(32) {
            return Err(shape_err(format!(
                "{name}: rows of {k} values; a Q8_0 row runs in blocks of 32 values"
            )));
        }
        let bytes = gguf_tensor(split, name, GgmlType::Q8_0, &[k, take.file_rows])?;
        let row_bytes = k / 32 * 34;
        let mut blocks = Vec::with_capacity(take.rows * (k / 32));
        for r in 0..take.rows {
            let src = take.first + (take.pick)(r);
            let row = bytes
                .get(src * row_bytes..(src + 1) * row_bytes)
                .ok_or_else(|| {
                    shape_err(format!(
                        "{name}: row {src} of a matrix of {} rows",
                        take.file_rows
                    ))
                })?;
            blocks.extend(row.as_chunks::<34>().0.iter().map(Q8Block::from_bytes));
        }
        upload_blocks(gpu, &blocks, take.rows, k)
    }
}

/// `blocks`, `rows` rows of `k` values, on the card in the two-plane layout of `q8_0_planes`.
fn upload_blocks(gpu: &Gpu, blocks: &[Q8Block], rows: usize, k: usize) -> Result<Q8Rows, GpuError> {
    let (qs, d) = q8_0_planes(blocks);
    let stream = gpu.stream();
    Ok(Q8Rows {
        qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
        d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
    })
}

/// Tensor `name` of `split`'s bytes, refused by name when it is missing or is not of type `ty` and
/// shape `dims` (ggml's `ne` order).
fn gguf_tensor<'a>(
    split: &'a Split,
    name: &str,
    ty: GgmlType,
    dims: &[usize],
) -> Result<&'a [u8], GpuError> {
    let (shard, info) = split
        .find(name)
        .ok_or_else(|| GpuError::tensor(WHAT, name, "in the draft file"))?;
    let want: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
    if info.ty != ty || info.dims != want {
        return Err(shape_err(format!(
            "{name} is {} {:?}; the layers read {ty} {want:?}",
            info.ty, info.dims
        )));
    }
    let file = split
        .shard(shard)
        .ok_or_else(|| shape_err(format!("{name} names shard {shard}")))?;
    Ok(file.data(info)?)
}

/// Tensor `name` of `split`, f32 values of shape `dims` (ggml's `ne` order), refused as
/// [`Q8Rows::from_gguf`] refuses.
pub fn gguf_f32(split: &Split, name: &str, dims: &[usize]) -> Result<Vec<f32>, GpuError> {
    let bytes = gguf_tensor(split, name, GgmlType::F32, dims)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// Which rows of a matrix a load takes: `rows` rows, row `r` the file's row `first + pick(r)` of
/// `file_rows` rows of `k` values.
struct Take<'a> {
    file_rows: usize,
    k: usize,
    first: usize,
    rows: usize,
    pick: &'a (dyn Fn(usize) -> usize + Sync),
}

impl<'a> Take<'a> {
    /// Every row of a matrix of `rows` rows of `k`, in `pick`'s order.
    fn all(rows: usize, k: usize, pick: &'a (dyn Fn(usize) -> usize + Sync)) -> Take<'a> {
        Take {
            file_rows: rows,
            k,
            first: 0,
            rows,
            pick,
        }
    }

    /// Key head `head`'s `hd` rows of a matrix of `n_kv` such heads of rows of `k`, in `pick`'s order.
    fn head(
        n_kv: usize,
        hd: usize,
        k: usize,
        head: usize,
        pick: &'a (dyn Fn(usize) -> usize + Sync),
    ) -> Take<'a> {
        Take {
            file_rows: n_kv * hd,
            k,
            first: head * hd,
            rows: hd,
            pick,
        }
    }
}

/// A layer's tensors by role; each source names them in its file's terms.
#[derive(Clone, Copy, Debug)]
enum Role {
    InNorm,
    PostNorm,
    Q,
    K,
    V,
    QNorm,
    KNorm,
    O,
    Gate,
    Up,
    Down,
    AttnConvProj,
    AttnConvBase,
    FfnConvProj,
    FfnConvBase,
}

impl Role {
    /// The name's tail in a safetensors draft (`layers.{l}.` before it); its layers have no convolution.
    fn hf(self) -> Result<&'static str, GpuError> {
        Ok(match self {
            Role::InNorm => "input_layernorm.weight",
            Role::PostNorm => "post_attention_layernorm.weight",
            Role::Q => "self_attn.q_proj.weight",
            Role::K => "self_attn.k_proj.weight",
            Role::V => "self_attn.v_proj.weight",
            Role::QNorm => "self_attn.q_norm.weight",
            Role::KNorm => "self_attn.k_norm.weight",
            Role::O => "self_attn.o_proj.weight",
            Role::Gate => "mlp.gate_proj.weight",
            Role::Up => "mlp.up_proj.weight",
            Role::Down => "mlp.down_proj.weight",
            Role::AttnConvProj | Role::AttnConvBase | Role::FfnConvProj | Role::FfnConvBase => {
                return Err(shape_err(format!(
                    "{self:?}: a safetensors draft's layer has no convolution"
                )));
            }
        })
    }

    /// The name's tail in a DFlash2 GGUF draft (`dflash2::names::blk`).
    fn gguf(self) -> &'static str {
        match self {
            Role::InNorm => "attn_norm.weight",
            Role::PostNorm => "ffn_norm.weight",
            Role::Q => "attn_q.weight",
            Role::K => "attn_k.weight",
            Role::V => "attn_v.weight",
            Role::QNorm => "attn_q_norm.weight",
            Role::KNorm => "attn_k_norm.weight",
            Role::O => "attn_output.weight",
            Role::Gate => "ffn_gate.weight",
            Role::Up => "ffn_up.weight",
            Role::Down => "ffn_down.weight",
            Role::AttnConvProj => "attn_conv_proj.weight",
            Role::AttnConvBase => "attn_conv_base",
            Role::FfnConvProj => "ffn_conv_proj.weight",
            Role::FfnConvBase => "ffn_conv_base",
        }
    }
}

/// Where a load reads a layer's tensors.
trait LayerSource {
    /// The rows `take` names of layer `l`'s `role` matrix as Q8_0 rows on the card.
    fn rows(&self, gpu: &Gpu, l: usize, role: Role, take: &Take<'_>) -> Result<Q8Rows, GpuError>;

    /// Layer `l`'s `role` values of shape `dims` (ggml's `ne` order), widened to f32.
    fn values(&self, l: usize, role: Role, dims: &[usize]) -> Result<Vec<f32>, GpuError>;
}

/// A safetensors draft's layers: every matrix quantized to Q8_0 on the host.
struct HfDraft<'a>(&'a dyn Bf16Source);

impl LayerSource for HfDraft<'_> {
    fn rows(&self, gpu: &Gpu, l: usize, role: Role, take: &Take<'_>) -> Result<Q8Rows, GpuError> {
        let k = take.k;
        let bits = matrix(self.0, l, role.hf()?, take.file_rows, k)?;
        let part = bits
            .get(take.first * k..(take.first + take.rows) * k)
            .ok_or_else(|| {
                shape_err(format!(
                    "{role:?}: rows {}..{} of {}",
                    take.first,
                    take.first + take.rows,
                    take.file_rows
                ))
            })?;
        Q8Rows::from_bf16(gpu, part, take.rows, k, take.pick)
    }

    fn values(&self, l: usize, role: Role, dims: &[usize]) -> Result<Vec<f32>, GpuError> {
        match *dims {
            [len] => Ok(widen(vector(self.0, l, role.hf()?, len)?)),
            _ => Err(shape_err(format!(
                "{role:?} of shape {dims:?}: a safetensors draft's layer holds vectors only"
            ))),
        }
    }
}

/// A DFlash2 GGUF draft's layers: the Q8_0 rows and the f32 values as the file stores them.
struct GgufDraft<'a>(&'a Split);

impl LayerSource for GgufDraft<'_> {
    fn rows(&self, gpu: &Gpu, l: usize, role: Role, take: &Take<'_>) -> Result<Q8Rows, GpuError> {
        Q8Rows::gguf_rows(gpu, self.0, &dflash2::names::blk(l, role.gguf()), take)
    }

    fn values(&self, l: usize, role: Role, dims: &[usize]) -> Result<Vec<f32>, GpuError> {
        gguf_f32(self.0, &dflash2::names::blk(l, role.gguf()), dims)
    }
}

/// One layer's convolution weights (`super::conv`): each sub-layer's coefficient projection and base.
struct ConvWeights {
    attn_proj: Q8Rows,
    attn_base: DeviceBuffer<f32>,
    ffn_proj: Q8Rows,
    ffn_base: DeviceBuffer<f32>,
}

/// The convolution's module, shape and scratch.
struct ConvPass {
    k: ConvKernels,
    conv: Conv,
    /// The rows' width.
    n: usize,
    /// The rows' coefficients, [`BLOCK_MAX`] rows of [`Conv::coeffs`].
    coeff: DeviceBuffer<f32>,
    /// Side 0's output: the sub-layer's projections' input.
    xc: DeviceBuffer<f32>,
    /// Side 1's output: what the residual adds.
    cv: DeviceBuffer<f32>,
}

impl ConvPass {
    /// Side 0 over `xn`, a sub-layer's `m` normed rows: their coefficients from `proj`, then the mix
    /// into `xc`.
    fn input(
        &mut self,
        gpu: &Gpu,
        proj: &Q8Rows,
        base: &DeviceBuffer<f32>,
        xn: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        gemv(gpu, proj, xn, m, &mut self.coeff)?;
        self.mix(gpu, base, xn, m, Side::Input)
    }

    /// Side 1 over `out`, the sub-layer's `m` output rows, into `cv`, from the coefficients
    /// [`ConvPass::input`] left.
    fn output(
        &mut self,
        gpu: &Gpu,
        base: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.mix(gpu, base, out, m, Side::Output)
    }

    fn mix(
        &mut self,
        gpu: &Gpu,
        base: &DeviceBuffer<f32>,
        h: &DeviceBuffer<f32>,
        m: usize,
        side: Side,
    ) -> Result<(), GpuError> {
        let y = match side {
            Side::Input => &mut self.xc,
            Side::Output => &mut self.cv,
        };
        self.k.enqueue_conv(
            gpu.stream(),
            ConvArgs {
                h,
                coeff: &self.coeff,
                base,
                conv: self.conv,
                n: self.n,
                m,
                side,
                y,
            },
        )
    }
}

/// The layer's convolution: the pass's with the layer's weights, both or neither (one load makes them
/// together).
fn conv_of<'a>(
    pass: &'a mut Option<ConvPass>,
    w: &'a Option<ConvWeights>,
) -> Result<Option<(&'a mut ConvPass, &'a ConvWeights)>, GpuError> {
    match (pass, w) {
        (Some(p), Some(w)) => Ok(Some((p, w))),
        (None, None) => Ok(None),
        _ => Err(GpuError::State {
            what: WHAT,
            missing: "a layer's convolution weights beside the pass's convolution",
        }),
    }
}

/// The target's embedding the block's rows come from ([`super::EmbedSource::Target`]).
struct TargetEmbed {
    table: EmbedTable,
    /// [`BLOCK_MAX`] rows: the staged anchor's, then the mask's.
    rows: DeviceBuffer<f32>,
    /// The anchor's row on the host.
    row: Vec<f32>,
}

/// One layer on the card.
struct Layer {
    ln_in: DeviceBuffer<f32>,
    ln_post: DeviceBuffer<f32>,
    q: Q8Rows,
    /// Per key head: its `head_dim` rows of K and V.
    k: Vec<Q8Rows>,
    v: Vec<Q8Rows>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    o: Q8Rows,
    gate: Q8Rows,
    up: Q8Rows,
    down: Q8Rows,
    /// The convolution's weights, when the layers carry it.
    conv: Option<ConvWeights>,
    /// `GUARD + n_kv * planes` rows of `head_dim` f16.
    kc: DeviceTensor<u16>,
    vc: DeviceTensor<u16>,
}

/// The pass's activations and the staged images.
struct Scratch {
    xn: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    x1: DeviceBuffer<f32>,
    xn2: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    dn: DeviceBuffer<f32>,
    kb: Vec<DeviceBuffer<f32>>,
    kn: Vec<DeviceBuffer<f32>>,
    kr: Vec<DeviceBuffer<f32>>,
    vb: Vec<DeviceBuffer<f32>>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// The block's image: `n_kv` words (the cache row each key head's block append starts at), then
    /// [`BLOCK_MAX`] key counts; under the per-row window then the append's second rows and the rows'
    /// first ring rows ([`per_row_image`]).
    blk_words: DeviceBuffer<u32>,
    /// The block rows' cos/sin, `BLOCK_MAX` rows of `head_dim`.
    blk_cs: DeviceBuffer<f32>,
    /// The append's image: `n_kv` words (the cache row of the append's first launch for each key head),
    /// then `n_kv` (its second launch's).
    ctx_words: DeviceBuffer<u32>,
    /// The appended rows' cos/sin.
    ctx_cs: DeviceBuffer<f32>,
}

/// The dense draft layers on the card ([`DraftLayers`]).
pub struct DenseLayers {
    dims: DenseDims,
    spec: DenseSpec,
    layers: Vec<Layer>,
    flash: FlashGqaKernels,
    rope: RopeTable,
    /// What the load set beyond the plain layer.
    features: Features,
    /// The convolution's module and scratch, when the layers carry it.
    conv: Option<ConvPass>,
    /// The target's embedding, when the layers embed the block.
    embed: Option<TargetEmbed>,
    /// Context rows each plane holds.
    cap: usize,
    /// Rows of one key head's plane: `cap` and the block's scratch rows.
    planes: usize,
    s: Scratch,
    /// The position of the first context row since the last reset, and the rows appended since.
    base: Option<u32>,
    count: usize,
    /// The block rows' staged count, for the pass to check.
    staged_block: Option<usize>,
    staged_rows: Option<usize>,
    stats: DenseStats,
}

/// What the layers have staged since they were loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseStats {
    /// Appends staged ([`DraftLayers::stage_kv`]).
    pub appends: usize,
    /// Of them, the ones whose rows run past the ring's end and continue at its front.
    pub wrapped: usize,
    /// Block passes staged ([`DraftLayers::stage_block`]).
    pub blocks: usize,
}

fn up_f32(gpu: &Gpu, v: &[f32]) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::from_host(gpu.stream(), v)?)
}

fn zeros_f32(gpu: &Gpu, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::zeroed(gpu.stream(), n.max(1))?)
}

/// Key head `head`'s whole buffer of `bufs` as the view an append reads.
fn whole(bufs: &[DeviceBuffer<f32>], head: usize) -> Result<Window<'_, f32>, GpuError> {
    let buf = bufs
        .get(head)
        .ok_or_else(|| shape_err(format!("key head {head} of {}", bufs.len())))?;
    Window::of(buf, 0, buf.len())
}

impl DenseLayers {
    /// Load `spec.layers` layers from `src`: every matrix quantized to Q8_0 on the host (the bf16 is kept
    /// nowhere), the query and key rows permuted as the module doc says, the caches zeroed.
    pub fn load(
        gpu: &Gpu,
        spec: &DenseSpec,
        src: &dyn Bf16Source,
    ) -> Result<DenseLayers, GpuError> {
        let dims = DenseDims::read(src)?;
        Self::assemble(gpu, spec, dims, Features::NONE, &HfDraft(src), None)
    }

    /// Load `spec.layers` layers from `src` with the block embedded from the target's table `embed`:
    /// row 0 of a block is the anchor token's row and rows 1.. are `mask_id`'s
    /// ([`DenseLayers::load_dflash2`]'s embedding, without its convolution or per-row window). A mask
    /// id past `embed`'s vocabulary and every tensor of another type or shape are refused by name.
    pub fn load_target_embed(
        gpu: &Gpu,
        spec: &DenseSpec,
        src: &dyn Bf16Source,
        embed: EmbedTable,
        mask_id: u32,
    ) -> Result<DenseLayers, GpuError> {
        let dims = DenseDims::read(src)?;
        let features = Features {
            target_embed: true,
            ..Features::NONE
        };
        Self::assemble(
            gpu,
            spec,
            dims,
            features,
            &HfDraft(src),
            Some((embed, mask_id)),
        )
    }

    /// Load a DFlash2 draft's layers from its GGUF file `draft`, `hp` its header
    /// ([`dflash2::Hparams::read`]), for contexts of up to `ctx` positions, the block embedded from the
    /// target's table `embed`: the module doc's DFlash2 layer, the file's Q8_0 rows on the card as it
    /// stores them, the caches zeroed. A head, group, width, window or convolution the stack does not
    /// run, a mask id past `embed`'s vocabulary and every tensor of another type or shape are refused by
    /// name.
    pub fn load_dflash2(
        gpu: &Gpu,
        draft: &Split,
        hp: &Hparams,
        ctx: usize,
        embed: EmbedTable,
    ) -> Result<DenseLayers, GpuError> {
        let (dims, spec, features) = Self::dflash2_parts(hp, ctx)?;
        let embed = Some((embed, hp.mask_token));
        Self::assemble(gpu, &spec, dims, features, &GgufDraft(draft), embed)
    }

    /// Card bytes of a DFlash2 draft's layers ([`DenseLayers::load_dflash2`]) at `ctx` positions,
    /// refused as the load refuses `hp`.
    pub fn dflash2_card_bytes(hp: &Hparams, ctx: usize) -> Result<u64, GpuError> {
        let (dims, spec, features) = Self::dflash2_parts(hp, ctx)?;
        Ok(dims.bytes(&spec, features))
    }

    /// A DFlash2 header's widths, spec and features, refused by name where the stack does not run it.
    fn dflash2_parts(
        hp: &Hparams,
        ctx: usize,
    ) -> Result<(DenseDims, DenseSpec, Features), GpuError> {
        let dims = DenseDims {
            hidden: hp.n_embd,
            n_q: hp.n_head,
            n_kv: hp.n_head_kv,
            head_dim: hp.head_dim,
            inter: hp.n_ff,
        };
        let (n_q, n_kv) = (dims.n_q, dims.n_kv);
        if dims.head_dim != HEAD
            || n_kv == 0
            || n_q == 0
            || !n_q.is_multiple_of(n_kv)
            || !(n_q / n_kv).is_multiple_of(PACK_4)
        {
            return Err(shape_err(format!(
                "{n_q} query heads over {n_kv} key heads of {} values; the ring pass takes heads of \
                 {HEAD} in groups of a multiple of {PACK_4}",
                dims.head_dim
            )));
        }
        for (what, v) in [
            ("hidden", dims.hidden),
            ("inter", dims.inter),
            ("q width", n_q * HEAD),
        ] {
            if v == 0 || !v.is_multiple_of(32) {
                return Err(shape_err(format!(
                    "{what} {v}: the Q8_0 rows run in blocks of 32 values"
                )));
            }
        }
        let c = hp.conv;
        if c.kernel == 0 || c.group == 0 || !dims.hidden.is_multiple_of(c.group) {
            return Err(shape_err(format!(
                "a convolution of kernel {} in groups of {} over {} channels: both nonzero, the group \
                 dividing the channels",
                c.kernel, c.group, dims.hidden
            )));
        }
        if hp.window < BLOCK_MAX || hp.block_size == 0 || hp.block_size > BLOCK_MAX {
            return Err(shape_err(format!(
                "a window of {} and blocks of {}: a block of 1..={BLOCK_MAX} rows inside the window",
                hp.window, hp.block_size
            )));
        }
        let spec = DenseSpec {
            layers: hp.n_layer,
            eps: hp.rms_eps,
            rope_theta: hp.rope_base,
            window: Some(hp.window),
            block_causal: false,
            ctx,
        };
        let features = Features {
            per_row: Some(hp.window),
            conv: Some(c),
            target_embed: true,
        };
        Ok((dims, spec, features))
    }

    /// The layers of `src` at `dims` with `features` set: `spec.layers` of them, the query and key rows
    /// permuted as the module doc says, the caches zeroed, and with `embed` the target's table and the
    /// mask id the block's rows 1.. embed.
    fn assemble(
        gpu: &Gpu,
        spec: &DenseSpec,
        dims: DenseDims,
        features: Features,
        src: &dyn LayerSource,
        embed: Option<(EmbedTable, u32)>,
    ) -> Result<DenseLayers, GpuError> {
        let DenseDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = dims;
        if spec.layers == 0 || spec.ctx == 0 || spec.window == Some(0) {
            return Err(shape_err(format!(
                "{} layers, a context of {} and a window of {:?}",
                spec.layers, spec.ctx, spec.window
            )));
        }
        if features.target_embed != embed.is_some() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the target's table beside the features that embed from it",
            });
        }
        let cap = ring_cap(spec, features);
        let planes = ring_planes(cap);
        let rope = RopeTable::new(&RopeSpec::window(spec.rope_theta, hd))?;
        // The file rotates pair (i, i + hd/2); the card's kernel rotates adjacent pairs (2i, 2i + 1). Rows of
        // q and k are reordered inside each head the same way, so every q.k dot product sums the same
        // products in another order, and v, o_proj and the cache keep the file's order.
        let perm_head = move |r: usize| (r / hd) * hd + adjacent_pairs(hd, r % hd);
        let pairs = move |r: usize| adjacent_pairs(hd, r);
        let ident = |r: usize| r;
        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let permuted_gain = |role: Role| -> Result<DeviceBuffer<f32>, GpuError> {
                let g = src.values(l, role, &[hd])?;
                let p: Vec<f32> = (0..hd).map(|j| g[adjacent_pairs(hd, j)]).collect();
                up_f32(gpu, &p)
            };
            let plain_gain = |role: Role, n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
                up_f32(gpu, &src.values(l, role, &[n])?)
            };
            let all = |role: Role, rows: usize, k: usize| -> Result<Q8Rows, GpuError> {
                src.rows(gpu, l, role, &Take::all(rows, k, &ident))
            };
            let q = src.rows(gpu, l, Role::Q, &Take::all(n_q * hd, h, &perm_head))?;
            let mut k = Vec::with_capacity(n_kv);
            let mut v = Vec::with_capacity(n_kv);
            for head in 0..n_kv {
                let k_rows = Take::head(n_kv, hd, h, head, &pairs);
                k.push(src.rows(gpu, l, Role::K, &k_rows)?);
                let v_rows = Take::head(n_kv, hd, h, head, &ident);
                v.push(src.rows(gpu, l, Role::V, &v_rows)?);
            }
            let o = all(Role::O, h, n_q * hd)?;
            let gate = all(Role::Gate, inter, h)?;
            let up = all(Role::Up, inter, h)?;
            let down = all(Role::Down, h, inter)?;
            let ln_in = plain_gain(Role::InNorm, h)?;
            let ln_post = plain_gain(Role::PostNorm, h)?;
            let q_norm = permuted_gain(Role::QNorm)?;
            let k_norm = permuted_gain(Role::KNorm)?;
            let conv = match features.conv {
                None => None,
                Some(c) => {
                    let base = |role: Role| -> Result<DeviceBuffer<f32>, GpuError> {
                        up_f32(gpu, &src.values(l, role, &[h, c.kernel, 2])?)
                    };
                    Some(ConvWeights {
                        attn_proj: all(Role::AttnConvProj, c.coeffs(h), h)?,
                        attn_base: base(Role::AttnConvBase)?,
                        ffn_proj: all(Role::FfnConvProj, c.coeffs(h), h)?,
                        ffn_base: base(Role::FfnConvBase)?,
                    })
                }
            };
            layers.push(Layer {
                ln_in,
                ln_post,
                q,
                k,
                v,
                q_norm,
                k_norm,
                o,
                gate,
                up,
                down,
                conv,
                kc: DeviceTensor::zeroed(gpu.stream(), GUARD + n_kv * planes, hd)?,
                vc: DeviceTensor::zeroed(gpu.stream(), GUARD + n_kv * planes, hd)?,
            });
        }
        let m = BLOCK_MAX;
        let per_head = |gpu: &Gpu| -> Result<Vec<DeviceBuffer<f32>>, GpuError> {
            (0..n_kv).map(|_| zeros_f32(gpu, m * hd)).collect()
        };
        let s = Scratch {
            xn: zeros_f32(gpu, m * h)?,
            q: zeros_f32(gpu, m * n_q * hd)?,
            qn: zeros_f32(gpu, m * n_q * hd)?,
            qr: zeros_f32(gpu, m * n_q * hd)?,
            attn: zeros_f32(gpu, m * n_q * hd)?,
            o: zeros_f32(gpu, m * h)?,
            x1: zeros_f32(gpu, m * h)?,
            xn2: zeros_f32(gpu, m * h)?,
            g: zeros_f32(gpu, m * inter)?,
            u: zeros_f32(gpu, m * inter)?,
            act: zeros_f32(gpu, m * inter)?,
            dn: zeros_f32(gpu, m * h)?,
            kb: per_head(gpu)?,
            kn: per_head(gpu)?,
            kr: per_head(gpu)?,
            vb: per_head(gpu)?,
            part_v: zeros_f32(gpu, partials_v(m, n_q, hd))?,
            part_ms: zeros_f32(gpu, partials_ms_len(m, n_q))?,
            blk_words: DeviceBuffer::from_host(
                gpu.stream(),
                &vec![1u32; block_words(n_kv, features)],
            )?,
            blk_cs: zeros_f32(gpu, m * hd)?,
            ctx_words: DeviceBuffer::from_host(gpu.stream(), &vec![0u32; 2 * n_kv])?,
            ctx_cs: zeros_f32(gpu, m * hd)?,
        };
        let conv = match features.conv {
            None => None,
            Some(c) => Some(ConvPass {
                k: ConvKernels::load(gpu.context())?,
                conv: c,
                n: h,
                coeff: zeros_f32(gpu, m * c.coeffs(h))?,
                xc: zeros_f32(gpu, m * h)?,
                cv: zeros_f32(gpu, m * h)?,
            }),
        };
        let embed = match embed {
            None => None,
            Some((table, mask)) => {
                let mut mask_row = vec![0.0f32; h];
                table.row_into(mask, &mut mask_row)?;
                let mut rows = vec![0.0f32; h];
                for _ in 1..BLOCK_MAX {
                    rows.extend_from_slice(&mask_row);
                }
                Some(TargetEmbed {
                    table,
                    rows: up_f32(gpu, &rows)?,
                    row: vec![0.0f32; h],
                })
            }
        };
        Ok(DenseLayers {
            dims,
            spec: spec.clone(),
            layers,
            flash: FlashGqaKernels::load(gpu.context())?,
            rope,
            features,
            conv,
            embed,
            cap,
            planes,
            s,
            base: None,
            count: 0,
            staged_block: None,
            staged_rows: None,
            stats: DenseStats::default(),
        })
    }

    /// The spec the layers were loaded with.
    #[must_use]
    pub fn spec(&self) -> &DenseSpec {
        &self.spec
    }

    /// What the layers have staged.
    #[must_use]
    pub fn stats(&self) -> DenseStats {
        self.stats
    }

    /// The widths the layers run at.
    #[must_use]
    pub fn dims(&self) -> DenseDims {
        self.dims
    }

    /// Rows a key head's ring holds before it wraps: context rows, and under the per-row window the
    /// block's own rows among them.
    #[must_use]
    pub fn ring_rows(&self) -> usize {
        self.cap
    }

    /// Context rows the next block pass sees: the ring's fill, and under the per-row window the
    /// window's (row `t` reads `min(visible, window − 1 − t)` of them, [`PerRow`]).
    #[must_use]
    pub fn visible(&self) -> usize {
        match self.features.per_row {
            None => self.count.min(self.cap),
            Some(w) => self.count.min(w),
        }
    }

    /// The most context rows the ring takes since the last reset, or `None` when a window bounds
    /// what a block reads: a draft with no window keeps every row, and the per-row window's ring holds
    /// a whole window unless the context is shorter.
    fn capacity(&self) -> Option<usize> {
        match self.features.per_row {
            None => self.spec.window.is_none().then_some(self.cap),
            Some(w) => {
                let holds = self.cap - BLOCK_MAX;
                (holds < w - 1).then_some(holds)
            }
        }
    }

    /// The cos/sin rows of `n` positions from `first`, `head_dim` f32 each, padded to [`BLOCK_MAX`] rows.
    fn cos_sin(&self, first: u32, n: usize) -> Vec<f32> {
        let hd = self.dims.head_dim;
        let mut out = Vec::with_capacity(BLOCK_MAX * hd);
        for t in 0..n {
            self.rope.push(
                first + u32::try_from(t).unwrap_or(0),
                Direction::Forward,
                &mut out,
            );
        }
        out.resize(BLOCK_MAX * hd, 0.0);
        out
    }

    /// Cache row of slot `slot` of key head `h`.
    fn row(&self, h: usize, slot: usize) -> u32 {
        u32::try_from(GUARD + h * self.planes + slot).expect("a plane row fits u32")
    }

    /// The buffers this load allocated, counted from the buffers themselves.
    /// Weights are these layers', in the order [`DenseLayers::load`] allocated
    /// them, then the owner's ([`OwnHeld::weights`]). One layer's weights are
    /// its query, each key head beside that head's value, the output and the
    /// three MLP matrices, then its four norm gains, then, when the layers
    /// carry the convolution, its attention projection and base and its MLP
    /// projection and base. The rings are each layer's K plane and then its V
    /// plane. Working bytes are this pass's scratch and images (the
    /// convolution's and the target embedding's rows among them) plus the
    /// owner's. Kernel modules and captured
    /// graphs are not buffers. A zero-length buffer allocates nothing and is
    /// not listed.
    #[must_use]
    pub fn held(&self, own: &OwnHeld) -> DraftHeld {
        let mut weights = self.weight_buffers();
        weights.extend(own.weights.iter().copied().filter(|&n| n > 0));
        DraftHeld {
            weights,
            rings: self.ring_bytes(),
            working: self.working_bytes() + own.working,
        }
    }

    /// Weight buffers in [`DenseLayers::load`]'s order. See [`DenseLayers::held`].
    fn weight_buffers(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for layer in &self.layers {
            push_q8(&mut out, &layer.q);
            for (k, v) in layer.k.iter().zip(&layer.v) {
                push_q8(&mut out, k);
                push_q8(&mut out, v);
            }
            push_q8(&mut out, &layer.o);
            push_q8(&mut out, &layer.gate);
            push_q8(&mut out, &layer.up);
            push_q8(&mut out, &layer.down);
            push_live(&mut out, buffer_bytes(&layer.ln_in));
            push_live(&mut out, buffer_bytes(&layer.ln_post));
            push_live(&mut out, buffer_bytes(&layer.q_norm));
            push_live(&mut out, buffer_bytes(&layer.k_norm));
            if let Some(c) = &layer.conv {
                push_q8(&mut out, &c.attn_proj);
                push_live(&mut out, buffer_bytes(&c.attn_base));
                push_q8(&mut out, &c.ffn_proj);
                push_live(&mut out, buffer_bytes(&c.ffn_base));
            }
        }
        out
    }

    /// Each layer's K plane, then its V plane.
    fn ring_bytes(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for layer in &self.layers {
            push_live(&mut out, buffer_bytes(layer.kc.buf()));
            push_live(&mut out, buffer_bytes(layer.vc.buf()));
        }
        out
    }

    /// The pass's scratch and the staged images, in all.
    fn working_bytes(&self) -> u64 {
        let s = &self.s;
        let mut n = buffer_bytes(&s.xn)
            + buffer_bytes(&s.q)
            + buffer_bytes(&s.qn)
            + buffer_bytes(&s.qr)
            + buffer_bytes(&s.attn)
            + buffer_bytes(&s.o)
            + buffer_bytes(&s.x1)
            + buffer_bytes(&s.xn2)
            + buffer_bytes(&s.g)
            + buffer_bytes(&s.u)
            + buffer_bytes(&s.act)
            + buffer_bytes(&s.dn)
            + buffer_bytes(&s.part_v)
            + buffer_bytes(&s.part_ms)
            + buffer_bytes(&s.blk_words)
            + buffer_bytes(&s.blk_cs)
            + buffer_bytes(&s.ctx_words)
            + buffer_bytes(&s.ctx_cs);
        for b in s.kb.iter().chain(&s.kn).chain(&s.kr).chain(&s.vb) {
            n += buffer_bytes(b);
        }
        if let Some(c) = &self.conv {
            n += buffer_bytes(&c.coeff) + buffer_bytes(&c.xc) + buffer_bytes(&c.cv);
        }
        if let Some(e) = &self.embed {
            n += buffer_bytes(&e.rows);
        }
        n
    }
}

/// Bytes of `buf`.
fn buffer_bytes<T>(buf: &DeviceBuffer<T>) -> u64 {
    buf.num_bytes() as u64
}

/// List `bytes` when the buffer allocated anything.
fn push_live(out: &mut Vec<u64>, bytes: u64) {
    if bytes > 0 {
        out.push(bytes);
    }
}

/// The two planes of a Q8_0 matrix, quants then scales.
fn push_q8(out: &mut Vec<u64>, rows: &Q8Rows) {
    push_live(out, buffer_bytes(rows.qs.buf()));
    push_live(out, buffer_bytes(rows.d.buf()));
}

impl DraftLayers for DenseLayers {
    const MAX_ROWS: usize = BLOCK_MAX;
    const APPEND_GROUP: usize = APPEND_MAX;

    fn hidden(&self) -> usize {
        self.dims.hidden
    }

    fn replayable(&self) -> Result<(), GpuError> {
        // The owner stages, then asks, then launches. Both images empty means
        // this replay has no stage call behind it. An eager pass checks its
        // own count in the enqueue, including a count that does not match.
        if self.staged_rows.is_none() && self.staged_block.is_none() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a replay before stage_kv and stage_block",
            });
        }
        Ok(())
    }

    fn stage_kv(&mut self, gpu: &Gpu, first: u32, n: usize) -> Result<(), GpuError> {
        if n == 0 || n > APPEND_MAX {
            return Err(shape_err(format!(
                "an append of {n} rows; a stage holds 1..={APPEND_MAX}"
            )));
        }
        match self.base {
            None => self.base = Some(first),
            Some(base) => {
                let next = u64::from(base) + self.count as u64;
                if u64::from(first) != next {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "context rows from position {first}; the ring holds {} rows from \
                             {base}, so the next row is {next}",
                            self.count
                        ),
                    });
                }
            }
        }
        let past = self.count;
        if let Some(cap) = self.capacity().filter(|&cap| past + n > cap) {
            return Err(shape_err(format!(
                "{} context rows past a capacity of {cap}",
                past + n
            )));
        }
        let slot0 = past % self.cap;
        // Rows that run past the ring's end continue at its front: the second launch starts that many
        // rows before slot 0, so its rows from `wrap` on land at slots 0.., and its first `wrap` rows
        // land in the cache's guard (head 0) or the previous head's block scratch, which no pass reads.
        let wrap = if slot0 + n > self.cap {
            self.cap - slot0
        } else {
            0
        };
        let n_kv = self.dims.n_kv;
        let words: Vec<u32> = (0..n_kv)
            .map(|h| self.row(h, slot0))
            .chain((0..n_kv).map(|h| {
                if wrap == 0 {
                    self.row(h, slot0)
                } else {
                    self.row(h, 0) - u32::try_from(wrap).unwrap_or(0)
                }
            }))
            .collect();
        self.s.ctx_words.copy_from_host(gpu.stream(), &words)?;
        let cs = self.cos_sin(first, n);
        self.s.ctx_cs.copy_from_host(gpu.stream(), &cs)?;
        self.count += n;
        self.staged_rows = Some(n);
        self.stats.appends += 1;
        self.stats.wrapped += usize::from(wrap > 0);
        Ok(())
    }

    fn append_kv(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, n: usize) -> Result<(), GpuError> {
        // The owner records its graphs at load, before any stage call: the launches read the staged
        // words from device addresses at replay. Only an eager launch needs its stage call first.
        if self.staged_rows != Some(n) && !capturing(gpu.stream())? {
            return Err(GpuError::State {
                what: WHAT,
                missing: "an append of the rows stage_kv staged",
            });
        }
        let DenseDims {
            hidden: h,
            n_kv,
            head_dim: hd,
            ..
        } = self.dims;
        if x.len() < n * h {
            return Err(shape_err(format!(
                "x holds {} values; {n} rows are {}",
                x.len(),
                n * h
            )));
        }
        let stream = gpu.stream();
        let (elem, flash) = (gpu.elem(), gpu.flash());
        let fault = gpu.unlabelled_sink();
        let eps = self.spec.eps;
        let Self { layers, s, .. } = self;
        for layer in layers.iter_mut() {
            for head in 0..n_kv {
                gemv(gpu, &layer.k[head], x, n, &mut s.kb[head])?;
                gemv(gpu, &layer.v[head], x, n, &mut s.vb[head])?;
                elem.enqueue_rms_norm(
                    stream,
                    &s.kb[head],
                    &layer.k_norm,
                    eps,
                    hd,
                    n,
                    &mut s.kn[head],
                )?;
                elem.enqueue_rope(stream, &s.kn[head], &s.ctx_cs, hd, 1, n, &mut s.kr[head])?;
            }
            for second in 0..2 {
                let at = PlaneAppend {
                    words: &s.ctx_words,
                    first_word: second * n_kv,
                    heads: n_kv,
                    m: n,
                    fault,
                };
                flash.enqueue_kv_append_planes(stream, &at, |h| whole(&s.kr, h), &mut layer.kc)?;
                flash.enqueue_kv_append_planes(stream, &at, |h| whole(&s.vb, h), &mut layer.vc)?;
            }
        }
        Ok(())
    }

    fn stage_block(
        &mut self,
        gpu: &Gpu,
        last: u32,
        pos: u32,
        m: usize,
        ctx: usize,
    ) -> Result<(), GpuError> {
        if m == 0 || m > BLOCK_MAX {
            return Err(shape_err(format!(
                "a block of {m} rows; a pass holds 1..={BLOCK_MAX}"
            )));
        }
        if let Some(base) = self.base {
            let next = u64::from(base) + self.count as u64;
            if u64::from(pos) != next {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "a block at position {pos}; the {} context rows from {base} end at {next}",
                        self.count
                    ),
                });
            }
        }
        let valid = self.visible();
        if ctx != valid {
            return Err(shape_err(format!(
                "a block over {ctx} visible context rows; the ring holds {valid}"
            )));
        }
        let n_kv = self.dims.n_kv;
        let words = match self.features.per_row {
            None => {
                let mut words: Vec<u32> = (0..n_kv).map(|h| self.row(h, valid)).collect();
                for t in 0..BLOCK_MAX {
                    let seen = if self.spec.block_causal { t + 1 } else { m };
                    words.push(u32::try_from(valid + seen.min(BLOCK_MAX)).unwrap_or(u32::MAX));
                }
                words
            }
            Some(window) => {
                let p = PerRow {
                    window,
                    cap: self.cap,
                    count: self.count,
                    m,
                    causal: self.spec.block_causal,
                };
                per_row_image(p, n_kv, |h, slot| self.row(h, slot))
            }
        };
        self.s.blk_words.copy_from_host(gpu.stream(), &words)?;
        let cs = self.cos_sin(pos, m);
        self.s.blk_cs.copy_from_host(gpu.stream(), &cs)?;
        if let Some(e) = self.embed.as_mut() {
            e.table.row_into(last, &mut e.row)?;
            let mut first = WindowMut::<f32>::of_mut(&mut e.rows, 0, e.row.len())?;
            first.copy_from_host(gpu.stream(), &e.row)?;
        }
        self.staged_block = Some(m);
        self.stats.blocks += 1;
        Ok(())
    }

    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        // As in `append_kv`: a capture runs before any stage call, an eager launch after its own.
        if self.staged_block != Some(m) && !capturing(gpu.stream())? {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a block of the rows stage_block staged",
            });
        }
        let DenseDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = self.dims;
        if x.len() < m * h {
            return Err(shape_err(format!(
                "x holds {} values; {m} rows are {}",
                x.len(),
                m * h
            )));
        }
        let stream = gpu.stream();
        let (elem, flash) = (gpu.elem(), gpu.flash());
        let fault = gpu.unlabelled_sink();
        let eps = self.spec.eps;
        let (ring, planes) = (self.cap, self.planes);
        let per_row = self.features.per_row.is_some();
        let group = n_q / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let Self {
            layers,
            s,
            flash: gqa,
            conv,
            embed,
            ..
        } = self;
        if let Some(e) = embed.as_ref() {
            let rows = Window::<f32>::of(&e.rows, 0, m * h)?;
            let mut input = WindowMut::<f32>::of_mut(&mut *x, 0, m * h)?;
            input.copy_from_device_async(&rows, stream)?;
        }
        let n_keys = Window::<u32>::of(&s.blk_words, 4 * n_kv, BLOCK_MAX)?;
        // The per-row window's block append is two launches, as a context append is (the image's
        // second rows from word `n_kv + BLOCK_MAX`), and each row's run starts at its own ring row.
        let first_words = [0, n_kv + BLOCK_MAX];
        let appends = if per_row {
            &first_words[..]
        } else {
            &first_words[..1]
        };
        let row0 = if per_row {
            Some(Window::<u32>::of(
                &s.blk_words,
                4 * (2 * n_kv + BLOCK_MAX),
                BLOCK_MAX,
            )?)
        } else {
            None
        };
        for layer in layers.iter_mut() {
            elem.enqueue_rms_norm(stream, x, &layer.ln_in, eps, h, m, &mut s.xn)?;
            let input = match conv_of(conv, &layer.conv)? {
                Some((c, w)) => {
                    c.input(gpu, &w.attn_proj, &w.attn_base, &s.xn, m)?;
                    &c.xc
                }
                None => &s.xn,
            };
            gemv(gpu, &layer.q, input, m, &mut s.q)?;
            elem.enqueue_rms_norm(stream, &s.q, &layer.q_norm, eps, hd, m * n_q, &mut s.qn)?;
            elem.enqueue_rope(
                stream,
                &s.qn,
                &s.blk_cs,
                hd,
                u32::try_from(n_q).unwrap_or(0),
                m,
                &mut s.qr,
            )?;
            for head in 0..n_kv {
                gemv(gpu, &layer.k[head], input, m, &mut s.kb[head])?;
                gemv(gpu, &layer.v[head], input, m, &mut s.vb[head])?;
                elem.enqueue_rms_norm(
                    stream,
                    &s.kb[head],
                    &layer.k_norm,
                    eps,
                    hd,
                    m,
                    &mut s.kn[head],
                )?;
                elem.enqueue_rope(stream, &s.kn[head], &s.blk_cs, hd, 1, m, &mut s.kr[head])?;
            }
            for &first_word in appends {
                let at = PlaneAppend {
                    words: &s.blk_words,
                    first_word,
                    heads: n_kv,
                    m,
                    fault,
                };
                flash.enqueue_kv_append_planes(stream, &at, |h| whole(&s.kr, h), &mut layer.kc)?;
                flash.enqueue_kv_append_planes(stream, &at, |h| whole(&s.vb, h), &mut layer.vc)?;
            }
            let plane_bytes = 2 * GUARD * hd;
            let plane_len = n_kv * planes * hd;
            let kc = Window::<u16>::of(layer.kc.buf(), plane_bytes, plane_len)?;
            let vc = Window::<u16>::of(layer.vc.buf(), plane_bytes, plane_len)?;
            match &row0 {
                None => {
                    let args = GqaArgs {
                        q: &s.qr,
                        kc: &kc,
                        vc: &vc,
                        n_keys: &n_keys,
                        scale,
                        n_kv,
                        ctx: planes,
                        m,
                        part_v: &mut s.part_v,
                        part_ms: &mut s.part_ms,
                        fault,
                        y: &mut s.attn,
                    };
                    if group == GROUP {
                        gqa.enqueue_pass_256(stream, args, MMA)?;
                    } else {
                        gqa.enqueue_pass_256_p4(stream, args, n_q, MMA)?;
                    }
                }
                Some(row0) => {
                    let args = GqaRingArgs {
                        q: &s.qr,
                        kc: &kc,
                        vc: &vc,
                        n_keys: &n_keys,
                        row0,
                        scale,
                        n_kv,
                        ring,
                        stride: planes,
                        m,
                        part_v: &mut s.part_v,
                        part_ms: &mut s.part_ms,
                        fault,
                        y: &mut s.attn,
                    };
                    gqa.enqueue_pass_p4_ring(stream, args, n_q)?;
                }
            }
            gemv(gpu, &layer.o, &s.attn, m, &mut s.o)?;
            let attn_out = match conv_of(conv, &layer.conv)? {
                Some((c, w)) => {
                    c.output(gpu, &w.attn_base, &s.o, m)?;
                    &c.cv
                }
                None => &s.o,
            };
            elem.enqueue_add(stream, x, attn_out, m * h, &mut s.x1)?;
            elem.enqueue_rms_norm(stream, &s.x1, &layer.ln_post, eps, h, m, &mut s.xn2)?;
            let input = match conv_of(conv, &layer.conv)? {
                Some((c, w)) => {
                    c.input(gpu, &w.ffn_proj, &w.ffn_base, &s.xn2, m)?;
                    &c.xc
                }
                None => &s.xn2,
            };
            gemv(gpu, &layer.gate, input, m, &mut s.g)?;
            gemv(gpu, &layer.up, input, m, &mut s.u)?;
            elem.enqueue_swiglu(stream, &s.g, &s.u, m * inter, &mut s.act)?;
            gemv(gpu, &layer.down, &s.act, m, &mut s.dn)?;
            let mlp_out = match conv_of(conv, &layer.conv)? {
                Some((c, w)) => {
                    c.output(gpu, &w.ffn_base, &s.dn, m)?;
                    &c.cv
                }
                None => &s.dn,
            };
            elem.enqueue_add(stream, &s.x1, mlp_out, m * h, x)?;
        }
        Ok(())
    }

    fn reset(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        self.base = None;
        self.count = 0;
        self.staged_block = None;
        self.staged_rows = None;
        Ok(())
    }

    fn card_bytes(&self, ctx: usize) -> u64 {
        let spec = DenseSpec {
            ctx,
            ..self.spec.clone()
        };
        self.dims.bytes(&spec, self.features)
    }
}

/// `y = w · x` over `m` columns, token-major.
fn gemv(
    gpu: &Gpu,
    w: &Q8Rows,
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gpu.q8f32().enqueue_q8_0_gemv_mcol(
        gpu.stream(),
        Q8_0GemvMcolArgs {
            qs: &w.qs,
            d: &w.d,
            x,
            m,
            out: GemvOut::TokenMajor,
            y,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_permutation_makes_neox_pair_i_adjacent_pair_i() {
        let hd = 8;
        let src: Vec<usize> = (0..hd).map(|j| adjacent_pairs(hd, j)).collect();
        // New rows (0, 1) are the file's (0, 4): NEOX pair 0; (2, 3) the file's (1, 5); and so on.
        assert_eq!(src, [0, 4, 1, 5, 2, 6, 3, 7]);
        // It is a permutation: every file row once.
        let mut seen = src;
        seen.sort_unstable();
        assert_eq!(seen, (0..hd).collect::<Vec<_>>());
        // At 256 as well.
        let mut all: Vec<usize> = (0..256).map(|j| adjacent_pairs(256, j)).collect();
        all.sort_unstable();
        assert_eq!(all, (0..256).collect::<Vec<_>>());
        assert_eq!(adjacent_pairs(256, 255), 255);
        assert_eq!(adjacent_pairs(256, 254), 127);
    }

    #[test]
    fn a_ring_is_the_window_or_the_context() {
        let spec = |window: Option<usize>, ctx: usize| DenseSpec {
            layers: 5,
            eps: 1e-6,
            rope_theta: 1e7,
            window,
            block_causal: true,
            ctx,
        };
        assert_eq!(DenseDims::ring(&spec(Some(2048), 8192)), 2048);
        assert_eq!(DenseDims::ring(&spec(Some(2048), 1024)), 1024);
        assert_eq!(DenseDims::ring(&spec(None, 4096)), 4096);
    }

    /// The per-row window's image against llama.cpp's standard mask itself (a query at `q` reads a key
    /// at `p` when `q − p < window`): for every context count up to five rings, every block width and
    /// both masks, each row's run of the ring is its visible context rows' slots in position order, then
    /// the block's; no run is longer than the ring; no slot a row reads is one the block writes; and the
    /// block append's two launches write every block slot and, inside a ring, only block slots of
    /// their own key head.
    #[test]
    fn per_row_image_is_the_reference_mask() {
        let window = 12;
        let spec = DenseSpec {
            layers: 1,
            eps: 1e-6,
            rope_theta: 1e4,
            window: Some(window),
            block_causal: false,
            ctx: 4096,
        };
        let f = Features {
            per_row: Some(window),
            conv: None,
            target_embed: false,
        };
        let cap = ring_cap(&spec, f);
        assert_eq!(cap, window - 1 + BLOCK_MAX);
        let (n_kv, planes) = (2usize, ring_planes(cap));
        let row = |h: usize, slot: usize| u32::try_from(GUARD + h * planes + slot).expect("a row");
        for causal in [false, true] {
            for count in 0..5 * cap {
                for m in 1..=BLOCK_MAX {
                    let p = PerRow {
                        window,
                        cap,
                        count,
                        m,
                        causal,
                    };
                    let img = per_row_image(p, n_kv, row);
                    assert_eq!(img.len(), block_words(n_kv, f));
                    let at = |w: usize| img[w] as usize;
                    let what = format!("causal {causal}, count {count}, m {m}");
                    let mut read = Vec::new();
                    for t in 0..m {
                        let seen: Vec<usize> =
                            (0..count).filter(|&p| count + t - p < window).collect();
                        let own = if causal { t + 1 } else { m };
                        let (keys, first) = (at(n_kv + t), at(2 * n_kv + BLOCK_MAX + t));
                        let run: Vec<usize> = (0..keys).map(|j| (first + j) % cap).collect();
                        let want: Vec<usize> = seen
                            .iter()
                            .copied()
                            .chain(count..count + own)
                            .map(|p| p % cap)
                            .collect();
                        assert_eq!(run, want, "{what}, row {t}");
                        assert!(keys <= cap, "{what}, row {t}: {keys} keys");
                        read.extend(seen.iter().map(|p| p % cap));
                    }
                    let block: Vec<usize> = (count..count + m).map(|p| p % cap).collect();
                    assert!(
                        block.iter().all(|s| !read.contains(s)),
                        "{what}: the block writes a slot a row reads"
                    );
                    for h in 0..n_kv {
                        let mut written = Vec::new();
                        for r in [at(h), at(n_kv + BLOCK_MAX + h)]
                            .into_iter()
                            .flat_map(|r0| r0..r0 + m)
                        {
                            assert!(r < GUARD + n_kv * planes, "{what}: row {r} past the cache");
                            if r < GUARD {
                                continue;
                            }
                            let (owner, slot) = ((r - GUARD) / planes, (r - GUARD) % planes);
                            if slot < cap {
                                assert!(
                                    owner == h && block.contains(&slot),
                                    "{what}: key head {h} writes head {owner}'s slot {slot}"
                                );
                                written.push(slot);
                            }
                        }
                        assert!(
                            block.iter().all(|s| written.contains(s)),
                            "{what}: key head {h} leaves a block slot unwritten"
                        );
                    }
                }
            }
        }
    }

    // ------------------------------------------------- a DFlash2 draft at test size, on the card

    const H: usize = 256;
    const N_Q: usize = 4;
    const N_KV: usize = 1;
    const INTER: usize = 512;
    const LAYERS: usize = 2;
    const WINDOW: usize = 24;
    const VOCAB: usize = 16;
    const MASK: u32 = 15;
    const CONV: Conv = Conv {
        kernel: 2,
        group: 16,
    };

    fn hparams() -> Hparams {
        Hparams {
            n_layer: LAYERS,
            n_embd: H,
            n_ff: INTER,
            n_head: N_Q,
            n_head_kv: N_KV,
            head_dim: HEAD,
            rope_base: 1e4,
            rms_eps: 1e-6,
            window: WINDOW,
            n_ctx_train: 4096,
            block_size: BLOCK_MAX,
            conv: CONV,
            selector: dflash2::Selector { rank: 32, top_k: 4 },
            target_layers: vec![1],
            target_hidden: H,
            mask_token: MASK,
            n_vocab: VOCAB,
        }
    }

    /// Seeded values in [−1, 1).
    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// One layer's values as the file holds them, in f64; matrices row-major `[rows][k]`.
    struct HostLayer {
        attn_norm: Vec<f64>,
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        q_norm: Vec<f64>,
        k_norm: Vec<f64>,
        o: Vec<f64>,
        ffn_norm: Vec<f64>,
        gate: Vec<f64>,
        up: Vec<f64>,
        down: Vec<f64>,
        attn_proj: Vec<f64>,
        attn_base: Vec<f64>,
        ffn_proj: Vec<f64>,
        ffn_base: Vec<f64>,
    }

    /// The test file's tensors as they are declared and written.
    struct Draft {
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
        seed: u32,
    }

    impl Draft {
        /// A Q8_0 matrix of `rows` rows of `k` seeded values in `[−a, a)`, and its values.
        fn matrix(&mut self, name: String, rows: usize, k: usize, a: f32) -> Vec<f64> {
            self.seed += 1;
            let vals: Vec<f32> = seeded(rows * k, self.seed).iter().map(|v| v * a).collect();
            let (mut bytes, mut host) = (Vec::new(), Vec::new());
            for blk in vals.as_chunks::<32>().0 {
                let b = quantize_q8_0(blk);
                bytes.extend_from_slice(&b.to_bytes());
                let d = f64::from(gguf::quant::half_to_f32(b.d));
                host.extend(b.q.iter().map(|&q| d * f64::from(q)));
            }
            self.tensors
                .push((name, vec![k as u64, rows as u64], 8, bytes));
            host
        }

        /// An f32 tensor of shape `dims` of seeded values in `center ± a`, and its values.
        fn values(&mut self, name: String, dims: &[usize], center: f32, a: f32) -> Vec<f64> {
            self.seed += 1;
            let n = dims.iter().product();
            let vals: Vec<f32> = seeded(n, self.seed)
                .iter()
                .map(|v| center + v * a)
                .collect();
            let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dims = dims.iter().map(|&d| d as u64).collect();
            self.tensors.push((name, dims, 0, bytes));
            vals.iter().map(|&v| f64::from(v)).collect()
        }
    }

    /// The test draft at `path`, the target's token table in the same file: every matrix Q8_0, every
    /// gain and base f32. Returns the layers' values and the table's rows as the file holds them.
    fn write_draft(path: &std::path::Path) -> (Vec<HostLayer>, Vec<f64>) {
        use gguf::write::{Layout, TensorDecl, Writer};
        let mut d = Draft {
            tensors: Vec::new(),
            seed: 1,
        };
        let coeffs = CONV.coeffs(H);
        let unit = |k: usize| 1.0 / (k as f32).sqrt();
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let b = |what: &str| dflash2::names::blk(l, what);
            let base_dims = [H, CONV.kernel, 2];
            layers.push(HostLayer {
                attn_norm: d.values(b("attn_norm.weight"), &[H], 1.0, 0.2),
                attn_proj: d.matrix(b("attn_conv_proj.weight"), coeffs, H, 0.35 * unit(H)),
                attn_base: d.values(b("attn_conv_base"), &base_dims, 0.5, 0.3),
                q: d.matrix(b("attn_q.weight"), N_Q * HEAD, H, unit(H)),
                k: d.matrix(b("attn_k.weight"), N_KV * HEAD, H, unit(H)),
                v: d.matrix(b("attn_v.weight"), N_KV * HEAD, H, unit(H)),
                q_norm: d.values(b("attn_q_norm.weight"), &[HEAD], 1.0, 0.2),
                k_norm: d.values(b("attn_k_norm.weight"), &[HEAD], 1.0, 0.2),
                o: d.matrix(b("attn_output.weight"), H, N_Q * HEAD, unit(N_Q * HEAD)),
                ffn_norm: d.values(b("ffn_norm.weight"), &[H], 1.0, 0.2),
                ffn_proj: d.matrix(b("ffn_conv_proj.weight"), coeffs, H, 0.35 * unit(H)),
                ffn_base: d.values(b("ffn_conv_base"), &base_dims, 0.5, 0.3),
                gate: d.matrix(b("ffn_gate.weight"), INTER, H, unit(H)),
                up: d.matrix(b("ffn_up.weight"), INTER, H, unit(H)),
                down: d.matrix(b("ffn_down.weight"), H, INTER, unit(INTER)),
            });
        }
        let table = d.matrix("token_embd.weight".to_string(), VOCAB, H, 1.0);
        let decls = d
            .tensors
            .iter()
            .map(|(name, dims, type_id, bytes)| TensorDecl {
                name: name.clone(),
                dims: dims.clone(),
                type_id: *type_id,
                nbytes: bytes.len() as u64,
            })
            .collect();
        let layout = Layout::new(&[], decls).unwrap_or_else(|e| panic!("{e}"));
        let file =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| panic!("{e}")));
        let mut w = Writer::new(file, layout).unwrap_or_else(|e| panic!("{e}"));
        for (name, _, _, bytes) in &d.tensors {
            w.tensor(name, bytes).unwrap_or_else(|e| panic!("{e}"));
        }
        w.finish().unwrap_or_else(|e| panic!("{e}"));
        (layers, table)
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// `w · x`, `w` of `rows` rows of `x.len()`.
    fn matvec(w: &[f64], rows: usize, x: &[f64]) -> Vec<f64> {
        let k = x.len();
        (0..rows).map(|r| dot(&w[r * k..][..k], x)).collect()
    }

    /// `x / sqrt(mean(x²) + eps) · gain` over each `gain.len()` run of `x`.
    fn rms(x: &[f64], gain: &[f64]) -> Vec<f64> {
        let n = gain.len();
        x.chunks(n)
            .flat_map(|v| {
                let scale = 1.0 / (dot(v, v) / n as f64 + 1e-6).sqrt();
                v.iter().zip(gain).map(move |(a, g)| a * scale * g)
            })
            .collect()
    }

    /// NEOX rotation of each head of `x` by the card's table row `cs` (pair `i` is `(x_i, x_{i+64})`).
    fn neox(x: &mut [f64], cs: &[f32]) {
        for head in x.chunks_mut(HEAD) {
            for i in 0..HEAD / 2 {
                let (c, s) = (f64::from(cs[2 * i]), f64::from(cs[2 * i + 1]));
                let (a, b) = (head[i], head[i + HEAD / 2]);
                head[i] = a * c - b * s;
                head[i + HEAD / 2] = a * s + b * c;
            }
        }
    }

    /// `conv.rs`'s rule in f64 over `m` rows of `H`: `coeff` [`Conv::coeffs`] a row.
    fn conv(h: &[f64], coeff: &[f64], base: &[f64], m: usize, side: usize) -> Vec<f64> {
        let (k, g) = (CONV.kernel, CONV.group);
        let gs = H / g;
        let mut y = vec![0.0f64; m * H];
        for t in 0..m {
            for c in 0..H {
                for j in 0..k.min(t + 1) {
                    let w = coeff[t * CONV.coeffs(H) + (side * k + j) * gs + c / g]
                        + base[(side * k + j) * H + c];
                    y[t * H + c] += w * h[(t - j) * H + c];
                }
            }
        }
        y
    }

    /// Each card row within one f16 step of the host's: the cache holds `round(host)` or, for a value
    /// at a rounding boundary the card's f32 sits on the other side of, its neighbour.
    fn close16(card: &[f64], host: &[f64]) -> bool {
        let top = host.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        card.iter()
            .zip(host)
            .all(|(c, h)| (c - h).abs() <= h.abs() / 1024.0 + 1e-5 * top)
    }

    /// `max |a − b| / max |b|`.
    fn rel_max(a: &[f64], b: &[f64]) -> f64 {
        let top = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
            / top
    }

    /// One layer's caches as the card holds them, K then V.
    type Cache = (Vec<u16>, Vec<u16>);

    /// One layer's block keys and values in f64: K rows then V rows.
    type BlockKv = (Vec<f64>, Vec<f64>);

    /// The block pass on the host in f64 over the card's own cached K and V: a block of `m` rows after
    /// `count` context rows from position `base`, every row reading the positions the reference's
    /// mask lets it.
    struct HostPass<'a> {
        w: &'a [HostLayer],
        caches: &'a [Cache],
        cap: usize,
        planes: usize,
        count: usize,
        base: u32,
        rope: &'a RopeTable,
    }

    impl HostPass<'_> {
        /// The table row of the row `i` rows from `base`.
        fn cs(&self, i: usize) -> Vec<f32> {
            let mut out = Vec::new();
            let pos = self.base + u32::try_from(i).expect("a position");
            self.rope.push(pos, Direction::Forward, &mut out);
            out
        }

        /// Key head `kh`'s cached row for the row `i` rows from `base`, in the file's order (a K row
        /// stands permuted, the module doc).
        fn cached(&self, c: &[u16], kh: usize, i: usize, key: bool) -> Vec<f64> {
            let row = &c[(GUARD + kh * self.planes + i % self.cap) * HEAD..][..HEAD];
            let vals: Vec<f64> = row
                .iter()
                .map(|&b| f64::from(gguf::quant::half_to_f32(b)))
                .collect();
            if !key {
                return vals;
            }
            let mut file = vec![0.0f64; HEAD];
            for (j, v) in vals.into_iter().enumerate() {
                file[adjacent_pairs(HEAD, j)] = v;
            }
            file
        }

        /// The block's output rows from its embedded rows `x0`, and each layer's block K and V.
        fn run(&self, x0: &[f64], m: usize) -> (Vec<f64>, Vec<BlockKv>) {
            let (group, coeffs, count) = (N_Q / N_KV, CONV.coeffs(H), self.count);
            let scale = 1.0 / (HEAD as f64).sqrt();
            let mut x = x0.to_vec();
            let mut block_kv = Vec::new();
            for (w, (kc, vc)) in self.w.iter().zip(self.caches) {
                let rows = |x: &[f64], gain: &[f64], proj: &[f64]| -> (Vec<f64>, Vec<f64>) {
                    let xn = rms(x, gain);
                    let coeff = xn.chunks(H).flat_map(|r| matvec(proj, coeffs, r)).collect();
                    (xn, coeff)
                };
                let (xn, coeff) = rows(&x, &w.attn_norm, &w.attn_proj);
                let xc = conv(&xn, &coeff, &w.attn_base, m, 0);
                let (mut q, mut kb, mut vb) = (Vec::new(), Vec::new(), Vec::new());
                for t in 0..m {
                    let row = &xc[t * H..][..H];
                    let cs = self.cs(count + t);
                    let mut qt = rms(&matvec(&w.q, N_Q * HEAD, row), &w.q_norm);
                    neox(&mut qt, &cs);
                    let mut kt = rms(&matvec(&w.k, N_KV * HEAD, row), &w.k_norm);
                    neox(&mut kt, &cs);
                    q.push(qt);
                    kb.extend(kt);
                    vb.extend(matvec(&w.v, N_KV * HEAD, row));
                }
                let mut attn = vec![0.0f64; m * N_Q * HEAD];
                for t in 0..m {
                    let seen: Vec<usize> = (0..count)
                        .filter(|&p| count + t - p < WINDOW)
                        .chain(count..count + m)
                        .collect();
                    for hq in 0..N_Q {
                        let kh = hq / group;
                        let qv = &q[t][hq * HEAD..][..HEAD];
                        let keys: Vec<(Vec<f64>, Vec<f64>)> = seen
                            .iter()
                            .map(|&i| (self.cached(kc, kh, i, true), self.cached(vc, kh, i, false)))
                            .collect();
                        let s: Vec<f64> = keys.iter().map(|(k, _)| dot(qv, k) * scale).collect();
                        let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = s.iter().map(|v| (v - top).exp()).collect();
                        let z: f64 = e.iter().sum();
                        let out = &mut attn[(t * N_Q + hq) * HEAD..][..HEAD];
                        for ((_, v), p) in keys.iter().zip(&e) {
                            for (o, v) in out.iter_mut().zip(v) {
                                *o += p / z * v;
                            }
                        }
                    }
                }
                let o: Vec<f64> = attn
                    .chunks(N_Q * HEAD)
                    .flat_map(|a| matvec(&w.o, H, a))
                    .collect();
                let co = conv(&o, &coeff, &w.attn_base, m, 1);
                let x1: Vec<f64> = x.iter().zip(&co).map(|(a, b)| a + b).collect();
                let (xn2, coeff2) = rows(&x1, &w.ffn_norm, &w.ffn_proj);
                let xc2 = conv(&xn2, &coeff2, &w.ffn_base, m, 0);
                let dn: Vec<f64> = xc2
                    .chunks(H)
                    .flat_map(|r| {
                        let g = matvec(&w.gate, INTER, r);
                        let u = matvec(&w.up, INTER, r);
                        let act: Vec<f64> = g
                            .iter()
                            .zip(&u)
                            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                            .collect();
                        matvec(&w.down, H, &act)
                    })
                    .collect();
                let cd = conv(&dn, &coeff2, &w.ffn_base, m, 1);
                x = x1.iter().zip(&cd).map(|(a, b)| a + b).collect();
                block_kv.push((kb, vb));
            }
            (x, block_kv)
        }
    }

    /// The DFlash2 layers on the card are the host network: a two-layer draft at test size (heads of
    /// 128 in a group of 4, the convolution of kernel 2 in groups of 16, a window of 24, so a ring of
    /// 31) written as a GGUF, its context appended in uneven chunks from position 100 (an append that
    /// wraps), and blocks of 8, 8, 3 and 8 rows after 5, 26, 51 and 62 context rows (no window cut; a
    /// cut with the block's append wrapping; runs that wrap; the ring's start). At every block, every
    /// cached K and V row a block row reads and the block's own are within one f16 step of the host's
    /// (the rotation's positions, the slots, the convolution's side 0), and the block's output is
    /// within `8 · u · sqrt(terms · layers)` of the f64 network over the card's caches, `terms` the
    /// sums a layer adds (seven projections over 256 inputs, the output projection over 512, the
    /// down projection over 512, a score over 128 and the softmax over up to 31 keys). The input
    /// buffer starts as 77s, so the output is the embedded rows'. A context of 12 holds a ring of 20:
    /// its block over 12 rows reads all of it, and a 13th row is refused by name. The buffers counted
    /// are the card bytes.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_dflash2_layers_are_the_host_network() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let s = gpu.stream();
        let dir = std::env::temp_dir().join(format!("bloomery-dflash2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        let path = dir.join("draft.gguf");
        let (w, table) = write_draft(&path);
        let split = std::sync::Arc::new(Split::open(&path).unwrap_or_else(|e| panic!("{e}")));
        let hp = hparams();
        let embed = || {
            EmbedTable::new(std::sync::Arc::clone(&split), "token_embd.weight", H)
                .unwrap_or_else(|e| panic!("{e}"))
        };
        let ctx = 4096;
        let mut layers =
            DenseLayers::load_dflash2(&gpu, &split, &hp, ctx, embed()).expect("the layers");
        let held = layers.held(&OwnHeld {
            weights: Vec::new(),
            working: 0,
        });
        let total =
            held.weights.iter().sum::<u64>() + held.rings.iter().sum::<u64>() + held.working;
        assert_eq!(
            total,
            layers.card_bytes(ctx),
            "the buffers are the card bytes"
        );
        assert_eq!(
            total,
            DenseLayers::dflash2_card_bytes(&hp, ctx).expect("the sizing")
        );
        assert_eq!(layers.ring_rows(), WINDOW - 1 + BLOCK_MAX);

        let base = 100u32;
        let ctx_rows = seeded(70 * H, 977);
        let ctx_f64 = |i: usize| -> Vec<f64> {
            ctx_rows[i * H..][..H]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let terms = (7 * H + N_Q * HEAD + INTER + HEAD + WINDOW + BLOCK_MAX) as f64;
        let band = 8.0 / 16_777_216.0 * (terms * LAYERS as f64).sqrt();
        // One block of `m` rows after `count` context rows, its anchor `last`: the cache check, then
        // the output's distance from the host's.
        let block = |layers: &mut DenseLayers, count: usize, m: usize, last: u32| -> f64 {
            let pos = base + u32::try_from(count).expect("a position");
            layers
                .stage_block(&gpu, last, pos, m, count.min(WINDOW))
                .expect("stage the block");
            let mut x = DeviceBuffer::from_host(s, &vec![77.0f32; BLOCK_MAX * H]).expect("x");
            layers.enqueue_block(&gpu, &mut x, m).expect("the block");
            s.synchronize().expect("sync");
            let got: Vec<f64> = x.to_host_vec(s).expect("x back")[..m * H]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let caches: Vec<Cache> = layers
                .layers
                .iter()
                .map(|l| {
                    (
                        l.kc.buf().to_host_vec(s).expect("kc"),
                        l.vc.buf().to_host_vec(s).expect("vc"),
                    )
                })
                .collect();
            let mut x0 = table[last as usize * H..][..H].to_vec();
            for _ in 1..m {
                x0.extend_from_slice(&table[MASK as usize * H..][..H]);
            }
            let pass = HostPass {
                w: &w,
                caches: &caches,
                cap: layers.cap,
                planes: layers.planes,
                count,
                base,
                rope: &layers.rope,
            };
            let (want, block_kv) = pass.run(&x0, m);
            for (l, (wl, (kc, vc))) in w.iter().zip(&caches).enumerate() {
                for i in count - count.min(WINDOW - 1)..count {
                    let row = ctx_f64(i);
                    let mut k = rms(&matvec(&wl.k, N_KV * HEAD, &row), &wl.k_norm);
                    neox(&mut k, &pass.cs(i));
                    let v = matvec(&wl.v, N_KV * HEAD, &row);
                    for kh in 0..N_KV {
                        let at = kh * HEAD..(kh + 1) * HEAD;
                        assert!(
                            close16(&pass.cached(kc, kh, i, true), &k[at.clone()])
                                && close16(&pass.cached(vc, kh, i, false), &v[at]),
                            "layer {l}: the cached context row {i} after {count} rows"
                        );
                    }
                }
                let (kb, vb) = &block_kv[l];
                for t in 0..m {
                    for kh in 0..N_KV {
                        let at = (t * N_KV + kh) * HEAD..(t * N_KV + kh + 1) * HEAD;
                        assert!(
                            close16(&pass.cached(kc, kh, count + t, true), &kb[at.clone()])
                                && close16(&pass.cached(vc, kh, count + t, false), &vb[at]),
                            "layer {l}: the block's row {t} after {count} rows"
                        );
                    }
                }
            }
            let err = rel_max(&got, &want);
            println!("dflash2 block of {m} after {count} rows: rel_max {err:.3e}, band {band:.3e}");
            err
        };
        let append = |layers: &mut DenseLayers, count: usize, n: usize| {
            let first = base + u32::try_from(count).expect("a position");
            layers.stage_kv(&gpu, first, n).expect("stage the rows");
            let x =
                DeviceBuffer::from_host(s, &ctx_rows[count * H..(count + n) * H]).expect("rows");
            layers.append_kv(&gpu, &x, n).expect("append the rows");
        };

        let blocks = [(5usize, 8usize, 3u32), (26, 8, 7), (51, 3, 1), (62, 8, 9)];
        let mut worst = 0.0f64;
        let mut count = 0;
        for n in [5usize, 8, 8, 5, 8, 7, 8, 2, 8, 3, 8] {
            if let Some(&(_, m, last)) = blocks.iter().find(|b| b.0 == count) {
                worst = worst.max(block(&mut layers, count, m, last));
            }
            append(&mut layers, count, n);
            count += n;
        }
        assert_eq!(layers.stats().blocks, blocks.len(), "every block ran");
        assert!(
            layers.stats().wrapped > 0,
            "an append ran past the ring's end"
        );
        assert!(
            layers
                .stage_block(&gpu, 2, base + 70, 8, WINDOW - 1)
                .is_err(),
            "a block over another count than the window's fill is refused"
        );

        let mut short = DenseLayers::load_dflash2(&gpu, &split, &hp, 12, embed()).expect("short");
        assert_eq!(short.ring_rows(), 12 + BLOCK_MAX);
        append(&mut short, 0, 8);
        append(&mut short, 8, 4);
        worst = worst.max(block(&mut short, 12, 8, 5));
        let e = short.stage_kv(&gpu, base + 12, 1).expect_err("a 13th row");
        assert!(e.to_string().contains("capacity of 12"), "{e}");

        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            worst <= band,
            "the card's block is the host network: {worst:e} past {band:e}"
        );
    }
}
