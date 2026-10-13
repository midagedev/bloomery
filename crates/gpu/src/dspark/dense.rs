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
//! pack-of-four entry). Qwen3.6's own rotation kernel (`rope_neox::head_norm_neox_append_256`) turns 64
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

use std::sync::{Mutex, PoisonError};

use cuda_core::DeviceBuffer;
use gguf::quant::Q8Block;
use model::quant::quantize_q8_0;

use super::{DraftLayers, Q8Rows};
use crate::flash_gqa::{
    FlashGqaKernels, GROUP, GqaArgs, HEAD_256, PACK_4, partials_ms_len, partials_v_len_256,
};
use crate::graph::capturing;
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::rope_table::{Direction, RopeSpec, RopeTable};
use crate::weights::q8_0_planes;
use crate::{DeviceTensor, Gpu, GpuError, Window};

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
        let DenseDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = *self;
        let q8 = |rows: usize, k: usize| (rows * (k + k / 16)) as u64;
        let weights = q8(n_q * hd, h)
            + 2 * q8(n_kv * hd, h)
            + q8(h, n_q * hd)
            + 2 * q8(inter, h)
            + q8(h, inter)
            + 4 * (2 * h + 2 * hd) as u64;
        let planes = ring_planes(Self::ring(spec));
        let kv = 2 * 2 * ((GUARD + n_kv * planes) * hd) as u64;
        let m = BLOCK_MAX;
        let scratch = 4 * (m * (5 * h + 3 * inter + 4 * n_q * hd + 4 * n_kv * hd)) as u64
            + 4 * (partials_v_len_256(m, n_q) + partials_ms_len(m, n_q)) as u64
            + 4 * (2 * m * hd + n_kv + 8 + 2 * n_kv) as u64;
        spec.layers as u64 * (weights + kv) + scratch
    }
}

fn ring_planes(cap: usize) -> usize {
    cap + BLOCK_MAX
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
        let (qs, d) = q8_0_planes(&blocks);
        let stream = gpu.stream();
        Ok(Q8Rows {
            qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
            d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
        })
    }
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
    /// [`BLOCK_MAX`] key counts.
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

impl DenseLayers {
    /// Load `spec.layers` layers from `src`: every matrix quantized to Q8_0 on the host (the bf16 is kept
    /// nowhere), the query and key rows permuted as the module doc says, the caches zeroed.
    pub fn load(
        gpu: &Gpu,
        spec: &DenseSpec,
        src: &dyn Bf16Source,
    ) -> Result<DenseLayers, GpuError> {
        let dims = DenseDims::read(src)?;
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
        let cap = DenseDims::ring(spec);
        let planes = ring_planes(cap);
        let rope = RopeTable::new(&RopeSpec::window(spec.rope_theta, hd))?;
        // The file rotates pair (i, i + hd/2); the card's kernel rotates adjacent pairs (2i, 2i + 1). Rows of
        // q and k are reordered inside each head the same way, so every q.k dot product sums the same
        // products in another order, and v, o_proj and the cache keep the file's order.
        let perm_head = move |r: usize| (r / hd) * hd + adjacent_pairs(hd, r % hd);
        let ident = |r: usize| r;
        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let permuted_gain = |tail: &str| -> Result<DeviceBuffer<f32>, GpuError> {
                let g = widen(vector(src, l, tail, hd)?);
                let p: Vec<f32> = (0..hd).map(|j| g[adjacent_pairs(hd, j)]).collect();
                up_f32(gpu, &p)
            };
            let plain_gain = |tail: &str, n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
                up_f32(gpu, &widen(vector(src, l, tail, n)?))
            };
            let q = Q8Rows::from_bf16(
                gpu,
                matrix(src, l, "self_attn.q_proj.weight", n_q * hd, h)?,
                n_q * hd,
                h,
                &perm_head,
            )?;
            let k_all = matrix(src, l, "self_attn.k_proj.weight", n_kv * hd, h)?;
            let v_all = matrix(src, l, "self_attn.v_proj.weight", n_kv * hd, h)?;
            let mut k = Vec::with_capacity(n_kv);
            let mut v = Vec::with_capacity(n_kv);
            for head in 0..n_kv {
                let rows = &k_all[head * hd * h..(head + 1) * hd * h];
                k.push(Q8Rows::from_bf16(gpu, rows, hd, h, &move |r| {
                    adjacent_pairs(hd, r)
                })?);
                let rows = &v_all[head * hd * h..(head + 1) * hd * h];
                v.push(Q8Rows::from_bf16(gpu, rows, hd, h, &ident)?);
            }
            let o = Q8Rows::from_bf16(
                gpu,
                matrix(src, l, "self_attn.o_proj.weight", h, n_q * hd)?,
                h,
                n_q * hd,
                &ident,
            )?;
            let mlp = |tail: &str, rows: usize, k: usize| -> Result<Q8Rows, GpuError> {
                Q8Rows::from_bf16(gpu, matrix(src, l, tail, rows, k)?, rows, k, &ident)
            };
            let gate = mlp("mlp.gate_proj.weight", inter, h)?;
            let up = mlp("mlp.up_proj.weight", inter, h)?;
            let down = mlp("mlp.down_proj.weight", h, inter)?;
            layers.push(Layer {
                ln_in: plain_gain("input_layernorm.weight", h)?,
                ln_post: plain_gain("post_attention_layernorm.weight", h)?,
                q,
                k,
                v,
                q_norm: permuted_gain("self_attn.q_norm.weight")?,
                k_norm: permuted_gain("self_attn.k_norm.weight")?,
                o,
                gate,
                up,
                down,
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
            part_v: zeros_f32(gpu, partials_v_len_256(m, n_q))?,
            part_ms: zeros_f32(gpu, partials_ms_len(m, n_q))?,
            blk_words: DeviceBuffer::from_host(gpu.stream(), &vec![1u32; n_kv + BLOCK_MAX])?,
            blk_cs: zeros_f32(gpu, m * hd)?,
            ctx_words: DeviceBuffer::from_host(gpu.stream(), &vec![0u32; 2 * n_kv])?,
            ctx_cs: zeros_f32(gpu, m * hd)?,
        };
        Ok(DenseLayers {
            dims,
            spec: spec.clone(),
            layers,
            flash: FlashGqaKernels::load(gpu.context())?,
            rope,
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

    /// Context rows a key head's plane holds before the ring wraps.
    #[must_use]
    pub fn ring_rows(&self) -> usize {
        self.cap
    }

    /// Context rows the next block pass sees: the ring's fill.
    #[must_use]
    pub fn visible(&self) -> usize {
        self.count.min(self.cap)
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
}

impl DraftLayers for DenseLayers {
    const MAX_ROWS: usize = BLOCK_MAX;
    const APPEND_GROUP: usize = APPEND_MAX;

    fn hidden(&self) -> usize {
        self.dims.hidden
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
        if self.spec.window.is_none() && past + n > self.cap {
            return Err(shape_err(format!(
                "{} context rows past a capacity of {}",
                past + n,
                self.cap
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
                for second in 0..2 {
                    let word = Window::<u32>::of(&s.ctx_words, 4 * (second * n_kv + head), 1)?;
                    flash.enqueue_kv_append_pos_buf(
                        stream,
                        &s.kr[head],
                        &word,
                        &mut layer.kc,
                        n,
                        fault,
                    )?;
                    flash.enqueue_kv_append_pos_buf(
                        stream,
                        &s.vb[head],
                        &word,
                        &mut layer.vc,
                        n,
                        fault,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn stage_block(
        &mut self,
        gpu: &Gpu,
        _last: u32,
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
        let mut words: Vec<u32> = (0..n_kv).map(|h| self.row(h, valid)).collect();
        for t in 0..BLOCK_MAX {
            let seen = if self.spec.block_causal { t + 1 } else { m };
            words.push(u32::try_from(valid + seen.min(BLOCK_MAX)).unwrap_or(u32::MAX));
        }
        self.s.blk_words.copy_from_host(gpu.stream(), &words)?;
        let cs = self.cos_sin(pos, m);
        self.s.blk_cs.copy_from_host(gpu.stream(), &cs)?;
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
        let planes = self.planes;
        let group = n_q / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let Self {
            layers,
            s,
            flash: gqa,
            ..
        } = self;
        let n_keys = Window::<u32>::of(&s.blk_words, 4 * n_kv, BLOCK_MAX)?;
        for layer in layers.iter_mut() {
            elem.enqueue_rms_norm(stream, x, &layer.ln_in, eps, h, m, &mut s.xn)?;
            gemv(gpu, &layer.q, &s.xn, m, &mut s.q)?;
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
                gemv(gpu, &layer.k[head], &s.xn, m, &mut s.kb[head])?;
                gemv(gpu, &layer.v[head], &s.xn, m, &mut s.vb[head])?;
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
                let word = Window::<u32>::of(&s.blk_words, 4 * head, 1)?;
                flash.enqueue_kv_append_pos_buf(
                    stream,
                    &s.kr[head],
                    &word,
                    &mut layer.kc,
                    m,
                    fault,
                )?;
                flash.enqueue_kv_append_pos_buf(
                    stream,
                    &s.vb[head],
                    &word,
                    &mut layer.vc,
                    m,
                    fault,
                )?;
            }
            let plane_bytes = 2 * GUARD * hd;
            let plane_len = n_kv * planes * hd;
            let kc = Window::<u16>::of(layer.kc.buf(), plane_bytes, plane_len)?;
            let vc = Window::<u16>::of(layer.vc.buf(), plane_bytes, plane_len)?;
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
            gemv(gpu, &layer.o, &s.attn, m, &mut s.o)?;
            elem.enqueue_add(stream, x, &s.o, m * h, &mut s.x1)?;
            elem.enqueue_rms_norm(stream, &s.x1, &layer.ln_post, eps, h, m, &mut s.xn2)?;
            gemv(gpu, &layer.gate, &s.xn2, m, &mut s.g)?;
            gemv(gpu, &layer.up, &s.xn2, m, &mut s.u)?;
            elem.enqueue_swiglu(stream, &s.g, &s.u, m * inter, &mut s.act)?;
            gemv(gpu, &layer.down, &s.act, m, &mut s.dn)?;
            elem.enqueue_add(stream, &s.x1, &s.dn, m * h, x)?;
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
        self.dims.card_bytes(&spec)
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
        let mut seen = src.clone();
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
}
