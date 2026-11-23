//! Inkling 2's draft layer: Qwen3-shaped GQA at a group of three ([`super::block_attn`]) with q and
//! k norms and a NEOX rotation of the whole head at absolute positions, a SiLU-gated MLP, and
//! context rows of its own: each layer projects its own mix of the target's taps (`super::fuse`)
//! to keys and values (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`), where the block's rows take the
//! layer's `k_proj` and `v_proj`. The reference is llama.cpp PR #32218's `graph_inkling2`.
//!
//! # The block
//!
//! The block is always [`BLOCK`] rows, non-causal, whatever the head reads of it: row 0 the anchor
//! (the target's embedding row of the token at the target's position, RMS-normed by the target's
//! embedding norm, [`AnchorEmbed`], staged by [`DraftLayers::stage_block`]), rows 1.. the file's
//! `mask_embed`, written at load. Block row `t` stands at position `pos + t`. Every pass runs the
//! whole block and leaves its rows in `x`; a head of `m` rows reads the first `m`. A windowed layer
//! reads the last `window` context rows (the reference's sliding window is `window + 1` positions
//! with every block row reading the anchor's, so the anchor at `pos` reads positions `pos − window
//! ..`), a global layer every row.
//!
//! # The rotation
//!
//! As [`super::dense`]: the q and k rows (and their norm gains) of each head are permuted at load so
//! that NEOX pair `(i, i + head_dim/2)` becomes adjacent pair `i`, which `elem::rope` turns; the dot
//! of a query with a key is the same sum in another order, and v and o are untouched.
//!
//! # The planes
//!
//! Each layer holds `block_attn`'s planes: a ring of `cap` rows (its window, or the context capacity
//! of a global layer) and the block's rows behind it. One image of stage words serves every launch:
//! word `l` the ring rows layer `l` reads (`min(count, cap)`), word `layers + l` the ring row of an
//! append's first row in layer `l` (`count mod cap`), and a last word 0, the block's start.

use crate::DeviceBufferRaw;
use cuda_core::DeviceBuffer;
use model::embed::EmbedTable;

use super::block_attn::{
    AttnArgs, BlockAttn, GROUP, HEAD, PutArgs, ROWS, partials_ms_len, partials_v_len,
};
use super::dense::{Bf16Source, adjacent_pairs, widen};
use super::{DraftLayers, Q8Rows};
use crate::graph::capturing;
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::rope_table::{Direction, RopeSpec, RopeTable};
use crate::{Gpu, GpuError, Window, WindowMut};

const WHAT: &str = "dspark::ctx_dense";

/// Rows of every block pass.
pub const BLOCK: usize = ROWS;
/// Rows one [`DraftLayers::stage_kv`] stages.
pub const APPEND_MAX: usize = BLOCK;
const ROW_WORDS: usize = HEAD / 2;

fn shape_err(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

/// What a draft's config states about these layers; the widths come from the tensors' shapes.
#[derive(Clone, Debug, PartialEq)]
pub struct CtxSpec {
    /// Decoder layers.
    pub layers: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The rotary base.
    pub rope_theta: f32,
    /// Context rows a windowed layer reads back.
    pub window: usize,
    /// Which layers read the window; the others read the whole context.
    pub windowed: Vec<bool>,
    /// The most positions a context holds.
    pub ctx: usize,
}

/// The layers' widths, read from the first layer's tensors and held equal by every other layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxDims {
    pub hidden: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub inter: usize,
}

/// The block's anchor row from the target: its embedding table, its embedding norm's gain and
/// epsilon.
pub struct AnchorEmbed {
    pub table: EmbedTable,
    pub gain: Vec<f32>,
    pub eps: f32,
}

/// One layer on the card.
struct Layer {
    ln_in: DeviceBuffer<f32>,
    ln_post: DeviceBuffer<f32>,
    q: Q8Rows,
    k: Q8Rows,
    v: Q8Rows,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    k_ctx: Q8Rows,
    v_ctx: Q8Rows,
    k_norm_ctx: DeviceBuffer<f32>,
    o: Q8Rows,
    gate: Q8Rows,
    up: Q8Rows,
    down: Q8Rows,
    /// The ring's rows.
    cap: usize,
    /// `n_kv · (cap + BLOCK) · head_dim / 2` words each.
    kc: DeviceBuffer<u32>,
    vc: DeviceBuffer<u32>,
}

/// The pass's activations and the staged images.
struct Scratch {
    xn: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    kr: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    x1: DeviceBuffer<f32>,
    xn2: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    dn: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    ck: DeviceBuffer<f32>,
    ckn: DeviceBuffer<f32>,
    ckr: DeviceBuffer<f32>,
    cv: DeviceBuffer<f32>,
    /// The stage words (module doc).
    words: DeviceBuffer<u32>,
    /// The block rows' cos/sin, `BLOCK` rows of `head_dim`.
    blk_cs: DeviceBuffer<f32>,
    /// The appended rows' cos/sin, `APPEND_MAX` rows of `head_dim`.
    ctx_cs: DeviceBuffer<f32>,
    /// The block's input rows: the staged anchor's, then the mask's.
    rows: DeviceBuffer<f32>,
}

/// Inkling 2's draft layers on the card ([`DraftLayers`]).
pub struct CtxDenseLayers {
    dims: CtxDims,
    spec: CtxSpec,
    layers: Vec<Layer>,
    attn: BlockAttn,
    rope: RopeTable,
    anchor: AnchorEmbed,
    /// The anchor's row on the host.
    row: Vec<f32>,
    s: Scratch,
    /// The position of the first context row since the last reset, and the rows appended since.
    base: Option<u32>,
    count: usize,
    staged_block: Option<usize>,
    staged_rows: Option<usize>,
}

fn zeros_f32(gpu: &Gpu, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::zeroed(gpu.stream(), n.max(1))?)
}

fn up_f32(gpu: &Gpu, v: &[f32]) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::from_host(gpu.stream(), v)?)
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

fn layer_name(l: usize, tail: &str) -> String {
    format!("layers.{l}.{tail}")
}

/// Tensor `name` of `src` as bits of shape `want`, refused by name otherwise.
fn tensor<'a>(src: &'a dyn Bf16Source, name: &str, want: &[usize]) -> Result<&'a [u16], GpuError> {
    let (bits, shape) = src.bf16(name)?;
    if shape != want {
        return Err(shape_err(format!(
            "{name} is {shape:?}; the layers read {want:?}"
        )));
    }
    Ok(bits)
}

impl CtxDenseLayers {
    /// The layers of `src` (the draft's BF16 tensors, each matrix quantized to Q8_0 on the host)
    /// under `spec`, the anchor's rows from `anchor`. A width the attention has no entry for (a head
    /// of other than [`HEAD`] values, a group of other than [`GROUP`]) is refused by name, as a
    /// missing tensor or one of another shape. Load-time only.
    pub fn load(
        gpu: &Gpu,
        src: &dyn Bf16Source,
        spec: CtxSpec,
        anchor: AnchorEmbed,
    ) -> Result<CtxDenseLayers, GpuError> {
        if spec.layers == 0 || spec.windowed.len() != spec.layers || spec.ctx == 0 {
            return Err(shape_err(format!(
                "{} layers with {} window flags over a context of {}",
                spec.layers,
                spec.windowed.len(),
                spec.ctx
            )));
        }
        let (_, q_shape) = src.bf16(&layer_name(0, "self_attn.q_proj.weight"))?;
        let (_, k_shape) = src.bf16(&layer_name(0, "self_attn.k_proj.weight"))?;
        let (_, n_shape) = src.bf16(&layer_name(0, "self_attn.q_norm.weight"))?;
        let (_, g_shape) = src.bf16(&layer_name(0, "mlp.gate_proj.weight"))?;
        let (&[qr, h], &[kr, _], &[hd], &[inter, _]) = (q_shape, k_shape, n_shape, g_shape) else {
            return Err(shape_err(format!(
                "layer 0's q {q_shape:?}, k {k_shape:?}, q_norm {n_shape:?}, gate {g_shape:?}"
            )));
        };
        if hd != HEAD || qr % hd != 0 || kr % hd != 0 || qr / hd != GROUP * (kr / hd) {
            return Err(shape_err(format!(
                "heads of {hd} values, {qr} query rows over {kr} key rows; the block attention \
                 takes heads of {HEAD} at a group of {GROUP}"
            )));
        }
        if h == 0 || !h.is_multiple_of(32) || !inter.is_multiple_of(32) {
            return Err(shape_err(format!(
                "a hidden width of {h} and an MLP of {inter}: multiples of 32"
            )));
        }
        let dims = CtxDims {
            hidden: h,
            n_q: qr / hd,
            n_kv: kr / hd,
            head_dim: hd,
            inter,
        };
        if anchor.gain.len() != h || anchor.table.n_vocab() == 0 {
            return Err(shape_err(format!(
                "an anchor norm of {} gains over rows of {h}",
                anchor.gain.len()
            )));
        }
        let stream = gpu.stream();
        // Row r of a permuted stack of heads: the file's row of the same head at the adjacent
        // pair's NEOX index (module doc).
        let permuted = move |r: usize| (r / hd) * hd + adjacent_pairs(hd, r % hd);
        let q8 = |name: &str, rows: usize, k: usize, perm: bool| -> Result<Q8Rows, GpuError> {
            let bits = tensor(src, name, &[rows, k])?;
            if perm {
                Q8Rows::from_bf16(gpu, bits, rows, k, &permuted)
            } else {
                Q8Rows::from_bf16(gpu, bits, rows, k, &|r| r)
            }
        };
        let gain = |name: &str, n: usize, perm: bool| -> Result<DeviceBuffer<f32>, GpuError> {
            let v = widen(tensor(src, name, &[n])?);
            if perm {
                let p: Vec<f32> = (0..n).map(|j| v[adjacent_pairs(n, j)]).collect();
                up_f32(gpu, &p)
            } else {
                up_f32(gpu, &v)
            }
        };
        let (nq, nkv) = (dims.n_q * hd, dims.n_kv * hd);
        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let n = |tail: &str| layer_name(l, tail);
            let cap = if spec.windowed[l] {
                spec.window
            } else {
                spec.ctx
            };
            let plane = dims.n_kv * (cap + BLOCK) * ROW_WORDS;
            layers.push(Layer {
                ln_in: gain(&n("input_layernorm.weight"), h, false)?,
                ln_post: gain(&n("post_attention_layernorm.weight"), h, false)?,
                q: q8(&n("self_attn.q_proj.weight"), nq, h, true)?,
                k: q8(&n("self_attn.k_proj.weight"), nkv, h, true)?,
                v: q8(&n("self_attn.v_proj.weight"), nkv, h, false)?,
                q_norm: gain(&n("self_attn.q_norm.weight"), hd, true)?,
                k_norm: gain(&n("self_attn.k_norm.weight"), hd, true)?,
                k_ctx: q8(&n("self_attn.k_proj_ctx.weight"), nkv, h, true)?,
                v_ctx: q8(&n("self_attn.v_proj_ctx.weight"), nkv, h, false)?,
                k_norm_ctx: gain(&n("self_attn.k_norm_ctx.weight"), hd, true)?,
                o: q8(&n("self_attn.o_proj.weight"), h, nq, false)?,
                gate: q8(&n("mlp.gate_proj.weight"), inter, h, false)?,
                up: q8(&n("mlp.up_proj.weight"), inter, h, false)?,
                down: q8(&n("mlp.down_proj.weight"), h, inter, false)?,
                cap,
                kc: DeviceBuffer::zeroed(stream, plane)?,
                vc: DeviceBuffer::zeroed(stream, plane)?,
            });
        }
        let mask = widen(tensor(src, "mask_embed", &[h])?);
        let mut rows = vec![0.0f32; BLOCK * h];
        for r in rows.chunks_exact_mut(h).skip(1) {
            r.copy_from_slice(&mask);
        }
        let b = BLOCK;
        let s = Scratch {
            xn: zeros_f32(gpu, b * h)?,
            q: zeros_f32(gpu, b * nq)?,
            qn: zeros_f32(gpu, b * nq)?,
            qr: zeros_f32(gpu, b * nq)?,
            k: zeros_f32(gpu, b * nkv)?,
            kn: zeros_f32(gpu, b * nkv)?,
            kr: zeros_f32(gpu, b * nkv)?,
            v: zeros_f32(gpu, b * nkv)?,
            attn: zeros_f32(gpu, b * nq)?,
            o: zeros_f32(gpu, b * h)?,
            x1: zeros_f32(gpu, b * h)?,
            xn2: zeros_f32(gpu, b * h)?,
            g: zeros_f32(gpu, b * inter)?,
            u: zeros_f32(gpu, b * inter)?,
            act: zeros_f32(gpu, b * inter)?,
            dn: zeros_f32(gpu, b * h)?,
            part_v: zeros_f32(gpu, partials_v_len(b, dims.n_q))?,
            part_ms: zeros_f32(gpu, partials_ms_len(b, dims.n_q))?,
            ck: zeros_f32(gpu, APPEND_MAX * nkv)?,
            ckn: zeros_f32(gpu, APPEND_MAX * nkv)?,
            ckr: zeros_f32(gpu, APPEND_MAX * nkv)?,
            cv: zeros_f32(gpu, APPEND_MAX * nkv)?,
            words: DeviceBuffer::zeroed(stream, 2 * spec.layers + 1)?,
            blk_cs: zeros_f32(gpu, BLOCK * hd)?,
            ctx_cs: zeros_f32(gpu, APPEND_MAX * hd)?,
            rows: up_f32(gpu, &rows)?,
        };
        let rope = RopeTable::new(&RopeSpec::window(spec.rope_theta, hd))?;
        Ok(CtxDenseLayers {
            dims,
            spec,
            layers,
            attn: BlockAttn::load(gpu.context())?,
            rope,
            anchor,
            row: vec![0.0; h],
            s,
            base: None,
            count: 0,
            staged_block: None,
            staged_rows: None,
        })
    }

    /// The layers' widths.
    #[must_use]
    pub fn dims(&self) -> CtxDims {
        self.dims
    }

    /// The cos/sin rows of `n` positions from `first`, `head_dim` f32 each, padded to `rows` rows.
    fn cos_sin(&self, first: u32, n: usize, rows: usize) -> Vec<f32> {
        let hd = self.dims.head_dim;
        let mut out = Vec::with_capacity(rows * hd);
        for t in 0..n {
            self.rope.push(
                first + u32::try_from(t).unwrap_or(0),
                Direction::Forward,
                &mut out,
            );
        }
        out.resize(rows * hd, 0.0);
        out
    }

    /// The stage words (module doc) at `count` rows.
    fn words(&self) -> Vec<u32> {
        let at = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let vis = self.layers.iter().map(|l| at(self.count.min(l.cap)));
        let slot = self.layers.iter().map(|l| at(self.count % l.cap));
        vis.chain(slot).chain([0]).collect()
    }

    /// The anchor's row: the target's embedding row of `id`, RMS-normed by its gain.
    fn anchor_row(&mut self, id: u32) -> Result<(), GpuError> {
        self.anchor
            .table
            .row_into(id, &mut self.row)
            .map_err(|e| shape_err(format!("the anchor row of {id}: {e}")))?;
        let h = self.row.len();
        let ms = self
            .row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / h as f64;
        let scale = (1.0 / (ms + f64::from(self.anchor.eps)).sqrt()) as f32;
        for (v, &g) in self.row.iter_mut().zip(&self.anchor.gain) {
            *v = (*v * scale) * g;
        }
        Ok(())
    }
}

impl DraftLayers for CtxDenseLayers {
    const MAX_ROWS: usize = BLOCK;
    const APPEND_GROUP: usize = APPEND_MAX;

    fn hidden(&self) -> usize {
        self.dims.hidden
    }

    fn replayable(&self) -> Result<(), GpuError> {
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
                    return Err(shape_err(format!(
                        "context rows from position {first}; the ring holds {} rows from {base}, \
                         so the next row is {next}",
                        self.count
                    )));
                }
            }
        }
        if self.count + n > self.spec.ctx {
            return Err(shape_err(format!(
                "{} context rows past a capacity of {}",
                self.count + n,
                self.spec.ctx
            )));
        }
        let words = self.words();
        self.s.words.copy_from_host(gpu.stream(), &words)?;
        let cs = self.cos_sin(first, n, APPEND_MAX);
        self.s.ctx_cs.copy_from_host(gpu.stream(), &cs)?;
        self.count += n;
        self.staged_rows = Some(n);
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
        let CtxDims {
            hidden: h,
            n_kv,
            head_dim: hd,
            ..
        } = self.dims;
        let n_layers = self.layers.len();
        if n == 0 || n > APPEND_MAX || x.len() < n_layers * APPEND_MAX * h {
            return Err(shape_err(format!(
                "x holds {} values; {n_layers} mixes of {APPEND_MAX} rows are {}",
                x.len(),
                n_layers * APPEND_MAX * h
            )));
        }
        let stream = gpu.stream();
        let elem = gpu.elem();
        let eps = self.spec.eps;
        let Self {
            layers, s, attn, ..
        } = self;
        for (l, layer) in layers.iter_mut().enumerate() {
            let mix = Window::<f32>::of(x, 4 * l * APPEND_MAX * h, n * h)?;
            gemv(gpu, &layer.k_ctx, &mix, n, &mut s.ck)?;
            gemv(gpu, &layer.v_ctx, &mix, n, &mut s.cv)?;
            elem.enqueue_rms_norm(
                stream,
                &s.ck,
                &layer.k_norm_ctx,
                eps,
                hd,
                n * n_kv,
                &mut s.ckn,
            )?;
            let heads = u32::try_from(n_kv).unwrap_or(0);
            elem.enqueue_rope(stream, &s.ckn, &s.ctx_cs, hd, heads, n, &mut s.ckr)?;
            let planes = layer.cap + BLOCK;
            for (src, dst) in [(&s.ckr, &mut layer.kc), (&s.cv, &mut layer.vc)] {
                attn.enqueue_put(
                    stream,
                    PutArgs {
                        src,
                        words: &s.words,
                        word: n_layers + l,
                        offset: 0,
                        modulus: layer.cap,
                        n_kv,
                        planes,
                        n,
                        dst,
                    },
                )?;
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
        if m == 0 || m > BLOCK {
            return Err(shape_err(format!(
                "a block of {m} rows; a pass holds 1..={BLOCK}"
            )));
        }
        if let Some(base) = self.base {
            let next = u64::from(base) + self.count as u64;
            if u64::from(pos) != next {
                return Err(shape_err(format!(
                    "a block at position {pos}; the {} context rows from {base} end at {next}",
                    self.count
                )));
            }
        }
        if ctx != self.count {
            return Err(shape_err(format!(
                "a block over {ctx} context rows; the layers hold {}",
                self.count
            )));
        }
        let words = self.words();
        self.s.words.copy_from_host(gpu.stream(), &words)?;
        let cs = self.cos_sin(pos, BLOCK, BLOCK);
        self.s.blk_cs.copy_from_host(gpu.stream(), &cs)?;
        self.anchor_row(last)?;
        let h = self.dims.hidden;
        let mut first = WindowMut::<f32>::of_mut(&mut self.s.rows, 0, h)?;
        first.copy_from_host(gpu.stream(), &self.row)?;
        self.staged_block = Some(m);
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
        let CtxDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = self.dims;
        let b = BLOCK;
        if m == 0 || m > b || x.len() < b * h {
            return Err(shape_err(format!(
                "x holds {} values; the block's {b} rows are {}",
                x.len(),
                b * h
            )));
        }
        let stream = gpu.stream();
        let elem = gpu.elem();
        let fault = gpu.unlabelled_sink();
        let eps = self.spec.eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let n_layers = self.layers.len();
        let Self {
            layers, s, attn, ..
        } = self;
        {
            let rows = Window::<f32>::of(&s.rows, 0, b * h)?;
            let mut input = WindowMut::<f32>::of_mut(&mut *x, 0, b * h)?;
            input.copy_from_device_raw(&rows, stream)?;
        }
        let (qh, kh) = (
            u32::try_from(n_q).unwrap_or(0),
            u32::try_from(n_kv).unwrap_or(0),
        );
        for (l, layer) in layers.iter_mut().enumerate() {
            elem.enqueue_rms_norm(stream, x, &layer.ln_in, eps, h, b, &mut s.xn)?;
            gemv(gpu, &layer.q, &s.xn, b, &mut s.q)?;
            elem.enqueue_rms_norm(stream, &s.q, &layer.q_norm, eps, hd, b * n_q, &mut s.qn)?;
            elem.enqueue_rope(stream, &s.qn, &s.blk_cs, hd, qh, b, &mut s.qr)?;
            gemv(gpu, &layer.k, &s.xn, b, &mut s.k)?;
            elem.enqueue_rms_norm(stream, &s.k, &layer.k_norm, eps, hd, b * n_kv, &mut s.kn)?;
            elem.enqueue_rope(stream, &s.kn, &s.blk_cs, hd, kh, b, &mut s.kr)?;
            gemv(gpu, &layer.v, &s.xn, b, &mut s.v)?;
            let planes = layer.cap + BLOCK;
            for (src, dst) in [(&s.kr, &mut layer.kc), (&s.v, &mut layer.vc)] {
                attn.enqueue_put(
                    stream,
                    PutArgs {
                        src,
                        words: &s.words,
                        word: 2 * n_layers,
                        offset: layer.cap,
                        modulus: BLOCK,
                        n_kv,
                        planes,
                        n: b,
                        dst,
                    },
                )?;
            }
            attn.enqueue_attn(
                stream,
                AttnArgs {
                    q: &s.qr,
                    kc: &layer.kc,
                    vc: &layer.vc,
                    words: &s.words,
                    word: l,
                    cap: layer.cap,
                    n_kv,
                    m: b,
                    scale,
                    fault,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                    y: &mut s.attn,
                },
            )?;
            gemv(gpu, &layer.o, &s.attn, b, &mut s.o)?;
            elem.enqueue_add(stream, x, &s.o, b * h, &mut s.x1)?;
            elem.enqueue_rms_norm(stream, &s.x1, &layer.ln_post, eps, h, b, &mut s.xn2)?;
            gemv(gpu, &layer.gate, &s.xn2, b, &mut s.g)?;
            gemv(gpu, &layer.up, &s.xn2, b, &mut s.u)?;
            elem.enqueue_swiglu(stream, &s.g, &s.u, b * inter, &mut s.act)?;
            gemv(gpu, &layer.down, &s.act, b, &mut s.dn)?;
            elem.enqueue_add(stream, &s.x1, &s.dn, b * h, x)?;
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
        let q8 = |q: &Q8Rows| (q.qs.buf().num_bytes() + q.d.buf().num_bytes()) as u64;
        let f = |b: &DeviceBuffer<f32>| b.num_bytes() as u64;
        let weights: u64 = self
            .layers
            .iter()
            .map(|l| {
                [
                    &l.q, &l.k, &l.v, &l.k_ctx, &l.v_ctx, &l.o, &l.gate, &l.up, &l.down,
                ]
                .into_iter()
                .map(q8)
                .sum::<u64>()
                    + [&l.ln_in, &l.ln_post, &l.q_norm, &l.k_norm, &l.k_norm_ctx]
                        .into_iter()
                        .map(f)
                        .sum::<u64>()
            })
            .sum();
        let planes: u64 = self
            .layers
            .iter()
            .zip(&self.spec.windowed)
            .map(|(_, &w)| {
                let cap = if w { self.spec.window } else { ctx };
                2 * (self.dims.n_kv * (cap + BLOCK) * ROW_WORDS * 4) as u64
            })
            .sum();
        let s = &self.s;
        let scratch: u64 = [
            &s.xn, &s.q, &s.qn, &s.qr, &s.k, &s.kn, &s.kr, &s.v, &s.attn, &s.o, &s.x1, &s.xn2,
            &s.g, &s.u, &s.act, &s.dn, &s.part_v, &s.part_ms, &s.ck, &s.ckn, &s.ckr, &s.cv,
            &s.blk_cs, &s.ctx_cs, &s.rows,
        ]
        .into_iter()
        .map(f)
        .sum::<u64>()
            + s.words.num_bytes() as u64;
        weights + planes + scratch
    }
}
