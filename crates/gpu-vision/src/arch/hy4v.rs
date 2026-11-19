//! The `hy_v4_flash` tower (Hy4-flash's mmproj: llama.cpp's `hy_v4_flash` graph) on the card: the
//! weights of one encoder file uploaded once, the activations of the largest image its token limit
//! allows allocated once beside them, and [`Encoder::encode`], from an image's bf16 patches (merge
//! order, one frame a patch, [`vision::arch::hy4v::media`]) to the bf16 rows the text model reads in
//! place of the image's pad ids: the merged tokens, grid row by grid row, each grid row followed by
//! the file's newline row.
//!
//! The chain, op by op, with the graph node of mtmd's `hy_v4_flash` graph each launch transcribes (a
//! launch is one kernel; every op reads bf16, computes in f32 and rounds its output to bf16, the
//! contract of [`crate`]):
//!
//! | # | launch | graph node | tap |
//! |---|---|---|---|
//! | 1 | position rows: the 128×128 table resampled to the grid | `hunyuanvl_pos_embd` (computed on the host by mtmd) | `pos` |
//! | 2 | GEMM `patches · patch_embdᵀ + b`, the position rows as the residual operand (K = 3·p²) | `build_inp`, the learned position add of `build_vit` | `embed.patch` (without the residual), `embed` |
//! | | per block `b` of `n_layer`: | | |
//! | 3 | LayerNorm `ln1` | `ln1` | `blk{b}.norm1` |
//! | 4 | GEMM `h · [q \| k \| v]ᵀ + b` (one GEMM of `3·dim` columns) | `Qcur`/`Kcur`/`Vcur` | `blk{b}.qkv` |
//! | 5 | attention, heads of 72, non-causal, no RoPE | `build_attn` | `blk{b}.sdpa` |
//! | 6 | GEMM `a · woᵀ + b`, residual epilogue | `attn_out`, `ffn_inp` | `blk{b}.attn` (without the residual), `blk{b}.resid1` |
//! | 7 | LayerNorm `ln2` | `ffn_inp_normed` | `blk{b}.norm2` |
//! | 8 | GEMM `h · upᵀ + b` (the 4304 rows padded to [`card::ff_padded`]) | `ffn_up` | `blk{b}.up` |
//! | 9 | GELU (tanh), in place | `FFN_GELU` | `blk{b}.act` |
//! | 10 | GEMM `act · downᵀ + b`, residual epilogue | `ffn_down`, `layer_out` | `blk{b}.mlp` (without the residual), `blk{b}` |
//! | 11 | RMS norm `mm.pre_norm` | the adapter's `build_norm(…, NORM_TYPE_RMS)` | `merger.norm` |
//! | 12 | GEMM `[n/4, 4·dim] · mm0ᵀ + b`, GELU (erf) epilogue, the merge a re-read of the rows | `ggml_conv_2d` (k 2, s 2), its bias, `ggml_gelu_erf` | `conv.pre` (without the GELU), `conv` |
//! | 13 | GEMM `· mm2ᵀ + b` | the 1×1 `ggml_conv_2d` and its bias | `proj` |
//! | 14 | copy of each grid row's tokens to its place among the rows | the permute of the tokens | |
//! | 15 | copy of the newline row after each grid row | the `ggml_concat` of `v.image_newline` | `embd` (the rows) |
//!
//! Launches per image: [`Encoder::launches`]. Eager, as the other towers: the grid of every launch
//! depends on the image's patch count.
//!
//! The ViT runs in merge order: its blocks are per-patch but for the attention, which the same order
//! of the queries, keys and values leaves as the reference's raster order computes it, row for row;
//! the position rows are written in that order ([`crate::pos_hunyuan`]); the strided convolution
//! then reads a group of `merge²` patches as one row, and its tokens come out in the raster order of
//! the merged grid, the reference's.
//!
//! Where this tower differs from the shared kernels' other users, and how the difference enters: the
//! position table is resampled by HunyuanVL's rule, [`crate::pos_hunyuan`]'s own entry; the strided
//! convolution's K order is [`vision::arch::merger`]'s, applied at load; the newline rows are
//! [`crate::pointwise`]'s copy, as Pixtral's break rows, one after every grid row.
//!
//! Weights as the card holds them: [`vision::arch::hy4v::card::weights`] (every matrix bf16, the f32
//! patch kernel, position table and both convolutions narrowed only when every value round-trips
//! through bf16, else the load is refused by name; a block's query, key and value as one
//! `[3·dim, dim]` matrix with one bias; every norm and bias f32). The newline row is written once
//! into the newline rows of the scratch. The load refuses uploads that differ from the figure.
//!
//! Scratch is allocated once from the token limits ([`vision::arch::hy4v::card::ScratchLens`]); an
//! image allocates nothing, and one past the limit is refused by name.

use crate::attn::{Attn72Kernels, AttnArgs, HEAD_DIM_72, QkvLayout};
use crate::chain::{
    Encoded, NoTaps, TapSink, read_bf16_narrowed, read_f32, side, tap, to_host, up_f32,
};
use crate::gelu_tanh::GeluTanhKernels;
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels, TILE_N};
use crate::layer_norm::LayerNormKernels;
use crate::norm::NormKernels;
use crate::pointwise::{self, Op};
use crate::pos_hunyuan::{MERGE, PosHunyuanKernels};
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::hy4v::card::{self, ScratchLens, ff_padded};
use vision::arch::hy4v::{Hparams, POS_SIDE, TokenLimits, names, rows_of, tensors};
use vision::arch::merger::reorder_merger_weight;

const WHAT: &str = "hy_v4_flash tower load";

/// Bytes of weights the card holds for a file of `hp`: [`card::weights`].
#[must_use]
pub fn weight_bytes_of(hp: &Hparams) -> u64 {
    card::weights(hp)
}

/// Bytes of activations the card holds for images of at most `limits`' token count.
#[must_use]
pub fn scratch_bytes_of(hp: &Hparams, limits: TokenLimits) -> u64 {
    ScratchLens::of(hp, limits).bytes()
}

/// One block's weights on the card.
struct BlockWeights {
    ln1: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    /// `[3·dim, dim]`: the query's rows, the key's, then the value's.
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
    ln2: (DeviceBuffer<f32>, DeviceBuffer<f32>),
    /// `ffn_up`'s rows, zero-padded to [`ff_padded`].
    up_w: DeviceBuffer<u16>,
    up_b: DeviceBuffer<f32>,
    /// `ffn_down`'s rows, each zero-padded to [`ff_padded`] columns.
    down_w: DeviceBuffer<u16>,
    down_b: DeviceBuffer<f32>,
}

/// Every weight of the tower on the card.
struct Weights {
    /// `[dim, 3·p²]`, narrowed from the file's f32.
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<f32>,
    /// `POS_SIDE²` rows of `dim`, narrowed from the file's f32.
    pos: DeviceBuffer<u16>,
    blocks: Vec<BlockWeights>,
    pre_norm: DeviceBuffer<f32>,
    /// `[4·dim, 4·dim]` in the card's K order, narrowed from the file's f32.
    conv_w: DeviceBuffer<u16>,
    conv_b: DeviceBuffer<f32>,
    /// `[out, 4·dim]`, narrowed from the file's f32.
    proj_w: DeviceBuffer<u16>,
    proj_b: DeviceBuffer<f32>,
}

impl Weights {
    /// Bytes uploaded: two per bf16, four per f32.
    fn uploaded(&self) -> u64 {
        let pair = |p: &(DeviceBuffer<f32>, DeviceBuffer<f32>)| p.0.len() + p.1.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                2 * (b.qkv_w.len() + b.o_w.len() + b.up_w.len() + b.down_w.len())
                    + 4 * (pair(&b.ln1)
                        + pair(&b.ln2)
                        + b.qkv_b.len()
                        + b.o_b.len()
                        + b.up_b.len()
                        + b.down_b.len())
            })
            .sum();
        (blocks
            + 2 * (self.patch_w.len() + self.pos.len() + self.conv_w.len() + self.proj_w.len())
            + 4 * (self.patch_b.len()
                + self.pre_norm.len()
                + self.conv_b.len()
                + self.proj_b.len())) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels and the weights.
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    ln: LayerNormKernels,
    rms: NormKernels,
    gelu: GeluTanhKernels,
    pos: PosHunyuanKernels,
    attn: Attn72Kernels,
    ops: pointwise::Kernels,
    w: Weights,
}

/// The activations of the largest image the limit allows, allocated at load. An image of `n`
/// patches uses the first `n` rows of each buffer: its patches and position rows are written first,
/// and every launch reads only rows that a copy or an earlier launch of the same image wrote, so
/// nothing a larger image left behind is read. `newlines` is the one buffer written at load.
struct Scratch {
    lens: ScratchLens,
    patches: DeviceBuffer<u16>,
    pos: DeviceBuffer<u16>,
    /// The block input and output, the attention-residual stream, and the normed rows.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    /// The up projection, and the GELU of it in place.
    up: DeviceBuffer<u16>,
    /// The strided convolution's rows after their GELU.
    hid: DeviceBuffer<u16>,
    proj: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
    /// The file's newline row, bf16, `lens.max_tokens` times.
    newlines: DeviceBuffer<u16>,
}

/// The loaded tower: the chain of one file, the activations of the largest image its limit allows,
/// its bytes on the card, and the stream every call runs on. A call takes `&mut self`: one image at
/// a time owns the activations, and the rows it returns borrow them until the next call.
pub struct Encoder {
    chain: Chain,
    s: Scratch,
    weight_bytes: u64,
    scratch_bytes: u64,
    stream: Arc<CudaStream>,
}

impl crate::Tower for Encoder {
    fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        Encoder::encode(self, patches)
    }
    fn out_dim(&self) -> usize {
        self.hparams().out_dim
    }
    fn weight_bytes(&self) -> u64 {
        Encoder::weight_bytes(self)
    }
    fn scratch_bytes(&self) -> u64 {
        Encoder::scratch_bytes(self)
    }
}

fn err(what: &'static str, e: impl std::error::Error + Send + Sync + 'static) -> GpuError {
    GpuError::plan(what, e)
}

impl Encoder {
    /// Read the file's hyperparameters and tensor table (both refused by name when they are not a
    /// `hy_v4_flash` encoder's), upload every weight — refused by name when the uploads are not
    /// [`card::weights`]' figure — and allocate the activations for images of at most
    /// `limits.max()` merged tokens; every later call runs on `stream`. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &Gguf,
        limits: TokenLimits,
    ) -> Result<Encoder, GpuError> {
        let hp = Hparams::read(file).map_err(|e| err(WHAT, e))?;
        tensors::check(
            &hp,
            file.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .map_err(|e| err(WHAT, e))?;
        if hp.dim != hp.n_head * HEAD_DIM_72
            || hp.merge != MERGE
            || !hp.dim.is_multiple_of(TILE_N)
            || !hp.out_dim.is_multiple_of(TILE_N)
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} values in {} heads, merge {}, out {}; the tower runs heads of {HEAD_DIM_72}, merge {MERGE} and widths that are multiples of {TILE_N}",
                    hp.dim, hp.n_head, hp.merge, hp.out_dim
                ),
            });
        }
        let ff = ff_padded(&hp);
        let narrow = |name: String| read_bf16_narrowed(file, &name, WHAT);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(stream, &narrow(name)?)?)
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut qkv_w = narrow(names::attn_q_weight(b))?;
            qkv_w.extend(narrow(names::attn_k_weight(b))?);
            qkv_w.extend(narrow(names::attn_v_weight(b))?);
            let mut qkv_b = read_f32(file, &names::attn_q_bias(b), WHAT)?;
            qkv_b.extend(read_f32(file, &names::attn_k_bias(b), WHAT)?);
            qkv_b.extend(read_f32(file, &names::attn_v_bias(b), WHAT)?);
            let mut up_w = narrow(names::ffn_up_weight(b))?;
            up_w.resize(ff * hp.dim, 0);
            let mut up_b = read_f32(file, &names::ffn_up_bias(b), WHAT)?;
            up_b.resize(ff, 0.0);
            let down_w: Vec<u16> = narrow(names::ffn_down_weight(b))?
                .chunks_exact(hp.ff)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0, ff - hp.ff))
                })
                .collect();
            blocks.push(BlockWeights {
                ln1: (f32s(names::ln1_weight(b))?, f32s(names::ln1_bias(b))?),
                qkv_w: DeviceBuffer::from_host(stream, &qkv_w)?,
                qkv_b: DeviceBuffer::from_host(stream, &qkv_b)?,
                o_w: bf16(names::attn_out_weight(b))?,
                o_b: f32s(names::attn_out_bias(b))?,
                ln2: (f32s(names::ln2_weight(b))?, f32s(names::ln2_bias(b))?),
                up_w: DeviceBuffer::from_host(stream, &up_w)?,
                up_b: DeviceBuffer::from_host(stream, &up_b)?,
                down_w: DeviceBuffer::from_host(stream, &down_w)?,
                down_b: f32s(names::ffn_down_bias(b))?,
            });
        }
        let merged = hp.merged();
        let conv_w =
            reorder_merger_weight(&narrow(names::conv_weight())?, hp.dim, hp.merge, merged);
        let w = Weights {
            patch_w: bf16(names::patch_embd_weight())?,
            patch_b: f32s(names::patch_embd_bias())?,
            pos: bf16(names::position_embd())?,
            blocks,
            pre_norm: f32s(names::pre_norm_weight())?,
            conv_w: DeviceBuffer::from_host(stream, &conv_w)?,
            conv_b: f32s(names::conv_bias())?,
            proj_w: bf16(names::proj_weight())?,
            proj_b: f32s(names::proj_bias())?,
        };
        let (uploaded, figure) = (w.uploaded(), card::weights(&hp));
        if uploaded != figure {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "uploaded {uploaded} B of weights; the figure of the file's hyperparameters (card::weights) is {figure} B"
                ),
            });
        }
        let lens = ScratchLens::of(&hp, limits);
        let s = Scratch::new(stream, lens, &narrow(names::image_newline())?)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                ln: LayerNormKernels::load(ctx)?,
                rms: NormKernels::load(ctx)?,
                gelu: GeluTanhKernels::load(ctx)?,
                pos: PosHunyuanKernels::load(ctx)?,
                attn: Attn72Kernels::load(ctx)?,
                ops: pointwise::Kernels::load(ctx)?,
                hp,
                w,
            },
            s,
            weight_bytes: uploaded,
            scratch_bytes: lens.bytes(),
            stream: Arc::clone(stream),
        })
    }

    /// The file's hyperparameters.
    #[must_use]
    pub fn hparams(&self) -> &Hparams {
        &self.chain.hp
    }

    /// Bytes of weights on the card: what the load uploaded, held to [`card::weights`].
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Bytes of activations on the card: the buffers of the largest image the limit allows.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Chain launches of one encode at `n_layer` blocks: the position rows and the patch GEMM,
    /// eight per block, and the adapter's five (the norm, the two convolutions, the copy of the
    /// tokens and the copy of the newline rows).
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        2 + 8 * n_layer + 5
    }

    /// Encode one image's patches. Eager: copies the patches into the tower's buffers, enqueues
    /// the chain on its stream and returns the rows without waiting for it; read them on that
    /// stream.
    pub fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        self.encode_tapped(patches, &mut NoTaps)
    }

    /// [`Encoder::encode`] handing the named intermediate tensors to `taps` ([`TapSink`]).
    pub fn encode_tapped(
        &mut self,
        patches: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded<'_>, GpuError> {
        let what = "hy_v4_flash tower encode";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let (n_h, n_w) = (patches.n_vit_h, patches.n_vit_w);
        let n = n_h * n_w;
        let k_patch = 3 * hp.patch * hp.patch;
        if n == 0 || patches.patch_len != k_patch || patches.bf16.len() != n * k_patch {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_h}x{n_w} patches of {} values ({} in all); the tower takes {k_patch} per patch",
                    patches.patch_len,
                    patches.bf16.len()
                ),
            });
        }
        s.start(hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.patches, 0, n * k_patch)?
            .copy_from_host(stream, &patches.bf16)?;
        chain.pos.enqueue(
            stream,
            &chain.w.pos,
            (POS_SIDE, hp.dim),
            (n_h, n_w),
            &mut s.pos,
        )?;
        tap(stream, taps, "pos", &s.pos, n, hp.dim)?;
        side(
            &chain.gemm,
            stream,
            taps,
            "embed.patch",
            (&s.patches, &chain.w.patch_w, Some(&chain.w.patch_b)),
            (n, hp.dim, k_patch),
            &mut s.xb,
        )?;
        chain.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.patches,
                b: &chain.w.patch_w,
                bias: Some(&chain.w.patch_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.pos),
                m: n,
                n: hp.dim,
                k: k_patch,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, "embed", &s.xa, n, hp.dim)?;
        let mut launches = 2;
        for b in 0..hp.n_layer {
            launches += chain.block(stream, b, n, s, taps)?;
        }
        let grid = (n_w / hp.merge, n_h / hp.merge);
        let (n_rows, tail) = chain.tail(stream, grid, s, taps)?;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, n_rows, hp.out_dim)?,
            launches: launches + tail,
        })
    }

    /// The adapter alone on the host rows `vit` (`n_h · n_w` rows of `dim` bf16 in merge order, the
    /// last block's output): a teacher-forced step for a gate that feeds the reference's
    /// `layer_out` of the last block to it. Returns the adapter's rows before the newlines (the
    /// `proj` tap) and the output rows.
    pub fn forced_tail(
        &mut self,
        (n_h, n_w): (usize, usize),
        vit: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        let what = "hy_v4_flash tower forced step";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let n = n_h * n_w;
        if n_h == 0 || n_w == 0 || vit.len() != n * hp.dim {
            return Err(GpuError::Shape {
                what,
                detail: format!("{} values for {n_h}x{n_w} rows of {}", vit.len(), hp.dim),
            });
        }
        s.start(hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.xa, 0, vit.len())?.copy_from_host(stream, vit)?;
        let grid = (n_w / hp.merge, n_h / hp.merge);
        let (n_rows, _) = chain.tail(stream, grid, s, &mut NoTaps)?;
        Ok((
            to_host(stream, &s.proj, grid.0 * grid.1 * hp.out_dim)?,
            to_host(stream, &s.rows, n_rows * hp.out_dim)?,
        ))
    }
}

impl Scratch {
    /// The buffers at `lens`, zero-filled but the newline rows, which hold `newline` (`out_dim`
    /// bf16 values) `lens.max_tokens` times. Load-time only.
    fn new(stream: &CudaStream, lens: ScratchLens, newline: &[u16]) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
        let newlines = newline.repeat(lens.max_tokens);
        if newlines.len() != lens.newlines {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a newline row of {} values, {} times, is not the {} values of the newline rows",
                    newline.len(),
                    lens.max_tokens,
                    lens.newlines
                ),
            });
        }
        Ok(Scratch {
            lens,
            patches: bf16(lens.patches)?,
            pos: bf16(lens.pos)?,
            xa: bf16(lens.stream)?,
            xb: bf16(lens.stream)?,
            h: bf16(lens.stream)?,
            qkv: bf16(lens.qkv)?,
            att: bf16(lens.att)?,
            up: bf16(lens.up)?,
            hid: bf16(lens.hidden)?,
            proj: bf16(lens.proj)?,
            rows: bf16(lens.rows)?,
            newlines: DeviceBuffer::from_host(stream, &newlines)?,
        })
    }

    /// Start an `n_h × n_w` image: refuse a grid that is not whole merge groups or is larger than
    /// the buffers.
    fn start(
        &self,
        hp: &Hparams,
        (n_h, n_w): (usize, usize),
        what: &'static str,
    ) -> Result<(), GpuError> {
        let n = n_h * n_w;
        if !n_h.is_multiple_of(hp.merge) || !n_w.is_multiple_of(hp.merge) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid is not whole {0}x{0} merge groups",
                    hp.merge
                ),
            });
        }
        if n > self.lens.max_patches {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid ({n} patches, {} tokens) does not fit the tower's buffers, sized for {} tokens",
                    n / (hp.merge * hp.merge),
                    self.lens.max_tokens
                ),
            });
        }
        Ok(())
    }
}

impl Chain {
    /// Block `b` of the chain on the first `n` rows of `s.xa`, its output back in `s.xa`; returns
    /// the launches.
    fn block(
        &self,
        stream: &CudaStream,
        b: usize,
        n: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<usize, GpuError> {
        let hp = &self.hp;
        let bw = &self.w.blocks[b];
        let (dim, heads, ff) = (hp.dim, hp.n_head, ff_padded(hp));
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: heads,
        };
        let name = |op: &str| format!("blk{b}.{op}");
        self.ln
            .enqueue(stream, &s.xa, (&bw.ln1.0, &bw.ln1.1), hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm1"), &s.h, n, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.qkv_w,
                bias: Some(&bw.qkv_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: 3 * dim,
                k: dim,
                c: &mut s.qkv,
            },
        )?;
        tap(stream, taps, &name("qkv"), &s.qkv, n, 3 * dim)?;
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: lay,
                n,
                scale: 1.0 / (HEAD_DIM_72 as f32).sqrt(),
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        tap(stream, taps, &name("sdpa"), &s.att, n, dim)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("attn"),
            (&s.att, &bw.o_w, Some(&bw.o_b)),
            (n, dim, dim),
            &mut s.xb,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.att,
                b: &bw.o_w,
                bias: Some(&bw.o_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xa),
                m: n,
                n: dim,
                k: dim,
                c: &mut s.xb,
            },
        )?;
        tap(stream, taps, &name("resid1"), &s.xb, n, dim)?;
        self.ln
            .enqueue(stream, &s.xb, (&bw.ln2.0, &bw.ln2.1), hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm2"), &s.h, n, dim)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &bw.up_w,
                bias: Some(&bw.up_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: ff,
                k: dim,
                c: &mut s.up,
            },
        )?;
        tap(stream, taps, &name("up"), &s.up, n, ff)?;
        self.gelu.enqueue(stream, n * ff, &mut s.up)?;
        tap(stream, taps, &name("act"), &s.up, n, ff)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("mlp"),
            (&s.up, &bw.down_w, Some(&bw.down_b)),
            (n, dim, ff),
            &mut s.xa,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.up,
                b: &bw.down_w,
                bias: Some(&bw.down_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xb),
                m: n,
                n: dim,
                k: ff,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, n, dim)?;
        Ok(8)
    }

    /// The adapter on the last block's output in `s.xa`, for an image of `nx × ny` merged tokens:
    /// the RMS norm into `s.h`, the strided convolution — `s.h` re-read as rows of `merge²·dim`
    /// values — with its GELU into `s.hid`, the 1×1 convolution into `s.proj`, and the rows the
    /// text model reads into `s.rows`: grid row `r`'s tokens at row `r·(nx + 1)`, the newline row
    /// after each. Returns the rows ([`rows_of`]) and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        (nx, ny): (usize, usize),
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let (dim, out, merged) = (hp.dim, hp.out_dim, hp.merged());
        let n_tok = nx * ny;
        let n = n_tok * hp.merge * hp.merge;
        self.rms
            .enqueue(stream, &s.xa, &self.w.pre_norm, hp.eps, n, &mut s.h)?;
        tap(stream, taps, "merger.norm", &s.h, n, dim)?;
        side(
            &self.gemm,
            stream,
            taps,
            "conv.pre",
            (&s.h, &self.w.conv_w, Some(&self.w.conv_b)),
            (n_tok, merged, merged),
            &mut s.hid,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &self.w.conv_w,
                bias: Some(&self.w.conv_b),
                epilogue: Epilogue::Gelu,
                resid: None,
                m: n_tok,
                n: merged,
                k: merged,
                c: &mut s.hid,
            },
        )?;
        tap(stream, taps, "conv", &s.hid, n_tok, merged)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.hid,
                b: &self.w.proj_w,
                bias: Some(&self.w.proj_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: out,
                k: merged,
                c: &mut s.proj,
            },
        )?;
        tap(stream, taps, "proj", &s.proj, n_tok, out)?;
        // The copies read neither their second operand nor the gain and bias: the source and the
        // adapter's norm gain stand in for them.
        let stand_in = &self.w.pre_norm;
        for (x, width, out_col) in [(&s.proj, nx * out, 0), (&s.newlines, out, nx * out)] {
            self.ops.enqueue(
                stream,
                pointwise::Args {
                    op: Op::Copy,
                    x,
                    b: x,
                    gain: stand_in,
                    bias: stand_in,
                    rows: ny,
                    width,
                    out_width: (nx + 1) * out,
                    out_col,
                    scale: 1.0,
                    clamp: (0.0, 0.0),
                    out: &mut s.rows,
                },
            )?;
        }
        let n_rows = rows_of(nx, ny);
        tap(stream, taps, "embd", &s.rows, n_rows, out)?;
        Ok((n_rows, 5))
    }
}
