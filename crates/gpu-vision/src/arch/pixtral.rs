//! The `pixtral` tower (Mistral Large 4's mmproj: llama.cpp's `pixtral` graph) on the card: the
//! weights of one encoder file uploaded once, the activations of the largest image its token
//! limit allows allocated once beside them, and [`Encoder::encode`], from an image's bf16 patches
//! (merge order, one frame a patch, [`vision::arch::pixtral::media`]) to the bf16 rows the text
//! model reads in place of the image's `[IMG]` ids: the merged tokens, grid row by grid row, with
//! the file's `[IMG_BREAK]` row between two grid rows.
//!
//! The chain, op by op, with the graph node of `models/pixtral.cpp` / `clip.cpp` each launch
//! transcribes (a launch is one kernel; every op reads bf16, computes in f32 and rounds its
//! output to bf16, the contract of [`crate`]):
//!
//! | # | launch | graph node | tap |
//! |---|---|---|---|
//! | 1 | GEMM `patches · wᵀ` (K = 3·p²), no bias | the patch convolution (`build_inp`) | `embed` |
//! | 2 | RMS norm `pre_ln` | `build_vit`'s pre-norm | `pre_ln` |
//! | | per block `b` of `n_layer`: | | |
//! | 3 | RMS norm `ln1` | `layer_inp_normed` | `blk{b}.norm1` |
//! | 4 | GEMM `h · [q \| k \| v]ᵀ`, no bias | `Qcur`/`Kcur`/`Vcur` | `blk{b}.qkv` |
//! | 5 | 2D RoPE of q and k in place, heads of 128 | `build_rope_2d` (`Qcur_pos`, `Kcur_pos`) | `blk{b}.qrot`, `.krot` |
//! | 6 | attention, heads of 128, non-causal | `build_attn` | `blk{b}.sdpa` |
//! | 7 | GEMM `a · woᵀ`, residual epilogue | `attn_out`, `ffn_inp` | `blk{b}.attn` (without the residual), `blk{b}.resid1` |
//! | 8 | RMS norm `ln2` | `ffn_inp_normed` | `blk{b}.norm2` |
//! | 9 | GEMM `h · [gate \| up]ᵀ` (one GEMM of `2·ff` columns) | `ffn_gate`, `ffn_up` | `blk{b}.gu` |
//! | 10 | SiLU gate | `ffn_swiglu` | `blk{b}.act` |
//! | 11 | GEMM `act · downᵀ`, residual epilogue | `ffn_down`, `layer_out` | `blk{b}.mlp` (without the residual), `blk{b}` |
//! | 12 | RMS norm `input_norm` | the merger's `ggml_rms_norm · mm_input_norm_w` | `merger.norm` |
//! | 13 | GEMM `[n/4, 4·dim] · mergerᵀ`, the merge a re-read of the rows | `im2col` and `mm_patch_merger_w` | `merger` |
//! | 14 | GEMM `· mm1ᵀ` | `build_ffn`'s up | `mm1` |
//! | 15 | GELU (tanh), in place | `FFN_GELU` | `gelu` |
//! | 16 | GEMM `· mm2ᵀ` | `build_ffn`'s down | `proj` |
//! | 17 | copy of each grid row's tokens to its place among the rows | the `ggml_concat` of the tokens | |
//! | 18 | copy of the break row after each grid row but the last (none for one grid row) | its `[IMG_BREAK]` side | `embd` (the rows) |
//!
//! Launches per image: [`Encoder::launches`]. Eager, as the other towers: the grid of every launch
//! depends on the image's patch count.
//!
//! The ViT runs in merge order: its blocks are per-patch but for the attention, which the same
//! order of the queries, keys, values and RoPE positions leaves as the reference's raster order
//! computes it, row for row; the merger then reads a group of `merge²` patches as one row, and
//! its tokens come out in the raster order of the merged grid, the reference's.
//!
//! Where this tower differs from the shared kernels' other users, and how the difference enters:
//! the heads are 128 values, so the RoPE and the attention are [`crate::rope2d::Rope128Kernels`]
//! and [`crate::attn_h128`]; llama.cpp's `build_rope_2d` turns adjacent pairs of each half of a
//! head and the kernel turns `(j, j + 64)`, so the query's and key's rows are permuted at load by
//! [`rope_source_128`] (the same permutation on both leaves `q · k` and the value rows as they
//! are) and the table is [`RopeTable128::interleaved`]'s; the merger's K order is
//! [`vision::arch::merger`]'s, applied at load; the break rows are [`crate::pointwise`]'s copy.
//!
//! Weights as the card holds them: every matrix bf16 (the f32 patch kernel and patch merger
//! narrow only when every value round-trips through bf16, else the load is refused by name); a
//! block's query, key and value as one `[3·dim, dim]` matrix and its gate and up as one matrix of
//! the gate's rows then the up's; every norm gain f32. The `[IMG_BREAK]` row is written once into
//! the break rows of the scratch. The load refuses uploads that differ from
//! [`vision::arch::pixtral::card::weights`].
//!
//! Scratch is allocated once from the token limits ([`vision::arch::pixtral::card::ScratchLens`]);
//! an image allocates nothing, and one past the limit is refused by name.

use crate::attn::{AttnArgs, QkvLayout};
use crate::attn_h128::{Attn128Kernels, HEAD_DIM_128};
use crate::chain::{
    Encoded, NoTaps, TapSink, linear, read_bf16_narrowed, side, tap, tap_cols, to_host, up_f32,
};
use crate::gelu_tanh::GeluTanhKernels;
use crate::gemm_bf16::GemmKernels;
use crate::mlp::MlpKernels;
use crate::norm::NormKernels;
use crate::pointwise::{self, Op};
use crate::rope2d::{Rope128Kernels, RopeArgs, RopeTable128, merge_order, rope_source_128};
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::merger::reorder_merger_weight;
use vision::arch::pixtral::card::{self, ScratchLens};
use vision::arch::pixtral::{Hparams, ROPE_THETA, TokenLimits, names, rows_of, tensors};

const WHAT: &str = "pixtral tower load";

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

/// A `[dim, dim]` query or key matrix with each head's rows in the card's RoPE order: row
/// `h·128 + r` is the file's row `h·128 + rope_source_128(r)`.
fn rope_rows(m: &[u16], dim: usize) -> Vec<u16> {
    (0..dim)
        .flat_map(|o| {
            let src = o - o % HEAD_DIM_128 + rope_source_128(o % HEAD_DIM_128);
            m[src * dim..(src + 1) * dim].iter().copied()
        })
        .collect()
}

/// One block's weights on the card.
struct BlockWeights {
    ln1: DeviceBuffer<f32>,
    /// `[3·dim, dim]`: the query's rows and the key's, each head's in the card's RoPE order
    /// ([`rope_rows`]), then the value's as the file holds them.
    qkv_w: DeviceBuffer<u16>,
    o_w: DeviceBuffer<u16>,
    ln2: DeviceBuffer<f32>,
    /// The gate's rows then the up's: `2·ff` rows.
    gu_w: DeviceBuffer<u16>,
    down_w: DeviceBuffer<u16>,
}

/// Every weight of the tower on the card.
struct Weights {
    /// `[dim, 3·p²]`, narrowed from the file's f32.
    patch_w: DeviceBuffer<u16>,
    pre_ln: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    input_norm: DeviceBuffer<f32>,
    /// `[dim, merge²·dim]` in the card's K order, narrowed from the file's f32.
    merger_w: DeviceBuffer<u16>,
    mm1_w: DeviceBuffer<u16>,
    mm2_w: DeviceBuffer<u16>,
}

impl Weights {
    /// Bytes uploaded: two per bf16, four per f32.
    fn uploaded(&self) -> u64 {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                2 * (b.qkv_w.len() + b.o_w.len() + b.gu_w.len() + b.down_w.len())
                    + 4 * (b.ln1.len() + b.ln2.len())
            })
            .sum();
        (blocks
            + 2 * (self.patch_w.len() + self.merger_w.len() + self.mm1_w.len() + self.mm2_w.len())
            + 4 * (self.pre_ln.len() + self.input_norm.len())) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels and the weights.
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    norm: NormKernels,
    mlp: MlpKernels,
    gelu: GeluTanhKernels,
    rope: Rope128Kernels,
    attn: Attn128Kernels,
    ops: pointwise::Kernels,
    w: Weights,
}

/// The activations of the largest image the limit allows, allocated at load. An image of `n`
/// patches uses the first `n` rows of each buffer: its patches and RoPE table are written first,
/// and every launch reads only rows that a copy or an earlier launch of the same image wrote, so
/// nothing a larger image left behind is read. `breaks` is the one buffer written at load.
struct Scratch {
    lens: ScratchLens,
    patches: DeviceBuffer<u16>,
    cs: DeviceBuffer<f32>,
    /// The block input and output, the attention-residual stream (later the merger's output), and
    /// the normed rows.
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    /// The MLP's fused gate and up rows, and the SiLU gate's output.
    gu: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    /// The projector's first linear's rows, the GELU's in place, and its output rows.
    mid: DeviceBuffer<u16>,
    proj: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
    /// The file's `[IMG_BREAK]` row, bf16, `lens.max_tokens` times.
    breaks: DeviceBuffer<u16>,
}

/// The loaded tower: the chain of one file, the activations of the largest image its limit
/// allows, its bytes on the card, and the stream every call runs on. A call takes `&mut self`: one
/// image at a time owns the activations, and the rows it returns borrow them until the next call.
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
    /// Read the file's hyperparameters and tensor table (both refused by name when they are not
    /// a `pixtral` encoder's), upload every weight — refused by name when the uploads are not
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
        if hp.dim != hp.n_head * HEAD_DIM_128 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "heads of {} / {} values; the tower runs heads of {HEAD_DIM_128}",
                    hp.dim, hp.n_head
                ),
            });
        }
        let narrow = |name: String| read_bf16_narrowed(file, &name, WHAT);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(stream, &narrow(name)?)?)
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut qkv = rope_rows(&narrow(names::attn_q_weight(b))?, hp.dim);
            qkv.extend(rope_rows(&narrow(names::attn_k_weight(b))?, hp.dim));
            qkv.extend(narrow(names::attn_v_weight(b))?);
            let mut gu = narrow(names::ffn_gate_weight(b))?;
            gu.extend(narrow(names::ffn_up_weight(b))?);
            blocks.push(BlockWeights {
                ln1: f32s(names::ln1_weight(b))?,
                qkv_w: DeviceBuffer::from_host(stream, &qkv)?,
                o_w: bf16(names::attn_out_weight(b))?,
                ln2: f32s(names::ln2_weight(b))?,
                gu_w: DeviceBuffer::from_host(stream, &gu)?,
                down_w: bf16(names::ffn_down_weight(b))?,
            });
        }
        let merger_w = reorder_merger_weight(
            &narrow(names::patch_merger_weight())?,
            hp.dim,
            hp.merge,
            hp.dim,
        );
        let w = Weights {
            patch_w: bf16(names::patch_embd_weight())?,
            pre_ln: f32s(names::pre_ln_weight())?,
            blocks,
            input_norm: f32s(names::input_norm_weight())?,
            merger_w: DeviceBuffer::from_host(stream, &merger_w)?,
            mm1_w: bf16(names::mm1_weight())?,
            mm2_w: bf16(names::mm2_weight())?,
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
        let s = Scratch::new(stream, lens, &narrow(names::img_break())?)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                norm: NormKernels::load(ctx)?,
                mlp: MlpKernels::load(ctx)?,
                gelu: GeluTanhKernels::load(ctx)?,
                rope: Rope128Kernels::load(ctx)?,
                attn: Attn128Kernels::load(ctx)?,
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

    /// Chain launches of one encode at `n_layer` blocks for an image of `ny` grid rows: the patch
    /// GEMM and the pre-norm, nine per block, the projector's five, the copy of the tokens and,
    /// when the image has a second grid row, the copy of its break rows.
    #[must_use]
    pub fn launches(n_layer: usize, ny: usize) -> usize {
        2 + 9 * n_layer + 6 + usize::from(ny > 1)
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
        let what = "pixtral tower encode";
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
        s.start(stream, hp, (n_h, n_w), what)?;
        WindowMut::<u16>::of_mut(&mut s.patches, 0, n * k_patch)?
            .copy_from_host(stream, &patches.bf16)?;
        linear(
            &chain.gemm,
            stream,
            (&s.patches, &chain.w.patch_w, None),
            (n, hp.dim, k_patch),
            None,
            &mut s.xb,
        )?;
        tap(stream, taps, "embed", &s.xb, n, hp.dim)?;
        chain
            .norm
            .enqueue(stream, &s.xb, &chain.w.pre_ln, hp.eps, n, &mut s.xa)?;
        tap(stream, taps, "pre_ln", &s.xa, n, hp.dim)?;
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

    /// The projector alone on the host rows `vit` (`n_h · n_w` rows of `dim` bf16 in merge order,
    /// the last block's output): a teacher-forced step for a gate that feeds the reference's
    /// `layer_out` of the last block to it. Returns the projector's rows before the breaks (the
    /// `proj` tap) and the output rows.
    pub fn forced_tail(
        &mut self,
        (n_h, n_w): (usize, usize),
        vit: &[u16],
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        let what = "pixtral tower forced step";
        let (chain, s, stream) = (&self.chain, &mut self.s, &*self.stream);
        let hp = &chain.hp;
        let n = n_h * n_w;
        if n_h == 0 || n_w == 0 || vit.len() != n * hp.dim {
            return Err(GpuError::Shape {
                what,
                detail: format!("{} values for {n_h}x{n_w} rows of {}", vit.len(), hp.dim),
            });
        }
        s.start(stream, hp, (n_h, n_w), what)?;
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
    /// The buffers at `lens`, zero-filled but the break rows, which hold `brk` (`out_dim` bf16
    /// values) `lens.max_tokens` times. Load-time only.
    fn new(stream: &CudaStream, lens: ScratchLens, brk: &[u16]) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
        let breaks = brk.repeat(lens.max_tokens);
        if breaks.len() != lens.breaks {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a break row of {} values, {} times, is not the {} values of the break rows",
                    brk.len(),
                    lens.max_tokens,
                    lens.breaks
                ),
            });
        }
        Ok(Scratch {
            lens,
            patches: bf16(lens.patches)?,
            cs: DeviceBuffer::zeroed(stream, lens.cs)?,
            xa: bf16(lens.stream)?,
            xb: bf16(lens.stream)?,
            h: bf16(lens.stream)?,
            qkv: bf16(lens.qkv)?,
            att: bf16(lens.att)?,
            gu: bf16(lens.gu)?,
            act: bf16(lens.act)?,
            mid: bf16(lens.mid)?,
            proj: bf16(lens.proj)?,
            rows: bf16(lens.rows)?,
            breaks: DeviceBuffer::from_host(stream, &breaks)?,
        })
    }

    /// Start an `n_h × n_w` image: refuse a grid that is not whole merge groups or is larger
    /// than the buffers, then write its RoPE table.
    fn start(
        &mut self,
        stream: &CudaStream,
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
        let table =
            RopeTable128::interleaved(n_h, n_w, merge_order(n_h, n_w, hp.merge), ROPE_THETA);
        WindowMut::<f32>::of_mut(&mut self.cs, 0, table.cs.len())?
            .copy_from_host(stream, &table.cs)?;
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
        let (dim, heads, ff) = (hp.dim, hp.n_head, hp.ff);
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: heads,
        };
        let name = |op: &str| format!("blk{b}.{op}");
        self.norm
            .enqueue(stream, &s.xa, &bw.ln1, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm1"), &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &bw.qkv_w, None),
            (n, 3 * dim, dim),
            None,
            &mut s.qkv,
        )?;
        tap(stream, taps, &name("qkv"), &s.qkv, n, 3 * dim)?;
        self.rope.enqueue(
            stream,
            RopeArgs {
                cs: &s.cs,
                n,
                row_width: lay.row_width,
                q0: lay.q0,
                k0: lay.k0,
                n_heads: heads,
                x: &mut s.qkv,
            },
        )?;
        for (op, c0) in [("qrot", lay.q0), ("krot", lay.k0)] {
            tap_cols(stream, taps, &name(op), &s.qkv, (n, 3 * dim), (c0, dim))?;
        }
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: lay,
                n,
                scale: 1.0 / (HEAD_DIM_128 as f32).sqrt(),
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
            (&s.att, &bw.o_w, None),
            (n, dim, dim),
            &mut s.xb,
        )?;
        linear(
            &self.gemm,
            stream,
            (&s.att, &bw.o_w, None),
            (n, dim, dim),
            Some(&s.xa),
            &mut s.xb,
        )?;
        tap(stream, taps, &name("resid1"), &s.xb, n, dim)?;
        self.norm
            .enqueue(stream, &s.xb, &bw.ln2, hp.eps, n, &mut s.h)?;
        tap(stream, taps, &name("norm2"), &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &bw.gu_w, None),
            (n, 2 * ff, dim),
            None,
            &mut s.gu,
        )?;
        tap(stream, taps, &name("gu"), &s.gu, n, 2 * ff)?;
        self.mlp.enqueue(stream, &s.gu, ff, n, None, &mut s.act)?;
        tap(stream, taps, &name("act"), &s.act, n, ff)?;
        side(
            &self.gemm,
            stream,
            taps,
            &name("mlp"),
            (&s.act, &bw.down_w, None),
            (n, dim, ff),
            &mut s.xa,
        )?;
        linear(
            &self.gemm,
            stream,
            (&s.act, &bw.down_w, None),
            (n, dim, ff),
            Some(&s.xb),
            &mut s.xa,
        )?;
        tap(stream, taps, &format!("blk{b}"), &s.xa, n, dim)?;
        Ok(9)
    }

    /// The projector on the last block's output in `s.xa`, for an image of `nx × ny` merged
    /// tokens: the input norm into `s.h`, the merger — `s.h` re-read as rows of `merge²·dim`
    /// values — into `s.xb`, the GELU MLP into `s.proj`, and the rows the text model reads into
    /// `s.rows`: grid row `r`'s tokens at row `r·(nx + 1)`, a break row after each but the last.
    /// Returns the rows ([`rows_of`]) and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        (nx, ny): (usize, usize),
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let (dim, out) = (hp.dim, hp.out_dim);
        let n_tok = nx * ny;
        let n = n_tok * hp.merge * hp.merge;
        self.norm
            .enqueue(stream, &s.xa, &self.w.input_norm, hp.eps, n, &mut s.h)?;
        tap(stream, taps, "merger.norm", &s.h, n, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.h, &self.w.merger_w, None),
            (n_tok, dim, dim * hp.merge * hp.merge),
            None,
            &mut s.xb,
        )?;
        tap(stream, taps, "merger", &s.xb, n_tok, dim)?;
        linear(
            &self.gemm,
            stream,
            (&s.xb, &self.w.mm1_w, None),
            (n_tok, out, dim),
            None,
            &mut s.mid,
        )?;
        tap(stream, taps, "mm1", &s.mid, n_tok, out)?;
        self.gelu.enqueue(stream, n_tok * out, &mut s.mid)?;
        tap(stream, taps, "gelu", &s.mid, n_tok, out)?;
        linear(
            &self.gemm,
            stream,
            (&s.mid, &self.w.mm2_w, None),
            (n_tok, out, out),
            None,
            &mut s.proj,
        )?;
        tap(stream, taps, "proj", &s.proj, n_tok, out)?;
        // The copy reads neither its second operand nor the gain and bias: the source and the
        // RoPE table stand in for them.
        self.ops.enqueue(
            stream,
            pointwise::Args {
                op: Op::Copy,
                x: &s.proj,
                b: &s.proj,
                gain: &s.cs,
                bias: &s.cs,
                rows: ny,
                width: nx * out,
                out_width: (nx + 1) * out,
                out_col: 0,
                scale: 1.0,
                clamp: (0.0, 0.0),
                out: &mut s.rows,
            },
        )?;
        let mut launches = 6;
        if ny > 1 {
            self.ops.enqueue(
                stream,
                pointwise::Args {
                    op: Op::Copy,
                    x: &s.breaks,
                    b: &s.breaks,
                    gain: &s.cs,
                    bias: &s.cs,
                    rows: ny - 1,
                    width: out,
                    out_width: (nx + 1) * out,
                    out_col: nx * out,
                    scale: 1.0,
                    clamp: (0.0, 0.0),
                    out: &mut s.rows,
                },
            )?;
            launches += 1;
        }
        let n_rows = rows_of(nx, ny);
        tap(stream, taps, "embd", &s.rows, n_rows, out)?;
        Ok((n_rows, launches))
    }
}

#[cfg(test)]
mod tests {
    use super::{HEAD_DIM_128, rope_rows};
    use crate::rope2d::rope_source_128;

    /// Each head's rows move inside the head by the card's RoPE order, whole rows at a time.
    #[test]
    fn the_rope_rows_are_each_heads_rows_in_the_cards_order() {
        let dim = 2 * HEAD_DIM_128;
        let m: Vec<u16> = (0..dim * dim).map(|i| (i / dim) as u16).collect();
        let p = rope_rows(&m, dim);
        for o in 0..dim {
            let want = (o / HEAD_DIM_128) * HEAD_DIM_128 + rope_source_128(o % HEAD_DIM_128);
            assert!(
                p[o * dim..(o + 1) * dim]
                    .iter()
                    .all(|&v| usize::from(v) == want),
                "row {o}"
            );
        }
    }
}
