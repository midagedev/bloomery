//! The MiniCPM-V 4.7 tower (`minicpmv4_7`) on the card: the weights of one mmproj file uploaded
//! once, the activations of the largest tile its bound allows allocated once beside them, and
//! [`Encoder::encode`], from one tile's bf16 patches (row-major, [`vision::arch::minicpmv::group`])
//! to the bf16 rows of its merged tokens. The graph is llama.cpp's `clip_graph_minicpmv4_6::build`
//! (`tools/mtmd/models/minicpmv.cpp:116-353`), whose `n_merge = 4` arm this transcribes; every op
//! reads bf16, computes in f32 and rounds its output to bf16, the contract of [`crate`].
//!
//! Per 448 slice (1024 patches), the launches in order:
//!
//! | step | launch | rows | count |
//! |---|---|---|---|
//! | patch conv | GEMM `patches · patch_embdᵀ + b`, the position rows as the residual operand | 1024 | 1 |
//! | blocks 0–6 | LayerNorm, GEMM qkv (1152 → 3456), `vis_attn_h72`, GEMM out + residual, LayerNorm, GEMM up (→ 4352), GELU (tanh), GEMM down + residual | 1024 | 7 × 8 |
//! | vit_merger | LayerNorm, GEMM qkv, `vis_attn_group`, GEMM out + residual, `vis_mean_groups`, LayerNorm (4608), GEMM up (→ 17216), GELU (tanh), GEMM down + mean residual | 1024 → 256 | 9 |
//! | blocks 7–26 | the same eight | 256 | 20 × 8 |
//! | `post_ln` | LayerNorm | 256 | 1 |
//! | final merger | LayerNorm (4608), GEMM up with the erf GELU fused, GEMM down (→ 2048) | 64 | 3 |
//!
//! That is [`Encoder::launches`]: 230 at 27 blocks (115 GEMMs, 58 LayerNorms, 27 + 1 attentions,
//! 28 GELUs, one mean). Eager, like the other towers: every grid depends on the tile's patch count.
//!
//! # Merge order
//!
//! The file's rows are row-major (`Patches`); the tower lays them out in [`merge_rows`] order so
//! that both mergers are reshapes: patch `(i, j)` of an `n_h × n_w` grid goes to row
//! `((I·n_w/4 + J)·4 + q)·4 + p` with `(I, J) = (i/4, j/4)` the 4×4 block, `q = 2·((i%4)/2) +
//! (j%4)/2` the 2×2 window in the block and `p = 2·(i%2) + j%2` the patch in the window, each
//! in raster order.
//! * The vit_merger's window `(wi, wj)` is the 4 consecutive rows `p = 0..4`, in mainline's
//!   `window_idx` order (`clip.cpp:4904-4913`: `(2wi,2wj)`, `(2wi,2wj+1)`, `(2wi+1,2wj)`,
//!   `(2wi+1,2wj+1)`); its down-sample group is the same four rows, in `vit_merger_ds_idx_0..3`
//!   order (`:4935-4938`, built by `make_ds_idx`, `:4890-4898`).
//! * The windows follow each 4×4 block together, so the 256 rows after the merger come out in the
//!   order of `merger_ds_idx_0..3` (`:4945-4953`): 4 consecutive rows are the final merger's group
//!   `p0..p3`, and the 64 output rows are mainline's raster order of the groups.
//!
//! The position rows are the table's rows `bucket_h·70 + bucket_w`, `bucket = floor(70·i/n)`
//! (`clip.cpp:4873-4885`), gathered on the host in row order and rounded to bf16 as the patch
//! GEMM's residual operand; they are re-sent only when the grid changes (the slices of an image
//! share one grid). The patch kernel is narrowed f32 → bf16 at load ([`Encoder::narrowed`]).
//!
//! The two ops that are not the other towers' are `vis_attn_group` (the vit_merger's attention
//! inside windows of four rows) and `vis_mean_groups` (the down-sample's residual, the mean of each
//! window in bf16 like the stream it joins).

use crate::attn::{Attn72Kernels, AttnArgs, HEAD_DIM_72, QkvLayout};
use crate::attn_group::{AttnGroupArgs, AttnGroupKernels};
use crate::chain::{Encoded, NoTaps, TapSink, read_bf16, read_f32, tap, up_f32};
use crate::gelu_tanh::GeluTanhKernels;
use crate::gemm_bf16::{Epilogue, GemmArgs, GemmKernels};
use crate::layer_norm::LayerNormKernels;
use crate::mean_groups::MeanGroupsKernels;
use bloomery_gpu::{DeviceTensor, GpuError, WindowMut};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::minicpmv::{Hparams, names, tensors};

mod host;
mod order;

pub use host::{
    HostFinalMerger, HostLinear, HostNorm, HostVitMerger, final_merger_f32, host_layer_norm,
    host_linear, vit_merger_f32,
};
pub use order::{
    SEPARATORS, bucket, check_grid, merge_row, merge_rows, position_index, tile_embd, window_rows,
};

const WHAT: &str = "MiniCPM-V tower load";

/// Rows of a window and of a down-sample group, at both mergers.
pub const GROUP: usize = 4;
/// Patches in one merged token: both mergers fold 2×2.
pub const PATCHES_PER_TOKEN: usize = GROUP * GROUP;
/// Patches per side of the block a vit_merger window and a final-merger group span together.
pub const BLOCK_SIDE: usize = 4;
/// The learned position table's side: `v.position_embd.weight` has `POS_SIDE²` rows.
pub const POS_SIDE: usize = 70;
/// The tensors the load narrows (f32 on disk, bf16 on the card).
const NARROWED: [&str; 1] = ["v.patch_embd.weight"];

// ------------------------------------------------------------------ bytes

/// The columns `ffn_up` has after its zero padding: a multiple of the GEMM's column tile.
fn ff_padded(hp: &Hparams) -> usize {
    hp.ff.next_multiple_of(crate::gemm_bf16::TILE_N)
}

/// Bytes of weights the card holds for a file with these hyperparameters: bf16 matrices (the
/// MLP of each block padded to [`ff_padded`]) and the patch kernel narrowed to bf16, every vector
/// f32. The position table and the separator rows stay on the host.
#[must_use]
pub fn weight_bytes_of(hp: &Hparams) -> u64 {
    let d = |n: usize| n as u64;
    let (dim, ff, out) = (d(hp.dim), d(ff_padded(hp)), d(hp.out_dim));
    let merged = d(hp.merged_dim());
    let ds_ff = d(4 * hp.ff);
    let k_patch = d(3 * hp.patch * hp.patch);
    let attn_mats = 4 * dim * dim;
    let block_mats = attn_mats + 2 * ff * dim;
    let attn_vecs = 3 * dim + dim;
    let block_vecs = attn_vecs + 4 * dim + ff + dim;
    let merger_mats = attn_mats + (merged + dim) * ds_ff;
    let merger_vecs = attn_vecs + 2 * dim + 2 * merged + ds_ff + dim;
    2 * (d(hp.n_layer) * block_mats + merger_mats + dim * k_patch + merged * merged + merged * out)
        + 4 * (d(hp.n_layer) * block_vecs + merger_vecs + dim + 2 * dim + 2 * merged + merged + out)
}

/// The activation buffers' lengths for a tile of at most `max_patches` patches, in bf16 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lens {
    patches: usize,
    pos: usize,
    stream: usize,
    qkv: usize,
    att: usize,
    up: usize,
    mean: usize,
    rows: usize,
}

impl Lens {
    fn of(hp: &Hparams, max_patches: usize) -> Lens {
        let n = max_patches;
        Lens {
            patches: n * 3 * hp.patch * hp.patch,
            pos: n * hp.dim,
            stream: n * hp.dim,
            qkv: n * 3 * hp.dim,
            att: n * hp.dim,
            // Also the vit_merger's hidden rows (n/4 of 4·ff, which is n·ff) and the final
            // merger's up rows (n/16 of 4608).
            up: n * ff_padded(hp),
            mean: n / GROUP * hp.dim,
            rows: n / (GROUP * GROUP) * hp.out_dim,
        }
    }

    /// Bytes of every buffer: all bf16, the three streams (`xa`, `xb`, `h`) counted.
    fn bytes(&self) -> u64 {
        let bf16 = self.patches
            + self.pos
            + 3 * self.stream
            + self.qkv
            + self.att
            + self.up
            + self.mean
            + self.rows;
        2 * bf16 as u64
    }
}

/// Bytes of activations the card holds for a tile of at most `max_patches` patches.
#[must_use]
pub fn scratch_bytes_of(hp: &Hparams, max_patches: usize) -> u64 {
    Lens::of(hp, max_patches).bytes()
}

// ------------------------------------------------------------------ weights

/// A norm's gain and bias.
type Norm = (DeviceBuffer<f32>, DeviceBuffer<f32>);

/// A fused q/k/v projection and the output projection of one attention, on the card.
struct AttnWeights {
    qkv_w: DeviceBuffer<u16>,
    qkv_b: DeviceBuffer<f32>,
    o_w: DeviceBuffer<u16>,
    o_b: DeviceBuffer<f32>,
}

/// One ViT block's weights.
struct BlockWeights {
    ln1: Norm,
    attn: AttnWeights,
    ln2: Norm,
    /// `ffn_up`'s rows, zero-padded to [`ff_padded`].
    up_w: DeviceBuffer<u16>,
    up_b: DeviceBuffer<f32>,
    /// `ffn_down`'s rows, each zero-padded to [`ff_padded`] columns.
    down_w: DeviceBuffer<u16>,
    down_b: DeviceBuffer<f32>,
}

/// A linear layer: `[out, in]` bf16 rows and the bias.
struct Linear {
    w: DeviceBuffer<u16>,
    b: DeviceBuffer<f32>,
}

/// The ViT merger after block 6.
struct VitMerger {
    ln1: Norm,
    attn: AttnWeights,
    ds_ln: Norm,
    up: Linear,
    down: Linear,
}

/// Every weight of the tower on the card.
struct Weights {
    patch_w: DeviceBuffer<u16>,
    patch_b: DeviceBuffer<f32>,
    blocks: Vec<BlockWeights>,
    merger: VitMerger,
    post_ln: Norm,
    mm_norm: Norm,
    mm_up: Linear,
    mm_down: Linear,
}

impl Weights {
    /// Bytes uploaded: two per bf16, four per f32.
    fn uploaded(&self) -> u64 {
        let norm = |p: &Norm| 4 * (p.0.len() + p.1.len());
        let lin = |l: &Linear| 2 * l.w.len() + 4 * l.b.len();
        let attn =
            |a: &AttnWeights| 2 * (a.qkv_w.len() + a.o_w.len()) + 4 * (a.qkv_b.len() + a.o_b.len());
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                norm(&b.ln1)
                    + norm(&b.ln2)
                    + attn(&b.attn)
                    + 2 * (b.up_w.len() + b.down_w.len())
                    + 4 * (b.up_b.len() + b.down_b.len())
            })
            .sum();
        let m = &self.merger;
        (blocks
            + norm(&m.ln1)
            + attn(&m.attn)
            + norm(&m.ds_ln)
            + lin(&m.up)
            + lin(&m.down)
            + 2 * self.patch_w.len()
            + 4 * self.patch_b.len()
            + norm(&self.post_ln)
            + norm(&self.mm_norm)
            + lin(&self.mm_up)
            + lin(&self.mm_down)) as u64
    }
}

/// What an encode reads and never writes: the hyperparameters, the kernels, the weights and the
/// position rows (bf16, on the host: a tile gathers its rows from them).
struct Chain {
    hp: Hparams,
    gemm: GemmKernels,
    norm: LayerNormKernels,
    gelu: GeluTanhKernels,
    attn: Attn72Kernels,
    window: AttnGroupKernels,
    mean: MeanGroupsKernels,
    w: Weights,
    pos_table: Vec<u16>,
}

/// The activations of the largest tile the bound allows. A tile of `n` patches uses the first
/// `n` rows of each buffer; every launch reads only rows that a copy or an earlier launch of the
/// same tile wrote.
struct Scratch {
    lens: Lens,
    patches: DeviceBuffer<u16>,
    pos: DeviceBuffer<u16>,
    xa: DeviceBuffer<u16>,
    xb: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    qkv: DeviceBuffer<u16>,
    att: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    mean: DeviceBuffer<u16>,
    rows: DeviceBuffer<u16>,
}

/// The host side of the last grid: its row order, and whether `Scratch::pos` holds its rows.
struct Staging {
    grid: Option<(usize, usize)>,
    order: Vec<usize>,
    patches: Vec<u16>,
}

/// The loaded tower: the chain of one file, the activations of the largest tile, its bytes on
/// the card, and the stream every call runs on. A call takes `&mut self`: one tile at a time owns
/// the activations, and the rows it returns borrow them until the next call.
pub struct Encoder {
    chain: Chain,
    s: Scratch,
    host: Staging,
    separators: Vec<f32>,
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
    /// a `minicpmv4_7` encoder's), upload every weight, held to [`weight_bytes_of`], and allocate
    /// the activations for tiles of at most `max_patches` patches (a multiple of 16: whole
    /// merged tokens). Every later call runs on `stream`. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &Gguf,
        max_patches: usize,
    ) -> Result<Encoder, GpuError> {
        let hp = Hparams::read(file).map_err(|e| err(WHAT, e))?;
        tensors::check(
            &hp,
            file.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .map_err(|e| err(WHAT, e))?;
        if hp.dim / hp.n_head != HEAD_DIM_72
            || hp.merge != BLOCK_SIDE
            || hp.out_dim % crate::gemm_bf16::TILE_N != 0
            || max_patches == 0
            || !max_patches.is_multiple_of(GROUP * GROUP)
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "heads of {} values, merge {}, out {}, tiles of {max_patches} patches; the tower runs heads of {HEAD_DIM_72}, merge {BLOCK_SIDE}, an output width that is a multiple of {} and tiles of a multiple of {} patches",
                    hp.dim / hp.n_head,
                    hp.merge,
                    hp.out_dim,
                    crate::gemm_bf16::TILE_N,
                    GROUP * GROUP
                ),
            });
        }
        let ff = ff_padded(&hp);
        let f32s = |name: String| up_f32(stream, file, &name, WHAT);
        let bf16 = |name: String| -> Result<DeviceBuffer<u16>, GpuError> {
            Ok(DeviceBuffer::from_host(
                stream,
                &read_bf16(file, &name, WHAT)?,
            )?)
        };
        let norm = |w: String, b: String| -> Result<Norm, GpuError> { Ok((f32s(w)?, f32s(b)?)) };
        let linear = |w: String, b: String| -> Result<Linear, GpuError> {
            Ok(Linear {
                w: bf16(w)?,
                b: f32s(b)?,
            })
        };
        // q, k and v as one `[3·dim, dim]` matrix: the output columns are the fused layout.
        let attn = |q: [String; 2], k: [String; 2], v: [String; 2], o: [String; 2]| {
            let mut w = Vec::new();
            let mut b = Vec::new();
            for [wn, bn] in [q, k, v] {
                w.extend(read_bf16(file, &wn, WHAT)?);
                b.extend(read_f32(file, &bn, WHAT)?);
            }
            let [ow, ob] = o;
            Ok::<_, GpuError>(AttnWeights {
                qkv_w: DeviceBuffer::from_host(stream, &w)?,
                qkv_b: DeviceBuffer::from_host(stream, &b)?,
                o_w: bf16(ow)?,
                o_b: f32s(ob)?,
            })
        };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for b in 0..hp.n_layer {
            let mut up_w = read_bf16(file, &names::ffn_up_weight(b), WHAT)?;
            up_w.resize(ff * hp.dim, 0);
            let mut up_b = read_f32(file, &names::ffn_up_bias(b), WHAT)?;
            up_b.resize(ff, 0.0);
            let down_w: Vec<u16> = read_bf16(file, &names::ffn_down_weight(b), WHAT)?
                .chunks_exact(hp.ff)
                .flat_map(|row| {
                    row.iter()
                        .copied()
                        .chain(std::iter::repeat_n(0, ff - hp.ff))
                })
                .collect();
            blocks.push(BlockWeights {
                ln1: norm(names::ln1_weight(b), names::ln1_bias(b))?,
                attn: attn(
                    [names::attn_q_weight(b), names::attn_q_bias(b)],
                    [names::attn_k_weight(b), names::attn_k_bias(b)],
                    [names::attn_v_weight(b), names::attn_v_bias(b)],
                    [names::attn_out_weight(b), names::attn_out_bias(b)],
                )?,
                ln2: norm(names::ln2_weight(b), names::ln2_bias(b))?,
                up_w: DeviceBuffer::from_host(stream, &up_w)?,
                up_b: DeviceBuffer::from_host(stream, &up_b)?,
                down_w: DeviceBuffer::from_host(stream, &down_w)?,
                down_b: f32s(names::ffn_down_bias(b))?,
            });
        }
        let vm = |stem: &str| (names::vit_merger_weight(stem), names::vit_merger_bias(stem));
        let merger = VitMerger {
            ln1: norm(vm("ln1").0, vm("ln1").1)?,
            attn: attn(
                [vm("attn_q").0, vm("attn_q").1],
                [vm("attn_k").0, vm("attn_k").1],
                [vm("attn_v").0, vm("attn_v").1],
                [vm("attn_out").0, vm("attn_out").1],
            )?,
            ds_ln: norm(vm("ds_ln").0, vm("ds_ln").1)?,
            up: linear(vm("ds_ffn_up").0, vm("ds_ffn_up").1)?,
            down: linear(vm("ds_ffn_down").0, vm("ds_ffn_down").1)?,
        };
        // The patch kernel is f32 on disk; the GEMM reads bf16, so it is rounded (to nearest,
        // ties to even) once here. This is the one narrowed tensor (`NARROWED`).
        let patch_w: Vec<u16> = read_f32(file, &names::patch_embd_weight(), WHAT)?
            .into_iter()
            .map(crate::f32_bf16)
            .collect();
        let w = Weights {
            patch_w: DeviceBuffer::from_host(stream, &patch_w)?,
            patch_b: f32s(names::patch_embd_bias())?,
            blocks,
            merger,
            post_ln: norm(names::post_ln_weight(), names::post_ln_bias())?,
            mm_norm: norm(names::mm_weight("input_norm"), names::mm_bias("input_norm"))?,
            mm_up: linear(names::mm_weight("up"), names::mm_bias("up"))?,
            mm_down: linear(names::mm_weight("down"), names::mm_bias("down"))?,
        };
        let (uploaded, figure) = (w.uploaded(), weight_bytes_of(&hp));
        if uploaded != figure {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "uploaded {uploaded} B of weights; the figure of the file's hyperparameters (weight_bytes_of) is {figure} B"
                ),
            });
        }
        let pos_table: Vec<u16> = read_f32(file, &names::position_embd(), WHAT)?
            .into_iter()
            .map(crate::f32_bf16)
            .collect();
        let separators = read_f32(file, &names::tok_embd_sep(), WHAT)?;
        let lens = Lens::of(&hp, max_patches);
        let s = Scratch::new(stream, lens)?;
        stream.synchronize()?;
        Ok(Encoder {
            chain: Chain {
                gemm: GemmKernels::load(ctx, stream)?,
                norm: LayerNormKernels::load(ctx)?,
                gelu: GeluTanhKernels::load(ctx)?,
                attn: Attn72Kernels::load(ctx)?,
                window: AttnGroupKernels::load(ctx)?,
                mean: MeanGroupsKernels::load(ctx)?,
                hp,
                w,
                pos_table,
            },
            s,
            host: Staging {
                grid: None,
                order: Vec::new(),
                patches: Vec::new(),
            },
            separators,
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

    /// Bytes of weights on the card: what the load uploaded, held to [`weight_bytes_of`].
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Bytes of activations on the card: the buffers of the largest tile the bound allows.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// The tensors the load narrowed: f32 on disk, bf16 on the card.
    #[must_use]
    pub fn narrowed(&self) -> &'static [&'static str] {
        &NARROWED
    }

    /// The four separator rows of `v.tok_embd_sep` (`</image>`, `<slice>`, `</slice>`, `\n`),
    /// `out_dim` f32 values each, as the file holds them: mainline feeds the text model these rows
    /// after a tile's merged rows, not the tokens' embeddings.
    #[must_use]
    pub fn separators(&self) -> &[f32] {
        &self.separators
    }

    /// The tower's one print line: bytes, launches per 448 slice and what was narrowed.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "minicpmv4_7 tower: weights {} B, scratch {} B, {} launches a 448 slice; narrowed: {}",
            self.weight_bytes,
            self.scratch_bytes,
            Encoder::launches(self.chain.hp.n_layer),
            NARROWED.join(", ")
        )
    }

    /// Chain launches of one encode at `n_layer` blocks: the patch GEMM, eight per block, the
    /// vit_merger's nine, the final norm and the final merger's three.
    #[must_use]
    pub const fn launches(n_layer: usize) -> usize {
        1 + 8 * n_layer + 9 + 1 + 3
    }

    /// Encode one tile's patches. Eager: copies the patches (in merge order) and, for a new grid,
    /// the position rows into the tower's buffers, enqueues the chain on its stream and returns
    /// the rows without waiting for it; read them on that stream before the next call.
    pub fn encode(&mut self, patches: &Patches) -> Result<Encoded<'_>, GpuError> {
        self.encode_tapped(patches, &mut NoTaps)
    }

    /// [`Encoder::encode`] handing the named intermediate tensors to `taps`: `embed`, `blk{b}`
    /// (a block's output), `vit_merger.attn` (the window attention's residual stream),
    /// `vit_merger.mean`, `vit_merger` (its output rows), `vit` (`post_ln`), `merger.norm`,
    /// `merger.h`.
    pub fn encode_tapped(
        &mut self,
        patches: &Patches,
        taps: &mut dyn TapSink,
    ) -> Result<Encoded<'_>, GpuError> {
        let what = "MiniCPM-V tower encode";
        let (chain, s, host, stream) = (&self.chain, &mut self.s, &mut self.host, &*self.stream);
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
        s.check(hp, (n_h, n_w), what)?;
        if host.grid != Some((n_h, n_w)) {
            host.order = merge_rows(n_h, n_w);
            let index = position_index(n_h, n_w);
            let mut rows = Vec::with_capacity(n * hp.dim);
            for &raster in &host.order {
                let t = index[raster];
                rows.extend_from_slice(&chain.pos_table[t * hp.dim..(t + 1) * hp.dim]);
            }
            WindowMut::<u16>::of_mut(&mut s.pos, 0, n * hp.dim)?.copy_from_host(stream, &rows)?;
            host.grid = Some((n_h, n_w));
        }
        host.patches.clear();
        for &raster in &host.order {
            host.patches
                .extend_from_slice(&patches.bf16[raster * k_patch..(raster + 1) * k_patch]);
        }
        WindowMut::<u16>::of_mut(&mut s.patches, 0, n * k_patch)?
            .copy_from_host(stream, &host.patches)?;
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
        let mut launches = 1;
        for b in 0..=hp.merger_after {
            launches += chain.block(stream, b, n, s, taps)?;
        }
        launches += chain.vit_merger(stream, n, s, taps)?;
        let n2 = n / GROUP;
        for b in hp.merger_after + 1..hp.n_layer {
            launches += chain.block(stream, b, n2, s, taps)?;
        }
        let (n_tok, tail) = chain.tail(stream, n2, s, taps)?;
        Ok(Encoded {
            rows: DeviceTensor::window_of(&s.rows, 0, n_tok, hp.out_dim)?,
            launches: launches + tail,
        })
    }
}

impl Scratch {
    /// The buffers at `lens`, zero-filled. Load-time only.
    fn new(stream: &CudaStream, lens: Lens) -> Result<Scratch, GpuError> {
        let bf16 = |n: usize| DeviceBuffer::<u16>::zeroed(stream, n);
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
            mean: bf16(lens.mean)?,
            rows: bf16(lens.rows)?,
        })
    }

    /// Refuse a grid that is not whole 4×4 blocks or is larger than the buffers.
    fn check(
        &self,
        hp: &Hparams,
        (n_h, n_w): (usize, usize),
        what: &'static str,
    ) -> Result<(), GpuError> {
        check_grid(n_h, n_w).map_err(|detail| GpuError::Shape { what, detail })?;
        if n_h * n_w * hp.dim > self.lens.stream {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "a {n_h}x{n_w} grid ({} patches) does not fit the tower's buffers, sized for {} patches",
                    n_h * n_w,
                    self.lens.stream / hp.dim
                ),
            });
        }
        Ok(())
    }
}

impl Chain {
    /// Block `b` on the first `n` rows of `s.xa`, its output back in `s.xa`; returns the launches.
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
        let scale = 1.0 / ((dim / heads) as f32).sqrt();
        self.norm
            .enqueue(stream, &s.xa, (&bw.ln1.0, &bw.ln1.1), hp.eps, n, &mut s.h)?;
        self.project_qkv(stream, &bw.attn, n, s)?;
        self.attn.enqueue(
            stream,
            AttnArgs {
                qkv: &s.qkv,
                layout: qkv_layout(hp),
                n,
                scale,
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        self.project_out(stream, &bw.attn, n, s)?;
        self.norm
            .enqueue(stream, &s.xb, (&bw.ln2.0, &bw.ln2.1), hp.eps, n, &mut s.h)?;
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
        self.gelu.enqueue(stream, n * ff, &mut s.up)?;
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

    /// `s.h · wqkvᵀ + b` of `n` rows into `s.qkv`.
    fn project_qkv(
        &self,
        stream: &CudaStream,
        a: &AttnWeights,
        n: usize,
        s: &mut Scratch,
    ) -> Result<(), GpuError> {
        let dim = self.hp.dim;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &a.qkv_w,
                bias: Some(&a.qkv_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n,
                n: 3 * dim,
                k: dim,
                c: &mut s.qkv,
            },
        )
    }

    /// `s.att · woᵀ + b` plus the residual `s.xa` of `n` rows into `s.xb`.
    fn project_out(
        &self,
        stream: &CudaStream,
        a: &AttnWeights,
        n: usize,
        s: &mut Scratch,
    ) -> Result<(), GpuError> {
        let dim = self.hp.dim;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.att,
                b: &a.o_w,
                bias: Some(&a.o_b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.xa),
                m: n,
                n: dim,
                k: dim,
                c: &mut s.xb,
            },
        )
    }

    /// The vit_merger on the first `n` rows of `s.xa` (merge order): window attention with its
    /// residual into `s.xb`, then the 2×2 down-sample (four consecutive rows read as one of
    /// `4·dim`) into the first `n / 4` rows of `s.xa`; returns the launches.
    fn vit_merger(
        &self,
        stream: &CudaStream,
        n: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<usize, GpuError> {
        let hp = &self.hp;
        let m = &self.w.merger;
        let (dim, merged, ds_ff) = (hp.dim, hp.merged_dim(), 4 * hp.ff);
        let scale = 1.0 / ((dim / hp.n_head) as f32).sqrt();
        self.norm
            .enqueue(stream, &s.xa, (&m.ln1.0, &m.ln1.1), hp.eps, n, &mut s.h)?;
        self.project_qkv(stream, &m.attn, n, s)?;
        self.window.enqueue(
            stream,
            AttnGroupArgs {
                qkv: &s.qkv,
                layout: qkv_layout(hp),
                n,
                group: GROUP,
                scale,
                out_width: dim,
                out: &mut s.att,
            },
        )?;
        self.project_out(stream, &m.attn, n, s)?;
        tap(stream, taps, "vit_merger.attn", &s.xb, n, dim)?;
        self.mean
            .enqueue(stream, n, GROUP, dim, &s.xb, &mut s.mean)?;
        tap(stream, taps, "vit_merger.mean", &s.mean, n / GROUP, dim)?;
        self.norm.enqueue(
            stream,
            &s.xb,
            (&m.ds_ln.0, &m.ds_ln.1),
            hp.eps,
            n / GROUP,
            &mut s.h,
        )?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.h,
                b: &m.up.w,
                bias: Some(&m.up.b),
                epilogue: Epilogue::None,
                resid: None,
                m: n / GROUP,
                n: ds_ff,
                k: merged,
                c: &mut s.up,
            },
        )?;
        self.gelu.enqueue(stream, n / GROUP * ds_ff, &mut s.up)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.up,
                b: &m.down.w,
                bias: Some(&m.down.b),
                epilogue: Epilogue::Residual,
                resid: Some(&s.mean),
                m: n / GROUP,
                n: dim,
                k: ds_ff,
                c: &mut s.xa,
            },
        )?;
        tap(stream, taps, "vit_merger", &s.xa, n / GROUP, dim)?;
        Ok(9)
    }

    /// The chain after the last block on `n` rows of `s.xa`: `post_ln` into `s.h`, then the final
    /// merger (four consecutive rows read as one of `4·dim`) into `s.rows`; returns the merged
    /// tokens and the launches.
    fn tail(
        &self,
        stream: &CudaStream,
        n: usize,
        s: &mut Scratch,
        taps: &mut dyn TapSink,
    ) -> Result<(usize, usize), GpuError> {
        let hp = &self.hp;
        let (dim, merged) = (hp.dim, hp.merged_dim());
        let n_tok = n / GROUP;
        self.norm.enqueue(
            stream,
            &s.xa,
            (&self.w.post_ln.0, &self.w.post_ln.1),
            hp.eps,
            n,
            &mut s.h,
        )?;
        tap(stream, taps, "vit", &s.h, n, dim)?;
        self.norm.enqueue(
            stream,
            &s.h,
            (&self.w.mm_norm.0, &self.w.mm_norm.1),
            hp.eps,
            n_tok,
            &mut s.xb,
        )?;
        tap(stream, taps, "merger.norm", &s.xb, n_tok, merged)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.xb,
                b: &self.w.mm_up.w,
                bias: Some(&self.w.mm_up.b),
                epilogue: Epilogue::Gelu,
                resid: None,
                m: n_tok,
                n: merged,
                k: merged,
                c: &mut s.up,
            },
        )?;
        tap(stream, taps, "merger.h", &s.up, n_tok, merged)?;
        self.gemm.enqueue(
            stream,
            GemmArgs {
                a: &s.up,
                b: &self.w.mm_down.w,
                bias: Some(&self.w.mm_down.b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: hp.out_dim,
                k: merged,
                c: &mut s.rows,
            },
        )?;
        Ok((n_tok, 4))
    }
}

/// The fused q/k/v rows of a block: `[q | k | v]`, `dim` columns each.
fn qkv_layout(hp: &Hparams) -> QkvLayout {
    QkvLayout {
        row_width: 3 * hp.dim,
        q0: 0,
        k0: hp.dim,
        v0: 2 * hp.dim,
        n_heads: hp.n_head,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp() -> Hparams {
        Hparams {
            n_layer: 27,
            dim: 1152,
            n_head: 16,
            ff: 4304,
            patch: 14,
            slice_size: 448,
            merge: 4,
            max_slices: 9,
            merger_after: 6,
            out_dim: 2048,
            eps: 1e-6,
        }
    }

    /// The tower's bytes are the card figure a plan reserves for it (`vision::arch::minicpmv::
    /// card`), whole: the weights of the file's hyperparameters and the activations of tiles of
    /// 16, 1024 and 1120 patches (the 1000×700 image's slices), buffer lengths included.
    #[test]
    fn the_tower_holds_the_card_figure() {
        use vision::arch::minicpmv::card;
        let hp = hp();
        for n in [16, 1024, 1120] {
            let figure = card::of(&hp, n);
            assert_eq!(weight_bytes_of(&hp), figure.weights);
            assert_eq!(scratch_bytes_of(&hp, n), figure.scratch, "{n} patches");
            let (lens, planned) = (Lens::of(&hp, n), card::ScratchLens::of(&hp, n));
            assert_eq!(
                [
                    lens.patches,
                    lens.pos,
                    lens.stream,
                    lens.qkv,
                    lens.att,
                    lens.up
                ],
                [
                    planned.patches,
                    planned.pos,
                    planned.xa,
                    planned.qkv,
                    planned.attn,
                    planned.up
                ]
            );
            assert_eq!([lens.mean, lens.rows], [planned.mean, planned.out]);
            assert_eq!(lens.stream, planned.xb);
            assert_eq!(lens.stream, planned.h);
        }
        let lens = Lens::of(&hp, 1024);
        assert!(256 * 4 * hp.ff <= lens.up && 64 * hp.merged_dim() <= lens.up);
        assert_eq!(Encoder::launches(27), 230);
    }
}
