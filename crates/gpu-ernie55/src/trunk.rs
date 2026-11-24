//! ERNIE 5.5's trunk: what stands around every sub-layer of the layer walk. The
//! four hyper-connection streams — the embedding row repeated into them, a site's
//! mix of them and the fold its sub-layer reads, the post that writes the
//! sub-layer's output back — the norm a sub-layer's input takes, and the dense SiTU
//! block of the leading layers. Host code over `bloomery-gpu`'s kernels; the names
//! are `model::arch::ernie55`'s, and the description's facts come in as [`Facts`].
//!
//! A walk: [`Trunk::embed`] lays the tokens' embedding rows into stream buffer 0
//! (the host reads them from the file, each repeated into the four streams); then
//! per layer, for each sub-layer (the attention, then the feed-forward):
//! [`Trunk::enqueue_pre`] — the mix (`hc_pre_f32_rmsadd`: the site's F32 `hc_fn`
//! over the streams, the RMS factor `1 / (sqrt(mean x²) + eps)`, `pre`, `post` and
//! the Sinkhorn `comb`), the fold by that `pre` (`ds41_hc_fold`, the sub-layer's
//! input) and its norm — the sub-layer's own launches, reading [`Trunk::cur`] and
//! writing an output `[t, hidden]`; then [`Trunk::enqueue_post`] — `ds41_hc_post`
//! into the other stream buffer. A dense layer's block is [`Trunk::enqueue_dense`].
//! The two stream buffers alternate, so the walk of an even number of sites ends in
//! buffer 0. Every enqueue is allocation-free and asynchronous: a captured step
//! records the same launches.
//!
//! What the file forces here, each named where it is used: the RMS factor outside
//! the root; the fold by the site's own `pre` (the graph folds a sub-layer's input
//! by its own mix, where V4.1 lags that fold by one sub-layer); the F32 `hc_fn`;
//! the embedding repeated into four streams; the dense block's FP8 128-block
//! projections and its SiTU rule. The post and the fold are V4.1's common entries,
//! their fused multiply-adds where ik's build puts them.
//!
//! The importance collector reads the dense block's three projections as every
//! by-name site does (`site::resident_gemv_cols`, `site::resident_gemm`); the
//! hyper-connection sites feed it nothing: `hc::feed_hc_fn` normalizes by the V4
//! rule.

use bloomery_gpu::gemm::{Gemm32Kernels, GemmKernels, GemmRoute};
use bloomery_gpu::hc::{HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcPostArgs, HcPreScratch};
use bloomery_gpu::hc_f32::{HcF32Args, HcF32Params};
use bloomery_gpu::hc_f32_pieces::{HcF32PiecesKernels, LEAD};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::kquant::act::SITU_LINEAR_BETA;
use bloomery_gpu::site::{self, SiteTy, WideIn, WideKernels};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Gpu, GpuError, WindowMut, col_group_count};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::names::{attn_norm, ffn_norm};
use model::arch::names::{Sub, ffn_down, ffn_gate, ffn_up, hc_base, hc_fn, hc_scale, token_embd};
use model::embed::EmbedTable;
use std::sync::Arc;

/// What the refusals of this file name.
const WHAT: &str = "ernie55::Trunk";

/// Launches of one site's pre: the mix, the fold, the norm.
pub const PRE_LAUNCHES: usize = 3;
/// Launches of one site's post.
pub const POST_LAUNCHES: usize = 1;
/// Launches of a dense block over at most [`GEMV_COLS`] rows: the gate, the up,
/// the SiTU, the down. More rows add the route table's fill.
pub const DENSE_LAUNCHES: usize = 4;
/// The most rows a dense block projects through the m-column gemv; more run the
/// wide GEMM over a dense route table.
const GEMV_COLS: usize = 8;
/// The dense projections' families the by-name arm reads here: the checkpoint's
/// dense blocks are F8_E4M3 with an f32 scale a 128-value block of each row.
const DENSE_TYPES: &[SiteTy] = &[SiteTy::Fp8B128];

/// What the trunk reads of the description: the widths and rules its sites run at.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    /// Values of a row (`embedding_length`).
    pub hidden: usize,
    /// The trunk's layers (`block_count`).
    pub layers: usize,
    /// `hyper_connection.epsilon`.
    pub hc_eps: f32,
    /// `hyper_connection.sinkhorn_iterations`.
    pub sinkhorn_iters: u32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// The dense feed-forward blocks: `(layer, feed_forward_length)`, in layer order.
    pub dense: Vec<(usize, usize)>,
    /// `activation.situ_beta` of the dense blocks; the linear beta is the device
    /// rule's ([`SITU_LINEAR_BETA`]), which the rule refuses another value of.
    pub situ_beta: f32,
}

impl Facts {
    /// The description's facts. Refused by name: a stream count the common
    /// hyper-connection kernels do not hold, and a Sinkhorn count past the kernel's
    /// u32.
    pub fn of(hp: &Hparams) -> Result<Facts, GpuError> {
        if hp.hc.streams != HC_STREAMS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "hyper_connection.count is {}; the common hyper-connection kernels hold \
                     {HC_STREAMS} streams",
                    hp.hc.streams
                ),
            ));
        }
        let sinkhorn_iters = u32::try_from(hp.hc.sinkhorn_iters).map_err(|_| {
            GpuError::shape(
                WHAT,
                format!(
                    "hyper_connection.sinkhorn_iterations is {}; it does not fit the kernel's u32",
                    hp.hc.sinkhorn_iters
                ),
            )
        })?;
        Ok(Facts {
            hidden: hp.n_embd,
            layers: hp.n_layer,
            hc_eps: hp.hc.eps,
            sinkhorn_iters,
            rms_eps: hp.rms_eps,
            dense: (0..hp.n_layer)
                .filter(|&l| hp.dense(l))
                .map(|l| (l, hp.dense_ff))
                .collect(),
            situ_beta: hp.situ.gate_beta,
        })
    }
}

/// One sub-layer's mix parameters by name: the site is held to its resident form
/// at load ([`Site::load`]) and its [`HcF32Params`] borrow the weights at each
/// enqueue.
struct Site {
    /// `blk.{l}.hc_{attn,ffn}_fn.weight`: the mix's projection of the streams.
    proj: String,
    /// `blk.{l}.hc_{attn,ffn}_scale.weight`: the three affine scales.
    scale: String,
    /// `blk.{l}.hc_{attn,ffn}_base.weight`: the [`HC_MIX`] affine offsets.
    base: String,
}

impl Site {
    /// Layer `layer`'s `sub` site: its projection [`HC_MIX`] F32 rows of `k`
    /// values (the site runs the F32 piece entry), its scale three values and its
    /// base [`HC_MIX`]; anything else is refused by name.
    fn load(w: &Weights, (layer, k): (usize, usize), sub: Sub) -> Result<Site, GpuError> {
        let (proj, scale, base) = (hc_fn(layer, sub), hc_scale(layer, sub), hc_base(layer, sub));
        match w.resident(WHAT, &proj)? {
            DevWeight::F32 { w, .. } if (w.rows(), w.cols()) == (HC_MIX, k) => {}
            DevWeight::F32 { w, .. } => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{proj} is {} rows of {} values; the site takes {HC_MIX} of {k}",
                        w.rows(),
                        w.cols()
                    ),
                ));
            }
            _ => {
                return Err(GpuError::tensor(
                    WHAT,
                    proj,
                    "F32 rows (the site runs the F32 piece entry)",
                ));
            }
        }
        let scale = site::vector(w, WHAT, scale, 3)?;
        let base = site::vector(w, WHAT, base, HC_MIX)?;
        Ok(Site { proj, scale, base })
    }

    /// The site's parameters over the weights, at `(hc epsilon, Sinkhorn iterations)`.
    fn params<'w>(
        &self,
        w: &'w Weights,
        (eps, iters): (f32, u32),
    ) -> Result<HcF32Params<'w>, GpuError> {
        let DevWeight::F32 { w: proj, .. } = w.resident(WHAT, &self.proj)? else {
            return Err(GpuError::tensor(
                WHAT,
                self.proj.clone(),
                "F32 rows (the site runs the F32 piece entry)",
            ));
        };
        Ok(HcF32Params {
            w: proj,
            scale: w.f32_buf(WHAT, &self.scale)?,
            base: w.f32_buf(WHAT, &self.base)?,
            eps,
            iters,
        })
    }
}

/// A dense layer's SiTU block: its three projections by name and the buffers the
/// gate, the up and the SiTU write.
struct Dense {
    layer: usize,
    /// The block's width: the gate's and the up's rows.
    ff: usize,
    gate: String,
    up: String,
    down: String,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    /// The wide arm's one-expert table: a column a token.
    route: GemmRoute,
}

impl Dense {
    /// Layer `layer`'s block of width `ff` for up to `rows` tokens: the gate and
    /// the up `ff` rows of `hidden`, the down `hidden` rows of `ff`, each of a
    /// family [`DENSE_TYPES`] reads; anything else is refused by name.
    fn load(
        stream: &CudaStream,
        w: &Weights,
        (layer, ff, hidden, rows): (usize, usize, usize, usize),
    ) -> Result<Dense, GpuError> {
        let (gate, up, down) = (ffn_gate(layer), ffn_up(layer), ffn_down(layer));
        for (name, shape) in [
            (&gate, (ff, hidden)),
            (&up, (ff, hidden)),
            (&down, (hidden, ff)),
        ] {
            site::resident_site(w, WHAT, name, shape, DENSE_TYPES)?;
        }
        let z = |len: usize| DeviceBuffer::zeroed(stream, len.max(1));
        Ok(Dense {
            layer,
            ff,
            gate,
            up,
            down,
            g: z(rows * ff)?,
            u: z(rows * ff)?,
            a: z(rows * ff)?,
            route: GemmRoute::new(stream, rows, 1)?,
        })
    }
}

/// The kernels a dense block's wide arm launches.
struct Wide {
    gemm: GemmKernels,
    g32: Gemm32Kernels,
}

/// One dense projection by name: the site, the token-major rows it reads and
/// where its token-major rows land.
struct DenseSite<'a> {
    name: &'a str,
    x: &'a DeviceBuffer<f32>,
    y: &'a mut DeviceBuffer<f32>,
}

/// `y = W · x` of the dense site `s` over `t` tokens: at most [`GEMV_COLS`]
/// tokens the m-column gemv of the weight's resident form; more the wide GEMM on
/// the block's dense route table `route` ([`site::resident_gemm`]), which the
/// caller filled for the `t` tokens.
fn project(
    gpu: &Gpu,
    w: &Weights,
    s: DenseSite<'_>,
    t: usize,
    (wide, route): (Option<&Wide>, &GemmRoute),
) -> Result<(), GpuError> {
    const PROJECT: &str = "ernie55::dense";
    if t <= GEMV_COLS {
        return site::resident_gemv_cols(gpu, None, (w, s.name), s.x, t, s.y);
    }
    let wide = wide.ok_or_else(|| {
        GpuError::shape(
            PROJECT,
            "the wide kernels (made with the trunk's dense blocks)",
        )
    })?;
    let kernels = WideKernels {
        gemm: &wide.gemm,
        g32: Some(&wide.g32),
    };
    let input = WideIn {
        q128: None,
        q32: None,
        f32: Some(s.x),
    };
    site::resident_gemm(gpu, &kernels, (w, s.name), &input, route, t, s.y)
}

/// The index of a sub-layer's site among a layer's two.
fn sub_index(sub: Sub) -> usize {
    match sub {
        Sub::Attn => 0,
        Sub::Ffn => 1,
    }
}

/// The index of layer `layer` among `layers`, refused by name past them.
fn layer_index(layer: usize, layers: usize) -> Result<usize, GpuError> {
    if layer < layers {
        Ok(layer)
    } else {
        Err(GpuError::shape(
            WHAT,
            format!("layer {layer} is past the trunk's {layers} layers"),
        ))
    }
}

/// The trunk around ERNIE's sub-layers: the streams, every site's parameters by
/// name, the buffers a walk writes, and the embedding table it reads on the host.
pub struct Trunk {
    facts: Facts,
    /// Two a layer: the attention site, then the feed-forward's.
    sites: Vec<Site>,
    /// The gain of each sub-layer's input norm, two a layer.
    norms: Vec<[String; 2]>,
    act: Act,
    pieces: HcF32PiecesKernels,
    hc: HcKernels,
    scratch: HcPreScratch,
    wide: Option<Wide>,
    dense: Vec<Dense>,
    embed: EmbedTable,
    /// Rows a walk holds at most: a prompt chunk's, the step's one included.
    rows: usize,
    /// The tokens of the walk in progress.
    tokens: usize,
    /// The stream buffers, and the index of the one holding the current streams.
    streams: [DeviceBuffer<f32>; 2],
    cur: usize,
    mixes: DeviceBuffer<f32>,
    /// The last pre's rows: `pre`, `post`, `comb` a token.
    hc_rows: DeviceBuffer<f32>,
    lin: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    /// `ds41_hc_post`'s fold of the new streams by the site's `pre`: the graph
    /// folds by the site's own mix before the sub-layer, so nothing reads this.
    post_fold: DeviceBuffer<f32>,
    /// The host rows of an embed: a token's row in each of the four streams.
    row: Vec<f32>,
}

impl Trunk {
    /// The trunk of the description's `facts` over the resident `w`, holding up to
    /// `rows` tokens a walk and reading the embedding from `file`. Refused by name:
    /// no rows; a width whose four streams are not whole pieces; a site, a norm
    /// gain or a dense projection of another form or shape; a SiTU rule the device
    /// does not run; and the embedding table's own refusals. Load-time only.
    pub fn load(
        gpu: &Gpu,
        w: &Weights,
        facts: &Facts,
        file: Arc<Split>,
        rows: usize,
    ) -> Result<Trunk, GpuError> {
        let stream = gpu.stream();
        let (hidden, layers) = (facts.hidden, facts.layers);
        let k = HC_STREAMS * hidden;
        if rows == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a trunk holds at least one token a walk",
            ));
        }
        if !k.is_multiple_of(HC_PIECE) {
            return Err(GpuError::shape(
                WHAT,
                format!("four streams of {hidden} values are {k}, not whole pieces of {HC_PIECE}"),
            ));
        }
        if facts.sinkhorn_iters == 0 {
            return Err(GpuError::shape(
                WHAT,
                "hyper_connection.sinkhorn_iterations is 0: the mix runs at least one round",
            ));
        }
        let act = Act::Situ {
            beta: facts.situ_beta,
            linear_beta: SITU_LINEAR_BETA,
        };
        act.code(WHAT)?;
        let mut sites = Vec::with_capacity(2 * layers);
        let mut norms = Vec::with_capacity(layers);
        for layer in 0..layers {
            for sub in [Sub::Attn, Sub::Ffn] {
                sites.push(Site::load(w, (layer, k), sub)?);
            }
            let attn = attn_norm(layer);
            let ffn = ffn_norm(layer);
            site::vector(w, WHAT, attn.clone(), hidden)?;
            site::vector(w, WHAT, ffn.clone(), hidden)?;
            norms.push([attn, ffn]);
        }
        let dense: Vec<Dense> = facts
            .dense
            .iter()
            .map(|&(layer, ff)| Dense::load(stream, w, (layer, ff, hidden, rows)))
            .collect::<Result<_, _>>()?;
        let wide = if dense.is_empty() {
            None
        } else {
            Some(Wide {
                gemm: GemmKernels::load(gpu.context())?,
                g32: Gemm32Kernels::load(gpu.context())?,
            })
        };
        let embed = EmbedTable::new(Arc::clone(&file), &token_embd(), hidden)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let scratch = HcPreScratch::with_groups(stream, k, col_group_count(LEAD, rows))?;
        let z = |len: usize| DeviceBuffer::zeroed(stream, len);
        Ok(Trunk {
            facts: facts.clone(),
            sites,
            norms,
            act,
            pieces: HcF32PiecesKernels::load(gpu.context())?,
            hc: HcKernels::load(gpu.context())?,
            scratch,
            wide,
            dense,
            embed,
            rows,
            tokens: 0,
            streams: [z(rows * k)?, z(rows * k)?],
            cur: 0,
            mixes: z(rows * HC_MIX)?,
            hc_rows: z(rows * HC_MIX)?,
            lin: z(rows * hidden)?,
            normed: z(rows * hidden)?,
            post_fold: z(rows * hidden)?,
            row: vec![0.0; rows * k],
        })
    }

    /// The width of a row.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.facts.hidden
    }

    /// The tokens a walk holds at most.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The tokens of the walk in progress.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The walk of the decode chain: one token, over the streams of the last embed. The chain is
    /// captured before any embed, and one enqueued after a prompt pass finds the pass's rows
    /// here; either way its launches are for the one token each refresh embeds.
    pub fn begin_step(&mut self) {
        self.tokens = 1;
    }

    /// The streams of the walk in progress: `tokens` of four streams, token-major.
    #[must_use]
    pub fn streams(&self) -> &DeviceBuffer<f32> {
        &self.streams[self.cur]
    }

    /// A sub-layer's input after its site's pre: the fold of the streams, `tokens`
    /// rows of [`Trunk::hidden`] values.
    #[must_use]
    pub fn lin(&self) -> &DeviceBuffer<f32> {
        &self.lin
    }

    /// A sub-layer's input, normed: what the sub-layer's own launches read.
    #[must_use]
    pub fn cur(&self) -> &DeviceBuffer<f32> {
        &self.normed
    }

    /// The last pre's rows, [`HC_MIX`] values a token (`pre`, `post`, `comb`): what
    /// the site's post reads, kept readable for the gate and a walk's debug rings.
    #[must_use]
    pub fn hc_rows(&self) -> &DeviceBuffer<f32> {
        &self.hc_rows
    }

    /// Layer `layer`'s dense block's last gate sums, up sums and SiTU product, each
    /// `tokens` rows of the block's width token-major (the product is what the down
    /// projection read); `None` for a layer with no dense block. The gate's read
    /// side, so each stage is held to its own rule.
    #[must_use]
    pub fn dense_taps(
        &self,
        layer: usize,
    ) -> Option<(&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>)> {
        self.dense
            .iter()
            .find(|d| d.layer == layer)
            .map(|d| (&d.g, &d.u, &d.a))
    }

    /// Layer `layer`'s `sub` site, refused by name past the trunk's layers.
    fn site(&self, layer: usize, sub: Sub) -> Result<&Site, GpuError> {
        let at = 2 * layer_index(layer, self.facts.layers)? + sub_index(sub);
        self.sites
            .get(at)
            .ok_or_else(|| GpuError::shape(WHAT, "the site's parameters (made with the trunk)"))
    }

    /// Start a walk of `tokens` in stream buffer 0: each token's embedding row, read
    /// from the file and dequantized on the host, repeated into its four streams.
    /// Synchronizing copies on the engine stream; never inside a capture. Refused by
    /// name: no token, more than [`Trunk::rows`], a token past the embedding's rows.
    pub fn embed(&mut self, stream: &CudaStream, tokens: &[u32]) -> Result<(), GpuError> {
        const EMBED: &str = "ernie55::Trunk::embed";
        let n = self.facts.hidden;
        if tokens.is_empty() || tokens.len() > self.rows {
            return Err(GpuError::shape(
                EMBED,
                format!("{} tokens; a walk holds 1 ..= {}", tokens.len(), self.rows),
            ));
        }
        let span = HC_STREAMS * n;
        let host = &mut self.row[..tokens.len() * span];
        for (t, &token) in tokens.iter().enumerate() {
            let rows = &mut host[t * span..][..span];
            let (first, rest) = rows.split_at_mut(n);
            self.embed
                .row_into(token, first)
                .map_err(|e| GpuError::plan(EMBED, e))?;
            for other in rest.chunks_exact_mut(n) {
                other.copy_from_slice(first);
            }
        }
        // `copy_from_host` takes the whole buffer; the walk's tokens are a prefix of it.
        WindowMut::<f32>::of_mut(&mut self.streams[0], 0, host.len())?
            .copy_from_host(stream, host)?;
        self.cur = 0;
        self.tokens = tokens.len();
        Ok(())
    }

    /// Sub-layer `sub` of layer `layer`'s pre over the streams: its mix
    /// ([`HcF32PiecesKernels::enqueue_pre`]: `hc_pre_f32_rmsadd`), the fold of the
    /// streams by that mix's `pre` into [`Trunk::lin`] (`ds41_hc_fold`), and the
    /// norm of the fold by the sub-layer's gain into [`Trunk::cur`]. The mix's
    /// `post` and `comb` stay for the site's post. The site's mix raises into
    /// `layer`'s fault sink.
    pub fn enqueue_pre(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        sub: Sub,
    ) -> Result<(), GpuError> {
        let (hidden, tokens) = (self.facts.hidden, self.tokens);
        let site = self.site(layer, sub)?;
        let params = site.params(w, (self.facts.hc_eps, self.facts.sinkhorn_iters))?;
        let stream = gpu.stream();
        let streams = &self.streams[self.cur];
        let a = HcF32Args {
            params: &params,
            x: streams,
            tokens,
            rms_eps: self.facts.rms_eps,
            fault: gpu.layer_sink(layer)?,
        };
        self.pieces.enqueue_pre(
            stream,
            &a,
            LEAD,
            &mut self.scratch,
            &mut self.mixes,
            &mut self.hc_rows,
        )?;
        self.hc.enqueue_fold(
            stream,
            streams,
            &self.hc_rows,
            hidden,
            tokens,
            &mut self.lin,
        )?;
        let gain = w.f32_buf(
            WHAT,
            &self.norms[layer_index(layer, self.facts.layers)?][sub_index(sub)],
        )?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &self.lin,
            gain,
            self.facts.rms_eps,
            hidden,
            tokens,
            &mut self.normed,
        )
    }

    /// Sub-layer `sub` of layer `layer`'s post: the sub-layer's output `x` (`tokens`
    /// rows of [`Trunk::hidden`] values) written back into the streams by the
    /// site's `post` and `comb` (`ds41_hc_post`), into the other stream buffer,
    /// which becomes the current one.
    pub fn enqueue_post(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        sub: Sub,
        x: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.site(layer, sub)?;
        let (left, right) = self.streams.split_at_mut(1);
        let (res, out, cur) = if self.cur == 0 {
            (&left[0], &mut right[0], 1)
        } else {
            (&right[0], &mut left[0], 0)
        };
        self.hc.enqueue_post(
            gpu.stream(),
            &HcPostArgs {
                x,
                res,
                hc: &self.hc_rows,
                n_embd: self.facts.hidden,
                tokens: self.tokens,
            },
            out,
            &mut self.post_fold,
        )?;
        self.cur = cur;
        Ok(())
    }

    /// Layer `layer`'s dense SiTU block over its normed input `x` (`tokens`
    /// rows token-major) into `y` ([`dense_block`]).
    pub fn enqueue_dense(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Trunk {
            facts,
            act,
            wide,
            dense,
            tokens,
            ..
        } = self;
        let run = DenseRun {
            hidden: facts.hidden,
            tokens: *tokens,
            act: *act,
            wide: wide.as_ref(),
        };
        dense_block(gpu, w, run, dense, layer, (x, y))
    }

    /// [`Trunk::enqueue_dense`] over the last pre's normed rows ([`Trunk::cur`]).
    pub fn enqueue_dense_normed(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Trunk {
            facts,
            act,
            wide,
            dense,
            tokens,
            normed,
            ..
        } = self;
        let run = DenseRun {
            hidden: facts.hidden,
            tokens: *tokens,
            act: *act,
            wide: wide.as_ref(),
        };
        dense_block(gpu, w, run, dense, layer, (normed, y))
    }

    /// The bytes it holds on the card: the two stream buffers, the mixes and the pre's rows, the
    /// fold, its norm and the post's fold, the pre's partials, and the dense blocks' rows and
    /// route tables.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let f = |b: &DeviceBuffer<f32>| b.num_bytes();
        f(&self.streams[0])
            + f(&self.streams[1])
            + f(&self.mixes)
            + f(&self.hc_rows)
            + f(&self.lin)
            + f(&self.normed)
            + f(&self.post_fold)
            + self.scratch.bytes()
            + self
                .dense
                .iter()
                .map(|d| f(&d.g) + f(&d.u) + f(&d.a) + d.route.bytes())
                .sum::<usize>()
    }
}

/// What a dense block's run reads of the trunk beside the block: the row width, the walk's tokens,
/// the SiTU rule and the wide arm's kernels.
struct DenseRun<'a> {
    hidden: usize,
    tokens: usize,
    act: Act,
    wide: Option<&'a Wide>,
}

/// Layer `layer`'s dense SiTU block of `dense` over its normed input `x` (`run.tokens` rows
/// token-major): the gate and the up projections, SiTU (`ElemKernels::enqueue_situ`), the down
/// projection, into `y`. At most [`GEMV_COLS`] tokens each projection is the m-column gemv of its
/// resident form; more run the wide GEMM over the block's dense route table.
fn dense_block(
    gpu: &Gpu,
    w: &Weights,
    run: DenseRun<'_>,
    dense: &mut [Dense],
    layer: usize,
    (x, y): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    const DENSE: &str = "ernie55::Trunk::enqueue_dense";
    let (t, hidden) = (run.tokens, run.hidden);
    let block = dense
        .iter_mut()
        .find(|d| d.layer == layer)
        .ok_or_else(|| GpuError::shape(DENSE, format!("layer {layer} carries no dense block")))?;
    if t == 0 || x.len() < t * hidden || y.len() < t * hidden {
        return Err(GpuError::shape(
            DENSE,
            format!(
                "tokens = {t}: x.len() {} / y.len() {} (need {})",
                x.len(),
                y.len(),
                t * hidden
            ),
        ));
    }
    let (stream, fault) = (gpu.stream(), gpu.layer_sink(layer)?);
    let names = [block.gate.as_str(), block.up.as_str(), block.down.as_str()];
    let feeds = |name: &str, rows: &DeviceBuffer<f32>, k: usize| -> Result<(), GpuError> {
        match bloomery_gpu::imatrix::collector() {
            Some(c) => c.feed_dense(gpu, name, rows, (t, k)),
            None => Ok(()),
        }
    };
    let wide = run.wide;
    if t > GEMV_COLS {
        wide.ok_or_else(|| {
            GpuError::shape(
                DENSE,
                "the wide kernels (made with the trunk's dense blocks)",
            )
        })?
        .gemm
        .enqueue_route_dense(stream, t, &mut block.route, fault)?;
    }
    for (name, buf) in [(names[0], &mut block.g), (names[1], &mut block.u)] {
        feeds(name, x, hidden)?;
        project(
            gpu,
            w,
            DenseSite { name, x, y: buf },
            t,
            (wide, &block.route),
        )?;
    }
    gpu.elem().enqueue_situ(
        stream,
        run.act,
        (&block.g, &block.u),
        t * block.ff,
        &mut block.a,
    )?;
    feeds(names[2], &block.a, block.ff)?;
    project(
        gpu,
        w,
        DenseSite {
            name: names[2],
            x: &block.a,
            y,
        },
        t,
        (wide, &block.route),
    )
}
