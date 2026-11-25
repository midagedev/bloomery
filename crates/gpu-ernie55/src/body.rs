//! The ERNIE 5.5 body: the layer walk over the parts and the load's pieces.
//!
//! One token a step ([`Body`] behind `GpuModel`):
//!
//! 1. the token's embedding row in each of the four hyper-connection streams ([`Trunk::embed`],
//!    on the host before the step's launches, as the step's other host inputs);
//! 2. each layer `l` of `0..n_layer`, in two sub-layers each wrapped by the layer's mHC mix
//!    ([`Trunk::enqueue_pre`] folds the streams into one row and norms it by the sub-layer's
//!    gain, [`Trunk::enqueue_post`] mixes the sub-layer's row back into them):
//!    - the attention: an HCA layer (`attention.compress_ratios` non-zero) or a latent MLA-DSA
//!      layer ([`Hparams::latent_layer`]), each a [`Mixer`];
//!    - the feed-forward block on the normed fold: the dense SiTU block on the leading layer
//!      ([`Hparams::dense`], [`Trunk::enqueue_dense`]) and the latent routed block on the rest
//!      ([`RoutedFfn`]);
//! 3. the head's mix and fold of the streams ([`HeadMix::enqueue_fold`]), the common head's norm
//!    and projection, MultiMax over the logits ([`HeadMix::enqueue_multimax`], the head's
//!    epilogue) and the argmax.
//!
//! A load whose routed experts the card cannot hold runs each layer with a host share as the
//! routed block's three parts around the common host tier's go and wait ([`crate::placed`]); its
//! prompt runs the same three parts around the host tier's batch port, a unit's rows at once
//! ([`crate::pass`]).
//!
//! The parts are the component modules' (`attn_hca`, `attn_mla`, `trunk`, `head`, `moe`), each
//! behind the narrow trait the walk calls; [`crate::parts`] builds them from the file. Every part
//! keeps its own state (its caches, words, streams and scratch) for the one resident sequence,
//! sized from the context and the walk's rows at the load. What the file forces here, each named
//! where it is used: the four streams, the attention kind a layer, the dense lead, the MultiMax
//! epilogue.

use std::ops::Range;

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::placed::PlacedHost;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::stream_side::StreamSide;
use bloomery_gpu::hybrid::{Chain, Hybrid};
use bloomery_gpu::model::{ChainBody, Rollback};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, Window, WindowMut, copy_row};
use cuda_core::{CudaStream, DeviceBuffer};
use model::ModelError;
use model::arch::Arch;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::names::Sub;
use model::moe::HostLayer;
use model::r8file::R8Source;
use runtime::seqstate::{Kept, Take};

/// What the body's refusals name.
pub(crate) const WHAT: &str = "ernie55::Body";

/// The hyper-connection streams every layer reads and writes (`hyper_connection.count`).
pub const HC: usize = 4;

/// The offset of the cache digests' FNV-1a 64 fold.
pub(crate) const DIGEST_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// `h` folded with `words`, each word's four bytes in order (FNV-1a 64).
pub(crate) fn fold(h: u64, words: impl IntoIterator<Item = u32>) -> u64 {
    words.into_iter().fold(h, |h, w| {
        w.to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

/// Bytes of an f32 on the card: the windows are byte-offset.
const F32: usize = size_of::<f32>();

/// The spacing of a prompt call's inner checkpoints: none. A call takes a checkpoint at its start
/// and at its end ([`Checkpoints::marks`]), so the sequence holds the two newest points
/// ([`Price::budget`](runtime::seqstate::Price::budget) of a ledger with no spacing: two slots,
/// whatever the context). A request that extends the last call's prompt returns to its end.
pub const CHECKPOINT_EVERY: u32 = 0;

/// The card and the weights a part's launch reads.
#[derive(Clone, Copy)]
pub struct Cx<'a> {
    pub gpu: &'a Gpu,
    pub w: &'a Weights,
}

/// A sub-layer's input as the trunk's pre leaves it: the fold of the streams and that fold normed
/// by the sub-layer's gain (`attn_norm`, `ffn_norm`), row-major `[t, hidden]` f32. A kind reads
/// the one its chain starts from.
#[derive(Clone, Copy)]
pub struct SubIn<'a> {
    pub fold: &'a DeviceBuffer<f32>,
    pub normed: &'a DeviceBuffer<f32>,
}

/// An attention kind's layers: every layer of the kind and the one resident sequence's caches.
/// Rows are row-major `[t, hidden]` f32.
pub trait Mixer {
    /// The kind's name in the walk's refusals.
    fn kind(&self) -> &'static str;

    /// Write the host words of the run of `m` tokens from position `b` (positions, visible
    /// counts, rope rows) on `stream`: outside a captured graph, before the run's launches. A run
    /// the words already hold is left as it is.
    fn set_run(&mut self, stream: &CudaStream, b: u32, m: usize) -> Result<(), GpuError>;

    /// Layer `l` at `pos` over the one row `x` into `out`, the token's cache row written first.
    /// The words hold the run `(pos, 1)`. Asynchronous, allocation-free, capturable.
    fn enqueue_step(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos: u32,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// Layer `l` over `t` rows at `pos0 .. pos0 + t` into `out`; each row the step's at its
    /// position. Writes its runs' words itself, on the stream, outside a capture.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the layer, the place, the length and the two buffers (rust-quality R8)"
    )]
    fn enqueue_prompt(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos0: u32,
        t: usize,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The rows one prompt launch takes at most.
    fn rows(&self) -> usize;

    /// Every cache back to the load's state: the next prompt writes the bits a fresh process
    /// writes.
    fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError>;

    /// The FNV-1a 64 of every cache's bits, read back (synchronizes `stream`): what a gate holds
    /// [`Mixer::reset`] to.
    fn cache_digest(&self, stream: &CudaStream) -> Result<u64, GpuError>;

    /// The stores a checkpoint copies, each as the f32 words of its bytes, in an order that is the
    /// kind's alone and the same on every call: what a position keeps that the positions after it
    /// overwrite. A kind whose stores are rows by position, cut by position, lends none.
    fn copied(&mut self) -> Result<Vec<WindowMut<'_, f32>>, GpuError> {
        Ok(Vec::new())
    }

    /// The device bytes the caches, words and scratch hold.
    fn bytes(&self) -> usize;

    /// The device bytes of the kind's stores alone, every layer's at the load's context: what the
    /// description's KV layout counts (`model::arch::place_hybrid::KvLayout`), held to the plan's
    /// at the load ([`crate::rooms::check`]).
    fn cache_bytes(&self) -> u64;

    /// The positions layer `l`'s last step or prompt row attended by selection (the indexer's
    /// top-k, ascending), read back for a gate; `None` for a kind that selects nothing or a row
    /// under the selection threshold. Synchronizes the stream.
    fn selection(&self, stream: &CudaStream, l: usize) -> Result<Option<Vec<u32>>, GpuError> {
        let _ = (stream, l);
        Ok(None)
    }
}

/// The routed layers' latent mixture of experts on the normed row, every expert on the card or
/// the plan's share on the card and the rest on the host tier.
pub trait RoutedFfn {
    /// Layer `l` over the one normed row `x` into `out`, every expert on the card.
    fn enqueue_step(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The front of a hosted layer's step: the router and the latent down of `x` into `lat`, the
    /// row the host leg reads. `slots` is the layer's `(slot map card copy, row offset)`, which a
    /// resident bonus reads ([`RoutedFfn::set_route_dials`]).
    fn enqueue_front(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        slots_lat: (Option<(&DeviceBuffer<u32>, usize)>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError>;

    /// The router levers of the load over a model of `n_layer` layers: the resident bonus, the
    /// top-k cut, the banned-experts mask and the census ([`bloomery_gpu::route_dial::RouteSetup`]).
    /// Returns the cut as a slot count, `None` the model's own. Load-time only.
    fn set_route_dials(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        n_layer: usize,
        setup: &bloomery_gpu::route_dial::RouteSetup,
    ) -> Result<Option<usize>, GpuError>;

    /// Write the routing census now; nothing with the census off.
    fn flush_census(&self) -> Result<(), GpuError>;

    /// The last launch's picks as the router left them on the card, row-major `[m, n_used]`: the
    /// ids and their final weights (renormalized, times the learned scale). The handoff's input
    /// and the taps' source.
    fn route(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<f32>);

    /// The shadow of a hosted layer's step: the shared expert of `x` and the card's slots over
    /// `lat` at the places `sel` (a card slot below `n_card`, or the host mark). Whether the
    /// card's slots ran.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the layer, the two rows and the places (rust-quality R8)"
    )]
    fn enqueue_shadow(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        lat: &DeviceBuffer<f32>,
        sel: &DeviceBuffer<u32>,
        n_card: usize,
    ) -> Result<bool, GpuError>;

    /// The back of a hosted layer's step (`card_hsum`): the card's sum where its flag is set, plus
    /// the host's sum, through the latent norm and up, plus the shared expert, into `out`.
    fn enqueue_back(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        card_hsum: (bool, Option<&DeviceBuffer<f32>>),
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// Layer `l` over the `m` normed rows of `x` into `out`, every expert on the card.
    fn enqueue_prompt(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The front of a hosted layer over the `m` normed rows of `x`: the router over the rows (the
    /// picks in [`RoutedFfn::route`], `[m, n_used]`) and the latent down of each row into the
    /// buffers' latent rows ([`RoutedFfn::latent_rows`]), the rows a prompt batch's download hands
    /// the host. The caller runs the download after this, and the places between the unit's pick
    /// and [`RoutedFfn::enqueue_card_rows`].
    fn enqueue_front_rows(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The last front's latent rows, `m` of the latent's width: the experts' input the card's
    /// slots read and a prompt batch's download hands the host.
    fn latent_rows(&self) -> &DeviceBuffer<f32>;

    /// [`RoutedFfn::enqueue_shadow`]'s shared expert over the `m` rows of `x`, into the buffers'
    /// `shared` rows: the half of the shadow that reads no places, which a prompt unit runs under
    /// the host's work before the unit's pick ([`RoutedFfn::enqueue_card_rows`] after it).
    fn enqueue_shared_rows(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// [`RoutedFfn::enqueue_shadow`]'s card slots over the `m` rows: one run over the latent rows
    /// ([`RoutedFfn::latent_rows`]) at the places `sel` (`m · n_used`, a card slot below `n_card`
    /// or the host mark), the picks' weights the sum's. Whether the card's slots ran.
    fn enqueue_card_rows(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        sel_n_card: (&DeviceBuffer<u32>, usize),
    ) -> Result<bool, GpuError>;

    /// [`RoutedFfn::enqueue_back`] over the `m` rows: the routed sums plus the host's `hsum` (`m`
    /// rows of the latent's width) through the norm, the latent up and the shared expert, into
    /// `out`.
    fn enqueue_back_rows(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        card_hsum: (bool, Option<&DeviceBuffer<f32>>),
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The rows one prompt launch takes at most.
    fn rows(&self) -> usize;

    /// The placed load's check of layer `l` with `on_card` experts on the card: the card's stacks
    /// hold exactly those, of the shape the card leg runs.
    fn check_card_stacks(&self, w: &Weights, l: usize, on_card: usize) -> Result<(), GpuError>;

    /// Layer `l`'s routed experts as the common host tier serves them. Load-time only.
    fn host_layer(&self, src: R8Source<'_>, l: usize) -> Result<HostLayer, ModelError>;

    /// The device bytes of its buffers.
    fn bytes(&self) -> usize;
}

/// The trunk around the layers: the walk's streams (two buffers of `[rows, HC, hidden]` f32, one
/// current), each sub-layer's mHC pre and post and the dense lead's block. A walk is the tokens of
/// the last [`Trunk::embed`]: every launch after it runs over those rows.
pub trait Trunk {
    /// Start a walk of `tokens` in stream buffer 0: each token's embedding row read on the host
    /// and repeated into its four streams, copied on `stream`. Synchronizing, outside a capture.
    fn embed(&mut self, stream: &CudaStream, tokens: &[u32]) -> Result<(), GpuError>;

    /// The walk of the decode chain: one token, over the streams of the last embed; the chain's
    /// launches are for it whether the chain is captured before any embed or enqueued after a
    /// prompt pass.
    fn begin_step(&mut self);

    /// Sub-layer `sub` of layer `l`'s pre over the walk's rows: the mix, the fold into
    /// [`Trunk::fold`], its norm by the sub-layer's gain into [`Trunk::normed`]; the post and comb
    /// mixes kept for [`Trunk::enqueue_post`].
    fn enqueue_pre(&mut self, cx: Cx<'_>, l: usize, sub: Sub) -> Result<(), GpuError>;

    /// The last pre's fold, un-normed.
    fn fold(&self) -> &DeviceBuffer<f32>;

    /// The last pre's fold normed by its sub-layer's gain.
    fn normed(&self) -> &DeviceBuffer<f32>;

    /// The post of the same sub-layer: `y` (the walk's rows) written back into the streams by the
    /// site's post and comb, into the other buffer, which becomes the current one.
    fn enqueue_post(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        sub: Sub,
        y: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// Layer `l`'s dense SiTU block over the last pre's normed rows into `y`.
    fn enqueue_dense(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// The current streams: the walk's rows of four streams, token-major.
    fn streams(&self) -> &DeviceBuffer<f32>;

    /// The rows a walk holds at most.
    fn rows(&self) -> usize;

    /// The device bytes of its streams and scratch.
    fn bytes(&self) -> usize;
}

/// The head's side of the output head: the four streams' mix and fold into the common head's
/// input, and the MultiMax epilogue on its logits.
pub trait HeadMix {
    /// The head's mix of one row's four streams `streams` and their fold into `input`
    /// (`hidden` values).
    fn enqueue_fold(
        &mut self,
        cx: Cx<'_>,
        streams: &DeviceBuffer<f32>,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;

    /// MultiMax over the `n` logits in place: the head's epilogue, before the argmax.
    fn enqueue_multimax(
        &mut self,
        cx: Cx<'_>,
        logits: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError>;

    /// The device bytes of its scratch.
    fn bytes(&self) -> usize;
}

/// Every part of the program, built from the file by [`crate::parts::load`].
pub struct Parts {
    pub hca: Box<dyn Mixer + Send>,
    pub mla: Box<dyn Mixer + Send>,
    pub moe: Box<dyn RoutedFfn + Send>,
    pub trunk: Box<dyn Trunk + Send>,
    pub head: Box<dyn HeadMix + Send>,
}

/// Where a sub-layer runs: the step's position, or a prompt unit's first position and rows.
#[derive(Clone, Copy)]
pub(crate) enum Run {
    Step(u32),
    Prompt(u32, usize),
}

impl Parts {
    /// The rows one prompt unit takes: the narrowest part's.
    pub(crate) fn rows(&self) -> usize {
        [
            self.hca.rows(),
            self.mla.rows(),
            self.moe.rows(),
            self.trunk.rows(),
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
    }

    /// Write both mixers' words for the run `(b, m)`.
    pub(crate) fn set_run(
        &mut self,
        stream: &CudaStream,
        b: u32,
        m: usize,
    ) -> Result<(), GpuError> {
        self.hca.set_run(stream, b, m)?;
        self.mla.set_run(stream, b, m)
    }

    /// Layer `l`'s attention over the trunk's last pre into `y`: the mixer of its kind (`latent`)
    /// at `at`.
    pub(crate) fn mix(
        &mut self,
        cx: Cx<'_>,
        (l, latent): (usize, bool),
        at: Run,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Parts {
            hca, mla, trunk, ..
        } = self;
        let mixer = if latent { mla } else { hca };
        let x = SubIn {
            fold: trunk.fold(),
            normed: trunk.normed(),
        };
        match at {
            Run::Step(pos) => mixer.enqueue_step(cx, l, pos, x, y),
            Run::Prompt(pos0, t) => mixer.enqueue_prompt(cx, l, pos0, t, x, y),
        }
    }

    fn bytes(&self) -> usize {
        self.hca.bytes()
            + self.mla.bytes()
            + self.moe.bytes()
            + self.trunk.bytes()
            + self.head.bytes()
    }
}

/// The widths and kinds the walk reads, from the file's header.
#[derive(Clone, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub latent_width: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
    /// `expert_used_count`: the picks a routed row keeps.
    pub n_used: usize,
    pub ctx: usize,
    pub eps: f32,
    /// Per layer: an MLA-DSA layer (`true`) or an HCA one.
    pub latent: Vec<bool>,
    /// Per layer: the dense feed-forward block (`true`) or the routed one.
    pub dense: Vec<bool>,
}

impl Shape {
    /// The shape of `hp`'s file at `ctx` positions; a context past the trained one, or none, is
    /// refused by name.
    pub fn of(hp: &Hparams, ctx: usize) -> Result<Shape, GpuError> {
        if ctx == 0 || ctx > hp.n_ctx_train {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a context of {ctx} positions; the file trains {}",
                    hp.n_ctx_train
                ),
            ));
        }
        Ok(Shape {
            hidden: hp.n_embd,
            latent_width: hp.latent,
            n_layer: hp.n_layer,
            n_vocab: hp.n_vocab,
            n_used: hp.experts.n_used,
            ctx,
            eps: hp.rms_eps,
            latent: (0..hp.n_layer).map(|l| hp.latent_layer(l)).collect(),
            dense: (0..hp.n_layer).map(|l| hp.dense(l)).collect(),
        })
    }

    /// The four streams of `m` rows.
    #[must_use]
    pub fn streams(&self, m: usize) -> usize {
        m * HC * self.hidden
    }
}

/// The rows a gate reads of the last step or prompt: each layer's streams after it, at the
/// tapped rows, and the head's logits before MultiMax. Armed by [`Body::arm_taps`].
pub(crate) struct Taps {
    /// Per layer, `rows` rows of the four streams.
    pub(crate) l_out: Vec<DeviceBuffer<f32>>,
    /// Per layer, `rows` rows of the picks' ids (their bits) and weights; unwritten on the dense
    /// layer.
    pub(crate) ids: Vec<DeviceBuffer<f32>>,
    pub(crate) weights: Vec<DeviceBuffer<f32>>,
    /// The rows the last walk left: the tapped rows are the last `filled` of the walk's.
    pub(crate) filled: usize,
    /// The projection's logits before the epilogue, one row.
    pub(crate) raw: DeviceBuffer<f32>,
    pub(crate) rows: usize,
}

/// What a gate reads back of the taps.
#[derive(Clone, Debug, Default)]
pub struct TapRows {
    /// Per layer, the tapped rows' four streams, row-major `[rows, HC, hidden]`.
    pub l_out: Vec<Vec<f32>>,
    /// Per layer, the tapped rows' picks, row-major `[rows, n_used]`; empty on the dense layer.
    pub ids: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
    /// The logits before MultiMax at the last row.
    pub raw: Vec<f32>,
    /// The rows each layer's taps hold: the walk's last ones.
    pub rows: usize,
}

/// A step's host input.
pub struct DecodeInput {
    pub token: u32,
    pub pos: u32,
}

/// The ERNIE 5.5 body on one stage card, its routed experts on the card or shared with the host
/// tier by the plan.
pub struct Body {
    pub(crate) sh: Shape,
    pub(crate) parts: Parts,
    /// A sub-layer's output over the trunk's rows.
    pub(crate) y: DeviceBuffer<f32>,
    /// The position the stores stand at: the next step's.
    pub(crate) held: u32,
    pub(crate) taps: Option<Taps>,
    /// The host tier's side of a placed load; `None` when the card holds every expert.
    pub(crate) placed: Option<PlacedHost>,
    /// The prompt pass's rows a unit ([`crate::pass`]); `None` on a load that steps its prompt.
    pub(crate) pass: Option<usize>,
    /// The hosted pass's buffers beside the placed side ([`crate::placed::HostedPass`]) on a load
    /// whose plan hosts experts and that has the pass; `None` otherwise.
    pub(crate) hosted: Option<crate::placed::HostedPass>,
    /// The HCA layers' window rings and compressor rings on the host at chosen positions
    /// ([`Mixer::copied`]): what a cut goes back to.
    pub(crate) ckpt: Checkpoints,
    /// The prompt calls' host streaming ([`crate::stream`]): the `BLOOMERY_XSTREAM` lever's side.
    pub(crate) xs: StreamSide,
}

impl Body {
    /// The body of `hp`'s file at `ctx` positions over `parts`: a sub-layer's output buffer over
    /// the trunk's rows, no host tier and no pass. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        hp: &Hparams,
        ctx: usize,
        mut parts: Parts,
    ) -> Result<Body, GpuError> {
        let sh = Shape::of(hp, ctx)?;
        let y = DeviceBuffer::zeroed(gpu.stream(), parts.trunk.rows().max(1) * sh.hidden)?;
        let lens: Vec<usize> = parts.hca.copied()?.iter().map(|w| w.len()).collect();
        let ckpt = if lens.is_empty() {
            Checkpoints::stateless(gpu.context(), CHECKPOINT_EVERY)
        } else {
            Checkpoints::sequence(gpu.context(), lens, ctx, CHECKPOINT_EVERY)?
        };
        Ok(Body {
            sh,
            parts,
            y,
            held: 0,
            taps: None,
            placed: None,
            pass: None,
            hosted: None,
            ckpt,
            xs: StreamSide::default(),
        })
    }

    /// What a cut to at most `n` positions of a model standing at `pos` keeps, and why: every
    /// position, the empty model, or the nearest checkpoint at or below `n`. After a call that
    /// failed part-way the stores hold other positions than `pos`, so only a checkpoint is kept.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if self.held == pos {
            self.ckpt.kept(n, pos)
        } else {
            self.ckpt.kept(n.min(pos), self.held)
        }
    }

    /// The checkpoints: their positions, their slots, what they have done.
    #[must_use]
    pub fn checkpoints(&self) -> &Checkpoints {
        &self.ckpt
    }

    /// A checkpoint at `pos`, the position the model stands at, after any waiting cut: the HCA
    /// layers' window rings and compressor rings copied to a host slot, or nothing where one
    /// stands. Refused by name when the stores hold other positions. Waits for the copies.
    pub fn checkpoint(&mut self, gpu: &Gpu, pos: u32) -> Result<Take, GpuError> {
        self.held_at(pos)?;
        self.apply_cut(gpu.stream())?;
        let Body { parts, ckpt, .. } = self;
        let mut lent = parts.hca.copied()?;
        let mut stores: Vec<&mut DeviceBuffer<f32>> = lent.iter_mut().map(|w| &mut **w).collect();
        ckpt.take(gpu.stream(), pos, &mut stores)
    }

    /// The waiting cut carried out: each store from its checkpoint's slot, or zeroed for the empty
    /// model. Nothing, and no allocation, when no cut waits. Waits for the copies; the first
    /// device call after a cut runs it ([`ChainBody::refresh`], a prompt call).
    pub(crate) fn apply_cut(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        let Body { parts, ckpt, .. } = self;
        let mut lent = parts.hca.copied()?;
        let mut stores: Vec<&mut DeviceBuffer<f32>> = lent.iter_mut().map(|w| &mut **w).collect();
        ckpt.apply(stream, &mut stores)
    }

    /// A cut to `pos`: nothing at the position the stores stand at; the empty model at 0; else the
    /// checkpoint at `pos`, copied back at the next device call. Any other position is refused by
    /// name ([`Body::kept`] says what a cut keeps).
    pub fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
        self.ckpt.cut(pos, self.held)?;
        self.held = pos;
        Ok(())
    }

    /// Refused by name unless the stores hold `pos` positions: a step that failed past its inputs
    /// has written the position into the window ring and a prompt call that failed part-way left
    /// the stores between positions, so a checkpoint of either would hold a state no position has.
    fn held_at(&self, pos: u32) -> Result<(), GpuError> {
        if self.held == pos {
            return Ok(());
        }
        Err(GpuError::shape(
            WHAT,
            format!(
                "a checkpoint at position {pos} on stores that hold {} (a call failed part-way: \
                 cut back to a checkpoint or reset first)",
                if self.held == u32::MAX {
                    "no position".to_string()
                } else {
                    format!("{} positions", self.held)
                }
            ),
        ))
    }

    /// The widths and kinds.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.sh
    }

    /// Each layer's kinds, (latent MLA-DSA, dense lead): the kinds a gate holds the cut to.
    #[must_use]
    pub fn kinds(&self) -> Vec<(bool, bool)> {
        self.sh
            .latent
            .iter()
            .zip(&self.sh.dense)
            .map(|(&a, &d)| (a, d))
            .collect()
    }

    /// The layers with a host share: empty on a load whose card holds every expert.
    #[must_use]
    pub fn hosted(&self) -> Vec<usize> {
        self.placed
            .as_ref()
            .map_or_else(Vec::new, |p| p.side.hosted_set())
    }

    /// Arm the taps for `rows` rows a layer: the next step or prompt writes them. Load-time.
    pub fn arm_taps(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let rows = rows.max(1);
        let bufs = |n: usize| {
            (0..self.sh.n_layer)
                .map(|_| DeviceBuffer::zeroed(stream, n))
                .collect::<Result<Vec<_>, _>>()
        };
        self.taps = Some(Taps {
            l_out: bufs(self.sh.streams(rows))?,
            ids: bufs(rows * self.sh.n_used)?,
            weights: bufs(rows * self.sh.n_used)?,
            filled: 0,
            raw: DeviceBuffer::zeroed(stream, self.sh.n_vocab)?,
            rows,
        });
        Ok(())
    }

    /// The tapped rows, read back (synchronizes the stream); refused by name when unarmed.
    pub fn taps(&self, stream: &CudaStream) -> Result<TapRows, GpuError> {
        let t = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps",
        })?;
        let (n, k) = (t.filled, self.sh.n_used);
        let read = |buf: &DeviceBuffer<f32>, width: usize| -> Result<Vec<f32>, GpuError> {
            let mut v = buf.to_host_vec(stream)?;
            v.truncate(n * width);
            Ok(v)
        };
        let mut rows = TapRows {
            rows: n,
            raw: t.raw.to_host_vec(stream)?,
            ..TapRows::default()
        };
        for l in 0..self.sh.n_layer {
            rows.l_out.push(read(&t.l_out[l], self.sh.streams(1))?);
            let (ids, weights) = if self.sh.dense[l] {
                (Vec::new(), Vec::new())
            } else {
                let ids = read(&t.ids[l], k)?.into_iter().map(f32::to_bits).collect();
                (ids, read(&t.weights[l], k)?)
            };
            rows.ids.push(ids);
            rows.weights.push(weights);
        }
        Ok(rows)
    }

    /// The digests of the HCA caches and of the MLA-DSA stores ([`Mixer::cache_digest`]).
    pub fn cache_digests(&self, stream: &CudaStream) -> Result<[u64; 2], GpuError> {
        Ok([
            self.parts.hca.cache_digest(stream)?,
            self.parts.mla.cache_digest(stream)?,
        ])
    }

    /// Layer `l`'s last selection (synchronizes); `None` on an HCA layer or under the threshold.
    pub fn selection(&self, stream: &CudaStream, l: usize) -> Result<Option<Vec<u32>>, GpuError> {
        if !self.sh.latent[l] {
            return Ok(None);
        }
        self.parts.mla.selection(stream, l)
    }

    /// The host inputs of the step at `pos`: both mixers' words and the token's streams, and the
    /// taps' count of one row, which a replayed chain's tap copies write without the enqueue that
    /// counts them.
    pub(crate) fn host_inputs(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: u32,
    ) -> Result<(), GpuError> {
        self.parts.set_run(stream, pos, 1)?;
        if let Some(t) = self.taps.as_mut() {
            t.filled = t.rows.min(1);
        }
        self.parts.trunk.embed(stream, &[token])
    }

    /// The device bytes the body holds beside the weights.
    pub(crate) fn device_bytes(&self) -> usize {
        self.parts.bytes()
            + F32 * self.y.len()
            + self.placed.as_ref().map_or(0, PlacedHost::device_bytes)
            + self
                .hosted
                .as_ref()
                .map_or(0, crate::placed::HostedPass::bytes)
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = PlacedHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::Ernie55
    }

    /// The step at `pos`; a position past the stores is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("a step at position {pos} in stores of {}", self.sh.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// A waiting cut carried out, then the step's host inputs, refused by name off the position
    /// the stores stand at: a window ring holds the position it stands at alone.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        if input.pos != self.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a step at position {}; the stores stand at {} (reset after a call that \
                     failed part-way)",
                    input.pos, self.held
                ),
            ));
        }
        self.apply_cut(stream)?;
        self.host_inputs(stream, input.token, input.pos)?;
        self.held = input.pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        crate::step::enqueue(gpu, w, self, head)
    }

    /// Every part's caches back to the load's state, the position to 0, the host tier settled and
    /// every checkpoint dropped.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.parts.hca.reset(stream)?;
        self.parts.mla.reset(stream)?;
        if let Some(p) = self.placed.as_mut() {
            p.reset(stream)?;
        }
        self.held = 0;
        self.ckpt.clear();
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every norm's, the head's included.
    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn resident_bytes(&self) -> usize {
        self.device_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.sh.n_layer
    }

    fn host(&mut self) -> Option<&mut PlacedHost> {
        self.placed.as_mut()
    }
}

/// Why a tier accessor that demands the tier panics on a load with no host share: a reader that
/// serves either load asks [`TierBody::host_tier`].
const NO_TIER: &str = "ernie55::Body: a load whose card holds every expert has no host tier";

impl TierBody for Body {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.placed.as_ref().expect(NO_TIER).hybrid()
    }

    fn host_tier(&self) -> Option<&Hybrid<HostRun>> {
        self.placed.as_ref().map(TierBody::hybrid)
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.placed.as_mut().expect(NO_TIER).hybrid_mut()
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.placed.as_mut().and_then(TierBody::residency_parts)
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(placed) => placed.serve_chain(chain),
            None => Err(GpuError::shape(WHAT, NO_TIER)),
        }
    }
}

impl Rollback for Body {
    /// A cut to `pos` ([`Body::cut`]): a position where a checkpoint stands, refused by name at any
    /// other (an HCA layer's window ring holds the position it stands at alone); the card work
    /// waits for the next device call.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
    }
}

/// Copy the last of the walk's `m` rows of the trunk's current streams into layer `l`'s tap, when
/// armed; at most the tap's rows.
pub(crate) fn tap_l_out(gpu: &Gpu, body: &mut Body, l: usize, m: usize) -> Result<(), GpuError> {
    let Body {
        sh, parts, taps, ..
    } = body;
    let Some(t) = taps.as_mut() else {
        return Ok(());
    };
    let width = sh.streams(1);
    let n = m.min(t.rows);
    t.filled = n;
    tail_rows(gpu, parts.trunk.streams(), &mut t.l_out[l], (m, n, width))
}

/// Copy the last `n` of the routed block's `m` rows of picks (`[m, n_used]`) into layer `l`'s
/// taps, when armed: the ids' bits unchanged.
pub(crate) fn tap_picks(gpu: &Gpu, body: &mut Body, l: usize, m: usize) -> Result<(), GpuError> {
    let Body {
        sh, parts, taps, ..
    } = body;
    let Some(t) = taps.as_mut() else {
        return Ok(());
    };
    let n = m.min(t.rows);
    let (ids, weights) = parts.moe.route();
    tail_rows(gpu, ids, &mut t.ids[l], (m, n, sh.n_used))?;
    tail_rows(gpu, weights, &mut t.weights[l], (m, n, sh.n_used))
}

/// The last `n` of the `m` rows of `width` 4-byte words of `src` into the head of `dst`, bits
/// unchanged.
fn tail_rows<T>(
    gpu: &Gpu,
    src: &DeviceBuffer<T>,
    dst: &mut DeviceBuffer<f32>,
    (m, n, width): (usize, usize, usize),
) -> Result<(), GpuError> {
    let from = Window::<f32>::of(src, (m - n) * width * F32, n * width)?;
    let mut to = WindowMut::<f32>::of_mut(dst, 0, n * width)?;
    copy_row(gpu.stream(), &from, &mut to, n * width)
}
