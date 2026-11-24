//! The parts of the program built from the file, each behind the trait the walk calls
//! ([`crate::body`]): the trunk and the head's mix as themselves, the HCA and the latent layers as
//! a [`Mixer`] each over their component and its state, the routed blocks as a [`RoutedFfn`]
//! over the block and its buffers. What an adapter adds is only what the walk's trait asks and the
//! component leaves to its caller: the latent chain's spans and head mix, which layer's state a
//! call reads, and the contexts the components' launches take.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::GqaCx;
use bloomery_gpu::flash_gqa::FlashGqaKernels;
use bloomery_gpu::gemm::{Gemm32Kernels, GemmKernels};
use bloomery_gpu::site::{ByName, KGemvKernels};
use bloomery_gpu::vha::VhaKernels;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, Window, WindowMut, copy_row};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::ModelError;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::names::Sub;
use model::arch::ernie55::place::MLA_UNIT;
use model::moe::HostLayer;
use model::r8file::R8Source;

use crate::attn_hca::{HcaDims, HcaIn, HcaKernels, HcaLayer, HcaState, vha_rows};
use crate::attn_mla::{self, Call, Mla, MlaState, Scratch, SpanWords, Vha};
use crate::body::{self, Cx, Mixer, Parts, RoutedFfn, SubIn};
use crate::head::Head;
use crate::moe::{self, Moe, MoeBufs};
use crate::trunk::{Facts, Trunk};

/// What the HCA adapter's refusals name.
const HCA: &str = "ernie55 parts: HCA";
/// What the latent adapter's refusals name.
const MLA: &str = "ernie55 parts: MLA-DSA";
/// What the routed adapter's contexts name.
const MOE: &str = "ernie55 parts: routed";

/// Bytes of an f32 on the card: the windows are byte-offset.
const F32: usize = size_of::<f32>();

/// The card tensors the parts derive from the file's at the load, into `w` beside the resident
/// layers `layers` of `file`: the latent layers' absorbed key and value planes
/// ([`attn_mla::derive`], the split of `attn_kv_b`). Load-time only.
pub fn derive(
    stream: &CudaStream,
    file: &Split,
    hp: &Hparams,
    layers: Range<usize>,
    w: &mut Weights,
) -> Result<(), GpuError> {
    attn_mla::derive(stream, file, hp, layers, w)
}

/// Every part of `file` at the load's context `ctx`, its prompt form over units of up to `rows`
/// rows, the card's tensors in `w` (the derived ones included, [`derive`]). Load-time only.
pub fn load(
    gpu: &Gpu,
    file: &Arc<Split>,
    w: &Weights,
    hp: &Hparams,
    (ctx, rows): (usize, usize),
) -> Result<Parts, GpuError> {
    let facts = Facts::of(hp)?;
    Ok(Parts {
        hca: Box::new(HcaMixer::load(gpu, file, w, hp, (ctx, rows))?),
        mla: Box::new(LatentMixer::load(gpu, file, w, hp, (ctx, rows))?),
        moe: Box::new(Routed::load(gpu, file, w, hp, rows)?),
        trunk: Box::new(Trunk::load(gpu, w, &facts, Arc::clone(file), rows)?),
        head: Box::new(Head::load(gpu, w, &facts, 1)?),
    })
}

/// The head's mix factors of layer `l` of `file` (the card copies `w` holds), folded for the card
/// ([`bloomery_gpu::vha::fold`]).
fn head_mix(
    stream: &CudaStream,
    (file, w): (&Split, &Weights),
    hp: &Hparams,
    l: usize,
) -> Result<DeviceBuffer<f32>, GpuError> {
    let vha = vha_rows(stream, (file, w), l, (hp.heads, hp.vha_rank))?;
    let mix = bloomery_gpu::vha::fold(&vha.u, &vha.v, hp.heads, vha.rank, MLA)?;
    Ok(DeviceBuffer::from_host(stream, &mix)?)
}

// ------------------------------------------------------------------------------------- the trunk

impl body::Trunk for Trunk {
    fn begin_step(&mut self) {
        Trunk::begin_step(self);
    }

    fn embed(&mut self, stream: &CudaStream, tokens: &[u32]) -> Result<(), GpuError> {
        Trunk::embed(self, stream, tokens)
    }

    fn enqueue_pre(&mut self, cx: Cx<'_>, l: usize, sub: Sub) -> Result<(), GpuError> {
        Trunk::enqueue_pre(self, cx.gpu, cx.w, l, sub)
    }

    fn fold(&self) -> &DeviceBuffer<f32> {
        self.lin()
    }

    fn normed(&self) -> &DeviceBuffer<f32> {
        self.cur()
    }

    fn enqueue_post(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        sub: Sub,
        y: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        Trunk::enqueue_post(self, cx.gpu, l, sub, y)
    }

    fn enqueue_dense(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_dense_normed(cx.gpu, cx.w, l, y)
    }

    fn streams(&self) -> &DeviceBuffer<f32> {
        Trunk::streams(self)
    }

    fn rows(&self) -> usize {
        Trunk::rows(self)
    }

    fn bytes(&self) -> usize {
        Trunk::bytes(self)
    }
}

// --------------------------------------------------------------------------------- the head's mix

impl body::HeadMix for Head {
    /// The one row's fold: the head part is loaded for one row a run ([`load`]).
    fn enqueue_fold(
        &mut self,
        cx: Cx<'_>,
        streams: &DeviceBuffer<f32>,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        Head::enqueue_fold(self, cx.gpu, cx.w, streams, 1, input)
    }

    fn enqueue_multimax(
        &mut self,
        cx: Cx<'_>,
        logits: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        Head::enqueue_multimax(self, cx.gpu, logits, n)
    }

    fn bytes(&self) -> usize {
        Head::bytes(self)
    }
}

// ------------------------------------------------------------------------------------ the HCA layers

/// The HCA layers: the kernels, each HCA layer's tensors and the one sequence's state. A layer
/// reads the trunk's normed fold ([`HcaIn::Normed`]): its `attn_norm` is the trunk's pre's, the
/// same launch over the same gain and epsilon, so it runs once.
struct HcaMixer {
    k: HcaKernels,
    /// Per layer of the file, `None` on a latent one.
    layers: Vec<Option<HcaLayer>>,
    st: HcaState,
}

impl HcaMixer {
    /// The HCA layers of `hp`'s `file` at `ctx` positions, their prompt pieces the multiple of the
    /// projections' columns at or above `rows`. Load-time only.
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: &Hparams,
        (ctx, rows): (usize, usize),
    ) -> Result<HcaMixer, GpuError> {
        let stream = gpu.stream();
        let d = HcaDims::of(hp, ctx, rows.max(1).next_multiple_of(HcaDims::COLS))?;
        let layers = (0..hp.n_layer)
            .map(|l| {
                if hp.latent_layer(l) {
                    return Ok(None);
                }
                let vha = vha_rows(stream, (file, w), l, (hp.heads, hp.vha_rank))?;
                HcaLayer::load(stream, w, &d, l, &vha).map(Some)
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(HcaMixer {
            k: HcaKernels::load(gpu.context())?,
            st: HcaState::new(stream, &d, hp.n_layer, |l| !hp.latent_layer(l))?,
            layers,
        })
    }
}

/// Layer `l` of `layers`, refused by name when the load holds no HCA layer there.
fn hca_layer(layers: &[Option<HcaLayer>], l: usize) -> Result<&HcaLayer, GpuError> {
    layers
        .get(l)
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::shape(HCA, format!("layer {l} is no HCA layer of the load")))
}

impl Mixer for HcaMixer {
    fn kind(&self) -> &'static str {
        "HCA"
    }

    fn set_run(&mut self, stream: &CudaStream, b: u32, m: usize) -> Result<(), GpuError> {
        self.st.set_run(stream, b, m)
    }

    fn enqueue_step(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos: u32,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let HcaMixer { k, layers, st } = self;
        let stream = cx.gpu.stream();
        hca_layer(layers, l)?.enqueue_step(
            cx.gpu,
            k,
            cx.w,
            st,
            stream,
            pos,
            HcaIn::Normed(x.normed),
            out,
        )
    }

    fn enqueue_prompt(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos0: u32,
        t: usize,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let HcaMixer { k, layers, st } = self;
        let stream = cx.gpu.stream();
        hca_layer(layers, l)?.enqueue_prompt(
            cx.gpu,
            k,
            cx.w,
            st,
            stream,
            pos0,
            t,
            HcaIn::Normed(x.normed),
            out,
        )
    }

    fn rows(&self) -> usize {
        self.st.dims().piece
    }

    fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.st.reset(stream)
    }

    fn cache_digest(&self, stream: &CudaStream) -> Result<u64, GpuError> {
        self.st.digest(stream)
    }

    /// The state's bytes (its caches, staging, words and scratch) and the layers' folded head
    /// mixes.
    fn bytes(&self) -> usize {
        self.st.bytes()
            + self
                .layers
                .iter()
                .flatten()
                .map(|l| l.mix.num_bytes())
                .sum::<usize>()
    }

    fn cache_bytes(&self) -> u64 {
        (self.st.dims().cache_bytes() * self.layers.iter().flatten().count()) as u64
    }
}

// ---------------------------------------------------------------------------- the latent layers

/// The latent layers: the component over every latent layer, a state a layer, the one span's
/// scratch, words and forms the layers share ([`MLA_UNIT`] columns, the plan's), the head mix's
/// kernels and each layer's folded mix. A prompt runs in spans of the scratch's columns.
struct LatentMixer {
    mla: Mla,
    /// Per layer of the file, `None` on an HCA one.
    states: Vec<Option<MlaState>>,
    scratch: Scratch,
    words: SpanWords,
    forms: ByName,
    vha: VhaKernels,
    /// Per layer of the file, the folded head mix; `None` on an HCA one.
    mixes: Vec<Option<DeviceBuffer<f32>>>,
    /// The rows a prompt call takes at most: the load's unit.
    rows: usize,
    /// The last latent layer: the scratch holds its list after a walk.
    top: usize,
    /// The position and the scratch column of the last row the words or a prompt span set.
    last: Option<(u32, usize)>,
}

impl LatentMixer {
    /// The latent layers of `hp`'s `file` at `ctx` positions, prompt calls of up to `rows` rows.
    /// Refused by name: a file with no latent layer, and the component's refusals. Load-time only.
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: &Hparams,
        (ctx, rows): (usize, usize),
    ) -> Result<LatentMixer, GpuError> {
        let stream = gpu.stream();
        let latent: Vec<usize> = (0..hp.n_layer).filter(|&l| hp.latent_layer(l)).collect();
        let mla = Mla::load(gpu, file, w, hp, ctx, &latent)?;
        let mut states: Vec<Option<MlaState>> = (0..hp.n_layer).map(|_| None).collect();
        let mut mixes: Vec<Option<DeviceBuffer<f32>>> = (0..hp.n_layer).map(|_| None).collect();
        for (at, &l) in latent.iter().enumerate() {
            states[l] = Some(mla.state(stream, at, ctx)?);
            mixes[l] = Some(head_mix(stream, (file, w), hp, l)?);
        }
        let top = latent
            .last()
            .copied()
            .ok_or_else(|| GpuError::shape(MLA, "the file holds no latent layer".to_string()))?;
        Ok(LatentMixer {
            scratch: mla.scratch(stream, MLA_UNIT)?,
            words: mla.words(stream, MLA_UNIT)?,
            forms: mla.forms(gpu, w, MLA_UNIT)?,
            vha: VhaKernels::load(gpu.context())?,
            mla,
            states,
            mixes,
            rows: rows.max(1),
            top,
            last: None,
        })
    }
}

/// The head mix of the latent layers through the chain's hook: layer `l`'s folded mix over its
/// heads' values, raised into `l`'s fault sink.
struct LatentMix<'a> {
    gpu: &'a Gpu,
    k: &'a VhaKernels,
    mixes: &'a [Option<DeviceBuffer<f32>>],
    heads: usize,
    width: usize,
}

impl Vha for LatentMix<'_> {
    fn mix(
        &self,
        stream: &CudaStream,
        layer: usize,
        t: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let mt = self
            .mixes
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(MLA, format!("layer {layer} has no head mix")))?;
        copy_row(stream, x, y, t * self.heads * self.width)?;
        self.k.enqueue_mix(
            stream,
            mt,
            y,
            self.heads,
            self.width,
            t,
            self.gpu.layer_sink(layer)?,
        )
    }
}

impl LatentMixer {
    /// The chain over `n` rows at `pos` of `x` into `out` (one row the step's chain), the span's
    /// words set by the caller: layer `l`'s state, the call's head mix.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the layer, the place, the length and the two buffers (rust-quality R8)"
    )]
    fn run(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos: u32,
        n: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let LatentMixer {
            mla,
            states,
            scratch,
            words,
            forms,
            vha,
            mixes,
            ..
        } = self;
        let sh = mla.shape();
        let st = states.get_mut(l).and_then(Option::as_mut).ok_or_else(|| {
            GpuError::shape(MLA, format!("layer {l} is no latent layer of the load"))
        })?;
        let mix = LatentMix {
            gpu: cx.gpu,
            k: vha,
            mixes,
            heads: sh.heads,
            width: sh.head_v,
        };
        let mut call = Call {
            gpu: cx.gpu,
            w: cx.w,
            vha: &mix,
            words,
            bn: forms,
            s: scratch,
        };
        if n == 1 {
            mla.enqueue_step(&mut call, st, pos, x, out)
        } else {
            mla.enqueue_prompt(&mut call, st, pos, n, x, out)
        }
    }
}

impl Mixer for LatentMixer {
    fn kind(&self) -> &'static str {
        "MLA-DSA"
    }

    fn set_run(&mut self, stream: &CudaStream, b: u32, m: usize) -> Result<(), GpuError> {
        if self.words.span() != (b, m) {
            self.words.set(stream, b, m)?;
        }
        self.last = Some((b + m as u32 - 1, (m - 1) % self.scratch.unit));
        Ok(())
    }

    fn enqueue_step(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos: u32,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.run(cx, l, pos, 1, x.normed, out)
    }

    /// The prompt in spans of the scratch's columns, each span's words set before its chain.
    fn enqueue_prompt(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        pos0: u32,
        t: usize,
        x: SubIn<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (hidden, unit) = (self.mla.shape().hidden, self.scratch.unit);
        if t == 0 {
            return Err(GpuError::shape(MLA, "a prompt of no row"));
        }
        let mut done = 0;
        while done < t {
            let n = unit.min(t - done);
            let at =
                pos0 + u32::try_from(done).map_err(|_| GpuError::shape(MLA, "a row past u32"))?;
            self.set_run(cx.gpu.stream(), at, n)?;
            let xs = Window::<f32>::of(x.normed, done * hidden * F32, n * hidden)?;
            let mut os = WindowMut::<f32>::of_mut(out, done * hidden * F32, n * hidden)?;
            self.run(cx, l, at, n, &xs, &mut os)?;
            done += n;
        }
        Ok(())
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        for st in self.states.iter_mut().flatten() {
            st.reset(stream)?;
        }
        self.last = None;
        Ok(())
    }

    fn cache_digest(&self, stream: &CudaStream) -> Result<u64, GpuError> {
        self.states
            .iter()
            .flatten()
            .try_fold(crate::body::DIGEST_BASIS, |h, st| st.digest(h, stream))
    }

    /// The states, the span's scratch, words and forms, the gather's tables and the folded head
    /// mixes.
    fn bytes(&self) -> usize {
        self.states
            .iter()
            .flatten()
            .map(MlaState::bytes)
            .sum::<usize>()
            + self.scratch.bytes()
            + self.words.bytes()
            + self.forms.bytes()
            + self.mla.bytes()
            + self
                .mixes
                .iter()
                .flatten()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
    }

    fn cache_bytes(&self) -> u64 {
        self.states.iter().flatten().map(|s| s.bytes() as u64).sum()
    }

    /// The top-k part of the last row's list, ascending, of the last latent layer (the scratch
    /// holds that layer's list alone after a walk); `None` when the row's candidates older than
    /// the window are not more than `top_k`. Refused by name: another latent layer, no walk, a
    /// selecting row's list of another length.
    fn selection(&self, stream: &CudaStream, l: usize) -> Result<Option<Vec<u32>>, GpuError> {
        if l != self.top {
            return Err(GpuError::shape(
                MLA,
                format!(
                    "layer {l}'s selection: the scratch keeps the list of the last latent layer, \
                     {}",
                    self.top
                ),
            ));
        }
        let (pos, col) = self.last.ok_or(GpuError::State {
            what: MLA,
            missing: "a walk's last row",
        })?;
        let sh = self.mla.shape();
        let cands = (pos as usize + 1).saturating_sub(sh.window);
        if cands <= sh.top_k {
            return Ok(None);
        }
        let (list, count) = self.scratch.selection(stream, col)?;
        if count as usize != sh.stride() || list.len() < sh.top_k {
            return Err(GpuError::shape(
                MLA,
                format!(
                    "a selecting row at {pos} lists {count} keys in {} cells; the list holds \
                     top_k {} and the window {}",
                    list.len(),
                    sh.top_k,
                    sh.window
                ),
            ));
        }
        let mut top = list[..sh.top_k].to_vec();
        top.sort_unstable();
        Ok(Some(top))
    }
}

// -------------------------------------------------------------------------------- the routed block

/// The kernels the routed block's sites and prompt launches take ([`GqaCx`], [`GemmKernels`]).
/// The context's flash entry is a field the common sites' context carries; no routed launch
/// reads it.
struct SiteKernels {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    gemm: GemmKernels,
}

/// The routed block of every routed layer and its buffers, every launch through the common sites'
/// context.
struct Routed {
    moe: Moe,
    bufs: MoeBufs,
    k: SiteKernels,
    hidden: usize,
    eps: f32,
}

impl Routed {
    /// Every routed layer of `hp`'s `file`, buffers for prompt units of up to `rows` rows.
    /// Load-time only.
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: &Hparams,
        rows: usize,
    ) -> Result<Routed, GpuError> {
        let (moe, bufs) = Moe::load(gpu, file, w, hp, rows.max(1))?;
        Ok(Routed {
            moe,
            bufs,
            k: SiteKernels {
                kgemv: KGemvKernels::load(gpu)?,
                g32: Gemm32Kernels::load(gpu.context())?,
                flash: FlashGqaKernels::load(gpu.context())?,
                gemm: GemmKernels::load(gpu.context())?,
            },
            hidden: hp.n_embd,
            eps: hp.rms_eps,
        })
    }
}

/// The common sites' context over `cx` and `k`.
fn sites<'a>(cx: Cx<'a>, k: &'a SiteKernels, (hidden, eps): (usize, f32)) -> GqaCx<'a> {
    GqaCx {
        what: MOE,
        gpu: cx.gpu,
        w: cx.w,
        kgemv: &k.kgemv,
        g32: &k.g32,
        flash: &k.flash,
        mma: false,
        out_gate: None,
        hidden,
        eps,
        n_kv: 1,
        ctx: 1,
    }
}

impl RoutedFfn for Routed {
    fn enqueue_step(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let g = sites(cx, &self.k, (self.hidden, self.eps));
        self.moe
            .enqueue_step((cx.gpu, cx.w, &g), &mut self.bufs, l, x, out)
    }

    fn enqueue_front(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        lat: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let g = sites(cx, &self.k, (self.hidden, self.eps));
        self.moe
            .enqueue_front((cx.gpu, cx.w, &g), &mut self.bufs, l, x, lat)
    }

    fn route(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<f32>) {
        self.bufs.route()
    }

    fn enqueue_shadow(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        lat: &DeviceBuffer<f32>,
        sel: &DeviceBuffer<u32>,
        n_card: usize,
    ) -> Result<bool, GpuError> {
        let g = sites(cx, &self.k, (self.hidden, self.eps));
        self.moe
            .enqueue_shadow((cx.gpu, cx.w, &g), &mut self.bufs, l, x, lat, (sel, n_card))
    }

    fn enqueue_back(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        card_hsum: (bool, Option<&DeviceBuffer<f32>>),
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let g = sites(cx, &self.k, (self.hidden, self.eps));
        self.moe
            .enqueue_back((cx.gpu, cx.w, &g), &mut self.bufs, l, card_hsum, out)
    }

    fn enqueue_prompt(
        &mut self,
        cx: Cx<'_>,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let g = sites(cx, &self.k, (self.hidden, self.eps));
        self.moe.enqueue_prompt(
            (cx.gpu, cx.w, &g, &self.k.gemm),
            &mut self.bufs,
            l,
            m,
            x,
            out,
        )
    }

    fn rows(&self) -> usize {
        self.bufs.rows()
    }

    fn check_card_stacks(&self, w: &Weights, l: usize, on_card: usize) -> Result<(), GpuError> {
        moe::check_card_stacks(w, self.moe.dims(), l, on_card)
    }

    fn host_layer(&self, src: R8Source<'_>, l: usize) -> Result<HostLayer, ModelError> {
        moe::host_layer(src, self.moe.dims(), l)
    }

    fn bytes(&self) -> usize {
        self.bufs.bytes()
    }
}
