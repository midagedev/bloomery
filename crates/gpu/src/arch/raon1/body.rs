//! The Raon-1 `ChainBody` over one sequence on one card: the file's keys read
//! once through [`Hparams`], each layer's sites, router and stores resolved
//! once at load, the step's buffers allocated once, and the token and
//! position refreshed before every enqueue or replay.
//!
//! A step (llama.cpp `src/models/raon1.cpp`): the embedding row repeated into
//! the [`STREAMS`] residual streams; every layer then runs its two
//! sub-layers through Raon_mHC ([`super::mhc`]): the route of the streams,
//! their fold into the sub-layer's input, the sub-layer, the post mix of the
//! streams and its output. The attention sub-layer is an RDA layer
//! ([`super::rda`]) or a latent one ([`super::mla`]), each applying its own
//! `attn_norm`; the feed-forward one is `ffn_norm` then the dense or routed
//! SiTU block ([`super::ffn`]). The head reads the streams' mean.
//!
//! A pass of `m` rows ([`Run::Rows`]: a verify's or a prompt chunk's,
//! [`super::rows`]) is the same walk over the rows the staging holds: the
//! embedding of their ids, the mHC kernels, norms and SiTU over every row,
//! the RDA and latent layers' row forms and the feed-forward block's
//! ([`ffn::ffn_rows`]), each row the one-row step's bits. A walk of one row
//! runs the step's own launches.

use super::ffn::{self, Ffn, FfnKernels, FfnRows, FfnScratch, Glu, Routed};
use super::mhc::{self, HC_OUT, MhcKernels, STREAMS, SiteVecs};
use super::mla::{self, MlaBufs, MlaDims, MlaKernels, MlaSites};
use super::rda::{self, RdaSites, RowKernels, RowsKit};
use super::rows::{RowState, RowViews};
use super::state::{self, PASS_ROWS, Pass};
use super::wide::{self, WIDE_ROWS, Wide, WideDims};
use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Site};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::FlashGqaKernels;
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::kquant::Act;
use crate::linear::mixer_kda::{KdaBufs, KdaDims};
use crate::linear::{HEAD, LinearKernels};
use crate::model::{ChainBody, GpuModel, NoHost, block_count};
use crate::recur::RecStore;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::tensor::Window;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::KdaGate;
use model::arch::raon1::hparams::{Hparams, Kind};
use model::arch::raon1::names::{self, Sub};
use model::placement::CardFormat;
use model::placement::kernels;
use std::ops::Range;

const WHAT: &str = "raon1::Body::load";
const STEP: &str = "raon1::Body::enqueue_chain";

/// The types a dense or shared-expert projection may have.
const FFN_TYPES: &[SiteTy] = &[
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::Q5_0,
];

/// A layer's attention sub-layer and its store.
enum Mixer {
    /// The recurrent state's lanes, their stamps and the conv ring
    /// ([`state::rda_store`]).
    Rda {
        sites: Box<RdaSites>,
        store: RecStore,
    },
    /// The latent cache, `[ctx][row]` f16.
    Mla {
        sites: Box<MlaSites>,
        cache: DeviceTensor<u16>,
    },
}

/// One layer: its attention sub-layer, its feed-forward block and the names
/// of its two hyper-connection sites' vectors (`fn`, `scale`, `base`).
struct Layer {
    mixer: Mixer,
    ffn: Ffn,
    hc: [[String; 3]; 2],
}

impl Layer {
    /// The attention sub-layer's sites.
    fn mixer_sites(&self) -> Vec<&Site> {
        match &self.mixer {
            Mixer::Rda { sites, .. } => sites.sites().to_vec(),
            Mixer::Mla { sites, .. } => sites.sites().to_vec(),
        }
    }

    fn sites(&self) -> Vec<&Site> {
        let mut out = self.mixer_sites();
        match &self.ffn {
            Ffn::Dense(g) => out.extend(g.sites()),
            Ffn::Routed(r) => out.extend(r.shared.sites()),
        }
        out
    }

    fn store_bytes(&self) -> usize {
        match &self.mixer {
            Mixer::Rda { store, .. } => store.bytes(),
            Mixer::Mla { cache, .. } => cache.buf().num_bytes(),
        }
    }
}

/// The kernels the step launches beyond the `Gpu`'s own.
struct Kernels {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    linear: LinearKernels,
    rows: RowKernels,
    hc: MhcKernels,
    ffn: FfnKernels,
    mla: MlaKernels,
}

impl Kernels {
    fn load(gpu: &Gpu, mla: &MlaDims) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            rows: RowKernels::load(ctx)?,
            hc: MhcKernels::load(ctx)?,
            ffn: FfnKernels::load(gpu)?,
            mla: MlaKernels::load(gpu, mla)?,
        })
    }
}

/// The RDA and latent layers' widths `hp` names, or a named refusal of a
/// shape the card does not run: the softplus decay, a latent row of another
/// width.
fn dims(hp: &Hparams) -> Result<(KdaDims, MlaDims), GpuError> {
    let KdaGate::Sigmoid { lower_bound } = hp.gate else {
        return Err(GpuError::shape(
            WHAT,
            "the RDA layers' decay is the softplus form; the card runs the bounded sigmoid \
             (kda.gate_lower_bound)",
        ));
    };
    let n = hp.n_embd;
    let kda = rda::dims(n, hp.n_head, hp.rms_eps, lower_bound);
    kda.rule.check(WHAT)?;
    let mla = MlaDims {
        embd: n,
        heads: hp.n_head,
        q_lora: hp.mla.q_lora,
        nope: hp.mla.nope(),
        extra: hp.mla.pe,
        v_head: hp.mla.head_v,
        rms_eps: hp.rms_eps,
        latent_eps: hp.latent_eps,
    };
    if mla.row() != hp.mla.row() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the latent row is {} values in the file; the card's latent layer caches {}",
                hp.mla.row(),
                mla.row()
            ),
        ));
    }
    Ok((kda, mla))
}

/// The widths a wide unit is cut for, or a named refusal of a shape the
/// wide arm's forms do not cover: the two mixers' output projections read
/// one width, the routed block's slots a card sum of the router's pick count.
fn wide_dims(hp: &Hparams, kda: &KdaDims, mla: &MlaDims) -> Result<WideDims, GpuError> {
    let attn = kda.shape.n_v * HEAD;
    if attn != mla.heads * mla.v_head {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the RDA layers' output projection reads {attn} values, the latent layers' {}: the wide arm holds one form of its input",
                mla.heads * mla.v_head
            ),
        ));
    }
    let e = hp.experts;
    Ok(WideDims {
        embd: hp.n_embd,
        attn,
        q_lora: hp.mla.q_lora,
        heads: mla.heads,
        dense_ff: hp.dense_ff,
        shared_ff: e.shared_ff(),
        expert_ff: e.ff,
        ff_rows: hp.dense_ff.max(ffn::sigmoid_128_6::N_USED * e.ff),
    })
}

/// A walk's buffers, `rows` rows of each width, token-major: the step's
/// one row, or a pass's [`PASS_ROWS`].
pub(super) struct Scratch {
    /// The streams the layer reads and the ones it writes, `STREAMS · n` a
    /// row.
    x: DeviceBuffer<f32>,
    x2: DeviceBuffer<f32>,
    /// The sub-layer's route ([`HC_OUT`] values a row) and its partial sums
    /// ([`mhc::parts_len`] a row).
    hc: DeviceBuffer<f32>,
    part: DeviceBuffer<f32>,
    /// The fold, its norm and the sub-layer's output, `n` a row each.
    u: DeviceBuffer<f32>,
    un: DeviceBuffer<f32>,
    br: DeviceBuffer<f32>,
    /// The RDA mixer's: qkv, fa, ga, beta_raw, write_raw, f, z, conv, beta,
    /// decay, o, gated.
    kda: [DeviceBuffer<f32>; 12],
    /// The latent layer's: qa, qr, kv, gate, q, qabs, att, av, gated,
    /// part_v, part_ms.
    mla: [DeviceBuffer<f32>; 11],
}

/// The f32 lengths of a [`Scratch`]'s buffers, in its fields' order: the
/// streams, the route and the sub-layer's rows, the RDA mixer's twelve, the
/// latent layer's eleven. Each buffer holds at least one value.
type ScratchLens = ([usize; 7], [usize; 12], [usize; 11]);

impl Scratch {
    /// The lengths of a walk's buffers of `rows` rows over a latent cache of
    /// `ctx` rows: the one formula [`Scratch::new`] allocates and
    /// [`Body::pass_reserve`] counts. Refused by name when the streams are
    /// not whole route pieces ([`mhc::parts_len`]).
    fn lens(
        hp: &Hparams,
        (d, m): (&KdaDims, &MlaDims),
        ctx: usize,
        rows: usize,
    ) -> Result<ScratchLens, GpuError> {
        let z = |n: usize| (rows * n).max(1);
        let n = hp.n_embd;
        let parts = mhc::parts_len(n).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "n_embd {n}: the route takes the streams' {} values in {}-value pieces",
                    STREAMS * n,
                    mhc::HC_PIECE
                ),
            )
        })?;
        let lin = d.shape;
        let (channels, v, n_v) = (lin.channels(), lin.n_v * HEAD, lin.n_v);
        // The latent attention takes at most a pass's rows a launch: a wider
        // unit runs it in chunks that share one set of partials.
        let (part_v, part_ms) = mla::partials_lens_rows(m, ctx, rows.min(PASS_ROWS));
        let (heads, row) = (m.heads, m.row());
        Ok((
            [
                z(STREAMS * n),
                z(STREAMS * n),
                z(HC_OUT),
                z(parts),
                z(n),
                z(n),
                z(n),
            ],
            [
                z(channels),
                z(hp.decay_rank),
                z(hp.decay_rank),
                z(n_v),
                z(n_v),
                z(v),
                z(v),
                z(channels),
                z(n_v),
                z(v),
                z(v),
                z(v),
            ],
            [
                z(m.q_lora),
                z(m.q_lora),
                z(row),
                z(heads * m.v_head),
                z(heads * m.head_k()),
                z(heads * row),
                z(heads * hp.mla.kv_lora),
                z(heads * m.v_head),
                z(heads * m.v_head),
                part_v.max(1),
                part_ms.max(1),
            ],
        ))
    }

    /// Card bytes of the buffers [`Scratch::lens`] names, before any load.
    fn bytes_of(lens: &ScratchLens) -> usize {
        let (a, b, c) = lens;
        let values: usize = a.iter().chain(b).chain(c).sum();
        values * std::mem::size_of::<f32>()
    }

    fn new(
        stream: &CudaStream,
        hp: &Hparams,
        dims: (&KdaDims, &MlaDims),
        ctx: usize,
        rows: usize,
    ) -> Result<Scratch, GpuError> {
        fn zeroed<const N: usize>(
            stream: &CudaStream,
            lens: [usize; N],
        ) -> Result<[DeviceBuffer<f32>; N], GpuError> {
            let mut out = Vec::with_capacity(N);
            for n in lens {
                out.push(DeviceBuffer::<f32>::zeroed(stream, n)?);
            }
            Ok(out
                .try_into()
                .unwrap_or_else(|_| unreachable!("{N} buffers made of {N} lengths")))
        }
        let (streams, kda, mla) = Scratch::lens(hp, dims, ctx, rows)?;
        let [x, x2, hc, part, u, un, br] = zeroed(stream, streams)?;
        Ok(Scratch {
            x,
            x2,
            hc,
            part,
            u,
            un,
            br,
            kda: zeroed(stream, kda)?,
            mla: zeroed(stream, mla)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        [
            &self.x, &self.x2, &self.hc, &self.part, &self.u, &self.un, &self.br,
        ]
        .into_iter()
        .chain(self.kda.iter())
        .chain(self.mla.iter())
        .map(DeviceBuffer::num_bytes)
        .sum()
    }
}

/// The step's input, rewritten by every refresh: the token once a stream
/// (the embedding writes the streams in one launch), the position, and the
/// embedding's per-row position and live key count, row 0's the step's.
struct StepIn {
    ids: DeviceBuffer<u32>,
    pos0: DeviceBuffer<u32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
}

impl StepIn {
    fn new(stream: &CudaStream) -> Result<StepIn, GpuError> {
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        Ok(StepIn {
            ids: u(STREAMS)?,
            pos0: u(1)?,
            pos: u(STREAMS)?,
            n_keys: u(STREAMS)?,
        })
    }

    fn bytes(&self) -> usize {
        [&self.ids, &self.pos0, &self.pos, &self.n_keys]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum()
    }
}

/// What a walk runs: the step's one row from its refreshed input, or `m`
/// rows of the staged pass from staging row `at`
/// ([`RowState`](super::rows::RowState)), their RDA states as `pass` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Run {
    Step,
    Rows { at: usize, m: usize, pass: Pass },
}

/// Where a walk's rows go past the last layer.
pub(super) enum Tail<'h> {
    /// Every row's mean of the streams into `head`, a head of the walk's
    /// rows, then the head.
    Head(&'h mut Head),
    /// The last row's mean into `head`, a head of one row, then the head: a
    /// prompt call's last chunk.
    Last(&'h mut Head),
    /// No head: a prompt chunk before the call's last.
    Open,
}

/// A walk's feed-forward buffers: the step's, or a pass's rows'.
enum FfnBufs<'a> {
    Step(&'a mut FfnScratch),
    Rows(&'a mut FfnRows),
}

/// The Raon-1 chain over one sequence on one card.
pub struct Body {
    hp: Hparams,
    kda: KdaDims,
    mla: MlaDims,
    act: Act,
    layers: Vec<Layer>,
    k: Kernels,
    acts: Acts,
    /// The step's one row.
    s: Scratch,
    step_in: StepIn,
    ffn_s: FfnScratch,
    /// A pass's rows, and the stage a row-major site's columns land in
    /// before their copy into token-major rows ([`RowsKit`]).
    pass: Scratch,
    ffn_rows: FfnRows,
    stage: DeviceBuffer<f32>,
    /// A wide unit's rows, forms and kernels ([`super::wide`]).
    pub(super) wide: Wide,
    /// The passes' staging, the lane word, the taps and the position the
    /// stores stand at.
    pub(super) rows: RowState,
    /// The token once a stream, rewritten by every refresh.
    ids: [u32; STREAMS],
    ctx: usize,
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format (the F16 ones widened to
    /// f32), the latent layers' caches of `ctx` rows, the RDA layers' stores
    /// of one lane and the walks' buffers. The model takes the file and
    /// closes it once the weights are resident. A verify of drafted rows
    /// takes [`GpuModel::open_verifying`]'s load.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        GpuModel::open_lanes(gpu, file, ctx, 1)
    }

    /// [`GpuModel::open`] with [`PASS_ROWS`] lanes an RDA store, so a
    /// verify of up to that many rows keeps any `1..=m` of them
    /// ([`super::state`]); the steps and prompt calls are the plain load's,
    /// bit for bit.
    pub fn open_verifying(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        GpuModel::open_lanes(gpu, file, ctx, PASS_ROWS)
    }

    fn open_lanes(
        gpu: Gpu,
        file: Split,
        ctx: usize,
        lanes: usize,
    ) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "raon1 GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
            for l in (0..hp.n_layer).filter(|&l| hp.kind(l) == Kind::Rda) {
                rda::derive(gpu.stream(), w, l)?;
            }
            for l in (0..hp.n_layer).filter(|&l| !hp.dense(l)) {
                down_words(gpu.stream(), &file, w, &names::ffn_down_exps(l))?;
            }
            Body::load(gpu, &file, w, (hp, ctx), lanes)
        })
    }
}

/// The routed down stack `name` in the layout the card leg reads it in
/// ([`LegStacks::of_resident`](crate::card_leg::LegStacks::of_resident)): a
/// type whose down entry reads the file's blocks as words, and which
/// [`Weights::load`] laid out otherwise (Q8_0's planes), is uploaded again
/// as words. Load-time only.
fn down_words(
    stream: &CudaStream,
    file: &Split,
    w: &mut Weights,
    name: &str,
) -> Result<(), GpuError> {
    let (_, t) = file
        .find(name)
        .ok_or_else(|| GpuError::tensor(WHAT, name, "a routed down stack of the file"))?;
    if kernels::card_down_runs(t.ty) && CardFormat::of(t.ty) != Some(CardFormat::KQuant) {
        w.relayout(stream, file, name, CardFormat::KQuant)?;
    }
    Ok(())
}

/// The site of `name` at `(rows, k)`, its type read from the file among
/// `allowed` and held to the resident weight.
fn typed(
    file: &Split,
    w: &Weights,
    name: String,
    (rows, k): (usize, usize),
    allowed: &[SiteTy],
) -> Result<Site, GpuError> {
    let ty = site::file_site(file, WHAT, &name, (rows, k), allowed)?;
    site::site(w, WHAT, &name, (rows, k), ty)?;
    Ok(Site::new(name, ty, k))
}

/// The resident f32 vector or matrix `name`, held to `len` values.
fn exact(w: &Weights, name: String, len: usize) -> Result<String, GpuError> {
    let got = w.f32_buf(WHAT, &name)?.len();
    if got != len {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} holds {got} values; the step reads {len}"),
        ));
    }
    Ok(name)
}

impl Body {
    /// Card bytes of the walk buffers of the rows pass ([`PASS_ROWS`] rows, the
    /// latent layers' attention partials among them) and of a wide unit
    /// ([`WIDE_ROWS`] rows, its forms and feed-forward rows), which every load of
    /// the file `hp` describes makes beside a plan's terms: what its plan
    /// keeps out of the expert budget (`PlanInputs::with_prompt_reserve`).
    /// Counted at a cache taller than one attention segment, the most any
    /// context takes. Refused as the load refuses `hp`.
    pub fn pass_reserve(hp: &Hparams) -> Result<u64, GpuError> {
        let (kda, mla) = dims(hp)?;
        let lens = Scratch::lens(hp, (&kda, &mla), usize::MAX, PASS_ROWS)?;
        let wide_lens = Scratch::lens(hp, (&kda, &mla), usize::MAX, WIDE_ROWS)?;
        let wide = Scratch::bytes_of(&wide_lens) + Wide::bytes_of(&wide_dims(hp, &kda, &mla)?);
        Ok((Scratch::bytes_of(&lens) + wide) as u64)
    }

    /// The last hyper-connection site's route as the step left it: the last
    /// layer's feed-forward site, `pre[4]`, `post[4]`, then `comb[16]` at
    /// `src · 4 + dst` ([`HC_OUT`] values). Synchronizes the stream.
    pub fn last_route(&self, stream: &CudaStream) -> Result<Vec<f32>, GpuError> {
        Ok(self.s.hc.to_host_vec(stream)?)
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        (hp, ctx): (Hparams, usize),
        lanes: usize,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let n = hp.n_embd;
        if hp.hc.streams != STREAMS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "hyper_connection.count is {}; the card's mHC runs {STREAMS} streams",
                    hp.hc.streams
                ),
            ));
        }
        let e = hp.experts;
        if e.n_expert != ffn::sigmoid_128_6::N_EXPERT
            || e.n_used != ffn::sigmoid_128_6::N_USED
            || !e.weights_norm
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the router is {} experts, top {}, renormalized {}; the card's router is \
                     {} experts, top {}, renormalized",
                    e.n_expert,
                    e.n_used,
                    e.weights_norm,
                    ffn::sigmoid_128_6::N_EXPERT,
                    ffn::sigmoid_128_6::N_USED
                ),
            ));
        }
        let (kda, mla) = dims(&hp)?;
        let act = Act::Situ {
            beta: hp.situ.gate_beta,
            linear_beta: hp.situ.value_beta,
        };
        act.code(WHAT)?;
        let glu = |l: usize, (gate, up, down): (String, String, String), ff: usize| {
            Ok::<Glu, GpuError>(Glu {
                gate: typed(file, w, gate, (ff, n), FFN_TYPES)?,
                up: typed(file, w, up, (ff, n), FFN_TYPES)?,
                down: typed(file, w, down, (n, ff), FFN_TYPES)?,
                ff,
            })
            .map_err(|err| GpuError::shape(WHAT, format!("layer {l}'s feed-forward block: {err}")))
        };
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mixer = match hp.kind(l) {
                Kind::Rda => Mixer::Rda {
                    sites: Box::new(RdaSites::load(file, w, &kda, l)?),
                    store: state::rda_store(stream, kda.shape, lanes)?,
                },
                Kind::Mla => Mixer::Mla {
                    sites: Box::new(MlaSites::load(file, w, &mla, l)?),
                    cache: DeviceTensor::zeroed(stream, ctx, mla.row())?,
                },
            };
            let ffn = if hp.dense(l) {
                Ffn::Dense(glu(
                    l,
                    (names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)),
                    hp.dense_ff,
                )?)
            } else {
                Ffn::Routed(Routed {
                    router: exact(w, names::ffn_gate_inp(l), e.n_expert * n)?,
                    bias: exact(w, names::exp_probs_b(l), e.n_expert)?,
                    stacks: [
                        names::ffn_gate_exps(l),
                        names::ffn_up_exps(l),
                        names::ffn_down_exps(l),
                    ],
                    shared: glu(
                        l,
                        (
                            names::ffn_gate_shexp(l),
                            names::ffn_up_shexp(l),
                            names::ffn_down_shexp(l),
                        ),
                        e.shared_ff(),
                    )?,
                })
            };
            let hc = [Sub::Attn, Sub::Ffn].map(|sub| {
                [
                    names::hc_fn(l, sub),
                    names::hc_scale(l, sub),
                    names::hc_base(l, sub),
                ]
            });
            for [f, scale, base] in &hc {
                exact(w, f.clone(), hp.hc.mix_dim() * STREAMS * n)?;
                exact(w, scale.clone(), 4)?;
                exact(w, base.clone(), hp.hc.mix_dim())?;
            }
            exact(w, names::ffn_norm(l), n)?;
            layers.push(Layer { mixer, ffn, hc });
        }
        let mut acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        rda::add_row_forms(
            stream,
            &mut acts,
            layers.iter().flat_map(Layer::mixer_sites),
        )?;
        let ff = (hp.dense_ff.max(e.shared_ff()), e.ff);
        let stage = rda::stage_len(&kda)
            .max(mla::stage_len(&mla))
            .max(ffn::stage_len(ff.0.max(ff.1)));
        Ok(Body {
            k: Kernels::load(gpu, &mla)?,
            s: Scratch::new(stream, &hp, (&kda, &mla), ctx, 1)?,
            step_in: StepIn::new(stream)?,
            ffn_s: FfnScratch::new(stream, n, ff.0, ff.1)?,
            pass: Scratch::new(stream, &hp, (&kda, &mla), ctx, PASS_ROWS)?,
            ffn_rows: FfnRows::new(stream, n, ff, PASS_ROWS)?,
            stage: DeviceBuffer::zeroed(stream, stage)?,
            wide: Wide::new(
                gpu,
                &wide_dims(&hp, &kda, &mla)?,
                Scratch::new(stream, &hp, (&kda, &mla), ctx, WIDE_ROWS)?,
            )?,
            rows: RowState::new(stream, lanes)?,
            kda,
            mla,
            act,
            layers,
            acts,
            ids: [0; STREAMS],
            ctx,
            hp,
        })
    }

    /// Device bytes the body holds past the weights and the stores.
    fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.step_in.bytes()
            + self.ffn_s.bytes()
            + self.pass.bytes()
            + self.ffn_rows.bytes()
            + self.stage.num_bytes()
            + self.wide.bytes()
            + self.rows.bytes()
            + self.acts.bytes()
            + self.k.kgemv.device_bytes()
    }

    /// Device bytes of the sequence's stores: the RDA layers' lanes and
    /// rings, the latent layers' caches.
    pub(super) fn store_bytes(&self) -> usize {
        self.layers.iter().map(Layer::store_bytes).sum()
    }

    /// The layers a tap may name and the floats of one tapped row of a
    /// layer: the streams, `STREAMS · n`.
    pub(super) fn tap_shape(&self) -> (usize, usize) {
        (self.layers.len(), STREAMS * self.hp.n_embd)
    }

    /// One walk (module doc): the embedding of `run`'s ids into the
    /// streams, every layer — the rows entering a tapped layer laid into the
    /// taps first — then `tail`.
    pub(super) fn walk(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        run: Run,
        tail: Tail<'_>,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let Body {
            hp,
            kda: kd,
            mla: md,
            act,
            layers,
            k,
            acts,
            s,
            step_in,
            ffn_s,
            pass: ps,
            ffn_rows,
            stage,
            rows,
            wide,
            ctx,
            ..
        } = self;
        let (n, hcf) = (hp.n_embd, (hp.hc.eps, hp.hc.magnitude));
        let cx = GqaCx {
            what: STEP,
            gpu,
            w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash,
            mma: false,
            out_gate: None,
            hidden: n,
            eps: hp.rms_eps,
            n_kv: 1,
            ctx: *ctx,
        };
        let embd = names::token_embd();
        let Some(DevWeight::KQuant { ty, w: table, .. }) = w.get(&embd) else {
            return Err(GpuError::tensor(STEP, &embd, "K-quant rows"));
        };
        let RowState {
            rin, lanes, taps, ..
        } = rows;
        let pos_w: Window<'_, u32>;
        let keys_w: Window<'_, u32>;
        let mut wide_kit = None;
        let (sc, mut ffn_b, pos, keys, (at, m, pass)) = match run {
            Run::Step => {
                gpu.elem().enqueue_embed_rows_kquant(
                    stream,
                    *ty,
                    EmbedRowsArgs {
                        w: table,
                        ids: &step_in.ids,
                        pos0: &step_in.pos0,
                        first: 0,
                        y: &mut s.x,
                        pos: &mut step_in.pos,
                        n_keys: &mut step_in.n_keys,
                    },
                )?;
                let (pos, keys) = (&step_in.pos, &step_in.n_keys);
                (s, FfnBufs::Step(ffn_s), pos, keys, (0, 1, Pass::InPlace))
            }
            Run::Rows { at, m, pass } if m > PASS_ROWS => {
                if pass != Pass::InPlace {
                    return Err(GpuError::shape(
                        STEP,
                        format!("a {pass:?} pass of {m} rows: only a prompt's units run wide"),
                    ));
                }
                wide.route_dense(gpu, m)?;
                let (kit, ws, wrin) = wide.kit(&k.g32, &k.ffn.router);
                let RowViews {
                    ids,
                    pos,
                    keys,
                    epos,
                    ekeys,
                } = wrin.views(at, m)?;
                gpu.elem().enqueue_embed_rows_kquant(
                    stream,
                    *ty,
                    EmbedRowsArgs {
                        w: table,
                        ids: &ids,
                        pos0: &pos,
                        first: 0,
                        y: &mut ws.x,
                        pos: epos,
                        n_keys: ekeys,
                    },
                )?;
                pos_w = pos;
                keys_w = keys;
                wide_kit = Some(kit);
                (
                    ws,
                    FfnBufs::Rows(ffn_rows),
                    &*pos_w,
                    &*keys_w,
                    (at, m, pass),
                )
            }
            Run::Rows { at, m, pass } => {
                let RowViews {
                    ids,
                    pos,
                    keys,
                    epos,
                    ekeys,
                } = rin.views(at, m)?;
                // The embedding's own positions are its rows', a stream each:
                // they land in buffers no layer reads; the layers read the
                // staged ones.
                gpu.elem().enqueue_embed_rows_kquant(
                    stream,
                    *ty,
                    EmbedRowsArgs {
                        w: table,
                        ids: &ids,
                        pos0: &pos,
                        first: 0,
                        y: &mut ps.x,
                        pos: epos,
                        n_keys: ekeys,
                    },
                )?;
                pos_w = pos;
                keys_w = keys;
                (
                    ps,
                    FfnBufs::Rows(ffn_rows),
                    &*pos_w,
                    &*keys_w,
                    (at, m, pass),
                )
            }
        };
        let Scratch {
            x,
            x2,
            hc,
            part,
            u,
            un,
            br,
            kda,
            mla: mla_b,
        } = sc;
        let mhc = &k.hc;
        for (l, layer) in layers.iter_mut().enumerate() {
            if let Some((t, slot)) = taps.as_mut().and_then(|t| t.slot(l).map(|i| (t, i))) {
                t.enter_at(stream, slot, x, (at, m))?;
            }
            let [attn_hc, ffn_hc] = &layer.hc;
            let vecs = |[f, scale, base]: &[String; 3]| -> Result<SiteVecs<'_>, GpuError> {
                Ok(SiteVecs {
                    w: w.f32_buf(STEP, f)?,
                    scale: w.f32_buf(STEP, scale)?,
                    base: w.f32_buf(STEP, base)?,
                })
            };
            mhc.enqueue_route(
                stream,
                &vecs(attn_hc)?,
                x,
                (n, m),
                hcf,
                (&mut *part, &mut *hc),
            )?;
            mhc.enqueue_fold(stream, x, hc, (n, m), u)?;
            match &mut layer.mixer {
                Mixer::Rda { sites, store } => {
                    let [
                        qkv,
                        fa,
                        ga,
                        beta_raw,
                        write_raw,
                        f,
                        z,
                        conv,
                        beta,
                        decay,
                        o,
                        gated,
                    ] = &mut *kda;
                    // The pass's own kind, not the lane book's: a verify's
                    // capture records it before its plan opens it.
                    let mut st = lanes.state(store)?;
                    st.each = pass == Pass::Verify;
                    let bufs = KdaBufs {
                        x: u,
                        pos,
                        xn: un,
                        qkv,
                        fa,
                        ga,
                        beta_raw,
                        write_raw: Some(write_raw),
                        f,
                        z,
                        conv,
                        beta,
                        decay,
                        o,
                        gated,
                        out: br,
                    };
                    if m == 1 {
                        rda::step(&cx, &k.linear, acts, sites, (l, &*kd), st, bufs)?;
                    } else if let Some(kit) = wide_kit.as_mut() {
                        wide::rda_rows(&cx, kit, &k.linear, sites, (l, &*kd, m), st, bufs)?;
                    } else {
                        let kit = RowsKit {
                            linear: &k.linear,
                            rows: &k.rows,
                            stage: &mut *stage,
                        };
                        rda::rows(&cx, kit, acts, sites, (l, &*kd, m), st, bufs)?;
                    }
                }
                Mixer::Mla { sites, cache } => {
                    let [qa, qr, kv, gate, q, qabs, att, av, gated, part_v, part_ms] = &mut *mla_b;
                    let bufs = MlaBufs {
                        x: u,
                        pos,
                        n_keys: keys,
                        xn: un,
                        qa,
                        qr,
                        kv,
                        gate,
                        q,
                        qabs,
                        att,
                        av,
                        gated,
                        part_v,
                        part_ms,
                        out: br,
                        cache,
                    };
                    if m == 1 {
                        mla::step(&cx, &k.mla, acts, (&**sites, &*md), l, bufs)?;
                    } else if let Some(kit) = wide_kit.as_mut() {
                        mla::rows_wide(&cx, &k.mla, kit, (&**sites, &*md), (l, m), bufs)?;
                    } else {
                        let kit = RowsKit {
                            linear: &k.linear,
                            rows: &k.rows,
                            stage: &mut *stage,
                        };
                        mla::rows(&cx, &k.mla, acts, (&**sites, &*md), (l, m), bufs, kit)?;
                    }
                }
            }
            mhc.enqueue_post(stream, (x, br, hc), (n, m), x2)?;
            mhc.enqueue_route(
                stream,
                &vecs(ffn_hc)?,
                x2,
                (n, m),
                hcf,
                (&mut *part, &mut *hc),
            )?;
            mhc.enqueue_fold(stream, x2, hc, (n, m), u)?;
            let gain = w.f32_buf(STEP, &names::ffn_norm(l))?;
            gpu.elem()
                .enqueue_rms_norm(stream, u, gain, hp.rms_eps, n, m, un)?;
            let rule = (*act, hp.experts.weights_scale);
            let at_l = (l, &layer.ffn);
            if let Some(kit) = wide_kit.as_mut() {
                wide::ffn(&cx, kit, rule, at_l, m, un, br)?;
            } else {
                match &mut ffn_b {
                    FfnBufs::Step(f) => ffn::ffn(&cx, &k.ffn, acts, rule, at_l, un, f, br)?,
                    FfnBufs::Rows(f) => {
                        let kit = RowsKit {
                            linear: &k.linear,
                            rows: &k.rows,
                            stage: &mut *stage,
                        };
                        ffn::ffn_rows(&cx, &k.ffn, acts, rule, at_l, m, un, f, br, kit)?
                    }
                }
            }
            mhc.enqueue_post(stream, (x2, br, hc), (n, m), x)?;
        }
        match tail {
            Tail::Head(head) => {
                if head.m() != m {
                    return Err(GpuError::shape(
                        STEP,
                        format!("a walk of {m} rows into a head of {}", head.m()),
                    ));
                }
                mhc.enqueue_mean(stream, x, (n, m), head.input_mut())?;
                head.enqueue(gpu, w)
            }
            Tail::Last(head) => {
                let row = STREAMS * n;
                let last = Window::<f32>::of(&*x, (m - 1) * row * size_of::<f32>(), row)?;
                mhc.enqueue_mean(stream, &last, (n, 1), head.input_mut())?;
                head.enqueue(gpu, w)
            }
            Tail::Open => Ok(()),
        }
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = NoHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::Raon1
    }

    /// The step at `pos`; a position past the caches is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(GpuError::shape(
                "raon1::Body::decode_input",
                format!("a step at position {pos} in caches of {}", self.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// The token once a stream and the position: two copies. Refused by
    /// name, before either, while a verify waits for its commit or off the
    /// position the stores stand at ([`RowState::step_at`]).
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        self.rows.step_at(input.pos)?;
        self.ids = [input.token; STREAMS];
        self.step_in.ids.copy_from_host(stream, &self.ids)?;
        self.step_in.pos0.copy_from_host(stream, &[input.pos])?;
        self.rows.held = input.pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.walk(gpu, w, Run::Step, Tail::Head(head))
    }

    /// Every RDA store back to a fresh one in place (a captured chain keeps
    /// their addresses, [`state::clear_store`]), the lane word at lane 0 and
    /// any verify waiting dropped. The latent caches need nothing: no row at
    /// or past the live count is read, and every row below it is written by
    /// its own walk first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.layers {
            if let Mixer::Rda { store, .. } = &mut layer.mixer {
                state::clear_store(stream, store)?;
            }
        }
        self.rows.reset(stream)
    }

    /// `attention.layer_norm_rms_epsilon`: every block norm's, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes() + self.scratch_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}
