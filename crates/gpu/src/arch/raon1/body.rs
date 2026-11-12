//! The Raon-1 `ChainBody` over one sequence on one card, both dialects' files
//! (`src/models/raon1.cpp` and `src/models/raon1nano.cpp`): the file's keys
//! read once through [`Hparams`], each layer's sites, router and stores
//! resolved once at load — the Nano file's fused stems split into the buffers
//! the split dialect's sites read ([`super::rda::fold`]) — the step's buffers
//! allocated once, and the token and position refreshed before every enqueue
//! or replay.
//!
//! A step: the embedding row repeated into the [`STREAMS`] residual streams;
//! every layer then runs its two sub-layers through Raon_mHC ([`super::mhc`],
//! the file's dialect's route): the route of the streams, their fold into the
//! sub-layer's input, the sub-layer, the post mix of the streams and its
//! output. The attention sub-layer is an RDA layer ([`super::rda`]) or a
//! latent one ([`super::mla`]), each applying its own `attn_norm`; the
//! feed-forward one is `ffn_norm` then the dense block of the description's
//! activation or the routed SiTU one ([`crate::arch::routed_ffn`]). A file
//! with a per-layer input ([`super::ple`]) adds its gated sum to every stream
//! after the feed-forward post-mix. The head reads the streams' mean — the
//! token table's own rows when the file ties it.
//!
//! A pass of `m` rows ([`Run::Rows`]: a verify's or a prompt chunk's,
//! [`super::rows`]) is the same walk over the rows the staging holds: the
//! embedding of their ids, the mHC kernels, norms and the block's activation
//! over every row, the RDA and latent layers' row forms and the feed-forward
//! block's ([`ffn::ffn_rows`]), each row the one-row step's bits. A walk of
//! one row runs the step's own launches.

use super::gather;
use super::mhc::{self, CHART_LEN, ChartVecs, HC_OUT, MhcKernels, STREAMS, SiteVecs};
use super::mla;
use super::ple::{Ple, PleBufs, PleKernels};
use super::rda::{self, RdaSites, RowKernels, RowsKit};
use super::rows::{RowState, RowViews};
use super::state::{self, PASS_ROWS, Pass};
use super::wide::{self, WIDE_ROWS, Wide, WideDims};
use crate::arch::gated_mla::{self, MlaBufs, MlaSites};
use crate::arch::hybrid::{Acts, DecodeInput, GqaCx, Site};
use crate::arch::routed_ffn::{
    self as ffn, Ffn, FfnKernels, FfnRows, FfnScratch, Glu, Routed, RoutedShape,
};
use crate::flash_gqa::FlashGqaKernels;
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::kquant::Act;
use crate::latent_gated::{self, MlaDims, MlaKernels};
use crate::linear::mixer_kda::{KdaBufs, KdaDims};
use crate::linear::{HEAD, LinearKernels};
use crate::model::{ChainBody, GpuModel, NoHost};
use crate::q38::Q38Kernels;
use crate::recur::RecStore;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::tensor::Window;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::KdaGate;
use model::arch::noaux::Experts;
use model::arch::raon1::hparams::{Hparams, Kind};
use model::arch::raon1::names::{self, Sub};
use model::arch::raon1::roles::Dialect;
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
            Ffn::Routed(r) => out.extend(r.sites()),
        }
        out
    }

    fn store_bytes(&self) -> usize {
        match &self.mixer {
            Mixer::Rda { store, .. } => store.bytes(),
            Mixer::Mla { cache, .. } => cache.buf().num_bytes(),
        }
    }

    /// `Ok` when the wide arm runs layer `l`: neither its mixer's projections
    /// nor its feed-forward block of resident weights `w` meet a refusal the wide
    /// arm raises ([`Body::wide_ok`]).
    fn wide_runs(&self, l: usize, w: &Weights, d: &WideDims) -> Result<(), GpuError> {
        let mixer = match &self.mixer {
            Mixer::Rda { sites, .. } => wide::rda_runs(sites),
            Mixer::Mla { sites, .. } => mla::wide_runs(sites),
        };
        mixer.map_err(|refusal| GpuError::shape(WHAT, format!("layer {l}'s mixer: {refusal}")))?;
        wide::ffn_runs(l, &self.ffn, w, d)
    }
}

/// The first refusal the wide arm raises over `layers` in file order, or `None`
/// when it runs every one ([`Body::wide_ok`]).
fn wide_refusal(layers: &[Layer], w: &Weights, d: &WideDims) -> Option<String> {
    layers
        .iter()
        .enumerate()
        .find_map(|(l, layer)| layer.wide_runs(l, w, d).err())
        .map(|refusal| refusal.to_string())
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
    /// The Q8_0 row gathers a Nano file's tables take.
    q38: Q38Kernels,
    /// The per-layer input's two entries.
    ple: PleKernels,
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
            mla: MlaKernels::load(gpu, mla, WIDE_ROWS)?,
            q38: Q38Kernels::load(ctx)?,
            ple: PleKernels::load(ctx)?,
        })
    }
}

/// The RDA and latent layers' widths `hp` names, or a named refusal of a
/// shape the card does not run: the softplus decay, a latent row of another
/// width, latent layers of more than one head count (the one latent dims
/// sizes every latent layer's sites, kernels and caches).
fn dims(hp: &Hparams) -> Result<(KdaDims, MlaDims), GpuError> {
    let KdaGate::Sigmoid { lower_bound } = hp.gate else {
        return Err(GpuError::shape(
            WHAT,
            "the RDA layers' decay is the softplus form; the card runs the bounded sigmoid \
             (kda.gate_lower_bound)",
        ));
    };
    let latent: Vec<usize> = (0..hp.n_layer)
        .filter(|&l| hp.kind(l) == Kind::Mla)
        .map(|l| hp.head(l))
        .collect();
    let Some(&heads) = latent.first() else {
        return Err(GpuError::shape(
            WHAT,
            "no latent layer: the card runs both mixers (attention.head_count_kv)",
        ));
    };
    if latent.iter().any(|&h| h != heads) {
        let other = latent.iter().copied().find(|&h| h != heads).expect("one");
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the latent layers' head counts differ: {heads} and {other} \
                 (attention.head_count); the card holds one latent shape"
            ),
        ));
    }
    let n = hp.n_embd;
    let kda = rda::dims(n, hp.n_head, hp.rms_eps, lower_bound);
    kda.rule.check(WHAT)?;
    let mla = MlaDims {
        embd: n,
        heads,
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

/// The blocks' activation as the card runs it, from the description's
/// ([`Hparams::act`]): the Mini file's `situ` keys' own, the Nano row's
/// SwiGLU plain or clamped. Another activation is refused by name.
fn act_of(hp: &Hparams) -> Result<Act, GpuError> {
    let act = match hp.act {
        model::arch::models::Act::Situ {
            gate_beta: beta,
            value_beta: linear_beta,
        } => Act::Situ { beta, linear_beta },
        model::arch::models::Act::SwiGlu { limit: None } => Act::SiluMul,
        model::arch::models::Act::SwiGlu { limit: Some(limit) } => Act::SwigluClamp { limit },
        other => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the blocks' activation is {}; the card runs situ and swiglu",
                    other.what()
                ),
            ));
        }
    };
    act.code(WHAT)?;
    Ok(act)
}

/// The widths a wide unit is cut for and why a file of shapes the wide
/// arm's forms do not cover keeps every prompt on the rows pass
/// ([`Body::wide_off`]): the two mixers' output projections reading one
/// width, a file that routes no experts (the Nano dialect's dense blocks).
/// The dims a dense file names are its own — the routed widths zero — for
/// [`Wide::bytes_of`] and [`Body::pass_reserve`] to count; a file the reason
/// names holds no wide kit at all.
fn wide_dims(
    hp: &Hparams,
    kda: &KdaDims,
    mla: &MlaDims,
) -> Result<(WideDims, Option<String>), GpuError> {
    let attn = kda.shape.n_v * HEAD;
    if attn != mla.heads * mla.v_head {
        return Ok((
            wide_dims_of(hp, attn, mla, 0, 0, hp.dense_ff),
            Some(format!(
                "the RDA layers' output projection reads {attn} values, the latent layers' {}: \
                 the wide arm holds one form of its input",
                mla.heads * mla.v_head
            )),
        ));
    }
    let Some(e) = &hp.experts else {
        return Ok((
            wide_dims_of(hp, attn, mla, 0, 0, hp.dense_ff),
            Some(
                "the file routes no experts; the card's wide arm reads the mini dialect's routed \
                 blocks"
                    .to_owned(),
            ),
        ));
    };
    let d = wide_dims_of(
        hp,
        attn,
        mla,
        e.shared_ff(),
        e.ff,
        wide_ff_rows(hp.dense_ff, e),
    );
    Ok((d, None))
}

/// [`wide_dims`]' widths at the given routed terms (`ff_rows` the widest
/// feed-forward block a unit's gate rows hold).
fn wide_dims_of(
    hp: &Hparams,
    attn: usize,
    mla: &MlaDims,
    shared_ff: usize,
    expert_ff: usize,
    ff_rows: usize,
) -> WideDims {
    WideDims {
        embd: hp.n_embd,
        attn,
        q_lora: hp.mla.q_lora,
        heads: mla.heads,
        nope: mla.nope,
        dense_ff: hp.dense_ff,
        shared_ff,
        expert_ff,
        ff_rows,
    }
}

/// The width of the widest block the dense forms' buffers hold rows of: the dense
/// layers' and the shared experts' (`n_shared` experts of `ff` run as one block
/// of `ff · n_shared` values).
fn dense_block_ff(dense_ff: usize, e: &Experts) -> usize {
    dense_ff.max(e.shared_ff())
}

/// The gate and up rows' width of a wide unit's feed-forward buffers: the widest
/// dense-form block, or the routed slots' (`N_USED` experts of `ff`).
fn wide_ff_rows(dense_ff: usize, e: &Experts) -> usize {
    dense_block_ff(dense_ff, e).max(ffn::sigmoid_128_6::N_USED * e.ff)
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
        // The latent attention takes at most [`mla::ATTN_CHUNK`] rows a launch:
        // a wider unit runs it in chunks that share one set of partials.
        let (part_v, part_ms) = latent_gated::partials_lens_rows(m, ctx, rows.min(mla::ATTN_CHUNK));
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
    /// A wide unit's rows, forms and kernels ([`super::wide`]); `None` for a
    /// file whose shapes or per-layer input keep every prompt on the rows
    /// pass ([`Body::wide_off`]).
    pub(super) wide: Option<Wide>,
    /// What keeps the wide arm off this file's prompts, found at load, or `None`
    /// when it runs every layer ([`Body::wide_ok`]).
    wide_off: Option<String>,
    /// The per-layer input site of a file that carries one, and its buffers.
    pub(super) ple: Option<Ple>,
    ple_b: Option<PleBufs>,
    /// The tensor dialect the file speaks ([`Hparams::dialect`]).
    pub(super) dialect: Dialect,
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
        // The header read once, before the blocks load: the trunk's own
        // count names the range, so the next-token blocks a file carries past
        // it (described as its `mtp`, run by no program) load nowhere and
        // their tensors stay unread — the reader's own rule.
        let hp = Hparams::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
        let trunk = 0..hp.n_trunk;
        GpuModel::load_blocks(gpu, &file, ctx, trunk, true, |gpu, w| {
            for l in (0..hp.n_trunk).filter(|&l| hp.kind(l) == Kind::Rda) {
                match hp.dialect {
                    Dialect::Split => rda::derive(gpu.stream(), w, l)?,
                    Dialect::Fused => rda::fold(gpu.stream(), w, l, (hp.n_head, hp.decay_rank))?,
                }
            }
            for l in (0..hp.n_trunk).filter(|&l| !hp.dense(l)) {
                down_words(gpu.stream(), &file, w, &names::ffn_down_exps(l))?;
            }
            for l in (0..hp.n_trunk).filter(|&l| hp.dense(l) && hp.dialect == Dialect::Fused) {
                ffn_fold(gpu.stream(), w, l, hp.dense_ff)?;
            }
            Body::load(gpu, &file, w, (hp.clone(), ctx), lanes)
        })
    }
}

/// The Nano dialect's fused feed-forward stem of dense layer `l`, split at
/// load into the split dialect's sites: the file's one `ffn_up` holds the
/// SwiGLU's gate rows then its up rows (raon1nano.cpp:152-154). Load-time
/// only, before the sites load.
fn ffn_fold(stream: &CudaStream, w: &mut Weights, l: usize, ff: usize) -> Result<(), GpuError> {
    w.split_rows(
        stream,
        &names::ffn_up(l),
        &[
            (format!("derived.blk.{l}.ffn_gate"), ff),
            (format!("derived.blk.{l}.ffn_up"), ff),
        ],
    )
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
        let (wd, dims_off) = wide_dims(hp, &kda, &mla)?;
        let ple = match hp.ple {
            Some(width) => PleBufs::bytes_of((hp.n_embd, hp.n_trunk, width)),
            None => 0,
        };
        // A file the dims refuse holds no wide kit: every prompt runs the
        // rows pass ([`Body::wide_off`]).
        let wide = if dims_off.is_some() || hp.ple.is_some() {
            0
        } else {
            let wide_lens = Scratch::lens(hp, (&kda, &mla), usize::MAX, WIDE_ROWS)?;
            Scratch::bytes_of(&wide_lens) + Wide::bytes_of(&wd)
        };
        Ok((Scratch::bytes_of(&lens) + wide + ple) as u64)
    }

    /// The last hyper-connection site's route as the step left it: the last
    /// layer's feed-forward site, `pre[4]`, `post[4]`, then `comb[16]` at
    /// `src · 4 + dst` ([`HC_OUT`] values). Synchronizes the stream.
    pub fn last_route(&self, stream: &CudaStream) -> Result<Vec<f32>, GpuError> {
        Ok(self.s.hc.to_host_vec(stream)?)
    }

    /// Whether a prompt of more than [`PASS_ROWS`] ids runs as wide units
    /// ([`super::wide`]), decided once at load over every layer. The wide arm
    /// refuses a weight the loader admits — an output projection or `attn_q_b`
    /// that is not Q8_0, a forget projection that is not F32, a dense block or
    /// shared expert whose down has no form at its width (a K-quant down over
    /// the dense width, a Q8_0 down over the shared width), routed stacks that
    /// are not one grouped K-quant type with a down the 32-value GEMM reads
    /// ([`wide::rda_runs`], [`mla::wide_runs`], [`wide::ffn_runs`]). One
    /// such layer keeps the whole file on the rows pass for prompts of every
    /// length, where a prompt past [`PASS_ROWS`] ids would otherwise stop at
    /// that layer by name.
    #[must_use]
    pub fn wide_ok(&self) -> bool {
        self.wide_off.is_none()
    }

    /// The refusal the load found that keeps the wide arm off this file's
    /// prompts, the first layer's in file order; `None` when
    /// [`Body::wide_ok`].
    #[must_use]
    pub fn wide_off(&self) -> Option<&str> {
        self.wide_off.as_deref()
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
        let (kda, mla) = dims(&hp)?;
        let act = act_of(&hp)?;
        if let Some(e) = &hp.experts
            && (e.n_expert != ffn::sigmoid_128_6::N_EXPERT
                || e.n_used != ffn::sigmoid_128_6::N_USED
                || !e.weights_norm)
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
        let ple = Ple::load(file, w, &hp)?;
        let glu = |l: usize, (gate, up, down): (String, String, String), ff: usize| {
            Ok::<Glu, GpuError>(Glu {
                gate: typed(file, w, gate, (ff, n), FFN_TYPES)?,
                up: typed(file, w, up, (ff, n), FFN_TYPES)?,
                down: typed(file, w, down, (n, ff), FFN_TYPES)?,
                ff,
            })
            .map_err(|err| GpuError::shape(WHAT, format!("layer {l}'s feed-forward block: {err}")))
        };
        let mut layers = Vec::with_capacity(hp.n_trunk);
        for l in 0..hp.n_trunk {
            let mixer = match hp.kind(l) {
                Kind::Rda => Mixer::Rda {
                    sites: Box::new(RdaSites::load(file, w, &kda, l, hp.dialect)?),
                    store: state::rda_store(stream, kda.shape, lanes)?,
                },
                Kind::Mla => Mixer::Mla {
                    sites: Box::new(MlaSites::load(file, w, &mla, l)?),
                    cache: DeviceTensor::zeroed(stream, ctx, mla.row())?,
                },
            };
            let ffn = if hp.dense(l) {
                // The dense block's gate and up: the split dialect's own file
                // tensors, or the load-time halves of the fused `ffn_up`.
                let (gate, up) = match hp.dialect {
                    Dialect::Split => (names::ffn_gate(l), names::ffn_up(l)),
                    Dialect::Fused => (
                        format!("derived.blk.{l}.ffn_gate"),
                        format!("derived.blk.{l}.ffn_up"),
                    ),
                };
                Ffn::Dense(glu(l, (gate, up, names::ffn_down(l)), hp.dense_ff)?)
            } else {
                let e = hp
                    .experts
                    .as_ref()
                    .expect("a routed layer: the read refused a file whose router keys are absent");
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
                    latent: None,
                    sum_norm: None,
                })
            };
            // A sublayer's mix vectors in the file's dialect: the split
            // dialect's `fn`, `scale` and `base`, or the folded chart's `fn`,
            // `base` and `chart`.
            let hc = [Sub::Attn, Sub::Ffn].map(|sub| match hp.dialect {
                Dialect::Split => [
                    names::hc_fn(l, sub),
                    names::hc_scale(l, sub),
                    names::hc_base(l, sub),
                ],
                Dialect::Fused => [
                    names::hc_fn(l, sub),
                    names::hc_base(l, sub),
                    names::hc_chart(l, sub),
                ],
            });
            for sub in &hc {
                match hp.dialect {
                    Dialect::Split => {
                        let [f, scale, base] = sub;
                        exact(w, f.clone(), hp.hc.mix_dim() * STREAMS * n)?;
                        exact(w, scale.clone(), 4)?;
                        exact(w, base.clone(), hp.hc.mix_dim())?;
                    }
                    Dialect::Fused => {
                        let [f, base, chart] = sub;
                        exact(w, f.clone(), hp.hc.mix_dim() * STREAMS * n)?;
                        exact(w, base.clone(), hp.hc.mix_dim())?;
                        exact(w, chart.clone(), CHART_LEN)?;
                    }
                }
            }
            exact(w, names::ffn_norm(l), n)?;
            layers.push(Layer { mixer, ffn, hc });
        }
        let mut acts = Acts::new(
            stream,
            layers
                .iter()
                .flat_map(Layer::sites)
                .chain(ple.iter().flat_map(|p| p.sites())),
        )?;
        rda::add_row_forms(
            stream,
            &mut acts,
            layers.iter().flat_map(Layer::mixer_sites),
        )?;
        let (ff, ff_expert) = match &hp.experts {
            Some(e) => (dense_block_ff(hp.dense_ff, e), e.ff),
            None => (hp.dense_ff, 0),
        };
        let (wd, dims_off) = wide_dims(&hp, &kda, &mla)?;
        // A file the dims refuse holds no wide kit at all, and so does one
        // with a per-layer input: every prompt takes the rows pass.
        let hold_wide = dims_off.is_none() && ple.is_none();
        let mut wide_off = dims_off.or_else(|| wide_refusal(&layers, w, &wd));
        if wide_off.is_none() && ple.is_some() {
            wide_off = Some(
                "the file's per-layer input: a wide unit runs none (the step and the rows pass \
                 do)"
                .to_owned(),
            );
        }
        let stage = rda::stage_len(&kda)
            .max(gated_mla::stage_len(&mla))
            .max(ffn::stage_len(ff.max(ff_expert)));
        // A file whose shapes the wide arm holds one form of holds the wide
        // kit whether a weight later refuses it or not, as a file of a
        // refusing weight type does; a file the dims refuse, or one with a
        // per-layer input, holds none: every prompt takes the rows pass.
        let wide = if hold_wide {
            Some(Wide::new(
                gpu,
                &wd,
                Scratch::new(stream, &hp, (&kda, &mla), ctx, WIDE_ROWS)?,
            )?)
        } else {
            None
        };
        Ok(Body {
            k: Kernels::load(gpu, &mla)?,
            s: Scratch::new(stream, &hp, (&kda, &mla), ctx, 1)?,
            step_in: StepIn::new(stream)?,
            ffn_s: FfnScratch::new(stream, n, ff, RoutedShape::plain(ff_expert))?,
            pass: Scratch::new(stream, &hp, (&kda, &mla), ctx, PASS_ROWS)?,
            ffn_rows: FfnRows::new(stream, n, (ff, RoutedShape::plain(ff_expert)), PASS_ROWS)?,
            stage: DeviceBuffer::zeroed(stream, stage)?,
            wide,
            wide_off,
            ple_b: ple
                .as_ref()
                .map(|p| PleBufs::new(stream, (n, p.trunk, p.width)))
                .transpose()?,
            rows: RowState::new(stream, lanes)?,
            kda,
            mla,
            act,
            ple,
            dialect: hp.dialect,
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
            + self.wide.as_ref().map_or(0, Wide::bytes)
            + self.ple_b.as_ref().map_or(0, PleBufs::bytes)
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
            wide_off,
            ple,
            ple_b,
            dialect,
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
        let embd = gather::RowsTable::of(w, STEP, &names::token_embd())?;
        let RowState {
            rin, lanes, taps, ..
        } = rows;
        let pos_w: Window<'_, u32>;
        let keys_w: Window<'_, u32>;
        let ids_w: Window<'_, u32>;
        // The per-layer input's gathers read the same ids as the streams'
        // own: the step's one token, or the staged rows'.
        let ple_step_ids: Window<'_, u32>;
        let ple_ids: &DeviceBuffer<u32>;
        let ple_pos0: &DeviceBuffer<u32>;
        let mut wide_kit = None;
        let (sc, mut ffn_b, pos, keys, (at, m, pass)) = match run {
            Run::Step => {
                gather::enqueue_rows(
                    gpu,
                    &k.q38,
                    &embd,
                    (&step_in.ids, &step_in.pos0),
                    (&mut s.x, &mut step_in.pos, &mut step_in.n_keys),
                )?;
                let (pos, keys) = (&step_in.pos, &step_in.n_keys);
                // The step's ids are the one token once a stream; the
                // per-layer input's gather reads the row once.
                ple_step_ids = Window::<u32>::of(&step_in.ids, 0, 1)?;
                ple_ids = &ple_step_ids;
                ple_pos0 = &step_in.pos0;
                (s, FfnBufs::Step(ffn_s), pos, keys, (0, 1, Pass::InPlace))
            }
            Run::Rows { at, m, pass } if m > PASS_ROWS => {
                if pass != Pass::InPlace {
                    return Err(GpuError::shape(
                        STEP,
                        format!("a {pass:?} pass of {m} rows: only a prompt's units run wide"),
                    ));
                }
                if let Some(why) = wide_off {
                    return Err(GpuError::shape(
                        STEP,
                        format!("a wide unit of {m} rows with the wide arm off: {why}"),
                    ));
                }
                let Some(wide) = wide.as_mut() else {
                    return Err(GpuError::state(
                        STEP,
                        "the wide kit: the load found the arm off",
                    ));
                };
                wide.route_dense(gpu, m)?;
                let (kit, ws, wrin) = wide.kit(&k.g32, &k.ffn.router);
                let RowViews {
                    ids,
                    pos,
                    keys,
                    epos,
                    ekeys,
                } = wrin.views(at, m)?;
                gather::enqueue_rows(gpu, &k.q38, &embd, (&ids, &pos), (&mut ws.x, epos, ekeys))?;
                pos_w = pos;
                keys_w = keys;
                ids_w = ids;
                ple_ids = &ids_w;
                ple_pos0 = &*pos_w;
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
                gather::enqueue_rows(gpu, &k.q38, &embd, (&ids, &pos), (&mut ps.x, epos, ekeys))?;
                pos_w = pos;
                keys_w = keys;
                ids_w = ids;
                ple_ids = &ids_w;
                ple_pos0 = &*pos_w;
                (
                    ps,
                    FfnBufs::Rows(ffn_rows),
                    &*pos_w,
                    &*keys_w,
                    (at, m, pass),
                )
            }
        };
        // The per-layer inputs' combine once a walk, before the layer loop.
        // A wide unit never runs here: a file with a per-layer input loads
        // the wide arm off.
        if let (Some(p), Some(pb)) = (ple.as_ref(), ple_b.as_mut()) {
            let mut kit = RowsKit {
                linear: &k.linear,
                rows: &k.rows,
                stage: &mut *stage,
            };
            p.enqueue_setup(
                &cx,
                (acts, &mut kit, &k.q38, &k.ple),
                (ple_ids, ple_pos0),
                m,
                pb,
            )?;
        }
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
            let charts = |[f, base, chart]: &[String; 3]| -> Result<ChartVecs<'_>, GpuError> {
                Ok(ChartVecs {
                    w: w.f32_buf(STEP, f)?,
                    base: w.f32_buf(STEP, base)?,
                    chart: w.f32_buf(STEP, chart)?,
                })
            };
            // The route the file's dialect parametrizes: the split dialect's
            // scale scalars, or the folded chart's baked mix. The partials
            // and the route are borrowed per call, for the fold and the post
            // mix read them between the two routes.
            let route = |vecs3: &[String; 3],
                         src: &DeviceBuffer<f32>,
                         (part, hc): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>)|
             -> Result<(), GpuError> {
                match dialect {
                    Dialect::Split => {
                        mhc.enqueue_route(stream, &vecs(vecs3)?, src, (n, m), hcf, (part, hc))
                    }
                    Dialect::Fused => mhc.enqueue_route_folded(
                        stream,
                        &charts(vecs3)?,
                        src,
                        (n, m),
                        hcf,
                        (part, hc),
                    ),
                }
            };
            route(attn_hc, x, (part, hc))?;
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
                        gated_mla::step(&cx, &k.mla, acts, (&**sites, &*md), l, bufs)?;
                    } else if let Some(kit) = wide_kit.as_mut() {
                        mla::rows_wide(&cx, &k.mla, kit, (&**sites, &*md), (l, m), bufs)?;
                    } else {
                        let kit = RowsKit {
                            linear: &k.linear,
                            rows: &k.rows,
                            stage: &mut *stage,
                        };
                        gated_mla::rows(&cx, &k.mla, acts, (&**sites, &*md), (l, m), bufs, kit)?;
                    }
                }
            }
            mhc.enqueue_post(stream, (x, br, hc), (n, m), x2)?;
            route(ffn_hc, x2, (part, hc))?;
            mhc.enqueue_fold(stream, x2, hc, (n, m), u)?;
            let gain = w.f32_buf(STEP, &names::ffn_norm(l))?;
            gpu.elem()
                .enqueue_rms_norm(stream, u, gain, hp.rms_eps, n, m, un)?;
            let rule = (*act, hp.experts.as_ref().map_or(1.0, |e| e.weights_scale));
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
            // The per-layer input's gated add, after the feed-forward
            // post-mix (module doc).
            if let (Some(p), Some(pb)) = (ple.as_ref(), ple_b.as_mut()) {
                let mut kit = RowsKit {
                    linear: &k.linear,
                    rows: &k.rows,
                    stage: &mut *stage,
                };
                p.enqueue_layer(&cx, (acts, &mut kit, mhc, &k.ple), (l, m), x, pb)?;
            }
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

#[cfg(test)]
mod tests {
    use super::mla::tests::typed_sites;
    use super::{
        Act, Ffn, Glu, Layer, Mixer, STREAMS, Site, SiteTy, act_of, dims, wide_dims, wide_refusal,
    };
    use crate::arch::routed_ffn::sigmoid_128_6::{N_EXPERT, N_USED};
    use crate::weights::Weights;
    use crate::{DeviceTensor, Gpu};
    use model::arch::delta_mla::{Mla, Situ};
    use model::arch::kda::Keys;
    use model::arch::models::KdaGate;
    use model::arch::noaux::Experts;
    use model::arch::raon1::hparams::{Hc, Hparams, Kind};

    /// A header with Raon's widths and a dense width, a routed expert's width and a
    /// count of shared experts of the caller's.
    fn header(dense_ff: usize, ff: usize, n_shared: usize) -> Hparams {
        Hparams {
            arch: model::arch::models::Arch::Raon1,
            n_layer: 3,
            n_trunk: 3,
            n_embd: 2560,
            heads: vec![40; 3],
            n_head: 40,
            kda: Keys {
                conv: 4,
                head_dim: 128,
            },
            decay_rank: 128,
            gate: KdaGate::Sigmoid { lower_bound: -5.0 },
            mla: Mla {
                q_lora: 576,
                kv_lora: 512,
                head_k: 192,
                head_v: 128,
                pe: 64,
            },
            rms_eps: 1e-5,
            latent_eps: 1e-6,
            hc: Hc {
                streams: STREAMS,
                eps: 1e-5,
                magnitude: 2.0,
            },
            situ: Some(Situ {
                gate_beta: 4.0,
                value_beta: 4.0,
            }),
            act: model::arch::models::Act::Situ {
                gate_beta: 4.0,
                value_beta: 4.0,
            },
            ple: None,
            head_tied: false,
            n_vocab: 1024,
            n_ctx_train: 4096,
            dense_lead: 1,
            dense_ff,
            experts: Some(Experts {
                n_expert: N_EXPERT,
                n_used: N_USED,
                ff,
                n_shared,
                shared_width: ff * n_shared,
                weights_scale: 1.0,
                weights_norm: true,
            }),
            dialect: model::arch::raon1::roles::Dialect::Split,
            kinds: vec![Kind::Rda, Kind::Rda, Kind::Mla],
            defaults: Vec::new(),
        }
    }

    /// A wide unit's gate and up rows hold the widest block that writes them: the
    /// dense layer's, a routed layer's shared experts' (`n_shared` of `ff` run as
    /// one block) and the routed slots' (`N_USED` of `ff`). Seven shared experts of
    /// 256 beside a dense width of 1024 are 1792 values, past the dense width and
    /// the slots' 1536. The buffers' size and the plan's reserve
    /// ([`Body::pass_reserve`]) both read these widths.
    #[test]
    fn a_wide_units_gate_rows_hold_the_shared_block() {
        for (dense_ff, ff, n_shared) in [(1024, 256, 7), (12032, 1088, 1), (512, 128, 1)] {
            let hp = header(dense_ff, ff, n_shared);
            let (kda, mla) = dims(&hp).expect("the header's widths");
            let (d, off) = wide_dims(&hp, &kda, &mla).expect("the wide arm's widths");
            assert!(off.is_none(), "the header's shapes run wide");
            for (what, width) in [
                ("dense", dense_ff),
                ("shared", ff * n_shared),
                ("routed slots", N_USED * ff),
            ] {
                assert!(
                    d.ff_rows >= width,
                    "dense {dense_ff}, {n_shared} shared of {ff}: gate rows of {} values hold \
                     the {what} block of {width}",
                    d.ff_rows
                );
            }
        }
    }

    /// One layer the wide arm refuses keeps the whole file off it: a latent layer
    /// whose output projection is Q6_K among Q8_0 ones is the first refusal, named
    /// by its layer, and a file of Raon's types has none.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_one_layer_with_a_q6k_output_keeps_the_file_off_the_wide_arm() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let hp = header(1024, 256, 1);
        let (kda, mla) = dims(&hp).expect("the header's widths");
        let (d, off) = wide_dims(&hp, &kda, &mla).expect("the wide arm's widths");
        assert!(off.is_none(), "the header's shapes run wide");
        let site = |name: &str, ty: SiteTy, k: usize| Site::new(name.to_owned(), ty, k);
        let layer = |out: SiteTy| Layer {
            mixer: Mixer::Mla {
                sites: Box::new(typed_sites(SiteTy::Q8_0, out)),
                cache: DeviceTensor::zeroed(gpu.stream(), 1, mla.row()).expect("a cache row"),
            },
            ffn: Ffn::Dense(Glu {
                gate: site("gate", SiteTy::Q4K, hp.n_embd),
                up: site("up", SiteTy::Q4K, hp.n_embd),
                down: site("down", SiteTy::Q6K, hp.dense_ff),
                ff: hp.dense_ff,
            }),
            hc: std::array::from_fn(|_| std::array::from_fn(|_| String::new())),
        };
        let w = Weights::default();
        let raon = [layer(SiteTy::Q8_0), layer(SiteTy::Q8_0)];
        assert_eq!(wide_refusal(&raon, &w, &d), None);
        let mixed = [layer(SiteTy::Q8_0), layer(SiteTy::Q6K), layer(SiteTy::F32)];
        let refusal = wide_refusal(&mixed, &w, &d).expect("a refusal");
        assert!(
            refusal.contains("layer 1's mixer") && refusal.contains("an output projection of Q6_K"),
            "{refusal}"
        );
    }

    /// The blocks' activation comes from the description: the situ keys' own,
    /// the Nano row's SwiGLU plain and clamped, each the entry its rule names;
    /// another activation is refused by name. Mutant: the SwiGlu arm answering
    /// the situ rule.
    #[test]
    fn the_act_is_the_descriptions() {
        let mut hp = header(1024, 256, 1);
        hp.act = model::arch::models::Act::Situ {
            gate_beta: 4.0,
            value_beta: 25.0,
        };
        assert_eq!(
            act_of(&hp).expect("the situ rule"),
            Act::Situ {
                beta: 4.0,
                linear_beta: 25.0
            }
        );
        for (limit, want) in [
            (None, Act::SiluMul),
            (Some(7.0f32), Act::SwigluClamp { limit: 7.0 }),
        ] {
            hp.act = model::arch::models::Act::SwiGlu { limit };
            assert_eq!(act_of(&hp).expect("a swiglu rule"), want);
        }
        hp.act = model::arch::models::Act::GeGlu;
        let err = act_of(&hp)
            .expect_err("a rule the card does not run")
            .to_string();
        assert!(err.contains("the blocks' activation is GELU GLU"), "{err}");
    }

    /// The wide arm's dims carry why a file of shapes they do not hold keeps
    /// every prompt on the rows pass: the Nano dialect's own two attention
    /// widths and its routing no experts, the Mini header's shapes neither.
    /// Mutant: the experts arm answered as though the widths differed.
    #[test]
    fn the_wide_dims_name_the_shapes_they_do_not_hold() {
        let hp = header(1024, 256, 1);
        let (kda, mla) = dims(&hp).expect("the header's widths");
        let (_, off) = wide_dims(&hp, &kda, &mla).expect("the wide arm's widths");
        assert!(off.is_none(), "the Mini header's shapes run wide");
        let mut nano = hp;
        nano.experts = None;
        nano.dense_lead = nano.n_trunk;
        let (kda, mla) = dims(&nano).expect("the header's widths");
        let (d, off) = wide_dims(&nano, &kda, &mla).expect("the wide arm's widths");
        let why = off.expect("a dense file runs no wide units");
        assert!(why.contains("the file routes no experts"), "{why}");
        assert_eq!(d.ff_rows, nano.dense_ff, "the dense width alone");
        // The Nano dialect's two attention widths: the latent layers' heads
        // their own, not the delta layers'.
        nano.heads[2] = 8;
        let (kda, mla) = dims(&nano).expect("the header's widths");
        let (_, off) = wide_dims(&nano, &kda, &mla).expect("the wide arm's widths");
        let why = off.expect("the widths' refusal");
        assert!(
            why.contains(
                "the RDA layers' output projection reads 4096 values, the latent \
                          layers' 1024"
            ),
            "{why}"
        );
    }

    /// The latent layers share one head count — the one latent dims sizes
    /// every latent layer's sites, kernels and caches — and a file whose
    /// differ is refused by name; a file of no latent layer is too. Mutant:
    /// the uniformity check dropped.
    #[test]
    fn the_latent_layers_share_one_head_count() {
        let mut hp = header(1024, 256, 1);
        hp.heads[2] += 1;
        let err = dims(&hp)
            .expect_err("differing latent head counts")
            .to_string();
        assert!(
            err.contains("the latent layers' head counts differ: 40 and 41"),
            "{err}"
        );
        hp.kinds = vec![Kind::Rda; 3];
        let err = dims(&hp).expect_err("no latent layer").to_string();
        assert!(err.contains("no latent layer"), "{err}");
    }
}
