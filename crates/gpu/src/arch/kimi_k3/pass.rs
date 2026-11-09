//! Kimi K3's placed prompt pass: up to a plan's [`PassShape::rows`] consecutive positions through
//! the placed chain layer by layer as one walk `(1, rows, Batch)` over the host tier's batch port,
//! so each hosted layer's host experts are served once a pass for every row ([`BatchLeg`]: one
//! download, one union serve, one upload) and the paged ones are read once a pass. Each row is the
//! placed step's bits ([`super::placed`]'s front, shadow and back at one row), the stores left as
//! that many steps leave them.
//!
//! What crosses layers or the host boundary is the pass's rows wide ([`PassLens`]): the sums, the
//! banks, the block's normed rows, the shared stack's output, the latent rows the host and the
//! tier read, the card's, the tier's and the host's sums, the router's results and the places.
//! What a layer's launches need inside is a chunk of [`CHUNK`] rows, today's rows-pass width: the
//! delta mixer's intermediates (its conv ring holds a pass of [`CHUNK`]), the latent mixer's (its
//! attention's partials at [`PassShape::mla_rows`] rows, the plan's other choice), the stage, the
//! shared stack's intermediates and the card leg's buffers.
//!
//! Why each row is the step's bits, launch by launch:
//!
//! - the embedding is the host table's row per id, the positions and key counts per row;
//! - the AttnRes mixes are one block a row, the banks at a stride of the pass's rows;
//! - a mixer runs its chunks in position order through the rows pass's own launches
//!   ([`super::rows`]; one row is the step's launch); on a load that selects, a latent layer picks
//!   each row's list and runs its core a row at a time over the rows' count words, staged with the
//!   positions ([`super::index::attend_rows_sel`]);
//! - every projection runs at most [`CHUNK`] columns ([`rows_kit::project_rows`]): each column the
//!   one-column launch's bits;
//! - the router over the rows is what that many one-row routes write (`route_rows`);
//! - the host serves each column's sum in its list order from zero, as the one-column serve does
//!   (the union port);
//! - the card's slots and the tier's are per (row, slot) launches summed in slot order per row;
//! - the back adds `(card + tier) + host` per row, as the step's back does;
//! - a layer the card holds whole runs the step's whole-card block on each row.
//!
//! A pass of more rows than the plan's shape, at a position the stores do not stand at, or on a
//! load made without the pass's buffers is refused by name.

use super::super::kda;
use super::index::{self, WORDS};
use super::placed::{Placed, Side};
use super::rows::{PassSel, run_mixer_rows};
use super::{
    Ffn, Instance, KdaBuffers, Kernels, Layer, Mixer, MixerIo, MlaBuffers, mla_dims, step_cx,
};
use crate::arch::gated_mla as mla;
use crate::arch::hybrid::{Acts, GqaCx};
use crate::arch::routed_ffn::{Routed, glu_rows};
use crate::arch::rows_kit::{self, RowKernels, RowsKit, add_row_forms};
use crate::attn_res::AttnResArgs;
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::card_slots::{CardAcc, Places};
use crate::head::{HOST_HEAD_ROWS, Head};
use crate::host::BatchLeg;
use crate::host::run::HostRun;
use crate::kquant::Act;
use crate::latent_gated::{self, MlaDims};
use crate::linear::mixer_kda::KdaDims;
use crate::linear::{HEAD, PASS_ROWS};
use crate::model::GpuModel;
use crate::routed_block::{RouteIn, RoutedRows, RouterEntry};
use crate::tensor::{Window, WindowMut, copy_row};
use crate::weights::Weights;
use crate::{Gpu, GpuError, LAYER_HEAD, launch_u32};
use cuda_core::DeviceBuffer;
use model::arch::kimi_k3::hparams::{Hparams, INDEX_DIM, INDEX_HEADS, Indexer};
use model::arch::kimi_k3::names;
use model::placement::kernels::{DownAct, DownForms};
use model::placement::workstation::TierLegScratch;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

/// What the pass's refusals name.
const PASS: &str = "kimi_k3 placed pass";

/// The rows a layer's launches take at once: the rows pass's width, which the delta mixer's conv
/// ring holds.
pub const CHUNK: usize = PASS_ROWS;

/// The rows the placed pass holds and the rows the latent mixer's attention runs at a time. The
/// load makes the pass's buffers at this shape ([`pass_bytes`]), the plan reserves them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassShape {
    /// Rows a pass holds: 1 to the host union's [`model::moe::UNION_MAX_COLS`].
    pub rows: usize,
    /// Rows the latent mixer's attention runs at a time: 1 to [`CHUNK`]. Its partials are these
    /// rows' alone.
    pub mla_rows: usize,
}

/// The shapes a plan tries, widest first: the rows from the union's widest down to one chunk, and
/// at each the latent attention a chunk, then a row, at a time.
pub const SHAPES: [PassShape; 14] = {
    let rows = [512, 256, 128, 64, 32, 16, 8];
    let mut out = [PassShape {
        rows: 1,
        mla_rows: 1,
    }; 14];
    let mut i = 0;
    while i < 7 {
        out[2 * i] = PassShape {
            rows: rows[i],
            mla_rows: CHUNK,
        };
        out[2 * i + 1] = PassShape {
            rows: rows[i],
            mla_rows: 1,
        };
        i += 1;
    }
    out
};

const _: () = assert!(SHAPES[0].rows == model::moe::UNION_MAX_COLS);

impl PassShape {
    /// Refused by name: rows outside 1 to the union's widest, latent rows outside 1 to
    /// [`CHUNK`].
    pub(super) fn check(self) -> Result<PassShape, GpuError> {
        if !(1..=model::moe::UNION_MAX_COLS).contains(&self.rows)
            || !(1..=CHUNK).contains(&self.mla_rows)
        {
            return Err(GpuError::shape(
                PASS,
                format!(
                    "a pass of {} rows with the latent attention at {}: 1 to {} rows, 1 to \
                     {CHUNK} latent rows",
                    self.rows,
                    self.mla_rows,
                    model::moe::UNION_MAX_COLS
                ),
            ));
        }
        Ok(self)
    }
}

/// The lengths of every buffer of a pass at a shape: the one reading of the pass's bytes, which
/// [`PlacedPass::new`] allocates and [`pass_bytes`] counts.
struct PassLens {
    /// `(f32 values)` of each f32 buffer, in [`PlacedPass`]'s field order.
    f32s: [usize; F32S],
    /// `(u32 values)` of each u32 buffer.
    u32s: [usize; U32S],
    /// The card leg's run of [`CHUNK`] rows.
    leg: LegShape,
    /// The card leg's bytes, the common leg's formula ([`TierLegScratch`]).
    leg_bytes: u64,
}

/// The f32 buffers: the rows' (sum ×2, mixed, update, ffn_xn, shared, banks, act, acc, tier,
/// routed, hsum, probs, weights), then a chunk's (xn, the delta mixer's eleven, the latent
/// mixer's nine and its two partials, the stage, the shared stack's three, the index queries and
/// weights on a load that selects).
const F32S: usize = 14 + 1 + 11 + 9 + 2 + 1 + 3 + 2;

/// The u32 buffers: the rows' ids, card places, tier places, positions, key counts and count
/// words on a load that selects.
const U32S: usize = 6;

impl PassLens {
    fn of(
        hp: &Hparams,
        (kd, md): (&KdaDims, &MlaDims),
        ctx: usize,
        shape: PassShape,
    ) -> Result<PassLens, GpuError> {
        let shape = shape.check()?;
        let (r, c) = (shape.rows, CHUNK);
        let n = hp.n_embd;
        let lat = hp.latent;
        let (n_expert, n_used) = (hp.experts.n_expert, hp.experts.n_used);
        let banks = hp.n_layer.div_ceil(hp.attn_res_block);
        let lin = kd.shape;
        let (channels, v, n_v) = (lin.channels(), lin.n_v * HEAD, lin.n_v);
        let (part_v, part_ms) = latent_gated::partials_lens_rows(md, ctx, shape.mla_rows);
        let (heads, row) = (md.heads, md.row());
        let ff = hp.dense_ff.max(hp.experts.shared_ff());
        let pick = |len: usize| if index::selects(hp, ctx) { len } else { 0 };
        let wide = n.max(lat).max(ff);
        let stage = [kda::stage_len(kd), mla::stage_len(md), c * wide]
            .into_iter()
            .max()
            .unwrap_or(1);
        let f32s = [
            r * n,
            r * n,
            r * n,
            r * n,
            r * n,
            r * n,
            banks * r * n,
            r * lat,
            r * lat,
            r * lat,
            r * lat,
            r * lat,
            r * n_expert,
            r * n_used,
            c * n,
            c * channels,
            c * HEAD,
            c,
            c * n_v,
            c * v,
            c * v,
            c * channels,
            c * n_v,
            c * v,
            c * v,
            c * v,
            c * md.q_lora,
            c * md.q_lora,
            c * row,
            c * heads * md.v_head,
            c * heads * md.head_k(),
            c * heads * row,
            c * heads * crate::latent::LATENT,
            c * heads * md.v_head,
            c * heads * md.v_head,
            part_v,
            part_ms,
            stage,
            c * ff,
            c * ff,
            c * ff,
            pick(c * INDEX_HEADS * INDEX_DIM),
            pick(c * INDEX_HEADS),
        ];
        let u32s = [r * n_used, r * n_used, r * n_used, r, r, pick(r * WORDS)];
        let leg = LegShape {
            cap: c,
            slots_per_col: n_used,
            n: lat,
            ff: hp.experts.ff,
        };
        let downs = DownForms::of(DownAct::Q8Act);
        let leg_bytes = TierLegScratch {
            n_embd: lat as u64,
            ff: hp.experts.ff as u64,
            n_used: n_used as u64,
            chunk: c as u64,
            downs,
            input_scales: false,
        }
        .bytes(downs)
        .map_err(|e| GpuError::plan(PASS, e))?;
        Ok(PassLens {
            f32s,
            u32s,
            leg,
            leg_bytes,
        })
    }

    /// Device bytes of a pass at these lengths, each buffer at least one value.
    fn bytes(&self) -> u64 {
        let words: usize = self.f32s.iter().chain(&self.u32s).map(|&l| l.max(1)).sum();
        4 * words as u64 + self.leg_bytes
    }
}

/// The device bytes the placed pass of `shape` makes on the stage card for a file of `hp` at a
/// context of `ctx` positions: the plan's prompt reserve for a load that runs it. Refused by name:
/// a shape outside [`PassShape`]'s bounds and a geometry the common leg's bytes do not count.
pub fn pass_bytes(hp: &Hparams, ctx: usize, shape: PassShape) -> Result<u64, GpuError> {
    let kd = kda::dims(hp)?;
    let md = mla_dims(hp);
    Ok(PassLens::of(hp, (&kd, &md), ctx, shape)?.bytes())
}

/// The placed pass's buffers ([`PassLens`]) and its row kernels, made once at load beside
/// [`Placed`] and lent to every pass. It holds no state a pass leaves for the next.
pub(super) struct PlacedPass {
    shape: PassShape,
    sum: [DeviceBuffer<f32>; 2],
    mixed: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    ffn_xn: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    /// `banks · rows · n` values: bank `b`'s row `t` at `b · rows · n + t · n`.
    banks: DeviceBuffer<f32>,
    /// The latent rows the host and the tier read, the card's sum, the tier's, the routed sum.
    act: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    tier: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    /// The host's sums the batch port uploads.
    hsum: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    weights: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    sel: DeviceBuffer<u32>,
    tsel: DeviceBuffer<u32>,
    pos: DeviceBuffer<u32>,
    keys: DeviceBuffer<u32>,
    /// `[rows][WORDS]`: the rows' count words on a load that selects ([`index::pass_words`]).
    words: DeviceBuffer<u32>,
    /// A chunk's: the mixer's normed rows, the mixers' intermediates, the stage, the shared
    /// stack's gate, up and SiTU rows, the index queries and weights on a load that selects.
    xn: DeviceBuffer<f32>,
    kb: KdaBuffers,
    mb: MlaBuffers,
    stage: DeviceBuffer<f32>,
    glu: [DeviceBuffer<f32>; 3],
    iq: DeviceBuffer<f32>,
    iw: DeviceBuffer<f32>,
    leg: LegBufs,
    rk: RowKernels,
    /// The host's staging of the rows' embeddings, positions, key counts and count words.
    rows_host: Vec<f32>,
    pos_host: Vec<u32>,
    keys_host: Vec<u32>,
    words_host: Vec<u32>,
    /// What [`pass_bytes`] counts for this pass.
    planned: u64,
}

impl PlacedPass {
    /// The pass's buffers at `shape` for a model of `hp` at `ctx` positions, and the q8_1 forms
    /// of 2 to [`CHUNK`] columns its K-quant sites read, which `acts` gains. Refused by name: the
    /// made bytes other than [`pass_bytes`]'s. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        (hp, dims, ctx): (&Hparams, (&KdaDims, &MlaDims), usize),
        (acts, layers): (&mut Acts, &[Layer]),
        shape: PassShape,
    ) -> Result<PlacedPass, GpuError> {
        let stream = gpu.stream();
        let lens = PassLens::of(hp, dims, ctx, shape)?;
        add_row_forms(stream, acts, layers.iter().flat_map(Layer::sites))?;
        let mut f = lens
            .f32s
            .iter()
            .map(|&l| DeviceBuffer::<f32>::zeroed(stream, l.max(1)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let mut u = lens
            .u32s
            .iter()
            .map(|&l| DeviceBuffer::<u32>::zeroed(stream, l.max(1)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let mut next_f = || {
            f.next()
                .ok_or(GpuError::state(PASS, "an f32 buffer of the lens"))
        };
        let mut next_u = || {
            u.next()
                .ok_or(GpuError::state(PASS, "a u32 buffer of the lens"))
        };
        let pass = PlacedPass {
            shape,
            sum: [next_f()?, next_f()?],
            mixed: next_f()?,
            update: next_f()?,
            ffn_xn: next_f()?,
            shared: next_f()?,
            banks: next_f()?,
            act: next_f()?,
            acc: next_f()?,
            tier: next_f()?,
            routed: next_f()?,
            hsum: next_f()?,
            probs: next_f()?,
            weights: next_f()?,
            xn: next_f()?,
            kb: KdaBuffers {
                qkv: next_f()?,
                fa: next_f()?,
                ga: next_f()?,
                beta_raw: next_f()?,
                f: next_f()?,
                z: next_f()?,
                conv: next_f()?,
                beta: next_f()?,
                decay: next_f()?,
                o: next_f()?,
                gated: next_f()?,
            },
            mb: MlaBuffers {
                sel: None,
                qa: next_f()?,
                qr: next_f()?,
                kv: next_f()?,
                gate: next_f()?,
                q: next_f()?,
                qabs: next_f()?,
                att: next_f()?,
                av: next_f()?,
                gated: next_f()?,
                part_v: next_f()?,
                part_ms: next_f()?,
            },
            stage: next_f()?,
            glu: [next_f()?, next_f()?, next_f()?],
            iq: next_f()?,
            iw: next_f()?,
            ids: next_u()?,
            sel: next_u()?,
            tsel: next_u()?,
            pos: next_u()?,
            keys: next_u()?,
            words: next_u()?,
            leg: LegBufs::for_down(stream, lens.leg, DownAct::Q8Act)?,
            rk: RowKernels::load(gpu.context())?,
            rows_host: vec![0.0; shape.rows * hp.n_embd],
            pos_host: vec![0; shape.rows],
            keys_host: vec![0; shape.rows],
            words_host: vec![0; shape.rows * WORDS],
            planned: lens.bytes(),
        };
        let made = pass.bytes() as u64;
        if made != pass.planned {
            return Err(GpuError::shape(
                PASS,
                format!(
                    "the pass's buffers take {made} B; its formula (pass_bytes) counts {}",
                    pass.planned
                ),
            ));
        }
        Ok(pass)
    }

    /// The shape the pass was made at.
    pub(super) fn shape(&self) -> PassShape {
        self.shape
    }

    /// Device bytes the pass holds.
    pub(super) fn bytes(&self) -> usize {
        let k = &self.kb;
        let m = &self.mb;
        let f: [&DeviceBuffer<f32>; F32S] = [
            &self.sum[0],
            &self.sum[1],
            &self.mixed,
            &self.update,
            &self.ffn_xn,
            &self.shared,
            &self.banks,
            &self.act,
            &self.acc,
            &self.tier,
            &self.routed,
            &self.hsum,
            &self.probs,
            &self.weights,
            &self.xn,
            &k.qkv,
            &k.fa,
            &k.ga,
            &k.beta_raw,
            &k.f,
            &k.z,
            &k.conv,
            &k.beta,
            &k.decay,
            &k.o,
            &k.gated,
            &m.qa,
            &m.qr,
            &m.kv,
            &m.gate,
            &m.q,
            &m.qabs,
            &m.att,
            &m.av,
            &m.gated,
            &m.part_v,
            &m.part_ms,
            &self.stage,
            &self.glu[0],
            &self.glu[1],
            &self.glu[2],
            &self.iq,
            &self.iw,
        ];
        let u: [&DeviceBuffer<u32>; U32S] = [
            &self.ids,
            &self.sel,
            &self.tsel,
            &self.pos,
            &self.keys,
            &self.words,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + u.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.leg.bytes()
    }
}

/// Where a placed pass's final mix lands.
pub enum PassTail<'a> {
    /// The last row's mix into `head`, a head of one row, and the head's launches.
    Last(&'a mut Head),
    /// Every row's mix, left in the pass's buffers for [`GpuModel::score_placed`]'s readback.
    Rows,
    /// No mix: the pass leaves the stores and caches for the next.
    Open,
}

/// `len` values of `buf` from value `at`: a pass's rows of one buffer.
fn rows_of(buf: &DeviceBuffer<f32>, at: usize, len: usize) -> Result<Window<'_, f32>, GpuError> {
    Window::<f32>::of(buf, at * 4, len)
}

/// [`rows_of`], writable.
fn rows_of_mut(
    buf: &mut DeviceBuffer<f32>,
    at: usize,
    len: usize,
) -> Result<WindowMut<'_, f32>, GpuError> {
    WindowMut::<f32>::of_mut(buf, at * 4, len)
}

/// [`rows_of`] of a u32 buffer.
fn words_of(buf: &DeviceBuffer<u32>, at: usize, len: usize) -> Result<Window<'_, u32>, GpuError> {
    Window::<u32>::of(buf, at * 4, len)
}

/// The routed block of hosted layer `l` of `layers`.
fn routed_of(layers: &[Layer], l: usize) -> Result<&Routed, GpuError> {
    match &layers[l].ffn {
        Ffn::Routed(r) => Ok(r),
        Ffn::Dense(_) => Err(GpuError::shape(
            PASS,
            format!("layer {l}: a dense block with experts on the host tier"),
        )),
    }
}

/// `0..m` in runs of at most `c`.
fn chunks(m: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..m).step_by(c.max(1)).map(move |c0| (c0, c.min(m - c0)))
}

/// What one pass's walk reads and writes: the body's parts, the placed side and the pass's
/// buffers, at `m` rows.
struct PassWalk<'a> {
    cx: GqaCx<'a>,
    k: &'a Kernels,
    acts: &'a mut Acts,
    lane: &'a super::super::state::Lane,
    dims: (&'a KdaDims, &'a MlaDims),
    layers: &'a mut [Layer],
    router: &'a mut Instance,
    side: &'a Side,
    /// The step's selection buffers on a load that selects, which a latent layer's rows reuse.
    step_sel: Option<&'a mut index::Sel>,
    p: PassParts<'a>,
    /// The blocks' rule and the router's scale.
    rule: (Act, f32),
    n: usize,
    latent: usize,
    n_expert: usize,
    n_used: usize,
    m: usize,
    pushed: usize,
}

/// The pass's buffers but the host sums, which the batch leg holds.
struct PassParts<'a> {
    shape: PassShape,
    sum: &'a mut [DeviceBuffer<f32>; 2],
    mixed: &'a mut DeviceBuffer<f32>,
    update: &'a mut DeviceBuffer<f32>,
    ffn_xn: &'a mut DeviceBuffer<f32>,
    shared: &'a mut DeviceBuffer<f32>,
    banks: &'a mut DeviceBuffer<f32>,
    act: &'a mut DeviceBuffer<f32>,
    acc: &'a mut DeviceBuffer<f32>,
    tier: &'a mut DeviceBuffer<f32>,
    routed: &'a mut DeviceBuffer<f32>,
    probs: &'a mut DeviceBuffer<f32>,
    weights: &'a mut DeviceBuffer<f32>,
    ids: &'a mut DeviceBuffer<u32>,
    sel: &'a mut DeviceBuffer<u32>,
    tsel: &'a mut DeviceBuffer<u32>,
    pos: &'a DeviceBuffer<u32>,
    keys: &'a DeviceBuffer<u32>,
    words: &'a DeviceBuffer<u32>,
    xn: &'a mut DeviceBuffer<f32>,
    kb: &'a mut KdaBuffers,
    mb: &'a mut MlaBuffers,
    stage: &'a mut DeviceBuffer<f32>,
    glu: &'a mut [DeviceBuffer<f32>; 3],
    iq: &'a mut DeviceBuffer<f32>,
    iw: &'a mut DeviceBuffer<f32>,
    leg: &'a mut LegBufs,
    rk: &'a RowKernels,
}

impl PassWalk<'_> {
    /// Layer `l`'s steps 1 to 5 over the pass's rows up to the block ([`super::Walk::attn_half`]
    /// at `m` rows): the mix for the mixer, the bank, the mixer in chunks, the sum after it, the
    /// block's mix normed into `ffn_xn`.
    fn attn_half(&mut self, l: usize) -> Result<(), GpuError> {
        let PassWalk {
            cx,
            k,
            acts,
            lane,
            dims,
            layers,
            step_sel,
            p,
            n,
            m,
            pushed,
            ..
        } = self;
        let (n, m) = (*n, *m);
        let (gpu, w) = (cx.gpu, cx.w);
        let stream = gpu.stream();
        let eps = cx.eps;
        let stride = p.shape.rows * n;
        let layer = &mut layers[l];
        let fault = gpu.layer_sink(l)?;
        let [attn_score, ffn_score] = &layer.scores;
        let [pa, pb] = &mut *p.sum;
        let input: &DeviceBuffer<f32> = if *pushed > 0 {
            k.attn_res.enqueue(
                stream,
                AttnResArgs {
                    banks: &*p.banks,
                    n_banks: *pushed,
                    stride,
                    cur: &*pa,
                    score: w.f32_buf(PASS, attn_score)?,
                    eps,
                    n,
                    m,
                    fault,
                },
                p.mixed,
            )?;
            &*p.mixed
        } else {
            &*pa
        };
        if layer.banks {
            let mut slot = rows_of_mut(p.banks, *pushed * stride, m * n)?;
            copy_row(stream, &*pa, &mut slot, m * n)?;
            *pushed += 1;
        }
        let banks = layer.banks;
        let out: &mut DeviceBuffer<f32> = if banks { &mut *pb } else { &mut *p.update };
        let c = match layer.mixer {
            Mixer::Kda { .. } => CHUNK,
            Mixer::Mla { .. } => p.shape.mla_rows,
        };
        for (c0, cm) in chunks(m, c) {
            let x = rows_of(input, c0 * n, cm * n)?;
            let pos = words_of(p.pos, c0, cm)?;
            let keys = words_of(p.keys, c0, cm)?;
            let words = if step_sel.is_some() {
                Some(words_of(p.words, c0 * WORDS, cm * WORDS)?)
            } else {
                None
            };
            let mut y = rows_of_mut(out, c0 * n, cm * n)?;
            let io = MixerIo {
                input: &x,
                pos: &pos,
                keys: &keys,
                xn: &mut *p.xn,
                kb: &mut *p.kb,
                mb: &mut *p.mb,
                out: &mut y,
            };
            run_mixer_rows(
                cx,
                (*k, &mut **acts, *lane),
                RowsKit {
                    linear: &k.linear,
                    rows: p.rk,
                    stage: &mut *p.stage,
                },
                (l, cm, &mut layer.mixer),
                *dims,
                io,
                step_sel
                    .as_deref_mut()
                    .zip(words.as_deref())
                    .map(|(sel, words)| PassSel {
                        iq: &mut *p.iq,
                        iw: &mut *p.iw,
                        words,
                        sel,
                    }),
            )?;
        }
        if !banks {
            gpu.elem()
                .enqueue_add(stream, &*pa, &*p.update, m * n, pb)?;
        }
        k.attn_res.enqueue(
            stream,
            AttnResArgs {
                banks: &*p.banks,
                n_banks: *pushed,
                stride,
                cur: &*pb,
                score: w.f32_buf(PASS, ffn_score)?,
                eps,
                n,
                m,
                fault,
            },
            p.mixed,
        )?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &*p.mixed,
            w.f32_buf(PASS, &layer.ffn_norm)?,
            eps,
            n,
            m,
            p.ffn_xn,
        )
    }

    /// The block of a layer the card holds whole: the step's whole-card block on each row.
    fn whole(&mut self, l: usize) -> Result<(), GpuError> {
        let n = self.n;
        for t in 0..self.m {
            let xn = rows_of(self.p.ffn_xn, t * n, n)?;
            let mut y = rows_of_mut(self.p.update, t * n, n)?;
            self.router.run(
                &self.cx,
                &mut *self.acts,
                self.rule,
                (l, &self.layers[l].ffn),
                &xn,
                &mut y,
            )?;
        }
        Ok(())
    }

    /// The update added to the sum after the mixer into the first sum buffer.
    fn close(&mut self) -> Result<(), GpuError> {
        let [pa, pb] = &mut *self.p.sum;
        self.cx.gpu.elem().enqueue_add(
            self.cx.gpu.stream(),
            &*pb,
            &*self.p.update,
            self.m * self.n,
            pa,
        )
    }

    /// A hosted layer's front from the block's norm: the router over the rows, the latent down
    /// into the rows the host and the tier read, the card's places and the tier's, and the
    /// download of the rows to the host's batch set.
    fn hosted_front(
        &mut self,
        port: &mut BatchLeg<'_, HostRun>,
        at: At,
        tier: Option<&super::super::tier::TierParts>,
    ) -> Result<(), GpuError> {
        let l = at.layer;
        let (m, n, lat, used) = (self.m, self.n, self.latent, self.n_used);
        let gpu = self.cx.gpu;
        let w = self.cx.w;
        let stream = gpu.stream();
        let r = routed_of(self.layers, l)?;
        let latent = r.latent.as_deref().ok_or_else(|| {
            GpuError::shape(
                PASS,
                format!("layer {l}: a routed block with no latent pair"),
            )
        })?;
        let rin = RouteIn {
            w: w.f32_tensor(PASS, &r.router)?,
            bias: Some(w.f32_buf(PASS, &r.bias)?),
            scale: self.rule.1,
            fault: gpu.layer_sink(l)?,
        };
        let p = &mut self.p;
        self.router.route_rows(
            &self.cx,
            &*p.ffn_xn,
            &rin,
            m,
            RoutedRows {
                probs: &mut *p.probs,
                logits: None,
                ids: &mut *p.ids,
                weights: &mut *p.weights,
            },
        )?;
        let mut kit = RowsKit {
            linear: &self.k.linear,
            rows: p.rk,
            stage: &mut *p.stage,
        };
        for (c0, cm) in chunks(m, CHUNK) {
            let x = rows_of(p.ffn_xn, c0 * n, cm * n)?;
            let mut y = rows_of_mut(p.act, c0 * lat, cm * lat)?;
            rows_kit::project_rows(
                &self.cx,
                &mut kit,
                &mut *self.acts,
                (l, cm),
                (&latent.down, lat),
                &x,
                &mut y,
            )?;
        }
        let side = self.side;
        let n_expert = self.n_expert;
        side.acc_k.enqueue_places(
            stream,
            &Places {
                ids: &*p.ids,
                n: m * used,
                map: side.slots.buf(),
                row_off: side.row_off[l],
                n_expert,
            },
            gpu.layer_sink(l)?,
            &mut *p.sel,
        )?;
        let key = port.key(at);
        match tier.filter(|t| t.k(l) > 0) {
            Some(t) => {
                side.acc_k.enqueue_places(
                    stream,
                    &Places {
                        ids: &*p.ids,
                        n: m * used,
                        map: t.places(),
                        row_off: side.row_off[l],
                        n_expert,
                    },
                    gpu.layer_sink(l)?,
                    &mut *p.tsel,
                )?;
                port.hybrid().enqueue_download_tiered(
                    stream,
                    [&*p.act, &*p.weights],
                    &*p.ids,
                    &[&*p.tsel],
                    key,
                )
            }
            None => port
                .hybrid()
                .enqueue_download(stream, [&*p.act, &*p.weights], &*p.ids, key),
        }
    }

    /// A hosted layer's card work under the host's: the shared stack over the rows and the
    /// card's slots, in chunks of [`CHUNK`] rows.
    fn hosted_shadow(&mut self, l: usize) -> Result<(), GpuError> {
        let (m, n, lat, used) = (self.m, self.n, self.latent, self.n_used);
        let rule = self.rule.0;
        let r = routed_of(self.layers, l)?;
        let n_card = self.side.n_card[l];
        let leg = self.router.leg();
        let p = &mut self.p;
        let mut kit = RowsKit {
            linear: &self.k.linear,
            rows: p.rk,
            stage: &mut *p.stage,
        };
        let stacks = (n_card > 0)
            .then(|| LegStacks::of_resident(self.cx.w, r.stacks.each_ref().map(String::as_str)))
            .transpose()?;
        for (c0, cm) in chunks(m, CHUNK) {
            let x = rows_of(p.ffn_xn, c0 * n, cm * n)?;
            let mut y = rows_of_mut(p.shared, c0 * n, cm * n)?;
            let [gate, up, act] = &mut *p.glu;
            glu_rows(
                &self.cx,
                &mut kit,
                (&mut *self.acts, rule),
                (l, cm),
                &r.shared,
                &x,
                (gate, up, act),
                &mut y,
            )?;
            let Some(stacks) = stacks.as_ref() else {
                continue;
            };
            let normed = rows_of(p.act, c0 * lat, cm * lat)?;
            let sel = words_of(p.sel, c0 * used, cm * used)?;
            let weights = rows_of(p.weights, c0 * used, cm * used)?;
            let mut acc = rows_of_mut(p.acc, c0 * lat, cm * lat)?;
            let run = LegRun {
                layer: l,
                stacks,
                normed: &normed,
                m: cm,
                sel: &sel,
                n_card,
                rule,
            };
            leg.enqueue(
                self.cx.gpu,
                &mut *p.leg,
                &run,
                Some(LegSum {
                    w: &weights,
                    acc: &mut acc,
                }),
            )?;
        }
        Ok(())
    }

    /// A hosted layer's back from the join on: the tier's sum, the cards' and the host's sums
    /// added `(card + tier) + host`, the routed norm, the latent up, the update and the sum.
    fn hosted_back(
        &mut self,
        port: &mut BatchLeg<'_, HostRun>,
        at: At,
        tier: Option<&super::super::tier::TierParts>,
    ) -> Result<(), GpuError> {
        let l = at.layer;
        let (m, n, lat, used) = (self.m, self.n, self.latent, self.n_used);
        let gpu = self.cx.gpu;
        let stream = gpu.stream();
        let n_card = self.side.n_card[l];
        let n_tier = tier.map_or(0, |t| t.k(l));
        let acc_k = &self.side.acc_k;
        if n_tier > 0 {
            let p = &mut self.p;
            let (weights, tsel, tier_sum) = (&*p.weights, &*p.tsel, &mut *p.tier);
            let fault = gpu.layer_sink(l)?;
            port.join_tiered(at, |stream, trows| {
                acc_k.enqueue_card_acc(
                    stream,
                    &CardAcc {
                        down: trows,
                        w: weights,
                        sel: tsel,
                        n: lat,
                        m,
                        n_card: n_tier,
                        n_used: used,
                        w_pitch: used,
                        fault,
                    },
                    tier_sum,
                )
            })?;
        }
        let r = routed_of(self.layers, l)?;
        let latent = r.latent.as_deref().ok_or_else(|| {
            GpuError::shape(
                PASS,
                format!("layer {l}: a routed block with no latent pair"),
            )
        })?;
        let gain = r.sum_norm.as_deref().ok_or_else(|| {
            GpuError::shape(PASS, format!("layer {l}: a routed block with no sum norm"))
        })?;
        let p = &mut self.p;
        let hsum = port.hsum();
        let len = m * lat;
        // The card's slots, then the tier's, then the host's: the step's back.
        let routed: &DeviceBuffer<f32> = match (n_card > 0, n_tier > 0) {
            (true, true) => {
                gpu.elem()
                    .enqueue_add(stream, &*p.acc, &*p.tier, len, &mut *p.routed)?;
                gpu.elem()
                    .enqueue_add(stream, &*p.routed, hsum, len, &mut *p.tier)?;
                &*p.tier
            }
            (true, false) => {
                gpu.elem()
                    .enqueue_add(stream, &*p.acc, hsum, len, &mut *p.routed)?;
                &*p.routed
            }
            (false, true) => {
                gpu.elem()
                    .enqueue_add(stream, &*p.tier, hsum, len, &mut *p.routed)?;
                &*p.routed
            }
            (false, false) => hsum,
        };
        gpu.elem().enqueue_rms_norm(
            stream,
            routed,
            self.cx.w.f32_buf(PASS, gain)?,
            self.cx.eps,
            lat,
            m,
            &mut *p.acc,
        )?;
        let mut kit = RowsKit {
            linear: &self.k.linear,
            rows: p.rk,
            stage: &mut *p.stage,
        };
        for (c0, cm) in chunks(m, CHUNK) {
            let x = rows_of(p.acc, c0 * lat, cm * lat)?;
            let mut y = rows_of_mut(p.mixed, c0 * n, cm * n)?;
            rows_kit::project_rows(
                &self.cx,
                &mut kit,
                &mut *self.acts,
                (l, cm),
                (&latent.up, n),
                &x,
                &mut y,
            )?;
        }
        gpu.elem()
            .enqueue_add(stream, &*p.mixed, &*p.shared, m * n, &mut *p.update)
    }
}

/// The pass's walk over the host tier's batch port, with the tier side the load hangs.
struct PassProgram<'a, 't> {
    walk: PassWalk<'a>,
    tier: Option<&'t super::super::tier::TierParts>,
}

impl<'a> LayerProgram for PassProgram<'a, '_> {
    type Port = BatchLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.walk.side.hosted[at.layer]
    }

    /// A layer the card holds whole entire; a hosted one up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.walk.attn_half(l)?;
        if !self.walk.side.hosted[l] {
            self.walk.whole(l)?;
            return self.walk.close();
        }
        self.walk.hosted_front(port, at, self.tier)
    }

    fn shadow(&mut self, _port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.walk.side.hosted[at.layer] {
            return Ok(());
        }
        self.walk.hosted_shadow(at.layer)
    }

    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.walk.side.hosted[at.layer] {
            return Ok(());
        }
        self.walk.hosted_back(port, at, self.tier)?;
        self.walk.close()
    }
}

impl Instance {
    /// The router over `m` rows of `xn` into `rows` ([`RouterEntry::route_rows`]).
    fn route_rows(
        &self,
        cx: &GqaCx<'_>,
        xn: &DeviceBuffer<f32>,
        rin: &RouteIn<'_>,
        m: usize,
        rows: RoutedRows<'_>,
    ) -> Result<(), GpuError> {
        let launch = (cx.gpu.stream(), cx.gpu, cx.g32);
        match self {
            Instance::E896(r) => r.k.router.route_rows(launch, xn, rin, m, rows),
            Instance::E8(r) => r.k.router.route_rows(launch, xn, rin, m, rows),
        }
    }

    /// The card leg's kernels.
    fn leg(&self) -> &CardLeg {
        match self {
            Instance::E896(r) => &r.k.leg,
            Instance::E8(r) => &r.k.leg,
        }
    }
}

impl super::Body {
    /// The placed pass at positions `pos .. pos + ids.len()` of the sequence the stores stand at:
    /// 1 to the pass's rows, each row the placed step's bits, the stores and caches left as that
    /// many steps leave them and the position book on `pos + ids.len()`. `tail` says where the
    /// last layer's output goes.
    ///
    /// # Errors
    /// [`GpuError`], before anything is enqueued, for a load made without the pass, no ids or
    /// more than the pass's rows, a `pos` the stores do not stand at, rows past the caches, taps
    /// armed, a [`PassTail::Last`] head of more than one row; from the launches otherwise.
    pub(super) fn placed_rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (ids, pos): (&[u32], u32),
        tail: PassTail<'_>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let Some(placed) = self.placed.as_mut() else {
            return Err(GpuError::shape(
                PASS,
                "a whole-card load: its passes are rows passes",
            ));
        };
        let Some(pass) = placed.pass.as_deref_mut() else {
            return Err(GpuError::shape(
                PASS,
                "a placed load made without the pass's buffers: its plan reserved none, so its \
                 prompt steps one id at a time",
            ));
        };
        let rows = pass.shape.rows;
        if !(1..=rows).contains(&m) {
            return Err(GpuError::shape(
                PASS,
                format!("a pass of {m} rows; the load's pass runs 1 to {rows}"),
            ));
        }
        let last = launch_u32(PASS, "rows", m)?;
        let end = pos.checked_add(last).filter(|&e| e as usize <= self.ctx);
        let Some(end) = end else {
            return Err(GpuError::shape(
                PASS,
                format!(
                    "a pass of {m} rows at position {pos} in caches of {}",
                    self.ctx
                ),
            ));
        };
        if pos != self.held {
            return Err(GpuError::shape(
                PASS,
                format!(
                    "a pass at position {pos}; the stores stand at {} (a call that failed \
                     part-way: reset)",
                    self.held
                ),
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::shape(
                PASS,
                "taps are armed: a pass of rows does not tap",
            ));
        }
        if let PassTail::Last(h) = &tail
            && h.m() != 1
        {
            return Err(GpuError::shape(
                PASS,
                format!("a head of {} rows on the last row alone", h.m()),
            ));
        }
        let n = self.hp.n_embd;
        let stream = gpu.stream();
        for (t, &id) in ids.iter().enumerate() {
            placed
                .embed
                .table
                .row_into(id, &mut pass.rows_host[t * n..(t + 1) * n])
                .map_err(|e| GpuError::shape(PASS, format!("token {id}: {e}")))?;
            pass.pos_host[t] = pos + t as u32;
            pass.keys_host[t] = pos + t as u32 + 1;
        }
        rows_of_mut(&mut pass.sum[0], 0, m * n)?
            .copy_from_host(stream, &pass.rows_host[..m * n])?;
        WindowMut::<u32>::of_mut(&mut pass.pos, 0, m)?
            .copy_from_host(stream, &pass.pos_host[..m])?;
        WindowMut::<u32>::of_mut(&mut pass.keys, 0, m)?
            .copy_from_host(stream, &pass.keys_host[..m])?;
        if let Some(sel) = self.s.mla.sel.as_ref() {
            let host = &mut pass.words_host[..m * WORDS];
            index::pass_words(host, pos, sel.top_k())?;
            WindowMut::<u32>::of_mut(&mut pass.words, 0, m * WORDS)?
                .copy_from_host(stream, host)?;
        }
        self.walk_placed(gpu, w, m, tail)?;
        self.held = end;
        Ok(())
    }

    /// The pass's launches: every layer through the batch port, then the final mix.
    fn walk_placed(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        tail: PassTail<'_>,
    ) -> Result<(), GpuError> {
        let super::Body {
            hp,
            kda: kd,
            mla: md,
            act,
            layers,
            k,
            acts,
            lane,
            router,
            ctx,
            out_score,
            placed,
            s: step,
            ..
        } = self;
        let placed = placed
            .as_mut()
            .ok_or(GpuError::state(PASS, "the placed side of a placed pass"))?;
        let Placed {
            shell, side, pass, ..
        } = placed;
        let pass = pass
            .as_deref_mut()
            .ok_or(GpuError::state(PASS, "the pass's buffers"))?;
        // The card's routed block feeds the importance collector; a hosted layer's experts run on
        // the host tier, whose activations the card never holds.
        if crate::imatrix::collector().is_some()
            && let Some(l) = side.hosted.iter().position(|&h| h)
        {
            return Err(GpuError::shape(
                PASS,
                format!(
                    "importance collection: layer {l} has experts on the host tier, whose \
                     activations the card does not hold (place every expert on the cards)"
                ),
            ));
        }
        let n = hp.n_embd;
        let stride = pass.shape.rows * n;
        let mut cx = step_cx(gpu, w, k, hp, *ctx);
        cx.what = PASS;
        let tier = side.tier.as_ref();
        let PlacedPass {
            shape,
            sum,
            mixed,
            update,
            ffn_xn,
            shared,
            banks,
            act: lat_act,
            acc,
            tier: tier_sum,
            routed,
            hsum,
            probs,
            weights,
            ids,
            sel,
            tsel,
            pos,
            keys,
            words,
            xn,
            kb,
            mb,
            stage,
            glu,
            iq,
            iw,
            leg,
            rk,
            ..
        } = pass;
        let walk = PassWalk {
            cx,
            k: &*k,
            acts,
            lane: &*lane,
            dims: (&*kd, &*md),
            layers: &mut layers[..],
            router,
            side: &*side,
            step_sel: step.mla.sel.as_mut(),
            p: PassParts {
                shape: *shape,
                sum,
                mixed,
                update,
                ffn_xn,
                shared,
                banks,
                act: lat_act,
                acc,
                tier: tier_sum,
                routed,
                probs,
                weights,
                ids,
                sel,
                tsel,
                pos: &*pos,
                keys: &*keys,
                words: &*words,
                xn,
                kb,
                mb,
                stage,
                glu,
                iq,
                iw,
                leg,
                rk: &*rk,
            },
            rule: (*act, hp.experts.weights_scale),
            n,
            latent: hp.latent,
            n_expert: hp.experts.n_expert,
            n_used: hp.experts.n_used,
            m,
            pushed: 0,
        };
        let mut prog = PassProgram { walk, tier };
        let n_layers = prog.walk.layers.len();
        let o = Overlap {
            units: 1,
            cols: m,
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(gpu.stream(), &mut shell.hybrid, hsum, shape.rows);
        sched::walk(o, n_layers, &mut leg, &mut prog)?;
        let PassWalk { p, pushed, .. } = prog.walk;
        let mix_into = |y: &mut DeviceBuffer<f32>| {
            k.attn_res.enqueue(
                gpu.stream(),
                AttnResArgs {
                    banks: &*p.banks,
                    n_banks: pushed,
                    stride,
                    cur: &p.sum[0],
                    score: out_score
                        .as_ref()
                        .unwrap_or(w)
                        .f32_buf(PASS, names::OUTPUT_RES_SCORE)?,
                    eps: hp.rms_eps,
                    n,
                    m,
                    fault: gpu.fault_sink(LAYER_HEAD),
                },
                y,
            )
        };
        match tail {
            PassTail::Last(head) => {
                mix_into(&mut *p.mixed)?;
                let last = rows_of(p.mixed, (m - 1) * n, n)?;
                copy_row(gpu.stream(), &last, head.input_mut(), n)?;
                head.enqueue(gpu, w)
            }
            PassTail::Rows => mix_into(&mut *p.mixed),
            PassTail::Open => Ok(()),
        }
    }

    /// The final mix of the last pass's rows, token-major, as [`PassTail::Rows`] left them.
    fn pass_rows_out(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.placed
            .as_ref()
            .and_then(|p| p.pass.as_deref())
            .map(|p| &p.mixed)
            .ok_or(GpuError::state(PASS, "the pass's buffers"))
    }

    /// The rows the load's pass holds; `None` on a load made without it.
    #[must_use]
    pub fn pass_shape(&self) -> Option<PassShape> {
        self.placed.as_ref()?.pass.as_deref().map(PlacedPass::shape)
    }

    /// The file's lightning indexer, `None` on a K3 file: past its top-k a latent layer attends
    /// the rows it picks, at a step and at every row of a pass ([`super::index`]).
    #[must_use]
    pub fn indexer(&self) -> Option<&Indexer> {
        self.hp.indexer.as_ref()
    }
}

impl GpuModel<super::Body> {
    /// Run `ids` from the model's position as placed passes of up to the load's pass rows and
    /// return the argmax after the last: every row the placed step's bits, the head on the last
    /// row alone. Refused by name with no ids, on a load made without the pass and as
    /// [`super::Body::placed_rows`] refuses.
    ///
    /// # Errors
    /// [`GpuError`] as above, and the head's fault.
    pub fn prompt_placed(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        const WHAT: &str = "kimi_k3 GpuModel::prompt_placed";
        let rows = self
            .body(WHAT)?
            .pass_shape()
            .ok_or(GpuError::shape(WHAT, "a load made without the placed pass"))?
            .rows;
        self.run_rows(ids.len(), WHAT, |gpu, w, body, head, pos| {
            let total = ids.len();
            for (c, chunk) in ids.chunks(rows).enumerate() {
                let at = c * rows;
                let tail = if at + chunk.len() == total {
                    PassTail::Last(&mut *head)
                } else {
                    PassTail::Open
                };
                let first = pos + launch_u32(WHAT, "row", at)?;
                body.placed_rows(gpu, w, (chunk, first), tail)?;
            }
            Ok(true)
        })?
        .ok_or(GpuError::state(
            WHAT,
            "the head's token of the call's last row",
        ))
    }

    /// Run `ids` from the model's position as placed passes and return the logits of the last
    /// `n` rows, row-major (`n` rows of the vocabulary), each row the placed step's logits at its
    /// position: the passes are cut from the end, so the last holds the last `n` rows, and its
    /// rows run through the head (the host's projection [`HOST_HEAD_ROWS`] rows at a time, or a
    /// card head's launches one row at a time). Refused by name: no ids, `n` of 0 or past the ids
    /// or the pass's rows, a load made without the pass, and as [`super::Body::placed_rows`]
    /// refuses.
    ///
    /// # Errors
    /// [`GpuError`] as above, and the head's fault.
    pub fn score_placed(&mut self, ids: &[u32], n: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT: &str = "kimi_k3 GpuModel::score_placed";
        let rows = self
            .body(WHAT)?
            .pass_shape()
            .ok_or(GpuError::shape(WHAT, "a load made without the placed pass"))?
            .rows;
        if n == 0 || n > ids.len() || n > rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the last {n} rows of {} ids: 1 to the ids and to the pass's {rows} rows",
                    ids.len()
                ),
            ));
        }
        let total = ids.len();
        let head_len = total - rows.min(total);
        let mut out = Vec::new();
        self.run_rows(total, WHAT, |gpu, w, body, head, pos| {
            for (c, chunk) in ids[..head_len].chunks(rows).enumerate() {
                let first = pos + launch_u32(WHAT, "row", c * rows)?;
                body.placed_rows(gpu, w, (chunk, first), PassTail::Open)?;
            }
            let last = &ids[head_len..];
            let first = pos + launch_u32(WHAT, "row", head_len)?;
            body.placed_rows(gpu, w, (last, first), PassTail::Rows)?;
            let width = body.hp.n_embd;
            let mixed = body.pass_rows_out()?;
            let from = last.len() - n;
            let window = rows_of(mixed, from * width, n * width)?;
            for (r0, rn) in chunks(n, HOST_HEAD_ROWS) {
                let part = rows_of(&window, r0 * width, rn * width)?;
                match head.host_logits_rows(gpu, &part, rn)? {
                    Some(logits) => out.extend_from_slice(&logits),
                    None => {
                        for t in 0..rn {
                            let row = rows_of(&part, t * width, width)?;
                            copy_row(gpu.stream(), &row, head.input_mut(), width)?;
                            head.enqueue(gpu, w)?;
                            out.extend_from_slice(&head.logits_to_host(gpu)?);
                        }
                    }
                }
            }
            let tail = rows_of(&window, (n - 1) * width, width)?;
            copy_row(gpu.stream(), &tail, head.input_mut(), width)?;
            head.enqueue(gpu, w)?;
            Ok(true)
        })?;
        Ok(out)
    }
}
