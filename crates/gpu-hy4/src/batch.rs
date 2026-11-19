//! The prompt batch: a call of `T` ids walked as one pass per layer over every row, which leaves the
//! stores where `T` decode steps over the same ids leave them (every sparse layer's cache rows, index
//! ring, pooled keys and every window layer's ring at those positions, the engram history, the
//! position) and the logits of the last row.
//!
//! The walk is layer-major over the call's rows, as the expert stream needs it: a layer's experts are
//! read once for the whole call. Per layer:
//! - an engram site, in chunks of [`Cfg::q_rows`] rows;
//! - the attention sub-layer in the same chunks, each chunk after the one before it so its keys are in
//!   the layer's store: the identity mix's pre and the norm, every projection one grouped int8 GEMM
//!   over the chunk's rows ([`project_wide`]), the query's rope and the key's norm, rope and append at
//!   each row's own position and rope row, on a sparse layer the selector over the chunk's rows (the
//!   index rows appended first, then each row's pool, score and list: one launch each) and the flash
//!   over each row's list, on a window layer the chunk's rows staged beside the ring (the flash reads
//!   ring then stage, the ring takes the stage after) and the per-head q8_0 gemvs of the absorbed
//!   query and of the values, [`MCOL`] rows a launch;
//! - the feed-forward sub-layer in units of [`Cfg::ffn_rows`] rows: the dense lead's block, or the
//!   routed block, whose experts are one grouped GEMM over the unit's `rows × 16` slots (the common
//!   `RouteBlock`: the route table, the gate and up GEMMs, the gate-first clamp into the down's form
//!   and the down GEMM) summed in slot order by the card sum the decode step uses.
//!
//! The head runs on the call's last row alone. Every launch is a kernel the decode step runs, over
//! `m` rows, so a row's values differ from the step's only where a wide GEMM accumulates in another
//! order than the gemv (both quantize their input the same way).
//!
//! A unit's expert slots are at most [`GEMM_MAX_SLOTS`], so [`Cfg::ffn_rows`] is at most 2,560 at the
//! top 16. A walk over a tier's experts (the host union, the expert stream) is the tier's port on this
//! walk's routed block, which this file does not carry.

use std::ops::Range;

use bloomery_gpu::arch::hybrid::{Site, WideCx, project_wide, wide_acts_at};
use bloomery_gpu::card_leg::{LegStacks, RouteIo, RouteW};
use bloomery_gpu::card_slots::{CardAcc, CardSlotKernels};
use bloomery_gpu::engram_delta::{DeltaArgs, GateOnlyArgs, KeyNormArgs, RowsArgs};
use bloomery_gpu::flash::{
    FlashGeom, FlashSelArgs, FlashWindowArgs, RingCommitArgs, partials_ms_len, partials_v_len,
    window_partials_lens,
};
use bloomery_gpu::gemm::{
    GEMM_MAX_SLOTS, Gemm32Kernels, GemmAct, GemmAct32, GemmKernels, GemmRoute,
};
use bloomery_gpu::hc_ident::{HC, HeadArgs, MIX, PostArgs, PreArgs};
use bloomery_gpu::head::Head;
use bloomery_gpu::kpool::{TailScoreArgs, weights_scale};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::latent::{INDEX_HEAD, IndexKeyRmsArgs, IndexPoolRopeArgs, IndexProj, POOL, Rows};
use bloomery_gpu::linear::out_gate::OutGateArgs;
use bloomery_gpu::media::{CallRows, widen_bf16};
use bloomery_gpu::q8f32::Q8_0GemvHeadsMcolArgs;
use bloomery_gpu::qsa::TopkTailArgs;
use bloomery_gpu::rope_table::{Direction, RopeRows};
use bloomery_gpu::routed_block::{RouteIn, RoutedRows, RouterEntry};
use bloomery_gpu::site::{WideActs, WideKernels};
use bloomery_gpu::token_major::TokenMajor;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel, Window, WindowMut, copy_row};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::arch::hy4::names::{self, Sub};
use model::arch::models::OutGate;

use crate::alone::Selection;
use crate::body::{
    Body, ENGRAM_EPS, ENGRAM_W, Ffn, HostIn, Kernels, LATENT, Layer, ROPE, ROW, Router, Shape,
    Store, TAP_STAGES,
};
use crate::step::{ATTN_NORM, ATTN_OUT, ENGRAM_OUT, FFN_NORM, FFN_OUT, HC_ATTN_OUT, L_OUT};

mod pass;

pub use pass::{PASS_ROWS, PassShape, PlacedPass, SHAPES, WALK_UNITS, pass_bytes, prompt_placed};

/// What this file's refusals name.
const WHAT: &str = "hy4::batch";

/// The most rows an attention chunk holds.
pub const Q_MAX: usize = 256;

/// The rows of a sparse layer's index ring: a chunk's appends and the three positions before them,
/// which the pool a chunk's first rows complete reads (`latent::enqueue_index_pool_norm_rope`).
pub(crate) const INDEX_RING: usize = Q_MAX + POOL - 1;

/// The columns of one per-head q8_0 launch: the launcher's most.
const MCOL: usize = 8;

/// The router's picks a token and its experts.
const N_USED: usize = <Router as RouterEntry>::N_USED;
const N_EXPERT: usize = <Router as RouterEntry>::N_EXPERT;

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::shape(WHAT, detail)
}

/// The walk's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cfg {
    /// The most rows one walk holds: the streams' rows.
    pub cap: usize,
    /// The rows of an engram or attention chunk, at most [`Q_MAX`].
    pub q_rows: usize,
    /// The rows of a feed-forward unit: its experts' slots (`rows × 16`) are at most
    /// [`GEMM_MAX_SLOTS`].
    pub ffn_rows: usize,
}

impl Cfg {
    /// The released model's prompt batch: 4,096 rows a walk.
    pub const PROMPT: Cfg = Cfg {
        cap: 4096,
        q_rows: Q_MAX,
        ffn_rows: GEMM_MAX_SLOTS / N_USED,
    };

    fn check(self) -> Result<(), GpuError> {
        let Cfg {
            cap,
            q_rows,
            ffn_rows,
        } = self;
        if cap == 0 || q_rows == 0 || ffn_rows == 0 {
            return Err(shape(format!("{self:?}: a walk needs rows")));
        }
        if q_rows > Q_MAX {
            return Err(shape(format!(
                "{self:?}: an attention chunk holds at most {Q_MAX} rows (the index ring's)"
            )));
        }
        if ffn_rows * N_USED > GEMM_MAX_SLOTS {
            return Err(shape(format!(
                "{self:?}: {ffn_rows} rows of {N_USED} slots pass the GEMM's {GEMM_MAX_SLOTS}"
            )));
        }
        Ok(())
    }
}

/// The input forms the routed block reads, held at the unit's rows.
struct ExpertActs {
    /// The block's input rows (the latent rows) in q8_1 blocks of 128, or in q8 blocks of 32.
    x128: Option<GemmAct>,
    x32: Option<GemmAct32>,
    /// The down GEMM's rows (the SwiGLU of the slots) in the same forms.
    h128: Option<GemmAct>,
    h32: Option<GemmAct32>,
}

/// A walk's device rows. The streams and the per-row words are the call's (`cap` rows); every other
/// buffer is a chunk's or a unit's, its row 0 the chunk's first row.
struct Buf {
    hs: [DeviceBuffer<f32>; 2],
    /// Each row's position, its live key count, and the chunks' first positions.
    pos: DeviceBuffer<u32>,
    keys: DeviceBuffer<u32>,
    bases: DeviceBuffer<u32>,
    /// The rows of a window chunk's stage: row `r` is the chunk's row `r`.
    iota: DeviceBuffer<u32>,
    stage: DeviceTensor<u16>,
    /// Each row's cos/sin pair at the sparse and the window rope base.
    cs: [DeviceBuffer<f32>; 2],
    e_rows: Vec<DeviceBuffer<f32>>,
    e_valid: Vec<DeviceBuffer<f32>>,
    mix: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    qa: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    q_cur: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    iq: DeviceBuffer<f32>,
    ik: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    /// The selector's query and head weights a feature at a time, a token to a column: the layout
    /// the score pass reads (`iq`, `wi` are a token a row, as the projections write them).
    iq_cols: DeviceBuffer<f32>,
    wi_cols: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    list: DeviceBuffer<u32>,
    n_sel: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
    fg: DeviceBuffer<f32>,
    fu: DeviceBuffer<f32>,
    fa: DeviceBuffer<f32>,
    lat: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    lat_up: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    f: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The routed GEMMs' rows: the gate's and up's, and the down's, a slot each.
    gate_rows: DeviceBuffer<f32>,
    up_rows: DeviceBuffer<f32>,
    down_rows: DeviceBuffer<f32>,
    e_norm: DeviceBuffer<f32>,
    e_kv: DeviceBuffer<f32>,
    e_kn: DeviceBuffer<f32>,
    e_gate: DeviceBuffer<f32>,
    e_ones: DeviceBuffer<f32>,
}

/// The host's buffers of one walk.
struct HostRows {
    streams: Vec<f32>,
    cs: [Vec<f32>; 2],
    pos: Vec<u32>,
    keys: Vec<u32>,
    bases: Vec<u32>,
    tokens: Vec<Option<u32>>,
    ids: Vec<u32>,
    live: Vec<bool>,
    vals: Vec<f32>,
    valid: Vec<f32>,
}

/// The gate's copies of a walk's stages ([`TAP_STAGES`], every layer) for every row, of the head's
/// fold at the last row, of each routed layer's picked experts for every row (the ids' bits) and of
/// each sparse layer's selection at the last row (the list's rows, their count and the pool scores,
/// bits in that order).
struct Taps {
    stages: Vec<DeviceBuffer<f32>>,
    head: DeviceBuffer<f32>,
    picks: Vec<DeviceBuffer<f32>>,
    sel: Vec<DeviceBuffer<f32>>,
}

/// What a walk's taps hold, read back ([`Prompt::taps`]).
pub struct WalkTaps {
    /// Layer-major, [`TAP_STAGES`] a layer, each `rows × width` ([`stage_width`]).
    pub stages: Vec<Vec<f32>>,
    /// The head's fold at the last row.
    pub head: Vec<f32>,
    /// Each routed layer's picked experts of every row, `16` a row in the router's slot order;
    /// empty on a dense layer.
    pub picks: Vec<Vec<u32>>,
    /// Each sparse layer's selection at the last row; `None` on a window layer.
    pub sel: Vec<Option<Selection>>,
}

/// The prompt batch's state: its kernels, the route tables, the wide forms and the rows.
pub struct Prompt {
    cfg: Cfg,
    layers: usize,
    hidden: usize,
    /// Each layer's kind (routed, sparse) and the selector's widths.
    kinds: Vec<(bool, bool)>,
    top_k: usize,
    pools: usize,
    gemm: GemmKernels,
    slots: CardSlotKernels,
    cols_of: TokenMajor,
    /// The one-expert table every projection of a chunk runs over, and the routed block's.
    dense: GemmRoute,
    experts: GemmRoute,
    /// The wide forms at each input width a site reads.
    wide: Vec<(usize, WideActs)>,
    acts: ExpertActs,
    /// The rope tail's gather tables of a chunk at each head count a layer has: source and
    /// destination.
    gather: Vec<(usize, DeviceBuffer<u32>, DeviceBuffer<u32>)>,
    cols: usize,
    buf: Buf,
    host: HostRows,
    taps: Option<Taps>,
    bytes: usize,
}

struct Alloc<'a> {
    stream: &'a CudaStream,
    bytes: usize,
}

impl Alloc<'_> {
    fn f(&mut self, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
        let b = DeviceBuffer::<f32>::zeroed(self.stream, n.max(1))?;
        self.bytes += b.num_bytes();
        Ok(b)
    }

    fn u(&mut self, n: usize) -> Result<DeviceBuffer<u32>, GpuError> {
        let b = DeviceBuffer::<u32>::zeroed(self.stream, n.max(1))?;
        self.bytes += b.num_bytes();
        Ok(b)
    }
}

impl Prompt {
    /// The batch's buffers for walks of `cfg` over `body`'s layers. Load-time only.
    pub fn new(gpu: &Gpu, w: &Weights, body: &Body, cfg: Cfg) -> Result<Prompt, GpuError> {
        cfg.check()?;
        let stream = gpu.stream();
        let sh = &body.sh;
        let (n, q) = (sh.hidden, cfg.q_rows);
        let r = q.max(cfg.ffn_rows);
        let slots = cfg.ffn_rows * N_USED;
        let heads = sh.max_heads;
        let cols = body.host.cols;
        let sites_n = body.layers.iter().filter(|l| l.engram.is_some()).count();
        let pools = sh.ctx.div_ceil(POOL);
        let ff = body
            .layers
            .iter()
            .map(|l| match &l.ffn {
                Ffn::Dense(g) => g.ff,
                Ffn::Routed { shared, .. } => shared.ff,
            })
            .max()
            .unwrap_or(1);
        let mut a = Alloc { stream, bytes: 0 };

        // The wide forms at each input width a site reads, at the most rows a unit holds.
        let sites: Vec<&Site> = body.layers.iter().flat_map(Layer::sites).collect();
        let mut widths: Vec<usize> = sites.iter().map(|s| s.k()).collect();
        widths.sort_unstable();
        widths.dedup();
        let wide = widths
            .into_iter()
            .map(|k| Ok((k, wide_acts_at(stream, sites.iter().copied(), (r, k))?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        a.bytes += wide.iter().map(|(_, a)| a.bytes()).sum::<usize>();

        // The forms the routed blocks of the file's stacks read.
        let mut forms = [false; 4];
        for layer in &body.layers {
            let Ffn::Routed { stacks, .. } = &layer.ffn else {
                continue;
            };
            let st = LegStacks::of(w, stacks.each_ref().map(String::as_str))?;
            let blk = st.block(sh.expert_ff, sh.latent)?;
            match (blk.gate, blk.up) {
                (RouteW::Grouped(..), RouteW::Grouped(..)) => forms[0] = true,
                (RouteW::Blocks32(_), RouteW::Blocks32(_)) => forms[1] = true,
                _ => {
                    return Err(shape(format!(
                        "{}: a gate and an up of two forms",
                        stacks[0]
                    )));
                }
            }
            match blk.down {
                RouteW::Grouped(..) => forms[2] = true,
                RouteW::Blocks32(_) => forms[3] = true,
            }
        }
        let acts = ExpertActs {
            x128: forms[0]
                .then(|| GemmAct::new(stream, cfg.ffn_rows, sh.latent))
                .transpose()?,
            x32: forms[1]
                .then(|| GemmAct32::new(stream, cfg.ffn_rows, sh.latent))
                .transpose()?,
            h128: forms[2]
                .then(|| GemmAct::new(stream, slots, sh.expert_ff))
                .transpose()?,
            h32: forms[3]
                .then(|| GemmAct32::new(stream, slots, sh.expert_ff))
                .transpose()?,
        };
        a.bytes += acts.x128.as_ref().map_or(0, GemmAct::bytes)
            + acts.x32.as_ref().map_or(0, GemmAct32::bytes)
            + acts.h128.as_ref().map_or(0, GemmAct::bytes)
            + acts.h32.as_ref().map_or(0, GemmAct32::bytes);

        let dense = GemmRoute::new(stream, r, 1)?;
        let experts = GemmRoute::new(stream, slots, N_EXPERT)?;
        a.bytes += dense.bytes() + experts.bytes();

        // The rope tail's gather: head `h`'s value `j` of row `t` from the rotated query's
        // `(t·heads + h)·head_k + nope + j` to the cache-width query's `(t·heads + h)·ROW + j`.
        let index = |v: usize| u32::try_from(v).map_err(|_| shape("a query index past u32"));
        let mut counts: Vec<usize> = body.layers.iter().map(|l| l.heads).collect();
        counts.sort_unstable();
        counts.dedup();
        let mut gather = Vec::new();
        for hd in counts {
            let (mut src, mut dst) = (Vec::new(), Vec::new());
            for t in 0..q {
                for h in 0..hd {
                    for j in 0..ROPE {
                        src.push(index((t * hd + h) * sh.head_k + sh.nope + j)?);
                        dst.push(index((t * hd + h) * ROW + j)?);
                    }
                }
            }
            let (s, d) = (
                DeviceBuffer::from_host(stream, &src)?,
                DeviceBuffer::from_host(stream, &dst)?,
            );
            a.bytes += s.num_bytes() + d.num_bytes();
            gather.push((hd, s, d));
        }

        let (wv, wms) = window_partials_lens(q * heads, sh.window);
        let part_v = partials_v_len(q * heads, sh.top_k).max(wv);
        let part_ms = partials_ms_len(q * heads, sh.top_k).max(wms);
        let chunks = cfg.cap.div_ceil(q);
        let buf = Buf {
            hs: [a.f(cfg.cap * HC * n)?, a.f(cfg.cap * HC * n)?],
            pos: a.u(cfg.cap)?,
            keys: a.u(cfg.cap)?,
            bases: a.u(chunks)?,
            iota: {
                let v: Vec<u32> = (0..q).map(|i| i as u32).collect();
                let b = DeviceBuffer::from_host(stream, &v)?;
                a.bytes += b.num_bytes();
                b
            },
            stage: DeviceTensor::zeroed(stream, q, ROW)?,
            cs: [a.f(cfg.cap * ROPE)?, a.f(cfg.cap * ROPE)?],
            e_rows: (0..sites_n)
                .map(|_| a.f(cfg.cap * cols * ENGRAM_W))
                .collect::<Result<_, _>>()?,
            e_valid: (0..sites_n)
                .map(|_| a.f(cfg.cap * cols))
                .collect::<Result<_, _>>()?,
            mix: a.f(r * MIX)?,
            y: a.f(r * n)?,
            xn: a.f(r * n)?,
            qa: a.f(q * sh.q_lora)?,
            qr: a.f(q * sh.q_lora)?,
            q: a.f(q * heads * sh.head_k)?,
            q_rope: a.f(q * heads * sh.head_k)?,
            q_cur: a.f(q * heads * ROW)?,
            kv_a: a.f(q * ROW)?,
            kv_s: a.f(q * ROW)?,
            kvr: a.f(q * ROW)?,
            iq: a.f(q * sh.idx_heads * INDEX_HEAD)?,
            ik: a.f(q * INDEX_HEAD)?,
            wi: a.f(q * sh.idx_heads)?,
            iq_cols: a.f(q * sh.idx_heads * INDEX_HEAD)?,
            wi_cols: a.f(q * sh.idx_heads)?,
            part_v: a.f(part_v)?,
            part_ms: a.f(part_ms)?,
            att: a.f(q * heads * LATENT)?,
            av: a.f(q * heads * sh.head_v)?,
            g: a.f(q * heads * sh.head_v)?,
            gated: a.f(q * heads * sh.head_v)?,
            o: a.f(r * n)?,
            out: a.f(r * n)?,
            list: a.u(q * sh.top_k)?,
            n_sel: a.u(q)?,
            scores: a.f(q * pools)?,
            fg: a.f(r * ff)?,
            fu: a.f(r * ff)?,
            fa: a.f(r * ff)?,
            lat: a.f(r * sh.latent)?,
            routed: a.f(r * sh.latent)?,
            lat_up: a.f(r * n)?,
            shared: a.f(r * n)?,
            f: a.f(r * n)?,
            probs: a.f(cfg.ffn_rows * N_EXPERT)?,
            ids: a.u(slots)?,
            weights: a.f(slots)?,
            gate_rows: a.f(slots * sh.expert_ff)?,
            up_rows: a.f(slots * sh.expert_ff)?,
            down_rows: a.f(slots * sh.latent)?,
            e_norm: a.f(q * cols * ENGRAM_W)?,
            e_kv: a.f(q * (HC + 1) * n)?,
            e_kn: a.f(q * HC * n)?,
            e_gate: a.f(q * HC)?,
            e_ones: {
                let b = DeviceBuffer::from_host(stream, &vec![1.0f32; HC * n])?;
                a.bytes += b.num_bytes();
                b
            },
        };
        Ok(Prompt {
            cfg,
            layers: body.layers.len(),
            hidden: n,
            kinds: body
                .layers
                .iter()
                .map(|l| (matches!(l.ffn, Ffn::Routed { .. }), l.dsa))
                .collect(),
            top_k: sh.top_k,
            pools,
            gemm: GemmKernels::load(gpu.context())?,
            slots: CardSlotKernels::load(gpu.context())?,
            cols_of: TokenMajor::load(gpu.context())?,
            dense,
            experts,
            wide,
            acts,
            gather,
            cols,
            buf,
            host: HostRows {
                streams: vec![0.0; q * HC * n],
                cs: [Vec::new(), Vec::new()],
                pos: Vec::new(),
                keys: Vec::new(),
                bases: Vec::new(),
                tokens: Vec::new(),
                ids: Vec::new(),
                live: Vec::new(),
                vals: Vec::new(),
                valid: Vec::new(),
            },
            taps: None,
            bytes: a.bytes,
        })
    }

    /// The device bytes the batch holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The walk's geometry.
    #[must_use]
    pub fn cfg(&self) -> Cfg {
        self.cfg
    }

    /// Arm the taps: every later walk copies every stage ([`TAP_STAGES`]) of every layer for each
    /// of its rows, the head's fold at its last row, each routed layer's picks and each sparse
    /// layer's selection at the last row.
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let (cap, n) = (self.cfg.cap, self.hidden);
        let mut stages = Vec::with_capacity(self.layers * TAP_STAGES.len());
        for _ in 0..self.layers {
            for stage in 0..TAP_STAGES.len() {
                stages.push(DeviceBuffer::zeroed(stream, cap * stage_width(stage, n))?);
            }
        }
        let width = |on: bool, w: usize| if on { w } else { 1 };
        let picks = self
            .kinds
            .iter()
            .map(|&(routed, _)| DeviceBuffer::zeroed(stream, cap * width(routed, N_USED)))
            .collect::<Result<_, _>>()?;
        let sel = self
            .kinds
            .iter()
            .map(|&(_, dsa)| DeviceBuffer::zeroed(stream, width(dsa, self.top_k + 1 + self.pools)))
            .collect::<Result<_, _>>()?;
        self.taps = Some(Taps {
            stages,
            head: DeviceBuffer::zeroed(stream, n)?,
            picks,
            sel,
        });
        Ok(())
    }

    /// Drop the taps.
    pub fn disarm_taps(&mut self) {
        self.taps = None;
    }

    /// The last walk's taps for its first `rows` rows, read back. Synchronizes `stream`.
    pub fn taps(&self, stream: &CudaStream, rows: usize) -> Result<WalkTaps, GpuError> {
        let t = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps",
        })?;
        let stages = t
            .stages
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let mut v = b.to_host_vec(stream)?;
                v.truncate(rows * stage_width(i % TAP_STAGES.len(), self.hidden));
                Ok(v)
            })
            .collect::<Result<_, GpuError>>()?;
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        let picks = t
            .picks
            .iter()
            .zip(&self.kinds)
            .map(|(b, &(routed, _))| {
                if !routed {
                    return Ok(Vec::new());
                }
                let mut v = bits(b.to_host_vec(stream)?);
                v.truncate(rows * N_USED);
                Ok(v)
            })
            .collect::<Result<_, GpuError>>()?;
        let sel = t
            .sel
            .iter()
            .zip(&self.kinds)
            .map(|(b, &(_, dsa))| {
                if !dsa {
                    return Ok(None);
                }
                let v = b.to_host_vec(stream)?;
                let count = (v[self.top_k].to_bits() as usize).min(self.top_k);
                let list = bits(v[..count].to_vec());
                Ok(Some((list, v[self.top_k + 1..].to_vec())))
            })
            .collect::<Result<_, GpuError>>()?;
        Ok(WalkTaps {
            stages,
            head: t.head.to_host_vec(stream)?,
            picks,
            sel,
        })
    }
}

/// The values a row of tap stage `stage` holds.
#[must_use]
pub fn stage_width(stage: usize, hidden: usize) -> usize {
    match stage {
        ENGRAM_OUT | HC_ATTN_OUT | L_OUT => HC * hidden,
        _ => hidden,
    }
}

/// `rows` rows of `width` values of `b` from row `r0`.
fn win(
    b: &DeviceBuffer<f32>,
    r0: usize,
    rows: usize,
    width: usize,
) -> Result<Window<'_, f32>, GpuError> {
    Window::<f32>::of(b, 4 * r0 * width, rows * width)
}

fn win_mut(
    b: &mut DeviceBuffer<f32>,
    r0: usize,
    rows: usize,
    width: usize,
) -> Result<WindowMut<'_, f32>, GpuError> {
    WindowMut::<f32>::of_mut(b, 4 * r0 * width, rows * width)
}

fn winu(b: &DeviceBuffer<u32>, r0: usize, rows: usize) -> Result<Window<'_, u32>, GpuError> {
    Window::<u32>::of(b, 4 * r0, rows)
}

/// The consecutive ranges of `len` rows `step` at a time.
fn chunks(len: usize, step: usize) -> impl Iterator<Item = Range<usize>> {
    (0..len)
        .step_by(step)
        .map(move |lo| lo..(lo + step).min(len))
}

/// What the walk reads of the body and does not write.
#[derive(Clone, Copy)]
struct Env<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    sh: Shape,
    layers: &'a [Layer],
    k: &'a Kernels,
    rope_rows: &'a RopeRows,
}

/// One chunk or unit of a layer: the streams' buffer (`cur` holds its input) and the rows.
#[derive(Clone, Copy)]
struct Span {
    cur: usize,
    /// The first row in the call and the chunk's index among a layer's chunks.
    r0: usize,
    index: usize,
    n: usize,
}

/// The wide projections' kernels and table, beside the forms they quantize into.
struct WideP<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    g32: &'a Gemm32Kernels,
    gemm: &'a GemmKernels,
    dense: &'a GemmRoute,
}

/// `y = W · x` of each of `sites` over `m` rows of `x`.
fn project(
    wp: &WideP<'_>,
    arms: &mut [(usize, WideActs)],
    l: usize,
    m: usize,
    x: &DeviceBuffer<f32>,
    sites: &mut [(&Site, usize, &mut DeviceBuffer<f32>)],
) -> Result<(), GpuError> {
    let k = sites
        .first()
        .map(|s| s.0.k())
        .ok_or_else(|| shape("a projection of no site"))?;
    let acts = arms
        .iter_mut()
        .find(|a| a.0 == k)
        .map(|a| &mut a.1)
        .ok_or_else(|| shape(format!("no wide forms of width {k}")))?;
    let cx = WideCx {
        what: WHAT,
        gpu: wp.gpu,
        w: wp.w,
        wk: WideKernels {
            gemm: wp.gemm,
            g32: Some(wp.g32),
        },
        m,
    };
    project_wide(&cx, (acts, wp.dense), l, x, sites)
}

/// Copy the first `n` rows of `src`, from row `src_r0`, into the tap of `stage` of layer `l` at
/// row `dst_r0`.
fn tap(
    gpu: &Gpu,
    taps: &mut Option<Taps>,
    (l, stage): (usize, usize),
    src: (&DeviceBuffer<f32>, usize),
    (dst_r0, n, width): (usize, usize, usize),
) -> Result<(), GpuError> {
    let Some(t) = taps.as_mut() else {
        return Ok(());
    };
    let from = win(src.0, src.1, n, width)?;
    let mut to = win_mut(
        &mut t.stages[l * TAP_STAGES.len() + stage],
        dst_r0,
        n,
        width,
    )?;
    copy_row(gpu.stream(), &from, &mut to, n * width)
}

/// Copy `n` 4-byte words of `src` from word `src_at` into the tap buffer `dst` at word `dst_at`,
/// bits unchanged.
fn tap_words<T>(
    gpu: &Gpu,
    dst: &mut DeviceBuffer<f32>,
    src: &DeviceBuffer<T>,
    (src_at, dst_at, n): (usize, usize, usize),
) -> Result<(), GpuError> {
    let from = Window::<f32>::of(src, 4 * src_at, n)?;
    let mut to = WindowMut::<f32>::of_mut(dst, 4 * dst_at, n)?;
    copy_row(gpu.stream(), &from, &mut to, n)
}

/// Walk `toks` (positions `pos0 ..`) through the layers; the head runs after the last layer when
/// `last`.
fn walk(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    p: &mut Prompt,
    (toks, pos0): (&[u32], u32),
    last: bool,
) -> Result<(), GpuError> {
    let Body {
        sh,
        layers,
        k,
        stores,
        host,
        rope_rows,
        ..
    } = body;
    let sh = *sh;
    let t = toks.len();
    if t == 0 || t > p.cfg.cap {
        return Err(shape(format!(
            "a walk of {t} rows in buffers of {}",
            p.cfg.cap
        )));
    }
    stage(host, p, gpu.stream(), &sh, (toks, pos0), None)?;
    let env = Env {
        gpu,
        w,
        sh,
        layers,
        k,
        rope_rows,
    };
    let mut cur = 0usize;
    for (l, layer) in layers.iter().enumerate() {
        if layer.engram.is_some() {
            for (index, rows) in chunks(t, p.cfg.q_rows).enumerate() {
                let sp = Span {
                    cur,
                    r0: rows.start,
                    index,
                    n: rows.len(),
                };
                engram(&env, p, l, sp)?;
            }
            cur ^= 1;
        }
        for (index, rows) in chunks(t, p.cfg.q_rows).enumerate() {
            let sp = Span {
                cur,
                r0: rows.start,
                index,
                n: rows.len(),
            };
            attention(&env, stores, p, l, sp)?;
        }
        cur ^= 1;
        for (index, rows) in chunks(t, p.cfg.ffn_rows).enumerate() {
            let sp = Span {
                cur,
                r0: rows.start,
                index,
                n: rows.len(),
            };
            ffn(&env, p, l, sp)?;
        }
        cur ^= 1;
    }
    if last {
        let n = sh.hidden;
        let x = win(&p.buf.hs[cur], t - 1, 1, HC * n)?;
        k.hc.enqueue_head(
            gpu.stream(),
            HeadArgs {
                x: &x,
                w: w.f32_buf(WHAT, names::OUTPUT_HC_FN)?,
                scale: w.f32_buf(WHAT, names::OUTPUT_HC_SCALE)?,
                base: w.f32_buf(WHAT, names::OUTPUT_HC_BASE)?,
                eps_norm: sh.eps,
                eps_hc: sh.hc_eps,
                n,
                m: 1,
                y: head.input_mut(),
                fault: gpu.unlabelled_sink(),
            },
        )?;
        if let Some(tp) = p.taps.as_mut() {
            copy_row(gpu.stream(), head.input(), &mut tp.head, n)?;
        }
        head.enqueue(gpu, w)?;
    }
    Ok(())
}

/// The host side of a walk: the embedding rows into the first stream buffer, every row's
/// position, live count and rope rows, the chunks' first positions, and each engram site's rows and
/// mask. A position `media` holds a row for takes that row in place of its id's embedding (widened
/// from bf16, [`widen_bf16`]) and enters the engram hash as no token, as the reference's embedding
/// input does.
fn stage(
    host: &mut HostIn,
    p: &mut Prompt,
    stream: &CudaStream,
    sh: &Shape,
    (toks, pos0): (&[u32], u32),
    media: Option<&CallRows<'_>>,
) -> Result<(), GpuError> {
    let n = sh.hidden;
    let t = toks.len();
    let q = p.cfg.q_rows;
    if let Some(m) = media
        && m.n_embd() != n
    {
        return Err(shape(format!(
            "media rows of {} values for a model of width {n}",
            m.n_embd()
        )));
    }
    let row_of = |i: usize| media.and_then(|m| m.row(pos0 + i as u32));
    // 1. The embedding rows, dequantized from the file (or a media span's row) and repeated into
    //    the four streams.
    let (ty, row_bytes) = host.embd;
    let embd = host.file.data(&names::token_embd())?;
    for (c, ids) in toks.chunks(q).enumerate() {
        let rows = &mut p.host.streams[..ids.len() * HC * n];
        for (j, (row, &token)) in rows.chunks_mut(HC * n).zip(ids).enumerate() {
            if let Some(m) = row_of(c * q + j) {
                widen_bf16(m, &mut row[..n]);
            } else {
                let at = token as usize * row_bytes;
                let src = embd
                    .get(at..at + row_bytes)
                    .ok_or_else(|| shape(format!("token {token} past the embedding's rows")))?;
                gguf::dequant_row(ty, src, &mut row[..n]).map_err(|e| GpuError::plan(WHAT, e))?;
            }
            let (first, rest) = row.split_at_mut(n);
            for c in rest.chunks_mut(n) {
                c.copy_from_slice(first);
            }
        }
        let mut dst = win_mut(&mut p.buf.hs[0], c * q, ids.len(), HC * n)?;
        dst.copy_from_host(stream, rows)?;
    }
    // 2. The positions, the live key counts and the chunks' first positions.
    let h = &mut p.host;
    h.pos.clear();
    h.pos.extend(pos0..pos0 + t as u32);
    h.keys.clear();
    h.keys.extend(pos0 + 1..=pos0 + t as u32);
    h.bases.clear();
    h.bases.extend((0..t).step_by(q).map(|r| pos0 + r as u32));
    for (dst, v) in [
        (&mut p.buf.pos, &h.pos),
        (&mut p.buf.keys, &h.keys),
        (&mut p.buf.bases, &h.bases),
    ] {
        let mut w = WindowMut::<u32>::of_mut(dst, 0, v.len())?;
        w.copy_from_host(stream, v)?;
    }
    // 3. The cos/sin rows of both rope bases.
    for ((table, dst), cs) in host.rope.iter().zip(&mut p.buf.cs).zip(&mut p.host.cs) {
        cs.clear();
        for pos in pos0..pos0 + t as u32 {
            table.push(pos, Direction::Forward, cs);
        }
        let mut w = win_mut(dst, 0, t, ROPE)?;
        w.copy_from_host(stream, cs)?;
    }
    // 4. The engram rows: every token's hash over the history, each site's rows dequantized from
    //    the table, and the token's mask on every column.
    if let Some((hash, hist)) = host.hash.as_mut() {
        let sites = p.buf.e_rows.len();
        let cols = p.cols;
        let h = &mut p.host;
        h.tokens.clear();
        h.tokens.extend(
            toks.iter()
                .enumerate()
                .map(|(i, &x)| row_of(i).is_none().then_some(x)),
        );
        h.ids.resize(sites * t * cols, 0);
        h.live.resize(t, false);
        hash.mapped_rows_into(hist, u64::from(pos0), &h.tokens, &mut h.ids, &mut h.live)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let table = host.file.data(names::ENGRAM_TABLE)?;
        h.vals.resize(t * cols * ENGRAM_W, 0.0);
        h.valid.resize(t * cols, 0.0);
        for (e, (rows, mask)) in p.buf.e_rows.iter_mut().zip(&mut p.buf.e_valid).enumerate() {
            for i in 0..t {
                let live = if h.live[i] { 1.0 } else { 0.0 };
                for c in 0..cols {
                    h.valid[i * cols + c] = live;
                    let r = h.ids[(e * t + i) * cols + c] as usize;
                    let src = table
                        .get(r * ENGRAM_W..(r + 1) * ENGRAM_W)
                        .ok_or_else(|| shape(format!("engram row {r} past the table")))?;
                    let at = (i * cols + c) * ENGRAM_W;
                    gguf::dequant_row(GgmlType::F8E4M3, src, &mut h.vals[at..at + ENGRAM_W])
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                }
            }
            let mut w = win_mut(rows, 0, t, cols * ENGRAM_W)?;
            w.copy_from_host(stream, &h.vals)?;
            let mut w = win_mut(mask, 0, t, cols)?;
            w.copy_from_host(stream, &h.valid)?;
        }
    }
    Ok(())
}

/// The streams of `hs` as (input, output) for the buffer `cur` holds the input in.
fn io(hs: &mut [DeviceBuffer<f32>; 2], cur: usize) -> (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) {
    let [a, b] = hs;
    if cur == 0 { (&*a, b) } else { (&*b, a) }
}

/// Layer `l`'s engram site over the chunk `sp` (llama.cpp `build_engram`): the rows normed per row,
/// the key and value projection, the per-stream gate and the masked delta into the other buffer.
fn engram(env: &Env<'_>, p: &mut Prompt, l: usize, sp: Span) -> Result<(), GpuError> {
    let Env {
        gpu,
        w,
        sh,
        layers,
        k,
        ..
    } = *env;
    let Some((e, kv_site)) = layers[l].engram.as_ref() else {
        return Ok(());
    };
    let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
    let (n, hid, cols) = (sp.n, sh.hidden, p.cols);
    let Prompt {
        gemm,
        dense,
        wide,
        buf,
        taps,
        ..
    } = p;
    gemm.enqueue_route_dense(stream, n, dense, fault)?;
    let wp = WideP {
        gpu,
        w,
        g32: &k.g32,
        gemm,
        dense,
    };
    let Buf {
        hs,
        e_rows,
        e_valid,
        e_norm,
        e_kv,
        e_kn,
        e_gate,
        e_ones,
        ..
    } = buf;
    let rows = win(&e_rows[*e], sp.r0, n, cols * ENGRAM_W)?;
    let valid = win(&e_valid[*e], sp.r0, n, cols)?;
    k.engram.enqueue_rows_norm(
        stream,
        RowsArgs {
            rows: &rows,
            lookup: w.f32_buf(WHAT, &names::engram_lookup_norm(l))?,
            valid: &valid,
            eps: ENGRAM_EPS,
            width: ENGRAM_W,
            cols,
            m: n,
            out: e_norm,
            fault,
        },
    )?;
    project(
        &wp,
        wide,
        l,
        n,
        e_norm,
        &mut [(kv_site, (HC + 1) * hid, e_kv)],
    )?;
    k.engram
        .enqueue_kv_bias(stream, w.f32_buf(WHAT, &names::engram_kv_b(l))?, n, e_kv)?;
    k.engram.gate.enqueue_key_norm(
        stream,
        KeyNormArgs {
            kv: e_kv,
            gain: e_ones,
            eps: ENGRAM_EPS,
            hc: HC,
            m: n,
            kn: e_kn,
            fault,
        },
    )?;
    let (x, out) = io(hs, sp.cur);
    let xw = win(x, sp.r0, n, HC * hid)?;
    k.engram.gate.enqueue_gate(
        stream,
        GateOnlyArgs {
            x: &xw,
            kn: e_kn,
            gain: w.f32_buf(WHAT, &names::engram_gate(l))?,
            eps: ENGRAM_EPS,
            hc: HC,
            m: n,
            gate: e_gate,
            fault,
        },
    )?;
    let mut ow = win_mut(out, sp.r0, n, HC * hid)?;
    k.engram.enqueue_delta(
        stream,
        DeltaArgs {
            x: &xw,
            kv: e_kv,
            gate: e_gate,
            gamma: w.f32_buf(WHAT, &names::engram_pre_gamma(l))?,
            hc_scale: w.f32_buf(WHAT, &names::engram_hc_scale(l))?,
            valid: &valid,
            eps: ENGRAM_EPS,
            cols,
            m: n,
            out: &mut ow,
            fault,
        },
    )?;
    tap(gpu, taps, (l, ENGRAM_OUT), (&ow, 0), (sp.r0, n, HC * hid))
}

/// The identity mix's pre of sub-layer `sub` of layer `l` over the rows `sp` of the streams: the
/// gates into `mix`, the fold into `y`, and the norm `gain` of the fold into `xn`.
fn pre_norm(
    env: &Env<'_>,
    (l, sub): (usize, Sub),
    sp: Span,
    (hs, mix, y, xn): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    gain: &str,
) -> Result<(), GpuError> {
    let Env { gpu, w, sh, k, .. } = *env;
    let x = win(hs, sp.r0, sp.n, HC * sh.hidden)?;
    k.hc.enqueue_pre(
        gpu.stream(),
        PreArgs {
            x: &x,
            w: w.f32_buf(WHAT, &names::hc_fn(l, sub))?,
            scale: w.f32_buf(WHAT, &names::hc_scale(l, sub))?,
            base: w.f32_buf(WHAT, &names::hc_base(l, sub))?,
            eps_norm: sh.eps,
            eps_hc: sh.hc_eps,
            magnitude: sh.hc_mag,
            n: sh.hidden,
            m: sp.n,
            mix,
            y,
            fault: gpu.layer_sink(l)?,
        },
    )?;
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        y,
        w.f32_buf(WHAT, gain)?,
        sh.eps,
        sh.hidden,
        sp.n,
        xn,
    )
}

/// The identity mix's post over the rows `sp`: the streams plus the sub-layer's output `x` times
/// the mix's post gates, into the other buffer.
fn post(
    env: &Env<'_>,
    hs: &mut [DeviceBuffer<f32>; 2],
    sp: Span,
    (x, mix): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let Env { gpu, sh, k, .. } = *env;
    let (res, out) = io(hs, sp.cur);
    let res = win(res, sp.r0, sp.n, HC * sh.hidden)?;
    let mut out = win_mut(out, sp.r0, sp.n, HC * sh.hidden)?;
    k.hc.enqueue_post(
        gpu.stream(),
        PostArgs {
            x,
            res: &res,
            mix,
            n: sh.hidden,
            m: sp.n,
            out: &mut out,
        },
    )
}

/// Layer `l`'s attention sub-layer over the chunk `sp` (llama.cpp `build_attention`, then the
/// sandwich norm and the identity mix around it).
fn attention(
    env: &Env<'_>,
    stores: &mut [Store],
    p: &mut Prompt,
    l: usize,
    sp: Span,
) -> Result<(), GpuError> {
    let Env {
        gpu,
        w,
        sh,
        layers,
        k,
        rope_rows,
    } = *env;
    let (stream, elem) = (gpu.stream(), gpu.elem());
    let fault = gpu.layer_sink(l)?;
    let a = &layers[l];
    let (heads, n, hid) = (a.heads, sp.n, sh.hidden);
    let Prompt {
        gemm,
        cols_of,
        dense,
        wide,
        gather,
        buf,
        taps,
        ..
    } = p;
    gemm.enqueue_route_dense(stream, n, dense, fault)?;
    let wp = WideP {
        gpu,
        w,
        g32: &k.g32,
        gemm,
        dense,
    };
    let Buf {
        hs,
        pos,
        keys,
        bases,
        iota,
        stage,
        cs,
        mix,
        y,
        xn,
        qa,
        qr,
        q,
        q_rope,
        q_cur,
        kv_a,
        kv_s,
        kvr,
        iq,
        ik,
        wi,
        iq_cols,
        wi_cols,
        part_v,
        part_ms,
        att,
        av,
        g,
        gated,
        o,
        out,
        list,
        n_sel,
        scores,
        ..
    } = buf;
    // 1. The mix's fold and the norm the sub-layer reads, then every projection of the normed
    //    rows: the query's down, the key row, the output gate and a sparse layer's index key and
    //    head weights.
    pre_norm(
        env,
        (l, Sub::Attn),
        sp,
        (&hs[sp.cur], mix, y, xn),
        &names::attn_norm(l),
    )?;
    tap(gpu, taps, (l, ATTN_NORM), (xn, 0), (sp.r0, n, hid))?;
    let mut sites: Vec<(&Site, usize, &mut DeviceBuffer<f32>)> = vec![
        (&a.q_a, sh.q_lora, qa),
        (&a.kv_a, ROW, kv_a),
        (&a.gate, heads * sh.head_v, g),
    ];
    if let Some(ix) = a.idx.as_ref() {
        sites.push((&ix.k, INDEX_HEAD, ik));
        sites.push((&ix.proj, sh.idx_heads, wi));
    }
    project(&wp, wide, l, n, xn, &mut sites)?;
    drop(sites);
    // 2. The query: the down's norm (the q residual), the up, and every head's rotated tail.
    elem.enqueue_rms_norm(
        stream,
        qa,
        w.f32_buf(WHAT, &names::attn_q_a_norm(l))?,
        sh.eps,
        sh.q_lora,
        n,
        qr,
    )?;
    let mut sites: Vec<(&Site, usize, &mut DeviceBuffer<f32>)> =
        vec![(&a.q_b, heads * sh.head_k, q)];
    if let Some(ix) = a.idx.as_ref() {
        sites.push((&ix.q_b, sh.idx_heads * INDEX_HEAD, iq));
    }
    project(&wp, wide, l, n, qr, &mut sites)?;
    drop(sites);
    let kind = usize::from(!a.dsa);
    let cs_rows = win(&cs[kind], sp.r0, n, ROPE)?;
    let runs =
        u32::try_from(heads * sh.head_k / ROPE).map_err(|_| shape("a query past u32 rope runs"))?;
    elem.enqueue_rope(stream, q, &cs_rows, ROPE, runs, n, q_rope)?;
    // 3. The key: the latent and the rotated part, normed, turned and appended as one f16 row
    //    `[k_rope | latent]` per row, at its position on a sparse layer and into the stage on a
    //    window layer.
    let kv_gain = w.f32_buf(WHAT, &names::attn_kv_a_norm(l))?;
    let pos_rows = winu(pos, sp.r0, n)?;
    let key_rows = winu(keys, sp.r0, n)?;
    let (at, cache): (&DeviceBuffer<u32>, &mut DeviceTensor<u16>) = match &mut stores[l] {
        Store::Dsa(d) => (&pos_rows, &mut *d.cache),
        Store::Swa { .. } => (iota, stage),
    };
    k.fused.enqueue_kv_norm_rope_append_rows(
        stream, kv_a, kv_gain, &cs_rows, at, sh.eps, LATENT, ROPE, n, kv_s, kvr, cache, fault,
    )?;
    // 4. A sparse layer's k-pool selector over the rows: the index rows appended, the pools they
    //    complete, each row's scores of its closed pools and its list of the kept pools' rows and
    //    its own pool's.
    if let (Some(ix), Store::Dsa(d)) = (a.idx.as_ref(), &mut stores[l]) {
        let (vq, vd) = w.q8_planes(WHAT, &names::indexer_compressor_value(l))?;
        let (gq, gd) = w.q8_planes(WHAT, &names::indexer_compressor_gate(l))?;
        let _ = ix;
        k.latent.enqueue_index_key_rms_append(
            stream,
            IndexKeyRmsArgs {
                rows: Rows {
                    x: ik,
                    stride: INDEX_HEAD,
                    m: n,
                },
                k_off: 0,
                gain: w.f32_buf(WHAT, &names::indexer_k_norm(l))?,
                value: IndexProj { qs: vq, d: vd },
                gate: IndexProj { qs: gq, d: gd },
                pos: &pos_rows,
                eps: sh.eps,
                ctx: sh.ctx,
                fault,
                cache: &mut d.index,
            },
        )?;
        k.latent.enqueue_index_pool_norm_rope(
            stream,
            IndexPoolRopeArgs {
                cache: &d.index,
                ape: w.f32_buf(WHAT, &names::indexer_compressor_ape(l))?,
                gain: w.f32_buf(WHAT, &names::indexer_compressor_norm(l))?,
                table: &rope_rows.table,
                n_keys: &key_rows,
                eps: sh.eps,
                ctx: sh.ctx,
                m: n,
                fault,
                pooled: &mut d.pooled,
            },
        )?;
        // The score pass reads a token a column; the projections wrote a token a row.
        cols_of.enqueue(stream, iq, n, sh.idx_heads * INDEX_HEAD, iq_cols)?;
        cols_of.enqueue(stream, wi, n, sh.idx_heads, wi_cols)?;
        k.kpool.enqueue_score_tail(
            stream,
            TailScoreArgs {
                q: iq_cols,
                w: wi_cols,
                n_keys: &key_rows,
                table: &rope_rows.table,
                pooled: &d.pooled,
                heads: sh.idx_heads,
                tokens: n,
                ctx: sh.ctx,
                kept: sh.kept,
                scale: weights_scale(sh.idx_heads, INDEX_HEAD),
                fault,
                scores,
            },
        )?;
        k.qsa.enqueue_topk_tail(
            stream,
            TopkTailArgs {
                n_keys: &key_rows,
                scores,
                ctx: sh.ctx,
                kept: sh.kept,
                width: sh.top_k,
                m: n,
                fault,
                list,
                n_sel,
            },
        )?;
        if let Some(t) = taps.as_mut() {
            let (top_k, pools) = (sh.top_k, sh.ctx.div_ceil(POOL));
            let dst = &mut t.sel[l];
            tap_words(gpu, dst, list, ((n - 1) * top_k, 0, top_k))?;
            tap_words(gpu, dst, n_sel, (n - 1, top_k, 1))?;
            tap_words(gpu, dst, scores, ((n - 1) * pools, top_k + 1, pools))?;
        }
    }
    // 5. The flash's query a head, `[q_rope | k_b · q_nope]`, MCOL rows a launch.
    let (qs, d) = w.q8_planes(WHAT, &names::attn_k_b(l))?;
    let q8 = gpu.q8f32();
    for r in chunks(n, MCOL) {
        let (xw, mut yw) = (
            win(q, r.start, r.len(), heads * sh.head_k)?,
            win_mut(q_cur, r.start, r.len(), heads * ROW)?,
        );
        q8.enqueue_q8_0_gemv_heads_mcol(
            stream,
            Q8_0GemvHeadsMcolArgs {
                qs,
                d,
                x: &xw,
                rows_per_head: LATENT,
                x_head_stride: sh.head_k,
                y_head_stride: ROW,
                y_off: ROPE,
                m: r.len(),
                x_col_stride: heads * sh.head_k,
                y_col_stride: heads * ROW,
                y: &mut yw,
            },
        )?;
    }
    let (_, src, dst) = gather
        .iter()
        .find(|g| g.0 == heads)
        .ok_or_else(|| shape(format!("no gather table of {heads} heads")))?;
    k.step
        .enqueue_gather(stream, q_rope, src, dst, n * heads * ROPE, q_cur)?;
    // 6. The latent attention with a sink a head: over each row's list on a sparse layer, over
    //    the ring and the stage on a window layer, the ring taking the stage after.
    let sinks = w.f32_buf(WHAT, &names::attn_sinks(l))?;
    let geom = FlashGeom {
        tokens: n,
        heads,
        rope_dims: ROPE,
        latent_dims: LATENT,
    };
    match &mut stores[l] {
        Store::Dsa(d) => k.flash.enqueue_flash_latent_sel_sink(
            stream,
            FlashSelArgs {
                q: q_cur,
                kv: &d.cache,
                list,
                counts: n_sel,
                stride: sh.top_k,
                kq_scale: sh.kq_scale(),
                geom,
                part_v,
                part_ms,
                fault,
            },
            sinks,
            att,
        )?,
        Store::Swa { ring } => {
            let base = winu(bases, sp.index, 1)?;
            k.flash.enqueue_flash_window(
                stream,
                FlashWindowArgs {
                    q: q_cur,
                    ring,
                    staged: Some(stage),
                    pos: &base,
                    sinks,
                    kq_scale: sh.kq_scale(),
                    geom,
                    part_v,
                    part_ms,
                    y: att,
                },
            )?;
            k.flash.enqueue_kv_ring_commit(
                stream,
                RingCommitArgs {
                    stage,
                    base: &base,
                    tokens: n,
                    ring,
                },
            )?;
        }
    }
    // 7. Each head's values, the output gate, the output projection and its sandwich norm.
    let (qs, d) = w.q8_planes(WHAT, &names::attn_v_b(l))?;
    for r in chunks(n, MCOL) {
        let (xw, mut yw) = (
            win(att, r.start, r.len(), heads * LATENT)?,
            win_mut(av, r.start, r.len(), heads * sh.head_v)?,
        );
        q8.enqueue_q8_0_gemv_heads_mcol(
            stream,
            Q8_0GemvHeadsMcolArgs {
                qs,
                d,
                x: &xw,
                rows_per_head: sh.head_v,
                x_head_stride: LATENT,
                y_head_stride: sh.head_v,
                y_off: 0,
                m: r.len(),
                x_col_stride: heads * LATENT,
                y_col_stride: heads * sh.head_v,
                y: &mut yw,
            },
        )?;
    }
    k.out_gate.enqueue_out_gate_head(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: av,
            gate: g,
            n_head: heads,
            m: n,
            fault,
            y: gated,
        },
        sh.head_v,
    )?;
    project(&wp, wide, l, n, gated, &mut [(&a.wo, hid, o)])?;
    elem.enqueue_rms_norm(
        stream,
        o,
        w.f32_buf(WHAT, &names::post_attention_norm(l))?,
        sh.eps,
        hid,
        n,
        out,
    )?;
    tap(gpu, taps, (l, ATTN_OUT), (out, 0), (sp.r0, n, hid))?;
    post(env, hs, sp, (out, mix))?;
    let (_, after) = io(hs, sp.cur);
    tap(
        gpu,
        taps,
        (l, HC_ATTN_OUT),
        (after, sp.r0),
        (sp.r0, n, HC * hid),
    )
}

/// The buffers a routed block's experts read and write: the latent rows and the routed ids, and the
/// gate's, the up's and the down's rows a slot each.
struct ExpertIo<'a> {
    lat: &'a DeviceBuffer<f32>,
    ids: &'a DeviceBuffer<u32>,
    gate: &'a mut DeviceBuffer<f32>,
    up: &'a mut DeviceBuffer<f32>,
    down: &'a mut DeviceBuffer<f32>,
}

/// Layer `l`'s routed experts over `n` rows of the latent: the rows quantized into the forms the
/// stacks read, the slot table over the routed ids (through `map`, each expert's place in the
/// card's stacks or `HOST`, when the card holds a share of the layer: the host's slots get no
/// tile and their down rows stay as they were; the raw ids otherwise), the gate and up GEMMs, the
/// gate-first clamp into the down's form and the down GEMM, the slots' rows into `io.down`. The
/// slot sum and the rest of the block are the caller's.
fn expert_rows(
    env: &Env<'_>,
    (l, n): (usize, usize),
    (gemm, acts, table): (&GemmKernels, &mut ExpertActs, &mut GemmRoute),
    (map, stacks, clamp): (Option<&DeviceBuffer<u32>>, &[String; 3], f32),
    io: ExpertIo<'_>,
) -> Result<(), GpuError> {
    let Env { gpu, w, sh, k, .. } = *env;
    let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
    let st = LegStacks::of(w, stacks.each_ref().map(String::as_str))?;
    let blk = st.block(sh.expert_ff, sh.latent)?;
    if let Some(x) = acts.x128.as_mut() {
        gpu.enqueue_quantize_gemm(io.lat, n, x, fault)?;
    }
    if let Some(x) = acts.x32.as_mut() {
        k.g32.enqueue_quantize_gemm32(stream, io.lat, n, x, fault)?;
    }
    let n_slots = n * N_USED;
    match map {
        None => gemm.enqueue_route(stream, io.ids, n_slots, table, fault)?,
        Some(map) => k
            .g32
            .enqueue_route_remap(stream, io.ids, map, n_slots, table, fault)?,
    }
    blk.enqueue_rule(
        gpu,
        (Some(gemm), &k.g32),
        (table, N_USED),
        n_slots,
        (
            RouteIo {
                x128: acts.x128.as_ref(),
                x32: acts.x32.as_ref(),
                g: io.gate,
                u: io.up,
                h128: acts.h128.as_mut(),
                h32: acts.h32.as_mut(),
                y: io.down,
            },
            Act::SwigluGateClamp { limit: clamp },
        ),
        fault,
    )
}

/// Layer `l`'s feed-forward sub-layer over the unit `sp`: the dense SiLU block, or the routed
/// block through the latent pair plus the shared SiLU expert; then the sandwich norm and the
/// identity mix around it.
fn ffn(env: &Env<'_>, p: &mut Prompt, l: usize, sp: Span) -> Result<(), GpuError> {
    let Env {
        gpu,
        w,
        sh,
        layers,
        k,
        ..
    } = *env;
    let (stream, elem) = (gpu.stream(), gpu.elem());
    let fault = gpu.layer_sink(l)?;
    let (n, hid) = (sp.n, sh.hidden);
    let Prompt {
        gemm,
        slots,
        dense,
        experts,
        wide,
        acts,
        buf,
        taps,
        ..
    } = p;
    gemm.enqueue_route_dense(stream, n, dense, fault)?;
    let wp = WideP {
        gpu,
        w,
        g32: &k.g32,
        gemm,
        dense,
    };
    let Buf {
        hs,
        mix,
        y,
        xn,
        out,
        fg,
        fu,
        fa,
        lat,
        routed,
        lat_up,
        shared,
        f,
        probs,
        ids,
        weights,
        gate_rows,
        up_rows,
        down_rows,
        ..
    } = buf;
    pre_norm(
        env,
        (l, Sub::Ffn),
        sp,
        (&hs[sp.cur], mix, y, xn),
        &names::ffn_norm(l),
    )?;
    tap(gpu, taps, (l, FFN_NORM), (xn, 0), (sp.r0, n, hid))?;
    match &layers[l].ffn {
        Ffn::Dense(glu) => {
            project(
                &wp,
                wide,
                l,
                n,
                xn,
                &mut [(&glu.gate, glu.ff, fg), (&glu.up, glu.ff, fu)],
            )?;
            elem.enqueue_swiglu(stream, fg, fu, n * glu.ff, fa)?;
            project(&wp, wide, l, n, fa, &mut [(&glu.down, hid, f)])?;
        }
        Ffn::Routed {
            down_lat,
            up_lat,
            stacks,
            shared: glu,
            clamp,
        } => {
            // The router over the unit's rows, the latent down and the shared expert's gate and
            // up on the same quantization of them.
            let r = RouteIn {
                w: w.f32_tensor(WHAT, &names::ffn_gate_inp(l))?,
                bias: Some(w.f32_buf(WHAT, &names::exp_probs_b(l))?),
                scale: sh.weights_scale,
                fault,
            };
            k.router.route_rows(
                (stream, gpu, &k.g32),
                xn,
                &r,
                n,
                RoutedRows {
                    probs,
                    logits: None,
                    ids,
                    weights,
                },
            )?;
            if let Some(t) = taps.as_mut() {
                tap_words(gpu, &mut t.picks[l], ids, (0, sp.r0 * N_USED, n * N_USED))?;
            }
            project(
                &wp,
                wide,
                l,
                n,
                xn,
                &mut [
                    (down_lat, sh.latent, lat),
                    (&glu.gate, glu.ff, fg),
                    (&glu.up, glu.ff, fu),
                ],
            )?;
            // The experts: one grouped GEMM over the unit's slots.
            expert_rows(
                env,
                (l, n),
                (&*gemm, &mut *acts, &mut *experts),
                (None, stacks, *clamp),
                ExpertIo {
                    lat: &*lat,
                    ids: &*ids,
                    gate: gate_rows,
                    up: up_rows,
                    down: down_rows,
                },
            )?;
            slots.enqueue_card_acc(
                stream,
                &CardAcc {
                    down: down_rows,
                    w: weights,
                    sel: ids,
                    n: sh.latent,
                    m: n,
                    n_card: N_EXPERT,
                    n_used: N_USED,
                    w_pitch: N_USED,
                    fault,
                },
                routed,
            )?;
            project(&wp, wide, l, n, routed, &mut [(up_lat, hid, lat_up)])?;
            elem.enqueue_swiglu(stream, fg, fu, n * glu.ff, fa)?;
            project(&wp, wide, l, n, fa, &mut [(&glu.down, hid, shared)])?;
            elem.enqueue_add(stream, lat_up, shared, n * hid, f)?;
        }
    }
    elem.enqueue_rms_norm(
        stream,
        f,
        w.f32_buf(WHAT, &names::post_ffw_norm(l))?,
        sh.eps,
        hid,
        n,
        out,
    )?;
    tap(gpu, taps, (l, FFN_OUT), (out, 0), (sp.r0, n, hid))?;
    post(env, hs, sp, (out, mix))?;
    let (_, after) = io(hs, sp.cur);
    tap(gpu, taps, (l, L_OUT), (after, sp.r0), (sp.r0, n, HC * hid))
}

/// The prompt `tokens` through `model` from its position as one batch: the stores, the engram
/// history and the position stand where `tokens.len()` decode steps leave them, and the return is
/// the greedy id after the last row (its logits are the model's, [`GpuModel::logits`]). A call of
/// more rows than [`Cfg::cap`] runs in walks of that many. A position the stores do not stand at,
/// a context the rows pass and a fault are refused by name; a call that fails part-way leaves the
/// stores to a reset.
pub fn prompt(model: &mut GpuModel<Body>, p: &mut Prompt, tokens: &[u32]) -> Result<u32, GpuError> {
    const CALL: &str = "hy4::batch::prompt";
    if tokens.is_empty() {
        return Err(GpuError::shape(CALL, "a prompt of no id"));
    }
    let n = u32::try_from(tokens.len()).map_err(|_| GpuError::shape(CALL, "ids past u32"))?;
    let token = model.run_rows(tokens.len(), CALL, |gpu, w, body, head, pos| {
        if pos != body.held {
            return Err(GpuError::shape(
                CALL,
                format!(
                    "a call at position {pos}; the stores stand at {} (reset after a call that \
                     failed part-way)",
                    body.held
                ),
            ));
        }
        // The cache rooms backed through the call's last position before any walk launches.
        body.fit(pos as usize, tokens.len())?;
        // Anything past this point that fails leaves the engram history and the stores part-way.
        body.held = u32::MAX;
        let cap = p.cfg.cap;
        let walks = tokens.len().div_ceil(cap);
        for (i, chunk) in tokens.chunks(cap).enumerate() {
            let at = pos + (i * cap) as u32;
            walk(gpu, w, body, head, p, (chunk, at), i + 1 == walks)?;
        }
        body.held = pos + n;
        Ok(true)
    })?;
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))
}
