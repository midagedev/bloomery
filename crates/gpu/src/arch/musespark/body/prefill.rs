//! The prompt path: a prompt runs layer-major over passes of the rows the
//! load chose ([`choose_pass_rows`]: at most [`PASS_ROWS_MAX`] on a
//! whole-card load, [`placed_rows`] on a placed one), each launch over the
//! pass's rows, in the step's order ([`super`]'s module doc):
//!
//! - a projection is one GEMM over the pass's rows on the pass's one-expert
//!   table ([`project_wide`]: the Q6_K q and o on q8_1 blocks of 128, the
//!   Q8_0 k, v, gate and latent pair on q8 blocks of 32);
//! - the norms, the per-head q/k norms, the gate's σ and product and the
//!   weightless norms are the step's launches at `m` rows; the α/β mixes are
//!   `mix_rows` over the pass's rows against each gain tiled to them
//!   ([`tile_rows`]: the mix itself is the step's arithmetic);
//! - a sliding layer ropes each row at its own position (the step's host
//!   table, one row a position, uploaded once a pass); the pass's K/V rows
//!   are appended to the planes before the attention, and the head-64 pass
//!   runs over runs of [`ATTN_ROWS`] rows at the layer's window, each row
//!   over the keys below its own live count — its segments and its sums are
//!   its step's, so on the step's q, K and V it is the step's bits;
//! - the router is its instance's two-launch rows form (the per-token
//!   launches' bits), the experts the step's `_sel` launches over the pass's
//!   `top_k · m` slots (each slot's own row: the step's bits on the step's
//!   latent), each slot's down row scaled and normed, the weighted sum.
//!
//! A placed load ([`super::placed`]) walks the same layers through the host
//! tier's batch port ([`PlacedPass`], [`sched::walk`]): a layer with no host
//! share runs whole; a hosted layer runs three parts around the host's union
//! over the pass's rows —
//!
//! - **front**: the attention half, the FFN half's norm, the router, the
//!   latent down and its weighted norm (the rows the host's experts read),
//!   and the download of those rows, the weights and the ids;
//! - **shadow**, under the host's union: the call's pick of the layer, then
//!   each slot's place in the card's stacks (`ds41_ffn_places`) over the
//!   slot map the pick left, and the card's slots over those places (a
//!   [`HOST`](crate::hybrid::HOST) slot's warps return first), none when the
//!   layer holds no expert on the card;
//! - **back**: the card slots' weighted sum (`ds41_ffn_card_acc_9`) plus the
//!   host's sums, the latent up, the scale-then-norm and the α/β mix.
//!
//! A layer an expert tier card holds experts of ([`super::tier`]) also
//! downloads each slot's tier place with the host's rows, and its back joins
//! the tier's rows through the port ([`BatchLeg::join_tiered`]): normed and
//! summed beside the card's slots before the host's sums add.
//!
//! The head reads the last row of the prompt's last pass only. After
//! [`GpuModel::prefill_with`] the K/V planes and the position are those of
//! the prompt's ids run as steps; the next step continues from them.
//! [`GpuModel::prefill_layer`] runs one layer alone over the passes on rows a
//! gate supplies, the steps' own input to it: the batch's arithmetic of that
//! layer without the layers before it between.
//!
//! The GEMMs sum each output in another order than the step's gemv and read
//! the Q8_0 sites' rows as q8 blocks of 32 where the step reads f32, so a
//! pass's rows agree with the steps' to the GEMMs' band, not bit for bit;
//! every other launch is the step's own at `m` rows. The pass's buffers live
//! in [`Prompt`], cut at load, so a pass allocates nothing.

use super::placed::{Placed, PlacedSide};
use super::stream::{RingPass, XSide};
use super::tier::{TierSum, enqueue_tier_sum};
use super::{Body, Kernels, Layer, RouteOut, Shape, Taps, enqueue_head, stack};
use crate::arch::hybrid::{Site, WideCx, norm_rows, project_wide, wide_acts_at};
use crate::card_leg::{LegStacks, with_ring};
use crate::card_slots::ColPlaces;
use crate::elem::EmbedRowsArgs;
use crate::flash::PlaneAppend;
use crate::flash_gqa::{Gqa64Args, HEAD_64, SEG_KEYS, SEGMENTS, window_segments};
use crate::gemm::{GEMM_BN, GEMM32_STEP, GemmKernels, GemmRoute};
use crate::head::Head;
use crate::host::BatchLeg;
use crate::host::run::HostRun;
use crate::host::served::prompt_pass;
use crate::host::xstream::LandBatch;
use crate::iq_sel::IqGateUp;
use crate::kquant::{Act as GateAct, SelDown};
use crate::model::GpuModel;
use crate::model::fit::{FIT_RESERVE, largest_fit};
use crate::rope_table::Direction;
use crate::site::{WideActs, WideKernels};
use crate::tensor::{Pitched, dtod_2d};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Q8Act, Window, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::arch::musespark::hparams::Hparams;
use model::arch::musespark::place::{self, PassForms, PassShape};
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::collections::BTreeMap;
use std::ops::Range;

const WHAT: &str = "musespark::prefill";

/// The most rows one pass of a whole-card load takes.
pub(super) const PASS_ROWS_MAX: usize = 512;

/// The rows one head-64 attention launch takes: its partials hold every
/// segment of every head of a row, so a run of rows bounds them.
const ATTN_ROWS: usize = 32;

/// The most rows one pass of a placed load takes: every routed slot of a
/// post-norm layer is a column of the host's union call
/// ([`HostRun::set_post_norm`](crate::host::run::HostRun::set_post_norm)),
/// so a pass of `rows` rows is a call of `rows · n_used` columns, at most the
/// union tail's (the plan's one formula of it,
/// [`pass_rows_max`](model::arch::musespark::place::pass_rows_max)).
#[must_use]
pub(super) fn placed_rows(n_used: usize) -> usize {
    model::arch::musespark::place::pass_rows_max(n_used)
}

/// The pass's word record: the first position, then one row word a key head
/// (head `h` at `P_ROWS + h`, the plane row `h · ctx + pos` the pass's first
/// K/V row lands at), then the pass's ids from `P_ROWS + n_kv`.
const P_POS: usize = 0;
const P_ROWS: usize = 1;

/// The pass's sizes: every width its buffers are cut from, collected once at
/// load and read by both the fit check ([`PassSizes::bytes`]) and the
/// allocation ([`Prompt::new`]), so the bytes the one counts and the other
/// holds cannot drift. The bytes are the plan's formula ([`place::pass_bytes`])
/// over the shape the load collected, which a placed load holds equal to the
/// shape the plan's reserve was counted over ([`PassSizes::refuse_unplanned`]).
pub(super) struct PassSizes {
    sh: Shape,
    shape: PassShape,
}

// The plan's formula counts the pass's buffers by constants of its own; each is held equal to
// the body's and the kernels' here.
const _: () = {
    assert!(place::PASS_ATTN_ROWS == ATTN_ROWS);
    assert!(place::PASS_POS_WORDS == P_ROWS);
    assert!(place::FLASH_SEGMENTS == SEGMENTS);
    assert!(place::SEGMENT_KEYS == SEG_KEYS);
    assert!(place::GEMM_COLS == GEMM_BN);
    assert!(place::GEMM_STEP == GEMM32_STEP);
};

impl PassSizes {
    /// The sizes for `sh` and `layers`' widths and windows: the segments the
    /// widest cut of a row takes, and the wide forms each site width is held
    /// in ([`WideActs::wants`] over the sites of that width, less the f32
    /// rows: the input's are the caller's). Host-side only.
    pub(super) fn of(sh: Shape, layers: &[Layer]) -> PassSizes {
        let segs = layers
            .iter()
            .map(|l| window_segments(l.window))
            .max()
            .unwrap_or_else(|| window_segments(0));
        let mut wide: BTreeMap<usize, PassForms> = BTreeMap::new();
        for s in layers.iter().flat_map(Layer::sites) {
            let forms = WideActs::wants(s.ty).without_f32();
            let slot = wide.entry(s.k).or_default();
            *slot = slot.union(PassForms {
                q128: forms.q128,
                q32: forms.q32,
            });
        }
        let shape = PassShape {
            hidden: sh.hidden,
            head: sh.head,
            n_head: sh.n_head,
            n_kv: sh.n_kv,
            latent: sh.latent,
            n_expert: sh.n_expert,
            top_k: sh.top_k,
            ff: sh.ff,
            segs,
            wide,
        };
        PassSizes { sh, shape }
    }

    /// The device bytes [`Prompt::new`] allocates for passes of `rows` rows:
    /// what the fit check counts, and what the plan reserves at the most rows
    /// a pass takes ([`place::prompt_reserve_bytes`]). Pure.
    pub(super) fn bytes(&self, rows: usize) -> usize {
        usize::try_from(place::pass_bytes(&self.shape, rows))
            .expect("the pass buffers' bytes fit a usize")
    }

    /// Refused by name when the plan's reserve for these buffers
    /// ([`place::prompt_reserve_bytes`]) was counted over another shape than
    /// the one this load cuts them from: the plan then keeps other bytes out
    /// of the expert budget than the load allocates.
    pub(super) fn refuse_unplanned(&self, hp: &Hparams, plan: &Plan<'_>) -> Result<(), GpuError> {
        let planned = place::pass_shape(hp, plan.model).map_err(|e| GpuError::plan(WHAT, e))?;
        if planned != self.shape {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the pass buffers are cut from {:?}, the plan's reserve counts {planned:?}",
                    self.shape
                ),
            ));
        }
        Ok(())
    }
}

/// The rows one pass takes on `free` card bytes, at most `cap`: the largest
/// multiple of 64 in `64..=min(cap, ctx)` — the whole `ctx` where that is
/// below 64 — whose buffers plus [`FIT_RESERVE`] fit ([`largest_fit`]), or a
/// named refusal when not even the smallest does. Load-time only; allocates
/// nothing.
pub(super) fn choose_pass_rows(
    sizes: &PassSizes,
    ctx: usize,
    free: usize,
    cap: usize,
) -> Result<usize, GpuError> {
    const WHAT_FIT: &str = "musespark::pass_rows";
    let hi = cap.min(ctx);
    if hi == 0 {
        return Err(GpuError::shape(
            WHAT_FIT,
            "a pass holds at least one row; the context or the cap left none",
        ));
    }
    let candidates: Vec<usize> = if hi < 64 {
        vec![hi]
    } else {
        (1..=hi / 64).map(|k| 64 * k).rev().collect()
    };
    let need = |rows| sizes.bytes(rows);
    largest_fit(candidates.iter().copied(), need, free, FIT_RESERVE).ok_or_else(|| {
        let smallest = candidates.last().copied().unwrap_or(hi);
        GpuError::shape(
            WHAT_FIT,
            format!(
                "the prefill pass buffers for passes of {smallest} rows need {} bytes and \
                 {FIT_RESERVE} more stay free; the card has {free} free: no pass size fits",
                need(smallest)
            ),
        )
    })
}

/// The head-64 attention's run of [`ATTN_ROWS`] rows: the launch takes whole
/// buffers, so a run's query rows, live counts and output rows are copied
/// through these.
struct AttnRun {
    q: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    n_keys: DeviceBuffer<u32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
}

/// A gain vector tiled to the pass's rows, twice, and the spare the tiling
/// doubles through ([`tile_rows`]).
struct Tiles {
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    spare: DeviceBuffer<f32>,
}

/// The buffers one pass runs over, `rows` rows each.
pub(super) struct Prompt {
    rows: usize,
    /// The grouped GEMM the projections run on, and the pass's one-expert
    /// table, filled once a pass.
    gemm: GemmKernels,
    route: GemmRoute,
    /// The host's sums for the pass's rows, `latent` values a row: what the
    /// host tier's batch port uploads into on a placed load.
    hsum: DeviceBuffer<f32>,
    b: PassBufs,
}

/// [`Prompt`]'s buffers beside the GEMM and the host's sums.
struct PassBufs {
    /// The layer's input and output rows, and the rows between its halves.
    x: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    delta: DeviceBuffer<f32>,
    post: DeviceBuffer<f32>,
    tiles: Tiles,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    k2: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    /// The pass's K and V rows regrouped head-major: head `h`'s rows at
    /// `h · m · head`, what the plane append reads.
    kh: DeviceBuffer<f32>,
    vh: DeviceBuffer<f32>,
    /// The latent down's rows and their weighted norm (what the experts
    /// read), the experts' weighted sum.
    lat: DeviceBuffer<f32>,
    lat2: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    /// The router's sigmoid scores (`n_expert` a row), its `top_k` ids and
    /// weights a row, and each routed slot's place in the card's stacks.
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    /// The slots' SiLU-gated up rows, their down rows, those scaled and
    /// normed.
    up: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    down2: DeviceBuffer<f32>,
    /// The word record ([`P_POS`], [`P_ROWS`], the ids) and its host copy.
    words: DeviceBuffer<u32>,
    host_words: Vec<u32>,
    /// The embedding's row outputs: each row's position and live key count.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// Each row's rope cos/sin row, from the step's host table.
    cs: DeviceBuffer<f32>,
    host_cs: Vec<f32>,
    run: AttnRun,
    /// The q8_1 slot forms of the experts' `_sel` launches: the latent rows
    /// and the slots' up rows.
    lat_q: Q8Act,
    up_q: Q8Act,
    /// The wide forms of each site width ([`project_wide`]).
    wide: Vec<(usize, WideActs)>,
}

impl Prompt {
    /// The buffers for passes of up to `rows` rows over `sizes`' widths and
    /// `layers`' sites, which must hold exactly the bytes
    /// [`PassSizes::bytes`] counted for them or the load fails by name. The
    /// rope is the step's table, one row of the head's width a position
    /// (`Body::load` refuses any other). Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        sizes: &PassSizes,
        layers: &[Layer],
        rows: usize,
    ) -> Result<Prompt, GpuError> {
        let sh = sizes.sh;
        if rows == 0 {
            return Err(GpuError::shape(WHAT, "a pass of no rows"));
        }
        let stream = gpu.stream();
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let zu = |n: usize| -> Result<DeviceBuffer<u32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n)?)
        };
        let q_width = sh.n_head * sh.head;
        let kv_width = sh.n_kv * sh.head;
        let mut wide = Vec::with_capacity(sizes.shape.wide.len());
        for &k in sizes.shape.wide.keys() {
            wide.push((
                k,
                wide_acts_at(stream, layers.iter().flat_map(Layer::sites), (rows, k))?,
            ));
        }
        let n_words = P_ROWS + sh.n_kv + rows;
        let part = ATTN_ROWS * sh.n_head * sizes.shape.segs;
        let prompt = Prompt {
            rows,
            gemm: GemmKernels::load(gpu.context())?,
            route: GemmRoute::new(stream, rows, 1)?,
            hsum: z(rows * sh.latent)?,
            b: PassBufs {
                x: z(rows * sh.hidden)?,
                mid: z(rows * sh.hidden)?,
                xn: z(rows * sh.hidden)?,
                delta: z(rows * sh.hidden)?,
                post: z(rows * sh.hidden)?,
                tiles: Tiles {
                    a: z(rows * sh.hidden)?,
                    b: z(rows * sh.hidden)?,
                    spare: z(rows * sh.hidden)?,
                },
                q: z(rows * q_width)?,
                qn: z(rows * q_width)?,
                gate: z(rows * q_width)?,
                attn: z(rows * q_width)?,
                k: z(rows * kv_width)?,
                k2: z(rows * kv_width)?,
                v: z(rows * kv_width)?,
                kh: z(rows * kv_width)?,
                vh: z(rows * kv_width)?,
                lat: z(rows * sh.latent)?,
                lat2: z(rows * sh.latent)?,
                moe: z(rows * sh.latent)?,
                probs: z(rows * sh.n_expert)?,
                ids: zu(rows * sh.top_k)?,
                weights: z(rows * sh.top_k)?,
                sel: zu(rows * sh.top_k)?,
                up: z(rows * sh.top_k * sh.ff)?,
                down: z(rows * sh.top_k * sh.latent)?,
                down2: z(rows * sh.top_k * sh.latent)?,
                words: zu(n_words)?,
                host_words: vec![0; n_words],
                pos: zu(rows)?,
                n_keys: zu(rows)?,
                cs: z(rows * sh.head)?,
                host_cs: Vec::with_capacity(rows * sh.head),
                run: AttnRun {
                    q: z(ATTN_ROWS * q_width)?,
                    y: z(ATTN_ROWS * q_width)?,
                    n_keys: zu(ATTN_ROWS)?,
                    part_v: z(part * HEAD_64)?,
                    part_ms: z(part * 2)?,
                },
                lat_q: Q8Act::with_slots(stream, rows, sh.latent)?,
                up_q: Q8Act::with_slots(stream, sh.top_k * rows, sh.ff)?,
                wide,
            },
        };
        let need = sizes.bytes(rows);
        if prompt.bytes() != need {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the pass buffers for passes of {rows} rows hold {} bytes, their fit check \
                     counted {need}",
                    prompt.bytes()
                ),
            ));
        }
        Ok(prompt)
    }

    /// The most ids a pass takes.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// The device bytes the pass's buffers hold (the GEMM module's own
    /// excepted: its code, not the pass's buffers).
    pub(super) fn bytes(&self) -> usize {
        let b = &self.b;
        let f = [
            &self.hsum,
            &b.x,
            &b.mid,
            &b.xn,
            &b.delta,
            &b.post,
            &b.tiles.a,
            &b.tiles.b,
            &b.tiles.spare,
            &b.q,
            &b.qn,
            &b.gate,
            &b.attn,
            &b.k,
            &b.k2,
            &b.v,
            &b.kh,
            &b.vh,
            &b.lat,
            &b.lat2,
            &b.moe,
            &b.probs,
            &b.weights,
            &b.up,
            &b.down,
            &b.down2,
            &b.cs,
            &b.run.q,
            &b.run.y,
            &b.run.part_v,
            &b.run.part_ms,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        let u = [&b.ids, &b.sel, &b.words, &b.pos, &b.n_keys, &b.run.n_keys]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>();
        f + u
            + b.lat_q.device_bytes()
            + b.up_q.device_bytes()
            + b.wide.iter().map(|w| w.1.bytes()).sum::<usize>()
            + self.route.bytes()
    }
}

/// `n` f32 of `src` from value `s0` copied into `dst` at value `d0`.
fn copy_f32(
    stream: &CudaStream,
    (src, s0): (&DeviceBuffer<f32>, usize),
    (dst, d0): (&mut DeviceBuffer<f32>, usize),
    n: usize,
) -> Result<(), GpuError> {
    let (f, width) = (size_of::<f32>(), n * size_of::<f32>());
    dtod_2d(
        stream,
        (src, s0 * f),
        (dst, d0 * f),
        Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )
}

/// `n` u32 of `src` from value `s0` copied into `dst`'s start.
fn copy_u32(
    stream: &CudaStream,
    (src, s0): (&DeviceBuffer<u32>, usize),
    dst: &mut DeviceBuffer<u32>,
    n: usize,
) -> Result<(), GpuError> {
    let (u, width) = (size_of::<u32>(), n * size_of::<u32>());
    dtod_2d(
        stream,
        (src, s0 * u),
        (dst, 0),
        Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )
}

/// `g`'s `n` values tiled `m` times into `dst`, doubling the tiled rows a
/// round through `spare` (two copies a round, `⌈log₂ m⌉` rounds): the rows a
/// flat `mix_rows` reads as each row's gain.
fn tile_rows(
    stream: &CudaStream,
    g: &DeviceBuffer<f32>,
    (n, m): (usize, usize),
    (dst, spare): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    copy_f32(stream, (g, 0), (&mut *dst, 0), n)?;
    let mut have = 1;
    while have < m {
        let c = have.min(m - have);
        copy_f32(stream, (&*dst, 0), (&mut *spare, 0), c * n)?;
        copy_f32(stream, (&*spare, 0), (&mut *dst, have * n), c * n)?;
        have += c;
    }
    Ok(())
}

/// `y = a ⊙ α + b ⊙ β` over `m` rows of `n`, `α` and `β` the layer's named
/// per-channel gains tiled to the rows: the step's `mix_rows` at `m` rows.
#[allow(
    clippy::too_many_arguments,
    reason = "the card, the weights, the two gains, the two branches, the widths, the tiles and the output (rust-quality R8)"
)]
fn mix_rows(
    (gpu, w): (&Gpu, &Weights),
    (alpha, beta): (&str, &str),
    (a, b): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    (n, m): (usize, usize),
    tiles: &mut Tiles,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let Tiles {
        a: ta,
        b: tb,
        spare,
    } = tiles;
    tile_rows(
        stream,
        w.f32_buf(WHAT, alpha)?,
        (n, m),
        (&mut *ta, &mut *spare),
    )?;
    tile_rows(
        stream,
        w.f32_buf(WHAT, beta)?,
        (n, m),
        (&mut *tb, &mut *spare),
    )?;
    gpu.elem().enqueue_mix_rows(stream, a, ta, b, tb, n * m, y)
}

/// What every launch of a pass reads.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    gemm: &'a GemmKernels,
    route: &'a GemmRoute,
    sh: Shape,
    /// The ones the weightless norms read, at the hidden and the head.
    ones_hidden: &'a DeviceBuffer<f32>,
    ones_head: &'a DeviceBuffer<f32>,
    m: usize,
}

impl Cx<'_> {
    /// The wide projections' context over the pass's rows.
    fn wide(&self) -> WideCx<'_> {
        WideCx {
            what: WHAT,
            gpu: self.gpu,
            w: self.w,
            wk: WideKernels {
                gemm: self.gemm,
                g32: Some(&self.k.g32),
            },
            m: self.m,
        }
    }

    /// `y = W · x` for each of `sites` (all of one width) over the pass's rows.
    fn project(
        &self,
        wide: &mut [(usize, WideActs)],
        layer: usize,
        x: &DeviceBuffer<f32>,
        sites: &mut [(&Site, usize, &mut DeviceBuffer<f32>)],
    ) -> Result<(), GpuError> {
        let Some(k) = sites.first().map(|s| s.0.k) else {
            return Ok(());
        };
        let acts = wide
            .iter_mut()
            .find(|a| a.0 == k)
            .map(|a| &mut a.1)
            .ok_or_else(|| GpuError::shape(WHAT, format!("no wide forms of width {k}")))?;
        project_wide(&self.wide(), (acts, self.route), layer, x, sites)
    }
}

/// Where a layer's card slots read their experts from.
#[derive(Clone, Copy)]
enum Places {
    /// Every expert of the layer is resident, in the router's id order: the
    /// ids are the places.
    Ids,
    /// The plan's share of the layer is resident: `sel` holds each slot's
    /// place in the card's stacks, or [`HOST`](crate::hybrid::HOST).
    Cut,
}

/// A pass's layer walk: what every layer of the pass reads and writes. Each
/// layer reads `b.x` and writes it back through `b.mid`.
struct PassWalk<'a> {
    cx: Cx<'a>,
    layers: &'a mut [Layer],
    b: &'a mut PassBufs,
    taps: Option<&'a mut Taps>,
    /// The layers the pass runs: every layer, or the one a gate probes
    /// ([`Forced`]).
    span: Range<usize>,
}

impl PassWalk<'_> {
    /// Whether the pass runs layer `l`.
    fn runs(&self, l: usize) -> bool {
        self.span.contains(&l)
    }

    /// Layer `l`'s attention half over the pass's rows, `b.x` into `b.mid`:
    /// the step's [`super::attention`] at `m` rows.
    fn attention(&mut self, l: usize) -> Result<(), GpuError> {
        let PassWalk { cx, layers, b, .. } = self;
        let (gpu, w, sh, m) = (cx.gpu, cx.w, cx.sh, cx.m);
        let stream = gpu.stream();
        let elem = gpu.elem();
        let fault = gpu.layer_sink(l)?;
        let a = &mut layers[l];
        let (q_width, kv_width) = (sh.n_head * sh.head, sh.n_kv * sh.head);
        let b = &mut **b;
        norm_rows(
            (gpu, w),
            (sh.eps, sh.hidden, m),
            &a.attn_norm,
            &b.x,
            &mut b.xn,
        )?;
        cx.project(
            &mut b.wide,
            l,
            &b.xn,
            &mut [
                (&a.q, q_width, &mut b.q),
                (&a.k, kv_width, &mut b.k),
                (&a.v, kv_width, &mut b.v),
                (&a.gate, q_width, &mut b.gate),
            ],
        )?;
        elem.enqueue_rms_norm(
            stream,
            &b.q,
            w.f32_buf(WHAT, &a.q_norm)?,
            sh.eps,
            sh.head,
            sh.n_head * m,
            &mut b.qn,
        )?;
        elem.enqueue_rms_norm(
            stream,
            &b.k,
            w.f32_buf(WHAT, &a.k_norm)?,
            sh.eps,
            sh.head,
            sh.n_kv * m,
            &mut b.k2,
        )?;
        // The sliding layers rope each row at its own position; the full
        // layers rope nothing.
        let (q_in, k_in) = if a.sliding {
            elem.enqueue_rope(stream, &b.qn, &b.cs, sh.head, sh.n_head as u32, m, &mut b.q)?;
            elem.enqueue_rope(stream, &b.k2, &b.cs, sh.head, sh.n_kv as u32, m, &mut b.k)?;
            (&b.q, &b.k)
        } else {
            (&b.qn, &b.k2)
        };
        let f = size_of::<f32>();
        let shape = Pitched {
            width: sh.head * f,
            rows: m,
            dev_pitch: kv_width * f,
            host_pitch: sh.head * f,
        };
        for h in 0..sh.n_kv {
            dtod_2d(
                stream,
                (k_in, h * sh.head * f),
                (&mut b.kh, h * m * sh.head * f),
                shape,
            )?;
            dtod_2d(
                stream,
                (&b.v, h * sh.head * f),
                (&mut b.vh, h * m * sh.head * f),
                shape,
            )?;
        }
        let at = PlaneAppend {
            words: &b.words,
            first_word: P_ROWS,
            heads: sh.n_kv,
            m,
            fault,
        };
        let (kh, vh, head) = (&b.kh, &b.vh, sh.head);
        gpu.flash().enqueue_kv_append_planes(
            stream,
            &at,
            move |h| Window::<f32>::of(kh, h * m * head * f, m * head),
            &mut a.kc,
        )?;
        gpu.flash().enqueue_kv_append_planes(
            stream,
            &at,
            move |h| Window::<f32>::of(vh, h * m * head * f, m * head),
            &mut a.vc,
        )?;
        // Runs of ATTN_ROWS rows, each row over its own live keys at the
        // layer's window: the step's launch at the run's rows.
        let run = &mut b.run;
        let mut r0 = 0;
        while r0 < m {
            let mm = (m - r0).min(ATTN_ROWS);
            copy_f32(stream, (q_in, r0 * q_width), (&mut run.q, 0), mm * q_width)?;
            copy_u32(stream, (&b.n_keys, r0), &mut run.n_keys, mm)?;
            cx.k.flash.enqueue_pass_64(
                stream,
                Gqa64Args {
                    q: &run.q,
                    kc: a.kc.buf(),
                    vc: a.vc.buf(),
                    n_keys: &run.n_keys,
                    scale: 1.0 / (sh.head as f32).sqrt(),
                    n_kv: sh.n_kv,
                    ctx: sh.ctx,
                    m: mm,
                    window: a.window,
                    part_v: &mut run.part_v,
                    part_ms: &mut run.part_ms,
                    fault,
                    y: &mut run.y,
                },
                false,
            )?;
            copy_f32(
                stream,
                (&run.y, 0),
                (&mut b.attn, r0 * q_width),
                mm * q_width,
            )?;
            r0 += mm;
        }
        // σ of the gate in place, the per-head weightless norm of the
        // attention output, their product, the output projection, the
        // weightless post-norm and the α/β mix.
        elem.enqueue_sigmoid_rows(stream, &mut b.gate, q_width * m)?;
        elem.enqueue_rms_norm(
            stream,
            &b.attn,
            cx.ones_head,
            sh.eps,
            sh.head,
            sh.n_head * m,
            &mut b.qn,
        )?;
        elem.enqueue_mul_rows(stream, &b.qn, &b.gate, q_width * m, &mut b.attn)?;
        cx.project(
            &mut b.wide,
            l,
            &b.attn,
            &mut [(&a.o, sh.hidden, &mut b.delta)],
        )?;
        elem.enqueue_rms_norm(
            stream,
            &b.delta,
            cx.ones_hidden,
            sh.post_eps,
            sh.hidden,
            m,
            &mut b.post,
        )?;
        mix_rows(
            (gpu, w),
            (&a.attn_alpha, &a.attn_beta),
            (&b.x, &b.post),
            (sh.hidden, m),
            &mut b.tiles,
            &mut b.mid,
        )
    }

    /// Layer `l`'s FFN front over the pass's rows of `b.mid`: the norm, the
    /// router's rows form at the layer's instance, the latent down and its
    /// weighted norm into `b.lat2` (the rows the experts read).
    fn ffn_pre(&mut self, l: usize) -> Result<(), GpuError> {
        let PassWalk { cx, layers, b, .. } = self;
        let (gpu, w, sh, m) = (cx.gpu, cx.w, cx.sh, cx.m);
        let stream = gpu.stream();
        let fault = gpu.layer_sink(l)?;
        let a = &layers[l];
        let b = &mut **b;
        norm_rows(
            (gpu, w),
            (sh.eps, sh.hidden, m),
            &a.ffn_norm,
            &b.mid,
            &mut b.xn,
        )?;
        let (rw, bias) = (w.f32_tensor(WHAT, &a.router)?, w.f32_buf(WHAT, &a.bias)?);
        let routers = &cx.k.routers;
        let (probs, ids, weights) = (&mut b.probs, &mut b.ids, &mut b.weights);
        match &a.route {
            RouteOut::E256(_) => routers.e256.enqueue_router_rows(
                stream,
                gpu,
                &cx.k.g32,
                rw,
                &b.xn,
                bias,
                sh.weights_scale,
                m,
                probs,
                ids,
                weights,
                fault,
            )?,
            RouteOut::E16(_) => routers.e16.enqueue_router_rows(
                stream,
                gpu,
                &cx.k.g32,
                rw,
                &b.xn,
                bias,
                sh.weights_scale,
                m,
                probs,
                ids,
                weights,
                fault,
            )?,
        }
        cx.project(
            &mut b.wide,
            l,
            &b.xn,
            &mut [(&a.latent_down, sh.latent, &mut b.lat)],
        )?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.lat,
            w.f32_buf(WHAT, &a.pre_exp_norm)?,
            sh.eps,
            sh.latent,
            m,
            &mut b.lat2,
        )
    }

    /// Layer `l`'s card experts over the pass's `top_k · m` slots from the
    /// rows `b.lat2`: the step's `_sel` launches (each slot reads its row's
    /// q8_1 column), each slot's down row scaled by `ffn_post_exp_norm` then
    /// normed into `b.down2`. Under [`Places::Cut`] a slot the host serves is
    /// neither read nor written: its up and down rows are cleared first, so
    /// the quantize after the up and the norm after the down read zeros
    /// there, never a row an earlier layer or a faulted pass left.
    fn experts(&mut self, l: usize, places: Places) -> Result<(), GpuError> {
        self.experts_with(l, places, None).map(|_| ())
    }

    /// Hosted layer `l`'s card route under the host's union ([`PlacedPass`]'s
    /// shadow), after the layer's pick: each routed slot's place in the
    /// card's stacks (`ds41_ffn_places`) over `side`'s slot map as the pick
    /// left it (its admits at their slots, its victims
    /// [`HOST`](crate::hybrid::HOST), as the host's serve reads them), the
    /// engine stream's wait on each landing batch of the pick (`land`, the
    /// residency machine's admits into the stage stacks), then the stage
    /// slots ([`PassWalk::experts`] under [`Places::Cut`]) and, when the
    /// layer streams, the ring's ([`PassWalk::experts_with`]). The streamed
    /// experts the ring pass joined, if it ran.
    fn card_route(
        &mut self,
        side: &PlacedSide,
        l: usize,
        land: &[LandBatch],
        ring: Option<RingPass<'_>>,
    ) -> Result<Option<usize>, GpuError> {
        let (gpu, sh, m) = (self.cx.gpu, self.cx.sh, self.cx.m);
        let stream = gpu.stream();
        let b = &mut *self.b;
        side.acc_k.enqueue_places_cols(
            stream,
            &ColPlaces {
                ids: &b.ids,
                ids_pitch: sh.top_k,
                used: sh.top_k,
                sel_pitch: sh.top_k,
                m,
                map: side.slots.buf(),
                row_off: l * side.stride,
                n_expert: sh.n_expert,
            },
            gpu.layer_sink(l)?,
            &mut b.sel,
        )?;
        for event in land.iter().filter_map(|lb| lb.event.as_deref()) {
            stream.wait(event)?;
        }
        self.experts_with(l, Places::Cut, ring)
    }

    /// [`PassWalk::experts`], and with `ring` the stream's slots of the layer
    /// between the stage slots' down and the norm: the latent rows in the
    /// route GEMM's form, the union's places of the pass's slots (the stage's
    /// below `n_card`, `n_card + i` for the `i`-th streamed expert) into
    /// `ring.union` for the back's sum, then per landing batch, behind its own
    /// landed event, the route GEMM over the half's stacks of the batch's
    /// slots ([`RingGemm::enqueue_batch`](crate::ring_gemm::RingGemm::enqueue_batch))
    /// — a slot outside the batch neither read nor written — each slot's down
    /// row into its row of the slots' downs beside the stage slots'. The
    /// streamed experts the ring pass joined; `None` without a ring.
    fn experts_with(
        &mut self,
        l: usize,
        places: Places,
        ring: Option<RingPass<'_>>,
    ) -> Result<Option<usize>, GpuError> {
        let PassWalk { cx, layers, b, .. } = self;
        let (gpu, w, sh, m) = (cx.gpu, cx.w, cx.sh, cx.m);
        let stream = gpu.stream();
        let fault = gpu.layer_sink(l)?;
        let a = &layers[l];
        let b = &mut **b;
        let stacks = a.stacks(WHAT)?;
        let slots = sh.top_k * m;
        let sel = match places {
            Places::Ids => &b.ids,
            Places::Cut => {
                b.up.zero_async(stream)?;
                b.down.zero_async(stream)?;
                &b.sel
            }
        };
        gpu.quantize_q8_1_cols_at(&b.lat2, 0, &mut b.lat_q, m, fault)?;
        cx.k.iq.enqueue_gate_up(
            stream,
            &IqGateUp {
                ty: GgmlType::IQ2_XXS,
                wg: stack(w, &stacks.gate)?,
                wu: stack(w, &stacks.up)?,
                act: &b.lat_q,
                sel,
                n_slots: slots,
                rows_per_expert: sh.ff,
                // Row `t`'s slots read column `t`.
                slots_per_col: sh.top_k,
                rule: GateAct::SiluMul,
            },
            fault,
            &mut b.up,
        )?;
        gpu.quantize_q8_1_cols_at(&b.up, 0, &mut b.up_q, slots, fault)?;
        cx.k.iq.enqueue_gemv_sel(
            stream,
            GgmlType::IQ2_XXS,
            &SelDown {
                w: stack(w, &stacks.down)?,
                act: &b.up_q,
                sel,
                n_slots: slots,
                rows_per_expert: sh.latent,
            },
            fault,
            &mut b.down,
        )?;
        let joined = match ring {
            None => None,
            Some(RingPass {
                r,
                acc_k,
                union,
                gemm: rg,
            }) => {
                let ctx = gpu.context();
                let card = LegStacks::of(w, [&stacks.gate, &stacks.up, &stacks.down])?;
                let kernels = (Some(cx.gemm), &cx.k.g32);
                rg.enqueue_rows(gpu, &cx.k.g32, &b.lat2, m, fault)?;
                r.with_union(ctx, sh.n_expert, |union_map| {
                    acc_k.enqueue_places_cols(
                        stream,
                        &ring_places(&b.ids, union_map, sh, m),
                        fault,
                        union,
                    )
                })?;
                with_ring(&card, a.n_card, &r, ctx, (sh.ff, sh.latent), |blk| {
                    for batch in &r.batches {
                        rg.enqueue_batch(
                            gpu,
                            kernels,
                            blk,
                            batch,
                            (&b.ids, slots, sh.top_k, sh.n_expert),
                            (&mut b.down, GateAct::SiluMul),
                            fault,
                        )?;
                    }
                    Ok(())
                })?;
                Some(r.n)
            }
        };
        gpu.elem().enqueue_gain_rms_norm(
            stream,
            &b.down,
            w.f32_buf(WHAT, &a.post_exp_norm)?,
            sh.post_eps,
            sh.latent,
            slots,
            &mut b.down2,
        )?;
        Ok(joined)
    }

    /// Layer `l`'s FFN back from the experts' weighted sum `sum` (the pass's
    /// `latent` rows): the latent up, the scale-then-norm and the α/β mix of
    /// `b.mid` into `b.x`; the layer is over.
    fn ffn_post(&mut self, l: usize, sum: Sum) -> Result<(), GpuError> {
        let PassWalk { cx, layers, b, .. } = self;
        let (gpu, w, sh, m) = (cx.gpu, cx.w, cx.sh, cx.m);
        let stream = gpu.stream();
        let a = &layers[l];
        let b = &mut **b;
        let sum = match sum {
            Sum::Moe => &b.moe,
            Sum::Lat => &b.lat,
        };
        cx.project(
            &mut b.wide,
            l,
            sum,
            &mut [(&a.latent_up, sh.hidden, &mut b.delta)],
        )?;
        gpu.elem().enqueue_gain_rms_norm(
            stream,
            &b.delta,
            w.f32_buf(WHAT, &a.post_ffw_norm)?,
            sh.post_eps,
            sh.hidden,
            m,
            &mut b.post,
        )?;
        mix_rows(
            (gpu, w),
            (&a.ffn_alpha, &a.ffn_beta),
            (&b.mid, &b.post),
            (sh.hidden, m),
            &mut b.tiles,
            &mut b.x,
        )?;
        self.snap(l)
    }

    /// Layer `l` whole over the pass's rows, every expert it routes to on the
    /// card: the step's [`super::ffn`] at `m` rows.
    fn layer(&mut self, l: usize) -> Result<(), GpuError> {
        self.attention(l)?;
        self.ffn_pre(l)?;
        self.experts(l, Places::Ids)?;
        let (gpu, sh, m) = (self.cx.gpu, self.cx.sh, self.cx.m);
        let b = &mut *self.b;
        gpu.elem().enqueue_weighted_sum(
            gpu.stream(),
            &b.down2,
            &b.weights,
            sh.latent,
            sh.top_k as u32,
            m,
            &mut b.moe,
        )?;
        self.ffn_post(l, Sum::Moe)
    }

    /// Layer `l`'s compared rows, when the taps are armed: the last row's
    /// `ffn_inp` (`b.mid`) and `l_out` (`b.x`) — the step's two residual
    /// taps at the pass's last position — and that row's routing: its kept
    /// ids, their weights and the router's scores, the pass's own (the
    /// route's logits keep the last step's). The other taps are the steps'.
    fn snap(&mut self, l: usize) -> Result<(), GpuError> {
        let Some(t) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let (stream, n, m) = (self.cx.gpu.stream(), self.cx.sh.hidden, self.cx.m);
        let (k, n_expert) = (self.cx.sh.top_k, self.cx.sh.n_expert);
        let tap = &mut t.layers[l];
        copy_f32(stream, (&self.b.mid, (m - 1) * n), (&mut tap.ffn_inp, 0), n)?;
        copy_f32(stream, (&self.b.x, (m - 1) * n), (&mut tap.l_out, 0), n)?;
        // The last row's routing in the layer's route buffers, where the
        // step leaves its own: the node readback's ids and weights.
        let (ids, weights, probs) = match &mut self.layers[l].route {
            RouteOut::E256(o) => (&mut o.ids, &mut o.weights, &mut o.probs),
            RouteOut::E16(o) => (&mut o.ids, &mut o.weights, &mut o.probs),
        };
        copy_u32(stream, (&self.b.ids, (m - 1) * k), ids, k)?;
        copy_f32(stream, (&self.b.weights, (m - 1) * k), (weights, 0), k)?;
        copy_f32(
            stream,
            (&self.b.probs, (m - 1) * n_expert),
            (probs, 0),
            n_expert,
        )
    }

    /// Hosted layer `at.layer`'s front: the attention half, the FFN front,
    /// the download of the experts' rows, the weights and the ids to the
    /// host ([`BatchLeg::key`]'s exchange) — with each slot's tier place on a
    /// layer the expert tier holds experts of. The slots' places in the
    /// card's stacks wait for the shadow's pick ([`PassWalk::card_route`]).
    fn hosted_front(
        &mut self,
        side: &mut PlacedSide,
        leg: &mut BatchLeg<'_, HostRun>,
        at: At,
    ) -> Result<(), GpuError> {
        let l = at.layer;
        self.attention(l)?;
        self.ffn_pre(l)?;
        let (gpu, sh, m) = (self.cx.gpu, self.cx.sh, self.cx.m);
        let stream = gpu.stream();
        let b = &*self.b;
        let key = leg.key(at);
        match side.tier.as_mut().filter(|t| t.k(l) > 0) {
            Some(t) => {
                t.enqueue_pass_places(
                    stream,
                    &side.acc_k,
                    (&b.ids, l, m),
                    (sh.top_k, sh.n_expert),
                    gpu.layer_sink(l)?,
                )?;
                leg.hybrid().enqueue_download_tiered_pitched(
                    stream,
                    [&b.lat2, &b.weights],
                    &b.ids,
                    sh.top_k,
                    &[t.pass_tsel()],
                    key,
                )
            }
            None => leg.hybrid().enqueue_download_pitched(
                stream,
                [&b.lat2, &b.weights],
                &b.ids,
                sh.top_k,
                key,
            ),
        }
    }

    /// Hosted layer `at.layer`'s back: the card slots' weighted sum, the
    /// tier's on a layer the expert tier holds experts of, and the host's
    /// sums, then the FFN back. A layer that streamed sums its card slots by
    /// the stream's union places (`union`: the places and their count, the
    /// stage's `n_card` and the streamed experts past it), so the streamed
    /// experts' rows join the card's sum, on a tier layer before the tier's
    /// join.
    fn hosted_back(
        &mut self,
        side: &PlacedSide,
        leg: &mut BatchLeg<'_, HostRun>,
        (at, union): (At, Option<(&DeviceBuffer<u32>, usize)>),
    ) -> Result<(), GpuError> {
        let l = at.layer;
        let n_tier = side.tier.as_ref().map_or(0, |t| t.k(l));
        if n_tier > 0 {
            return self.tier_back(side, leg, (at, n_tier), union);
        }
        let (gpu, sh, m) = (self.cx.gpu, self.cx.sh, self.cx.m);
        let stream = gpu.stream();
        let b = &mut *self.b;
        let (sel, n_card) = union.unwrap_or((&b.sel, self.layers[l].n_card));
        let hsum = leg.hsum();
        if n_card > 0 {
            side.acc_k.enqueue_card_acc(
                stream,
                &sh.card_acc((&b.down2, &b.weights, sel), (m, n_card), gpu.layer_sink(l)?),
                &mut b.moe,
            )?;
            // The latent down's rows are spent: they take the card's sum
            // plus the host's.
            gpu.elem()
                .enqueue_add(stream, &b.moe, hsum, m * sh.latent, &mut b.lat)?;
        } else {
            copy_f32(stream, (hsum, 0), (&mut b.lat, 0), m * sh.latent)?;
        }
        self.ffn_post(l, Sum::Lat)
    }

    /// [`PassWalk::hosted_back`] of a layer whose `n_tier` experts the expert
    /// tier holds: the card slots' sum into `moe` (by `union`'s places and
    /// count when the layer streamed, else the stage's), the tier's rows joined
    /// through the port — normed into the spent `down` rows and summed into
    /// the spent `lat2` rows ([`enqueue_tier_sum`]) — the two cards' sums, then
    /// the host's, into `moe`.
    fn tier_back(
        &mut self,
        side: &PlacedSide,
        leg: &mut BatchLeg<'_, HostRun>,
        (at, n_tier): (At, usize),
        union: Option<(&DeviceBuffer<u32>, usize)>,
    ) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, sh, m) = (self.cx.gpu, self.cx.sh, self.cx.m);
        let stream = gpu.stream();
        let a = &self.layers[l];
        let fault = gpu.layer_sink(l)?;
        let t = side.tier.as_ref().ok_or(GpuError::state(
            WHAT,
            "the stage card's tier side of a tier layer",
        ))?;
        let gain = self.cx.w.f32_buf(WHAT, &a.post_exp_norm)?;
        let b = &mut *self.b;
        let (sel, n_card) = union.unwrap_or((&b.sel, a.n_card));
        if n_card > 0 {
            side.acc_k.enqueue_card_acc(
                stream,
                &sh.card_acc((&b.down2, &b.weights, sel), (m, n_card), fault),
                &mut b.moe,
            )?;
        }
        let (down, lat2, weights) = (&mut b.down, &mut b.lat2, &b.weights);
        leg.join_tiered(at, |_, trows| {
            let sum = TierSum {
                trows,
                gain,
                post_eps: sh.post_eps,
                w: weights,
                tsel: t.pass_tsel(),
                n_tier,
                n: sh.latent,
                m,
                n_used: sh.top_k,
                fault,
            };
            enqueue_tier_sum(gpu, &side.acc_k, &sum, (down, lat2))
        })?;
        let hsum = leg.hsum();
        let n = m * sh.latent;
        if n_card > 0 {
            gpu.elem()
                .enqueue_add(stream, &b.moe, &b.lat2, n, &mut b.lat)?;
            gpu.elem()
                .enqueue_add(stream, &b.lat, hsum, n, &mut b.moe)?;
        } else {
            gpu.elem()
                .enqueue_add(stream, &b.lat2, hsum, n, &mut b.moe)?;
        }
        self.ffn_post(l, Sum::Moe)
    }
}

/// The places of the pass's `m` rows of routed slots (`ids`, `top_k` a row)
/// through `map`, one word a router expert: a row of the stream's (its union
/// row, or a landing batch's ring slots).
fn ring_places<'a>(
    ids: &'a DeviceBuffer<u32>,
    map: &'a DeviceBuffer<u32>,
    sh: Shape,
    m: usize,
) -> ColPlaces<'a> {
    ColPlaces {
        ids,
        ids_pitch: sh.top_k,
        used: sh.top_k,
        sel_pitch: sh.top_k,
        m,
        map,
        row_off: 0,
        n_expert: sh.n_expert,
    }
}

/// Which pass buffer holds a layer's experts' weighted sum.
#[derive(Clone, Copy)]
enum Sum {
    /// The card's whole sum.
    Moe,
    /// The card's slots' sum plus the host's.
    Lat,
}

/// A placed load's pass `(1, m, Batch)` through the host tier's batch port
/// (module doc): the layers with no host share run whole in their front, a
/// hosted one in its three parts around the host's union. The pass is one
/// walk of one unit of the call's host streaming (`xs`, [`super::stream`]),
/// ending at `walk_end`: a hosted layer's shadow picks, its card route reads
/// what the pick landed and the layer's stream, and its back owes the
/// layer's return.
struct PlacedPass<'a, 'b> {
    walk: PassWalk<'a>,
    side: &'b mut PlacedSide,
    xs: &'b mut XSide,
    walk_end: usize,
    /// The union's place count of the layer whose shadow streamed, until its
    /// back sums by it.
    streamed: Option<usize>,
}

/// Layer `l` of the pass picks at its shadow: the pass runs it, the host
/// serves a share of it and the card holds its stage stacks — the layers the
/// call's host streaming moves ([`PromptStream`](crate::host::prompt_stream::PromptStream)'s `card`).
fn picks(walk: &PassWalk<'_>, side: &PlacedSide, l: usize) -> bool {
    walk.runs(l) && side.hosted[l] && walk.layers[l].n_card > 0
}

impl<'a> LayerProgram for PlacedPass<'a, '_> {
    type Port = BatchLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted[at.layer] && self.walk.runs(at.layer)
    }

    fn front(&mut self, leg: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.walk.runs(at.layer) {
            Ok(())
        } else if self.side.hosted[at.layer] {
            self.walk.hosted_front(&mut *self.side, leg, at)
        } else {
            self.walk.layer(at.layer)
        }
    }

    /// A hosted layer's pick ([`PromptStream::unit_pick`](crate::host::prompt_stream::PromptStream::unit_pick):
    /// nothing when no call streams), its card route behind the pick's
    /// landings over the stage slots and the layer's stream, and the unit's
    /// read of the layer.
    fn shadow(&mut self, leg: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let PlacedPass {
            walk,
            side,
            xs,
            walk_end,
            streamed,
        } = self;
        if !(side.hosted[l] && walk.runs(l)) {
            return Ok(());
        }
        let m = walk.cx.m;
        let (ring, land) = xs
            .stream
            .unit_pick(leg, at, (m, 1), *walk_end, |k| picks(walk, side, k))?;
        *streamed = None;
        if walk.layers[l].n_card > 0 {
            let n_card = walk.layers[l].n_card;
            let ring = match ring {
                Some(r) => Some(xs.ring_pass(r, &side.acc_k)?),
                None => None,
            };
            if let Some(n) = walk.card_route(side, l, &land, ring)? {
                xs.joined(l, n);
                *streamed = Some(n_card + n);
            }
        }
        let streams = streamed.is_some();
        xs.stream
            .unit_read(leg, (at, 1), (land, streams), |k| picks(walk, side, k))
    }

    /// A hosted layer's back, summed by the stream's union places when its
    /// shadow streamed; then the layer's return, owed in a returning call's
    /// last walk.
    fn back(&mut self, leg: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !(self.side.hosted[l] && self.walk.runs(l)) {
            return Ok(());
        }
        let union = match self.streamed.take() {
            Some(n) => Some((self.xs.union_places()?, n)),
            None => None,
        };
        self.walk.hosted_back(&*self.side, leg, (at, union))?;
        let card = picks(&self.walk, self.side, l);
        self.xs.stream.unit_back((at, 1), self.walk_end, card);
        Ok(())
    }
}

/// A pass that runs one layer alone on the rows a gate supplies: the
/// teacher-forced arm of the prompt path ([`GpuModel::prefill_layer`]).
struct Forced<'a> {
    layer: usize,
    /// The layer's input rows for the pass, `hidden` values a row: the rows
    /// the steps' layer before it left. `None` keeps the embedding rows,
    /// layer 0's own input.
    x: Option<&'a [f32]>,
}

impl Body {
    /// One pass of `ids` (at most the prompt buffers' rows) from position
    /// `pos`, layer-major; with `head`, the head over the pass's last row. A
    /// placed load walks the layers through the host tier's batch port
    /// ([`PlacedPass`]). With `only`, the pass runs that one layer on its
    /// rows (the embedding's `pos` and key counts are the pass's own). A
    /// gate's layer span or input override is the steps' instrument and is
    /// refused by name here.
    fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (ids, pos): (&[u32], u32),
        only: Option<Forced<'_>>,
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let sh = self.sh;
        let stream = gpu.stream();
        let rows = self.prompt.rows;
        if m == 0 || m > rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} ids; the prompt buffers hold 1..={rows}"),
            ));
        }
        let n = self.layers.len();
        let span = match &only {
            None => 0..n,
            Some(f) if f.layer >= n => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {} of a model of {n} layers", f.layer),
                ));
            }
            Some(Forced { layer: l, x }) => {
                match x {
                    Some(x) if x.len() != m * sh.hidden => {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "{} input values for a pass of {m} rows of {}",
                                x.len(),
                                sh.hidden
                            ),
                        ));
                    }
                    None if *l != 0 => {
                        return Err(GpuError::shape(
                            WHAT,
                            format!("layer {l} alone needs the rows the layer before it left"),
                        ));
                    }
                    _ => {}
                }
                *l..*l + 1
            }
        };
        if self
            .taps
            .as_ref()
            .is_some_and(|t| t.span.is_some() || t.inject.is_some())
        {
            return Err(GpuError::state(
                WHAT,
                "a pass runs every layer on its own rows: no layer span or input override",
            ));
        }
        let Prompt {
            gemm,
            route,
            hsum,
            b,
            ..
        } = &mut self.prompt;
        b.host_words[P_POS] = pos;
        for h in 0..sh.n_kv {
            b.host_words[P_ROWS + h] = h
                .checked_mul(sh.ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("row {h} · {} + {pos} does not fit a u32 row word", sh.ctx),
                    )
                })?;
        }
        let at = P_ROWS + sh.n_kv;
        b.host_words[at..at + m].copy_from_slice(ids);
        b.words.copy_from_host(stream, &b.host_words)?;
        // Each row's rope row, the step's table at its position.
        b.host_cs.clear();
        for t in 0..m {
            let p = pos + launch_u32(WHAT, "row", t)?;
            self.table.push(p, Direction::Forward, &mut b.host_cs);
        }
        // The copy fills the whole buffer: the rows past the pass's `m` are never read.
        b.host_cs.resize(b.cs.len(), 0.0);
        b.cs.copy_from_host(stream, &b.host_cs)?;
        gemm.enqueue_route_dense(stream, m, route, gpu.unlabelled_sink())?;
        // The embedding rows and their weightless norm: the first layer's
        // input.
        {
            let f = size_of::<u32>();
            let ids_w = Window::<u32>::of(&b.words, at * f, m)?;
            let pos0 = Window::<u32>::of(&b.words, P_POS * f, 1)?;
            gpu.elem().enqueue_embed_rows_q4k(
                stream,
                EmbedRowsArgs {
                    w: stack(w, &self.embed)?,
                    ids: &ids_w,
                    pos0: &pos0,
                    first: 0,
                    y: &mut b.xn,
                    pos: &mut b.pos,
                    n_keys: &mut b.n_keys,
                },
            )?;
            gpu.elem().enqueue_rms_norm(
                stream,
                &b.xn,
                &self.s.ones_hidden,
                sh.eps,
                sh.hidden,
                m,
                &mut b.x,
            )?;
        }
        // A layer alone reads the rows the gate gives it in place of the
        // embedding's; the copy fills the whole buffer, the rows past the
        // pass's are never read.
        if let Some(Forced { x: Some(x), .. }) = &only {
            let mut staged = x.to_vec();
            staged.resize(b.x.len(), 0.0);
            b.x.copy_from_host(stream, &staged)?;
        }
        let cx = Cx {
            gpu,
            w,
            k: &self.k,
            gemm,
            route,
            sh,
            ones_hidden: &self.s.ones_hidden,
            ones_head: &self.s.ones_head,
            m,
        };
        let mut walk = PassWalk {
            cx,
            layers: &mut self.layers,
            b,
            taps: self.taps.as_mut(),
            span: span.clone(),
        };
        match self.placed.as_mut() {
            None => {
                for l in span {
                    walk.layer(l)?;
                }
            }
            Some(Placed { hybrid, side, .. }) => {
                let mut leg = BatchLeg::new(stream, hybrid, hsum, rows);
                let o = Overlap {
                    units: 1,
                    cols: m,
                    port: PortKind::Batch,
                };
                let mut pass = PlacedPass {
                    walk,
                    side,
                    xs: &mut self.xs,
                    walk_end: pos as usize + m,
                    streamed: None,
                };
                sched::walk(o, n, &mut leg, &mut pass)?;
                pass.xs.stream.end_walk(leg.hybrid(), stream)?;
                pass.xs.walk_done();
                walk = pass.walk;
            }
        }
        if let Some(head) = head {
            let f = size_of::<f32>();
            let row = Window::<f32>::of(&walk.b.x, (m - 1) * sh.hidden * f, sh.hidden)?;
            head.input_mut().copy_from_device_async(&row, stream)?;
            enqueue_head(gpu, w, head, self.epilogue)?;
        }
        Ok(())
    }
}

impl GpuModel<Body> {
    /// The rows one pass of [`GpuModel::prefill`] takes: the size the load
    /// chose from the card's room ([`choose_pass_rows`]).
    pub fn pass_rows(&self) -> Result<usize, GpuError> {
        Ok(self.body(WHAT)?.prompt.rows())
    }

    /// [`GpuModel::prefill_with`] in passes of the most ids the prompt
    /// buffers hold.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        let rows = self.body(WHAT)?.prompt.rows();
        self.prefill_with(ids, rows)
    }

    /// Feed `ids` through the chain in passes of up to `rows` ids (module
    /// doc) and return the greedy next token after the last one. Positions
    /// continue from wherever the model stands, and the K/V planes after it
    /// are those of the ids run as steps. `rows` past the prompt buffers'
    /// rows, or 0, and an empty `ids` are refused by name. The passes run
    /// eager in either step mode; a fault any pass raised is read behind the
    /// last pass's head (`GpuModel::run_rows`, the call-end read).
    pub fn prefill_with(&mut self, ids: &[u32], rows: usize) -> Result<u32, GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let cap = self.body(WHAT)?.prompt.rows();
        if rows == 0 || rows > cap {
            return Err(GpuError::shape(
                WHAT,
                format!("passes of {rows} ids; the prompt buffers hold 1..={cap}"),
            ));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "ids", ids.len())? - 1, WHAT)?;
        let n = ids.len().div_ceil(rows);
        prompt_pass(self, |m| {
            m.streamed(ids.len(), |m| {
                let mut next = None;
                for (i, pass) in ids.chunks(rows).enumerate() {
                    let last = i + 1 == n;
                    next = m.run_rows(pass.len(), WHAT, |gpu, w, body, head, pos| {
                        body.enqueue_pass(gpu, w, (pass, pos), None, last.then_some(head))?;
                        Ok(last)
                    })?;
                }
                next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
            })
        })
    }

    /// `run`, a prompt call of `n` ids from the model's position, inside the
    /// call's host streaming ([`Body::stream_begin`], [`Body::stream_end`]):
    /// before a decode history the call returns each layer to the set it
    /// started with, else its placement stays. The call's own error first:
    /// an end refused after a failed call is its echo. Nothing more on a
    /// whole-card load or while the lever is off.
    fn streamed<T>(
        &mut self,
        n: usize,
        run: impl FnOnce(&mut GpuModel<Body>) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        let end = self.pos() as usize + n;
        let history = {
            let (gpu, _, body) = self.body_parts(WHAT)?;
            // Read once: the call keeps no row, so the history does not move until its end.
            let history = body.call_history();
            body.stream_begin(gpu.stream(), n, end, history)?;
            history
        };
        let r = run(self);
        let (gpu, _, body) = self.body_parts(WHAT)?;
        let ended = body.stream_end(gpu.stream(), r.is_ok() && !history);
        let v = r?;
        ended?;
        Ok(v)
    }

    /// Layer `layer` alone over `ids`' rows in passes of up to `rows` ids, the
    /// teacher-forced arm of [`GpuModel::prefill_with`]: every pass reads the
    /// layer's input from `x` (`hidden` values an id, the rows the steps'
    /// layer before it left; `None` for layer 0, which reads the embedding)
    /// and the layer's K/V planes get the rows those inputs make, so a gate
    /// holds each layer's batch arithmetic to the steps' on the steps' own
    /// input, with none of the layers before it between. Positions continue
    /// from wherever the model stands; the taps must be armed, and after the
    /// call the layer's `l_out` tap is the last row's. The head does not run,
    /// no token is read, and the other layers' planes and taps are as they
    /// were. Refused by name: an empty `ids`, `rows` of 0 or past the prompt
    /// buffers', a layer past the model's, `x` of another length than the
    /// ids', no `x` for a layer other than 0.
    pub fn prefill_layer(
        &mut self,
        ids: &[u32],
        rows: usize,
        (layer, x): (usize, Option<&[f32]>),
    ) -> Result<(), GpuError> {
        const WHAT_LAYER: &str = "musespark::prefill_layer";
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT_LAYER, "empty token slice"));
        }
        let (cap, hidden) = {
            let body = self.body(WHAT_LAYER)?;
            (body.prompt.rows(), body.sh.hidden)
        };
        if rows == 0 || rows > cap {
            return Err(GpuError::shape(
                WHAT_LAYER,
                format!("passes of {rows} ids; the prompt buffers hold 1..={cap}"),
            ));
        }
        if let Some(x) = x
            && x.len() != ids.len() * hidden
        {
            return Err(GpuError::shape(
                WHAT_LAYER,
                format!("{} input values for {} ids of {hidden}", x.len(), ids.len()),
            ));
        }
        prompt_pass(self, |m| {
            m.streamed(ids.len(), |m| {
                for (i, pass) in ids.chunks(rows).enumerate() {
                    let x = x.map(|x| &x[i * rows * hidden..][..pass.len() * hidden]);
                    m.run_rows(pass.len(), WHAT_LAYER, |gpu, w, body, _head, pos| {
                        body.enqueue_pass(gpu, w, (pass, pos), Some(Forced { layer, x }), None)?;
                        Ok(false)
                    })?;
                }
                Ok(())
            })
        })
    }

    /// Every layer's output row at the last position of the last call, the
    /// `l_out` tap alone (`layers · hidden` values): a gate's instrument
    /// (it synchronizes the stream, and the taps must be armed).
    pub fn layer_outs(&mut self) -> Result<Vec<f32>, GpuError> {
        const WHAT_OUTS: &str = "musespark::layer_outs";
        let (gpu, _w, body) = self.body_parts(WHAT_OUTS)?;
        let stream = gpu.stream();
        let taps = body
            .taps
            .as_ref()
            .ok_or_else(|| GpuError::state(WHAT_OUTS, "the taps are not armed (set_taps)"))?;
        let mut out = Vec::with_capacity(taps.layers.len() * body.sh.hidden);
        for t in &taps.layers {
            out.extend(t.l_out.to_host_vec(stream)?);
        }
        Ok(out)
    }

    /// Layer `layer`'s output row at the last position of the last call: [`GpuModel::layer_outs`]' one entry, a gate's
    /// instrument (it synchronizes the stream, and the taps must be armed). Refused by name: a layer past the model's.
    pub fn layer_out(&mut self, layer: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT_OUT: &str = "musespark::layer_out";
        let (gpu, _w, body) = self.body_parts(WHAT_OUT)?;
        let taps = body
            .taps
            .as_ref()
            .ok_or_else(|| GpuError::state(WHAT_OUT, "the taps are not armed (set_taps)"))?;
        let tap = taps.layers.get(layer).ok_or_else(|| {
            GpuError::shape(
                WHAT_OUT,
                format!(
                    "layer {layer} is past the model's {} layers",
                    taps.layers.len()
                ),
            )
        })?;
        Ok(tap.l_out.to_host_vec(gpu.stream())?)
    }

    /// Every layer's K and V planes over the positions written so far, by
    /// name (`blk.L.k`, `blk.L.v`), each key head's rows in head order as
    /// little-endian f16 bytes: a gate's instrument for the stores a prompt
    /// and the steps leave (it synchronizes the stream).
    pub fn store_planes(&mut self) -> Result<Vec<(String, Vec<u8>)>, GpuError> {
        let pos = self.pos() as usize;
        let (gpu, _w, body) = self.body_parts(WHAT)?;
        let stream = gpu.stream();
        let (n_kv, ctx, head) = (body.sh.n_kv, body.sh.ctx, body.sh.head);
        let mut out = Vec::with_capacity(2 * body.layers.len());
        for (l, a) in body.layers.iter().enumerate() {
            for (name, plane) in [("k", &a.kc), ("v", &a.vc)] {
                let all = plane.buf().to_host_vec(stream)?;
                let mut bytes = Vec::with_capacity(n_kv * pos * head * 2);
                for h in 0..n_kv {
                    let rows = &all[h * ctx * head..(h * ctx + pos) * head];
                    bytes.extend(rows.iter().flat_map(|v| v.to_le_bytes()));
                }
                out.push((format!("blk.{l}.{name}"), bytes));
            }
        }
        Ok(out)
    }
}
