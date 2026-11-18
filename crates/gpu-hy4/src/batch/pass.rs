//! The placed prompt pass: up to [`PASS_ROWS`] consecutive prompt positions of a placed load
//! (some of the routed layers' experts on the host tier) walked layer by layer as one unit over
//! the host tier's batch port, so each hosted layer's host experts are read once for every row
//! ([`BatchLeg`]: one download, one union serve, one upload) and not once a row, as the decode step
//! does. The walk is the common `sched::walk` of `(1, rows, Batch)`; what it leaves in the stores
//! is what the prompt batch ([`super::prompt`]) leaves, and each row is the placed step's
//! feed-forward block up to rounding (the experts' sum is `card + host`, the card's slots one
//! grouped GEMM over the rows).
//!
//! A layer the card holds whole runs the whole-card block of [`super::ffn`] over the rows; a
//! hosted routed layer runs it in three parts around the host's serve, in the host leg's order:
//!
//! - **front**: the identity mix's fold and the feed-forward norm, the router over the rows, the
//!   latent down into the rows the host reads, the card's places of every slot
//!   ([`PlacedSide::places_rows`]) and the download of the latent rows, weights and ids
//!   ([`PlacedSide::download_rows`]);
//! - **shadow**, under the host's work: the shared expert, and the card's slots as one grouped GEMM
//!   over the layer's card stacks, whose table is the route over the card's slot ids and leaves the
//!   host's slots out ([`PlacedSide::remap_row`], `enqueue_route_remap`), summed in slot order over
//!   the slots the card holds;
//! - **back**: the card's and the host's sums added (the routed sum is linear in the experts'
//!   rows), the latent up, the shared expert added, the sandwich norm and the identity mix's post.
//!
//! The pass's buffers are the prompt batch's at a call of `rows` rows in one unit, beside the host's
//! sums, the layer's map, the slots' places and the card slots' route tables ([`PlacedPass`]); the
//! plan reserves their bytes ([`pass_bytes`]) and the load holds them to it.

use bloomery_gpu::card_slots::CardAcc;
use bloomery_gpu::gemm::{GEMM_MAX_SLOTS, GemmRoute};
use bloomery_gpu::hc_ident::{HC, HeadArgs, MIX};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::census::TierCensus;
use bloomery_gpu::host::placed::{PlacedHost, PlacedSide};
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::{BatchLeg, HybridStats};
use bloomery_gpu::latent::{INDEX_HEAD, POOL};
use bloomery_gpu::routed_block::{RouteIn, RoutedRows, RouterEntry};
use bloomery_gpu::site::WideForms;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::names::{self, Sub};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use super::{
    Buf, Cfg, Env, ExpertIo, FFN_NORM, FFN_OUT, L_OUT, N_EXPERT, N_USED, Prompt, Q_MAX, Span, WHAT,
    WalkTaps, WideP, attention, chunks, engram, expert_rows, ffn, io, post, pre_norm, project,
    shape, stage, tap, tap_words,
};
use crate::body::{ENGRAM_W, Ffn, LATENT, ROPE, ROW, Shape, Store};

/// The most rows one pass holds: the host union's widest unit.
pub const PASS_ROWS: usize = model::moe::UNION_MAX_COLS;

/// What this file's refusals name.
const PASS: &str = "hy4 placed pass";

/// A pass shape: the rows of one pass and the rows of its attention chunk (the engram and attention
/// sub-layers run `q_rows` rows a launch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassShape {
    pub rows: usize,
    pub q_rows: usize,
}

impl PassShape {
    /// The shape of a pass of `rows` rows.
    #[must_use]
    pub const fn of(rows: usize) -> PassShape {
        PassShape {
            rows,
            q_rows: if rows < Q_MAX { rows } else { Q_MAX },
        }
    }
}

/// The shapes a plan may hold, widest first: the host union's widest unit down to the narrowest
/// the card's room may be left with. A narrower pass reads each hosted layer's experts once for
/// fewer rows.
pub const SHAPES: [PassShape; 4] = [
    PassShape::of(PASS_ROWS),
    PassShape::of(PASS_ROWS / 2),
    PassShape::of(PASS_ROWS / 4),
    PassShape::of(PASS_ROWS / 8),
];

/// The prompt batch's geometry for a pass of `rows` rows: one walk, one unit. Refused by name:
/// no rows, or more than [`PASS_ROWS`].
fn pass_cfg(rows: usize) -> Result<Cfg, GpuError> {
    if !(1..=PASS_ROWS).contains(&rows) {
        return Err(GpuError::shape(
            PASS,
            format!("a pass of {rows} rows: 1 to {PASS_ROWS} rows"),
        ));
    }
    Ok(Cfg {
        cap: rows,
        q_rows: PassShape::of(rows).q_rows,
        ffn_rows: rows,
    })
}

// The unit's slots fit one grouped GEMM.
const _: () = assert!(PASS_ROWS * N_USED <= GEMM_MAX_SLOTS);

/// The device bytes a placed pass of `rows` rows holds on a load of `ctx` positions, never below
/// what [`PlacedPass::new`] makes: the plan's prompt reserve ([`PlanInputs::with_prompt_reserve`]),
/// which the load holds the made bytes to. Every buffer of the prompt batch at the pass's
/// geometry, counted by the widths it is allocated at; the quantized forms the stacks and sites
/// read are counted in both of the forms the kernels have (the file's types pick one, which the
/// plan cannot read), so the bound sits a few megabytes over the made bytes at 512 rows.
///
/// # Errors
/// [`GpuError`] by name for a pass of no row or past [`PASS_ROWS`], and a shape the body is not
/// built for.
///
/// [`PlanInputs::with_prompt_reserve`]: model::arch::place_hybrid::PlanInputs::with_prompt_reserve
pub fn pass_bytes(hp: &Hparams, ctx: usize, rows: usize) -> Result<u64, GpuError> {
    let cfg = pass_cfg(rows)?;
    let sh = Shape::of(hp, ctx)?;
    let (n, q) = (sh.hidden, cfg.q_rows);
    let (cap, r) = (cfg.cap, q.max(cfg.ffn_rows));
    let slots = cfg.ffn_rows * N_USED;
    let heads = sh.max_heads;
    let pools = ctx.div_ceil(POOL);
    let (sites, cols) = hp
        .engram
        .as_ref()
        .map_or((0, 1), |e| (e.layers.len(), e.n_cols()));
    let ff = hp.dense_ff.max(hp.experts.shared_width);
    let (wv, wms) = bloomery_gpu::flash::window_partials_lens(q * heads, sh.window);
    let part_v = bloomery_gpu::flash::partials_v_len(q * heads, sh.top_k).max(wv);
    let part_ms = bloomery_gpu::flash::partials_ms_len(q * heads, sh.top_k).max(wms);
    let f32s = 2 * cap * HC * n
        + 2 * cap * ROPE
        + sites * cap * cols * (ENGRAM_W + 1)
        + r * MIX
        + 2 * r * n
        + 2 * q * sh.q_lora
        + 2 * q * heads * sh.head_k
        + q * heads * ROW
        + 3 * q * ROW
        + q * sh.idx_heads * INDEX_HEAD
        + q * INDEX_HEAD
        + q * sh.idx_heads
        + q * sh.idx_heads * INDEX_HEAD
        + q * sh.idx_heads
        + part_v
        + part_ms
        + q * heads * LATENT
        + 3 * q * heads * sh.head_v
        + 2 * r * n
        + q * pools
        + 3 * r * ff
        + 2 * r * sh.latent
        + 3 * r * n
        + cfg.ffn_rows * N_EXPERT
        + slots
        + 2 * slots * sh.expert_ff
        + slots * sh.latent
        + q * cols * ENGRAM_W
        + q * (HC + 1) * n
        + q * HC * n
        + q * HC
        + HC * n;
    let u32s = 2 * cap + cap.div_ceil(q) + q + q * sh.top_k + q + slots;
    let u16s = q * ROW;
    // The rope tail's gather tables at each head count a layer has.
    let mut counts: Vec<usize> = (0..hp.n_layer).map(|l| hp.heads(l)).collect();
    counts.sort_unstable();
    counts.dedup();
    let gather: usize = counts.iter().map(|hd| 2 * q * hd * ROPE).sum();
    // The wide forms at each input width a site reads, in both forms.
    let mut widths = vec![
        n,
        sh.q_lora,
        sh.latent,
        hp.dense_ff,
        hp.experts.shared_width,
    ];
    widths.extend(counts.iter().map(|hd| hd * sh.head_v));
    if let Some(e) = &hp.engram {
        widths.push(e.n_cols() * e.key_length);
    }
    widths.sort_unstable();
    widths.dedup();
    let both = WideForms::q8(true, true);
    let wide: usize = widths.iter().map(|&k| both.bytes(r, k)).sum();
    let experts = both.bytes(cfg.ffn_rows, sh.latent) + both.bytes(slots, sh.expert_ff);
    // A hosted layer's card slots run over a table of its card count, one table a distinct count
    // ([`PlacedPass`]); a plan does not know the placement, so one a routed layer, each no wider
    // than the whole stack's.
    let routed = (0..hp.n_layer).filter(|&l| !hp.dense(l)).count();
    let routes = GemmRoute::bytes_at(r, 1) + (1 + routed) * GemmRoute::bytes_at(slots, N_EXPERT);
    // The host's sums, the layer's map and the slots' places.
    let beside = rows * sh.latent * 4 + N_EXPERT * 4 + rows * N_USED * 4;
    let total = 4 * f32s + 4 * u32s + 2 * u16s + 4 * gather + wide + experts + routes + beside;
    u64::try_from(total).map_err(|_| GpuError::shape(PASS, "a pass past u64 bytes"))
}

/// The placed pass's buffers: the prompt batch at the pass's geometry, the host's sums, the layer's
/// map of its experts' places, the places of the unit's routed slots and the route tables of the
/// hosted layers' card slots. Held for the load's life.
pub struct PlacedPass {
    prompt: Prompt,
    /// The host's sums of the unit's rows, in the latent's width, which the batch leg lands them in.
    hsum: DeviceBuffer<f32>,
    /// A hosted layer's row of the slot map: each expert's place in the card's stacks, or `HOST`.
    map: DeviceBuffer<u32>,
    /// The places of the unit's routed slots, from the same map.
    sel: DeviceBuffer<u32>,
    /// The route tables of the hosted layers' card slots, one a distinct card count: a grouped GEMM's
    /// table holds the experts of the one stack it routes over, and a hosted layer's stacks hold
    /// only its card's experts.
    routes: Vec<GemmRoute>,
    bytes: usize,
}

/// The distinct card counts of `side`'s hosted layers that hold experts on the card, ascending.
fn hosted_counts(side: &PlacedSide) -> Vec<usize> {
    let mut counts: Vec<usize> = side
        .hosted_set()
        .into_iter()
        .map(|l| side.n_card(l))
        .filter(|&n| n > 0)
        .collect();
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// The table of the hosted layers whose card holds `n_card` experts, from `routes`
/// ([`PlacedPass::routes`]). Refused by name when the pass made none for the count.
fn route_of(
    routes: &mut [GemmRoute],
    (l, n_card): (usize, usize),
) -> Result<&mut GemmRoute, GpuError> {
    routes
        .iter_mut()
        .find(|t| t.n_experts() == n_card)
        .ok_or_else(|| {
            shape(format!(
                "layer {l}: {n_card} card experts and no route table made for the count"
            ))
        })
}

impl PlacedPass {
    /// The pass's buffers for `rows` rows over `body`'s layers and `side`'s hosted layers. Load-time
    /// only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        body: &crate::body::Body,
        side: &PlacedSide,
        rows: usize,
    ) -> Result<PlacedPass, GpuError> {
        let cfg = pass_cfg(rows)?;
        let prompt = Prompt::new(gpu, w, body, cfg)?;
        let stream = gpu.stream();
        let hsum = DeviceBuffer::zeroed(stream, rows * body.sh.latent)?;
        let map = DeviceBuffer::zeroed(stream, N_EXPERT)?;
        let sel = DeviceBuffer::zeroed(stream, rows * N_USED)?;
        let slots = cfg.ffn_rows * N_USED;
        let routes = hosted_counts(side)
            .into_iter()
            .map(|n| GemmRoute::new(stream, slots, n))
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = prompt.bytes()
            + hsum.num_bytes()
            + map.num_bytes()
            + sel.num_bytes()
            + routes.iter().map(GemmRoute::bytes).sum::<usize>();
        Ok(PlacedPass {
            prompt,
            hsum,
            map,
            sel,
            routes,
            bytes,
        })
    }

    /// The rows a pass holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.prompt.cfg.cap
    }

    /// Arm the pass's taps: every later walk copies every stage ([`crate::TAP_STAGES`]) of every
    /// layer for its rows ([`Prompt::arm_taps`]).
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.prompt.arm_taps(gpu)
    }

    /// The last walk's taps for its first `rows` rows, read back ([`Prompt::taps`]).
    pub fn taps(&self, stream: &CudaStream, rows: usize) -> Result<WalkTaps, GpuError> {
        self.prompt.taps(stream, rows)
    }

    /// The device bytes the pass holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// The pass's walk `(1, rows, Batch)`: layer `l`'s attention (and engram site) over the rows, then
/// its feed-forward block whole, or as the front, shadow and back around the host's serve.
struct PassWalk<'a> {
    env: Env<'a>,
    stores: &'a mut [Store],
    p: &'a mut Prompt,
    side: &'a PlacedSide,
    map: &'a mut DeviceBuffer<u32>,
    sel: &'a mut DeviceBuffer<u32>,
    /// The hosted layers' card route tables ([`PlacedPass::routes`]).
    routes: &'a mut [GemmRoute],
    /// The call's rows, all one unit.
    t: usize,
    /// The streams' buffer holding the layer's input, as the prompt batch's walk keeps it.
    cur: usize,
}

impl PassWalk<'_> {
    /// The span of the whole call's rows over the streams' buffer `cur`.
    fn span(&self) -> Span {
        Span {
            cur: self.cur,
            r0: 0,
            index: 0,
            n: self.t,
        }
    }

    /// Layer `l`'s engram site and attention sub-layer over every row, in the prompt batch's
    /// chunks.
    fn attention_half(&mut self, l: usize) -> Result<(), GpuError> {
        let t = self.t;
        let q = self.p.cfg.q_rows;
        if self.env.layers[l].engram.is_some() {
            for (index, rows) in chunks(t, q).enumerate() {
                let sp = Span {
                    cur: self.cur,
                    r0: rows.start,
                    index,
                    n: rows.len(),
                };
                engram(&self.env, self.p, l, sp)?;
            }
            self.cur ^= 1;
        }
        for (index, rows) in chunks(t, q).enumerate() {
            let sp = Span {
                cur: self.cur,
                r0: rows.start,
                index,
                n: rows.len(),
            };
            attention(&self.env, self.stores, self.p, l, sp)?;
        }
        self.cur ^= 1;
        Ok(())
    }

    /// A hosted layer's front from the feed-forward norm: the router over the rows, the latent
    /// down into the rows the host reads, the card's places and the download.
    fn hosted_front(&mut self, port: &mut BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let n = self.t;
        let sp = self.span();
        let Ffn::Routed { down_lat, .. } = &layers[l].ffn else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let Prompt {
            gemm,
            dense,
            wide,
            buf,
            taps,
            ..
        } = &mut *self.p;
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
            lat,
            probs,
            ids,
            weights,
            ..
        } = buf;
        pre_norm(
            &env,
            (l, Sub::Ffn),
            sp,
            (&hs[sp.cur], mix, y, xn),
            &names::ffn_norm(l),
        )?;
        tap(gpu, taps, (l, FFN_NORM), (&*xn, 0), (0, n, sh.hidden))?;
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
            tap_words(gpu, &mut t.picks[l], ids, (0, 0, n * N_USED))?;
        }
        project(&wp, wide, l, n, xn, &mut [(down_lat, sh.latent, lat)])?;
        self.side.places_rows(gpu, l, (ids, n), self.sel)?;
        self.side
            .download_rows(port, at, (&*lat, &*weights, &*ids), &[])
    }

    /// A hosted layer's card work under the host's: the shared expert over the rows and the
    /// card's slots, summed in slot order into the card's routed sum.
    fn hosted_shadow(&mut self, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let (n, hid) = (self.t, sh.hidden);
        let Ffn::Routed {
            stacks,
            shared: glu,
            clamp,
            ..
        } = &layers[l].ffn
        else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let Prompt {
            gemm,
            slots,
            dense,
            wide,
            acts,
            buf,
            ..
        } = &mut *self.p;
        gemm.enqueue_route_dense(stream, n, dense, fault)?;
        let wp = WideP {
            gpu,
            w,
            g32: &k.g32,
            gemm,
            dense,
        };
        let Buf {
            xn,
            lat,
            routed,
            fg,
            fu,
            fa,
            shared,
            ids,
            weights,
            gate_rows,
            up_rows,
            down_rows,
            ..
        } = buf;
        project(
            &wp,
            wide,
            l,
            n,
            xn,
            &mut [(&glu.gate, glu.ff, fg), (&glu.up, glu.ff, fu)],
        )?;
        gpu.elem().enqueue_swiglu(stream, fg, fu, n * glu.ff, fa)?;
        project(&wp, wide, l, n, fa, &mut [(&glu.down, hid, shared)])?;
        let n_card = self.side.n_card(l);
        if n_card == 0 {
            return Ok(());
        }
        let table = route_of(self.routes, (l, n_card))?;
        self.side.remap_row(stream, l, self.map)?;
        expert_rows(
            &env,
            (l, n),
            (&*gemm, &mut *acts, table),
            (Some(&*self.map), stacks, *clamp),
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
                sel: self.sel,
                n: sh.latent,
                m: n,
                n_card,
                n_used: N_USED,
                w_pitch: N_USED,
                fault,
            },
            routed,
        )
    }

    /// A hosted layer's back from the join on: the card's and the host's sums added, the latent
    /// up, the shared expert added, the sandwich norm, the identity mix's post.
    fn hosted_back(&mut self, port: &BatchLeg<'_, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let env = self.env;
        let Env {
            gpu,
            w,
            sh,
            layers,
            k,
            ..
        } = env;
        let (stream, fault) = (gpu.stream(), gpu.layer_sink(l)?);
        let (n, hid) = (self.t, sh.hidden);
        let sp = self.span();
        let Ffn::Routed { up_lat, .. } = &layers[l].ffn else {
            return Err(shape(format!(
                "layer {l}: a hosted layer with no routed block"
            )));
        };
        let hsum = port.hsum();
        let Prompt {
            gemm,
            dense,
            wide,
            buf,
            taps,
            ..
        } = &mut *self.p;
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
            lat,
            routed,
            lat_up,
            shared,
            f,
            out,
            ..
        } = buf;
        // The card's slots, then the host's: the step's back.
        let sum: &DeviceBuffer<f32> = if self.side.n_card(l) > 0 {
            gpu.elem()
                .enqueue_add(stream, routed, hsum, n * sh.latent, lat)?;
            &*lat
        } else {
            hsum
        };
        project(&wp, wide, l, n, sum, &mut [(up_lat, hid, lat_up)])?;
        let elem = gpu.elem();
        elem.enqueue_add(stream, lat_up, shared, n * hid, f)?;
        elem.enqueue_rms_norm(
            stream,
            f,
            w.f32_buf(WHAT, &names::post_ffw_norm(l))?,
            sh.eps,
            hid,
            n,
            out,
        )?;
        tap(gpu, taps, (l, FFN_OUT), (&*out, 0), (0, n, hid))?;
        post(&env, hs, sp, (out, mix))?;
        let (_, after) = io(hs, sp.cur);
        tap(gpu, taps, (l, L_OUT), (after, 0), (0, n, HC * hid))?;
        self.cur ^= 1;
        Ok(())
    }
}

impl<'a> LayerProgram for PassWalk<'a> {
    type Port = BatchLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    /// A layer the card holds whole entire; a hosted one up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.attention_half(l)?;
        if self.side.hosted(l) {
            return self.hosted_front(port, at);
        }
        let ffn_rows = self.p.cfg.ffn_rows;
        for (index, rows) in chunks(self.t, ffn_rows).enumerate() {
            let sp = Span {
                cur: self.cur,
                r0: rows.start,
                index,
                n: rows.len(),
            };
            ffn(&self.env, self.p, l, sp)?;
        }
        self.cur ^= 1;
        Ok(())
    }

    fn shadow(&mut self, _port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        self.hosted_shadow(at)
    }

    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if !self.side.hosted(at.layer) {
            return Ok(());
        }
        self.hosted_back(port, at)
    }
}

/// Walk `toks` (positions `pos0 ..`, at most the pass's rows) through the layers by the host
/// tier's batch port; the head runs after the last layer when `last`.
fn walk_pass(
    gpu: &Gpu,
    w: &Weights,
    body: &mut crate::body::Body,
    head: &mut Head,
    (toks, pos0): (&[u32], u32),
    last: bool,
) -> Result<(), GpuError> {
    let crate::body::Body {
        sh,
        layers,
        k,
        stores,
        host,
        rope_rows,
        placed,
        pass,
        taps,
        ..
    } = body;
    if taps.is_some() {
        return Err(GpuError::shape(PASS, "taps are armed: a pass does not tap"));
    }
    let PlacedHost { shell, side, .. } = placed.as_mut().ok_or(GpuError::State {
        what: PASS,
        missing: "the placed side of a placed load",
    })?;
    let PlacedPass {
        prompt,
        hsum,
        map,
        sel,
        routes,
        ..
    } = pass.as_deref_mut().ok_or(GpuError::State {
        what: PASS,
        missing: "the pass's buffers (a load opened with its pass)",
    })?;
    let t = toks.len();
    if t == 0 || t > prompt.cfg.cap {
        return Err(shape(format!(
            "a walk of {t} rows in buffers of {}",
            prompt.cfg.cap
        )));
    }
    let sh = *sh;
    stage(host, prompt, gpu.stream(), &sh, (toks, pos0))?;
    let env = Env {
        gpu,
        w,
        sh,
        layers,
        k,
        rope_rows,
    };
    let n_layers = layers.len();
    let mut leg = BatchLeg::new(gpu.stream(), &mut shell.hybrid, hsum, prompt.cfg.cap);
    let mut prog = PassWalk {
        env,
        stores,
        p: &mut *prompt,
        side: &*side,
        map,
        sel,
        routes,
        t,
        cur: 0,
    };
    let o = Overlap {
        units: 1,
        cols: t,
        port: PortKind::Batch,
    };
    sched::walk(o, n_layers, &mut leg, &mut prog)?;
    let cur = prog.cur;
    if last {
        let n = sh.hidden;
        let x = super::win(&prompt.buf.hs[cur], t - 1, 1, HC * n)?;
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
        head.enqueue(gpu, w)?;
    }
    Ok(())
}

/// The prompt `ids` through `model`'s placed pass from its position, in passes of up to the load's
/// rows: the stores, the engram history and the position stand where `ids.len()` decode steps
/// leave them, and the return is the greedy id after the last row. The call is one residency pass
/// ([`prompt_pass`]: its boundary before the first walk, none inside, 0 rows kept), so every walk
/// reads one slot map. Refused by name: no ids, a load made without the pass, a position the
/// stores do not stand at; a call that fails part-way leaves the stores to a reset.
pub fn prompt_placed(
    model: &mut GpuModel<crate::body::Body>,
    ids: &[u32],
) -> Result<u32, GpuError> {
    const CALL: &str = "hy4::batch::prompt_placed";
    if ids.is_empty() {
        return Err(GpuError::shape(CALL, "a prompt of no id"));
    }
    let rows = model
        .body(CALL)?
        .pass_rows()
        .ok_or_else(|| GpuError::shape(CALL, "a load made without the placed pass"))?;
    let n = u32::try_from(ids.len()).map_err(|_| GpuError::shape(CALL, "ids past u32"))?;
    let token = prompt_pass(model, |model| {
        model.run_rows(ids.len(), CALL, |gpu, w, body, head, pos| {
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
            body.fit(pos as usize, ids.len())?;
            // Anything past this point that fails leaves the engram history and the stores part-way.
            body.held = u32::MAX;
            let walks = ids.len().div_ceil(rows);
            for (i, chunk) in ids.chunks(rows).enumerate() {
                let at = pos + (i * rows) as u32;
                walk_pass(gpu, w, body, head, (chunk, at), i + 1 == walks)?;
            }
            body.held = pos + n;
            Ok(true)
        })
    })?;
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))
}

impl crate::body::Body {
    /// The rows the load's placed pass holds; `None` on a load made without it.
    #[must_use]
    pub fn pass_rows(&self) -> Option<usize> {
        self.pass.as_deref().map(PlacedPass::rows)
    }

    /// The load's placed pass, for a gate that arms its taps and reads them back.
    pub fn pass_mut(&mut self) -> Option<&mut PlacedPass> {
        self.pass.as_deref_mut()
    }

    /// The load's placed pass.
    #[must_use]
    pub fn pass(&self) -> Option<&PlacedPass> {
        self.pass.as_deref()
    }

    /// The routed layers the host tier serves a share of; 0 on a whole-card load.
    #[must_use]
    pub fn hosted_layers(&self) -> usize {
        self.placed.as_ref().map_or(0, |p| {
            (0..self.layers.len()).filter(|&l| p.side.hosted(l)).count()
        })
    }

    /// The device bytes the load's placed pass holds; 0 on a load made without it.
    #[must_use]
    pub fn pass_device_bytes(&self) -> usize {
        self.pass.as_deref().map_or(0, PlacedPass::bytes)
    }

    /// The host tier's census since load ([`TierCensus`]: the services' picks and where the host
    /// read them, and the slots a prompt left out for the card's route); `None` on a whole-card
    /// load and on a host that counts no reads.
    #[must_use]
    pub fn host_census(&self) -> Option<TierCensus> {
        self.placed.as_ref()?.shell.hybrid.census(None)
    }

    /// The host tier's service counters since load ([`HybridStats`]: the batch serves and their
    /// columns among them); `None` on a whole-card load.
    #[must_use]
    pub fn host_stats(&self) -> Option<HybridStats> {
        Some(self.placed.as_ref()?.shell.hybrid.stats())
    }
}
