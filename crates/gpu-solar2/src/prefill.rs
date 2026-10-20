//! The prompt batch: a prompt of `P` ids fed in units of at most
//! [`T_MAX`](crate::body::T_MAX) positions ([`prefill`]), which leaves the
//! model where `P` decode steps over the same ids leave it in everything a
//! later step reads: every layer's recurrent state and conv ring, every GQA
//! layer's key and value rows at those positions, and the last position's
//! logits. Each layer runs over the whole unit, one launch a site a layer
//! (not a token or a chunk): the KDA mixer is [`mixer_kda::enqueue`] at the
//! unit's `m` rows over [`BatchSites`] (the GEMM arm of the typed sites), the
//! GQA layer is [`gqa_rows`], and the routed block is [`Routed`]'s row parts
//! over the same rows, so a unit writes the steps' bits, whatever the cut.
//!
//! A call is cut by [`runtime::prompt::call_batches`] into units, and its
//! units run in groups of [`Extra::set_group`]'s size ([`groups`]); a group
//! is one walk of the runtime's layer schedule at the point `(g, T, Batch)`
//! ([`sched::walk`]) through the host tier's batch port ([`BatchLeg`]). Per
//! layer and unit:
//! - the front: the mixer by the layer's kind (`Gqa`: [`gqa_rows`];
//!   `DeltaRule`: the KDA mixer and the residual add), then the routed
//!   block's norm and router over the unit's rows and the download of the
//!   rows, ids and weights to the host;
//! - the shadow: the shared expert and the card experts, under the host leg;
//! - the back: the host sums (the walk's serve uploaded them) joined with the
//!   shadow's and the residual after the mixer, into the unit's `x`.
//!
//! The head runs after the last layer of the last unit, for its last
//! position alone.
//!
//! A group's units share every buffer one part of one item writes and the
//! same or the next part of that item reads (the mixers' intermediates, the
//! routed block's rows, the activation forms, the host sums); what a unit's
//! back, or its later layers, read after the next unit's front ran is the
//! unit's own ([`Unit`]: the layer's input and output rows, the residual
//! after the mixer, the positions and the GQA word record). All 48 layers
//! carry a routed block, so no dense layer sits behind a routed one.
//!
//! The body is reached through two functions: [`parts`] (its fields as
//! disjoint borrows) and [`extra_dims`] (the batch buffers for a change of
//! group); each has its contract written at its definition.

use std::ops::Range;

use bloomery_gpu::arch::hybrid::{GqaBufs, GqaCx, GqaWide, WideCx, gqa_rows, project_wide};
use bloomery_gpu::flash_gqa::HEAD;
use bloomery_gpu::flash_gqa_prefill::FlashGqaPrefill;
use bloomery_gpu::gemm::{GemmKernels, GemmRoute};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::read_fault;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::linear::mixer_kda::{self, KdaBufs, KdaDims, KdaIn, KdaProj, KdaState, KdaVecs};
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::site::{WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, Pitched, dtod_2d};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::DeviceBuffer;
use model::embed::EmbedTable;
use runtime::layer::MixerKind;
use runtime::prompt::{GROUP_LEVER, call_batches, call_end, check_group, group_sets, groups};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::{
    Batch, Body, Dims, KdaSites, LayerCfg, Scratch, Store, T_MAX, WORD_POS, WORD_ROWS,
};
use crate::ffn::Routed;
use crate::step::Kernels;

/// What this file's refusals name.
const WHAT: &str = "solar2 prefill";

/// Query heads a key head serves: the GQA layers' group.
const GROUP: usize = 8;

/// The largest `BLOOMERY_PREFILL_GROUP`: the lever registry's.
const GROUP_MAX: usize = bloomery_levers::PREFILL_GROUP_MAX as usize;
const _: () = assert!(GROUP_MAX as u64 == bloomery_levers::PREFILL_GROUP_MAX);

// One unit's columns are one host union call's.
const _: () = assert!(T_MAX <= model::moe::UNION_MAX_COLS);

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// The body's fields the batch walks, split into the disjoint borrows the
/// walk holds at once; [`parts`] is the one function that names them.
pub(crate) struct Parts<'a> {
    pub hybrid: &'a mut Hybrid<HostRun>,
    pub dims: &'a Dims,
    pub cfg: &'a [LayerCfg],
    pub stores: &'a mut [Store],
    pub batch: &'a mut Batch,
    pub extra: &'a mut Extra,
    pub k: &'a Kernels,
    pub embed: &'a EmbedTable,
    pub ffn: &'a mut Routed,
    /// The body's highest stored position, raised when a call succeeds.
    pub at: &'a mut u32,
    /// Each layer's tap, when a gate armed them: the layer's output at the
    /// last position of the call.
    pub taps: Option<&'a mut [DeviceBuffer<f32>]>,
}

/// `body` as the walk's [`Parts`].
fn parts(body: &mut Body) -> Parts<'_> {
    let Body {
        shell,
        dims,
        cfg,
        stores,
        batch,
        prefill,
        k,
        ffn,
        at,
        embd,
        ..
    } = body;
    Parts {
        taps: shell.taps.as_deref_mut(),
        hybrid: &mut shell.hybrid,
        dims,
        cfg,
        stores,
        batch,
        extra: prefill,
        k,
        embed: &embd.table,
        ffn,
        at,
    }
}

/// The body's batch buffers and dims, for a change of group.
fn extra_dims(body: &mut Body) -> (&mut Extra, &Dims) {
    (&mut body.prefill, &body.dims)
}

/// A unit's host staging: the embedding rows before their upload and the
/// unit's positions, live key counts and GQA word record.
struct Stage {
    rows: Vec<f32>,
    pos: Vec<u32>,
    keys: Vec<u32>,
    words: Vec<u32>,
}

impl Stage {
    fn new(embd: usize, n_kv: usize) -> Stage {
        Stage {
            rows: vec![0.0; T_MAX * embd],
            pos: vec![0; T_MAX],
            keys: vec![0; T_MAX],
            words: vec![0; WORD_ROWS + n_kv],
        }
    }

    /// The unit `run`'s positions, live key counts (the position plus one)
    /// and word record over stores of `ctx` positions.
    fn set(&mut self, run: &Range<u32>, ctx: usize) -> Result<(), GpuError> {
        let p0 = run.start;
        let t = (run.end - run.start) as usize;
        for ((pos, keys), p) in self.pos[..t]
            .iter_mut()
            .zip(&mut self.keys[..t])
            .zip(run.clone())
        {
            *pos = p;
            *keys = p + 1;
        }
        self.words[WORD_POS] = p0;
        for (h, word) in self.words[WORD_ROWS..].iter_mut().enumerate() {
            *word = h
                .checked_mul(ctx)
                .and_then(|r| r.checked_add(p0 as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    shape(format!(
                        "row {h} · {ctx} + {p0} does not fit a u32 row word"
                    ))
                })?;
        }
        Ok(())
    }
}

/// A group's unit's own buffers beyond the first unit's (those are the
/// batch's [`Scratch`]): see the module doc.
struct UnitBufs {
    x: DeviceBuffer<f32>,
    x1: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    words: DeviceBuffer<u32>,
}

impl UnitBufs {
    fn new(gpu: &Gpu, embd: usize, n_kv: usize) -> Result<UnitBufs, GpuError> {
        let stream = gpu.stream();
        Ok(UnitBufs {
            x: DeviceBuffer::zeroed(stream, T_MAX * embd)?,
            x1: DeviceBuffer::zeroed(stream, T_MAX * embd)?,
            pos: DeviceBuffer::zeroed(stream, T_MAX)?,
            n_keys: DeviceBuffer::zeroed(stream, T_MAX)?,
            words: DeviceBuffer::zeroed(stream, WORD_ROWS + n_kv)?,
        })
    }

    fn bytes(&self) -> usize {
        self.x.num_bytes()
            + self.x1.num_bytes()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + self.words.num_bytes()
    }
}

/// What the prompt batch holds beyond the batch's [`Scratch`] and wide forms:
/// the host sums the batch port uploads, the grouped GEMM's and the prefill
/// flash's kernels, the dense one-expert table, the wide forms of the widths
/// the batch's `wide` does not hold, the qkv window's staging, the host
/// staging of every unit and the units a group of two or more needs. Made
/// once, so that a timed prompt allocates nothing.
pub struct Extra {
    group: usize,
    units: Vec<UnitBufs>,
    stages: Vec<Stage>,
    hsum: DeviceBuffer<f32>,
    gemm: GemmKernels,
    route: GemmRoute,
    flash: FlashGqaPrefill,
    /// The KDA qkv sites' outputs before their copy into the joined qkv rows'
    /// three windows: a GEMM writes token-major rows of one site's width.
    stage: [DeviceBuffer<f32>; 3],
    /// The wide forms by input width `k`, for the widths the batch's `wide`
    /// does not hold: the Q5_0 low-rank halves' and the output projections'.
    arms: Vec<(usize, WideActs)>,
}

impl Extra {
    /// The buffers for a group of `group` units over `stores`' sites.
    /// Refused by name for a group outside the lever's range and for a load
    /// whose heads are not [`GROUP`] a key head. Load-time only.
    pub fn new(gpu: &Gpu, d: &Dims, stores: &[Store], group: usize) -> Result<Extra, GpuError> {
        check_group(group, GROUP_MAX).map_err(|r| GpuError::Shape {
            what: GROUP_LEVER,
            detail: r.to_string(),
        })?;
        if d.heads != GROUP * d.kv_heads {
            return Err(shape(format!(
                "{} query heads over {} key heads: the batch's GQA layers run {GROUP} a key head",
                d.heads, d.kv_heads
            )));
        }
        let stream = gpu.stream();
        let mut arms: Vec<(usize, (bool, bool))> = Vec::new();
        for s in stores.iter().flat_map(Store::sites) {
            let k = s.k();
            if k == d.embd {
                continue;
            }
            let (q128, q32) = WideActs::wants(s.ty());
            match arms.iter_mut().find(|a| a.0 == k) {
                Some(a) => a.1 = (a.1.0 | q128, a.1.1 | q32),
                None => arms.push((k, (q128, q32))),
            }
        }
        let arms = arms
            .into_iter()
            .map(|(k, forms)| Ok((k, WideActs::new(stream, forms, T_MAX, k)?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        let (qk, kv) = (d.kda.shape.n_k * HEAD, d.kda.shape.n_v * HEAD);
        let sets = group_sets(group);
        Ok(Extra {
            group,
            units: (1..sets)
                .map(|_| UnitBufs::new(gpu, d.embd, d.kv_heads))
                .collect::<Result<_, _>>()?,
            stages: (0..sets).map(|_| Stage::new(d.embd, d.kv_heads)).collect(),
            hsum: DeviceBuffer::zeroed(stream, T_MAX * d.embd)?,
            gemm: GemmKernels::load(gpu.context())?,
            route: GemmRoute::new(stream, T_MAX, 1)?,
            flash: FlashGqaPrefill::load(gpu.context())?,
            stage: [
                DeviceBuffer::zeroed(stream, T_MAX * qk)?,
                DeviceBuffer::zeroed(stream, T_MAX * qk)?,
                DeviceBuffer::zeroed(stream, T_MAX * kv)?,
            ],
            arms,
        })
    }

    /// Set the group to `group` units a walk: the units it lacks are made
    /// here, between calls. Returns whether it made any.
    pub fn set_group(&mut self, gpu: &Gpu, d: &Dims, group: usize) -> Result<bool, GpuError> {
        check_group(group, GROUP_MAX).map_err(|r| GpuError::Shape {
            what: GROUP_LEVER,
            detail: r.to_string(),
        })?;
        let sets = group_sets(group);
        let have = self.units.len() + 1;
        for _ in have..sets {
            self.units.push(UnitBufs::new(gpu, d.embd, d.kv_heads)?);
            self.stages.push(Stage::new(d.embd, d.kv_heads));
        }
        self.group = group;
        Ok(sets > have)
    }

    /// The device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.units.iter().map(UnitBufs::bytes).sum::<usize>()
            + self.hsum.num_bytes()
            + self.route.bytes()
            + self
                .stage
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.arms.iter().map(|a| a.1.bytes()).sum::<usize>()
    }
}

/// The host tier's batch sets and the host union's slabs for units of
/// [`T_MAX`] columns, made once so that a timed prompt allocates nothing.
/// Load-time only.
pub fn prepare_host(gpu: &Gpu, hybrid: &mut Hybrid<HostRun>) -> Result<(), GpuError> {
    hybrid.prepare_batch(gpu.context(), T_MAX)?;
    hybrid.host_mut().prepare_union(T_MAX)
}

/// Set the prompt group to `g` units a walk: a group's units run their layers
/// together, so the card work of one unit overlaps the previous one's host
/// serve; 1 runs each unit alone, and every `g` writes the same bits.
/// Refused by name for a `g` outside 1 to the lever's most. When the buffers
/// hold fewer units than `g` needs
/// ([`group_sets`]), the units it lacks are made here, between calls.
/// Returns whether it made any.
pub fn set_prefill_group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let (extra, dims) = extra_dims(body);
    extra.set_group(gpu, dims, g)
}

/// Feed `ids` from position `at` in units of up to [`T_MAX`] positions and
/// enqueue the head after the last: the pass
/// [`GpuModel::run_rows`](bloomery_gpu::GpuModel::run_rows) runs, returning
/// whether the head was enqueued (always, on success). Refused by name
/// before anything runs: for no id and past the stores' positions. With the
/// taps armed, each layer's output at the call's last position is copied to its
/// tap. A group waits for its launches and reads the fault word once
/// at its end, so a fault ends the call before the next group; the last
/// group's word rides the head's readback.
pub fn prefill(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    ids: &[u32],
    at: u32,
) -> Result<bool, GpuError> {
    let mut p = parts(body);
    let to = call_end(at, ids.len(), p.dims.ctx).map_err(|r| shape(r.to_string()))?;
    let units = call_batches(at, to, &[to], T_MAX);
    for gr in groups(units.len(), p.extra.group) {
        let runs = &units[gr];
        let (Some(first), Some(end)) = (runs.first(), runs.last()) else {
            continue;
        };
        let seg = &ids[(first.start - at) as usize..(end.end - at) as usize];
        enqueue_group(gpu, w, &mut p, head, seg, runs, end.end == to)?;
    }
    Ok(true)
}

/// A unit's own buffers as the walk borrows them: the first unit's are the
/// batch's [`Scratch`], the others' [`Extra`]'s.
struct Unit<'a> {
    x: &'a mut DeviceBuffer<f32>,
    x1: &'a mut DeviceBuffer<f32>,
    pos: &'a mut DeviceBuffer<u32>,
    n_keys: &'a mut DeviceBuffer<u32>,
    words: &'a mut DeviceBuffer<u32>,
}

/// One group of `ids` over the consecutive units `runs`: each unit's rows and
/// positions on the card, one walk `(units, T, Batch)` over every layer, and
/// the head after the last unit when `last`.
fn enqueue_group(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    head: &mut Head,
    ids: &[u32],
    runs: &[Range<u32>],
    last: bool,
) -> Result<(), GpuError> {
    let g = runs.len();
    let ts: Vec<usize> = runs.iter().map(|r| (r.end - r.start) as usize).collect();
    if g == 0
        || g > p.extra.units.len() + 1
        || ts.iter().any(|&t| t == 0 || t > T_MAX)
        || runs.windows(2).any(|w| w[0].end != w[1].start)
        || ts.iter().sum::<usize>() != ids.len()
    {
        return Err(shape(format!(
            "a group of units {runs:?} over {} ids on buffers of {} units: 1 to {T_MAX} \
             positions a unit, consecutive",
            ids.len(),
            p.extra.units.len() + 1
        )));
    }
    let stream = gpu.stream();
    let (n, ctx) = (p.dims.embd, p.dims.ctx);
    let Scratch {
        x,
        x1,
        xn,
        pos,
        words,
        n_keys,
        kda,
        gqa,
        delta,
        ..
    } = &mut p.batch.rows;
    let Extra {
        units: more,
        stages,
        hsum,
        gemm,
        route,
        flash,
        stage,
        arms,
        ..
    } = &mut *p.extra;
    let mut units = vec![Unit {
        x,
        x1,
        pos,
        n_keys,
        words,
    }];
    units.extend(more.iter_mut().map(|u| Unit {
        x: &mut u.x,
        x1: &mut u.x1,
        pos: &mut u.pos,
        n_keys: &mut u.n_keys,
        words: &mut u.words,
    }));
    units.truncate(g);
    let mut from = 0;
    for ((unit, stage), (run, &t)) in units
        .iter_mut()
        .zip(stages.iter_mut())
        .zip(runs.iter().zip(&ts))
    {
        for (&token, row) in ids[from..from + t]
            .iter()
            .zip(stage.rows[..t * n].chunks_exact_mut(n))
        {
            p.embed.row_into(token, row)?;
        }
        stage.set(run, ctx)?;
        span_mut(WHAT, &mut *unit.x, 0, t * n)?.copy_from_host(stream, &stage.rows[..t * n])?;
        span_mut(WHAT, &mut *unit.pos, 0, t)?.copy_from_host(stream, &stage.pos[..t])?;
        span_mut(WHAT, &mut *unit.n_keys, 0, t)?.copy_from_host(stream, &stage.keys[..t])?;
        unit.words.copy_from_host(stream, &stage.words)?;
        from += t;
    }
    let end = runs.last().map_or(0, |r| r.end);
    let mut walk = Walk {
        gpu,
        w,
        d: p.dims,
        cfg: p.cfg,
        stores: &mut *p.stores,
        k: p.k,
        ffn: &mut *p.ffn,
        units,
        ts: &ts,
        xn,
        kda,
        gqa,
        delta,
        wide: &mut p.batch.wide,
        gemm,
        route,
        flash,
        stage,
        arms,
        taps: p.taps.as_deref_mut(),
    };
    let o = Overlap {
        units: g,
        cols: ts.iter().copied().max().unwrap_or(1),
        port: PortKind::Batch,
    };
    let mut leg = BatchLeg::new(stream, &mut *p.hybrid, hsum, T_MAX);
    leg.set_unit_cols(&ts);
    sched::walk(o, p.cfg.len(), &mut leg, &mut walk)?;
    if last {
        walk.head(head)?;
    }
    read_fault(WHAT, gpu, p.hybrid, last)?;
    *p.at = (*p.at).max(end);
    Ok(())
}

/// One group's walk: the body's parts, the shared buffers and each unit's
/// own.
struct Walk<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    d: &'a Dims,
    cfg: &'a [LayerCfg],
    stores: &'a mut [Store],
    k: &'a Kernels,
    ffn: &'a mut Routed,
    units: Vec<Unit<'a>>,
    ts: &'a [usize],
    xn: &'a mut DeviceBuffer<f32>,
    kda: &'a mut [DeviceBuffer<f32>; 11],
    gqa: &'a mut [DeviceBuffer<f32>; 7],
    delta: &'a mut DeviceBuffer<f32>,
    wide: &'a mut WideActs,
    gemm: &'a GemmKernels,
    route: &'a mut GemmRoute,
    flash: &'a FlashGqaPrefill,
    stage: &'a mut [DeviceBuffer<f32>; 3],
    arms: &'a mut [(usize, WideActs)],
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
}

impl Walk<'_> {
    /// The dense one-expert table filled for the `t` rows of a part.
    fn fill_route(&mut self, t: usize) -> Result<(), GpuError> {
        self.gemm
            .enqueue_route_dense(self.gpu.stream(), t, self.route, self.gpu.unlabelled_sink())
    }

    /// GQA layer `l` over unit `u`: `x` in, the residual `x1` out.
    fn gqa(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        let t = self.ts[u];
        let Store::Gqa(layer) = &mut self.stores[l] else {
            return Err(other_store(l, "a GQA layer's"));
        };
        let cx = GqaCx {
            what: WHAT,
            gpu: self.gpu,
            w: self.w,
            kgemv: &self.k.kgemv,
            g32: &self.k.g32,
            flash: &self.k.flash,
            // The step flash's score pass; the rows pass launches the prefill flash.
            mma: true,
            out_gate: Some(&self.k.linear.out_gate),
            hidden: self.d.embd,
            eps: self.d.rms_eps,
            n_kv: self.d.kv_heads,
            ctx: self.d.ctx,
        };
        let unit = &mut self.units[u];
        let [q, k_rows, v, g, attn, part_v, part_ms] = &mut *self.gqa;
        let bufs = GqaBufs {
            words: &*unit.words,
            row_word: WORD_ROWS,
            n_keys: &*unit.n_keys,
            xn: &mut *self.xn,
            q,
            k: k_rows,
            v,
            g: Some(g),
            attn,
            part_v,
            part_ms,
            delta: &mut *self.delta,
        };
        let wide = GqaWide {
            gemm: self.gemm,
            prefill: self.flash,
            route: &*self.route,
            xn: &mut *self.wide,
            o: arm_at(&mut *self.arms, self.d.heads * HEAD)?,
        };
        gqa_rows::<GROUP>(&cx, (l, layer), t, (bufs, wide), (&*unit.x, &mut *unit.x1))
    }

    /// KDA layer `l` over unit `u`: the mixer over `x`, then the residual
    /// add into `x1`.
    fn kda(&mut self, l: usize, u: usize) -> Result<(), GpuError> {
        let (gpu, w, t) = (self.gpu, self.w, self.ts[u]);
        let Store::Kda {
            sites,
            state,
            stamp,
            ring,
        } = &mut self.stores[l]
        else {
            return Err(other_store(l, "a KDA layer's"));
        };
        let vec = |i: usize| w.f32_buf(WHAT, &sites.vecs[i]);
        let vecs = KdaVecs {
            norm: vec(0)?,
            conv: vec(1)?,
            dt_bias: vec(2)?,
            a: vec(3)?,
            gate_norm: vec(4)?,
        };
        let unit = &mut self.units[u];
        let [qkv, fa, ga, beta_raw, f, z, conv, beta, decay, o, gated] = &mut *self.kda;
        let st = KdaState {
            state,
            stamp,
            ring,
            lane: &self.k.lane,
            lanes: 1,
            each: false,
            row: 0,
        };
        let mut proj = BatchSites {
            wk: WideKernels {
                gemm: self.gemm,
                g32: Some(&self.k.g32),
            },
            w,
            route: &*self.route,
            sites,
            x_forms: &mut *self.wide,
            arms: &mut *self.arms,
            stage: &mut *self.stage,
            kda: &self.d.kda,
        };
        let bufs = KdaBufs {
            x: &*unit.x,
            pos: &*unit.pos,
            xn: &mut *self.xn,
            qkv,
            fa,
            ga,
            beta_raw,
            f,
            z,
            conv,
            beta,
            decay,
            o,
            gated,
            out: &mut *self.delta,
        };
        mixer_kda::enqueue(
            gpu,
            gpu.stream(),
            &self.k.linear,
            &mut proj,
            (l, t, &self.d.kda),
            &vecs,
            st,
            bufs,
        )?;
        gpu.elem().enqueue_add(
            gpu.stream(),
            &*unit.x,
            &*self.delta,
            t * self.d.embd,
            &mut *unit.x1,
        )
    }

    /// After the last layer: the last unit's last row into the head's input,
    /// and the head.
    fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let n = self.d.embd;
        let (Some(unit), Some(&t)) = (self.units.last(), self.ts.last()) else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a unit of the group for the head",
            });
        };
        let row = span(WHAT, &*unit.x, (t - 1) * n, n)?;
        head.input_mut()
            .copy_from_device_async(&row, self.gpu.stream())?;
        head.enqueue(self.gpu, self.w)
    }
}

/// The wide forms of input width `k` beside the model width's.
fn arm_at(arms: &mut [(usize, WideActs)], k: usize) -> Result<&mut WideActs, GpuError> {
    arms.iter_mut()
        .find(|a| a.0 == k)
        .map(|a| &mut a.1)
        .ok_or_else(|| shape(format!("no wide forms of width {k}")))
}

/// Layer `l`'s store is not the kind its description names.
fn other_store(l: usize, want: &str) -> GpuError {
    shape(format!("layer {l}: the store is not {want} store"))
}

impl<'a> LayerProgram for Walk<'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// Every layer carries a routed block.
    fn host_leg(&self, at: At) -> bool {
        self.cfg.get(at.layer).is_some_and(|c| c.kind.host_leg())
    }

    /// The mixer, then the routed block up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let (l, u) = (at.layer, at.unit);
        let t = self.ts[u];
        self.fill_route(t)?;
        match self.cfg[l].kind.mixer {
            MixerKind::Gqa => self.gqa(l, u)?,
            MixerKind::DeltaRule => self.kda(l, u)?,
            _ => return Err(shape(format!("layer {l}: a mixer solar2 has none of"))),
        }
        let gw = (self.gpu, self.w);
        self.ffn.front_rows(gw, (l, t), &*self.units[u].x1)?;
        let key = port.key(at);
        self.ffn.download(port.hybrid(), self.gpu.stream(), key)
    }

    /// The shared expert and the card experts under the host leg.
    fn shadow(&mut self, _port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let gw = (self.gpu, self.w);
        let lm = (at.layer, self.ts[at.unit]);
        self.ffn.shadow_rows(gw, lm, &*self.units[at.unit].x1)
    }

    /// The host sums joined with the shadow's, into the unit's `x`, and the
    /// layer's tap where a gate armed them.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let n = self.d.embd;
        let unit = &mut self.units[at.unit];
        let t = self.ts[at.unit];
        self.ffn
            .back_rows(self.gpu, (at.layer, t), (port.hsum(), &mut *unit.x))?;
        // The call's last row is the last unit's last: a later group, if any,
        // overwrites what this one copied.
        if at.unit + 1 != self.ts.len() {
            return Ok(());
        }
        let Some(taps) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let tap = taps.get_mut(at.layer).ok_or(GpuError::State {
            what: WHAT,
            missing: "the layer's tap",
        })?;
        let row = span(WHAT, &*unit.x, (t - 1) * n, n)?;
        tap.copy_from_device_async(&row, self.gpu.stream())?;
        Ok(())
    }
}

/// The KDA projections over a unit's rows: the batch arm of
/// `crate::kda::Sites`. One q8_1 form of `xn` serves the six input sites,
/// and each low-rank half is quantized into the form its Q5_0 site reads.
struct BatchSites<'a> {
    wk: WideKernels<'a>,
    w: &'a Weights,
    /// The dense one-expert table, filled for the unit's rows.
    route: &'a GemmRoute,
    sites: &'a KdaSites,
    /// The forms at the model width: the input sites read them.
    x_forms: &'a mut WideActs,
    /// The forms at the other widths ([`Extra`]).
    arms: &'a mut [(usize, WideActs)],
    /// The qkv sites' outputs before their copy ([`Extra`]).
    stage: &'a mut [DeviceBuffer<f32>; 3],
    kda: &'a KdaDims,
}

impl<'a> BatchSites<'a> {
    /// The projections' launch context over `m` rows; it borrows nothing of
    /// `self` but the weights and kernels it lends.
    fn cx<'g>(&self, gpu: &'g Gpu, m: usize) -> WideCx<'g>
    where
        'a: 'g,
    {
        WideCx {
            what: WHAT,
            gpu,
            w: self.w,
            wk: WideKernels {
                gemm: self.wk.gemm,
                g32: self.wk.g32,
            },
            m,
        }
    }
}

impl KdaProj for BatchSites<'_> {
    /// q·k·v, `f_a`, `g_a` and β's raw rows of `xn`, then `f = f_b · fa` and
    /// `z = g_b · ga`.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        let (lin, s) = (self.kda.shape, self.sites);
        let (qk, kv) = (lin.n_k * HEAD, lin.n_v * HEAD);
        let (f, channels) = (size_of::<f32>(), lin.channels());
        let cx = self.cx(gpu, m);
        // One quantization of `xn` serves the six sites that read it. The step
        // writes q, k and v into the joined rows' three windows; a GEMM writes a
        // site's rows contiguously, so each lands in its stage and is copied to
        // its window.
        let [q, k, v] = &s.qkv;
        let [sq, sk, sv] = &mut *self.stage;
        let low = s.f_b.k();
        project_wide(
            &cx,
            (&mut *self.x_forms, self.route),
            l,
            xn,
            &mut [
                (q, qk, sq),
                (k, qk, sk),
                (v, kv, sv),
                (&s.f_a, low, &mut *y.fa),
                (&s.g_a, low, &mut *y.ga),
                (&s.beta, lin.n_v, &mut *y.beta_raw),
            ],
        )?;
        let mut at = 0;
        for (stage, rows) in self.stage.iter().zip([qk, qk, kv]) {
            let window = Pitched {
                width: rows * f,
                rows: m,
                dev_pitch: rows * f,
                host_pitch: channels * f,
            };
            dtod_2d(gpu.stream(), (stage, 0), (&mut *y.qkv, at * f), window)?;
            at += rows;
        }
        for (site, x, out) in [(&s.f_b, &*y.fa, y.f), (&s.g_b, &*y.ga, y.z)] {
            let forms = arm_at(&mut *self.arms, site.k())?;
            project_wide(&cx, (forms, self.route), l, x, &mut [(site, kv, out)])?;
        }
        Ok(())
    }

    /// `out = o_proj · gated`.
    fn output(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let site = &self.sites.out;
        let cx = self.cx(gpu, m);
        let forms = if site.k() == self.kda.embd {
            &mut *self.x_forms
        } else {
            arm_at(&mut *self.arms, site.k())?
        };
        project_wide(
            &cx,
            (forms, self.route),
            l,
            gated,
            &mut [(site, self.kda.embd, out)],
        )
    }
}
