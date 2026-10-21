//! The prompt batch: a prompt of `P` ids fed in units of at most
//! [`T_MAX`](crate::body::T_MAX) positions ([`prefill`]), which leaves the
//! model where `P` decode steps over the same ids leave it in everything a
//! later step reads: every layer's recurrent state and conv ring, every GQA
//! layer's key and value rows at those positions, and the last position's
//! logits.
//!
//! The walk is the common owner's ([`prompt_walk`]): the units of a group, the
//! host staging of their rows, the routed block over the unit's slots, the
//! head and the fault word. This file holds what Solar-Open2's architecture
//! decides: the mixers over the unit's rows (the NoPE GQA layer with its out
//! gate, and the KDA mixer), their intermediates, the wide forms their
//! projections read, and the GQA word record.
//!
//! Each layer runs over the whole unit, one launch a site a layer (not a token
//! or a chunk): the KDA mixer is [`mixer_kda::enqueue`] at the unit's `m` rows
//! over [`BatchSites`] (the GEMM arm of the typed sites), the GQA layer is
//! [`gqa_rows`], and the routed block is [`Routed`]'s row parts over the same
//! rows.
//!
//! A group's units share every buffer one part of one item writes and the
//! same or the next part of that item reads (the mixers' intermediates, the
//! routed block's rows, the activation forms, the host sums); what a unit's
//! back, or its later layers, read after the next unit's front ran is the
//! unit's own (the layer's input and output rows, the residual after the
//! mixer, the positions and the live key counts). All 48 layers carry a
//! routed block, so no dense layer sits behind a routed one.

use bloomery_gpu::arch::hybrid::{GqaBufs, GqaCx, GqaWide, WideCx, gqa_rows, project_wide};
use bloomery_gpu::flash_gqa::HEAD;
use bloomery_gpu::flash_gqa_prefill::FlashGqaPrefill;
use bloomery_gpu::gemm::{GemmKernels, GemmRoute};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::linear::mixer_kda::{self, KdaBufs, KdaDims, KdaIn, KdaProj, KdaState, KdaVecs};
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::site::{WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, Pitched, dtod_2d};
use bloomery_gpu_deepseek41::prompt_walk::{self, Batch, BatchAttention, BatchBufs, UnitRows};
use cuda_core::DeviceBuffer;
use model::embed::EmbedTable;
use runtime::layer::{FfnKind, MixerKind};

use crate::body::{Body, Dims, KdaSites, LayerCfg, MixerBufs, Store, T_MAX, WORD_POS, WORD_ROWS};
use crate::ffn::Routed;
use crate::step::Kernels;

/// What this file's refusals name.
const WHAT: &str = "solar2 prefill";

/// Query heads a key head serves: the GQA layers' group.
const GROUP: usize = 8;

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
struct Parts<'a> {
    hybrid: &'a mut Hybrid<HostRun>,
    dims: &'a Dims,
    cfg: &'a [LayerCfg],
    stores: &'a mut [Store],
    /// The mixers' rows: the normed rows and every mixer's intermediate.
    rows: &'a mut MixerBufs,
    /// The wide forms the projections read at the model width.
    wide: &'a mut WideActs,
    extra: &'a mut Extra,
    /// The units of a group: the walker's own buffers.
    units: &'a mut BatchBufs,
    k: &'a Kernels,
    embed: &'a EmbedTable,
    ffn: &'a mut Routed,
    /// The body's highest stored position, raised when a call succeeds.
    at: &'a mut u32,
    /// Each layer's tap, when a gate armed them: the layer's output at the
    /// last position of the call.
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
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
        units,
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
        rows: &mut batch.rows,
        wide: &mut batch.wide,
        extra: prefill,
        units,
        k,
        embed: &embd.table,
        ffn,
        at,
    }
}

/// A unit's GQA word record: the unit's first position and the key heads' row
/// bases, on the host and on the card.
struct Word {
    host: Vec<u32>,
    dev: DeviceBuffer<u32>,
}

impl Word {
    /// The record of a unit over `n_kv` key heads. Load-time only.
    fn new(gpu: &Gpu, n_kv: usize) -> Result<Word, GpuError> {
        Ok(Word {
            host: vec![0; WORD_ROWS + n_kv],
            dev: DeviceBuffer::zeroed(gpu.stream(), WORD_ROWS + n_kv)?,
        })
    }

    /// The device bytes of the record.
    fn bytes(&self) -> usize {
        self.dev.num_bytes()
    }
}

/// What the prompt batch holds beyond the mixers' rows and the wide forms:
/// the GQA word record of every unit, the grouped GEMM's and the prefill
/// flash's kernels, the dense one-expert table, the wide forms of the widths
/// the batch's `wide` does not hold and the qkv window's staging. Made once,
/// so that a timed prompt allocates nothing.
pub struct Extra {
    /// The GQA word record of every unit of a group.
    words: Vec<Word>,
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
    /// Refused by name for a load whose heads are not [`GROUP`] a key head.
    /// Load-time only.
    pub fn new(gpu: &Gpu, d: &Dims, stores: &[Store], group: usize) -> Result<Extra, GpuError> {
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
        let sets = prompt_walk::group_sets(group);
        Ok(Extra {
            words: (0..sets)
                .map(|_| Word::new(gpu, d.kv_heads))
                .collect::<Result<_, _>>()?,
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
        let sets = prompt_walk::group_sets(group);
        let have = self.words.len();
        for _ in have..sets {
            self.words.push(Word::new(gpu, d.kv_heads)?);
        }
        Ok(sets > have)
    }

    /// The device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.words.iter().map(Word::bytes).sum::<usize>()
            + self.route.bytes()
            + self
                .stage
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.arms.iter().map(|a| a.1.bytes()).sum::<usize>()
    }
}

/// Set the prompt group to `g` units a walk: a group's units run their layers
/// together, so the card work of one unit overlaps the previous one's host
/// serve; 1 runs each unit alone, and every `g` writes the same bits.
/// Refused by name for a `g` outside 1 to the lever's most. When the buffers
/// hold fewer units than `g` needs
/// ([`prompt_walk::group_sets`]), the units it lacks are made here, between
/// calls. Returns whether it made any.
pub fn set_prefill_group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    prompt_walk::check_group(g, prompt_walk::GROUP_MAX).map_err(|r| GpuError::Shape {
        what: prompt_walk::GROUP_LEVER,
        detail: r.to_string(),
    })?;
    let (extra, units, dims) = (&mut body.prefill, &mut body.units, &body.dims);
    let words = extra.set_group(gpu, dims, g)?;
    let bufs = units.set_group(gpu, dims.embd, g)?;
    Ok(words || bufs)
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
    let p = parts(body);
    let mut attention = Attention {
        gpu,
        w,
        d: p.dims,
        cfg: p.cfg,
        stores: p.stores,
        k: p.k,
        rows: p.rows,
        wide: p.wide,
        gemm: &p.extra.gemm,
        route: &mut p.extra.route,
        flash: &p.extra.flash,
        stage: &mut p.extra.stage,
        arms: &mut p.extra.arms,
        words: &mut p.extra.words,
    };
    let mut batch = Batch {
        gpu,
        w,
        bufs: p.units,
        attention: &mut attention,
        ffn: p.ffn,
        hybrid: p.hybrid,
        embed: p.embed,
        head,
        taps: p.taps,
        embd: p.dims.embd,
        ctx: p.dims.ctx,
        layers: p.cfg.len(),
    };
    let mut at = at;
    batch.prompt(ids, &mut at)?;
    *p.at = (*p.at).max(at);
    Ok(true)
}

/// The mixers over a unit's rows: what Solar-Open2's architecture decides, as
/// the walk's attention.
struct Attention<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    d: &'a Dims,
    cfg: &'a [LayerCfg],
    stores: &'a mut [Store],
    k: &'a Kernels,
    /// The normed rows and every mixer's intermediate.
    rows: &'a mut MixerBufs,
    /// The wide forms at the model width.
    wide: &'a mut WideActs,
    gemm: &'a GemmKernels,
    route: &'a mut GemmRoute,
    flash: &'a FlashGqaPrefill,
    stage: &'a mut [DeviceBuffer<f32>; 3],
    arms: &'a mut [(usize, WideActs)],
    /// The GQA word record of every unit of a group.
    words: &'a mut [Word],
}

impl Attention<'_> {
    /// The dense one-expert table filled for the `t` rows of a part.
    fn fill_route(&mut self, t: usize) -> Result<(), GpuError> {
        self.gemm
            .enqueue_route_dense(self.gpu.stream(), t, self.route, self.gpu.unlabelled_sink())
    }

    /// GQA layer `l` over unit `u`: `x` in, the residual `x1` out.
    fn gqa(&mut self, l: usize, u: usize, t: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
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
        let words = &self.words[u].dev;
        let MixerBufs { xn, gqa, delta, .. } = &mut *self.rows;
        let [q, k_rows, v, g, attn, part_v, part_ms] = &mut *gqa;
        let bufs = GqaBufs {
            words,
            row_word: WORD_ROWS,
            n_keys: rows.n_keys,
            xn,
            q,
            k: k_rows,
            v,
            g: Some(g),
            attn,
            part_v,
            part_ms,
            delta,
        };
        let wide = GqaWide {
            gemm: self.gemm,
            prefill: self.flash,
            route: &*self.route,
            xn: &mut *self.wide,
            o: arm_at(&mut *self.arms, self.d.heads * HEAD)?,
        };
        gqa_rows::<GROUP>(&cx, (l, layer), t, (bufs, wide), (rows.x, rows.x1))
    }

    /// KDA layer `l` over unit `u`: the mixer over `x`, then the residual
    /// add into `x1`.
    fn kda(&mut self, l: usize, _u: usize, t: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
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
        let MixerBufs { xn, kda, delta, .. } = &mut *self.rows;
        let [qkv, fa, ga, beta_raw, f, z, conv, beta, decay, o, gated] = &mut *kda;
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
            x: rows.x,
            pos: rows.pos,
            xn,
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
            out: delta,
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
        gpu.elem()
            .enqueue_add(gpu.stream(), rows.x, delta, t * self.d.embd, rows.x1)
    }
}

impl BatchAttention for Attention<'_> {
    /// The unit's GQA word record: its first position and the key heads' row
    /// bases over the stores' positions.
    fn stage(&mut self, u: usize, at: u32, _m: usize) -> Result<(), GpuError> {
        let word = &mut self.words[u];
        word.host[WORD_POS] = at;
        for (h, w) in word.host[WORD_ROWS..].iter_mut().enumerate() {
            *w = h
                .checked_mul(self.d.ctx)
                .and_then(|r| r.checked_add(at as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    shape(format!(
                        "row {h} · {} + {at} does not fit a u32 row word",
                        self.d.ctx
                    ))
                })?;
        }
        word.dev.copy_from_host(self.gpu.stream(), &word.host)?;
        Ok(())
    }

    /// The mixers by the layer's kind, over the unit's rows.
    fn attend(&mut self, l: usize, u: usize, m: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
        self.fill_route(m)?;
        match self.cfg[l].kind.mixer {
            MixerKind::Gqa => self.gqa(l, u, m, rows),
            MixerKind::DeltaRule => self.kda(l, u, m, rows),
            _ => Err(shape(format!("layer {l}: a mixer solar2 has none of"))),
        }
    }

    /// Every layer carries a routed block.
    fn ffn(&self, _l: usize) -> FfnKind {
        FfnKind::Moe
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
