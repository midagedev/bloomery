//! One token's chain on the card (the crate doc's steps 2 to 5): every layer's engram site, its
//! attention and feed-forward sub-layers between the identity mix's pre and post, then the head's
//! fold of the streams into the output head's input, and the head. The four streams ping-pong
//! between two buffers: every write goes to the other one, so a launch never reads what it writes.
//! Every position-dependent value is a device word or row the refresh wrote, so the chain is the
//! same launches at every position and one capture replays at any of them.

use bloomery_gpu::arch::hybrid::{Acts, GqaCx};
use bloomery_gpu::card_leg::{LegRun, LegStacks, LegSum};
use bloomery_gpu::engram_delta::{DeltaArgs, GateOnlyArgs, KeyNormArgs, RowsArgs};
use bloomery_gpu::flash::{FlashGeom, FlashSelArgs, FlashWindowArgs};
use bloomery_gpu::hc_ident::{HC, HeadArgs, PostArgs, PreArgs};
use bloomery_gpu::head::Head;
use bloomery_gpu::kpool::{TailScoreArgs, weights_scale};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::latent::{INDEX_HEAD, IndexKeyRmsArgs, IndexPoolRopeArgs, IndexProj, Rows};
use bloomery_gpu::linear::out_gate::OutGateArgs;
use bloomery_gpu::model::Q8_0GemvHeadsArgs;
use bloomery_gpu::qsa::TopkTailArgs;
use bloomery_gpu::routed_block::{RouteIn, RouterEntry};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{FaultSink, Gpu, GpuError, copy_row};
use model::arch::hy4::names::{self, Sub};
use model::arch::models::OutGate;

use crate::body::{
    Body, ENGRAM_EPS, ENGRAM_W, Ffn, Glu, LATENT, ROPE, ROW, Router, Scratch, Sparse, Store,
    TAP_STAGES, WHAT, cx,
};

/// The tap slots of [`TAP_STAGES`].
pub(crate) const ENGRAM_OUT: usize = 0;
pub(crate) const ATTN_NORM: usize = 1;
pub(crate) const ATTN_OUT: usize = 2;
pub(crate) const HC_ATTN_OUT: usize = 3;
pub(crate) const FFN_NORM: usize = 4;
pub(crate) const FFN_OUT: usize = 5;
pub(crate) const L_OUT: usize = 6;

/// Enqueue the step's chain of `body` over `w`, the head last: every layer on the card, or on a
/// placed load through the host tier's step port ([`crate::placed::walk`]).
pub(crate) fn enqueue(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    // The streams enter in buffer 0: the refresh laid the embedding there.
    let mut cur = 0usize;
    match body.placed.take() {
        None => {
            for l in 0..body.layers.len() {
                let fault = gpu.layer_sink(l)?;
                attn_half(gpu, w, body, l, &mut cur, fault)?;
                ffn(gpu, w, body, l, fault)?;
                close(gpu, body, l, &mut cur)?;
            }
        }
        Some(mut placed) => {
            let walked = crate::placed::walk((gpu, w), body, &mut placed, &mut cur);
            body.placed = Some(placed);
            walked?;
        }
    }
    let sh = body.sh;
    body.k.hc.enqueue_head(
        gpu.stream(),
        HeadArgs {
            x: &body.s.hs[cur],
            w: w.f32_buf(WHAT, names::OUTPUT_HC_FN)?,
            scale: w.f32_buf(WHAT, names::OUTPUT_HC_SCALE)?,
            base: w.f32_buf(WHAT, names::OUTPUT_HC_BASE)?,
            eps_norm: sh.eps,
            eps_hc: sh.hc_eps,
            n: sh.hidden,
            m: 1,
            y: head.input_mut(),
            fault: gpu.unlabelled_sink(),
        },
    )?;
    if let Some(t) = body.taps.as_mut() {
        copy_row(gpu.stream(), head.input(), &mut t.head, sh.hidden)?;
    }
    head.enqueue(gpu, w)
}

/// Layer `l` from the streams in `*cur` to the feed-forward block's normed input `s.xn`: the
/// engram site, the attention sub-layer between the identity mix's pre and post, the feed-forward
/// pre and norm.
pub(crate) fn attn_half(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    cur: &mut usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    if body.layers[l].engram.is_some() {
        engram(gpu, w, body, l, cur, fault)?;
        tap(gpu, body, l, ENGRAM_OUT, Src::Streams(*cur))?;
    }
    pre(gpu, w, body, (l, Sub::Attn), *cur, fault)?;
    norm(gpu, w, body, &names::attn_norm(l))?;
    tap(gpu, body, l, ATTN_NORM, Src::Xn)?;
    attention(gpu, w, body, l, fault)?;
    tap(gpu, body, l, ATTN_OUT, Src::Out)?;
    post(gpu, body, cur)?;
    tap(gpu, body, l, HC_ATTN_OUT, Src::Streams(*cur))?;
    pre(gpu, w, body, (l, Sub::Ffn), *cur, fault)?;
    norm(gpu, w, body, &names::ffn_norm(l))?;
    tap(gpu, body, l, FFN_NORM, Src::Xn)
}

/// Layer `l`'s close: the feed-forward output `s.out` through the identity mix's post.
pub(crate) fn close(gpu: &Gpu, body: &mut Body, l: usize, cur: &mut usize) -> Result<(), GpuError> {
    tap(gpu, body, l, FFN_OUT, Src::Out)?;
    post(gpu, body, cur)?;
    tap(gpu, body, l, L_OUT, Src::Streams(*cur))
}

/// What a tap copies.
enum Src {
    /// The four streams in buffer `i`.
    Streams(usize),
    /// The normed row a sub-layer reads.
    Xn,
    /// A sub-layer's output after its sandwich norm.
    Out,
}

fn tap(gpu: &Gpu, body: &mut Body, l: usize, stage: usize, src: Src) -> Result<(), GpuError> {
    let Some(t) = body.taps.as_mut() else {
        return Ok(());
    };
    let n = body.sh.hidden;
    let dst = &mut t.stages[l * TAP_STAGES.len() + stage];
    let (from, len) = match src {
        Src::Streams(i) => (&body.s.hs[i], HC * n),
        Src::Xn => (&body.s.xn, n),
        Src::Out => (&body.s.out, n),
    };
    copy_row(gpu.stream(), from, dst, len)
}

/// The identity mix's pre of sub-layer `sub` of layer `l` over the streams in buffer `cur`: the
/// gates into `s.mix`, the fold into `s.y`.
pub(crate) fn pre(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    (l, sub): (usize, Sub),
    cur: usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let sh = body.sh;
    let s = &mut body.s;
    body.k.hc.enqueue_pre(
        gpu.stream(),
        PreArgs {
            x: &s.hs[cur],
            w: w.f32_buf(WHAT, &names::hc_fn(l, sub))?,
            scale: w.f32_buf(WHAT, &names::hc_scale(l, sub))?,
            base: w.f32_buf(WHAT, &names::hc_base(l, sub))?,
            eps_norm: sh.eps,
            eps_hc: sh.hc_eps,
            magnitude: sh.hc_mag,
            n: sh.hidden,
            m: 1,
            mix: &mut s.mix,
            y: &mut s.y,
            fault,
        },
    )
}

/// The RMS norm of the fold `s.y` by `gain` into `s.xn`.
pub(crate) fn norm(gpu: &Gpu, w: &Weights, body: &mut Body, gain: &str) -> Result<(), GpuError> {
    let sh = body.sh;
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        &body.s.y,
        w.f32_buf(WHAT, gain)?,
        sh.eps,
        sh.hidden,
        1,
        &mut body.s.xn,
    )
}

/// The identity mix's post: the streams in `*cur` plus the sub-layer's output `s.out` times the
/// mix's post gates, into the other buffer, which becomes `*cur`.
pub(crate) fn post(gpu: &Gpu, body: &mut Body, cur: &mut usize) -> Result<(), GpuError> {
    let n = body.sh.hidden;
    let s = &mut body.s;
    let [a, b] = &mut s.hs;
    let (res, out) = if *cur == 0 { (&*a, b) } else { (&*b, a) };
    body.k.hc.enqueue_post(
        gpu.stream(),
        PostArgs {
            x: &s.out,
            res,
            mix: &s.mix,
            n,
            m: 1,
            out,
        },
    )?;
    *cur ^= 1;
    Ok(())
}

/// Layer `l`'s engram site over the streams in `*cur` (llama.cpp `build_engram`): the refresh's
/// gathered rows normed per row and scaled by their column's lookup norm and the mask, the fused
/// key and value projection and its bias, the per-stream gate, the masked delta into the other
/// buffer, which becomes `*cur`.
pub(crate) fn engram(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    cur: &mut usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        words,
        host,
        ..
    } = body;
    let Some((e, kv_site)) = layers[l].engram.as_ref() else {
        return Ok(());
    };
    let cx = cx(gpu, w, k, sh);
    let cols = host.cols;
    let (rows, valid) = (&words.e_rows[*e], &words.e_valid[*e]);
    k.engram.enqueue_rows_norm(
        stream,
        RowsArgs {
            rows,
            lookup: w.f32_buf(WHAT, &names::engram_lookup_norm(l))?,
            valid,
            eps: ENGRAM_EPS,
            width: ENGRAM_W,
            cols,
            m: 1,
            out: &mut s.e_norm,
            fault,
        },
    )?;
    kv_site.project(&cx, acts, l, &s.e_norm, &mut s.e_kv)?;
    k.engram.enqueue_kv_bias(
        stream,
        w.f32_buf(WHAT, &names::engram_kv_b(l))?,
        1,
        &mut s.e_kv,
    )?;
    k.engram.gate.enqueue_key_norm(
        stream,
        KeyNormArgs {
            kv: &s.e_kv,
            gain: &s.e_ones,
            eps: ENGRAM_EPS,
            hc: HC,
            m: 1,
            kn: &mut s.e_kn,
            fault,
        },
    )?;
    let [a, b] = &mut s.hs;
    let (x, out) = if *cur == 0 { (&*a, b) } else { (&*b, a) };
    k.engram.gate.enqueue_gate(
        stream,
        GateOnlyArgs {
            x,
            kn: &s.e_kn,
            gain: w.f32_buf(WHAT, &names::engram_gate(l))?,
            eps: ENGRAM_EPS,
            hc: HC,
            m: 1,
            gate: &mut s.e_gate,
            fault,
        },
    )?;
    k.engram.enqueue_delta(
        stream,
        DeltaArgs {
            x,
            kv: &s.e_kv,
            gate: &s.e_gate,
            gamma: w.f32_buf(WHAT, &names::engram_pre_gamma(l))?,
            hc_scale: w.f32_buf(WHAT, &names::engram_hc_scale(l))?,
            valid,
            eps: ENGRAM_EPS,
            cols,
            m: 1,
            out,
            fault,
        },
    )?;
    *cur ^= 1;
    Ok(())
}

/// Layer `l`'s gated latent attention over the normed row `s.xn` into `s.out` (llama.cpp
/// `build_attention`, then the sandwich norm).
pub(crate) fn attention(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    attention_keys(gpu, w, body, l, fault)?;
    attention_read(gpu, w, body, l, fault)
}

/// [`attention`]'s first half: the query, the key's append and, on a sparse layer, the
/// selector's list.
pub(crate) fn attention_keys(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        words,
        stores,
        rope_rows,
        ..
    } = body;
    let sh = *sh;
    let cx = cx(gpu, w, k, &sh);
    let a = &layers[l];
    let heads = a.heads;
    // 1. The query: q_a, its norm (the q residual), q_b; every head's rotated tail turned at the
    //    layer's base (the rope runs over each 64-value run of the head, the gather keeps the
    //    tail's).
    a.q_a.project(&cx, acts, l, &s.xn, &mut s.qa)?;
    elem.enqueue_rms_norm(
        stream,
        &s.qa,
        w.f32_buf(WHAT, &names::attn_q_a_norm(l))?,
        sh.eps,
        sh.q_lora,
        1,
        &mut s.qr,
    )?;
    a.q_b.project(&cx, acts, l, &s.qr, &mut s.q)?;
    let cs = if a.dsa { &words.cs_dsa } else { &words.cs_swa };
    let runs = u32::try_from(heads * sh.head_k / ROPE)
        .map_err(|_| GpuError::shape(WHAT, "a query past u32 rope runs"))?;
    elem.enqueue_rope(stream, &s.q, cs, ROPE, runs, 1, &mut s.q_rope)?;
    // 2. The key: the latent and the rotated part, normed, turned and appended as one f16 row
    //    `[k_rope | latent]`, at the position on a sparse layer and at its window slot on a
    //    window layer.
    a.kv_a.project(&cx, acts, l, &s.xn, &mut s.kv_a)?;
    let kv_gain = w.f32_buf(WHAT, &names::attn_kv_a_norm(l))?;
    let (at, cache) = match &mut stores[l] {
        Store::Dsa(d) => (&words.pos, &mut d.cache),
        Store::Swa { ring } => (&words.ring_pos, ring),
    };
    k.fused.enqueue_kv_norm_rope_append(
        stream,
        &s.kv_a,
        kv_gain,
        cs,
        at,
        sh.eps,
        LATENT,
        ROPE,
        &mut s.kv_s,
        &mut s.kvr,
        cache,
        fault,
    )?;
    // 3. A sparse layer's k-pool selector: the token key's index row, the pool it completes, the
    //    scores of the closed pools and the list of the kept pools' rows and its own pool's.
    if let (Some(ix), Store::Dsa(d)) = (a.idx.as_ref(), &mut stores[l]) {
        let Sparse {
            index,
            pooled,
            scores,
            list,
            n_sel,
            ..
        } = &mut **d;
        ix.k.project(&cx, acts, l, &s.xn, &mut s.ik)?;
        let (vq, vd) = w.q8_planes(WHAT, &names::indexer_compressor_value(l))?;
        let (gq, gd) = w.q8_planes(WHAT, &names::indexer_compressor_gate(l))?;
        k.latent.enqueue_index_key_rms_append(
            stream,
            IndexKeyRmsArgs {
                rows: Rows {
                    x: &s.ik,
                    stride: INDEX_HEAD,
                    m: 1,
                },
                k_off: 0,
                gain: w.f32_buf(WHAT, &names::indexer_k_norm(l))?,
                value: IndexProj { qs: vq, d: vd },
                gate: IndexProj { qs: gq, d: gd },
                pos: &words.pos,
                eps: sh.eps,
                ctx: sh.ctx,
                fault,
                cache: index,
            },
        )?;
        k.latent.enqueue_index_pool_norm_rope(
            stream,
            IndexPoolRopeArgs {
                cache: index,
                ape: w.f32_buf(WHAT, &names::indexer_compressor_ape(l))?,
                gain: w.f32_buf(WHAT, &names::indexer_compressor_norm(l))?,
                table: &rope_rows.table,
                n_keys: &words.n_keys,
                eps: sh.eps,
                ctx: sh.ctx,
                m: 1,
                fault,
                pooled,
            },
        )?;
        ix.q_b.project(&cx, acts, l, &s.qr, &mut s.iq)?;
        ix.proj.project(&cx, acts, l, &s.xn, &mut s.wi)?;
        k.kpool.enqueue_score_tail(
            stream,
            TailScoreArgs {
                q: &s.iq,
                w: &s.wi,
                n_keys: &words.n_keys,
                table: &rope_rows.table,
                pooled,
                heads: sh.idx_heads,
                tokens: 1,
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
                n_keys: &words.n_keys,
                scores,
                ctx: sh.ctx,
                kept: sh.kept,
                width: sh.top_k,
                m: 1,
                fault,
                list,
                n_sel,
            },
        )?;
    }
    Ok(())
}

/// [`attention`]'s second half: the flash over the list or the ring, the values, the output gate,
/// the output projection and the sandwich norm.
pub(crate) fn attention_read(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        words,
        stores,
        ..
    } = body;
    let sh = *sh;
    let cx = cx(gpu, w, k, &sh);
    let a = &layers[l];
    let heads = a.heads;
    // 4. The flash's query a head: `[q_rope | k_b · q_nope]`.
    let (qs, d) = w.q8_planes(WHAT, &names::attn_k_b(l))?;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d,
            x: &s.q,
            rows_per_head: LATENT,
            x_head_stride: sh.head_k,
            y_head_stride: ROW,
            y_off: ROPE,
            y: &mut s.q_cur,
        },
    )?;
    k.step.enqueue_gather(
        stream,
        &s.q_rope,
        &s.rope_src,
        &s.rope_dst,
        heads * ROPE,
        &mut s.q_cur,
    )?;
    // 5. The latent attention with a sink a head: over the selector's list on a sparse layer,
    //    over the window ring on a window layer.
    let sinks = w.f32_buf(WHAT, &names::attn_sinks(l))?;
    let geom = FlashGeom {
        tokens: 1,
        heads,
        rope_dims: ROPE,
        latent_dims: LATENT,
    };
    match &stores[l] {
        Store::Dsa(d) => k.flash.enqueue_flash_latent_sel_sink(
            stream,
            FlashSelArgs {
                q: &s.q_cur,
                kv: &d.cache,
                list: &d.list,
                counts: &d.n_sel,
                stride: sh.top_k,
                kq_scale: sh.kq_scale(),
                geom,
                part_v: &mut s.part_v,
                part_ms: &mut s.part_ms,
                fault,
            },
            sinks,
            &mut s.att,
        )?,
        Store::Swa { ring } => k.flash.enqueue_flash_window(
            stream,
            FlashWindowArgs {
                q: &s.q_cur,
                ring,
                staged: None,
                pos: &words.pos,
                sinks,
                kq_scale: sh.kq_scale(),
                geom,
                part_v: &mut s.part_v,
                part_ms: &mut s.part_ms,
                y: &mut s.att,
            },
        )?,
    }
    // 6. Each head's values, the output gate, the output projection and its sandwich norm.
    let (qs, d) = w.q8_planes(WHAT, &names::attn_v_b(l))?;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d,
            x: &s.att,
            rows_per_head: sh.head_v,
            x_head_stride: LATENT,
            y_head_stride: sh.head_v,
            y_off: 0,
            y: &mut s.av,
        },
    )?;
    a.gate.project(&cx, acts, l, &s.xn, &mut s.g)?;
    k.out_gate.enqueue_out_gate_head(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: &s.av,
            gate: &s.g,
            n_head: heads,
            m: 1,
            fault,
            y: &mut s.gated,
        },
        sh.head_v,
    )?;
    a.wo.project(&cx, acts, l, &s.gated, &mut s.o)?;
    elem.enqueue_rms_norm(
        stream,
        &s.o,
        w.f32_buf(WHAT, &names::post_attention_norm(l))?,
        sh.eps,
        sh.hidden,
        1,
        &mut s.out,
    )
}

/// The SiLU-gated block `g` over the normed row `s.xn` into `s.shared` (`shared`) or `s.f`.
pub(crate) fn glu(
    gpu: &Gpu,
    (cx, acts): (&GqaCx<'_>, &mut Acts),
    l: usize,
    g: &Glu,
    s: &mut Scratch,
    shared: bool,
) -> Result<(), GpuError> {
    g.gate.project(cx, acts, l, &s.xn, &mut s.fg)?;
    g.up.project(cx, acts, l, &s.xn, &mut s.fu)?;
    gpu.elem()
        .enqueue_swiglu(gpu.stream(), &s.fg, &s.fu, g.ff, &mut s.fa)?;
    let y = if shared { &mut s.shared } else { &mut s.f };
    g.down.project(cx, acts, l, &s.fa, y)
}

/// Layer `l`'s feed-forward block over the normed row `s.xn` into `s.out`: the dense SiLU block,
/// or the routed sum through the latent pair plus the shared SiLU expert; then the sandwich norm.
pub(crate) fn ffn(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    fault: FaultSink,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let elem = gpu.elem();
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        routes,
        ..
    } = body;
    let sh = *sh;
    let cx = cx(gpu, w, k, &sh);
    match &layers[l].ffn {
        Ffn::Dense(g) => glu(gpu, (&cx, acts), l, g, s, false)?,
        Ffn::Routed {
            down_lat,
            up_lat,
            stacks,
            shared,
            clamp,
        } => {
            let route = routes[l].as_mut().ok_or(GpuError::State {
                what: WHAT,
                missing: "a routed layer's router output",
            })?;
            let r = RouteIn {
                w: w.f32_tensor(WHAT, &names::ffn_gate_inp(l))?,
                bias: Some(w.f32_buf(WHAT, &names::exp_probs_b(l))?),
                scale: sh.weights_scale,
                fault,
            };
            k.router.route(stream, &s.xn, &r, route)?;
            down_lat.project(&cx, acts, l, &s.xn, &mut s.lat)?;
            let st = LegStacks::of(w, stacks.each_ref().map(String::as_str))?;
            let run = LegRun {
                layer: l,
                stacks: &st,
                normed: &s.lat,
                m: 1,
                sel: Router::out_ids(route),
                n_card: Router::N_EXPERT,
                rule: Act::SwigluGateClamp { limit: *clamp },
            };
            k.leg.enqueue(
                gpu,
                &mut s.leg,
                &run,
                Some(LegSum {
                    w: Router::out_weights(route),
                    acc: &mut s.routed,
                }),
            )?;
            up_lat.project(&cx, acts, l, &s.routed, &mut s.lat_up)?;
            glu(gpu, (&cx, acts), l, shared, s, true)?;
            elem.enqueue_add(stream, &s.lat_up, &s.shared, sh.hidden, &mut s.f)?;
        }
    }
    ffn_norm(gpu, w, body, l)
}

/// Layer `l`'s feed-forward sandwich norm: the block's output `s.f` by `post_ffw_norm` into
/// `s.out`.
pub(crate) fn ffn_norm(gpu: &Gpu, w: &Weights, body: &mut Body, l: usize) -> Result<(), GpuError> {
    let sh = body.sh;
    let s = &mut body.s;
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        &s.f,
        w.f32_buf(WHAT, &names::post_ffw_norm(l))?,
        sh.eps,
        sh.hidden,
        1,
        &mut s.out,
    )
}
