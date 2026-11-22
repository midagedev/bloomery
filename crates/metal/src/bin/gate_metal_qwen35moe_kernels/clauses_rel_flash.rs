//! The rel-bias flash's clauses (`kernels/rel_flash.metal`): one decode step's
//! attention against its f64 oracle at the Mini's geometry (24 query heads over
//! 8 key heads of 128 values, 16 relative dims, a 1024-column bias table), over
//! a one-pass depth, a split depth, a window, a ring shorter than the context
//! and a depth past the table's extent; the sink's two defined edges and its
//! two refused ones; a live count the flash refuses; and the launch's
//! refusals. The kernel's rule, per query head `h` and live key `j`:
//! `s_j = scale · dot(q_h, k_j) + Σ_d bias[d, live − 1 − j] · r_hd` (the
//! column of a distance at or past the extent is 0), and
//! `out = Σ_j exp(s_j − t) v_j / (1 + Σ_j exp(s_j − t))`, `t` the head's sink.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::attention_geom::{SEGMENT_CAP, segments_for_window};
use bloomery_metal::inkling2_geom::{D_REL, EXTENT, GROUP, HEAD, KV_HEADS, QUERY_HEADS};
use bloomery_metal::kernels::{RelFlashParams, rel_partials_len};
use gguf::quant::f32_to_f16_bits;

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::{ClauseRun, Harness};

struct Case {
    ctx: usize,
    rows: usize,
    keys: usize,
    window: usize,
    q: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
    r: Vec<f32>,
    bias: Vec<f32>,
    sinks: Vec<f32>,
}

/// The value of an f16 word.
fn f16(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f64::from(bits & 0x3ff);
    sign * match exp {
        0 => man * 2f64.powi(-24),
        31 => f64::NAN,
        e => (1.0 + man / 1024.0) * 2f64.powi(e - 15),
    }
}

fn case(ctx: usize, rows: usize, keys: usize, window: usize, seed: u64) -> Case {
    let plane = KV_HEADS * rows * HEAD;
    Case {
        ctx,
        rows,
        keys,
        window,
        q: ink2::draw(QUERY_HEADS * HEAD, seed, 1.6),
        k: ink2::draw(plane, seed + 1, 1.6)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect(),
        v: ink2::draw(plane, seed + 2, 1.0)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect(),
        r: ink2::draw(QUERY_HEADS * D_REL, seed + 3, 1.0),
        bias: ink2::draw(D_REL * EXTENT, seed + 4, 0.25),
        sinks: ink2::draw(QUERY_HEADS, seed + 5, 2.0),
    }
}

fn scale() -> f32 {
    (1.0 / (HEAD as f64).sqrt()) as f32
}

fn launch(h: &Harness<'_>, c: &Case) -> Result<Vec<f32>, String> {
    let window = (c.window != 0).then_some(c.window as u32);
    let segments = segments_for_window(c.keys as u32, window) as usize;
    let q = ink2::up(h, &c.q)?;
    let k = ink2::up(h, &c.k)?;
    let v = ink2::up(h, &c.v)?;
    let r = ink2::up(h, &c.r)?;
    let bias = ink2::up(h, &c.bias)?;
    let sinks = ink2::up(h, &c.sinks)?;
    let plen = rel_partials_len(segments).ok_or("partials wrap")?;
    let mut partials = ink2::up(h, &ink2::blank(plen))?;
    let mut out = ink2::up(h, &ink2::blank(QUERY_HEADS * HEAD))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.rel_flash(
            encoder,
            RelFlashParams {
                segments,
                ctx: c.ctx,
                rows: c.rows,
                keys: c.keys,
                window: c.window,
                scale: scale(),
                q: q.window(Elems(0), Elems(c.q.len()))?,
                cache_k: k.window(Elems(0), Elems(c.k.len()))?,
                cache_v: v.window(Elems(0), Elems(c.v.len()))?,
                r: r.window(Elems(0), Elems(c.r.len()))?,
                bias: bias.window(Elems(0), Elems(c.bias.len()))?,
                sinks: sinks.window(Elems(0), Elems(QUERY_HEADS))?,
                partials: partials.window_mut(Elems(0), Elems(plen))?,
                out: out.window_mut(Elems(0), Elems(QUERY_HEADS * HEAD))?,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    ink2::read(h, &out)
}

/// The oracle's output of head `head`, with its band.
fn oracle(c: &Case, head: usize) -> (Vec<f64>, Vec<f64>) {
    let kv = head / GROUP;
    let lo = if c.window != 0 && c.keys > c.window {
        c.keys - c.window
    } else {
        0
    };
    let sink = f64::from(c.sinks[head]);
    let mut logits = Vec::new();
    let mut smag = 0.0f64;
    for j in lo..c.keys {
        let row = kv * c.rows * HEAD + (j % c.rows) * HEAD;
        let (mut dot, mut mag) = (0.0f64, 0.0f64);
        for d in 0..HEAD {
            let t = f64::from(c.q[head * HEAD + d]) * f16(c.k[row + d]);
            dot += t;
            mag += t.abs();
        }
        let dist = c.keys - 1 - j;
        let (mut b, mut bmag) = (0.0f64, 0.0f64);
        if dist < EXTENT {
            for d in 0..D_REL {
                let t = f64::from(c.r[head * D_REL + d]) * f64::from(c.bias[d * EXTENT + dist]);
                b += t;
                bmag += t.abs();
            }
        }
        logits.push(dot * f64::from(scale()) + b);
        smag = smag.max(mag * f64::from(scale()) + bmag);
    }
    let top = logits.iter().copied().fold(sink, f64::max);
    let denom = (sink - top).exp() + logits.iter().map(|s| (s - top).exp()).sum::<f64>();
    let mut out = vec![0.0f64; HEAD];
    let mut mag = vec![0.0f64; HEAD];
    for (n, s) in logits.iter().enumerate() {
        let p = (s - top).exp() / denom;
        let row = kv * c.rows * HEAD + ((lo + n) % c.rows) * HEAD;
        for d in 0..HEAD {
            let v = f16(c.v[row + d]);
            out[d] += p * v;
            mag[d] += p * v.abs();
        }
    }
    // A score carries the dot's `dim/…` fused roundings, the butterfly's, the
    // scale's and the bias dot's; the softmax passes it on at most doubled; the
    // online rescale adds a rounding a tile; the value sum, a lane's fmas.
    let walk = ink2::gamma_walk(HEAD / 4 + 12 + D_REL);
    let tiles = c.keys.div_ceil(16);
    let rel = 2.0 * walk * smag + ink2::gamma_walk(tiles + 8) + 8.0 * U;
    let band = mag.iter().map(|m| rel * m + 1.0e-30).collect();
    (out, band)
}

fn fit_case(h: &Harness<'_>, what: &str, c: &Case) -> Result<f64, String> {
    let got = launch(h, c)?;
    common::expect_clean(h, what)?;
    ink2::written(what, &got)?;
    let mut worst = 0.0f64;
    for head in 0..QUERY_HEADS {
        let (want, band) = oracle(c, head);
        worst = worst.max(ink2::fit(
            &format!("{what}, head {head}"),
            &got[head * HEAD..(head + 1) * HEAD],
            &want,
            &band,
        )?);
    }
    Ok(worst)
}

/// The flash is the oracle's in band at: a short visit (one key, a tile and
/// one over), a one-pass depth, a split depth, a window over a deep context, a
/// ring shorter than the context holding a window, and a depth past the bias
/// table's extent, where the column is 0.
pub fn rel_flash_rows(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (i, (label, ctx, rows, keys, window)) in [
        ("1 key", 64, 64, 1, 0),
        ("17 keys", 64, 64, 17, 0),
        ("500 keys", 512, 512, 500, 0),
        ("1000 keys", 1024, 1024, 1000, 0),
        ("window 600 of 1500", 2048, 2048, 1500, 600),
        ("ring 256 window 200", 1024, 256, 700, 200),
        ("1100 keys past the extent", 2048, 2048, 1100, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let c = case(ctx, rows, keys, window, 0x4601 + 16 * i as u64);
        let worst = fit_case(h, label, &c)?;
        parts.push(format!("{label} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "the attention in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A sink of `-inf` weighs nothing and is neutral; a sink that dwarfs every
/// score leaves a row of near zero; a NaN or `+inf` sink raises the product
/// site and its head's row is NaN alone.
pub fn rel_flash_sink(h: &Harness<'_>) -> Clause {
    let mut c = case(512, 512, 300, 0, 0x4701);
    c.sinks[4] = f32::NEG_INFINITY;
    let got = launch(h, &c)?;
    common::expect_clean(h, "a -inf sink")?;
    let (want, band) = oracle(&c, 4);
    ink2::fit("a -inf sink", &got[4 * HEAD..5 * HEAD], &want, &band)?;
    let clean = {
        let mut d = case(512, 512, 300, 0, 0x4701);
        d.sinks[4] = 0.0;
        launch(h, &d)?
    };
    for head in (0..QUERY_HEADS).filter(|h| *h != 4) {
        ink2::same(
            &format!("head {head} beside a changed sink"),
            &got[head * HEAD..(head + 1) * HEAD],
            &clean[head * HEAD..(head + 1) * HEAD],
        )?;
    }
    for (what, sink) in [("a NaN sink", f32::NAN), ("a +inf sink", f32::INFINITY)] {
        let mut d = case(512, 512, 300, 0, 0x4701);
        d.sinks[11] = sink;
        let got = launch(h, &d)?;
        common::expect_fault(h, what, FaultSite::F32Product)?;
        for head in 0..QUERY_HEADS {
            let row = &got[head * HEAD..(head + 1) * HEAD];
            if head == 11 {
                if !row.iter().all(|v| v.is_nan()) {
                    return Err(format!("{what} left a row value that is a number"));
                }
            } else {
                ink2::same(
                    &format!("head {head} beside {what}"),
                    row,
                    &clean[head * HEAD..(head + 1) * HEAD],
                )
                .or_else(|_| {
                    let (want, band) = oracle(&d, head);
                    ink2::fit(&format!("head {head} beside {what}"), row, &want, &band).map(|_| ())
                })?;
            }
        }
    }
    Ok(Verdict::Noted(
        "a -inf sink is neutral in band; a NaN or +inf sink raises the product site and is NaN in its head alone".to_string(),
    ))
}

/// A live count of zero or past the context raises the key-count site, and
/// every row is NaN: no plausible attention over nothing.
pub fn rel_flash_keys(h: &Harness<'_>) -> Clause {
    for (what, keys) in [("no keys", 0usize), ("keys past the context", 513)] {
        let c = case(512, 512, keys, 0, 0x4801);
        let got = launch(h, &c)?;
        common::expect_fault(h, what, FaultSite::KeyCount)?;
        if !got.iter().all(|v| v.is_nan()) {
            return Err(format!("{what} left a row value that is a number"));
        }
    }
    Ok(Verdict::Noted(
        "zero keys and keys past the context raise the key-count site and the rows are NaN"
            .to_string(),
    ))
}

/// The launch's refusals by name, nothing encoded.
pub fn rel_flash_bounds(h: &Harness<'_>) -> Clause {
    let c = case(64, 64, 10, 0, 0x4901);
    let q = ink2::up(h, &c.q)?;
    let k = ink2::up(h, &c.k)?;
    let v = ink2::up(h, &c.v)?;
    let r = ink2::up(h, &c.r)?;
    let bias = ink2::up(h, &c.bias)?;
    let sinks = ink2::up(h, &c.sinks)?;
    let cap_len = rel_partials_len(SEGMENT_CAP).ok_or("partials wrap")?;
    let mut partials = ink2::up(h, &ink2::blank(cap_len))?;
    let mut out = ink2::up(h, &ink2::blank(QUERY_HEADS * HEAD))?;
    let plane = KV_HEADS * 64 * HEAD;
    let mut run = |label: &str,
                   segments: usize,
                   rows: usize,
                   scale: f32,
                   k_len: usize,
                   bias_len: usize,
                   p_len: usize,
                   needle: &str|
     -> Result<(), String> {
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.rel_flash(
                    encoder,
                    RelFlashParams {
                        segments,
                        ctx: 64,
                        rows,
                        keys: 10,
                        window: 0,
                        scale,
                        q: q.window(Elems(0), Elems(QUERY_HEADS * HEAD))?,
                        cache_k: k.window(Elems(0), Elems(k_len))?,
                        cache_v: v.window(Elems(0), Elems(plane))?,
                        r: r.window(Elems(0), Elems(QUERY_HEADS * D_REL))?,
                        bias: bias.window(Elems(0), Elems(bias_len))?,
                        sinks: sinks.window(Elems(0), Elems(QUERY_HEADS))?,
                        partials: partials.window_mut(Elems(0), Elems(p_len))?,
                        out: out.window_mut(Elems(0), Elems(QUERY_HEADS * HEAD))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let s = scale();
    let p1 = rel_partials_len(1).ok_or("partials wrap")?;
    run(
        "no segments",
        0,
        64,
        s,
        plane,
        D_REL * EXTENT,
        p1,
        "segments",
    )?;
    run(
        "segments past the cap",
        SEGMENT_CAP + 1,
        64,
        s,
        plane,
        D_REL * EXTENT,
        cap_len,
        "segments",
    )?;
    run("no rows", 1, 0, s, plane, D_REL * EXTENT, p1, "rows")?;
    run(
        "a zero scale",
        1,
        64,
        0.0,
        plane,
        D_REL * EXTENT,
        p1,
        "scale",
    )?;
    run(
        "a short key plane",
        1,
        64,
        s,
        plane - 1,
        D_REL * EXTENT,
        p1,
        "key plane",
    )?;
    run(
        "a short bias table",
        1,
        64,
        s,
        plane,
        D_REL * EXTENT - 1,
        p1,
        "bias",
    )?;
    run(
        "short partials",
        1,
        64,
        s,
        plane,
        D_REL * EXTENT,
        p1 - 1,
        "partials",
    )?;
    Ok(Verdict::Noted(
        "seven refusals by name, each before an encode".to_string(),
    ))
}

/// The flash's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        (
            "rel_flash_rows",
            Box::new(|| rel_flash_rows(h)) as ClauseRun,
        ),
        ("rel_flash_sink", Box::new(|| rel_flash_sink(h))),
        ("rel_flash_keys", Box::new(|| rel_flash_keys(h))),
        ("rel_flash_bounds", Box::new(|| rel_flash_bounds(h))),
    ]
}
