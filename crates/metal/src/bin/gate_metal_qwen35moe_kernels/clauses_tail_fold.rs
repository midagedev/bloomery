//! The folded tails' clauses: an Inkling 2 decode visit's elementwise tails
//! run in the launch that makes their input (`kernels/conv_tail.metal`, the
//! tails of `qk_clip.metal` and `router_situ.metal`). Each folded launch is held
//! to the program it replaces, bit for bit: every buffer it writes and the fault
//! word, over the ring's reach (a position before, at and past the taps, a ring
//! that wraps, a plane that wraps under a position the ring does not), a slot
//! stride wider than the row, the windowed and windowless convs, an input that
//! is not a number, and the launches' refusals.
//!
//! The programs are the unfolded entries, which the kernels keep for this:
//!
//! - `combine_conv_join`: `combine_rows`, `conv_res`, `bias_rows`;
//! - `conv_res_join`: `conv_res`, `bias_rows`;
//! - `qk_clip_conv_append`: `conv_res` of the key and of the value rows,
//!   `qk_clip_append`;
//! - `router_sigmoid_norm`: `norm_rows`, `router_sigmoid`.
//!
//! Their own clauses hold each entry to its f64 oracle; these hold the fold to
//! the entry, so a fold cannot move a bit the entries pinned.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::{CONV_CHANNELS, EXPERTS, HEAD, HIDDEN, KV_HEADS, TAPS, TOP_K};
use bloomery_metal::kernels::{
    BiasRowsParams, CombineConvJoinParams, CombineParams, ConvResJoinParams, ConvResParams,
    KvConvParams, NormRowsParams, QkClipConvParams, QkClipParams, RouterSigmoidNormParams,
    RouterSigmoidParams, TailConv,
};

use super::clauses_qk_clip::{
    CTX, PLANE_FILL, QK_THRESH, R_THRESH, Rows, V_EPS, rows as clip_rows,
};
use super::common::{self, LAYER};
use super::ink2;
use super::{ClauseRun, Harness};

/// The attention post-conv's first channel in the visit's ring, and the
/// feed-forward post-conv's.
const BASE_ATTN: usize = 2 * KV_HEADS * HEAD;
const BASE_FFN: usize = BASE_ATTN + HIDDEN;
/// The value conv's first channel in the ring; the key conv's is 0.
const BASE_V: usize = KV_HEADS * HEAD;
const SCALE: f32 = 2.5;
const EPS: f32 = 1.0e-6;

/// What a program left behind: the words of every buffer it writes, in a fixed
/// order, and the fault word's three fields.
struct Seen {
    words: Vec<(&'static str, Vec<u32>)>,
    fault: Option<(u32, u32, u32)>,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn seen(h: &Harness<'_>, words: Vec<(&'static str, Vec<u32>)>) -> Result<Seen, String> {
    Ok(Seen {
        words,
        fault: h.fault()?.map(|f| (f.layer, f.code, f.sites)),
    })
}

/// The fault word `sites` raised in [`LAYER`] leave: the smallest code and
/// the mask; none for no site.
fn fault_of(sites: &[FaultSite]) -> Option<(u32, u32, u32)> {
    let code = sites.iter().map(|s| *s as u32).min()?;
    let mask = sites.iter().fold(0u32, |m, s| m | 1u32 << (*s as u32));
    Some((LAYER, code, mask))
}

/// The folded launch left the unfolded program's words and its fault word, and
/// that fault word is `sites`.
fn equal(what: &str, folded: &Seen, plain: &Seen, sites: &[FaultSite]) -> Result<(), String> {
    for ((name, a), (_, b)) in folded.words.iter().zip(&plain.words) {
        if let Some((i, x, y)) = common::first_diff(a, b) {
            return Err(format!(
                "{what}: {name} word {i} is {x:#010x}, the unfolded program's {y:#010x}"
            ));
        }
    }
    if folded.fault != plain.fault {
        return Err(format!(
            "{what}: the fault word is {:?}, the unfolded program's {:?}",
            folded.fault, plain.fault
        ));
    }
    if plain.fault != fault_of(sites) {
        return Err(format!(
            "{what}: the unfolded program raised {:?}, the case is for {:?}",
            plain.fault,
            fault_of(sites)
        ));
    }
    Ok(())
}

/// Exactly `width` words of `out` are NaN: the bad channels, written as NaN.
fn nan_channels(what: &str, out: &[u32], width: usize) -> Result<(), String> {
    let nan = out.iter().filter(|w| f32::from_bits(**w).is_nan()).count();
    if nan == width {
        Ok(())
    } else {
        Err(format!(
            "{what}: {nan} conv outputs are NaN, the bad channels are {width}"
        ))
    }
}

fn garbage(len: usize, seed: u64) -> Vec<f32> {
    ink2::draw(len, seed, 50.0)
        .iter()
        .map(|v| 100.0 + v)
        .collect()
}

// ------------------------------------------------------------- the post-convs

/// A visit's post-conv inputs: the slots' rows (or the attention output), the
/// row the sum is added into, the filter, the ring and the stream row.
struct Tail {
    /// `TOP_K` rows `stride` apart for the combine; the conv's input row for the join.
    y: Vec<f32>,
    stride: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    ring: Vec<f32>,
    stream: Vec<f32>,
    rows: usize,
    pos: u32,
}

fn tail(seed: u64, rows: usize, pos: u32, stride: usize) -> Tail {
    Tail {
        y: ink2::draw(TOP_K * stride, seed, 1.5),
        stride,
        x: ink2::draw(HIDDEN, seed + 1, 2.0),
        w: ink2::draw(HIDDEN * TAPS, seed + 2, 0.6),
        ring: garbage(rows * CONV_CHANNELS, seed + 3),
        stream: ink2::draw(HIDDEN, seed + 4, 3.0),
        rows,
        pos,
    }
}

/// The experts' sum, the feed-forward's post-conv and the stream's add: the
/// folded launch, or the three launches it replaces.
fn run_combine(h: &Harness<'_>, t: &Tail, folded: bool) -> Result<Seen, String> {
    let y = ink2::up(h, &t.y)?;
    let inv = ink2::up(h, &(0..TOP_K as u32).collect::<Vec<_>>())?;
    let mut x = ink2::up(h, &t.x)?;
    let w = ink2::up(h, &t.w)?;
    let mut ring = ink2::up(h, &t.ring)?;
    let mut out = ink2::up(h, &ink2::blank(HIDDEN))?;
    let mut stream = ink2::up(h, &t.stream)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        if folded {
            h.kernels.combine_conv_join(
                encoder,
                CombineConvJoinParams {
                    cols: HIDDEN,
                    slots: TOP_K,
                    y: y.window(Elems(0), Elems(t.y.len()))?,
                    y_stride: t.stride,
                    x: x.window_mut(Elems(0), Elems(HIDDEN))?,
                    conv: TailConv {
                        w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                        taps: TAPS,
                        ring: ring.window_mut(Elems(0), Elems(t.rows * CONV_CHANNELS))?,
                        ring_rows: t.rows,
                        ch: CONV_CHANNELS,
                        base: BASE_FFN,
                        pos: t.pos,
                    },
                    conv_out: out.window_mut(Elems(0), Elems(HIDDEN))?,
                    stream: stream.window_mut(Elems(0), Elems(HIDDEN))?,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        } else {
            h.kernels.combine_rows(
                encoder,
                CombineParams {
                    rows: 1,
                    per_row: TOP_K,
                    cols: HIDDEN,
                    y: y.window(Elems(0), Elems(t.y.len()))?,
                    y_rows: TOP_K,
                    y_stride: t.stride,
                    inv: inv.window(Elems(0), Elems(TOP_K))?,
                    x: x.window_mut(Elems(0), Elems(HIDDEN))?,
                    x_stride: HIDDEN,
                },
            )?;
            encoder.memory_barrier();
            h.kernels.conv_res(
                encoder,
                ConvResParams {
                    d: HIDDEN,
                    taps: TAPS,
                    heads: 0,
                    rows: t.rows,
                    ch: CONV_CHANNELS,
                    base: BASE_FFN,
                    m: 1,
                    pos: t.pos,
                    x: x.window(Elems(0), Elems(HIDDEN))?,
                    w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                    dyn_taps: None,
                    y: out.window_mut(Elems(0), Elems(HIDDEN))?,
                    ring: ring.window_mut(Elems(0), Elems(t.rows * CONV_CHANNELS))?,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                },
            )?;
            encoder.memory_barrier();
            h.kernels.bias_rows(
                encoder,
                BiasRowsParams {
                    rows: 1,
                    cols: HIDDEN,
                    y: stream.window_mut(Elems(0), Elems(HIDDEN))?,
                    y_stride: HIDDEN,
                    bias: out.window(Elems(0), Elems(HIDDEN))?,
                },
            )
        }
    })?;
    seen(
        h,
        vec![
            ("the visit's row", bits(&ink2::read(h, &x)?)),
            ("the conv's output", bits(&ink2::read(h, &out)?)),
            ("the ring", bits(&ink2::read(h, &ring)?)),
            ("the stream", bits(&ink2::read(h, &stream)?)),
        ],
    )
}

/// The attention's post-conv and the stream's add: the folded launch, or the
/// two launches it replaces. `t.y` is the conv's input row.
fn run_join(h: &Harness<'_>, t: &Tail, folded: bool) -> Result<Seen, String> {
    let ao = ink2::up(h, &t.y[..HIDDEN])?;
    let w = ink2::up(h, &t.w)?;
    let mut ring = ink2::up(h, &t.ring)?;
    let mut ac = ink2::up(h, &ink2::blank(HIDDEN))?;
    let mut stream = ink2::up(h, &t.stream)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        if folded {
            h.kernels.conv_res_join(
                encoder,
                ConvResJoinParams {
                    cols: HIDDEN,
                    x: ao.window(Elems(0), Elems(HIDDEN))?,
                    conv: TailConv {
                        w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                        taps: TAPS,
                        ring: ring.window_mut(Elems(0), Elems(t.rows * CONV_CHANNELS))?,
                        ring_rows: t.rows,
                        ch: CONV_CHANNELS,
                        base: BASE_ATTN,
                        pos: t.pos,
                    },
                    y: ac.window_mut(Elems(0), Elems(HIDDEN))?,
                    stream: stream.window_mut(Elems(0), Elems(HIDDEN))?,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        } else {
            h.kernels.conv_res(
                encoder,
                ConvResParams {
                    d: HIDDEN,
                    taps: TAPS,
                    heads: 0,
                    rows: t.rows,
                    ch: CONV_CHANNELS,
                    base: BASE_ATTN,
                    m: 1,
                    pos: t.pos,
                    x: ao.window(Elems(0), Elems(HIDDEN))?,
                    w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                    dyn_taps: None,
                    y: ac.window_mut(Elems(0), Elems(HIDDEN))?,
                    ring: ring.window_mut(Elems(0), Elems(t.rows * CONV_CHANNELS))?,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                },
            )?;
            encoder.memory_barrier();
            h.kernels.bias_rows(
                encoder,
                BiasRowsParams {
                    rows: 1,
                    cols: HIDDEN,
                    y: stream.window_mut(Elems(0), Elems(HIDDEN))?,
                    y_stride: HIDDEN,
                    bias: ac.window(Elems(0), Elems(HIDDEN))?,
                },
            )
        }
    })?;
    seen(
        h,
        vec![
            ("the conv's output", bits(&ink2::read(h, &ac)?)),
            ("the ring", bits(&ink2::read(h, &ring)?)),
            ("the stream", bits(&ink2::read(h, &stream)?)),
        ],
    )
}

/// The ring's rows, the token's position and the slots' stride of each case:
/// a position before the taps, at them and past them, a ring that wraps and
/// one longer than the reach, a stride a row's width and one padded.
const CASES: [(usize, u32, usize); 8] = [
    (TAPS - 1, 0, HIDDEN),
    (TAPS - 1, 1, HIDDEN + 16),
    (TAPS - 1, 2, HIDDEN),
    (TAPS - 1, 3, HIDDEN + 16),
    (TAPS - 1, 11, HIDDEN),
    (TAPS + 1, 11, HIDDEN + 16),
    (TAPS + 1, 4, HIDDEN),
    (TAPS - 1, 4_099, HIDDEN),
];

/// The folded sum, post-conv and add are the three launches' bits and fault
/// word at every case, and an input that is not a number raises the conv site
/// in both and is the same NaN.
pub fn combine_conv_join(h: &Harness<'_>) -> Clause {
    for (i, (rows, pos, stride)) in CASES.into_iter().enumerate() {
        let t = tail(0x4A00 + 16 * i as u64, rows, pos, stride);
        let (folded, plain) = (run_combine(h, &t, true)?, run_combine(h, &t, false)?);
        equal(
            &format!("ring {rows} position {pos} stride {stride}"),
            &folded,
            &plain,
            &[],
        )?;
    }
    // A slot value that is not a number is NaN in its channel's output; an
    // infinite one over sixteen channels leaves some of them infinite in the
    // conv's own arithmetic, which the launch must still write as NaN.
    for (what, poison, width) in [
        ("a NaN slot value", f32::NAN, 1),
        ("an infinite slot value", f32::INFINITY, 16),
    ] {
        let mut t = tail(0x4A80, TAPS - 1, 5, HIDDEN);
        for c in 0..width {
            t.y[3 * HIDDEN + 700 + c] = poison;
        }
        let (folded, plain) = (run_combine(h, &t, true)?, run_combine(h, &t, false)?);
        equal(what, &folded, &plain, &[FaultSite::LinearConv])?;
        nan_channels(what, &folded.words[1].1, width)?;
    }
    Ok(Verdict::Noted(format!(
        "{} cases and two non-finite inputs: the visit's row, the conv's output, the ring and the stream are the three launches' bits, the fault word the same",
        CASES.len()
    )))
}

/// The folded post-conv and add are the two launches' bits and fault word at
/// every case, and an input that is not a number the same in both.
pub fn conv_res_join(h: &Harness<'_>) -> Clause {
    for (i, (rows, pos, _)) in CASES.into_iter().enumerate() {
        let t = tail(0x4B00 + 16 * i as u64, rows, pos, HIDDEN);
        let (folded, plain) = (run_join(h, &t, true)?, run_join(h, &t, false)?);
        equal(&format!("ring {rows} position {pos}"), &folded, &plain, &[])?;
    }
    for (what, poison, width) in [
        ("a NaN input", f32::NAN, 1),
        ("an infinite input", f32::INFINITY, 16),
    ] {
        let mut t = tail(0x4B80, TAPS - 1, 5, HIDDEN);
        for c in 0..width {
            t.y[700 + c] = poison;
        }
        let (folded, plain) = (run_join(h, &t, true)?, run_join(h, &t, false)?);
        equal(what, &folded, &plain, &[FaultSite::LinearConv])?;
        nan_channels(what, &folded.words[0].1, width)?;
    }
    Ok(Verdict::Noted(format!(
        "{} cases and two non-finite inputs: the conv's output, the ring and the stream are the two launches' bits, the fault word the same",
        CASES.len()
    )))
}

// --------------------------------------------------------- the clip's tail convs

/// A key and value conv's inputs beside the clip's rows.
struct Clip {
    rows: Rows,
    wk: Vec<f32>,
    wv: Vec<f32>,
    dyn_k: Vec<f32>,
    dyn_v: Vec<f32>,
    ring: Vec<f32>,
    ring_rows: usize,
    windowed: bool,
    /// The convs' position in the sequence.
    pos: u32,
    /// The plane's append position: the sequence's, modulo the plane.
    plane_pos: u32,
}

fn clip_case(seed: u64, windowed: bool, pos: u32, plane_pos: u32, ring_rows: usize) -> Clip {
    Clip {
        rows: clip_rows(seed),
        wk: ink2::draw(KV_HEADS * HEAD * TAPS, seed + 10, 0.6),
        wv: ink2::draw(KV_HEADS * HEAD * TAPS, seed + 11, 0.6),
        dyn_k: ink2::draw(KV_HEADS * TAPS, seed + 12, 0.5),
        dyn_v: ink2::draw(KV_HEADS * TAPS, seed + 13, 0.5),
        ring: garbage(ring_rows * CONV_CHANNELS, seed + 14),
        ring_rows,
        windowed,
        pos,
        plane_pos,
    }
}

/// The clips' windows over the buffers, a launch's `QkClipParams`.
macro_rules! clip_params {
    ($h:expr, $c:expr, $q:ident, $k:ident, $v:ident, $r:ident, $gq:ident, $gr:ident, $ck:ident, $cv:ident) => {
        QkClipParams {
            q: $q.window_mut(Elems(0), Elems($c.rows.q.len()))?,
            k: $k.window_mut(Elems(0), Elems(KV_HEADS * HEAD))?,
            v: $v.window_mut(Elems(0), Elems(KV_HEADS * HEAD))?,
            r: $r.window_mut(Elems(0), Elems($c.rows.r.len()))?,
            gq: $gq.window(Elems(0), Elems($c.rows.gq.len()))?,
            gr: $gr.window(Elems(0), Elems($c.rows.gr.len()))?,
            cache_k: $ck.window_mut(Elems(0), Elems(KV_HEADS * CTX * HEAD))?,
            cache_v: $cv.window_mut(Elems(0), Elems(KV_HEADS * CTX * HEAD))?,
            ctx: CTX,
            pos: $c.plane_pos,
            v_eps: V_EPS,
            qk_thresh: QK_THRESH,
            r_thresh: R_THRESH,
            tau: if $c.windowed { 1.0 } else { 0.8125 },
            faults: $h.device.fault_word(),
            layer: LAYER,
        }
    };
}

/// A windowless visit's dynamic taps window of `buf`, none for a windowed one.
macro_rules! taps {
    ($c:expr, $buf:expr) => {
        (!$c.windowed)
            .then(|| $buf.window(Elems(0), Elems(KV_HEADS * TAPS)))
            .transpose()?
    };
}

/// The key and value convs and the clips with the append: the folded launch, or
/// the launches it replaces.
fn run_clip(h: &Harness<'_>, c: &Clip, folded: bool) -> Result<Seen, String> {
    let plane = vec![PLANE_FILL; KV_HEADS * CTX * HEAD];
    let (mut q, mut r) = (ink2::up(h, &c.rows.q)?, ink2::up(h, &c.rows.r)?);
    let (k_raw, v_raw) = (ink2::up(h, &c.rows.k)?, ink2::up(h, &c.rows.v)?);
    let (mut k, mut v) = (
        ink2::up(h, &ink2::blank(KV_HEADS * HEAD))?,
        ink2::up(h, &ink2::blank(KV_HEADS * HEAD))?,
    );
    let (gq, gr) = (ink2::up(h, &c.rows.gq)?, ink2::up(h, &c.rows.gr)?);
    let (mut ck, mut cv) = (ink2::up(h, &plane)?, ink2::up(h, &plane)?);
    let (wk, wv) = (ink2::up(h, &c.wk)?, ink2::up(h, &c.wv)?);
    let (dk, dv) = (ink2::up(h, &c.dyn_k)?, ink2::up(h, &c.dyn_v)?);
    let mut ring = ink2::up(h, &c.ring)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        if folded {
            h.kernels.qk_clip_conv_append(
                encoder,
                QkClipConvParams {
                    clip: clip_params!(h, c, q, k, v, r, gq, gr, ck, cv),
                    conv: KvConvParams {
                        k_raw: k_raw.window(Elems(0), Elems(KV_HEADS * HEAD))?,
                        v_raw: v_raw.window(Elems(0), Elems(KV_HEADS * HEAD))?,
                        wk: wk.window(Elems(0), Elems(c.wk.len()))?,
                        wv: wv.window(Elems(0), Elems(c.wv.len()))?,
                        dyn_k: taps!(c, dk),
                        dyn_v: taps!(c, dv),
                        taps: TAPS,
                        ring: ring.window_mut(Elems(0), Elems(c.ring_rows * CONV_CHANNELS))?,
                        ring_rows: c.ring_rows,
                        ch: CONV_CHANNELS,
                        base_k: 0,
                        base_v: BASE_V,
                        pos: c.pos,
                    },
                },
            )
        } else {
            for (base, raw, w, dynamic, y) in [
                (0, &k_raw, &wk, &dk, &mut k),
                (BASE_V, &v_raw, &wv, &dv, &mut v),
            ] {
                h.kernels.conv_res(
                    encoder,
                    ConvResParams {
                        d: KV_HEADS * HEAD,
                        taps: TAPS,
                        heads: if c.windowed { 0 } else { KV_HEADS },
                        rows: c.ring_rows,
                        ch: CONV_CHANNELS,
                        base,
                        m: 1,
                        pos: c.pos,
                        x: raw.window(Elems(0), Elems(KV_HEADS * HEAD))?,
                        w: w.window(Elems(0), Elems(c.wk.len()))?,
                        dyn_taps: taps!(c, dynamic),
                        y: y.window_mut(Elems(0), Elems(KV_HEADS * HEAD))?,
                        ring: ring.window_mut(Elems(0), Elems(c.ring_rows * CONV_CHANNELS))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )?;
            }
            encoder.memory_barrier();
            h.kernels
                .qk_clip_append(encoder, clip_params!(h, c, q, k, v, r, gq, gr, ck, cv))
        }
    })?;
    let words16 = |b: &bloomery_metal::MetalBuffer<u16>| -> Result<Vec<u32>, String> {
        Ok(ink2::read(h, b)?.into_iter().map(u32::from).collect())
    };
    seen(
        h,
        vec![
            ("the q rows", bits(&ink2::read(h, &q)?)),
            ("the k rows", bits(&ink2::read(h, &k)?)),
            ("the v rows", bits(&ink2::read(h, &v)?)),
            ("the r rows", bits(&ink2::read(h, &r)?)),
            ("the key plane", words16(&ck)?),
            ("the value plane", words16(&cv)?),
            ("the ring", bits(&ink2::read(h, &ring)?)),
        ],
    )
}

/// The folded convs, clips and append are the launches' bits and fault word,
/// windowed and windowless, at positions the plane wraps under (the convs'
/// position is the sequence's, the append's the plane's), and an input that is
/// not a number, a position past the plane and a value past the half range
/// leave the same words and fault word.
pub fn qk_clip_conv_append(h: &Harness<'_>) -> Clause {
    let cases = [
        (true, 0, 0, TAPS - 1),
        (false, 1, 1, TAPS - 1),
        (true, 2, 2, TAPS - 1),
        (false, 3, 3, TAPS - 1),
        (true, 5, 5, TAPS + 1),
        (false, 1_030, 1_030 % CTX as u32, TAPS - 1),
        (true, 1_031, 1_031 % CTX as u32, TAPS - 1),
        (false, 4_099, 4_099 % CTX as u32, TAPS + 1),
    ];
    for (i, (windowed, pos, plane_pos, ring_rows)) in cases.into_iter().enumerate() {
        let c = clip_case(0x4C00 + 100 * i as u64, windowed, pos, plane_pos, ring_rows);
        let (folded, plain) = (run_clip(h, &c, true)?, run_clip(h, &c, false)?);
        equal(
            &format!("windowed {windowed} position {pos} plane position {plane_pos}"),
            &folded,
            &plain,
            &[],
        )?;
    }
    // A key input that is not a number: the conv site, then the clip's.
    let mut c = clip_case(0x4D00, false, 6, 3, TAPS - 1);
    c.rows.k[700] = f32::NAN;
    let (folded, plain) = (run_clip(h, &c, true)?, run_clip(h, &c, false)?);
    equal(
        "a NaN key input",
        &folded,
        &plain,
        &[FaultSite::LinearConv, FaultSite::CacheValue],
    )?;
    // A key input that is infinite over sixteen channels: the conv writes NaN
    // for each, so the clip's norm is not a number and raises, where a conv
    // that passed an infinity on would leave the clip a zero factor.
    let mut c = clip_case(0x4D20, false, 6, 3, TAPS - 1);
    for k in &mut c.rows.k[5 * HEAD..5 * HEAD + 16] {
        *k = f32::INFINITY;
    }
    let (folded, plain) = (run_clip(h, &c, true)?, run_clip(h, &c, false)?);
    equal(
        "an infinite key input",
        &folded,
        &plain,
        &[FaultSite::LinearConv, FaultSite::CacheValue],
    )?;
    // A value input that is infinite.
    let mut c = clip_case(0x4D40, true, 6, 3, TAPS - 1);
    for v in &mut c.rows.v[HEAD..HEAD + 16] {
        *v = f32::INFINITY;
    }
    let (folded, plain) = (run_clip(h, &c, true)?, run_clip(h, &c, false)?);
    equal(
        "an infinite value input",
        &folded,
        &plain,
        &[FaultSite::LinearConv, FaultSite::CacheValue],
    )?;
    // A plane position past the plane: the convs ran, nothing is appended.
    let c = clip_case(0x4D80, false, 9, CTX as u32, TAPS - 1);
    let (folded, plain) = (run_clip(h, &c, true)?, run_clip(h, &c, false)?);
    equal(
        "a position past the plane",
        &folded,
        &plain,
        &[FaultSite::CachePos],
    )?;
    Ok(Verdict::Noted(format!(
        "{} cases and four faulting inputs: q, k, v and r rows, the planes and the ring are the launches' bits, the fault word the same",
        cases.len()
    )))
}

// -------------------------------------------------------------- the router's norm

/// The router's inputs: the row before its norm, the norm's gain, the experts'
/// rows and the selection bias.
struct Route {
    x: Vec<f32>,
    gain: Vec<f32>,
    rows: Vec<f32>,
    bias: Vec<f32>,
    top_k: usize,
}

fn route(seed: u64, amp: f32, top_k: usize) -> Route {
    Route {
        x: ink2::draw(HIDDEN, seed, amp),
        gain: ink2::draw(HIDDEN, seed + 1, 0.5)
            .iter()
            .map(|g| 1.0 + g)
            .collect(),
        rows: ink2::draw(EXPERTS * HIDDEN, seed + 2, 0.06),
        bias: ink2::draw(EXPERTS, seed + 3, 0.3),
        top_k,
    }
}

/// The norm and the router: the folded launch, or the two launches it replaces.
fn run_route(h: &Harness<'_>, c: &Route, folded: bool) -> Result<Seen, String> {
    let rows = ink2::up(h, &c.rows)?;
    let x = ink2::up(h, &c.x)?;
    let gain = ink2::up(h, &c.gain)?;
    let bias = ink2::up(h, &c.bias)?;
    let mut normed = ink2::up(h, &ink2::blank(HIDDEN))?;
    let mut partials = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ticket = ink2::up(h, &[0u32])?;
    let mut logits = ink2::up(h, &ink2::blank(EXPERTS))?;
    let mut ids = ink2::up(h, &vec![u32::MAX; c.top_k])?;
    let mut weights = ink2::up(h, &ink2::blank(c.top_k))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        macro_rules! router {
            ($x:expr) => {
                RouterSigmoidParams {
                    k: HIDDEN,
                    top_k: c.top_k,
                    rows: rows.window(Elems(0), Elems(c.rows.len()))?,
                    x: $x,
                    bias: bias.window(Elems(0), Elems(EXPERTS))?,
                    partials: partials.window_mut(Elems(0), Elems(EXPERTS))?,
                    ticket: ticket.window_mut(Elems(0), Elems(1))?,
                    logits: logits.window_mut(Elems(0), Elems(EXPERTS))?,
                    ids: ids.window_mut(Elems(0), Elems(c.top_k))?,
                    weights: weights.window_mut(Elems(0), Elems(c.top_k))?,
                    scale: SCALE,
                    faults: h.device.fault_word(),
                    layer: LAYER,
                }
            };
        }
        if folded {
            h.kernels.router_sigmoid_norm(
                encoder,
                RouterSigmoidNormParams {
                    router: router!(x.window(Elems(0), Elems(HIDDEN))?),
                    gain: gain.window(Elems(0), Elems(HIDDEN))?,
                    normed: normed.window_mut(Elems(0), Elems(HIDDEN))?,
                    eps: EPS,
                },
            )
        } else {
            h.kernels.norm_rows(
                encoder,
                NormRowsParams {
                    rows: 1,
                    cols: HIDDEN,
                    x: x.window(Elems(0), Elems(HIDDEN))?,
                    x_stride: HIDDEN,
                    gain: gain.window(Elems(0), Elems(HIDDEN))?,
                    y: normed.window_mut(Elems(0), Elems(HIDDEN))?,
                    y_stride: HIDDEN,
                    eps: EPS,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )?;
            encoder.memory_barrier();
            let in_row = normed.window(Elems(0), Elems(HIDDEN))?;
            h.kernels.router_sigmoid(encoder, router!(in_row))
        }
    })?;
    seen(
        h,
        vec![
            ("the normed row", bits(&ink2::read(h, &normed)?)),
            ("the scores", bits(&ink2::read(h, &logits)?)),
            ("the kept ids", ink2::read(h, &ids)?),
            ("the kept weights", bits(&ink2::read(h, &weights)?)),
            ("the ticket", ink2::read(h, &ticket)?),
        ],
    )
}

/// The folded norm and router are the two launches' bits and fault word: the
/// normed row, the scores, the kept ids and weights and the ticket, over a
/// plain row, a zero row (every score equal, the tie order alone), a top-k
/// under eight and a row the norm cannot take (a NaN value, a sum that
/// overflows).
pub fn router_sigmoid_norm(h: &Harness<'_>) -> Clause {
    let mut cases = 0;
    for (i, (amp, top_k)) in [(1.7f32, TOP_K), (40.0, TOP_K), (1.7, 3)]
        .into_iter()
        .enumerate()
    {
        let c = route(0x4E00 + 16 * i as u64, amp, top_k);
        let (folded, plain) = (run_route(h, &c, true)?, run_route(h, &c, false)?);
        equal(
            &format!("amplitude {amp} top-k {top_k}"),
            &folded,
            &plain,
            &[],
        )?;
        cases += 1;
    }
    let mut zero = route(0x4E80, 1.0, TOP_K);
    zero.x = vec![0.0; HIDDEN];
    zero.bias = vec![0.0; EXPERTS];
    let (folded, plain) = (run_route(h, &zero, true)?, run_route(h, &zero, false)?);
    equal("a zero row", &folded, &plain, &[])?;
    let ids = &folded.words[2].1;
    if ids != &(0..TOP_K as u32).collect::<Vec<_>>() {
        return Err(format!("a zero row kept {ids:?}, not the lowest ids"));
    }
    for (what, poison) in [
        ("a NaN value", (f32::NAN, 1.0)),
        ("a sum that overflows", (1.0e30, 1.0)),
        ("a gain that overflows a value", (3.0, 3.0e38)),
    ] {
        let mut bad = route(0x4EC0, 1.7, TOP_K);
        bad.x[5] = poison.0;
        bad.gain[5] = poison.1;
        let (folded, plain) = (run_route(h, &bad, true)?, run_route(h, &bad, false)?);
        if plain.fault.is_none() {
            return Err(format!("{what}: the unfolded program raised nothing"));
        }
        // Whatever the unfolded program raised is what the fold raised.
        let sites = [FaultSite::NormQuant, FaultSite::Router];
        let want: Vec<FaultSite> = sites
            .into_iter()
            .filter(|s| {
                plain
                    .fault
                    .is_some_and(|(_, _, mask)| mask >> (*s as u32) & 1 == 1)
            })
            .collect();
        equal(what, &folded, &plain, &want)?;
    }
    Ok(Verdict::Noted(format!(
        "{} plain cases, a zero row and three rows the norm cannot take: the normed row, scores, ids, weights and ticket are the two launches' bits, the fault word the same",
        cases
    )))
}

// -------------------------------------------------------------------- refusals

/// The shapes and windows the folded launches refuse by name, each before an
/// encode.
pub fn tail_fold_bounds(h: &Harness<'_>) -> Clause {
    let t = tail(0x4F00, TAPS - 1, 0, HIDDEN);
    let c = clip_case(0x4F40, false, 0, 0, TAPS - 1);
    let r = route(0x4F80, 1.0, TOP_K);

    let combine = |label: &str,
                   taps: usize,
                   slots: usize,
                   stride: usize,
                   ring_rows: usize,
                   base: usize,
                   x_len: usize,
                   needle: &str|
     -> Result<(), String> {
        let y = ink2::up(h, &t.y)?;
        let mut x = ink2::up(h, &t.x)?;
        let w = ink2::up(h, &t.w)?;
        let mut ring = ink2::up(h, &vec![0f32; t.rows.max(ring_rows) * CONV_CHANNELS])?;
        let mut out = ink2::up(h, &ink2::blank(HIDDEN))?;
        let mut stream = ink2::up(h, &t.stream)?;
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.combine_conv_join(
                    encoder,
                    CombineConvJoinParams {
                        cols: HIDDEN,
                        slots,
                        y: y.window(Elems(0), Elems(t.y.len()))?,
                        y_stride: stride,
                        x: x.window_mut(Elems(0), Elems(x_len))?,
                        conv: TailConv {
                            w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                            taps,
                            ring: ring.window_mut(Elems(0), Elems(ring_rows * CONV_CHANNELS))?,
                            ring_rows,
                            ch: CONV_CHANNELS,
                            base,
                            pos: 0,
                        },
                        conv_out: out.window_mut(Elems(0), Elems(HIDDEN))?,
                        stream: stream.window_mut(Elems(0), Elems(HIDDEN))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let ok = (TAPS, TOP_K, HIDDEN, TAPS - 1, BASE_FFN, HIDDEN);
    combine("three taps", 3, ok.1, ok.2, ok.3, ok.4, ok.5, "taps")?;
    combine("seven slots", ok.0, 7, ok.2, ok.3, ok.4, ok.5, "slots")?;
    combine(
        "a stride off the float4s",
        ok.0,
        ok.1,
        HIDDEN + 2,
        ok.3,
        ok.4,
        ok.5,
        "stride",
    )?;
    combine(
        "a ring under the reach",
        ok.0,
        ok.1,
        ok.2,
        TAPS - 2,
        ok.4,
        ok.5,
        "ring rows",
    )?;
    combine(
        "a base past the ring",
        ok.0,
        ok.1,
        ok.2,
        ok.3,
        CONV_CHANNELS - HIDDEN + 4,
        ok.5,
        "pass the ring",
    )?;
    combine(
        "a short row",
        ok.0,
        ok.1,
        ok.2,
        ok.3,
        ok.4,
        HIDDEN - 4,
        "x window",
    )?;

    let join = |label: &str, cols: usize, stream_len: usize, needle: &str| -> Result<(), String> {
        let ao = ink2::up(h, &t.y[..HIDDEN])?;
        let w = ink2::up(h, &t.w)?;
        let mut ring = ink2::up(h, &t.ring)?;
        let mut ac = ink2::up(h, &ink2::blank(HIDDEN))?;
        let mut stream = ink2::up(h, &t.stream)?;
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.conv_res_join(
                    encoder,
                    ConvResJoinParams {
                        cols,
                        x: ao.window(Elems(0), Elems(HIDDEN))?,
                        conv: TailConv {
                            w: w.window(Elems(0), Elems(HIDDEN * TAPS))?,
                            taps: TAPS,
                            ring: ring.window_mut(Elems(0), Elems(t.rows * CONV_CHANNELS))?,
                            ring_rows: t.rows,
                            ch: CONV_CHANNELS,
                            base: BASE_ATTN,
                            pos: 0,
                        },
                        y: ac.window_mut(Elems(0), Elems(HIDDEN))?,
                        stream: stream.window_mut(Elems(0), Elems(stream_len))?,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    join("a width off the float4s", HIDDEN - 1, HIDDEN, "float4")?;
    join("a short stream row", HIDDEN, HIDDEN - 4, "stream")?;

    let clip_refusal =
        |label: &str, only_key_taps: bool, ring_rows: usize, needle: &str| -> Result<(), String> {
            let plane = vec![PLANE_FILL; KV_HEADS * CTX * HEAD];
            let (mut q, mut rb) = (ink2::up(h, &c.rows.q)?, ink2::up(h, &c.rows.r)?);
            let (k_raw, v_raw) = (ink2::up(h, &c.rows.k)?, ink2::up(h, &c.rows.v)?);
            let (mut k, mut v) = (ink2::up(h, &c.rows.k)?, ink2::up(h, &c.rows.v)?);
            let (gq, gr) = (ink2::up(h, &c.rows.gq)?, ink2::up(h, &c.rows.gr)?);
            let (mut ck, mut cv) = (ink2::up(h, &plane)?, ink2::up(h, &plane)?);
            let (wk, wv) = (ink2::up(h, &c.wk)?, ink2::up(h, &c.wv)?);
            let (dk, dv) = (ink2::up(h, &c.dyn_k)?, ink2::up(h, &c.dyn_v)?);
            let mut ring = ink2::up(h, &vec![0f32; ring_rows.max(1) * CONV_CHANNELS])?;
            ink2::refused(
                label,
                h.encode(|encoder| {
                    h.kernels.qk_clip_conv_append(
                        encoder,
                        QkClipConvParams {
                            clip: clip_params!(h, c, q, k, v, rb, gq, gr, ck, cv),
                            conv: KvConvParams {
                                k_raw: k_raw.window(Elems(0), Elems(KV_HEADS * HEAD))?,
                                v_raw: v_raw.window(Elems(0), Elems(KV_HEADS * HEAD))?,
                                wk: wk.window(Elems(0), Elems(c.wk.len()))?,
                                wv: wv.window(Elems(0), Elems(c.wv.len()))?,
                                dyn_k: Some(dk.window(Elems(0), Elems(KV_HEADS * TAPS))?),
                                dyn_v: (!only_key_taps)
                                    .then(|| dv.window(Elems(0), Elems(KV_HEADS * TAPS)))
                                    .transpose()?,
                                taps: TAPS,
                                ring: ring
                                    .window_mut(Elems(0), Elems(ring_rows * CONV_CHANNELS))?,
                                ring_rows,
                                ch: CONV_CHANNELS,
                                base_k: 0,
                                base_v: BASE_V,
                                pos: 0,
                            },
                        },
                    )
                }),
                needle,
            )
        };
    clip_refusal("taps for the key alone", true, TAPS - 1, "held together")?;
    clip_refusal("a ring under the reach", false, TAPS - 2, "ring rows")?;

    let route_run = |label: &str, eps: f32, gain_len: usize, needle: &str| -> Result<(), String> {
        let rows = ink2::up(h, &r.rows)?;
        let x = ink2::up(h, &r.x)?;
        let gain = ink2::up(h, &r.gain)?;
        let bias = ink2::up(h, &r.bias)?;
        let mut normed = ink2::up(h, &ink2::blank(HIDDEN))?;
        let mut partials = ink2::up(h, &ink2::blank(EXPERTS))?;
        let mut ticket = ink2::up(h, &[0u32])?;
        let mut logits = ink2::up(h, &ink2::blank(EXPERTS))?;
        let mut ids = ink2::up(h, &[u32::MAX; TOP_K])?;
        let mut weights = ink2::up(h, &ink2::blank(TOP_K))?;
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.router_sigmoid_norm(
                    encoder,
                    RouterSigmoidNormParams {
                        router: RouterSigmoidParams {
                            k: HIDDEN,
                            top_k: TOP_K,
                            rows: rows.window(Elems(0), Elems(r.rows.len()))?,
                            x: x.window(Elems(0), Elems(HIDDEN))?,
                            bias: bias.window(Elems(0), Elems(EXPERTS))?,
                            partials: partials.window_mut(Elems(0), Elems(EXPERTS))?,
                            ticket: ticket.window_mut(Elems(0), Elems(1))?,
                            logits: logits.window_mut(Elems(0), Elems(EXPERTS))?,
                            ids: ids.window_mut(Elems(0), Elems(TOP_K))?,
                            weights: weights.window_mut(Elems(0), Elems(TOP_K))?,
                            scale: SCALE,
                            faults: h.device.fault_word(),
                            layer: LAYER,
                        },
                        gain: gain.window(Elems(0), Elems(gain_len))?,
                        normed: normed.window_mut(Elems(0), Elems(HIDDEN))?,
                        eps,
                    },
                )
            }),
            needle,
        )
    };
    route_run("a zero epsilon", 0.0, HIDDEN, "epsilon")?;
    route_run("a short gain", EPS, HIDDEN - 4, "gain")?;
    Ok(Verdict::Noted(
        "twelve refusals by name, each before an encode".to_string(),
    ))
}

/// The folded tails' clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        (
            "combine_conv_join",
            Box::new(|| combine_conv_join(h)) as ClauseRun,
        ),
        ("conv_res_join", Box::new(|| conv_res_join(h))),
        ("qk_clip_conv_append", Box::new(|| qk_clip_conv_append(h))),
        ("router_sigmoid_norm", Box::new(|| router_sigmoid_norm(h))),
        ("tail_fold_bounds", Box::new(|| tail_fold_bounds(h))),
    ]
}
