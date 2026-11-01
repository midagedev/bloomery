//! The window, per-head gate and rope clauses of the attention kernels, at the
//! geometry of the row's instance. Each scene plants a cache, launches one of
//! the three paths (the decode flash, the rows flash, the prefill flash) with
//! the layer a case names, and holds every output of every query head to the
//! f64 attention over the last `window` of its keys (`oracle_attention`), gated
//! by the head's own value when the instance gates per head. The appended rows
//! are the oracle's f16 or q8_0 bits and every other cache word is the planted
//! one. The cache rows before a layer's first window key are poisoned with the
//! largest finite f16 in keys and values, so a read of one is red wherever the
//! kernel reaches it.
//!
//! The kernels read a host table of `(cos, sin)` a pair; the rope factors are
//! that table's recipe ([`oracle_attention::rope_row_f64`]). The launch is given
//! the table the case builds and the oracle rotates with the recipe as it
//! should be, so a table defect is a mutant of the case, named on the gate's
//! command line (`--mutant NAME`), which the kernels' source does not carry.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::{Buffer, Device, DeviceError, Elems, Fault, Window};
use bloomery_metal::geom::AttnGeom;
use bloomery_metal::kernels::{
    AttentionParams, AttentionPrefillParams, AttentionRowsParams, AttnLayer, segments_for_window,
};
use bloomery_metal::{MetalBuffer, MetalDevice};

use super::clauses_attention::{
    CacheBack, KIND_K, KIND_Q, KIND_V, cache_of_for, dequant_k_for, dequant_v_for, gate_of,
    head_row_for, read_cache, take_qg_for, u, upload_cache,
};
use super::common::{LAYER, first_diff, ratio};
use super::oracle::sigmoid;
use super::oracle_attention::{
    attend, band, head_gate_row, norm_rope, pack_f16, pack_q8, prefill_band, rope_row_f64,
    window_tail,
};
use super::{ClauseCtx, ClauseRun, Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD};

/// Cache rows of every scene: the deepest scene's keys and the rows it appends.
const CTX: usize = 1056;
/// The norm's epsilon of the query and key heads.
const EPS: f32 = 1e-5;
/// The largest finite f16: what a cache row before the window holds.
const POISON: u16 = 0x7BFF;
/// An f16 word no packed value of a planted row uses, where no row was planted.
const HOLE: u16 = 0x0BAD;
/// A q8_0 code word and scale where no row was planted.
const CODE_HOLE: u32 = 0x0BAD_0BAD;
const SCALE_HOLE: u16 = 0x0BAD;
/// The kinds of the planted rows, apart from the launch's own projections.
const KIND_PRE_K: u32 = 11;
const KIND_PRE_V: u32 = 13;
/// The base of a full layer's rope and of a sliding layer's.
const BASE_FULL: f64 = 1_000_000.0;
const BASE_SLIDING: f64 = 10_000.0;

/// The rope recipe a table is built from: ggml's, with an optional divisor a pair.
#[derive(Clone)]
struct Rope {
    base: f64,
    /// The width the exponent `-2i/dims` runs over.
    dims: usize,
    factors: Option<Vec<f32>>,
}

impl Rope {
    /// A plain rope over the whole turned width.
    fn plain(attn: &AttnGeom, base: f64) -> Self {
        Self {
            base,
            dims: attn.rot,
            factors: None,
        }
    }

    /// `ctx` rows of `attn.rot` values.
    fn table(&self, attn: &AttnGeom, ctx: usize) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(ctx * attn.rot);
        for pos in 0..ctx {
            out.extend(rope_row_f64(
                pos,
                self.base,
                self.dims,
                attn.rot,
                self.factors.as_deref(),
            )?);
        }
        Ok(out)
    }
}

/// LFM3's `rope_freqs.weight`: half the pairs unfactored, the rest `1e30`.
fn lfm3_factors(pairs: usize) -> Vec<f32> {
    (0..pairs)
        .map(|i| if i < pairs / 2 { 1.0 } else { 1e30 })
        .collect()
}

/// The tables of a scene: the gains, the table the launch reads and the one
/// the oracle rotates with.
struct Tables {
    q_gain: Vec<f32>,
    k_gain: Vec<f32>,
    rope: Vec<f32>,
    reference: Vec<f32>,
}

fn gain_of(dim: usize, kind: u32) -> f32 {
    let n = (u(dim).wrapping_mul(17).wrapping_add(kind.wrapping_mul(997))) % 1000;
    0.75 + 0.25 * (f32::from(u16::try_from(n).expect("under 1000")) / 999.0 - 0.5)
}

impl Tables {
    /// `launch` is the recipe the kernel's table is built from, `reference`
    /// the one the oracle rotates with.
    fn of(attn: &AttnGeom, launch: &Rope, reference: &Rope) -> Result<Self, String> {
        Ok(Self {
            q_gain: (0..attn.head).map(|d| gain_of(d, KIND_Q)).collect(),
            k_gain: (0..attn.head).map(|d| gain_of(d, KIND_K)).collect(),
            rope: launch.table(attn, CTX)?,
            reference: reference.table(attn, CTX)?,
        })
    }

    /// A plain full layer's tables.
    fn plain(attn: &AttnGeom) -> Result<Self, String> {
        let rope = Rope::plain(attn, BASE_FULL);
        Self::of(attn, &rope, &rope)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    /// One decode flash launch.
    Decode,
    /// One rows launch.
    Rows,
    /// A decode flash launch a row, in one command buffer.
    Stepped,
    /// One prefill launch.
    Prefill,
}

/// One scene: a path, the chunk of positions it launches and the layer's window.
#[derive(Clone)]
struct Case {
    path: Path,
    q8: bool,
    /// The first position launched; a decode scene's is its position.
    from: usize,
    rows: usize,
    /// Split-K segments of the decode and rows flashes.
    segments: usize,
    window: Option<u32>,
    /// The cache rows before the first window key hold the poison.
    poison: bool,
    label: String,
}

impl Case {
    /// The decode flash at `n_keys` live keys, the segments of the keys the
    /// window leaves.
    fn decode(n_keys: usize, window: Option<u32>) -> Self {
        Self {
            path: Path::Decode,
            q8: false,
            from: n_keys - 1,
            rows: 1,
            segments: segments_for_window(u(n_keys), window) as usize,
            window,
            poison: window.is_some(),
            label: format!("decode keys {n_keys} window {window:?}"),
        }
    }

    /// A chunk of `rows` rows from `from`, one flash launch a row or one rows
    /// launch, with the segments of the last row's keys.
    fn rows(path: Path, from: usize, rows: usize, window: Option<u32>) -> Self {
        let last = from + rows;
        Self {
            path,
            q8: false,
            from,
            rows,
            segments: segments_for_window(u(last), window) as usize,
            window,
            poison: window.is_some(),
            label: format!("{path:?} {rows} rows from {from} window {window:?}"),
        }
    }

    /// The scene launched in `segments` segments, off the rule: the split
    /// flash and its fold at a key count the one-pass flash takes.
    fn split(mut self, segments: usize) -> Self {
        self.segments = segments;
        self.label.push_str(&format!(" segments {segments}"));
        self
    }

    /// The first key the first row of the chunk attends.
    fn first_key(&self) -> usize {
        self.window.map_or(0, |w| {
            (self.from + 1).saturating_sub(usize::try_from(w).expect("a window fits usize"))
        })
    }
}

/// The projections of a chunk, rows end to end.
struct Inputs {
    q: Vec<f32>,
    /// One raw value a row and query head; empty without a per-head gate.
    gate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// The raw gate of query head `head` of row `r`: in `[-2, 2]` and unlike the
/// next head's and the next row's.
fn head_gate_of(r: usize, head: usize) -> f32 {
    (((r * 7 + head * 5) % 9) as f32 - 4.0) * 0.5
}

fn inputs(attn: &AttnGeom, rows: usize) -> Result<Inputs, String> {
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let mut inp = Inputs {
        q: vec![0f32; rows * q_n],
        gate: Vec::new(),
        k: vec![0f32; rows * kv_n],
        v: vec![0f32; rows * kv_n],
    };
    for r in 0..rows {
        for head in 0..attn.query_heads {
            let at = r * q_n + head * attn.qg_stride;
            inp.q[at..at + attn.head].copy_from_slice(&head_row_for(
                attn,
                u(r * attn.query_heads + head),
                KIND_Q,
            ));
            if attn.gated {
                for dim in 0..attn.head {
                    inp.q[at + attn.head + dim] = gate_of(u(dim + r));
                }
            }
            if attn.head_gate {
                inp.gate.push(head_gate_of(r, head));
            }
        }
        for kv in 0..attn.kv_heads {
            let at = r * kv_n + kv * attn.head;
            let row = u(r * attn.kv_heads + kv);
            inp.k[at..at + attn.head].copy_from_slice(&head_row_for(attn, row, KIND_K));
            inp.v[at..at + attn.head].copy_from_slice(&head_row_for(attn, row, KIND_V));
        }
    }
    Ok(inp)
}

/// The cache of a scene: the rows `[0, from)` planted, the ones before the
/// first window key poisoned, the rest holes.
fn plant(attn: &AttnGeom, case: &Case) -> Result<CacheBack, String> {
    let poison_below = if case.poison { case.first_key() } else { 0 };
    let head = attn.head;
    let poison_row = vec![65_504.0f32; head];
    if case.q8 {
        let (words, blocks) = (head / 4, head / 32);
        let codes_n = attn.q8_code_len(CTX).ok_or("the code length wraps")?;
        let scales_n = attn.q8_scale_len(CTX).ok_or("the scale length wraps")?;
        let mut k_codes = vec![CODE_HOLE; codes_n];
        let mut v_codes = vec![CODE_HOLE; codes_n];
        let mut k_scales = vec![SCALE_HOLE; scales_n];
        let mut v_scales = vec![SCALE_HOLE; scales_n];
        for kv in 0..attn.kv_heads {
            for row in 0..case.from {
                let index = kv * CTX + row;
                let (kp, vp) = if row < poison_below {
                    (poison_row.clone(), poison_row.clone())
                } else {
                    (
                        head_row_for(attn, u(row), KIND_PRE_K + u(kv)),
                        head_row_for(attn, u(row), KIND_PRE_V + u(kv)),
                    )
                };
                let (kc, ks) = pack_q8(&kp);
                let (vc, vs) = pack_q8(&vp);
                k_codes[index * words..(index + 1) * words].copy_from_slice(&kc);
                v_codes[index * words..(index + 1) * words].copy_from_slice(&vc);
                k_scales[index * blocks..(index + 1) * blocks].copy_from_slice(&ks);
                v_scales[index * blocks..(index + 1) * blocks].copy_from_slice(&vs);
            }
        }
        return Ok(CacheBack::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        });
    }
    let n = attn.f16_len(CTX).ok_or("the f16 cache length wraps")?;
    let mut k = vec![HOLE; n];
    let mut v = vec![HOLE; n];
    for kv in 0..attn.kv_heads {
        for row in 0..case.from {
            let at = (kv * CTX + row) * head;
            if row < poison_below {
                k[at..at + head].fill(POISON);
                v[at..at + head].fill(POISON);
            } else {
                k[at..at + head].copy_from_slice(&pack_f16(&head_row_for(
                    attn,
                    u(row),
                    KIND_PRE_K + u(kv),
                )));
                v[at..at + head].copy_from_slice(&pack_f16(&head_row_for(
                    attn,
                    u(row),
                    KIND_PRE_V + u(kv),
                )));
            }
        }
    }
    Ok(CacheBack::F16 { k, v })
}

/// Rows `first..first + n` of the per-head gate buffer, when the instance has one.
fn gate_window(
    buf: &Option<MetalBuffer<f32>>,
    first: usize,
    n: usize,
    heads: usize,
) -> Result<Option<Window<'_, f32, MetalDevice>>, DeviceError> {
    match buf {
        Some(b) => b
            .window(Elems(VALUE_PAD + first * heads), Elems(n * heads))
            .map(Some),
        None => Ok(None),
    }
}

/// What a scene's launch left.
struct Seen {
    out: Vec<f32>,
    cache: CacheBack,
    fault: Option<Fault>,
    ticket: Vec<u32>,
}

fn launch(
    h: &Harness<'_>,
    case: &Case,
    tables: &Tables,
    inp: &Inputs,
    planted: &CacheBack,
) -> Result<Seen, String> {
    let attn = &h.geom().attn;
    let (device, stream) = (h.device, h.stream);
    h.clear_fault()?;
    let rows = case.rows;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let heads = attn.query_heads;
    let partial_n = match case.path {
        Path::Prefill => 0,
        _ => attn
            .partials_len(case.segments)
            .ok_or("the partial length wraps")?,
    };
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let up = |body: &[f32]| -> Result<MetalBuffer<f32>, String> {
        device
            .upload(stream, &super::padded(sentinel, VALUE_PAD, body, 1))
            .map_err(why)
    };
    let q_buf = up(&inp.q)?;
    let k_buf = up(&inp.k)?;
    let v_buf = up(&inp.v)?;
    let q_gain = up(&tables.q_gain)?;
    let k_gain = up(&tables.k_gain)?;
    let rope = up(&tables.rope)?;
    // The gate's buffer holds a query projection's worth of poison past it,
    // so a read of a per-channel index or the next head's lands on NaN.
    let gate_buf = if attn.head_gate {
        let host = super::padded(sentinel, VALUE_PAD, &inp.gate, attn.query_heads * attn.head);
        Some(device.upload(stream, &host).map_err(why)?)
    } else {
        None
    };
    let mut partials = device
        .upload(stream, &vec![0f32; rows * partial_n + 1])
        .map_err(why)?;
    let mut ticket = device
        .upload(stream, &vec![0u32; rows * attn.kv_heads])
        .map_err(why)?;
    let out_host = super::padded(
        sentinel,
        OUT_GUARD,
        &vec![sentinel; rows * out_n],
        OUT_GUARD,
    );
    let mut out_buf = device.upload(stream, &out_host).map_err(why)?;
    let mut uploaded = upload_cache(device, stream, planted)?;
    let (eps, from, ctx) = (EPS, case.from, CTX);
    let gate_rows = |first: usize, n: usize| gate_window(&gate_buf, first, n, heads);
    h.encode(|encoder| match case.path {
        Path::Decode => h.kernels.attention_flash_layer(
            encoder,
            AttentionParams {
                head_dim: attn.head,
                query_heads: heads,
                kv_heads: attn.kv_heads,
                segments: case.segments,
                ctx,
                eps,
                layer: LAYER,
                pos: u(from),
                n_keys: u(from + 1),
                q: q_buf.window(Elems(VALUE_PAD), Elems(q_n))?,
                k: k_buf.window(Elems(VALUE_PAD), Elems(kv_n))?,
                v: v_buf.window(Elems(VALUE_PAD), Elems(kv_n))?,
                q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                rope: rope.window(Elems(VALUE_PAD), Elems(ctx * attn.rot))?,
                cache: cache_of_for(attn, &mut uploaded, ctx)?,
                partials: partials.window_mut(Elems(0), Elems(partial_n))?,
                out: out_buf.window_mut(Elems(OUT_GUARD), Elems(out_n))?,
                fault: device.fault_word(),
                ticket: ticket.window_mut(Elems(0), Elems(attn.kv_heads))?,
            },
            AttnLayer {
                window: case.window,
                head_gate: gate_rows(0, 1)?,
            },
        ),
        Path::Rows => h.kernels.attention_rows_layer(
            encoder,
            AttentionRowsParams {
                head_dim: attn.head,
                query_heads: heads,
                kv_heads: attn.kv_heads,
                segments: case.segments,
                ctx,
                eps,
                layer: LAYER,
                from: u(from),
                rows,
                q: q_buf.window(Elems(VALUE_PAD), Elems(rows * q_n))?,
                q_stride: q_n,
                k: k_buf.window(Elems(VALUE_PAD), Elems(rows * kv_n))?,
                k_stride: kv_n,
                v: v_buf.window(Elems(VALUE_PAD), Elems(rows * kv_n))?,
                v_stride: kv_n,
                q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                rope: rope.window(Elems(VALUE_PAD), Elems(ctx * attn.rot))?,
                cache: cache_of_for(attn, &mut uploaded, ctx)?,
                partials: partials.window_mut(Elems(0), Elems(rows * partial_n))?,
                out: out_buf.window_mut(Elems(OUT_GUARD), Elems(rows * out_n))?,
                out_stride: out_n,
                fault: device.fault_word(),
                ticket: ticket.window_mut(Elems(0), Elems(rows * attn.kv_heads))?,
            },
            AttnLayer {
                window: case.window,
                head_gate: gate_rows(0, rows)?,
            },
        ),
        Path::Stepped => {
            for r in 0..rows {
                h.kernels.attention_flash_layer(
                    encoder,
                    AttentionParams {
                        head_dim: attn.head,
                        query_heads: heads,
                        kv_heads: attn.kv_heads,
                        segments: case.segments,
                        ctx,
                        eps,
                        layer: LAYER,
                        pos: u(from + r),
                        n_keys: u(from + r + 1),
                        q: q_buf.window(Elems(VALUE_PAD + r * q_n), Elems(q_n))?,
                        k: k_buf.window(Elems(VALUE_PAD + r * kv_n), Elems(kv_n))?,
                        v: v_buf.window(Elems(VALUE_PAD + r * kv_n), Elems(kv_n))?,
                        q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                        k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                        rope: rope.window(Elems(VALUE_PAD), Elems(ctx * attn.rot))?,
                        cache: cache_of_for(attn, &mut uploaded, ctx)?,
                        partials: partials.window_mut(Elems(r * partial_n), Elems(partial_n))?,
                        out: out_buf.window_mut(Elems(OUT_GUARD + r * out_n), Elems(out_n))?,
                        fault: device.fault_word(),
                        ticket: ticket
                            .window_mut(Elems(r * attn.kv_heads), Elems(attn.kv_heads))?,
                    },
                    AttnLayer {
                        window: case.window,
                        head_gate: gate_rows(r, 1)?,
                    },
                )?;
            }
            Ok(())
        }
        Path::Prefill => h.kernels.attention_prefill_layer(
            encoder,
            AttentionPrefillParams {
                head_dim: attn.head,
                query_heads: heads,
                kv_heads: attn.kv_heads,
                ctx,
                eps,
                layer: LAYER,
                from: u(from),
                rows,
                q: q_buf.window(Elems(VALUE_PAD), Elems(rows * q_n))?,
                q_stride: q_n,
                k: k_buf.window(Elems(VALUE_PAD), Elems(rows * kv_n))?,
                k_stride: kv_n,
                v: v_buf.window(Elems(VALUE_PAD), Elems(rows * kv_n))?,
                v_stride: kv_n,
                q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                rope: rope.window(Elems(VALUE_PAD), Elems(ctx * attn.rot))?,
                cache: cache_of_for(attn, &mut uploaded, ctx)?,
                out: out_buf.window_mut(Elems(OUT_GUARD), Elems(rows * out_n))?,
                out_stride: out_n,
                fault: device.fault_word(),
            },
            AttnLayer {
                window: case.window,
                head_gate: gate_rows(0, rows)?,
            },
        ),
    })?;
    let mut tickets = vec![1u32; rows * attn.kv_heads];
    device
        .read(stream, ticket.whole().map_err(why)?, &mut tickets)
        .map_err(why)?;
    let mut back = vec![0f32; out_host.len()];
    device
        .read(stream, out_buf.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    super::check_guards(&back, rows * out_n)?;
    Ok(Seen {
        out: back[OUT_GUARD..OUT_GUARD + rows * out_n].to_vec(),
        cache: read_cache(device, stream, &uploaded)?,
        fault: h.fault()?,
        ticket: tickets,
    })
}

/// The first row of `got` that is not `want`'s, named.
fn same_row<T: PartialEq + Copy + core::fmt::Debug>(
    got: &[T],
    row: usize,
    want: &[T],
    what: &str,
) -> Result<(), String> {
    let at = row * want.len();
    let have = got
        .get(at..at + want.len())
        .ok_or_else(|| format!("{what} row {row} is past the cache"))?;
    match first_diff(have, want) {
        None => Ok(()),
        Some((dim, g, w)) => Err(format!("{what} row {row} dim {dim}: got {g:?}, want {w:?}")),
    }
}

/// The row of key head `kv` the launch appended at `pos` is the oracle's bits:
/// the key normed and roped, the value raw.
fn expect_appended(
    attn: &AttnGeom,
    tables: &Tables,
    inp: &Inputs,
    got: &CacheBack,
    r: usize,
    pos: usize,
    kv: usize,
) -> Result<(), String> {
    let kv_n = attn.kv_heads * attn.head;
    let at = r * kv_n + kv * attn.head;
    let rope = tables
        .reference
        .get(pos * attn.rot..(pos + 1) * attn.rot)
        .ok_or("the rope row is past the table")?;
    let key = norm_rope(&inp.k[at..at + attn.head], &tables.k_gain, EPS, rope)?;
    let value = &inp.v[at..at + attn.head];
    let row = kv * CTX + pos;
    match got {
        CacheBack::F16 { k, v } => {
            same_row(k, row, &pack_f16(&key), "k")?;
            same_row(v, row, &pack_f16(value), "v")
        }
        CacheBack::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let ((kc, ks), (vc, vs)) = (pack_q8(&key), pack_q8(value));
            same_row(k_codes, row, &kc, "k codes")?;
            same_row(k_scales, row, &ks, "k scales")?;
            same_row(v_codes, row, &vc, "v codes")?;
            same_row(v_scales, row, &vs, "v scales")
        }
    }
}

/// Every cache row outside the appended positions is the planted one, the
/// poison and the holes included.
fn expect_untouched(
    attn: &AttnGeom,
    case: &Case,
    planted: &CacheBack,
    got: &CacheBack,
) -> Result<(), String> {
    let appended = case.from..case.from + case.rows;
    let (head, words, blocks) = (attn.head, attn.head / 4, attn.head / 32);
    for kv in 0..attn.kv_heads {
        for pos in (0..CTX).filter(|p| !appended.contains(p)) {
            let row = kv * CTX + pos;
            let at = format!("kv {kv} position {pos}");
            match (planted, got) {
                (CacheBack::F16 { k: pk, v: pv }, CacheBack::F16 { k, v }) => {
                    same_row(k, row, &pk[row * head..(row + 1) * head], "k")
                        .map_err(|e| format!("{at}: {e}"))?;
                    same_row(v, row, &pv[row * head..(row + 1) * head], "v")
                        .map_err(|e| format!("{at}: {e}"))?;
                }
                (
                    CacheBack::Q8 {
                        k_codes: pkc,
                        k_scales: pks,
                        v_codes: pvc,
                        v_scales: pvs,
                    },
                    CacheBack::Q8 {
                        k_codes,
                        k_scales,
                        v_codes,
                        v_scales,
                    },
                ) => {
                    let w = row * words..(row + 1) * words;
                    let b = row * blocks..(row + 1) * blocks;
                    same_row(k_codes, row, &pkc[w.clone()], "k codes")
                        .and_then(|()| same_row(k_scales, row, &pks[b.clone()], "k scales"))
                        .and_then(|()| same_row(v_codes, row, &pvc[w], "v codes"))
                        .and_then(|()| same_row(v_scales, row, &pvs[b], "v scales"))
                        .map_err(|e| format!("{at}: {e}"))?;
                }
                _ => return Err("the read-back cache is another format".to_string()),
            }
        }
    }
    Ok(())
}

/// The launch left no fault, no ticket set, the appended and the planted rows
/// as they should be, and every output of every query head within its band of
/// the f64 attention over the keys the layer attends. The worst ratio of an
/// error to its band.
fn check(
    attn: &AttnGeom,
    case: &Case,
    tables: &Tables,
    inp: &Inputs,
    planted: &CacheBack,
    seen: &Seen,
) -> Result<f64, String> {
    let at = |e: String| format!("{}: {e}", case.label);
    if let Some(fault) = seen.fault {
        return Err(at(format!("a finite attention raised {fault}")));
    }
    if seen.ticket.iter().any(|&t| t != 0) {
        return Err(at(format!(
            "a fold's ticket reads {:?}, not zero",
            seen.ticket
        )));
    }
    for r in 0..case.rows {
        for kv in 0..attn.kv_heads {
            expect_appended(attn, tables, inp, &seen.cache, r, case.from + r, kv).map_err(at)?;
        }
    }
    expect_untouched(attn, case, planted, &seen.cache).map_err(at)?;
    let end = case.from + case.rows;
    let rows_of = |dequant: fn(&AttnGeom, &CacheBack, usize, usize, usize) -> Vec<f32>| {
        (0..attn.kv_heads)
            .map(|kv| {
                (0..end)
                    .map(|i| dequant(attn, &seen.cache, CTX, kv, i))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let (keys, vals) = (rows_of(dequant_k_for), rows_of(dequant_v_for));
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let mut worst = 0.0f64;
    for r in 0..case.rows {
        let pos = case.from + r;
        let rope = tables
            .reference
            .get(pos * attn.rot..(pos + 1) * attn.rot)
            .ok_or("the rope row is past the table")?;
        let q_row = &inp.q[r * q_n..(r + 1) * q_n];
        for head in 0..attn.query_heads {
            let qn = norm_rope(
                take_qg_for(attn, q_row, head, 0)?,
                &tables.q_gain,
                EPS,
                rope,
            )?;
            let gate_row: Option<Vec<f32>> = if attn.head_gate {
                Some(head_gate_row(
                    inp.gate[r * attn.query_heads + head],
                    attn.head,
                ))
            } else if attn.gated {
                Some(take_qg_for(attn, q_row, head, attn.head)?.to_vec())
            } else {
                None
            };
            let gate = gate_row.as_deref();
            let kv = head / attn.gqa;
            let ks = window_tail(&keys[kv][..=pos], case.window);
            let vs = window_tail(&vals[kv][..=pos], case.window);
            let refer = attend(&qn, ks, vs)?;
            let bound = if case.path == Path::Prefill {
                prefill_band(&qn, ks, vs, gate)?
            } else {
                band(&qn, ks, vs, gate, case.segments)?
            };
            for dim in 0..attn.head {
                let got = seen.out[r * out_n + head * attn.head + dim];
                let want = gate.map_or(refer[dim], |g| refer[dim] * sigmoid(f64::from(g[dim])));
                let diff = (f64::from(got) - want).abs();
                if !got.is_finite() || !want.is_finite() || diff > bound[dim] {
                    return Err(at(format!(
                        "row {r} (position {pos}) head {head} dim {dim}: got {got}, f64 attention {want} over {} keys, |diff| {diff} over band {}",
                        ks.len(),
                        bound[dim]
                    )));
                }
                worst = worst.max(ratio(diff, bound[dim]));
            }
        }
    }
    Ok(worst)
}

/// One scene, from planting to the oracle: the outputs and the worst ratio.
fn run(h: &Harness<'_>, tables: &Tables, case: &Case) -> Result<(Seen, f64), String> {
    let attn = &h.geom().attn;
    let inp = inputs(attn, case.rows)?;
    let planted = plant(attn, case)?;
    let seen = launch(h, case, tables, &inp, &planted)?;
    let worst = check(attn, case, tables, &inp, &planted, &seen)?;
    Ok((seen, worst))
}

fn same_bits(what: &str, a: &[f32], b: &[f32]) -> Result<(), String> {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    match first_diff(&bits(a), &bits(b)) {
        None => Ok(()),
        Some((i, x, y)) => Err(format!(
            "{what}: word {i} is {x:#x}, the other launch's {y:#x}"
        )),
    }
}

/// A noted verdict of the scenes' worst ratio.
fn noted(what: &str, parts: &[String], worst: f64) -> Clause {
    Ok(Verdict::Noted(format!(
        "{what}: {}; worst |got-ref|/band {worst:.4}",
        parts.join(", ")
    )))
}

/// The decode flash attends the last `window` of its live keys: window 256 at
/// depths 300 and 1025 against the full layer, the window at, one past and
/// far past the depth, one key, a window narrower than a segment, the segments
/// a caller counts by the unwindowed rule, and the q8_0 cache. A window past
/// the depth is the full layer's bits.
pub fn attn_window(h: &Harness<'_>) -> Clause {
    let attn = &h.geom().attn;
    let tables = Tables::plain(attn)?;
    let mut cases = vec![
        Case::decode(1025, None),
        Case::decode(300, Some(256)),
        Case::decode(1025, Some(256)),
        Case::decode(256, Some(256)),
        Case::decode(257, Some(256)),
        Case::decode(300, Some(4096)),
        Case::decode(300, Some(64)),
        Case::decode(1025, Some(1)),
    ];
    // The split's count of 1025 unwindowed keys over a window of 100: the
    // segments past the window's seven tiles are empty.
    cases.push(Case::decode(1025, Some(100)).split(segments_for_window(1025, None) as usize));
    let mut q8 = Case::decode(1025, Some(256));
    q8.q8 = true;
    q8.label.push_str(" q8_0");
    cases.push(q8);
    let mut worst = 0.0f64;
    let mut parts = Vec::new();
    for case in &cases {
        let (_, w) = run(h, &tables, case)?;
        worst = worst.max(w);
        parts.push(format!(
            "{}/{}{}",
            case.from + 1,
            case.window.map_or("full".to_string(), |w| w.to_string()),
            if case.q8 { "q8" } else { "" }
        ));
    }
    let (full, _) = run(h, &tables, &Case::decode(300, None))?;
    let (past, _) = run(h, &tables, &Case::decode(300, Some(4096)))?;
    same_bits(
        "a window past the depth against the full layer",
        &past.out,
        &full.out,
    )?;
    noted("keys/window", &parts, worst)
}

/// The chunks of the rows launch with a window: the window past the first
/// rows' positions, a chunk that crosses it (also split, off the rule), a deep
/// chunk, a narrow window, and no window. Every row is within the band of the
/// oracle and is the bits of the decode launch at its position with the same
/// segment count.
pub fn attn_window_rows(h: &Harness<'_>) -> Clause {
    let attn = &h.geom().attn;
    let tables = Tables::plain(attn)?;
    let chunks = [
        (0usize, 9usize, Some(256u32), None),
        (250, 20, Some(256), None),
        (250, 20, Some(256), Some(8usize)),
        (300, 17, Some(100), None),
        (1000, 33, Some(256), None),
        (300, 9, None, None),
    ];
    let mut worst = 0.0f64;
    let mut parts = Vec::new();
    for (from, rows, window, split) in chunks {
        let at = |path| {
            let case = Case::rows(path, from, rows, window);
            match split {
                Some(segments) => case.split(segments),
                None => case,
            }
        };
        let rows_case = at(Path::Rows);
        let (a, w) = run(h, &tables, &rows_case)?;
        worst = worst.max(w);
        let stepped = at(Path::Stepped);
        let (b, _) = run(h, &tables, &stepped)?;
        same_bits(&rows_case.label, &a.out, &b.out)?;
        parts.push(format!(
            "{rows}@{from}/{}{}",
            window.map_or("full".to_string(), |w| w.to_string()),
            split.map_or(String::new(), |s| format!(" S={s}"))
        ));
    }
    noted("rows@first position/window", &parts, worst)
}

/// The prefill flash with a window: chunks whose blocks start inside and
/// before the window, a block of one, a narrow window and no window, each
/// within the prefill band of the oracle.
pub fn attn_window_prefill(h: &Harness<'_>) -> Clause {
    let attn = &h.geom().attn;
    let tables = Tables::plain(attn)?;
    let chunks = [
        (0usize, 9usize, Some(256u32)),
        (250, 20, Some(256)),
        (300, 64, Some(256)),
        (1000, 40, Some(256)),
        (1000, 33, Some(100)),
        (700, 1, Some(256)),
        (300, 9, None),
    ];
    let mut worst = 0.0f64;
    let mut parts = Vec::new();
    for (from, rows, window) in chunks {
        let (_, w) = run(h, &tables, &Case::rows(Path::Prefill, from, rows, window))?;
        worst = worst.max(w);
        parts.push(format!(
            "{rows}@{from}/{}",
            window.map_or("full".to_string(), |w| w.to_string())
        ));
    }
    noted("rows@first position/window", &parts, worst)
}

/// The per-head gate multiplies every value of a head by the sigmoid of that
/// head's own raw value, on every path: the one-pass flash (one segment), the
/// split's fold, the rows launch of one segment and with a fold, the prefill
/// launch. The gate values differ head to head and row to row. The instances with no per-head gate are the other
/// rows' attention clauses, with the launch's gate window refused by name
/// (`attn_layer_refusals`).
pub fn attn_head_gate(h: &Harness<'_>) -> Clause {
    let attn = &h.geom().attn;
    if !attn.head_gate {
        return Ok(Verdict::Skip(
            "the instance's output gate is not one value a head".to_string(),
        ));
    }
    let tables = Tables::plain(attn)?;
    let cases = [
        Case::decode(20, None),
        Case::decode(64, None),
        Case::decode(300, None),
        Case::decode(300, None).split(10),
        Case::rows(Path::Rows, 200, 9, None),
        Case::rows(Path::Rows, 200, 9, None).split(7),
        Case::rows(Path::Prefill, 37, 9, None),
        Case::rows(Path::Prefill, 300, 9, Some(256)),
    ];
    let mut worst = 0.0f64;
    let mut parts = Vec::new();
    for case in &cases {
        let (_, w) = run(h, &tables, case)?;
        worst = worst.max(w);
        parts.push(format!("{:?} {}@{}", case.path, case.rows, case.from));
    }
    noted("gated launches", &parts, worst)
}

/// The rope table with factors turns the kernels as the recipe says, and the
/// derivation of LFM3's `1e30` holds in f32: its unrotated pairs have a cosine
/// of 1 and a sine under `pos * 1e-30`, with no special case in the kernel. The
/// scenes are LFM3's full layer, a table of factors that all differ, and a
/// sliding layer's (no factors, its own base). A table defect named on the
/// command line (`--mutant NAME`) is built into the launch's table only.
pub fn attn_rope_factors(h: &Harness<'_>, mutant: Option<&str>) -> Clause {
    let attn = &h.geom().attn;
    let pairs = attn.rot / 2;
    let lfm3 = Rope {
        base: BASE_FULL,
        dims: attn.rot,
        factors: Some(lfm3_factors(pairs)),
    };
    let skew = Rope {
        base: BASE_FULL,
        dims: attn.rot,
        factors: Some((0..pairs).map(|i| 1.0 + i as f32 / 16.0).collect()),
    };
    let sliding = Rope::plain(attn, BASE_SLIDING);
    let scenes = [
        ("full layer, LFM3's factors", lfm3, None),
        ("full layer, factors 1 + i/16", skew, None),
        ("sliding layer, no factors", sliding, Some(256u32)),
    ];
    let mut worst = 0.0f64;
    let mut parts = Vec::new();
    for (name, reference, window) in scenes {
        let mut launch_rope = reference.clone();
        match mutant {
            Some("rope_factors_ignored") => launch_rope.factors = None,
            Some("rope_exponent_64") => launch_rope.dims = 64,
            _ => {}
        }
        let tables = Tables::of(attn, &launch_rope, &reference)?;
        if name.starts_with("full layer, LFM3") {
            expect_unrotated_tail(attn, &reference)?;
        }
        let case = Case::decode(1025, window);
        let (_, w) = run(h, &tables, &case)?;
        worst = worst.max(w);
        let chunk = Case::rows(Path::Prefill, 300, 9, window);
        let (_, w) = run(h, &tables, &chunk)?;
        worst = worst.max(w);
        parts.push(name.to_string());
    }
    noted("scenes", &parts, worst)
}

/// The pairs LFM3's table divides by `1e30` come out of the recipe unrotated
/// within f32, at every position of the cache: the cosine is 1, the sine is
/// under `pos * 1e-30` (the angle's bound, `theta_i <= pos`), and one added to
/// it is 1.
fn expect_unrotated_tail(attn: &AttnGeom, reference: &Rope) -> Result<(), String> {
    let table = reference.table(attn, CTX)?;
    for pos in 0..CTX {
        let row = &table[pos * attn.rot..(pos + 1) * attn.rot];
        for i in attn.rot / 4..attn.rot / 2 {
            let (c, s) = (row[2 * i], row[2 * i + 1]);
            let bound = (pos as f32) * 1e-30;
            if c != 1.0 || s.abs() > bound || 1.0 + s.abs() != 1.0 {
                return Err(format!(
                    "position {pos} pair {i}: cos {c}, sin {s}, not 1 and under {bound:e} within f32"
                ));
            }
        }
    }
    Ok(())
}

/// The launches of a layer the instance cannot take are refused by name and
/// write nothing: a window of no keys, a window where the flash cuts none, a
/// gate window where there is no per-head gate, none or one of another length
/// where the gate is per head, and the launch with no layer argument.
pub fn attn_layer_refusals(h: &Harness<'_>) -> Clause {
    let attn = h.geom().attn;
    let (device, stream) = (h.device, h.stream);
    let tables = Tables::plain(&attn)?;
    let case = Case::decode(64, None);
    let inp = inputs(&attn, 1)?;
    let planted = plant(&attn, &case)?;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let heads = attn.query_heads;
    let up = |body: &[f32]| -> Result<MetalBuffer<f32>, String> {
        device
            .upload(stream, &super::padded(sentinel, VALUE_PAD, body, 1))
            .map_err(why)
    };
    let (q_buf, k_buf, v_buf) = (up(&inp.q)?, up(&inp.k)?, up(&inp.v)?);
    let (q_gain, k_gain, rope) = (up(&tables.q_gain)?, up(&tables.k_gain)?, up(&tables.rope)?);
    let gate_buf = up(&vec![0.5f32; heads + 1])?;
    let partial_n = attn.partials_len(1).ok_or("the partial length wraps")?;
    let mut partials = device.upload(stream, &vec![0f32; partial_n]).map_err(why)?;
    let mut ticket = device
        .upload(stream, &vec![0u32; attn.kv_heads])
        .map_err(why)?;
    let out_host = super::padded(sentinel, OUT_GUARD, &vec![sentinel; out_n], OUT_GUARD);
    let mut out_buf = device.upload(stream, &out_host).map_err(why)?;
    let mut uploaded = upload_cache(device, stream, &planted)?;
    let mut refused = Vec::new();
    for (what, needle, window, gate_len, plain) in [
        (
            "a window of no keys",
            "window of 0",
            Some(0u32),
            Some(heads),
            false,
        ),
        (
            "a window where the flash cuts none",
            "cuts no window",
            Some(64),
            Some(heads),
            false,
        ),
        (
            "a gate window where the instance has none",
            "no per-head gate",
            None,
            Some(heads),
            false,
        ),
        (
            "no gate window where the gate is per head",
            "holds no head gate window",
            None,
            None,
            false,
        ),
        (
            "a gate window of another length",
            "head gate window holds",
            None,
            Some(heads + 1),
            false,
        ),
        (
            "a launch with no layer argument",
            "holds no head gate window",
            None,
            None,
            true,
        ),
    ] {
        // A case is a refusal only on the instances it names: a window needs
        // none cut, a gate window needs none to be taken, and the reverse.
        let applies = match needle {
            "window of 0" => attn.windowed,
            "cuts no window" => !attn.windowed,
            "no per-head gate" => !attn.head_gate,
            _ => attn.head_gate,
        };
        if !applies {
            continue;
        }
        let result = h.encode(|encoder| {
            let params = AttentionParams {
                head_dim: attn.head,
                query_heads: heads,
                kv_heads: attn.kv_heads,
                segments: 1,
                ctx: CTX,
                eps: EPS,
                layer: LAYER,
                pos: 63,
                n_keys: 64,
                q: q_buf.window(Elems(VALUE_PAD), Elems(q_n))?,
                k: k_buf.window(Elems(VALUE_PAD), Elems(kv_n))?,
                v: v_buf.window(Elems(VALUE_PAD), Elems(kv_n))?,
                q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
                rope: rope.window(Elems(VALUE_PAD), Elems(CTX * attn.rot))?,
                cache: cache_of_for(&attn, &mut uploaded, CTX)?,
                partials: partials.window_mut(Elems(0), Elems(partial_n))?,
                out: out_buf.window_mut(Elems(OUT_GUARD), Elems(out_n))?,
                fault: device.fault_word(),
                ticket: ticket.window_mut(Elems(0), Elems(attn.kv_heads))?,
            };
            if plain {
                return h.kernels.attention_flash(encoder, params);
            }
            let head_gate = match gate_len {
                Some(n) => Some(gate_buf.window(Elems(VALUE_PAD), Elems(n))?),
                None => None,
            };
            h.kernels
                .attention_flash_layer(encoder, params, AttnLayer { window, head_gate })
        });
        match result {
            Ok(()) => return Err(format!("{what} was accepted")),
            Err(text) if text.contains(needle) => refused.push(what),
            Err(text) => {
                return Err(format!(
                    "{what} was refused without naming {needle}: {text}"
                ));
            }
        }
    }
    let mut back = vec![0f32; out_host.len()];
    device
        .read(stream, out_buf.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if back.iter().any(|v| v.to_bits() != SENTINEL_BITS) {
        return Err("a refused launch wrote".to_string());
    }
    if refused.is_empty() {
        return Err("no refusal case applies to this instance".to_string());
    }
    Ok(Verdict::Noted(format!(
        "{} refused by name, nothing written",
        refused.join(", ")
    )))
}

/// The clauses of the instance whose flash cuts a window and gates per head,
/// in gate order.
pub fn table<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    let mutant = super::flag(ctx.args, "--mutant");
    vec![
        (
            "attn_window",
            Box::new(move || attn_window(h)) as ClauseRun<'a>,
        ),
        ("attn_window_rows", Box::new(move || attn_window_rows(h))),
        (
            "attn_window_prefill",
            Box::new(move || attn_window_prefill(h)),
        ),
        ("attn_head_gate", Box::new(move || attn_head_gate(h))),
        (
            "attn_rope_factors",
            Box::new(move || attn_rope_factors(h, mutant)),
        ),
        (
            "attn_layer_refusals",
            Box::new(move || attn_layer_refusals(h)),
        ),
    ]
}

/// The layer refusals alone: a row whose flash cuts no window and has no
/// per-head gate refuses a launch that names either.
pub fn refusals<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![(
        "attn_layer_refusals",
        Box::new(move || attn_layer_refusals(h)) as ClauseRun<'a>,
    )]
}
