//! Decode attention clauses. The output is the attention context. The
//! appended cache row is the host quantizer's bits: `f32_to_f16_bits` or
//! `gguf::quant::quantize_q8_0`. The norm and the partial rope are the f32
//! tree in `oracle_attention`.

use bloomery_device::conformance::{Clause, Verdict, expect_range, why};
use bloomery_device::fault::{Fault, FaultSite, site_mask};
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::kernels::{
    AttentionParams, AttnCache, HEAD_DIM, KV_HEADS, QUERY_HEADS, ROT, f16_len, out_len,
    partials_len, q8_code_len, q8_scale_len,
};
use bloomery_metal::{MetalBuffer, MetalDevice};
use gguf::quant::{GgmlType, dequant_row};

use super::fixture::Fixture;
use super::oracle_attention::{
    attend, band, dequant_f16, dequant_q8, norm_rope, pack_f16, pack_q8,
};
use super::{Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD};

/// The layer a raise is recorded in.
const LAYER: u32 = 5;
/// Cache rows. The append sits at 63, and rows past it stay unread.
const CTX: usize = 80;
/// An f16 cache word no packed value of the planted rows uses as its check.
const F16_HOLE: u16 = 0x0BAD;
/// A q8_0 code word planted where no row was written.
const CODE_HOLE: u32 = 0x0BAD_0BAD;
/// A q8_0 scale planted where no row was written.
const SCALE_HOLE: u16 = 0x0BAD;
/// The q8_0 scale a non-finite block stores.
const Q8_NAN_SCALE: u16 = 0x7e00;

const KIND_Q: u32 = 3;
const KIND_K: u32 = 5;
const KIND_V: u32 = 7;
const KIND_PRE_K: u32 = 11;
const KIND_PRE_V: u32 = 13;

struct Tables {
    eps: f32,
    q_gain: Vec<f32>,
    k_gain: Vec<f32>,
    rope: Vec<f32>,
    note: String,
}

struct Scene {
    queries: usize,
    segments: usize,
    ctx: usize,
    pos: [u32; 2],
    n_keys: [u32; 2],
    /// Rows `[0, prefill)` planted before the launch. Append positions sit past it.
    prefill: usize,
    reread: bool,
    q8: bool,
}

enum CacheBack {
    F16 {
        k: Vec<u16>,
        v: Vec<u16>,
    },
    Q8 {
        k_codes: Vec<u32>,
        k_scales: Vec<u16>,
        v_codes: Vec<u32>,
        v_scales: Vec<u16>,
    },
}

struct Seen {
    out: Vec<f32>,
    cache: CacheBack,
    fault: Option<Fault>,
}

struct Checked {
    out: Vec<f32>,
    band: Vec<f64>,
    worst: f64,
}

fn u(n: usize) -> u32 {
    u32::try_from(n).expect("a gate index fits u32")
}

fn synth(index: u32, dim: u32, kind: u32) -> f32 {
    let n = index
        .wrapping_mul(131)
        .wrapping_add(dim.wrapping_mul(17))
        .wrapping_add(kind.wrapping_mul(997))
        % 1000;
    f32::from(u16::try_from(n).expect("under 1000")) / 999.0 - 0.5
}

fn head_row(index: u32, kind: u32) -> [f32; HEAD_DIM] {
    let mut row = [0f32; HEAD_DIM];
    for (dim, value) in row.iter_mut().enumerate() {
        *value = synth(index, u(dim), kind);
    }
    row
}

fn gain_array(values: &[f32]) -> Result<[f32; HEAD_DIM], String> {
    <[f32; HEAD_DIM]>::try_from(values).map_err(|_| "a gain holds 256 values".to_string())
}

fn take_head(
    flat: &[f32],
    query: usize,
    heads: usize,
    head: usize,
) -> Result<[f32; HEAD_DIM], String> {
    let at = (query * heads + head) * HEAD_DIM;
    flat.get(at..at + HEAD_DIM)
        .and_then(|row| <[f32; HEAD_DIM]>::try_from(row).ok())
        .ok_or_else(|| format!("query {query} head {head} is past the buffer"))
}

fn wider(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b).is_gt() { a } else { b }
}

fn load_gain(fixture: &Fixture, name: &str, kind: u32) -> (Vec<f32>, String) {
    if let Ok(values) = fixture.f32s(name, HEAD_DIM)
        && values.iter().all(|v| v.is_finite())
    {
        return (values, name.to_string());
    }
    let values = (0..HEAD_DIM)
        .map(|dim| 0.75 + 0.25 * synth(0, u(dim), kind))
        .collect();
    (values, format!("synthetic {name}"))
}

fn rope_from_base(base: f32, ctx: usize) -> Vec<f32> {
    let mut table = vec![0f32; ctx * ROT];
    let b = f64::from(base);
    for pos in 0..ctx {
        for i in 0..ROT / 2 {
            let exp = -f64::from(u(2 * i)) / f64::from(u(ROT));
            let theta = f64::from(u(pos)) * b.powf(exp);
            let at = pos * ROT + 2 * i;
            table[at] = theta.cos() as f32;
            table[at + 1] = theta.sin() as f32;
        }
    }
    table
}

fn tensor_len(dims: &[u64]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(usize::try_from(dim).ok()?)
    })
}

fn load_rope(fixture: &Fixture, ctx: usize) -> Result<(Vec<f32>, String), String> {
    let file =
        gguf::Gguf::open(&fixture.path).map_err(|e| format!("the fixture did not reopen: {e}"))?;
    let need = ctx * ROT;
    for tensor in file.iter_tensors() {
        if !tensor.name.to_ascii_lowercase().contains("rope") || tensor.ty != GgmlType::F32 {
            continue;
        }
        if tensor_len(&tensor.dims).is_none_or(|n| n < need) {
            continue;
        }
        let bytes = file
            .data(tensor)
            .map_err(|e| format!("{}: {e}", tensor.name))?;
        let mut out = vec![0f32; need];
        dequant_row(GgmlType::F32, bytes, &mut out).map_err(|e| format!("{}: {e}", tensor.name))?;
        return Ok((out, tensor.name.clone()));
    }
    let base = file.arch_get_f32("rope.freq_base").ok_or_else(|| {
        "the fixture has no f32 rope tensor and names no rope.freq_base".to_string()
    })?;
    if !base.is_finite() || base <= 0.0 {
        return Err(format!(
            "rope.freq_base {base} is not a positive finite base"
        ));
    }
    Ok((rope_from_base(base, ctx), format!("rope.freq_base {base}")))
}

fn load_tables(fixture: &Fixture) -> Result<Tables, String> {
    let eps = fixture.rms_eps()?;
    if !eps.is_finite() || eps < 0.0 {
        return Err(format!(
            "the norm epsilon {eps} is not finite and non-negative"
        ));
    }
    let (q_gain, q_note) = load_gain(fixture, "blk.0.attn_q_norm.weight", KIND_Q);
    let (k_gain, k_note) = load_gain(fixture, "blk.0.attn_k_norm.weight", KIND_K);
    let (rope, rope_note) = load_rope(fixture, CTX)?;
    Ok(Tables {
        eps,
        q_gain,
        k_gain,
        rope,
        note: format!("q gain {q_note}, k gain {k_note}, rope {rope_note}"),
    })
}

fn scene_one(segments: usize, q8: bool, reread: bool) -> Scene {
    Scene {
        queries: 1,
        segments,
        ctx: CTX,
        pos: [63, 0],
        n_keys: [64, 0],
        prefill: 63,
        reread,
        q8,
    }
}

fn scene_two(q8: bool) -> Scene {
    Scene {
        queries: 2,
        segments: 1,
        ctx: CTX,
        pos: [62, 63],
        n_keys: [63, 64],
        prefill: 62,
        reread: false,
        q8,
    }
}

fn inputs(scene: &Scene) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0f32; scene.queries * QUERY_HEADS * HEAD_DIM];
    let mut k = vec![0f32; scene.queries * KV_HEADS * HEAD_DIM];
    let mut v = vec![0f32; scene.queries * KV_HEADS * HEAD_DIM];
    for query in 0..scene.queries {
        for head in 0..QUERY_HEADS {
            let row = head_row(u(head), KIND_Q + u(query));
            let at = (query * QUERY_HEADS + head) * HEAD_DIM;
            q[at..at + HEAD_DIM].copy_from_slice(&row);
        }
        for kv in 0..KV_HEADS {
            let at = (query * KV_HEADS + kv) * HEAD_DIM;
            k[at..at + HEAD_DIM].copy_from_slice(&head_row(u(kv), KIND_K + u(query)));
            v[at..at + HEAD_DIM].copy_from_slice(&head_row(u(kv), KIND_V + u(query)));
        }
    }
    (q, k, v)
}

fn plant_f16(scene: &Scene) -> Result<CacheBack, String> {
    let n = f16_len(scene.ctx).ok_or("the f16 cache length wraps")?;
    let mut k = vec![F16_HOLE; n];
    let mut v = vec![F16_HOLE; n];
    for kv in 0..KV_HEADS {
        for row in 0..scene.prefill {
            let at = (kv * scene.ctx + row) * HEAD_DIM;
            k[at..at + HEAD_DIM].copy_from_slice(&pack_f16(&head_row(u(row), KIND_PRE_K + u(kv))));
            v[at..at + HEAD_DIM].copy_from_slice(&pack_f16(&head_row(u(row), KIND_PRE_V + u(kv))));
        }
    }
    Ok(CacheBack::F16 { k, v })
}

fn plant_q8(scene: &Scene) -> Result<CacheBack, String> {
    let codes_n = q8_code_len(scene.ctx).ok_or("the code length wraps")?;
    let scales_n = q8_scale_len(scene.ctx).ok_or("the scale length wraps")?;
    let mut k_codes = vec![CODE_HOLE; codes_n];
    let mut v_codes = vec![CODE_HOLE; codes_n];
    let mut k_scales = vec![SCALE_HOLE; scales_n];
    let mut v_scales = vec![SCALE_HOLE; scales_n];
    for kv in 0..KV_HEADS {
        for row in 0..scene.prefill {
            let index = kv * scene.ctx + row;
            let (kc, ks) = pack_q8(&head_row(u(row), KIND_PRE_K + u(kv)));
            let (vc, vs) = pack_q8(&head_row(u(row), KIND_PRE_V + u(kv)));
            k_codes[index * 64..index * 64 + 64].copy_from_slice(&kc);
            v_codes[index * 64..index * 64 + 64].copy_from_slice(&vc);
            k_scales[index * 8..index * 8 + 8].copy_from_slice(&ks);
            v_scales[index * 8..index * 8 + 8].copy_from_slice(&vs);
        }
    }
    Ok(CacheBack::Q8 {
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    })
}

fn plant(scene: &Scene) -> Result<CacheBack, String> {
    if scene.prefill > scene.ctx {
        return Err("the prefill is past the cache".to_string());
    }
    if scene.q8 {
        plant_q8(scene)
    } else {
        plant_f16(scene)
    }
}

fn upload_f32<'a>(
    device: &'a MetalDevice,
    stream: &'a bloomery_metal::stream::MetalStream,
    body: &[f32],
) -> Result<(MetalBuffer<f32>, usize), String> {
    let host = super::padded(f32::from_bits(SENTINEL_BITS), VALUE_PAD, body, 1);
    let buffer = device.upload(stream, &host).map_err(why)?;
    Ok((buffer, host.len()))
}

enum Uploaded {
    F16 {
        k: MetalBuffer<u16>,
        v: MetalBuffer<u16>,
    },
    Q8 {
        k_codes: MetalBuffer<u32>,
        k_scales: MetalBuffer<u16>,
        v_codes: MetalBuffer<u32>,
        v_scales: MetalBuffer<u16>,
    },
}

fn upload_cache(
    device: &MetalDevice,
    stream: &bloomery_metal::stream::MetalStream,
    planted: &CacheBack,
) -> Result<Uploaded, String> {
    match planted {
        CacheBack::F16 { k, v } => Ok(Uploaded::F16 {
            k: device.upload(stream, k).map_err(why)?,
            v: device.upload(stream, v).map_err(why)?,
        }),
        CacheBack::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => Ok(Uploaded::Q8 {
            k_codes: device.upload(stream, k_codes).map_err(why)?,
            k_scales: device.upload(stream, k_scales).map_err(why)?,
            v_codes: device.upload(stream, v_codes).map_err(why)?,
            v_scales: device.upload(stream, v_scales).map_err(why)?,
        }),
    }
}

fn read_cache(
    device: &MetalDevice,
    stream: &bloomery_metal::stream::MetalStream,
    uploaded: &Uploaded,
) -> Result<CacheBack, String> {
    match uploaded {
        Uploaded::F16 { k, v } => {
            let mut kb = vec![0u16; k.len().get()];
            let mut vb = vec![0u16; v.len().get()];
            device
                .read(stream, k.whole().map_err(why)?, &mut kb)
                .map_err(why)?;
            device
                .read(stream, v.whole().map_err(why)?, &mut vb)
                .map_err(why)?;
            Ok(CacheBack::F16 { k: kb, v: vb })
        }
        Uploaded::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let mut kc = vec![0u32; k_codes.len().get()];
            let mut ks = vec![0u16; k_scales.len().get()];
            let mut vc = vec![0u32; v_codes.len().get()];
            let mut vs = vec![0u16; v_scales.len().get()];
            device
                .read(stream, k_codes.whole().map_err(why)?, &mut kc)
                .map_err(why)?;
            device
                .read(stream, k_scales.whole().map_err(why)?, &mut ks)
                .map_err(why)?;
            device
                .read(stream, v_codes.whole().map_err(why)?, &mut vc)
                .map_err(why)?;
            device
                .read(stream, v_scales.whole().map_err(why)?, &mut vs)
                .map_err(why)?;
            Ok(CacheBack::Q8 {
                k_codes: kc,
                k_scales: ks,
                v_codes: vc,
                v_scales: vs,
            })
        }
    }
}

fn cache_of(uploaded: &mut Uploaded, ctx: usize) -> Result<AttnCache<'_>, DeviceError> {
    match uploaded {
        Uploaded::F16 { k, v } => {
            let n = f16_len(ctx).ok_or_else(|| DeviceError::Range {
                what: "attention",
                detail: "the f16 cache length wraps".to_string(),
            })?;
            Ok(AttnCache::F16 {
                k: k.window_mut(Elems(0), Elems(n))?,
                v: v.window_mut(Elems(0), Elems(n))?,
            })
        }
        Uploaded::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let codes = q8_code_len(ctx).ok_or_else(|| DeviceError::Range {
                what: "attention",
                detail: "the code length wraps".to_string(),
            })?;
            let scales = q8_scale_len(ctx).ok_or_else(|| DeviceError::Range {
                what: "attention",
                detail: "the scale length wraps".to_string(),
            })?;
            Ok(AttnCache::Q8 {
                k_codes: k_codes.window_mut(Elems(0), Elems(codes))?,
                k_scales: k_scales.window_mut(Elems(0), Elems(scales))?,
                v_codes: v_codes.window_mut(Elems(0), Elems(codes))?,
                v_scales: v_scales.window_mut(Elems(0), Elems(scales))?,
            })
        }
    }
}

fn launch(
    h: &Harness<'_>,
    tables: &Tables,
    scene: &Scene,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    planted: &CacheBack,
) -> Result<Seen, String> {
    let (device, stream) = (h.device, h.stream);
    h.clear_fault()?;
    if tables.rope.len() != scene.ctx * ROT {
        return Err("the rope table is not one row per cache row".to_string());
    }
    let out_n = out_len(scene.queries).ok_or("the output length wraps")?;
    let partial_n =
        partials_len(scene.queries, scene.segments).ok_or("the partial length wraps")?;
    if q.len() != out_n || k.len() != scene.queries * KV_HEADS * HEAD_DIM || k.len() != v.len() {
        return Err("the query or the key buffer is not the scene's shape".to_string());
    }
    let (q_buf, _) = upload_f32(device, stream, q)?;
    let (k_buf, _) = upload_f32(device, stream, k)?;
    let (v_buf, _) = upload_f32(device, stream, v)?;
    let (q_gain, _) = upload_f32(device, stream, &tables.q_gain)?;
    let (k_gain, _) = upload_f32(device, stream, &tables.k_gain)?;
    let (rope, _) = upload_f32(device, stream, &tables.rope)?;
    let mut partials = device.upload(stream, &vec![0f32; partial_n]).map_err(why)?;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let out_host = super::padded(sentinel, OUT_GUARD, &vec![sentinel; out_n], OUT_GUARD);
    let mut out_buf = device.upload(stream, &out_host).map_err(why)?;
    let mut uploaded = upload_cache(device, stream, planted)?;
    let queries = scene.queries;
    let segments = scene.segments;
    let ctx = scene.ctx;
    let reread = scene.reread;
    let pos = scene.pos;
    let n_keys = scene.n_keys;
    let eps = tables.eps;
    h.encode(|encoder| {
        let cache = cache_of(&mut uploaded, ctx)?;
        h.kernels.attention(
            encoder,
            AttentionParams {
                head_dim: HEAD_DIM,
                query_heads: QUERY_HEADS,
                kv_heads: KV_HEADS,
                queries,
                segments,
                ctx,
                eps,
                layer: LAYER,
                pos,
                n_keys,
                reread,
                q: q_buf.window(Elems(VALUE_PAD), Elems(q.len()))?,
                k: k_buf.window(Elems(VALUE_PAD), Elems(k.len()))?,
                v: v_buf.window(Elems(VALUE_PAD), Elems(v.len()))?,
                q_gain: q_gain.window(Elems(VALUE_PAD), Elems(HEAD_DIM))?,
                k_gain: k_gain.window(Elems(VALUE_PAD), Elems(HEAD_DIM))?,
                rope: rope.window(Elems(VALUE_PAD), Elems(ctx * ROT))?,
                cache,
                partials: partials.window_mut(Elems(0), Elems(partial_n))?,
                out: out_buf.window_mut(Elems(OUT_GUARD), Elems(out_n))?,
                fault: device.fault_word(),
            },
        )
    })?;
    let mut back = vec![0f32; out_host.len()];
    device
        .read(stream, out_buf.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    super::check_guards(&back, out_n)?;
    let cache = read_cache(device, stream, &uploaded)?;
    Ok(Seen {
        out: back[OUT_GUARD..OUT_GUARD + out_n].to_vec(),
        cache,
        fault: h.fault()?,
    })
}

fn dequant_k(cache: &CacheBack, ctx: usize, kv: usize, index: usize) -> [f32; HEAD_DIM] {
    let row = kv * ctx + index;
    match cache {
        CacheBack::F16 { k, .. } => {
            let mut out = [0f32; HEAD_DIM];
            for (dim, out) in out.iter_mut().enumerate() {
                *out = dequant_f16(k[row * HEAD_DIM + dim]);
            }
            out
        }
        CacheBack::Q8 {
            k_codes, k_scales, ..
        } => dequant_q8(
            &k_codes[row * 64..row * 64 + 64],
            &k_scales[row * 8..row * 8 + 8],
        ),
    }
}

fn dequant_v(cache: &CacheBack, ctx: usize, kv: usize, index: usize) -> [f32; HEAD_DIM] {
    let row = kv * ctx + index;
    match cache {
        CacheBack::F16 { v, .. } => {
            let mut out = [0f32; HEAD_DIM];
            for (dim, out) in out.iter_mut().enumerate() {
                *out = dequant_f16(v[row * HEAD_DIM + dim]);
            }
            out
        }
        CacheBack::Q8 {
            v_codes, v_scales, ..
        } => dequant_q8(
            &v_codes[row * 64..row * 64 + 64],
            &v_scales[row * 8..row * 8 + 8],
        ),
    }
}

fn first_diff<T: PartialEq + Copy>(got: &[T], want: &[T]) -> Option<(usize, T, T)> {
    got.iter()
        .zip(want)
        .enumerate()
        .find_map(|(i, (g, w))| (*g != *w).then_some((i, *g, *w)))
}

fn quant_label(scene: &Scene) -> &'static str {
    if scene.q8 {
        "gguf::quant::quantize_q8_0"
    } else {
        "gguf::quant::f32_to_f16_bits"
    }
}

fn expect_f16_row(
    got: &[u16],
    row: usize,
    want: &[u16; HEAD_DIM],
    what: &str,
) -> Result<(), String> {
    let at = row * HEAD_DIM;
    let have = got
        .get(at..at + HEAD_DIM)
        .ok_or_else(|| format!("{what} row {row} is past the cache"))?;
    if let Some((dim, g, w)) = first_diff(have, want) {
        return Err(format!(
            "{what} row {row} dim {dim}: got {g:#x}, f32_to_f16_bits has {w:#x}"
        ));
    }
    Ok(())
}

fn expect_q8_row(
    codes: &[u32],
    scales: &[u16],
    row: usize,
    want_codes: &[u32; 64],
    want_scales: &[u16; 8],
    what: &str,
) -> Result<(), String> {
    let code = codes
        .get(row * 64..row * 64 + 64)
        .ok_or_else(|| format!("{what} row {row} codes are past the cache"))?;
    let scale = scales
        .get(row * 8..row * 8 + 8)
        .ok_or_else(|| format!("{what} row {row} scales are past the cache"))?;
    if let Some((i, g, w)) = first_diff(code, want_codes) {
        return Err(format!(
            "{what} row {row} code {i}: got {g:#x}, gguf::quant::quantize_q8_0 has {w:#x}"
        ));
    }
    if let Some((i, g, w)) = first_diff(scale, want_scales) {
        return Err(format!(
            "{what} row {row} scale {i}: got {g:#x}, gguf::quant::quantize_q8_0 has {w:#x}"
        ));
    }
    Ok(())
}

fn append_of(scene: &Scene, row: usize) -> Result<Option<usize>, String> {
    let mut found = None;
    for (query, &pos) in scene.pos.iter().enumerate().take(scene.queries) {
        if usize::try_from(pos).map_err(|_| "a position does not fit".to_string())? == row {
            if found.is_some() {
                return Err(format!("two queries append row {row}"));
            }
            found = Some(query);
        }
    }
    Ok(found)
}

fn expect_append_row(
    tables: &Tables,
    scene: &Scene,
    k: &[f32],
    v: &[f32],
    cache: &CacheBack,
    query: usize,
    kv: usize,
) -> Result<(), String> {
    let pos =
        usize::try_from(scene.pos[query]).map_err(|_| "a position does not fit".to_string())?;
    let row = kv * scene.ctx + pos;
    let rope = tables
        .rope
        .get(pos * ROT..pos * ROT + ROT)
        .ok_or("the rope row is past the table")?;
    let key = norm_rope(
        &take_head(k, query, KV_HEADS, kv)?,
        &gain_array(&tables.k_gain)?,
        tables.eps,
        rope,
    )?;
    let value = take_head(v, query, KV_HEADS, kv)?;
    match cache {
        CacheBack::F16 { k, v } => {
            expect_f16_row(k, row, &pack_f16(&key), "k")?;
            expect_f16_row(v, row, &pack_f16(&value), "v")
        }
        CacheBack::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let (kc, ks) = pack_q8(&key);
            let (vc, vs) = pack_q8(&value);
            expect_q8_row(k_codes, k_scales, row, &kc, &ks, "k")?;
            expect_q8_row(v_codes, v_scales, row, &vc, &vs, "v")
        }
    }
}

fn expect_planted_row(
    scene: &Scene,
    planted: &CacheBack,
    got: &CacheBack,
    kv: usize,
    row: usize,
) -> Result<(), String> {
    let index = kv * scene.ctx + row;
    match (planted, got) {
        (CacheBack::F16 { k: pk, v: pv }, CacheBack::F16 { k, v }) => {
            let planted_k = pk
                .get(index * HEAD_DIM..index * HEAD_DIM + HEAD_DIM)
                .and_then(|r| <[u16; HEAD_DIM]>::try_from(r).ok())
                .ok_or("the planted k row is short")?;
            let planted_v = pv
                .get(index * HEAD_DIM..index * HEAD_DIM + HEAD_DIM)
                .and_then(|r| <[u16; HEAD_DIM]>::try_from(r).ok())
                .ok_or("the planted v row is short")?;
            expect_f16_row(k, index, &planted_k, "k")?;
            expect_f16_row(v, index, &planted_v, "v")
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
            let kc = pkc
                .get(index * 64..index * 64 + 64)
                .and_then(|r| <[u32; 64]>::try_from(r).ok())
                .ok_or("the planted k codes are short")?;
            let ks = pks
                .get(index * 8..index * 8 + 8)
                .and_then(|r| <[u16; 8]>::try_from(r).ok())
                .ok_or("the planted k scales are short")?;
            let vc = pvc
                .get(index * 64..index * 64 + 64)
                .and_then(|r| <[u32; 64]>::try_from(r).ok())
                .ok_or("the planted v codes are short")?;
            let vs = pvs
                .get(index * 8..index * 8 + 8)
                .and_then(|r| <[u16; 8]>::try_from(r).ok())
                .ok_or("the planted v scales are short")?;
            expect_q8_row(k_codes, k_scales, index, &kc, &ks, "k")?;
            expect_q8_row(v_codes, v_scales, index, &vc, &vs, "v")
        }
        _ => Err("the read-back cache is a different format".to_string()),
    }
}

fn expect_cache(
    tables: &Tables,
    scene: &Scene,
    k: &[f32],
    v: &[f32],
    planted: &CacheBack,
    got: &CacheBack,
) -> Result<(), String> {
    for kv in 0..KV_HEADS {
        for row in 0..scene.ctx {
            match append_of(scene, row)? {
                Some(query) => expect_append_row(tables, scene, k, v, got, query, kv)?,
                None => expect_planted_row(scene, planted, got, kv, row)?,
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct QueryIn<'a> {
    tables: &'a Tables,
    scene: &'a Scene,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    planted: &'a CacheBack,
    seen: &'a Seen,
}

fn check_head(
    inp: &QueryIn<'_>,
    query: usize,
    head: usize,
    band_out: &mut [f64],
    worst: &mut f64,
) -> Result<(), String> {
    let QueryIn {
        tables,
        scene,
        q,
        seen,
        ..
    } = *inp;
    let kv = head / 8;
    let pos =
        usize::try_from(scene.pos[query]).map_err(|_| "a position does not fit".to_string())?;
    let keys_n =
        usize::try_from(scene.n_keys[query]).map_err(|_| "a key count does not fit".to_string())?;
    let rope = tables
        .rope
        .get(pos * ROT..pos * ROT + ROT)
        .ok_or("the rope row is past the table")?;
    let qn = norm_rope(
        &take_head(q, query, QUERY_HEADS, head)?,
        &gain_array(&tables.q_gain)?,
        tables.eps,
        rope,
    )?;
    let mut keys = Vec::with_capacity(keys_n);
    let mut vals = Vec::with_capacity(keys_n);
    for index in 0..keys_n {
        keys.push(dequant_k(&seen.cache, scene.ctx, kv, index));
        vals.push(dequant_v(&seen.cache, scene.ctx, kv, index));
    }
    let refer = attend(&qn, &keys, &vals)?;
    let bound = band(&qn, &keys, &vals, scene.segments);
    for dim in 0..HEAD_DIM {
        let at = (query * QUERY_HEADS + head) * HEAD_DIM + dim;
        let got = seen.out.get(at).copied().ok_or("the output is short")?;
        let b = bound[dim];
        let diff = (f64::from(got) - refer[dim]).abs();
        if !b.is_finite() {
            return Err(format!(
                "query {query} head {head} dim {dim}: the band is not finite"
            ));
        }
        let ratio = if b > 0.0 {
            diff / b
        } else if diff == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        *worst = wider(*worst, ratio);
        band_out[at] = b;
        if !got.is_finite() || !refer[dim].is_finite() || diff.total_cmp(&b).is_gt() {
            return Err(format!(
                "query {query} head {head} dim {dim}: got {got}, f64 attention {}, |diff| {diff} over band {b} (ratio {ratio}) via {}",
                refer[dim],
                quant_label(scene)
            ));
        }
    }
    Ok(())
}

fn check_queries(inp: &QueryIn<'_>, heads: std::ops::Range<usize>) -> Result<Checked, String> {
    let QueryIn {
        tables,
        scene,
        k,
        v,
        planted,
        seen,
        ..
    } = *inp;
    if let Some(fault) = seen.fault {
        return Err(format!("a finite attention raised {fault}"));
    }
    expect_cache(tables, scene, k, v, planted, &seen.cache)?;
    let mut band_out = vec![0.0f64; seen.out.len()];
    let mut worst = 0.0f64;
    for query in 0..scene.queries {
        for head in heads.clone() {
            check_head(inp, query, head, &mut band_out, &mut worst)?;
        }
    }
    Ok(Checked {
        out: seen.out.clone(),
        band: band_out,
        worst,
    })
}

fn require_live_append(scene: &Scene) -> Result<(), String> {
    for query in 0..scene.queries {
        let pos = scene.pos[query];
        let keys = scene.n_keys[query];
        if keys == 0 || usize::try_from(keys).unwrap_or(usize::MAX) > scene.ctx {
            return Err(format!(
                "query {query} key count {keys} is not a live count"
            ));
        }
        if pos >= keys || usize::try_from(pos).unwrap_or(usize::MAX) >= scene.ctx {
            return Err(format!(
                "query {query} position {pos} is not an append in the cache"
            ));
        }
        if usize::try_from(pos).unwrap_or(usize::MAX) < scene.prefill {
            return Err(format!("query {query} appends a planted row"));
        }
    }
    Ok(())
}

fn run(h: &Harness<'_>, tables: &Tables, scene: &Scene) -> Result<Checked, String> {
    require_live_append(scene)?;
    let (q, k, v) = inputs(scene);
    let planted = plant(scene)?;
    let seen = launch(h, tables, scene, &q, &k, &v, &planted)?;
    check_queries(
        &QueryIn {
            tables,
            scene,
            q: &q,
            k: &k,
            v: &v,
            planted: &planted,
            seen: &seen,
        },
        0..QUERY_HEADS,
    )
}

fn agree(a: &Checked, b: &Checked) -> Result<(), String> {
    if a.out.len() != b.out.len() || a.band.len() != a.out.len() || b.band.len() != b.out.len() {
        return Err("the two attentions differ in length".to_string());
    }
    for i in 0..a.out.len() {
        let limit = a.band[i] + b.band[i];
        let diff = (f64::from(a.out[i]) - f64::from(b.out[i])).abs();
        if !a.out[i].is_finite()
            || !b.out[i].is_finite()
            || !limit.is_finite()
            || diff.total_cmp(&limit).is_gt()
        {
            return Err(format!(
                "position {i}: {} and {} differ by {diff}, the bands sum to {limit}",
                a.out[i], b.out[i]
            ));
        }
    }
    Ok(())
}

fn noted(tables: &Tables, worst: f64, what: &str) -> Verdict {
    Verdict::Noted(format!(
        "{what}, worst |got-ref|/band {worst:.4}, {}",
        tables.note
    ))
}

/// f16 attention and the appended row match the oracle at one segment and at four.
pub fn attn_f16(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let one = run(h, &tables, &scene_one(1, false, false))?;
    let four = run(h, &tables, &scene_one(4, false, false))?;
    Ok(noted(
        &tables,
        wider(one.worst, four.worst),
        "f16 S=1 and S=4",
    ))
}

/// The appended q8_0 row matches `gguf::quant::quantize_q8_0`, and attention over it is inside the band.
pub fn attn_q8(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let one = run(h, &tables, &scene_one(1, true, false))?;
    let four = run(h, &tables, &scene_one(4, true, false))?;
    Ok(noted(
        &tables,
        wider(one.worst, four.worst),
        "q8_0 S=1 and S=4, append gguf::quant::quantize_q8_0",
    ))
}

/// S = 1 and S = 4 agree inside the sum of their bands, and each matches the oracle.
pub fn attn_s(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let one = run(h, &tables, &scene_one(1, false, false))?;
    let four = run(h, &tables, &scene_one(4, false, false))?;
    agree(&one, &four)?;
    Ok(noted(&tables, wider(one.worst, four.worst), "S=1 and S=4"))
}

/// The re-fetch arm and the threadgroup-tile arm both match the oracle, and each other inside the band.
pub fn attn_reread(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let mut worst = 0.0f64;
    for q8 in [false, true] {
        for segments in [1usize, 4] {
            let tile = run(h, &tables, &scene_one(segments, q8, false))?;
            let reread = run(h, &tables, &scene_one(segments, q8, true))?;
            agree(&tile, &reread)?;
            worst = wider(worst, wider(tile.worst, reread.worst));
        }
    }
    Ok(noted(
        &tables,
        worst,
        "tile and reread, f16 and q8_0, S=1 and S=4",
    ))
}

/// Two query positions in one tile match the oracle, f16 and q8_0.
pub fn attn_m(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let f16 = run(h, &tables, &scene_two(false))?;
    let q8 = run(h, &tables, &scene_two(true))?;
    Ok(noted(&tables, wider(f16.worst, q8.worst), "queries=2"))
}

fn expect_site(fault: Option<Fault>, site: FaultSite) -> Result<(), String> {
    match fault {
        Some(f) if f.layer == LAYER && f.site() == Some(site) && f.sites == site_mask(site) => {
            Ok(())
        }
        Some(f) => Err(format!("raised {f}, not {} in layer {LAYER}", site.name())),
        None => Err(format!("{} raised nothing", site.name())),
    }
}

fn expect_written_nan(out: &[f32]) -> Result<(), String> {
    for (i, value) in out.iter().enumerate() {
        if value.to_bits() == SENTINEL_BITS {
            return Err(format!("value {i} was not written"));
        }
        if !value.is_nan() {
            return Err(format!("value {i} is {value:?}, not NaN"));
        }
    }
    Ok(())
}

fn cache_same(planted: &CacheBack, got: &CacheBack) -> Result<(), String> {
    match (planted, got) {
        (CacheBack::F16 { k: pk, v: pv }, CacheBack::F16 { k, v }) => {
            if let Some((i, g, w)) = first_diff(k, pk) {
                return Err(format!(
                    "k cache word {i} is {g:#x}, the planted word is {w:#x}"
                ));
            }
            if let Some((i, g, w)) = first_diff(v, pv) {
                return Err(format!(
                    "v cache word {i} is {g:#x}, the planted word is {w:#x}"
                ));
            }
            Ok(())
        }
        _ => Err("a fault that must not write changed the cache format".to_string()),
    }
}

fn expect_nan_q8_row(codes: &[u32], scales: &[u16], row: usize) -> Result<(), String> {
    let scale = scales
        .get(row * 8..row * 8 + 8)
        .ok_or("the bad q8_0 row's scales are past the cache")?;
    let code = codes
        .get(row * 64..row * 64 + 64)
        .ok_or("the bad q8_0 row's codes are past the cache")?;
    for (i, &bits) in scale.iter().enumerate() {
        if bits != Q8_NAN_SCALE {
            return Err(format!(
                "k row {row} scale {i} is {bits:#x}, a non-finite block stores {Q8_NAN_SCALE:#x}"
            ));
        }
    }
    for (i, &word) in code.iter().enumerate() {
        if word != 0 {
            return Err(format!(
                "k row {row} code {i} is {word:#x}, a non-finite block stores zero codes"
            ));
        }
    }
    Ok(())
}

/// KeyCount and CachePos write NaN and leave the cache untouched. KvQuant stores a NaN scale and zero codes.
pub fn attn_fault(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    for (pos, keys, site) in [
        (63u32, 0u32, FaultSite::KeyCount),
        (63, u(CTX + 1), FaultSite::KeyCount),
        (u(CTX), 64, FaultSite::CachePos),
    ] {
        let scene = Scene {
            queries: 1,
            segments: 1,
            ctx: CTX,
            pos: [pos, 0],
            n_keys: [keys, 0],
            prefill: 0,
            reread: false,
            q8: false,
        };
        let (q, k, v) = inputs(&scene);
        let planted = plant(&scene)?;
        let seen = launch(h, &tables, &scene, &q, &k, &v, &planted)?;
        expect_site(seen.fault, site)?;
        expect_written_nan(&seen.out)?;
        cache_same(&planted, &seen.cache)?;
    }

    let scene = Scene {
        queries: 1,
        segments: 1,
        ctx: CTX,
        pos: [63, 0],
        n_keys: [64, 0],
        prefill: 0,
        reread: false,
        q8: true,
    };
    let (q, mut k, v) = inputs(&scene);
    k[0] = f32::NAN;
    let planted = plant(&scene)?;
    let seen = launch(h, &tables, &scene, &q, &k, &v, &planted)?;
    expect_site(seen.fault, FaultSite::KvQuant)?;
    let CacheBack::Q8 {
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    } = &seen.cache
    else {
        return Err("the q8_0 fault read back another format".to_string());
    };
    let row = 63;
    expect_nan_q8_row(k_codes, k_scales, row)?;
    let (vc, vs) = pack_q8(&take_head(&v, 0, KV_HEADS, 0)?);
    expect_q8_row(v_codes, v_scales, row, &vc, &vs, "v")?;
    for kv in 1..KV_HEADS {
        expect_append_row(&tables, &scene, &k, &v, &seen.cache, 0, kv)?;
    }
    for kv in 0..KV_HEADS {
        for index in 0..scene.ctx {
            if index == row {
                continue;
            }
            expect_planted_row(&scene, &planted, &seen.cache, kv, index)?;
        }
    }
    for (i, value) in seen.out[..8 * HEAD_DIM].iter().enumerate() {
        if value.to_bits() == SENTINEL_BITS || !value.is_nan() {
            return Err(format!(
                "kv 0 value {i} is {value:?}, the NaN key's heads are NaN"
            ));
        }
    }
    // The fault word is raised, and kv 0's key is the NaN row. The other key
    // head's outputs still match the oracle over the cache that was written.
    let mut ignored = vec![0.0f64; seen.out.len()];
    let mut worst = 0.0f64;
    let inp = QueryIn {
        tables: &tables,
        scene: &scene,
        q: &q,
        k: &k,
        v: &v,
        planted: &planted,
        seen: &seen,
    };
    for head in 8..QUERY_HEADS {
        check_head(&inp, 0, head, &mut ignored, &mut worst)?;
    }
    Ok(Verdict::Noted(format!(
        "key_count at 0 and past the cache, cache_pos, kv_quant with a NaN scale and zero codes, other key head worst ratio {worst:.4}"
    )))
}

struct Bounds {
    q: MetalBuffer<f32>,
    k: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    q_gain: MetalBuffer<f32>,
    k_gain: MetalBuffer<f32>,
    rope: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    out: MetalBuffer<f32>,
    k_cache: MetalBuffer<u16>,
    v_cache: MetalBuffer<u16>,
    out_bytes: Vec<f32>,
}

struct Refusal {
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    queries: usize,
    segments: usize,
    ctx: usize,
    q_window: usize,
    needle: &'static str,
}

fn refuse_one(
    h: &Harness<'_>,
    bounds: &mut Bounds,
    eps: f32,
    refusal: &Refusal,
) -> Result<(), String> {
    let result = {
        let mut recording = h.stream.begin().map_err(why)?;
        let mut encoder = recording.compute_encoder().map_err(why)?;
        h.kernels.attention(
            &mut encoder,
            AttentionParams {
                head_dim: refusal.head_dim,
                query_heads: refusal.query_heads,
                kv_heads: refusal.kv_heads,
                queries: refusal.queries,
                segments: refusal.segments,
                ctx: refusal.ctx,
                eps,
                layer: LAYER,
                pos: [63, 0],
                n_keys: [64, 0],
                reread: false,
                q: bounds
                    .q
                    .window(Elems(0), Elems(refusal.q_window))
                    .map_err(why)?,
                k: bounds
                    .k
                    .window(Elems(0), Elems(KV_HEADS * HEAD_DIM))
                    .map_err(why)?,
                v: bounds
                    .v
                    .window(Elems(0), Elems(KV_HEADS * HEAD_DIM))
                    .map_err(why)?,
                q_gain: bounds
                    .q_gain
                    .window(Elems(0), Elems(HEAD_DIM))
                    .map_err(why)?,
                k_gain: bounds
                    .k_gain
                    .window(Elems(0), Elems(HEAD_DIM))
                    .map_err(why)?,
                rope: bounds
                    .rope
                    .window(Elems(0), Elems(CTX * ROT))
                    .map_err(why)?,
                cache: AttnCache::F16 {
                    k: bounds
                        .k_cache
                        .window_mut(Elems(0), Elems(f16_len(CTX).ok_or("cache")?))
                        .map_err(why)?,
                    v: bounds
                        .v_cache
                        .window_mut(Elems(0), Elems(f16_len(CTX).ok_or("cache")?))
                        .map_err(why)?,
                },
                partials: bounds
                    .partials
                    .window_mut(Elems(0), Elems(partials_len(1, 1).ok_or("partials")?))
                    .map_err(why)?,
                out: bounds
                    .out
                    .window_mut(Elems(0), Elems(out_len(1).ok_or("out")?))
                    .map_err(why)?,
                fault: h.device.fault_word(),
            },
        )
    };
    expect_range(result, "attention", refusal.needle)
}

/// A head dim, head count, query count or window the kernel does not take is refused by name.
pub fn attn_bounds(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables(fixture)?;
    let (device, stream) = (h.device, h.stream);
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let out_n = out_len(1).ok_or("the output length wraps")?;
    let out_bytes = vec![sentinel; out_n];
    let mut bounds = Bounds {
        q: device.upload(stream, &vec![0f32; out_n]).map_err(why)?,
        k: device
            .upload(stream, &vec![0f32; KV_HEADS * HEAD_DIM])
            .map_err(why)?,
        v: device
            .upload(stream, &vec![0f32; KV_HEADS * HEAD_DIM])
            .map_err(why)?,
        q_gain: device.upload(stream, &tables.q_gain).map_err(why)?,
        k_gain: device.upload(stream, &tables.k_gain).map_err(why)?,
        rope: device.upload(stream, &tables.rope).map_err(why)?,
        partials: device
            .upload(stream, &vec![0f32; partials_len(1, 1).ok_or("partials")?])
            .map_err(why)?,
        out: device.upload(stream, &out_bytes).map_err(why)?,
        k_cache: device
            .upload(stream, &vec![F16_HOLE; f16_len(CTX).ok_or("cache")?])
            .map_err(why)?,
        v_cache: device
            .upload(stream, &vec![F16_HOLE; f16_len(CTX).ok_or("cache")?])
            .map_err(why)?,
        out_bytes,
    };
    let full = out_n;
    let cases = [
        Refusal {
            head_dim: 128,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 1,
            segments: 1,
            ctx: CTX,
            q_window: full,
            needle: "head dim",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: 8,
            kv_heads: KV_HEADS,
            queries: 1,
            segments: 1,
            ctx: CTX,
            q_window: full,
            needle: "query heads",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: 1,
            queries: 1,
            segments: 1,
            ctx: CTX,
            q_window: full,
            needle: "kv heads",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 0,
            segments: 1,
            ctx: CTX,
            q_window: full,
            needle: "queries",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 3,
            segments: 1,
            ctx: CTX,
            q_window: full,
            needle: "queries",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 1,
            segments: 1,
            ctx: CTX,
            q_window: 1,
            needle: "window",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 1,
            segments: 0,
            ctx: CTX,
            q_window: full,
            needle: "segments",
        },
        Refusal {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries: 1,
            segments: 1,
            ctx: 0,
            q_window: full,
            needle: "ctx",
        },
    ];
    for case in &cases {
        refuse_one(h, &mut bounds, tables.eps, case)?;
    }
    let mut back = vec![0f32; bounds.out_bytes.len()];
    device
        .read(stream, bounds.out.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if let Some((i, (g, w))) = back
        .iter()
        .zip(&bounds.out_bytes)
        .enumerate()
        .find(|(_, (g, w))| g.to_bits() != w.to_bits())
    {
        return Err(format!(
            "a refused launch wrote output word {i}: {g:?}, it was {w:?}"
        ));
    }
    Ok(Verdict::Noted(
        "head dim, query heads, kv heads, queries, window, segments and ctx refused, nothing written"
            .to_string(),
    ))
}

/// The decode-attention clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("attn-f16", Box::new(|| ctx.with(attn_f16))),
        ("attn-q8", Box::new(|| ctx.with(attn_q8))),
        ("attn-s", Box::new(|| ctx.with(attn_s))),
        ("attn-reread", Box::new(|| ctx.with(attn_reread))),
        ("attn-m", Box::new(|| ctx.with(attn_m))),
        ("attn-fault", Box::new(|| ctx.with(attn_fault))),
        ("attn-bounds", Box::new(|| ctx.with(attn_bounds))),
    ]
}
