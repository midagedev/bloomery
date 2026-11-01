//! Decode attention clauses at the row's instance's geometry. The output is
//! the attention, gated when the instance's query carries a gate. The appended
//! cache row is the host quantizer's bits: `f32_to_f16_bits` or
//! `gguf::quant::quantize_q8_0`. The norm and the partial rope are the f32
//! tree in `oracle_attention`.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::{Fault, FaultSite};
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::attention_geom::TILE;
use bloomery_metal::encode::{Binding, ComputeEncoder};
use bloomery_metal::geom::AttnGeom;
use bloomery_metal::kernels::{AttentionParams, AttnCache, ONE_PASS_KEYS, segments_for_keys};
use bloomery_metal::library::Library;
use bloomery_metal::{MetalBuffer, MetalDevice};
use gguf::quant::{GgmlType, dequant_row, half_to_f32};

use super::common::{self, BadArg, LAYER, expect_fault, first_diff};
use super::fixture::Fixture;
use super::oracle::sigmoid;
use super::oracle_attention::{
    attend, band, dequant_f16, dequant_q8, norm_rope, pack_f16, pack_q8,
};
use super::{Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD};

/// Cache rows of the fault and refusal scenes. The append sits at 63, and
/// rows past it stay unread.
pub(super) const CTX: usize = 80;
/// The widest key-count scene's cache rows: the 4096-key scene plus rows
/// past the live keys that stay unread.
const DEEP_CTX: usize = 4112;
/// The key counts the attention scenes run at by the rule. On the f16 cache's
/// one-pass flash, up to its waterline (`ONE_PASS_KEYS`): one key, one round
/// of the SIMD groups' chunks, then two and eight. Past it, the split at the
/// cap: one key past the waterline, whose segments of three tiles leave the
/// last five holding no key, then four and sixteen tiles a segment, and the
/// deepest scene, whose last segment holds two of its seventeen tiles.
const SCENE_KEYS: [u32; 8] = [1, 64, 100, 512, 513, 1024, 4096, 4112];
const _: () = assert!(
    SCENE_KEYS[3] as usize == ONE_PASS_KEYS && SCENE_KEYS[4] as usize == ONE_PASS_KEYS + 1,
    "the rule's scenes hold the one-pass waterline from both sides"
);
/// The split flash, off the rule: a tile in each of the first four segments
/// with the last twelve holding no key, a tile a segment with a part tile
/// last, two tiles a segment at the cap, and three segments of uneven tiles.
const SPLIT_SCENES: [(u32, usize); 4] = [(64, 16), (100, 7), (512, 16), (1024, 3)];
/// An f16 cache word no packed value of the planted rows uses as its check.
const F16_HOLE: u16 = 0x0BAD;
/// A q8_0 code word planted where no row was written.
const CODE_HOLE: u32 = 0x0BAD_0BAD;
/// A q8_0 scale planted where no row was written.
const SCALE_HOLE: u16 = 0x0BAD;
/// The q8_0 scale a non-finite block stores.
const Q8_NAN_SCALE: u16 = 0x7e00;
/// The key count of the fold's underflow scene, which the rule splits into
/// the cap's segments.
const UNDERFLOW_KEYS: u32 = 1024;
/// How far under the other keys' scores the planted segment's scores sit:
/// past what an f32 weight reaches.
const UNDERFLOW_DROP: f32 = 200.0;
/// The least gap between the planted segment's scores and the other keys' at
/// which an f32 weight of the segment is 0.
const UNDERFLOW_GAP: f32 = 110.0;
/// An f16 quiet NaN.
const F16_NAN: u16 = 0x7e00;
/// A q8_0 cache row's code words.
fn q8_words(attn: &AttnGeom) -> usize {
    attn.head / 4
}

/// A q8_0 cache row's scales.
fn q8_scales(attn: &AttnGeom) -> usize {
    attn.head / 32
}

/// The pattern-expansion kernel, the `attn_half` clause's own: the shipped
/// library does not carry it, and the clause compiles it once beside the
/// source this run's kernels came from, so it still pins the production
/// `attn_half` the attention kernel reads its cache with.
const EXPAND: &str = "\n// Every f16 bit pattern as f32, one value per thread. The host compares it\n// with `half_to_f32`.\nkernel void gqa256_f16_expand(device float *out [[buffer(0)]],\n                              uint gid [[thread_position_in_grid]]) {\n    if (gid >= 65536u) {\n        return;\n    }\n    out[gid] = attn_half(ushort(gid));\n}\n";

pub(super) const KIND_Q: u32 = 3;
pub(super) const KIND_K: u32 = 5;
pub(super) const KIND_V: u32 = 7;
const KIND_PRE_K: u32 = 11;
const KIND_PRE_V: u32 = 13;

pub(super) struct Tables {
    pub(super) eps: f32,
    pub(super) q_gain: Vec<f32>,
    pub(super) k_gain: Vec<f32>,
    pub(super) rope: Vec<f32>,
    pub(super) note: String,
}

pub(super) struct Scene {
    pub(super) segments: usize,
    pub(super) ctx: usize,
    pub(super) pos: u32,
    pub(super) n_keys: u32,
    /// Rows `[0, prefill)` planted before the launch. The append sits past it.
    pub(super) prefill: usize,
    pub(super) q8: bool,
}

pub(super) enum CacheBack {
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
    worst: f64,
}

pub(super) fn u(n: usize) -> u32 {
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

/// One head of the synthetic rows `index` and `kind` name.
pub(super) fn head_row_for(attn: &AttnGeom, index: u32, kind: u32) -> Vec<f32> {
    (0..attn.head)
        .map(|dim| synth(index, u(dim), kind))
        .collect()
}

fn take_head<'f>(attn: &AttnGeom, flat: &'f [f32], head: usize) -> Result<&'f [f32], String> {
    let at = head * attn.head;
    flat.get(at..at + attn.head)
        .ok_or_else(|| format!("head {head} is past the buffer"))
}

/// The `offset`th value of query head `head`'s `qg_stride` values, one head's
/// width of them.
pub(super) fn take_qg_for<'f>(
    attn: &AttnGeom,
    flat: &'f [f32],
    head: usize,
    offset: usize,
) -> Result<&'f [f32], String> {
    let at = head * attn.qg_stride + offset;
    flat.get(at..at + attn.head)
        .ok_or_else(|| format!("head {head} offset {offset} is past the query"))
}

/// A finite gate in about `[-2, 2]`, so the sigmoid is not 1.
pub(super) fn gate_of(dim: u32) -> f32 {
    ((dim % 9) as f32 - 4.0) * 0.5
}

fn wider(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b).is_gt() { a } else { b }
}

fn load_gain(attn: &AttnGeom, fixture: &Fixture, name: &str, kind: u32) -> (Vec<f32>, String) {
    if let Ok(values) = fixture.f32s(name, attn.head)
        && values.iter().all(|v| v.is_finite())
    {
        return (values, name.to_string());
    }
    let values = (0..attn.head)
        .map(|dim| 0.75 + 0.25 * synth(0, u(dim), kind))
        .collect();
    (values, format!("synthetic {name}"))
}

fn rope_from_base(attn: &AttnGeom, base: f32, ctx: usize) -> Vec<f32> {
    let mut table = vec![0f32; ctx * attn.rot];
    let b = f64::from(base);
    for pos in 0..ctx {
        for i in 0..attn.rot / 2 {
            let exp = -f64::from(u(2 * i)) / f64::from(u(attn.rot));
            let theta = f64::from(u(pos)) * b.powf(exp);
            let at = pos * attn.rot + 2 * i;
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

fn load_rope(attn: &AttnGeom, fixture: &Fixture, ctx: usize) -> Result<(Vec<f32>, String), String> {
    let file =
        gguf::Gguf::open(&fixture.path).map_err(|e| format!("the fixture did not reopen: {e}"))?;
    let need = ctx * attn.rot;
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
    Ok((
        rope_from_base(attn, base, ctx),
        format!("rope.freq_base {base}"),
    ))
}

/// The fixture's norm epsilon, the query and key gains (or synthetic ones of
/// the head's width) and a rope table of the deepest scene's rows.
pub(super) fn load_tables_for(attn: &AttnGeom, fixture: &Fixture) -> Result<Tables, String> {
    let eps = fixture.rms_eps()?;
    if !eps.is_finite() || eps < 0.0 {
        return Err(format!(
            "the norm epsilon {eps} is not finite and non-negative"
        ));
    }
    let (q_gain, q_note) = load_gain(attn, fixture, "blk.0.attn_q_norm.weight", KIND_Q);
    let (k_gain, k_note) = load_gain(attn, fixture, "blk.0.attn_k_norm.weight", KIND_K);
    let (rope, rope_note) = load_rope(attn, fixture, DEEP_CTX)?;
    Ok(Tables {
        eps,
        q_gain,
        k_gain,
        rope,
        note: format!("q gain {q_note}, k gain {k_note}, rope {rope_note}"),
    })
}

fn scene_one(segments: usize, q8: bool) -> Scene {
    Scene {
        segments,
        ctx: CTX,
        pos: 63,
        n_keys: 64,
        prefill: 63,
        q8,
    }
}

/// The split flash off the rule: `segments` segments of `n_keys` live keys,
/// the append at the last live row.
fn scene_split(n_keys: u32, segments: usize, q8: bool) -> Scene {
    Scene {
        segments,
        ..scene_keys(n_keys, q8)
    }
}

/// The scene of a live key count: the rule's segments, the append at the
/// last live row, every earlier row planted.
fn scene_keys(n_keys: u32, q8: bool) -> Scene {
    let pos = n_keys - 1;
    Scene {
        segments: segments_for_keys(n_keys) as usize,
        ctx: DEEP_CTX,
        pos,
        n_keys,
        prefill: (pos) as usize,
        q8,
    }
}

fn inputs(attn: &AttnGeom, _scene: &Scene) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_n = attn.qg_len().expect("the query length wraps");
    let mut q = vec![0f32; q_n];
    let mut k = vec![0f32; attn.kv_heads * attn.head];
    let mut v = vec![0f32; attn.kv_heads * attn.head];
    for head in 0..attn.query_heads {
        let at = head * attn.qg_stride;
        q[at..at + attn.head].copy_from_slice(&head_row_for(attn, u(head), KIND_Q));
        if attn.gated {
            for dim in 0..attn.head {
                q[at + attn.head + dim] = gate_of(u(dim));
            }
        }
    }
    for kv in 0..attn.kv_heads {
        let at = kv * attn.head;
        k[at..at + attn.head].copy_from_slice(&head_row_for(attn, u(kv), KIND_K));
        v[at..at + attn.head].copy_from_slice(&head_row_for(attn, u(kv), KIND_V));
    }
    (q, k, v)
}

pub(super) fn plant_f16_for(attn: &AttnGeom, scene: &Scene) -> Result<CacheBack, String> {
    let n = attn
        .f16_len(scene.ctx)
        .ok_or("the f16 cache length wraps")?;
    let mut k = vec![F16_HOLE; n];
    let mut v = vec![F16_HOLE; n];
    for kv in 0..attn.kv_heads {
        for row in 0..scene.prefill {
            let at = (kv * scene.ctx + row) * attn.head;
            let pre_k = head_row_for(attn, u(row), KIND_PRE_K + u(kv));
            let pre_v = head_row_for(attn, u(row), KIND_PRE_V + u(kv));
            k[at..at + attn.head].copy_from_slice(&pack_f16(&pre_k));
            v[at..at + attn.head].copy_from_slice(&pack_f16(&pre_v));
        }
    }
    Ok(CacheBack::F16 { k, v })
}

fn plant_q8(attn: &AttnGeom, scene: &Scene) -> Result<CacheBack, String> {
    let codes_n = attn.q8_code_len(scene.ctx).ok_or("the code length wraps")?;
    let scales_n = attn
        .q8_scale_len(scene.ctx)
        .ok_or("the scale length wraps")?;
    let (words, scales) = (q8_words(attn), q8_scales(attn));
    let mut k_codes = vec![CODE_HOLE; codes_n];
    let mut v_codes = vec![CODE_HOLE; codes_n];
    let mut k_scales = vec![SCALE_HOLE; scales_n];
    let mut v_scales = vec![SCALE_HOLE; scales_n];
    for kv in 0..attn.kv_heads {
        for row in 0..scene.prefill {
            let index = kv * scene.ctx + row;
            let (kc, ks) = pack_q8(&head_row_for(attn, u(row), KIND_PRE_K + u(kv)));
            let (vc, vs) = pack_q8(&head_row_for(attn, u(row), KIND_PRE_V + u(kv)));
            k_codes[index * words..(index + 1) * words].copy_from_slice(&kc);
            v_codes[index * words..(index + 1) * words].copy_from_slice(&vc);
            k_scales[index * scales..(index + 1) * scales].copy_from_slice(&ks);
            v_scales[index * scales..(index + 1) * scales].copy_from_slice(&vs);
        }
    }
    Ok(CacheBack::Q8 {
        k_codes,
        k_scales,
        v_codes,
        v_scales,
    })
}

fn plant(attn: &AttnGeom, scene: &Scene) -> Result<CacheBack, String> {
    if scene.prefill > scene.ctx {
        return Err("the prefill is past the cache".to_string());
    }
    if scene.q8 {
        plant_q8(attn, scene)
    } else {
        plant_f16_for(attn, scene)
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

pub(super) enum Uploaded {
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

pub(super) fn upload_cache(
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

pub(super) fn read_cache(
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

pub(super) fn cache_of_for<'u>(
    attn: &AttnGeom,
    uploaded: &'u mut Uploaded,
    ctx: usize,
) -> Result<AttnCache<'u>, DeviceError> {
    match uploaded {
        Uploaded::F16 { k, v } => {
            let n = attn.f16_len(ctx).ok_or_else(|| DeviceError::Range {
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
            let codes = attn.q8_code_len(ctx).ok_or_else(|| DeviceError::Range {
                what: "attention",
                detail: "the code length wraps".to_string(),
            })?;
            let scales = attn.q8_scale_len(ctx).ok_or_else(|| DeviceError::Range {
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
    let attn = &h.geom().attn;
    let (device, stream) = (h.device, h.stream);
    h.clear_fault()?;
    if tables.rope.len() < scene.ctx * attn.rot {
        return Err("the rope table is shorter than the scene's cache".to_string());
    }
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let partial_n = attn
        .partials_len(scene.segments)
        .ok_or("the partial length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    if q.len() != q_n || k.len() != kv_n || k.len() != v.len() {
        return Err("the query or the key buffer is not the scene's shape".to_string());
    }
    let (q_buf, _) = upload_f32(device, stream, q)?;
    let (k_buf, _) = upload_f32(device, stream, k)?;
    let (v_buf, _) = upload_f32(device, stream, v)?;
    let (q_gain, _) = upload_f32(device, stream, &tables.q_gain)?;
    let (k_gain, _) = upload_f32(device, stream, &tables.k_gain)?;
    let (rope, _) = upload_f32(device, stream, &tables.rope[..scene.ctx * attn.rot])?;
    let mut partials = device.upload(stream, &vec![0f32; partial_n]).map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32]).map_err(why)?;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let out_host = super::padded(sentinel, OUT_GUARD, &vec![sentinel; out_n], OUT_GUARD);
    let mut out_buf = device.upload(stream, &out_host).map_err(why)?;
    let mut uploaded = upload_cache(device, stream, planted)?;
    let segments = scene.segments;
    let ctx = scene.ctx;
    let pos = scene.pos;
    let n_keys = scene.n_keys;
    let eps = tables.eps;
    let q_len = q.len();
    let k_len = k.len();
    h.encode(|encoder| {
        let flash = AttentionParams {
            head_dim: attn.head,
            query_heads: attn.query_heads,
            kv_heads: attn.kv_heads,
            segments,
            ctx,
            eps,
            layer: LAYER,
            pos,
            n_keys,
            q: q_buf.window(Elems(VALUE_PAD), Elems(q_len))?,
            k: k_buf.window(Elems(VALUE_PAD), Elems(k_len))?,
            v: v_buf.window(Elems(VALUE_PAD), Elems(k_len))?,
            q_gain: q_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
            k_gain: k_gain.window(Elems(VALUE_PAD), Elems(attn.head))?,
            rope: rope.window(Elems(VALUE_PAD), Elems(ctx * attn.rot))?,
            cache: cache_of_for(attn, &mut uploaded, ctx)?,
            partials: partials.window_mut(Elems(0), Elems(partial_n))?,
            out: out_buf.window_mut(Elems(OUT_GUARD), Elems(out_n))?,
            fault: device.fault_word(),
            ticket: ticket.window_mut(Elems(0), Elems(1))?,
        };
        // One dispatch: past one segment its last threadgroup folds the
        // partials and writes the gated output.
        h.kernels.attention_flash(encoder, flash)?;
        Ok(())
    })?;
    let mut ticket_back = [1u32; 1];
    device
        .read(stream, ticket.whole().map_err(why)?, &mut ticket_back)
        .map_err(why)?;
    if ticket_back[0] != 0 {
        return Err(format!(
            "the fold's ticket reads back {}, not the zero it resets to",
            ticket_back[0]
        ));
    }
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

/// One key row of the read-back cache, dequantized.
pub(super) fn dequant_k_for(
    attn: &AttnGeom,
    cache: &CacheBack,
    ctx: usize,
    kv: usize,
    index: usize,
) -> Vec<f32> {
    let row = kv * ctx + index;
    match cache {
        CacheBack::F16 { k, .. } => k[row * attn.head..(row + 1) * attn.head]
            .iter()
            .map(|&bits| dequant_f16(bits))
            .collect(),
        CacheBack::Q8 {
            k_codes, k_scales, ..
        } => {
            let (words, scales) = (q8_words(attn), q8_scales(attn));
            dequant_q8(
                &k_codes[row * words..(row + 1) * words],
                &k_scales[row * scales..(row + 1) * scales],
            )
        }
    }
}

/// One value row of the read-back cache, dequantized.
pub(super) fn dequant_v_for(
    attn: &AttnGeom,
    cache: &CacheBack,
    ctx: usize,
    kv: usize,
    index: usize,
) -> Vec<f32> {
    let row = kv * ctx + index;
    match cache {
        CacheBack::F16 { v, .. } => v[row * attn.head..(row + 1) * attn.head]
            .iter()
            .map(|&bits| dequant_f16(bits))
            .collect(),
        CacheBack::Q8 {
            v_codes, v_scales, ..
        } => {
            let (words, scales) = (q8_words(attn), q8_scales(attn));
            dequant_q8(
                &v_codes[row * words..(row + 1) * words],
                &v_scales[row * scales..(row + 1) * scales],
            )
        }
    }
}

fn quant_label(scene: &Scene) -> &'static str {
    if scene.q8 {
        "gguf::quant::quantize_q8_0"
    } else {
        "gguf::quant::f32_to_f16_bits"
    }
}

fn expect_f16_row(got: &[u16], row: usize, want: &[u16], what: &str) -> Result<(), String> {
    let at = row * want.len();
    let have = got
        .get(at..at + want.len())
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
    want_codes: &[u32],
    want_scales: &[u16],
    what: &str,
) -> Result<(), String> {
    let (words, blocks) = (want_codes.len(), want_scales.len());
    let code = codes
        .get(row * words..(row + 1) * words)
        .ok_or_else(|| format!("{what} row {row} codes are past the cache"))?;
    let scale = scales
        .get(row * blocks..(row + 1) * blocks)
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

fn is_append(scene: &Scene, row: usize) -> Result<bool, String> {
    Ok(usize::try_from(scene.pos).map_err(|_| "a position does not fit".to_string())? == row)
}

fn expect_append_row(
    attn: &AttnGeom,
    tables: &Tables,
    scene: &Scene,
    k: &[f32],
    v: &[f32],
    cache: &CacheBack,
    kv: usize,
) -> Result<(), String> {
    let pos = usize::try_from(scene.pos).map_err(|_| "a position does not fit".to_string())?;
    let row = kv * scene.ctx + pos;
    let rope = tables
        .rope
        .get(pos * attn.rot..(pos + 1) * attn.rot)
        .ok_or("the rope row is past the table")?;
    let key = norm_rope(take_head(attn, k, kv)?, &tables.k_gain, tables.eps, rope)?;
    let value = take_head(attn, v, kv)?;
    match cache {
        CacheBack::F16 { k, v } => {
            expect_f16_row(k, row, &pack_f16(&key), "k")?;
            expect_f16_row(v, row, &pack_f16(value), "v")
        }
        CacheBack::Q8 {
            k_codes,
            k_scales,
            v_codes,
            v_scales,
        } => {
            let (kc, ks) = pack_q8(&key);
            let (vc, vs) = pack_q8(value);
            expect_q8_row(k_codes, k_scales, row, &kc, &ks, "k")?;
            expect_q8_row(v_codes, v_scales, row, &vc, &vs, "v")
        }
    }
}

fn expect_planted_row(
    attn: &AttnGeom,
    scene: &Scene,
    planted: &CacheBack,
    got: &CacheBack,
    kv: usize,
    row: usize,
) -> Result<(), String> {
    let index = kv * scene.ctx + row;
    let (head, words, blocks) = (attn.head, q8_words(attn), q8_scales(attn));
    match (planted, got) {
        (CacheBack::F16 { k: pk, v: pv }, CacheBack::F16 { k, v }) => {
            let planted_k = pk
                .get(index * head..(index + 1) * head)
                .ok_or("the planted k row is short")?;
            let planted_v = pv
                .get(index * head..(index + 1) * head)
                .ok_or("the planted v row is short")?;
            expect_f16_row(k, index, planted_k, "k")?;
            expect_f16_row(v, index, planted_v, "v")
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
                .get(index * words..(index + 1) * words)
                .ok_or("the planted k codes are short")?;
            let ks = pks
                .get(index * blocks..(index + 1) * blocks)
                .ok_or("the planted k scales are short")?;
            let vc = pvc
                .get(index * words..(index + 1) * words)
                .ok_or("the planted v codes are short")?;
            let vs = pvs
                .get(index * blocks..(index + 1) * blocks)
                .ok_or("the planted v scales are short")?;
            expect_q8_row(k_codes, k_scales, index, kc, ks, "k")?;
            expect_q8_row(v_codes, v_scales, index, vc, vs, "v")
        }
        _ => Err("the read-back cache is a different format".to_string()),
    }
}

fn expect_cache(
    attn: &AttnGeom,
    tables: &Tables,
    scene: &Scene,
    k: &[f32],
    v: &[f32],
    planted: &CacheBack,
    got: &CacheBack,
) -> Result<(), String> {
    for kv in 0..attn.kv_heads {
        for row in 0..scene.ctx {
            if is_append(scene, row)? {
                expect_append_row(attn, tables, scene, k, v, got, kv)?;
            } else {
                expect_planted_row(attn, scene, planted, got, kv, row)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct QueryIn<'a> {
    attn: &'a AttnGeom,
    tables: &'a Tables,
    scene: &'a Scene,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    planted: &'a CacheBack,
    seen: &'a Seen,
}

/// One head against the oracle: its worst ratio of the error to the band.
fn check_head(inp: &QueryIn<'_>, head: usize) -> Result<f64, String> {
    let QueryIn {
        attn,
        tables,
        scene,
        q,
        seen,
        ..
    } = *inp;
    let kv = head / attn.gqa;
    let pos = usize::try_from(scene.pos).map_err(|_| "a position does not fit".to_string())?;
    let keys_n =
        usize::try_from(scene.n_keys).map_err(|_| "a key count does not fit".to_string())?;
    let rope = tables
        .rope
        .get(pos * attn.rot..(pos + 1) * attn.rot)
        .ok_or("the rope row is past the table")?;
    let qn = norm_rope(
        take_qg_for(attn, q, head, 0)?,
        &tables.q_gain,
        tables.eps,
        rope,
    )?;
    let gate = if attn.gated {
        Some(take_qg_for(attn, q, head, attn.head)?)
    } else {
        None
    };
    let mut keys = Vec::with_capacity(keys_n);
    let mut vals = Vec::with_capacity(keys_n);
    for index in 0..keys_n {
        keys.push(dequant_k_for(attn, &seen.cache, scene.ctx, kv, index));
        vals.push(dequant_v_for(attn, &seen.cache, scene.ctx, kv, index));
    }
    let refer = attend(&qn, &keys, &vals)?;
    let bound = band(&qn, &keys, &vals, gate, scene.segments)?;
    let mut worst = 0.0f64;
    for dim in 0..attn.head {
        let at = head * attn.head + dim;
        let got = seen.out.get(at).copied().ok_or("the output is short")?;
        let want = gate.map_or(refer[dim], |g| refer[dim] * sigmoid(f64::from(g[dim])));
        let b = bound[dim];
        let diff = (f64::from(got) - want).abs();
        if !b.is_finite() {
            return Err(format!("head {head} dim {dim}: the band is not finite"));
        }
        let ratio = if b > 0.0 {
            diff / b
        } else if diff == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        worst = wider(worst, ratio);
        if !got.is_finite() || !want.is_finite() || diff.total_cmp(&b).is_gt() {
            return Err(format!(
                "head {head} dim {dim}: got {got}, f64 attention {want}, |diff| {diff} over band {b} (ratio {ratio}) via {}",
                quant_label(scene)
            ));
        }
    }
    Ok(worst)
}

fn check_queries(inp: &QueryIn<'_>, heads: std::ops::Range<usize>) -> Result<Checked, String> {
    let QueryIn {
        attn,
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
    expect_cache(attn, tables, scene, k, v, planted, &seen.cache)?;
    let mut worst = 0.0f64;
    for head in heads {
        worst = wider(worst, check_head(inp, head)?);
    }
    Ok(Checked { worst })
}

fn require_live_append(scene: &Scene) -> Result<(), String> {
    let pos = scene.pos;
    let keys = scene.n_keys;
    if keys == 0 || usize::try_from(keys).unwrap_or(usize::MAX) > scene.ctx {
        return Err(format!("key count {keys} is not a live count"));
    }
    if pos >= keys || usize::try_from(pos).unwrap_or(usize::MAX) >= scene.ctx {
        return Err(format!("position {pos} is not an append in the cache"));
    }
    if usize::try_from(pos).unwrap_or(usize::MAX) < scene.prefill {
        return Err("the append lands on a planted row".to_string());
    }
    Ok(())
}

fn run(h: &Harness<'_>, tables: &Tables, scene: &Scene) -> Result<Checked, String> {
    require_live_append(scene)?;
    let attn = &h.geom().attn;
    let (q, k, v) = inputs(attn, scene);
    let planted = plant(attn, scene)?;
    let seen = launch(h, tables, scene, &q, &k, &v, &planted)?;
    check_queries(
        &QueryIn {
            attn,
            tables,
            scene,
            q: &q,
            k: &k,
            v: &v,
            planted: &planted,
            seen: &seen,
        },
        0..attn.query_heads,
    )
}

fn noted_band(tables: &Tables, worst: f64, what: &str) -> Verdict {
    Verdict::Noted(format!(
        "{what}, worst |got-ref|/band {worst:.4}, {}",
        tables.note
    ))
}

/// f16 attention and the appended row match the oracle at the rule's key
/// counts (the one-pass flash), at the split's own counts off the rule (the
/// tiles and the ticket's fold), and in the four-segment scene the split rule
/// leaves.
pub fn attn_f16(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables_for(&h.geom().attn, fixture)?;
    let (worst, note) = key_scenes(h, &tables, false)?;
    Ok(noted_band(&tables, worst, &format!("f16{note}")))
}

/// The appended q8_0 row matches `gguf::quant::quantize_q8_0`, and attention
/// over it is inside the band, at the same key counts.
pub fn attn_q8(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables_for(&h.geom().attn, fixture)?;
    let (worst, note) = key_scenes(h, &tables, true)?;
    Ok(noted_band(
        &tables,
        worst,
        &format!("q8_0{note}, append gguf::quant::quantize_q8_0"),
    ))
}

/// Every key-count scene of the f16 or the q8_0 cache: the rule's, the split
/// at the segment counts [`SPLIT_SCENES`] names, and four segments of 64 keys.
/// The worst ratio and the scenes' note.
fn key_scenes(h: &Harness<'_>, tables: &Tables, q8: bool) -> Result<(f64, String), String> {
    let mut worst = 0.0f64;
    let mut note = String::new();
    for &n_keys in &SCENE_KEYS {
        let checked = run(h, tables, &scene_keys(n_keys, q8))?;
        worst = wider(worst, checked.worst);
        note.push_str(&format!(
            " keys={n_keys}/S={} by the rule",
            segments_for_keys(n_keys)
        ));
    }
    for (n_keys, segments) in SPLIT_SCENES {
        let checked = run(h, tables, &scene_split(n_keys, segments, q8))?;
        worst = wider(worst, checked.worst);
        note.push_str(&format!(" keys={n_keys}/S={segments}"));
    }
    let four = run(h, tables, &scene_one(4, q8))?;
    note.push_str(" keys=64/S=4 off the rule");
    Ok((wider(worst, four.worst), note))
}

/// The hardware half conversion matches `half_to_f32` on every pattern.
/// A finite value matches bit for bit. A NaN matches as NaN-ness and sign.
pub fn attn_half(h: &Harness<'_>, source: &str) -> Clause {
    const N: usize = 65_536;
    const THREADS: usize = 256;
    let (device, stream) = (h.device, h.stream);
    let text = format!("{source}{EXPAND}");
    let library = Library::from_source_text(device, &text).map_err(why)?;
    let pipeline = library.pipeline("gqa256_f16_expand").map_err(why)?;
    let host = vec![f32::from_bits(SENTINEL_BITS); N];
    let mut out = device.upload(stream, &host).map_err(why)?;
    h.encode(|encoder| {
        encoder.set_pipeline(&pipeline);
        encoder.set_binding(0, Binding::of_mut(&out.window_mut(Elems(0), Elems(N))?)?)?;
        encoder.dispatch_groups(N / THREADS, 1, THREADS)
    })?;
    let mut back = vec![0f32; N];
    device
        .read(stream, out.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    let mut finite_off = 0u32;
    let mut nan_off = 0u32;
    let mut first = String::new();
    for (gid, &got) in back.iter().enumerate() {
        let bits = u16::try_from(gid).expect("a pattern fits u16");
        let want = half_to_f32(bits);
        let nan_bad =
            want.is_nan() && (!got.is_nan() || (got.to_bits() >> 31) != (want.to_bits() >> 31));
        let finite_bad = !want.is_nan() && got.to_bits() != want.to_bits();
        if nan_bad {
            nan_off += 1;
        }
        if finite_bad {
            finite_off += 1;
        }
        if (nan_bad || finite_bad) && first.is_empty() {
            first = format!(
                "pattern {gid:#x}: got {got:?} ({:#x}), half_to_f32 is {want:?} ({:#x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
    if finite_off != 0 || nan_off != 0 {
        return Err(format!(
            "{finite_off} finite patterns and {nan_off} NaN patterns differ; {first}"
        ));
    }
    Ok(Verdict::Noted(
        "65536 patterns: finite bits match half_to_f32, NaN compared as NaN-ness and sign"
            .to_string(),
    ))
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

fn expect_nan_q8_row(
    attn: &AttnGeom,
    codes: &[u32],
    scales: &[u16],
    row: usize,
) -> Result<(), String> {
    let (words, blocks) = (q8_words(attn), q8_scales(attn));
    let scale = scales
        .get(row * blocks..(row + 1) * blocks)
        .ok_or("the bad q8_0 row's scales are past the cache")?;
    let code = codes
        .get(row * words..(row + 1) * words)
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
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    for (pos, keys, site, what) in [
        (63u32, 0u32, FaultSite::KeyCount, "a key count of 0"),
        (
            63,
            u(CTX + 1),
            FaultSite::KeyCount,
            "a key count past the cache",
        ),
        (
            u(CTX),
            64,
            FaultSite::CachePos,
            "a position at the cache end",
        ),
    ] {
        let scene = Scene {
            segments: 1,
            ctx: CTX,
            pos,
            n_keys: keys,
            prefill: 0,
            q8: false,
        };
        let (q, k, v) = inputs(attn, &scene);
        let planted = plant(attn, &scene)?;
        let seen = launch(h, &tables, &scene, &q, &k, &v, &planted)?;
        expect_fault(h, what, site)?;
        expect_written_nan(&seen.out)?;
        cache_same(&planted, &seen.cache)?;
    }

    let scene = Scene {
        segments: 1,
        ctx: CTX,
        pos: 63,
        n_keys: 64,
        prefill: 0,
        q8: true,
    };
    let (q, mut k, v) = inputs(attn, &scene);
    k[0] = f32::NAN;
    let planted = plant(attn, &scene)?;
    let seen = launch(h, &tables, &scene, &q, &k, &v, &planted)?;
    expect_fault(h, "a non-finite key", FaultSite::KvQuant)?;
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
    expect_nan_q8_row(attn, k_codes, k_scales, row)?;
    let (vc, vs) = pack_q8(take_head(attn, &v, 0)?);
    expect_q8_row(v_codes, v_scales, row, &vc, &vs, "v")?;
    for kv in 1..attn.kv_heads {
        expect_append_row(attn, &tables, &scene, &k, &v, &seen.cache, kv)?;
    }
    for kv in 0..attn.kv_heads {
        for index in 0..scene.ctx {
            if index == row {
                continue;
            }
            expect_planted_row(attn, &scene, &planted, &seen.cache, kv, index)?;
        }
    }
    for (i, value) in seen.out[..attn.gqa * attn.head].iter().enumerate() {
        if value.to_bits() == SENTINEL_BITS || !value.is_nan() {
            return Err(format!(
                "kv 0 value {i} is {value:?}, the NaN key's heads are NaN"
            ));
        }
    }
    // The fault word is raised, and kv 0's key is the NaN row. The other key
    // head's outputs still match the oracle over the cache that was written.
    let mut worst = 0.0f64;
    let inp = QueryIn {
        attn,
        tables: &tables,
        scene: &scene,
        q: &q,
        k: &k,
        v: &v,
        planted: &planted,
        seen: &seen,
    };
    for head in attn.gqa..attn.query_heads {
        worst = wider(worst, check_head(&inp, head)?);
    }
    Ok(Verdict::Noted(format!(
        "key_count at 0 and past the cache, cache_pos, kv_quant with a NaN scale and zero codes, other key head worst ratio {worst:.4}"
    )))
}

/// The keys one segment holds: `attn_span`'s rule (attention.metal), the live
/// keys over the segments, rounded up to a tile.
fn segment_span(n_keys: usize, segments: usize) -> usize {
    n_keys.div_ceil(segments).div_ceil(TILE) * TILE
}

/// A segment whose weights underflow still carries its values to the output.
/// The second of the rule's segments holds only keys that score
/// [`UNDERFLOW_DROP`] under the other keys of query head 0, so its fold
/// factor is 0 in f32, and one value row of NaN: every output of key head 0
/// is NaN, not the finite attention of the other segments.
pub fn attn_fold_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    let scene = scene_keys(UNDERFLOW_KEYS, false);
    require_live_append(&scene)?;
    let segments = scene.segments;
    if segments < 3 {
        return Err(format!(
            "{UNDERFLOW_KEYS} keys split into {segments} segments, the scene needs 3"
        ));
    }
    let n_keys =
        usize::try_from(scene.n_keys).map_err(|_| "a key count does not fit".to_string())?;
    let pos = usize::try_from(scene.pos).map_err(|_| "a position does not fit".to_string())?;
    let span = segment_span(n_keys, segments);
    let (lo, hi) = (span, 2 * span);
    let nan_row = lo + span / 2;
    let (q, k, v) = inputs(attn, &scene);
    let mut planted = plant(attn, &scene)?;
    let CacheBack::F16 {
        k: k_rows,
        v: v_rows,
    } = &mut planted
    else {
        return Err("the f16 scene planted another format".to_string());
    };
    let rope = tables
        .rope
        .get(pos * attn.rot..(pos + 1) * attn.rot)
        .ok_or("the rope row is past the table")?;
    let qn = norm_rope(
        take_qg_for(attn, &q, 0, 0)?,
        &tables.q_gain,
        tables.eps,
        rope,
    )?;
    // Head 0's keys of the segment are one row against its query, scaled to
    // the drop.
    let norm: f32 = qn.iter().map(|x| x * x).sum();
    let against: Vec<f32> = qn
        .iter()
        .map(|x| -(UNDERFLOW_DROP / attn.scale) * x / norm)
        .collect();
    let row = pack_f16(&against);
    if row.iter().any(|&bits| !dequant_f16(bits).is_finite()) {
        return Err("the segment's key row is past what f16 holds".to_string());
    }
    let score = |bits: &[u16]| -> f32 {
        attn.scale
            * qn.iter()
                .zip(bits)
                .map(|(a, &b)| a * dequant_f16(b))
                .sum::<f32>()
    };
    for key in lo..hi {
        k_rows[key * attn.head..(key + 1) * attn.head].copy_from_slice(&row);
    }
    v_rows[nan_row * attn.head..(nan_row + 1) * attn.head].fill(F16_NAN);
    // The planted rows outside the segment bound the head's maximum from
    // below (the append can only raise it).
    let others = (0..scene.prefill)
        .filter(|key| !(lo..hi).contains(key))
        .map(|key| score(&k_rows[key * attn.head..(key + 1) * attn.head]))
        .fold(f32::NEG_INFINITY, f32::max);
    let gap = others - score(&row);
    if gap.is_nan() || gap < UNDERFLOW_GAP {
        return Err(format!(
            "the segment scores {gap} under the other keys, not the {UNDERFLOW_GAP} an f32 weight needs"
        ));
    }
    let seen = launch(h, &tables, &scene, &q, &k, &v, &planted)?;
    expect_written_nan(&seen.out[..attn.gqa * attn.head]).map_err(|e| {
        format!("a NaN value of a segment whose factor underflows did not reach key head 0's outputs: {e}")
    })?;
    Ok(Verdict::Noted(format!(
        "{segments} segments of {span} keys, segment 1 scored {gap:.0} under the other keys on head 0, value row {nan_row} NaN: key head 0's {} heads are NaN",
        attn.gqa
    )))
}

/// The buffers of the refusal table.
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
    ticket: MetalBuffer<u32>,
}

/// The launch of one case: a head dim, a head count, a segment count, a cache
/// width or the query window, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    segments: usize,
    ctx: usize,
    q_window: usize,
}

impl Args {
    fn whole(attn: &AttnGeom, q_window: usize) -> Self {
        Self {
            head_dim: attn.head,
            query_heads: attn.query_heads,
            kv_heads: attn.kv_heads,
            segments: 1,
            ctx: CTX,
            q_window,
        }
    }
}

fn refuse(
    h: &Harness<'_>,
    b: &mut Bounds,
    tables: &Tables,
    a: &Args,
    encoder: &mut ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    let attn = &h.geom().attn;
    let wrap = |what: &str| DeviceError::Range {
        what: "attention",
        detail: format!("the {what} length wraps"),
    };
    let cache_n = attn.f16_len(CTX).ok_or_else(|| wrap("cache"))?;
    let partial_n = attn.partials_len(1).ok_or_else(|| wrap("partials"))?;
    let out_n = attn.out_len().ok_or_else(|| wrap("out"))?;
    h.kernels.attention_flash(
        encoder,
        AttentionParams {
            head_dim: a.head_dim,
            query_heads: a.query_heads,
            kv_heads: a.kv_heads,
            segments: a.segments,
            ctx: a.ctx,
            eps: tables.eps,
            layer: LAYER,
            pos: 63,
            n_keys: 64,
            q: b.q.window(Elems(0), Elems(a.q_window))?,
            k: b.k.window(Elems(0), Elems(attn.kv_heads * attn.head))?,
            v: b.v.window(Elems(0), Elems(attn.kv_heads * attn.head))?,
            q_gain: b.q_gain.window(Elems(0), Elems(attn.head))?,
            k_gain: b.k_gain.window(Elems(0), Elems(attn.head))?,
            rope: b.rope.window(Elems(0), Elems(CTX * attn.rot))?,
            cache: AttnCache::F16 {
                k: b.k_cache.window_mut(Elems(0), Elems(cache_n))?,
                v: b.v_cache.window_mut(Elems(0), Elems(cache_n))?,
            },
            partials: b.partials.window_mut(Elems(0), Elems(partial_n))?,
            out: b.out.window_mut(Elems(0), Elems(out_n))?,
            fault: h.device.fault_word(),
            ticket: b.ticket.window_mut(Elems(0), Elems(1))?,
        },
    )
}

/// One case of the table: the whole arguments with one of them made bad.
fn bad_arg<'a>(
    what: &'static str,
    needle: &'static str,
    edit: Box<dyn FnOnce(&mut Args) + 'a>,
    h: &'a Harness<'_>,
    tables: &'a Tables,
    whole: Args,
) -> BadArg<'a, Bounds> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b: &mut Bounds, encoder: &mut ComputeEncoder<'_>| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, b, tables, &a, encoder)
        }),
    )
}

/// A head dim, head count or window the kernel does not take is refused by name.
/// The same clause then runs one segment, the one-pass flash, so a key it
/// drops is red.
pub fn attn_bounds(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = h.geom().attn;
    let tables = load_tables_for(&attn, fixture)?;
    let (device, stream) = (h.device, h.stream);
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let mut bufs = Bounds {
        q: device.upload(stream, &vec![0f32; q_n]).map_err(why)?,
        k: device
            .upload(stream, &vec![0f32; attn.kv_heads * attn.head])
            .map_err(why)?,
        v: device
            .upload(stream, &vec![0f32; attn.kv_heads * attn.head])
            .map_err(why)?,
        q_gain: device.upload(stream, &tables.q_gain).map_err(why)?,
        k_gain: device.upload(stream, &tables.k_gain).map_err(why)?,
        rope: device.upload(stream, &tables.rope).map_err(why)?,
        partials: device
            .upload(stream, &vec![0f32; attn.partials_len(1).ok_or("partials")?])
            .map_err(why)?,
        out: device
            .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); out_n])
            .map_err(why)?,
        k_cache: device
            .upload(stream, &vec![F16_HOLE; attn.f16_len(CTX).ok_or("cache")?])
            .map_err(why)?,
        v_cache: device
            .upload(stream, &vec![F16_HOLE; attn.f16_len(CTX).ok_or("cache")?])
            .map_err(why)?,
        ticket: device.upload(stream, &[0u32]).map_err(why)?,
    };
    let whole = Args::whole(&attn, q_n);
    let case = |what: &'static str, needle: &'static str, edit: Box<dyn FnOnce(&mut Args)>| {
        bad_arg(what, needle, edit, h, &tables, whole)
    };
    let cases = vec![
        case(
            "half the head dim",
            "head dim",
            Box::new(move |a| a.head_dim = attn.head / 2),
        ),
        case(
            "half the query heads",
            "query heads",
            Box::new(move |a| a.query_heads = attn.query_heads / 2),
        ),
        case("one kv head", "kv heads", Box::new(|a| a.kv_heads = 1)),
        case(
            "a query window of one value",
            "window",
            Box::new(|a| a.q_window = 1),
        ),
        case(
            "a segment count of zero",
            "segments",
            Box::new(|a| a.segments = 0),
        ),
        case("a cache of no rows", "ctx", Box::new(|a| a.ctx = 0)),
    ];
    let unchanged =
        |h: &Harness<'_>, b: &Bounds| common::still(h, &b.out, |v| v.to_bits() == SENTINEL_BITS);
    common::refuse_launches(h, "attention", &mut bufs, cases, unchanged)?;
    // The well-formed launch is the one-segment scene below: the oracle holds
    // its every output, so an unwritten word is red there.
    let checked = run(h, &tables, &scene_one(1, false))?;
    Ok(Verdict::Noted(format!(
        "head dim, query heads, kv heads, window, segments and ctx refused, nothing written; S=1 worst ratio {:.4}",
        checked.worst
    )))
}

/// The decode-attention clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("attn_f16", Box::new(|| ctx.with(attn_f16))),
        ("attn_q8", Box::new(|| ctx.with(attn_q8))),
        (
            "attn_half",
            Box::new(|| match ctx.fixture {
                Ok(_fixture) => attn_half(ctx.harness, ctx.source),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("attn_fault", Box::new(|| ctx.with(attn_fault))),
        ("attn_fold_nan", Box::new(|| ctx.with(attn_fold_nan))),
        ("attn_bounds", Box::new(|| ctx.with(attn_bounds))),
    ]
}
