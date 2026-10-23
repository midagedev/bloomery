//! The decode-attention timing probe's body (macOS only).
//!
//! The method is the mat-vec probe's (`metal_matvec_time`): a timed command
//! buffer holds many dispatches, its time is its own `GPUEndTime` minus
//! `GPUStartTime`, read after it completed, and the figure is the median
//! over the buffers. Two arms time the production launches through
//! `Kernels`: `packed`, many dispatches in one serial command buffer with a
//! memory barrier between each (the steady-state kernel time), and `lone`,
//! one dispatch in its own command buffer (what a session sees when each
//! group is its own buffer). Phase arms split the flash's time: a probe
//! kernel compiled from the shipped source beside it, the shipped phases
//! gated at run time by one word — append, q prepare, tile loads, tile
//! compute — so `full − no_append` is the append and so on. The merge is
//! timed the same two ways past one segment.
//!
//! Each row is one line, tab-separated: kernel, keys, arm, the GPU
//! microseconds per dispatch (`us_median`, `us_p90`), the K/V bytes one
//! dispatch reads and their GB/s. A development-loop instrument: a figure it
//! prints is this machine's, taken with no timing lease, and is not a
//! published number.
//!
//! Flags: `--trials N` (timed command buffers per arm, default 16; `lone`
//! takes twice that), `--keys a,b,c` (default 100,512,1024,4096),
//! `--no-phases` (skip the phase arms), `--help`.

use std::process::ExitCode;
use std::time::Duration;

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::{Binding, ComputeEncoder};
use bloomery_metal::kernels::{
    AttentionParams, AttnCache, KV_HEADS, Kernels, QUERY_HEADS, SCALE, f16_len, out_len,
    partials_len, qg_len, segments_for_keys,
};
use bloomery_metal::library::{Library, Pipeline, SOURCE};
use bloomery_metal::{GpuInterval, MetalStream};

/// The flash threadgroup's width, `attention_geom::FLASH_THREADS`.
const FLASH_THREADS: usize = 256;

/// The cache rows the probe holds: the largest key count it times.
const CTX: u32 = 4096;

/// The warm-up buffers ahead of the timed ones.
const WARMUPS: usize = 2;

/// How long the host waits for one timed command buffer.
const BUFFER_WAIT: Duration = Duration::from_secs(30);

/// The bytes one timed command buffer's dispatches read, so every buffer
/// moves about the same total.
const PACKED_BYTES: usize = 64 << 20;

/// The layer word a raise is recorded in. Any word serves a probe that
/// raises nothing; the gate's own is 3.
const LAYER: u32 = 3;

const _: () = assert!(FLASH_THREADS == 256);

/// The probe kernel: the shipped flash's phases, gated at run time by one
/// word. The bodies are the shipped flash's, copied here so the shipped
/// kernel carries no switch; the statics it calls are the shipped ones, so
/// the split is of the production program. Compiled from [`SOURCE`] beside
/// the library, the way the kernel gate compiles its mutants.
const PROBE: &str = r#"
// The attention timing probe. Not a shipped kernel: the gate's library does
// not carry it. The phase word's bits: 1 the append, 2 the q prepare, 4 the
// tile loads, 8 the tile compute, 16 the fold off, 32 the epilogue off,
// 64 the tile loop off; 15 is the whole flash.
#pragma METAL fp contract(off)

struct AttnProbeArgs {
    uint phase;
};

static_assert(sizeof(AttnProbeArgs) == 4, "AttnProbeArgs is 4 bytes: the probe packs one word");

// The shipped tile loop with the row loads and the arithmetic gated apart.
// A gated compute leaves the running state the loop's entry state.
template <typename Cache>
static void probe_tile_loop(Cache cache,
                            constant AttnArgs &args,
                            bool ok,
                            uint kv,
                            uint seg,
                            uint span,
                            uint max_live,
                            uint tid,
                            uint lane,
                            uint phase,
                            thread float qv[8],
                            thread float *running_m,
                            thread float *running_l,
                            thread float acc[8]) {
    *running_m = -INFINITY;
    *running_l = 0.0f;
    for (uint d = 0u; d < 8u; ++d) {
        acc[d] = 0.0f;
    }
    if (!ok) {
        return;
    }
    uint lo0 = seg * span;
    uint seg_hi = min((seg + 1u) * span, max_live);
    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {
        uint hi = min(lo + BLOOMERY_ATTN_TILE, seg_hi);
        if ((phase & 4u) != 0u) {
            cache.load(tid, kv, args.ctx, lo, hi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scores[BLOOMERY_ATTN_TILE];
        bool live[BLOOMERY_ATTN_TILE];
        float tile_max = -INFINITY;
        uint nvalid = 0u;
        float kvv[8];
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            uint index = lo + t;
            bool on = index < hi && index < args.keys;
            live[t] = on;
            float dot = 0.0f;
            if (on && (phase & 4u) != 0u) {
                cache.key(t, lane, kvv);
            }
            if (on && (phase & 8u) != 0u) {
                for (uint d = 0u; d < 8u; ++d) {
                    dot = fma(qv[d], kvv[d], dot);
                }
            }
            dot = simd_sum(dot);
            float score = dot * args.scale;
            scores[t] = 0.0f;
            if (on) {
                scores[t] = score;
                tile_max = max(tile_max, score);
                nvalid += 1u;
            }
        }
        if (nvalid == 0u) {
            continue;
        }
        float new_m = max(*running_m, tile_max);
        float old_m = *running_m;
        float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
        *running_l *= factor;
        for (uint d = 0u; d < 8u; ++d) {
            acc[d] *= factor;
        }
        *running_m = new_m;
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            if (!live[t]) {
                continue;
            }
            if ((phase & 4u) != 0u) {
                cache.value(t, lane, kvv);
            }
            float weight = exp(scores[t] - new_m);
            *running_l += weight;
            for (uint d = 0u; d < 8u; ++d) {
                acc[d] = fma(weight, kvv[d], acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The shipped flash with its phases gated by the probe's word.
template <typename Cache>
static void probe_flash(Cache cache,
                        constant AttnArgs &args,
                        device float const *q,
                        device float const *k,
                        device float const *v,
                        device float const *q_gain,
                        device float const *k_gain,
                        device float const *rope,
                        device coherent(device) float *partials,
                        device float *out,
                        device atomic_uint *faults,
                        device atomic_uint *ticket,
                        uint groups,
                        uint phase,
                        uint group,
                        uint tid,
                        threadgroup float *scratch,
                        threadgroup uint &last_flag) {
    uint sg = tid / BLOOMERY_ATTN_LANES;
    uint lane = tid % BLOOMERY_ATTN_LANES;
    uint kv = group / args.segments;
    uint seg = group - kv * args.segments;
    bool active = kv < BLOOMERY_ATTN_KV && seg < args.segments;

    bool key_bad = args.keys == 0u || args.keys > args.ctx;
    bool pos_bad = args.pos >= args.ctx;
    bool ok = active && !key_bad && !pos_bad;
    uint max_live = key_bad ? 0u : args.keys;
    uint span = attn_span(max_live, args.segments);
    if (active) {
        if ((phase & 1u) != 0u) {
            attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, sg, lane,
                        scratch);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (active) {
        float qv[8];
        if ((phase & 2u) != 0u) {
            attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);
        } else {
            for (uint d = 0u; d < 8u; ++d) {
                qv[d] = 0.0f;
            }
        }
        float running_m;
        float running_l;
        float acc[8];
        if ((phase & 64u) == 0u) {
            probe_tile_loop(cache, args, ok, kv, seg, span, max_live, tid, lane, phase, qv,
                            &running_m, &running_l, acc);
        } else {
            running_m = -INFINITY;
            running_l = 0.0f;
            for (uint d = 0u; d < 8u; ++d) {
                acc[d] = 0.0f;
            }
        }
        if ((phase & 32u) != 0u) {
            attn_epilogue(args, q, partials, out, faults, ok, kv, seg, sg, lane, &running_m,
                          &running_l, acc);
        }
    }

    if (args.segments > 1u && (phase & 16u) == 0u) {
        ticket_is_last(ticket, groups, tid, last_flag);
        if (last_flag == 1u) {
            attn_fold(partials, ticket, args, q, out, faults, tid, (threadgroup float *)scratch);
        }
    }
}

kernel void probe_noop(uint tid [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[BLOOMERY_ATTN_SG * BLOOMERY_ATTN_HEAD];
    if (tid == 0u) {
        scratch[0] = float(group.x);
    }
}

// The ticket handshake alone: is_last's device barrier and atomic, the
// acquire fence, the reset.
kernel void probe_ticket(device atomic_uint *ticket [[buffer(0)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 tpg [[threadgroups_per_grid]]) {
    threadgroup uint last_flag = 0u;
    ticket_is_last(ticket, tpg.x, tid, last_flag);
    if (last_flag == 1u) {
        ticket_acquire();
        if (tid == 0u) {
            ticket_reset(ticket);
        }
    }
}

// N coherent loads of the fold's shape: a thread's 16 dims of one segment
// partial, N segments.
kernel void probe_fold_loads(device coherent(device) float *partials [[buffer(0)]],
                             constant AttnProbeArgs &probe [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             uint tid [[thread_index_in_threadgroup]]) {
    uint head = tid / 16u;
    uint dim0 = (tid % 16u) * 16u;
    float sink = 0.0f;
    for (uint j = 0u; j < probe.phase; ++j) {
        uint base = (head * 64u + j) * BLOOMERY_ATTN_PARTIAL;
        float mj = ticket_load(partials, base);
        float lj = ticket_load(partials, base + 1u);
        sink += mj + lj;
        for (uint d = 0u; d < 16u; ++d) {
            sink += ticket_load(partials, base + 2u + dim0 + d);
        }
    }
    if (sink == 12345.678f) {
        out[0] = sink;
    }
}

kernel void probe_flash_f16(device float const *q [[buffer(0)]],
                            device float const *k [[buffer(1)]],
                            device float const *v [[buffer(2)]],
                            device float const *q_gain [[buffer(3)]],
                            device float const *k_gain [[buffer(4)]],
                            device float const *rope [[buffer(5)]],
                            device ushort *cache_k [[buffer(6)]],
                            device ushort *cache_v [[buffer(7)]],
                            device coherent(device) float *partials [[buffer(8)]],
                            device float *out [[buffer(9)]],
                            constant AttnArgs &args [[buffer(10)]],
                            device atomic_uint *faults [[buffer(11)]],
                            device atomic_uint *ticket [[buffer(12)]],
                            constant AttnProbeArgs &probe [[buffer(13)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 tpg [[threadgroups_per_grid]]) {
    threadgroup uint4 shared[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 4];
    threadgroup uint last_flag = 0u;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = shared;
    cache.tile_v = shared + BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 8;
    probe_flash(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults, ticket, tpg.x,
                probe.phase, group.x, tid, (threadgroup float *)shared, last_flag);
}
"#;

/// One arm's figure: the per-dispatch microseconds over the timed buffers.
struct Timed {
    us_median: f64,
    us_p90: f64,
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Nearest rank of `p`: the value at `ceil(p * len)`, counting from one.
fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = (p * v.len() as f64).ceil() as usize;
    v[rank.saturating_sub(1)]
}

/// The buffers one attention launch reads and writes, planted once.
struct Bufs {
    q: bloomery_metal::buffer::MetalBuffer<f32>,
    k: bloomery_metal::buffer::MetalBuffer<f32>,
    v: bloomery_metal::buffer::MetalBuffer<f32>,
    q_gain: bloomery_metal::buffer::MetalBuffer<f32>,
    k_gain: bloomery_metal::buffer::MetalBuffer<f32>,
    rope: bloomery_metal::buffer::MetalBuffer<f32>,
    cache_k: bloomery_metal::buffer::MetalBuffer<u16>,
    cache_v: bloomery_metal::buffer::MetalBuffer<u16>,
    partials: bloomery_metal::buffer::MetalBuffer<f32>,
    out: bloomery_metal::buffer::MetalBuffer<f32>,
    ticket: bloomery_metal::buffer::MetalBuffer<u32>,
}

/// A small xorshift* the host plants its patterns with.
fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// One deterministic value of a planted pattern, in `[-span, span)`.
fn value(state: &mut u64, span: f32) -> f32 {
    let unit = (next(state) >> 40) as f32 / 16_777_216.0;
    (unit - 0.5) * 2.0 * span
}

/// The rope table, `ctx` rows of the interleaved cos/sin pairs of a base.
fn rope_rows(ctx: u32) -> Vec<f32> {
    let base = 10_000.0f64;
    let mut out = vec![0f32; ctx as usize * 64];
    for pos in 0..ctx {
        for i in 0..32 {
            let exp = -(i as f64) * 2.0 / 64.0;
            let theta = f64::from(pos) * base.powf(exp);
            let at = pos as usize * 64 + 2 * i;
            out[at] = theta.cos() as f32;
            out[at + 1] = theta.sin() as f32;
        }
    }
    out
}

/// The K/V bytes one flash dispatch reads: `keys` rows of K and V, both key
/// heads, f16.
fn kv_bytes(keys: u32) -> u64 {
    u64::from(keys) * 256 * 2 * 2 * KV_HEADS as u64
}

/// The attention's launch parameters over the planted buffers.
fn params<'w>(
    b: &'w mut Bufs,
    segments: usize,
    pos: u32,
    keys: u32,
    fault: &'w bloomery_metal::fault::MetalFaultWord,
) -> Result<AttentionParams<'w>, DeviceError> {
    let parts = partials_len(segments).ok_or_else(|| DeviceError::Range {
        what: "attention",
        detail: format!("the partials of {segments} segments do not fit"),
    })?;
    let cache_n = f16_len(CTX as usize).ok_or_else(|| DeviceError::Range {
        what: "attention",
        detail: "the cache length wraps".to_string(),
    })?;
    Ok(AttentionParams {
        head_dim: 256,
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        segments,
        ctx: CTX as usize,
        eps: 1e-6,
        layer: LAYER,
        pos,
        n_keys: keys,
        q: b.q
            .window(Elems(0), Elems(qg_len().expect("the query length wraps")))?,
        k: b.k.window(Elems(0), Elems(KV_HEADS * 256))?,
        v: b.v.window(Elems(0), Elems(KV_HEADS * 256))?,
        q_gain: b.q_gain.window(Elems(0), Elems(256))?,
        k_gain: b.k_gain.window(Elems(0), Elems(256))?,
        rope: b.rope.window(Elems(0), Elems(CTX as usize * 64))?,
        cache: AttnCache::F16 {
            k: b.cache_k.window_mut(Elems(0), Elems(cache_n))?,
            v: b.cache_v.window_mut(Elems(0), Elems(cache_n))?,
        },
        partials: b.partials.window_mut(Elems(0), Elems(parts))?,
        out: b
            .out
            .window_mut(Elems(0), Elems(out_len().expect("the out length wraps")))?,
        fault,
        ticket: b.ticket.window_mut(Elems(0), Elems(1))?,
    })
}

/// Encode one timed command buffer per repeat, submit them back to back and
/// read each buffer's GPU interval. `encode` launches dispatch `n`, counted
/// across the buffers. The warm-up buffers are not part of the figure.
fn time_arm(
    device: &MetalDevice,
    stream: &MetalStream,
    dispatches: usize,
    trials: usize,
    mut encode: impl FnMut(&mut ComputeEncoder<'_>, usize) -> Result<(), DeviceError>,
) -> Result<Timed, String> {
    if dispatches == 0 || trials == 0 {
        return Err("a timed arm needs a dispatch and a repeat".to_string());
    }
    let buffers = WARMUPS + trials;
    let mut pending = Vec::with_capacity(buffers);
    for _ in 0..buffers {
        let mut recording = stream.begin().map_err(why)?;
        let stamps = recording.stamps();
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            for n in 0..dispatches {
                if n > 0 {
                    encoder.memory_barrier();
                }
                encode(&mut encoder, n).map_err(why)?;
            }
            encoder.end();
        }
        pending.push((recording, stamps));
    }
    let mut all_stamps = Vec::with_capacity(buffers);
    for (recording, stamps) in pending {
        stream.submit(recording).map_err(why)?;
        all_stamps.push(stamps);
    }
    let mut gaps = Vec::with_capacity(trials);
    for (i, stamps) in all_stamps.iter().enumerate() {
        let gpu = match stamps.interval(BUFFER_WAIT) {
            Ok(GpuInterval::Ran(gpu)) => gpu,
            Ok(GpuInterval::StillRunning) => {
                return Err(
                    "the command buffer was still running on the GPU when the wait ended"
                        .to_string(),
                );
            }
            Err(e) => {
                let _drained = stream.sync();
                return Err(why(e));
            }
        };
        if i >= WARMUPS {
            gaps.push(gpu / dispatches as f64 * 1e6);
        }
    }
    stream.sync().map_err(why)?;
    if let Some(fault) = device.fault_word().read(stream).map_err(why)? {
        return Err(format!("the launch raised {fault}"));
    }
    device.fault_word().clear(stream).map_err(why)?;
    Ok(Timed {
        us_median: median(&gaps),
        us_p90: percentile(&gaps, 0.9),
    })
}

/// One probe kernel dispatch, bound by hand: the same buffers the flash
/// launch binds, in its order, with the phase word behind the ticket.
fn encode_phase(
    encoder: &mut ComputeEncoder<'_>,
    pipeline: &Pipeline,
    b: &mut Bufs,
    fault: &bloomery_metal::fault::MetalFaultWord,
    args: &[[u8; 4]; 7],
    phase: u32,
    segments: usize,
) -> Result<(), DeviceError> {
    let qg = qg_len().expect("the query length wraps");
    let parts = partials_len(segments).ok_or_else(|| DeviceError::Range {
        what: "attention",
        detail: format!("the partials of {segments} segments do not fit"),
    })?;
    let cache_n = f16_len(CTX as usize).ok_or_else(|| DeviceError::Range {
        what: "attention",
        detail: "the cache length wraps".to_string(),
    })?;
    let out_n = out_len().expect("the out length wraps");
    let mut words: Vec<u8> = Vec::with_capacity(28);
    for w in args {
        words.extend_from_slice(w);
    }
    let probe = phase.to_ne_bytes();
    encoder.set_pipeline(pipeline);
    encoder.set_binding(0, Binding::of(&b.q.window(Elems(0), Elems(qg))?)?)?;
    encoder.set_binding(
        1,
        Binding::of(&b.k.window(Elems(0), Elems(KV_HEADS * 256))?)?,
    )?;
    encoder.set_binding(
        2,
        Binding::of(&b.v.window(Elems(0), Elems(KV_HEADS * 256))?)?,
    )?;
    encoder.set_binding(3, Binding::of(&b.q_gain.window(Elems(0), Elems(256))?)?)?;
    encoder.set_binding(4, Binding::of(&b.k_gain.window(Elems(0), Elems(256))?)?)?;
    encoder.set_binding(
        5,
        Binding::of(&b.rope.window(Elems(0), Elems(CTX as usize * 64))?)?,
    )?;
    encoder.set_binding(
        6,
        Binding::of_mut(&b.cache_k.window_mut(Elems(0), Elems(cache_n))?)?,
    )?;
    encoder.set_binding(
        7,
        Binding::of_mut(&b.cache_v.window_mut(Elems(0), Elems(cache_n))?)?,
    )?;
    encoder.set_binding(
        8,
        Binding::of_mut(&b.partials.window_mut(Elems(0), Elems(parts))?)?,
    )?;
    encoder.set_binding(
        9,
        Binding::of_mut(&b.out.window_mut(Elems(0), Elems(out_n))?)?,
    )?;
    encoder.set_binding(10, Binding::bytes(&words)?)?;
    encoder.set_binding(11, fault.binding())?;
    encoder.set_binding(
        12,
        Binding::of_mut(&b.ticket.window_mut(Elems(0), Elems(1))?)?,
    )?;
    encoder.set_binding(13, Binding::bytes(&probe)?)?;
    encoder.dispatch_groups(KV_HEADS * segments, 1, FLASH_THREADS)
}

/// The flash's argument words, in the launch module's order.
fn args_words(segments: u32, pos: u32, keys: u32) -> [[u8; 4]; 7] {
    [
        segments,
        CTX,
        LAYER,
        1e-6f32.to_bits(),
        SCALE.to_bits(),
        pos,
        keys,
    ]
    .map(u32::to_ne_bytes)
}

/// The dispatches a packed arm holds, so a buffer moves about
/// [`PACKED_BYTES`].
fn packed_dispatches(keys: u32) -> usize {
    (PACKED_BYTES / kv_bytes(keys) as usize).clamp(4, 4096)
}

fn help() {
    println!(
        "\
metal_attention_time: development-loop timings of the decode attention.

A number this prints is this machine's, from no timing lease, and is not a
published result. Rows are tab-separated: kernel, keys, arm, us_median,
us_p90, kv_bytes, gbps.

  --help        print this and exit 0
  --trials N    timed command buffers per arm (default 16; `lone` takes twice)
  --keys LIST   comma-separated key counts (default 100,512,1024,4096)
  --no-phases   skip the phase arms"
    );
}

struct Run {
    trials: usize,
    keys: Vec<u32>,
    phases: bool,
}

fn parse(args: &[String]) -> Result<Run, String> {
    let mut trials = 16usize;
    let mut keys = vec![100u32, 512, 1024, 4096];
    let mut phases = true;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = || args.get(i + 1);
        match arg {
            "--no-phases" => phases = false,
            "--trials" => {
                let value = next().ok_or("--trials takes a value")?;
                trials = value
                    .parse()
                    .map_err(|_| format!("--trials takes a whole number, not {value:?}"))?;
                if trials == 0 {
                    return Err("--trials takes at least one timed buffer".to_string());
                }
                i += 1;
            }
            "--keys" => {
                let value = next().ok_or("--keys takes a value")?;
                keys = value
                    .split(',')
                    .map(str::trim)
                    .map(|tok| {
                        let n: u32 = tok
                            .parse()
                            .map_err(|_| format!("--keys takes key counts, not {tok:?}"))?;
                        if n == 0 || n > CTX {
                            return Err(format!("a key count of {n} is not 1..={CTX}"));
                        }
                        Ok(n)
                    })
                    .collect::<Result<_, _>>()?;
                i += 1;
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; --help lists the arguments"
                ));
            }
        }
        i += 1;
    }
    Ok(Run {
        trials,
        keys,
        phases,
    })
}

fn print_row(kernel: &str, keys: u32, arm: &str, n: usize, t: &Timed) {
    let bytes = kv_bytes(keys);
    let gbps = bytes as f64 / t.us_median / 1000.0;
    println!(
        "attn\tkernel={kernel}\tkeys={keys}\tarm={arm}\tn={n}\tus_median={:.3}\tus_p90={:.3}\tkv_bytes={bytes}\tgbps={gbps:.2}",
        t.us_median, t.us_p90
    );
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        help();
        return ExitCode::SUCCESS;
    }
    let run = match parse(&args) {
        Ok(run) => run,
        Err(e) => {
            println!("metal_attention_time: {e}");
            return ExitCode::from(64);
        }
    };
    let device = match MetalDevice::open() {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("metal_attention_time: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("metal_attention_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let kernels = match Kernels::new(&device) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("metal_attention_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("metal_attention_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    match timed(&run, &device, &kernels, &info.name) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            println!("metal_attention_time: FAIL ({e})");
            ExitCode::FAILURE
        }
    }
}

/// The whole table. An arm that fails prints its row and turns the exit
/// code; the arms behind it still run.
fn timed(
    run: &Run,
    device: &MetalDevice,
    kernels: &Kernels,
    info_name: &str,
) -> Result<(), String> {
    let stream = device.stream();
    println!(
        "info\tinstrument=development-loop\tdevice={info_name}\tlib={}\ttrials={}\tlone_trials={}\twarmups={WARMUPS}",
        device.library().path().name(),
        run.trials,
        2 * run.trials,
    );

    // The planted buffers: a query with finite gates, one K and V row per
    // key head, gains of one, the rope table, and a cache of small
    // deterministic f16 rows.
    let mut state = 0x6B70_726F_6265u64;
    let qg = qg_len().expect("the query length wraps");
    let mut q_host = (0..qg).map(|_| value(&mut state, 0.5)).collect::<Vec<_>>();
    for head in 0..QUERY_HEADS {
        for dim in 0..256 {
            q_host[head * 512 + 256 + dim] = value(&mut state, 2.0);
        }
    }
    let k_host = (0..KV_HEADS * 256)
        .map(|_| value(&mut state, 0.5))
        .collect::<Vec<_>>();
    let v_host = (0..KV_HEADS * 256)
        .map(|_| value(&mut state, 0.5))
        .collect::<Vec<_>>();
    let gain_host = vec![1.0f32; 256];
    let mut cache_host = vec![0u16; f16_len(CTX as usize).expect("the cache length wraps")];
    {
        let mut cstate = 0x6B63_6163_6865u64;
        for word in &mut cache_host {
            *word = gguf::quant::f32_to_f16_bits(value(&mut cstate, 0.1));
        }
    }
    let partials_host = vec![0f32; partials_len(segments_for_keys(CTX) as usize).expect("fits")];
    let out_host = vec![0f32; out_len().expect("the out length wraps")];
    let upload = |host: &[f32]| -> Result<bloomery_metal::buffer::MetalBuffer<f32>, String> {
        device.upload(stream, host).map_err(why)
    };
    let mut bufs = Bufs {
        q: upload(&q_host)?,
        k: upload(&k_host)?,
        v: upload(&v_host)?,
        q_gain: upload(&gain_host)?,
        k_gain: upload(&gain_host)?,
        rope: upload(&rope_rows(CTX))?,
        cache_k: device.upload(stream, &cache_host).map_err(why)?,
        cache_v: device.upload(stream, &cache_host).map_err(why)?,
        partials: upload(&partials_host)?,
        out: upload(&out_host)?,
        ticket: device.upload(stream, &[0u32]).map_err(why)?,
    };
    device.fault_word().clear(stream).map_err(why)?;

    let mut failed = false;
    for &keys in &run.keys {
        let pos = keys - 1;
        let segments = segments_for_keys(keys) as usize;
        let tgs = KV_HEADS * segments;
        let args = args_words(segments as u32, pos, keys);
        let packed = packed_dispatches(keys);
        println!(
            "scene\tkeys={keys}\tpos={pos}\tsegments={segments}\tthreadgroups={tgs}\tthreads={FLASH_THREADS}\tpacked_n={packed}"
        );
        // The production flash: many dispatches in one buffer, then one per
        // buffer.
        for (name, dispatches, trials) in
            [("packed", packed, run.trials), ("lone", 1, 2 * run.trials)]
        {
            match time_arm(device, stream, dispatches, trials, |encoder, _| {
                let p = params(&mut bufs, segments, pos, keys, device.fault_word())?;
                kernels.attention_flash(encoder, p)
            }) {
                Ok(t) => print_row("gqa256_flash_f16", keys, name, dispatches, &t),
                Err(e) => {
                    println!("attn\tkernel=gqa256_flash_f16\tkeys={keys}\tarm={name}\tFAIL\t{e}");
                    failed = true;
                }
            }
        }
        if !run.phases {
            continue;
        }
        let point = Point {
            keys,
            segments,
            args: &args,
            packed,
            trials: run.trials,
        };
        let phases = phase_arms(device, stream, &mut bufs, &point);
        failed |= phases;
    }
    if failed {
        Err("a timed arm failed".to_string())
    } else {
        Ok(())
    }
}

/// One key count's launch shape, as the phase arms time it.
#[derive(Clone, Copy)]
struct Point<'a> {
    keys: u32,
    segments: usize,
    args: &'a [[u8; 4]; 7],
    packed: usize,
    trials: usize,
}

/// The phase arms of one key count: the probe kernel, the whole flash and
/// one phase off at a time, then the split. Whether any arm failed.
#[allow(clippy::type_complexity)]
fn phase_arms(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    point: &Point<'_>,
) -> bool {
    let Point {
        keys,
        segments,
        args,
        packed,
        trials,
    } = *point;
    // The phase word's bits: 1 the append, 2 the q prepare, 4 the tile
    // loads, 8 the tile compute, 16 the fold off, 32 the epilogue off,
    // 64 the tile loop off. 15 is the whole flash.
    let phases: [(&str, u32); 6] = [
        ("full", 15),
        ("no_append", 14),
        ("no_prepare", 13),
        ("loads_only", 7),
        ("no_tile", 3),
        ("no_fold", 15 | 16),
    ];
    let library = match Library::from_source_text(device, &format!("{SOURCE}{PROBE}")) {
        Ok(library) => library,
        Err(e) => {
            println!(
                "attn\tkernel=probe_flash_f16\tkeys={keys}\tarm=phase\tFAIL\t{}",
                why(e)
            );
            return true;
        }
    };
    let pipeline = match library.pipeline("probe_flash_f16") {
        Ok(pipeline) => pipeline,
        Err(e) => {
            println!(
                "attn\tkernel=probe_flash_f16\tkeys={keys}\tarm=phase\tFAIL\t{}",
                why(e)
            );
            return true;
        }
    };
    let mut figures: Vec<(&str, f64)> = Vec::new();
    let mut failed = false;
    for (name, phase) in phases {
        // A fresh ticket an arm: the fold resets it, but the arms that skip
        // the fold leave it raised, and the next arm must start from zero.
        bufs.ticket = match device.upload(stream, &[0u32]) {
            Ok(ticket) => ticket,
            Err(e) => {
                println!(
                    "attn\tkernel=probe_flash_f16\tkeys={keys}\tarm={name}\tFAIL\t{}",
                    why(e)
                );
                return true;
            }
        };
        match time_arm(device, stream, packed, trials, |encoder, _| {
            encode_phase(
                encoder,
                &pipeline,
                bufs,
                device.fault_word(),
                args,
                phase,
                segments,
            )
        }) {
            Ok(t) => {
                print_row("probe_flash_f16", keys, name, packed, &t);
                figures.push((name, t.us_median));
            }
            Err(e) => {
                println!("attn\tkernel=probe_flash_f16\tkeys={keys}\tarm={name}\tFAIL\t{e}");
                failed = true;
            }
        }
    }
    let at = |name: &str| figures.iter().find(|(n, _)| *n == name).map(|(_, us)| *us);
    if let (Some(full), Some(no_append), Some(no_prepare), Some(loads_only), Some(no_tile)) = (
        at("full"),
        at("no_append"),
        at("no_prepare"),
        at("loads_only"),
        at("no_tile"),
    ) {
        println!(
            "split\tkeys={keys}\tappend_us={:.3}\tprepare_us={:.3}\ttile_us={:.3}\tloads_us={:.3}\tcompute_us={:.3}\tskeleton_us={:.3}",
            full - no_append,
            full - no_prepare,
            full - no_tile,
            loads_only - no_tile,
            no_append - loads_only,
            no_tile,
        );
    }
    failed
}
