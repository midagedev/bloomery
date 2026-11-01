//! The decode-attention timing probe's body (macOS only).
//!
//! The method is the mat-vec probe's (`metal_matvec_time`): a timed command
//! buffer holds many dispatches, its time is its own `GPUEndTime` minus
//! `GPUStartTime`, read after it completed, and the figure is the median
//! over the buffers. The buffers go one at a time (`shared/gpu_time.rs`:
//! committed, waited, read), so no buffer's interval holds a neighbour's
//! queue time. Two arms time the production launches through `Kernels`:
//! `packed`, many dispatches in one serial command buffer with a memory
//! barrier between each (the steady-state kernel time), and `lone`, one
//! dispatch in its own command buffer (what a session sees when each group is
//! its own buffer). Phase arms split the flash's time: a probe kernel
//! compiled from the shipped source beside it, the shipped phases gated at run
//! time by one word — append, q prepare, the tile loop, the epilogue, the
//! ticket, the fold — so `full − no_append` is the append and so on. Before the phase arms run, a self-check holds the probe to the shipped
//! flash: the arm table says what each name does to the phase word
//! ([`check_arms`]), the `full` arm's output equals the shipped dispatch's
//! bit for bit, and the epilogue bit decides whether the partials are
//! written. A probe that fails it prints no phase row.
//!
//! The phase arms time in one of two units (`--phase-arms`): `lone`, one
//! dispatch to a command buffer like the shipped `lone` arm (the default),
//! or `packed`, the shipped `packed` arm's many dispatches to a buffer. The
//! split's `probe_over_shipped` divides by the shipped arm of the same unit.
//!
//! Each row is one line, tab-separated: kernel, keys, arm, the GPU
//! microseconds per dispatch (`us_median`, `us_min`, `us_p90`), the K/V bytes
//! one dispatch reads and their GB/s. A development-loop instrument: a figure
//! it prints is this machine's, taken with no timing lease, and is not a
//! published number.
//!
//! Flags: `--trials N` (timed command buffers per arm, default 16; `lone`
//! takes twice that), `--keys a,b,c` (default 100,512,1024,4096), `--segs
//! a,b,c` (segment counts timed at every key count, in place of the rule's),
//! `--phase-arms lone|packed` (the phase arms' unit, default `lone`),
//! `--no-phases` (skip the phase arms and the self-check), `--no-mlx` (skip
//! MLX's kernels, `mlx.rs`), `--variants` (the source variants and the merge
//! dispatch), `--help`. The shapes are the build's host instance's: its head,
//! query and key heads, its rope width and whether its query carries a gate.

use std::process::ExitCode;

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord, Stream};
use bloomery_metal::MetalStream;
use bloomery_metal::attention_geom::{FLASH_THREADS, ONE_PASS_THREADS};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::{Binding, ComputeEncoder};
use bloomery_metal::instance::HOST;
use bloomery_metal::kernels::{
    AttentionParams, AttnCache, HEAD_DIM, KV_HEADS, Kernels, QG_STRIDE, QUERY_HEADS, ROT, SCALE,
    SEGMENT_CAP, f16_len, out_len, partials_len, qg_len, segments_for_keys,
};
use bloomery_metal::library::{Library, Pipeline, SOURCE};

#[path = "../shared/gpu_time.rs"]
mod gpu_time;
#[path = "mlx.rs"]
mod mlx;

use gpu_time::{Timed, WARMUPS, time_arm};

/// The cache rows the probe holds: the largest key count it times.
const CTX: u32 = 16384;

/// The most segments the probe's buffers hold: the fold's one round of
/// lanes. Past the host's [`SEGMENT_CAP`] only the source variants time, the
/// shipped launch refusing the count.
const PROBE_SEG_MAX: usize = 32;

/// The bytes one timed command buffer's dispatches read, so every buffer
/// moves about the same total.
const PACKED_BYTES: usize = 64 << 20;

/// The layer word a raise is recorded in. Any word serves a probe that
/// raises nothing; the gate's own is 3.
const LAYER: u32 = 3;

/// The key count the self-check launches at, and its segments: past one
/// segment, so the flash writes partials and a fold follows, with every
/// segment holding live keys.
const CHECK_KEYS: u32 = 1024;
const CHECK_SEGMENTS: usize = 16;

/// The phase word's bits. The probe source reads them as the `PROBE_PHASE_*`
/// macros [`phase_macros`] writes in front of it and the arm table builds its
/// words from these constants, so a bit means one thing in the kernel and in
/// the table.
const PH_APPEND: u32 = 1;
const PH_PREPARE: u32 = 2;
/// Set: the ticket handshake and the fold are skipped.
const PH_NO_FOLD: u32 = 16;
/// Set: the epilogue, the partial or output write, is skipped.
const PH_NO_EPILOGUE: u32 = 32;
/// Set: the tile loop is skipped.
const PH_NO_LOOP: u32 = 64;
/// Set: the ticket handshake runs and the fold's body is skipped (the last
/// threadgroup only resets the ticket).
const PH_TICKET_ONLY: u32 = 128;
/// Set: the segment's keys stream from the cache straight into each SIMD
/// group's registers (the one-pass flash's walk over the segment) in place of
/// the threadgroup tiles. Not a phase of the shipped flash: a candidate loop.
const PH_DIRECT: u32 = 256;
/// The shipped flash: the phases on, nothing skipped.
const PH_FULL: u32 = PH_APPEND | PH_PREPARE;

/// The phase arms, each a name and a phase word. The two bits that empty the
/// partials (the tile loop, the epilogue) change what the fold reads, so the
/// pairs that toggle them keep the fold off in both arms.
const ARMS: [(&str, u32); 11] = [
    ("full", PH_FULL),
    ("no_append", PH_FULL & !PH_APPEND),
    ("no_prepare", PH_FULL & !PH_PREPARE),
    ("no_loop", PH_FULL | PH_NO_LOOP),
    ("no_loop_no_fold", PH_FULL | PH_NO_LOOP | PH_NO_FOLD),
    ("no_fold", PH_FULL | PH_NO_FOLD),
    ("no_epilogue", PH_FULL | PH_NO_EPILOGUE),
    ("no_fold_no_epilogue", PH_FULL | PH_NO_FOLD | PH_NO_EPILOGUE),
    ("ticket_only", PH_FULL | PH_TICKET_ONLY),
    ("direct", PH_FULL | PH_DIRECT),
    ("direct_no_fold", PH_FULL | PH_DIRECT | PH_NO_FOLD),
];

/// The arms the split row reads, in the order [`phase_arms`] binds them.
const SPLIT_READS: [&str; 9] = [
    "full",
    "no_append",
    "no_prepare",
    "no_loop",
    "no_loop_no_fold",
    "no_fold",
    "no_epilogue",
    "no_fold_no_epilogue",
    "ticket_only",
];

/// The bits whose set state skips a phase.
const SKIP_BITS: u32 = PH_NO_FOLD | PH_NO_EPILOGUE | PH_NO_LOOP | PH_TICKET_ONLY;

/// The phase word an arm's name says. `full` is [`PH_FULL`] and `ticket_only`
/// adds [`PH_TICKET_ONLY`]. `no_<phase>`, joined by `_no_`, turns each named
/// phase off from [`PH_FULL`]: the append and the prepare by clearing their
/// bit, the loop, the fold and the epilogue by setting their skip bit.
/// `direct`, alone or before `_no_<phase>`, adds [`PH_DIRECT`] to the word the
/// rest names.
fn word_named(name: &str) -> Result<u32, String> {
    if let Some(rest) = name.strip_prefix("direct") {
        let base = match rest {
            "" => "full",
            _ => rest
                .strip_prefix('_')
                .ok_or_else(|| format!("arm {name}: `direct` is followed by `_no_<phase>`"))?,
        };
        return word_named(base).map(|word| word | PH_DIRECT);
    }
    match name {
        "full" => return Ok(PH_FULL),
        "ticket_only" => return Ok(PH_FULL | PH_TICKET_ONLY),
        _ => {}
    }
    let Some(phases) = name.strip_prefix("no_") else {
        return Err(format!(
            "arm {name}: a name is `full`, `ticket_only` or `no_<phase>`"
        ));
    };
    let mut word = PH_FULL;
    for phase in phases.split("_no_") {
        match phase {
            "append" => word &= !PH_APPEND,
            "prepare" => word &= !PH_PREPARE,
            "loop" => word |= PH_NO_LOOP,
            "fold" => word |= PH_NO_FOLD,
            "epilogue" => word |= PH_NO_EPILOGUE,
            other => return Err(format!("arm {name}: there is no phase `{other}`")),
        }
    }
    Ok(word)
}

/// Holds an arm table to the phase bits' meaning: the bits are one each and
/// distinct, `full` is the shipped flash's work (the append and the prepare on,
/// no skip bit set, so the epilogue runs), every arm's word is the one its
/// name says ([`word_named`]), no name or word repeats, and every arm the
/// split row reads is in the table. Every failed clause is named.
fn check_arms(arms: &[(&str, u32)]) -> Result<(), String> {
    let mut failures: Vec<String> = Vec::new();
    let mut seen = 0u32;
    for bit in [
        PH_APPEND,
        PH_PREPARE,
        PH_NO_FOLD,
        PH_NO_EPILOGUE,
        PH_NO_LOOP,
        PH_TICKET_ONLY,
        PH_DIRECT,
    ] {
        if !bit.is_power_of_two() || (seen & bit) != 0 {
            failures.push(format!("phase bit {bit} is not a bit of its own"));
        }
        seen |= bit;
    }
    if (PH_FULL & SKIP_BITS) != 0
        || (PH_FULL & (PH_APPEND | PH_PREPARE)) != (PH_APPEND | PH_PREPARE)
    {
        failures.push(format!(
            "full word {PH_FULL} is not the shipped flash: the append and the prepare on, no skip bit"
        ));
    }
    for (i, (name, word)) in arms.iter().enumerate() {
        match word_named(name) {
            Ok(said) if said == *word => {}
            Ok(said) => failures.push(format!(
                "arm {name}: the table's word is {word}, its name says {said}"
            )),
            Err(e) => failures.push(e),
        }
        if arms[..i].iter().any(|(other, _)| other == name) {
            failures.push(format!("arm {name} stands twice"));
        }
        if let Some((other, _)) = arms[..i].iter().find(|(_, w)| w == word) {
            failures.push(format!("arms {other} and {name} share the word {word}"));
        }
    }
    for name in SPLIT_READS {
        if !arms.iter().any(|(arm, _)| *arm == name) {
            failures.push(format!(
                "the split row reads arm {name}, which the table lacks"
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// The macros the probe source reads the phase word with.
fn phase_macros() -> String {
    [
        ("APPEND", PH_APPEND),
        ("PREPARE", PH_PREPARE),
        ("NO_FOLD", PH_NO_FOLD),
        ("NO_EPILOGUE", PH_NO_EPILOGUE),
        ("NO_LOOP", PH_NO_LOOP),
        ("TICKET_ONLY", PH_TICKET_ONLY),
        ("DIRECT", PH_DIRECT),
    ]
    .iter()
    .map(|(name, bit)| format!("#define PROBE_PHASE_{name} {bit}u\n"))
    .collect()
}

/// The shipped source with the probe kernel and MLX's kernels after it, the
/// way the kernel gate compiles its mutants.
fn probe_source() -> String {
    format!("{SOURCE}\n{}{PROBE}\n{}", phase_macros(), mlx::SOURCE)
}

/// The probe kernel: the shipped flash's phases, gated at run time by one
/// word. The bodies are the shipped flash's, copied here so the shipped
/// kernel carries no switch; the statics it calls are the shipped ones, so
/// the split is of the production program, which the self-check holds it to.
const PROBE: &str = r#"
// The attention timing probe. Not a shipped kernel: the gate's library does
// not carry it. The phase word's bits are the PROBE_PHASE_* macros written in
// front of this text.
#pragma METAL fp contract(off)

struct AttnProbeArgs {
    uint phase;
};

static_assert(sizeof(AttnProbeArgs) == 4, "AttnProbeArgs is 4 bytes: the probe packs one word");

// The candidate walk: the segmented flash's grid, each SIMD group (a query
// head) streaming the segment's keys from the cache into registers four at a
// time with the one-pass flash's arithmetic, no threadgroup tiles and no
// barrier. The f16 cache only.
#define PROBE_CHUNK 4u
static void probe_direct_walk(device ushort *cache_k,
                              device ushort *cache_v,
                              thread const AttnArgs &args,
                              bool ok,
                              uint kv,
                              uint seg,
                              uint span,
                              uint max_live,
                              uint lane,
                              thread float qv[BLOOMERY_ATTN_LANE_DIMS],
                              thread float *running_m,
                              thread float *running_l,
                              thread float acc[BLOOMERY_ATTN_LANE_DIMS]) {
    float m = -INFINITY;
    float l = 0.0f;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        acc[d] = 0.0f;
    }
    if (ok) {
        uint lo0 = seg * span;
        uint seg_hi = min((seg + 1u) * span, max_live);
        const device attn_lane_word *ck =
            (const device attn_lane_word *)cache_k + kv * args.ctx * BLOOMERY_ATTN_LANES + lane;
        const device attn_lane_word *cv =
            (const device attn_lane_word *)cache_v + kv * args.ctx * BLOOMERY_ATTN_LANES + lane;
        for (uint base = lo0; base < seg_hi; base += PROBE_CHUNK) {
            attn_lane_word wk[PROBE_CHUNK];
            attn_lane_word wv[PROBE_CHUNK];
            for (uint c = 0u; c < PROBE_CHUNK; ++c) {
                uint j = base + c;
                attn_lane_word a = 0u;
                attn_lane_word b = 0u;
                if (j < seg_hi) {
                    a = ck[j * BLOOMERY_ATTN_LANES];
                    b = cv[j * BLOOMERY_ATTN_LANES];
                }
                wk[c] = a;
                wv[c] = b;
            }
            float s[PROBE_CHUNK];
            float chunk_max = -INFINITY;
            for (uint c = 0u; c < PROBE_CHUNK; ++c) {
                float kk[BLOOMERY_ATTN_LANE_DIMS];
                attn_widen(wk[c], kk);
                float dot = 0.0f;
                for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                    dot = fma(qv[d], kk[d], dot);
                }
                dot = simd_sum(dot);
                float score = dot * args.scale;
                s[c] = base + c < seg_hi ? score : -INFINITY;
                chunk_max = max(chunk_max, s[c]);
            }
            float new_m = max(m, chunk_max);
            float factor = m > -INFINITY ? exp(m - new_m) : 0.0f;
            l *= factor;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                acc[d] *= factor;
            }
            m = new_m;
            for (uint c = 0u; c < PROBE_CHUNK; ++c) {
                float vv[BLOOMERY_ATTN_LANE_DIMS];
                attn_widen(wv[c], vv);
                float weight = base + c < seg_hi ? exp(s[c] - new_m) : 0.0f;
                l += weight;
                for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                    acc[d] = fma(weight, vv[d], acc[d]);
                }
            }
        }
    }
    *running_m = m;
    *running_l = l;
}

// The shipped flash with its phases gated by the probe's word.
template <typename Cache>
static void probe_flash(Cache cache,
                        thread const AttnArgs &args,
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

    uint max_live;
    bool ok = attn_bounds(args, faults, active, tid, max_live);
    uint span = attn_span(max_live, args.segments);
    if (active && (phase & PROBE_PHASE_APPEND) != 0u) {
        attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, sg, lane,
                    scratch);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (active) {
        float qv[BLOOMERY_ATTN_LANE_DIMS];
        if ((phase & PROBE_PHASE_PREPARE) != 0u) {
            attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);
        } else {
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                qv[d] = 0.0f;
            }
        }
        float running_m;
        float running_l;
        float acc[BLOOMERY_ATTN_LANE_DIMS];
        if ((phase & PROBE_PHASE_NO_LOOP) == 0u && (phase & PROBE_PHASE_DIRECT) != 0u) {
            probe_direct_walk(cache.k, cache.v, args, ok, kv, seg, span, max_live, lane, qv,
                              &running_m, &running_l, acc);
        } else if ((phase & PROBE_PHASE_NO_LOOP) == 0u) {
            attn_tile_loop(cache, args, ok, kv, seg, span, max_live, lane, qv, &running_m,
                           &running_l, acc);
        } else {
            running_m = -INFINITY;
            running_l = 0.0f;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                acc[d] = 0.0f;
            }
        }
        if ((phase & PROBE_PHASE_NO_EPILOGUE) == 0u) {
            attn_epilogue(args, q, partials, out, faults, ok, kv, seg, sg, lane, &running_m,
                          &running_l, acc);
        }
    }

    if (args.segments > 1u && active && (phase & PROBE_PHASE_NO_FOLD) == 0u) {
        ticket_is_last(ticket + kv, args.segments, tid, last_flag);
        if (last_flag == 1u) {
            if ((phase & PROBE_PHASE_TICKET_ONLY) != 0u) {
                if (tid == 0u) {
                    ticket_reset(ticket + kv);
                }
            } else {
                attn_fold(partials, ticket + kv, args, q, out, faults, tid, kv);
            }
        }
    }
}

ATTN_FLASH_HOLD
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
                            constant AttnArgs &launch [[buffer(10)]],
                            device atomic_uint *faults [[buffer(11)]],
                            device atomic_uint *ticket [[buffer(12)]],
                            constant AttnProbeArgs &probe [[buffer(13)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    probe_flash(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults, ticket,
                probe.phase, group.x, tid, (threadgroup float *)shared, last_flag);
}

// The fold as a dispatch of its own, after a flash whose fold is off (the
// `nofold` variant): a threadgroup a query head, a thread a value of it, plain
// loads (the dispatch boundary publishes the partials). Every thread reads
// the head's segment maxima and weight sums; the values add by fma in
// ascending segment order times exp(m - M). The packed gate is applied; an
// instance that gates per head writes the ungated value (a timing arm).
kernel void probe_merge_f16(device float const *q [[buffer(0)]],
                            device float const *partials [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant AttnArgs &args [[buffer(3)]],
                            device atomic_uint *faults [[buffer(4)]],
                            uint dim [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
    uint head = group.x;
    float big = -INFINITY;
    bool refused = false;
    for (uint s = 0u; s < args.segments; ++s) {
        uint base = (head * args.segments + s) * BLOOMERY_ATTN_PARTIAL;
        float m = partials[base];
        float l = partials[base + 1u];
        if (isnan(m) || isnan(l)) {
            refused = true;
        } else if (l > 0.0f) {
            big = max(big, m);
        }
    }
    float l = 0.0f;
    float acc = 0.0f;
    for (uint s = 0u; s < args.segments; ++s) {
        uint base = (head * args.segments + s) * BLOOMERY_ATTN_PARTIAL;
        float ls = partials[base + 1u];
        float f = !refused && ls > 0.0f ? exp(partials[base] - big) : 0.0f;
        l = fma(ls, f, l);
        acc = fma(partials[base + 2u + dim], f, acc);
    }
    uint o = head * BLOOMERY_ATTN_HEAD + dim;
    if (refused || !(l > 0.0f) || !isfinite(big)) {
        out[o] = NAN;
        return;
    }
    float attn = acc * (1.0f / l);
#if BLOOMERY_ATTN_GATED
    float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim];
    attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);
#else
    attn_commit_plain(out, o, attn);
#endif
}
"#;

/// The buffers one attention launch reads and writes, planted once.
struct Bufs {
    q: MetalBuffer<f32>,
    k: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    q_gain: MetalBuffer<f32>,
    k_gain: MetalBuffer<f32>,
    rope: MetalBuffer<f32>,
    cache_k: MetalBuffer<u16>,
    cache_v: MetalBuffer<u16>,
    partials: MetalBuffer<f32>,
    out: MetalBuffer<f32>,
    ticket: MetalBuffer<u32>,
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

/// The rope table, `ctx` rows of [`ROT`] values: the interleaved cos/sin
/// pairs of a base.
fn rope_rows(ctx: u32) -> Vec<f32> {
    let base = 10_000.0f64;
    let mut out = vec![0f32; ctx as usize * ROT];
    for pos in 0..ctx {
        for i in 0..ROT / 2 {
            let exp = -(i as f64) * 2.0 / ROT as f64;
            let theta = f64::from(pos) * base.powf(exp);
            let at = pos as usize * ROT + 2 * i;
            out[at] = theta.cos() as f32;
            out[at + 1] = theta.sin() as f32;
        }
    }
    out
}

/// The K/V bytes one flash dispatch reads: `keys` rows of K and V, every key
/// head, f16.
fn kv_bytes(keys: u32) -> u64 {
    u64::from(keys) * HEAD_DIM as u64 * 2 * 2 * KV_HEADS as u64
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
        head_dim: HEAD_DIM,
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
        k: b.k.window(Elems(0), Elems(KV_HEADS * HEAD_DIM))?,
        v: b.v.window(Elems(0), Elems(KV_HEADS * HEAD_DIM))?,
        q_gain: b.q_gain.window(Elems(0), Elems(HEAD_DIM))?,
        k_gain: b.k_gain.window(Elems(0), Elems(HEAD_DIM))?,
        rope: b.rope.window(Elems(0), Elems(CTX as usize * ROT))?,
        cache: AttnCache::F16 {
            k: b.cache_k.window_mut(Elems(0), Elems(cache_n))?,
            v: b.cache_v.window_mut(Elems(0), Elems(cache_n))?,
        },
        partials: b.partials.window_mut(Elems(0), Elems(parts))?,
        out: b
            .out
            .window_mut(Elems(0), Elems(out_len().expect("the out length wraps")))?,
        fault,
        ticket: b.ticket.window_mut(Elems(0), Elems(KV_HEADS))?,
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
        Binding::of(&b.k.window(Elems(0), Elems(KV_HEADS * HEAD_DIM))?)?,
    )?;
    encoder.set_binding(
        2,
        Binding::of(&b.v.window(Elems(0), Elems(KV_HEADS * HEAD_DIM))?)?,
    )?;
    encoder.set_binding(
        3,
        Binding::of(&b.q_gain.window(Elems(0), Elems(HEAD_DIM))?)?,
    )?;
    encoder.set_binding(
        4,
        Binding::of(&b.k_gain.window(Elems(0), Elems(HEAD_DIM))?)?,
    )?;
    encoder.set_binding(
        5,
        Binding::of(&b.rope.window(Elems(0), Elems(CTX as usize * ROT))?)?,
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
    // Every key head's ticket word: each key head's fold reads its own.
    encoder.set_binding(
        12,
        Binding::of_mut(&b.ticket.window_mut(Elems(0), Elems(KV_HEADS))?)?,
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
us_min, us_p90, kv_bytes, gbps.

  --help        print this and exit 0
  --trials N    timed command buffers per arm (default 16; `lone` takes twice)
  --keys LIST   comma-separated key counts (default 100,512,1024,4096)
  --segs LIST   comma-separated segment counts timed at every key count, in place of the rule's
                (at most 32; past the host's cap only the variants time)
  --phase-arms U  the phase arms' unit: `lone` (default), one dispatch to a command buffer like
                the shipped `lone` arm, or `packed`, many dispatches to a buffer like the shipped
                `packed` arm
  --no-phases   skip the phase arms and the self-check they need
  --no-mlx      skip MLX's kernels (sdpa_vector and its two passes) on the same cache
  --variants    time the segmented flash of each source variant (VARIANTS) at every split scene,
                and the fold as a merge dispatch after the `nofold` variant"
    );
}

/// What one timed command buffer of a phase arm holds.
#[derive(Clone, Copy)]
enum PhaseUnit {
    /// One dispatch, the shipped `lone` arm's unit.
    Lone,
    /// The packed rule's dispatches with a barrier between each, the shipped
    /// `packed` arm's unit.
    Packed,
}

impl PhaseUnit {
    fn name(self) -> &'static str {
        match self {
            Self::Lone => "lone",
            Self::Packed => "packed",
        }
    }
}

struct Run {
    trials: usize,
    keys: Vec<u32>,
    /// Segment counts to time at every key count, in place of the rule's.
    segs: Vec<usize>,
    phases: bool,
    phase_unit: PhaseUnit,
    mlx: bool,
    variants: bool,
}

fn parse(args: &[String]) -> Result<Run, String> {
    let mut trials = 16usize;
    let mut keys = vec![100u32, 512, 1024, 4096];
    let mut segs: Vec<usize> = Vec::new();
    let mut phases = true;
    let mut mlx = true;
    let mut variants = false;
    let mut phase_unit = PhaseUnit::Lone;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = || args.get(i + 1);
        match arg {
            "--no-phases" => phases = false,
            "--no-mlx" => mlx = false,
            "--variants" => variants = true,
            "--phase-arms" => {
                let value = next().ok_or("--phase-arms takes lone or packed")?;
                phase_unit = match value.as_str() {
                    "lone" => PhaseUnit::Lone,
                    "packed" => PhaseUnit::Packed,
                    other => {
                        return Err(format!("--phase-arms takes lone or packed, not {other:?}"));
                    }
                };
                i += 1;
            }
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
            "--segs" => {
                let value = next().ok_or("--segs takes a value")?;
                segs = value
                    .split(',')
                    .map(str::trim)
                    .map(|tok| {
                        let n: usize = tok
                            .parse()
                            .map_err(|_| format!("--segs takes segment counts, not {tok:?}"))?;
                        if n == 0 || n > PROBE_SEG_MAX {
                            return Err(format!(
                                "a segment count of {n} is not 1..={PROBE_SEG_MAX}"
                            ));
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
        segs,
        phases,
        phase_unit,
        mlx,
        variants,
    })
}

fn print_row(kernel: &str, keys: u32, arm: &str, n: usize, t: &Timed) {
    let bytes = kv_bytes(keys);
    let gbps = bytes as f64 / t.us_median / 1000.0;
    println!(
        "attn\tkernel={kernel}\tkeys={keys}\tarm={arm}\tn={n}\tus_median={:.3}\tus_min={:.3}\tus_p90={:.3}\tkv_bytes={bytes}\tgbps={gbps:.2}",
        t.us_median, t.us_min, t.us_p90
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
    let device = match MetalDevice::open(&HOST) {
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
        "info\tinstrument=development-loop\tdevice={info_name}\tlib={}\ttrials={}\tlone_trials={}\twarmups={WARMUPS}\tphase_arms={}",
        device.library().path().name(),
        run.trials,
        2 * run.trials,
        run.phase_unit.name(),
    );

    // The planted buffers: a query (with finite gates when the instance's
    // query carries them), one K and V row per key head, gains of one, the
    // rope table, and a cache of small deterministic f16 rows.
    let mut state = 0x6B70_726F_6265u64;
    let qg = qg_len().expect("the query length wraps");
    let mut q_host = (0..qg).map(|_| value(&mut state, 0.5)).collect::<Vec<_>>();
    if QG_STRIDE == 2 * HEAD_DIM {
        for head in 0..QUERY_HEADS {
            for dim in 0..HEAD_DIM {
                q_host[head * QG_STRIDE + HEAD_DIM + dim] = value(&mut state, 2.0);
            }
        }
    }
    let k_host = (0..KV_HEADS * HEAD_DIM)
        .map(|_| value(&mut state, 0.5))
        .collect::<Vec<_>>();
    let v_host = (0..KV_HEADS * HEAD_DIM)
        .map(|_| value(&mut state, 0.5))
        .collect::<Vec<_>>();
    let gain_host = vec![1.0f32; HEAD_DIM];
    let mut cache_host = vec![0u16; f16_len(CTX as usize).expect("the cache length wraps")];
    {
        let mut cstate = 0x6B63_6163_6865u64;
        for word in &mut cache_host {
            *word = gguf::quant::f32_to_f16_bits(value(&mut cstate, 0.1));
        }
    }
    let partials_host = vec![0f32; partials_len(PROBE_SEG_MAX).expect("fits")];
    let out_host = vec![0f32; out_len().expect("the out length wraps")];
    let upload = |host: &[f32]| -> Result<MetalBuffer<f32>, String> {
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
        ticket: device.upload(stream, &[0u32; KV_HEADS]).map_err(why)?,
    };
    device.fault_word().clear(stream).map_err(why)?;

    let mut failed = false;
    let library = if run.phases || run.mlx {
        let library = Library::from_source_text(device, &probe_source()).map_err(why)?;
        pipeline_rows(&library);
        Some(library)
    } else {
        None
    };
    // The phase arms need the probe's pipeline, which exists only once its
    // self-check holds: a probe that is not the shipped flash prints no phase
    // row.
    let probe = if let (true, Some(library)) = (run.phases, &library) {
        match probe_pipeline(device, stream, kernels, &mut bufs, library) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                println!("attn\tkernel=probe_flash_f16\tarm=self_check\tFAIL\t{e}");
                failed = true;
                None
            }
        }
    } else {
        None
    };
    // MLX's kernels time only once their two paths agree on the self-check
    // scene's output.
    let mut mlx = match (run.mlx, &library) {
        (true, Some(library)) => match mlx_setup(device, stream, &mut bufs, library) {
            Ok(m) => Some(m),
            Err(e) => {
                println!("attn\tkernel=mlx\tarm=self_check\tFAIL\t{e}");
                failed = true;
                None
            }
        },
        _ => None,
    };
    let variants = if run.variants {
        variant_pipelines(device)?
    } else {
        Variants::default()
    };
    let mut mlx_timed: Vec<u32> = Vec::new();
    let scenes: Vec<(u32, usize)> = run
        .keys
        .iter()
        .flat_map(|&keys| {
            if run.segs.is_empty() {
                vec![(keys, segments_for_keys(keys) as usize)]
            } else {
                run.segs.iter().map(|&s| (keys, s)).collect()
            }
        })
        .collect();
    for (keys, segments) in scenes {
        let pos = keys - 1;
        // One segment of the f16 cache is the one-pass flash, a threadgroup a
        // query head; more are the segmented flash, a threadgroup a (key
        // head, segment).
        let (shipped, tgs, threads) = if segments == 1 {
            ("attn_flash_one_f16", QUERY_HEADS, ONE_PASS_THREADS)
        } else {
            ("gqa256_flash_f16", KV_HEADS * segments, FLASH_THREADS)
        };
        let args = args_words(segments as u32, pos, keys);
        let packed = packed_dispatches(keys);
        let (dispatches, trials) = match run.phase_unit {
            PhaseUnit::Lone => (1, 2 * run.trials),
            PhaseUnit::Packed => (packed, run.trials),
        };
        println!(
            "scene\tkeys={keys}\tpos={pos}\tsegments={segments}\tkernel={shipped}\tthreadgroups={tgs}\tthreads={threads}\tpacked_n={packed}\tphase_n={dispatches}"
        );
        // The production flash: many dispatches in one buffer, then one per
        // buffer. The phase arms divide by the one of their own unit.
        let mut shipped_us = None;
        let shipped_arms = if segments <= SEGMENT_CAP {
            vec![("packed", packed, run.trials), ("lone", 1, 2 * run.trials)]
        } else {
            println!(
                "attn\tkernel={shipped}\tkeys={keys}\tskipped=segments of {segments} past the host's {SEGMENT_CAP}"
            );
            Vec::new()
        };
        for (name, arm_dispatches, arm_trials) in shipped_arms {
            match time_arm(device, stream, arm_dispatches, arm_trials, |encoder, _| {
                let p = params(&mut bufs, segments, pos, keys, device.fault_word())?;
                kernels.attention_flash(encoder, p)
            }) {
                Ok(t) => {
                    print_row(shipped, keys, name, arm_dispatches, &t);
                    if arm_dispatches == dispatches {
                        shipped_us = Some(t.us_median);
                    }
                }
                Err(e) => {
                    println!("attn\tkernel={shipped}\tkeys={keys}\tarm={name}\tFAIL\t{e}");
                    failed = true;
                }
            }
        }
        if let Some(m) = &mut mlx
            && !mlx_timed.contains(&keys)
        {
            mlx_timed.push(keys);
            failed |= mlx_arms(device, stream, &mut bufs, m, keys, packed, run.trials);
        }
        if segments > 1 {
            for (name, pipeline) in &variants.flash {
                failed |= variant_arm(
                    device, stream, &mut bufs, kernels, name, pipeline, keys, segments, &args,
                    packed, run.trials,
                );
            }
            if let Some((nofold, merge)) = variants.merge() {
                failed |= merge_arms(
                    device, stream, &mut bufs, kernels, nofold, merge, keys, segments, &args,
                    packed, run.trials,
                );
            }
        }
        let Some(pipeline) = &probe else {
            continue;
        };
        // The probe is the segmented flash: at one segment it would time a
        // program the shipped launch does not run.
        if segments == 1 {
            println!(
                "split\tkeys={keys}\tskipped=one segment is the one-pass flash, which the probe is not"
            );
            continue;
        }
        let point = Point {
            keys,
            segments,
            args: &args,
            dispatches,
            trials,
            shipped_us,
        };
        failed |= phase_arms(device, stream, &mut bufs, &point, pipeline);
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
    /// The dispatches of each timed command buffer.
    dispatches: usize,
    /// The timed command buffers of each arm.
    trials: usize,
    /// The shipped flash's figure of the same unit at this key count, when it ran.
    shipped_us: Option<f64>,
}

/// One command buffer holding one encoded dispatch, committed and waited.
fn dispatch_once(
    stream: &MetalStream,
    encode: impl FnOnce(&mut ComputeEncoder<'_>) -> Result<(), DeviceError>,
) -> Result<(), String> {
    let mut recording = stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder().map_err(why)?;
        encode(&mut encoder).map_err(why)?;
        encoder.end();
    }
    stream.submit(recording).map_err(why)?;
    stream.sync().map_err(why)
}

/// The scratch a launch writes (the partials, the output and the ticket),
/// zeroed, so an arm that skips a write reads what no earlier arm left.
fn clear_scratch(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
) -> Result<(), String> {
    let parts = partials_len(PROBE_SEG_MAX).expect("the partials fit");
    bufs.partials = device.zeroed::<f32>(stream, Elems(parts)).map_err(why)?;
    bufs.out = device
        .zeroed::<f32>(stream, Elems(out_len().expect("the out length wraps")))
        .map_err(why)?;
    bufs.ticket = device.zeroed::<u32>(stream, Elems(KV_HEADS)).map_err(why)?;
    Ok(())
}

/// The first `n` values of `buf`, read once the stream's work is done.
fn read_f32(
    device: &MetalDevice,
    stream: &MetalStream,
    buf: &MetalBuffer<f32>,
    n: usize,
) -> Result<Vec<f32>, String> {
    let mut host = vec![0f32; n];
    device
        .read(
            stream,
            buf.window(Elems(0), Elems(n)).map_err(why)?,
            &mut host,
        )
        .map_err(why)?;
    Ok(host)
}

/// The probe's pipeline, built once the arm table holds to the phase bits'
/// meaning and the probe has held to the shipped flash.
fn probe_pipeline(
    device: &MetalDevice,
    stream: &MetalStream,
    kernels: &Kernels,
    bufs: &mut Bufs,
    library: &Library,
) -> Result<Pipeline, String> {
    check_arms(&ARMS)?;
    println!("selfcheck\tclause=arm_table\tarms={}", ARMS.len());
    let pipeline = library.pipeline("probe_flash_f16").map_err(why)?;
    self_check(device, stream, kernels, bufs, &pipeline)?;
    Ok(pipeline)
}

/// Holds the probe to what its phase table says, at a scene past one segment.
/// Every clause prints a `selfcheck` row; every failed clause is named in the
/// error, and a failed probe times nothing.
///
/// - `full_equals_shipped`: the probe's all-phases arm writes the output
///   the shipped flash writes, bit for bit, so what the phase arms time is
///   the production program.
/// - `epilogue_bit`: with the epilogue on (and the fold off) every segment
///   of every head writes a partial with a positive weight sum; with the
///   epilogue bit set nothing is written. The two arms differ in what they
///   do, in the direction the bit's name says.
fn self_check(
    device: &MetalDevice,
    stream: &MetalStream,
    kernels: &Kernels,
    bufs: &mut Bufs,
    pipeline: &Pipeline,
) -> Result<(), String> {
    let keys = CHECK_KEYS;
    let pos = keys - 1;
    let segments = CHECK_SEGMENTS;
    let args = args_words(segments as u32, pos, keys);
    let out_n = out_len().expect("the out length wraps");
    let parts_all = partials_len(PROBE_SEG_MAX).expect("the partials fit");
    let parts_used = partials_len(segments).ok_or("the self-check partials do not fit")?;
    // The floats of one (head, segment) partial: the max, the weight sum and
    // the head's values.
    let stride = partials_len(1).expect("the partials fit") / QUERY_HEADS;
    let slots = QUERY_HEADS * segments;
    let mut failures: Vec<String> = Vec::new();

    clear_scratch(device, stream, bufs)?;
    dispatch_once(stream, |encoder| {
        let p = params(bufs, segments, pos, keys, device.fault_word())?;
        kernels.attention_flash(encoder, p)
    })?;
    let shipped = read_f32(device, stream, &bufs.out, out_n)?;
    clear_scratch(device, stream, bufs)?;
    dispatch_once(stream, |encoder| {
        encode_phase(
            encoder,
            pipeline,
            bufs,
            device.fault_word(),
            &args,
            PH_FULL,
            segments,
        )
    })?;
    let probed = read_f32(device, stream, &bufs.out, out_n)?;
    let finite = shipped.iter().all(|x| x.is_finite());
    let differing = shipped
        .iter()
        .zip(&probed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "selfcheck\tclause=full_equals_shipped\tkeys={keys}\tvalues={out_n}\tfinite={finite}\tdiffering={differing}"
    );
    if !finite {
        failures.push(
            "full_equals_shipped: the shipped flash's output is not finite at the self-check scene"
                .to_string(),
        );
    }
    if differing != 0 {
        failures.push(format!(
            "full_equals_shipped: {differing} of {out_n} output values differ from the shipped flash's"
        ));
    }

    // The candidate walk is not the shipped program: it is held to the
    // shipped output within the rounding of its other rescale points.
    clear_scratch(device, stream, bufs)?;
    dispatch_once(stream, |encoder| {
        encode_phase(
            encoder,
            pipeline,
            bufs,
            device.fault_word(),
            &args,
            PH_FULL | PH_DIRECT,
            segments,
        )
    })?;
    let direct = read_f32(device, stream, &bufs.out, out_n)?;
    let big = shipped.iter().fold(0f32, |a, x| a.max(x.abs()));
    let diff = shipped
        .iter()
        .zip(&direct)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    println!("selfcheck\tclause=direct_close\tkeys={keys}\tmax_abs={big:e}\tmax_diff={diff:e}");
    if diff > 1e-4 * big || direct.iter().any(|x| !x.is_finite()) {
        failures.push(format!(
            "direct_close: the direct walk's output is {diff:e} from the shipped flash's (max {big:e})"
        ));
    }

    let mut partials_of = |phase: u32| -> Result<Vec<f32>, String> {
        clear_scratch(device, stream, bufs)?;
        dispatch_once(stream, |encoder| {
            encode_phase(
                encoder,
                pipeline,
                bufs,
                device.fault_word(),
                &args,
                phase,
                segments,
            )
        })?;
        read_f32(device, stream, &bufs.partials, parts_all)
    };
    let on = partials_of(PH_FULL | PH_NO_FOLD)?;
    let off = partials_of(PH_FULL | PH_NO_FOLD | PH_NO_EPILOGUE)?;
    let weighted_on = (0..slots)
        .filter(|slot| on[slot * stride + 1] > 0.0)
        .count();
    let written_off = off[..parts_used]
        .iter()
        .filter(|x| x.to_bits() != 0)
        .count();
    println!(
        "selfcheck\tclause=epilogue_bit\tkeys={keys}\tsegments={segments}\tslots={slots}\tweighted_slots_on={weighted_on}\twritten_words_off={written_off}"
    );
    if weighted_on != slots {
        failures.push(format!(
            "epilogue_bit: the epilogue-on arm wrote a weighted partial in {weighted_on} of {slots} slots"
        ));
    }
    if written_off != 0 {
        failures.push(format!(
            "epilogue_bit: the epilogue-off arm wrote {written_off} partial words"
        ));
    }

    if let Some(fault) = device.fault_word().read(stream).map_err(why)? {
        failures.push(format!("the self-check launches raised {fault}"));
    }
    device.fault_word().clear(stream).map_err(why)?;
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// The phase arms of one key count: the probe kernel, the whole flash and
/// one phase off at a time, then the split. Whether any arm failed.
fn phase_arms(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    point: &Point<'_>,
    pipeline: &Pipeline,
) -> bool {
    let Point {
        keys,
        segments,
        args,
        dispatches,
        trials,
        shipped_us,
    } = *point;
    let mut figures: Vec<(&str, f64)> = Vec::new();
    let mut failed = false;
    for (name, phase) in ARMS {
        if let Err(e) = clear_scratch(device, stream, bufs) {
            println!("attn\tkernel=probe_flash_f16\tkeys={keys}\tarm={name}\tFAIL\t{e}");
            return true;
        }
        match time_arm(device, stream, dispatches, trials, |encoder, _| {
            encode_phase(
                encoder,
                pipeline,
                bufs,
                device.fault_word(),
                args,
                phase,
                segments,
            )
        }) {
            Ok(t) => {
                print_row("probe_flash_f16", keys, name, dispatches, &t);
                figures.push((name, t.us_median));
            }
            Err(e) => {
                println!("attn\tkernel=probe_flash_f16\tkeys={keys}\tarm={name}\tFAIL\t{e}");
                failed = true;
            }
        }
    }
    let at = |name: &str| figures.iter().find(|(n, _)| *n == name).map(|(_, us)| *us);
    if let [
        Some(full),
        Some(no_append),
        Some(no_prepare),
        Some(no_loop),
        Some(no_loop_no_fold),
        Some(no_fold),
        Some(no_epilogue),
        Some(bare),
        Some(ticket_only),
    ] = SPLIT_READS.map(at)
    {
        // Each term is a difference of two arms that differ in one phase.
        // `loop` is the tile loop whole, `fold` the ticket and the fold,
        // `ticket` the handshake alone and `fold_body` the fold's own work,
        // `epilogue` the partial write alone, `epilogue_and_fold` that write
        // and the fold work it lets the fold do, and `floor` what stays with
        // the tile loop, the epilogue's partials and the fold off (the
        // launch, the append, the prepare). `probe_over_shipped` is the
        // probe's whole-flash arm over the shipped flash's arm of the same unit: the
        // run-time gates are not free, so the split is of a program that may
        // time apart from the shipped one even where its output equals it.
        let ratio = shipped_us.map_or_else(|| "none".to_string(), |us| format!("{:.3}", full / us));
        println!(
            "split\tkeys={keys}\tappend_us={:.3}\tprepare_us={:.3}\tloop_us={:.3}\tfold_us={:.3}\tepilogue_us={:.3}\tepilogue_and_fold_us={:.3}\tfloor_us={:.3}\tticket_us={:.3}\tfold_body_us={:.3}\tprobe_over_shipped={ratio}",
            full - no_append,
            full - no_prepare,
            full - no_loop,
            full - no_fold,
            no_fold - bare,
            full - no_epilogue,
            no_loop_no_fold,
            ticket_only - no_fold,
            full - ticket_only,
        );
    }
    if let (Some(full), Some(no_fold), Some(direct), Some(direct_no_fold)) = (
        at("full"),
        at("no_fold"),
        at("direct"),
        at("direct_no_fold"),
    ) {
        // The tile loop against the candidate walk on the same grid and fold.
        println!(
            "walk\tkeys={keys}\tsegments={segments}\ttiles_us={full:.3}\tdirect_us={direct:.3}\ttiles_no_fold_us={no_fold:.3}\tdirect_no_fold_us={direct_no_fold:.3}"
        );
    }
    failed
}

/// Source edits timed against the shipped segmented flash: each is the shipped
/// source with each of its finds replaced (a find must stand exactly once in
/// the probe's text), compiled as a library of its own, its `gqa256_flash_f16`
/// launched with the shipped bindings. A development-loop instrument: a
/// variant is a question about one term.
///
/// - `nofold`: no ticket and no fold; the partials are written and left.
/// - `onefold`: one ticket word for the launch; its last threadgroup folds
///   every key head's query heads, the key heads one after another.
/// - `nobf`: each key's score summed by the SIMD-group sum, not the tile's
///   butterfly; lane t keeps tile key t's score.
/// - `noattr`: the flash declares no maximum threads, so the compiler picks
///   its registers (and the threadgroups a core holds) alone.
/// - `vpre`: the value phase's first rows issued before the tile's softmax.
/// - `interleave`: segment s walks tiles s, s + S, ... of the keys.
const VARIANTS: &[(&str, &[(&str, &str)])] = &[
    (
        "nofold",
        &[(
            "if (args.segments > 1u && active) {\n        ticket_is_last(ticket + kv, args.segments, tid, last_flag);",
            "if (false) {\n        ticket_is_last(ticket + kv, args.segments, tid, last_flag);",
        )],
    ),
    (
        "onefold",
        &[(
            "    if (args.segments > 1u && active) {\n        ticket_is_last(ticket + kv, args.segments, tid, last_flag);\n        if (last_flag == 1u) {\n            attn_fold(partials, ticket + kv, args, ATTN_GATE_SRC, out, faults, tid, kv);\n        }",
            "    if (args.segments > 1u) {\n        ticket_is_last(ticket, BLOOMERY_ATTN_KV * args.segments, tid, last_flag);\n        if (last_flag == 1u) {\n            for (uint f = 0u; f < BLOOMERY_ATTN_KV; ++f) {\n                attn_fold(partials, ticket, args, ATTN_GATE_SRC, out, faults, tid, f);\n            }\n        }",
        )],
    ),
    (
        "nobf",
        &[(
            "    bool b0 = (lane & 1u) != 0u;
    bool b1 = (lane & 2u) != 0u;
    float quads[BLOOMERY_ATTN_TILE / ATTN_WALK_AHEAD];
    for (uint t0 = 0u; t0 < BLOOMERY_ATTN_TILE; t0 += ATTN_WALK_AHEAD) {
        typename Cache::Word words[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            uint key = lo + t0 + c;
            words[c] = FULL || key < hi ? cache.key_word(row0 + key, lane) : Cache::zero();
        }
        float part[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            Cache::widen(words[c], kvv);
            float dot = 0.0f;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                dot = fma(qv[d], kvv[d], dot);
            }
            part[c] = dot;
        }
        float pair0 = (b0 ? part[1] : part[0]) + simd_shuffle_xor(b0 ? part[0] : part[1], ushort(1));
        float pair1 = (b0 ? part[3] : part[2]) + simd_shuffle_xor(b0 ? part[2] : part[3], ushort(1));
        quads[t0 / ATTN_WALK_AHEAD] = (b1 ? pair1 : pair0) + simd_shuffle_xor(b1 ? pair0 : pair1, ushort(2));
    }
    bool b2 = (lane & 4u) != 0u;
    bool b3 = (lane & 8u) != 0u;
    float half0 = (b2 ? quads[1] : quads[0]) + simd_shuffle_xor(b2 ? quads[0] : quads[1], ushort(4));
    float half1 = (b2 ? quads[3] : quads[2]) + simd_shuffle_xor(b2 ? quads[2] : quads[3], ushort(4));
    float sixteen = (b3 ? half1 : half0) + simd_shuffle_xor(b3 ? half0 : half1, ushort(8));
    float score = (sixteen + simd_shuffle_xor(sixteen, ushort(16))) * args.scale;
    uint t = lane & (BLOOMERY_ATTN_TILE - 1u);
    float mine = FULL || lo + t < hi ? score : -INFINITY;
",
            "    float mine = -INFINITY;
    for (uint t0 = 0u; t0 < BLOOMERY_ATTN_TILE; t0 += ATTN_WALK_AHEAD) {
        typename Cache::Word words[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            uint key = lo + t0 + c;
            words[c] = FULL || key < hi ? cache.key_word(row0 + key, lane) : Cache::zero();
        }
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            uint t = t0 + c;
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            Cache::widen(words[c], kvv);
            float dot = 0.0f;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                dot = fma(qv[d], kvv[d], dot);
            }
            dot = simd_sum(dot);
            float score = dot * args.scale;
            float scored = FULL || lo + t < hi ? score : -INFINITY;
            mine = lane == t ? scored : mine;
        }
    }
",
        )],
    ),
    (
        "noattr",
        &[(
            "#define ATTN_FLASH_HOLD [[max_total_threads_per_threadgroup(BLOOMERY_ATTN_THREADS * BLOOMERY_ATTN_KV)]]",
            "#define ATTN_FLASH_HOLD",
        )],
    ),
    (
        "vpre",
        &[
            (
                "    float new_m = max(*running_m, simd_max(mine));\n    float old_m = *running_m;",
                "    typename Cache::Word vfirst[ATTN_WALK_AHEAD];\n    for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {\n        uint key = lo + c;\n        vfirst[c] = FULL || key < hi ? cache.value_word(row0 + key, lane) : Cache::zero();\n    }\n    float new_m = max(*running_m, simd_max(mine));\n    float old_m = *running_m;",
            ),
            (
                "            words[c] = FULL || key < hi ? cache.value_word(row0 + key, lane) : Cache::zero();",
                "            words[c] = t0 == 0u ? vfirst[c] : (FULL || key < hi ? cache.value_word(row0 + key, lane) : Cache::zero());",
            ),
        ],
    ),
    (
        "interleave",
        &[
            (
                "    bool owns = ok && args.pos < args.keys && args.pos / span == seg;",
                "    bool owns = ok && args.pos < args.keys && (args.pos / BLOOMERY_ATTN_TILE) % args.segments == seg;",
            ),
            (
                "    uint lo0 = seg * span;\n    uint seg_hi = min((seg + 1u) * span, max_live);\n#endif\n    uint row0 = kv * args.ctx;\n    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {",
                "    uint lo0 = seg * BLOOMERY_ATTN_TILE;\n    uint seg_hi = max_live;\n#endif\n    uint row0 = kv * args.ctx;\n    for (uint lo = lo0; lo < seg_hi; lo += args.segments * BLOOMERY_ATTN_TILE) {",
            ),
        ],
    ),
];

/// Variants made of other variants' edits, applied in the order named.
const COMBOS: &[(&str, &[&str])] = &[
    ("nobf_noattr", &["nobf", "noattr"]),
    ("before", &["onefold", "nobf", "noattr"]),
];

/// The variants' pipelines, and the merge dispatch the `nofold` variant pairs
/// with.
#[derive(Default)]
struct Variants {
    flash: Vec<(&'static str, Pipeline)>,
    merge: Option<Pipeline>,
}

impl Variants {
    /// The `nofold` flash and the merge it pairs with, when both were built.
    fn merge(&self) -> Option<(&Pipeline, &Pipeline)> {
        let nofold = self
            .flash
            .iter()
            .find_map(|(n, p)| (*n == "nofold").then_some(p))?;
        Some((nofold, self.merge.as_ref()?))
    }
}

/// Each variant's `gqa256_flash_f16` pipeline, with its resource row, and the
/// probe's merge kernel.
fn variant_pipelines(device: &MetalDevice) -> Result<Variants, String> {
    let mut out = Variants::default();
    let mut all: Vec<(&str, Vec<(&str, &str)>)> = VARIANTS
        .iter()
        .map(|&(name, edits)| (name, edits.to_vec()))
        .collect();
    for &(name, parts) in COMBOS {
        let mut edits = Vec::new();
        for part in parts {
            let (_, e) = VARIANTS
                .iter()
                .find(|(n, _)| n == part)
                .ok_or_else(|| format!("combo {name}: no variant named {part}"))?;
            edits.extend_from_slice(e);
        }
        all.push((name, edits));
    }
    for (name, edits) in all {
        let mut text = probe_source();
        for &(find, replace) in &edits {
            let hits = text.matches(find).count();
            if hits != 1 {
                return Err(format!(
                    "variant {name}: the source holds {hits} of {find:?}, not one"
                ));
            }
            text = text.replace(find, replace);
        }
        let library = Library::from_source_text(device, &text).map_err(why)?;
        let pipeline = library.pipeline("gqa256_flash_f16").map_err(why)?;
        println!(
            "pipe\tname=var_{name}\tmax_threads={}\texec_width={}\tstatic_tg_bytes={}",
            pipeline.max_threads_per_group(),
            pipeline.execution_width(),
            pipeline.static_group_bytes()
        );
        out.flash.push((name, pipeline));
    }
    let library = Library::from_source_text(device, &probe_source()).map_err(why)?;
    out.merge = Some(library.pipeline("probe_merge_f16").map_err(why)?);
    Ok(out)
}

/// One variant at one split scene: its output against the shipped flash's
/// (the largest difference; skipped past the host's segment cap, where the
/// shipped launch refuses), then its packed time.
#[allow(clippy::too_many_arguments)]
fn variant_arm(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    kernels: &Kernels,
    name: &str,
    pipeline: &Pipeline,
    keys: u32,
    segments: usize,
    args: &[[u8; 4]; 7],
    packed: usize,
    trials: usize,
) -> bool {
    let pos = keys - 1;
    let out_n = out_len().expect("the out length wraps");
    let fault = device.fault_word();
    let compared = (|| -> Result<Option<f32>, String> {
        if segments > SEGMENT_CAP {
            return Ok(None);
        }
        clear_scratch(device, stream, bufs)?;
        dispatch_once(stream, |encoder| {
            let p = params(bufs, segments, pos, keys, fault)?;
            kernels.attention_flash(encoder, p)
        })?;
        let shipped = read_f32(device, stream, &bufs.out, out_n)?;
        clear_scratch(device, stream, bufs)?;
        dispatch_once(stream, |encoder| {
            encode_phase(encoder, pipeline, bufs, fault, args, 0, segments)
        })?;
        let got = read_f32(device, stream, &bufs.out, out_n)?;
        Ok(Some(
            shipped
                .iter()
                .zip(&got)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs())),
        ))
    })();
    let timed = time_arm(device, stream, packed, trials, |encoder, _| {
        encode_phase(encoder, pipeline, bufs, fault, args, 0, segments)
    });
    match (compared, timed) {
        (Ok(diff), Ok(t)) => {
            let diff = diff.map_or_else(|| "skipped".to_string(), |d| format!("{d:e}"));
            println!(
                "variant\tname={name}\tkeys={keys}\tsegments={segments}\tmax_diff_vs_shipped={diff}"
            );
            print_row(&format!("var_{name}"), keys, "packed", packed, &t);
            false
        }
        (Err(e), _) | (_, Err(e)) => {
            println!("attn\tkernel=var_{name}\tkeys={keys}\tarm=packed\tFAIL\t{e}");
            true
        }
    }
}

/// The probe's merge dispatch over the partials a flash left: a threadgroup
/// a query head, a thread a value.
fn encode_merge(
    encoder: &mut ComputeEncoder<'_>,
    merge: &Pipeline,
    b: &mut Bufs,
    fault: &bloomery_metal::fault::MetalFaultWord,
    args: &[[u8; 4]; 7],
    segments: usize,
) -> Result<(), DeviceError> {
    let qg = qg_len().expect("the query length wraps");
    let parts = partials_len(segments).ok_or_else(|| DeviceError::Range {
        what: "attention",
        detail: format!("the partials of {segments} segments do not fit"),
    })?;
    let out_n = out_len().expect("the out length wraps");
    let words: Vec<u8> = args.iter().flatten().copied().collect();
    encoder.set_pipeline(merge);
    encoder.set_binding(0, Binding::of(&b.q.window(Elems(0), Elems(qg))?)?)?;
    encoder.set_binding(1, Binding::of(&b.partials.window(Elems(0), Elems(parts))?)?)?;
    encoder.set_binding(
        2,
        Binding::of_mut(&b.out.window_mut(Elems(0), Elems(out_n))?)?,
    )?;
    encoder.set_binding(3, Binding::bytes(&words)?)?;
    encoder.set_binding(4, fault.binding())?;
    encoder.dispatch_groups(QUERY_HEADS, 1, HEAD_DIM)
}

/// The fold priced as a dispatch of its own: the `nofold` flash then the
/// merge behind a barrier (`merge_pair`, its output against the shipped
/// flash's), and the merge alone over the partials the pair left. Whether an
/// arm failed.
#[allow(clippy::too_many_arguments)]
fn merge_arms(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    kernels: &Kernels,
    nofold: &Pipeline,
    merge: &Pipeline,
    keys: u32,
    segments: usize,
    args: &[[u8; 4]; 7],
    packed: usize,
    trials: usize,
) -> bool {
    let pos = keys - 1;
    let out_n = out_len().expect("the out length wraps");
    let fault = device.fault_word();
    let compared = (|| -> Result<String, String> {
        let shipped = if segments <= SEGMENT_CAP {
            clear_scratch(device, stream, bufs)?;
            dispatch_once(stream, |encoder| {
                let p = params(bufs, segments, pos, keys, fault)?;
                kernels.attention_flash(encoder, p)
            })?;
            Some(read_f32(device, stream, &bufs.out, out_n)?)
        } else {
            None
        };
        clear_scratch(device, stream, bufs)?;
        dispatch_once(stream, |encoder| {
            encode_phase(encoder, nofold, bufs, fault, args, 0, segments)?;
            encoder.memory_barrier();
            encode_merge(encoder, merge, bufs, fault, args, segments)
        })?;
        let got = read_f32(device, stream, &bufs.out, out_n)?;
        if got.iter().any(|x| !x.is_finite()) {
            return Err("the merge wrote a value that is not finite".to_string());
        }
        Ok(shipped.map_or_else(
            || "skipped".to_string(),
            |shipped| {
                let d = shipped
                    .iter()
                    .zip(&got)
                    .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                format!("{d:e}")
            },
        ))
    })();
    match compared {
        Ok(diff) => println!(
            "variant\tname=merge_pair\tkeys={keys}\tsegments={segments}\tmax_diff_vs_shipped={diff}"
        ),
        Err(e) => {
            println!("attn\tkernel=merge_pair\tkeys={keys}\tarm=packed\tFAIL\t{e}");
            return true;
        }
    }
    let mut failed = false;
    for (name, with_flash) in [("merge_pair", true), ("merge_alone", false)] {
        match time_arm(device, stream, packed, trials, |encoder, _| {
            if with_flash {
                encode_phase(encoder, nofold, bufs, fault, args, 0, segments)?;
                encoder.memory_barrier();
            }
            encode_merge(encoder, merge, bufs, fault, args, segments)
        }) {
            Ok(t) => print_row(name, keys, "packed", packed, &t),
            Err(e) => {
                println!("attn\tkernel={name}\tkeys={keys}\tarm=packed\tFAIL\t{e}");
                failed = true;
            }
        }
    }
    failed
}

/// Every flash pipeline's resource figures, as Metal reports them: the most
/// threads a group may hold (below 1024 when registers bind), the SIMD width
/// and the static threadgroup memory.
fn pipeline_rows(library: &Library) {
    for name in [
        "attn_flash_one_f16",
        "gqa256_flash_f16",
        "probe_flash_f16",
        "mlx_vec_f16",
        "mlx_2p1_f16",
        "mlx_2p2_f16",
    ]
    .into_iter()
    .chain(mlx::ARMS)
    {
        match library.pipeline(name) {
            Ok(p) => println!(
                "pipe\tname={name}\tmax_threads={}\texec_width={}\tstatic_tg_bytes={}",
                p.max_threads_per_group(),
                p.execution_width(),
                p.static_group_bytes()
            ),
            Err(e) => println!("pipe\tname={name}\tFAIL\t{}", why(e)),
        }
    }
}

/// MLX's pipelines and partials, once its two paths agree.
struct Mlx {
    pipes: mlx::MlxPipes,
    bufs: mlx::MlxBufs,
}

/// The MLX launch over the planted query and cache at `keys`.
fn mlx_launch(bufs: &mut Bufs, keys: u32) -> mlx::MlxLaunch<'_> {
    mlx::MlxLaunch {
        q: &bufs.q,
        qg: qg_len().expect("the query length wraps"),
        cache_k: &mut bufs.cache_k,
        cache_v: &mut bufs.cache_v,
        cache_n: f16_len(CTX as usize).expect("the cache length wraps"),
        keys,
        ctx: CTX,
        scale: SCALE,
        head: HEAD_DIM,
    }
}

/// Builds MLX's pipelines and holds the port to one clause: at the
/// self-check's key count its single-pass kernel and its two passes write
/// the same output within the half partials' rounding, finite and not all
/// zero. A port that reads the wrong rows fails it.
fn mlx_setup(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    library: &Library,
) -> Result<Mlx, String> {
    let pipes = mlx::MlxPipes {
        vec: library.pipeline("mlx_vec_f16").map_err(why)?,
        pass1: library.pipeline("mlx_2p1_f16").map_err(why)?,
        pass2: library.pipeline("mlx_2p2_f16").map_err(why)?,
        arms: mlx::ARMS
            .iter()
            .map(|&name| Ok((name, library.pipeline(name).map_err(why)?)))
            .collect::<Result<_, String>>()?,
    };
    let mut m = mlx::MlxBufs::new(device, stream, HEAD_DIM)?;
    let keys = CHECK_KEYS;
    let blocks = mlx::blocks_g13s(keys, QUERY_HEADS / KV_HEADS);
    let out_n = QUERY_HEADS * HEAD_DIM;
    dispatch_once(stream, |encoder| {
        mlx::encode_vec(encoder, &pipes, &mut mlx_launch(bufs, keys), &mut m)
    })?;
    let single = read_f32(device, stream, &m.out, out_n)?;
    m.out = device.zeroed::<f32>(stream, Elems(out_n)).map_err(why)?;
    dispatch_once(stream, |encoder| {
        let mut l = mlx_launch(bufs, keys);
        mlx::encode_pass1(encoder, &pipes.pass1, &mut l, &mut m, blocks)?;
        encoder.memory_barrier();
        mlx::encode_pass2(encoder, &pipes, &l, &mut m, blocks)
    })?;
    let two = read_f32(device, stream, &m.out, out_n)?;
    let finite = single.iter().chain(&two).all(|x| x.is_finite());
    let big = single.iter().fold(0f32, |a, x| a.max(x.abs()));
    let diff = single
        .iter()
        .zip(&two)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    println!(
        "selfcheck\tclause=mlx_paths_agree\tkeys={keys}\tblocks={blocks}\tvalues={out_n}\tfinite={finite}\tmax_abs={big:e}\tmax_diff={diff:e}"
    );
    if !finite || big == 0.0 || diff > 5e-3 * big {
        return Err(format!(
            "mlx_paths_agree: finite {finite}, max |out| {big:e}, max diff {diff:e} (bound 5e-3 of max)"
        ));
    }
    // Every arm's pass 1, folded by pass 2, writes the single pass's output.
    for (name, arm) in &pipes.arms {
        m.out = device.zeroed::<f32>(stream, Elems(out_n)).map_err(why)?;
        dispatch_once(stream, |encoder| {
            let mut l = mlx_launch(bufs, keys);
            mlx::encode_pass1(encoder, arm, &mut l, &mut m, blocks)?;
            encoder.memory_barrier();
            mlx::encode_pass2(encoder, &pipes, &l, &mut m, blocks)
        })?;
        let got = read_f32(device, stream, &m.out, out_n)?;
        let finite = got.iter().all(|x| x.is_finite());
        let diff = single
            .iter()
            .zip(&got)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        println!(
            "selfcheck\tclause=mlx_arm_agrees\tarm={name}\tkeys={keys}\tfinite={finite}\tmax_diff={diff:e}"
        );
        if !finite || diff > 5e-3 * big {
            return Err(format!(
                "mlx_arm_agrees {name}: finite {finite}, max diff {diff:e} (bound 5e-3 of {big:e})"
            ));
        }
    }
    Ok(Mlx { pipes, bufs: m })
}

/// MLX's arms at one key count, in the shipped `packed` unit: the
/// single-pass kernel, pass 1 alone, pass 2 alone and both passes, at MLX's
/// block count for this GPU. Whether any arm failed.
fn mlx_arms(
    device: &MetalDevice,
    stream: &MetalStream,
    bufs: &mut Bufs,
    m: &mut Mlx,
    keys: u32,
    packed: usize,
    trials: usize,
) -> bool {
    let mut failed = false;
    let blocks = mlx::blocks_g13s(keys, QUERY_HEADS / KV_HEADS);
    let mut report = |name: &str, r: Result<Timed, String>| match r {
        Ok(t) => print_row(name, keys, "packed", packed, &t),
        Err(e) => {
            println!("attn\tkernel={name}\tkeys={keys}\tarm=packed\tFAIL\t{e}");
            failed = true;
        }
    };
    let Mlx { pipes, bufs: mb } = m;
    report(
        "mlx_vec",
        time_arm(device, stream, packed, trials, |encoder, _| {
            mlx::encode_vec(encoder, pipes, &mut mlx_launch(bufs, keys), mb)
        }),
    );
    report(
        &format!("mlx_2p_b{blocks}"),
        time_arm(device, stream, packed, trials, |encoder, _| {
            let mut l = mlx_launch(bufs, keys);
            mlx::encode_pass1(encoder, &pipes.pass1, &mut l, mb, blocks)?;
            encoder.memory_barrier();
            mlx::encode_pass2(encoder, pipes, &l, mb, blocks)
        }),
    );
    report(
        &format!("mlx_2p1_b{blocks}"),
        time_arm(device, stream, packed, trials, |encoder, _| {
            mlx::encode_pass1(
                encoder,
                &pipes.pass1,
                &mut mlx_launch(bufs, keys),
                mb,
                blocks,
            )
        }),
    );
    report(
        &format!("mlx_2p2_b{blocks}"),
        time_arm(device, stream, packed, trials, |encoder, _| {
            mlx::encode_pass2(encoder, pipes, &mlx_launch(bufs, keys), mb, blocks)
        }),
    );
    for (name, arm) in &pipes.arms {
        report(
            &format!("{name}_b{blocks}"),
            time_arm(device, stream, packed, trials, |encoder, _| {
                mlx::encode_pass1(encoder, arm, &mut mlx_launch(bufs, keys), mb, blocks)
            }),
        );
    }
    failed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped table with the word of the arm `name` replaced.
    fn with_word(name: &str, word: u32) -> Vec<(&'static str, u32)> {
        ARMS.iter()
            .map(|&(n, w)| (n, if n == name { word } else { w }))
            .collect()
    }

    #[test]
    fn the_arm_table_is_held_to_the_phase_bit_meaning() {
        check_arms(&ARMS).expect("the shipped table holds");
        let mut dropped = ARMS.to_vec();
        dropped.retain(|(n, _)| *n != "ticket_only");
        let mut twice = ARMS.to_vec();
        twice.push(("no_fold", PH_FULL | PH_NO_FOLD));
        let mut unnamed = ARMS.to_vec();
        unnamed.push(("no_prefetch", PH_FULL));
        let cases = [
            (
                "an epilogue arm that never sets its bit",
                with_word("no_epilogue", PH_FULL | PH_NO_FOLD),
                "arm no_epilogue: the table's word",
            ),
            (
                "a full arm that skips the epilogue",
                with_word("full", PH_FULL | PH_NO_EPILOGUE),
                "arm full: the table's word",
            ),
            (
                "an arm the split reads, dropped",
                dropped,
                "reads arm ticket_only",
            ),
            ("a name twice", twice, "arm no_fold stands twice"),
            (
                "a name no phase says",
                unnamed,
                "there is no phase `prefetch`",
            ),
        ];
        for (what, table, said) in cases {
            let err = check_arms(&table).expect_err(what);
            assert!(err.contains(said), "{what}: {err}");
        }
    }
}
