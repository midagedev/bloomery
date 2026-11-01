//! The row kernels' timing probe's body (macOS only).
//!
//! The method is the attention probe's: a timed command buffer holds many
//! dispatches, its time is its own `GPUEndTime` minus `GPUStartTime`, and the
//! figure is the median over the buffers. The delta arms launch
//! `Kernels::delta_rows` over a chunk of `m` rows and print the microseconds a
//! row of a layer takes; `step` is the decode step, one row a dispatch.
//!
//! Each row is one line, tab-separated. A development-loop instrument: a
//! figure it prints is this machine's, taken with no timing lease, and is not
//! a published number.
//!
//! The attention arms launch `Kernels::attention_prefill` over a chunk of `P`
//! causal queries from position 0 (the append of the chunk's keys and values,
//! then the simdgroup-matrix flash) and print the milliseconds a launch takes;
//! `--v0` also launches the decode flash's rows launch (`attention_rows`), the
//! bit-exact oracle, for the sizes it is practical at.
//!
//! Flags: `--trials N` (timed command buffers per arm, default 7),
//! `--delta a,b,c` (chunk row counts, default 64,512), `--attn a,b,c` (prefill
//! attention sizes, default none), `--v0` (add the decode flash's rows arm),
//! `--help`.

use std::process::ExitCode;

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord};
use bloomery_metal::MetalBuffer;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::Binding;
use bloomery_metal::instance::HOST;
use bloomery_metal::kernels::{
    AttentionPrefillParams, AttentionRowsParams, AttnCache, CONV_TAPS, DeltaParams,
    DeltaRowStrides, DeltaRowsParams, HEAD_COLS, HEAD_DIM, HeadMap, KV_HEADS, Kernels, QUERY_HEADS,
    ROT, VALUE_HEADS, channels, f16_len, out_len, partials_len, prepared_elems, qg_len, ring_elems,
    state_elems,
};
use bloomery_metal::library::{Library, SOURCE};

#[path = "../shared/gpu_time.rs"]
mod gpu_time;

use gpu_time::{Timed, time_arm};

/// The delta layer's key heads.
const N_K: usize = 16;

/// The layer word a raise is recorded in.
const LAYER: u32 = 3;

/// The gains a launch binds: the head's ones where the flash norms each
/// head, none where it applies no head norm (the buffer then goes unread).
fn gains_of(
    norm: bool,
    ones: &MetalBuffer<f32>,
) -> Result<Option<bloomery_device::Window<'_, f32, MetalDevice>>, DeviceError> {
    norm.then(|| ones.window(Elems(0), Elems(HEAD_DIM)))
        .transpose()
}

fn help() {
    println!(
        "\
metal_rows_time: development-loop timings of the row kernels.

A number this prints is this machine's, from no timing lease, and is not a
published result. Rows are tab-separated.

  --help         print this and exit 0
  --trials N     timed command buffers per arm (default 7)
  --delta LIST   comma-separated chunk row counts (default 64,512)
  --attn LIST    comma-separated prefill attention sizes (default none)
  --v0           add the decode flash's rows arm to the attention sizes
  --phases       add the prefill kernel's phase arms to the attention sizes
  --mma          time the simdgroup matrix product alone, by operand type
  --chunk N      launch the prefill attention in chunks of N rows, one command buffer"
    );
}

struct Run {
    trials: usize,
    delta: Vec<usize>,
    attn: Vec<usize>,
    v0: bool,
    phases: bool,
    mma: bool,
    chunk: usize,
}

fn parse_list(value: &str, what: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(str::trim)
        .map(|tok| {
            tok.parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("{what} takes whole numbers, not {tok:?}"))
        })
        .collect()
}

fn parse(args: &[String]) -> Result<Run, String> {
    let mut run = Run {
        trials: 7,
        delta: vec![64, 512],
        attn: Vec::new(),
        v0: false,
        phases: false,
        mma: false,
        chunk: 0,
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = args.get(i + 1);
        match arg {
            "--trials" => {
                let v = value.ok_or("--trials takes a value")?;
                run.trials = parse_list(v, "--trials")?[0];
                i += 1;
            }
            "--v0" => run.v0 = true,
            "--phases" => run.phases = true,
            "--mma" => run.mma = true,
            "--chunk" => {
                run.chunk = parse_list(value.ok_or("--chunk takes a value")?, "--chunk")?[0];
                i += 1;
            }
            "--attn" => {
                run.attn = parse_list(value.ok_or("--attn takes a value")?, "--attn")?;
                i += 1;
            }
            "--delta" => {
                run.delta = parse_list(value.ok_or("--delta takes a value")?, "--delta")?;
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
    Ok(run)
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

fn planted(state: &mut u64, n: usize, span: f32) -> Vec<f32> {
    (0..n).map(|_| value(state, span)).collect()
}

const PREP_OFF: (&str, &str) = (
    "    uint kh = tgp.x;\n    uint r = tgp.y;\n    uint tid = tpt.x;\n    uint ch = (2u * n_k + n_v) * head;",
    "    if (args.rows != 0xFFFFFFFFu) {\n        return;\n    }\n    uint kh = tgp.x;\n    uint r = tgp.y;\n    uint tid = tpt.x;\n    uint ch = (2u * n_k + n_v) * head;",
);
const RECUR_OFF: (&str, &str) = (
    "    bool row_ok = lane_ok;\n    for (uint r = 0u; r < args.rows; ++r) {",
    "    bool row_ok = lane_ok;\n    for (uint r = 0u; r < args.rows && args.rows == 0xFFFFFFFFu; ++r) {",
);
const POST_OFF: (&str, &str) = (
    "    float oval = numerator[tid];\n    o2[tid] = oval * oval;",
    "    if (args.rows != 0xFFFFFFFFu) {\n        return;\n    }\n    float oval = numerator[tid];\n    o2[tid] = oval * oval;",
);

/// The phase arms: each is the launch with the other phases gated off.
const PHASES: &[(&str, &[(&str, &str)])] = &[
    ("prep", &[RECUR_OFF, POST_OFF]),
    ("recur", &[PREP_OFF, POST_OFF]),
    ("post", &[PREP_OFF, RECUR_OFF]),
];

fn print_delta(arm: &str, rows: usize, t: &Timed) {
    println!(
        "delta\tarm={arm}\trows={rows}\tus_dispatch={:.1}\tus_min={:.1}\tus_p90={:.1}\tus_row_layer={:.3}\tus_row_layer_min={:.3}",
        t.us_median,
        t.us_min,
        t.us_p90,
        t.us_median / rows as f64,
        t.us_min / rows as f64
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
            println!("metal_rows_time: {e}");
            return ExitCode::from(64);
        }
    };
    let device = match MetalDevice::open(&HOST) {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("metal_rows_time: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("metal_rows_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let kernels = match Kernels::new(&device) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("metal_rows_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let name = device.info().map(|i| i.name).unwrap_or_default();
    println!(
        "info\tinstrument=development-loop\tdevice={name}\tlib={}\ttrials={}",
        device.library().path().name(),
        run.trials
    );
    let result = delta_arms(&run, &device, &kernels)
        .and_then(|()| attn_arms(&run, &device, &kernels))
        .and_then(|()| {
            if run.mma {
                mma_arms(&run, &device)
            } else {
                Ok(())
            }
        });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            println!("metal_rows_time: FAIL ({e})");
            ExitCode::FAILURE
        }
    }
}

/// The matrix product probe: a SIMD group's loop of products over registers
/// only, eight independent accumulators, by operand and accumulator type.
const MMA_PROBE: &str = r#"
#include <metal_simdgroup_matrix>
struct MmaArgs { uint iters; };
#define MMA_KERNEL(NAME, AT, CT) \
kernel void NAME(device float *out [[buffer(0)]], constant MmaArgs &args [[buffer(1)]], \
                 uint gid [[thread_position_in_grid]]) { \
    simdgroup_matrix<AT, 8, 8> a = make_filled_simdgroup_matrix<AT, 8, 8>(AT(0.5f)); \
    simdgroup_matrix<AT, 8, 8> b = make_filled_simdgroup_matrix<AT, 8, 8>(AT(0.25f)); \
    simdgroup_matrix<CT, 8, 8> c[8]; \
    for (uint i = 0u; i < 8u; ++i) { c[i] = make_filled_simdgroup_matrix<CT, 8, 8>(CT(0.0f)); } \
    for (uint it = 0u; it < args.iters; ++it) { \
        for (uint i = 0u; i < 8u; ++i) { simdgroup_multiply_accumulate(c[i], a, b, c[i]); } \
    } \
    float sum = 0.0f; \
    for (uint i = 0u; i < 8u; ++i) { sum += float(c[i].thread_elements()[0]); } \
    if (sum == 123456.0f) { out[gid] = sum; } \
}
MMA_KERNEL(mma_hhf, half, float)
MMA_KERNEL(mma_fff, float, float)
MMA_KERNEL(mma_hhh, half, half)
"#;

fn mma_arms(run: &Run, device: &MetalDevice) -> Result<(), String> {
    let stream = device.stream();
    let library =
        Library::from_source_text(device, &format!("{SOURCE}{MMA_PROBE}")).map_err(why)?;
    let groups = 256usize;
    let threads = 128usize;
    let iters = 4000u32;
    let mut out = device
        .upload(stream, &vec![0f32; groups * threads])
        .map_err(why)?;
    for name in ["mma_hhf", "mma_fff", "mma_hhh"] {
        let pipeline = library.pipeline(name).map_err(why)?;
        let timed = time_arm(device, stream, 1, run.trials, |encoder, _| {
            encoder.set_pipeline(&pipeline);
            encoder.set_binding(
                0,
                Binding::of_mut(&out.window_mut(Elems(0), Elems(groups * threads))?)?,
            )?;
            encoder.set_binding(1, Binding::bytes(&iters.to_ne_bytes())?)?;
            encoder.dispatch_groups(groups, 1, threads)
        })?;
        let products = (groups * threads / 32) as f64 * f64::from(iters) * 8.0;
        println!(
            "mma\tarm={name}\tus={:.1}\tus_min={:.1}\ttflops_min={:.3}\tcycles_per_mma_core_at_1.3ghz={:.2}",
            timed.us_median,
            timed.us_min,
            products * 1024.0 / (timed.us_min * 1e-6) / 1e12,
            timed.us_min * 1e-6 * 1.296e9 * 16.0 / products
        );
    }
    Ok(())
}

/// The rope table, `ctx` rows of the interleaved cos/sin pairs of a base.
fn rope_rows(ctx: usize) -> Vec<f32> {
    let base = 10_000.0f64;
    let mut out = vec![0f32; ctx * ROT];
    for pos in 0..ctx {
        for i in 0..ROT / 2 {
            let theta = pos as f64 * base.powf(-(i as f64) * 2.0 / ROT as f64);
            out[pos * ROT + 2 * i] = theta.cos() as f32;
            out[pos * ROT + 2 * i + 1] = theta.sin() as f32;
        }
    }
    out
}

fn print_attn(arm: &str, p: usize, t: &Timed) {
    // Causal attention of 16 heads over p positions: 4 · 256 flops a (query,
    // key) pair, p(p+1)/2 pairs a head.
    let flops = 16.0 * 4.0 * 256.0 * (p as f64) * (p as f64 + 1.0) / 2.0;
    println!(
        "attn\tarm={arm}\tp={p}\tms={:.3}\tms_min={:.3}\tms_p90={:.3}\ttflops_min={:.3}",
        t.us_median / 1000.0,
        t.us_min / 1000.0,
        t.us_p90 / 1000.0,
        flops / (t.us_min * 1e-6) / 1e12
    );
}

/// The shipped rescale loop's text, which the three rescale arms replace.
const RESCALE_FIND: &str = "        #pragma clang loop unroll(full)\n    for (uint c = 0u; c < PF_DCHUNKS; ++c) {\n            acc[c].thread_elements()[0] *= factor;\n            acc[c].thread_elements()[1] *= factor;\n        }\n";

/// The prefill kernel's phase arms: the shipped source with one phase of the
/// tile loop left out or rewritten by a text edit, so the difference from the
/// full launch is that phase's time (with the overlap the other phases hide it
/// under).
const ATTN_PHASES: &[(&str, &str, &str)] = &[
    (
        "append_only",
        "    // The heaviest blocks (the last positions) are the first threadgroups.\n    uint kv = group.y;",
        "    if (args.rows != 0xFFFFFFFFu) {\n        return;\n    }\n    uint kv = group.y;",
    ),
    (
        "no_load",
        "            if (key < block_keys) {\n                ulong at",
        "            if (key < 0u) {\n                ulong at",
    ),
    ("no_rescale", RESCALE_FIND, ""),
    // The rescale as it was: a diagonal fragment product behind the vote.
    (
        "diag_rescale",
        RESCALE_FIND,
        "        if (t > 0u && simd_any(factor != 1.0f)) {\n            simdgroup_float8x8 diag;\n            diag.thread_elements()[0] = fn == fm ? factor : 0.0f;\n            diag.thread_elements()[1] = fn + 1u == fm ? factor : 0.0f;\n            #pragma clang loop unroll(full)\n    for (uint c = 0u; c < PF_DCHUNKS; ++c) {\n                simdgroup_float8x8 scaled;\n                simdgroup_multiply(scaled, diag, acc[c]);\n                acc[c] = scaled;\n            }\n        }\n",
    ),
    // The elementwise rescale behind the vote.
    (
        "cond_ew",
        RESCALE_FIND,
        "        if (t > 0u && simd_any(factor != 1.0f)) {\n            #pragma clang loop unroll(full)\n    for (uint c = 0u; c < PF_DCHUNKS; ++c) {\n                acc[c].thread_elements()[0] *= factor;\n                acc[c].thread_elements()[1] *= factor;\n            }\n        }\n",
    ),
    (
        "no_exp",
        "p[j] = live[j] ? exp(sv[j] - new_m) : 0.0f;",
        "p[j] = live[j] ? (sv[j] - new_m) : 0.0f;",
    ),
];

fn attn_arms(run: &Run, device: &MetalDevice, kernels: &Kernels) -> Result<(), String> {
    let variants = if run.phases {
        ATTN_PHASES
            .iter()
            .map(|(arm, find, replace)| {
                let text = SOURCE.to_string();
                if text.matches(find).count() != 1 {
                    return Err(format!("phase {arm}: {find:?} does not occur once"));
                }
                let library =
                    Library::from_source_text(device, &text.replace(find, replace)).map_err(why)?;
                Ok((*arm, Kernels::from_library(&library).map_err(why)?))
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        Vec::new()
    };
    let stream = device.stream();
    if !run.attn.is_empty() {
        for name in [
            "attn_prefill_f16",
            "gqa256_flash_f16_rows",
            "gdn_rows",
            "gdn_prep_rows",
        ] {
            let pipeline = device.library().pipeline(name).map_err(why)?;
            println!(
                "pipeline\tname={name}\tmax_threads_per_group={}",
                pipeline.max_threads_per_group()
            );
        }
    }
    for &p in &run.attn {
        let mut s = 0x00A7_7E17u64 ^ p as u64;
        let q_n = qg_len().ok_or("q")?;
        let out_n = out_len().ok_or("out")?;
        let kv_n = KV_HEADS * HEAD_DIM;
        let cache_n = f16_len(p).ok_or("cache")?;
        let up = |v: &[f32]| device.upload(stream, v).map_err(why);
        let mut q = planted(&mut s, p * q_n, 0.5);
        // The gate halves of a gated query (the host's packed `[q | gate]` heads); a host with no gate has none.
        if q_n == QUERY_HEADS * 2 * HEAD_DIM {
            for r in 0..p {
                for head in 0..QUERY_HEADS {
                    for d in 0..HEAD_DIM {
                        q[r * q_n + head * 2 * HEAD_DIM + HEAD_DIM + d] =
                            (d % 9) as f32 * 0.5 - 2.0;
                    }
                }
            }
        }
        let q = up(&q)?;
        let k = up(&planted(&mut s, p * kv_n, 0.5))?;
        let v = up(&planted(&mut s, p * kv_n, 0.5))?;
        let ones = up(&vec![1.0f32; HEAD_DIM])?;
        let rope = up(&rope_rows(p))?;
        let mut cache_k = device.upload(stream, &vec![0u16; cache_n]).map_err(why)?;
        let mut cache_v = device.upload(stream, &vec![0u16; cache_n]).map_err(why)?;
        let mut out = up(&vec![0f32; p * out_n])?;
        device.fault_word().clear(stream).map_err(why)?;
        let step = if run.chunk == 0 { p } else { run.chunk.min(p) };
        // FNV-1a over the output's f32 bit words: two builds of the kernel agree bit for bit when the sums do.
        let sum = |arm: &str, out: &MetalBuffer<f32>| -> Result<(), String> {
            let mut host = vec![0f32; p * out_n];
            device
                .read(stream, out.whole().map_err(why)?, &mut host)
                .map_err(why)?;
            let fnv = host.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
            });
            let nan = host.iter().filter(|v| v.is_nan()).count();
            println!("attn_sum\tarm={arm}\tp={p}\tfnv={fnv:016x}\tnan={nan}");
            Ok(())
        };
        let timed = time_arm(device, stream, 1, run.trials, |encoder, _| {
            let mut from = 0usize;
            while from < p {
                let rows = step.min(p - from);
                if from > 0 {
                    encoder.memory_barrier();
                }
                kernels.attention_prefill(
                    encoder,
                    AttentionPrefillParams {
                        head_dim: HEAD_DIM,
                        query_heads: QUERY_HEADS,
                        kv_heads: KV_HEADS,
                        ctx: p,
                        eps: 1e-6,
                        layer: LAYER,
                        from: from as u32,
                        rows,
                        q: q.window(Elems(from * q_n), Elems(rows * q_n))?,
                        q_stride: q_n,
                        k: k.window(Elems(from * kv_n), Elems(rows * kv_n))?,
                        k_stride: kv_n,
                        v: v.window(Elems(from * kv_n), Elems(rows * kv_n))?,
                        v_stride: kv_n,
                        q_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        k_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        rope: rope.window(Elems(0), Elems(p * ROT))?,
                        cache: AttnCache::F16 {
                            k: cache_k.window_mut(Elems(0), Elems(cache_n))?,
                            v: cache_v.window_mut(Elems(0), Elems(cache_n))?,
                        },
                        out: out.window_mut(Elems(from * out_n), Elems(rows * out_n))?,
                        out_stride: out_n,
                        fault: device.fault_word(),
                    },
                )?;
                from += rows;
            }
            Ok(())
        })?;
        print_attn("prefill", p, &timed);
        sum("prefill", &out)?;
        for (arm, variant) in &variants {
            let timed = time_arm(device, stream, 1, run.trials, |encoder, _| {
                variant.attention_prefill(
                    encoder,
                    AttentionPrefillParams {
                        head_dim: HEAD_DIM,
                        query_heads: QUERY_HEADS,
                        kv_heads: KV_HEADS,
                        ctx: p,
                        eps: 1e-6,
                        layer: LAYER,
                        from: 0,
                        rows: p,
                        q: q.window(Elems(0), Elems(p * q_n))?,
                        q_stride: q_n,
                        k: k.window(Elems(0), Elems(p * kv_n))?,
                        k_stride: kv_n,
                        v: v.window(Elems(0), Elems(p * kv_n))?,
                        v_stride: kv_n,
                        q_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        k_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        rope: rope.window(Elems(0), Elems(p * ROT))?,
                        cache: AttnCache::F16 {
                            k: cache_k.window_mut(Elems(0), Elems(cache_n))?,
                            v: cache_v.window_mut(Elems(0), Elems(cache_n))?,
                        },
                        out: out.window_mut(Elems(0), Elems(p * out_n))?,
                        out_stride: out_n,
                        fault: device.fault_word(),
                    },
                )
            })?;
            print_attn(arm, p, &timed);
            sum(arm, &out)?;
        }
        if run.v0 && p <= 1024 {
            let segments = 5usize;
            let partials_n = partials_len(segments).ok_or("partials")? * p;
            let mut partials = up(&vec![0f32; partials_n])?;
            let mut ticket = device
                .upload(stream, &vec![0u32; p * KV_HEADS])
                .map_err(why)?;
            let timed = time_arm(device, stream, 1, run.trials, |encoder, _| {
                kernels.attention_rows(
                    encoder,
                    AttentionRowsParams {
                        head_dim: HEAD_DIM,
                        query_heads: QUERY_HEADS,
                        kv_heads: KV_HEADS,
                        segments,
                        ctx: p,
                        eps: 1e-6,
                        layer: LAYER,
                        from: 0,
                        rows: p,
                        q: q.window(Elems(0), Elems(p * q_n))?,
                        q_stride: q_n,
                        k: k.window(Elems(0), Elems(p * kv_n))?,
                        k_stride: kv_n,
                        v: v.window(Elems(0), Elems(p * kv_n))?,
                        v_stride: kv_n,
                        q_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        k_gain: gains_of(kernels.geom().attn.qk_norm, &ones)?,
                        rope: rope.window(Elems(0), Elems(p * ROT))?,
                        cache: AttnCache::F16 {
                            k: cache_k.window_mut(Elems(0), Elems(cache_n))?,
                            v: cache_v.window_mut(Elems(0), Elems(cache_n))?,
                        },
                        partials: partials.window_mut(Elems(0), Elems(partials_n))?,
                        out: out.window_mut(Elems(0), Elems(p * out_n))?,
                        out_stride: out_n,
                        fault: device.fault_word(),
                        ticket: ticket.window_mut(Elems(0), Elems(p * KV_HEADS))?,
                    },
                )
            })?;
            print_attn("v0", p, &timed);
        }
    }
    Ok(())
}

/// The planted inputs of the delta arms, `rows` rows of the largest chunk.
struct DeltaBufs {
    x: MetalBuffer<f32>,
    conv_w: MetalBuffer<f32>,
    b: MetalBuffer<f32>,
    a: MetalBuffer<f32>,
    dt: MetalBuffer<f32>,
    ssm_a: MetalBuffer<f32>,
    z: MetalBuffer<f32>,
    norm_w: MetalBuffer<f32>,
    lane: MetalBuffer<u32>,
    state: MetalBuffer<f32>,
    ring: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    g: MetalBuffer<f32>,
    prep: MetalBuffer<f32>,
}

const G_ROW: usize = VALUE_HEADS * HEAD_COLS;
const P_ROW: usize = VALUE_HEADS * 8;

fn delta_arms(run: &Run, device: &MetalDevice, kernels: &Kernels) -> Result<(), String> {
    let stream = device.stream();
    let top = *run.delta.iter().max().ok_or("no chunk sizes")?;
    let ch = channels(N_K).ok_or("channels")?;
    let mut s = 0xD311_7A5Eu64;
    let up = |v: &[f32]| device.upload(stream, v).map_err(why);
    let neg: Vec<f32> = planted(&mut s, VALUE_HEADS, 2.0)
        .into_iter()
        .map(|v| -v.abs() - 0.1)
        .collect();
    let mut bufs = DeltaBufs {
        x: up(&planted(&mut s, top * ch, 0.3))?,
        conv_w: up(&planted(&mut s, CONV_TAPS * ch, 0.5))?,
        b: up(&planted(&mut s, top * VALUE_HEADS, 1.0))?,
        a: up(&planted(&mut s, top * VALUE_HEADS, 2.0))?,
        dt: up(&planted(&mut s, VALUE_HEADS, 0.5))?,
        ssm_a: up(&neg)?,
        z: up(&planted(&mut s, top * G_ROW, 1.0))?,
        norm_w: up(&vec![1.0f32; HEAD_COLS])?,
        lane: device.upload(stream, &[0u32, 1]).map_err(why)?,
        state: up(&planted(&mut s, state_elems(2).ok_or("state")?, 0.04))?,
        ring: up(&planted(&mut s, ring_elems(N_K).ok_or("ring")?, 0.05))?,
        partials: up(&vec![0f32; top * P_ROW])?,
        g: up(&vec![0f32; top * G_ROW])?,
        prep: up(&vec![0f32; top * prepared_elems(N_K).ok_or("prep")?])?,
    };
    device.fault_word().clear(stream).map_err(why)?;
    let strides = DeltaRowStrides::dense(N_K).ok_or("strides")?;
    let mut rows_arm = |arm: &str, kernels: &Kernels, rows: usize| -> Result<(), String> {
        let dispatches = if rows >= 256 { 2 } else { 8 };
        let timed = time_arm(device, stream, dispatches, run.trials, |encoder, _| {
            kernels.delta_rows(
                encoder,
                DeltaRowsParams {
                    x: bufs.x.window(Elems(0), Elems(rows * ch))?,
                    conv_w: bufs.conv_w.window(Elems(0), Elems(CONV_TAPS * ch))?,
                    b_raw: bufs.b.window(Elems(0), Elems(rows * VALUE_HEADS))?,
                    a_raw: bufs.a.window(Elems(0), Elems(rows * VALUE_HEADS))?,
                    dt_bias: bufs.dt.window(Elems(0), Elems(VALUE_HEADS))?,
                    ssm_a: bufs.ssm_a.window(Elems(0), Elems(VALUE_HEADS))?,
                    z: bufs.z.window(Elems(0), Elems(rows * G_ROW))?,
                    norm_w: bufs.norm_w.window(Elems(0), Elems(HEAD_COLS))?,
                    lane: bufs.lane.window(Elems(0), Elems(2))?,
                    state: bufs
                        .state
                        .window_mut(Elems(0), Elems(state_elems(2).unwrap()))?,
                    ring: bufs
                        .ring
                        .window_mut(Elems(0), Elems(ring_elems(N_K).unwrap()))?,
                    partials: bufs.partials.window_mut(Elems(0), Elems(rows * P_ROW))?,
                    g: bufs.g.window_mut(Elems(0), Elems(rows * G_ROW))?,
                    prep: bufs
                        .prep
                        .window_mut(Elems(0), Elems(rows * prepared_elems(N_K).unwrap()))?,
                    faults: device.fault_word(),
                    n_v: VALUE_HEADS,
                    head: HEAD_COLS,
                    taps: CONV_TAPS,
                    n_k: N_K,
                    map: HeadMap::Tiled,
                    lanes: 2,
                    lane_at: 0,
                    lane_to: 0,
                    layer: LAYER,
                    rows,
                    pos: 100,
                    strides,
                    eps: 1e-6,
                    snaps: None,
                },
            )
        })?;
        print_delta(arm, rows, &timed);
        Ok(())
    };
    for &rows in &run.delta {
        rows_arm("rows", kernels, rows)?;
    }
    // The phase arms: the shipped source with a phase's kernel or loop gated
    // off by one text edit, as the kernel gate edits a mutant in. A phase arm
    // runs the launch with the other phases off, so the three figures sum to
    // the full launch's, less the dispatch gaps.
    for (arm, edits) in PHASES {
        let mut text = SOURCE.to_string();
        for (find, replace) in *edits {
            if text.matches(find).count() != 1 {
                return Err(format!("phase {arm}: {find:?} does not occur once"));
            }
            text = text.replace(find, replace);
        }
        let library = Library::from_source_text(device, &text).map_err(why)?;
        let variant = Kernels::from_library(&library).map_err(why)?;
        for &rows in &run.delta {
            rows_arm(arm, &variant, rows)?;
        }
    }
    let timed = time_arm(device, stream, 64, run.trials, |encoder, n| {
        let pos = [100 + n as u32];
        kernels.delta(
            encoder,
            DeltaParams {
                x: bufs.x.window(Elems(0), Elems(ch))?,
                conv_w: bufs.conv_w.window(Elems(0), Elems(CONV_TAPS * ch))?,
                b_raw: bufs.b.window(Elems(0), Elems(VALUE_HEADS))?,
                a_raw: bufs.a.window(Elems(0), Elems(VALUE_HEADS))?,
                dt_bias: bufs.dt.window(Elems(0), Elems(VALUE_HEADS))?,
                ssm_a: bufs.ssm_a.window(Elems(0), Elems(VALUE_HEADS))?,
                z: bufs.z.window(Elems(0), Elems(G_ROW))?,
                norm_w: bufs.norm_w.window(Elems(0), Elems(HEAD_COLS))?,
                lane: bufs.lane.window(Elems(0), Elems(2))?,
                state: bufs
                    .state
                    .window_mut(Elems(0), Elems(state_elems(2).unwrap()))?,
                ring: bufs
                    .ring
                    .window_mut(Elems(0), Elems(ring_elems(N_K).unwrap()))?,
                partials: bufs.partials.window_mut(Elems(0), Elems(P_ROW))?,
                g: bufs.g.window_mut(Elems(0), Elems(G_ROW))?,
                faults: device.fault_word(),
                n_v: VALUE_HEADS,
                head: HEAD_COLS,
                taps: CONV_TAPS,
                n_k: N_K,
                map: HeadMap::Tiled,
                lanes: 2,
                lane_at: 0,
                lane_to: 0,
                layer: LAYER,
                pos: &pos,
                eps: 1e-6,
            },
        )
    })?;
    print_delta("step", 1, &timed);
    Ok(())
}
