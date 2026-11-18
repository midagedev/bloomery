//! V4.1 host expert tier gate, on seeded inputs with no reference engine's dump behind them. Two
//! arms over every routed layer of the file the tree runs, each over `N_TOKENS` seeded activation
//! columns (`columns::seeded`) with per-token lists of `n_used` distinct experts at positive
//! weights.
//!
//! The routed clamp is crossed on purpose, on any file: each layer's experts for token 0 run again
//! on its row scaled by the power of two that brings the largest gate, up and −up to twice the
//! limit, so `silu(g) > L`, `u > L` and `u < −L` all occur; every slot's combine must be
//! `qdot::swiglu_clamp`'s bits on the same dots and the clamp's f64 statement within
//! [`COMBINE_EVAL`].
//!
//! The r8 arm: with `BLOOMERY_R8` on (unset), every layer is built a second time with its routed
//! gate and up read from the file's r8 sidecar (`host::layer` over the file's r8 pair) and must
//! give every token's partial sum bit for bit as the file's layer does; no sidecar at its path
//! refuses the run by name (`HostR8::at_gate`). `BLOOMERY_R8=off` runs the file's layer alone.
//!
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries the routed partial sum against ik's `ffn_moe_out` (the band over the host tier's three stages) and the layers' routing and activations are seeded
//!
//! `hw_`: needs the V4.1 shards on the box (`just gate-ds41-host`); `BLOOMERY_V41_MODEL` names
//! another first shard.

#[path = "common/columns.rs"]
mod columns;

use gguf::Split;
use model::arch::deepseek41::host;
use model::arch::deepseek41::hparams::Hparams;
use model::moe::{HostLayer, HostScratch};
use model::ops::Tensor2;
use model::placement::workstation;
use model::r8file::{HostR8, R8Source};

/// The seeded tokens a layer's columns and lists have.
const N_TOKENS: usize = 5;

/// f32's unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;
/// `v_silu`'s relative error: `v_expf` within 1.5 ulp, then one add and one
/// divide.
const SILU_EVAL: f64 = 6.0 * U;
/// The clamped combine's relative error against its f64 statement: `v_silu`
/// within [`SILU_EVAL`], the clamps exact (they move no value further than
/// their input moved), then one product rounding.
const COMBINE_EVAL: f64 = SILU_EVAL + U + SILU_EVAL * U;
/// Half the spacing of f32's subnormals, 2^-150: a product that lands below
/// `f32::MIN_POSITIVE` rounds by at most this much, whatever its relative
/// error.
const HALF_SUBNORMAL: f64 = f32::from_bits(1) as f64 / 2.0;

/// `BLOOMERY_R8` as the engine takes it: the registry's on/off row, parsed by
/// the levers crate (`Levers::host`); unset is on.
fn r8_lever() -> bool {
    bloomery_levers::Levers::from_env()
        .unwrap_or_else(|e| panic!("the levers: {e}"))
        .host()
        .r8
}

/// `|got − want| / band`, a NaN anywhere reading as a miss.
fn ratio(got: f64, want: f64, band: f64) -> f64 {
    let r = (got - want).abs() / band;
    if r.is_nan() { f64::INFINITY } else { r }
}

/// `x·σ(x)` in f64.
fn silu64(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// The smallest power of two `c` with `c·v >= target`, or 1 when `v` is not
/// positive: a row whose dots never reach that side keeps its crossings at 0
/// and [`synthetic_clamp`] fails by name.
fn pow2_reaching(v: f32, target: f32) -> f32 {
    if v.is_nan() || v <= 0.0 {
        return 1.0;
    }
    let mut c = 1.0f32;
    while c * v < target && c < 1e30 {
        c *= 2.0;
    }
    c
}

/// Every slot's gate and up dots of the last call, slot after slot.
fn slot_dots(scratch: &HostScratch, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut g, mut u) = (Vec::new(), Vec::new());
    for s in 0..n {
        g.extend_from_slice(&scratch.gate(s).data);
        u.extend_from_slice(&scratch.up(s).data);
    }
    (g, u)
}

/// The routed clamp crossed on purpose (module doc): layer `l`'s experts for
/// `list` on `x` scaled by a power of two — exact through the q8_K
/// quantization, so the dots scale with it — until the largest gate, up and
/// −up reach twice the limit. Every slot's combine must equal
/// `qdot::swiglu_clamp` on its own dots bit for bit and the f64 statement
/// `clamp(u, ±L) · min(silu(g), L)` within [`COMBINE_EVAL`] and
/// [`HALF_SUBNORMAL`]; where `e^-g`, within its 4u, can pass `f32::MAX`,
/// `v_silu` may flush to -0, so there the value itself is the band. All three
/// crossings must occur. Prints its line and returns the verdict.
fn synthetic_clamp(
    layer: &HostLayer,
    src: R8Source<'_>,
    l: usize,
    x: &[f32],
    list: &[(u32, f32)],
    scratch: &mut HostScratch,
) -> bool {
    let limit = layer.swiglu_limit();
    let ff = scratch.gate(0).data.len();
    let mut out = vec![0.0f32; x.len()];
    let mut run = |xs: Vec<f32>, scratch: &mut HostScratch| {
        let t = Tensor2::from_vec(xs.len(), 1, xs);
        layer
            .experts_into(src, &t, list, &mut out, scratch)
            .unwrap_or_else(|e| panic!("layer {l} synthetic clamp: {e}"));
    };
    run(x.to_vec(), scratch);
    let (g1, u1) = slot_dots(scratch, list.len());
    let top = |v: &[f32], sign: f32| v.iter().fold(0.0f32, |m, &a| m.max(sign * a));
    let c = [top(&g1, 1.0), top(&u1, 1.0), top(&u1, -1.0)]
        .iter()
        .map(|&v| pow2_reaching(v, 2.0 * limit))
        .fold(1.0f32, f32::max);
    run(x.iter().map(|&v| v * c).collect(), scratch);
    let (g, u) = slot_dots(scratch, list.len());
    let lim = f64::from(limit);
    let (mut bits_equal, mut stated) = (true, 0.0f64);
    let (mut silu_hi, mut up_hi, mut up_lo) = (0usize, 0usize, 0usize);
    for s in 0..list.len() {
        let (gs, us) = (&g[s * ff..(s + 1) * ff], &u[s * ff..(s + 1) * ff]);
        let h = &scratch.par(s).data;
        let mut want = vec![f32::NAN; ff];
        qdot::swiglu_clamp(gs, us, limit, &mut want);
        bits_equal &= h.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
        for ((&hv, &gv), &uv) in h.iter().zip(gs).zip(us) {
            let (gd, ud) = (f64::from(gv), f64::from(uv));
            let sj = silu64(gd);
            silu_hi += usize::from(sj > lim);
            up_hi += usize::from(ud > lim);
            up_lo += usize::from(ud < -lim);
            let e = sj.min(lim) * ud.clamp(-lim, lim);
            let flush = (-gd).exp() * (1.0 + 4.0 * U) >= f64::from(f32::MAX);
            let rel = if flush {
                1.0 + COMBINE_EVAL
            } else {
                COMBINE_EVAL
            };
            if f64::from(hv) != e {
                stated = stated.max(ratio(f64::from(hv), e, rel * e.abs() + HALF_SUBNORMAL));
            }
        }
    }
    let pass = bits_equal && stated <= 1.0 && silu_hi > 0 && up_hi > 0 && up_lo > 0;
    println!(
        "clamp_synthetic layer={l} slots={} x_scale={c} limit={limit} silu>L={silu_hi} u>L={up_hi} \
         u<-L={up_lo} par_eq_swiglu_clamp={bits_equal} stated_ratio={stated:.3} {}",
        list.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Layer `l`'s seeded activation columns, `N_TOKENS` of `embd`.
fn seeded_x(l: usize, embd: usize) -> Vec<f32> {
    columns::seeded(N_TOKENS * embd, 0x7141_0000 + l as u64)
}

/// Layer `l`'s seeded routing: per token `n_used` distinct experts of `n_expert` at positive
/// weights.
fn seeded_lists(l: usize, n_expert: usize, n_used: usize) -> Vec<Vec<(u32, f32)>> {
    let mut state = 0x7141_5EED ^ (l as u64 + 1);
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    (0..N_TOKENS)
        .map(|_| {
            let mut ids: Vec<u32> = Vec::with_capacity(n_used);
            while ids.len() < n_used {
                let e = u32::try_from(next() % n_expert as u64).expect("an expert id");
                if !ids.contains(&e) {
                    ids.push(e);
                }
            }
            ids.into_iter()
                .map(|e| (e, 0.25 + ((next() >> 40) as f32) / 16_777_216.0))
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "hw: needs the box and the V4.1 shards"]
fn hw_ds41_host_clamp_and_r8_arms() {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    // The MoE module's own metadata reader takes the file's expert count as it is.
    let meta = model::moe::Meta::read(split.shard(0).expect("a split has shard 0"))
        .unwrap_or_else(|e| panic!("moe::Meta of {path}: {e}"));
    assert_eq!(
        (meta.n_expert, meta.n_used, meta.ff),
        (hp.experts.n_expert, hp.experts.n_used, hp.experts.ff),
        "moe::Meta must read V4.1's expert shape as Hparams does"
    );
    println!(
        "moe::Meta n_expert={} n_used={} ff={} (as Hparams)",
        meta.n_expert, meta.n_used, meta.ff
    );
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let mut scratch = HostScratch::new(embd, ff, n_used).expect("a scratch of n_used experts");
    let r8 = HostR8::at_gate(&split, r8_lever())
        .unwrap_or_else(|e| panic!("the r8 reading of {path}: {e}"));
    let rows = R8Source::rows(&split);
    let pair = R8Source::of(&split, &r8).unwrap_or_else(|e| panic!("the r8 pair of {path}: {e}"));
    let mut scratch_r8 = HostScratch::new(embd, ff, n_used).expect("a scratch of n_used experts");
    let mut failed: Vec<usize> = Vec::new();
    let mut clamp_failed: Vec<usize> = Vec::new();
    let mut clamped = 0;
    let mut routed = 0;
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(rows, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
        else {
            println!("layer={l} no routed experts");
            continue;
        };
        routed += 1;
        let r8_layer = r8.sidecar().map(|_| {
            host::layer(pair, &hp, l)
                .unwrap_or_else(|e| panic!("layer {l} from the r8 sidecar: {e}"))
                .unwrap_or_else(|| panic!("layer {l} routes in the file and not from the sidecar"))
        });
        let limit = layer.swiglu_limit();
        let x_all = seeded_x(l, embd);
        let lists = seeded_lists(l, hp.experts.n_expert, n_used);
        let mut r8_cells = 0usize;
        if let Some(r8_layer) = &r8_layer {
            for (t, list) in lists.iter().enumerate() {
                let x = Tensor2::from_vec(embd, 1, x_all[t * embd..(t + 1) * embd].to_vec());
                let mut out = vec![0.0f32; embd];
                layer
                    .experts_into(rows, &x, list, &mut out, &mut scratch)
                    .unwrap_or_else(|e| panic!("layer {l} token {t}: {e}"));
                let mut out_r8 = vec![0.0f32; embd];
                r8_layer
                    .experts_into(pair, &x, list, &mut out_r8, &mut scratch_r8)
                    .unwrap_or_else(|e| panic!("layer {l} token {t} r8: {e}"));
                r8_cells += columns::diff_cells(&out_r8, &out);
            }
        }
        println!(
            "layer={l} limit={limit} tokens={N_TOKENS} r8_cells_off={} {}",
            if r8_layer.is_some() {
                r8_cells.to_string()
            } else {
                format!("- ({r8})")
            },
            if r8_cells == 0 { "PASS" } else { "FAIL" }
        );
        if r8_cells != 0 {
            failed.push(l);
        }
        if limit > 1e-6 {
            clamped += 1;
            if !synthetic_clamp(&layer, rows, l, &x_all[..embd], &lists[0], &mut scratch) {
                clamp_failed.push(l);
            }
        } else {
            println!("clamp_synthetic layer={l} limit={limit} no routed clamp");
        }
    }
    println!("ds41_host layers={routed} r8 failed={failed:?}");
    assert!(
        clamped > 0,
        "no routed layer carries a SwiGLU limit above 1e-6: the clamp is not exercised"
    );
    assert!(
        failed.is_empty() && clamp_failed.is_empty(),
        "layers where the r8 sidecar's partial sums differ: {failed:?}; layers where the crossed clamp does not hold: {clamp_failed:?}"
    );
    println!(
        "PASSED: ds41_host — the routed clamp crossed on all three sides on {clamped} layers; {r8}{}",
        if r8.sidecar().is_some() {
            ", its partial sums the file's bit for bit on every layer"
        } else {
            ""
        }
    );
}
