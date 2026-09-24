//! V4.1 host expert tier gate. For every layer of the 5-token oracle set
//! `ref_deepseek41` (`# build db517b69`, ik on the CPU), the host tier
//! (`moe::HostLayer`) serves each token's routed experts with ik's own routing
//! injected — the ids of `ffn_moe_topk-L` (its logical twin; the flat one is
//! the view's raw memory) and the weights of `ffn_moe_weights_scaled-L`, the
//! node ik's `MUL_MULTI_ADD` reads — and its partial sum is compared with
//! `ffn_moe_out-L`, the weighted expert sum before the shared expert joins
//! (`ffn_out-L = ffn_moe_out-L + ffn_shexp-L`). Every layer is in it: 0 and 1
//! (down q5_K) and the shard-crossing 7, 14, 27 and 34 among them. Each layer
//! prints its line: the three stages against their bands, the flips, and the
//! clamp hits the set's routing reaches — which layers those are is the file's.
//!
//! The routed clamp is crossed on purpose, on any file: each layer's experts
//! for token 0 run again on its row scaled by the power of two that brings the
//! largest gate, up and −up to twice the limit, so `silu(g) > L`, `u > L` and
//! `u < −L` all occur; every slot's combine must be `qdot::swiglu_clamp`'s bits
//! on the same dots and the clamp's f64 statement within [`COMBINE_EVAL`].
//!
//! `hw_`: needs the V4.1 shards and the oracle set on the box
//! (`just gate-ds41-host`); `BLOOMERY_V41_MODEL` names another first shard.
//!
//! The band, derived before the run. Both engines quantize alike: for the
//! q3_K gate and up, ik's q8_K (`iqk_quantize_row_q8_K_T`, AVX2: codes
//! `round(fl(127/M)·x)`, `d = fl(M/127)`) and ours (qdot's q8_K: codes
//! `nearest_int(fl(-127/max)·x)`, `d = fl(1/iscale)`) give the same codes up to
//! sign and scales within three roundings, `|δd/d| ≤ γ_3`; for the q4_K/q5_K
//! down, both run the same q8_2_x4 encoder (bit for bit, `gate-qdot`). A
//! difference has three sources, each bounded or counted:
//!
//! 1. Float order inside a dot. Every kernel here sums leaves — a float scale
//!    times an exact integer sum — into f32 lanes by FMA and ends in an
//!    eight-lane hsum. A leaf's path holds one product rounding, at most three
//!    FMAs per super-block (ik's q8_2_x4 K-quant kernel: the min term and two
//!    halves, the longest of the four kernels; ggml's q3_K takes one, iqk's
//!    two) and three hsum levels, so `n(k) = 4·(k/256) + 8` roundings bound it:
//!    `|dot − exact| ≤ γ_n · Σ|leaves|`. `Σ|leaves|` is bounded per super-block
//!    `b` from the row's own scales and the activation's magnitudes
//!    `A_b = Σ|a|`: q3_K ours (signed `q`, `|sc| ≤ 32`, `|q| ≤ 4`)
//!    `128·d_b·A_b`, ik's (the offset as a min term: `q + 4 ≤ 7` plus 4)
//!    `352·d_b·A_b`; q4_K/q5_K, both engines, `63·(15|31)·d_b·A_b +
//!    63·dmin_b·A_b`.
//! 2. The gate/up difference through the combine: SiLU's slope is at most
//!    1.1, its f32 evaluation (`v_silu`, the same code in both engines) is
//!    within 6u, `min` and the clamp are 1-Lipschitz, and the product rounds
//!    once — `E_h` below.
//! 3. h's quantization. Where the two h sit on either side of a rounding
//!    boundary, a code moves by one: not bounded but counted. Our down kernel
//!    run on ik's h gives `d̃`; `F = d_ours − d̃` is the flip term, exactly, and
//!    `d̃ − d_ik` is kernel against kernel on the same bytes (1.). Every code of
//!    the two quantized h must be within one of the other — more is not a flip.
//!
//! The partial sum then satisfies `|out_ours − out_ik − Σ w·F| ≤ Σ|w|·E_R +
//! γ_{2·n_used}·(Σ|w·d_ours| + Σ|w·d_ik|)`: each engine's weighted sum rounds
//! at most twice per term.

use std::collections::HashMap;
use std::path::PathBuf;

use gguf::quant::half_to_f32;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::host;
use model::arch::deepseek41::hparams::Hparams;
use model::moe::{HostLayer, HostScratch};
use model::ops::{Tensor2, Weight, matmul_q_group_into};
use model::placement::workstation;

/// The oracle set and the ik tree it was dumped from.
const SET: &str = "ref_deepseek41";
// PIN(2026-09-23): the sink-fixed oracle tree; this set's bytes are those
// 49ef19d0 dumped, since the fixed branch does not run in a 5-token prefill.
const BUILD: &str = "db517b69";

/// f32's unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;
/// The largest slope of `x·σ(x)` (1.0998 at x ≈ 2.4).
const SILU_SLOPE: f64 = 1.1;
/// `v_silu`'s relative error: `v_expf` within 1.5 ulp, then one add and one
/// divide.
const SILU_EVAL: f64 = 6.0 * U;
/// `Σ|leaves|` per unit `d_b·A_b` of a q3_K row: ours, ik's (see the module doc).
const Q3K_LEAF_OURS: f64 = 128.0;
const Q3K_LEAF_IK: f64 = 352.0;
/// The largest 6-bit scale or min of a q4_K/q5_K sub-block.
const KQ_SCALE_MAX: f64 = 63.0;
/// The clamped combine's relative error against its f64 statement: `v_silu`
/// within [`SILU_EVAL`], the clamps exact (they move no value further than
/// their input moved), then one product rounding.
const COMBINE_EVAL: f64 = SILU_EVAL + U + SILU_EVAL * U;
/// Half the spacing of f32's subnormals, 2^-150: a product that lands below
/// `f32::MIN_POSITIVE` rounds by at most this much, whatever its relative
/// error.
const HALF_SUBNORMAL: f64 = f32::from_bits(1) as f64 / 2.0;

/// `n·u / (1 − n·u)`: the relative bound of `n` roundings.
fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
}

/// Roundings on the longest leaf path of a `k`-long dot (module doc, 1.).
fn n_dot(k: usize) -> f64 {
    4.0 * (k / 256) as f64 + 8.0
}

/// One oracle set: its header, its tensor rows' shapes and the files of its
/// logical integer twins.
struct Set {
    dir: PathBuf,
    shapes: HashMap<String, [usize; 3]>,
    logical: HashMap<String, String>,
}

impl Set {
    fn open(split: &Split) -> Set {
        let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
        // The set of that name for the V4.1 file the tree runs.
        let dir = PathBuf::from(base).join(gguf::v41::set(SET));
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "no oracle manifest at {} ({e}). The lead produces it (tools/ref/dump.sh). \
                 Do not run it yourself and do not skip this test.",
                path.display()
            )
        });
        let mut header = HashMap::new();
        let mut shapes = HashMap::new();
        let mut logical = HashMap::new();
        for line in text.lines() {
            if let Some(h) = line.strip_prefix("# ") {
                if let Some((k, v)) = h.split_once('\t') {
                    header.insert(k.to_string(), v.to_string());
                }
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "tensor" if f[2] == "0" && f[11] == "1" => {
                    let ne = |i: usize| {
                        f[i].parse::<usize>()
                            .unwrap_or_else(|e| panic!("{line}: {e}"))
                    };
                    shapes.insert(f[1].to_string(), [ne(4), ne(5), ne(6)]);
                }
                "int" if f[2] == "0" && f[6] == "logical" => {
                    logical.insert(f[1].to_string(), f[11].to_string());
                }
                _ => {}
            }
        }
        let get = |k: &str| header.get(k).map(String::as_str);
        assert_eq!(
            get("build"),
            Some(BUILD),
            "{SET} was dumped from another ik tree"
        );
        assert_eq!(get("arch"), Some("deepseek41"), "{SET} is not a V4.1 set");
        assert!(
            header.contains_key("complete"),
            "{SET}: the manifest has no completion trailer — the dump that wrote it did not finish"
        );
        let ours = split
            .shard_path(0)
            .and_then(std::path::Path::file_name)
            .map(|n| n.to_string_lossy().into_owned());
        assert_eq!(
            get("model_file").map(str::to_string),
            ours,
            "{SET} was dumped from another file"
        );
        // Both V4.1 files name their shards alike: the path tells them apart.
        assert_eq!(
            get("model").map(str::to_string),
            split
                .shard_path(0)
                .map(|p| p.to_string_lossy().into_owned()),
            "{SET} was dumped from another file"
        );
        Set {
            dir,
            shapes,
            logical,
        }
    }

    fn bytes(&self, file: &str) -> Vec<u8> {
        let path = self.dir.join(file);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// A contiguous f32 node of shape `ne` (`[ne0, ne1, ne2]`).
    fn f32s(&self, name: &str, ne: [usize; 3]) -> Vec<f32> {
        let got = self
            .shapes
            .get(name)
            .unwrap_or_else(|| panic!("{SET}: no contiguous node {name}"));
        assert_eq!(*got, ne, "{SET}: {name}'s shape");
        let b = self.bytes(&format!("{name}.0.f32"));
        assert_eq!(
            b.len(),
            4 * ne.iter().product::<usize>(),
            "{name}: file length"
        );
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// The logical twin of an integer node: `count` values in logical order.
    fn i32s_logical(&self, name: &str, count: usize) -> Vec<i32> {
        let file = self
            .logical
            .get(name)
            .unwrap_or_else(|| panic!("{SET}: no logical twin of {name}"));
        let b = self.bytes(file);
        assert_eq!(b.len(), 4 * count, "{name}: logical twin length");
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }
}

/// `A_b = Σ|a|` per 256-value block of `x`'s q8_K activation (ours).
fn q8k_block_abs(x: &[f32]) -> Vec<f64> {
    let mut col = vec![0u8; qdot::col_bytes(GgmlType::Q3_K, x.len())];
    qdot::quantize_col(GgmlType::Q3_K, x, &mut col);
    col.as_chunks::<296>()
        .0
        .iter()
        .map(|b| {
            let d = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).abs();
            d * b[8..264]
                .iter()
                .map(|&q| f64::from(i8::from_le_bytes([q])).abs())
                .sum::<f64>()
        })
        .collect()
}

/// `h`'s q8_2_x4 activation for weight type `ty`: per 32-value block its
/// bf16 scale bits and its codes.
fn q82x4_blocks(ty: GgmlType, h: &[f32]) -> Vec<(u16, [i8; 32])> {
    let mut col = vec![0u8; qdot::col_bytes(ty, h.len())];
    qdot::quantize_col(ty, h, &mut col);
    col.as_chunks::<144>()
        .0
        .iter()
        .flat_map(|g| {
            (0..4).map(move |ir| {
                let bits = u16::from_le_bytes([g[2 * ir], g[2 * ir + 1]]);
                let mut q = [0i8; 32];
                for (qi, &c) in q.iter_mut().zip(&g[16 + 32 * ir..16 + 32 * (ir + 1)]) {
                    *qi = i8::from_le_bytes([c]);
                }
                (bits, q)
            })
        })
        .collect()
}

/// The q3_K bound of each row of `w`: `D_j = Σ_b d_b·A_b` (`d_b` the row's
/// super-block scale, f16 at byte 108 of its 110).
fn q3k_rows(w: &Weight<'_>, a: &[f64]) -> Vec<f64> {
    assert_eq!(w.ty(), GgmlType::Q3_K);
    let rb = w.bytes().len() / w.n();
    (0..w.n())
        .map(|j| {
            let row = &w.bytes()[j * rb..(j + 1) * rb];
            row.as_chunks::<110>()
                .0
                .iter()
                .zip(a)
                .map(|(b, &ab)| {
                    f64::from(half_to_f32(u16::from_le_bytes([b[108], b[109]]))).abs() * ab
                })
                .sum()
        })
        .collect()
}

/// The q4_K/q5_K `Σ|leaves|` bound of each row of `w` against block
/// magnitudes `a` (per 256): `Σ_b 63·(q_max·d_b + dmin_b)·A_b`, `d` and
/// `dmin` the f16 pair opening each super-block.
fn kquant_rows(w: &Weight<'_>, a: &[f64]) -> Vec<f64> {
    let (q_max, block) = match w.ty() {
        GgmlType::Q4_K => (15.0, 144),
        GgmlType::Q5_K => (31.0, 176),
        other => panic!("no down bound for {other}"),
    };
    let rb = w.bytes().len() / w.n();
    (0..w.n())
        .map(|i| {
            let row = &w.bytes()[i * rb..(i + 1) * rb];
            row.chunks_exact(block)
                .zip(a)
                .map(|(b, &ab)| {
                    let d = f64::from(half_to_f32(u16::from_le_bytes([b[0], b[1]]))).abs();
                    let dmin = f64::from(half_to_f32(u16::from_le_bytes([b[2], b[3]]))).abs();
                    KQ_SCALE_MAX * (q_max * d + dmin) * ab
                })
                .sum()
        })
        .collect()
}

/// A layer's verdicts and counts.
#[derive(Default)]
struct Layer {
    /// Worst `|Δ| / band` of each stage: h, the down on the same bytes, the sum.
    h: f64,
    down: f64,
    out: f64,
    /// Slots whose down on ik's h equals ik's down bit for bit.
    down_bits: usize,
    slots: usize,
    /// Codes of the two quantized h that differ, 32-value blocks whose scale
    /// moved, and the largest code step.
    flips: usize,
    scale_moves: usize,
    step: i32,
    /// `max |out_ours − out_ik|` and `max |Σ w·F|` over the layer.
    raw: f64,
    flip_sum: f64,
    /// Clamp hits of the routed combine: `silu(g) > L`, `u > L`, `u < −L`, and
    /// the hits an unclamped h would put outside the h band.
    silu_hi: usize,
    up_hi: usize,
    up_lo: usize,
    visible: usize,
}

impl Layer {
    fn pass(&self) -> bool {
        self.h <= 1.0 && self.down <= 1.0 && self.out <= 1.0 && self.step <= 1
    }
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
    split: &Split,
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
            .experts_into(split, &t, list, &mut out, scratch)
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

#[test]
#[ignore = "hw: needs the box, the V4.1 shards and $BLOOMERY_DATA/ref_deepseek41"]
fn hw_ds41_host_matches_ik_routed_sum() {
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
    let set = Set::open(&split);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    let mut scratch = HostScratch::new(embd, ff);
    let (e_gu, e_down) = (gamma(n_dot(embd)), gamma(n_dot(ff)));
    let g3 = gamma(3.0);
    let mut failed: Vec<usize> = Vec::new();
    let mut clamp_failed: Vec<usize> = Vec::new();
    let mut clamped = 0;
    let mut worst = Layer::default();
    let mut routed = 0;
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(&split, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
        else {
            println!("layer={l} no routed experts");
            continue;
        };
        routed += 1;
        let limit = layer.swiglu_limit();
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        let ids = set.i32s_logical(&format!("ffn_moe_topk-{l}"), n_used * n_tokens);
        let ws = set.f32s(
            &format!("ffn_moe_weights_scaled-{l}"),
            [1, n_used, n_tokens],
        );
        let h_ik_all = set.f32s(&format!("ffn_moe_gate_par-{l}"), [ff, n_used, n_tokens]);
        let d_ik_all = set.f32s(&format!("ffn_moe_down-{l}"), [embd, n_used, n_tokens]);
        let out_ik_all = set.f32s(&format!("ffn_moe_out-{l}"), [embd, n_tokens, 1]);
        let [sg, su, sd] = layer.stacks();
        let mut st = Layer::default();
        let mut down_ty = GgmlType::F32;
        for t in 0..n_tokens {
            let x = Tensor2::from_vec(embd, 1, x_all[t * embd..(t + 1) * embd].to_vec());
            let list: Vec<(u32, f32)> = (0..n_used)
                .map(|s| {
                    let id = ids[t * n_used + s];
                    let e = u32::try_from(id).unwrap_or_else(|_| {
                        panic!("layer {l} token {t}: routed id {id} is negative")
                    });
                    (e, ws[t * n_used + s])
                })
                .collect();
            let mut out = vec![0.0f32; embd];
            layer
                .experts_into(&split, &x, &list, &mut out, &mut scratch)
                .unwrap_or_else(|e| panic!("layer {l} token {t}: {e}"));
            // x's magnitudes per block, ik's within γ_3 of ours.
            let ax: Vec<f64> = q8k_block_abs(&x.data);
            let (mut flip, mut band_r, mut terms) =
                (vec![0.0f64; embd], vec![0.0f64; embd], vec![0.0f64; embd]);
            for (s, &(e, w)) in list.iter().enumerate() {
                let e = e as usize;
                let w = f64::from(w);
                let (wg, wu, wd) = (
                    sg.expert(&split, e).unwrap(),
                    su.expert(&split, e).unwrap(),
                    sd.expert(&split, e).unwrap(),
                );
                down_ty = wd.ty();
                let (g, u, h, d) = (
                    &scratch.gate(s).data,
                    &scratch.up(s).data,
                    &scratch.par(s).data,
                    &scratch.down(s).data,
                );
                let h_ik = &h_ik_all[(t * n_used + s) * ff..(t * n_used + s + 1) * ff];
                let d_ik = &d_ik_all[(t * n_used + s) * embd..(t * n_used + s + 1) * embd];

                // Stage A: h against ik's, E_h per element (module doc, 1. and 2.).
                let (dg, du) = (q3k_rows(&wg, &ax), q3k_rows(&wu, &ax));
                let leaf = e_gu * (Q3K_LEAF_OURS + Q3K_LEAF_IK * (1.0 + g3)) + g3 * Q3K_LEAF_OURS;
                let lim = f64::from(limit);
                for j in 0..ff {
                    let (gj, uj) = (f64::from(g[j]), f64::from(u[j]));
                    let (e_g, e_u) = (leaf * dg[j], leaf * du[j]);
                    let sj = gj / (1.0 + (-gj).exp());
                    let m = sj.min(lim);
                    let c = uj.clamp(-lim, lim);
                    let e_m = SILU_SLOPE * e_g + SILU_EVAL * (2.0 * sj.abs() + SILU_SLOPE * e_g);
                    let e_h = (c.abs() * e_m
                        + (m.abs() + SILU_EVAL * sj.abs() + e_m) * e_u
                        + 2.0 * U * f64::from(h[j]).abs())
                        * (1.0 + 4.0 * U);
                    st.h = st.h.max(ratio(f64::from(h[j]), f64::from(h_ik[j]), e_h));
                    let hit = (sj > lim, uj > lim, uj < -lim);
                    st.silu_hi += usize::from(hit.0);
                    st.up_hi += usize::from(hit.1);
                    st.up_lo += usize::from(hit.2);
                    if (hit.0 || hit.1 || hit.2) && ratio(sj * uj, f64::from(h_ik[j]), e_h) > 1.0 {
                        st.visible += 1;
                    }
                }

                // Stage B: the two quantized h differ by flips only; our down
                // on ik's h against ik's down (module doc, 3. and 1.).
                let (qo, qi) = (q82x4_blocks(wd.ty(), h), q82x4_blocks(wd.ty(), h_ik));
                for ((so, co), (si, ci)) in qo.iter().zip(&qi) {
                    st.scale_moves += usize::from(so != si);
                    for (&a, &b) in co.iter().zip(ci) {
                        let step = (i32::from(a) - i32::from(b)).abs();
                        st.flips += usize::from(step != 0);
                        st.step = st.step.max(step);
                    }
                }
                let a_h: Vec<f64> = qi
                    .chunks(8)
                    .map(|sb| {
                        sb.iter()
                            .map(|(bits, q)| {
                                let dq = f64::from(f32::from_bits(u32::from(*bits) << 16)).abs();
                                dq * q.iter().map(|&v| f64::from(v).abs()).sum::<f64>()
                            })
                            .sum()
                    })
                    .collect();
                let sdn = kquant_rows(&wd, &a_h);
                let h_ik_t = Tensor2::from_vec(ff, 1, h_ik.to_vec());
                let mut on_ik = [Tensor2::from_vec(embd, 1, vec![f32::NAN; embd])];
                matmul_q_group_into("ds41_host_gate", &[wd], &[&h_ik_t], &mut on_ik)
                    .unwrap_or_else(|err| panic!("layer {l} token {t} expert {e}: {err}"));
                let dt = &on_ik[0].data;
                st.slots += 1;
                st.down_bits +=
                    usize::from(dt.iter().zip(d_ik).all(|(a, b)| a.to_bits() == b.to_bits()));
                for i in 0..embd {
                    let e_r = 2.0 * e_down * sdn[i];
                    st.down = st
                        .down
                        .max(ratio(f64::from(dt[i]), f64::from(d_ik[i]), e_r));
                    flip[i] += w * (f64::from(d[i]) - f64::from(dt[i]));
                    band_r[i] += w.abs() * e_r;
                    terms[i] += (w * f64::from(d[i])).abs() + (w * f64::from(d_ik[i])).abs();
                }
            }

            // Stage C: the partial sum, flips accounted.
            let out_ik = &out_ik_all[t * embd..(t + 1) * embd];
            let e_sum = gamma(2.0 * n_used as f64);
            for i in 0..embd {
                let (o, oi) = (f64::from(out[i]), f64::from(out_ik[i]));
                let band = band_r[i] + e_sum * terms[i];
                st.out = st.out.max(ratio(o - flip[i], oi, band));
                st.raw = st.raw.max((o - oi).abs());
                st.flip_sum = st.flip_sum.max(flip[i].abs());
            }
        }
        let shards = [sg.shard(), su.shard(), sd.shard()];
        println!(
            "layer={l} down={down_ty} shards(gate,up,down)={shards:?} limit={limit} tokens={n_tokens} slots={} \
             h_ratio={:.3e} flips={} scale_moves={} max_code_step={} down_ratio={:.3e} down_bits_equal={}/{} \
             out_ratio={:.3e} max_abs_diff={:.3e} max_flip_term={:.3e} \
             clamp_hits silu>L={} u>L={} u<-L={} visible={} {}",
            st.slots,
            st.h,
            st.flips,
            st.scale_moves,
            st.step,
            st.down,
            st.down_bits,
            st.slots,
            st.out,
            st.raw,
            st.flip_sum,
            st.silu_hi,
            st.up_hi,
            st.up_lo,
            st.visible,
            if st.pass() { "PASS" } else { "FAIL" }
        );
        if !st.pass() {
            failed.push(l);
        }
        worst.h = worst.h.max(st.h);
        worst.down = worst.down.max(st.down);
        worst.out = worst.out.max(st.out);
        worst.flips += st.flips;
        worst.silu_hi += st.silu_hi;
        worst.up_hi += st.up_hi;
        worst.up_lo += st.up_lo;
        worst.visible += st.visible;
        if limit > 1e-6 {
            clamped += 1;
            let list: Vec<(u32, f32)> = (0..n_used)
                .map(|s| {
                    let e = u32::try_from(ids[s])
                        .unwrap_or_else(|_| panic!("layer {l}: routed id {} is negative", ids[s]));
                    (e, ws[s])
                })
                .collect();
            if !synthetic_clamp(&layer, &split, l, &x_all[..embd], &list, &mut scratch) {
                clamp_failed.push(l);
            }
        } else {
            println!("clamp_synthetic layer={l} limit={limit} no routed clamp");
        }
    }
    println!(
        "ds41_host layers={routed} worst h_ratio={:.3e} down_ratio={:.3e} out_ratio={:.3e} flips={} \
         clamp_hits silu>L={} u>L={} u<-L={} visible={} failed={failed:?}",
        worst.h,
        worst.down,
        worst.out,
        worst.flips,
        worst.silu_hi,
        worst.up_hi,
        worst.up_lo,
        worst.visible
    );
    assert!(
        clamped > 0,
        "no routed layer carries a SwiGLU limit above 1e-6: the clamp is not exercised"
    );
    assert!(
        failed.is_empty() && clamp_failed.is_empty(),
        "layers outside the band: {failed:?}; layers where the crossed clamp does not hold: {clamp_failed:?}"
    );
    println!(
        "PASSED: ds41_host — the host tier's routed partial sum equals ik's ffn_moe_out on every layer \
         within the derived band, flips counted; the routed clamp crossed on all three sides on {clamped} layers"
    );
}
