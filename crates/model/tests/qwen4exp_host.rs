//! qwen4exp host expert gate: what the host tier serves of the Qwen3.8 file
//! (`arch::qwen35moe::host`), on the file's own stacks.
//!
//! - The routed types by layer, as the header gives them: which layers the
//!   host serves: every stack fused by qdot at its row width, the five layers
//!   with a q8_0 down (2, 4, 30, 46, 47 — layer 2's q5_K gate and up among
//!   them) through qdot's q8_0 x q8_2_x4 kernel, so the whole run builds.
//! - Four layers — 0 and 43 (q4_K gate and up, q5_1 down; a GDN and an
//!   attention layer), 2 (q5_K gate and up, q8_0 down) and 4 (q4_K gate and
//!   up, q8_0 down) — at ten slots, one slot and none, against an f64
//!   reference on the file's dequantized rows, stage by stage from the call's
//!   own scratch: each gate and up row against its dot with the activation's
//!   q8_2_x4 round trip; each combine against `silu(g)·u` of that call's own
//!   g and u; each down row against its dot with the combine's round trip;
//!   the sum against the list-order f32 fold `o + w·d` from zero, bit for bit.
//!   Eleven slots are refused by name (the scratch's routed width is ten).
//!
//! The bands, derived before the run. A fused dot sums leaves — a float scale
//! times an exact integer sum, the K-quants' and q5_1's min term a leaf of its
//! own — into f32 lanes by FMA and ends in an eight-lane hsum, so `|dot −
//! exact| ≤ γ_n · Σ|leaves|`, with `n` one product rounding, one FMA per later
//! block of its lane and three hsum levels, over-counted here as `k/32 + 12`
//! (q5_1 and q8_0, a block of 32) and `4·(k/256) + 12` (K-quants, a
//! super-block of 256), and the dequantized f32 row within three roundings of the exact
//! values (`γ_3`, added). `Σ|leaves| ≤ Σ_i (|w_i − b_i| + |b_i|)·|a_i|`,
//! `b_i` value `i`'s min term (`−dmin·m_j` of its sub-block, q5_1's `m`),
//! read from the block headers (q8_0 has none: `b_i` = 0). The activations are the same bits on both
//! sides (qdot's q8_2_x4 encoder is `quantize_row_q8_2_x4_roundtrip`'s,
//! `gate-qdot`), and a block's min term reads the block's exact integer
//! code sum. The combine: `v_silu` within 6u and one product rounding,
//! `8u·|s·u|` plus the least normal (the value itself where it flushes to
//! zero).
//!
//! `hw_`: needs the four shards on the box. Reads 30 experts' rows of four
//! layers: seconds.

use gguf::quant::half_to_f32;
use gguf::{GgmlType, Split, dequant_row, quantize_activations};
use model::arch::qwen35moe::hparams::Hparams;
use model::arch::qwen35moe::{host, names};
use model::moe::HostScratch;
use model::r8file::R8Source;
use model::{ModelError, Tensor2};

const Q38: &str = "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

// PIN(2026-09-27): the routed stacks' types by layer, from the header dump (1,224 tensors): gate and
// up q4_K but layer 2's q5_K; down q5_1 but the q8_0 of layers 2, 4, 30, 46 and 47.
const Q8_0_DOWN: [usize; 5] = [2, 4, 30, 46, 47];
const Q5_K_GATE_UP: [usize; 1] = [2];

/// The served layers the stages run on: a GDN layer and an attention layer
/// (q5_1 downs), layer 2 (q5_K gate and up, q8_0 down) and layer 4 (a q8_0 down).
// PIN(2026-09-27): layers 2 and 4 join once qdot fuses q8_0 (q8qdot); the run's refusal clause
// becomes the whole run served.
const SERVED: [usize; 4] = [0, 43, 2, 4];

/// Ten slots over the stack's first, last and middle experts, weights of a
/// renormalized softmax's shape.
const IDS: [u32; 10] = [0, 511, 7, 300, 128, 255, 64, 401, 33, 480];
const WTS: [f32; 10] = [0.3, 0.15, 0.1, 0.1, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04];

const U: f64 = f32::EPSILON as f64 / 2.0;

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// A stack's raw bytes and its per-expert row geometry.
struct Stack<'a> {
    bytes: &'a [u8],
    ty: GgmlType,
    k: usize,
    rows: usize,
    per: usize,
}

impl<'a> Stack<'a> {
    fn of(split: &'a Split, name: &str, n_expert: usize) -> Stack<'a> {
        let (s, t) = split
            .find(name)
            .unwrap_or_else(|| panic!("{name}: not in the file"));
        let bytes = split
            .shard(s)
            .and_then(|g| g.data(t).ok())
            .unwrap_or_else(|| panic!("{name}: no data"));
        let k = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        Stack {
            bytes,
            ty: t.ty,
            k,
            rows,
            per: bytes.len() / n_expert,
        }
    }

    /// Row `r` of expert `e`: its dequantized values and each value's min
    /// term `b_i` (see the module doc).
    fn row(&self, e: usize, r: usize) -> (Vec<f32>, Vec<f64>) {
        let rb = self.per / self.rows;
        let src = &self.bytes[e * self.per + r * rb..][..rb];
        let mut w = vec![0.0f32; self.k];
        dequant_row(self.ty, src, &mut w).expect("dequant");
        let b = min_terms(self.ty, src, self.k);
        (w, b)
    }
}

/// Each value's min term from the block headers: a K-quant super-block's
/// `−dmin·m_j` per sub-block of 32, q5_1's `m` per block of 32.
fn min_terms(ty: GgmlType, src: &[u8], k: usize) -> Vec<f64> {
    let h = |o: usize| f64::from(half_to_f32(u16::from_le_bytes([src[o], src[o + 1]])));
    let mut b = vec![0.0f64; k];
    match ty {
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let sb = if ty == GgmlType::Q4_K { 144 } else { 176 };
            for s in 0..k / 256 {
                let blk = &src[s * sb..];
                let dmin = h(s * sb + 2);
                let q = &blk[4..16];
                for j in 0..8 {
                    let m = if j < 4 {
                        q[j + 4] & 63
                    } else {
                        (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
                    };
                    for v in &mut b[s * 256 + j * 32..][..32] {
                        *v = -dmin * f64::from(m);
                    }
                }
            }
        }
        GgmlType::Q5_1 => {
            for blk in 0..k / 32 {
                let m = h(blk * 24 + 2);
                for v in &mut b[blk * 32..][..32] {
                    *v = m;
                }
            }
        }
        GgmlType::Q8_0 => {}
        other => panic!("no min-term reading for {other}"),
    }
    b
}

/// The f64 dot of `w` with `a` and its band (module doc) at `n` roundings.
fn dot_band(w: &[f32], b: &[f64], a: &[f32], n: usize) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut leaves = 0.0f64;
    for ((&wi, &bi), &ai) in w.iter().zip(b).zip(a) {
        let (wi, ai) = (f64::from(wi), f64::from(ai));
        dot += wi * ai;
        leaves += ((wi - bi).abs() + bi.abs()) * ai.abs();
    }
    (dot, gamma(n + 3) * leaves)
}

/// Roundings a fused dot's leaf takes at `k` values of `ty` (module doc).
fn roundings(ty: GgmlType, k: usize) -> usize {
    match ty {
        GgmlType::Q5_1 | GgmlType::Q8_0 => k / 32 + 12,
        _ => 4 * (k / 256) + 12,
    }
}

/// Deterministic activations in `[-2, 2)`.
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_host() {
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hparams: {e}"));
    let src = R8Source::rows(&split);
    let mut failed: Vec<String> = Vec::new();
    let mut check = |what: String, ok: bool| {
        println!("{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failed.push(what);
        }
    };

    // The routed types by layer, and the whole run served, the q8_0 downs among it.
    let mut q8_down = Vec::new();
    let mut q5k_gu = Vec::new();
    for l in 0..hp.n_layer {
        let ty = |n: String| split.find(&n).map(|(_, t)| t.ty);
        let (g, u, d) = (
            ty(names::ffn_gate_exps(l)),
            ty(names::ffn_up_exps(l)),
            ty(names::ffn_down_exps(l)),
        );
        println!("layer {l}: gate {g:?} up {u:?} down {d:?}");
        if d == Some(GgmlType::Q8_0) {
            q8_down.push(l);
        }
        if g == Some(GgmlType::Q5_K) && u == Some(GgmlType::Q5_K) {
            q5k_gu.push(l);
        }
    }
    check(
        format!(
            "q8_0 downs {q8_down:?} (want {Q8_0_DOWN:?}), q5_K gate and up {q5k_gu:?} (want {Q5_K_GATE_UP:?})"
        ),
        q8_down == Q8_0_DOWN && q5k_gu == Q5_K_GATE_UP,
    );
    let whole = host::layers(src, &hp, 0..hp.n_layer);
    check(
        format!(
            "the whole run of {} layers: {}",
            hp.n_layer,
            whole.as_ref().map_or_else(ToString::to_string, |v| format!(
                "served, {} layers",
                v.len()
            ))
        ),
        whole.as_ref().is_ok_and(|v| v.len() == hp.n_layer),
    );
    drop(whole);

    let n_used = hp.n_used;
    let mut scratch = HostScratch::new(hp.n_embd, hp.expert_ff, n_used).expect("scratch");
    for (case, &l) in SERVED.iter().enumerate() {
        let layer = host::layer(src, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"));
        let gate = Stack::of(&split, &names::ffn_gate_exps(l), hp.n_expert);
        let up = Stack::of(&split, &names::ffn_up_exps(l), hp.n_expert);
        let down = Stack::of(&split, &names::ffn_down_exps(l), hp.n_expert);
        let xv = lcg(hp.n_embd, 17 + case as u64);
        let x = Tensor2::from_vec(hp.n_embd, 1, xv.clone());
        let mut xq = vec![0.0f32; hp.n_embd];
        quantize_activations(gate.ty, &xv, &mut xq).expect("x round trip");
        for slots in [n_used, 1, 0] {
            let list: Vec<(u32, f32)> = IDS.iter().copied().zip(WTS).take(slots).collect();
            let mut out = vec![f32::NAN; hp.n_embd];
            layer
                .experts_into(src, &x, &list, &mut out, &mut scratch)
                .unwrap_or_else(|e| panic!("layer {l} {slots} slots: {e}"));
            let (mut worst_gu, mut worst_h, mut worst_d) = (0.0f64, 0.0f64, 0.0f64);
            for (i, &(e, _)) in list.iter().enumerate() {
                let e = e as usize;
                for (stack, got) in [(&gate, scratch.gate(i)), (&up, scratch.up(i))] {
                    for (r, &v) in got.data.iter().enumerate().take(stack.rows) {
                        let (w, b) = stack.row(e, r);
                        let (want, band) = dot_band(&w, &b, &xq, roundings(stack.ty, stack.k));
                        let d = (f64::from(v) - want).abs() / band;
                        worst_gu = worst_gu.max(if d.is_nan() { f64::INFINITY } else { d });
                    }
                }
                let (g, u, h) = (
                    &scratch.gate(i).data,
                    &scratch.up(i).data,
                    &scratch.par(i).data,
                );
                for r in 0..hp.expert_ff {
                    let (gv, uv) = (f64::from(g[r]), f64::from(u[r]));
                    let want = gv / (1.0 + (-gv).exp()) * uv;
                    // A combine `v_silu` flushed to zero is off by the value itself.
                    let flushed = if h[r] == 0.0 { want.abs() } else { 0.0 };
                    let band = (8.0 * U * want.abs()).max(flushed) + f64::from(f32::MIN_POSITIVE);
                    let d = (f64::from(h[r]) - want).abs() / band;
                    worst_h = worst_h.max(if d.is_nan() { f64::INFINITY } else { d });
                }
                let mut hq = vec![0.0f32; hp.expert_ff];
                quantize_activations(down.ty, h, &mut hq).expect("h round trip");
                let got = &scratch.down(i).data;
                for (r, &v) in got.iter().enumerate().take(down.rows) {
                    let (w, b) = down.row(e, r);
                    let (want, band) = dot_band(&w, &b, &hq, roundings(down.ty, down.k));
                    let d = (f64::from(v) - want).abs() / band;
                    worst_d = worst_d.max(if d.is_nan() { f64::INFINITY } else { d });
                }
            }
            let mut want = vec![0.0f32; hp.n_embd];
            for (i, &(_, w)) in list.iter().enumerate() {
                for (o, &d) in want.iter_mut().zip(&scratch.down(i).data) {
                    *o += w * d;
                }
            }
            let sum_bits = out
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            check(
                format!(
                    "layer {l} ({:?}/{:?}/{:?}) {slots} slots: gate·up worst |Δ|/band {worst_gu:.3}, \
                     combine {worst_h:.3}, down {worst_d:.3}; the sum's list-order fold, {sum_bits} \
                     values differ",
                    gate.ty, up.ty, down.ty
                ),
                worst_gu <= 1.0 && worst_h <= 1.0 && worst_d <= 1.0 && sum_bits == 0,
            );
        }
        let eleven: Vec<(u32, f32)> = (0..=n_used as u32).map(|e| (e, 0.1)).collect();
        let mut out = vec![0.0f32; hp.n_embd];
        let got = layer.experts_into(src, &x, &eleven, &mut out, &mut scratch);
        let text = got
            .as_ref()
            .err()
            .map_or("served".to_string(), ToString::to_string);
        check(
            format!("layer {l}: {} slots: {text}", n_used + 1),
            matches!(got, Err(ModelError::Shape { want_ne0, got_ne0, .. })
                if want_ne0 == n_used && got_ne0 == n_used + 1),
        );
    }
    assert!(
        failed.is_empty(),
        "{} clause(s) failed:\n  {}",
        failed.len(),
        failed.join("\n  ")
    );
}
