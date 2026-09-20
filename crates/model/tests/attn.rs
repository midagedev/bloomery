//! Gate for `model::attn`: one block's MLA attention, end to end, against the
//! ik_llama.cpp oracle — blocks 0 and 1, every intermediate the round owns.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
//!
//! Each tensor is asserted in the same order the computation produces it, shape before
//! values. The oracle dumps flatten in ggml order (ne0 contiguous), which is exactly
//! `Tensor2`'s layout with trailing dims folded, so a shape failure here is a real
//! mismatch and not a convention gap.
//!
//! Tolerance policy: the five projection/rope/norm tensors gate at 1e-4. The last three
//! carry documented bounds instead — every one of them fails a uniform 1e-4 today
//! (FAIL-first measured 2026-09-19 against this exact source, manifest-free probe over
//! the same files), and the mechanism is the same in each case: `ops::matmul_q`'s f32
//! lane order differs from the reference SIMD kernels by last-ulps (~1.5e-5 on `q`),
//! and `quantize_act`'s int8 codes are a step function — an activation value sitting
//! on a rounding tie flips one code, which moves a whole q_nope2 column. Measured, the
//! flip lands in exactly one (head, token) column per block (col 32 = head 5 token 2;
//! col 57 = head 9 token 3). [2026-09-20, MUL-21: after the Q3_K fused wiring the
//! seats moved — block 0 flips cols 32 and 62, block 1 flips none; two flips total,
//! same as before, redistributed. The `max_cols` pin followed to 2; see the call site.]
//! The attention arithmetic itself is a transcription: fed the
//! oracle's own q_rope/q_nope2/kvr, the scalar path reproduces `kqv_compressed`
//! with max |diff| = 0 (`hw_attn_exact_input_stages` asserts that at tolerance
//! zero, on the `flash_attn_latent_scalar` leg). Since MUL-36 the default
//! dispatch runs the AVX2+FMA+F16C twin, whose kq dot sums in 8-lane groups
//! instead of the fa4 two-partial chain — ULP-scale off ik's bits, banded
//! against the scalar path by `hw_flash_simd_bands_against_scalar` and by the
//! dispatch leg of the exact-input test. The bounds cover two simultaneous tie flips, so an upstream `matmul_q` change that
//! moves a tie re-reds the gate instead of silently widening it.
#[path = "common/oracle.rs"]
mod oracle;

use model::attn::block_attn_trace;
use model::{Slot, Tensor2};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// Oracle `ne[]` (ggml order) folded to the `Tensor2` pair: rows = ne0, columns = the
/// product of the rest. q_rope {64,16,6}, q_nope2 {512,6,16} and kqv_compressed
/// {512,16,6} all land here.
fn folded(ne: &[i64; 4]) -> [i64; 2] {
    [ne[0], ne[1] * ne[2] * ne[3]]
}

fn run_block(o: &oracle::Oracle, g: &gguf::Gguf, n: usize) -> model::attn::AttnTrace {
    let (xs, xinf) = o.load(&format!("attn_norm-{n}"), 0);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);
    // The dump is one prefill batch: tokens at positions 0..5, sequence 0.
    let slots: Vec<Slot> = (0..x.ne1 as u32).map(|t| Slot { seq: 0, pos: t }).collect();
    let derived = model::derived::Derived::new(g).unwrap();
    block_attn_trace(g, n, &x, &slots, &derived).expect("block_attn_trace")
}

fn check(o: &oracle::Oracle, got: &Tensor2, name: &str, occ: u32, what: &str, tol: f32) {
    let (want, winf) = o.load(name, occ);
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        folded(&winf.ne),
        "{what}: shape must match the reference before the values can mean anything"
    );
    oracle::assert_close(&got.data, &want, tol, what);
}

/// The same check, plus the shape of the deviation — for a tensor whose bound exists only
/// to admit a tie flip. Lead-added 2026-09-19 on adoption.
///
/// A blanket `3e-2` on `q_nope2` says "anything in these 49152 values may be 0.5 % wrong",
/// which is not what the round measured and not what we want to allow. What it measured is
/// that the flip is confined to ONE (head, token) column per block — 373 and 377 elements,
/// all sharing a column index. So the gate states that instead: every element is within
/// `strict`, EXCEPT elements lying in at most `max_cols` columns, which get `loose`.
///
/// `max_cols` was 1 at adoption (2026-09-19) on the argument below — that a SECOND
/// flipped column is the "upstream `matmul_q` change" event and must re-red the gate.
/// That event then happened (MUL-21, 2026-09-20: the Q3_K fused wiring re-seated the
/// ties; two columns in block 0, none in block 1, flip total unchanged) — the gate
/// reded, it was looked at with a both-ways control, and the pin moved to 2 WITH the
/// same discipline: a third column re-reds. The argument is not retired; it has fired
/// once and was adjudicated in the call site's dated note.
///
/// This is a tightening, so no re-authoring evidence is owed — but it was measured both
/// ways anyway (2026-09-19, box): it passes on this source at 1 column per block, and
/// adding 5e-4 to a single element in column 7 reds it with
/// `2 columns deviate past 1e-4 ... Columns: [7, 32]`, while the blanket 3e-2 it replaces
/// stayed green through that same perturbation.
fn check_confined(
    o: &oracle::Oracle,
    got: &Tensor2,
    name: &str,
    occ: u32,
    what: &str,
    strict: f32,
    loose: f32,
    max_cols: usize,
) {
    let (want, winf) = o.load(name, occ);
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        folded(&winf.ne),
        "{what}: shape must match the reference before the values can mean anything"
    );
    assert_eq!(got.data.len(), want.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut over: std::collections::BTreeSet<usize> = Default::default();
    let mut n_over = 0usize;
    for (i, (&g, &w)) in got.data.iter().zip(&want).enumerate() {
        let d = (g - w).abs();
        worst = worst.max(d);
        assert!(
            d <= loose,
            "{what}: {d:e} at index {i} (col {}) exceeds even the tie-flip bound {loose:e}",
            i / got.ne0
        );
        if d > strict {
            over.insert(i / got.ne0);
            n_over += 1;
        }
    }
    assert!(
        over.len() <= max_cols,
        "{what}: {} columns deviate past {strict:e} ({n_over} elements) — a tie flip moves \
         ONE column, so this is a different fault. Columns: {:?}",
        over.len(),
        over.iter().take(8).collect::<Vec<_>>()
    );
    eprintln!(
        "{what:38} max|diff| = {worst:e}   ok (≤{strict:e} outside {} column(s) {:?}, ≤{loose:e} inside)",
        over.len(),
        over.iter().collect::<Vec<_>>()
    );
}

fn check_block(o: &oracle::Oracle, n: usize) {
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tr = run_block(o, &g, n);

    // The eight contract tensors, in computation order. Occurrences verified against
    // the manifest: q, kv_rope_compressed, kvr, q_nope2, kqv_compressed, kqv_out are
    // occurrence 0; q_rope and kv_compressed are occurrence 1 (occurrence 0 is the
    // pre-ROPE / pre-norm VIEW).
    //
    // Measured max |diff| (2026-09-19): blk0 / blk1.
    check(o, &tr.q, &format!("q-{n}"), 0, &format!("q-{n}"), 1e-4); // 1.5e-5 / 1.5e-5
    check(
        o,
        &tr.kv_rope_compressed,
        &format!("kv_rope_compressed-{n}"),
        0,
        &format!("kv_rope_compressed-{n}"),
        1e-4,
    ); // 1.9e-5 / 5.1e-5
    check(
        o,
        &tr.q_rope,
        &format!("q_rope-{n}"),
        1,
        &format!("q_rope-{n} (rope out)"),
        1e-4,
    ); // 1.5e-5 / 1.5e-5
    check(
        o,
        &tr.kv_compressed,
        &format!("kv_compressed-{n}"),
        1,
        &format!("kv_compressed-{n} (normed latent)"),
        1e-4,
    ); // 1.4e-6 / 1.1e-6
    check(
        o,
        &tr.kvr,
        &format!("kvr-{n}"),
        0,
        &format!("kvr-{n}"),
        1e-4,
    ); // 8.6e-6 / 5.1e-5
    // One flipped activation code moves one whole column; bound covers two flips.
    // Measured 2.5e-3 / 1.1e-2 (373 and 377 of 49152 elements above 1e-4, one column
    // each); exact-input companion is 3.8e-6 / 9.5e-7.
    //
    // Re-pinned 1 -> 2 columns by the LEAD on 2026-09-20, on the MUL-21 fused wiring.
    // This is the event the max_cols=1 pin was built to catch, and it was caught and
    // looked at: routing Q3_K through `crates/qdot` moved every Q3_K logit, which
    // re-seated `quantize_act`'s ties. Measured on the box, same tree, both ways:
    // fused -> block 0 deviates in cols [32, 62] and block 1 in none; scalar restore
    // -> block 0 [32], block 1 [57], green. The flip TOTAL across blocks is unchanged
    // at two — the upstream moved them between blocks, it did not add a fault class,
    // and the exact-input companion still passes (bit-exact stage B). A THIRD column
    // reds the gate again; that discipline moves with the pin.
    check_confined(
        o,
        &tr.q_nope2,
        &format!("q_nope2-{n}"),
        0,
        &format!("q_nope2-{n} (absorbed)"),
        1e-4,
        3e-2,
        2,
    );
    // Softmax damps the same tie-flip spikes ~100x. Measured 1.2e-4 / 8.1e-5; the
    // exact-input companion is bit-exact (0), so nothing here is attention arithmetic.
    check(
        o,
        &tr.kqv_compressed,
        &format!("kqv_compressed-{n}"),
        0,
        &format!("kqv_compressed-{n} (attention out)"),
        5e-4,
    );
    // wv_b (Q3_K) and the output projection amplify the kqv_compressed residual.
    // Measured 8.9e-4 / 2.7e-4; exact-input companion 1.4e-5 / 3.3e-7.
    check(
        o,
        &tr.kqv_out,
        &format!("kqv_out-{n}"),
        0,
        &format!("kqv_out-{n}"),
        5e-3,
    );
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_block0_matches_ggml() {
    let o = oracle::Oracle::open();
    check_block(&o, 0);
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_block1_matches_ggml() {
    let o = oracle::Oracle::open();
    check_block(&o, 1);
}

/// Exact-input companions: feed the oracle's own tensors into each stage and assert the
/// stage's own arithmetic at its noise floor. This is what makes the documented bounds
/// above honest — they are inherited from upstream `matmul_q` slack, not from anything
/// in this crate's attention path.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_exact_input_stages() {
    let o = oracle::Oracle::open();
    let load2 = |name: &str, occ: u32| -> Tensor2 {
        let (v, inf) = o.load(name, occ);
        Tensor2::from_vec(
            inf.ne[0] as usize,
            (inf.ne[1] * inf.ne[2] * inf.ne[3]) as usize,
            v,
        )
    };
    let maxdiff = |a: &Tensor2, b: &Tensor2| {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    for blk in [0usize, 1usize] {
        let g = gguf::Gguf::open(model_path()).unwrap();
        let p = model::attn::MlaParams::read(&g, blk).unwrap();
        let wkb = g.find(&format!("blk.{blk}.attn_kv_b.weight")).unwrap();
        let n_tok = 6usize;
        let slots: Vec<Slot> = (0..n_tok as u32).map(|t| Slot { seq: 0, pos: t }).collect();

        // A: the absorption (Q8_0 requant + block_q8_2 activation + f64 block dot).
        // Measured 3.8e-6 / 9.5e-7 — quantizer and dot conventions, no order slack
        // beyond the f64-accumulated block sum. The Q8_0 weights come from `Derived`
        // now (byte-identical to the in-place build by `tests/derived.rs`).
        let q_exact = load2(&format!("q-{blk}"), 0);
        let derived = model::derived::Derived::new(&g).unwrap();
        let out = model::attn::q_nope2_absorbed(derived.wk_b_all_heads(blk).unwrap(), &q_exact, &p)
            .unwrap();
        let want = load2(&format!("q_nope2-{blk}"), 0);
        assert_eq!([out.ne0, out.ne1], [want.ne0, want.ne1], "stage A shape");
        assert!(
            maxdiff(&out, &want) <= 1e-5,
            "stage A: absorption arithmetic"
        );

        // B: the attention proper — two legs since MUL-36 (2026-09-20).
        //
        // The scalar leg (`flash_attn_latent_scalar`, the no-AVX2 fallback)
        // keeps tolerance ZERO, the property this gate has always proven:
        // fed the kernel's exact inputs, the scalar transcription reproduces
        // the kernel's exact outputs bit for bit. It is also the A/B control
        // for the dispatch leg in the same run — the wiring around the
        // kernel changed nothing else.
        //
        // The dispatch leg (AVX2+FMA+F16C on this box) re-pins the former
        // 0.0. Its kq dot sums in 8-lane groups (four FMA partials + a
        // fixed lane tree, `kq_dot_fa4_avx2`) instead of the fa4 two-partial
        // chain, so the outputs leave ik's bits by ULP-scale amounts. The
        // band is derived, not measured-in-advance: this test computes δ =
        // max |kq_simd − kq_scalar| over the oracle's own (row, key) pairs,
        // then |Δout| ≤ 2·kq_scale·δ·n_keys·spread(V) — each key's weight
        // moves by at most its s moved (kq_scale·δ, and the same again for
        // the max scan's m, the factor 2), the softmax output is a convex
        // combination with S ≥ 1 (the argmax key's weight is exactly
        // `v_expf(0)` = 1), and a convex combination moves by at most
        // max|Δw| times the spread of the values it mixes. Attribution: the
        // V stage cannot contribute — its j-axis lanes preserve every output
        // element's key-order FMA chain, bit-identical by construction — so
        // the whole leg-B difference is the kq sum order. Measured 2026-09-20
        // on this tree (both blocks, joint worst): δ-band 2.13e-4, realized
        // max|diff| 1.31e-6 — ~160x headroom, and the scalar leg at 0 in the
        // same run is the wiring control.
        let q_rope_exact = load2(&format!("q_rope-{blk}"), 1);
        let q_nope2_exact = load2(&format!("q_nope2-{blk}"), 0);
        let kvr_exact = load2(&format!("kvr-{blk}"), 0);
        let cache16: Vec<Vec<u16>> = (0..n_tok)
            .map(|t| {
                kvr_exact
                    .col(t)
                    .iter()
                    .map(|&v| model::attn::f32_to_f16_bits(v))
                    .collect()
            })
            .collect();
        let want = load2(&format!("kqv_compressed-{blk}"), 0);

        let kqv_scalar = model::attn::flash_attn_latent_scalar(
            &q_rope_exact,
            &q_nope2_exact,
            &cache16,
            &slots,
            &slots,
            &p,
        );
        assert_eq!(
            [kqv_scalar.ne0, kqv_scalar.ne1],
            [want.ne0, want.ne1],
            "stage B shape"
        );
        oracle::assert_close(
            &kqv_scalar.data,
            &want.data,
            0.0,
            "stage B scalar: attention bit-exact",
        );

        let kqv = model::attn::flash_attn_latent(
            &q_rope_exact,
            &q_nope2_exact,
            &cache16,
            &slots,
            &slots,
            &p,
        );
        assert_eq!([kqv.ne0, kqv.ne1], [want.ne0, want.ne1], "stage B shape");
        // δ over the oracle's own (row, key) pairs, and the V spread of the
        // same rows — both from the data the legs just consumed. The FA row
        // is rope + ABSORBED nope (576), not kq_head (192): q_nope2 already
        // carries the wk_b absorption.
        let mut delta = 0.0f32;
        for t in 0..n_tok {
            for h in 0..p.n_head {
                let mut qrow = vec![0.0f32; p.rope_dims + p.latent];
                qrow[..p.rope_dims].copy_from_slice(q_rope_exact.col(t * p.n_head + h));
                qrow[p.rope_dims..].copy_from_slice(q_nope2_exact.col(h * n_tok + t));
                for k in &cache16 {
                    delta = delta.max(
                        (model::attn::kq_dot_simd(&qrow, k) - model::attn::kq_dot_fa4(&qrow, k))
                            .abs(),
                    );
                }
            }
        }
        let mut v_lo = f32::INFINITY;
        let mut v_hi = f32::NEG_INFINITY;
        for row in &cache16 {
            for &b in &row[p.rope_dims..] {
                let v = gguf::quant::half_to_f32(b);
                v_lo = v_lo.min(v);
                v_hi = v_hi.max(v);
            }
        }
        let band = 2.0 * p.kq_scale * delta * n_tok as f32 * (v_hi - v_lo);
        oracle::assert_close(&kqv.data, &want.data, band, "stage B simd: banded");

        // C: wv_b (Q3_K view) + output projection — generic `matmul_q` order slack.
        // Measured 1.4e-5 / 3.3e-7.
        let kqv_exact = load2(&format!("kqv_compressed-{blk}"), 0);
        let kqv_2d = model::attn::wv_b_heads(&g, wkb, &kqv_exact, &p).unwrap();
        let wo = g.find(&format!("blk.{blk}.attn_output.weight")).unwrap();
        let out = model::ops::matmul_q(&g, wo, &kqv_2d).unwrap();
        let want = load2(&format!("kqv_out-{blk}"), 0);
        assert_eq!([out.ne0, out.ne1], [want.ne0, want.ne1], "stage C shape");
        assert!(
            maxdiff(&out, &want) <= 1e-4,
            "stage C: wv_b + output projection"
        );
    }
}

/// Deterministic uniform noise for the synthetic band test — no model file,
/// no oracle, only the box's ISA.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// MUL-36: the AVX2 flash kernel against its scalar twin on synthetic data —
/// the scalar path IS the oracle here (same values, ik's own sum order), so
/// this needs no model file and no dump, only AVX2+FMA+F16C.
///
/// The kq band is the textbook reassociation bound: both sum orders round at
/// most `d_head` times, each rounding at most one unit-in-the-last-place
/// (2⁻²⁴) of a running partial no larger than Σ|q·k|. The flash band then
/// propagates the REALIZED dot deviation δ (measured by the kq leg in this
/// same run, not the loose bound): each key's weight moves by at most its s
/// moved (kq_scale·δ, and the same again for the max scan's m — the factor
/// 2), the softmax output is a convex combination with S ≥ 1 (the argmax
/// key's weight is exactly `v_expf(0)` = 1), so the output moves by at most
/// 2·kq_scale·δ·n_keys·spread(V). Prints the realized worst against both so
/// a run says how much headroom each leaves. Measured 2026-09-20 on this
/// tree: kq realized 1.30e-4 against the bound 1.34e-1 (~1000x — the
/// textbook bound assumes every rounding aligns, which random data never
/// does), flash realized 1.17e-5 against its band 1.62e-2.
#[test]
#[ignore = "hw: needs AVX2+FMA+F16C (the box); synthetic data, no model file"]
fn hw_flash_simd_bands_against_scalar() {
    // The real MLA's shapes (576-wide row, 512 latent) at a small head/token
    // count. The key count (100) crosses three whole 32-key blocks plus a
    // partial fourth (the padding break), and the first query's position
    // admits 50 keys — past the block boundary, into the M-bump rescale.
    let p = model::attn::MlaParams {
        n_head: 3,
        kq_head: 576,
        nope: 512,
        rope_dims: 64,
        v_head: 128,
        latent: 512,
        eps: 1e-6,
        // The rope fields are dead here — flash reads only the dims.
        rope: model::attn::RopeParams {
            n_dims: 64,
            freq_base: 1e4,
            freq_scale: 0.025,
            ext_factor: 1.0,
            mscale_param: 0.7,
            corr_dims: [10.0, 23.0],
            theta_scale: 0.8,
        },
        kq_scale: 0.078,
    };
    let mut rng = Lcg(0x5eed_1234_5678_9abc);
    let n_tok = 5usize;
    let n_keys = 100usize;

    let mut q_rope = Tensor2::zeros(p.rope_dims, p.n_head * n_tok);
    let mut q_nope2 = Tensor2::zeros(p.latent, p.n_head * n_tok);
    for t in 0..n_tok {
        for h in 0..p.n_head {
            for v in q_rope.col_mut(t * p.n_head + h).iter_mut() {
                *v = rng.range(-3.0, 3.0);
            }
            for v in q_nope2.col_mut(h * n_tok + t).iter_mut() {
                *v = rng.range(-3.0, 3.0);
            }
        }
    }
    let mut keys16: Vec<Vec<u16>> = Vec::with_capacity(n_keys);
    let mut v_lo = f32::INFINITY;
    let mut v_hi = f32::NEG_INFINITY;
    for _ in 0..n_keys {
        let row: Vec<u16> = (0..p.rope_dims + p.latent)
            .map(|_| model::attn::f32_to_f16_bits(rng.range(-4.0, 4.0)))
            .collect();
        for &b in &row[p.rope_dims..] {
            let v = gguf::quant::half_to_f32(b);
            v_lo = v_lo.min(v);
            v_hi = v_hi.max(v);
        }
        keys16.push(row);
    }
    // Sequences and positions chosen so every mask path fires: half the keys
    // are another sequence, positions wrap past the queries (future
    // masking), and every query keeps at least one allowed key.
    let key_slots: Vec<Slot> = (0..n_keys)
        .map(|u| Slot {
            seq: if u < 50 { 0 } else { 1 },
            pos: (u % 25) as u32,
        })
        .collect();
    let q_slots: Vec<Slot> = [24u32, 0, 3, 5, 24]
        .iter()
        .map(|&pos| Slot { seq: 0, pos })
        .collect();

    // Leg 1 — the kq dot itself, against the reassociation bound.
    let mut worst_kq = 0.0f32;
    let mut worst_band = 0.0f32;
    for t in 0..n_tok {
        for h in 0..p.n_head {
            let mut qrow = vec![0.0f32; p.rope_dims + p.latent];
            qrow[..p.rope_dims].copy_from_slice(q_rope.col(t * p.n_head + h));
            qrow[p.rope_dims..].copy_from_slice(q_nope2.col(h * n_tok + t));
            for (u, k) in keys16.iter().enumerate() {
                let d =
                    (model::attn::kq_dot_simd(&qrow, k) - model::attn::kq_dot_fa4(&qrow, k)).abs();
                let band = kq_reassoc_band(&qrow, k);
                worst_kq = worst_kq.max(d);
                worst_band = worst_band.max(band);
                assert!(
                    d <= band,
                    "kq dot: |simd - scalar| = {d:e} exceeds the reassociation band \
                     {band:e} (t {t}, h {h}, key {u})"
                );
            }
        }
    }
    eprintln!(
        "kq dot, simd vs scalar ({} pairs)   max|diff| = {worst_kq:e}   band {worst_band:e}",
        n_tok * p.n_head * n_keys
    );

    // Leg 2 — the whole kernel, propagating the realized δ of leg 1.
    let a = model::attn::flash_attn_latent(&q_rope, &q_nope2, &keys16, &key_slots, &q_slots, &p);
    let b =
        model::attn::flash_attn_latent_scalar(&q_rope, &q_nope2, &keys16, &key_slots, &q_slots, &p);
    let flash_band = 2.0 * p.kq_scale * worst_kq * n_keys as f32 * (v_hi - v_lo);
    let mut worst_flash = 0.0f32;
    for (i, (&x, &y)) in a.data.iter().zip(&b.data).enumerate() {
        let d = (x - y).abs();
        worst_flash = worst_flash.max(d);
        assert!(
            d <= flash_band,
            "flash: |simd - scalar| = {d:e} at index {i} exceeds the band {flash_band:e}"
        );
    }
    eprintln!(
        "flash, simd vs scalar ({} values)  max|diff| = {worst_flash:e}   band {flash_band:e}",
        a.data.len()
    );
}

/// The reassociation band for one (q row, f16 key row) pair: both sum orders
/// round at most `d_head` times, each rounding at most one ulp (2⁻²⁴) of a
/// running partial no larger than Σ|q·k|.
fn kq_reassoc_band(q: &[f32], k: &[u16]) -> f32 {
    let sum_abs: f32 = q
        .iter()
        .zip(k)
        .map(|(&qi, &ki)| qi.abs() * gguf::quant::half_to_f32(ki).abs())
        .sum();
    2.0 * q.len() as f32 * 2.0f32.powi(-24) * sum_abs
}
