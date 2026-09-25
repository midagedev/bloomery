//! Gate for `model::arch::deepseek2::attn`: one block's MLA attention, end to end, against the
//! ik_llama.cpp oracle — blocks 0 and 1, every intermediate the trace holds.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
//!
//! Each tensor is asserted in the same order the computation produces it, shape before
//! values. The oracle dumps flatten in ggml order (ne0 contiguous), which is exactly
//! `Tensor2`'s layout with trailing dims folded, so a shape failure here is a real
//! mismatch and not a convention gap.
//!
//! Tolerance policy: the five projection/rope/norm tensors gate at 1e-4. The last
//! three carry documented bounds instead — every one of them fails a uniform 1e-4,
//! and the mechanism is the same in each case: `ops::matmul_q`'s f32 lane order
//! differs from the reference SIMD kernels by last-ulps, and `quantize_act`'s int8
//! codes are a step function — an activation value sitting on a rounding tie flips
//! one code, which moves a whole q_nope2 column. The bounds cover two simultaneous
//! tie flips, so an upstream `matmul_q` change that moves a tie re-reds the gate
//! instead of silently widening it. The attention arithmetic itself is a
//! transcription: fed the oracle's own q_rope/q_nope2/kvr, the scalar path
//! reproduces `kqv_compressed` bit-exactly (`hw_attn_exact_input_stages`, tolerance
//! zero, on the `flash_attn_latent_scalar` leg). The default dispatch runs the
//! AVX2+FMA+F16C twin, whose kq dot sums in 8-lane groups instead of the fa4
//! two-partial chain — ULP-scale off ik's bits, banded against the scalar path by
//! `hw_flash_simd_bands_against_scalar` and by the dispatch leg of the exact-input
//! test.
#[path = "common/asserts.rs"]
mod asserts;
#[path = "common/manifest.rs"]
mod manifest;
#[path = "common/model_path.rs"]
mod model_path;
#[path = "common/oracle.rs"]
mod oracle;

use model::arch::deepseek2::attn::{self, block_attn_trace};
use model::arch::deepseek2::derived::Derived;
use model::{Slot, Tensor2};

/// Oracle `ne[]` (ggml order) folded to the `Tensor2` pair: rows = ne0, columns = the
/// product of the rest. q_rope {64,16,6}, q_nope2 {512,6,16} and kqv_compressed
/// {512,16,6} all land here.
fn folded(ne: &[i64; 4]) -> [i64; 2] {
    [ne[0], ne[1] * ne[2] * ne[3]]
}

fn run_block(o: &oracle::Oracle, g: &gguf::Gguf, n: usize) -> attn::AttnTrace {
    let (xs, xinf) = o.load(&format!("attn_norm-{n}"), 0);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);
    // The dump is one prefill batch: tokens at positions 0..5, sequence 0.
    let slots: Vec<Slot> = (0..x.ne1 as u32).map(|t| Slot { seq: 0, pos: t }).collect();
    let derived = Derived::new(g).unwrap();
    block_attn_trace(g, n, &x, &slots, &derived).expect("block_attn_trace")
}

fn check(o: &oracle::Oracle, got: &Tensor2, name: &str, occ: u32, what: &str, tol: f32) {
    let (want, winf) = o.load(name, occ);
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        folded(&winf.ne),
        "{what}: shape must match the reference before the values can mean anything"
    );
    asserts::assert_close(&got.data, &want, tol, what);
}

/// The bound [`check_confined`] enforces, one field per term of its contract.
struct Confinement {
    strict: f32,
    loose: f32,
    max_cols: usize,
}

/// The same check, plus the shape of the deviation — for a tensor whose bound
/// exists only to admit a tie flip.
///
/// A blanket `3e-2` on `q_nope2` says "anything in these 49152 values may be 0.5 %
/// wrong", which is not what we want to allow. The gate states the confinement
/// instead: every element is within `strict`, EXCEPT elements lying in at most
/// `max_cols` columns, which get `loose`. A tie flip moves ONE column; a SECOND
/// deviating column is an upstream `matmul_q` change event and must re-red the gate.
fn check_confined(
    o: &oracle::Oracle,
    got: &Tensor2,
    name: &str,
    occ: u32,
    what: &str,
    bound: Confinement,
) {
    let Confinement {
        strict,
        loose,
        max_cols,
    } = bound;
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
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let tr = run_block(o, &g, n);

    // The eight contract tensors, in computation order. Occurrences verified against
    // the manifest: q, kv_rope_compressed, kvr, q_nope2, kqv_compressed, kqv_out are
    // occurrence 0; q_rope and kv_compressed are occurrence 1 (occurrence 0 is the
    // pre-ROPE / pre-norm VIEW).
    check(o, &tr.q, &format!("q-{n}"), 0, &format!("q-{n}"), 1e-4);
    check(
        o,
        &tr.kv_rope_compressed,
        &format!("kv_rope_compressed-{n}"),
        0,
        &format!("kv_rope_compressed-{n}"),
        1e-4,
    );
    check(
        o,
        &tr.q_rope,
        &format!("q_rope-{n}"),
        1,
        &format!("q_rope-{n} (rope out)"),
        1e-4,
    );
    check(
        o,
        &tr.kv_compressed,
        &format!("kv_compressed-{n}"),
        1,
        &format!("kv_compressed-{n} (normed latent)"),
        1e-4,
    );
    check(
        o,
        &tr.kvr,
        &format!("kvr-{n}"),
        0,
        &format!("kvr-{n}"),
        1e-4,
    );
    // One flipped activation code moves one whole column; the bound covers two
    // simultaneous flips (see `check_confined`).
    // PIN(2026-09-20): max_cols 1 -> 2 — the Q3_K fused wiring re-seated
    // `quantize_act`'s ties (two columns in block 0, none in block 1; flip total
    // across blocks unchanged, exact-input companion still bit-exact). A THIRD
    // column reds the gate.
    check_confined(
        o,
        &tr.q_nope2,
        &format!("q_nope2-{n}"),
        0,
        &format!("q_nope2-{n} (absorbed)"),
        Confinement {
            strict: 1e-4,
            loose: 3e-2,
            max_cols: 2,
        },
    );
    // Softmax damps the same tie-flip spikes; the exact-input companion is
    // bit-exact, so nothing here is attention arithmetic.
    check(
        o,
        &tr.kqv_compressed,
        &format!("kqv_compressed-{n}"),
        0,
        &format!("kqv_compressed-{n} (attention out)"),
        5e-4,
    );
    // wv_b (Q3_K) and the output projection amplify the kqv_compressed residual.
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
        let g = gguf::Gguf::open(model_path::model_path()).unwrap();
        let p = attn::MlaParams::read(&g, blk).unwrap();
        let wkb = g.find(&format!("blk.{blk}.attn_kv_b.weight")).unwrap();
        let n_tok = 6usize;
        let slots: Vec<Slot> = (0..n_tok as u32).map(|t| Slot { seq: 0, pos: t }).collect();

        // A: the absorption (Q8_0 requant + block_q8_2 activation + f64 block dot)
        // — quantizer and dot conventions, no order slack beyond the f64-accumulated
        // block sum. The Q8_0 weights come from `Derived` (byte-identical to the
        // in-place build by `tests/derived.rs`).
        let q_exact = load2(&format!("q-{blk}"), 0);
        let derived = Derived::new(&g).unwrap();
        let out =
            attn::q_nope2_absorbed(derived.wk_b_all_heads(blk).unwrap(), &q_exact, &p).unwrap();
        let want = load2(&format!("q_nope2-{blk}"), 0);
        assert_eq!([out.ne0, out.ne1], [want.ne0, want.ne1], "stage A shape");
        assert!(
            maxdiff(&out, &want) <= 1e-5,
            "stage A: absorption arithmetic"
        );

        // B: the attention proper — two legs.
        //
        // The scalar leg (`flash_attn_latent_scalar`, the no-AVX2 fallback)
        // keeps tolerance ZERO: fed the kernel's exact inputs, the scalar
        // transcription reproduces the kernel's exact outputs bit for bit,
        // and doubles as the wiring control for the dispatch leg in the same
        // run.
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
        // max|Δw| times the spread of the values it mixes. The V stage
        // cannot contribute — its j-axis lanes preserve every output
        // element's key-order FMA chain, bit-identical by construction — so
        // the whole leg-B difference is the kq sum order.
        let q_rope_exact = load2(&format!("q_rope-{blk}"), 1);
        let q_nope2_exact = load2(&format!("q_nope2-{blk}"), 0);
        let kvr_exact = load2(&format!("kvr-{blk}"), 0);
        // One flat row-major buffer, the cache's own layout, columns in token
        // order — the same f16 bits the per-row fixture held.
        let d_head = p.rope_dims + p.latent;
        let mut cache16: Vec<u16> = Vec::with_capacity(n_tok * d_head);
        for t in 0..n_tok {
            cache16.extend(
                kvr_exact
                    .col(t)
                    .iter()
                    .map(|&v| gguf::quant::f32_to_f16_bits(v)),
            );
        }
        let cache = model::kv::KvRows::new(&cache16, d_head);
        let want = load2(&format!("kqv_compressed-{blk}"), 0);

        let kqv_scalar = attn::flash_attn_latent_scalar(
            &q_rope_exact,
            &q_nope2_exact,
            cache,
            &slots,
            &slots,
            &p,
        );
        assert_eq!(
            [kqv_scalar.ne0, kqv_scalar.ne1],
            [want.ne0, want.ne1],
            "stage B shape"
        );
        asserts::assert_close(
            &kqv_scalar.data,
            &want.data,
            0.0,
            "stage B scalar: attention bit-exact",
        );

        let kqv = attn::flash_attn_latent(&q_rope_exact, &q_nope2_exact, cache, &slots, &slots, &p);
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
                for u in 0..n_tok {
                    let k = cache.row(u);
                    delta =
                        delta.max((attn::kq_dot_simd(&qrow, k) - attn::kq_dot_fa4(&qrow, k)).abs());
                }
            }
        }
        let mut v_lo = f32::INFINITY;
        let mut v_hi = f32::NEG_INFINITY;
        for u in 0..n_tok {
            for &b in &cache.row(u)[p.rope_dims..] {
                let v = gguf::quant::half_to_f32(b);
                v_lo = v_lo.min(v);
                v_hi = v_hi.max(v);
            }
        }
        let band = 2.0 * p.kq_scale * delta * n_tok as f32 * (v_hi - v_lo);
        asserts::assert_close(&kqv.data, &want.data, band, "stage B simd: banded");

        // C: wv_b (Q3_K view) + output projection — generic `matmul_q` order slack.
        let kqv_exact = load2(&format!("kqv_compressed-{blk}"), 0);
        let kqv_2d = attn::wv_b_heads(&g, wkb, &kqv_exact, &p).unwrap();
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

/// The fused (token, head) attention dispatch must produce the three tensors
/// of the three-call chain BIT-identically — `to_bits` on every element, no
/// tolerance: the fuse moves calls between threads, it must not move a bit.
/// The oracle's own `q`/`q_rope`/`kvr` feed both sides, so any difference is
/// the fused wiring's, not upstream slack. `n_tokens` 1 is the decode shape,
/// 6 the prefill shape (a chunk straddling more than one (t, h) row).
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_heads_fused_bit_identical() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let blk = 0usize;
    let p = attn::MlaParams::read(&g, blk).unwrap();
    let derived = Derived::new(&g).unwrap();
    let views = &derived.attn_plan(blk).unwrap().v_up_views;
    let wblocks = derived.wk_b_all_heads(blk).unwrap();

    let load2 = |name: &str, occ: u32| -> Tensor2 {
        let (v, inf) = o.load(name, occ);
        Tensor2::from_vec(
            inf.ne[0] as usize,
            (inf.ne[1] * inf.ne[2] * inf.ne[3]) as usize,
            v,
        )
    };
    let q_full = load2(&format!("q-{blk}"), 0);
    let q_rope_full = load2(&format!("q_rope-{blk}"), 1);
    let kvr_full = load2(&format!("kvr-{blk}"), 0);

    for n_tok in [1usize, 6] {
        // The batch's own tokens are the KV entries, the scratch-cache shape.
        let slots: Vec<Slot> = (0..n_tok as u32).map(|t| Slot { seq: 0, pos: t }).collect();
        // One flat row-major buffer, the cache's own layout, columns in token
        // order — the same f16 bits the per-row fixture held.
        let d_head = p.rope_dims + p.latent;
        let mut cache16: Vec<u16> = Vec::with_capacity(n_tok * d_head);
        for t in 0..n_tok {
            cache16.extend(
                kvr_full
                    .col(t)
                    .iter()
                    .map(|&v| gguf::quant::f32_to_f16_bits(v)),
            );
        }
        let cache = model::kv::KvRows::new(&cache16, d_head);
        let first_cols = |src: &Tensor2, n: usize| {
            Tensor2::from_vec(src.ne0, n, src.data[..src.ne0 * n].to_vec())
        };
        let q = first_cols(&q_full, n_tok);
        let q_rope = first_cols(&q_rope_full, p.n_head * n_tok);

        // The chain the fused dispatch replaces, stage by stage.
        let q_nope2 = attn::q_nope2_absorbed(wblocks, &q, &p).unwrap();
        let kqv_compressed = attn::flash_attn_latent(&q_rope, &q_nope2, cache, &slots, &slots, &p);
        let kqv_2d = attn::wv_b_heads_with(&g, views, &kqv_compressed, &p).unwrap();

        // The fused dispatch, same inputs.
        let (fq, fc, f2) =
            attn::attn_heads_fused(&g, wblocks, &q, &q_rope, cache, &slots, &slots, views, &p)
                .unwrap();

        let cmp = |name: &str, chain: &Tensor2, fused: &Tensor2| {
            assert_eq!(
                (fused.ne0, fused.ne1),
                (chain.ne0, chain.ne1),
                "{name} shape, n_tok {n_tok}"
            );
            let at = chain
                .data
                .iter()
                .zip(&fused.data)
                .position(|(x, y)| x.to_bits() != y.to_bits());
            assert!(
                at.is_none(),
                "{name} (n_tok {n_tok}): fused differs from the chain at element \
                 {at:?} of {} — the fuse moved a bit",
                chain.data.len()
            );
        };
        cmp("q_nope2", &q_nope2, &fq);
        cmp("kqv_compressed", &kqv_compressed, &fc);
        cmp("kqv_2d", &kqv_2d, &f2);
        eprintln!(
            "attn_heads_fused, n_tok {n_tok}:        q_nope2, kqv_compressed, kqv_2d \
             bit-identical to the chain ({} + {} + {} values)",
            fq.data.len(),
            fc.data.len(),
            f2.data.len()
        );
    }
}

/// The per-row latent split of the fused dispatch (`attn_heads_split`) must
/// produce the chain's three tensors BIT-identically at both splits — one
/// participant per (token, head) row and two, each accumulating half the
/// latent — at the decode shape against a cache 6, 1024 and 4096 keys deep
/// (plus the decode token's own row), and at the 6-token prefill shape.
/// `to_bits` on every element: the split moves calls between threads and
/// narrows the V accumulation to a latent range, it must not move a bit.
///
/// The query is the oracle's token 0; the cache rows are the oracle's six
/// `kvr` rows, cycled and scaled element-wise by 1 ± 0.25 noise — real
/// magnitudes, so the softmax spreads over many keys and the running max
/// bumps between blocks (the rescale path), and every row distinct, so a
/// range reading another key's row cannot agree by accident. Key counts
/// 7, 1025, 4097 end on a partial 32-key block.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_heads_split_bit_identical() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let blk = 0usize;
    let p = attn::MlaParams::read(&g, blk).unwrap();
    let derived = Derived::new(&g).unwrap();
    let views = &derived.attn_plan(blk).unwrap().v_up_views;
    let wblocks = derived.wk_b_all_heads(blk).unwrap();

    let load2 = |name: &str, occ: u32| -> Tensor2 {
        let (v, inf) = o.load(name, occ);
        Tensor2::from_vec(
            inf.ne[0] as usize,
            (inf.ne[1] * inf.ne[2] * inf.ne[3]) as usize,
            v,
        )
    };
    let q_full = load2(&format!("q-{blk}"), 0);
    let q_rope_full = load2(&format!("q_rope-{blk}"), 1);
    let kvr_full = load2(&format!("kvr-{blk}"), 0);
    let d_head = p.rope_dims + p.latent;
    let first_cols =
        |src: &Tensor2, n: usize| Tensor2::from_vec(src.ne0, n, src.data[..src.ne0 * n].to_vec());

    // (label, query count, cached key rows): the three decode depths, then
    // the prefill shape — six queries over their own six rows, causal.
    let shapes: [(&str, usize, usize); 4] = [
        ("decode, depth 6", 1, 7),
        ("decode, depth 1024", 1, 1025),
        ("decode, depth 4096", 1, 4097),
        ("prefill, 6 tokens", 6, 6),
    ];
    let mut rng = Lcg(0x5911_7a11_0c0d_e5ed);
    for (label, n_tok, n_keys) in shapes {
        let cache16: Vec<u16> = if n_tok == n_keys {
            // The prefill rows are the oracle's own, the scratch-cache shape.
            (0..n_keys)
                .flat_map(|u| kvr_full.col(u).to_vec())
                .map(gguf::quant::f32_to_f16_bits)
                .collect()
        } else {
            let mut c = Vec::with_capacity(n_keys * d_head);
            for u in 0..n_keys {
                for &v in kvr_full.col(u % kvr_full.ne1) {
                    c.push(gguf::quant::f32_to_f16_bits(
                        v * (1.0 + 0.25 * rng.range(-1.0, 1.0)),
                    ));
                }
            }
            c
        };
        let cache = model::kv::KvRows::new(&cache16, d_head);
        // Keys at positions 0..n_keys; a decode query is the last of them.
        let key_slots: Vec<Slot> = (0..n_keys as u32)
            .map(|u| Slot { seq: 0, pos: u })
            .collect();
        let q_slots: Vec<Slot> = if n_tok == n_keys {
            key_slots.clone()
        } else {
            vec![Slot {
                seq: 0,
                pos: (n_keys - 1) as u32,
            }]
        };
        let q = first_cols(&q_full, n_tok);
        let q_rope = first_cols(&q_rope_full, p.n_head * n_tok);

        let q_nope2 = attn::q_nope2_absorbed(wblocks, &q, &p).unwrap();
        let kqv_compressed =
            attn::flash_attn_latent(&q_rope, &q_nope2, cache, &key_slots, &q_slots, &p);
        let kqv_2d = attn::wv_b_heads_with(&g, views, &kqv_compressed, &p).unwrap();

        for halves in [1usize, 2] {
            let (fq, fc, f2) = attn::attn_heads_split(
                &g, wblocks, &q, &q_rope, cache, &key_slots, &q_slots, views, &p, halves,
            )
            .unwrap();
            let cmp = |name: &str, chain: &Tensor2, split: &Tensor2| {
                assert_eq!(
                    (split.ne0, split.ne1),
                    (chain.ne0, chain.ne1),
                    "{name} shape, {label}, halves {halves}"
                );
                let at = chain
                    .data
                    .iter()
                    .zip(&split.data)
                    .position(|(x, y)| x.to_bits() != y.to_bits());
                assert!(
                    at.is_none(),
                    "{name} ({label}, halves {halves}): split differs from the chain at \
                     element {at:?} of {} — the split moved a bit",
                    chain.data.len()
                );
            };
            cmp("q_nope2", &q_nope2, &fq);
            cmp("kqv_compressed", &kqv_compressed, &fc);
            cmp("kqv_2d", &kqv_2d, &f2);
            eprintln!(
                "attn_heads_split, {label:<19} halves {halves}: q_nope2, kqv_compressed, kqv_2d \
                 bit-identical to the chain ({} keys, {} + {} + {} values)",
                n_keys,
                fq.data.len(),
                fc.data.len(),
                f2.data.len()
            );
        }
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

/// The AVX2 flash kernel against its scalar twin on synthetic data — the
/// scalar path IS the oracle here (same values, ik's own sum order), so this
/// needs no model file and no dump, only AVX2+FMA+F16C.
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
/// a run says how much headroom each leaves.
#[test]
#[ignore = "hw: needs AVX2+FMA+F16C (the box); synthetic data, no model file"]
fn hw_flash_simd_bands_against_scalar() {
    // The real MLA's shapes (576-wide row, 512 latent) at a small head/token
    // count. The key count (100) crosses three whole 32-key blocks plus a
    // partial fourth (the padding break), and the first query's position
    // admits 50 keys — past the block boundary, into the M-bump rescale.
    let p = attn::MlaParams {
        n_head: 3,
        kq_head: 576,
        nope: 512,
        rope_dims: 64,
        v_head: 128,
        latent: 512,
        eps: 1e-6,
        // The rope fields are dead here — flash reads only the dims.
        rope: attn::RopeParams {
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
    // One flat row-major buffer, the cache's own layout: the RNG draws run in
    // the same per-row, per-element order as before, so the values are the
    // same bytes the per-row fixture held.
    let width = p.rope_dims + p.latent;
    let mut keys16: Vec<u16> = Vec::with_capacity(n_keys * width);
    let mut v_lo = f32::INFINITY;
    let mut v_hi = f32::NEG_INFINITY;
    for _ in 0..n_keys {
        let start = keys16.len();
        for _ in 0..width {
            keys16.push(gguf::quant::f32_to_f16_bits(rng.range(-4.0, 4.0)));
        }
        for &b in &keys16[start + p.rope_dims..] {
            let v = gguf::quant::half_to_f32(b);
            v_lo = v_lo.min(v);
            v_hi = v_hi.max(v);
        }
    }
    let keys = model::kv::KvRows::new(&keys16, width);
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
            for u in 0..n_keys {
                let k = keys.row(u);
                let d = (attn::kq_dot_simd(&qrow, k) - attn::kq_dot_fa4(&qrow, k)).abs();
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
    let a = attn::flash_attn_latent(&q_rope, &q_nope2, keys, &key_slots, &q_slots, &p);
    let b = attn::flash_attn_latent_scalar(&q_rope, &q_nope2, keys, &key_slots, &q_slots, &p);
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

/// The split-out V-accumulation twins against each other, bit for bit — the
/// property the kernel-level "given the same weights, bit-identical" claim
/// rests on, pinned at helper granularity so latents the full kernel would
/// never vectorize (64 and 72: one tile, and tile + tail) still cover the
/// tile tail. Both `r` buffers start from the same random non-zero state:
/// the accumulator carries across blocks in the kernel, never re-zeroed
/// between them.
#[test]
#[ignore = "hw: needs AVX2+FMA+F16C (the box); synthetic data, no model file"]
fn hw_flash_v_accum_avx2_matches_scalar() {
    if !(std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c"))
    {
        eprintln!(
            "hw_flash_v_accum_avx2_matches_scalar: no AVX2+FMA+F16C on this machine — skipped"
        );
        return;
    }
    let rope = 64usize;
    let mut rng = Lcg(0x0acc_6512_2311_9f10);
    let nz = |rng: &mut Lcg| {
        let v = rng.range(-2.0, 2.0);
        if v == 0.0 { 0.25 } else { v }
    };
    let mut compared = 0usize;
    // 512 = the model's latent (8 whole tiles), 64 = one tile, 72 = tile +
    // single-vector tail.
    for &latent in &[512usize, 64, 72] {
        let width = rope + latent;
        // 40 rows: a whole 32-lane block at 0, and a short block at 32 whose
        // lanes past the cache end carry w = 0 — the padding shape the row
        // kernel's `break` leaves behind.
        let mut keys16: Vec<u16> = Vec::with_capacity(40 * width);
        for _ in 0..40 * width {
            keys16.push(gguf::quant::f32_to_f16_bits(rng.range(-4.0, 4.0)));
        }
        let keys = model::kv::KvRows::new(&keys16, width);
        // (blk, half-masked?, shape name) — the three weight shapes the row
        // kernel produces: every lane active, a random ~half masked to
        // exactly 0.0, and the short last block with the past-the-end lanes
        // masked.
        for (blk, half, shape) in [
            (0usize, false, "all 32 lanes active"),
            (0usize, true, "random half masked"),
            (
                32usize,
                false,
                "short last block, lanes past the end masked",
            ),
        ] {
            let mut w = [0.0f32; 32];
            for (l, wl) in w.iter_mut().enumerate() {
                *wl = if blk + l >= 40 || (half && rng.unit() < 0.5) {
                    0.0
                } else {
                    rng.range(1e-6, 1.0)
                };
            }
            let mut r_s: Vec<f32> = (0..latent).map(|_| nz(&mut rng)).collect();
            let mut r_a = r_s.clone();
            attn::flash_v_accum_scalar(&w, keys, blk, rope, &mut r_s);
            // SAFETY: the ISA check at the top owns the fn's ISA half;
            // `keys` is `rope + latent` wide, `r` is `latent` long (a
            // multiple of 8 in all three shapes), and every lane with
            // w[l] != 0 satisfies `blk + l < 40 = keys.len()` by the mask
            // construction above.
            unsafe { attn::flash_v_accum_avx2(&w, keys, blk, rope, &mut r_a) };
            for (d, (&a, &b)) in r_s.iter().zip(r_a.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "V accumulation, latent {latent}, blk {blk} ({shape}): element {d} of {} — scalar {a:e} vs avx2 {b:e}",
                    r_s.len()
                );
            }
            compared += r_s.len();
        }
    }
    eprintln!(
        "hw_flash_v_accum_avx2_matches_scalar: {compared} accumulator elements bit-identical \
         (latent 512/64/72 x all-active/half-masked/short-block)"
    );
}

/// The synthetic deterministic batch the flash-SIMD lever test runs on: the
/// real MLA's shapes (576-wide row, 512 latent — a shape the SIMD path WOULD
/// take: row width % 32 == 0, latent % 8 == 0) at a small head/token count.
/// No model file, no oracle — only the box's ISA. One struct so the re-exec
/// child and its parent test consume the same bytes.
struct FlashLeverFixture {
    p: attn::MlaParams,
    q_rope: Tensor2,
    q_nope2: Tensor2,
    /// The KV rows as one flat row-major buffer, the cache's own layout.
    keys16: Vec<u16>,
    key_slots: Vec<Slot>,
    q_slots: Vec<Slot>,
}

impl FlashLeverFixture {
    /// The cache view over the flat buffer, at the real row width.
    fn keys(&self) -> model::kv::KvRows<'_> {
        model::kv::KvRows::new(&self.keys16, self.p.rope_dims + self.p.latent)
    }
}

fn flash_lever_fixture() -> FlashLeverFixture {
    let p = attn::MlaParams {
        n_head: 2,
        kq_head: 576,
        nope: 512,
        rope_dims: 64,
        v_head: 128,
        latent: 512,
        eps: 1e-6,
        // The rope fields are dead here — flash reads only the dims.
        rope: attn::RopeParams {
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
    let mut rng = Lcg(0x1eed_0000_beef_cafe);
    let n_tok = 3usize;
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
    // One flat row-major buffer, the cache's own layout: the RNG draws run in
    // the same per-row, per-element order as before, so the fixture's bytes are
    // the ones the per-row fixture held.
    let width = p.rope_dims + p.latent;
    let mut keys16: Vec<u16> = Vec::with_capacity(n_keys * width);
    for _ in 0..n_keys {
        for _ in 0..width {
            keys16.push(gguf::quant::f32_to_f16_bits(rng.range(-4.0, 4.0)));
        }
    }
    // Every query keeps at least one allowed key; the key count crosses
    // three whole 32-key blocks plus a partial fourth.
    let key_slots: Vec<Slot> = (0..n_keys)
        .map(|u| Slot {
            seq: if u < 50 { 0 } else { 1 },
            pos: (u % 25) as u32,
        })
        .collect();
    let q_slots: Vec<Slot> = [24u32, 0, 5]
        .iter()
        .map(|&pos| Slot { seq: 0, pos })
        .collect();
    FlashLeverFixture {
        p,
        q_rope,
        q_nope2,
        keys16,
        key_slots,
        q_slots,
    }
}

/// The re-exec entry point of [`hw_flash_simd_lever_forces_scalar`]. Not a
/// test of its own: when `BLOOMERY_FLASH_SIMD_CHILD_DUMP` is absent (i.e.
/// someone ran the file directly) it returns without doing anything — the
/// pattern `tests/mt.rs`'s child helper set.
#[test]
#[ignore = "hw: re-exec child of hw_flash_simd_lever_forces_scalar; standalone it is a no-op"]
fn hw_flash_simd_lever_child_dump() {
    let Ok(dump) = std::env::var("BLOOMERY_FLASH_SIMD_CHILD_DUMP") else {
        eprintln!("lever child: no BLOOMERY_FLASH_SIMD_CHILD_DUMP, nothing to do");
        return;
    };
    let f = flash_lever_fixture();
    // The public dispatch: with BLOOMERY_FLASH_SIMD=0 the OnceLock in
    // `flash_simd` must force the scalar row twin for the whole process.
    let out = attn::flash_attn_latent(
        &f.q_rope,
        &f.q_nope2,
        f.keys(),
        &f.key_slots,
        &f.q_slots,
        &f.p,
    );
    let bytes: Vec<u8> = out.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&dump, &bytes).unwrap();
    eprintln!(
        "lever child: {} f32 ({} bytes) dumped to {dump}",
        out.data.len(),
        bytes.len()
    );
}

/// `BLOOMERY_FLASH_SIMD=0` is the A/B instrument of record for re-pins, so
/// the env branch (`flash_simd`'s OnceLock) must be exercised. The lever is
/// proven the only way a process-wide flag can be: a re-exec'd child of this binary —
/// same code, same fixture, the env var set — runs the PUBLIC
/// `flash_attn_latent` and dumps its output bytes; the parent requires those
/// bytes to equal its own in-process `flash_attn_latent_scalar` run on the
/// same fixture. A broken lever (the child silently running the SIMD twin)
/// moves the output by the kq sum-order ULPs and reds the byte compare.
#[test]
#[ignore = "hw: needs AVX2+FMA+F16C (the box); synthetic data, no model file"]
fn hw_flash_simd_lever_forces_scalar() {
    let f = flash_lever_fixture();

    // The scalar oracle, in-process: the bit-exact twin the SIMD path is
    // banded against by `hw_flash_simd_bands_against_scalar`.
    let want = attn::flash_attn_latent_scalar(
        &f.q_rope,
        &f.q_nope2,
        f.keys(),
        &f.key_slots,
        &f.q_slots,
        &f.p,
    );
    let want_bytes: Vec<u8> = want.data.iter().flat_map(|v| v.to_le_bytes()).collect();

    // The child: fresh process, BLOOMERY_FLASH_SIMD=0, public entry point.
    let exe = std::env::current_exe().unwrap();
    let dump =
        std::env::temp_dir().join(format!("bloomery-flash-lever-{}.f32", std::process::id()));
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "hw_flash_simd_lever_child_dump",
            "--ignored",
            "--nocapture",
        ])
        .env("BLOOMERY_FLASH_SIMD", "0")
        .env("BLOOMERY_FLASH_SIMD_CHILD_DUMP", &dump)
        .output()
        .expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "lever child failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let got = std::fs::read(&dump).unwrap();
    let at = got.iter().zip(&want_bytes).position(|(x, y)| x != y);
    assert_eq!(
        got,
        want_bytes,
        "BLOOMERY_FLASH_SIMD=0 did not force the scalar path: the child's \
         flash_attn_latent output differs from flash_attn_latent_scalar \
         (first differing byte at {at:?} of {})",
        got.len()
    );
    eprintln!(
        "lever child (env-forced) vs in-process scalar: byte-identical ({} bytes)",
        got.len()
    );

    // Evidence only, deliberately NOT asserted: the parent's own dispatch
    // leg (env unset, SIMD on this box) against the same scalar twin. It
    // differing is expected — the documented kq reassociation band; it
    // being byte-identical would mean this fixture cannot tell the two row
    // twins apart, in which case the compare above proves nothing.
    let simd = attn::flash_attn_latent(
        &f.q_rope,
        &f.q_nope2,
        f.keys(),
        &f.key_slots,
        &f.q_slots,
        &f.p,
    );
    let same = simd
        .data
        .iter()
        .zip(&want.data)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!(
        "parent in-process flash_attn_latent (SIMD, env unset): {} the scalar twin on this fixture",
        if same {
            "byte-identical to"
        } else {
            "differs from"
        }
    );
}

/// The KV prefetch lever is observed, not assumed: on a one-query row the
/// AVX2 twin reports which arm and which distance it took, every arm is
/// bit-identical (a prefetch is a hint, never a value), and a multi-query row never
/// prefetches whatever the lever says. The observable is process-global and
/// every simd row writes it, so the assertions run in a fresh child process
/// where this is the only test — the same re-exec shape as
/// `hw_flash_simd_lever_forces_scalar`; in the parent's `--ignored` run a
/// neighbouring test's multi-query row could land between the call and
/// the read.
#[test]
#[ignore]
fn hw_kv_prefetch_lever_is_observed() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "hw_kv_prefetch_lever_child",
            "--ignored",
            "--nocapture",
        ])
        .env("BLOOMERY_KV_PREFETCH_CHILD", "1")
        .output()
        .expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "kv prefetch lever child failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("kv prefetch lever: one-query on/off observed"),
        "child ran but did not report the observation\n{stdout}"
    );
}

/// The child half of `hw_kv_prefetch_lever_is_observed`: only meaningful in
/// the re-exec, where no other row writes the observable.
#[test]
#[ignore]
fn hw_kv_prefetch_lever_child() {
    if std::env::var_os("BLOOMERY_KV_PREFETCH_CHILD").is_none() {
        eprintln!("hw_kv_prefetch_lever_child: run through hw_kv_prefetch_lever_is_observed");
        return;
    }
    if !(std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c"))
    {
        panic!(
            "hw_kv_prefetch_lever_child: no AVX2+FMA+F16C on this box — the gate has nothing to observe"
        );
    }
    let f = flash_lever_fixture();
    let n_tok = f.q_slots.len();
    // One query (token 0) against the whole cache: `q_rope` columns are
    // `t * n_head + h`, `q_nope2` columns are `h * n_tok + t`.
    let mut q_rope = Tensor2::zeros(f.p.rope_dims, f.p.n_head);
    let mut q_nope2 = Tensor2::zeros(f.p.latent, f.p.n_head);
    for h in 0..f.p.n_head {
        q_rope.col_mut(h).copy_from_slice(f.q_rope.col(h));
        q_nope2.col_mut(h).copy_from_slice(f.q_nope2.col(h * n_tok));
    }
    let q1 = &f.q_slots[..1];

    attn::set_kv_prefetch(Some(true));
    let on = attn::flash_attn_latent(&q_rope, &q_nope2, f.keys(), &f.key_slots, q1, &f.p);
    assert_eq!(
        attn::last_kv_prefetch(),
        Some(true),
        "one-query row with the lever forced on must report prefetching"
    );
    attn::set_kv_prefetch(Some(false));
    let off = attn::flash_attn_latent(&q_rope, &q_nope2, f.keys(), &f.key_slots, q1, &f.p);
    assert_eq!(
        attn::last_kv_prefetch(),
        Some(false),
        "one-query row with the lever forced off must report no prefetch"
    );
    let same = on
        .data
        .iter()
        .zip(off.data.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        same,
        "prefetch on vs off changed a value — a hint touched arithmetic"
    );
    assert_eq!(
        attn::last_kv_prefetch_rows(),
        Some(0),
        "the off arm must report distance 0"
    );

    // The distance lever: each forced distance is the one the row reports,
    // and none of them moves a bit. 100 keys, so every distance here issues
    // hints inside the row.
    let distances = [1usize, 2, 4, 8];
    for d in distances {
        attn::set_kv_prefetch_rows(Some(d));
        let at_d = attn::flash_attn_latent(&q_rope, &q_nope2, f.keys(), &f.key_slots, q1, &f.p);
        assert_eq!(
            attn::last_kv_prefetch_rows(),
            Some(d),
            "one-query row with the distance forced to {d} must report {d}"
        );
        assert!(
            at_d.data
                .iter()
                .zip(off.data.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "prefetch distance {d} vs off changed a value — a hint touched arithmetic"
        );
    }

    // Multi-query rows (prefill) never prefetch, lever or no lever.
    attn::set_kv_prefetch_rows(Some(4));
    let _ = attn::flash_attn_latent(
        &f.q_rope,
        &f.q_nope2,
        f.keys(),
        &f.key_slots,
        &f.q_slots,
        &f.p,
    );
    assert_eq!(
        attn::last_kv_prefetch(),
        Some(false),
        "a {n_tok}-query row must not prefetch"
    );
    attn::set_kv_prefetch_rows(None);
    println!(
        "kv prefetch lever: one-query on/off observed, distances {distances:?} observed, {} values bit-identical, {n_tok}-query never prefetches",
        on.data.len()
    );
}

/// One shape of the split-K band fixture: synthetic queries and keys at the
/// real MLA row width, positions `0..n_keys` in one sequence.
struct SplitKCase {
    label: &'static str,
    q_rope: Tensor2,
    q_nope2: Tensor2,
    keys16: Vec<u16>,
    key_slots: Vec<Slot>,
    q_slots: Vec<Slot>,
    v_max: f32,
}

/// The synthetic MLA shape the split-K band test runs at (576-wide rows,
/// 512 latent, three heads). The rope fields are dead — flash reads only
/// the dims.
fn split_k_params() -> attn::MlaParams {
    attn::MlaParams {
        n_head: 3,
        kq_head: 576,
        nope: 512,
        rope_dims: 64,
        v_head: 128,
        latent: 512,
        eps: 1e-6,
        rope: attn::RopeParams {
            n_dims: 64,
            freq_base: 1e4,
            freq_scale: 0.025,
            ext_factor: 1.0,
            mscale_param: 0.7,
            corr_dims: [10.0, 23.0],
            theta_scale: 0.8,
        },
        kq_scale: 0.078,
    }
}

/// The band fixture, deterministic so the re-exec child and its parent
/// consume the same bytes: a decode query over 1025 and 4097 keys (33 and 129
/// blocks — uneven segments, the last block partial) and a four-query causal
/// batch over 1100 keys whose rows see 1, 40, 700 and 1100 of them (one
/// segment, two, many). Small queries keep the softmax spread over many keys,
/// so every segment carries weight and the running max bumps.
fn split_k_cases(p: &attn::MlaParams) -> Vec<SplitKCase> {
    let mut rng = Lcg(0x5911_7a11_ba5e_0001);
    let width = p.rope_dims + p.latent;
    let shapes: [(&'static str, usize, &[u32]); 3] = [
        ("decode, 1025 keys", 1025, &[1024]),
        ("decode, 4097 keys", 4097, &[4096]),
        ("4 queries, 1100 keys", 1100, &[0, 39, 699, 1099]),
    ];
    shapes
        .iter()
        .map(|&(label, n_keys, qpos)| {
            let n_tok = qpos.len();
            let mut q_rope = Tensor2::zeros(p.rope_dims, p.n_head * n_tok);
            let mut q_nope2 = Tensor2::zeros(p.latent, p.n_head * n_tok);
            for t in 0..n_tok {
                for h in 0..p.n_head {
                    for v in q_rope.col_mut(t * p.n_head + h).iter_mut() {
                        *v = rng.range(-0.3, 0.3);
                    }
                    for v in q_nope2.col_mut(h * n_tok + t).iter_mut() {
                        *v = rng.range(-0.3, 0.3);
                    }
                }
            }
            let mut keys16: Vec<u16> = Vec::with_capacity(n_keys * width);
            let mut v_max = 0.0f32;
            for _ in 0..n_keys {
                let start = keys16.len();
                for _ in 0..width {
                    keys16.push(gguf::quant::f32_to_f16_bits(rng.range(-4.0, 4.0)));
                }
                for &b in &keys16[start + p.rope_dims..] {
                    v_max = v_max.max(gguf::quant::half_to_f32(b).abs());
                }
            }
            SplitKCase {
                label,
                q_rope,
                q_nope2,
                keys16,
                key_slots: (0..n_keys as u32)
                    .map(|u| Slot { seq: 0, pos: u })
                    .collect(),
                q_slots: qpos.iter().map(|&pos| Slot { seq: 0, pos }).collect(),
                v_max,
            }
        })
        .collect()
}

/// Both row twins on every case, in order: the dispatch leg, then the scalar.
fn split_k_outputs(p: &attn::MlaParams, cases: &[SplitKCase]) -> Vec<Tensor2> {
    let width = p.rope_dims + p.latent;
    let mut outs = Vec::new();
    for c in cases {
        let keys = model::kv::KvRows::new(&c.keys16, width);
        outs.push(attn::flash_attn_latent(
            &c.q_rope,
            &c.q_nope2,
            keys,
            &c.key_slots,
            &c.q_slots,
            p,
        ));
        outs.push(attn::flash_attn_latent_scalar(
            &c.q_rope,
            &c.q_nope2,
            keys,
            &c.key_slots,
            &c.q_slots,
            p,
        ));
    }
    outs
}

/// The re-exec entry point of [`hw_flash_split_k_bands_against_one_segment`]:
/// without `BLOOMERY_SPLITK_BAND_DUMP` it does nothing.
#[test]
#[ignore = "hw: re-exec child of hw_flash_split_k_bands_against_one_segment; standalone it is a no-op"]
fn hw_flash_split_k_band_child() {
    let Ok(dump) = std::env::var("BLOOMERY_SPLITK_BAND_DUMP") else {
        eprintln!("split-K band child: no BLOOMERY_SPLITK_BAND_DUMP, nothing to do");
        return;
    };
    let p = split_k_params();
    let outs = split_k_outputs(&p, &split_k_cases(&p));
    let bytes: Vec<u8> = outs
        .iter()
        .flat_map(|t| t.data.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    std::fs::write(&dump, &bytes).unwrap();
    eprintln!("split-K band child: {} bytes dumped to {dump}", bytes.len());
}

/// Split-K against the single-pass online softmax it replaced, on synthetic
/// data (no model file): the same row twins, the same scores bit for bit (the
/// kq dot per key is untouched), only the plan differs — the default
/// segments merged by `combine_segments` in this process, one segment in a
/// re-exec'd child run with `BLOOMERY_FLASH_SEGMENTS=1` (the lever is read
/// once per process). This is the gate on the merge itself: the chain and
/// the fused dispatch share it, so their bit identity cannot catch a wrong
/// merge, and the oracle's 6-token rows have one segment, where the merge is
/// the identity.
///
/// The band is the textbook reassociation bound, derived, with `u = 2⁻²⁴`:
/// both plans sum `S` and every `R[d]` over the same `n` weighted terms in
/// different association, each rounding at most `u` of a running partial no
/// larger than `Σ w·|V|` (`n·u` per path); each weight carries at most
/// `n_blocks` rescale factors in one plan and a segment's rescales plus one
/// merge factor in the other, each off by at most `2u` (exp + multiply) plus
/// `v_expf`'s few ulp (`8` rescales of slack for both). So `R` and `S` each move
/// by at most `γ = (2n + 4·(n_blocks + 8))·u` relative to `Σ w·|V|` and `Σ w`,
/// and `R/S` by at most `2γ·max|V|` plus the final multiply's `u`. A dropped or
/// misweighted segment moves the output by about its weight share times the V
/// spread — orders above this band. A child that ignored the lever would
/// return the parent's own bytes: the test fails on that too.
#[test]
#[ignore = "hw: needs AVX2+FMA+F16C (the box); synthetic data, no model file"]
fn hw_flash_split_k_bands_against_one_segment() {
    let p = split_k_params();
    let cases = split_k_cases(&p);
    let split = split_k_outputs(&p, &cases);

    let exe = std::env::current_exe().unwrap();
    let dump =
        std::env::temp_dir().join(format!("bloomery-splitk-band-{}.f32", std::process::id()));
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "hw_flash_split_k_band_child",
            "--ignored",
            "--nocapture",
        ])
        .env("BLOOMERY_FLASH_SEGMENTS", "1")
        .env("BLOOMERY_SPLITK_BAND_DUMP", &dump)
        .output()
        .expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "split-K band child failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let raw = std::fs::read(&dump).unwrap();
    let _ = std::fs::remove_file(&dump);
    let one: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let total: usize = split.iter().map(|t| t.data.len()).sum();
    assert_eq!(one.len(), total, "child output length");

    let u = 2.0f32.powi(-24);
    let mut at = 0usize;
    let mut moved_total = 0usize;
    for (i, got) in split.iter().enumerate() {
        let c = &cases[i / 2];
        let leg = if i % 2 == 0 { "dispatch" } else { "scalar" };
        let n = c.key_slots.len() as f32;
        let n_blocks = c.key_slots.len().div_ceil(32) as f32;
        let gamma = (2.0 * n + 4.0 * (n_blocks + 8.0)) * u;
        let band = 2.0 * gamma * c.v_max + u * c.v_max;
        let want = &one[at..at + got.data.len()];
        at += got.data.len();
        let mut worst = 0.0f32;
        let mut sq = 0.0f64;
        let mut moved = 0usize;
        for (k, (&a, &b)) in got.data.iter().zip(want).enumerate() {
            let d = (a - b).abs();
            assert!(
                d <= band,
                "{} ({leg}): split-K vs one segment |diff| {d:e} at index {k} exceeds the \
                 reassociation band {band:e}",
                c.label
            );
            worst = worst.max(d);
            sq += f64::from(d) * f64::from(d);
            moved += usize::from(a.to_bits() != b.to_bits());
        }
        moved_total += moved;
        let rms = (sq / got.data.len() as f64).sqrt();
        eprintln!(
            "split-K vs one segment, {:<21} {leg:<8} max|diff| = {worst:e}  rms {rms:.3e}  \
             ({moved} of {} values moved; max|V| {}, band {band:e})",
            c.label,
            got.data.len(),
            c.v_max
        );
    }
    assert!(
        moved_total > 0,
        "the one-segment child returned the split's own bytes — BLOOMERY_FLASH_SEGMENTS=1 \
         did not reach the plan, so the band above compared a thing with itself"
    );
}

/// The oracle's decode-shaped fixture at `n_keys` cached keys (the query is
/// the last): the oracle's token-0 `q`/`q_rope`, the six `kvr` rows cycled and
/// scaled element-wise by 1 ± 0.25 noise — the recipe of
/// `hw_attn_heads_split_bit_identical`.
struct DecodeFixture {
    q: Tensor2,
    q_rope: Tensor2,
    cache16: Vec<u16>,
    key_slots: Vec<Slot>,
    q_slots: Vec<Slot>,
}

fn decode_fixture(
    o: &oracle::Oracle,
    p: &attn::MlaParams,
    n_keys: usize,
    seed: u64,
) -> DecodeFixture {
    let blk = 0usize;
    let load2 = |name: &str, occ: u32| -> Tensor2 {
        let (v, inf) = o.load(name, occ);
        Tensor2::from_vec(
            inf.ne[0] as usize,
            (inf.ne[1] * inf.ne[2] * inf.ne[3]) as usize,
            v,
        )
    };
    let q_full = load2(&format!("q-{blk}"), 0);
    let q_rope_full = load2(&format!("q_rope-{blk}"), 1);
    let kvr_full = load2(&format!("kvr-{blk}"), 0);
    let first_cols =
        |src: &Tensor2, n: usize| Tensor2::from_vec(src.ne0, n, src.data[..src.ne0 * n].to_vec());
    let mut rng = Lcg(seed);
    let mut cache16 = Vec::with_capacity(n_keys * (p.rope_dims + p.latent));
    for u in 0..n_keys {
        for &v in kvr_full.col(u % kvr_full.ne1) {
            cache16.push(gguf::quant::f32_to_f16_bits(
                v * (1.0 + 0.25 * rng.range(-1.0, 1.0)),
            ));
        }
    }
    DecodeFixture {
        q: first_cols(&q_full, 1),
        q_rope: first_cols(&q_rope_full, p.n_head),
        cache16,
        key_slots: (0..n_keys as u32)
            .map(|u| Slot { seq: 0, pos: u })
            .collect(),
        q_slots: vec![Slot {
            seq: 0,
            pos: (n_keys - 1) as u32,
        }],
    }
}

/// The re-exec entry point of [`hw_attn_split_k_is_thread_count_invariant`]:
/// without `BLOOMERY_SPLITK_CHILD_DUMP` it does nothing.
#[test]
#[ignore = "hw: re-exec child of hw_attn_split_k_is_thread_count_invariant; standalone it is a no-op"]
fn hw_attn_split_k_threads_child() {
    let Ok(dump) = std::env::var("BLOOMERY_SPLITK_CHILD_DUMP") else {
        eprintln!("split-K child: no BLOOMERY_SPLITK_CHILD_DUMP, nothing to do");
        return;
    };
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let blk = 0usize;
    let p = attn::MlaParams::read(&g, blk).unwrap();
    let derived = Derived::new(&g).unwrap();
    let views = &derived.attn_plan(blk).unwrap().v_up_views;
    let wblocks = derived.wk_b_all_heads(blk).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    for (n_keys, seed) in [(1025usize, 0x7e11_0001u64), (4097, 0x7e11_0002)] {
        let f = decode_fixture(&o, &p, n_keys, seed);
        let cache = model::kv::KvRows::new(&f.cache16, p.rope_dims + p.latent);
        let (fq, fc, f2) = attn::attn_heads_fused(
            &g,
            wblocks,
            &f.q,
            &f.q_rope,
            cache,
            &f.key_slots,
            &f.q_slots,
            views,
            &p,
        )
        .unwrap();
        let chain = attn::flash_attn_latent(&f.q_rope, &fq, cache, &f.key_slots, &f.q_slots, &p);
        for t in [&fq, &fc, &f2, &chain] {
            bytes.extend(t.data.iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    std::fs::write(&dump, &bytes).unwrap();
    eprintln!(
        "split-K child: {} bytes with threads::pool().threads() = {} dumped to {dump}",
        bytes.len(),
        threads::pool().threads()
    );
}

/// The split-K plan must not depend on the thread count, not by one bit:
/// children at `BLOOMERY_THREADS` 1, 32 and 3 run the fused decode dispatch
/// (and the chain's flash) at 1025 and 4097 cached keys and dump every output
/// byte; the parent compares them. `tests/mt.rs` cannot see this — its prompt
/// is six tokens, one 32-key block, one segment. The children's pool sizes are
/// printed as evidence, the `tests/mt.rs` pattern.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_split_k_is_thread_count_invariant() {
    let exe = std::env::current_exe().unwrap();
    let run = |threads_env: &str| -> Vec<u8> {
        let dump = std::env::temp_dir().join(format!(
            "bloomery-splitk-child-{}-t{threads_env}.bin",
            std::process::id()
        ));
        let out = std::process::Command::new(&exe)
            .args([
                "--exact",
                "hw_attn_split_k_threads_child",
                "--ignored",
                "--nocapture",
            ])
            .env("BLOOMERY_SPLITK_CHILD_DUMP", &dump)
            .env("BLOOMERY_THREADS", threads_env)
            .output()
            .expect("re-exec of this test binary");
        assert!(
            out.status.success(),
            "child at BLOOMERY_THREADS={threads_env} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        let bytes = std::fs::read(&dump).unwrap();
        let _ = std::fs::remove_file(&dump);
        bytes
    };
    let one = run("1");
    for other in ["32", "3"] {
        let got = run(other);
        let at = one.iter().zip(&got).position(|(a, b)| a != b);
        assert!(
            got.len() == one.len() && at.is_none(),
            "split-K outputs differ between BLOOMERY_THREADS=1 and ={other}: first differing \
             byte at {at:?} of {} (lengths {} vs {})",
            one.len(),
            one.len(),
            got.len()
        );
        eprintln!(
            "split-K decode, threads=1 vs threads={other:<3} byte-identical ({} bytes)",
            one.len()
        );
    }
}

/// The decode depths the bundle band runs at: 257 keys (9 blocks) stays on
/// the one-dispatch shape, where every row runs the tile one head at a time;
/// 1025 and 4097 take split-K, where an item runs it on a bundle of 8.
const BUNDLE_CASES: [(usize, u64); 3] =
    [(257, 0xb0d1_e001), (1025, 0xb0d1_e002), (4097, 0xb0d1_e003)];

/// Every `kqv_compressed` of the bundle cases through the fused dispatch,
/// and the realized kq-dot deviation of this process's AVX2 kernel from the
/// scalar fa4 order over the same (head, key) pairs:
/// `max |kq_dot_simd − kq_dot_fa4|` (`kq_dot_simd` follows the lever).
/// Also `max|V|` over every case's cache.
fn bundle_outputs(o: &oracle::Oracle) -> (Vec<Tensor2>, f32, f32) {
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let blk = 0usize;
    let p = attn::MlaParams::read(&g, blk).unwrap();
    let derived = Derived::new(&g).unwrap();
    let views = &derived.attn_plan(blk).unwrap().v_up_views;
    let wblocks = derived.wk_b_all_heads(blk).unwrap();
    let d_head = p.rope_dims + p.latent;
    let mut outs = Vec::new();
    let mut delta = 0.0f32;
    let mut v_max = 0.0f32;
    for (n_keys, seed) in BUNDLE_CASES {
        let f = decode_fixture(o, &p, n_keys, seed);
        let cache = model::kv::KvRows::new(&f.cache16, d_head);
        let (fq, fc, _) = attn::attn_heads_fused(
            &g,
            wblocks,
            &f.q,
            &f.q_rope,
            cache,
            &f.key_slots,
            &f.q_slots,
            views,
            &p,
        )
        .unwrap();
        for h in 0..p.n_head {
            let mut qrow = vec![0.0f32; d_head];
            qrow[..p.rope_dims].copy_from_slice(f.q_rope.col(h));
            qrow[p.rope_dims..].copy_from_slice(fq.col(h));
            for u in 0..n_keys {
                let k = cache.row(u);
                delta = delta.max((attn::kq_dot_simd(&qrow, k) - attn::kq_dot_fa4(&qrow, k)).abs());
            }
        }
        for u in 0..n_keys {
            for &b in &cache.row(u)[p.rope_dims..] {
                v_max = v_max.max(gguf::quant::half_to_f32(b).abs());
            }
        }
        outs.push(fc);
    }
    (outs, delta, v_max)
}

/// The re-exec entry point of [`hw_attn_bundle_bands_against_per_head`]:
/// without `BLOOMERY_BUNDLE_BAND_DUMP` it does nothing.
#[test]
#[ignore = "hw: re-exec child of hw_attn_bundle_bands_against_per_head; standalone it is a no-op"]
fn hw_attn_bundle_band_child() {
    let Ok(dump) = std::env::var("BLOOMERY_BUNDLE_BAND_DUMP") else {
        eprintln!("bundle band child: no BLOOMERY_BUNDLE_BAND_DUMP, nothing to do");
        return;
    };
    let o = oracle::Oracle::open();
    let (outs, delta, _) = bundle_outputs(&o);
    let mut bytes: Vec<u8> = outs
        .iter()
        .flat_map(|t| t.data.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    bytes.extend(delta.to_le_bytes());
    std::fs::write(&dump, &bytes).unwrap();
    eprintln!("bundle band child: {} bytes dumped to {dump}", bytes.len());
}

/// The head-bundle tile (the default) against the per-head online kernel it
/// replaced (`BLOOMERY_ATTN_BUNDLE=1`, a re-exec'd child — the lever is read
/// once per process), through the fused decode dispatch on the oracle's
/// decode fixture at 257, 1025 and 4097 keys: the same plan and merge, a
/// different segment kernel. This is the float-class proof of the tile:
/// the bit gates above compare the chain and the dispatch, which share the
/// kernel whichever it is.
///
/// The band is derived. The two kernels differ in (a) the kq dot's sum
/// order — each process measures its own kernel's realized deviation from
/// the scalar fa4 order, so a score moves by at most `ε = kq_scale·(δ₈ +
/// δ₁)`; a common shift of all scores cancels in the softmax, so each weight
/// moves by a relative `e^ε − 1 ≤ 2ε` and the output, a convex combination
/// of V, by at most `2·2ε·max|V|`; and (b) rounding — the online kernel
/// applies up to `n_blocks` rescales per weight, the tile none, and both
/// sum over the same `n` terms in different association: the split-K
/// gate's `γ = (2n + 4·(n_blocks + 8))·u`, moving `R/S` by `2γ·max|V|` plus
/// the final multiply's `u·max|V|`. A skipped or misrouted head moves its
/// whole output — orders above the band. A child that ignored the lever
/// would return the parent's own bytes: the test fails on that too.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_attn_bundle_bands_against_per_head() {
    let o = oracle::Oracle::open();
    let (tile, delta8, v_max) = bundle_outputs(&o);

    let exe = std::env::current_exe().unwrap();
    let dump =
        std::env::temp_dir().join(format!("bloomery-bundle-band-{}.f32", std::process::id()));
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "hw_attn_bundle_band_child",
            "--ignored",
            "--nocapture",
        ])
        .env("BLOOMERY_ATTN_BUNDLE", "1")
        .env("BLOOMERY_BUNDLE_BAND_DUMP", &dump)
        .output()
        .expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "bundle band child failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let raw = std::fs::read(&dump).unwrap();
    let _ = std::fs::remove_file(&dump);
    let vals: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let total: usize = tile.iter().map(|t| t.data.len()).sum();
    assert_eq!(vals.len(), total + 1, "child output length");
    let delta1 = vals[total];

    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let p = attn::MlaParams::read(&g, 0).unwrap();
    let u = 2.0f32.powi(-24);
    let eps = p.kq_scale * (delta8 + delta1);
    eprintln!(
        "bundle band: kq dot vs fa4, tile δ₈ = {delta8:e}, per-head δ₁ = {delta1:e}, \
         score ε = {eps:e}, max|V| {v_max}"
    );
    let mut at = 0usize;
    let mut moved_total = 0usize;
    for (got, (n_keys, _)) in tile.iter().zip(BUNDLE_CASES) {
        let n = n_keys as f32;
        let n_blocks = n_keys.div_ceil(32) as f32;
        let gamma = (2.0 * n + 4.0 * (n_blocks + 8.0)) * u;
        let band = 2.0 * (2.0 * eps + gamma) * v_max + u * v_max;
        let want = &vals[at..at + got.data.len()];
        at += got.data.len();
        let mut worst = 0.0f32;
        let mut sq = 0.0f64;
        let mut moved = 0usize;
        for (k, (&a, &b)) in got.data.iter().zip(want).enumerate() {
            let d = (a - b).abs();
            assert!(
                d <= band,
                "decode, {n_keys} keys: bundle 8 vs bundle 1 |diff| {d:e} at index {k} \
                 (head {}) exceeds the derived band {band:e}",
                k / p.latent
            );
            worst = worst.max(d);
            sq += f64::from(d) * f64::from(d);
            moved += usize::from(a.to_bits() != b.to_bits());
        }
        moved_total += moved;
        let rms = (sq / got.data.len() as f64).sqrt();
        eprintln!(
            "bundle 8 vs bundle 1, decode, {n_keys:>4} keys  max|diff| = {worst:e}  rms {rms:.3e}  \
             ({moved} of {} values moved; band {band:e})",
            got.data.len()
        );
    }
    assert!(
        moved_total > 0,
        "the bundle-1 child returned the tile's own bytes — BLOOMERY_ATTN_BUNDLE=1 did not \
         reach the kernel, so the band above compared a thing with itself"
    );
}
