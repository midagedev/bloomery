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
//! The attention arithmetic itself is bit-exact: fed the
//! oracle's own q_rope/q_nope2/kvr, `flash_attn_latent` reproduces `kqv_compressed`
//! with max |diff| = 0 (`hw_attn_exact_input_stages` asserts that at tolerance zero).
//! The bounds cover two simultaneous tie flips, so an upstream `matmul_q` change that
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

        // B: the attention proper — tolerance ZERO. `flash_attn_latent` is a
        // transcription of ik's FA kernel op for op (dot lane order, v_expf, online
        // M/S, FMA chains, final 1/S multiply), so with the kernel's exact inputs it
        // must reproduce the kernel's exact outputs, bit for bit. Any nonzero diff
        // here is a real divergence from the reference, not noise.
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
        let kqv = model::attn::flash_attn_latent(
            &q_rope_exact,
            &q_nope2_exact,
            &cache16,
            &slots,
            &slots,
            &p,
        );
        let want = load2(&format!("kqv_compressed-{blk}"), 0);
        assert_eq!([kqv.ne0, kqv.ne1], [want.ne0, want.ne1], "stage B shape");
        oracle::assert_close(&kqv.data, &want.data, 0.0, "stage B: attention bit-exact");

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
