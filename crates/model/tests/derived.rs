//! Gate for `Derived`: precomputing the wk_b Q8_0 requant must change **nothing**.
//!
//! The claim is layered, and every layer is an exact-zero assertion — no
//! tolerance, for the same reason `tests/kv.rs` has none: a tolerance here
//! would let "rebuilt the same weights differently" hide behind drift that
//! looks like noise.
//!
//!   1. **Bytes.** What `Derived::new` builds for a (block, head) is the same
//!      bytes the pre-`Derived` code built in place, over **every** block and
//!      head — checked against a reference copy of the old two loops kept in
//!      this file. The copy is deliberate: a helper shared with `derived.rs`
//!      could be wrong once and green twice.
//!   2. **Output.** `q_nope2_absorbed` fed the cached blocks produces the same
//!      tensor as the same function fed the reference-built blocks, on the
//!      oracle's own `q` for blocks 0 and 1 (the same exact-input convention
//!      as `tests/attn.rs`). Given layer 1 this should be trivial; asserting it
//!      anyway pins the production path end to end.
//!   3. **End to end.** `forward` (which builds its own `Derived` inside) and
//!      `step` (explicit `Derived`) land on identical logits — the wiring
//!      claim that block `b`'s absorption gets block `b`'s blocks.
//!
//! `hw_` prefix: needs the box and the model file. Like `tests/kv.rs`, this
//! compares our paths against each other — `tests/attn.rs` and
//! `tests/forward.rs` are what tie them to ik.
#[path = "common/oracle.rs"]
mod oracle;

use model::Tensor2;
use model::attn::{MlaParams, Q8Block, q_nope2_absorbed, quantize_q8_0};
use model::derived::Derived;
use model::forward::{forward, new_cache, step};

/// The pre-`Derived` build, verbatim: the two loops `q_nope2_absorbed` used to
/// run per call — dequantize the head's k-up rows of `attn_kv_b`, requant
/// column `j`'s 32-value spans along the q_nope axis into Q8_0. This file's
/// copy is the "current code" the gate compares against; if `derived.rs` and
/// this ever disagree, one of them stopped being that code.
fn reference_wblocks(
    g: &gguf::Gguf,
    wkb: &gguf::TensorInfo,
    p: &MlaParams,
    h: usize,
) -> Vec<Q8Block> {
    let bytes = g.data(wkb).unwrap();
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    let nblocks = p.nope / 32;

    let mut rows = vec![0.0f32; p.nope * p.latent];
    let mut wk_row = vec![0.0f32; p.latent];
    for d in 0..p.nope {
        let off = (h * (p.nope + p.v_head) + d) * row_bytes;
        gguf::dequant_row(wkb.ty, &bytes[off..off + row_bytes], wk_row.as_mut_slice()).unwrap();
        rows[d * p.latent..(d + 1) * p.latent].copy_from_slice(&wk_row);
    }
    let mut wblocks = Vec::with_capacity(p.latent * nblocks);
    let mut vals = [0.0f32; 32];
    for j in 0..p.latent {
        for b in 0..nblocks {
            for (l, v) in vals.iter_mut().enumerate() {
                *v = rows[(32 * b + l) * p.latent + j];
            }
            wblocks.push(quantize_q8_0(&vals));
        }
    }
    wblocks
}

/// Layers 1 and 2: the bytes, everywhere, and the output they produce.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_derived_wblocks_bit_identical() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let d = Derived::new(&g).unwrap();
    let n_block = g.block_count().unwrap() as usize;

    // 1. Bytes: every block, every head, against the in-place reference build.
    for b in 0..n_block {
        let p = MlaParams::read(&g, b).unwrap();
        let wkb = g.find(&format!("blk.{b}.attn_kv_b.weight")).unwrap();
        for h in 0..p.n_head {
            let want = reference_wblocks(&g, wkb, &p, h);
            let got = d.wk_b_blocks(b, h).unwrap();
            assert_eq!(
                got,
                want.as_slice(),
                "block {b} head {h}: Derived's blocks are not the in-place bytes"
            );
        }
    }
    eprintln!("wblocks   {n_block} blocks, every head, byte-identical to the in-place build");

    // 2. Output: the production function over Derived's blocks vs over the
    //    reference-built blocks, on the oracle's own q.
    for blk in [0usize, 1] {
        let (qs, qinf) = o.load(&format!("q-{blk}"), 0);
        let q = Tensor2::from_vec(
            qinf.ne[0] as usize,
            (qinf.ne[1] * qinf.ne[2] * qinf.ne[3]) as usize,
            qs,
        );
        let p = MlaParams::read(&g, blk).unwrap();
        let wkb = g.find(&format!("blk.{blk}.attn_kv_b.weight")).unwrap();
        let mut reference_flat: Vec<Q8Block> = Vec::new();
        for h in 0..p.n_head {
            reference_flat.extend(reference_wblocks(&g, wkb, &p, h));
        }
        assert_eq!(
            d.wk_b_all_heads(blk).unwrap(),
            reference_flat.as_slice(),
            "layer 2 precondition: the flat form must be the heads concatenated"
        );
        let from_reference = q_nope2_absorbed(&reference_flat, &q, &p).unwrap();
        let from_derived = q_nope2_absorbed(d.wk_b_all_heads(blk).unwrap(), &q, &p).unwrap();
        assert_eq!(from_derived.data.len(), from_reference.data.len(), "shape");
        let worst = from_derived
            .data
            .iter()
            .zip(&from_reference.data)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "q_nope2-{blk}: cached-wblocks vs in-place-wblocks max|diff| {worst:e}"
        );
        eprintln!("q_nope2-{blk:<4} cached == in-place   max|diff| = 0   exact");
    }
}

/// Every slot the model will ever ask for is filled, and shaped as the
/// absorption indexes it: `latent * (nope/32)` blocks per head.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_derived_all_blocks_all_heads_filled() {
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let d = Derived::new(&g).unwrap();
    let n_block = g.block_count().unwrap() as usize;
    let p0 = MlaParams::read(&g, 0).unwrap();

    assert_eq!(
        std::mem::size_of::<Q8Block>(),
        34,
        "Q8_0 block is a 2-byte f16 scale + 32 int8 codes; the size math in the \
         decode binary and the MB figures assume it"
    );
    assert_eq!(
        d.filled_blocks(),
        n_block,
        "this file carries attn_kv_b in every block"
    );
    let mut total = 0usize;
    for b in 0..n_block {
        let p = MlaParams::read(&g, b).unwrap();
        let expected = p.latent * (p.nope / 32);
        for h in 0..p.n_head {
            assert_eq!(
                d.wk_b_blocks(b, h).unwrap().len(),
                expected,
                "block {b} head {h}: expected latent·nope/32 = {expected} blocks"
            );
            total += expected;
        }
    }
    eprintln!(
        "filled    {n_block} blocks × {} heads × {} blocks = {} Q8_0 blocks, {:.1} MB",
        p0.n_head,
        p0.latent * (p0.nope / 32),
        total,
        d.size_bytes() as f64 / 1e6
    );
}

/// Layer 3: the logits of the wrapper (`forward`, own `Derived`) and the
/// explicit path (`step`, handed the same `Derived`) agree bit for bit.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_derived_step_equals_forward_bit_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    let wrapper = forward(&g, &tokens).unwrap();
    let derived = Derived::new(&g).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let explicit = step(&g, &tokens, &mut cache, &derived).unwrap();

    assert_eq!(
        explicit.data.len(),
        wrapper.data.len(),
        "logit count {} vs {}",
        explicit.data.len(),
        wrapper.data.len()
    );
    let worst = explicit
        .data
        .iter()
        .zip(&wrapper.data)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "step(explicit Derived) vs forward(wrapper Derived): max|diff| {worst:e}"
    );
    eprintln!("step vs forward   max|diff| = 0   exact");
}
