//! GPU gate for the output head (docs/gpu-design.md P8's head piece):
//! `result_norm → lm_head (Q6_K) → argmax` over resident scratch, m = 1,
//! against the ik CUDA oracle (`ref_cuda_v2`). What is asserted:
//! - (a) every tap finite, and inside the pinned head bands (`BANDS`);
//! - a structural fence per tap: `rel <= 0.25`, an order of magnitude above
//!   the head's expected rels and below the O(0.5..1) rels a wrong
//!   gain/geometry produces (measured with the FAIL-first mutation below). A
//!   violation here is a layout defect, not noise;
//! - (b) an eager rerun bit-identical;
//! - (c) the argmax: the device token equals the oracle's argmax over
//!   `result_output`'s last row (both proven against the pinned ik token) and
//!   equals the host argmax of our own logits under the kernel's tie rule —
//!   strictly greater replaces, so equal values keep the LOWER index.
//!
//! The head's input is the dump's `l_out-26` (occurrence 0, the residual ADD,
//! `{2048, 1}`) — the same tensor the CPU head gate feeds
//! (`crates/model/tests/head.rs`).
//!
//! `nvfp4_head_k2688`, on a synthetic file: an NVFP4 `output.weight` of 2,688 values a row
//! (Lightning's hidden width: whole 64-value blocks and no whole q8_1 super-block of 256, so the
//! head reads the 32-value rows through `site::nvfp4_sel32_gemv`) beside its `.scale`:
//! - each logit within the band of the f64 `scale · Σ w_j x_j` of the host's dequantized row and
//!   the head's own normed row (`NVFP4_BAND` of the row's own scale, the 32-value activation
//!   rounding), the argmax the logits' maximum under the tie rule above;
//! - a head of three rows: row `c`'s logits and token are the one-row head's on that row, bit for
//!   bit (the gemv is token-major and the copy puts it in the argmax's layout).
//!
//! `nvfp4_head_k5120`, the same file at Qwen3.8-27B's hidden width, 5,120 values a row (whole q8_1
//! super-blocks of 256, so the head reads the q8_1 rows through `site::kgemv`):
//! - each logit equal bit for bit to the gemv's host rule over the head's own normed row (the
//!   host's q8_1 codes, `lane_partial_host::<Nvfp4, 2>` per lane, the butterfly, the multiply by the
//!   scale) and within the derived band of the f64 `scale · Σ w_j x_j` (the activation's half
//!   block scale a value, the gemv's roundings), the argmax the logits' maximum under the tie rule
//!   above;
//! - a head of three rows: row `c`'s logits and token are the one-row head's, bit for bit.

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::head::Head;
#[cfg(feature = "gpu")]
use bloomery_gpu::weights::Weights;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::block::{self, Bands, BlockKind, M_TOKENS, TapKind, TapResult};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::oracle::deepseek2::L_OUT_26;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, bits_equal, find_ref_row, ref_dir, ref_manifest, ref_model_path,
    ref_tensor_logical_in, verdict,
};
#[cfg(feature = "gpu")]
use gguf::Split;

/// The oracle's own greedy token for the prompt the dump sets were made with
/// (`gate_block`'s pin: both of ik's backends argmax here). Re-proved here so
/// a wrong dump set fails before any engine number is read.
#[cfg(feature = "gpu")]
const IK_ARGMAX: usize = 8913;

/// Structural fence — see the module doc. Not a band.
#[cfg(feature = "gpu")]
const FENCE: f32 = 0.25;

/// Head bands, PIN(2026-09-21) from this gate's first table (engine vs
/// `ref_cuda_v2`, fed the oracle's own `l_out-26`: result_norm 8.4e-8,
/// result_output 1.21e-2) and the two oracles' own distance on the logits
/// (ik CPU vs CUDA, `gate_block`: result_output 2.9e-2). Derivation as in
/// `gate_p8`: the larger of 2 x the measured rel and the oracle-pair
/// distance, rounded up to one digit. `result_norm` is exact arithmetic on an
/// identical input, so its band is 1e-6. The logits' 1.2e-2 is the q8_1
/// activation noise of one Q6_K gemv over 102400 rows (the lm_head's raw-x
/// rel measured 1.27e-2; the table is in `docs/gpu-design.md`).
#[cfg(feature = "gpu")]
const BANDS: [(TapKind, usize, f32); 2] = [
    (TapKind::ResultNorm, 27, 1e-6),
    (TapKind::ResultOutput, 27, 3e-2),
];

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_head_gpu", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let gpu = Gpu::new()?;
    let mut ok = nvfp4_head::run(&gpu)?;
    let file = Split::open(ref_model_path()?)?;
    let man = ref_manifest()?;
    let dir = ref_dir();

    // The head's input: the dump's l_out-26 — the last block's residual ADD,
    // last position only (the reference runs the head there alone). The same
    // contract the CPU head gate pins.
    let in_row = find_ref_row(&man, L_OUT_26, 0)?;
    if in_row.ty != "f32" || in_row.op != "ADD" || in_row.ne != [2048, 1, 1, 1] {
        return Err(format!(
            "gate_head_gpu: l_out-26/0 is {} {} {:?}, want f32 ADD [2048, 1]",
            in_row.ty, in_row.op, in_row.ne
        )
        .into());
    }
    let x_in = ref_tensor_logical_in(&dir, in_row)?;

    // The architecture-wide rms eps, read here independently of the caller
    // that hands it to Head (the CPU head gate's second-reader pattern).
    let eps = file
        .arch_get_f32("attention.layer_norm_rms_epsilon")
        .ok_or("gate_head_gpu: rms eps key missing or not f32")?;

    // Globals only: the head reads no block tensors and no derived weights,
    // so nothing is derived after the load.
    let w = Weights::load(gpu.stream(), &file, 0..0, true)?;
    let mut head = Head::new(&gpu, &w, eps)?;
    println!(
        "resident head_scratch_bytes={} hidden={} n_vocab={} m=1 input=l_out-26/0",
        head.resident_bytes(),
        head.hidden(),
        head.n_vocab()
    );

    // ---- eager run on the dump's last-position head input
    head.set_input(&gpu, &x_in)?;
    head.enqueue(&gpu, &w)?;
    let normed1 = head.normed_to_host(&gpu)?;
    let logits1 = head.logits_to_host(&gpu)?;
    let token1 = head.token(&gpu)? as usize;
    for (name, v) in [("result_norm", &normed1), ("result_output", &logits1)] {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            eprintln!("FAIL: {name} holds a non-finite value at {i}");
            ok = false;
        }
    }

    // ---- tap table against the oracle, inside the pinned bands
    let out_row = find_ref_row(&man, "result_output", 0)?;
    let oracle = ref_tensor_logical_in(&dir, out_row)?;
    if oracle.len() != head.n_vocab() {
        return Err(format!(
            "gate_head_gpu: result_output holds {} values, output.weight has {} rows",
            oracle.len(),
            head.n_vocab()
        )
        .into());
    }
    let mut results: Vec<TapResult> = Vec::new();
    for tap in block::taps(BlockKind::Head, 27) {
        let got: &[f32] = match tap.kind {
            TapKind::ResultNorm => &normed1,
            TapKind::ResultOutput => &logits1,
            _ => unreachable!("the head tap list holds only the two kinds above"),
        };
        results.push(block::compare_in(&dir, &man, &tap, got, M_TOKENS)?);
    }
    let bands = Bands::pinned(&BANDS);
    block::print_table(
        "head",
        "gpu_head",
        "ref_cuda[last_token]",
        &results,
        Some(&bands),
    );
    if let Err(v) = bands.assert_within(&results) {
        eprintln!("FAIL: {v}");
        ok = false;
    }
    for r in &results {
        if r.rel > FENCE {
            eprintln!(
                "FAIL: tap {} rel={:.3e} leaves the structural fence {FENCE}",
                r.tap.name(),
                r.rel
            );
            ok = false;
        }
    }

    // ---- (c) argmax: device token vs the oracle row vs our own logits,
    // one tie rule everywhere (equal values -> lower index).
    let oracle_am = argmax_low(&oracle);
    let host_am = argmax_low(&logits1);
    let am_ok = token1 == oracle_am && token1 == host_am && oracle_am == IK_ARGMAX;
    println!(
        "argmax device={token1} host={host_am} oracle={oracle_am} ik_pin={IK_ARGMAX} {}",
        verdict(am_ok)
    );
    if !am_ok {
        ok = false;
    }

    // ---- (b) eager rerun bit-identical
    head.set_input(&gpu, &x_in)?;
    head.enqueue(&gpu, &w)?;
    let rerun_same = bits_equal(&normed1, &head.normed_to_host(&gpu)?)
        && bits_equal(&logits1, &head.logits_to_host(&gpu)?)
        && token1 == head.token(&gpu)? as usize;
    println!(
        "rerun eager_bit_identical={rerun_same} {}",
        verdict(rerun_same)
    );
    if !rerun_same {
        ok = false;
    }

    // ---- for the record: top-5 of both sides
    println!("top-5 ours: {:?}", top5(&logits1));
    println!("top-5 oracle: {:?}", top5(&oracle));

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_head_gpu — head chain taps printed against the dump's last \
         token inside the pinned bands, rerun bit-identical, argmax equal to the \
         oracle's under one tie rule"
    );
    Ok(())
}

/// Index of the maximum, ties to the LOWER index — `elem::argmax_fault`'s rule
/// (`argmax_take`: strictly greater replaces, equal never does under an
/// ascending scan). Every argmax in this gate runs through it.
#[cfg(feature = "gpu")]
fn argmax_low(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Top-5 as (id, logit), sorted by (logit desc, id asc) — the same total
/// order the argmax walks, so the printed ranking cannot disagree with the
/// asserted token.
#[cfg(feature = "gpu")]
fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(5);
    idx.into_iter().map(|i| (i, v[i])).collect()
}

/// The `nvfp4_head_k2688` and `nvfp4_head_k5120` clauses (module doc).
#[cfg(feature = "gpu")]
mod nvfp4_head {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::head::Head;
    use bloomery_gpu::mxfp4::{Nvfp4, lane_partial_host};
    use bloomery_gpu::weights::{HEAD_TENSOR, Weights};
    use bloomery_gpu_gates::act_rule::{Q8_1_BLOCK, q8_1_codes};
    use bloomery_gpu_gates::rounding::{U64, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, GateRng, par_map, verdict, write_gguf};
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use model::arch::names::output_norm;

    /// Lightning's hidden width and a vocabulary of a few blocks of rows.
    const K: usize = 2688;
    /// Qwen3.8-27B's hidden width: whole q8_1 super-blocks of 256 values, so its head reads the q8_1
    /// rows through `site::kgemv`.
    const K_Q8_1: usize = 5120;
    const N_VOCAB: usize = 1000;
    /// Rows of the multi-row head.
    const M: usize = 3;
    /// The tensor scale, no power of two, and far enough from 1 that a logit without it leaves the
    /// band on nearly every row.
    const SCALE: f32 = 0.371_9;
    const EPS: f32 = 1e-6;
    /// The band of a logit about the f64 reference, a share of the row's own scale `scale ·
    /// sqrt(Σ (w_j x_j)²)`: the 32-value activation rounds each value by at most 1/254 of its block's
    /// largest magnitude, which keeps the dot's error near 0.4 % of that scale [derived]; 3 % is the
    /// Q5_K head test's band over the same kind of rounding.
    const NVFP4_BAND: f64 = 0.03;

    /// `rows` NVFP4 rows of `k` values drawn from `seed`: each block's four scale bytes uniform
    /// over the 127 UE4M3 bytes the format defines, its 32 code bytes uniform over all 256.
    fn nvfp4_rows(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let mut out = Vec::with_capacity(rows * k / 64 * 36);
        for _ in 0..rows * k / 64 {
            out.extend((0..4).map(|_| rng.below(0x7F) as u8));
            out.extend((0..32).map(|_| rng.below(256) as u8));
        }
        out
    }

    /// Row `c` of a head of `M` rows of `k` values: the inputs the one-row head is run on one at a
    /// time.
    fn input(k: usize, c: usize) -> Vec<f32> {
        (0..k)
            .map(|i| (((i + 7 * c) * 37 % 101) as f32 - 50.0 + c as f32) / 25.0)
            .collect()
    }

    /// A synthetic file holding an NVFP4 `output.weight` of `k` values a row from `seed`, its
    /// `.scale` and the final norm's gain, loaded: the weight's bytes and the weights.
    fn load(gpu: &Gpu, k: usize, seed: u64) -> Result<(Vec<u8>, Weights), GateError> {
        let rows = nvfp4_rows(N_VOCAB, k, seed);
        let gain: Vec<u8> = (0..k)
            .flat_map(|i| (0.5 + (i % 7) as f32 / 8.0).to_le_bytes())
            .collect();
        let decls = vec![
            (
                HEAD_TENSOR.to_string(),
                vec![k as u64, N_VOCAB as u64],
                GgmlType::NVFP4,
                rows.clone(),
            ),
            (
                "output.scale".to_string(),
                vec![1],
                GgmlType::F32,
                SCALE.to_le_bytes().to_vec(),
            ),
            (output_norm(), vec![k as u64], GgmlType::F32, gain),
        ];
        let dir = std::env::temp_dir().join(format!("bloomery-nvfp4-head-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("head.gguf");
        let loaded = write_gguf(&path, &[], &decls).and_then(|()| {
            let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            let w = Weights::load_where(gpu.stream(), &split, |n| {
                n == HEAD_TENSOR || n == output_norm()
            })?;
            Ok(w)
        });
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        let w = loaded?;
        cleanup?;
        Ok((rows, w))
    }

    /// One row through the head: its normed row, its logits and its token.
    fn one(
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, u32), GateError> {
        head.set_input(gpu, x)?;
        head.enqueue(gpu, w)?;
        let token = head.token(gpu)?;
        Ok((head.normed_to_host(gpu)?, head.logits_to_host(gpu)?, token))
    }

    /// The index of the logits' maximum under the kernel's tie rule: strictly greater replaces, so
    /// equal values keep the lower index.
    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0
    }

    /// A head of `M` rows of `k` values: row `c`'s logits and token are the one-row head's on that
    /// row, bit for bit (the gemv is token-major and the copy puts it in the argmax's layout).
    fn three_rows(gpu: &Gpu, w: &Weights, head: &mut Head, k: usize) -> Result<bool, GateError> {
        let inputs: Vec<Vec<f32>> = (0..M).map(|c| input(k, c)).collect();
        let mut multi = Head::with_m(gpu, w, EPS, M)?;
        multi.set_input(gpu, &inputs.concat())?;
        multi.enqueue(gpu, w)?;
        let got = multi.logits_to_host(gpu)?;
        let tokens = multi.tokens(gpu)?;
        let mut rows_equal = got.len() == M * N_VOCAB && tokens.len() == M;
        for (c, x) in inputs.iter().enumerate() {
            let (_, want, want_token) = one(gpu, w, head, x)?;
            rows_equal &= tokens.get(c) == Some(&want_token)
                && want
                    .iter()
                    .enumerate()
                    .all(|(v, w1)| got.get(v * M + c).map(|g| g.to_bits()) == Some(w1.to_bits()));
        }
        println!(
            "nvfp4 clause=nvfp4_head_k{k} rows={M} three_rows_are_the_one_row_heads_bit_for_bit={rows_equal} {}",
            verdict(rows_equal)
        );
        Ok(rows_equal)
    }

    pub(super) fn run(gpu: &Gpu) -> Result<bool, GateError> {
        let sel32 = sel32_head(gpu)?;
        let q8_1 = q8_1_head(gpu)?;
        Ok(sel32 && q8_1)
    }

    /// The `nvfp4_head_k2688` clause (module doc).
    fn sel32_head(gpu: &Gpu) -> Result<bool, GateError> {
        let (rows, w) = load(gpu, K, 0x5eed_0301)?;
        let mut head = Head::new(gpu, &w, EPS)?;
        println!(
            "nvfp4_head hidden={} n_vocab={} resident_bytes={}",
            head.hidden(),
            head.n_vocab(),
            head.resident_bytes()
        );
        let (normed, logits, token) = one(gpu, &w, &mut head, &input(K, 0))?;
        let row_bytes = K / 64 * 36;
        let per: Vec<(bool, f64)> = par_map(N_VOCAB, |r| {
            let mut wv = vec![0.0f32; K];
            dequant_row(
                GgmlType::NVFP4,
                &rows[r * row_bytes..(r + 1) * row_bytes],
                &mut wv,
            )
            .unwrap_or_else(|e| panic!("row {r}: {e}"));
            let terms = wv
                .iter()
                .zip(&normed)
                .map(|(&a, &b)| f64::from(a) * f64::from(b));
            let scale = f64::from(SCALE);
            let want = scale * terms.clone().sum::<f64>();
            let own = scale * terms.map(|t| t * t).sum::<f64>().sqrt();
            let err = (f64::from(logits[r]) - want).abs();
            (err <= NVFP4_BAND * own + 1e-6, err / own.max(1e-30))
        });
        let in_band = per.iter().all(|p| p.0);
        let worst = per.iter().map(|p| p.1).fold(0.0f64, f64::max);
        let argmax_ok = token as usize == argmax(&logits);
        println!(
            "nvfp4 clause=nvfp4_head_k2688 rows={N_VOCAB} k={K} in_band={in_band} \
             worst_err_over_row_scale={worst:.3e} argmax_is_the_maximum={argmax_ok} {}",
            verdict(in_band && argmax_ok)
        );
        let rows_equal = three_rows(gpu, &w, &mut head, K)?;
        Ok(in_band && argmax_ok && rows_equal)
    }

    /// The `nvfp4_head_k5120` clause (module doc). The band is derived per row, not a share of the
    /// row's scale: each activation value is off its q8_1 value by at most half its 128-block's
    /// scale `d`, so the logit moves by at most `scale · Σ |w_j| d_j / 2`, and the gemv's own
    /// roundings are `gate_nvfp4`'s `γ(2 · ⌈K/1024⌉ + 6) + K/16 · u64` of `scale · Σ |w_j x̃_j|`,
    /// with `|x̃_j|` at most `|x_j| + d_j / 2`.
    fn q8_1_head(gpu: &Gpu) -> Result<bool, GateError> {
        const K: usize = K_Q8_1;
        let (rows, w) = load(gpu, K, 0x5eed_0302)?;
        let mut head = Head::new(gpu, &w, EPS)?;
        let (normed, logits, token) = one(gpu, &w, &mut head, &input(K, 0))?;
        let (q, d) = q8_1_codes(&normed);
        let row_bytes = K / 64 * 36;
        let scale = f64::from(SCALE);
        let rounding = gamma(2 * (K / 32).div_ceil(32) + 6) + (K / 16) as f64 * U64;
        let per: Vec<(bool, f64)> = par_map(N_VOCAB, |r| {
            let row = &rows[r * row_bytes..][..row_bytes];
            let mut wv = vec![0.0f32; K];
            dequant_row(GgmlType::NVFP4, row, &mut wv).unwrap_or_else(|e| panic!("row {r}: {e}"));
            let lanes: [f32; 32] =
                std::array::from_fn(|lane| lane_partial_host::<Nvfp4, 2>(row, &q, &d, lane));
            let bits = (butterfly(lanes) * SCALE).to_bits() == logits[r].to_bits();
            let (mut want, mut mag, mut step) = (0.0f64, 0.0f64, 0.0f64);
            for (j, (&a, &x)) in wv.iter().zip(&normed).enumerate() {
                let term = f64::from(a) * f64::from(x);
                want += term;
                mag += term.abs();
                step += f64::from(a).abs() * f64::from(d[j / Q8_1_BLOCK]);
            }
            let half = 0.5001 * step;
            let band = scale.abs() * (half + rounding * (mag + half)) + 1e-6;
            let err = (f64::from(logits[r]) - scale * want).abs();
            (bits, err / band)
        });
        let bits = per.iter().filter(|p| p.0).count();
        let worst = per.iter().map(|p| p.1).fold(0.0f64, f64::max);
        let argmax_ok = token as usize == argmax(&logits);
        let pass = bits == N_VOCAB && worst <= 1.0 && argmax_ok;
        println!(
            "nvfp4 clause=nvfp4_head_k5120 rows={N_VOCAB} k={K} bits_equal={bits}/{N_VOCAB} \
             worst_err_over_band={worst:.3e} argmax_is_the_maximum={argmax_ok} {}",
            verdict(pass)
        );
        let rows_equal = three_rows(gpu, &w, &mut head, K)?;
        Ok(pass && rows_equal)
    }
}
