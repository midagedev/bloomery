//! The A4 clauses: MLX's affine 4-bit decode (`quant.metal`'s `A4` owner)
//! against the f64 oracle, over the MLX-format fixture split. The Q4_K
//! clauses' own shapes and fault inputs, every new entry: the five mat-vec
//! modes, the pair, the fold, the head and the embedding.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::{
    EXPERT_K, EXPERT_SLOTS, HEAD_COLS, HEAD_K, HEAD_VOCAB, PARTIALS_PER_HEAD, Quant, ROUTED_IDS,
    VALUE_HEADS,
};

use super::clauses_embed::run_embed;
use super::clauses_experts::{
    self, PairIn, check_fold_quant, check_pair_quant, ffn_norm, fold_bad_id, fold_weights_of,
    launch_pair,
};
use super::clauses_head;
use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::{Harness, HeadNormLaunch, Launch, ModeSpec, SENTINEL_BITS};

/// The store dot at the Q4_K clause's shapes and the head's tail rows.
pub fn a4_matvec(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let tensors = [
        common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
        common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048),
        common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
        common::Tensor::a4("blk.1.attn_q.weight", 2048, 8192),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x1A34 + i as u64, 1.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    // The head's last rows: a count no threadgroup divides, deep in the
    // 248320-row tensor, in the Q6_K clause's shape.
    let tail_rows = 1153;
    let head = fix.rows_at(
        "output.weight",
        gguf::quant::GgmlType::MLX_A4,
        HEAD_K,
        HEAD_VOCAB - tail_rows,
        tail_rows,
    )?;
    let x = oracle::activations(HEAD_K, 0x1A3A, 1.0);
    let worst = common::run_checked(
        h,
        &Launch {
            quant: Quant::A4,
            rows: tail_rows,
            cols: HEAD_K,
            weights: head,
            x: &x,
            y0: &common::sentinel_rows(tail_rows),
            mode: ModeSpec::Store,
        },
    )
    .map_err(|e| {
        format!(
            "output.weight rows {}..{HEAD_VOCAB}: {e}",
            HEAD_VOCAB - tail_rows
        )
    })?;
    parts.push(format!(
        "output.weight tail {tail_rows}x{HEAD_K} {worst:.3}"
    ));
    common::noted(parts)
}

/// Row counts no threadgroup divides: every row written, no word past the
/// last, in the Q4_K clause's counts.
pub fn a4_row_tail(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    let t = common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192);
    let all = t.bytes(fix)?;
    let row_bytes = 2048 / Quant::A4.values_per_block() * Quant::A4.block_bytes();
    let x = oracle::activations(2048, 0x1A60, 1.0);
    for rows in counts {
        common::run_checked(
            h,
            &Launch {
                quant: Quant::A4,
                rows,
                cols: 2048,
                weights: &all[..rows * row_bytes],
                x: &x,
                y0: &common::sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{} rows of {}: {e}", rows, t.label()))?;
    }
    Ok(Verdict::Noted(format!(
        "{} row counts, every row written, no word past the last",
        counts.len()
    )))
}

/// The norm on load and the residual epilogue, with the fault inputs the
/// Q4_K clause runs: a NaN input, an infinite one, a zero input with a zero
/// epsilon, each raising the norm site and writing NaN rows.
pub fn a4_norm(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let eps = fix.rms_eps()?;
    let cases = [
        (
            common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
            Some("blk.0.attn_norm.weight"),
        ),
        (common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048), None),
        (
            common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
            None,
        ),
        (
            common::Tensor::a4("output.weight", 2048, 4096),
            Some("output_norm.weight"),
        ),
    ];
    let mut parts = Vec::new();
    for (i, (t, gain_name)) in cases.iter().enumerate() {
        let gain = common::gain_for(fix, *gain_name, t.cols, 0x1AA1 + i as u64)?;
        let x = oracle::activations(t.cols, 0x1A40 + i as u64, 3.7);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }

    let t = common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048);
    let gain = common::gain_for(fix, None, t.cols, 0x1AB0)?;
    // A zero input norms to zero with a positive epsilon: not a fault.
    common::run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fix)?,
            x: &vec![0f32; t.cols],
            y0: &common::sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {eps}: {e}"))?;

    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(t.cols, 1, 1.0), eps),
        (
            "an infinite input",
            oracle::activations(t.cols, 2, 1.0),
            eps,
        ),
        ("a zero input and a zero epsilon", vec![0f32; t.cols], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[t.cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fix)?,
            x: &x,
            y0: &common::sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        })?;
        common::expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;

    // The residual epilogue: the store's rows plus the output's own.
    let tensors = [
        common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
        common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048),
        common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
    ];
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x1A50 + i as u64, 1.0);
        let resid = oracle::activations(t.rows, 0x1BE5 + i as u64, 2.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("resid {} {worst:.3}", t.label()));
    }
    common::noted(parts)
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone, in the Q4_K clause's layer shapes.
pub fn a4_seg(h: &Harness<'_>, fix: &Fixture, layer_norm: &str) -> Clause {
    let eps = fix.rms_eps()?;
    let cols = 2048;
    let groups = [
        vec![
            ("blk.1.attn_q.weight", 8192usize),
            ("blk.1.attn_k.weight", 512),
            ("blk.1.attn_v.weight", 512),
        ],
        vec![
            ("blk.0.attn_qkv.weight", 8192),
            ("blk.0.attn_gate.weight", 4096),
        ],
    ];
    let ty = gguf::quant::GgmlType::MLX_A4;
    let gain = fix.f32s(layer_norm, cols)?;
    let mut parts = Vec::new();
    for (i, tensors) in groups.iter().enumerate() {
        let x = oracle::activations(cols, 0x1E86 + i as u64, 1.0);
        let mut segments: Vec<(usize, &[u8])> = Vec::new();
        let mut rows = 0usize;
        for (name, n) in tensors {
            segments.push((*n, fix.rows(name, ty, cols, *n)?));
            rows += n;
        }
        let got = h.seg(&super::SegLaunch {
            quant: Quant::A4,
            cols,
            segments: segments.clone(),
            x: &x,
            y0: &common::sentinel_rows(rows),
            gain: &gain,
            eps,
            layer: LAYER,
        })?;
        if let Some(fault) = h.fault()? {
            return Err(format!("the segmented launch raised {fault}"));
        }
        let mut at = 0usize;
        for (j, (n, bytes)) in segments.iter().enumerate() {
            let alone = h.matvec(&Launch {
                quant: Quant::A4,
                rows: *n,
                cols,
                weights: bytes,
                x: &x,
                y0: &common::sentinel_rows(*n),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            })?;
            for (k, (a, b)) in got[at..at + n].iter().zip(alone.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "{} segment {j} row {k}: the segment read {a:?}, the tensor alone read {b:?}",
                        tensors[j].0
                    ));
                }
            }
            at += n;
        }
        parts.push(format!("{} {} segments", tensors[0].0, tensors.len()));
    }
    common::noted(parts)
}

/// The head-normed residual mat-vec: the gated numerator times the scale of
/// the head each group sits in, against the f64 oracle, with the NaN partials
/// and numerator the Q4_K clause runs.
pub fn a4_headnorm(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let eps = fix.rms_eps()?;
    let cols = HEAD_COLS * VALUE_HEADS;
    let t = common::Tensor::a4("blk.0.ssm_out.weight", cols, 2048);
    let weights = t.bytes(fix)?;
    let g = oracle::activations(cols, 0x1E88, 1.0);
    let partials = head_partials();
    let x = common::head_scaled(
        &g,
        &partials,
        VALUE_HEADS,
        HEAD_COLS,
        PARTIALS_PER_HEAD,
        eps,
    );
    let reference = oracle::reference(Quant::A4, weights, 2048, cols, &x)?;
    let resid = oracle::activations(2048, 0x1E89, 2.0);
    let got = h.headnorm(&HeadNormLaunch {
        quant: Quant::A4,
        rows: 2048,
        cols,
        weights,
        g: &g,
        partials: &partials,
        y0: &resid,
        eps,
        layer: LAYER,
    })?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let kind = Kind::HeadNorm(oracle::HeadGeometry {
        heads: VALUE_HEADS,
        cols_per_head: HEAD_COLS,
        partials_per_head: PARTIALS_PER_HEAD,
    });
    let band = oracle::band(Quant::A4, cols / Quant::A4.values_per_block(), kind);
    let worst = oracle::check(&got, &reference, band, kind, Some(&resid))?;
    let mut noted = format!("worst error over its band: {worst:.3}");

    // A head the norm cannot take: the gate's site is raised and the head is
    // NaN, not a zero head.
    let mut bad_partials = partials.clone();
    bad_partials[5 * PARTIALS_PER_HEAD + 3] = f32::INFINITY;
    let mut bad_g = g.clone();
    bad_g[17] = f32::NAN;
    for (what, g, partials) in [
        ("a partial that is not finite", &g, &bad_partials),
        ("a gated numerator that is not finite", &bad_g, &partials),
    ] {
        h.clear_fault()?;
        let got = h.headnorm(&HeadNormLaunch {
            quant: Quant::A4,
            rows: 2048,
            cols,
            weights,
            g,
            partials,
            y0: &resid,
            eps,
            layer: LAYER,
        })?;
        common::expect_fault(h, what, FaultSite::LinearGate)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;
    noted.push_str(
        "; a partial and a gated numerator that are not finite each raised the gate site, every row NaN",
    );
    Ok(Verdict::Noted(noted))
}

/// The heads' partial sums of the squares of their values: the Q4_K clause's.
fn head_partials() -> Vec<f32> {
    (0..VALUE_HEADS * PARTIALS_PER_HEAD)
        .map(|i| 0.5 + (i % 7) as f32 * 0.25)
        .collect()
}

/// The gate-up pair over the MLX stacks: the oracle band, an id past the
/// stack, and the norm's fault inputs.
pub fn a4_pair(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let worst = check_pair_quant(h, fix, Quant::A4)?;
    let weights = clauses_experts::pair_weights(fix, Quant::A4)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(2048, 0x1B1D, 1.0);
    let y_len = EXPERT_SLOTS * EXPERT_K;
    let ids: [i32; ROUTED_IDS] = [0, 1, 2, 3, 1, 2, 3, 0];

    // An id past the stack: that slot's rows NaN, the others the clean bits.
    h.clear_fault()?;
    let clean = launch_pair(
        h,
        PairIn {
            quant: Quant::A4,
            n_experts: 4,
            ids: &ids,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm: &norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); y_len],
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("the clean pair raised {f}"));
    }
    for bad in [-1, 4] {
        let mut bad_ids = ids;
        bad_ids[3] = bad;
        h.clear_fault()?;
        let got = launch_pair(
            h,
            PairIn {
                quant: Quant::A4,
                n_experts: 4,
                ids: &bad_ids,
                gate: &weights.gate,
                up: &weights.up,
                shared_gate: &weights.shared_gate,
                shared_up: &weights.shared_up,
                x: &x,
                norm: &norm,
                y0: &vec![1.0; y_len],
            },
        )?;
        common::expect_fault(h, &format!("id {bad}"), FaultSite::ExpertId)?;
        for slot in 0..EXPERT_SLOTS {
            let row = &got[slot * EXPERT_K..(slot + 1) * EXPERT_K];
            let clean_row = &clean[slot * EXPERT_K..(slot + 1) * EXPERT_K];
            if slot == 3 {
                if let Some(k) = row.iter().position(|v| !v.is_nan()) {
                    return Err(format!("id {bad} slot 3 row {k} is {:?}, not NaN", row[k]));
                }
            } else if let Some(k) = row
                .iter()
                .zip(clean_row)
                .position(|(g, c)| g.to_bits() != c.to_bits())
            {
                return Err(format!(
                    "id {bad} slot {slot} row {k} changed to {:?} (clean {:?})",
                    row[k], clean_row[k]
                ));
            }
        }
    }

    // The norm's fault inputs: NaN rows in all nine slots' rows.
    for (what, mut x, eps) in [
        (
            "a NaN input",
            oracle::activations(2048, 0x1A0, 1.0),
            norm.eps,
        ),
        (
            "an infinite input",
            oracle::activations(2048, 0x1A1, 1.0),
            norm.eps,
        ),
        ("a zero input and a zero epsilon", vec![0f32; 2048], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[2047] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = launch_pair(
            h,
            PairIn {
                quant: Quant::A4,
                n_experts: 4,
                ids: &ids,
                gate: &weights.gate,
                up: &weights.up,
                shared_gate: &weights.shared_gate,
                shared_up: &weights.shared_up,
                x: &x,
                norm: &clauses_experts::PairNorm {
                    gain: norm.gain.clone(),
                    eps,
                },
                y0: &vec![1.0; y_len],
            },
        )?;
        common::expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(k) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "{what}: slot {} row {} is {:?}, not NaN",
                k / EXPERT_K,
                k % EXPERT_K,
                got[k]
            ));
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; ids -1 and the stack length raised expert_id, \
         NaN in that slot only; the norm's fault inputs raised norm_quant, NaN in all nine \
         slots' rows"
    )))
}

/// The folded down over the MLX stacks: the oracle band, a row tail, and a
/// bad id's NaN rows.
pub fn a4_fold(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let worst = check_fold_quant(h, fix, Quant::A4, 2048)?;
    let tail = check_fold_quant(h, fix, Quant::A4, 2045)?;
    fold_bad_id(h, &fold_weights_of(fix, Quant::A4, 2048)?, Quant::A4)?;
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; tail 2045 rows: {tail}; a bad id writes NaN and \
         raises expert_id"
    )))
}

/// The head over the MLX table: the logits against the norm-on-load oracle
/// band, the id against the host argmax of the read-back logits, head and
/// tail rows.
pub fn a4_head(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let eps = fix.rms_eps()?;
    let gain = fix.f32s("output_norm.weight", HEAD_K)?;
    let samples = [
        ("output.weight head", 0usize, 32usize, 0x1EAD_u64),
        ("output.weight tail", HEAD_VOCAB - 17, 17, 0x1EAE),
    ];
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, at, rows, seed) in samples {
        let weights = fix.rows_at(
            "output.weight",
            gguf::quant::GgmlType::MLX_A4,
            HEAD_K,
            at,
            rows,
        )?;
        let x = oracle::activations(HEAD_K, seed, 0.5);
        let out = clauses_head::run(
            h,
            clauses_head::HeadRun {
                quant: Quant::A4,
                weights,
                x: &x,
                gain: &gain,
                rows,
                k: HEAD_K,
                eps,
            },
        )?;
        if let Some(f) = h.fault()? {
            return Err(format!("{name} raised {f}"));
        }
        if out.ticket != 0 {
            return Err(format!(
                "{name}: the ticket counter is {}, not 0",
                out.ticket
            ));
        }
        let xhat = oracle::rms_norm(&x, &gain, eps);
        let reference = oracle::reference(Quant::A4, weights, rows, HEAD_K, &xhat)?;
        let band = oracle::band(Quant::A4, HEAD_K / Quant::A4.values_per_block(), Kind::Norm);
        let ratio = oracle::check(&out.logits, &reference, band, Kind::Norm, None)
            .map_err(|e| format!("{name}: {e}"))?;
        let (_, want) = clauses_head::host_argmax(&out.logits)?;
        if out.id != want {
            return Err(format!(
                "{name}: step id is {}, the logits select {want}",
                out.id
            ));
        }
        worst = worst.max(ratio);
        notes.push(format!("{name} {rows}x{HEAD_K} {ratio:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; {}",
        notes.join(", ")
    )))
}

/// The embedding over the MLX table: the owner's value decode bit-equal to
/// the host reader, row by row, and the ids past the vocabulary.
pub fn a4_embed(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let (table, vocab, ty) = fix.table("token_embd.weight", HEAD_K)?;
    if ty != gguf::quant::GgmlType::MLX_A4 {
        return Err(format!("token_embd.weight is {ty:?}, not MLX_A4"));
    }
    let cols = HEAD_K;

    let good = [0u32, 1, 5, 4095, (vocab - 1) as u32];
    let rows = run_embed(h, Quant::A4, table, vocab, &good)?;
    common::expect_clean(h, "ids in range")?;
    for (t, &id) in good.iter().enumerate() {
        let want = oracle::table_row(Quant::A4, table, cols, id as usize)?;
        for (k, (g, w)) in rows[t * cols..(t + 1) * cols].iter().zip(&want).enumerate() {
            if g.to_bits() != w.to_bits() {
                return Err(format!(
                    "id {id} value {k}: got {g:?}, the reader has {w:?}"
                ));
            }
        }
    }

    let mixed = [7u32, vocab as u32, 9, u32::MAX, (vocab - 1) as u32];
    let rows = run_embed(h, Quant::A4, table, vocab, &mixed)?;
    common::expect_fault(h, "ids past the vocabulary", FaultSite::TokenId)?;
    h.clear_fault()?;
    for (t, &id) in mixed.iter().enumerate() {
        let row = &rows[t * cols..(t + 1) * cols];
        if (id as usize) < vocab {
            let want = oracle::table_row(Quant::A4, table, cols, id as usize)?;
            if let Some(k) = row
                .iter()
                .zip(&want)
                .position(|(g, w)| g.to_bits() != w.to_bits())
            {
                return Err(format!(
                    "id {id} beside a bad id, value {k}: got {:?}",
                    row[k]
                ));
            }
        } else if let Some(k) = row.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "id {id} (vocab {vocab}) value {k} is {:?}, not NaN",
                row[k]
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "{} ids bit-equal to the reader, ids {} and {} raised token_id with NaN rows",
        good.len(),
        vocab,
        u32::MAX
    )))
}

/// The clause table: one clause per entry, in the token's order.
pub fn table<'a>(
    h: &'a super::Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    let with = |f: fn(&super::Harness<'_>, &Fixture) -> Clause| {
        Box::new(move || match mlx {
            Ok(fix) => f(h, fix),
            Err(e) => Err(e.clone()),
        }) as super::ClauseRun<'a>
    };
    vec![
        ("a4_matvec", with(a4_matvec)),
        ("a4_row_tail", with(a4_row_tail)),
        ("a4_norm", with(a4_norm)),
        (
            "a4_seg",
            Box::new(move || match mlx {
                Ok(fix) => a4_seg(h, fix, "blk.1.attn_norm.weight"),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("a4_headnorm", with(a4_headnorm)),
        ("a4_pair", with(a4_pair)),
        ("a4_fold", with(a4_fold)),
        ("a4_head", with(a4_head)),
        ("a4_embed", with(a4_embed)),
    ]
}
