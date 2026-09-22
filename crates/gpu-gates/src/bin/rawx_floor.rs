//! Host-only probe: the q8_1 activation-quantization floor on real
//! activations. The kernel gates judge a gemv against a reference fed the
//! same q8_1-quantized activations the kernel consumes; judged against the
//! raw activations instead, even a bit-exact kernel sits at the floor this
//! probe prints, because Q(x) is not x. Each site pairs one intermediate
//! dumped by the reference engine ($BLOOMERY_DATA/ref_cuda, raw
//! little-endian f32, `<tensor>-<layer>.0.f32`) with the real weight rows of
//! the gemv that consumes it, and reports
//! `max|W.Q(x) - W.x| / max|W.x|` per token column, the max over columns,
//! and the global form of the same fraction (the gates' ungated raw-x
//! column). The synthetic generator's floor (`activations(k, 1, 1)`) sits
//! in the same table, and per-block `amax / rms` spread of the activations
//! (median and max over blocks) explains the gap: q8_1 spends its 127
//! levels on the block amax, so a spiky block quantizes its bulk relatively
//! worse. Arithmetic is the gates' own reference (`ref_gemv`: dequantize +
//! f64 dot); no device code, no timing, no assertions on the numbers — this
//! is a measurement, not a gate. Sites whose file length, tensor type or
//! K do not match the expected geometry are skipped with what was found
//! rather than measured on a guessed layout.

use std::path::Path;

use bloomery_gpu_gates::{
    DEFAULT_MODEL, activations, max_rel_err, open_model, ref_dir_named, ref_gemv, row_bytes,
    tensor_bytes,
};
use gguf::Gguf;
use gguf::quant::GgmlType;

type ProbeError = Box<dyn std::error::Error>;

/// One measurement site: the dumped intermediate `x_file`, the quantized
/// `tensor` whose gemv consumes it, and the q8_1 `block` size that gemv
/// quantizes activations with (128 for the K-quants, 32 for Q5_0/Q5_1).
struct Site {
    name: &'static str,
    layer: Option<u32>,
    x_file: String,
    tensor: String,
    ty: GgmlType,
    k: usize,
    block: usize,
    rows_cap: usize,
    note: &'static str,
}

/// The site list: layers 1 and 13 of every gemv site the kernel gates
/// cover, plus the lm head on the final-norm output. Expert stacks use
/// expert 0's rows only. `attn_output_post` measures the o-projection's
/// OUTPUT as activations (the raw tensor named `kqv_out`), kept beside
/// `attn_output` (its input `kqv_2d`) so both readings of the site exist;
/// the input row is the one that grounds the gemv floor.
fn sites() -> Vec<Site> {
    let mut v = Vec::new();
    for layer in [1u32, 13] {
        // (site, x file stem, tensor infix, type, K, q8 block, row cap, note)
        let per_layer: [(&str, &str, &str, GgmlType, usize, usize, usize, &str); 6] = [
            (
                "attn_q",
                "attn_norm",
                "attn_q",
                GgmlType::Q3_K,
                2048,
                128,
                2048,
                "",
            ),
            (
                "attn_output",
                "kqv_2d",
                "attn_output",
                GgmlType::Q4_K,
                2048,
                128,
                2048,
                "x is the o-projection's input: the reshaped attention output",
            ),
            (
                "attn_output_post",
                "kqv_out",
                "attn_output",
                GgmlType::Q4_K,
                2048,
                128,
                2048,
                "x is the o-projection's output (kqv_out), measured for comparison",
            ),
            (
                "moe_gate",
                "ffn_norm",
                "ffn_gate_exps",
                GgmlType::Q3_K,
                2048,
                128,
                2048,
                "",
            ),
            (
                "shexp_down",
                "ffn_up_gate",
                "ffn_down_shexp",
                GgmlType::Q4_K,
                2816,
                128,
                2048,
                "x is the fused shared-expert silu(gate)*up product, width = the down K",
            ),
            (
                "moe_down",
                "ffn_moe_gate_par",
                "ffn_down_exps",
                GgmlType::Q5_0,
                1408,
                32,
                2048,
                "x is tokens x experts x 1408; every 1408-value slice is one column",
            ),
        ];
        for (name, stem, infix, ty, k, block, cap, note) in per_layer {
            v.push(Site {
                name,
                layer: Some(layer),
                x_file: format!("{stem}-{layer}.0.f32"),
                tensor: format!("blk.{layer}.{infix}.weight"),
                ty,
                k,
                block,
                rows_cap: cap,
                note,
            });
        }
    }
    v.push(Site {
        name: "lm_head",
        layer: None,
        x_file: "result_norm.0.f32".to_string(),
        tensor: "output.weight".to_string(),
        ty: GgmlType::Q6_K,
        k: 2048,
        block: 128,
        rows_cap: 4096,
        note: "x is the final-norm output, one token (the last)",
    });
    v
}

fn main() -> Result<(), ProbeError> {
    let model = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    // The probe reads the pre-v2 plain-file set by name, not the environment:
    // its sites name `<tensor>-<layer>.0.f32` files that only that set carries.
    let dir = ref_dir_named("ref_cuda");
    let gguf = open_model()?;
    println!("rawx_floor: model {model}");
    println!("rawx_floor: activations dir {}", dir.display());
    println!();
    println!(
        "{:<18} {:>2} {:<26} {:<30} {:<4} {:>5} {:>4} {:>5} {:>5} {:>11} {:>5} {:>11} {:>11} {:>12}",
        "site",
        "L",
        "x-file",
        "weight",
        "type",
        "K",
        "blk",
        "rows",
        "cols",
        "real_floor",
        "worst",
        "global",
        "synth_floor",
        "synth_global"
    );
    for site in sites() {
        run_site(&gguf, &dir, &site)?;
    }
    println!();
    println!(
        "real_floor : max over token columns of max|W.Q(x) - W.x| / max|W.x| within the column"
    );
    println!(
        "global     : numerator and denominator over the whole output (the gates' raw-x column)"
    );
    println!("synth_*    : the same site with activations(K, 1, 1) as x");
    Ok(())
}

fn run_site(gguf: &Gguf, dir: &Path, s: &Site) -> Result<(), ProbeError> {
    let layer = s.layer.map_or_else(|| "-".to_string(), |l| l.to_string());
    let x_path = dir.join(&s.x_file);
    let x_bytes = match std::fs::read(&x_path) {
        Ok(b) => b,
        Err(e) => {
            println!(
                "{:<18} {:>2} SKIP: no x file at {} ({e})",
                s.name,
                layer,
                x_path.display()
            );
            return Ok(());
        }
    };
    if x_bytes.len() % (4 * s.k) != 0 {
        println!(
            "{:<18} {:>2} SKIP: {} is {} bytes, not a multiple of 4*{} — refusing to guess the layout",
            s.name,
            layer,
            s.x_file,
            x_bytes.len(),
            s.k
        );
        return Ok(());
    }
    let x: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let m = x.len() / s.k;

    let (info, bytes) = match tensor_bytes(gguf, &s.tensor) {
        Ok(v) => v,
        Err(e) => {
            println!("{:<18} {:>2} SKIP: tensor {}: {e}", s.name, layer, s.tensor);
            return Ok(());
        }
    };
    if info.ty != s.ty || info.dims.first().copied() != Some(s.k as u64) || info.dims.len() < 2 {
        println!(
            "{:<18} {:>2} SKIP: {} is {:?} dims {:?}, expected type {:?} with dims[0] = {}",
            s.name, layer, s.tensor, info.ty, info.dims, s.ty, s.k
        );
        return Ok(());
    }
    // dims[1] is the row count of expert 0 for the 3-D expert stacks (rows
    // are expert-major in the flat byte order), so the first `rows` raw rows
    // are expert 0's rows for those and just the leading rows for 2-D ones.
    let rows = usize::try_from(info.dims[1])?.min(s.rows_cap);
    let rb = row_bytes(s.ty, s.k)?;
    if bytes.len() < rb * rows {
        println!(
            "{:<18} {:>2} SKIP: {} holds {} bytes, fewer than {rows} rows of {rb}",
            s.name,
            layer,
            s.tensor,
            bytes.len()
        );
        return Ok(());
    }
    let w = &bytes[..rb * rows];

    // Real activations: the floor of a bit-exact kernel judged against x.
    let y_exact = ref_gemv(s.ty, w, s.k, rows, &x, m)?;
    let y_q = ref_gemv(s.ty, w, s.k, rows, &q8_1_quantize(&x, s.k, s.block), m)?;
    let cols = col_rel_errs(&y_q, &y_exact, rows, m)?;
    let global = max_rel_err(&y_q, &y_exact)?;
    let floor = cols.iter().copied().fold(0.0f32, f32::max);
    let worst = cols
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);

    // Synthetic activations through the same weights, for one table.
    let xs = activations(s.k, 1, 1);
    let ys_exact = ref_gemv(s.ty, w, s.k, rows, &xs, 1)?;
    let ys_q = ref_gemv(s.ty, w, s.k, rows, &q8_1_quantize(&xs, s.k, s.block), 1)?;
    let cols_s = col_rel_errs(&ys_q, &ys_exact, rows, 1)?;
    let global_s = max_rel_err(&ys_q, &ys_exact)?;

    println!(
        "{:<18} {:>2} {:<26} {:<30} {:<4} {:>5} {:>4} {:>5} {:>5} {:>11} {:>5} {:>11} {:>11} {:>12}",
        s.name,
        layer,
        s.x_file,
        s.tensor,
        format!("{:?}", s.ty),
        s.k,
        s.block,
        rows,
        m,
        format!("{floor:.3e}"),
        worst,
        format!("{global:.3e}"),
        format!("{:.3e}", cols_s[0]),
        format!("{global_s:.3e}")
    );
    println!("  real  per_col : {}", fmt_errs(&cols));
    println!("  synth per_col : {}", fmt_errs(&cols_s));
    let (r_med, r_max, r_n, r_zero) = block_spread(&x, s.block);
    let (s_med, s_max, s_n, s_zero) = block_spread(&xs, s.block);
    println!(
        "  blk amax/rms  : real median {r_med:.2} max {r_max:.2} ({r_n} blocks, {r_zero} zero-rms) | synth median {s_med:.2} max {s_max:.2} ({s_n} blocks, {s_zero} zero-rms)"
    );
    if !s.note.is_empty() {
        println!("  note: {}", s.note);
    }
    Ok(())
}

/// Host mirror of the device q8_1 quantizer's activation treatment, block
/// size as a parameter (128 for the K-quant sites, 32 for the Q5_0/Q5_1
/// sites): per block `d = amax/127` (1.0 for an all-zero block) and
/// `q = round(x/d)` clamped to ±127; the probe dots the reconstructed
/// `q*d` values. Same semantics as the gate binaries' host mirrors; kept
/// here because those live behind `#[cfg(feature = "gpu")]`.
fn q8_1_quantize(x: &[f32], k: usize, block: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for col in 0..x.len() / k {
        for b in 0..k / block {
            let vals = &x[col * k + b * block..][..block];
            let amax = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (o, &v) in out[col * k + b * block..][..block].iter_mut().zip(vals) {
                *o = (v / d).round().clamp(-127.0, 127.0) * d;
            }
        }
    }
    out
}

/// The `max_rel_err` metric restricted to one token column of the row-major
/// `m`-outputs-per-row layout `ref_gemv` produces.
fn col_rel_errs(
    y_q: &[f32],
    y_exact: &[f32],
    rows: usize,
    m: usize,
) -> Result<Vec<f32>, ProbeError> {
    let mut out = Vec::with_capacity(m);
    for c in 0..m {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for r in 0..rows {
            let i = r * m + c;
            num = num.max((y_q[i] - y_exact[i]).abs());
            den = den.max(y_exact[i].abs());
        }
        if den == 0.0 {
            return Err(format!("column {c}: exact reference is all zero").into());
        }
        out.push(num / den);
    }
    Ok(out)
}

/// (median, max, block count, zero-rms block count) of per-block
/// `amax / rms`, rms taken in f64, over every block of every column of `x`.
/// Zero-rms blocks are excluded from median/max and counted instead: the
/// quantizer maps them to zeros, so their ratio does not exist.
fn block_spread(x: &[f32], block: usize) -> (f64, f64, usize, usize) {
    let mut ratios = Vec::with_capacity(x.len() / block);
    let mut zeros = 0usize;
    for blk in x.chunks_exact(block) {
        let mut amax = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &v in blk {
            let a = f64::from(v.abs());
            amax = amax.max(a);
            sum_sq += a * a;
        }
        let rms = (sum_sq / block as f64).sqrt();
        if rms > 0.0 {
            ratios.push(amax / rms);
        } else {
            zeros += 1;
        }
    }
    ratios.sort_by(|a, b| a.total_cmp(b));
    let n = ratios.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, 0, zeros);
    }
    let median = if n % 2 == 1 {
        ratios[n / 2]
    } else {
        (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0
    };
    (median, ratios[n - 1], n, zeros)
}

fn fmt_errs(v: &[f32]) -> String {
    v.iter()
        .map(|e| format!("{e:.3e}"))
        .collect::<Vec<_>>()
        .join(" ")
}
