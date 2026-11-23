//! The fold of a recurrent head's correction through the target's output matrix
//! ([`runtime::dspark::Fold`], Inkling 2's draft). The file's correction `hidden_out · e + b`
//! (`hidden × rank` and `hidden` values) runs through the target's output matrix on every row; the
//! load computes `W2 = output · hidden_out` (`n_head × rank`) and `c = output · b` once, so a row's
//! correction is one gemv over `W2` plus `c` ([`super::rnn`]'s anchor form).
//!
//! [`fold`] runs on the card over the target's matrix in place: `W2`'s columns are the matrix's
//! products with `hidden_out`'s columns, eight at a time through the matrix's row kernel
//! (token-major, so column `j` comes back as one run of `n_head` values), gathered into rows on the
//! host and quantized row by row to Q8_0 (`quantize_q8_0`, the engine's rule); `c` stays f32 on the
//! card. The rounding this adds over the unfolded rule is Q8_0's on `W2`: value `v` of a row's
//! correction moves by `Σ_r (q_vr − w_vr) · e_r`, of standard deviation `sqrt(Σ_r d_v,⌊r/32⌋² ·
//! e_r² / 12)` for the block scales `d` of row `v` ([`fold_sigma`]).

use std::sync::{Mutex, PoisonError};

use cuda_core::DeviceBuffer;
use gguf::quant::Q8Block;
use model::quant::quantize_q8_0;

use super::dense::upload_blocks;
use super::{Q8Rows, TargetHead};
use crate::q8f32::{Bf16GemvMcolArgs, GemvOut, Q8_0GemvMcolArgs};
use crate::{Gpu, GpuError};

const WHAT: &str = "dspark::fold";

/// Columns of one gemv: the row kernels take eight.
const COLS: usize = 8;

/// The folded correction on the card.
pub struct Folded {
    /// `W2`: one Q8_0 row of `rank` values per head id.
    pub w2: Q8Rows,
    /// `c`: one value per head id.
    pub c: DeviceBuffer<f32>,
}

/// `y = output · x` over `m` token-major columns of `hidden`, the matrix read in place.
fn product(
    gpu: &Gpu,
    output: &TargetHead,
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let s = gpu.stream();
    match output {
        TargetHead::Q8_0InPlace { qs, d } => gpu.q8f32().enqueue_q8_0_gemv_mcol(
            s,
            Q8_0GemvMcolArgs {
                qs: &qs.w,
                d: &d.w,
                x,
                m,
                out: GemvOut::TokenMajor,
                y,
            },
        ),
        TargetHead::Bf16InPlace(p) => gpu.q8f32().enqueue_bf16_gemv_mcol(
            s,
            Bf16GemvMcolArgs {
                w: &p.w,
                x,
                m,
                out: GemvOut::TokenMajor,
                y,
            },
        ),
        other => Err(GpuError::shape(
            WHAT,
            format!(
                "a fold through a {:?} matrix: the fold reads a Q8_0 or BF16 matrix in place",
                other.base()
            ),
        )),
    }
}

/// The matrix's rows: the head ids.
fn head_rows(output: &TargetHead) -> Result<usize, GpuError> {
    match output {
        TargetHead::Q8_0InPlace { qs, .. } => Ok(qs.n_rows()),
        TargetHead::Bf16InPlace(p) => Ok(p.n_rows()),
        other => Err(GpuError::shape(
            WHAT,
            format!(
                "a fold through a {:?} matrix: the fold reads a Q8_0 or BF16 matrix in place",
                other.base()
            ),
        )),
    }
}

/// `W2 = output · hidden_out` and `c = output · b` (module doc): `hidden_out` is `hidden` rows of
/// `rank` f32 (the file's `[hidden, rank]`), `b` its `hidden` bias values, `output` the target's
/// matrix in place over the head ids. Load-time only; blocks.
pub fn fold(
    gpu: &Gpu,
    output: &TargetHead,
    hidden_out: &[f32],
    b: &[f32],
    hidden: usize,
    rank: usize,
) -> Result<Folded, GpuError> {
    let n = head_rows(output)?;
    if rank == 0
        || !rank.is_multiple_of(32)
        || hidden == 0
        || hidden_out.len() != hidden * rank
        || b.len() != hidden
    {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "hidden_out of {} values and a bias of {}: {hidden} rows of a rank of {rank} (a \
                 multiple of 32) and {hidden} values",
                hidden_out.len(),
                b.len()
            ),
        ));
    }
    let s = gpu.stream();
    let mut x = DeviceBuffer::zeroed(s, COLS * hidden)?;
    let mut y = DeviceBuffer::zeroed(s, COLS * n)?;
    let mut w2 = vec![0.0f32; n * rank];
    let mut cols = vec![0.0f32; COLS * hidden];
    for r0 in (0..rank).step_by(COLS) {
        let m = COLS.min(rank - r0);
        for j in 0..m {
            for d in 0..hidden {
                cols[j * hidden + d] = hidden_out[d * rank + r0 + j];
            }
        }
        x.copy_from_host(s, &cols)?;
        product(gpu, output, &x, m, &mut y)?;
        let back = y.to_host_vec(s)?;
        for j in 0..m {
            for (v, &w) in back[j * n..(j + 1) * n].iter().enumerate() {
                w2[v * rank + r0 + j] = w;
            }
        }
    }
    let bias = DeviceBuffer::from_host(s, b)?;
    let mut c = DeviceBuffer::zeroed(s, n)?;
    product(gpu, output, &bias, 1, &mut c)?;
    Ok(Folded {
        w2: quantized(gpu, &w2, n, rank)?,
        c,
    })
}

/// `rows` rows of `k` f32 quantized to Q8_0 rows on the card, `quantize_q8_0`'s blocks on the
/// resident pool.
fn quantized(gpu: &Gpu, v: &[f32], rows: usize, k: usize) -> Result<Q8Rows, GpuError> {
    let parts: Mutex<Vec<(usize, Vec<Q8Block>)>> = Mutex::new(Vec::new());
    threads::pool().for_each_chunk(rows, |range| {
        let mut blocks = Vec::with_capacity(range.len() * (k / 32));
        for r in range.clone() {
            for blk in v[r * k..(r + 1) * k].as_chunks::<32>().0 {
                blocks.push(quantize_q8_0(blk));
            }
        }
        parts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((range.start, blocks));
    });
    let mut parts = parts.into_inner().unwrap_or_else(PoisonError::into_inner);
    parts.sort_by_key(|(first, _)| *first);
    let blocks: Vec<Q8Block> = parts.into_iter().flat_map(|(_, b)| b).collect();
    upload_blocks(gpu, &blocks, rows, k)
}

/// The standard deviation of value `v`'s correction under the fold's Q8_0 rounding (module doc):
/// `scales` are row `v`'s `rank / 32` block scales of `W2`, `e` the row's correction vector.
#[must_use]
pub fn fold_sigma(scales: &[f32], e: &[f32]) -> f64 {
    let var: f64 = e
        .iter()
        .enumerate()
        .map(|(r, &er)| {
            let d = f64::from(scales[r / 32]);
            d * d * f64::from(er) * f64::from(er) / 12.0
        })
        .sum();
    var.sqrt()
}
