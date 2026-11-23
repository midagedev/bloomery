//! The split of a fused w13 routed stack into the gate and up stacks the card
//! leg reads: a family whose file fuses the two stacks (`ffn_gate_up_exps`,
//! the converter's un-interleaved form — each expert's gate rows above its up
//! rows, the reference's graph splitting the product by row offset,
//! llama-graph.cpp `build_moe_ffn`'s merged gate·up path) uploads the fused
//! tensor once and splits it here at load. Any later fused-w13 family splits
//! through this one owner; the family's own code names the fused stem, the
//! architecture fact that forced it.
//!
//! The split is a load-time repack with no numeric change: element `(i, e)` of
//! the fused product is the half stack's element `(i, e)` bit for bit, and the
//! per-expert tensor scale multiplies each element after its dot over either
//! stack (llama-graph.cpp `build_lora_mm_id`'s `w_s` path), so the halves
//! share the one scale row — each carries the fused stack's values.
//!
//! Three resident forms split ([`split_w13`]): an NVFP4 stack's planes (the
//! type's own upload), a Q8_0 stack's two planes (its type's own upload too —
//! `CardFormat::of` gives Q8_0 `Q8_0Planes`, so a plain [`crate::weights::
//! Weights::load`] holds it there), and a K-quant's words (a stack a plan
//! holds in the file's blocks, a Q8_0 one through [`crate::weights::Weights::
//! relayout`], whose routed `_sel` entries read words). The plane forms split
//! by one pitched device copy a plane — every expert's `rows_half` rows of its
//! `2 · rows_half` — the words by one copy. The copies are asynchronous on the
//! caller's stream; [`crate::weights::Weights::split_fused_w13`] is the filing
//! wrapper, which waits for them before the fused stack's buffers are freed.

use crate::GpuError;
use crate::mxfp4_sel::Nvfp4Stack;
use crate::tensor::{DeviceTensor, Pitched, copy_row, dtod_2d};
use crate::weights::DevWeight;
use cuda_core::{CudaStream, DeviceCopy};

/// What the refusals of this file and its filing wrapper name.
pub(crate) const WHAT: &str = "Weights::split_fused_w13";

/// The derived name of a fused stack's gate half.
#[must_use]
pub fn gate_of(fused: &str) -> String {
    format!("derived.{fused}.gate")
}

/// The derived name of a fused stack's up half.
#[must_use]
pub fn up_of(fused: &str) -> String {
    format!("derived.{fused}.up")
}

/// `rows_half` rows of each of `experts` experts of the plane `p` — its
/// experts `per_expert` rows apart, the half's rows at `at_row` of each
/// expert — as one `experts · rows_half`-row plane: one pitched device copy,
/// asynchronous on `stream`. Refused by name before anything is allocated
/// when the bytes of a half, an expert or the copy's source offset pass
/// usize.
fn half_plane<T: DeviceCopy>(
    stream: &CudaStream,
    p: &DeviceTensor<T>,
    (experts, per_expert): (usize, usize),
    (rows_half, at_row): (usize, usize),
) -> Result<DeviceTensor<T>, GpuError> {
    let row = p.cols().checked_mul(size_of::<T>());
    let Some((width, pitch, at)) = row.and_then(|r| {
        Some((
            rows_half.checked_mul(r)?,
            per_expert.checked_mul(r)?,
            at_row.checked_mul(r)?,
        ))
    }) else {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a plane of {} rows of {} words: a half's bytes pass usize",
                p.rows(),
                p.cols()
            ),
        ));
    };
    let rows = experts.checked_mul(rows_half).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("{experts} experts of {rows_half} rows pass usize"),
        )
    })?;
    // SAFETY: the pitched copy below writes every byte of the tensor — its
    // `experts` rows of `width` bytes at the pitch of a contiguous half — on
    // `stream`; when it fails the tensor is dropped here unread.
    let mut half = unsafe { DeviceTensor::<T>::unfilled(stream, rows, p.cols())? };
    dtod_2d(
        stream,
        (p.buf(), at),
        (half.buf_mut(), 0),
        Pitched {
            width,
            rows: experts,
            dev_pitch: pitch,
            host_pitch: width,
        },
    )?;
    Ok(half)
}

/// The fused stack's per-expert scale values as one half's scale plane: one
/// row copy of the whole `experts`-value row, asynchronous on `stream`.
fn scale_row(
    stream: &CudaStream,
    scales: &DeviceTensor<f32>,
    experts: usize,
) -> Result<DeviceTensor<f32>, GpuError> {
    // SAFETY: the row copy below writes every byte of the tensor (`experts`
    // f32) on `stream`; when it fails the tensor is dropped here unread.
    let mut row = unsafe { DeviceTensor::<f32>::unfilled(stream, experts, 1)? };
    copy_row(stream, scales.buf(), row.buf_mut(), experts)?;
    Ok(row)
}

/// The experts `rows` rows hold as whole experts of `per_expert` rows, when
/// both planes of a two-plane stack agree on them; `None` when they do not
/// divide or disagree.
fn whole_experts((qs_rows, d_rows): (usize, usize), per_expert: usize) -> Option<usize> {
    let experts = qs_rows.checked_div(per_expert).filter(|&n| n > 0)?;
    (experts * per_expert == qs_rows && d_rows == qs_rows).then_some(experts)
}

/// The refusal for a plane pair that is not whole experts: its rows.
fn plane_rows(rows: usize, per_expert: usize) -> String {
    format!("{rows} rows are not whole experts of {per_expert}")
}

/// Split the resident fused w13 stack `name` — every expert `2 · rows_half`
/// rows, its gate rows `[0, rows_half)` above its up rows — into its gate and
/// up stacks of the same resident form: an NVFP4 stack's planes, each half
/// carrying the fused stack's per-expert scale values; a Q8_0 or an MXFP8
/// stack's two planes (the codes and the block scales, one pitched copy
/// each); or a
/// K-quant's words (its row bytes a whole number of words, as the words the
/// half's kernels address by byte offset need). Any other resident form is
/// refused by name, as is a stack that is not whole experts of that many
/// rows. The copies are asynchronous on `stream`: the caller keeps the fused
/// weight alive until they end ([`crate::weights::Weights::split_fused_w13`]
/// does).
pub fn split_w13(
    stream: &CudaStream,
    (w, name): (&DevWeight, &str),
    rows_half: usize,
) -> Result<(DevWeight, DevWeight), GpuError> {
    let refuse = |detail: String| GpuError::shape(WHAT, format!("tensor {name}: {detail}"));
    let Some(per_expert) = rows_half.checked_mul(2) else {
        return Err(refuse(format!("{rows_half} rows a half pass usize")));
    };
    if rows_half == 0 {
        return Err(refuse("a fused stack of no rows a half".to_string()));
    }
    match w {
        DevWeight::Nvfp4Stack { qs, e, scales, k } => {
            let stack = Nvfp4Stack { qs, e, scales };
            let experts = stack
                .experts(WHAT, *k, per_expert)
                .map_err(|err| match err {
                    GpuError::Shape { detail, .. } => refuse(detail),
                    other => other,
                })?;
            let plane =
                |p, at_row| half_plane(stream, p, (experts, per_expert), (rows_half, at_row));
            let gate = DevWeight::Nvfp4Stack {
                qs: plane(qs, 0)?,
                e: plane(e, 0)?,
                scales: scale_row(stream, scales, experts)?,
                k: *k,
            };
            let up = DevWeight::Nvfp4Stack {
                qs: plane(qs, rows_half)?,
                e: plane(e, rows_half)?,
                scales: scale_row(stream, scales, experts)?,
                k: *k,
            };
            Ok((gate, up))
        }
        DevWeight::Q8_0 { qs, d, k } => {
            let experts = whole_experts((qs.rows(), d.rows()), per_expert)
                .ok_or_else(|| refuse(plane_rows(qs.rows(), per_expert)))?;
            let geom = (experts, per_expert);
            let rows = (rows_half, 0);
            let gate = DevWeight::Q8_0 {
                qs: half_plane(stream, qs, geom, rows)?,
                d: half_plane(stream, d, geom, rows)?,
                k: *k,
            };
            let rows = (rows_half, rows_half);
            let up = DevWeight::Q8_0 {
                qs: half_plane(stream, qs, geom, rows)?,
                d: half_plane(stream, d, geom, rows)?,
                k: *k,
            };
            Ok((gate, up))
        }
        DevWeight::Mxfp8 { qs, e, k } => {
            let experts = whole_experts((qs.rows(), e.rows()), per_expert)
                .ok_or_else(|| refuse(plane_rows(qs.rows(), per_expert)))?;
            let geom = (experts, per_expert);
            let half = |at_row| -> Result<DevWeight, GpuError> {
                Ok(DevWeight::Mxfp8 {
                    qs: half_plane(stream, qs, geom, (rows_half, at_row))?,
                    e: half_plane(stream, e, geom, (rows_half, at_row))?,
                    k: *k,
                })
            };
            Ok((half(0)?, half(rows_half)?))
        }
        DevWeight::KQuant { ty, w, k } => {
            let (rows, cols) = (w.rows(), w.cols());
            let experts = rows
                .checked_div(per_expert)
                .filter(|&n| n > 0 && n * per_expert == rows)
                .ok_or_else(|| {
                    refuse(format!(
                        "{rows} rows of {cols} words are not whole experts of {per_expert}"
                    ))
                })?;
            let block = (*ty)
                .blck_size()
                .and_then(|b| usize::try_from(b).ok())
                .filter(|b| k.is_multiple_of(*b));
            let row_bytes = block
                .zip((*ty).type_size().and_then(|s| usize::try_from(s).ok()))
                .map(|(b, s)| s * (k / b))
                .filter(|bytes| bytes.is_multiple_of(4));
            if row_bytes.is_none() {
                return Err(refuse(format!(
                    "a {ty} stack of K={k}: its row bytes are not a whole number of words"
                )));
            }
            let words = |at_row| half_plane(stream, w, (experts, per_expert), (rows_half, at_row));
            Ok((
                DevWeight::KQuant {
                    ty: *ty,
                    w: words(0)?,
                    k: *k,
                },
                DevWeight::KQuant {
                    ty: *ty,
                    w: words(rows_half)?,
                    k: *k,
                },
            ))
        }
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a resident fused w13 stack (an NVFP4 or Q8_0 stack's planes, or a K-quant's words)",
        )),
    }
}
