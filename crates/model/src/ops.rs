//! The two primitives every module needs, owned here so three rounds do not write three
//! of them. Reference implementations: correctness first, `crates/q3k-cpu` holds the fast
//! path and stage 1 does not call it.
//!
//! Accumulation is f32 in ggml's own order (one row at a time, k ascending). That choice is
//! the reason the gates can be tight: a different order gives a different last bit, and a
//! gate at 1e-3 would hide a real error to leave room for it.

use crate::profile;
use gguf::{GgmlType, Gguf, TensorInfo, dequant_row, quantize_activations};
use std::cell::RefCell;
use std::ops::Range;
use std::sync::Mutex;
use std::time::Instant;

/// A 2-D activation block in ggml's layout: `ne0` is contiguous, `ne1` strides by `ne0`.
///
/// Verified against the oracle (`docs/oracle.md`): a one-token run's `inp_embd` equals the
/// first `ne0` floats of a two-token run, difference exactly 0. So token `t` lives at
/// `data[t * ne0 .. (t + 1) * ne0]`.
#[derive(Clone, PartialEq, Debug)]
pub struct Tensor2 {
    pub ne0: usize,
    pub ne1: usize,
    pub data: Vec<f32>,
}

impl Tensor2 {
    pub fn zeros(ne0: usize, ne1: usize) -> Self {
        Self {
            ne0,
            ne1,
            data: vec![0.0; ne0 * ne1],
        }
    }

    pub fn from_vec(ne0: usize, ne1: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            ne0 * ne1,
            "Tensor2 data length must be ne0 * ne1"
        );
        Self { ne0, ne1, data }
    }

    /// Column `i`, i.e. token `i`'s `ne0` contiguous values.
    pub fn col(&self, i: usize) -> &[f32] {
        &self.data[i * self.ne0..(i + 1) * self.ne0]
    }

    pub fn col_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.ne0..(i + 1) * self.ne0]
    }
}

/// An f32 tensor read straight out of the file — norm gains, router biases, anything
/// the quantizer left alone.
///
/// One owner for the byte walk: several call sites need it, and copies of a loop that
/// reads f32 little-endian are several places for an endianness or stride assumption
/// to drift apart.
pub fn f32_tensor(gguf: &Gguf, t: &TensorInfo) -> Result<Vec<f32>, crate::ModelError> {
    // Profiler hook (crate::profile): the byte walk itself. `rows` and `k` are 0
    // on purpose — a flat read has no contraction and no row structure; the
    // weight MB column is the work. Level 1 only.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes = gguf.data(t)?;
    let out = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if let Some(t_call) = t_call {
        let acc = profile::CallAcc::new();
        profile::record(
            "f32_tensor",
            GgmlType::F32,
            0,
            0,
            bytes.len() as u64,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    Ok(out)
}

/// RMS norm with a learned gain, per column.
///
/// ggml computes the mean of squares over the whole row, adds eps, and multiplies by the
/// reciprocal square root — `eps` is inside the sqrt, not added to it. Getting that wrong
/// is a ~1e-4 error that a loose gate would absorb.
pub fn rms_norm(x: &Tensor2, gain: &[f32], eps: f32) -> Tensor2 {
    // Profiler hook (crate::profile): level-1 call timer; fires twice per block
    // plus once in the head. No contraction here: `rows` counts the columns
    // normed, `k` the elements each column walks, and the only weight read is
    // the F32 gain.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    assert_eq!(gain.len(), x.ne0, "rms_norm gain must be ne0 long");
    let mut out = Tensor2::zeros(x.ne0, x.ne1);
    for t in 0..x.ne1 {
        let src = x.col(t);
        let mut sum = 0.0f32;
        for &v in src {
            sum += v * v;
        }
        let scale = 1.0f32 / (sum / x.ne0 as f32 + eps).sqrt();
        let dst = out.col_mut(t);
        for i in 0..x.ne0 {
            dst[i] = src[i] * scale * gain[i];
        }
    }
    if let Some(t_call) = t_call {
        let acc = profile::CallAcc::new();
        profile::record(
            "rms_norm",
            GgmlType::F32,
            x.ne1 as u64,
            (x.ne0 * x.ne1) as u64,
            gain.len() as u64 * 4,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    out
}

// Per-thread dequantized-weight-row scratch for `matmul_q`: rows run on the pool,
// so every worker needs its own buffer, and a thread-local grown to the largest
// `k` seen beats allocating per chunk at the call rates `matmul_q` runs at.
// (Plain comment, not a doc comment: `thread_local!` is a macro invocation and
// has nothing to attach a doc to.)
thread_local! {
    static ROW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// One pool chunk's finished bookkeeping: its row-range start (for the
/// lowest-row-first error precedence), its share of the level-2 stage timers,
/// and the first error it hit (if any — a quant error on the scalar path, a
/// `qdot` error on the fused one). The output values themselves go straight
/// into `out` through `out_ptr`; workers hand these small structs over through
/// one mutex take per chunk — the only shared mutable state on the parallel
/// path, and it is never touched from inside the row loop.
struct RowChunk {
    start: usize,
    acc: profile::CallAcc,
    err: Option<crate::ModelError>,
}

/// `matmul_q`'s activation columns in whichever representation the row loop will
/// consume: the scalar path dots the f32 round trip, the fused qdot path dots the
/// bytes `qdot::quantize_col` produced (`cb` per column). One enum, not a flag
/// plus two buffers, so the two paths cannot be handed each other's input.
enum QuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round-trip representation.
    F32(Vec<f32>),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, buf: Vec<u8> },
}

/// `matmul_q`'s output pointer, in the one form the pool closure can capture:
/// raw pointers are neither `Send` nor `Sync`, and the closure must be `Sync`.
/// The impls carry no new safety — they point at the argument documented at
/// the construction site (disjoint cells by the row split, published by the
/// pool's completion protocol), which is what makes sharing the pointer sound.
pub(crate) struct SharedOut(pub(crate) *mut f32);
// SAFETY: see the struct doc — the pointer is only dereferenced under the
// disjoint-cell and join-ordering argument at its construction site.
unsafe impl Send for SharedOut {}
// SAFETY: same argument; workers only write cells their own row range owns.
unsafe impl Sync for SharedOut {}

impl SharedOut {
    /// Write one output cell. A method on purpose: a bare `out_ptr.0` inside
    /// the pool closure would capture the *field* — a raw pointer, which is
    /// not `Sync` — while the call captures `&SharedOut`, whose `Sync` is the
    /// argument above.
    ///
    /// # Safety
    ///
    /// `idx` must name a cell inside the caller's own row range of the split
    /// `threads::chunks` produced — the construction-site comment owns the
    /// full argument.
    pub(crate) unsafe fn write(&self, idx: usize, v: f32) {
        // SAFETY: the caller guarantees the disjointness contract above.
        unsafe { *self.0.add(idx) = v };
    }
}

/// One distinct activation input's quantized-column buffer, in the one form
/// the quant pre-pass pool closure can capture: the [`SharedOut`] trick
/// again — raw pointers are neither `Send` nor `Sync`, and the closure must
/// be `Sync`. The impls carry no new safety; they point at the buffer
/// `quantize_distinct` owns for the whole call, and what makes sharing them
/// sound is the column-split argument at the dispatch site (disjoint cells,
/// published by the pool's completion protocol).
enum SharedQuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round-trip representation,
    /// `k` cells per column.
    F32(*mut f32),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, ptr: *mut u8 },
}
// SAFETY: see the enum doc — the pointer is only dereferenced under the
// disjoint-cell and join-ordering argument at its construction site.
unsafe impl Send for SharedQuantCols {}
// SAFETY: same argument; workers only write cells their own column range
// of the split owns.
unsafe impl Sync for SharedQuantCols {}

impl SharedQuantCols {
    /// Quantize one column into its cell of this buffer. These are the exact
    /// calls a serial pre-pass would make per column, untouched — the pool
    /// split changes WHO calls them, never the calls.
    ///
    /// # Safety
    ///
    /// `t` must name a column inside the caller's own sub-range of the
    /// column split, and `x_col` must be column `t` of the input this
    /// buffer was built for — the construction-site comment in
    /// `quantize_distinct` owns the full argument.
    unsafe fn quantize_into(&self, ty: GgmlType, k: usize, x_col: &[f32], t: usize) {
        match self {
            SharedQuantCols::F32(ptr) => {
                // SAFETY: the caller guarantees the disjoint-cell contract
                // above; the cell is `k` f32s, as allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * k), k) };
                quantize_activations(ty, x_col, cell);
            }
            SharedQuantCols::Bytes { cb, ptr } => {
                // SAFETY: same contract; the cell is `cb` bytes, as
                // allocated by `col_bytes`.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * cb), *cb) };
                qdot::quantize_col(ty, x_col, cell);
            }
        }
    }
}

/// `y = W · x` where `W` is a quantized 2-D tensor straight out of the file.
///
/// ggml's convention, kept verbatim: `W.dims == [k, n]` with `k` contiguous, so a row of
/// `W` is `k` long and the result is `n` long. `x` must be `[k, n_tokens]`.
///
/// This dequantizes a row at a time and drops it — stage 1 is about being right, and the
/// row buffer keeps the working set in L1 rather than materializing the whole matrix.
///
/// **Activations are quantized first, in the format this weight type implies** — Q8_K for
/// Q3_K, Q8_2_X4 for Q4_K/Q5_K/Q6_K/Q5_0/Q5_1, none for F32. That is what ggml does before
/// a quantized dot, and the oracle is ggml's output: an f32 reference, and the *wrong*
/// quantized format, are both far enough away to force every gate below open.
/// `gguf::activation_format` owns the table.
///
/// **A quantized type with a fused kernel takes the fused path** (`crates/qdot`):
/// Q3_K/Q4_K/Q6_K for k a multiple of 256, Q5_0/Q5_1 for k a multiple of 32
/// (`qdot::k_granularity` owns the per-type contract). The dot runs straight off the
/// quantized codes and no f32 weight row is ever materialized. That kernel is *more*
/// accurate than this crate's dequant-then-round-trip f32 dot, so its output is
/// deliberately NOT bit-identical to the scalar path — gates that cross such a matmul
/// assert closeness to the oracle, never bit equality with the scalar path. Every other
/// type, and rows whose k breaks the type's block contract, keep the scalar path
/// unchanged.
pub fn matmul_q(gguf: &Gguf, w: &TensorInfo, x: &Tensor2) -> Result<Tensor2, crate::ModelError> {
    let mut outs = matmul_q_multi("matmul_q", gguf, &[w], &[x])?;
    Ok(outs.pop().expect("one pair in, one tensor out"))
}

/// The batched sibling of [`matmul_q`]: `y_i = W_i · x_i` for a list of
/// same-`(k, n)` weight views, in **one** pool dispatch over the concatenated
/// row space (plus the quant pre-pass's own). The MoE expert loop is the
/// caller this exists for: it interleaves a layer's gate+up views in one call
/// and its downs in another. Rows still split only on the output row of one
/// pair — the bit-identity argument of [`matmul_q`] is unchanged, and
/// `tests/ops.rs` holds batched ≡ sequential as a byte compare. Pairs may
/// share an activation block by reference (`x_i == x_j`): each DISTINCT
/// block is quantized once and the sharers read the same bytes —
/// deterministic encoders make that identical to quantizing per pair.
///
/// Every `W_i` must agree on dims and type (they are slices of one expert
/// stack); the `x_i` may differ in token count (buckets do).
pub fn matmul_q_batch(
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    matmul_q_multi("matmul_q_batch", gguf, ws, xs)
}

/// The single owner of the row computation both dispatch shapes ride: one
/// (weight view, activation block) pair plus everything a row of it needs.
/// A duplicated row loop is how the single and batched paths would drift
/// apart bit for bit, so the loop exists once, here.
struct PairWork<'a> {
    bytes: &'a [u8],
    row_bytes: usize,
    /// Fused qdot columns: (stride `cb`, quantized byte buffer).
    fused_cols: Option<(usize, &'a [u8])>,
    /// Scalar-path f32 round trip; empty on the fused path.
    q_f32: &'a [f32],
    ne1: usize,
    out: SharedOut,
}

impl PairWork<'_> {
    /// Rows `rows` (local row ids of this pair) across every token column,
    /// written straight into the pair's output at cells `t * n + r`. The
    /// scalar path takes the `ROW_BUF` borrow once per call, not per row —
    /// a `RefCell` borrow per row would put the borrow-check on the row loop.
    fn compute_rows(
        &self,
        ty: GgmlType,
        k: usize,
        n: usize,
        rows: Range<usize>,
        lvl: u8,
        acc: &mut profile::CallAcc,
    ) -> Result<(), crate::ModelError> {
        if let Some((cb, acol)) = self.fused_cols {
            // Fused rows: `qdot::dot_row` consumes the weight bytes and the
            // quantized column directly, so no f32 row is materialized and
            // ROW_BUF stays untouched. `add_dequant_w` staying 0 here is by
            // design — the dequant is fused into the dot, so the dot timer
            // covers the whole row, tokens included.
            for r in rows {
                let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                for t in 0..self.ne1 {
                    let v = qdot::dot_row(ty, src, &acol[t * cb..(t + 1) * cb], k)?;
                    // SAFETY: cell `t * n + r` belongs to this chunk's row
                    // range alone — see the SharedOut construction site.
                    unsafe { self.out.write(t * n + r, v) };
                }
                if let Some(t_dot) = t_dot {
                    acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                }
            }
            Ok(())
        } else {
            ROW_BUF.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if scratch.len() < k {
                    scratch.resize(k, 0.0);
                }
                let row = &mut scratch[..k];
                for r in rows {
                    let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                    // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                    let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
                    if ty == GgmlType::F32 {
                        for (i, v) in row.iter_mut().enumerate() {
                            *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
                        }
                    } else {
                        dequant_row(ty, src, row)?;
                    }
                    if let Some(t_d) = t_d {
                        acc.add_dequant_w(t_d.elapsed().as_nanos() as u64);
                    }
                    let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                    for t in 0..self.ne1 {
                        let xc = &self.q_f32[t * k..(t + 1) * k];
                        let mut a = 0.0f32;
                        for i in 0..k {
                            a += row[i] * xc[i];
                        }
                        // SAFETY: cell `t * n + r` belongs to this chunk's row
                        // range alone — see the SharedOut construction site.
                        unsafe { self.out.write(t * n + r, a) };
                    }
                    if let Some(t_dot) = t_dot {
                        acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                    }
                }
                Ok(())
            })
        }
    }
}

/// Shape contract of a batch: the two lists must be the same length, every
/// `W_i` must agree on `k`, `n` and type (the views are slices of one expert
/// stack), and every input must be `k` wide. `Ok(None)` is the empty batch —
/// not an error; the caller returns no outputs.
fn batch_shape(
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Option<(usize, usize, GgmlType)>, crate::ModelError> {
    if ws.len() != xs.len() {
        return Err(crate::ModelError::Shape {
            what: "matmul_q batch",
            want_ne0: ws.len(),
            want_ne1: ws.len(),
            got_ne0: xs.len(),
            got_ne1: xs.len(),
        });
    }
    let Some(w0) = ws.first() else {
        return Ok(None);
    };
    let k = w0.dims[0] as usize;
    let n = if w0.dims.len() > 1 {
        w0.dims[1] as usize
    } else {
        1
    };
    let ty = w0.ty;
    for (w, x) in ws.iter().zip(xs) {
        let wk = w.dims.first().copied().unwrap_or(0) as usize;
        let wn = if w.dims.len() > 1 {
            w.dims[1] as usize
        } else {
            1
        };
        if wk != k || wn != n || w.ty != ty {
            return Err(crate::ModelError::Shape {
                what: "matmul_q batch",
                want_ne0: k,
                want_ne1: n,
                got_ne0: wk,
                got_ne1: wn,
            });
        }
        if x.ne0 != k {
            return Err(crate::ModelError::Shape {
                what: "matmul_q input",
                want_ne0: k,
                want_ne1: x.ne1,
                got_ne0: x.ne0,
                got_ne1: x.ne1,
            });
        }
    }
    Ok(Some((k, n, ty)))
}

/// Map each pair's input to its slot in the distinct-input table. Identity is
/// reference identity (`std::ptr::eq` on the `&Tensor2`): the MoE gate/up
/// batch interleaves two weight views over ONE `xb`, and two distinct blocks
/// cannot report `ptr::eq` by accident. The quantizers are pure functions of
/// the column, so one quantization shared by both consumers is byte-identical
/// to two; a miss is only a lost optimization, never a wrong byte.
fn dedupe_inputs<'a>(xs: &'a [&'a Tensor2]) -> (Vec<usize>, Vec<&'a Tensor2>) {
    let mut slot_of: Vec<usize> = Vec::with_capacity(xs.len());
    let mut distinct: Vec<&Tensor2> = Vec::with_capacity(xs.len());
    for x in xs {
        match distinct.iter().position(|d| std::ptr::eq(*d, *x)) {
            Some(s) => slot_of.push(s),
            None => {
                slot_of.push(distinct.len());
                distinct.push(x);
            }
        }
    }
    (slot_of, distinct)
}

/// Quantize every activation column once per DISTINCT input, on the resident
/// pool, split over the concatenated column space. WHICH format is a property
/// of the WEIGHT type, not a global choice — `gguf::activation_format` owns
/// the table; the fused path quantizes into `qdot::quantize_col`'s byte
/// layout, the scalar path into the f32 round trip.
///
/// Bit identity: each column's output depends only on that column's input
/// values through a deterministic encoder, and the split partitions the output
/// cells, so which worker quantizes which column cannot change a byte.
/// Splitting WITHIN a column is not done and would need its own argument
/// (block-local scales).
///
/// SAFETY (construction site, referenced by every write): the raw pointers
/// alias the buffers `quantized` owns for the whole call. Every participant
/// quantizes only columns inside its own contiguous sub-range of the split,
/// and the split partitions the concatenated column space of the distinct
/// inputs, so no two participants ever write the same cell. The pool's
/// completion protocol publishes the writes to this thread before
/// `for_each_chunk` returns, and the buffers are not read until the PairWork
/// build, after the join. A column straddling a chunk boundary is walked as
/// two segments, never quantized twice: the walker hands each segment the
/// intersection of the chunk and one input, exactly the row walker's
/// pair-boundary rule.
fn quantize_distinct(
    ty: GgmlType,
    k: usize,
    distinct: &[&Tensor2],
    fused: bool,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> Vec<QuantCols> {
    let cb = if fused {
        Some(qdot::col_bytes(ty, k))
    } else {
        None
    };
    let mut quantized: Vec<QuantCols> = distinct
        .iter()
        .map(|x| match cb {
            Some(cb) => QuantCols::Bytes {
                cb,
                buf: vec![0u8; x.ne1 * cb],
            },
            None => QuantCols::F32(vec![0.0f32; x.data.len()]),
        })
        .collect();
    let mut col_starts: Vec<usize> = Vec::with_capacity(distinct.len());
    let mut total_cols = 0usize;
    for x in distinct {
        col_starts.push(total_cols);
        total_cols += x.ne1;
    }
    let shared: Vec<SharedQuantCols> = quantized
        .iter_mut()
        .map(|q| match q {
            QuantCols::Bytes { cb, buf } => SharedQuantCols::Bytes {
                cb: *cb,
                ptr: buf.as_mut_ptr(),
            },
            QuantCols::F32(q) => SharedQuantCols::F32(q.as_mut_ptr()),
        })
        .collect();
    let collected: Mutex<Vec<profile::CallAcc>> =
        Mutex::new(Vec::with_capacity(threads::pool().threads()));
    let quant_pass = |cols: Range<usize>| {
        let mut acc = profile::CallAcc::new();
        let mut c = cols.start;
        while c < cols.end {
            // The input whose column range contains `c`. `col_starts` is
            // sorted, so this is a partition point; a zero-width input can
            // never be selected — its start equals its successor's, and the
            // partition point lands on the LAST start `<= c`, so the walk
            // always takes a nonempty segment and always advances.
            let d = col_starts.partition_point(|&s| s <= c) - 1;
            let t0 = c - col_starts[d];
            let t_end = t0 + (cols.end - c).min(distinct[d].ne1 - t0);
            let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
            for t in t0..t_end {
                // SAFETY: column `t` of input `d` is inside this chunk's
                // sub-range of the column split — see the construction-site
                // comment above.
                unsafe { shared[d].quantize_into(ty, k, distinct[d].col(t), t) };
            }
            if let Some(t_q) = t_q {
                acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
            }
            c = col_starts[d] + t_end;
        }
        // The one lock of the chunk — never per column, for the same reason
        // the row walker locks once: a mutex inside the column loop would
        // profile the mutex.
        collected
            .lock()
            .expect("matmul_q quant chunk collector")
            .push(acc);
    };
    if total_cols > 1 {
        threads::pool().for_each_chunk(total_cols, quant_pass);
    } else {
        // One column (or none): there is no split to make, and waking pool
        // workers to hand one of them a single column would pay the dispatch
        // for nothing. The same closure runs inline on the caller —
        // `quantize_into` is deterministic, so the bytes are the bytes the
        // pool would have produced; `tests/mt.rs`'s cross-thread byte compare
        // covers this branch too, because `total_cols` does not depend on the
        // thread count.
        quant_pass(0..total_cols);
    }
    for acc in collected
        .into_inner()
        .expect("matmul_q quant chunk collector")
    {
        pacc.add_acc(&acc);
    }
    quantized
}

/// One `PairWork` per pair: its weight bytes, its (possibly shared) quantized
/// columns in the representation the row loop consumes, and a shared pointer
/// to its output. Pairs that share an activation block (the MoE gate/up
/// interleave) share one quantized buffer — read-only sharing from here on;
/// the quant pre-pass was the only writer and it is done.
fn build_pair_work<'a>(
    outs: &'a mut [Tensor2],
    bytes: &'a [&'a [u8]],
    quantized: &'a [QuantCols],
    slot_of: &[usize],
    row_bytes: usize,
) -> Vec<PairWork<'a>> {
    outs.iter_mut()
        .zip(bytes)
        .zip(slot_of.iter().map(|s| &quantized[*s]))
        .map(|((out, bytes), q)| {
            // SAFETY (construction site, referenced by every write): the raw
            // pointer aliases this pair's `out.data`, owned by the caller for
            // the whole call. Every participant writes only cells `t * n + r`
            // with `r` inside its own sub-range of the row split, and the
            // split partitions the rows of every pair, so no two participants
            // ever write the same cell. The pool's completion protocol
            // publishes the writes to this thread before `for_each_chunk`
            // returns. The outputs are not read until after the join, and a
            // chunk that errors early leaves its unwritten cells at the zeros
            // `Tensor2::zeros` gave them — the error discards the output
            // either way.
            let out_ptr = SharedOut(out.data.as_mut_ptr());
            let (fused_cols, q_f32) = match q {
                QuantCols::Bytes { cb, buf } => (Some((*cb, buf.as_slice())), &[][..]),
                QuantCols::F32(q) => (None, q.as_slice()),
            };
            PairWork {
                bytes,
                row_bytes,
                fused_cols,
                q_f32,
                ne1: out.ne1,
                out: out_ptr,
            }
        })
        .collect()
}

/// The row dispatch and its gather: one pool job over `Σ_i n` rows (the rows
/// of one pair at a time — a chunk may straddle pair boundaries, and the
/// walker hands each pair its own sub-range so no row ever belongs to two
/// pairs), then the sort, the error scan and the accumulator fold.
///
/// The split axis is the output row `r` and only that: each output element
/// still accumulates its k products ascending, in the order it always did, so
/// which worker computes which row cannot change a bit — `tests/mt.rs` holds
/// that as a byte compare across thread counts and `tests/ops.rs` holds
/// batched ≡ sequential. Splitting k (or the token axis) would reorder the
/// accumulation and is forbidden.
fn run_row_pool(
    pairs: &[PairWork<'_>],
    ty: GgmlType,
    k: usize,
    n: usize,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> Result<(), crate::ModelError> {
    let total_rows = n * pairs.len();
    let collected: Mutex<Vec<RowChunk>> = Mutex::new(Vec::with_capacity(threads::pool().threads()));
    threads::pool().for_each_chunk(total_rows, |rows| {
        let mut chunk = RowChunk {
            start: rows.start,
            acc: profile::CallAcc::new(),
            err: None,
        };
        let mut gr = rows.start;
        while gr < rows.end {
            let p = gr / n;
            let sub_end = rows.end.min((p + 1) * n);
            let r0 = gr - p * n;
            if let Err(e) =
                pairs[p].compute_rows(ty, k, n, r0..r0 + (sub_end - gr), lvl, &mut chunk.acc)
            {
                chunk.err = Some(e);
                break;
            }
            gr = sub_end;
        }
        // The one lock of the chunk — never per row: a mutex inside the row loop
        // would profile the mutex.
        collected
            .lock()
            .expect("matmul_q row-chunk collector")
            .push(chunk);
    });

    // Arrival order is nondeterministic; row order is not. Sorting by `start`
    // keeps the first error the lowest failing row's — the same error a
    // sequential `?` would have returned (a quant error on the scalar path, a
    // `qdot` error on the fused one) — and the caller's `record` is skipped on
    // that path exactly as it was before. Level 2 times this bookkeeping as
    // the `gather` stage.
    let t_gather = if lvl >= 2 { Some(Instant::now()) } else { None };
    let mut chunks = collected
        .into_inner()
        .expect("matmul_q row-chunk collector");
    chunks.sort_by_key(|c| c.start);
    if let Some(e) = chunks.iter_mut().find_map(|c| c.err.take()) {
        return Err(e);
    }
    for c in &chunks {
        pacc.add_acc(&c.acc);
    }
    if let Some(t_gather) = t_gather {
        pacc.add_gather(t_gather.elapsed().as_nanos() as u64);
    }
    Ok(())
}

/// One pool dispatch over `Σ_i n` rows for all `(W_i, x_i)` pairs. `site`
/// names the profiler row so the single and batched shapes stay separable in
/// the stage table — folding them under one name would hide the very count
/// the batch exists to change.
fn matmul_q_multi(
    site: &'static str,
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    let Some((k, n, ty)) = batch_shape(ws, xs)? else {
        return Ok(Vec::new());
    };
    // Profiler hook (crate::profile): level 0 is one compare per call, level 1
    // one `Instant` pair for the whole call, level 2 two more per row. None of
    // it touches the arithmetic — the gate proves that bit for bit. The row
    // and quant chunks each accumulate into one `CallAcc` per chunk, merged
    // after the join, so `profile::record` still fires exactly once per call.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes: Vec<&[u8]> = ws.iter().map(|w| gguf.data(w)).collect::<Result<_, _>>()?;
    let row_bytes = bytes[0].len() / n;
    // Fused path: qdot dots the quantized codes directly, so activations are
    // quantized into `qdot::quantize_col`'s layout. `k_granularity` owns the
    // per-type block contract (256 for the K-quants, 32 for Q5_0/Q5_1).
    // Decided once per batch: every pair shares `ty`, so no pair can disagree
    // with its own row loop.
    let fused = qdot::supports(ty) && k.is_multiple_of(qdot::k_granularity(ty));

    let (slot_of, distinct) = dedupe_inputs(xs);
    let quantized = quantize_distinct(ty, k, &distinct, fused, lvl, &mut pacc);

    let mut outs: Vec<Tensor2> = xs.iter().map(|x| Tensor2::zeros(n, x.ne1)).collect();
    let pairs = build_pair_work(&mut outs, &bytes, &quantized, &slot_of, row_bytes);
    run_row_pool(&pairs, ty, k, n, lvl, &mut pacc)?;

    let total_rows = n * pairs.len();
    let weight_bytes: u64 = bytes.iter().map(|b| b.len() as u64).sum();
    if let Some(t_call) = t_call {
        profile::record(
            site,
            ty,
            total_rows as u64,
            k as u64 * total_rows as u64,
            weight_bytes,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
    }
    Ok(outs)
}
