//! The two primitives every module needs, written once — reference
//! implementations, fast path in `crates/q3k-cpu`. Accumulation is f32 in
//! ggml's own order (one row at a time, k ascending): a different order moves
//! the last bit, which is what lets the gates stay tight.

use crate::profile;
use gguf::{GgmlType, Gguf, TensorInfo, dequant_row, quantize_activations};
use std::cell::RefCell;
use std::ops::Range;
use std::time::Instant;

/// A 2-D activation block in ggml's layout: `ne0` contiguous, `ne1` strides by
/// `ne0`; token `t` lives at `data[t*ne0..(t+1)*ne0]` (oracle, `docs/oracle.md`).
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

/// An f32 tensor straight from the file — norm gains, router biases, anything the quantizer left alone; the crate's one little-endian byte walk.
pub fn f32_tensor(gguf: &Gguf, t: &TensorInfo) -> Result<Vec<f32>, crate::ModelError> {
    // Profiler hook: a flat read — `rows`/`k` are 0; the weight MB is the work.
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

/// RMS norm with a learned gain, per column — ggml's form: mean of squares over the row, `eps` inside the sqrt (added to the mean, not the result).
pub fn rms_norm(x: &Tensor2, gain: &[f32], eps: f32) -> Tensor2 {
    // Profiler hook: `rows` = columns normed, `k` = elements each walks.
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

// Per-thread dequantized-row scratch for `matmul_q`: rows run on the pool, so every worker needs its own, grown to the largest `k` seen.
thread_local! {
    static ROW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// One pool chunk's bookkeeping: row-range start (lowest-row-first error
/// precedence), stage timers, first error (quant or `qdot`). Values go to
/// `out` via `out_ptr`; crosses threads in one mutex take per chunk.
struct RowChunk {
    start: usize,
    acc: profile::CallAcc,
    err: Option<crate::ModelError>,
}

/// `matmul_q`'s activation columns in the form the row loop consumes — f32 round trip (scalar) or `qdot::quantize_col` bytes (fused, `cb`/column); an enum so the paths cannot swap inputs.
enum QuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round-trip representation.
    F32(Vec<f32>),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, buf: Vec<u8> },
}

/// `matmul_q`'s output pointer in the one form the pool closure can capture
/// (raw pointers are neither `Send` nor `Sync`); the impls add no safety of
/// their own — see the construction site.
pub(crate) struct SharedOut(pub(crate) *mut f32);
// SAFETY: see the struct doc — disjoint cells and join ordering, argued at the construction site.
unsafe impl Send for SharedOut {}
// SAFETY: same argument; workers write only cells their own row range owns.
unsafe impl Sync for SharedOut {}

impl SharedOut {
    /// Write one output cell — a method so the call captures `&SharedOut` (`Sync`), not the raw pointer.
    ///
    /// # Safety
    ///
    /// `idx` must name a cell inside the caller's own row range of the split.
    pub(crate) unsafe fn write(&self, idx: usize, v: f32) {
        // SAFETY: the caller guarantees the disjointness contract above.
        unsafe { *self.0.add(idx) = v };
    }
}

/// [`SharedOut`] for a distinct input's quantized-column buffer; the impls add no safety of their own — see the `quantize_distinct` construction site.
enum SharedQuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round trip, `k` cells per column.
    F32(*mut f32),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, ptr: *mut u8 },
}
// SAFETY: see the enum doc — disjoint cells and join ordering, argued at the construction site.
unsafe impl Send for SharedQuantCols {}
// SAFETY: same argument; workers write only their own column range's cells.
unsafe impl Sync for SharedQuantCols {}

impl SharedQuantCols {
    /// Quantize one column into its cell — the exact call a serial pre-pass would make; the split changes WHO calls it, never the call.
    ///
    /// # Safety
    ///
    /// `t` inside the caller's sub-range of the split; `x_col` is column
    /// `t` of this buffer's input.
    unsafe fn quantize_into(&self, ty: GgmlType, k: usize, x_col: &[f32], t: usize) {
        match self {
            SharedQuantCols::F32(ptr) => {
                // SAFETY: the disjoint-cell contract above; the cell is `k` f32s, as allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * k), k) };
                quantize_activations(ty, x_col, cell);
            }
            SharedQuantCols::Bytes { cb, ptr } => {
                // SAFETY: same contract; the cell is `cb` bytes, as `col_bytes` allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * cb), *cb) };
                qdot::quantize_col(ty, x_col, cell);
            }
        }
    }
}

/// `y = W · x` for a quantized 2-D `W` straight from the file, ggml's
/// convention: `W.dims == [k, n]`, `k` contiguous; `x` must be `[k, n_tokens]`;
/// dequantizes one row at a time (the row buffer stays in L1).
///
/// Activations are quantized first into the format the weight type implies
/// (Q8_K for Q3_K, Q8_2_X4 for Q4_K/Q5_K/Q6_K/Q5_0/Q5_1, none for F32;
/// `gguf::activation_format` owns the table) — the oracle is ggml's output,
/// and the wrong format would force every gate open.
///
/// A quantized type with a fused kernel takes the fused path (`crates/qdot`):
/// k a multiple of `qdot::k_granularity(ty)` — 256 for Q3_K/Q4_K/Q6_K, 32
/// for Q5_0/Q5_1. The fused dot is deliberately NOT bit-identical to the
/// scalar path (it is more accurate than the dequant round trip) — gates
/// across it assert closeness to the oracle, never bit equality; all other
/// types keep the scalar path.
pub fn matmul_q(gguf: &Gguf, w: &TensorInfo, x: &Tensor2) -> Result<Tensor2, crate::ModelError> {
    let mut outs = matmul_q_multi("matmul_q", gguf, &[w], &[x])?;
    Ok(outs.pop().expect("one pair in, one tensor out"))
}

/// The batched sibling of [`matmul_q`]: `y_i = W_i · x_i` for same-`(k, n)`
/// weight views, one pool dispatch over the concatenated row space; rows split
/// only on the output row of one pair, so bit identity is unchanged
/// (`tests/ops.rs`: batched ≡ sequential). Pairs may share an activation
/// block — each DISTINCT block is quantized once (deterministic encoders);
/// `W_i` all agree on dims and type, `x_i` may differ in token count.
pub fn matmul_q_batch(
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    matmul_q_multi("matmul_q_batch", gguf, ws, xs)
}

/// One (weight view, activation block) pair plus everything a row needs; the
/// row loop exists once, here — a duplicate is how the shapes would drift bit for bit.
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
    /// Rows `rows` of this pair across every token column → cells `t * n + r`; ROW_BUF is borrowed once per call.
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
            // Fused rows: `qdot::dot_row` dots the weight bytes and the
            // quantized column directly — no f32 row, ROW_BUF untouched;
            // `add_dequant_w` stays 0 (the dequant is in the dot timer).
            for r in rows {
                let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                for t in 0..self.ne1 {
                    let v = qdot::dot_row(ty, src, &acol[t * cb..(t + 1) * cb], k)?;
                    // SAFETY: cell `t * n + r` is inside this chunk's row range — see the SharedOut construction site.
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
                        // SAFETY: cell `t * n + r` is inside this chunk's row range — see the SharedOut construction site.
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

/// Shape contract of a batch: equal-length lists; every `W_i` agrees on
/// `k`, `n`, type (one expert stack) and every input is `k` wide. `Ok(None)`
/// is the empty batch, not an error.
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

/// Each pair's input → its slot in the distinct-input table, by reference
/// identity (`std::ptr::eq`) — distinct blocks cannot collide. Quantizers are
/// pure functions of the column, so one shared quantization is byte-identical
/// to two; a miss only loses the optimization.
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

/// Quantize every activation column once per DISTINCT input on the pool,
/// split over the concatenated column space; the format is a property of the
/// WEIGHT type (`gguf::activation_format`): fused → `qdot::quantize_col`
/// bytes, scalar → the f32 round trip.
///
/// Bit identity: each column is a pure function of its own input, and the
/// split partitions the output cells; splitting WITHIN a column would need
/// its own argument (block-local scales).
///
/// SAFETY (construction site, referenced by every write): the pointers alias
/// buffers owned for the whole call; each participant writes only its own
/// sub-range's columns, the join publishes the writes, and a
/// boundary-straddling column is two segments, never quantized twice.
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
    let collected: profile::ChunkSlots<profile::CallAcc> = profile::ChunkSlots::new();
    let quant_pass = |cols: Range<usize>| {
        let mut acc = profile::CallAcc::new();
        let mut c = cols.start;
        while c < cols.end {
            // The input whose column range contains `c` — `col_starts` is
            // sorted, so a partition point landing on the LAST start `<= c`;
            // never a zero-width input, always advances.
            let d = col_starts.partition_point(|&s| s <= c) - 1;
            let t0 = c - col_starts[d];
            let t_end = t0 + (cols.end - c).min(distinct[d].ne1 - t0);
            let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
            for t in t0..t_end {
                // SAFETY: column `t` of input `d` is inside this chunk's sub-range — see the construction site above.
                unsafe { shared[d].quantize_into(ty, k, distinct[d].col(t), t) };
            }
            if let Some(t_q) = t_q {
                acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
            }
            c = col_starts[d] + t_end;
        }
        // Only a profiled chunk has anything to report.
        if lvl > 0 {
            collected.push(acc);
        }
    };
    if total_cols > 1 {
        threads::pool().for_each_chunk(total_cols, quant_pass);
    } else {
        // One column (or none): no split — the closure runs inline on the caller; deterministic, `tests/mt.rs` covers this branch too.
        quant_pass(0..total_cols);
    }
    for acc in collected.into_vec() {
        pacc.add_acc(&acc);
    }
    quantized
}

/// One `PairWork` per pair: weight bytes, (possibly shared) quantized
/// columns, and a shared output pointer; the sharing is read-only from here —
/// the quant pre-pass was the only writer.
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
            // SAFETY (construction site, referenced by every write): the
            // pointer aliases this pair's `out.data`, caller-owned for the
            // whole call. Each participant writes only cells `t * n + r` with
            // `r` inside its own sub-range of the row split, which partitions
            // the rows; the join publishes the writes; an early error leaves
            // zeros and discards the output either way.
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

/// The row dispatch and its gather: one pool job over `Σ_i n` rows (a chunk
/// may straddle pair boundaries; the walker hands each pair its own
/// sub-range), then the sort, error scan and accumulator fold. The split
/// axis is the output row `r` and only that — each element still accumulates
/// its k products ascending, so the worker split cannot change a bit
/// (`tests/mt.rs`, `tests/ops.rs`); splitting k or the token axis is forbidden.
fn run_row_pool(
    pairs: &[PairWork<'_>],
    ty: GgmlType,
    k: usize,
    n: usize,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> Result<(), crate::ModelError> {
    let total_rows = n * pairs.len();
    let collected: profile::ChunkSlots<RowChunk> = profile::ChunkSlots::new();
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
        // A clean unprofiled chunk has nothing to report.
        if lvl > 0 || chunk.err.is_some() {
            collected.push(chunk);
        }
    });

    // Arrival order is nondeterministic; sorting by `start` keeps the first
    // error the lowest failing row's, as a sequential `?` would return.
    let t_gather = if lvl >= 2 { Some(Instant::now()) } else { None };
    let mut chunks = collected.into_vec();
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

/// One pool dispatch over `Σ_i n` rows for all `(W_i, x_i)` pairs; `site`
/// names the profiler row so the two shapes stay separable in the stage table.
fn matmul_q_multi(
    site: &'static str,
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    let Some((k, n, ty)) = batch_shape(ws, xs)? else {
        return Ok(Vec::new());
    };
    // Profiler hook: level 0 one compare, level 1 one `Instant` pair, level 2
    // two more per row, none touching the arithmetic; chunks merge into one
    // `CallAcc`, so `record` fires once per call.
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
