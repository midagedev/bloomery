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
///
/// Buffers are recycled: `Drop` hands `data` to a per-thread free list and the
/// constructors take from it, so a steady-state step does not go to malloc for
/// activations (`tests/alloc.rs` pins the count).
#[derive(PartialEq, Debug)]
pub struct Tensor2 {
    pub ne0: usize,
    pub ne1: usize,
    pub data: Vec<f32>,
}

thread_local! {
    /// Per-thread recycled `Vec<f32>` storage, bucketed by size class: the
    /// activations (`Tensor2`) and the scalar-path quantization buffers are the
    /// same shape of allocation, and a step cycles several sizes at once — a
    /// flat list that fills rejects every later drop and starves exactly the
    /// sizes it exists to serve.
    static F32_BLOCKS: RefCell<Vec<Vec<Vec<f32>>>> = const { RefCell::new(Vec::new()) };
    /// The byte-shaped sibling, for the fused path's quantized columns.
    static U8_BLOCKS: RefCell<Vec<Vec<Vec<u8>>>> = const { RefCell::new(Vec::new()) };
}
/// Free-list depth per size class. One decode step's working set fits: the
/// widest class holds the sixteen per-head activation blocks `wv_b_heads`
/// gathers, plus headroom for transients.
const CLASS_DEPTH: usize = 32;

/// The bucket a capacity of `n` lives in: the power-of-two floor. A take's
/// capacity window `[n, 2n)` never crosses more than one bucket boundary, so a
/// search walks at most two.
fn class_of(n: usize) -> usize {
    n.ilog2() as usize
}

fn take_f32(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let hit = F32_BLOCKS
        .try_with(|p| {
            let mut p = p.borrow_mut();
            let c = class_of(n);
            [c, c + 1].into_iter().find_map(|cls| {
                p.get_mut(cls).and_then(|b| {
                    let i = b
                        .iter()
                        .position(|b| b.capacity() >= n && b.capacity() <= 2 * n)?;
                    Some(b.swap_remove(i))
                })
            })
        })
        .ok()
        .flatten();
    match hit {
        Some(mut b) => {
            // Stale cells stay: only a block shorter than `n` is extended.
            if b.len() >= n {
                b.truncate(n);
            } else {
                b.resize(n, 0.0);
            }
            b
        }
        None => vec![0.0f32; n],
    }
}

fn give_f32(b: Vec<f32>) {
    if b.capacity() == 0 {
        return;
    }
    let _ = F32_BLOCKS.try_with(|p| {
        let mut p = p.borrow_mut();
        let cls = class_of(b.capacity());
        if p.len() <= cls {
            p.resize_with(cls + 1, Vec::new);
        }
        if p[cls].len() < CLASS_DEPTH {
            p[cls].push(b);
        }
    });
}

fn take_u8(n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let hit = U8_BLOCKS
        .try_with(|p| {
            let mut p = p.borrow_mut();
            let c = class_of(n);
            [c, c + 1].into_iter().find_map(|cls| {
                p.get_mut(cls).and_then(|b| {
                    let i = b
                        .iter()
                        .position(|b| b.capacity() >= n && b.capacity() <= 2 * n)?;
                    Some(b.swap_remove(i))
                })
            })
        })
        .ok()
        .flatten();
    match hit {
        Some(mut b) => {
            // Stale cells stay: only a block shorter than `n` is extended.
            if b.len() >= n {
                b.truncate(n);
            } else {
                b.resize(n, 0);
            }
            b
        }
        None => vec![0u8; n],
    }
}

fn give_u8(b: Vec<u8>) {
    if b.capacity() == 0 {
        return;
    }
    let _ = U8_BLOCKS.try_with(|p| {
        let mut p = p.borrow_mut();
        let cls = class_of(b.capacity());
        if p.len() <= cls {
            p.resize_with(cls + 1, Vec::new);
        }
        if p[cls].len() < CLASS_DEPTH {
            p[cls].push(b);
        }
    });
}

/// `BLOOMERY_POISON=1` fills every `scratch` block with NaN, so a kernel that
/// leaves a cell unwritten fails the gates loudly instead of reading a stale value.
fn poison() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("BLOOMERY_POISON").is_ok_and(|v| v != "0"))
}

/// A block of `n` f32 off the free list, contents unspecified (stale, never
/// uninitialized). The capacity window keeps a logits-sized block from being spent
/// on a 2048-wide request.
fn take_block(n: usize) -> Vec<f32> {
    take_f32(n)
}

impl Drop for Tensor2 {
    fn drop(&mut self) {
        // `try_with`: a block dropped during thread teardown just frees.
        give_f32(std::mem::take(&mut self.data));
    }
}

impl Clone for Tensor2 {
    fn clone(&self) -> Self {
        let mut out = Self::scratch(self.ne0, self.ne1);
        out.data.copy_from_slice(&self.data);
        out
    }
}

impl Tensor2 {
    pub fn zeros(ne0: usize, ne1: usize) -> Self {
        let mut data = take_block(ne0 * ne1);
        data.fill(0.0);
        Self { ne0, ne1, data }
    }

    /// A block whose every cell the caller is about to write; reading one first
    /// is a bug (`BLOOMERY_POISON` makes it a NaN).
    pub fn scratch(ne0: usize, ne1: usize) -> Self {
        let mut data = take_block(ne0 * ne1);
        if poison() {
            data.fill(f32::NAN);
        }
        Self { ne0, ne1, data }
    }

    pub fn from_vec(ne0: usize, ne1: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            ne0 * ne1,
            "Tensor2 data length must be ne0 * ne1"
        );
        Self { ne0, ne1, data }
    }

    /// The values, leaving the free list out of it (`Drop` forbids moving `data` out).
    pub fn into_data(mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
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
    let mut out = Tensor2::scratch(x.ne0, x.ne1);
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
    /// The chunk's own wall, profiled runs only.
    busy_ns: u64,
    acc: profile::CallAcc,
    err: Option<crate::ModelError>,
}

/// `matmul_q`'s activation columns in the form the row loop consumes — f32 round trip (scalar) or `qdot::quantize_col` bytes (fused, `cb`/column); an enum so the paths cannot swap inputs.
///
/// The buffers are recycled per thread: quantized activations are per-call
/// values in stable storage, so a steady decode step takes from the pool and
/// returns instead of going to malloc (`Tensor2`'s free list is the same
/// argument). Contents are rewritten before use — every column is a pure
/// function of its input, and the quantizer writes whole columns.
enum QuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round-trip representation.
    F32(Vec<f32>),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, buf: Vec<u8> },
}

impl QuantCols {
    /// A buffer for one distinct input of `ne1` columns, off the pool.
    fn new(cb: Option<usize>, x: &Tensor2) -> QuantCols {
        // Stale contents, like `Tensor2::scratch`; poisoned the same way, so a
        // quantizer that leaves a byte of its column unwritten shows in the gates.
        match cb {
            Some(cb) => {
                let mut buf = take_u8(x.ne1 * cb);
                if poison() {
                    buf.fill(0xA5);
                }
                QuantCols::Bytes { cb, buf }
            }
            None => {
                let mut buf = take_f32(x.data.len());
                if poison() {
                    buf.fill(f32::NAN);
                }
                QuantCols::F32(buf)
            }
        }
    }

    /// The write view the pool workers quantize through; SAFETY: the pointer
    /// aliases this buffer, owned by the caller for the whole dispatch — the
    /// disjoint-cells argument at `quantize_distinct`'s construction site.
    fn shared(&mut self) -> SharedQuantCols {
        match self {
            QuantCols::Bytes { cb, buf } => SharedQuantCols::Bytes {
                cb: *cb,
                ptr: buf.as_mut_ptr(),
            },
            QuantCols::F32(q) => SharedQuantCols::F32(q.as_mut_ptr()),
        }
    }

    /// Hand the storage back to the pool.
    fn release(self) {
        match self {
            QuantCols::Bytes { buf, .. } => give_u8(buf),
            QuantCols::F32(q) => give_f32(q),
        }
    }
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
#[derive(Clone, Copy)]
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
///
/// The single-pair shape: no `Vec` of pairs, no dedupe table, the one
/// quantization buffer off the per-thread pool — the form a decode step calls
/// hundreds of times per second.
pub fn matmul_q(gguf: &Gguf, w: &TensorInfo, x: &Tensor2) -> Result<Tensor2, crate::ModelError> {
    matmul_q_one("matmul_q", gguf, w, x)
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

impl<'a> PairWork<'a> {
    /// The one construction path both dispatch shapes ride; the SAFETY
    /// argument for the output pointer lives at the callers, where the split
    /// is decided.
    fn new(out: &mut Tensor2, bytes: &'a [u8], q: &'a QuantCols, row_bytes: usize) -> PairWork<'a> {
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
    }

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

/// The column walker both dispatch shapes ride: quantize columns `cols` of the
/// concatenated distinct-input space, one `quantize_into` call per column,
/// chunk-straddling inputs handled segment-wise.
///
/// SAFETY (construction site, referenced by every write): the pointers alias
/// buffers owned for the whole call; each participant writes only its own
/// sub-range's columns, the join publishes the writes, and a
/// boundary-straddling column is two segments, never quantized twice.
#[allow(clippy::too_many_arguments)]
fn quant_range(
    ty: GgmlType,
    k: usize,
    distinct: &[&Tensor2],
    col_starts: &[usize],
    shared: &[SharedQuantCols],
    cols: Range<usize>,
    lvl: u8,
    acc: &mut profile::CallAcc,
) {
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
}

/// Quantize one input's columns — the single-pair shape: no `Vec` of buffers,
/// the collector only when profiling. Bit-identity argument as
/// [`quantize_distinct`].
fn quantize_one(
    ty: GgmlType,
    k: usize,
    x: &Tensor2,
    fused: bool,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> QuantCols {
    let cb = if fused {
        Some(qdot::col_bytes(ty, k))
    } else {
        None
    };
    let mut qc = QuantCols::new(cb, x);
    let shared = [qc.shared()];
    let distinct = [x];
    let starts = [0usize];
    let total_cols = x.ne1;
    // Only a profiled dispatch pays for the chunk collector; the quantizer
    // cannot fail, so an unprofiled one needs no error channel either.
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<profile::CallAcc>::new())
    } else {
        None
    };
    if total_cols > 1 {
        threads::pool().for_each_chunk(total_cols, |cols| {
            let mut acc = profile::CallAcc::new();
            quant_range(ty, k, &distinct, &starts, &shared, cols, lvl, &mut acc);
            if let Some(collected) = &collected {
                collected.push(acc);
            }
        });
    } else if total_cols == 1 {
        // One column: no split — the walker runs inline on the caller;
        // deterministic, `tests/mt.rs` covers this branch too.
        let mut acc = profile::CallAcc::new();
        quant_range(ty, k, &distinct, &starts, &shared, 0..1, lvl, &mut acc);
        if let Some(collected) = &collected {
            collected.push(acc);
        }
    }
    if let Some(collected) = collected {
        for acc in collected.into_vec() {
            pacc.add_acc(&acc);
        }
    }
    qc
}

/// Quantize every activation column once per DISTINCT input on the pool,
/// split over the concatenated column space; the format is a property of the
/// WEIGHT type (`gguf::activation_format`): fused → `qdot::quantize_col`
/// bytes, scalar → the f32 round trip.
///
/// Bit identity: each column is a pure function of its own input, and the
/// split partitions the output cells; splitting WITHIN a column would need
/// its own argument (block-local scales).
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
    let mut quantized: Vec<QuantCols> = distinct.iter().map(|x| QuantCols::new(cb, x)).collect();
    let n = distinct.len();
    // Column starts and shared views: on the stack for the small batches a
    // decode step builds (one distinct input per routed expert), on the heap
    // past that.
    let mut starts_small = [0usize; 8];
    let mut starts_big: Vec<usize> = Vec::new();
    let mut total_cols = 0usize;
    let col_starts: &[usize] = if n <= 8 {
        for (i, x) in distinct.iter().enumerate() {
            starts_small[i] = total_cols;
            total_cols += x.ne1;
        }
        &starts_small[..n]
    } else {
        starts_big.reserve(n);
        for x in distinct {
            starts_big.push(total_cols);
            total_cols += x.ne1;
        }
        &starts_big
    };
    let mut shared_small = [SharedQuantCols::F32(std::ptr::null_mut()); 8];
    let shared_big: Vec<SharedQuantCols>;
    let shared: &[SharedQuantCols] = if n <= 8 {
        for i in 0..n {
            shared_small[i] = quantized[i].shared();
        }
        &shared_small[..n]
    } else {
        shared_big = quantized.iter_mut().map(|q| q.shared()).collect();
        &shared_big
    };
    // Only a profiled dispatch pays for the chunk collector; the quantizer
    // cannot fail, so an unprofiled one needs no error channel either.
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<profile::CallAcc>::new())
    } else {
        None
    };
    if total_cols > 1 {
        threads::pool().for_each_chunk(total_cols, |cols| {
            let mut acc = profile::CallAcc::new();
            quant_range(ty, k, distinct, col_starts, shared, cols, lvl, &mut acc);
            if let Some(collected) = &collected {
                collected.push(acc);
            }
        });
    } else if total_cols == 1 {
        // One column: no split — the walker runs inline on the caller;
        // deterministic, `tests/mt.rs` covers this branch too.
        let mut acc = profile::CallAcc::new();
        quant_range(ty, k, distinct, col_starts, shared, 0..1, lvl, &mut acc);
        if let Some(collected) = &collected {
            collected.push(acc);
        }
    }
    if let Some(collected) = collected {
        for acc in collected.into_vec() {
            pacc.add_acc(&acc);
        }
    }
    quantized
}

/// One `PairWork` per pair: weight bytes, (possibly shared) quantized
/// columns, and a shared output pointer; the sharing is read-only from here —
/// the quant pre-pass was the only writer.
///
/// SAFETY (construction site, referenced by every write): each `PairWork`'s
/// output pointer aliases that pair's `out.data`, caller-owned for the whole
/// call. Each participant writes only cells `t * n + r` with `r` inside its
/// own sub-range of the row split, which partitions the rows; the join
/// publishes the writes; an early error leaves zeros and discards the output
/// either way.
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
        .map(|((out, bytes), q)| PairWork::new(out, bytes, q, row_bytes))
        .collect()
}

/// One lane of a row dispatch: the next unclaimed row and the lane's end, on
/// its own cache line so an owner's claims do not fight its neighbours'.
#[repr(align(64))]
struct Lane {
    next: std::sync::atomic::AtomicUsize,
    end: usize,
}
const MAX_LANES: usize = 64;
fn steal_enabled() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("BLOOMERY_STEAL").map_or(true, |v| v != "0"))
}
/// Blocks a lane is cut into, and the smallest block worth a claim.
const STEAL_BLOCKS: usize = 4;
const STEAL_MIN_ROWS: usize = 8;

/// The unprofiled dispatch's error channel: no chunk collector, no
/// allocation — a clean chunk touches nothing, and the mutex only locks when
/// an error exists. The winning error is the lowest failing row's, exactly
/// what the sorted scan decided on the profiled path.
struct ErrGate {
    any: std::sync::atomic::AtomicBool,
    slot: std::sync::Mutex<Option<(usize, crate::ModelError)>>,
}

impl ErrGate {
    fn new() -> Self {
        ErrGate {
            any: std::sync::atomic::AtomicBool::new(false),
            slot: std::sync::Mutex::new(None),
        }
    }

    /// Offer the error of the chunk that starts at `start`; a lower start
    /// replaces a higher one. Compared under the lock, so the payload and the
    /// minimum cannot come from different chunks.
    fn offer(&self, start: usize, e: crate::ModelError) {
        let mut slot = self.slot.lock().expect("row error slot");
        if slot.as_ref().is_none_or(|(s, _)| start < *s) {
            *slot = Some((start, e));
        }
        self.any.store(true, std::sync::atomic::Ordering::Release);
    }

    fn take(&self) -> Option<crate::ModelError> {
        if !self.any.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        self.slot.lock().expect("row error slot").take().map(|(_, e)| e)
    }
}

/// The row dispatch and its gather: one pool job over `Σ_i n` rows (a chunk
/// may straddle pair boundaries; the walker hands each pair its own
/// sub-range), then the sort, error scan and accumulator fold. The split
/// axis is the output row `r` and only that — each element still accumulates
/// its k products ascending, so the worker split cannot change a bit
/// (`tests/mt.rs`, `tests/ops.rs`); splitting k or the token axis is forbidden.
///
/// A profiled dispatch pays the chunk collector (one `Vec` of pool-width
/// slots); an unprofiled one runs the error channel above and allocates
/// nothing unless a chunk fails.
fn run_row_pool(
    pairs: &[PairWork<'_>],
    ty: GgmlType,
    k: usize,
    n: usize,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> Result<(), crate::ModelError> {
    let total_rows = n * pairs.len();
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<RowChunk>::new())
    } else {
        None
    };
    let gate = ErrGate::new();
    let t_span = if lvl >= 1 { Some(Instant::now()) } else { None };
    // Lanes: the pool's own static split, one cursor each. A participant walks
    // its home lane front to back in blocks, then takes blocks off the front of
    // whatever lanes are still unfinished. The slowest chunk of a dispatch runs
    // 20–40 % over the mean and which chunk that is changes every dispatch, so
    // the barrier waits on noise; the tail is the only part worth sharing, and
    // the owner still streams its rows contiguously. A row's value does not
    // depend on who computes it.
    let nlanes = threads::pool().threads();
    let lanes: [Lane; MAX_LANES] = std::array::from_fn(|t| {
        let (start, end) = if t < nlanes {
            threads::chunk_bounds(total_rows, nlanes, t)
        } else {
            (0, 0)
        };
        Lane {
            next: std::sync::atomic::AtomicUsize::new(start),
            end,
        }
    });
    assert!(nlanes <= MAX_LANES, "row pool lanes: {nlanes} threads");
    // `BLOOMERY_STEAL=0` is the A/B lever: whole-lane blocks, home lane only.
    let steal = steal_enabled();
    let block = if steal {
        (total_rows / nlanes / STEAL_BLOCKS).max(STEAL_MIN_ROWS)
    } else {
        total_rows.max(1)
    };
    threads::pool().for_each_chunk(total_rows, |rows| {
        let t_busy = if lvl >= 1 { Some(Instant::now()) } else { None };
        let mut chunk = RowChunk {
            start: rows.start,
            busy_ns: 0,
            acc: profile::CallAcc::new(),
            err: None,
        };
        let home = (0..nlanes)
            .find(|&t| !rows.is_empty() && threads::chunk_bounds(total_rows, nlanes, t).0 == rows.start)
            .unwrap_or(0);
        'lanes: for off in 0..if steal { nlanes } else { 1 } {
            let lane = &lanes[(home + off) % nlanes];
            loop {
                // A cheap look first: a finished lane costs a read, not a write.
                if lane.next.load(std::sync::atomic::Ordering::Relaxed) >= lane.end {
                    break;
                }
                let from = lane.next.fetch_add(block, std::sync::atomic::Ordering::Relaxed);
                if from >= lane.end {
                    break;
                }
                let to = lane.end.min(from + block);
                let mut gr = from;
                while gr < to {
                    let p = gr / n;
                    let sub_end = to.min((p + 1) * n);
                    let r0 = gr - p * n;
                    if let Err(e) = pairs[p].compute_rows(
                        ty,
                        k,
                        n,
                        r0..r0 + (sub_end - gr),
                        lvl,
                        &mut chunk.acc,
                    ) {
                        chunk.err = Some(e);
                        break 'lanes;
                    }
                    gr = sub_end;
                }
            }
        }
        if let Some(t_busy) = t_busy {
            chunk.busy_ns = t_busy.elapsed().as_nanos() as u64;
        }
        // A clean unprofiled chunk has nothing to report.
        if let Some(collected) = &collected {
            collected.push(chunk);
        } else if let Some(e) = chunk.err {
            gate.offer(chunk.start, e);
        }
    });

    let span_ns = t_span.map(|t| t.elapsed().as_nanos() as u64);
    if let Some(collected) = collected {
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
        if lvl >= 1 {
            for (i, c) in chunks.iter().enumerate() {
                profile::add_chunk_busy(i, c.busy_ns);
            }
        }
        if let Some(span_ns) = span_ns {
            pacc.add_span(span_ns, chunks.iter().map(|c| c.busy_ns).max().unwrap_or(0));
        }
        if let Some(t_gather) = t_gather {
            pacc.add_gather(t_gather.elapsed().as_nanos() as u64);
        }
    } else if let Some(e) = gate.take() {
        return Err(e);
    }
    Ok(())
}

/// The single-pair dispatch `matmul_q` rides: one weight, one input, no
/// per-call `Vec`s — the quantization buffer off the thread pool, the one
/// `PairWork` on the stack. Same row loop, same `batch_shape` contract, same
/// profiler row as the batch shape; only the bookkeeping is narrower.
fn matmul_q_one(
    site: &'static str,
    gguf: &Gguf,
    w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, crate::ModelError> {
    let Some((k, n, ty)) = batch_shape(std::slice::from_ref(&w), std::slice::from_ref(&x))? else {
        unreachable!("a one-pair batch is not empty")
    };
    // Profiler hook: level 0 one compare, level 1 one `Instant` pair, level 2
    // two more per row, none touching the arithmetic; chunks merge into one
    // `CallAcc`, so `record` fires once per call.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / n;
    let fused = qdot::supports(ty) && k.is_multiple_of(qdot::k_granularity(ty));

    let qc = quantize_one(ty, k, x, fused, lvl, &mut pacc);
    let mut out = Tensor2::scratch(n, x.ne1);
    // SAFETY: the construction-site argument in `build_pair_work`'s doc — one
    // pair, the whole row split, the join publishes.
    let pw = PairWork::new(&mut out, bytes, &qc, row_bytes);
    run_row_pool(std::slice::from_ref(&pw), ty, k, n, lvl, &mut pacc)?;
    qc.release();

    if let Some(t_call) = t_call {
        profile::record(
            site,
            ty,
            n as u64,
            k as u64 * n as u64,
            bytes.len() as u64,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
    }
    Ok(out)
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

    let mut outs: Vec<Tensor2> = xs.iter().map(|x| Tensor2::scratch(n, x.ne1)).collect();
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
    for q in quantized {
        q.release();
    }
    Ok(outs)
}
