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
use std::sync::Mutex;
use std::time::Instant;

/// A 2-D activation block in ggml's layout: `ne0` is contiguous, `ne1` strides by `ne0`.
///
/// Verified against the oracle (2026-09-19, `docs/oracle.md`): a one-token run's `inp_embd`
/// equals the first `ne0` floats of a two-token run, difference exactly 0. So token `t`
/// lives at `data[t * ne0 .. (t + 1) * ne0]`.
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
/// One owner for the byte walk: this was written out three times (attn, head, and the
/// ops gate inline) before `forward` needed a fourth, and three copies of a loop that
/// reads f32 little-endian is three places for an endianness or stride assumption to
/// drift apart.
pub fn f32_tensor(gguf: &Gguf, t: &TensorInfo) -> Result<Vec<f32>, crate::ModelError> {
    // Profiler hook (crate::profile): the byte walk itself. `rows` and `k` are 0
    // on purpose — a flat read has no contraction and no row structure; the
    // weight MB column is the work. Level 1 only.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes = gguf.data(t)?;
    let out = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    // Profiler hook (crate::profile): level-1 call timer. This fires twice per
    // block plus once in the head (~55 times per decode step), so one Instant
    // pair is the whole instrumentation. The shape statement is not a
    // contraction because there is none: `rows` counts the columns normed, `k`
    // the elements each column walks, and the only weight read is the F32 gain.
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

// Per-thread dequantized-weight-row scratch for `matmul_q`. The single-threaded
// version reused one `vec![0.0f32; k]` for the whole call; with the rows on the
// pool every worker needs its own, and growing a thread-local to the largest `k`
// seen beats allocating per chunk by the call count — `matmul_q` fires over a
// thousand times per token. (Plain comment, not a doc comment: `thread_local!`
// is a macro invocation and has nothing to attach a doc to.)
thread_local! {
    static ROW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// One pool chunk's finished bookkeeping: its row-range start (for the
/// lowest-row-first error precedence), its share of the level-2 stage timers,
/// and the first error it hit (if any — a quant error on the scalar path, a
/// `qdot` error on the fused one). The output values themselves go straight
/// into `out` through `out_ptr` — the MUL-23 measurement (2026-09-20, level 2)
/// put the old staged handover at ~7 ms per decode step: a private buffer per
/// chunk (32 allocations per call at ~850 calls per step), a collector push,
/// a sort, and a transpose copy, none of which computes anything. Workers
/// hand these small structs over through one mutex take per chunk — the only
/// shared mutable state on the parallel path, and it is never touched from
/// inside the row loop.
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
struct SharedOut(*mut f32);
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
    unsafe fn write(&self, idx: usize, v: f32) {
        // SAFETY: the caller guarantees the disjointness contract above.
        unsafe { *self.0.add(idx) = v };
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
/// a quantized dot, and the oracle is ggml's output. An f32 reference is 0.6 % away
/// (measured), and the *wrong* quantized format is still 0.1 % away — both would force
/// every gate below to open. `gguf::activation_format` owns the table.
///
/// **Q3_K with k a multiple of 256 takes the fused path** (`crates/qdot`): the dot runs
/// straight off the quantized codes and no f32 weight row is ever materialized. That
/// kernel is *more* accurate than this crate's dequant-then-round-trip f32 dot (measured
/// against an f64 exact answer in the qdot round, 2026-09-20), so its output is
/// deliberately NOT bit-identical to the scalar path — gates that cross a Q3_K matmul
/// assert closeness to the oracle, never bit equality with the scalar path. Every other
/// type, and Q3_K rows whose k is not a multiple of 256, keep the scalar path unchanged.
pub fn matmul_q(gguf: &Gguf, w: &TensorInfo, x: &Tensor2) -> Result<Tensor2, crate::ModelError> {
    // Profiler hook (crate::profile): level 0 is one compare per call, level 1 one
    // `Instant` pair for the whole call, level 2 two more per row. None of it touches
    // the arithmetic — the gate proves that bit for bit. Since the rows moved onto
    // the pool, level 2 accumulates into one `CallAcc` per chunk, merged after the
    // join, so `profile::record` still fires exactly once per call.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let k = w.dims[0] as usize;
    let n = if w.dims.len() > 1 {
        w.dims[1] as usize
    } else {
        1
    };
    if x.ne0 != k {
        return Err(crate::ModelError::Shape {
            what: "matmul_q input",
            want_ne0: k,
            want_ne1: x.ne1,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / n;
    let mut out = Tensor2::zeros(n, x.ne1);

    // Quantize every activation column once, not once per weight row. WHICH format is a
    // property of the WEIGHT type, not a global choice — `gguf::activation_format` owns the
    // table. Using Q8_K for everything was wrong by 2e-3 on the Q5_1 down projection
    // (found 2026-09-19 by the ffn round, proven against libggml's own quantizer).
    // Single-threaded on purpose: a measured 0.2 % of one decode step (level 2,
    // 12.8 of 7588 ms) — parallelizing it would cost more than it could return.
    //
    // The fused qdot kernel (crates/qdot) dots the quantized codes directly, so its
    // activation input is `qdot::quantize_col`'s byte layout, not the f32 round trip
    // above. `supports(w.ty)` implies Q3_K — the only type the qdot API is built for —
    // and the k check mirrors the kernel's whole-super-block contract (256 values per
    // block). Decided once per call here; the row loop branches on the same fact.
    let fused = qdot::supports(w.ty) && k.is_multiple_of(256);
    let quantized = {
        let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
        let q = if fused {
            let cb = qdot::col_bytes(w.ty, k);
            let mut buf = vec![0u8; x.ne1 * cb];
            for t in 0..x.ne1 {
                qdot::quantize_col(w.ty, x.col(t), &mut buf[t * cb..(t + 1) * cb]);
            }
            QuantCols::Bytes { cb, buf }
        } else {
            let mut q = vec![0.0f32; x.data.len()];
            for t in 0..x.ne1 {
                quantize_activations(w.ty, x.col(t), &mut q[t * k..(t + 1) * k]);
            }
            QuantCols::F32(q)
        };
        if let Some(t_q) = t_q {
            pacc.add_quant_act(t_q.elapsed().as_nanos() as u64);
        }
        q
    };

    // The rows run on the resident pool, split on the OUTPUT ROW r and nothing else.
    // That is the whole bit-identity argument: each output element still accumulates
    // its k products ascending, in the same order it always did, so which thread
    // computes which row cannot change a bit — `tests/mt.rs` holds that as a byte
    // compare across thread counts. Splitting k (or the token axis) would reorder
    // the accumulation and is forbidden. The fused path rides the same argument:
    // `qdot::dot_row` computes one row's whole k in a single call, so a row is
    // still the only unit that moves between threads and k is never split.
    //
    // Workers write their results straight into `out` (token-major `t*n + r`):
    // the row split partitions the cells, so no two participants alias, and the
    // pool's completion protocol orders the writes before this function reads
    // `out` again — the same ordering argument that publishes the job slot.
    // The staged handover this replaced (private chunk buffer, collector push,
    // sort, transpose copy) was measured at ~7 ms per decode step (MUL-23,
    // level 2, 2026-09-20) without computing anything.
    let ne1 = x.ne1;
    let ty = w.ty;
    // The fused branch's column stride and buffer, hoisted so the worker closure
    // captures plain data and the scalar branch stays literally the code it was.
    let fused_cols: Option<(usize, &[u8])> = match &quantized {
        QuantCols::Bytes { cb, buf } => Some((*cb, buf.as_slice())),
        QuantCols::F32(_) => None,
    };
    // The scalar path's f32 round trip, shadowed under the name its row loop has
    // always used. On the fused path the binding is dead — that row loop reads
    // `fused_cols` instead — so an empty slice stands in for it rather than an
    // `Option` unwrap inside the loop. The two row loops run one per call, never
    // both: the same `fused` fact chose what `quantized` holds.
    let quantized: &[f32] = match &quantized {
        QuantCols::F32(q) => q,
        QuantCols::Bytes { .. } => &[],
    };
    // SAFETY (taken once, before the closure, so the &mut borrow ends here and
    // only the raw pointer crosses into the pool): `out_ptr` aliases `out.data`,
    // which this function owns for the whole call. Every participant — workers
    // and the calling thread's own chunk — writes only cells `t * n + r` with
    // `r` inside its own contiguous range of the row split, and
    // `threads::chunks(n, T)` partitions `0..n`, so no two participants ever
    // write the same cell. The writes are published to this thread by the
    // pool's completion protocol (each worker's `remaining` fetch-sub release
    // happens-after its writes; the dispatcher's acquire load of `remaining ==
    // 0` happens before `for_each_chunk` returns), the same argument that
    // already publishes the job slot. `out` is not read until after the join,
    // and a chunk that errors early simply leaves its unwritten cells at the
    // zeros `Tensor2::zeros` gave them — the error discards the output either
    // way, exactly as the staged handover's partially-filled buffers did.
    let out_ptr = SharedOut(out.data.as_mut_ptr());
    let collected: Mutex<Vec<RowChunk>> = Mutex::new(Vec::with_capacity(threads::pool().threads()));
    threads::pool().for_each_chunk(n, |rows| {
        let mut chunk = RowChunk {
            start: rows.start,
            acc: profile::CallAcc::new(),
            err: None,
        };
        if let Some((cb, acol)) = fused_cols {
            // Fused rows: `qdot::dot_row` consumes the weight bytes and the
            // quantized column directly, so no f32 row is materialized and
            // ROW_BUF stays untouched. `add_dequant_w` staying 0 here is by
            // design — Q3_K's dequant stage dropping to 0.00 in the profile
            // table is this wiring's signature, not missing instrumentation.
            // The dequant is fused into the dot, so the dot timer covers the
            // whole row, tokens included.
            for r in rows {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                for t in 0..ne1 {
                    match qdot::dot_row(ty, src, &acol[t * cb..(t + 1) * cb], k) {
                        Ok(v) => {
                            // SAFETY: cell `t * n + r` belongs to this chunk's
                            // row range alone — see the comment on `out_ptr`.
                            unsafe { out_ptr.write(t * n + r, v) };
                        }
                        Err(e) => {
                            chunk.err = Some(e.into());
                            break;
                        }
                    }
                }
                if let Some(t_dot) = t_dot {
                    chunk.acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                }
                if chunk.err.is_some() {
                    break;
                }
            }
        } else {
            ROW_BUF.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if scratch.len() < k {
                    scratch.resize(k, 0.0);
                }
                let row = &mut scratch[..k];
                for r in rows {
                    let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                    // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                    let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
                    if ty == GgmlType::F32 {
                        for (i, v) in row.iter_mut().enumerate() {
                            *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
                        }
                    } else if let Err(e) = dequant_row(ty, src, row) {
                        chunk.err = Some(e.into());
                        break;
                    }
                    if let Some(t_d) = t_d {
                        chunk.acc.add_dequant_w(t_d.elapsed().as_nanos() as u64);
                    }
                    let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                    for t in 0..ne1 {
                        let xc = &quantized[t * k..(t + 1) * k];
                        let mut acc = 0.0f32;
                        for i in 0..k {
                            acc += row[i] * xc[i];
                        }
                        // SAFETY: cell `t * n + r` belongs to this chunk's row
                        // range alone — see the comment on `out_ptr`.
                        unsafe { out_ptr.write(t * n + r, acc) };
                    }
                    if let Some(t_dot) = t_dot {
                        chunk.acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                    }
                }
            });
        }
        // The one lock of the chunk — never per row: a mutex inside the row loop
        // would profile the mutex.
        collected
            .lock()
            .expect("matmul_q row-chunk collector")
            .push(chunk);
    });

    // Arrival order is nondeterministic; row order is not. Sorting by `start`
    // keeps the first error the lowest failing row's — the same error the
    // sequential `?` this replaced would have returned (a quant error on the
    // scalar path, a `qdot` error on the fused one), and the record call below
    // is skipped on that path exactly as it was before. The values themselves
    // are already in `out`; since MUL-23 this section is bookkeeping only, and
    // level 2 times it as the `gather` stage to keep that honest.
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
    if let Some(t_call) = t_call {
        profile::record(
            "matmul_q",
            w.ty,
            n as u64,
            k as u64 * n as u64,
            bytes.len() as u64,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
    }
    Ok(out)
}
