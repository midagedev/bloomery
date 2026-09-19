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

/// One pool chunk's finished work: its output rows in a private contiguous buffer
/// laid out `[r_local][t]`, its share of the level-2 stage timers, and the first
/// quant error it hit (if any). Workers hand these over through one mutex take per
/// chunk — the only shared mutable state on the parallel path, and it is never
/// touched from inside the row loop.
struct RowChunk {
    start: usize,
    nrows: usize,
    buf: Vec<f32>,
    acc: profile::CallAcc,
    err: Option<gguf::QuantError>,
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
    let quantized = {
        let mut q = vec![0.0f32; x.data.len()];
        let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
        for t in 0..x.ne1 {
            quantize_activations(w.ty, x.col(t), &mut q[t * k..(t + 1) * k]);
        }
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
    // the accumulation and is forbidden.
    //
    // Workers write into a private contiguous buffer laid out [r_local][t]; the
    // gather below moves them into `out` single-threaded. Direct `out.data[t*n + r]`
    // writes from several threads would alias without new `unsafe`, and this round
    // adds none.
    let ne1 = x.ne1;
    let ty = w.ty;
    let collected: Mutex<Vec<RowChunk>> = Mutex::new(Vec::new());
    threads::pool().for_each_chunk(n, |rows| {
        let mut chunk = RowChunk {
            start: rows.start,
            nrows: rows.len(),
            buf: vec![0.0f32; rows.len() * ne1],
            acc: profile::CallAcc::new(),
            err: None,
        };
        ROW_BUF.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < k {
                scratch.resize(k, 0.0);
            }
            let row = &mut scratch[..k];
            for (rl, r) in rows.enumerate() {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
                if ty == GgmlType::F32 {
                    for (i, v) in row.iter_mut().enumerate() {
                        *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
                    }
                } else if let Err(e) = dequant_row(ty, src, row) {
                    chunk.err = Some(e);
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
                    chunk.buf[rl * ne1 + t] = acc;
                }
                if let Some(t_dot) = t_dot {
                    chunk.acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                }
            }
        });
        // The one lock of the chunk — never per row: a mutex inside the row loop
        // would profile the mutex.
        collected
            .lock()
            .expect("matmul_q row-chunk collector")
            .push(chunk);
    });

    // Arrival order is nondeterministic; row order is not. Sorting by `start` makes
    // the gather sequential and the first quant error the lowest failing row's —
    // the same error the sequential `?` this replaced would have returned, and the
    // record call below is skipped on that path exactly as it was before.
    let mut chunks = collected
        .into_inner()
        .expect("matmul_q row-chunk collector");
    chunks.sort_by_key(|c| c.start);
    if let Some(e) = chunks.iter_mut().find_map(|c| c.err.take()) {
        return Err(e.into());
    }
    for c in &chunks {
        for rl in 0..c.nrows {
            let r = c.start + rl;
            for t in 0..ne1 {
                out.data[t * n + r] = c.buf[rl * ne1 + t];
            }
        }
        pacc.add_acc(&c.acc);
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
