//! The two primitives every module needs, owned here so three rounds do not write three
//! of them. Reference implementations: correctness first, `crates/q3k-cpu` holds the fast
//! path and stage 1 does not call it.
//!
//! Accumulation is f32 in ggml's own order (one row at a time, k ascending). That choice is
//! the reason the gates can be tight: a different order gives a different last bit, and a
//! gate at 1e-3 would hide a real error to leave room for it.

use gguf::{GgmlType, Gguf, TensorInfo, dequant_row, quantize_activations};

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

/// RMS norm with a learned gain, per column.
///
/// ggml computes the mean of squares over the whole row, adds eps, and multiplies by the
/// reciprocal square root — `eps` is inside the sqrt, not added to it. Getting that wrong
/// is a ~1e-4 error that a loose gate would absorb.
pub fn rms_norm(x: &Tensor2, gain: &[f32], eps: f32) -> Tensor2 {
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
    out
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
    let mut row = vec![0.0f32; k];
    let mut out = Tensor2::zeros(n, x.ne1);

    // Quantize every activation column once, not once per weight row. WHICH format is a
    // property of the WEIGHT type, not a global choice — `gguf::activation_format` owns the
    // table. Using Q8_K for everything was wrong by 2e-3 on the Q5_1 down projection
    // (found 2026-09-19 by the ffn round, proven against libggml's own quantizer).
    let quantized = {
        let mut q = vec![0.0f32; x.data.len()];
        for t in 0..x.ne1 {
            quantize_activations(w.ty, x.col(t), &mut q[t * k..(t + 1) * k]);
        }
        q
    };
    for r in 0..n {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        if w.ty == GgmlType::F32 {
            for (i, v) in row.iter_mut().enumerate() {
                *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            }
        } else {
            dequant_row(w.ty, src, &mut row)?;
        }
        for t in 0..x.ne1 {
            let xc = &quantized[t * k..(t + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += row[i] * xc[i];
            }
            out.data[t * n + r] = acc;
        }
    }
    Ok(out)
}
