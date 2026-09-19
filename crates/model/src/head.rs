//! Output head: `l_out-26` in, `result_norm` then `result_output` out.
//!
//! Gate: `crates/model/tests/head.rs` against the oracle. Owned by the head round.
//!
//! Both stages are position-wise, so a batch in is a batch out: `head` returns one
//! logits column per input column and never special-cases the last position — picking
//! the position is the caller's job. That is the batch-first decision in `lib.rs`
//! applied to the last op pair in the graph.
use crate::ops::{f32_tensor, matmul_q, rms_norm};
use crate::{ModelError, Tensor2};
use gguf::Gguf;

/// Returns the logits for every token position; the oracle only holds the last one.
///
/// `x` is the final block's output `[embd, n_tokens]`; the result is
/// `[vocab, n_tokens]`, where the vocab axis is `output.weight`'s own width — the
/// gate checks it against `deepseek2.vocab_size` from the file, never a literal.
pub fn head(gguf: &Gguf, x: &Tensor2) -> Result<Tensor2, ModelError> {
    // deepseek2 carries one architecture-wide rms eps; the file has no separate
    // final-norm key. The 1e-4 gate on `result_norm` is the numeric proof that
    // this key is the one the final norm runs with.
    let eps = gguf
        .architecture()
        .and_then(|a| gguf.value(&format!("{a}.attention.layer_norm_rms_epsilon")))
        .and_then(gguf::Value::as_f32)
        .expect("rms eps must come from the file, never from a literal");

    let norm_t = gguf
        .find("output_norm.weight")
        .ok_or_else(|| ModelError::MissingTensor("output_norm.weight".into()))?;
    let gain = f32_tensor(gguf, norm_t)?;

    if x.ne0 != gain.len() {
        return Err(ModelError::Shape {
            what: "head input",
            want_ne0: gain.len(),
            want_ne1: x.ne1,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }

    let normed = rms_norm(x, &gain, eps);

    let out_t = gguf
        .find("output.weight")
        .ok_or_else(|| ModelError::MissingTensor("output.weight".into()))?;
    matmul_q(gguf, out_t, &normed)
}
