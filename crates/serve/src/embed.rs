//! An embedding model the server holds beside its engine ([`crate::Server::with_side`]): the one
//! seam between the embeddings endpoints and the model, and the normalization the endpoints own.
//!
//! The server reads a request as llama-server's embeddings handler does and tokenizes its text
//! with the model's own vocabulary ([`Embed::tokenizer`]); the model runs one pass over one input's
//! ids, including its pooling and learned projections ([`Embed::embed`]). The cut to a
//! `dimensions` width, the normalization
//! ([`normalize`], llama.cpp's `common_embd_normalize`) and the answer's shape are the server's.

use std::sync::Arc;

use crate::engine::Tokenizer;

/// The model failed on a valid input (a 500), which ends the server.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(pub String);

/// An embedding model: one input's ids in, its pooled vector out. The server calls it one
/// request at a time, behind a mutex, and reads the facts below once, when it takes the model.
pub trait Embed: Send {
    /// The model's vocabulary, which the server tokenizes the inputs' text with.
    fn tokenizer(&self) -> Arc<dyn Tokenizer>;
    /// The model's name: its `/v1/models` id and the `model` every OpenAI answer carries.
    fn name(&self) -> String;
    /// The model file, reported as `/props`' `embedding_model.path`.
    fn path(&self) -> String;
    /// The most positions one input takes.
    fn ctx(&self) -> usize;
    /// The widths a request's `dimensions` may cut the pooled vector to, ascending; the last is
    /// the pooled vector's own width.
    fn dims(&self) -> Vec<usize>;
    /// The side model's image preparer, present when a projector is loaded.
    fn media(&self) -> Option<crate::media::SharedMediaModel> {
        None
    }
    /// One pass over one input: at least one id and at most [`Embed::ctx`], each below the
    /// vocabulary's size. The pooled vector, the full width (the last of [`Embed::dims`]).
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError>;
    /// One pass with prepared images replacing the positions they occupy.
    fn embed_media(
        &mut self,
        _ids: &[u32],
        _images: &[crate::media::MediaFeed],
    ) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError(
            "the embedding model has no image encoder (--embd-mmproj)".into(),
        ))
    }
}

/// A normalization llama.cpp's `common_embd_normalize` names (`embd_normalize`): `-1` none, `0`
/// by the largest magnitude onto the int16 range, `2` euclidean (the default), any other `p > 0`
/// the p-norm (`1` taxicab).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Norm(i32);

impl Default for Norm {
    fn default() -> Norm {
        Norm(2)
    }
}

impl Norm {
    /// `p` as a normalization; below `-1`, which names none, is refused by name.
    pub fn of(p: i32) -> Result<Norm, String> {
        if p < -1 {
            return Err(
                "-1 (none), 0 (max absolute, int16), 1 (taxicab), 2 (euclidean) or a p-norm \
                 above 2"
                    .to_owned(),
            );
        }
        Ok(Norm(p))
    }
}

/// `v` normalized as `common_embd_normalize` does: the norm summed in f64 in order and its
/// reciprocal applied in f32; a vector of zeros comes back as zeros. A vector with a value that
/// is not zero whose norm comes to zero or past the f64 range, or whose reciprocal is past the
/// f32 range (a large `p` underflows or overflows the sum), is refused by name, where
/// `common_embd_normalize` gives zeros or infinities.
pub fn normalize(v: &[f32], norm: Norm) -> Result<Vec<f32>, String> {
    let sum = match norm.0 {
        -1 => 1.0,
        0 => {
            v.iter().fold(0.0f64, |m, &x| {
                let a = f64::from(x.abs());
                if m < a { a } else { m }
            }) / 32760.0
        }
        2 => v.iter().fold(0.0f64, |s, &x| s + f64::from(x * x)).sqrt(),
        p => {
            let p = f64::from(p);
            v.iter()
                .fold(0.0f64, |s, &x| s + f64::from(x.abs()).powf(p))
                .powf(1.0 / p)
        }
    };
    if v.iter().all(|&x| x == 0.0) {
        return Ok(vec![0.0; v.len()]);
    }
    let scale = (1.0 / sum) as f32;
    if !(sum > 0.0 && sum.is_finite() && scale > 0.0 && scale.is_finite()) {
        return Err(format!(
            "embd_normalize {}: this vector's norm comes to {sum:e}, which f32 cannot scale by",
            norm.0
        ));
    }
    Ok(v.iter().map(|&x| x * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each normalization against the arithmetic `common_embd_normalize` does, and the norms
    /// it cannot scale by refused.
    #[test]
    fn normalize_is_llama_cpps() {
        let n = |p| Norm::of(p).unwrap();
        let v = [3.0f32, -4.0];
        assert_eq!(normalize(&v, n(-1)).unwrap(), v);
        let int16 = (1.0 / (4.0f64 / 32760.0)) as f32;
        assert_eq!(normalize(&v, n(0)).unwrap(), [3.0 * int16, -4.0 * int16]);
        let l1 = (1.0 / 7.0f64) as f32;
        assert_eq!(normalize(&v, n(1)).unwrap(), [3.0 * l1, -4.0 * l1]);
        assert_eq!(normalize(&v, Norm::default()).unwrap(), [0.6, -0.8]);
        let p3 = (1.0 / (27.0f64 + 64.0).powf(1.0 / 3.0)) as f32;
        assert_eq!(normalize(&v, n(3)).unwrap(), [3.0 * p3, -4.0 * p3]);
        assert_eq!(normalize(&[0.0, 0.0], n(2)).unwrap(), [0.0, 0.0]);
        assert!(Norm::of(-2).is_err());
        for (v, p) in [
            ([0.5f32, 0.25], 2000),
            ([3.0f32, 4.0], 2000),
            ([1e-40f32, 0.0], 0),
        ] {
            let e = normalize(&v, n(p)).expect_err("refused");
            assert!(
                e.starts_with(&format!("embd_normalize {p}: ")),
                "{v:?} {p}: {e}"
            );
        }
    }
}
