//! The host rule of the sandwich residual (`Residual::Sandwich`): a sublayer's output goes through
//! its own weighted RMS norm before it joins the stream, `x += add · gain ⊙ rms_norm(out)`, with
//! `add` the sublayer's scale (the mixer's, then the feed-forward block's) and `gain` the norm's
//! own weight, which the file stores as the gain itself.
//!
//! The norm is the CPU engine's ([`model::ops::rms_norm`], ggml's form: the mean of the squares
//! over the row with `eps` inside the root, then `(scale · gain) · out`); the join is `x + add · y`
//! with the product rounded. The two scales a model carries so far, 1 and ½, are powers of two, so
//! a fused multiply-add rounds the same bits as the product and the sum do.
//!
//! No silent failure: a length that does not fit, an `eps` that is not finite and non-negative,
//! and a non-finite input or normed value (an all-zero output with `eps` 0 has no scale) are
//! refused by name, and a refused call leaves `x` as it was.

use model::ops::{Tensor2, rms_norm};

use crate::ChainError;

/// `x += add · gain ⊙ rms_norm(out)` over the rows of `gain.len()` values `x` and `out` hold,
/// each row normed alone.
pub fn sandwich_add(
    x: &mut [f32],
    out: &[f32],
    gain: &[f32],
    eps: f32,
    add: f32,
) -> Result<(), ChainError> {
    const WHAT: &str = "sandwich_add";
    let n = gain.len();
    if n == 0 || x.is_empty() || x.len() != out.len() || !x.len().is_multiple_of(n) {
        return Err(ChainError::shape(
            WHAT,
            format!(
                "x.len() {}, out.len() {}, gain.len() {n}; need equal, whole rows of the gain's \
                 width, at least one",
                x.len(),
                out.len()
            ),
        ));
    }
    if !(eps.is_finite() && eps >= 0.0 && add.is_finite()) {
        return Err(ChainError::shape(
            WHAT,
            format!("eps {eps} and add {add}; eps is finite and non-negative, add finite"),
        ));
    }
    let normed = rms_norm(&Tensor2::from_vec(n, x.len() / n, out.to_vec()), gain, eps).into_data();
    if let Some(i) = normed.iter().position(|v| !v.is_finite()) {
        return Err(ChainError::shape(
            WHAT,
            format!(
                "value {i}: the normed output is {}; an input is not finite, or a row is all \
                 zero with eps 0",
                normed[i]
            ),
        ));
    }
    for (x, y) in x.iter_mut().zip(&normed) {
        *x += add * y;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::sandwich_add;

    /// The output `[2, 2]` has mean square 4, so the scale is ½ and the normed value is `½ · gain ·
    /// 2` = `[1, 3]` for the gain `[1, 3]`; at `add` ½ the stream `[10, 20]` becomes `[10.5, 21.5]`.
    /// Mutant: the join drops `add` (`[11, 23]`), the gain is left out (`[10.5, 20.5]`), or the
    /// norm is not applied.
    #[test]
    fn the_output_is_normed_weighted_and_scaled_into_the_stream() {
        let mut x = [10.0, 20.0];
        sandwich_add(&mut x, &[2.0, 2.0], &[1.0, 3.0], 0.0, 0.5).expect("a valid call");
        assert_eq!(x, [10.5, 21.5]);
    }

    /// `eps` sits inside the root, added to the mean square: the output `[1, 1]` at `eps` 3 has
    /// scale `1 / √(1 + 3) = ½`, so the gain `[2, 4]` gives `[1, 2]` (a root taken first, `1 / (1 +
    /// 3)`, would give `[½, 1]`). Mutant: `eps` added after the root.
    #[test]
    fn eps_is_inside_the_root() {
        let mut x = [0.0, 0.0];
        sandwich_add(&mut x, &[1.0, 1.0], &[2.0, 4.0], 3.0, 1.0).expect("a valid call");
        assert_eq!(x, [1.0, 2.0]);
    }

    /// Each row is normed alone: the rows `[1, 1, 1, 1]` (scale 1) and `[4, 0, 0, 0]` (mean square
    /// 4, scale ½) give `[1, 1, 1, 1]` and `[2, 0, 0, 0]`; normed over both rows together the mean
    /// square would be 2.5. Mutant: the norm runs over the whole buffer.
    #[test]
    fn each_row_is_normed_alone() {
        let mut x = [0.0; 8];
        let out = [1.0, 1.0, 1.0, 1.0, 4.0, 0.0, 0.0, 0.0];
        sandwich_add(&mut x, &out, &[1.0; 4], 0.0, 1.0).expect("a valid call");
        assert_eq!(x, [1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    /// A length that does not fit, a negative `eps`, a NaN output and an all-zero row at `eps` 0
    /// are refused by name, and `x` is as it was. Mutant: add before the finite check.
    #[test]
    fn what_has_no_defined_norm_is_refused_and_leaves_the_stream() {
        let mut x = [5.0, 6.0];
        for (out, gain, eps, want) in [
            (&[1.0][..], &[1.0, 1.0][..], 0.0, "x.len() 2, out.len() 1"),
            (&[1.0, 1.0][..], &[1.0, 1.0][..], -1.0, "eps -1"),
            (&[f32::NAN, 1.0][..], &[1.0, 1.0][..], 0.0, "value 0"),
            (&[0.0, 0.0][..], &[1.0, 1.0][..], 0.0, "all zero with eps 0"),
        ] {
            let err = sandwich_add(&mut x, out, gain, eps, 1.0)
                .expect_err("a refused call")
                .to_string();
            assert!(err.contains(want), "{out:?}: {err}");
        }
        assert_eq!(x, [5.0, 6.0]);
    }
}
