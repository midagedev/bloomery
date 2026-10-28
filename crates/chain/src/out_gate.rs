//! The host rule of the sigmoid output gate of a gated softmax-attention layer, `y = attn ·
//! sigmoid(g)`, over `m` tokens of `n_head` heads of `head_dim` values. `g` is the gate
//! projection's own output (`x · W_g`, an ordinary linear outside the rule). [`OutGate::Proj`]
//! gates every value by a `g` of its own, `[m][n_head][head_dim]`; [`OutGate::ProjHead`] gates all
//! the values of a head by the head's one `g`, `[m][n_head]`, so value `i` reads `g[i / head_dim]`.
//!
//! Arithmetic: `sigmoid(g) = 1 / (1 + e^−g)` in f32 with the host libm's `exp` (ggml's
//! `ggml_vec_sigmoid_f32`), and `y = attn · sigmoid(g)` rounded once. A card kernel's `exp` is its
//! own, an ulp or two apart, so a kernel is held to this within its band, not bit for bit.
//!
//! No silent failure: a gate of another kind or a length that does not fit is refused by name, and
//! so is a non-finite input or product, as the kernels raise their fault word for it (`y` then
//! holds nothing meaningful).

use model::arch::models::OutGate;

use crate::ChainError;

/// `attn · sigmoid(gate)` into `y`, for the gate kind `kind` ([`OutGate::Proj`] or
/// [`OutGate::ProjHead`]); `attn` and `y` are `[m][n_head][head_dim]`, `m` the whole number of
/// tokens they hold.
pub fn out_gate(
    kind: OutGate,
    attn: &[f32],
    gate: &[f32],
    n_head: usize,
    head_dim: usize,
    y: &mut [f32],
) -> Result<(), ChainError> {
    const WHAT: &str = "out_gate";
    let width = n_head * head_dim;
    if width == 0 || attn.is_empty() || !attn.len().is_multiple_of(width) {
        return Err(ChainError::shape(
            WHAT,
            format!(
                "attn holds {} values; need a whole number, at least one, of tokens of \
                 {n_head} heads of {head_dim}",
                attn.len()
            ),
        ));
    }
    let m = attn.len() / width;
    let gates = match kind {
        OutGate::Proj => m * width,
        OutGate::ProjHead => m * n_head,
        OutGate::None | OutGate::Packed => {
            return Err(ChainError::shape(
                WHAT,
                format!("{kind:?} is not a gate of its own: this rule takes Proj and ProjHead"),
            ));
        }
    };
    if gate.len() != gates || y.len() != attn.len() {
        return Err(ChainError::shape(
            WHAT,
            format!(
                "gate.len() {} (want {gates}), y.len() {} (want {}) for {kind:?} over m={m} \
                 n_head={n_head} head_dim={head_dim}",
                gate.len(),
                y.len(),
                attn.len()
            ),
        ));
    }
    let per_head = kind == OutGate::ProjHead;
    for (i, (&a, out)) in attn.iter().zip(y.iter_mut()).enumerate() {
        let g = gate[if per_head { i / head_dim } else { i }];
        let product = a * (1.0 / (1.0 + (-g).exp()));
        if !(a.is_finite() && g.is_finite() && product.is_finite()) {
            return Err(ChainError::shape(
                WHAT,
                format!("value {i}: attn {a}, gate {g}, product {product}; one is not finite"),
            ));
        }
        *out = product;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use model::arch::models::OutGate;

    use super::out_gate;

    /// `sigmoid(0)` is exactly ½ and `sigmoid(100)` rounds to exactly 1 (`e^−100` is far below half
    /// an ulp of 1), so the gates 0 and 100 keep half a value or all of it. Two heads of two values:
    /// per head the gate `[0, 100]` halves head 0 and keeps head 1, so `[2, 4, 6, 8]` becomes
    /// `[1, 2, 6, 8]`; per channel the gate `[0, 100, 100, 0]` gives `[1, 4, 6, 4]`. Mutant: the
    /// per-head index `i / head_dim` is `i % n_head` (the head's gate alternates by value), the
    /// per-channel index is `i / head_dim`, or the sigmoid is `1 / (1 + e^g)`.
    #[test]
    fn a_head_gates_all_its_values_by_one_number_and_a_channel_gates_its_own() {
        let attn = [2.0, 4.0, 6.0, 8.0];
        let mut y = [0.0; 4];
        out_gate(OutGate::ProjHead, &attn, &[0.0, 100.0], 2, 2, &mut y).expect("a valid call");
        assert_eq!(y, [1.0, 2.0, 6.0, 8.0]);
        out_gate(
            OutGate::Proj,
            &attn,
            &[0.0, 100.0, 100.0, 0.0],
            2,
            2,
            &mut y,
        )
        .expect("a valid call");
        assert_eq!(y, [1.0, 4.0, 6.0, 4.0]);
    }

    /// The gate of the second token is the second row of `[m][n_head]`: tokens `[2, 4, 6, 8]` and
    /// `[10, 12, 14, 16]` with the gates `[0, 100]` and `[100, 0]` give `[1, 2, 6, 8]` and
    /// `[10, 12, 7, 8]`. Mutant: the per-head index drops the token (`i / head_dim % n_head`).
    #[test]
    fn the_token_axis_of_a_per_head_gate_is_its_own_row() {
        let attn = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let mut y = [0.0; 8];
        out_gate(
            OutGate::ProjHead,
            &attn,
            &[0.0, 100.0, 100.0, 0.0],
            2,
            2,
            &mut y,
        )
        .expect("a valid call");
        assert_eq!(y, [1.0, 2.0, 6.0, 8.0, 10.0, 12.0, 7.0, 8.0]);
    }

    /// A gate of no projection of its own, a length that does not fit and a non-finite value are
    /// refused by name. Mutant: drop the finite check (`1 / (1 + e^−NaN)` is NaN, a plausible-looking
    /// output), or accept `Packed`.
    #[test]
    fn what_this_rule_does_not_take_is_refused_by_name() {
        let attn = [2.0, 4.0, 6.0, 8.0];
        let mut y = [0.0; 4];
        for kind in [OutGate::None, OutGate::Packed] {
            let err = out_gate(kind, &attn, &[0.0; 4], 2, 2, &mut y)
                .expect_err("not a projected gate")
                .to_string();
            assert!(err.contains("is not a gate of its own"), "{kind:?}: {err}");
        }
        let err = out_gate(OutGate::ProjHead, &attn, &[0.0; 3], 2, 2, &mut y)
            .expect_err("a short gate")
            .to_string();
        assert!(err.contains("gate.len() 3 (want 2)"), "{err}");
        let err = out_gate(OutGate::ProjHead, &attn, &[0.0; 2], 3, 2, &mut y)
            .expect_err("tokens that do not divide")
            .to_string();
        assert!(err.contains("a whole number"), "{err}");
        let err = out_gate(OutGate::ProjHead, &attn, &[f32::NAN, 0.0], 2, 2, &mut y)
            .expect_err("a NaN gate")
            .to_string();
        assert!(err.contains("value 0: attn 2, gate NaN"), "{err}");
    }
}
