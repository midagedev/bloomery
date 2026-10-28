//! The host rule of a gated short convolution, `y = C ⊙ conv(B ⊙ x)`: the reference a card or Metal
//! kernel is held to. The input projection writes three chunks of `d` values a token, `B | C | x`
//! in that order; the conv is causal and depthwise, one filter of `taps` values per channel, the
//! oldest tap first, with no bias and no activation, over the products `B ⊙ x` of the token's
//! last `taps` positions. The conv ring ([`Ring`]) keeps those products across calls. The input
//! and output projections around the rule are ordinary linears and are not here.
//!
//! Arithmetic, in the order the delta layers' conv runs so one kernel shape serves both: `B ⊙ x`
//! rounded to f32; `acc = x₀·w₀` rounded, then `acc = fma(xⱼ, wⱼ, acc)` for the later taps, the
//! oldest input first; `y = C · acc` rounded.
//!
//! No silent failure: a call whose rows are not consecutive positions has no defined
//! predecessors, and an output that is not finite means an input or a product was not; both are
//! refused by name, and a refused call leaves the ring as it was (`y` then holds nothing
//! meaningful).

use crate::ChainError;
use crate::linear::Ring;

/// One call of the short conv over `pos.len()` tokens at the consecutive positions `pos`, `d`
/// channels, `ring`'s taps.
///
/// - `bcx`: `[m][3d]`, each row `B | C | x`.
/// - `weights`: `[d][taps]`, the oldest tap first.
/// - `state`: the ring, `[rows][d]`, the product of position `p` in slot `p mod rows`. A
///   predecessor before the sequence's start (`pos < distance`) reads zero whatever the ring
///   holds. The call writes the products of its last `min(m, rows)` tokens.
/// - `y`: `[m][d]`.
pub fn short_conv(
    ring: Ring,
    d: usize,
    bcx: &[f32],
    weights: &[f32],
    pos: &[u32],
    state: &mut [f32],
    y: &mut [f32],
) -> Result<(), ChainError> {
    const WHAT: &str = "short_conv";
    let (m, taps, rows) = (pos.len(), ring.taps(), ring.rows());
    if d == 0 || m == 0 {
        return Err(ChainError::shape(
            WHAT,
            format!("need d >= 1 and a call of at least one token, got d={d} m={m}"),
        ));
    }
    let lens = [
        ("bcx", bcx.len(), m * 3 * d),
        ("weights", weights.len(), d * taps),
        ("state", state.len(), ring.values(d)),
        ("y", y.len(), m * d),
    ];
    if let Some((name, got, want)) = lens.iter().find(|(_, got, want)| got != want) {
        return Err(ChainError::shape(
            WHAT,
            format!("{name}.len() {got}, want {want} (d={d} m={m} taps={taps})"),
        ));
    }
    if let Some(t) = (1..m).find(|&t| u64::from(pos[t]) != u64::from(pos[0]) + t as u64) {
        return Err(ChainError::shape(
            WHAT,
            format!(
                "row {t} is position {}; the rows of a call are consecutive positions from {}",
                pos[t], pos[0]
            ),
        ));
    }
    let bx: Vec<f32> = (0..m * d)
        .map(|i| {
            let row = &bcx[i / d * 3 * d..][..3 * d];
            row[i % d] * row[2 * d + i % d]
        })
        .collect();
    for (t, &p) in pos.iter().enumerate() {
        let p = p as usize;
        for c in 0..d {
            // Tap `j` reads the input `reach − j` positions back: this call's row, or the ring.
            let input = |j: usize| -> f32 {
                let back = ring.reach() - j;
                if back <= t {
                    bx[(t - back) * d + c]
                } else {
                    ring.before(p, back).map_or(0.0, |slot| state[slot * d + c])
                }
            };
            let w = &weights[c * taps..][..taps];
            let mut acc = input(0) * w[0];
            for (j, &wj) in w.iter().enumerate().skip(1) {
                acc = input(j).mul_add(wj, acc);
            }
            let out = bcx[t * 3 * d + d + c] * acc;
            if !out.is_finite() {
                return Err(ChainError::shape(
                    WHAT,
                    format!(
                        "token {t} channel {c}: the output is {out}; an input or a product is not finite"
                    ),
                ));
            }
            y[t * d + c] = out;
        }
    }
    for (t, &p) in pos.iter().enumerate().skip(m.saturating_sub(rows)) {
        let slot = p as usize % rows;
        state[slot * d..][..d].copy_from_slice(&bx[t * d..][..d]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::short_conv;
    use crate::linear::Ring;

    /// The hand case's filters: `d = 2` channels of three taps, `[1, 2, 3]` and `[0, 0, 1]`.
    const W: [f32; 6] = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0];

    /// Tokens of the hand case, `B | C | x`: the products `B ⊙ x` are `[2, 5]`, `[4, 6]`,
    /// `[6, 7]`, then `[4, 8]`.
    const TOKENS: [[f32; 6]; 4] = [
        [2.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        [2.0, 1.0, 0.5, 1.0, 2.0, 6.0],
        [2.0, 1.0, 2.0, 1.0, 3.0, 7.0],
        [1.0, 1.0, 1.0, 1.0, 4.0, 8.0],
    ];

    fn call(
        tokens: &[[f32; 6]],
        from: u32,
        state: &mut [f32],
    ) -> Result<Vec<f32>, crate::ChainError> {
        let bcx: Vec<f32> = tokens.iter().flatten().copied().collect();
        let pos: Vec<u32> = (from..from + tokens.len() as u32).collect();
        let mut y = vec![0.0; tokens.len() * 2];
        short_conv(Ring::new(3), 2, &bcx, &W, &pos, state, &mut y)?;
        Ok(y)
    }

    /// The rule on a hand-derived case at the sequence start. Channel 0, filter `[1, 2, 3]` over
    /// the products `2, 4, 6` with `C = 1, ½, 2`: `3·2 = 6`; `2·2 + 3·4 = 16`, times ½ is 8;
    /// `1·2 + 2·4 + 3·6 = 28`, times 2 is 56. Channel 1, filter `[0, 0, 1]` is the current
    /// product, `5, 6, 7`, times `C = 1`. The ring then holds the three products in slots 0, 1, 2.
    /// Mutant: the taps read newest first (`input(reach − j)`), `C` left out of `out`, or `x` taken
    /// where `B ⊙ x` is.
    #[test]
    fn the_rule_on_a_hand_case_at_the_sequence_start() {
        let mut state = vec![0.0; 20];
        let y = call(&TOKENS[..3], 0, &mut state).expect("a valid call");
        assert_eq!(y, [6.0, 5.0, 8.0, 6.0, 56.0, 7.0]);
        assert_eq!(state[..6], [2.0, 5.0, 4.0, 6.0, 6.0, 7.0]);
        assert!(state[6..].iter().all(|&v| v == 0.0));
    }

    /// A call that continues reads its predecessors from the ring by position: the token at
    /// position 3, products `[4, 8]`, sees `4, 6` (positions 1, 2) and itself on channel 0:
    /// `1·4 + 2·6 + 3·4 = 28`; on channel 1 the current product, 8. And a call cut anywhere is the
    /// uncut call: the four tokens in one call, or as 3 + 1, or one by one, give the same rows and
    /// the same ring. Mutant: `ring.before` is given the row number `t` where the position `p` is
    /// (a continuing call reads the slots of its row numbers).
    #[test]
    fn a_call_cut_at_any_point_is_the_uncut_call() {
        let mut whole = vec![0.0; 20];
        let y_whole = call(&TOKENS, 0, &mut whole).expect("a valid call");
        assert_eq!(y_whole[6..], [28.0, 8.0]);

        let mut cut = vec![0.0; 20];
        let mut y_cut = call(&TOKENS[..3], 0, &mut cut).expect("a valid call");
        y_cut.extend(call(&TOKENS[3..], 3, &mut cut).expect("a valid call"));
        assert_eq!((&y_cut, &cut), (&y_whole, &whole));

        let mut single = vec![0.0; 20];
        let mut y_single = Vec::new();
        for (p, token) in TOKENS.iter().enumerate() {
            y_single.extend(call(&[*token], p as u32, &mut single).expect("a valid call"));
        }
        assert_eq!((&y_single, &single), (&y_whole, &whole));
    }

    /// More tokens than the ring has rows: the ring keeps the last `rows` products in the slots
    /// of their positions, and the outputs are the one-by-one run's. Fourteen tokens of small
    /// integers (every product and sum exact) from position 5 over a ring of ten rows. Mutant: the
    /// ring write takes the slot `t mod rows` where `pos[t] mod rows` is, `pos[t] mod taps`, or
    /// skips one token too many (`m.saturating_sub(rows) + 1`).
    #[test]
    fn a_call_longer_than_the_ring_keeps_the_last_rows() {
        let tokens: Vec<[f32; 6]> = (0..14)
            .map(|t| std::array::from_fn(|k| ((t * 7 + k * 3) % 5) as f32 - 2.0))
            .collect();
        let mut whole = vec![0.0; 20];
        let y_whole = call(&tokens, 5, &mut whole).expect("a valid call");
        let mut single = vec![0.0; 20];
        let mut y_single = Vec::new();
        for (t, token) in tokens.iter().enumerate() {
            y_single.extend(call(&[*token], 5 + t as u32, &mut single).expect("a valid call"));
        }
        assert_eq!((&y_whole, &whole), (&y_single, &single));
    }

    /// At the sequence start the ring is not read, whatever it holds: the hand case over a ring
    /// full of 99 gives the rows of a ring of zeros. Mutant: `Ring::before` returns a slot while
    /// `p < d`, or the call reads the ring for a predecessor the sequence does not have.
    #[test]
    fn the_sequence_start_reads_no_ring() {
        let mut junk = vec![99.0; 20];
        let y = call(&TOKENS[..3], 0, &mut junk).expect("a valid call");
        assert_eq!(y, [6.0, 5.0, 8.0, 6.0, 56.0, 7.0]);
    }

    /// A call whose rows are not consecutive positions, one of the wrong lengths and one with a
    /// value that is not finite are refused by name, and the ring is as it was. Mutant: drop the
    /// consecutive check, or write the ring before the finite check.
    #[test]
    fn a_call_with_no_defined_predecessors_is_refused_and_leaves_the_ring() {
        let ring = Ring::new(3);
        let bcx: Vec<f32> = TOKENS[..2].iter().flatten().copied().collect();
        let mut state = vec![7.0; 20];
        let mut y = vec![0.0; 4];
        let err = short_conv(ring, 2, &bcx, &W, &[4, 6], &mut state, &mut y)
            .expect_err("positions 4 and 6")
            .to_string();
        assert!(err.contains("row 1 is position 6"), "{err}");
        let err = short_conv(ring, 2, &bcx, &W[..5], &[4, 5], &mut state, &mut y)
            .expect_err("a short weight table")
            .to_string();
        assert!(err.contains("weights.len() 5, want 6"), "{err}");
        let mut nan = bcx;
        nan[4] = f32::NAN;
        let err = short_conv(ring, 2, &nan, &W, &[4, 5], &mut state, &mut y)
            .expect_err("a NaN input")
            .to_string();
        assert!(
            err.contains("token 0 channel 0: the output is NaN"),
            "{err}"
        );
        assert!(state.iter().all(|&v| v == 7.0));
    }
}
