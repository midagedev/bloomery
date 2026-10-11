//! The two facts of the `glm5v` tower that no shared kernel carries, as host rules the card's
//! kernels and their gate are held to: the K order of the merger's 2×2 convolution and the clamped
//! SwiGLU. Both are llama.cpp's graph: `glm4v.cpp`'s `ggml_conv_2d` over a merge group, and
//! `ggml_swiglu_clamp` (`ggml-cuda/unary.cuh`, `ggml_cuda_op_swiglu_clamp_single`).
//!
//! The convolution is a GEMM over `im2col` rows. `im2col` puts the channel slowest and the kernel
//! column fastest (`ggml-cuda/im2col.cu`: `iic * KH_KW + ikh * KW + ikw`), so the file's weight
//! `[kx, ky, c, o]` is, per output `o`, `K` values at `kx + merge·(ky + merge·c)`. The card reads
//! a merge group as its patch rows side by side, the patch `dy·merge + dx` at columns
//! `(dy·merge + dx)·dim ..`, so the weight is reordered once at load to
//! `(ky·merge + kx)·dim + c`: the channel fastest, then the column, then the row.

/// The flat index of `(kx, ky, c)` in output `o`'s row of the file's convolution weight, which
/// ggml stores `[kx, ky, c, o]` with `kx` contiguous: `kx + merge·(ky + merge·c)` within the
/// row of `merge²·dim` values.
#[must_use]
pub fn conv_k_in_file(merge: usize, kx: usize, ky: usize, c: usize) -> usize {
    kx + merge * (ky + merge * c)
}

/// The column of the same `(kx, ky, c)` in a merged row of the card: the group's patch
/// `ky·merge + kx` (the patch at row `dy = ky`, column `dx = kx` of its group), then channel `c`.
#[must_use]
pub fn conv_k_on_card(merge: usize, dim: usize, kx: usize, ky: usize, c: usize) -> usize {
    (ky * merge + kx) * dim + c
}

/// The merger's convolution weight as the card's GEMM reads it: `out` rows of `merge²·dim` values,
/// each row reordered from the file's order ([`conv_k_in_file`]) to the card's
/// ([`conv_k_on_card`]). A weight of another length is a caller's bug and panics by name.
#[must_use]
pub fn reorder_merger_weight<T: Copy>(file: &[T], dim: usize, merge: usize, out: usize) -> Vec<T> {
    let k = merge * merge * dim;
    assert_eq!(
        file.len(),
        out * k,
        "merger weight: {} values for {out} rows of {k}",
        file.len()
    );
    let mut card = file.to_vec();
    for (src, dst) in file.chunks_exact(k).zip(card.chunks_exact_mut(k)) {
        for c in 0..dim {
            for ky in 0..merge {
                for kx in 0..merge {
                    dst[conv_k_on_card(merge, dim, kx, ky, c)] =
                        src[conv_k_in_file(merge, kx, ky, c)];
                }
            }
        }
    }
    card
}

/// `ggml_swiglu_clamp(gate, up, limit)` on one f32: the gate is clamped from above only and the up
/// value from both sides, then `silu(gate) · up`. A NaN stays a NaN (llama.cpp's `fminf` and
/// `fmaxf` return the number and so hide it).
#[must_use]
pub fn swiglu_clamp(gate: f32, up: f32, limit: f32) -> f32 {
    let g = if gate > limit { limit } else { gate };
    let u = if up > limit {
        limit
    } else if up < -limit {
        -limit
    } else {
        up
    };
    g / (1.0 + (-g).exp()) * u
}

#[cfg(test)]
mod tests {
    use super::{conv_k_in_file, conv_k_on_card, reorder_merger_weight, swiglu_clamp};

    /// The reordered weight times the card's row (a merge group's patch rows side by side) is
    /// ggml's convolution of the permuted group. `glm4v.cpp` reshapes the group to
    /// `[dim, dx, dy]` and permutes it to `[dx, dy, dim]`; `ggml_conv_2d` is `im2col` (K index
    /// `kx + merge·(ky + merge·c)`, `ggml-cuda/im2col.cu`) times the weight `[kx, ky, c, o]`
    /// flattened per output. Integer values, so the sums are exact.
    #[test]
    fn the_reordered_weight_is_ggmls_convolution_of_a_group() {
        let (dim, merge, out) = (5usize, 2usize, 3usize);
        let k = merge * merge * dim;
        let file: Vec<i64> = (0..(out * k) as i64)
            .map(|i| (i * 7919) % 211 - 100)
            .collect();
        let rows: Vec<i64> = (0..k as i64).map(|i| (i * 104_729) % 97 - 40).collect();
        // the group as the conv sees it, `[dx, dy, c]` with dx contiguous
        let mut conv_input = vec![0i64; k];
        for c in 0..dim {
            for dy in 0..merge {
                for dx in 0..merge {
                    conv_input[dx + merge * (dy + merge * c)] = rows[c + dim * (dx + merge * dy)];
                }
            }
        }
        let card = reorder_merger_weight(&file, dim, merge, out);
        for o in 0..out {
            // the weight tensor `[kx, ky, c, o]`, kx contiguous, against the conv's input
            let want: i64 = (0..k).map(|i| file[i + k * o] * conv_input[i]).sum();
            let got: i64 = (0..k).map(|j| card[o * k + j] * rows[j]).sum();
            assert_eq!(got, want, "output {o}");
        }
    }

    /// The two index functions are the layouts they name: the file's has the column fastest, the
    /// card's the channel.
    #[test]
    fn the_indices_are_the_two_layouts() {
        assert_eq!(conv_k_in_file(2, 1, 0, 0), 1);
        assert_eq!(conv_k_in_file(2, 0, 1, 0), 2);
        assert_eq!(conv_k_in_file(2, 0, 0, 1), 4);
        assert_eq!(conv_k_on_card(2, 1024, 0, 0, 1), 1);
        assert_eq!(conv_k_on_card(2, 1024, 1, 0, 0), 1024);
        assert_eq!(conv_k_on_card(2, 1024, 0, 1, 0), 2048);
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// The gate is clamped from above only, the up value from both sides, at the same limit.
    #[test]
    fn the_gate_clamps_one_side_and_the_up_value_both() {
        let l = 10.0;
        assert_eq!(swiglu_clamp(25.0, 3.0, l), silu(10.0) * 3.0);
        // below -limit the gate is not clamped: silu(-20), not silu(-10)
        assert_eq!(swiglu_clamp(-20.0, 3.0, l), silu(-20.0) * 3.0);
        assert_ne!(silu(-20.0), silu(-10.0));
        assert_eq!(swiglu_clamp(1.5, 25.0, l), silu(1.5) * 10.0);
        assert_eq!(swiglu_clamp(1.5, -25.0, l), silu(1.5) * -10.0);
        assert_eq!(swiglu_clamp(1.5, 2.5, l), silu(1.5) * 2.5);
    }

    /// A NaN input is a NaN output, not a clamped number.
    #[test]
    fn a_nan_stays_a_nan() {
        assert!(swiglu_clamp(f32::NAN, 1.0, 10.0).is_nan());
        assert!(swiglu_clamp(1.0, f32::NAN, 10.0).is_nan());
    }
}
