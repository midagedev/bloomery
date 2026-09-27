//! The size rules of the stores a layer keeps beside its weights, one owner
//! for the plan that counts their bytes and the card body that allocates
//! them: the recurrent conv ring's rows, the PLE conv ring's rows, the pooled
//! plane's rows, and each layer's bytes by kind. Pure arithmetic over the
//! file's widths; a card constant that must equal one of these holds itself
//! to it with a `const` assertion where it is defined.

/// Bytes of an f16 value.
pub const F16_BYTES: u64 = 2;
/// Bytes of an f32 value.
pub const F32_BYTES: u64 = 4;
/// Bytes of a u32 word.
pub const U32_BYTES: u64 = 4;

/// The widest call a later call may roll back into, in positions: a ring
/// keeps its conv's reach plus this many inputs, so a cut anywhere inside a
/// pass of up to `PASS_ROWS` positions still finds every input its next
/// position reads.
pub const PASS_ROWS: usize = 8;

/// Rows of a recurrent layer's conv ring: the reach of a `conv`-tap conv and
/// a pass.
///
/// # Panics
///
/// On a conv of no tap, by name.
#[must_use]
pub const fn conv_ring_rows(conv: usize) -> usize {
    assert!(conv >= 1, "conv_ring_rows: a conv of at least one tap");
    conv - 1 + PASS_ROWS
}

/// Rows of a PLE conv ring: the reach of `taps` taps `dilation` apart and a
/// pass.
///
/// # Panics
///
/// On a conv of no tap, by name.
#[must_use]
pub const fn ple_ring_rows(taps: usize, dilation: usize) -> usize {
    assert!(taps >= 1, "ple_ring_rows: a conv of at least one tap");
    (taps - 1) * dilation + PASS_ROWS
}

/// Rows of a pooled key plane over `ctx` positions in pools of `pool`, the
/// last pool counted whole. `pool` is at least one.
#[must_use]
pub const fn pooled_rows(ctx: usize, pool: usize) -> usize {
    ctx.div_ceil(pool)
}

/// A gated-delta-rule layer's state and conv ring over `v_heads` value heads
/// and `k_heads` key heads of `state` values with a `conv`-tap conv, in
/// bytes: `v_heads` heads of `state × state` f32 and the ring's rows of the
/// conv's channels, `2·k_heads·state + v_heads·state`, in f32.
#[must_use]
pub const fn recurrent_bytes(v_heads: usize, k_heads: usize, state: usize, conv: usize) -> u64 {
    let (v, k, d) = (v_heads as u64, k_heads as u64, state as u64);
    let channels = 2 * k * d + v * d;
    v * d * d * F32_BYTES + conv_ring_rows(conv) as u64 * channels * F32_BYTES
}

/// Lanes of a gated-delta-rule layer's state on a model that verifies
/// drafted rows (Qwen3.8's `Body38`): a verify of up to this many rows keeps
/// the state after each of its rows in a lane of its own, and a commit moves
/// the lane word instead of copying.
pub const DELTA_LANES: usize = 4;

/// Bytes a gated-delta-rule layer holds past [`recurrent_bytes`]' one state
/// lane when it keeps `lanes` lanes (at least one) over `v_heads` value
/// heads of `state × state` f32: the other `lanes − 1` lanes and a u32
/// stamp a lane (the position the lane's state stands at).
///
/// # Panics
///
/// On no lane, by name.
#[must_use]
pub const fn delta_lane_bytes(v_heads: usize, state: usize, lanes: usize) -> u64 {
    assert!(lanes >= 1, "delta_lane_bytes: a state of at least one lane");
    let (v, d, l) = (v_heads as u64, state as u64, lanes as u64);
    (l - 1) * v * d * d * F32_BYTES + l * U32_BYTES
}

/// A PLE conv ring of `taps` taps `dilation` apart over `streams` streams
/// of `n_embd` values, in bytes (f32).
#[must_use]
pub const fn ple_ring_bytes(taps: usize, dilation: usize, streams: usize, n_embd: usize) -> u64 {
    ple_ring_rows(taps, dilation) as u64 * (streams * n_embd) as u64 * F32_BYTES
}

/// A GQA position's K and V over `kv_heads` heads of `head_dim`, in f16.
#[must_use]
pub const fn kv_row_bytes(kv_heads: usize, head_dim: usize) -> u64 {
    2 * (kv_heads * head_dim) as u64 * F16_BYTES
}

/// A dense attention layer's bytes at `ctx` positions: each position's K and
/// V ([`kv_row_bytes`]) and nothing else — the MTP draft layer's store, which
/// attends to every position it holds.
#[must_use]
pub const fn dense_kv_bytes(kv_heads: usize, head_dim: usize, ctx: usize) -> u64 {
    ctx as u64 * kv_row_bytes(kv_heads, head_dim)
}

/// A selecting attention layer's bytes at `ctx` positions: each position's
/// K and V ([`kv_row_bytes`]) and raw index key of `idx_dim` f16, and one
/// pooled key of `idx_dim` f16 a pool of `pool` positions
/// ([`pooled_rows`]).
#[must_use]
pub const fn selecting_bytes(
    kv_heads: usize,
    head_dim: usize,
    idx_dim: usize,
    pool: usize,
    ctx: usize,
) -> u64 {
    let raw = idx_dim as u64 * F16_BYTES;
    ctx as u64 * (kv_row_bytes(kv_heads, head_dim) + raw) + pooled_rows(ctx, pool) as u64 * raw
}

#[cfg(test)]
mod tests {
    use super::{
        DELTA_LANES, conv_ring_rows, delta_lane_bytes, dense_kv_bytes, kv_row_bytes,
        ple_ring_bytes, ple_ring_rows, pooled_rows, recurrent_bytes, selecting_bytes,
    };

    /// Qwen3.8's sizes: a GDN layer holds 48 heads of 128 × 128 f32 and
    /// eleven conv rows of 10,240 f32 channels; the PLE ring 17 rows of four
    /// 2,560-value streams; an attention position 2 × 2 × 256 f16 of K and V
    /// and a 128-value raw index key, and a 128-value pooled key per four
    /// positions, the last pool whole.
    #[test]
    fn qwen38_sizes() {
        assert_eq!(conv_ring_rows(4), 11);
        assert_eq!(ple_ring_rows(4, 3), 17);
        assert_eq!(recurrent_bytes(48, 16, 128, 4), 3_145_728 + 11 * 10_240 * 4);
        assert_eq!(ple_ring_bytes(4, 3, 4, 2560), 696_320);
        assert_eq!(kv_row_bytes(2, 256), 2048);
        assert_eq!(
            selecting_bytes(2, 256, 128, 4, 4096),
            4096 * 2304 + 1024 * 256
        );
        assert_eq!(
            selecting_bytes(2, 256, 128, 4, 4097),
            4097 * 2304 + 1025 * 256
        );
    }

    /// Qwen3.8's MTP layer: 2 × 2 × 256 f16 of K and V a position and no
    /// index key, at every position to the context.
    #[test]
    fn qwen38_mtp_store() {
        assert_eq!(dense_kv_bytes(2, 256, 4096), 8_388_608);
        assert_eq!(dense_kv_bytes(2, 256, 32_768), 67_108_864);
        assert_eq!(dense_kv_bytes(2, 256, 1), 2048);
        assert_eq!(dense_kv_bytes(2, 256, 0), 0);
    }

    /// Qwen3.8's lanes: past the one lane `recurrent_bytes` counts, three
    /// more lanes of 48 heads of 128 × 128 f32 and four u32 stamps; one lane
    /// adds only its stamp.
    #[test]
    fn qwen38_lanes() {
        assert_eq!(DELTA_LANES, 4);
        assert_eq!(
            delta_lane_bytes(48, 128, DELTA_LANES),
            3 * 3_145_728 + 4 * 4
        );
        assert_eq!(delta_lane_bytes(48, 128, 1), 4);
    }

    #[test]
    #[should_panic(expected = "delta_lane_bytes: a state of at least one lane")]
    fn a_state_of_no_lane_is_refused() {
        let _ = delta_lane_bytes(48, 128, std::hint::black_box(0));
    }

    /// A conv of no tap has no reach: both ring sizes refuse it by name
    /// rather than wrap to a ring of `usize::MAX` rows.
    #[test]
    #[should_panic(expected = "conv_ring_rows: a conv of at least one tap")]
    fn a_conv_of_no_tap_is_refused() {
        let _ = conv_ring_rows(std::hint::black_box(0));
    }

    #[test]
    #[should_panic(expected = "ple_ring_rows: a conv of at least one tap")]
    fn a_ple_conv_of_no_tap_is_refused() {
        let _ = ple_ring_rows(std::hint::black_box(0), 3);
    }

    /// The pooled plane counts a partial last pool whole and nothing past it.
    #[test]
    fn pooled_rows_round_up() {
        for (ctx, want) in [
            (0usize, 0usize),
            (1, 1),
            (4, 1),
            (5, 2),
            (8, 2),
            (2051, 513),
        ] {
            assert_eq!(pooled_rows(ctx, 4), want, "ctx {ctx}");
        }
    }
}
