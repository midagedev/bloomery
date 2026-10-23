/// Values in one head.
pub const HEAD_DIM: usize = 256;

/// The partial rope: the first 64 values turn, in 32 pairs.
pub const ROT: usize = 64;

/// Query heads.
pub const QUERY_HEADS: usize = 16;

/// Key and value heads.
pub const KV_HEADS: usize = 2;

/// Query heads per key head.
pub const GQA: usize = QUERY_HEADS / KV_HEADS;

/// Keys in one K/V tile.
pub const TILE: usize = 16;

/// SIMD groups in the flash threadgroup.
pub const SIMD_GROUPS: usize = 8;

/// The width of a SIMD group.
pub const SIMD_WIDTH: usize = 32;

/// Query heads one SIMD group holds.
pub const HEADS_PER_SG: usize = 1;

/// Values one lane holds. The SIMD group covers the head.
pub const LANE_DIMS: usize = HEAD_DIM / SIMD_WIDTH;

/// One segment partial: the max, the weight sum, then the head.
pub const PARTIAL_ELEMS: usize = HEAD_DIM + 2;

/// One head of the query projection: the query, then the gate.
pub const QG_STRIDE: usize = HEAD_DIM * 2;

/// The live-key ceiling one segment holds. At or under it the flash is one
/// dispatch that finishes in its epilogue; past it the segment rule splits
/// the keys and the flash's last threadgroup folds the partials (the count
/// model reads this as the waterline its two scenes sit either side of).
pub const SEGMENT_KEYS: usize = 64;

/// Threads in the flash threadgroup.
pub const FLASH_THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// `1/sqrt(256)`, exact.
pub const SCALE: f32 = 0.0625;

const _: () = {
    assert!(GQA == QUERY_HEADS / KV_HEADS);
    assert!(HEADS_PER_SG * SIMD_GROUPS == GQA);
    assert!(LANE_DIMS == 8);
    assert!(PARTIAL_ELEMS == HEAD_DIM + 2);
    assert!(QG_STRIDE == HEAD_DIM * 2);
    assert!(FLASH_THREADS == 256);
    assert!(SCALE == 1.0 / 16.0);
};

/// Segments a live key count takes.
///
/// At or under [`SEGMENT_KEYS`] the keys are one segment: the flash is one
/// dispatch that finishes in its epilogue, and the count model's waterline
/// holds. Past it, while a tile a segment still fills more of the card than
/// two tiles a segment would (up to twice [`SEGMENT_KEYS`]), the keys split
/// a tile a segment; past that, two tiles a segment, capped so the partials
/// the split writes and the fold reads stay a small share of the K/V bytes
/// at depth. Zero keys are still one segment: the flash raises on that
/// count.
#[must_use]
pub fn segments_for_keys(n_keys: u32) -> u32 {
    let n = u64::from(n_keys);
    let segments = if n <= SEGMENT_KEYS as u64 {
        1
    } else if n <= 2 * SEGMENT_KEYS as u64 {
        n.div_ceil(TILE as u64)
    } else {
        n.div_ceil((SEGMENT_KEYS / 2) as u64)
            .min(SEGMENT_CAP as u64)
    };
    u32::try_from(segments.max(1)).expect("a segment count fits u32")
}

/// The most segments a key count takes. The partials are
/// `QUERY_HEADS * SEGMENT_CAP * PARTIAL_ELEMS` values, written by the flash
/// and read by its fold; at the cap they are an eighth of the K/V bytes of
/// the keys that reach it.
pub const SEGMENT_CAP: usize = 64;

/// `attn_span`'s rule (attention.metal): the keys a segment holds, rounded
/// up to a tile, with the empty-count floor. This mirror exists so the
/// coverage test below can hold the Rust rule and the kernel's span to one
/// property; the kernel's own copies are the gate's scenes to catch.
#[cfg(test)]
fn span_mirror(live: u32, segments: u32) -> u32 {
    if live == 0 || segments == 0 {
        return TILE as u32;
    }
    let per = live.div_ceil(segments);
    per.div_ceil(TILE as u32) * TILE as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_segment_holds_the_waterline() {
        for n in [1u32, 2, 63, 64] {
            assert_eq!(segments_for_keys(n), 1, "{n} keys are one segment");
        }
    }

    #[test]
    fn the_rule_picks_the_counted_splits() {
        // The counted splits: a tile a segment to twice the waterline, then
        // two tiles a segment to the cap.
        for (n, want) in [
            (65u32, 5u32),
            (100, 7),
            (128, 8),
            (129, 5),
            (256, 8),
            (512, 16),
            (1024, 32),
            (2048, 64),
            (4096, 64),
            (8192, 64),
        ] {
            assert_eq!(segments_for_keys(n), want, "{n} keys");
        }
    }

    #[test]
    fn the_segments_cover_every_count() {
        for n in 1..=8192u32 {
            let segments = segments_for_keys(n);
            let span = span_mirror(n, segments);
            // The segments tile [0, segments * span) contiguously, so the
            // count is covered when the tiling reaches its end. A tile
            // rounding can leave empty trailing segments; the fold skips
            // their zero-weight partials, so only the cover matters here.
            assert!(
                u64::from(segments) * u64::from(span) >= u64::from(n),
                "{n} keys in {segments} segments of {span}"
            );
        }
    }
}
