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
pub const SIMD_GROUPS: usize = 4;

/// The width of a SIMD group.
pub const SIMD_WIDTH: usize = 32;

/// Query heads one SIMD group holds.
pub const HEADS_PER_SG: usize = 2;

/// Values one lane holds. The SIMD group covers the head.
pub const LANE_DIMS: usize = HEAD_DIM / SIMD_WIDTH;

/// One segment partial: the max, the weight sum, then the head.
pub const PARTIAL_ELEMS: usize = HEAD_DIM + 2;

/// One head of the query projection: the query, then the gate.
pub const QG_STRIDE: usize = HEAD_DIM * 2;

/// Keys one segment of the token count holds. A count past this is a second
/// dispatch, the merge.
pub const SEGMENT_KEYS: usize = 64;

/// Threads in the flash threadgroup.
pub const FLASH_THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// Threads in the merge threadgroup: one lane per [`LANE_DIMS`] values.
pub const MERGE_THREADS: usize = SIMD_WIDTH;

/// `1/sqrt(256)`, exact.
pub const SCALE: f32 = 0.0625;

const _: () = {
    assert!(GQA == QUERY_HEADS / KV_HEADS);
    assert!(HEADS_PER_SG * SIMD_GROUPS == GQA);
    assert!(LANE_DIMS == 8);
    assert!(PARTIAL_ELEMS == HEAD_DIM + 2);
    assert!(QG_STRIDE == HEAD_DIM * 2);
    assert!(FLASH_THREADS == 128);
    assert!(SCALE == 1.0 / 16.0);
};

/// Segments a live key count takes. Each holds [`SEGMENT_KEYS`] keys. Zero
/// keys are still one segment: the flash raises on that count.
#[must_use]
pub fn segments_for_keys(n_keys: u32) -> u32 {
    let n = u64::from(n_keys).div_ceil(SEGMENT_KEYS as u64);
    u32::try_from(n.max(1)).expect("a segment count fits u32")
}

/// Dispatches one attention layer encodes for `segments`.
///
/// One segment is the flash, which normalizes and applies the gate. More
/// than one is the flash and then the merge. The token count and the launch
/// both read this.
#[must_use]
pub fn attention_dispatches(segments: u32) -> u32 {
    if segments <= 1 { 1 } else { 2 }
}
