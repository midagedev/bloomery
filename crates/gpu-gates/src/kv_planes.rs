//! The cache planes of one GQA layer as the Qwen3.5 body allocates them, in allocation order. The
//! planes are the allocator's unit (a plan rounds each to the card's granule); the layer's raw
//! bytes are their sum, which the description's layout counts (`KvLayout::layer_bytes`).

use bloomery_gpu::rope_neox::q8_plane_lens;

/// One GQA layer's cache planes' bytes at `rows` positions onto `planes`, in the order the body
/// allocates them: f16's `k` and `v`, or with `q8` (the cache in q8_0's two-plane layout) `kq,kd,vq,vd` —
/// the buffers `KvPlanes::new` zeroes, their lengths and widths (`q8_plane_lens`'s u32 words and
/// u16 scales) that owner's own.
pub fn layer_planes(planes: &mut Vec<u64>, n_kv: usize, head: usize, q8: bool, rows: u64) {
    if q8 {
        let rows = usize::try_from(rows).expect("a cache of usize rows");
        let (words, scales) = q8_plane_lens(head, n_kv, rows);
        planes.push(4 * words as u64);
        planes.push(2 * scales as u64);
        planes.push(4 * words as u64);
        planes.push(2 * scales as u64);
    } else {
        let bytes = n_kv as u64 * rows * head as u64 * 2;
        planes.push(bytes);
        planes.push(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::stores::kv_plane_bytes;

    const KV_HEADS: usize = 8;
    const HEAD: usize = 128;

    /// A layer's planes are two (f16) or four (q8_0), and their sum is the byte count the runtime's
    /// store formula gives (the one the description's layout counts), so the allocator's list and
    /// the verdict's sum are one number.
    #[test]
    fn the_planes_of_a_layer_sum_to_the_stores_bytes() {
        for rows in [32_u64, 4352] {
            for (q8, count) in [(false, 2), (true, 4)] {
                let mut planes = Vec::new();
                layer_planes(&mut planes, KV_HEADS, HEAD, q8, rows);
                assert_eq!(planes.len(), count, "q8 {q8}");
                assert_eq!(
                    planes.iter().sum::<u64>(),
                    kv_plane_bytes(KV_HEADS, HEAD, q8, rows),
                    "q8 {q8} at {rows} rows"
                );
            }
        }
    }
}
