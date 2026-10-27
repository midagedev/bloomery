use core::fmt;

use crate::geom::{Geometry, KernelRefused, yes_no};
use crate::instance::{HOST, Instance};

/// Values in one head. A shim: the build's host instance's. Callers left:
/// the arch, the e2e and kernel gates, their oracles and the probes.
pub const HEAD_DIM: usize = HOST.key.attn.row.head as usize;

/// The partial rope: the values of a head that turn, in pairs from the
/// first. A shim: the build's host instance's. Callers left: the arch, the
/// kernel gate, its oracles and the probes.
pub const ROT: usize = HOST.key.attn.rot as usize;

/// Query heads. A shim: the build's host instance's. Callers left: the arch,
/// the kernel gate and the probes.
pub const QUERY_HEADS: usize = HOST.key.attn.n_head as usize;

/// Key and value heads. A shim: the build's host instance's. Callers left:
/// the arch, the e2e and kernel gates and the probes.
pub const KV_HEADS: usize = HOST.key.attn.n_kv as usize;

/// Query heads per key head of `inst`.
#[must_use]
pub const fn gqa(inst: &Instance) -> usize {
    (inst.key.attn.n_head / inst.key.attn.n_kv) as usize
}

/// Keys in one K/V tile.
pub const TILE: usize = 16;

/// SIMD groups in the flash threadgroup of `inst`: its group of query
/// heads, [`HEADS_PER_SG`] a SIMD group.
#[must_use]
pub const fn simd_groups(inst: &Instance) -> usize {
    gqa(inst) / HEADS_PER_SG
}

/// The width of a SIMD group.
pub const SIMD_WIDTH: usize = 32;

/// Query heads one SIMD group holds.
pub const HEADS_PER_SG: usize = 1;

/// Values one lane of `inst`'s flash holds: a SIMD group is one query head,
/// its lanes cover the head.
#[must_use]
pub const fn lane_dims(inst: &Instance) -> usize {
    inst.key.attn.row.head as usize / SIMD_WIDTH
}

/// One segment partial of a head of `head` values: the max, the weight sum,
/// then the head.
#[must_use]
pub const fn partial_elems(head: usize) -> usize {
    head + 2
}

/// One head of `inst`'s query projection: the query, then the gate when it
/// writes one.
#[must_use]
pub const fn qg_stride(inst: &Instance) -> usize {
    inst.key.attn.row.head as usize * (1 + inst.key.attn.q_gate as usize)
}

/// One head of the query projection: the query, then the gate. A shim: the
/// build's host instance's. Callers left: the kernel gate.
pub const QG_STRIDE: usize = qg_stride(&HOST);

/// The live-key ceiling one segment holds. At or under it the flash is one
/// dispatch that finishes in its epilogue; past it the segment rule splits
/// the keys and the flash's last threadgroup folds the partials (the count
/// model reads this as the waterline its two scenes sit either side of).
pub const SEGMENT_KEYS: usize = 64;

/// Threads in the flash threadgroup of `inst`.
#[must_use]
pub const fn flash_threads(inst: &Instance) -> usize {
    SIMD_WIDTH * simd_groups(inst)
}

/// Threads in the flash threadgroup. A shim: the build's host instance's.
/// Callers left: the probes.
pub const FLASH_THREADS: usize = flash_threads(&HOST);

/// The score's scale for a head of `head` values: ik's
/// `1.0f/sqrtf(float(head))` in f32 (exact at 256; at 128 the f32 nearest,
/// the delta net's own query scale). A head with no written scale has NaN,
/// which [`holds`] refuses.
#[must_use]
pub const fn scale_of(head: usize) -> f32 {
    match head {
        256 => 0.0625,
        128 => f32::from_bits(0x3db5_04f3),
        _ => f32::NAN,
    }
}

/// The score's scale. A shim: the build's host instance's. Callers left: the
/// kernel gate's oracles and the probes.
pub const SCALE: f32 = scale_of(HEAD_DIM);

const _: () = assert!(
    !SCALE.is_nan(),
    "no score scale is written for the host's head"
);

/// Positions one prefill threadgroup holds: a SIMD group each, whose rows
/// are the group's query heads of one key head at that position.
pub const PREFILL_POSITIONS: usize = 4;

/// Keys one prefill tile holds: two 8-key fragments.
pub const PREFILL_KEYS: usize = 16;

/// Halves of padding after each row of a prefill tile, so the eight rows of
/// a fragment load fall in different banks.
pub const PREFILL_PAD: usize = 8;

/// Threads in the prefill threadgroup.
pub const PREFILL_THREADS: usize = PREFILL_POSITIONS * SIMD_WIDTH;

/// The geometry every flash kernel is written for, held for `g`: the
/// kernels' own `static_assert`s, stated where the host sizes its launches.
///
/// # Errors
///
/// [`KernelRefused`] naming the first axis the flash cannot take.
pub fn holds(g: &Geometry) -> Result<(), KernelRefused> {
    let a = &g.attn;
    let refuse = |axis, value: &dyn fmt::Display, needs| {
        Err(KernelRefused::new("attention", axis, value, needs))
    };
    // A lane holds 8 values (a 16-byte word) or 4 (an 8-byte word), the
    // norm's pair tree is head/64 warps, and the score's scale is written
    // for those heads.
    if a.lane_dims * SIMD_WIDTH != a.head
        || !matches!(a.lane_dims, 4 | 8)
        || !a.head.is_multiple_of(64)
        || scale_of(a.head).is_nan()
    {
        return refuse("head", &a.head, "128 or 256 values");
    }
    let row = &g.instance.key.attn.row;
    if row.value as usize != a.head {
        return refuse("value width", &row.value, "the head's width");
    }
    if a.gqa * a.kv_heads != a.query_heads {
        return refuse("query heads", &a.query_heads, "a multiple of the key heads");
    }
    // One SIMD group a query head, and 256 threads in the flash threadgroup.
    if HEADS_PER_SG * a.simd_groups != a.gqa || a.flash_threads != 256 || a.gqa != 8 {
        return refuse(
            "query heads a key head",
            &a.gqa,
            "8 (one SIMD group each in a 256-thread flash)",
        );
    }
    // The fold: one thread a 16 values of a head, every head at once.
    if a.query_heads * a.head != a.flash_threads * 16 {
        return refuse(
            "query heads x head",
            &(a.query_heads * a.head),
            "4096 (the fold's 256 threads of 16 values)",
        );
    }
    // The norm's pair tree (rope_neox's `owned_pair`): the turning pairs
    // inside warp 0 or every value turning.
    if !a.rot.is_multiple_of(2) || a.rot > a.head || (a.rot / 2 > SIMD_WIDTH && a.rot != a.head) {
        return refuse(
            "rope dims",
            &a.rot,
            "an even count of at most 64, or the whole head",
        );
    }
    // The shape table's row that cuts a window, folds sinks or multiplies the
    // value rows is a layer that asks for one: the flash attends every
    // position and appends the values as they are.
    if row.window || row.sinks || row.value_scale {
        return refuse(
            "flash row",
            &format_args!(
                "window {}, sinks {}, value scale {}",
                yes_no(row.window),
                yes_no(row.sinks),
                yes_no(row.value_scale)
            ),
            "a row with no window, no sinks and no value multiplier",
        );
    }
    Ok(())
}

const _: () = {
    assert!(PREFILL_POSITIONS * SIMD_WIDTH == 128);
    assert!(PREFILL_KEYS == 16);
    // A prefill tile's row stride keeps the head's 8-value alignment.
    assert!(PREFILL_PAD.is_multiple_of(8));
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
/// `query_heads * SEGMENT_CAP * partial_elems` values, written by the flash
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
    use crate::instance::REGISTERED;

    #[test]
    fn the_scale_is_iks_for_every_registered_head() {
        for inst in REGISTERED {
            let head = inst.key.attn.row.head;
            let ik = 1.0f32 / (head as f32).sqrt();
            assert_eq!(
                scale_of(head as usize).to_bits(),
                ik.to_bits(),
                "{}: head {head}",
                inst.name
            );
        }
    }

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
