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
/// heads, [`HEADS_PER_SG`] a SIMD group. The threadgroup is as many SIMD
/// groups as the key head has query heads.
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

/// Whether the flash of `inst` cuts a window a launch names: its shape-table
/// row does.
#[must_use]
pub const fn windowed(inst: &Instance) -> bool {
    inst.key.attn.row.window
}

/// Whether the layers of `inst` gate their output with one value a query
/// head, from a projection of their own, in a buffer the launch binds.
#[must_use]
pub const fn head_gate(inst: &Instance) -> bool {
    inst.key.attn.head_gate
}

/// Whether the flash of `inst` norms each query and key head by the file's
/// gains before its rope.
#[must_use]
pub const fn qk_norm(inst: &Instance) -> bool {
    inst.key.attn.qk_norm
}

/// Whether the rope of `inst` pairs the neighbours `2i, 2i + 1` (GGUF's NORM
/// pairing), not the NEOX halves.
#[must_use]
pub const fn norm_pairs(inst: &Instance) -> bool {
    inst.key.attn.norm_pairs
}

/// The live keys the count model and the e2e gate take as one segment's
/// worth: their sessions this deep run one segment (it is under
/// [`ONE_PASS_KEYS`]).
pub const SEGMENT_KEYS: usize = 64;

/// The live-key ceiling one segment holds. At or under it the f16 launch is
/// the one-pass flash, a threadgroup a query head over every key the layer
/// attends, which writes no partial; past it the keys split into
/// [`SEGMENT_CAP`] segments and the flash's last threadgroup folds the
/// partials. Past it the split's threadgroups a key head outrun the one
/// threadgroup a query head the one pass has.
pub const ONE_PASS_KEYS: usize = 512;

/// Threads in the flash threadgroup of `inst`: a SIMD group to each query
/// head of a key head.
#[must_use]
pub const fn flash_threads(inst: &Instance) -> usize {
    SIMD_WIDTH * simd_groups(inst)
}

/// Threads in the flash threadgroup. A shim: the build's host instance's.
/// Callers left: the probes.
pub const FLASH_THREADS: usize = flash_threads(&HOST);

/// SIMD groups in the one-pass flash's threadgroup (the f16 launch of one
/// segment): one threadgroup a query head, its groups taking the keys
/// strided. Groups 0, 1 and 2 prepare the query, the key and the value.
pub const ONE_PASS_SG: usize = 16;

/// Keys one SIMD group of the one-pass flash streams into registers at a
/// time, between two rescales of its softmax.
pub const ONE_PASS_CHUNK: usize = 4;

/// Threads in the one-pass flash's threadgroup.
pub const ONE_PASS_THREADS: usize = SIMD_WIDTH * ONE_PASS_SG;

/// The one-pass flash's threadgroup memory at a head of `head` values
/// (`ATTN_ONE_FLOATS` in attention.metal, in 16-byte words): the query, the
/// key's scratch, the key's and value's halves, and each group's maximum,
/// weight sum and values.
#[must_use]
pub const fn one_pass_bytes(head: usize) -> usize {
    (3 * head + ONE_PASS_SG * (2 + head)).div_ceil(4) * 16
}

const _: () = {
    assert!(
        ONE_PASS_SG >= 3,
        "the one-pass prepare takes three SIMD groups"
    );
    // A round of chunks spans a tile, so a key meets no more rescales than the
    // band's tile count.
    assert!(ONE_PASS_SG * ONE_PASS_CHUNK >= TILE);
};

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

/// Rows of a prefill fragment: the query heads of a key head, padded with
/// zero queries when the key head has fewer.
pub const PREFILL_ROWS: usize = 8;

/// The threadgroup memory one threadgroup may hold, in bytes.
const SHARED_BYTES: usize = 32 * 1024;

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
    // One SIMD group a query head, so a flash of the key head's group of
    // SIMD groups; the prefill's fragments hold at most eight of them.
    if HEADS_PER_SG * a.simd_groups != a.gqa
        || a.flash_threads != SIMD_WIDTH * a.simd_groups
        || !(1..=PREFILL_ROWS).contains(&a.gqa)
    {
        return refuse(
            "query heads a key head",
            &a.gqa,
            "1 to 8 (a SIMD group each in a flash of that many)",
        );
    }
    // The one-pass flash merges a thread a value, and holds the query, the
    // key's scratch, the key's and value's halves and every group's values in
    // one threadgroup's memory.
    if ONE_PASS_THREADS < a.head || one_pass_bytes(a.head) > SHARED_BYTES {
        return refuse(
            "head",
            &a.head,
            "at most the one-pass flash's threads, its values in 32 KB",
        );
    }
    // The prologue's two axes name the entries the kernels are: the head
    // norm's sum-of-squares walks the NEOX pair tree, so a head norm with
    // NORM pairs is no entry, and a norm-free prologue is compiled for the
    // neighbour pairs alone.
    if (a.qk_norm && a.norm_pairs) || (!a.qk_norm && !a.norm_pairs) {
        return refuse(
            "head norm and rope pairs",
            &format_args!(
                "head norm {}, NORM pairs {}",
                yes_no(a.qk_norm),
                yes_no(a.norm_pairs)
            ),
            "a head norm with NEOX pairs, or none with NORM pairs",
        );
    }
    // The norm's pair tree (rope_neox's `owned_pair`): the turning pairs
    // inside warp 0 or every value turning. The neighbour pairs turn within
    // one lane, so they take no tree.
    if !a.rot.is_multiple_of(2)
        || a.rot > a.head
        || (!a.norm_pairs && a.rot / 2 > SIMD_WIDTH && a.rot != a.head)
    {
        return refuse(
            "rope dims",
            &a.rot,
            "an even count of at most 64, or the whole head",
        );
    }
    // The shape table's row that folds sinks or multiplies the value rows is
    // a layer that asks for one: the flash folds no sinks and appends the
    // values as they are. A window is the launch's word.
    if row.sinks || row.value_scale {
        return refuse(
            "flash row",
            &format_args!(
                "window {}, sinks {}, value scale {}",
                yes_no(row.window),
                yes_no(row.sinks),
                yes_no(row.value_scale)
            ),
            "a row with no sinks and no value multiplier",
        );
    }
    // The gate is packed in the query or one value a head, not both.
    if a.gated && a.head_gate {
        return refuse(
            "output gate",
            &"packed in the query and per head",
            "packed in the query or per head",
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

/// Segments the keys a layer attends take: [`segments_for_keys`] of the last
/// `window` of its `n_keys` live keys, or of all of them with no window. A
/// segment the rule does not count holds no key of the window and is never
/// launched.
#[must_use]
pub fn segments_for_window(n_keys: u32, window: Option<u32>) -> u32 {
    segments_for_keys(window.map_or(n_keys, |w| n_keys.min(w)))
}

/// Segments a live key count takes.
///
/// At or under [`ONE_PASS_KEYS`] the keys are one segment: the f16 flash is
/// the one-pass launch, one dispatch that finishes in its epilogue. Past it
/// the keys split into [`SEGMENT_CAP`] segments at every depth. Zero keys are
/// still one segment: the flash raises on that count.
#[must_use]
pub fn segments_for_keys(n_keys: u32) -> u32 {
    if n_keys as usize <= ONE_PASS_KEYS {
        1
    } else {
        SEGMENT_CAP as u32
    }
}

/// The most segments a key count takes, and the count every split takes:
/// fewer leave the split's threadgroups too few to keep the card's cores
/// busy, more lengthen the fold, which reads every segment's partial of every
/// head in the dispatch's last threadgroup. The partials are
/// `query_heads * SEGMENT_CAP * partial_elems` values, a small share of the
/// K/V bytes of the keys past [`ONE_PASS_KEYS`].
pub const SEGMENT_CAP: usize = 16;

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

    /// Qwen3.6's instance with the attention axes of `n_head` heads over
    /// `n_kv` key heads of 256 values.
    fn shaped(n_head: u32, n_kv: u32) -> Instance {
        let mut inst = crate::instance::QWEN35MOE_A3B;
        inst.key.attn.row = models::shape::gqa_row(256, n_head / n_kv)
            .expect("the shape table has a row for the group");
        inst.key.attn.n_head = n_head;
        inst.key.attn.n_kv = n_kv;
        inst
    }

    #[test]
    fn the_flash_takes_the_groups_of_a_fragment() {
        // Qwen3.6-35B-A3B (8 a key head), Qwen3.8-27B (6) and a 16-head 9B
        // shape over 4 key heads (4), and the group of two.
        for (n_head, n_kv, gqa, threads) in [
            (16u32, 2u32, 8usize, 256usize),
            (24, 4, 6, 192),
            (16, 4, 4, 128),
            (8, 4, 2, 64),
        ] {
            let g = Geometry::of(&shaped(n_head, n_kv));
            assert_eq!(holds(&g), Ok(()), "{n_head} heads over {n_kv}");
            assert_eq!(g.attn.gqa, gqa, "{n_head}/{n_kv}: group");
            assert_eq!(g.attn.flash_threads, threads, "{n_head}/{n_kv}: threads");
        }
    }

    #[test]
    fn the_flash_refuses_a_group_past_a_fragment_and_folds_any_key_heads() {
        let wide = Geometry::of(&shaped(20, 2));
        let refused = holds(&wide).expect_err("ten query heads a key head");
        assert_eq!(
            (refused.kernel, refused.axis),
            ("attention", "query heads a key head")
        );
        // The fold walks the heads a SIMD group at a time, so three key heads
        // and 128 query heads are shapes it takes.
        assert_eq!(holds(&Geometry::of(&shaped(24, 3))), Ok(()), "24 over 3");
        assert_eq!(
            holds(&Geometry::of(&shaped(128, 16))),
            Ok(()),
            "128 over 16"
        );
    }

    #[test]
    fn the_flash_takes_group_four_at_head_128_with_a_window_and_a_head_gate() {
        let g = Geometry::of(&crate::instance::LFM3_ATTN);
        assert_eq!(holds(&g), Ok(()));
        let a = g.attn;
        assert_eq!(
            (a.head, a.rot, a.query_heads, a.kv_heads, a.gqa),
            (128, 128, 16, 4, 4)
        );
        assert_eq!((a.simd_groups, a.flash_threads, a.lane_dims), (4, 128, 4));
        assert!(a.windowed && a.head_gate && !a.gated);
        assert_eq!(a.qg_stride, a.head, "a head gate adds nothing to the query");
        // A whole head turns: thread t of each of the two warps owns the
        // pair (t, t + 64).
        assert_eq!(a.rot, a.head);
    }

    #[test]
    fn the_prologues_two_axes_name_the_entries_and_the_mixes_are_refused() {
        // Qwen3.6's attention at the llama axes: the no-norm, neighbour-paired
        // prologue, which holds.
        let mut llama = crate::instance::LLAMA_ATTN;
        llama.key.hidden = 2048;
        let g = Geometry::of(&llama);
        assert_eq!(holds(&g), Ok(()));
        assert!(g.attn.qk_norm == llama.key.attn.qk_norm && !g.attn.qk_norm);
        assert!(g.attn.norm_pairs);
        // A head norm over the neighbour pairs: the norm's sum-of-squares
        // walks the NEOX pair tree, so the mix is no entry.
        let mut mixed = llama;
        mixed.key.attn.qk_norm = true;
        let refused = holds(&Geometry::of(&mixed)).expect_err("a head norm with NORM pairs");
        assert_eq!(
            (refused.kernel, refused.axis),
            ("attention", "head norm and rope pairs")
        );
        assert!(refused.value.starts_with("head norm yes, NORM pairs yes"));
        // No head norm over the NEOX pairs: no entry either.
        let mut mixed = llama;
        mixed.key.attn.norm_pairs = false;
        let refused = holds(&Geometry::of(&mixed)).expect_err("no head norm with NEOX pairs");
        assert_eq!(
            (refused.kernel, refused.axis),
            ("attention", "head norm and rope pairs")
        );
        assert!(refused.value.starts_with("head norm no, NORM pairs no"));
    }

    #[test]
    fn a_window_is_the_launchs_and_sinks_and_two_gates_are_refused_by_name() {
        let mut inst = crate::instance::LFM3_ATTN;
        inst.key.attn.row.sinks = true;
        let refused = holds(&Geometry::of(&inst)).expect_err("a row with sinks");
        assert_eq!((refused.kernel, refused.axis), ("attention", "flash row"));
        assert!(refused.value.starts_with("window yes, sinks yes"));
        let mut inst = crate::instance::LFM3_ATTN;
        inst.key.attn.q_gate = true;
        let refused = holds(&Geometry::of(&inst)).expect_err("both gates");
        assert_eq!((refused.kernel, refused.axis), ("attention", "output gate"));
    }

    #[test]
    fn a_window_takes_the_segments_of_its_own_keys() {
        for (n_keys, window, want) in [
            (300u32, Some(256u32), 1u32),
            (1025, Some(256), 1),
            (20_000, Some(256), 1),
            (20_000, None, 16),
            (100, Some(256), 1),
            (20_000, Some(512), 1),
            (20_000, Some(513), 16),
            (1025, Some(1), 1),
            (64, Some(64), 1),
        ] {
            assert_eq!(
                segments_for_window(n_keys, window),
                want,
                "{n_keys} keys, window {window:?}"
            );
        }
    }

    #[test]
    fn one_segment_holds_the_waterline() {
        for n in [1u32, 2, 63, 64, 65, 256, 511, 512] {
            assert_eq!(segments_for_keys(n), 1, "{n} keys are one segment");
        }
    }

    #[test]
    fn the_rule_picks_the_counted_splits() {
        // Every count past the waterline is the cap.
        for (n, want) in [(513u32, 16u32), (4096, 16), (16_385, 16), (262_144, 16)] {
            assert_eq!(segments_for_keys(n), want, "{n} keys");
        }
    }

    #[test]
    fn the_segments_cover_every_count() {
        for n in 1..=20_000u32 {
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
