// Inkling 2's kernel geometry: the numbers the family's new `.metal` drafts
// are written at — the rel-bias flash (`rel_flash.metal`), the normed mix
// (`res_mix.metal`), the loop controller (`loop_control.metal`), the residual
// short conv (`conv_res.metal`), the qk clip epilogue (`qk_clip.metal`), the
// NVFP4 mat-vec (`nvfp4.metal`) and the situ pair (`situ.metal`) — rendered
// into every translation unit's header as the `BLOOMERY_INK2_*` defines.
// This file is `include!`d by `build.rs`, so it holds no inner doc comment;
// the module's own doc lives at its declaration in `lib.rs`.
//
// Every number is a family fact of the Mini's file (the description its
// reader holds: 24 query heads over 8 key heads of 128 values, a window and
// a bias bank of 1024, `d_rel` 16, 4 conv taps, 128 sigmoid-biased experts of
// 1152 values at the top 8, a looped core of rank 64 over a hidden of 2304),
// the same in every unit because no registered instance carries it yet:
// `instance::INKLING2_MINI` is held out of `REGISTERED` until the shape
// table carries its router's 128-expert top-8 row. The Small's wider hidden
// is the one axis these kernels do not take whole: the controller's staging
// does not fit a threadgroup at 4,608 (`controller_holds`).
//
// The geometry of both Macs the family targets (an M4 Pro and an M1 Pro) is
// the same where these kernels care: a SIMD group of 32 lanes, a threadgroup
// of at most 1024 threads and at most `THREADGROUP_BYTES` of threadgroup
// memory. What differs between them is the dispatch cost and the byte wall
// the counts in `arch::inkling2::count` divide by, not any number here.

use crate::geom::KernelRefused;

/// The threadgroup memory one threadgroup may hold: 32 KiB on this backend's
/// devices (Apple GPUs), the bound every staged buffer below is held to.
pub const THREADGROUP_BYTES: usize = 32 * 1024;

/// The Mini's residual width: the mix's, the controller's and the router's
/// columns, the head's input.
pub const HIDDEN: usize = 2304;

/// Query heads a layer.
pub const QUERY_HEADS: usize = 24;

/// Key and value heads a layer.
pub const KV_HEADS: usize = 8;

/// Values a head: the key, the query and the value width.
pub const HEAD: usize = 128;

/// Query heads a key head: the flash's SIMD groups, one a query head.
pub const GROUP: usize = QUERY_HEADS / KV_HEADS;

/// Values one flash lane holds: a quarter of the head, one 8-byte word of
/// f16 cache values.
pub const LANE_DIMS: usize = HEAD / 32;

/// Threads in the rel-bias flash's threadgroup: the key head's group of
/// query heads, one SIMD group each, the decode attention's own shape at a
/// group of three.
pub const FLASH_THREADS: usize = GROUP * 32;

/// `d_rel`: values a head of the r projection, and the bias table's first
/// axis. The flash stages the tile's `[D_REL][TILE]` slice in threadgroup
/// memory and keeps the r row in registers.
pub const D_REL: usize = 16;

/// The bias bank's width: the windowed layers' window and the windowless
/// layers' `relative_extent`, 1024 of the Mini's file. A key distance at or
/// past it reads the pad column 0 (the launch uploads no column for it).
pub const EXTENT: usize = 1024;

/// The windowed layers' window, in positions.
pub const WINDOW: usize = 1024;

/// Bytes of the bias tile the flash stages a key tile: `D_REL × TILE` f32,
/// TILE being the decode attention's own tile.
pub const BIAS_TILE_BYTES: usize = D_REL * crate::attention_geom::TILE * 4;

/// Bytes of threadgroup memory the rel-bias flash's threadgroup holds: the
/// group's query rows (`GROUP × HEAD` f32, the q prepare's scratch) and the
/// bias tile.
pub const FLASH_THREADGROUP_BYTES: usize = GROUP * HEAD * 4 + BIAS_TILE_BYTES;

/// The f16 K and V bytes a windowed visit's flash reads: the window's rows
/// over every key head, K and V both.
pub const WINDOWED_KV_BYTES: usize = WINDOW * KV_HEADS * HEAD * 2 * 2;

/// The f16 K and V bytes a windowless visit's flash reads at `depth` live
/// positions: every position's rows over every key head, K and V both.
#[must_use]
pub fn global_kv_bytes(depth: usize) -> usize {
    depth * KV_HEADS * HEAD * 2 * 2
}

/// The most vectors one mix reads: the banked block inputs and the running
/// stream. The Mini's deepest bank is the loop's: a fresh block every sixth
/// visit over the coda's run, nine banks and the stream at the final norm;
/// the coda layers run at six and the message mixes of a core visit at
/// five, so the bound holds them all with room the launch counts down from.
pub const MIX_VECS: usize = 16;

/// The most query rows one mix scores in one launch: a windowless visit's
/// three input norms (q, k and v) beside the two message mixes' tables.
pub const MIX_NQ: usize = 4;

/// Threads in the mix's threadgroup: one a mix point, the norm's own
/// cooperative-load width.
pub const MIX_THREADS: usize = 256;

/// Bytes of threadgroup memory the mix's threadgroup holds: the
/// probabilities (`MIX_NQ × MIX_VECS` f32) and the vectors' RMS scales
/// (`MIX_VECS` f32). The states themselves stream from device memory twice
/// (the score pass, then the weighted sum), as the entry's body does.
pub const MIX_THREADGROUP_BYTES: usize = (MIX_NQ * MIX_VECS + MIX_VECS) * 4;

/// The looped core's gate rank: the width of the controllers' `h`.
pub const CONTROLLER_RANK: usize = 64;

/// Threads in the controller's threadgroup: one a row.
pub const CONTROLLER_THREADS: usize = 256;

/// Bytes of threadgroup memory the controller's threadgroup stages at a
/// hidden of `hidden`: the raw `[cur; msg]` row, `2 · hidden` f32. The Mini's
/// 2,304 fits a threadgroup whole; the Small's 4,608 does not (the staging
/// splits, a kernel the family does not carry yet).
#[must_use]
pub const fn controller_stage_bytes(hidden: usize) -> usize {
    2 * hidden * 4
}

/// The controller's geometry at a hidden of `hidden`, or the refusal naming
/// the bound: the staged row must fit one threadgroup's memory.
///
/// # Errors
///
/// [`KernelRefused`] naming the hidden width when the staging does not fit.
pub fn controller_holds(hidden: usize) -> Result<(), KernelRefused> {
    let bytes = controller_stage_bytes(hidden);
    if bytes <= THREADGROUP_BYTES {
        return Ok(());
    }
    Err(KernelRefused::new(
        "loop_control",
        "hidden",
        hidden,
        "at most 4096 (the staged [cur; msg] row in 32 KiB)",
    ))
}

/// The conv taps of the Mini's four short convolutions.
pub const TAPS: usize = 4;

/// The most taps the conv entry is launched with; the host refuses a wider
/// filter by name. The loops run over `taps` and no table is sized by it.
pub const MAX_TAPS: usize = 16;

/// Threads in the conv's threadgroup: one a channel.
pub const CONV_THREADS: usize = 128;

/// The conv's threadgroups along `channels` channels.
#[must_use]
pub fn conv_groups(channels: usize) -> usize {
    channels.div_ceil(CONV_THREADS)
}

/// The channel bases of a visit's four short convolutions in its one ring:
/// the k conv over the key heads' channels, the v conv over the same, the
/// attention output's over the hidden and the feed-forward block's over the
/// hidden — `2·kv·head + 2·hidden` channels in all, the reference's one
/// recurrent state buffer a visit.
pub const CONV_CHANNELS: usize = 2 * KV_HEADS * HEAD + 2 * HIDDEN;

/// The qk clip epilogue's rows a token: the query heads', the key heads',
/// the value heads' and the relative heads', each once, one SIMD group a
/// row.
pub const CLIP_ROWS: usize = 2 * (QUERY_HEADS + KV_HEADS);

/// Threads in the qk clip epilogue's threadgroup: eight SIMD groups of rows,
/// the mix's own width, so the visit's rows are two threadgroups.
pub const CLIP_THREADS: usize = 256;

/// The qk clip epilogue's threadgroups a token.
pub const CLIP_GROUPS: usize = CLIP_ROWS / (CLIP_THREADS / 32);

/// The experts the Mini's biased sigmoid router routes over.
pub const EXPERTS: usize = 128;

/// The experts a token keeps.
pub const TOP_K: usize = 8;

/// The least multiple the router variant's hidden width must be a multiple
/// of: one f32x4 a lane a pass over the 32 lanes, the norm's cooperative
/// load taking a short last pass over any multiple of it (the Mini's 2,304
/// is eighteen such passes exactly).
pub const ROUTER_WIDTH_STEP: usize = 128;

/// The router variant's hidden rule at `hidden`, or the refusal naming the
/// step. The registered instances' own router takes a multiple of 1024 (the
/// norm's lanes covering the input exactly); this variant is the family's
/// own kernel, and its launch counts the short last pass the way the rows
/// norm does.
///
/// # Errors
///
/// [`KernelRefused`] naming the hidden width when it is not a multiple of
/// [`ROUTER_WIDTH_STEP`].
pub fn router_holds(hidden: usize) -> Result<(), KernelRefused> {
    if hidden.is_multiple_of(ROUTER_WIDTH_STEP) {
        return Ok(());
    }
    Err(KernelRefused::new(
        "router_sigmoid",
        "hidden",
        hidden,
        "a multiple of 128 (a float4 a lane)",
    ))
}

/// Values one NVFP4 block holds: four 16-value sub-blocks.
pub const NVFP4_VALUES: usize = 64;

/// Bytes one NVFP4 block holds: the four sub-blocks' UE4M3 scale bytes and
/// 32 code bytes.
pub const NVFP4_BYTES: usize = 36;

/// The bytes an NVFP4 stack of `rows` rows over `k` columns holds: the
/// blocks of a row, `k / NVFP4_VALUES` of them, the rows stacked per expert.
/// `None` when `k` is not a whole number of blocks or the product does not
/// fit.
#[must_use]
pub fn nvfp4_stack_bytes(rows: usize, k: usize) -> Option<usize> {
    if !k.is_multiple_of(NVFP4_VALUES) {
        return None;
    }
    rows.checked_mul(k / NVFP4_VALUES)?.checked_mul(NVFP4_BYTES)
}

const _: () = {
    assert!(
        GROUP * KV_HEADS == QUERY_HEADS,
        "the group covers the heads"
    );
    assert!(LANE_DIMS * 32 == HEAD, "the lanes cover the head");
    assert!(
        WINDOW > crate::attention_geom::ONE_PASS_KEYS,
        "the window is past the one-pass waterline"
    );
    assert!(
        FLASH_THREADGROUP_BYTES <= THREADGROUP_BYTES,
        "the flash's threadgroup memory fits"
    );
    assert!(
        MIX_THREADGROUP_BYTES <= THREADGROUP_BYTES,
        "the mix's threadgroup memory fits"
    );
    assert!(MIX_VECS > 9, "the loop's deepest bank and the stream");
    assert!(
        controller_stage_bytes(HIDDEN) <= THREADGROUP_BYTES,
        "the Mini's controller staging fits a threadgroup"
    );
    assert!(
        CLIP_ROWS.is_multiple_of(CLIP_THREADS / 32),
        "the clip rows fill whole threadgroups"
    );
    assert!(D_REL <= 32, "the r row sits whole in one lane's registers");
    assert!(
        TAPS <= MAX_TAPS,
        "the file's taps are inside the entry's bound"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The flash's threadgroup at the Mini's axes: the group of three's 96
    /// threads, its staged bytes, and the f16 K and V a windowed visit reads
    /// (the window over every key head, K and V both, f16 a value). Mutant:
    /// `GROUP` is the query heads, or the bias tile is staged a key.
    #[test]
    fn the_flash_threadgroup_and_bytes() {
        assert_eq!((GROUP, FLASH_THREADS, LANE_DIMS), (3, 96, 4));
        assert_eq!(BIAS_TILE_BYTES, 1024, "16 x 16 f32");
        assert_eq!(
            FLASH_THREADGROUP_BYTES, 2560,
            "the scratch and the bias tile"
        );
        assert_eq!(
            WINDOWED_KV_BYTES, 4_194_304,
            "the window's rows, K and V, f16"
        );
        assert_eq!(global_kv_bytes(4096), 16_777_216, "every live row at depth");
    }

    /// The segments the visits' keys take: the window's 1,024 keys are past
    /// the decode attention's one-pass waterline, so the split path runs its
    /// segment cap and the fold joins the sink in the last threadgroup of
    /// each key head — one dispatch a visit at any depth, the grid the key
    /// heads times the cap. Mutant: the window taken as one segment.
    #[test]
    fn the_visits_keys_split_at_the_segment_cap() {
        use crate::attention_geom::{SEGMENT_CAP, segments_for_keys};
        assert_eq!(segments_for_keys(WINDOW as u32), SEGMENT_CAP as u32);
        assert_eq!(
            KV_HEADS * SEGMENT_CAP,
            128,
            "the flash's grid: the key heads times the cap"
        );
        // The windowless visits at a depth inside the waterline stay one
        // segment; past it they take the cap like the windowed ones.
        assert_eq!(segments_for_keys(512), 1);
        assert_eq!(segments_for_keys(262_144), SEGMENT_CAP as u32);
    }

    /// The controller stages the Mini's hidden whole and refuses the Small's:
    /// the staged `[cur; msg]` row is `2 · hidden` f32, 18 KiB at the Mini's
    /// 2,304 and 36 KiB at the Small's 4,608, over a threadgroup's 32 KiB.
    /// The staged row is the one place a family width meets the threadgroup
    /// bound. Mutant: the staging counts the hidden once.
    #[test]
    fn the_controller_stages_the_mini_and_refuses_the_small() {
        assert_eq!(controller_stage_bytes(HIDDEN), 2 * 2304 * 4);
        assert_eq!(controller_holds(HIDDEN), Ok(()));
        let refused = controller_holds(4608).expect_err("the Small's staging");
        assert_eq!(
            (refused.kernel, refused.axis, refused.value.as_str()),
            ("loop_control", "hidden", "4608")
        );
        assert!(refused.to_string().contains("at most 4096"), "{refused}");
        // The bound itself: the widest hidden whose staging fits.
        assert_eq!(controller_holds(4096), Ok(()));
        assert!(controller_holds(4100).is_err());
    }

    /// The mix's bounds: the bank and the stream under the vector cap, the
    /// query rows under the launch's, and the probabilities and scales the
    /// threadgroup holds while the states stream from device memory twice.
    /// Mutant: the probabilities staged a vector, not a query row.
    #[test]
    fn the_mixs_threadgroup_holds_the_probabilities_and_scales() {
        assert_eq!(MIX_NQ, 4, "three input norms beside the two message tables");
        assert_eq!(
            MIX_THREADGROUP_BYTES, 320,
            "a probability a (query row, vector), a scale a vector"
        );
        // The states stream twice: a mix point reads every vector's values
        // once for its two sums and once for the weighted sum.
        assert_eq!(2 * MIX_VECS * HIDDEN * 4, 294_912);
    }

    /// The conv's bounds: the visit's four convolutions read disjoint ranges
    /// of one ring of the reference's channel width, a threadgroup along
    /// every 128 channels, and the entry's tap bound holds the file's four
    /// with the headroom the launch checks. Mutant: a conv's range in the
    /// ring is its own channels, not its base in the visit's.
    #[test]
    fn the_convs_read_disjoint_ranges_of_one_ring() {
        assert_eq!(
            CONV_CHANNELS, 6656,
            "the k and v channels and the two hidden-width outputs"
        );
        assert_eq!(conv_groups(CONV_CHANNELS), 52);
        // The widest conv is a hidden-width one; the k and v convs are half
        // that.
        assert_eq!(conv_groups(HIDDEN), 18);
        assert_eq!(conv_groups(KV_HEADS * HEAD), 8);
    }

    /// The qk clip epilogue covers a visit's rows in whole threadgroups: the
    /// query, key, value and relative heads' rows, one SIMD group each, two
    /// threadgroups of eight. Mutant: a row kind counted once.
    #[test]
    fn the_clip_covers_the_visits_rows_in_whole_threadgroups() {
        assert_eq!(CLIP_ROWS, 2 * (QUERY_HEADS + KV_HEADS));
        assert_eq!((CLIP_ROWS, CLIP_GROUPS), (64, 8));
    }

    /// The router variant's width rule: the Mini's 2,304 is a multiple of
    /// the lane's float4 step, and a width that is not is refused by name —
    /// the registered instances' router holds its own, stricter multiple.
    /// Mutant: the rule dropped, so any width launches.
    #[test]
    fn the_router_variant_takes_the_minis_width() {
        assert_eq!(router_holds(HIDDEN), Ok(()));
        assert!(HIDDEN.is_multiple_of(ROUTER_WIDTH_STEP));
        let refused = router_holds(2300).expect_err("a width between the steps");
        assert_eq!(
            (refused.kernel, refused.axis, refused.value.as_str()),
            ("router_sigmoid", "hidden", "2300")
        );
        assert!(
            refused.to_string().contains("a multiple of 128"),
            "{refused}"
        );
    }

    /// The NVFP4 block: 36 bytes of 64 values, four sub-blocks of sixteen
    /// under their own UE4M3 scale, and the stack bytes of the Mini's fused
    /// expert rows — the per-expert f32 scale the launch applies after the
    /// dot is one word an expert, beside the blocks.
    #[test]
    fn the_nvfp4_block_and_stack_bytes() {
        assert_eq!((NVFP4_VALUES, NVFP4_BYTES), (64, 36));
        // The Mini's fused w13 stack an expert: 2,304 rows (the gate rows
        // above the up rows) over the hidden's columns.
        assert_eq!(
            nvfp4_stack_bytes(2 * 1152, HIDDEN),
            Some(2 * 1152 * (HIDDEN / NVFP4_VALUES) * NVFP4_BYTES)
        );
        assert_eq!(
            nvfp4_stack_bytes(1152, HIDDEN),
            Some(1152 * 36 * HIDDEN / 64)
        );
        assert_eq!(nvfp4_stack_bytes(1, 32), None, "a partial block");
        assert_eq!(nvfp4_stack_bytes(1, 63), None, "a partial block");
    }
}
