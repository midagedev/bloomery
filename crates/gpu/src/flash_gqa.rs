//! Grouped-query flash attention for `m` query rows — one decode position,
//! or the consecutive positions of a prompt chunk: [`GROUP`] query heads
//! share one key/value head of [`HEAD`] values, and the cache is a pair of
//! f16 planes, `[n_kv][ctx][HEAD]` each (the layout
//! `rope_neox::head_norm_neox_append` writes). Row `t` sees the keys below
//! its own live count `n_keys[t]` — the causal limit of its position. The
//! segment + merge shape of `flash.rs`: the key range is cut into
//! [`SEGMENTS`] segments, block `(segment, kv head, row)` walks its
//! segment for all the group's query heads of that row at once — one warp
//! per query head over one staging of the K and V tiles — and writes per
//! head the softmax partials `(max, Σ exp, Σ exp·V)`; the merge folds a
//! head's live segments in ascending order. A row's arithmetic does not
//! depend on `m` or on the other rows: one row is the one-position launch.
//! The segment count comes from [`SEGMENTS`], not the cache height, so a
//! captured graph's grid is fixed whatever context the load allocated; each
//! block computes its key range from its row's own live count through
//! [`seg_span`], a segment wholly past that count writes the
//! neutral partial (`m = −inf`, `s = 0`) and reads no key row, and the merge
//! stops at the row's last live segment.
//!
//! A live count of zero or past the cache has no defined attention. The
//! segment passes and the merge read the count through one rule
//! ([`live_keys`]): such a row reads no key, and the merge raises
//! [`FaultSite::KeyCount`] and writes NaN into the row's output.
//!
//! Reduction structure, the fixed contract of this kernel (reruns and
//! replays are bit-identical; bit identity with the reference is not
//! claimed — its exponential is another function):
//! - the score of key `j` is lane `j`'s dot of the query row with the key
//!   row over [`ILP`] rotating f32 partials (partial `p` takes the value
//!   pairs `p, p + ILP, …`, both values of a pair by one fused multiply-add
//!   each), combined as `(a0 + a1) + (a2 + a3)`, then one multiply by
//!   `scale`;
//! - online softmax in [`KEY_TILE`]-key tiles, keys ascending: the tile max
//!   by the five-step butterfly, the running max bumped, each weight
//!   `dev_exp(s − m)`, the weight sum by the five-step butterfly, the
//!   running sum and the value partials rescaled by `dev_exp(m_old − m)` on
//!   a bump;
//! - lane `l` accumulates dims `4l .. 4l + 3` over a tile's keys ascending,
//!   one fused multiply-add per key;
//! - the merge: `flash::online_fold` over the live segments ascending, then
//!   `r · (1/s)`.
//!
//! Two segment passes share everything after the scores (`fold_tile`) and
//! the merge: [`flash_gqa_kernels::gqa_flash_seg`] computes each score as
//! the f32 dot above, [`flash_gqa_kernels::gqa_flash_seg_mma`] on the tensor
//! cores from the query rounded to f16 (`mma.m16n8k16`, the group's eight
//! heads in rows 0–7 of a 16-row tile). The caller picks one per launch.
//!
//! Keys at or past the live count get weight exactly zero and are never
//! loaded: the staging writes zero for them, so a padded row holding a NaN
//! pattern reaches no result.
//!
//! The entries above serve heads of [`HEAD`] values. The `_256` entries
//! ([`FlashGqaKernels::enqueue_pass_256`]) serve heads of [`HEAD_256`] with the
//! same grid, reduction structure and refusals, lane `l` owning dims `4l ..
//! +3` and `128 + 4l .. +3`; their tensor-core pass sums each 128-value half's
//! score in its own accumulator and adds the halves, low first — the order of
//! the prefill flash at that width. They are instances of the head-generic
//! bodies `seg_scalar`, `seg_mma` and `merge_body`.
//!
//! The `_p4` entries ([`FlashGqaKernels::enqueue_pass_256_p4`]) serve heads of
//! [`HEAD_256`] in groups of any multiple of [`PACK_4`] query heads per key
//! head: a segment block takes four query heads — four warps, the tensor-core
//! pass's rows `0..4` — and a key head's group is `group / 4` neighbouring
//! blocks, each staging the key head's tiles for its own heads. Every row's
//! arithmetic is the `_256` pass's; the merge is `gqa_flash_merge_256`. They
//! are instances of the PACK-generic copies `seg_scalar_p` and `seg_mma_p`.
//!
//! The `_p2` entry ([`FlashGqaKernels::enqueue_pass_256_p2`]) is the `_p4`
//! scalar pass in blocks of [`PACK_2`] query heads — two warps — for a group
//! that is a multiple of two and not of four (Qwen3.5-9B's 24/4): an
//! instance of the same `seg_scalar_p`. The tensor-core body needs four warps
//! a tile, so no `_p2` tensor-core pass exists and a call asking for one is
//! refused by name.
//!
//! The `_p12_sel` entry ([`FlashGqaKernels::enqueue_pass_256_p12_sel`]) walks
//! each row's own list of cache rows (a token-pool selector's, `qsa`) instead
//! of the rows below its count, one block a (row, key head, segment) holding
//! the key head's twelve query heads: `seg_mma_ps`, a copy of the `_p4` tensor
//! pass whose row list (`Listed`) only stages a tile's rows and bounds the
//! count, and whose tile holds twelve heads and four zero rows. Over the list
//! `0 .. n` a row stages the same keys in the same tiles and segments as the
//! `_p4` entries at count `n`, and every head's arithmetic is theirs; the gate
//! holds the two bodies to the same bits there.
//!
//! The `_q8` entries ([`FlashGqaKernels::enqueue_pass_q8`] and
//! [`FlashGqaKernels::enqueue_pass_256_q8`]) are the eight-head tensor-core
//! segment passes over the Q8_0 cache the quantizing appends write
//! (`rope_neox`'s two-plane layout: a codes plane of `head/4` u32 a row, a
//! scales plane of `head/32` u16, one pair a side for K and for V). They and
//! the `_256` tensor-core pass are instances of one body, `seg_mma` over
//! [`Q8Kv`] or [`F16Kv`]: only a tile's staged words differ
//! ([`KvSrc::word`]), and the merges are the f16 passes'. The key and value
//! tiles stage the dequantized values — `code · d`, `d` the block's f16
//! scale widened, exact in f32 — rounded to f16, the tiles' own format,
//! through [`q8_word_values`] and `f32x2_to_f16x2_bits`. Their products carry
//! the f16 rounding the f16 cache never had, so the owning gates hold each
//! entry bit for bit to its f16 twin run on the f16 cache of those
//! dequantized values. No scalar or packed (`_p4`, `_p2`) q8 pass exists.
//!
//! A key at or past the segment's end stages zeros, so a refused block's NaN
//! scale reaches a live result only as the f16 twin's NaN rows do — through a
//! key below its row's count, never past it.
//!
//! The `_rows` entry ([`FlashGqaKernels::enqueue_pass_rows`]) is the
//! head-128 tensor-core segment pass over several sequences' f16 caches: row
//! `t` reads the planes of row `t` of a per-row table ([`RowPlanes`], the
//! launch's grid-constant parameter) and is otherwise `seg_mma`'s row over
//! them; the merge is [`flash_gqa_kernels::gqa_flash_merge`].
//!
//! The `_k192` entries ([`FlashGqaKernels::enqueue_pass_k192`]) serve MiMo-V2's
//! heads: scores over [`HEAD_K192`] key values, values of [`HEAD`], `packs`
//! blocks of [`GROUP`] query heads a key head (`seg_scalar_kv`, the scalar
//! reduction above over 96 key words), and per layer a window of positions
//! and a per-head sink. The window is one rule ([`window_cut`]): a row of
//! `limit` live keys attends the last `window` of them, cut into
//! [`window_segments`] segments from its first windowed key, so no segment,
//! tile or partial exists below it and no score is masked. The sink
//! (`gqa_flash_merge_sink`) joins the softmax last, as the partial `(sink, 1,
//! 0)`, unscaled: segments ascending, then the sink. A row of at most
//! `window` keys is cut into the same 64-key segments with or without a
//! window. Without a window and without sinks the merge is
//! [`flash_gqa_kernels::gqa_flash_merge`].
//!
//! Both merges skip a segment whose partial is neutral (`s = 0`). Over a
//! finite row that skip is unreachable: a row's segments are cut from its
//! first key (windowed or not), so each of its `⌈n / span⌉` live segments
//! holds a key. It guards a partial no kernel here writes for a live segment.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{
    MERGE_BATCH, MMA_K, MMA_NTILE, MMA_ROWS, dev_exp, f32_to_f16_bits, f32x2_to_f16x2_bits,
    half_bits_to_f32, mma_row_words, online_fold,
};
use crate::launch_u32;
use crate::rope_neox::q8_plane_lens;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
};
use cuda_host::cuda_module;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// Values per head.
pub const HEAD: usize = 128;
/// Query heads per key/value head, and so warps per segment block.
pub const GROUP: usize = 8;
/// Keys per online-softmax tile: one key per lane.
pub const KEY_TILE: usize = 32;
/// The shortest segment a row's cut walks ([`seg_span`]'s floor): a multiple
/// of [`KEY_TILE`]; the choice keeps the grid above the card's SM count from
/// about a thousand keys on (four key heads times one block per segment).
pub const SEG_KEYS: usize = 64;
/// Rotating partials of the score dot.
const ILP: usize = 4;

const THREADS: usize = GROUP * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS);
/// u32 words of one key row, and the padded stride the K tile is staged at:
/// lane `j` reads word `w` of key `j` at `j·65 + w`, a distinct bank per lane.
const ROW_WORDS: usize = HEAD / 2;
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of one value row: lane `l` reads word `l`, dims `4l .. 4l + 3`.
const ROW_QWORDS: usize = HEAD / 4;
const _: () = assert!(ROW_QWORDS == 32 && SEG_KEYS.is_multiple_of(KEY_TILE));
const _: () = assert!(ROW_WORDS.is_multiple_of(ILP));

/// u32 words per staged row of the tensor-core pass's query and key tiles.
const MMA_ROW_W: usize = mma_row_words(HEAD);
const _: () = assert!(MMA_ROW_W * 4 % 128 == 16 && MMA_ROWS == 2 * GROUP);
const _: () = assert!(KEY_TILE == 4 * MMA_NTILE && HEAD.is_multiple_of(2 * MMA_K));

/// Merge threads per head: one per dim.
const MERGE_THREADS: u32 = HEAD as u32;

/// Values per head of the wide instance (Qwen3.6's full-attention layers):
/// [`FlashGqaKernels::enqueue_pass_256`]'s head, the same [`GROUP`],
/// [`KEY_TILE`] and [`SEG_KEYS`].
pub const HEAD_256: usize = 256;
/// Its 128-value slices, the key tile's padded stride in u32 words, and its
/// value rows in u64 words.
const QW_256: usize = HEAD_256 / SLICE;
const K_STRIDE_256: usize = HEAD_256 / 2 + 1;
const ROW_QWORDS_256: usize = HEAD_256 / 4;
const MMA_ROW_W_256: usize = mma_row_words(HEAD_256);
const MERGE_THREADS_256: u32 = HEAD_256 as u32;
const _: () = assert!(MERGE_THREADS_256 as usize == HEAD_256);
// The entries' launch contracts spell GROUP and HEAD_256 out as 8 and 256.
const _: () = assert!(GROUP == 8 && HEAD_256 == 256);

/// Query heads one block of the `_p4` entries takes
/// ([`FlashGqaKernels::enqueue_pass_256_p4`]): a group of any multiple of four
/// query heads per key head runs as `group / 4` blocks per key head.
pub const PACK_4: usize = 4;
const THREADS_P4: usize = PACK_4 * 32;
const THREADS_P4_U32: u32 = THREADS_P4 as u32;
// The `_p4` launch contracts spell PACK_4 and its block out as 4 and 128.
const _: () = assert!(PACK_4 == 4 && THREADS_P4_U32 == 128);
// The `_p4` instance of `seg_mma_p`: its generic const blocks run only when
// the device build instantiates it, this one in every check.
const _: () = assert!(MMA_ROWS.is_multiple_of(PACK_4) && PACK_4 >= 4 && PACK_4 <= MMA_ROWS / 2);

/// Query heads one block of the `_p12_sel` entry takes
/// ([`FlashGqaKernels::enqueue_pass_256_p12_sel`]): a key head's whole group of
/// twelve (Qwen3.8's 24/2), one warp each, in rows `0..12` of the 16-row query
/// tile with [`PAD_12`] zero rows after them.
pub const PACK_12: usize = 12;
const PAD_12: usize = MMA_ROWS - PACK_12;
const THREADS_P12: usize = PACK_12 * 32;
const THREADS_P12_U32: u32 = THREADS_P12 as u32;
// The `_p12_sel` launch contract spells PACK_12 and its block out as 12 and
// 384; its instance of `seg_mma_ps` is checked in every build, as `_p4`'s above.
const _: () = assert!(PACK_12 == 12 && THREADS_P12_U32 == 384);
const _: () = assert!(PACK_12 + PAD_12 == MMA_ROWS && PAD_12 <= PACK_12 && PACK_12 > MMA_ROWS / 2);

/// Segments every decode launch cuts a row's keys into, whatever the cache
/// height: per (row, key head) unit about one block for each SM of either
/// workstation card (82 and 84 SMs), and whole merge batches. The grid is
/// `m · units · SEGMENTS`, so a captured graph's launch does not depend on
/// the context the load allocated.
pub const SEGMENTS: usize = 80;
const _: () = assert!(SEGMENTS.is_multiple_of(MERGE_BATCH));

/// Keys each segment of a row of `live` keys walks when the row is cut into
/// `segs` segments: `floor` while `segs · floor` covers the row, else
/// `⌈live / segs⌉` rounded up to whole [`KEY_TILE`]s. Segment `s` walks
/// `[s · span, min((s + 1) · span, live))`, and the first `⌈live / span⌉ <=
/// segs` hold keys. The one owner of the cut: every segment pass and both
/// merges compute it from the row's own count, on the device.
#[inline(always)]
#[must_use]
pub const fn seg_span(live: usize, segs: usize, floor: usize) -> usize {
    let per = live.div_ceil(segs).div_ceil(KEY_TILE) * KEY_TILE;
    if per > floor { per } else { floor }
}

/// Segments a listed launch ([`FlashGqaKernels::enqueue_pass_256_p12_sel`])
/// cuts a row's list of at most `width` entries into: whole
/// [`SEG_KEYS`]-key segments of the width.
#[must_use]
pub const fn listed_segments(width: usize) -> usize {
    let w = if width == 0 { 1 } else { width };
    w.div_ceil(SEG_KEYS)
}

/// The partials' length for `m` rows of `n_head` query heads cut into
/// `segs` segments, `per_seg` values a segment: the one product every
/// `partials_*` function below takes.
const fn partials_len(m: usize, n_head: usize, segs: usize, per_seg: usize) -> usize {
    m * n_head * segs * per_seg
}

/// The `Σ exp·V` partials' length for `m` rows of `n_head` query heads
/// ([`SEGMENTS`] a row's head).
#[must_use]
pub fn partials_v_len(m: usize, n_head: usize) -> usize {
    partials_len(m, n_head, SEGMENTS, HEAD)
}

/// The `(max, Σ exp)` partials' length.
#[must_use]
pub fn partials_ms_len(m: usize, n_head: usize) -> usize {
    partials_len(m, n_head, SEGMENTS, 2)
}

/// [`partials_v_len`] at [`HEAD_256`]; the `(max, Σ exp)` partials are
/// [`partials_ms_len`] at either width.
#[must_use]
pub fn partials_v_len_256(m: usize, n_head: usize) -> usize {
    partials_len(m, n_head, SEGMENTS, HEAD_256)
}

/// [`partials_v_len_256`] of a listed launch over lists of at most `width`
/// ([`listed_segments`]`(width)` a row's head).
#[must_use]
pub fn listed_partials_v_len_256(m: usize, n_head: usize, width: usize) -> usize {
    partials_len(m, n_head, listed_segments(width), HEAD_256)
}

/// [`partials_ms_len`] of a listed launch over lists of at most `width`.
#[must_use]
pub fn listed_partials_ms_len(m: usize, n_head: usize, width: usize) -> usize {
    partials_len(m, n_head, listed_segments(width), 2)
}

// The cut's invariants, the ones the segment bodies and both merges rely on
// (every range they walk and every partial they read): the span covers the
// row (`span · segs >= live`), so a row's live segments stay within the
// launch (`live.div_ceil(span) <= segs`) and the merges never read a partial
// past the buffer; the span is whole KEY_TILEs; and it is the floor's
// `SEG_KEYS` exactly while `segs · SEG_KEYS` covers the row, so every count
// at or below 80 · 64 = 5,120 keys is cut into 64-key segments whatever the
// cache height. The listed path's cut at Qwen3.8's width: `SEG_KEYS` for
// every live count, because its 33 launched segments of 64 keys cover 2,051.
const fn seg_span_holds(live: usize, segs: usize, floor: usize) -> bool {
    let span = seg_span(live, segs, floor);
    span * segs >= live
        && live.div_ceil(span) <= segs
        && span.is_multiple_of(KEY_TILE)
        && (span == floor) == (live <= segs * floor)
}
const _: () = {
    let mut live = 0;
    while live <= 65_536 {
        assert!(seg_span_holds(live, SEGMENTS, SEG_KEYS));
        live += 1;
    }
    let mut i = 0;
    let points = [261_120, 262_144, 1_048_576, 2_000_000, u32::MAX as usize];
    while i < points.len() {
        assert!(seg_span_holds(points[i], SEGMENTS, SEG_KEYS));
        i += 1;
    }
    let segs = listed_segments(2_051);
    assert!(segs == 33);
    let mut live = 0;
    while live <= 2_051 {
        assert!(seg_span_holds(live, segs, SEG_KEYS));
        live += 1;
    }
};

/// Key values per head of the MiMo-V2 instance
/// ([`FlashGqaKernels::enqueue_pass_k192`]): its value rows are [`HEAD`]
/// wide, so the key planes are `[n_kv][ctx][HEAD_K192]` and the value planes
/// `[n_kv][ctx][HEAD]`.
pub const HEAD_K192: usize = 192;
/// u32 words of one K192 key row and its staged stride (odd, so lane `j`
/// reads word `w` of key `j` at `97j + w` from a distinct bank), and its
/// u64 words.
const ROW_WORDS_K192: usize = HEAD_K192 / 2;
const K_STRIDE_K192: usize = ROW_WORDS_K192 + 1;
const ROW_QWORDS_K192: usize = HEAD_K192 / 4;
// The K192 entries' launch contracts spell HEAD_K192 and its words out as 192, 96 and 48.
const _: () = assert!(HEAD_K192 == 192 && ROW_WORDS_K192 == 96 && ROW_QWORDS_K192 == 48);
const _: () = assert!(ROW_WORDS_K192.is_multiple_of(ILP) && K_STRIDE_K192 % 2 == 1);
/// The merge of [`FlashGqaKernels::enqueue_pass_k192`] with the sinks: one
/// thread a value dim, [`HEAD`] of them.
const MERGE_THREADS_SINK: u32 = HEAD as u32;

/// The window a row of `limit` live keys attends: `(first, n_w)`, the keys
/// `[first, first + n_w)` — the last `window` of them, or all of them for
/// `window == 0` (no window). A key at position `p` attends the keys
/// `p − window < k <= p`, itself included, so the row whose live count is
/// `limit = p + 1` starts at `limit − min(limit, window)`. The one owner of
/// the window's lower bound: the K192 segment pass and its merge cut the
/// `n_w` keys with [`seg_span`] and walk them from `first`, so no segment,
/// tile or partial exists below `first`.
#[inline(always)]
#[must_use]
pub const fn window_cut(limit: usize, window: usize) -> (usize, usize) {
    let first = if window == 0 || limit < window {
        0
    } else {
        limit - window
    };
    (first, limit - first)
}

/// Segments a K192 launch with `window` cuts a row into: [`SEGMENTS`]
/// without a window, else whole [`SEG_KEYS`]-key segments of the window.
#[must_use]
pub const fn window_segments(window: usize) -> usize {
    if window == 0 {
        SEGMENTS
    } else {
        window.div_ceil(SEG_KEYS)
    }
}

// The window cut's invariants: the cut keeps `first + n_w = limit` and
// `n_w <= window`; at 128 positions in two segments of 64 the span is the
// floor for every live count (so `n_w <= 128` and an `n_w` of 128 or less is
// cut into the same 64-key segments with or without a window), and no row
// has a segment past its window; no window is today's cut.
const _: () = {
    assert!(window_segments(0) == SEGMENTS && window_segments(128) == 2);
    assert!(window_segments(129) == 3 && window_segments(1) == 1);
    let segs = window_segments(128);
    let mut limit = 0;
    while limit <= 65_536 {
        let (first, n_w) = window_cut(limit, 128);
        assert!(first + n_w == limit && n_w <= 128 && n_w <= limit);
        assert!(first == 0 || n_w == 128);
        assert!(seg_span(n_w, segs, SEG_KEYS) == SEG_KEYS);
        assert!(seg_span_holds(n_w, segs, SEG_KEYS));
        assert!(n_w.div_ceil(SEG_KEYS) <= segs);
        let (first, n_w) = window_cut(limit, 0);
        assert!(first == 0 && n_w == limit);
        limit += 1;
    }
};

/// A row's live key count as every kernel here reads it: `count` when it
/// lies in `1..=ctx`, else 0 — a refused row, whose segments are all
/// neutral and whose merge raises [`FaultSite::KeyCount`].
#[inline(always)]
fn live_keys(count: u32, ctx: usize) -> usize {
    let c = count as usize;
    if c > ctx { 0 } else { c }
}

/// One tile's online-softmax step for warp `w`'s head, lane `lane` holding
/// key `lane`'s score `sc` (`live` false past the segment's end): the tile
/// max by the five-step butterfly, the running `(mx, s_sum)` and the value
/// partials rescaled on a bump, the weight `dev_exp(sc − mx)` (0 when not
/// live), the weight sum by the five-step butterfly, and the value partials
/// of dims `4·lane .. +3` accumulated over the tile's keys ascending, one
/// fused multiply-add per key. Both segment passes run their tiles through
/// it, so their softmax and value orders are one.
///
/// SAFETY: `ws` is the block's `GROUP · KEY_TILE` f32 weight tile and `vs`
/// its staged `KEY_TILE · ROW_QWORDS` value tile (published by a block
/// barrier); `w < GROUP`; all 32 lanes of the warp call it together.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the tile step's state is the caller's registers, passed by reference"
)]
unsafe fn fold_tile(
    sc: f32,
    live: bool,
    mx: &mut f32,
    s_sum: &mut f32,
    acc: &mut [f32; 4],
    ws: *mut f32,
    vs: *const u64,
    w: usize,
    lane: usize,
) {
    let mut tmax = sc;
    let mut off = 16u32;
    while off > 0 {
        tmax = tmax.max(warp::shuffle_xor_f32(tmax, off));
        off >>= 1;
    }
    let m_new = mx.max(tmax);
    let pw = if live { dev_exp(sc - m_new) } else { 0.0 };
    let mut psum = pw;
    let mut off = 16u32;
    while off > 0 {
        psum += warp::shuffle_xor_f32(psum, off);
        off >>= 1;
    }
    if m_new > *mx {
        let f = if *mx > f32::NEG_INFINITY {
            dev_exp(*mx - m_new)
        } else {
            0.0
        };
        *s_sum *= f;
        acc[0] *= f;
        acc[1] *= f;
        acc[2] *= f;
        acc[3] *= f;
        *mx = m_new;
    }
    *s_sum += psum;

    // SAFETY: w·32 + lane < GROUP·KEY_TILE; this warp's own slots.
    unsafe { *ws.add(w * KEY_TILE + lane) = pw };
    warp::sync_mask(u32::MAX);
    let mut j = 0usize;
    while j < KEY_TILE {
        // SAFETY: j < 32 and lane < 32: inside WS and VS, written before
        // the caller's barrier (VS) or the warp sync (WS) above.
        let (pj, vw) = unsafe { (*ws.add(w * KEY_TILE + j), *vs.add(j * ROW_QWORDS + lane)) };
        acc[0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[0]);
        acc[1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[1]);
        acc[2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[2]);
        acc[3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[3]);
        j += 1;
    }
    warp::sync_mask(u32::MAX);
}

// ------------------------------------------------------- the cache formats
//
// The two KV cache formats a flash body reads, each the layout its append
// writes (`rope_neox`): the f16 planes ([`F16Kv`]) or the Q8_0 cache
// ([`Q8Kv`]). A body is written once over [`KvSrc`]; its two instances differ
// only in a decode tile's staged word ([`KvSrc::word`]) and in how a prefill
// tile lands (`flash_gqa_prefill`, on [`KvSrc::Q8`]).

/// A KV cache format as the flash bodies read it, the planes `n_kv · ctx`
/// rows of `HEAD` values (key head `kh`'s rows `kh·ctx ..`).
pub(crate) trait KvSrc: Copy {
    /// Whether the planes hold the Q8_0 form: a prefill tile lands as its
    /// rows' raw code words and scales and is widened in place to f16.
    const Q8: bool;
    /// What a walk over one key head's rows computes once, before its tiles.
    type Head: Copy;

    /// Key head `kh`'s rows of a cache of `ctx` rows.
    fn head<const HEAD: usize>(self, kh: usize, ctx: usize) -> Self::Head;

    /// Staged word `word` of row `key` of the head: four values as the f16
    /// bits the tiles hold, of the key row, then of the value row.
    ///
    /// # Safety
    /// `head` is [`KvSrc::head`] of a key head below the launch's `n_kv`,
    /// `key < ctx`, `word < HEAD/4`, and the entry's launch contract holds.
    unsafe fn word<const HEAD: usize>(head: Self::Head, key: usize, word: usize) -> (u64, u64);

    /// The planes as u32 words: the key rows, the key scales, the value rows
    /// and the value scales. An f16 row is its f16 pairs, and the f16 cache
    /// has no scales (null).
    fn words(self) -> [*const u32; 4];
}

/// The f16 cache: two planes of f16 rows, key and value.
#[derive(Clone, Copy)]
pub(crate) struct F16Kv<'a> {
    pub(crate) k: &'a [u16],
    pub(crate) v: &'a [u16],
}

impl KvSrc for F16Kv<'_> {
    const Q8: bool = false;
    /// The key head's first value, and the two planes as u64 words.
    type Head = (usize, *const u64, *const u64);

    #[inline(always)]
    fn head<const HEAD: usize>(self, kh: usize, ctx: usize) -> Self::Head {
        (
            kh * ctx * HEAD,
            self.k.as_ptr() as *const u64,
            self.v.as_ptr() as *const u64,
        )
    }

    #[inline(always)]
    unsafe fn word<const HEAD: usize>(
        (plane, k64, v64): Self::Head,
        key: usize,
        word: usize,
    ) -> (u64, u64) {
        // SAFETY: the row is inside key head kh's plane (this fn's contract),
        // and the u64 read is aligned: a row is HEAD/4 u64 and the planes
        // start 8-byte aligned (a device allocation).
        unsafe {
            let r = (plane + key * HEAD) / 4 + word;
            (*k64.add(r), *v64.add(r))
        }
    }

    #[inline(always)]
    fn words(self) -> [*const u32; 4] {
        [
            self.k.as_ptr().cast::<u32>(),
            std::ptr::null(),
            self.v.as_ptr().cast::<u32>(),
            std::ptr::null(),
        ]
    }
}

/// The Q8_0 cache: a side is a codes plane of `head/4` u32 a row (code `j`
/// of a 32-value block in byte `j % 4` of word `j / 4`) and a scales plane of
/// `head/32` u16 a row, each the block's f16 scale bits.
#[derive(Clone, Copy)]
pub(crate) struct Q8Kv<'a> {
    pub(crate) kq: &'a [u32],
    pub(crate) kd: &'a [u16],
    pub(crate) vq: &'a [u32],
    pub(crate) vd: &'a [u16],
}

impl KvSrc for Q8Kv<'_> {
    const Q8: bool = true;
    /// The key head's first row, and the planes.
    type Head = (usize, Self);

    #[inline(always)]
    fn head<const HEAD: usize>(self, kh: usize, ctx: usize) -> Self::Head {
        (kh * ctx, self)
    }

    /// The staged word's code word and its block's scale a side, widened by
    /// [`q8_value_word`].
    #[inline(always)]
    unsafe fn word<const HEAD: usize>(
        (row0, p): Self::Head,
        key: usize,
        word: usize,
    ) -> (u64, u64) {
        const { assert!(HEAD.is_multiple_of(32)) };
        let row = row0 + key;
        // SAFETY: the row is inside key head kh's rows of both sides' planes
        // (this fn's contract); word < HEAD/4 and word/8 < HEAD/32 are inside
        // the row.
        let (kc, ks, vc, vs) = unsafe {
            (
                *p.kq.get_unchecked(row * (HEAD / 4) + word),
                *p.kd.get_unchecked(row * (HEAD / 32) + word / 8),
                *p.vq.get_unchecked(row * (HEAD / 4) + word),
                *p.vd.get_unchecked(row * (HEAD / 32) + word / 8),
            )
        };
        (q8_value_word(kc, ks), q8_value_word(vc, vs))
    }

    #[inline(always)]
    fn words(self) -> [*const u32; 4] {
        [
            self.kq.as_ptr(),
            self.kd.as_ptr().cast::<u32>(),
            self.vq.as_ptr(),
            self.vd.as_ptr().cast::<u32>(),
        ]
    }
}

/// One code word's four values as the q8 reads widen them: `code·d` each,
/// exact in f32 — a code's 7 significant bits against the f16 scale's 11
/// leave 6 of f32's 24 — the dequantized value the f16 tiles round from.
#[inline(always)]
pub(crate) fn q8_word_values(cw: u32, d: f32) -> [f32; 4] {
    [
        (cw as u8 as i8 as f32) * d,
        ((cw >> 8) as u8 as i8 as f32) * d,
        ((cw >> 16) as u8 as i8 as f32) * d,
        ((cw >> 24) as u8 as i8 as f32) * d,
    ]
}

/// One staged word's four dequantized f16 values packed as the tiles' u64:
/// `cw` the row's code word, `dbits` its block's f16 scale.
#[inline(always)]
fn q8_value_word(cw: u32, dbits: u16) -> u64 {
    let d = half_bits_to_f32(dbits);
    let v = q8_word_values(cw, d);
    let lo = f32x2_to_f16x2_bits(v[0], v[1]);
    u64::from(lo) | (u64::from(f32x2_to_f16x2_bits(v[2], v[3])) << 32)
}

// --------------------------------------- the rows of several sequences
//
// A launch over the rows of several sequences — a pass of resident slots,
// each slot's rows over its own cache — finds each row's planes in a per-row
// table, the launch's grid-constant parameter ([`RowPlanes`]). A captured
// launch holds the table as it holds every buffer address it was given, so a
// replay reads the planes the capture read; the rows' other inputs (the
// queries, the live counts, the partials) are row-indexed as in a launch of
// one sequence. The `_rows` entries serve the f16 cache.

/// Rows one launch over several sequences' caches takes: a pass of resident
/// slots holds at most this many (`model::MAX_PASS_ROWS`).
pub(crate) const ROW_PLANES: usize = 8;
// The `_rows` launch contracts spell ROW_PLANES out as 8.
const _: () = assert!(ROW_PLANES == 8);

/// The per-row table of a launch over several sequences' caches, its
/// grid-constant parameter: row `t`'s planes in [`KvSrc::words`]'s order —
/// key rows, key scales, value rows, value scales, the f16 cache's scales
/// null — for each row below the launch's `m`, null past it. Built on the
/// host by [`RowTable`] alone.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RowPlanes {
    rows: [[*mut u32; 4]; ROW_PLANES],
}

impl RowPlanes {
    /// Row `t`'s f16 key and value planes, each `plane` u16 — the planes a
    /// launch of one sequence takes as its two slices.
    ///
    /// # Safety
    /// `t` is below the launch's rows, whose table [`RowTable::planes`] built
    /// over f16 planes of at least `plane` values each; nothing writes the
    /// planes while the launch reads them.
    #[inline(always)]
    pub(crate) unsafe fn f16(&self, t: usize, plane: usize) -> F16Kv<'_> {
        // SAFETY: t < m <= ROW_PLANES (this fn's contract).
        let w = unsafe { *self.rows.get_unchecked(t) };
        // SAFETY: the row's words 0 and 2 are live f16 planes of at least
        // `plane` values that only this launch's reads touch (this fn's
        // contract).
        unsafe {
            F16Kv {
                k: std::slice::from_raw_parts(w[0].cast_const().cast::<u16>(), plane),
                v: std::slice::from_raw_parts(w[2].cast_const().cast::<u16>(), plane),
            }
        }
    }

    /// Row `t`'s f16 key and value planes to write through, each `plane`
    /// u16: [`RowPlanes::f16`]'s planes as raw pointers.
    ///
    /// # Safety
    /// As [`RowPlanes::f16`], with the launch the one writer of the planes.
    #[inline(always)]
    pub(crate) unsafe fn f16_mut(&self, t: usize) -> (*mut u16, *mut u16) {
        // SAFETY: t < m <= ROW_PLANES (this fn's contract).
        let w = unsafe { *self.rows.get_unchecked(t) };
        (w[0].cast::<u16>(), w[2].cast::<u16>())
    }
}

/// [`RowPlanes`] built on the host: each sequence's planes put in row order,
/// each put borrowing them for the table's life, so the launches the table
/// feeds ([`FlashGqaKernels::enqueue_pass_rows`] and
/// `RopeNeoxKernels::enqueue_head_norm_neox_append_rows`) read and write
/// planes nothing else holds while they are enqueued.
pub(crate) struct RowTable<'a> {
    planes: RowPlanes,
    /// The rows put so far: `0..rows`.
    rows: usize,
    /// The shortest plane put, in u16.
    plane: usize,
    _planes: PhantomData<&'a mut DeviceBuffer<u16>>,
}

impl Default for RowTable<'_> {
    /// No row put.
    fn default() -> Self {
        RowTable {
            planes: RowPlanes {
                rows: [[ptr::null_mut(); 4]; ROW_PLANES],
            },
            rows: 0,
            plane: usize::MAX,
            _planes: PhantomData,
        }
    }
}

impl<'a> RowTable<'a> {
    /// Rows `rows` of the launch read and write the f16 planes `k` and `v`.
    /// Refused by name unless `rows` hold a row, start where the last put
    /// ended (the first at row 0) and end within [`ROW_PLANES`].
    pub(crate) fn put_f16(
        &mut self,
        rows: Range<usize>,
        k: &'a mut DeviceBuffer<u16>,
        v: &'a mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        if rows.start != self.rows || rows.is_empty() || rows.end > ROW_PLANES {
            return Err(GpuError::shape(
                "flash_gqa::RowTable::put_f16",
                format!(
                    "rows {rows:?} after rows 0..{}: a sequence's rows follow the last put's, \
                     hold a row and end within {ROW_PLANES}",
                    self.rows
                ),
            ));
        }
        let words = [
            k.cu_deviceptr() as *mut u32,
            ptr::null_mut(),
            v.cu_deviceptr() as *mut u32,
            ptr::null_mut(),
        ];
        for row in &mut self.planes.rows[rows.clone()] {
            *row = words;
        }
        self.rows = rows.end;
        self.plane = self.plane.min(k.len()).min(v.len());
        Ok(())
    }

    /// The table of a launch of `m` rows over planes of `need` u16 each, or
    /// the named refusal of a table of other rows or a shorter plane.
    pub(crate) fn planes(
        &self,
        what: &'static str,
        m: usize,
        need: usize,
    ) -> Result<RowPlanes, GpuError> {
        if self.rows != m || self.plane < need {
            return Err(GpuError::shape(
                what,
                format!(
                    "a launch of {m} rows over planes of {need} values; the table holds {} rows, \
                     its shortest plane {}",
                    self.rows, self.plane
                ),
            ));
        }
        Ok(self.planes)
    }
}

// ------------------------------------------------ the head-generic bodies
//
// The segment passes and the merge as `#[inline(always)]` bodies over the
// head width, each instance a thin `#[kernel]` entry that declares its shared
// tiles at the instance's size and hands them in (a `static` inside a generic
// function cannot take its size from the function's parameters). `QW` is the
// head's count of 128-value slices: lane `l` owns dims `4l + 128·i .. + 3` for
// `i < QW`, one u64 of each slice of a value row, so every value load of a
// warp covers 32 consecutive u64 words.

/// Values of the head slice one lane-word walk covers: 32 lanes × 4 dims.
const SLICE: usize = 128;

/// [`fold_tile`] over a head of `QW · 128` values: the same tile max, rescale
/// and weight sum, the value partials `acc[i]` of dims `4·lane + 128·i .. +3`
/// accumulated over the tile's keys ascending, one fused multiply-add per key
/// and dim.
///
/// SAFETY: as [`fold_tile`], with `vs` the staged `KEY_TILE · HEAD/4` value
/// tile.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the tile step's state is the caller's registers, passed by reference"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn fold_tile_w<const HEAD: usize, const QW: usize>(
    sc: f32,
    live: bool,
    mx: &mut f32,
    s_sum: &mut f32,
    acc: &mut [[f32; 4]; QW],
    ws: *mut f32,
    vs: *const u64,
    w: usize,
    lane: usize,
) {
    let row_qwords = HEAD / 4;
    let mut tmax = sc;
    let mut off = 16u32;
    while off > 0 {
        tmax = tmax.max(warp::shuffle_xor_f32(tmax, off));
        off >>= 1;
    }
    let m_new = mx.max(tmax);
    let pw = if live { dev_exp(sc - m_new) } else { 0.0 };
    let mut psum = pw;
    let mut off = 16u32;
    while off > 0 {
        psum += warp::shuffle_xor_f32(psum, off);
        off >>= 1;
    }
    if m_new > *mx {
        let f = if *mx > f32::NEG_INFINITY {
            dev_exp(*mx - m_new)
        } else {
            0.0
        };
        *s_sum *= f;
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            acc[i][0] *= f;
            acc[i][1] *= f;
            acc[i][2] *= f;
            acc[i][3] *= f;
        }
        *mx = m_new;
    }
    *s_sum += psum;

    // SAFETY: w·32 + lane < GROUP·KEY_TILE; this warp's own slots.
    unsafe { *ws.add(w * KEY_TILE + lane) = pw };
    warp::sync_mask(u32::MAX);
    let mut j = 0usize;
    while j < KEY_TILE {
        // SAFETY: j < 32 and lane < 32: inside WS, written before the warp
        // sync above.
        let pj = unsafe { *ws.add(w * KEY_TILE + j) };
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: j < KEY_TILE and lane + 32·i < HEAD/4: inside VS,
            // written before the caller's barrier.
            let vw = unsafe { *vs.add(j * row_qwords + lane + 32 * i) };
            acc[i][0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[i][0]);
            acc[i][1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[i][1]);
            acc[i][2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[i][2]);
            acc[i][3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[i][3]);
        }
        j += 1;
    }
    warp::sync_mask(u32::MAX);
}

/// The scalar segment pass of [`flash_gqa_kernels::gqa_flash_seg`] over a
/// head of `HEAD = QW · 128` values: the grid, the partial index, the
/// neutral segment, the staging and the reduction structure of the module
/// doc, lane `l` holding dims `4l + 128·i .. +3`.
///
/// SAFETY: the entry's launch contract at `HEAD` (every buffer bound it
/// names), a block of `GROUP · 32` threads, and the four tiles this block's
/// shared memory as the entry declared them: the group's query rows
/// (`GROUP · HEAD` f32), the key tile at stride `HEAD/2 + 1` words, the value
/// tile (`KEY_TILE · HEAD/4` u64) and the weight tile (`GROUP · KEY_TILE`).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn seg_scalar<const HEAD: usize, const QW: usize>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qs: *mut f32,
    ks: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(HEAD == QW * SLICE && HEAD.is_multiple_of(2 * ILP)) };
    let row_words = HEAD / 2;
    let k_stride = row_words + 1;
    let row_qwords = HEAD / 4;
    // q values and K/V tile words one thread stages per tile.
    let per_thread = HEAD / 32;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    if b >= rows * nkv * n_seg {
        return; // block-uniform
    }
    let t = b % rows;
    let sk = b / rows;
    let seg = sk / nkv;
    let kh = sk - seg * nkv;
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let n_head = nkv * GROUP;
    let h = kh * GROUP + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let span = seg_span(limit, n_seg, seg_keys as usize);
    let lo = seg * span;
    if lo >= limit {
        if lane == 0 {
            // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo and limit are the block's
    }
    let hi = (lo + span).min(limit);

    // Row t's group query rows: thread `tid` stages values
    // `per_thread·tid ..` of the group's GROUP·HEAD.
    let qb = (t * n_head + kh * GROUP) * HEAD;
    let mut i = 0usize;
    while i < per_thread {
        // SAFETY: qb + per_thread·tid + i < (t·n_head + (kh + 1)·GROUP)·HEAD
        // <= m·n_kv·GROUP·HEAD <= q.len(); the shared index < GROUP·HEAD.
        unsafe { *qs.add(per_thread * tid + i) = *q.get_unchecked(qb + per_thread * tid + i) };
        i += 1;
    }

    let plane = kh * ctx * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; QW];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        // Stage the tile: 32 keys × HEAD/4 u64 of K and of V; a key at or
        // past `hi` stages zeros.
        let mut i = 0usize;
        while i < per_thread {
            let e = tid + THREADS * i;
            let key = e / row_qwords;
            let word = e - key * row_qwords;
            let (kw, vw) = if t0 + key < hi {
                // SAFETY: t0 + key < hi <= ctx, so the row is inside key
                // head kh's plane; a row is HEAD u16 = HEAD/4 u64 and the
                // planes start 8-byte aligned (a device allocation), so the
                // u64 read is aligned and inside both planes.
                unsafe {
                    let r = (plane + (t0 + key) * HEAD) / 4 + word;
                    (*k64.add(r), *v64.add(r))
                }
            } else {
                (0u64, 0u64)
            };
            // SAFETY: key < 32 and word < HEAD/4: 2·word + 1 < k_stride and
            // key·HEAD/4 + word < KEY_TILE·HEAD/4.
            unsafe {
                *ks.add(key * k_stride + 2 * word) = kw as u32;
                *ks.add(key * k_stride + 2 * word + 1) = (kw >> 32) as u32;
                *vs.add(key * row_qwords + word) = vw;
            }
            i += 1;
        }
        thread::sync_threads();

        // The score of key t0 + lane for head h.
        let live = t0 + lane < hi;
        let mut a = [0.0f32; ILP];
        let mut wd = 0usize;
        while wd < row_words {
            let mut p = 0usize;
            while p < ILP {
                // SAFETY: lane < 32, wd + p < row_words: inside KS; the query
                // index w·HEAD + 2(wd + p) + 1 < GROUP·HEAD.
                let (kw, q0, q1) = unsafe {
                    (
                        *ks.add(lane * k_stride + wd + p),
                        *qs.add(w * HEAD + 2 * (wd + p)),
                        *qs.add(w * HEAD + 2 * (wd + p) + 1),
                    )
                };
                let k0 = half_bits_to_f32(kw as u16);
                let k1 = half_bits_to_f32((kw >> 16) as u16);
                a[p] = f32::mul_add(q0, k0, a[p]);
                a[p] = f32::mul_add(q1, k1, a[p]);
                p += 1;
            }
            wd += ILP;
        }
        let dot = (a[0] + a[1]) + (a[2] + a[3]);
        let sc = if live { dot * scale } else { f32::NEG_INFINITY };

        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barrier above; w < GROUP and lane < 32.
        unsafe {
            fold_tile_w::<HEAD, QW>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: idx < m·n_kv·GROUP·segs; dims 4·lane + 128·i .. +3 of the
    // partial row are this lane's alone, and lane 0 writes (m, s).
    unsafe {
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            let o = idx * HEAD + 4 * lane + SLICE * i;
            *part_v.get_unchecked_mut(o) = acc[i][0];
            *part_v.get_unchecked_mut(o + 1) = acc[i][1];
            *part_v.get_unchecked_mut(o + 2) = acc[i][2];
            *part_v.get_unchecked_mut(o + 3) = acc[i][3];
        }
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// The tensor-core segment pass of [`flash_gqa_kernels::gqa_flash_seg_mma`]
/// over a head of `HEAD = QW · 128` values and a cache in the format `S`: the
/// grid, the staging and every step after the scores are [`seg_scalar`]'s
/// (`fold_tile_w`), a tile's words read through [`KvSrc::word`]. Warp `w`
/// rounds its head's `HEAD/2` value pairs into row `w` of the 16-row query
/// tile (rows `8..16` zero); warps `0..4` each take eight keys of a tile and
/// accumulate the score of each 128-value slice `i` of the head in its own
/// f32 accumulator, the slice's eight k16 steps ascending from zero (two
/// `mma.m16n8k16` per `ldmatrix.x4` of the key), then add the slices in
/// ascending order — the order the prefill flash adds its two warps' half
/// scores in, so both compute a key's score to the same bits — and write
/// `scale · S`. Both staged strides are `mma_row_words(HEAD)` words.
///
/// SAFETY: the entry's launch contract at `HEAD` over `src`'s planes, a block
/// of `GROUP · 32` threads, and the four tiles this block's shared memory as
/// the entry declared them: the query tile (`MMA_ROWS · mma_row_words(HEAD)`
/// u32), the key tile (`KEY_TILE · mma_row_words(HEAD)` u32), the value tile
/// (`KEY_TILE · HEAD/4` u64) and the weight tile (`GROUP · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn seg_mma<const HEAD: usize, const QW: usize, S: KvSrc>(
    q: &[f32],
    src: S,
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qt: *mut u32,
    kt: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(HEAD == QW * SLICE) };
    const { assert!(mma_row_words(HEAD) * 4 % 128 == 16 && MMA_ROWS == 2 * GROUP) };
    const { assert!(KEY_TILE == 4 * MMA_NTILE && SLICE.is_multiple_of(2 * MMA_K)) };
    let row_w = mma_row_words(HEAD);
    let row_qwords = HEAD / 4;
    let per_thread = HEAD / 32;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    if b >= rows * nkv * n_seg {
        return; // block-uniform
    }
    let t = b % rows;
    let sk = b / rows;
    let seg = sk / nkv;
    let kh = sk - seg * nkv;
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let n_head = nkv * GROUP;
    let h = kh * GROUP + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let span = seg_span(limit, n_seg, seg_keys as usize);
    let lo = seg * span;
    if lo >= limit {
        if lane == 0 {
            // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo and limit are the block's
    }
    let hi = (lo + span).min(limit);

    // The query tile: warp `w` rounds row t's head `w`'s HEAD/2 value pairs,
    // HEAD/64 per lane, and writes the zero rows `8 + w`. Each pair is one
    // 8-byte load, and the loads come before the roundings.
    let qb = (t * n_head + kh * GROUP + w) * HEAD;
    let q64 = q.as_ptr() as *const u64;
    let mut raw = [[0u64; 2]; QW];
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: qb is a multiple of HEAD, so word qb/2 + wd holds values
            // qb + 2·wd and + 1 (wd = lane + 32·i + 64·s < HEAD/2), inside
            // (t·n_head + kh·GROUP + w + 1)·HEAD <= m·n_kv·GROUP·HEAD <=
            // q.len(); the buffer starts 8-byte aligned (a device allocation).
            unsafe { raw[s][i] = *q64.add(qb / 2 + lane + 32 * i + 64 * s) };
        }
    }
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let wd = lane + 32 * i + 64 * s;
            let lo16 = f32_to_f16_bits(f32::from_bits(raw[s][i] as u32)) as u32;
            let hi16 = f32_to_f16_bits(f32::from_bits((raw[s][i] >> 32) as u32)) as u32;
            // SAFETY: the tile words w·row_w + wd and (8 + w)·row_w + wd are
            // inside QT (wd < HEAD/2 < row_w, 8 + w < MMA_ROWS).
            unsafe {
                *qt.add(w * row_w + wd) = lo16 | (hi16 << 16);
                *qt.add((GROUP + w) * row_w + wd) = 0;
            }
        }
    }

    let head = src.head::<HEAD>(kh, ctx);
    let key0 = (w % 4) * MMA_NTILE;
    let arow = lane % MMA_ROWS;
    let ahalf = lane / MMA_ROWS;
    let bkey = lane % MMA_NTILE;
    let boct = lane / MMA_NTILE;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; QW];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        let mut i = 0usize;
        while i < per_thread {
            let e = tid + THREADS * i;
            let key = e / row_qwords;
            let word = e - key * row_qwords;
            let (kw, vw) = if t0 + key < hi {
                // SAFETY: t0 + key < hi <= ctx and word < HEAD/4, and head is
                // key head kh's (kh < n_kv).
                let (kw, vw) = unsafe { S::word::<HEAD>(head, t0 + key, word) };
                // Rebuilt here, not passed through: the call's own pair would
                // merge with the zeros in the other order, and the f16
                // entry's registers would move.
                (kw, vw)
            } else {
                (0u64, 0u64)
            };
            // SAFETY: key < 32 and 2·word + 1 < HEAD/2 < row_w: inside KT;
            // key·HEAD/4 + word inside VS.
            unsafe {
                *kt.add(key * row_w + 2 * word) = kw as u32;
                *kt.add(key * row_w + 2 * word + 1) = (kw >> 32) as u32;
                *vs.add(key * row_qwords + word) = vw;
            }
            i += 1;
        }
        thread::sync_threads();

        if w < 4 {
            let mut c = [[0.0f32; 4]; QW];
            for s in 0..QW {
                cuda_device::thread::__unroll_config::<0>();
                let mut d = SLICE * s;
                while d < SLICE * (s + 1) {
                    // SAFETY: key row key0 + bkey < KEY_TILE and words d/2 +
                    // 4·boct .. + 4 <= HEAD/2 are inside KT; every lane of the
                    // warp issues the load (the branch is on the warp index),
                    // after the barrier that staged it.
                    let bf = unsafe {
                        let bp = kt.add((key0 + bkey) * row_w + d / 2 + boct * 4);
                        wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            bp.cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: query row arow < MMA_ROWS and words d/2 +
                    // 4·ahalf .. + 4 + MMA_K/2 <= HEAD/2 are inside QT,
                    // published by the first barrier; the whole warp issues
                    // both loads and both `mma.sync` with its own fragments.
                    unsafe {
                        let ap0 = qt.add(arow * row_w + d / 2 + ahalf * (MMA_K / 4));
                        let af0 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap0.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af0, [bf[0], bf[1]]);
                        let ap1 = ap0.add(MMA_K / 2);
                        let af1 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap1.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af1, [bf[2], bf[3]]);
                    }
                    d += 2 * MMA_K;
                }
            }
            // The slices' scores added in ascending order.
            let mut sc0 = c[0][0];
            let mut sc1 = c[0][1];
            for s in 1..QW {
                cuda_device::thread::__unroll_config::<0>();
                sc0 = add_rn_f32(sc0, c[s][0]);
                sc1 = add_rn_f32(sc1, c[s][1]);
            }
            // sc0, sc1: head lane/4 at keys key0 + 2·(lane%4) + {0, 1}; each
            // slice's c[2], c[3] are the zero rows.
            let g = lane / 4;
            let kk = key0 + 2 * (lane % 4);
            let s0 = if t0 + kk < hi {
                mul_rn_f32(scale, sc0)
            } else {
                f32::NEG_INFINITY
            };
            let s1 = if t0 + kk + 1 < hi {
                mul_rn_f32(scale, sc1)
            } else {
                f32::NEG_INFINITY
            };
            // SAFETY: g < GROUP and kk + 1 < KEY_TILE: inside WS; each
            // (head, key) slot has one writer.
            unsafe {
                *ws.add(g * KEY_TILE + kk) = s0;
                *ws.add(g * KEY_TILE + kk + 1) = s1;
            }
        }
        thread::sync_threads();
        // SAFETY: w < GROUP and lane < KEY_TILE: inside WS, written before
        // the barrier above.
        let sc = unsafe { *ws.add(w * KEY_TILE + lane) };
        let live = t0 + lane < hi;
        warp::sync_mask(u32::MAX);
        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barriers above; w < GROUP and lane < 32.
        unsafe {
            fold_tile_w::<HEAD, QW>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: as in seg_scalar.
    unsafe {
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            let o = idx * HEAD + 4 * lane + SLICE * i;
            *part_v.get_unchecked_mut(o) = acc[i][0];
            *part_v.get_unchecked_mut(o + 1) = acc[i][1];
            *part_v.get_unchecked_mut(o + 2) = acc[i][2];
            *part_v.get_unchecked_mut(o + 3) = acc[i][3];
        }
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// [`fold_tile_w`] for the PACK-generic passes, the same steps in the same
/// order: the `_256` entries keep their own body, and this copy serves the
/// `_p4` and `_p12_sel` entries, whose weight tile holds one row per warp of
/// the block.
///
/// SAFETY: as [`fold_tile`], with `ws` a weight tile of `KEY_TILE` f32 per
/// warp of the block and `vs` the staged `KEY_TILE · HEAD/4` value tile.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the tile step's state is the caller's registers, passed by reference"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn fold_tile_w_p<const HEAD: usize, const QW: usize>(
    sc: f32,
    live: bool,
    mx: &mut f32,
    s_sum: &mut f32,
    acc: &mut [[f32; 4]; QW],
    ws: *mut f32,
    vs: *const u64,
    w: usize,
    lane: usize,
) {
    let row_qwords = HEAD / 4;
    let mut tmax = sc;
    let mut off = 16u32;
    while off > 0 {
        tmax = tmax.max(warp::shuffle_xor_f32(tmax, off));
        off >>= 1;
    }
    let m_new = mx.max(tmax);
    let pw = if live { dev_exp(sc - m_new) } else { 0.0 };
    let mut psum = pw;
    let mut off = 16u32;
    while off > 0 {
        psum += warp::shuffle_xor_f32(psum, off);
        off >>= 1;
    }
    if m_new > *mx {
        let f = if *mx > f32::NEG_INFINITY {
            dev_exp(*mx - m_new)
        } else {
            0.0
        };
        *s_sum *= f;
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            acc[i][0] *= f;
            acc[i][1] *= f;
            acc[i][2] *= f;
            acc[i][3] *= f;
        }
        *mx = m_new;
    }
    *s_sum += psum;

    // SAFETY: w·32 + lane is below the tile's KEY_TILE f32 per warp; this
    // warp's own slots.
    unsafe { *ws.add(w * KEY_TILE + lane) = pw };
    warp::sync_mask(u32::MAX);
    let mut j = 0usize;
    while j < KEY_TILE {
        // SAFETY: j < 32 and lane < 32: inside WS, written before the warp
        // sync above.
        let pj = unsafe { *ws.add(w * KEY_TILE + j) };
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: j < KEY_TILE and lane + 32·i < HEAD/4: inside VS,
            // written before the caller's barrier.
            let vw = unsafe { *vs.add(j * row_qwords + lane + 32 * i) };
            acc[i][0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[i][0]);
            acc[i][1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[i][1]);
            acc[i][2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[i][2]);
            acc[i][3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[i][3]);
        }
        j += 1;
    }
    warp::sync_mask(u32::MAX);
}

/// The block of the head-generic segment passes: block `b = ((seg·n_kv +
/// kh)·packs + p)·m + t` is row `t`, segment `seg` and query heads `(kh·packs
/// + p)·PACK ..` of key head `kh` — `packs` blocks of `PACK` heads per key
/// head, the rows and then the packs of one segment neighbouring blocks.
/// Returns `(t, seg, kh, head0, n_head)`, or `None` past the grid. At
/// `packs = 1` this is the one-block-per-group map of the `GROUP` entries.
#[inline(always)]
const fn seg_block<const PACK: usize>(
    b: usize,
    rows: usize,
    nkv: usize,
    packs: usize,
    n_seg: usize,
) -> Option<(usize, usize, usize, usize, usize)> {
    let units = nkv * packs;
    if b >= rows * units * n_seg {
        return None;
    }
    let t = b % rows;
    let sk = b / rows;
    let seg = sk / units;
    let r = sk - seg * units;
    Some((t, seg, r / packs, r * PACK, units * PACK))
}

/// [`seg_block`] over `rows · n_kv · packs · n_seg` blocks: every block below
/// the grid names a distinct (row, segment, query head) for each of its
/// `PACK` warps, every query head `head0 + w` reads key head `(head0 + w) /
/// (packs · PACK)` — its group's — and the blocks cover every (row, segment,
/// head) once; the block past the grid names nothing.
const fn seg_blocks_hold<const PACK: usize>(
    rows: usize,
    nkv: usize,
    packs: usize,
    n_seg: usize,
) -> bool {
    const CAP: usize = 1024;
    let blocks = rows * nkv * packs * n_seg;
    if blocks * PACK > CAP {
        return false;
    }
    let mut seen = [false; CAP];
    let mut b = 0;
    while b < blocks {
        let Some((t, seg, kh, head0, n_head)) = seg_block::<PACK>(b, rows, nkv, packs, n_seg)
        else {
            return false;
        };
        if t >= rows || seg >= n_seg || kh >= nkv || n_head != nkv * packs * PACK {
            return false;
        }
        let mut w = 0;
        while w < PACK {
            let h = head0 + w;
            let at = (t * n_head + h) * n_seg + seg;
            if h >= n_head || h / (packs * PACK) != kh || seen[at] {
                return false;
            }
            seen[at] = true;
            w += 1;
        }
        b += 1;
    }
    let mut i = 0;
    while i < blocks * PACK {
        if !seen[i] {
            return false;
        }
        i += 1;
    }
    seg_block::<PACK>(blocks, rows, nkv, packs, n_seg).is_none()
}

/// Query heads one block of the `_p2` entry takes
/// ([`FlashGqaKernels::enqueue_pass_256_p2`]): a group of any even count of
/// query heads per key head runs as `group / 2` blocks per key head.
pub const PACK_2: usize = 2;
const THREADS_P2: usize = PACK_2 * 32;
const THREADS_P2_U32: u32 = THREADS_P2 as u32;
// The `_p2` launch contract spells PACK_2 and its block out as 2 and 64.
const _: () = assert!(PACK_2 == 2 && THREADS_P2_U32 == 64);

// The block maps: at `GROUP` and `packs = 1` the map the eight-head passes
// compute inline, and the `_p4` entries' packs of four at groups 4, 8 and 12
// (Qwen3.8's 24/2), over several rows and segments.
const _: () = assert!(seg_blocks_hold::<GROUP>(3, 2, 1, 4) && seg_blocks_hold::<GROUP>(2, 4, 1, 3));
const _: () =
    assert!(seg_blocks_hold::<PACK_4>(3, 2, 3, 4) && seg_blocks_hold::<PACK_4>(2, 3, 2, 3));
const _: () =
    assert!(seg_blocks_hold::<PACK_4>(1, 2, 1, 5) && seg_blocks_hold::<PACK_4>(2, 1, 3, 7));
// The `_p2` entry's packs of two at group 6 (Qwen3.5-9B's 24/4) and 2.
const _: () =
    assert!(seg_blocks_hold::<PACK_2>(3, 4, 3, 4) && seg_blocks_hold::<PACK_2>(2, 3, 1, 5));

/// [`seg_scalar`] in blocks of `PACK` query heads — the `_256` entries keep
/// their own body, and this PACK-generic copy serves the `_p4` entries. The
/// scalar segment pass of [`flash_gqa_kernels::gqa_flash_seg`] over a head of
/// `HEAD = QW · 128` values and blocks of `PACK` query heads
/// ([`seg_block`], `packs` of them per key head): the partial index, the
/// neutral segment, the staging and the reduction structure of the module
/// doc, warp `w` query head `head0 + w`, lane `l` holding dims `4l + 128·i ..
/// +3`.
///
/// SAFETY: the entry's launch contract at `HEAD` and `packs · PACK` query
/// heads per key head (every buffer bound it names), a block of `PACK · 32`
/// threads, and the four tiles this block's shared memory as the entry
/// declared them: the pack's query rows (`PACK · HEAD` f32), the key tile at
/// stride `HEAD/2 + 1` words, the value tile (`KEY_TILE · HEAD/4` u64) and the
/// weight tile (`PACK · KEY_TILE`).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn seg_scalar_p<const HEAD: usize, const QW: usize, const PACK: usize>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    packs: usize,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qs: *mut f32,
    ks: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(HEAD == QW * SLICE && HEAD.is_multiple_of(2 * ILP)) };
    const { assert!((KEY_TILE * HEAD / 4).is_multiple_of(PACK * 32)) };
    let row_words = HEAD / 2;
    let k_stride = row_words + 1;
    let row_qwords = HEAD / 4;
    let threads = PACK * 32;
    // q values one thread stages, and K/V tile words per tile.
    let per_thread = HEAD / 32;
    let stage_per = KEY_TILE * row_qwords / threads;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    let Some((t, seg, kh, head0, n_head)) = seg_block::<PACK>(b, rows, nkv, packs, n_seg) else {
        return; // block-uniform
    };
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let h = head0 + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let span = seg_span(limit, n_seg, seg_keys as usize);
    let lo = seg * span;
    if lo >= limit {
        if lane == 0 {
            // SAFETY: idx < m·n_head·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo and limit are the block's
    }
    let hi = (lo + span).min(limit);

    // Row t's pack of query rows: thread `tid` stages values
    // `per_thread·tid ..` of the pack's PACK·HEAD.
    let qb = (t * n_head + head0) * HEAD;
    let mut i = 0usize;
    while i < per_thread {
        // SAFETY: qb + per_thread·tid + i < (t·n_head + head0 + PACK)·HEAD <=
        // m·n_head·HEAD <= q.len(); the shared index < PACK·HEAD.
        unsafe { *qs.add(per_thread * tid + i) = *q.get_unchecked(qb + per_thread * tid + i) };
        i += 1;
    }

    let plane = kh * ctx * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; QW];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        // Stage the tile: 32 keys × HEAD/4 u64 of K and of V; a key at or
        // past `hi` stages zeros.
        let mut i = 0usize;
        while i < stage_per {
            let e = tid + threads * i;
            let key = e / row_qwords;
            let word = e - key * row_qwords;
            let (kw, vw) = if t0 + key < hi {
                // SAFETY: t0 + key < hi <= ctx, so the row is inside key
                // head kh's plane; a row is HEAD u16 = HEAD/4 u64 and the
                // planes start 8-byte aligned (a device allocation), so the
                // u64 read is aligned and inside both planes.
                unsafe {
                    let r = (plane + (t0 + key) * HEAD) / 4 + word;
                    (*k64.add(r), *v64.add(r))
                }
            } else {
                (0u64, 0u64)
            };
            // SAFETY: key < 32 and word < HEAD/4: 2·word + 1 < k_stride and
            // key·HEAD/4 + word < KEY_TILE·HEAD/4.
            unsafe {
                *ks.add(key * k_stride + 2 * word) = kw as u32;
                *ks.add(key * k_stride + 2 * word + 1) = (kw >> 32) as u32;
                *vs.add(key * row_qwords + word) = vw;
            }
            i += 1;
        }
        thread::sync_threads();

        // The score of key t0 + lane for head h.
        let live = t0 + lane < hi;
        let mut a = [0.0f32; ILP];
        let mut wd = 0usize;
        while wd < row_words {
            let mut p = 0usize;
            while p < ILP {
                // SAFETY: lane < 32, wd + p < row_words: inside KS; the query
                // index w·HEAD + 2(wd + p) + 1 < PACK·HEAD.
                let (kw, q0, q1) = unsafe {
                    (
                        *ks.add(lane * k_stride + wd + p),
                        *qs.add(w * HEAD + 2 * (wd + p)),
                        *qs.add(w * HEAD + 2 * (wd + p) + 1),
                    )
                };
                let k0 = half_bits_to_f32(kw as u16);
                let k1 = half_bits_to_f32((kw >> 16) as u16);
                a[p] = f32::mul_add(q0, k0, a[p]);
                a[p] = f32::mul_add(q1, k1, a[p]);
                p += 1;
            }
            wd += ILP;
        }
        let dot = (a[0] + a[1]) + (a[2] + a[3]);
        let sc = if live { dot * scale } else { f32::NEG_INFINITY };

        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barrier above; w < PACK and lane < 32.
        unsafe {
            fold_tile_w_p::<HEAD, QW>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: idx < m·n_head·segs; dims 4·lane + 128·i .. +3 of the
    // partial row are this lane's alone, and lane 0 writes (m, s).
    unsafe {
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            let o = idx * HEAD + 4 * lane + SLICE * i;
            *part_v.get_unchecked_mut(o) = acc[i][0];
            *part_v.get_unchecked_mut(o + 1) = acc[i][1];
            *part_v.get_unchecked_mut(o + 2) = acc[i][2];
            *part_v.get_unchecked_mut(o + 3) = acc[i][3];
        }
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// [`seg_mma`] in blocks of `PACK` query heads — the `_256` entries keep
/// their own body, and this PACK-generic copy serves the `_p4` entries. The
/// tensor-core segment pass of [`flash_gqa_kernels::gqa_flash_seg_mma`] over a
/// head of `HEAD = QW · 128` values and blocks of `PACK` query heads: the
/// grid, the staging and every step after the scores are [`seg_scalar_p`]'s
/// (`fold_tile_w_p`). Warp `w` rounds its head's `HEAD/2` value pairs into row
/// `w` of the 16-row query tile, `TILE_PACKS = MMA_ROWS / PACK` packs of rows
/// (rows `PACK..16` zero); warps `0..4` each take eight keys of a tile and
/// accumulate the score of each 128-value slice `i` of the head in its own
/// f32 accumulator, the slice's eight k16 steps ascending from zero (two
/// `mma.m16n8k16` per `ldmatrix.x4` of the key), then add the slices in
/// ascending order — the order the prefill flash adds its two warps' half
/// scores in, so both compute a key's score to the same bits — and write
/// `scale · S`. Both staged strides are `mma_row_words(HEAD)` words.
///
/// SAFETY: the entry's launch contract at `HEAD` and `packs · PACK` query
/// heads per key head, a block of `PACK · 32` threads, and the four tiles this
/// block's shared memory as the entry declared them: the query tile
/// (`MMA_ROWS · mma_row_words(HEAD)` u32), the key tile (`KEY_TILE ·
/// mma_row_words(HEAD)` u32), the value tile (`KEY_TILE · HEAD/4` u64) and the
/// weight tile (`PACK · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn seg_mma_p<
    const HEAD: usize,
    const QW: usize,
    const PACK: usize,
    const TILE_PACKS: usize,
>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    packs: usize,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qt: *mut u32,
    kt: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(HEAD == QW * SLICE) };
    // The pack's heads are rows `0..PACK` of the 16-row tile, the fragment's
    // rows `lane/4 < 8`; four warps take a tile's keys.
    const { assert!(mma_row_words(HEAD) * 4 % 128 == 16 && MMA_ROWS.is_multiple_of(PACK)) };
    const { assert!(PACK >= 4 && PACK <= MMA_ROWS / 2) };
    // A parameter, not `MMA_ROWS / PACK`, so the zero-row loop's bound is a
    // constant the unroller reads.
    const { assert!(TILE_PACKS * PACK == MMA_ROWS) };
    const { assert!(KEY_TILE == 4 * MMA_NTILE && SLICE.is_multiple_of(2 * MMA_K)) };
    const { assert!((KEY_TILE * HEAD / 4).is_multiple_of(PACK * 32)) };
    let row_w = mma_row_words(HEAD);
    let row_qwords = HEAD / 4;
    let threads = PACK * 32;
    let stage_per = KEY_TILE * row_qwords / threads;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    let Some((t, seg, kh, head0, n_head)) = seg_block::<PACK>(b, rows, nkv, packs, n_seg) else {
        return; // block-uniform
    };
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let h = head0 + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let span = seg_span(limit, n_seg, seg_keys as usize);
    let lo = seg * span;
    if lo >= limit {
        if lane == 0 {
            // SAFETY: idx < m·n_head·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo and limit are the block's
    }
    let hi = (lo + span).min(limit);

    // The query tile: warp `w` rounds row t's head `head0 + w`'s HEAD/2 value
    // pairs, HEAD/64 per lane, into row `w`, and writes the zero rows `z·PACK +
    // w` for `z >= 1`. Each pair is one 8-byte load, and the loads come before
    // the roundings.
    let qb = (t * n_head + head0 + w) * HEAD;
    let q64 = q.as_ptr() as *const u64;
    let mut raw = [[0u64; 2]; QW];
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: qb is a multiple of HEAD, so word qb/2 + wd holds values
            // qb + 2·wd and + 1 (wd = lane + 32·i + 64·s < HEAD/2), inside
            // (t·n_head + h + 1)·HEAD <= m·n_head·HEAD <= q.len(); the buffer
            // starts 8-byte aligned (a device allocation).
            unsafe { raw[s][i] = *q64.add(qb / 2 + lane + 32 * i + 64 * s) };
        }
    }
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let wd = lane + 32 * i + 64 * s;
            let lo16 = f32_to_f16_bits(f32::from_bits(raw[s][i] as u32)) as u32;
            let hi16 = f32_to_f16_bits(f32::from_bits((raw[s][i] >> 32) as u32)) as u32;
            // SAFETY: the tile words w·row_w + wd and (z·PACK + w)·row_w + wd
            // are inside QT (wd < HEAD/2 < row_w, z·PACK + w < MMA_ROWS).
            unsafe {
                *qt.add(w * row_w + wd) = lo16 | (hi16 << 16);
                *qt.add((PACK + w) * row_w + wd) = 0;
                for z in 2..TILE_PACKS {
                    cuda_device::thread::__unroll_config::<0>();
                    *qt.add((z * PACK + w) * row_w + wd) = 0;
                }
            }
        }
    }

    let plane = kh * ctx * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let key0 = (w % 4) * MMA_NTILE;
    let arow = lane % MMA_ROWS;
    let ahalf = lane / MMA_ROWS;
    let bkey = lane % MMA_NTILE;
    let boct = lane / MMA_NTILE;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; QW];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        let mut i = 0usize;
        while i < stage_per {
            let e = tid + threads * i;
            let key = e / row_qwords;
            let word = e - key * row_qwords;
            let (kw, vw) = if t0 + key < hi {
                // SAFETY: as in seg_scalar — the row is inside key head kh's
                // plane and the u64 read is aligned.
                unsafe {
                    let r = (plane + (t0 + key) * HEAD) / 4 + word;
                    (*k64.add(r), *v64.add(r))
                }
            } else {
                (0u64, 0u64)
            };
            // SAFETY: key < 32 and 2·word + 1 < HEAD/2 < row_w: inside KT;
            // key·HEAD/4 + word inside VS.
            unsafe {
                *kt.add(key * row_w + 2 * word) = kw as u32;
                *kt.add(key * row_w + 2 * word + 1) = (kw >> 32) as u32;
                *vs.add(key * row_qwords + word) = vw;
            }
            i += 1;
        }
        thread::sync_threads();

        if PACK <= 4 || w < 4 {
            let mut c = [[0.0f32; 4]; QW];
            for s in 0..QW {
                cuda_device::thread::__unroll_config::<0>();
                let mut d = SLICE * s;
                while d < SLICE * (s + 1) {
                    // SAFETY: key row key0 + bkey < KEY_TILE and words d/2 +
                    // 4·boct .. + 4 <= HEAD/2 are inside KT; every lane of the
                    // warp issues the load (the branch is on the warp index),
                    // after the barrier that staged it.
                    let bf = unsafe {
                        let bp = kt.add((key0 + bkey) * row_w + d / 2 + boct * 4);
                        wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            bp.cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: query row arow < MMA_ROWS and words d/2 +
                    // 4·ahalf .. + 4 + MMA_K/2 <= HEAD/2 are inside QT,
                    // published by the first barrier; the whole warp issues
                    // both loads and both `mma.sync` with its own fragments.
                    unsafe {
                        let ap0 = qt.add(arow * row_w + d / 2 + ahalf * (MMA_K / 4));
                        let af0 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap0.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af0, [bf[0], bf[1]]);
                        let ap1 = ap0.add(MMA_K / 2);
                        let af1 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap1.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af1, [bf[2], bf[3]]);
                    }
                    d += 2 * MMA_K;
                }
            }
            // The slices' scores added in ascending order.
            let mut sc0 = c[0][0];
            let mut sc1 = c[0][1];
            for s in 1..QW {
                cuda_device::thread::__unroll_config::<0>();
                sc0 = add_rn_f32(sc0, c[s][0]);
                sc1 = add_rn_f32(sc1, c[s][1]);
            }
            // sc0, sc1: tile row lane/4 at keys key0 + 2·(lane%4) + {0, 1},
            // the pack's head when lane/4 < PACK; each slice's c[2], c[3] are
            // zero rows.
            let g = lane / 4;
            let kk = key0 + 2 * (lane % 4);
            let s0 = if t0 + kk < hi {
                mul_rn_f32(scale, sc0)
            } else {
                f32::NEG_INFINITY
            };
            let s1 = if t0 + kk + 1 < hi {
                mul_rn_f32(scale, sc1)
            } else {
                f32::NEG_INFINITY
            };
            // A pack of eight fills rows 0..8, so every lane's row is a head.
            if PACK >= MMA_ROWS / 2 || g < PACK {
                // SAFETY: g < PACK and kk + 1 < KEY_TILE: inside WS; each
                // (head, key) slot has one writer.
                unsafe {
                    *ws.add(g * KEY_TILE + kk) = s0;
                    *ws.add(g * KEY_TILE + kk + 1) = s1;
                }
            }
        }
        thread::sync_threads();
        // SAFETY: w < PACK and lane < KEY_TILE: inside WS, written before
        // the barrier above.
        let sc = unsafe { *ws.add(w * KEY_TILE + lane) };
        let live = t0 + lane < hi;
        warp::sync_mask(u32::MAX);
        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barriers above; w < PACK and lane < 32.
        unsafe {
            fold_tile_w_p::<HEAD, QW>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: as in seg_scalar.
    unsafe {
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            let o = idx * HEAD + 4 * lane + SLICE * i;
            *part_v.get_unchecked_mut(o) = acc[i][0];
            *part_v.get_unchecked_mut(o + 1) = acc[i][1];
            *part_v.get_unchecked_mut(o + 2) = acc[i][2];
            *part_v.get_unchecked_mut(o + 3) = acc[i][3];
        }
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// A `_p12_sel` row's keys: key `j` of row `t` is cache row
/// `list[t·width + j]`, the count at most `width`. [`seg_mma_ps`] takes a
/// row's count bound from [`Listed::count_cap`] and stages a tile through
/// [`Listed::stage_row`]; every other step is [`seg_mma_p`]'s. An entry at or
/// past the cache
/// raises [`FaultSite::PoolSelect`] and stages the f16 NaN pattern in both
/// planes, so the row's output is NaN, never a plausible value.
#[derive(Clone, Copy)]
struct Listed<'a> {
    list: &'a [u32],
    width: usize,
    fault: FaultSink,
}

impl Listed<'_> {
    /// The largest live count a row may carry: past it the row is refused
    /// ([`live_keys`]).
    #[inline(always)]
    fn count_cap(self) -> usize {
        self.width
    }

    /// Word `word` (u64) of key `j` of row `t`, in the key and the value
    /// plane: `plane` is the key head's first value, the planes `ctx` rows
    /// of `HEAD` f16 per key head.
    ///
    /// # Safety
    /// `j` is below row `t`'s live count ([`live_keys`] at
    /// [`Listed::count_cap`]), `t` a row of the launch, `word < HEAD/4`,
    /// `plane` a key head's first value, and the entry's launch contract
    /// holds.
    #[inline(always)]
    #[allow(
        clippy::too_many_arguments,
        reason = "the staging's state is the caller's registers, passed by value"
    )]
    unsafe fn stage_row<const HEAD: usize>(
        self,
        t: usize,
        j: usize,
        word: usize,
        plane: usize,
        ctx: usize,
        k64: *const u64,
        v64: *const u64,
    ) -> (u64, u64) {
        // SAFETY: j < count <= width and t < m, so t·width + j < m·width <=
        // list.len() by the launch contract.
        let row = unsafe { *self.list.get_unchecked(t * self.width + j) } as usize;
        if row >= ctx {
            self.fault.raise(FaultSite::PoolSelect);
            return (SEL_NAN4, SEL_NAN4);
        }
        // SAFETY: row < ctx, so the row is inside the key head's plane; a
        // row is HEAD u16 = HEAD/4 u64 and the planes start 8-byte aligned
        // (a device allocation), so the u64 read is aligned and inside both
        // planes.
        unsafe {
            let r = (plane + row * HEAD) / 4 + word;
            (*k64.add(r), *v64.add(r))
        }
    }
}

/// f16 NaN in each of a u64's four halves: the stage of a listed row past
/// the cache.
const SEL_NAN4: u64 = 0x7e00_7e00_7e00_7e00;

/// [`seg_mma_p`] over a row's [`Listed`] keys, for a pack that fills more
/// than half of the 16-row query tile — the `_p4` entries keep their own
/// body, and this one serves the `_p12_sel` entry. Every step is
/// [`seg_mma_p`]'s except the row's count bound ([`Listed::count_cap`]), the
/// tile's staging load ([`Listed::stage_row`]) and the pack's rows: `PACK`
/// heads in rows `0..PACK`, `PAD` zero rows after them, the heads past row 7
/// read back from the accumulator's `c[2]`, `c[3]` (rows `lane/4 + 8`).
///
/// SAFETY: [`seg_mma_p`]'s, with key `j < count` of row `t` read through
/// `walk` under [`Listed::stage_row`]'s contract, a block of `PACK · 32`
/// threads and a weight tile of `PACK · KEY_TILE` f32.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn seg_mma_ps<const HEAD: usize, const QW: usize, const PACK: usize, const PAD: usize>(
    walk: Listed<'_>,
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    packs: usize,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qt: *mut u32,
    kt: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(HEAD == QW * SLICE) };
    // The pack's heads are rows `0..PACK` of the 16-row tile, the fragment's
    // rows `lane/4` and, past 8, `lane/4 + 8`; four warps take a tile's keys,
    // and the `PAD` zero rows are written one a warp.
    const { assert!(mma_row_words(HEAD) * 4 % 128 == 16 && PACK >= 4) };
    const { assert!(PACK + PAD == MMA_ROWS && PAD <= PACK) };
    const { assert!(KEY_TILE == 4 * MMA_NTILE && SLICE.is_multiple_of(2 * MMA_K)) };
    let row_w = mma_row_words(HEAD);
    let row_qwords = HEAD / 4;
    let threads = PACK * 32;
    // The tile's words: the block's threads take `stage_per` of them each, the
    // last pass short when the block does not divide the tile.
    let tile_words = KEY_TILE * row_qwords;
    let stage_per = tile_words.div_ceil(threads);

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    let Some((t, seg, kh, head0, n_head)) = seg_block::<PACK>(b, rows, nkv, packs, n_seg) else {
        return; // block-uniform
    };
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let h = head0 + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, walk.count_cap());
    let span = seg_span(limit, n_seg, seg_keys as usize);
    let lo = seg * span;
    if lo >= limit {
        if lane == 0 {
            // SAFETY: idx < m·n_head·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo and limit are the block's
    }
    let hi = (lo + span).min(limit);

    // The query tile: warp `w` rounds row t's head `head0 + w`'s HEAD/2 value
    // pairs, HEAD/64 per lane, into row `w`, and warps `w < PAD` write the
    // zero row `PACK + w`. Each pair is one 8-byte load, and the loads come
    // before the roundings.
    let qb = (t * n_head + head0 + w) * HEAD;
    let q64 = q.as_ptr() as *const u64;
    let mut raw = [[0u64; 2]; QW];
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: qb is a multiple of HEAD, so word qb/2 + wd holds values
            // qb + 2·wd and + 1 (wd = lane + 32·i + 64·s < HEAD/2), inside
            // (t·n_head + h + 1)·HEAD <= m·n_head·HEAD <= q.len(); the buffer
            // starts 8-byte aligned (a device allocation).
            unsafe { raw[s][i] = *q64.add(qb / 2 + lane + 32 * i + 64 * s) };
        }
    }
    for s in 0..QW {
        cuda_device::thread::__unroll_config::<0>();
        for i in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let wd = lane + 32 * i + 64 * s;
            let lo16 = f32_to_f16_bits(f32::from_bits(raw[s][i] as u32)) as u32;
            let hi16 = f32_to_f16_bits(f32::from_bits((raw[s][i] >> 32) as u32)) as u32;
            // SAFETY: the tile words w·row_w + wd and (PACK + w)·row_w + wd
            // are inside QT (wd < HEAD/2 < row_w, w < PACK, PACK + w <
            // MMA_ROWS for w < PAD).
            unsafe {
                *qt.add(w * row_w + wd) = lo16 | (hi16 << 16);
                if w < PAD {
                    *qt.add((PACK + w) * row_w + wd) = 0;
                }
            }
        }
    }

    let plane = kh * ctx * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let key0 = (w % 4) * MMA_NTILE;
    let arow = lane % MMA_ROWS;
    let ahalf = lane / MMA_ROWS;
    let bkey = lane % MMA_NTILE;
    let boct = lane / MMA_NTILE;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; QW];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        let mut i = 0usize;
        while i < stage_per {
            let e = tid + threads * i;
            if e < tile_words {
                let key = e / row_qwords;
                let word = e - key * row_qwords;
                let (kw, vw) = if t0 + key < hi {
                    // SAFETY: t0 + key < hi <= the row's live count, t < m,
                    // word < HEAD/4 and plane is key head kh's first value.
                    unsafe { walk.stage_row::<HEAD>(t, t0 + key, word, plane, ctx, k64, v64) }
                } else {
                    (0u64, 0u64)
                };
                // SAFETY: e < tile_words, so key < 32 and 2·word + 1 <
                // HEAD/2 < row_w: inside KT; key·HEAD/4 + word inside VS.
                unsafe {
                    *kt.add(key * row_w + 2 * word) = kw as u32;
                    *kt.add(key * row_w + 2 * word + 1) = (kw >> 32) as u32;
                    *vs.add(key * row_qwords + word) = vw;
                }
            }
            i += 1;
        }
        thread::sync_threads();

        if PACK <= 4 || w < 4 {
            let mut c = [[0.0f32; 4]; QW];
            for s in 0..QW {
                cuda_device::thread::__unroll_config::<0>();
                let mut d = SLICE * s;
                while d < SLICE * (s + 1) {
                    // SAFETY: key row key0 + bkey < KEY_TILE and words d/2 +
                    // 4·boct .. + 4 <= HEAD/2 are inside KT; every lane of the
                    // warp issues the load (the branch is on the warp index),
                    // after the barrier that staged it.
                    let bf = unsafe {
                        let bp = kt.add((key0 + bkey) * row_w + d / 2 + boct * 4);
                        wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            bp.cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: query row arow < MMA_ROWS and words d/2 +
                    // 4·ahalf .. + 4 + MMA_K/2 <= HEAD/2 are inside QT,
                    // published by the first barrier; the whole warp issues
                    // both loads and both `mma.sync` with its own fragments.
                    unsafe {
                        let ap0 = qt.add(arow * row_w + d / 2 + ahalf * (MMA_K / 4));
                        let af0 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap0.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af0, [bf[0], bf[1]]);
                        let ap1 = ap0.add(MMA_K / 2);
                        let af1 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap1.cast_const().cast::<u8>(),
                        ));
                        c[s] = wmma::mma_m16n8k16_f32_f16(c[s], af1, [bf[2], bf[3]]);
                    }
                    d += 2 * MMA_K;
                }
            }
            // The slices' scores added in ascending order.
            let mut sc0 = c[0][0];
            let mut sc1 = c[0][1];
            let mut sc2 = c[0][2];
            let mut sc3 = c[0][3];
            for s in 1..QW {
                cuda_device::thread::__unroll_config::<0>();
                sc0 = add_rn_f32(sc0, c[s][0]);
                sc1 = add_rn_f32(sc1, c[s][1]);
                sc2 = add_rn_f32(sc2, c[s][2]);
                sc3 = add_rn_f32(sc3, c[s][3]);
            }
            // sc0, sc1: tile row lane/4 at keys key0 + 2·(lane%4) + {0, 1},
            // the pack's head when lane/4 < PACK; sc2, sc3: row lane/4 + 8 at
            // the same keys, a head when lane/4 + 8 < PACK, else a zero row.
            let g = lane / 4;
            let kk = key0 + 2 * (lane % 4);
            let live0 = t0 + kk < hi;
            let live1 = t0 + kk + 1 < hi;
            let s0 = if live0 {
                mul_rn_f32(scale, sc0)
            } else {
                f32::NEG_INFINITY
            };
            let s1 = if live1 {
                mul_rn_f32(scale, sc1)
            } else {
                f32::NEG_INFINITY
            };
            // A pack of eight or more fills rows 0..8, so every lane's row is
            // a head.
            if PACK >= MMA_ROWS / 2 || g < PACK {
                // SAFETY: g < PACK and kk + 1 < KEY_TILE: inside WS; each
                // (head, key) slot has one writer.
                unsafe {
                    *ws.add(g * KEY_TILE + kk) = s0;
                    *ws.add(g * KEY_TILE + kk + 1) = s1;
                }
            }
            if g + MMA_ROWS / 2 < PACK {
                let s2 = if live0 {
                    mul_rn_f32(scale, sc2)
                } else {
                    f32::NEG_INFINITY
                };
                let s3 = if live1 {
                    mul_rn_f32(scale, sc3)
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: g + 8 < PACK and kk + 1 < KEY_TILE: inside WS; a
                // different row from the arm above, one writer a slot.
                unsafe {
                    *ws.add((g + MMA_ROWS / 2) * KEY_TILE + kk) = s2;
                    *ws.add((g + MMA_ROWS / 2) * KEY_TILE + kk + 1) = s3;
                }
            }
        }
        thread::sync_threads();
        // SAFETY: w < PACK and lane < KEY_TILE: inside WS, written before
        // the barrier above.
        let sc = unsafe { *ws.add(w * KEY_TILE + lane) };
        let live = t0 + lane < hi;
        warp::sync_mask(u32::MAX);
        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barriers above; w < PACK and lane < 32.
        unsafe {
            fold_tile_w_p::<HEAD, QW>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: idx < m·n_head·segs; dims 4·lane + 128·i .. +3 of the
    // partial row are this lane's alone, and lane 0 writes (m, s).
    unsafe {
        for i in 0..QW {
            cuda_device::thread::__unroll_config::<0>();
            let o = idx * HEAD + 4 * lane + SLICE * i;
            *part_v.get_unchecked_mut(o) = acc[i][0];
            *part_v.get_unchecked_mut(o + 1) = acc[i][1];
            *part_v.get_unchecked_mut(o + 2) = acc[i][2];
            *part_v.get_unchecked_mut(o + 3) = acc[i][3];
        }
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// The merge of [`flash_gqa_kernels::gqa_flash_merge`] over a head of
/// `HEAD` values: block `b = t·n_head + h`, thread `d` owning dim `d`, the
/// live segments folded ascending through `online_fold` in batches of
/// [`MERGE_BATCH`] loads, a refused row raising [`FaultSite::KeyCount`] and
/// NaN in every dim.
///
/// SAFETY: the entry's launch contract at `HEAD` and a block of `HEAD`
/// threads.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn merge_body<const HEAD: usize>(
    part_v: &[f32],
    part_ms: &[f32],
    n_keys_buf: &[u32],
    n_head: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    fault: FaultSink,
    mut y: DisjointSlice<f32>,
) {
    let b = thread::blockIdx_x() as usize;
    let nh = n_head as usize;
    if b >= m as usize * nh {
        return; // block-uniform
    }
    let t = b / nh;
    let d = thread::threadIdx_x() as usize;
    let n_seg = segs as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx as usize);
    if limit == 0 {
        if d == 0 {
            fault.raise(FaultSite::KeyCount);
        }
        // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = f32::NAN };
        return; // block-uniform: the row is the block's
    }
    // seg_span keeps ⌈limit / span⌉ <= segs (its const check), so the walk
    // below stays inside the row's segments.
    let live = limit.div_ceil(seg_span(limit, n_seg, seg_keys as usize));
    let mut mx = f32::NEG_INFINITY;
    let mut s = 0.0f32;
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < live {
        let mut ms = [0.0f32; 2 * MERGE_BATCH];
        let mut vs = [0.0f32; MERGE_BATCH];
        for i in 0..MERGE_BATCH {
            cuda_device::thread::__unroll_config::<0>();
            if j + i < live {
                let idx = b * n_seg + j + i;
                // SAFETY: idx < m·n_head·segs: both slots inside part_ms,
                // and idx·HEAD + d < m·n_head·segs·HEAD <= part_v.len(). A
                // live segment wrote all three.
                unsafe {
                    ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                    ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                    vs[i] = *part_v.get_unchecked(idx * HEAD + d);
                }
            }
        }
        for i in 0..MERGE_BATCH {
            cuda_device::thread::__unroll_config::<0>();
            if j + i < live && ms[2 * i + 1] != 0.0 {
                (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
            }
        }
        j += MERGE_BATCH;
    }
    // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
    unsafe { *y.get_unchecked_mut(b * HEAD + d) = acc * (1.0 / s) };
}

// ------------------------------------------------------------ the K192 bodies
//
// MiMo-V2's heads: scores over 192 key values, values of 128, per layer a
// window of the positions (the sliding layers) and a per-head sink in the
// softmax. New bodies over [`HEAD_K192`] and [`HEAD`] that reuse the walks
// above without editing them: the tile step is `fold_tile_w_p` at the value
// width, the block map `seg_block`, the cut [`seg_span`] over the window's
// keys ([`window_cut`]).

/// The scalar segment pass of [`flash_gqa_kernels::gqa_flash_seg_k192`] in
/// blocks of `PACK` query heads ([`seg_block`], `packs` of them per key
/// head): [`seg_scalar_p`]'s grid, partial index, neutral segment, tile
/// walk and reduction structure at a key width of [`HEAD_K192`] and a value
/// width of [`HEAD`], over the keys [`window_cut`] names — segment `seg` of
/// a row of `n_w` windowed keys walks `[first + seg·span, first +
/// min((seg + 1)·span, n_w))` with `span` = [`seg_span`] of `n_w`. Lane `j`
/// scores key `t0 + j` over the 96 key words (partial `p` takes the word
/// pairs `p, p + 4, …`), lane `l` holds value dims `4l .. 4l + 3`.
///
/// SAFETY: the entry's launch contract (every buffer bound it names), a block
/// of `PACK · 32` threads, and the four tiles this block's shared memory as
/// the entry declared them: the pack's query rows (`PACK · HEAD_K192` f32),
/// the key tile at stride `K_STRIDE_K192` words, the value tile (`KEY_TILE ·
/// HEAD/4` u64) and the weight tile (`PACK · KEY_TILE`).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn seg_scalar_kv<const PACK: usize>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    packs: usize,
    window: usize,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qs: *mut f32,
    ks: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(PACK == GROUP) };
    const { assert!((KEY_TILE * ROW_QWORDS_K192).is_multiple_of(PACK * 32)) };
    const { assert!((KEY_TILE * HEAD / 4).is_multiple_of(PACK * 32)) };
    let threads = PACK * 32;
    // q values one thread stages, and K and V tile words per tile.
    let per_thread = HEAD_K192 / 32;
    let stage_k = KEY_TILE * ROW_QWORDS_K192 / threads;
    let stage_v = KEY_TILE * ROW_QWORDS / threads;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let rows = m as usize;
    let Some((t, seg, kh, head0, n_head)) = seg_block::<PACK>(b, rows, nkv, packs, n_seg) else {
        return; // block-uniform
    };
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let h = head0 + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let (first, n_w) = window_cut(limit, window);
    let span = seg_span(n_w, n_seg, seg_keys as usize);
    let lo_rel = seg * span;
    if lo_rel >= n_w {
        if lane == 0 {
            // SAFETY: idx < m·n_head·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: lo_rel and n_w are the block's
    }
    let lo = first + lo_rel;
    let hi = first + (lo_rel + span).min(n_w);

    // Row t's pack of query rows: thread `tid` stages values
    // `per_thread·tid ..` of the pack's PACK·HEAD_K192.
    let qb = (t * n_head + head0) * HEAD_K192;
    let mut i = 0usize;
    while i < per_thread {
        // SAFETY: qb + per_thread·tid + i < (t·n_head + head0 + PACK)·HEAD_K192
        // <= m·n_head·HEAD_K192 <= q.len(); the shared index < PACK·HEAD_K192.
        unsafe { *qs.add(per_thread * tid + i) = *q.get_unchecked(qb + per_thread * tid + i) };
        i += 1;
    }

    let k_plane = kh * ctx * HEAD_K192;
    let v_plane = kh * ctx * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [[0.0f32; 4]; 1];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        // Stage the tile: 32 keys × 48 u64 of K and × 32 u64 of V; a key at
        // or past `hi` stages zeros.
        let mut i = 0usize;
        while i < stage_k {
            let e = tid + threads * i;
            let key = e / ROW_QWORDS_K192;
            let word = e - key * ROW_QWORDS_K192;
            let kw = if t0 + key < hi {
                // SAFETY: t0 + key < hi <= ctx, so the row is inside key head
                // kh's key plane; a row is HEAD_K192 u16 = 48 u64 and the
                // plane starts 8-byte aligned (a device allocation), so the
                // u64 read is aligned and inside the plane.
                unsafe { *k64.add((k_plane + (t0 + key) * HEAD_K192) / 4 + word) }
            } else {
                0u64
            };
            // SAFETY: key < 32 and word < 48: 2·word + 1 < K_STRIDE_K192.
            unsafe {
                *ks.add(key * K_STRIDE_K192 + 2 * word) = kw as u32;
                *ks.add(key * K_STRIDE_K192 + 2 * word + 1) = (kw >> 32) as u32;
            }
            i += 1;
        }
        let mut i = 0usize;
        while i < stage_v {
            let e = tid + threads * i;
            let key = e / ROW_QWORDS;
            let word = e - key * ROW_QWORDS;
            let vw = if t0 + key < hi {
                // SAFETY: as the key row, in the value plane of HEAD u16 =
                // 32 u64 a row.
                unsafe { *v64.add((v_plane + (t0 + key) * HEAD) / 4 + word) }
            } else {
                0u64
            };
            // SAFETY: key < 32 and word < 32: inside VS.
            unsafe { *vs.add(key * ROW_QWORDS + word) = vw };
            i += 1;
        }
        thread::sync_threads();

        // The score of key t0 + lane for head h.
        let live = t0 + lane < hi;
        let mut a = [0.0f32; ILP];
        let mut wd = 0usize;
        while wd < ROW_WORDS_K192 {
            let mut p = 0usize;
            while p < ILP {
                // SAFETY: lane < 32, wd + p < ROW_WORDS_K192: inside KS; the
                // query index w·HEAD_K192 + 2(wd + p) + 1 < PACK·HEAD_K192.
                let (kw, q0, q1) = unsafe {
                    (
                        *ks.add(lane * K_STRIDE_K192 + wd + p),
                        *qs.add(w * HEAD_K192 + 2 * (wd + p)),
                        *qs.add(w * HEAD_K192 + 2 * (wd + p) + 1),
                    )
                };
                let k0 = half_bits_to_f32(kw as u16);
                let k1 = half_bits_to_f32((kw >> 16) as u16);
                a[p] = f32::mul_add(q0, k0, a[p]);
                a[p] = f32::mul_add(q1, k1, a[p]);
                p += 1;
            }
            wd += ILP;
        }
        let dot = (a[0] + a[1]) + (a[2] + a[3]);
        let sc = if live { dot * scale } else { f32::NEG_INFINITY };

        // SAFETY: WS and VS are this block's tiles, VS staged before the
        // barrier above; w < PACK and lane < 32.
        unsafe {
            fold_tile_w_p::<HEAD, 1>(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane)
        };
        t0 += KEY_TILE;
    }

    // SAFETY: idx < m·n_head·segs; dims 4·lane .. +3 of the partial row are
    // this lane's alone, and lane 0 writes (m, s).
    unsafe {
        let o = idx * HEAD + 4 * lane;
        *part_v.get_unchecked_mut(o) = acc[0][0];
        *part_v.get_unchecked_mut(o + 1) = acc[0][1];
        *part_v.get_unchecked_mut(o + 2) = acc[0][2];
        *part_v.get_unchecked_mut(o + 3) = acc[0][3];
        if lane == 0 {
            *part_ms.get_unchecked_mut(2 * idx) = mx;
            *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
        }
    }
}

/// The merge of [`flash_gqa_kernels::gqa_flash_merge_sink`]: [`merge_body`]'s
/// walk at [`HEAD`] over the window's segments (`window_cut`'s `n_w` keys cut
/// by [`seg_span`], the live segments folded ascending through
/// `online_fold`), then head `h`'s sink joins as the partial `(sink, 1, 0)`
/// — unscaled, one more logit in the denominator and nothing in the value —
/// and `acc · (1/s)`. Segments ascending, then the sink: this order is the
/// gate. A sink of `−inf` is neutral (`dev_exp(−inf − mx)` is 0, nothing
/// changes but the sign of a zero); a NaN or `+inf` sink has no defined
/// softmax and writes NaN into the row. A refused row raises
/// [`FaultSite::KeyCount`] and writes NaN in every dim, as [`merge_body`].
///
/// SAFETY: the entry's launch contract and a block of [`HEAD`] threads.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn merge_sink_body(
    part_v: &[f32],
    part_ms: &[f32],
    n_keys_buf: &[u32],
    sinks: &[f32],
    n_head: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    window: usize,
    fault: FaultSink,
    mut y: DisjointSlice<f32>,
) {
    let b = thread::blockIdx_x() as usize;
    let nh = n_head as usize;
    if b >= m as usize * nh {
        return; // block-uniform
    }
    let t = b / nh;
    let h = b - t * nh;
    let d = thread::threadIdx_x() as usize;
    let n_seg = segs as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx as usize);
    if limit == 0 {
        if d == 0 {
            fault.raise(FaultSite::KeyCount);
        }
        // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = f32::NAN };
        return; // block-uniform: the row is the block's
    }
    let (_, n_w) = window_cut(limit, window);
    // seg_span keeps ⌈n_w / span⌉ <= segs (its const check at the launch's
    // windows), so the walk below stays inside the row's segments.
    let live = n_w.div_ceil(seg_span(n_w, n_seg, seg_keys as usize));
    let mut mx = f32::NEG_INFINITY;
    let mut s = 0.0f32;
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < live {
        let mut ms = [0.0f32; 2 * MERGE_BATCH];
        let mut vs = [0.0f32; MERGE_BATCH];
        for i in 0..MERGE_BATCH {
            cuda_device::thread::__unroll_config::<0>();
            if j + i < live {
                let idx = b * n_seg + j + i;
                // SAFETY: idx < m·n_head·segs: both slots inside part_ms, and
                // idx·HEAD + d < m·n_head·segs·HEAD <= part_v.len(). A live
                // segment wrote all three.
                unsafe {
                    ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                    ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                    vs[i] = *part_v.get_unchecked(idx * HEAD + d);
                }
            }
        }
        for i in 0..MERGE_BATCH {
            cuda_device::thread::__unroll_config::<0>();
            if j + i < live && ms[2 * i + 1] != 0.0 {
                (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
            }
        }
        j += MERGE_BATCH;
    }
    // SAFETY: h < n_head <= sinks.len() by the launch contract.
    let sink = unsafe { *sinks.get_unchecked(h) };
    let (_, s, acc) = online_fold(mx, s, acc, sink, 1.0, 0.0);
    let out = if sink.is_nan() || sink == f32::INFINITY {
        f32::NAN
    } else {
        acc * (1.0 / s)
    };
    // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
    unsafe { *y.get_unchecked_mut(b * HEAD + d) = out };
}

#[cuda_module]
mod flash_gqa_kernels {
    use super::*;

    /// The segment pass. Block `b = (seg·n_kv + kh)·m + t` walks keys
    /// `[seg · span, min((seg + 1)·span, n_keys))` of key head `kh` —
    /// `span` = [`seg_span`] of the row's own `n_keys`, `segs` and
    /// `seg_keys` — for row `t`'s query heads `kh·GROUP ..`; warp `w` is
    /// query head `h =
    /// kh·GROUP + w`, whose partials land at index `(t·n_head + h)·segs +
    /// seg`. `n_keys` is [`live_keys`] of `n_keys_buf[t]`: a refused row's
    /// segments are all neutral. The rows of one segment are neighbouring
    /// blocks, so they read its key rows while they are in L2.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let n_seg = segs as usize;
        let rows = m as usize;
        if b >= rows * nkv * n_seg {
            return; // block-uniform
        }
        let t = b % rows;
        let sk = b / rows;
        let seg = sk / nkv;
        let kh = sk - seg * nkv;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let n_head = nkv * GROUP;
        let h = kh * GROUP + w;
        let idx = (t * n_head + h) * n_seg + seg;
        let ctx = ctx as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
        let span = seg_span(limit, n_seg, seg_keys as usize);
        let lo = seg * span;
        if lo >= limit {
            if lane == 0 {
                // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform: lo and limit are the block's
        }
        let hi = (lo + span).min(limit);

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. Every
        // index below is inside its array, and every write is ordered before
        // its reads by a block barrier (or, for WS, written and read by one
        // warp between two barriers).
        let (qs, ks, vs, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };

        // Row t's group query rows: thread `tid` stages values 4·tid .. +3 of
        // the group's 1024.
        let qb = (t * n_head + kh * GROUP) * HEAD;
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: qb + 4·tid + i < (t·n_head + (kh + 1)·GROUP)·HEAD <=
            // m·n_kv·1024 <= q.len(); the shared index 4·tid + i < GROUP·HEAD.
            unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
            i += 1;
        }

        let plane = kh * ctx * HEAD;
        let k64 = kc.as_ptr() as *const u64;
        let v64 = vc.as_ptr() as *const u64;
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = [0.0f32; 4];
        let mut t0 = lo;
        while t0 < hi {
            thread::sync_threads();
            // Stage the tile: 32 keys × 32 u64 of K and of V, four of each
            // per thread; a key at or past `hi` stages zeros.
            let mut i = 0usize;
            while i < 4 {
                let e = tid + THREADS * i;
                let key = e / ROW_QWORDS;
                let word = e - key * ROW_QWORDS;
                let (kw, vw) = if t0 + key < hi {
                    // SAFETY: t0 + key < hi <= ctx, so the row is inside key
                    // head kh's plane; a row is 128 u16 = 32 u64 and the
                    // planes start 8-byte aligned (a device allocation), so
                    // the u64 read is aligned and inside both planes.
                    unsafe {
                        let r = (plane + (t0 + key) * HEAD) / 4 + word;
                        (*k64.add(r), *v64.add(r))
                    }
                } else {
                    (0u64, 0u64)
                };
                // SAFETY: key < 32 and word < 32: 2·word + 1 < K_STRIDE and
                // key·32 + word < KEY_TILE·ROW_QWORDS.
                unsafe {
                    *ks.add(key * K_STRIDE + 2 * word) = kw as u32;
                    *ks.add(key * K_STRIDE + 2 * word + 1) = (kw >> 32) as u32;
                    *vs.add(key * ROW_QWORDS + word) = vw;
                }
                i += 1;
            }
            thread::sync_threads();

            // The score of key t0 + lane for head h.
            let live = t0 + lane < hi;
            let mut a = [0.0f32; ILP];
            let mut wd = 0usize;
            while wd < ROW_WORDS {
                let mut p = 0usize;
                while p < ILP {
                    // SAFETY: lane < 32, wd + p < ROW_WORDS: inside KS; the
                    // query index w·HEAD + 2(wd + p) + 1 < GROUP·HEAD.
                    let (kw, q0, q1) = unsafe {
                        (
                            *ks.add(lane * K_STRIDE + wd + p),
                            *qs.add(w * HEAD + 2 * (wd + p)),
                            *qs.add(w * HEAD + 2 * (wd + p) + 1),
                        )
                    };
                    let k0 = half_bits_to_f32(kw as u16);
                    let k1 = half_bits_to_f32((kw >> 16) as u16);
                    a[p] = f32::mul_add(q0, k0, a[p]);
                    a[p] = f32::mul_add(q1, k1, a[p]);
                    p += 1;
                }
                wd += ILP;
            }
            let dot = (a[0] + a[1]) + (a[2] + a[3]);
            let sc = if live { dot * scale } else { f32::NEG_INFINITY };

            // SAFETY: WS and VS are this block's tiles, VS staged before the
            // barrier above; w < GROUP and lane < 32.
            unsafe { fold_tile(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane) };
            t0 += KEY_TILE;
        }

        // SAFETY: idx < m·n_kv·GROUP·segs; the four dims 4·lane .. +3 of the
        // partial row are this lane's alone, and lane 0 writes (m, s).
        unsafe {
            let o = idx * HEAD + 4 * lane;
            *part_v.get_unchecked_mut(o) = acc[0];
            *part_v.get_unchecked_mut(o + 1) = acc[1];
            *part_v.get_unchecked_mut(o + 2) = acc[2];
            *part_v.get_unchecked_mut(o + 3) = acc[3];
            if lane == 0 {
                *part_ms.get_unchecked_mut(2 * idx) = mx;
                *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
            }
        }
    }

    /// The segment pass with `S = Q·Kᵀ` on the tensor cores: the grid,
    /// the partials and every step after the scores are [`gqa_flash_seg`]'s
    /// (`fold_tile`), so [`gqa_flash_merge`] serves both. The group's eight
    /// query rows are staged as f16 (rounded to nearest even) into rows
    /// `0..8` of a 16-row tile whose rows `8..16` stay zero, and a tile's 32
    /// keys are staged as the cache's own f16 pairs; warps `0..4` each take
    /// eight keys and walk the 128 dims in four `ldmatrix.x4` steps of two
    /// `mma.m16n8k16` each (the fragment roles of `flash_latent_mma`), and
    /// write `scale · S` for rows `0..8` into the logit tile. Both staged
    /// strides are `mma_row_words(128)` words, so eight rows of one
    /// `ldmatrix` phase cover the 32 banks once. Keys at or past the
    /// segment's end are staged zero and their logits are `−inf`.
    ///
    /// The query rows and the products are f16, a different arithmetic
    /// class from the f32 scalar pass, with its own band.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let n_seg = segs as usize;
        let rows = m as usize;
        if b >= rows * nkv * n_seg {
            return; // block-uniform
        }
        let t = b % rows;
        let sk = b / rows;
        let seg = sk / nkv;
        let kh = sk - seg * nkv;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let n_head = nkv * GROUP;
        let h = kh * GROUP + w;
        let idx = (t * n_head + h) * n_seg + seg;
        let ctx = ctx as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
        let span = seg_span(limit, n_seg, seg_keys as usize);
        let lo = seg * span;
        if lo >= limit {
            if lane == 0 {
                // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform: lo and limit are the block's
        }
        let hi = (lo + span).min(limit);

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. Every
        // index below is inside its array, and every write is ordered before
        // its reads by a block barrier (or, for WS, by the warp sync inside
        // `fold_tile`).
        let (qt, kt, vs, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };

        // The query tile: warp `w` rounds row t's head `w`'s 64 value pairs,
        // two per lane, and writes the zero rows `8 + w`. Each pair is one
        // 8-byte load, and the loads come before the roundings.
        let qb = (t * n_head + kh * GROUP + w) * HEAD;
        let q64 = q.as_ptr() as *const u64;
        let mut raw = [0u64; 2];
        #[unroll]
        for i in 0usize..2 {
            // SAFETY: qb is a multiple of HEAD, so word (qb + 2·wd)/2 holds
            // values qb + 2·wd and + 1, with qb + 2·wd + 1 < (t·n_head +
            // kh·GROUP + w + 1)·HEAD <= m·n_kv·1024 <= q.len(); the buffer
            // starts 8-byte aligned (a device allocation).
            unsafe { raw[i] = *q64.add(qb / 2 + lane + 32 * i) };
        }
        #[unroll]
        for i in 0usize..2 {
            let wd = lane + 32 * i;
            let lo16 = f32_to_f16_bits(f32::from_bits(raw[i] as u32)) as u32;
            let hi16 = f32_to_f16_bits(f32::from_bits((raw[i] >> 32) as u32)) as u32;
            // SAFETY: the tile words w·MMA_ROW_W + wd and (8 + w)·MMA_ROW_W +
            // wd are inside QT (wd < 64 < MMA_ROW_W, 8 + w < MMA_ROWS).
            unsafe {
                *qt.add(w * MMA_ROW_W + wd) = lo16 | (hi16 << 16);
                *qt.add((GROUP + w) * MMA_ROW_W + wd) = 0;
            }
        }

        let plane = kh * ctx * HEAD;
        let k64 = kc.as_ptr() as *const u64;
        let v64 = vc.as_ptr() as *const u64;
        let key0 = (w % 4) * MMA_NTILE;
        let arow = lane % MMA_ROWS;
        let ahalf = lane / MMA_ROWS;
        let bkey = lane % MMA_NTILE;
        let boct = lane / MMA_NTILE;
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = [0.0f32; 4];
        let mut t0 = lo;
        while t0 < hi {
            thread::sync_threads();
            let mut i = 0usize;
            while i < 4 {
                let e = tid + THREADS * i;
                let key = e / ROW_QWORDS;
                let word = e - key * ROW_QWORDS;
                let (kw, vw) = if t0 + key < hi {
                    // SAFETY: as in gqa_flash_seg — the row is inside key
                    // head kh's plane and the u64 read is aligned.
                    unsafe {
                        let r = (plane + (t0 + key) * HEAD) / 4 + word;
                        (*k64.add(r), *v64.add(r))
                    }
                } else {
                    (0u64, 0u64)
                };
                // SAFETY: key < 32 and 2·word + 1 < 64 < MMA_ROW_W: inside
                // KT; key·32 + word inside VS.
                unsafe {
                    *kt.add(key * MMA_ROW_W + 2 * word) = kw as u32;
                    *kt.add(key * MMA_ROW_W + 2 * word + 1) = (kw >> 32) as u32;
                    *vs.add(key * ROW_QWORDS + word) = vw;
                }
                i += 1;
            }
            thread::sync_threads();

            if w < 4 {
                let mut c = [0.0f32; 4];
                let mut d = 0usize;
                while d < HEAD {
                    // SAFETY: key row key0 + bkey < KEY_TILE and words
                    // d/2 + 4·boct .. + 4 <= HEAD/2 are inside KT; every
                    // lane of the warp issues the load (the branch is on the
                    // warp index), after the barrier that staged it.
                    let bf = unsafe {
                        let bp = kt.add((key0 + bkey) * MMA_ROW_W + d / 2 + boct * 4);
                        wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            bp.cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: query row arow < MMA_ROWS and words d/2 +
                    // 4·ahalf .. + 4 + MMA_K/2 <= HEAD/2 are inside QT,
                    // published by the first barrier; the whole warp issues
                    // both loads and both `mma.sync` with its own fragments.
                    unsafe {
                        let ap0 = qt.add(arow * MMA_ROW_W + d / 2 + ahalf * (MMA_K / 4));
                        let af0 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap0.cast_const().cast::<u8>(),
                        ));
                        c = wmma::mma_m16n8k16_f32_f16(c, af0, [bf[0], bf[1]]);
                        let ap1 = ap0.add(MMA_K / 2);
                        let af1 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap1.cast_const().cast::<u8>(),
                        ));
                        c = wmma::mma_m16n8k16_f32_f16(c, af1, [bf[2], bf[3]]);
                    }
                    d += 2 * MMA_K;
                }
                // c[0], c[1]: head lane/4 at keys key0 + 2·(lane%4) + {0, 1};
                // c[2], c[3] are the zero rows.
                let g = lane / 4;
                let kk = key0 + 2 * (lane % 4);
                let s0 = if t0 + kk < hi {
                    scale * c[0]
                } else {
                    f32::NEG_INFINITY
                };
                let s1 = if t0 + kk + 1 < hi {
                    scale * c[1]
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: g < GROUP and kk + 1 < KEY_TILE: inside WS; each
                // (head, key) slot has one writer.
                unsafe {
                    *ws.add(g * KEY_TILE + kk) = s0;
                    *ws.add(g * KEY_TILE + kk + 1) = s1;
                }
            }
            thread::sync_threads();
            // SAFETY: w < GROUP and lane < KEY_TILE: inside WS, written
            // before the barrier above.
            let sc = unsafe { *ws.add(w * KEY_TILE + lane) };
            let live = t0 + lane < hi;
            warp::sync_mask(u32::MAX);
            // SAFETY: WS and VS are this block's tiles, VS staged before the
            // barriers above; w < GROUP and lane < 32.
            unsafe { fold_tile(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane) };
            t0 += KEY_TILE;
        }

        // SAFETY: as in gqa_flash_seg.
        unsafe {
            let o = idx * HEAD + 4 * lane;
            *part_v.get_unchecked_mut(o) = acc[0];
            *part_v.get_unchecked_mut(o + 1) = acc[1];
            *part_v.get_unchecked_mut(o + 2) = acc[2];
            *part_v.get_unchecked_mut(o + 3) = acc[3];
            if lane == 0 {
                *part_ms.get_unchecked_mut(2 * idx) = mx;
                *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
            }
        }
    }

    /// The merge: block `b = t·n_head + h` folds row `t`'s query head `h`'s
    /// partials of the segments that hold its keys — the first
    /// `ceil(n_keys / span)` of `segs`, `span` = [`seg_span`] of the row's
    /// own `n_keys`, with `n_keys` [`live_keys`] of
    /// `n_keys_buf[t]` as the segment pass reads it — in ascending order
    /// through `online_fold`, skipping one whose `Σ exp` is zero, and writes
    /// `y[(t·n_head + h)·HEAD + d] = r · (1/s)`, thread `d` owning dim `d`.
    /// A refused row raises [`FaultSite::KeyCount`] on `fault` and gets NaN
    /// in every dim.
    /// The partials are loaded [`MERGE_BATCH`] segments at a time ahead of
    /// their folds; the folds and their order are the one-at-a-time walk's.
    /// A segment past the row's last live one holds the neutral partial and
    /// is never read.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn gqa_flash_merge(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let nh = n_head as usize;
        if b >= m as usize * nh {
            return; // block-uniform
        }
        let t = b / nh;
        let d = thread::threadIdx_x() as usize;
        let n_seg = segs as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx as usize);
        if limit == 0 {
            if d == 0 {
                fault.raise(FaultSite::KeyCount);
            }
            // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
            unsafe { *y.get_unchecked_mut(b * HEAD + d) = f32::NAN };
            return; // block-uniform: the row is the block's
        }
        // seg_span keeps ⌈limit / span⌉ <= segs (its const check), so the
        // walk below stays inside the row's segments.
        let live = limit.div_ceil(seg_span(limit, n_seg, seg_keys as usize));
        let mut mx = f32::NEG_INFINITY;
        let mut s = 0.0f32;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < live {
            // The batch's loads are issued ahead of its folds, so the walk
            // does not wait on memory twice per segment.
            let mut ms = [0.0f32; 2 * MERGE_BATCH];
            let mut vs = [0.0f32; MERGE_BATCH];
            #[unroll]
            for i in 0..MERGE_BATCH {
                if j + i < live {
                    let idx = b * n_seg + j + i;
                    // SAFETY: idx < m·n_head·segs: both slots inside part_ms,
                    // and idx·HEAD + d < m·n_head·segs·HEAD <= part_v.len().
                    // A live segment wrote all three.
                    unsafe {
                        ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                        ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                        vs[i] = *part_v.get_unchecked(idx * HEAD + d);
                    }
                }
            }
            #[unroll]
            for i in 0..MERGE_BATCH {
                if j + i < live && ms[2 * i + 1] != 0.0 {
                    (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
                }
            }
            j += MERGE_BATCH;
        }
        // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = acc * (1.0 / s) };
    }

    /// [`gqa_flash_seg`] at [`HEAD_256`]: lane `l` owns dims `4l .. +3` and
    /// `128 + 4l .. +3` (`seg_scalar`).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * 8 * segs * 256,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_256(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP * HEAD_256 }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256; the raw form reaches it without a
        // reference. The launch contract is `seg_scalar`'s at HEAD_256.
        unsafe {
            seg_scalar::<HEAD_256, QW_256>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma`] at [`HEAD_256`] (`seg_mma`): each score is the
    /// two 128-value slices' accumulators added, low slice first.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * 8 * segs * 256,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_256(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256; the raw form reaches it without a
        // reference. The launch contract is `seg_mma`'s at HEAD_256.
        unsafe {
            seg_mma::<HEAD_256, QW_256, _>(
                q,
                F16Kv { k: kc, v: vc },
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_merge`] at [`HEAD_256`]: 256 threads, one per dim.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            part_v.len() >= m * n_head * segs * 256,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 256
        )
    )]
    pub fn gqa_flash_merge_256(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `merge_body`'s at HEAD_256, and the
        // block is HEAD_256 threads.
        unsafe {
            merge_body::<HEAD_256>(
                part_v, part_ms, n_keys_buf, n_head, ctx, segs, seg_keys, m, fault, y,
            )
        };
    }
    /// them per key head (`seg_scalar_p` at `PACK_4`): block `b = ((seg·n_kv +
    /// kh)·packs + p)·m + t`, warp `w` query head `(kh·packs + p)·4 + w`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * packs * 4 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * packs * 4 * segs * 256,
            part_ms.len() >= m * n_kv * packs * 4 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_256_p4(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        packs: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { PACK_4 * HEAD_256 }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { PACK_4 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256 and PACK_4; the raw form reaches it
        // without a reference. The launch contract is `seg_scalar_p`'s at
        // HEAD_256 and `packs · PACK_4` heads per key head.
        unsafe {
            seg_scalar_p::<HEAD_256, QW_256, PACK_4>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                packs as usize,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_256`] in blocks of [`PACK_2`] query heads, `packs` of
    /// them per key head (`seg_scalar_p` at `PACK_2`): block `b = ((seg·n_kv +
    /// kh)·packs + p)·m + t`, warp `w` query head `(kh·packs + p)·2 + w`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * packs * 2 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * packs * 2 * segs * 256,
            part_ms.len() >= m * n_kv * packs * 2 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_256_p2(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        packs: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { PACK_2 * HEAD_256 }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { PACK_2 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256 and PACK_2; the raw form reaches it
        // without a reference. The launch contract is `seg_scalar_p`'s at
        // HEAD_256 and `packs · PACK_2` heads per key head.
        unsafe {
            seg_scalar_p::<HEAD_256, QW_256, PACK_2>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                packs as usize,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma_256`] in blocks of [`PACK_4`] query heads
    /// (`seg_mma_p` at `PACK_4`): the pack's heads are rows `0..4` of the 16-row query tile
    /// and all four warps take a tile's keys; the block map is
    /// [`gqa_flash_seg_256_p4`]'s.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * packs * 4 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * packs * 4 * segs * 256,
            part_ms.len() >= m * n_kv * packs * 4 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_256_p4(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        packs: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { PACK_4 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256 and PACK_4; the raw form reaches it
        // without a reference. The launch contract is `seg_mma_p`'s at
        // HEAD_256 and `packs · PACK_4` heads per key head.
        unsafe {
            seg_mma_p::<HEAD_256, QW_256, PACK_4, { MMA_ROWS / PACK_4 }>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                packs as usize,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma_256_p4`] over each row's listed keys (`seg_mma_ps` at
    /// `PACK_12` with the listed walk), one block of twelve warps a (row, key
    /// head, segment): the key head's twelve query heads in rows `0..12` of
    /// the tile, rows `12..16` zero, so a tile of keys is staged once for the
    /// group. Row `t`'s key `j` is cache row `list[t·width + j]`, its live
    /// count `n_sel[t]` (refused past `width`), the segments cut over list
    /// positions. A list entry at or past `ctx` raises
    /// [`FaultSite::PoolSelect`] on `fault` and stages NaN. With the list
    /// `0 .. n` a row stages [`gqa_flash_seg_mma_256_p4`]'s keys at count `n`,
    /// and every head's output is that entry's bit for bit; the gate holds the
    /// two bodies to the same bits there.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(384)]
    #[launch_contract(
        domain = 1,
        block = (384, 1, 1),
        requires = (
            n_sel.len() >= m,
            list.len() >= m * width,
            segs >= 1,
            q.len() >= m * n_kv * 12 * 256,
            kc.len() >= n_kv * ctx * 256,
            vc.len() >= n_kv * ctx * 256,
            part_v.len() >= m * n_kv * 12 * segs * 256,
            part_ms.len() >= m * n_kv * 12 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_256_p12_sel(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        list: &[u32],
        n_sel: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        width: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        fault: FaultSink,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { PACK_12 * KEY_TILE }> = SharedArray::UNINIT;

        let walk = Listed {
            list,
            width: width as usize,
            fault,
        };
        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256 and PACK_12; the raw form reaches it
        // without a reference. The launch contract is `seg_mma_ps`'s at
        // HEAD_256 and one pack of PACK_12 heads per key head, with the
        // list's.
        unsafe {
            seg_mma_ps::<HEAD_256, QW_256, PACK_12, PAD_12>(
                walk,
                q,
                kc,
                vc,
                n_sel,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                1,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma`] over the Q8_0 cache (`seg_mma` over [`Q8Kv`]):
    /// the key and value tiles hold the dequantized f16, so its scores carry
    /// the f16 rounding of `code·d` the f16 cache never had; everything after
    /// the staging is the f16 instance's.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            kq.len() >= n_kv * ctx * 32,
            kd.len() >= n_kv * ctx * 4,
            vq.len() >= n_kv * ctx * 32,
            vd.len() >= n_kv * ctx * 4,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_q8(
        q: &[f32],
        kq: &[u32],
        kd: &[u16],
        vq: &[u32],
        vd: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. The launch
        // contract is `seg_mma`'s at HEAD over the four Q8_0 planes.
        unsafe {
            seg_mma::<HEAD, 1, _>(
                q,
                Q8Kv { kq, kd, vq, vd },
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma`] over several sequences' f16 caches: row `t`'s
    /// keys and values are the planes of row `t` of `planes` ([`RowPlanes`],
    /// each `n_kv · ctx` rows of 128). `seg_mma` at [`HEAD`] over [`F16Kv`],
    /// the body of the one-sequence tensor-core passes: the grid, the
    /// partials' index and every row's arithmetic are a launch of one
    /// sequence's over that row's planes, and [`gqa_flash_merge`] folds them.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m <= 8,
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_rows(
        q: &[f32],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
        #[grid_constant] planes: &RowPlanes,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let rows = m as usize;
        if b >= rows * n_kv as usize * segs as usize {
            return; // block-uniform
        }
        // SAFETY: b % m < m <= ROW_PLANES (the launch contract), and the
        // launcher built the table over f16 planes of n_kv·ctx·HEAD values
        // that this launch only reads.
        let src = unsafe { planes.f16(b % rows, n_kv as usize * ctx as usize * HEAD) };
        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. The
        // launch contract is `seg_mma`'s at HEAD over the row's planes.
        unsafe {
            seg_mma::<HEAD, 1, _>(
                q,
                src,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`gqa_flash_seg_mma_256`] over the Q8_0 cache (`seg_mma` at
    /// [`HEAD_256`] over [`Q8Kv`]): [`gqa_flash_seg_mma_q8`] at the wide head.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 256,
            kq.len() >= n_kv * ctx * 64,
            kd.len() >= n_kv * ctx * 8,
            vq.len() >= n_kv * ctx * 64,
            vd.len() >= n_kv * ctx * 8,
            part_v.len() >= m * n_kv * 8 * segs * 256,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma_256_q8(
        q: &[f32],
        kq: &[u32],
        kd: &[u16],
        vq: &[u32],
        vd: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W_256 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS_256 }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_256; the raw form reaches it without a
        // reference. The launch contract is `seg_mma`'s at HEAD_256 over the
        // four Q8_0 planes.
        unsafe {
            seg_mma::<HEAD_256, QW_256, _>(
                q,
                Q8Kv { kq, kd, vq, vd },
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// The segment pass of MiMo-V2's heads: scores over [`HEAD_K192`] key
    /// values, values of [`HEAD`], blocks of [`GROUP`] query heads, `packs` of
    /// them per key head (`seg_scalar_kv`): block `b = ((seg·n_kv + kh)·packs
    /// + p)·m + t`, warp `w` query head `(kh·packs + p)·8 + w`. With `window`
    /// 0 a row walks every live key, as [`gqa_flash_seg`]; else the last
    /// `window` of them ([`window_cut`]), cut into `segs` segments from the
    /// row's first windowed key.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * packs * 8 * 192,
            kc.len() >= n_kv * ctx * 192,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * packs * 8 * segs * 128,
            part_ms.len() >= m * n_kv * packs * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_k192(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        packs: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP * HEAD_K192 }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE_K192 }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for HEAD_K192 and GROUP; the raw form reaches it
        // without a reference. The launch contract is `seg_scalar_kv`'s.
        unsafe {
            seg_scalar_kv::<GROUP>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                m,
                packs as usize,
                window as usize,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// The merge of MiMo-V2's heads (`merge_sink_body`): [`gqa_flash_merge`]
    /// over the window's segments with head `h`'s sink folded in last,
    /// `sinks[h]`, unscaled, `−inf` neutral.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs >= 1,
            sinks.len() >= n_head,
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn gqa_flash_merge_sink(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        sinks: &[f32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `merge_sink_body`'s, and the block is
        // HEAD threads.
        unsafe {
            merge_sink_body(
                part_v,
                part_ms,
                n_keys_buf,
                sinks,
                n_head,
                ctx,
                segs,
                seg_keys,
                m,
                window as usize,
                fault,
                y,
            )
        };
    }
}

/// The segment entry of a packed pass: the pack's scalar or tensor-core body.
#[derive(Clone, Copy)]
enum PackedSeg {
    Scalar4,
    Mma4,
    Scalar2,
}

impl PackedSeg {
    /// Query heads one block takes.
    fn pack(self) -> usize {
        match self {
            PackedSeg::Scalar4 | PackedSeg::Mma4 => PACK_4,
            PackedSeg::Scalar2 => PACK_2,
        }
    }
}

/// [`FlashGqaKernels::enqueue_pass`]'s arguments: `m` rows of `n_kv · GROUP`
/// query heads (roped, unscaled, token-major), the layer's two planes of
/// `n_kv · ctx` rows, each row's live key count on the device (`m` of them),
/// the partials scratch ([`partials_v_len`], [`partials_ms_len`] of `m`
/// rows), the sink a refused count raises on, and the output rows,
/// token-major.
pub struct GqaArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashGqaKernels::enqueue_pass_256_p12_sel`]'s arguments: those of
/// [`GqaArgs`] with each row's list (`m · width` cache rows, ascending) and its length
/// (`n_sel`, `m` of them) in place of the live counts.
pub struct GqaSelArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub list: &'a DeviceBuffer<u32>,
    pub n_sel: &'a DeviceBuffer<u32>,
    pub width: usize,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`GqaArgs`]' q8_0 form: the same query rows, counts, partials and output,
/// the cache the four planes of the quantizing appends ([`q8_plane_lens`]) —
/// per side the codes and the scales — served by the `_q8` entries.
pub struct GqaQ8Args<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kq: &'a DeviceBuffer<u32>,
    pub kd: &'a DeviceBuffer<u16>,
    pub vq: &'a DeviceBuffer<u32>,
    pub vd: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashGqaKernels::enqueue_pass_rows`]'s arguments: those of [`GqaArgs`]
/// without the planes, which the per-row table gives each row.
pub(crate) struct GqaRowsArgs<'a> {
    pub(crate) q: &'a DeviceBuffer<f32>,
    pub(crate) n_keys: &'a DeviceBuffer<u32>,
    pub(crate) scale: f32,
    pub(crate) n_kv: usize,
    pub(crate) ctx: usize,
    pub(crate) m: usize,
    pub(crate) part_v: &'a mut DeviceBuffer<f32>,
    pub(crate) part_ms: &'a mut DeviceBuffer<f32>,
    pub(crate) fault: FaultSink,
    pub(crate) y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashGqaKernels::enqueue_pass_k192`]'s arguments: `m` rows of `n_head`
/// query heads of [`HEAD_K192`] values (roped, unscaled, token-major), the
/// layer's key plane of `n_kv · ctx` rows of [`HEAD_K192`] and value plane of
/// `n_kv · ctx` rows of [`HEAD`] (the f16 planes
/// `RopeNeoxKernels::enqueue_neox_append_k192` writes), each row's live key
/// count on the device (`m` of them), the layer's window in positions (0:
/// every live key) and its per-head sinks (`n_head` f32, or none), the
/// partials scratch (`m · n_head ·` [`window_segments`]`(window)` rows, the
/// [`partials_v_len`] and [`partials_ms_len`] of a launch at that many
/// segments), the sink a refused count raises on, and the output rows of
/// [`HEAD`] values, token-major.
pub struct GqaK192Args<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub window: usize,
    pub sinks: Option<&'a DeviceBuffer<f32>>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashGqaKernels::enqueue_merge_k192`]'s arguments: the partials of a
/// segment pass at `window` ([`window_segments`] segments a row's head), each
/// row's live count, the layer's sinks (or none), the cache height the counts
/// are held to, `m` rows, the sink a refused count raises on, and the output
/// rows of [`HEAD`] values.
pub struct GqaK192MergeArgs<'a> {
    pub part_v: &'a DeviceBuffer<f32>,
    pub part_ms: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub sinks: Option<&'a DeviceBuffer<f32>>,
    pub ctx: usize,
    pub m: usize,
    pub window: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loader's check of a layer's sinks as the host holds them, before they
/// go to the card: every value must be finite, else refused by name with the
/// head. A device buffer is not read at launch, so this is where a non-finite
/// sink of a file is refused; `−inf` is the kernel's neutral value, not a
/// file's.
///
/// # Errors
///
/// [`GpuError`] naming the first head whose sink is NaN or infinite.
pub fn check_sinks(sinks: &[f32]) -> Result<(), GpuError> {
    match sinks.iter().position(|v| !v.is_finite()) {
        Some(h) => Err(GpuError::shape(
            "flash_gqa::check_sinks",
            format!("the sink of head {h} is {}, not finite", sinks[h]),
        )),
        None => Ok(()),
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct FlashGqaKernels {
    module: flash_gqa_kernels::LoadedModule,
}

impl FlashGqaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashGqaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(flash_gqa_kernels, ctx)? };
        Ok(FlashGqaKernels { module })
    }

    /// Enqueue `m` rows' attention: the segment pass (`m · n_kv ·
    /// [`SEGMENTS`]` blocks; the tensor-core one when `mma`) and the
    /// merge (`m · n_kv · GROUP` blocks). Each block cuts its row's range
    /// from the row's own count, so every count the kernels accept is
    /// walked in full; a count
    /// of zero or past `ctx` raises [`FaultSite::KeyCount`] on `args.fault`
    /// and its row is NaN. Two launches. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_pass(
        &self,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue";
        let GqaArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let n_head = n_kv * GROUP;
        let segs = SEGMENTS;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kc", kc.len(), n_kv * ctx * HEAD),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        if mma {
            let prep = self.module.prepare_gqa_flash_seg_mma(cfg)?;
            self.module.gqa_flash_seg_mma(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        } else {
            let prep = self.module.prepare_gqa_flash_seg(cfg)?;
            self.module.gqa_flash_seg(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        }
        let prep = self.module.prepare_gqa_flash_merge(LaunchConfig1D::new(
            merge_grid,
            MERGE_THREADS,
            0,
        ))?;
        self.module.gqa_flash_merge(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass`] over heads of [`HEAD_256`] values:
    /// `args.q` and `args.y` hold `m · n_kv · GROUP` rows of 256, the planes
    /// `n_kv · ctx` rows of 256 f16, `args.part_v` [`partials_v_len_256`].
    /// The segment pass (`gqa_flash_seg_mma_256` when `mma`, else
    /// `gqa_flash_seg_256`) and `gqa_flash_merge_256`: two launches, the same
    /// grid and refusals. Asynchronous, allocation-free, capturable.
    pub fn enqueue_pass_256(
        &self,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_256";
        let GqaArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let n_head = n_kv * GROUP;
        let segs = SEGMENTS;
        let lens = [
            ("q", q.len(), m * n_head * HEAD_256),
            ("kc", kc.len(), n_kv * ctx * HEAD_256),
            ("vc", vc.len(), n_kv * ctx * HEAD_256),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len_256(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        if mma {
            let prep = self.module.prepare_gqa_flash_seg_mma_256(cfg)?;
            self.module.gqa_flash_seg_mma_256(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        } else {
            let prep = self.module.prepare_gqa_flash_seg_256(cfg)?;
            self.module.gqa_flash_seg_256(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        }
        let prep = self
            .module
            .prepare_gqa_flash_merge_256(LaunchConfig1D::new(merge_grid, MERGE_THREADS_256, 0))?;
        self.module.gqa_flash_merge_256(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass_256`] for `n_head` query heads over
    /// `args.n_kv` key heads in blocks of [`PACK_4`]: `n_head` a nonzero
    /// multiple of `4 · n_kv` (refused by name otherwise), `args.q` and
    /// `args.y` `m · n_head` rows of 256, `args.part_v` [`partials_v_len_256`]
    /// and `args.part_ms` [`partials_ms_len`] of `n_head`. The segment pass
    /// (`gqa_flash_seg_mma_256_p4` when `mma`, else `gqa_flash_seg_256_p4`;
    /// `m · n_head / 4 · `[`SEGMENTS`]` blocks of 128 threads) and
    /// `gqa_flash_merge_256` (`m · n_head` blocks): two launches, the refusals
    /// of `enqueue_pass_256`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_pass_256_p4(
        &self,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        n_head: usize,
        mma: bool,
    ) -> Result<(), GpuError> {
        let seg = if mma {
            PackedSeg::Mma4
        } else {
            PackedSeg::Scalar4
        };
        self.pass_packed("flash_gqa::enqueue_256_p4", stream, args, n_head, seg)
    }

    /// [`FlashGqaKernels::enqueue_pass_256_p4`]'s scalar pass in blocks of
    /// [`PACK_2`]: `n_head` a nonzero multiple of `2 · n_kv` (refused by name
    /// otherwise), `gqa_flash_seg_256_p2` (`m · n_head / 2 · `[`SEGMENTS`]`
    /// blocks of 64 threads) and `gqa_flash_merge_256`: two launches, the
    /// refusals of `enqueue_pass_256`. Every row's bits are the `_256` scalar
    /// pass's. Asynchronous, allocation-free, capturable.
    pub fn enqueue_pass_256_p2(
        &self,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        self.pass_packed(
            "flash_gqa::enqueue_256_p2",
            stream,
            args,
            n_head,
            PackedSeg::Scalar2,
        )
    }

    /// The packed passes' one body: the group refused unless a nonzero
    /// multiple of `seg`'s pack, the buffers checked, then `seg`'s segment
    /// entry and the merge.
    fn pass_packed(
        &self,
        what: &'static str,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        n_head: usize,
        seg: PackedSeg,
    ) -> Result<(), GpuError> {
        let GqaArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let pack = seg.pack();
        if n_head == 0 || !n_head.is_multiple_of(n_kv * pack) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel packs {pack} query heads a block, so the group must be a \
                     multiple of {pack}; got {n_head} heads over {n_kv}"
                ),
            ));
        }
        let packs = n_head / (n_kv * pack);
        let segs = SEGMENTS;
        let lens = [
            ("q", q.len(), m * n_head * HEAD_256),
            ("kc", kc.len(), n_kv * ctx * HEAD_256),
            ("vc", vc.len(), n_kv * ctx * HEAD_256),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len_256(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * packs * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let packs = launch_u32(what, "packs", packs)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        match seg {
            PackedSeg::Mma4 => {
                let cfg = LaunchConfig1D::new(grid, THREADS_P4_U32, 0);
                let prep = self.module.prepare_gqa_flash_seg_mma_256_p4(cfg)?;
                self.module.gqa_flash_seg_mma_256_p4(
                    stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, packs,
                    part_v, part_ms,
                )?;
            }
            PackedSeg::Scalar4 => {
                let cfg = LaunchConfig1D::new(grid, THREADS_P4_U32, 0);
                let prep = self.module.prepare_gqa_flash_seg_256_p4(cfg)?;
                self.module.gqa_flash_seg_256_p4(
                    stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, packs,
                    part_v, part_ms,
                )?;
            }
            PackedSeg::Scalar2 => {
                let cfg = LaunchConfig1D::new(grid, THREADS_P2_U32, 0);
                let prep = self.module.prepare_gqa_flash_seg_256_p2(cfg)?;
                self.module.gqa_flash_seg_256_p2(
                    stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, packs,
                    part_v, part_ms,
                )?;
            }
        }
        let prep = self
            .module
            .prepare_gqa_flash_merge_256(LaunchConfig1D::new(merge_grid, MERGE_THREADS_256, 0))?;
        self.module.gqa_flash_merge_256(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass_256_p4`] over each row's listed keys:
    /// row `t` attends the cache rows `list[t·width .. t·width + n_sel[t]]`
    /// (ascending, as `qsa`'s top-k writes them), `n_head` query heads over
    /// `args.n_kv` in blocks of [`PACK_12`] — one block a key head, so
    /// `n_head` must be `args.n_kv · 12` (refused by name otherwise) — the
    /// segments cut over list positions: [`listed_segments`]`(width)` of them,
    /// so `args.part_v` is [`listed_partials_v_len_256`] and `args.part_ms`
    /// [`listed_partials_ms_len`] at `(m, n_head, width)`. The segment pass
    /// (`gqa_flash_seg_mma_256_p12_sel`) and `gqa_flash_merge_256` over the
    /// lists: two launches. A length of zero or past `width` raises
    /// [`FaultSite::KeyCount`] and its row is NaN; a list entry at or past
    /// the cache raises [`FaultSite::PoolSelect`] and its row is NaN. With
    /// every list `0 .. n` each row stages `enqueue_pass_256_p4`'s keys at
    /// count `n`, and the gate holds the two passes to the same bits there.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_pass_256_p12_sel(
        &self,
        stream: &CudaStream,
        args: GqaSelArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_256_p12_sel";
        let GqaSelArgs {
            q,
            kc,
            vc,
            list,
            n_sel,
            width,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 || width == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_kv, ctx, m and width >= 1, got n_kv={n_kv} ctx={ctx} m={m} \
                     width={width}"
                ),
            ));
        }
        if n_head != n_kv * PACK_12 {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel takes a key head's {PACK_12} query heads in one block, so \
                     the group must be {PACK_12}; got {n_head} heads over {n_kv}"
                ),
            ));
        }
        let segs = listed_segments(width);
        let lens = [
            ("q", q.len(), m * n_head * HEAD_256),
            ("kc", kc.len(), n_kv * ctx * HEAD_256),
            ("vc", vc.len(), n_kv * ctx * HEAD_256),
            ("list", list.len(), m * width),
            ("n_sel", n_sel.len(), m),
            (
                "part_v",
                part_v.len(),
                listed_partials_v_len_256(m, n_head, width),
            ),
            (
                "part_ms",
                part_ms.len(),
                listed_partials_ms_len(m, n_head, width),
            ),
            ("y", y.len(), m * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let width = launch_u32(what, "width", width)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_P12_U32, 0);
        let prep = self.module.prepare_gqa_flash_seg_mma_256_p12_sel(cfg)?;
        self.module.gqa_flash_seg_mma_256_p12_sel(
            stream, &prep, q, kc, vc, list, n_sel, scale, n_kv, ctx, width, segs, seg_keys, m,
            fault, part_v, part_ms,
        )?;
        // The merge reads a row's count through the same rule at the list's
        // width: its `ctx` argument is the width.
        let prep = self
            .module
            .prepare_gqa_flash_merge_256(LaunchConfig1D::new(merge_grid, MERGE_THREADS_256, 0))?;
        self.module.gqa_flash_merge_256(
            stream, &prep, part_v, part_ms, n_sel, heads, width, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass`]'s tensor-core pass over the Q8_0
    /// cache (the module doc's `_q8` paragraph): `args`' four planes in place
    /// of the f16 pair, the segment pass `gqa_flash_seg_mma_q8` — its key
    /// tile the dequantized f16 — and `gqa_flash_merge`. Two launches, the
    /// refusals of `enqueue_pass`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_pass_q8(
        &self,
        stream: &CudaStream,
        args: GqaQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_q8";
        let GqaQ8Args {
            q,
            kq,
            kd,
            vq,
            vd,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let n_head = n_kv * GROUP;
        let segs = SEGMENTS;
        let (words, scales) = q8_plane_lens(HEAD, n_kv, ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        let prep = self.module.prepare_gqa_flash_seg_mma_q8(cfg)?;
        self.module.gqa_flash_seg_mma_q8(
            stream, &prep, q, kq, kd, vq, vd, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
            part_ms,
        )?;
        let prep = self.module.prepare_gqa_flash_merge(LaunchConfig1D::new(
            merge_grid,
            MERGE_THREADS,
            0,
        ))?;
        self.module.gqa_flash_merge(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass`]'s tensor-core pass over several
    /// sequences' f16 caches: row `t` attends the planes `planes` gives it
    /// ([`RowTable`]), the rest of its inputs `args`' rows as
    /// `enqueue_pass`'s. The segment pass `gqa_flash_seg_mma_rows` and
    /// `gqa_flash_merge` at `m` rows: `enqueue_pass`'s grid, partials and
    /// refusals, and a table of other than `m` rows, more than
    /// [`ROW_PLANES`], or a plane shorter than `n_kv · ctx · HEAD` refused by
    /// name. Each row's segments and its merge are cut from its own count
    /// and the fixed [`SEGMENTS`], so its bits are a launch of one sequence's
    /// over its own planes. Two launches. Asynchronous, allocation-free,
    /// capturable.
    pub(crate) fn enqueue_pass_rows(
        &self,
        stream: &CudaStream,
        planes: &RowTable<'_>,
        args: GqaRowsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_rows";
        let GqaRowsArgs {
            q,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 || m > ROW_PLANES {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_kv, ctx >= 1 and 1 <= m <= {ROW_PLANES}, got n_kv={n_kv} ctx={ctx} \
                     m={m}"
                ),
            ));
        }
        let rows = planes.planes(what, m, n_kv * ctx * HEAD)?;
        let n_head = n_kv * GROUP;
        let segs = SEGMENTS;
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        let prep = self.module.prepare_gqa_flash_seg_mma_rows(cfg)?;
        // SAFETY: the table's rows are `planes`' puts, each an f16 plane pair
        // of at least n_kv·ctx·HEAD values (checked above), borrowed by the
        // table while this enqueue runs; a graph that captures the launch
        // replays it over the same planes, which the caller keeps alive and
        // in place while the graph lives, as for any buffer it captured.
        unsafe {
            self.module.gqa_flash_seg_mma_rows(
                stream, &prep, q, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v, part_ms,
                rows,
            )?;
        }
        let prep = self.module.prepare_gqa_flash_merge(LaunchConfig1D::new(
            merge_grid,
            MERGE_THREADS,
            0,
        ))?;
        self.module.gqa_flash_merge(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// [`FlashGqaKernels::enqueue_pass_256`]'s tensor-core pass over the
    /// Q8_0 cache: the segment pass `gqa_flash_seg_mma_256_q8` and
    /// `gqa_flash_merge_256`, the planes [`q8_plane_lens`] at [`HEAD_256`].
    /// Two launches, the refusals of `enqueue_pass_256`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_pass_256_q8(
        &self,
        stream: &CudaStream,
        args: GqaQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_256_q8";
        let GqaQ8Args {
            q,
            kq,
            kd,
            vq,
            vd,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let n_head = n_kv * GROUP;
        let segs = SEGMENTS;
        let (words, scales) = q8_plane_lens(HEAD_256, n_kv, ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD_256),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len_256(m, n_head)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head)),
            ("y", y.len(), m * n_head * HEAD_256),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        let prep = self.module.prepare_gqa_flash_seg_mma_256_q8(cfg)?;
        self.module.gqa_flash_seg_mma_256_q8(
            stream, &prep, q, kq, kd, vq, vd, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
            part_ms,
        )?;
        let prep = self
            .module
            .prepare_gqa_flash_merge_256(LaunchConfig1D::new(merge_grid, MERGE_THREADS_256, 0))?;
        self.module.gqa_flash_merge_256(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }

    /// Enqueue `m` rows' attention over heads of [`HEAD_K192`] key values and
    /// [`HEAD`] value values, `n_head` query heads over `args.n_kv` key heads
    /// in blocks of [`GROUP`] (`n_head` a nonzero multiple of `8 · n_kv`,
    /// refused by name otherwise; `packs = n_head / (8 · n_kv)` blocks a key
    /// head): the segment pass `gqa_flash_seg_k192` (`m · n_kv · packs ·
    /// segs` blocks of 256 threads) and a merge. Without a window the row's
    /// keys are cut into [`SEGMENTS`] segments from key 0; with `window`
    /// positions into [`window_segments`] segments from the row's first
    /// windowed key ([`window_cut`]). Without sinks the merge is
    /// [`flash_gqa_kernels::gqa_flash_merge`] and a window is refused by name
    /// (that merge cuts from key 0); with sinks it is `gqa_flash_merge_sink`
    /// at any window, head `h` folding `sinks[h]` last. A non-finite scale, a
    /// short buffer or a zero count is refused by name before the launch; a
    /// sink the device holds as NaN or `+inf` makes its row NaN (a device
    /// buffer is not read here — the loader refuses a non-finite sink). A
    /// count of zero or past `ctx` raises [`FaultSite::KeyCount`] on
    /// `args.fault` and its row is NaN. Two launches. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_pass_k192(
        &self,
        stream: &CudaStream,
        args: GqaK192Args<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_k192";
        let GqaK192Args {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            window,
            sinks,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        if n_head == 0 || !n_head.is_multiple_of(n_kv * GROUP) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernel packs {GROUP} query heads a block, so the group must be a \
                     multiple of {GROUP}; got {n_head} heads over {n_kv}"
                ),
            ));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("the score scale must be finite, got {scale}"),
            ));
        }
        if window > 0 && sinks.is_none() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a window of {window} positions needs the sink merge (the merge without \
                     sinks cuts every row from key 0); pass the layer's sinks"
                ),
            ));
        }
        let packs = n_head / (n_kv * GROUP);
        let segs = window_segments(window);
        let lens = [
            ("q", q.len(), m * n_head * HEAD_K192),
            ("kc", kc.len(), n_kv * ctx * HEAD_K192),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_len(m, n_head, segs, HEAD)),
            ("part_ms", part_ms.len(), partials_len(m, n_head, segs, 2)),
            ("y", y.len(), m * n_head * HEAD),
            ("sinks", sinks.map_or(n_head, |s| s.len()), n_head),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * packs * segs)?;
        let packs = launch_u32(what, "packs", packs)?;
        let n_kv_u = launch_u32(what, "n_kv", n_kv)?;
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m_u = launch_u32(what, "m", m)?;
        let window_u = launch_u32(what, "window", window)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        let prep = self.module.prepare_gqa_flash_seg_k192(cfg)?;
        self.module.gqa_flash_seg_k192(
            stream,
            &prep,
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv_u,
            ctx_u,
            segs,
            seg_keys,
            m_u,
            packs,
            window_u,
            &mut *part_v,
            &mut *part_ms,
        )?;
        self.enqueue_merge_k192(
            stream,
            GqaK192MergeArgs {
                part_v,
                part_ms,
                n_keys,
                sinks,
                ctx,
                m,
                window,
                fault,
                y,
            },
            n_head,
        )
    }

    /// The merge of [`FlashGqaKernels::enqueue_pass_k192`] alone, over the
    /// partials of a segment pass launched at the same `window`: `m · n_head`
    /// blocks of [`HEAD`] threads, `gqa_flash_merge` without sinks (and
    /// without a window) and `gqa_flash_merge_sink` with them. The refusals
    /// of the pass: a window without sinks, a short buffer, a zero count.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_merge_k192(
        &self,
        stream: &CudaStream,
        args: GqaK192MergeArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue_merge_k192";
        let GqaK192MergeArgs {
            part_v,
            part_ms,
            n_keys,
            sinks,
            ctx,
            m,
            window,
            fault,
            y,
        } = args;
        if n_head == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_head, ctx and m >= 1, got n_head={n_head} ctx={ctx} m={m}"),
            ));
        }
        if window > 0 && sinks.is_none() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a window of {window} positions needs the sink merge (the merge without \
                     sinks cuts every row from key 0); pass the layer's sinks"
                ),
            ));
        }
        let segs = window_segments(window);
        let lens = [
            ("part_v", part_v.len(), partials_len(m, n_head, segs, HEAD)),
            ("part_ms", part_ms.len(), partials_len(m, n_head, segs, 2)),
            ("n_keys", n_keys.len(), m),
            ("y", y.len(), m * n_head * HEAD),
            ("sinks", sinks.map_or(n_head, |s| s.len()), n_head),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let window = launch_u32(what, "window", window)?;
        let cfg = LaunchConfig1D::new(merge_grid, MERGE_THREADS_SINK, 0);
        match sinks {
            Some(sinks) => {
                let prep = self.module.prepare_gqa_flash_merge_sink(cfg)?;
                self.module.gqa_flash_merge_sink(
                    stream, &prep, part_v, part_ms, n_keys, sinks, heads, ctx, segs, seg_keys, m,
                    window, fault, y,
                )?;
            }
            None => {
                let prep = self.module.prepare_gqa_flash_merge(cfg)?;
                self.module.gqa_flash_merge(
                    stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
                )?;
            }
        }
        Ok(())
    }
}
