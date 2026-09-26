//! P5: the KV append (f32 -> f16) and the latent (MLA) flash attention over
//! the absorbed cache (docs/gpu-design.md 작업 꾸러미 P5). One cache row per
//! token per layer, `[rope_dims | latent]` u16 with the rope part first —
//! the row order the CPU engine's `kvr` columns carry (`crates/model/src/
//! kv.rs`, oracle-verified there) — and one f32 query row per (token, head),
//! `[q_rope | q_nope2]` in the same order. The attention output is the
//! softmax(scale · q·k)-weighted sum of the rows' latent tails, projected
//! later by `attn_kv_b`'s V part (not this package).
//!
//! The f16 rounding reuses `gguf::quant::f32_to_f16_bits` itself — the one
//! owner of the CPU oracle's conversion — so cache bits are equal by
//! construction, not by transcription; the gate asserts it element for
//! element on real `kvr-L` rows and on the IEEE edges.
//!
//! Launch geometry of the single-block kernel: one CUDA block of [`LATENT`]
//! threads per query row, so a thread owns exactly one latent dim of that
//! row's output for the whole run and needs no accumulator array. Everything a
//! thread must share with the rest of the block (the staged query row, a
//! tile's logits, its weights, the rescale) goes through shared memory.
//!
//! A decode step's 16 heads are 16 query rows. Sixteen blocks leave most of
//! the card idle, so the key range is cut into [`seg_keys`]-key segments and
//! the launch is two kernels: [`flash_latent_mma`], one block per (group of
//! [`MMA_ROWS`] query rows, segment) with the `Q·Kᵀ` product on the tensor
//! cores, writing each row's softmax partials per segment (running max,
//! `Σ exp` relative to it, and the un-normalised `Σ exp·V`), and `flash_merge`
//! folding the segments of a row into the final latent row. The segment count comes from
//! the cache height, not from the live key count, so a captured graph's grid
//! is fixed and the step still follows `n_keys_buf` — a segment wholly past
//! the causal limit writes a neutral partial (`m = −inf`, `s = 0`) and reads
//! no key row at all, which is also what keeps padded rows holding NaN out of
//! every result. A cache short enough to hold one segment takes the
//! single-launch [`flash_latent`] instead, which needs no partials and no
//! merge; that choice is made from the cache height when the launch is
//! enqueued, never per key count.
//!
//! Merge order is fixed (ascending segment), so reruns and graph replays are
//! bit-identical. The split launch rounds the query rows and the products to
//! f16 and the single-block form does not, so the two sit inside a band of
//! each other rather than being bit-equal.
//!
//! Reduction structure of the single-block kernel (the fixed, deterministic
//! contract of this family; reruns are bit-identical, CPU bit-identity is not
//! claimed):
//! - QK dot per key: the block splits as [`KEY_TILE`] keys x [`DIM_SPLIT`]
//!   threads. Thread `d` of a key walks dims `d, d + DIM_SPLIT, …` into
//!   [`ILP`] rotating f32 partials (one fused multiply-add each per group of
//!   `ILP` steps, so that many loads are in flight at once), combines them by
//!   `(a0+a1)+(a2+a3)` with the trailing steps folded into `a0`, and the
//!   key's threads combine by the fixed four-step xor butterfly.
//! - Online softmax in `KEY_TILE`-key tiles, keys ascending — the fa4 tile
//!   size. Warp 0 owns the running `(m, s)`: tile max by the five-step
//!   butterfly max, weights `exp(s − m)` per lane, weight sum by the
//!   five-step butterfly sum, state rescale on a max bump (`s` immediately,
//!   the V partials before the tile's accumulation), and publishes the
//!   tile's 32 weights and the rescale to the block.
//! - V accumulation: every thread accumulates its own latent dim over the
//!   tile's keys into [`ILP`] rotating partials (keys `l … l + ILP − 1`), one
//!   fused multiply-add each, combined by the same fixed tree once at the
//!   end.
//! - Final row: `r · (1/s)`, one plain multiply.
//!
//! Keys at or past the causal limit (mask, cache padding) carry weight
//! exactly `0.0`. Only the last tile of a run can reach past the limit, and
//! it takes a guarded path whose `wl == 0.0` test skips those loads — so
//! padded rows holding NaN bit patterns are never read into any result. A
//! tile wholly inside the limit reads only real rows and needs no guard.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::q8_1_quant_vals;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32_f16x2_lo};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, kernel, launch_bounds, launch_contract, thread,
    warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The CPU oracle's f32 -> f16 conversion, reused as this module's device
/// and host rounding so the cache bits cannot drift from the engine's.
pub use gguf::quant::f32_to_f16_bits;

/// Latent tail this kernel family's geometry is built for, and so the block's
/// thread count: one thread per latent dim. `enqueue_flash_latent` rejects
/// any other latent width.
pub(crate) const LATENT: usize = 512;
/// [`LATENT`] as the `u32` block width a launch takes.
const LATENT_U32: u32 = LATENT as u32;
const _: () = assert!(LATENT_U32 as usize == LATENT);
/// Keys per online-softmax tile — one warp's worth, so a tile's max and
/// weight sum are single warp butterflies.
pub(crate) const KEY_TILE: usize = 32;
/// Threads sharing one key's QK dot; the block's `LATENT` threads cover
/// `KEY_TILE` keys at a time.
pub(crate) const DIM_SPLIT: usize = LATENT / KEY_TILE;
const _: () = assert!(LATENT.is_multiple_of(KEY_TILE));
/// Widest `rope_dims + latent` row the shared staging buffer holds.
pub(crate) const MAX_WIDTH: usize = 640;
/// Rotating partials each hot loop carries, and so the loads a thread keeps
/// in flight. Four is measured, not derived: eight measured slower at depth
/// on this card, so the loops are not short of memory-level parallelism. The
/// partials are combined by a fixed tree, never by the loop order.
pub(crate) const ILP: usize = 4;

// ------------------------------------------------- tensor-core geometry

/// Query rows one block of [`flash_kernels::flash_latent_mma`] carries: a
/// decode step's whole head set, and so the `M` axis of its `mma.sync`.
pub const MMA_ROWS: usize = 16;
/// Warps in an `flash_latent_mma` block, one per query row of the group: the
/// block's thread count is the latent width, so the V accumulation gives each
/// thread one latent dim and the online softmax gives each warp one head.
/// Four warps issue the `S = Q·Kᵀ` product; every warp stages, reduces and
/// accumulates.
pub(crate) const MMA_WARPS: usize = MMA_ROWS;
/// Threads in an `flash_latent_mma` block.
pub const MMA_BLOCK: usize = MMA_WARPS * 32;
/// [`MMA_BLOCK`] as the `u32` block width a launch takes.
const MMA_BLOCK_U32: u32 = MMA_BLOCK as u32;
const _: () = assert!(MMA_BLOCK_U32 as usize == MMA_BLOCK);
/// Keys one warp's `mma.sync` `n`-tile covers — the instruction's `n`.
pub const MMA_NTILE: usize = 8;
/// Warps that issue the `S = Q·Kᵀ` `mma.sync`. Four of them cover a tile's
/// keys, and the accumulator's four values per lane then cover every
/// (head, key) slot of the tile exactly once.
pub const MMA_QK_WARPS: usize = 4;
/// Keys one `flash_latent_mma` tile covers: every issuing warp's `n`-tile at
/// once. It is also the lane count of a warp, which is what lets the softmax
/// finish a head's tile without a shared reduction.
pub const MMA_KEYS: usize = MMA_QK_WARPS * MMA_NTILE;
const _: () = assert!(MMA_KEYS == MMA_QK_WARPS * MMA_NTILE);
const _: () = assert!(MMA_KEYS == 32);
/// Dims one `mma.sync` step covers — the instruction's `k`.
pub const MMA_K: usize = 16;
/// The `rope_dims + latent` row width `flash_latent_mma` is built for. Its
/// shared tiles are sized for it and its `k` walk assumes the width divides
/// by `2 * MMA_K`, so the host entry rejects any other width rather than
/// reading past a tile.
pub(crate) const MMA_WIDTH: usize = 576;
const _: () = assert!(MMA_WIDTH.is_multiple_of(2 * MMA_K));
/// f16 lanes between two staged query rows. `MMA_QSTRIDE * 2 ≡ 16 (mod 128)`
/// is what makes every `ldmatrix` phase read eight rows across all thirty-two
/// banks exactly once; the unpadded 576 would put all sixteen rows in the
/// same bank.
pub(crate) const MMA_QSTRIDE: usize = MMA_WIDTH + 8;
const _: () = assert!(MMA_QSTRIDE * 2 % 128 == 16);
/// `MMA_QSTRIDE` as u32 words, the staged tile's element type.
pub(crate) const MMA_QROW_W: usize = MMA_QSTRIDE / 2;
/// u32 words one staged query tile takes.
pub(crate) const MMA_QWORDS: usize = MMA_ROWS * MMA_QROW_W;
// A lane stages `MMA_WIDTH / 64` words of a query or key row — a
// compile-time trip count, so the staging loop's loads all issue before any
// of them is consumed — and the staging loops cover the row exactly.
const _: () = assert!(MMA_WIDTH.is_multiple_of(64));
/// f16 lanes between two staged key rows — padded for the same reason as
/// [`MMA_QSTRIDE`], and by the same amount.
pub(crate) const MMA_KSTRIDE: usize = MMA_WIDTH + 8;
const _: () = assert!(MMA_KSTRIDE * 2 % 128 == 16);
/// `MMA_KSTRIDE` as u32 words.
pub(crate) const MMA_KROW_W: usize = MMA_KSTRIDE / 2;
/// u32 words the staged key tile takes. The tile is whole — every key row's
/// whole width, so the `k` axis is one walk and the row's latent tail is
/// still in shared memory when the V accumulation wants it. Together with
/// the query tile that is past the 48 KB a static allocation may take, so
/// both live in this kernel's dynamic shared memory.
pub(crate) const MMA_KWORDS: usize = MMA_KEYS * MMA_KROW_W;
/// Key rows one warp stages — a compile-time trip count, so the staging
/// loads all issue before any of them is consumed.
pub const MMA_KROWS_PER_WARP: usize = MMA_KEYS / MMA_WARPS;
const _: () = assert!(MMA_KEYS.is_multiple_of(MMA_WARPS));
/// Bytes of dynamic shared memory one `flash_latent_mma` block takes: the
/// query tile then the key tile, in that order. Both are `u32` arrays and
/// the base is sixteen-byte aligned, so the query tile's word count is the
/// key tile's offset.
pub(crate) const MMA_DYN_BYTES: usize = (MMA_QWORDS + MMA_KWORDS) * 4;
/// `#[launch_contract(dynamic_shared = ...)]` takes an integer literal and
/// not a constant, so the kernel's declaration spells the byte count out.
/// This is the two sides agreeing: change the geometry and the build stops
/// here rather than at a launch the driver rejects.
const _: () = assert!(MMA_DYN_BYTES == 56064);
/// [`MMA_DYN_BYTES`] as the `u32` dynamic shared-memory size a launch takes.
const MMA_DYN_BYTES_U32: u32 = MMA_DYN_BYTES as u32;
const _: () = assert!(MMA_DYN_BYTES_U32 as usize == MMA_DYN_BYTES);
/// Floats a tile's per-head logits (then weights) take.
pub const MMA_TILE: usize = MMA_ROWS * MMA_KEYS;
/// u32 words between two staged rows of a `width`-wide row — the padded
/// stride of [`MMA_QROW_W`] and [`MMA_KROW_W`] at any width: eight rows of
/// one `ldmatrix` phase then cover the thirty-two banks once.
pub const fn mma_row_words(width: usize) -> usize {
    (width + 8) / 2
}
/// u32 words the staged query tile of a `width`-wide row takes.
pub const fn mma_qwords(width: usize) -> usize {
    MMA_ROWS * mma_row_words(width)
}
/// u32 words the staged key tile of a `width`-wide row takes.
const fn mma_kwords(width: usize) -> usize {
    MMA_KEYS * mma_row_words(width)
}
/// Bytes of dynamic shared memory a tensor-core segment block of a
/// `width`-wide row takes: the query tile, then the key tile.
pub const fn mma_dyn_bytes(width: usize) -> usize {
    (mma_qwords(width) + mma_kwords(width)) * 4
}
const _: () = assert!(mma_row_words(MMA_WIDTH) == MMA_QROW_W);
const _: () = assert!(mma_row_words(MMA_WIDTH) == MMA_KROW_W);
const _: () = assert!(mma_qwords(MMA_WIDTH) == MMA_QWORDS);
const _: () = assert!(mma_kwords(MMA_WIDTH) == MMA_KWORDS);
const _: () = assert!(mma_dyn_bytes(MMA_WIDTH) == MMA_DYN_BYTES);
/// Keys one segment of the tensor-core pass walks, and so [`seg_keys`]'s
/// default. One block carries every head, so the grid is the segment count
/// alone: 128-key segments would leave a 4192-row cache at thirty-three
/// blocks on eighty-four SMs. Sixty-four keys give sixty-six blocks there and
/// keep the merge's fold half the length a thirty-two-key segment would.
pub const MMA_SEG_KEYS: usize = 64;

/// Segments whose partials a split-K merge loads as one batch before it
/// folds them: every merge kernel (this file's, the GQA flash's, V4.1's
/// attention merge) walks its segments in batches of this many.
pub const MERGE_BATCH: usize = 16;

/// Blocks the tensor-core pass needs for `q_rows` query rows: the rows are
/// cut into groups of [`MMA_ROWS`], and every group runs each segment.
pub fn mma_groups(q_rows: usize) -> usize {
    q_rows.div_ceil(MMA_ROWS)
}

/// Keys per segment: [`MMA_SEG_KEYS`], unless `BLOOMERY_FLASH_SEG` names
/// another multiple of [`KEY_TILE`]. Read once, at first use: the value fixes
/// a captured graph's grid, so it must not change between capture and replay.
/// A value that is set but unusable panics rather than falling back — a
/// sweep row that silently ran the default would be a wrong measurement.
///
/// `BLOOMERY_FLASH_MMA` is refused by name, whatever its value: the
/// tensor-core pass is the only segment pass, and a run that asks for another
/// must not silently time this one.
pub fn seg_keys() -> usize {
    static SEG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SEG.get_or_init(|| {
        if let Ok(v) = std::env::var("BLOOMERY_FLASH_MMA") {
            panic!(
                "BLOOMERY_FLASH_MMA={v} is not a lever: the tensor-core segment pass is the only \
                 segment pass; unset it"
            );
        }
        match std::env::var("BLOOMERY_FLASH_SEG") {
            Err(_) => MMA_SEG_KEYS,
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n >= KEY_TILE && n.is_multiple_of(KEY_TILE) => n,
                _ => panic!(
                    "BLOOMERY_FLASH_SEG={v} is not a positive multiple of KEY_TILE ({KEY_TILE})"
                ),
            },
        }
    })
}

/// Segments a `cache_rows`-tall cache is cut into. One means the cache fits
/// in a single segment and the single-launch [`flash_latent`] serves it.
pub fn segments_for(cache_rows: usize) -> usize {
    cache_rows.max(1).div_ceil(seg_keys())
}

/// Length the split launch's `Σ exp·V` partials buffer needs for `q_rows`
/// query rows over a `cache_rows`-tall cache.
pub fn partials_v_len(q_rows: usize, cache_rows: usize) -> usize {
    q_rows * segments_for(cache_rows) * LATENT
}

/// Length the split launch's `(max, Σ exp)` partials buffer needs — two f32
/// per (query row, segment).
pub fn partials_ms_len(q_rows: usize, cache_rows: usize) -> usize {
    q_rows * segments_for(cache_rows) * 2
}

// --------------------------------------------------------------- cores

/// `exp(x)` on device: the hardware's `ex2.approx.f32` on `x · log2(e)` — a
/// deterministic instruction whose ~1e-7 relative error sits far inside every
/// band this package gates against, with no dependence on host libm.
#[inline(always)]
pub fn dev_exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x * std::f32::consts::LOG2_E)
}

/// Two f32 rounded to f16 and packed, `lo` in the low half: the hardware's
/// `cvt.rn.f16x2.f32`, with a NaN lane replaced by `sign | 0x7e00`. Each half
/// is [`f32_to_f16_bits`] of its value on every f32 — the conversion is IEEE
/// nearest even as that function is, and only its canonical NaN differs,
/// which the select restores; `gate-gpu-p4` holds the two over all 2^32
/// inputs. Device only.
#[inline(always)]
pub fn f32x2_to_f16x2_bits(lo: f32, hi: f32) -> u32 {
    let p = cvt_f16x2_f32(lo, hi);
    let l = if lo.is_nan() {
        ((lo.to_bits() >> 16) & 0x8000) | 0x7e00
    } else {
        p & 0xffff
    };
    let h = if hi.is_nan() {
        ((hi.to_bits() >> 16) & 0x8000) | 0x7e00
    } else {
        p >> 16
    };
    l | (h << 16)
}

/// One `f16` bit pattern widened to `f32` by the hardware's widening
/// convert. Widening `f16` to `f32` is exact, so this is
/// `cores::half_to_f32`'s value for every finite and infinite input — one
/// instruction instead of a decode whose subnormal branch is a loop.
#[inline(always)]
pub(crate) fn half_bits_to_f32(bits: u16) -> f32 {
    cvt_f32_f16x2_lo(bits as u32)
}

/// One step of the online-softmax fold over `(max, Σ exp, Σ exp·v)`
/// partials: partial `(mj, sj, vj)` joins the running `(mx, s, acc)`, the
/// side with the smaller max rescaled, and the new running triple comes
/// back. `sj` is non-zero — a caller skips a neutral partial before loading
/// its `vj`. The merges fold their segments through this, and a head's sink
/// joins as the partial `(sink, 1, 0)`: one more logit in the denominator
/// and nothing in the value.
#[inline(always)]
pub fn online_fold(mx: f32, s: f32, acc: f32, mj: f32, sj: f32, vj: f32) -> (f32, f32, f32) {
    if mj > mx {
        // A first bump scales by 0.0 — the partials are still zero, so the
        // reset and the scale are the same value.
        let f = if mx > f32::NEG_INFINITY {
            dev_exp(mx - mj)
        } else {
            0.0
        };
        (mj, f32::mul_add(s, f, sj), f32::mul_add(acc, f, vj))
    } else {
        let g = dev_exp(mj - mx);
        (mx, f32::mul_add(sj, g, s), f32::mul_add(vj, g, acc))
    }
}

/// The tensor-core segment pass's walk — the query staging, the tile loop
/// (key staging, `S = Q·Kᵀ` on `mma.sync`, the per-head online softmax, the
/// V accumulation) and the partial stores — written once and expanded in
/// every entry that walks keys this way: [`flash_kernels::flash_latent_mma`]
/// (576-wide rows, rope then latent) and the V4.1 attention's segment pass
/// (512-wide rows, the value is the whole row). An entry keeps its own grid
/// decomposition, key source and per-head limits, and names its locals as
/// the arguments.
///
/// A macro, not an `#[inline(always)]` function: the same walk moved into a
/// function compiled `flash_latent_mma` to other machine code (another
/// register allocation and block layout) where this expansion keeps its PTX
/// to the byte. The loops the entry would mark `#[unroll]` call the importer's
/// unroll marker first, which is what that attribute expands to; the
/// attribute itself is rewritten only in a `#[kernel]`'s own text.
///
/// Arguments: `width`, the row width the shared tiles are sized for (a
/// constant, a multiple of 64); `q`, the query rows; `kvw`, the key
/// source's rows as u32 words; `key => row`, the name the walk binds a
/// staged key index to and the source row it reads for it — the key itself
/// for a contiguous source, an index-list load for a gathering one, asked
/// only below `hi`; `scratch`, the block's staged query tile, key tile,
/// logits, weights, rescale and per-head limit pointers, the limits written
/// before the walk; `rows`, the launch's query rows; `base_row`, this
/// block's first; `n_seg`/`seg`, the partial slots per row and this block's;
/// `lo`/`hi`, the key range; `width_rt`, `rope`, `lat`, the row's width and
/// its split (the value is the last `lat` dims, one per thread); `scale`;
/// `tid`; `part_v`/`part_ms`, the partial outputs.
///
/// The fragment coordinates, masking and reduction order are
/// `flash_latent_mma`'s, documented there. The expansion's `unsafe` blocks
/// rest on the caller's contract: `width` is a multiple of 64 with
/// `(width + 8) * 2 ≡ 16 (mod 128)` (the entry asserts both at compile
/// time); the scratch pointers are this block's shared memory sized for
/// `width`, the limits written; `q` holds `rows` rows of `width_rt = rope +
/// lat = width` f32, `rope` even; `row` maps every key below `hi` to a row
/// of the source, whose buffer is device-allocated; `lat` is the block's
/// thread count; the partial buffers hold `n_seg` slots per row. Every
/// thread of the block runs the expansion.
#[macro_export]
macro_rules! mma_segment_walk {
    (
        width: $width:expr,
        q: $q:ident,
        kvw: $kvw:ident,
        key: $key:ident => $row:expr,
        scratch: ($qs:ident, $kt:ident, $klog:ident, $kw:ident, $vms_sh:ident, $lim_sh:ident),
        rows: $rows:ident,
        base_row: $base_row:ident,
        n_seg: $n_seg:ident,
        seg: $seg:ident,
        lo: $lo:ident,
        hi: $hi_max:ident,
        width_rt: $width_rt:ident,
        rope: $rope:ident,
        lat: $lat:ident,
        scale: $scale:ident,
        tid: $tid:ident,
        part_v: $part_v:ident,
        part_ms: $part_ms:ident $(,)?
    ) => {{
        // The group's query rows, f32 pairs rounded to one f16 pair each:
        // warp `wid` stages row `wid`, its lanes walking the row's words. The
        // loads are issued as a batch and consumed as a batch, so the row
        // costs one global latency rather than `width / 64` of them.
        let lane = ::cuda_device::warp::lane_id() as usize;
        let wid = $tid / 32;
        {
            let row = $base_row + wid;
            let live = row < $rows;
            let mut raw = [0.0f32; 2 * ($width / 64)];
            let mut i = 0usize;
            while i < const { $width / 64 } {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                let w = lane + i * 32;
                if live {
                    // SAFETY: row < rows and 2 * w + 1 < width, so both
                    // loads are inside that query row (the caller's
                    // contract).
                    unsafe {
                        raw[2 * i] = *$q.get_unchecked(row * $width_rt + 2 * w);
                        raw[2 * i + 1] = *$q.get_unchecked(row * $width_rt + 2 * w + 1);
                    }
                }
                i += 1;
            }
            let mut i = 0usize;
            while i < const { $width / 64 } {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                let w = lane + i * 32;
                let a = $crate::flash::f32_to_f16_bits(raw[2 * i]) as u32;
                let c = $crate::flash::f32_to_f16_bits(raw[2 * i + 1]) as u32;
                // SAFETY: wid < MMA_ROWS and w < width / 2 < the padded
                // row stride bound the store inside the query tile.
                unsafe {
                    *$qs.add(wid * const { $crate::flash::mma_row_words($width) } + w) =
                        a | (c << 16);
                }
                i += 1;
            }
        }

        // This warp's eight keys inside a tile and its head in the softmax,
        // and this thread's latent dim in the V loop.
        let key0 = wid * $crate::flash::MMA_NTILE;
        let d0 = $tid;
        // `ldmatrix` lane roles: A addresses row `lane % 16` at dim half
        // `lane / 16`; B addresses key `lane % 8` of the warp's eight at dim
        // octet `lane / 8`.
        let arow = lane % $crate::flash::MMA_ROWS;
        let ahalf = lane / $crate::flash::MMA_ROWS;
        let bkey = lane % $crate::flash::MMA_NTILE;
        let boct = lane / $crate::flash::MMA_NTILE;

        let mut mx = f32::NEG_INFINITY;
        let mut ss = 0.0f32;
        let mut r = [0.0f32; $crate::flash::MMA_ROWS];

        let mut blk = $lo;
        while blk < $hi_max {
            // The previous tile's fragment, weight and value reads are done.
            // On the first pass this is also what publishes the query tile.
            ::cuda_device::thread::sync_threads();
            // This warp's key rows, issued as a batch: the cache rows are
            // already f16, so this stage is a copy. A row at or past
            // `hi_max` — which may hold NaN — is not read and stays zero,
            // and its weight is zero, so it leaves every partial alone.
            let mut rr = 0usize;
            while rr < $crate::flash::MMA_KROWS_PER_WARP {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                let row = wid * $crate::flash::MMA_KROWS_PER_WARP + rr;
                let $key = blk + row;
                let live = $key < $hi_max;
                let mut raw = [0u32; $width / 64];
                let mut i = 0usize;
                while i < const { $width / 64 } {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    let ww = lane + i * 32;
                    if live {
                        // The caller's row expression stays outside the
                        // unsafe block: a gathering source's index load is
                        // its own unsafe operation, with its own SAFETY.
                        let src_row: usize = $row;
                        // SAFETY: key < hi, so `src_row` names a row of the
                        // source, and 2 * ww < width puts the load inside
                        // it (the caller's contract).
                        unsafe {
                            raw[i] = *$kvw.add(src_row * ($width_rt / 2) + ww);
                        }
                    }
                    i += 1;
                }
                let mut i = 0usize;
                while i < const { $width / 64 } {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    let ww = lane + i * 32;
                    // SAFETY: row < MMA_KEYS and ww < width / 2 < the
                    // padded row stride bound the store inside the key tile.
                    unsafe {
                        *$kt.add(row * const { $crate::flash::mma_row_words($width) } + ww) =
                            raw[i];
                    }
                    i += 1;
                }
                rr += 1;
            }
            ::cuda_device::thread::sync_threads();

            // ---- QK: S = Q · Kᵀ for all sixteen heads, on the tensor cores
            let mut c = [0.0f32; 4];
            if wid < $crate::flash::MMA_QK_WARPS {
                // Two k-steps per B load: `ldmatrix.x4` returns four 8x8
                // tiles, which is thirty-two dims of this warp's eight keys.
                // `width` divides by `2 * MMA_K`, so this walk covers the
                // row exactly.
                let mut d = 0usize;
                while d < $width {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    // SAFETY: key row `key0 + bkey < MMA_KEYS` and words
                    // `d / 2 + boct * 4 .. + 4 <= width / 2` are inside the
                    // staged key tile.
                    let bp = unsafe {
                        $kt.add(
                            (key0 + bkey) * const { $crate::flash::mma_row_words($width) }
                                + d / 2
                                + boct * 4,
                        )
                    };
                    // SAFETY: every lane of the warp reaches this load with
                    // the same qualifiers and an address inside the key tile,
                    // and the barrier above orders the staging writes first.
                    let bf = unsafe {
                        ::cuda_device::wmma::ldmatrix_x4_shared_u32(
                            ::cuda_device::shared::cvta_generic_to_shared_u32(
                                bp.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    // SAFETY: query row `arow < MMA_ROWS` and words
                    // `d / 2 + ahalf * (MMA_K / 4) .. + 4 + MMA_K / 2 <=
                    // width / 2` keep this and `ap1` inside the query tile.
                    let ap0 = unsafe {
                        $qs.add(
                            arow * const { $crate::flash::mma_row_words($width) }
                                + d / 2
                                + ahalf * ($crate::flash::MMA_K / 4),
                        )
                    };
                    // SAFETY: every lane of the warp reaches this load with
                    // the same qualifiers and `ap0` inside the query tile,
                    // published by the first pass's barrier.
                    let af0 = unsafe {
                        ::cuda_device::wmma::ldmatrix_x4_shared_u32(
                            ::cuda_device::shared::cvta_generic_to_shared_u32(
                                ap0.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    // SAFETY: the whole warp issues this `mma.sync` (the
                    // branch is on the warp index) with fragments it loaded.
                    c = unsafe {
                        ::cuda_device::wmma::mma_m16n8k16_f32_f16(c, af0, [bf[0], bf[1]])
                    };
                    // SAFETY: `MMA_K / 2` words past `ap0` is still inside
                    // its query row — the bound on `ap0` includes this step.
                    let ap1 = unsafe { ap0.add($crate::flash::MMA_K / 2) };
                    // SAFETY: every lane of the warp reaches this load with
                    // the same qualifiers and `ap1` inside the query tile,
                    // published by the first pass's barrier.
                    let af1 = unsafe {
                        ::cuda_device::wmma::ldmatrix_x4_shared_u32(
                            ::cuda_device::shared::cvta_generic_to_shared_u32(
                                ap1.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    // SAFETY: the whole warp issues this `mma.sync` (the
                    // branch is on the warp index) with fragments it loaded.
                    c = unsafe {
                        ::cuda_device::wmma::mma_m16n8k16_f32_f16(c, af1, [bf[2], bf[3]])
                    };
                    d += 2 * $crate::flash::MMA_K;
                }
            }
            // The accumulator's four values are heads `group` and
            // `group + 8` at keys `key0 + 2 * thread + {0, 1}`. A key at or
            // past its head's limit is `−inf`, which also covers every key
            // at or past `hi_max` (their tile rows were staged zero).
            if wid < $crate::flash::MMA_QK_WARPS {
                let g = lane / 4;
                let t4 = lane % 4;
                let mut j = 0usize;
                while j < 4 {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    let head = g + if j >= 2 {
                        $crate::flash::MMA_ROWS / 2
                    } else {
                        0
                    };
                    let key = key0 + 2 * t4 + (j & 1);
                    // SAFETY: head < MMA_ROWS bounds the read inside the
                    // limits, written before the tile loop's first barrier.
                    let lim = unsafe { *$lim_sh.add(head) } as usize;
                    let sv = if blk + key < lim {
                        $scale * c[j]
                    } else {
                        f32::NEG_INFINITY
                    };
                    // SAFETY: head < MMA_ROWS and key < MMA_KEYS bound the
                    // slot, and one lane of one warp writes each.
                    unsafe {
                        *$klog.add(head * $crate::flash::MMA_KEYS + key) = sv;
                    }
                    j += 1;
                }
            }
            ::cuda_device::thread::sync_threads();

            // ---- online softmax: warp `wid` owns head `wid`, its
            // thirty-two lanes the tile's thirty-two keys, so the head's
            // (max, Σ exp) is a warp reduction and never leaves registers.
            // SAFETY: wid < MMA_ROWS and lane < MMA_KEYS bound the read
            // inside the logits, published by the barrier above.
            let sv = unsafe { *$klog.add(wid * $crate::flash::MMA_KEYS + lane) };
            let smax = ::cuda_device::warp::reduce_max_f32(sv);
            // FlashMS update: s is rescaled here, the V partials just
            // before this tile's accumulation (the CPU oracle's order).
            // A first bump scales by 0.0 — the partials are still zero,
            // so the reset and the scale are the same value.
            let mut vms = 1.0f32;
            if smax > mx {
                vms = if mx > f32::NEG_INFINITY {
                    $crate::flash::dev_exp(mx - smax)
                } else {
                    0.0
                };
                ss *= vms;
                mx = smax;
            }
            let w = if sv == f32::NEG_INFINITY {
                0.0
            } else {
                $crate::flash::dev_exp(sv - mx)
            };
            ss += ::cuda_device::warp::reduce_sum_f32(w);
            // SAFETY: wid < MMA_ROWS and lane < MMA_KEYS bound the slot
            // inside the weights, and one lane writes each.
            unsafe {
                *$kw.add(wid * $crate::flash::MMA_KEYS + lane) = w;
            }
            if lane == 0 {
                // SAFETY: wid < MMA_ROWS bounds the store, and lane 0 of
                // warp `wid` is its only writer.
                unsafe {
                    *$vms_sh.add(wid) = vms;
                }
            }
            ::cuda_device::thread::sync_threads();

            // ---- V accumulation: this thread's latent dim for every head,
            // the tile's keys ascending, one partial per head. The values
            // come out of the staged tile — the key row's latent tail is
            // still there — so this loop touches no global memory and a row
            // past `hi_max` reads the zero the staging left. The trip count
            // is the whole tile and not the live keys: a compile-time bound
            // is what lets the sixteen accumulators be scheduled across
            // keys, and it measured faster even at a depth where most of
            // the tile is dead.
            let mut h = 0usize;
            while h < $crate::flash::MMA_ROWS {
                ::cuda_device::thread::__unroll_config::<{ 0 }>();
                // SAFETY: h < MMA_ROWS bounds the read, and the rescales
                // hold this tile's (published before the barrier above).
                unsafe {
                    r[h] *= *$vms_sh.add(h);
                }
                h += 1;
            }
            let vword = ($rope + d0) / 2;
            let vlo = ($rope + d0).is_multiple_of(2);
            let mut l = 0usize;
            while l < $crate::flash::MMA_KEYS {
                // SAFETY: l < MMA_KEYS and vword < width / 2 < the padded
                // row stride bound the read inside the staged tile.
                let (v0, v1) = unsafe {
                    ::cuda_device::convert::cvt_f32x2_f16x2(
                        *$kt.add(l * const { $crate::flash::mma_row_words($width) } + vword),
                    )
                };
                let v = if vlo { v0 } else { v1 };
                let mut h = 0usize;
                while h < $crate::flash::MMA_ROWS {
                    ::cuda_device::thread::__unroll_config::<{ 0 }>();
                    // SAFETY: h < MMA_ROWS and l < MMA_KEYS bound the read
                    // inside the weights.
                    let wgt = unsafe { *$kw.add(h * $crate::flash::MMA_KEYS + l) };
                    r[h] = f32::mul_add(wgt, v, r[h]);
                    h += 1;
                }
                l += 1;
            }
            blk += $crate::flash::MMA_KEYS;
        }

        let mut h = 0usize;
        while h < $crate::flash::MMA_ROWS {
            ::cuda_device::thread::__unroll_config::<{ 0 }>();
            let row = $base_row + h;
            if row < $rows {
                // SAFETY: row < rows and d0 < lat, the block's thread
                // count, so the store is inside part_v (the caller's
                // contract).
                unsafe {
                    *$part_v.get_unchecked_mut((row * $n_seg + $seg) * $lat + d0) = r[h];
                }
            }
            h += 1;
        }
        if lane == 0 {
            let row = $base_row + wid;
            if row < $rows {
                let idx = row * $n_seg + $seg;
                // SAFETY: idx < rows * n_seg, so both slots are inside
                // part_ms; `mx` and `ss` are this warp's head's state.
                unsafe {
                    *$part_ms.get_unchecked_mut(2 * idx) = mx;
                    *$part_ms.get_unchecked_mut(2 * idx + 1) = ss;
                }
            }
        }
    }};
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod flash_kernels {
    use super::*;

    /// Convert `m` new KV rows (`m * width` f32, row-major, token-major) to
    /// f16 with the oracle's rounding and store them at rows
    /// `pos..pos + m` of the preallocated cache. `pos` is a launch scalar:
    /// a captured graph replays it frozen at its capture-time value — the
    /// captured step wants `kv_append_pos_buf`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= m * width, dst.len() >= (pos + m) * width)
    )]
    pub fn kv_append(m: u32, width: u32, pos: u32, src: &[f32], mut dst: DisjointSlice<u16>) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        let row = i / w;
        let col = i - row * w;
        // SAFETY: i < total <= src.len(); row < m and col < width, so the
        // store index is < (pos + m) * width <= dst.len() (launch contract).
        unsafe {
            *dst.get_unchecked_mut((pos as usize + row) * w + col) =
                f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// `kv_append` with `pos` read from `pos_buf[0]` at run time — the
    /// variant a captured decode step replays against a new position by
    /// rewriting the buffer between launches. A row that would land at or
    /// past `dst_rows` (the cache's allocated height) is refused: `pos`
    /// comes from device memory, so that bound cannot be a launch contract;
    /// the row is stored nowhere and raises [`FaultSite::CachePos`] on
    /// `fault`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            pos_buf.len() >= 1,
            src.len() >= m * width,
            dst.len() >= dst_rows * width
        )
    )]
    pub fn kv_append_pos_buf(
        m: u32,
        width: u32,
        dst_rows: u32,
        pos_buf: &[u32],
        src: &[f32],
        mut dst: DisjointSlice<u16>,
        fault: FaultSink,
    ) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        // SAFETY: pos_buf.len() >= 1 by the launch contract.
        let pos = unsafe { *pos_buf.get_unchecked(0) } as usize;
        let row = i / w;
        let col = i - row * w;
        if pos + row >= dst_rows as usize {
            if col == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return;
        }
        // SAFETY: i < total <= src.len(); pos + row < dst_rows and col <
        // width, so the store index is < dst_rows * width <= dst.len().
        unsafe {
            *dst.get_unchecked_mut((pos + row) * w + col) = f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// Latent flash attention over the absorbed KV cache, `m` query tokens
    /// (1..=8) with every head in one launch. `q` holds `m * n_heads` rows
    /// of `width = rope_dims + latent` f32, row `t * n_heads + h` (the
    /// `kqv_compressed` column order); `kv` is the `[dst_rows x width]` u16
    /// cache; the live key count is `n_keys_buf[0]`. Query `t` attends to
    /// keys `0..n_keys − m + t` (its own rows are already appended, so
    /// `n_keys >= m`); `y` holds `m * n_heads` rows of `latent` f32 in the
    /// same row order. Deterministic by the module doc's reduction contract.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            y.len() >= q_rows * latent
        )
    )]
    pub fn flash_latent(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // The query row, staged once: every key's dot reads all of it.
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        // The tile's scaled logits, then its weights, then the max-bump
        // rescale at [0] and the final `1/s` at [1].
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let rope = rope_dims as usize;
        let lat = latent as usize;
        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw pointer form is the only way to reach it
        // without a reference to a `static mut`. Every access below is
        // bounded by the array's length and ordered by `sync_threads`.
        let (qs, klog, kw, st) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QROW),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut ST),
            )
        };
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        // SAFETY: the pointers are this block's shared scratch, `row` is
        // inside q's rows and `0..limit` inside kv's (launch contract).
        let (_, s_sum, r) = unsafe {
            latent_range(
                q, kv, qs, klog, kw, st, row, tid, 0, limit, rope, lat, scale,
            )
        };

        if tid == 0 {
            // SAFETY: thread 0 alone writes ST[1]; no thread reads it before
            // the barrier below. `s_sum` is the block state, valid in warp 0.
            unsafe {
                *st.add(1) = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            }
        }
        thread::sync_threads();
        // SAFETY: ST[1] is published above.
        let s_inv = unsafe { *st.add(1) };
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * lat + tid) = s_inv * r;
        }
    }

    /// [`flash_latent`] with [`flash_merge_q8`]'s q8_1 side output. The
    /// single-segment path never reaches the merge, so without this twin a
    /// cache short enough to hold one segment would take the folded step
    /// down a path that does not quantize at all — the two paths have to
    /// agree or neither may fold.
    ///
    /// The staging buffer is `QROW`, the query row's own: the last read of
    /// it is inside `latent_range`, which ends at a `sync_threads` before
    /// this writes.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            y.len() >= q_rows * latent,
            q3a.len() >= m_lo * 64 * half_it,
            q4a.len() >= m_lo * 256 * quad_it,
            q6a.len() >= m_lo * 128 * half_it,
            s8a.len() >= m_lo * 8 * n_sb,
            d8a.len() >= m_lo * 2 * n_sb,
            q3b.len() >= (q_rows - m_lo) * 64 * half_it,
            q4b.len() >= (q_rows - m_lo) * 256 * quad_it,
            q6b.len() >= (q_rows - m_lo) * 128 * half_it,
            s8b.len() >= (q_rows - m_lo) * 8 * n_sb,
            d8b.len() >= (q_rows - m_lo) * 2 * n_sb
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_q8(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut y: DisjointSlice<f32>,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let rope = rope_dims as usize;
        let lat = latent as usize;
        // SAFETY: each `static mut` is this block's own shared allocation,
        // every access bounded and ordered by `sync_threads`.
        let (qs, klog, kw, st) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QROW),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut ST),
            )
        };
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        // SAFETY: the pointers are this block's shared scratch, `row` is
        // inside q's rows and `0..limit` inside kv's (launch contract).
        let (_, s_sum, r) = unsafe {
            latent_range(
                q, kv, qs, klog, kw, st, row, tid, 0, limit, rope, lat, scale,
            )
        };

        if tid == 0 {
            // SAFETY: thread 0 alone writes ST[1]; no thread reads it before
            // the barrier below. `s_sum` is the block state, valid in warp 0.
            unsafe {
                *st.add(1) = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            }
        }
        thread::sync_threads();
        // SAFETY: ST[1] is published above.
        let s_inv = unsafe { *st.add(1) };
        let v = s_inv * r;
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * lat + tid) = v;
        }
        // SAFETY: `qs` is this block's `MAX_WIDTH >= LATENT` shared array
        // and its last read is inside `latent_range`, before the barrier
        // above; the launch contract bounds both output sets and `row`
        // picks a column inside its own half.
        unsafe {
            quant_row(
                v, qs, tid, lat, row, m_lo, n_sb, half_it, quad_it, &mut q3a, &mut q4a, &mut q6a,
                &mut s8a, &mut d8a, &mut q3b, &mut q4b, &mut q6b, &mut s8b, &mut d8b, fault,
            );
        }
    }

    /// The segment pass of the split launch: block `(group, segment)` attends
    /// keys `[segment * seg_keys, min((segment + 1) * seg_keys, limit))` for
    /// the [`MMA_ROWS`] query rows `group * MMA_ROWS ..` at once and writes
    /// each row's partials for [`flash_merge`] and [`flash_merge_q8`] to fold:
    /// `part_ms` holds `(running max, Σ exp)` per (row, segment), `part_v` the
    /// row's un-normalised `Σ exp·V` relative to that max. Segment is the slow
    /// block index, so the blocks sharing a key slice run together.
    ///
    /// The heads are the `M` axis of `mma.sync.aligned.m16n8k16`: one
    /// instruction takes sixteen heads times eight keys times sixteen dims,
    /// where a scalar walk issues that many fused multiply-adds. The
    /// fragment coordinates, in `cuda_device::wmma`'s terms (`group =
    /// lane / 4`, `thread = lane % 4`):
    /// - `A` is the staged query tile, row-major 16x16 (head, dim). One
    ///   `ldmatrix.x4` with lane `l` addressing row `l % 16` at dim
    ///   `d + 8 * (l / 16)` returns exactly `a[0..4]`: `a[0]` is
    ///   `q[group][d + 2*thread + {0,1}]`, `a[1]` the same at head
    ///   `group + 8`, `a[2]` and `a[3]` those two at `d + 8`.
    /// - `B` is the staged key chunk, column-major 16x8 (dim, key) — which
    ///   is what a `[key][dim]` row already is, so nothing is transposed.
    ///   One `ldmatrix.x4` with lane `l` addressing key
    ///   `key0 + l % 8` at dim `d + 8 * (l / 8)` returns two whole `B`
    ///   fragments: `[r0, r1]` for dims `d..d+16` and `[r2, r3]` for
    ///   `d+16..d+32`, each element `kt[key0 + group][d + 2*thread + {0,1}]`
    ///   as `B`'s `(row = dim, col = key)` wants.
    /// - `C`/`D` element `j` is head `group + 8 * (j >= 2)` and key
    ///   `key0 + 2 * thread + (j & 1)`, which is where the logit store below
    ///   sends it.
    ///
    /// Both staged strides are padded so that eight rows of one `ldmatrix`
    /// phase cover the thirty-two shared banks exactly once.
    ///
    /// Every row of the group has its own causal limit, so a head whose
    /// limit the segment has already passed sees `−inf` at every key and
    /// leaves the neutral partial (`m = −inf`, `s = 0`) and a zero `Σ exp·V`
    /// behind, which the merge skips. A block whose group is wholly past its
    /// limits writes the neutral partial and returns before reading a key
    /// row. No key row at or past `hi_max` — the group's widest live key
    /// — is ever loaded: the staging leaves those tile rows zero and the
    /// logit store masks them, which is what keeps padded rows holding NaN
    /// out of every result.
    ///
    /// The query rows and the products are f16, so this entry is a different
    /// arithmetic class from the f32 single-block kernel and carries its own
    /// band.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        dynamic_shared = 56064,
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_mma(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        // Per head, the tile's scaled logits and then its weights.
        static mut KLOG: SharedArray<f32, MMA_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, MMA_TILE> = SharedArray::UNINIT;
        // Per head, the tile's max-bump rescale and the row's causal limit.
        static mut VMS: SharedArray<f32, MMA_ROWS> = SharedArray::UNINIT;
        static mut LIM: SharedArray<u32, MMA_ROWS> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let rows = q_rows as usize;
        let n_seg = segs as usize;
        let groups = rows.div_ceil(MMA_ROWS);
        if b >= groups * n_seg {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let seg = b / groups;
        let grp = b - seg * groups;
        let base_row = grp * MMA_ROWS;
        let rope = rope_dims as usize;
        let lat = latent as usize;
        let width = rope + lat;
        let lo = seg * seg_keys as usize;
        // The group's widest limit: limits rise with the row, and a row past
        // `q_rows` is not a row at all, so the last live row carries it.
        let last_row = (base_row + MMA_ROWS - 1).min(rows - 1);
        let lim_max = causal_limit(n_keys_buf, dst_rows, last_row, n_heads, m);
        if lo >= lim_max {
            if tid < MMA_ROWS && base_row + tid < rows {
                let idx = (base_row + tid) * n_seg + seg;
                // SAFETY: idx < q_rows * segs, so both slots are inside
                // part_ms (launch contract).
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform, and no key row of this segment is read
        }
        let hi_max = (lo + seg_keys as usize).min(lim_max);

        // The group's query rows and the tile's key rows as f16, both in the
        // block's dynamic shared memory: the query tile first, the key tile
        // at its end. The launch contract declares exactly
        // [`MMA_DYN_BYTES`], which is the two word counts.
        // SAFETY: each pointer below is this block's own shared allocation;
        // every access is bounded by the declared word count and ordered by
        // `sync_threads`.
        let (qs, kt, klog, kw, vms_sh, lim_sh) = unsafe {
            (
                DynamicSharedArray::<u32, 16>::get(),
                DynamicSharedArray::<u32, 16>::offset(MMA_QWORDS * 4),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut VMS),
                SharedArray::as_raw_mut_ptr(&raw mut LIM),
            )
        };
        // The cache row pair-wise. `width` and `rope` are even and the
        // buffer is device-allocated, so every u32 read below is aligned.
        let kvw = kv.as_ptr().cast::<u32>();

        // Each head's causal limit. A row past `q_rows` gets limit 0 and a
        // zero query row: its logits are `−inf` at every key.
        if tid < MMA_ROWS {
            let row = base_row + tid;
            let l = if row < rows {
                causal_limit(n_keys_buf, dst_rows, row, n_heads, m)
            } else {
                0
            };
            // SAFETY: tid < MMA_ROWS bounds the store.
            unsafe {
                *lim_sh.add(tid) = l as u32;
            }
        }
        crate::mma_segment_walk! {
            width: MMA_WIDTH,
            q: q,
            kvw: kvw,
            key: key => key,
            scratch: (qs, kt, klog, kw, vms_sh, lim_sh),
            rows: rows,
            base_row: base_row,
            n_seg: n_seg,
            seg: seg,
            lo: lo,
            hi: hi_max,
            width_rt: width,
            rope: rope,
            lat: lat,
            scale: scale,
            tid: tid,
            part_v: part_v,
            part_ms: part_ms,
        }
    }

    /// The merge pass of the split launch: one block per query row, one
    /// thread per latent dim, folding the row's partials by the standard
    /// online-softmax rescale in ascending segment order — a fixed order, so
    /// reruns and graph replays are bit-identical. The scan stops at the last
    /// segment the row's causal limit reaches, which is why this kernel reads
    /// `n_keys_buf` too: the grid is the cache's, the work is the live key
    /// count's. A neutral partial (`Σ exp == 0`) is skipped as well, so a
    /// segment past the limit is never folded in and its `part_v` slice is
    /// never read.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2,
            y.len() >= q_rows * latent
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_merge(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        q_rows: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        part_v: &[f32],
        part_ms: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let lat = latent as usize;
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract), and the
        // partial reads are inside theirs.
        unsafe {
            let v = merge_row(
                n_keys_buf, m, n_heads, dst_rows, segs, seg_keys, part_v, part_ms, row, tid, lat,
            );
            *y.get_unchecked_mut(row * lat + tid) = v;
        }
    }

    /// [`flash_merge`] with the q8_1 side output: the same fold, the same
    /// `y`, and the merged row also leaves the block as the quantized
    /// activation the `wv_b` gemv reads — so the quantize launch that used
    /// to follow is deleted rather than merged.
    ///
    /// A block already holds a whole head's `latent` = 512 values, which is
    /// exactly four whole q8_1 blocks, so the scale reduction's partition is
    /// unchanged: warp `w` of the first four takes 128-value block `w` with
    /// lane ℓ holding values `4ℓ .. +3`, the same 32 lanes over the same 128
    /// values the standalone quantizer's warp had. The values go through
    /// shared memory because this kernel's threads hold one latent dim each
    /// and the quantizer's lanes want four consecutive — a transpose inside
    /// the block, not a different set of values.
    ///
    /// Rows below `m_lo` quantize into the `_a` outputs at column `row`, the
    /// rest into `_b` at column `row − m_lo`: the two halves of `kqvc`, one
    /// scratch each. The branch is on the block index, so no warp splits
    /// across it.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2,
            y.len() >= q_rows * latent,
            q3a.len() >= m_lo * 64 * half_it,
            q4a.len() >= m_lo * 256 * quad_it,
            q6a.len() >= m_lo * 128 * half_it,
            s8a.len() >= m_lo * 8 * n_sb,
            d8a.len() >= m_lo * 2 * n_sb,
            q3b.len() >= (q_rows - m_lo) * 64 * half_it,
            q4b.len() >= (q_rows - m_lo) * 256 * quad_it,
            q6b.len() >= (q_rows - m_lo) * 128 * half_it,
            s8b.len() >= (q_rows - m_lo) * 8 * n_sb,
            d8b.len() >= (q_rows - m_lo) * 2 * n_sb
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_merge_q8(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        q_rows: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        part_v: &[f32],
        part_ms: &[f32],
        mut y: DisjointSlice<f32>,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut VROW: SharedArray<f32, LATENT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let lat = latent as usize;
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment and the partial reads are
        // inside theirs (launch contract).
        let v = unsafe {
            let v = merge_row(
                n_keys_buf, m, n_heads, dst_rows, segs, seg_keys, part_v, part_ms, row, tid, lat,
            );
            *y.get_unchecked_mut(row * lat + tid) = v;
            v
        };
        // SAFETY: VROW is this block's own shared allocation; tid < 512 =
        // LATENT and every read below is bounded by the same length and
        // ordered by the `sync_threads` inside the helper.
        let vs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VROW) };
        // SAFETY: the launch contract bounds both output sets; `row` picks a
        // column inside its own half, and the helper's preconditions on
        // lane, block and geometry are the ones this kernel's block shape
        // gives it.
        unsafe {
            quant_row(
                v, vs, tid, lat, row, m_lo, n_sb, half_it, quad_it, &mut q3a, &mut q4a, &mut q6a,
                &mut s8a, &mut d8a, &mut q3b, &mut q4b, &mut q6b, &mut s8b, &mut d8b, fault,
            );
        }
    }

    /// One query row's fold over its segments' partials, the standard
    /// online-softmax rescale in ascending segment order — the shared body
    /// of [`flash_merge`] and [`flash_merge_q8`], so the two cannot drift.
    /// Returns this thread's final latent value. The partials are loaded
    /// sixteen segments at a time ahead of their folds; the folds and their
    /// order are the one-at-a-time walk's.
    ///
    /// SAFETY: `row < q_rows`, `tid < latent`, and the partial buffers hold
    /// `q_rows * segs * latent` / `q_rows * segs * 2` elements.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn merge_row(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        part_v: &[f32],
        part_ms: &[f32],
        row: usize,
        tid: usize,
        lat: usize,
    ) -> f32 {
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        let n_seg = limit.div_ceil(seg_keys as usize).min(segs as usize);
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = 0.0f32;
        let mut seg = 0usize;
        while seg < n_seg {
            // The batch's loads are issued ahead of its folds, so the walk
            // does not wait on memory twice per segment.
            let mut ms = [0.0f32; 2 * MERGE_BATCH];
            let mut vs = [0.0f32; MERGE_BATCH];
            let mut i = 0usize;
            while i < MERGE_BATCH {
                cuda_device::thread::__unroll_config::<0>();
                if seg + i < n_seg {
                    let idx = row * segs as usize + seg + i;
                    // SAFETY: idx < q_rows * segs, so both slots are inside
                    // part_ms, and idx * lat + tid < q_rows * segs * latent
                    // (caller's contract). A neutral segment's value slot is
                    // loaded and not folded.
                    unsafe {
                        ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                        ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                        vs[i] = *part_v.get_unchecked(idx * lat + tid);
                    }
                }
                i += 1;
            }
            let mut i = 0usize;
            while i < MERGE_BATCH {
                cuda_device::thread::__unroll_config::<0>();
                if seg + i < n_seg && ms[2 * i + 1] != 0.0 {
                    (mx, s_sum, acc) = online_fold(mx, s_sum, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
                }
                i += 1;
            }
            seg += MERGE_BATCH;
        }
        let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
        s_inv * acc
    }

    /// The q8_1 side output of a block that holds one whole column: thread
    /// `tid` contributes its value `v`, the block transposes through shared
    /// memory, and the first `lat / 128` warps each quantize one 128-value
    /// block through `q8_1_quant_vals` — the one owner of that body, and of
    /// its refusal: a block holding a non-finite value is stored with a NaN
    /// scale and zero codes, and raises [`FaultSite::QuantColumn`] on
    /// `fault`, as the quantize launch it replaces does.
    ///
    /// The transpose is what makes the fold bit-identical rather than merely
    /// close: the quantizer's warp sees the same 128 values in the same
    /// lanes as it would reading the stored column back, so its `amax` is
    /// the same reduction over the same set.
    ///
    /// SAFETY: `vs` is the calling block's own `LATENT`-element shared
    /// array, `tid < lat <= LATENT`, `lat` is a multiple of 128, `col` is a
    /// column of the half `row` selects, and the launch contract bounds both
    /// output sets.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn quant_row(
        v: f32,
        vs: *mut f32,
        tid: usize,
        lat: usize,
        row: usize,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        q3a: &mut DisjointSlice<u64>,
        q4a: &mut DisjointSlice<u32>,
        q6a: &mut DisjointSlice<u32>,
        s8a: &mut DisjointSlice<i32>,
        d8a: &mut DisjointSlice<f32>,
        q3b: &mut DisjointSlice<u64>,
        q4b: &mut DisjointSlice<u32>,
        q6b: &mut DisjointSlice<u32>,
        s8b: &mut DisjointSlice<i32>,
        d8b: &mut DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        // SAFETY: tid < lat <= LATENT, the shared array's length.
        unsafe {
            *vs.add(tid) = v;
        }
        thread::sync_threads();
        let blk = tid / 32; // warp index, and the 128-value block it takes
        if blk >= lat / 128 {
            return; // warp-uniform: the collectives below stay full-warp
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: 128*blk + 4*lane + 3 < 128*(blk+1) <= lat <= LATENT.
        let vals = unsafe {
            let base = vs.add(128 * blk + 4 * lane);
            [*base, *base.add(1), *base.add(2), *base.add(3)]
        };
        let n_sb = n_sb as usize;
        // The branch is on the block index, so each arm runs block-uniform.
        let refused = if row < m_lo as usize {
            // SAFETY: the values are block `blk`'s, held four per lane in
            // value order (`q8_1_quant_vals`'s precondition); `row < m_lo` is
            // inside the first half and the launch contract bounds its outputs.
            unsafe {
                q8_1_quant_vals(
                    vals, row, blk, n_sb, half_it, quad_it, lane, q3a, q4a, q6a, s8a, d8a,
                )
            }
        } else {
            // SAFETY: the values are block `blk`'s, held four per lane in
            // value order (`q8_1_quant_vals`'s precondition); `row - m_lo` is
            // inside the second half and the launch contract bounds its outputs.
            unsafe {
                q8_1_quant_vals(
                    vals,
                    row - m_lo as usize,
                    blk,
                    n_sb,
                    half_it,
                    quad_it,
                    lane,
                    q3b,
                    q4b,
                    q6b,
                    s8b,
                    d8b,
                )
            }
        };
        if refused && lane == 0 {
            fault.raise(FaultSite::QuantColumn);
        }
    }

    /// The causal key limit of query row `row`: the live key count clamped to
    /// the cache's allocated rows (a lying `n_keys_buf` then reads at most
    /// what is allocated), plus the row's own offset inside the `m`-token
    /// batch. Query `t` of `m` attends keys `0..n_keys − m + t + 1`.
    #[inline(always)]
    fn causal_limit(n_keys_buf: &[u32], dst_rows: u32, row: usize, n_heads: u32, m: u32) -> usize {
        // SAFETY: n_keys_buf.len() >= 1 by every caller's launch contract.
        let n_keys = (unsafe { *n_keys_buf.get_unchecked(0) } as usize).min(dst_rows as usize);
        let t = row / n_heads as usize;
        (n_keys + t + 1).saturating_sub(m as usize)
    }

    /// One block's walk of keys `lo..hi` of query row `row`, the module doc's
    /// reduction contract: the query row is staged into `qs`, then the online
    /// softmax runs over [`KEY_TILE`]-key tiles. Returns this thread's
    /// `(running max, Σ exp, Σ exp·V)` — the first two are the block's shared
    /// state and are valid in warp 0 (thread 0 among them), the third is this
    /// thread's own latent dim, un-normalised and relative to that max.
    ///
    /// `lo` is a multiple of [`KEY_TILE`] and `hi <= limit <= dst_rows`, so
    /// only the final tile can reach past `hi` and it takes the guarded path
    /// whose `wl == 0.0` test skips those loads.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn latent_range(
        q: &[f32],
        kv: &[u16],
        qs: *mut f32,
        klog: *mut f32,
        kw: *mut f32,
        st: *mut f32,
        row: usize,
        tid: usize,
        lo: usize,
        hi: usize,
        rope: usize,
        lat: usize,
        scale: f32,
    ) -> (f32, f32, f32) {
        let lane = warp::lane_id() as usize;
        let width = rope + lat;

        let mut s = tid;
        while s < width {
            // SAFETY: row is inside q's rows and s < width bound the load by
            // the caller's launch contract; s < width <= MAX_WIDTH
            // (host-validated) bounds the shared store.
            unsafe {
                *qs.add(s) = *q.get_unchecked(row * width + s);
            }
            s += LATENT;
        }

        // This thread's key inside a tile, and its slice of that key's dims.
        let key = tid / DIM_SPLIT;
        let dim0 = tid % DIM_SPLIT;

        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut r0 = 0.0f32;
        let mut r1 = 0.0f32;
        let mut r2 = 0.0f32;
        let mut r3 = 0.0f32;

        thread::sync_threads(); // the staged query row is visible

        let mut blk = lo;
        while blk < hi {
            // ---- QK dot: DIM_SPLIT threads per key, four rotating partials
            let mut acc = 0.0f32;
            if blk + key < hi {
                let kb = (blk + key) * width;
                let mut a0 = 0.0f32;
                let mut a1 = 0.0f32;
                let mut a2 = 0.0f32;
                let mut a3 = 0.0f32;
                let mut i = dim0;
                while i + (ILP - 1) * DIM_SPLIT < width {
                    // SAFETY: blk + key < hi <= dst_rows and every dim index
                    // below is < width by this loop's test, so each load is
                    // inside the key's row of kv (launch contract); the
                    // shared reads are below width <= MAX_WIDTH.
                    unsafe {
                        let k0 = half_bits_to_f32(*kv.get_unchecked(kb + i));
                        let k1 = half_bits_to_f32(*kv.get_unchecked(kb + i + DIM_SPLIT));
                        let k2 = half_bits_to_f32(*kv.get_unchecked(kb + i + 2 * DIM_SPLIT));
                        let k3 = half_bits_to_f32(*kv.get_unchecked(kb + i + 3 * DIM_SPLIT));
                        a0 = f32::mul_add(*qs.add(i), k0, a0);
                        a1 = f32::mul_add(*qs.add(i + DIM_SPLIT), k1, a1);
                        a2 = f32::mul_add(*qs.add(i + 2 * DIM_SPLIT), k2, a2);
                        a3 = f32::mul_add(*qs.add(i + 3 * DIM_SPLIT), k3, a3);
                    }
                    i += ILP * DIM_SPLIT;
                }
                while i < width {
                    // SAFETY: blk + key < hi <= dst_rows and i < width <=
                    // MAX_WIDTH, so the kv load is inside the key's row and
                    // the shared read inside the staged query row.
                    unsafe {
                        a0 = f32::mul_add(
                            *qs.add(i),
                            half_bits_to_f32(*kv.get_unchecked(kb + i)),
                            a0,
                        );
                    }
                    i += DIM_SPLIT;
                }
                acc = (a0 + a1) + (a2 + a3);
            }
            // The key's DIM_SPLIT threads are one aligned lane group, so the
            // four-step butterfly stays inside the key. Every thread calls
            // it: the guard above shapes the value, never the control flow.
            acc += warp::shuffle_xor_f32(acc, 1);
            acc += warp::shuffle_xor_f32(acc, 2);
            acc += warp::shuffle_xor_f32(acc, 4);
            acc += warp::shuffle_xor_f32(acc, 8);
            if dim0 == 0 {
                let sv = if blk + key < hi {
                    scale * acc
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: key < KEY_TILE, and one thread per key writes it.
                unsafe {
                    *klog.add(key) = sv;
                }
            }
            thread::sync_threads();

            // ---- online softmax: warp 0 owns (m, s) and publishes the tile
            if tid < KEY_TILE {
                // SAFETY: lane == tid < KEY_TILE here.
                let sv = unsafe { *klog.add(lane) };
                let smax = warp::reduce_max_f32(sv);
                // FlashMS update: s is rescaled here, the V partials just
                // before this tile's accumulation (the CPU oracle's order).
                // A first bump scales by 0.0 — the partials are still zero,
                // so the reset and the scale are the same value.
                let mut vms = 1.0f32;
                if smax > mx {
                    vms = if mx > f32::NEG_INFINITY {
                        dev_exp(mx - smax)
                    } else {
                        0.0
                    };
                    s_sum *= vms;
                    mx = smax;
                }
                let w = if sv == f32::NEG_INFINITY {
                    0.0
                } else {
                    dev_exp(sv - mx)
                };
                s_sum += warp::reduce_sum_f32(w);
                // SAFETY: lane < KEY_TILE; one lane writes each slot, and
                // lane 0 alone writes the rescale.
                unsafe {
                    *kw.add(lane) = w;
                    if lane == 0 {
                        *st.add(0) = vms;
                    }
                }
            }
            thread::sync_threads();

            // ---- V accumulation: this thread's own latent dim, four keys
            // in flight. SAFETY: ST[0] and KW hold this tile's published
            // values (both barriers above).
            let vms = unsafe { *st.add(0) };
            r0 *= vms;
            r1 *= vms;
            r2 *= vms;
            r3 *= vms;
            let tail = rope + tid;
            if blk + KEY_TILE <= hi {
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: the whole tile is below hi <= dst_rows, so all
                    // ILP rows' tails are inside kv (launch contract); tail <
                    // width and l + ILP <= KEY_TILE.
                    unsafe {
                        let base = (blk + l) * width + tail;
                        let v0 = half_bits_to_f32(*kv.get_unchecked(base));
                        let v1 = half_bits_to_f32(*kv.get_unchecked(base + width));
                        let v2 = half_bits_to_f32(*kv.get_unchecked(base + 2 * width));
                        let v3 = half_bits_to_f32(*kv.get_unchecked(base + 3 * width));
                        r0 = f32::mul_add(*kw.add(l), v0, r0);
                        r1 = f32::mul_add(*kw.add(l + 1), v1, r1);
                        r2 = f32::mul_add(*kw.add(l + 2), v2, r2);
                        r3 = f32::mul_add(*kw.add(l + 3), v3, r3);
                    }
                    l += ILP;
                }
            } else {
                // The run's last tile is the only one that reaches past the
                // limit: a zero weight is exactly a row this must not load.
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: l < KEY_TILE.
                    let wl = unsafe { *kw.add(l) };
                    if wl != 0.0 {
                        // SAFETY: wl != 0 => key blk+l is live => blk+l < hi
                        // <= dst_rows, so its tail is inside kv.
                        unsafe {
                            r0 = f32::mul_add(
                                wl,
                                half_bits_to_f32(*kv.get_unchecked((blk + l) * width + tail)),
                                r0,
                            );
                        }
                    }
                    l += 1;
                }
            }
            blk += KEY_TILE;
        }

        (mx, s_sum, (r0 + r1) + (r2 + r3))
    }
}

/// The loaded P5 device module: `kv_append`, `kv_append_pos_buf`,
/// `flash_latent`, the tensor-core segment pass `flash_latent_mma`,
/// `flash_merge` and the q8 twins. Owns no context and no stream — every
/// enqueue takes the engine stream (`Gpu::stream()`), so launches order with
/// the rest of the step and are capturable.
pub struct FlashKernels {
    module: flash_kernels::LoadedModule,
}

/// The query geometry every flash launch over the cache shares: `tokens`
/// query tokens (1..=8) of `heads` heads each, so `tokens * heads` query
/// rows of `rope_dims + latent_dims` f32, row `t * heads + h`.
#[derive(Clone, Copy, Debug)]
pub struct FlashGeom {
    pub tokens: usize,
    pub heads: usize,
    pub rope_dims: usize,
    pub latent_dims: usize,
}

/// What every attending launch reads: the query rows `q`, the `[rows x
/// rope_dims + latent_dims]` f16 cache `kv`, the device-side live key count
/// `n_keys_buf[0]`, and `kq_scale` (`MlaParams::kq_scale`).
#[derive(Clone, Copy)]
pub struct FlashInputs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kv: &'a DeviceTensor<u16>,
    pub n_keys_buf: &'a DeviceBuffer<u32>,
    pub kq_scale: f32,
    pub geom: FlashGeom,
}

/// [`FlashKernels::enqueue_flash_latent`]'s arguments.
pub struct FlashLatentArgs<'a> {
    pub inputs: FlashInputs<'a>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashKernels::enqueue_flash_latent_q8`]'s arguments: the side
/// output's refusal raises on `fault`.
pub struct FlashLatentQ8Args<'a> {
    pub inputs: FlashInputs<'a>,
    pub y: &'a mut DeviceBuffer<f32>,
    pub lo: &'a mut Q8Act,
    pub hi: &'a mut Q8Act,
    pub fault: FaultSink,
}

/// [`FlashKernels::enqueue_flash_latent_split`]'s arguments.
pub struct FlashSplitArgs<'a> {
    pub inputs: FlashInputs<'a>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The segment pass's arguments — [`FlashKernels::enqueue_flash_latent_mma`].
pub struct FlashSegArgs<'a> {
    pub inputs: FlashInputs<'a>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
}

/// [`FlashKernels::enqueue_flash_merge`]'s arguments: `cache_rows` is the
/// height the partials were written for.
pub struct FlashMergeArgs<'a> {
    pub n_keys_buf: &'a DeviceBuffer<u32>,
    pub cache_rows: usize,
    pub tokens: usize,
    pub heads: usize,
    pub latent_dims: usize,
    pub part_v: &'a DeviceBuffer<f32>,
    pub part_ms: &'a DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`FlashKernels::enqueue_flash_merge_q8`]'s arguments: the side output's
/// refusal raises on `fault`.
pub struct FlashMergeQ8Args<'a> {
    pub merge: FlashMergeArgs<'a>,
    pub lo: &'a mut Q8Act,
    pub hi: &'a mut Q8Act,
    pub fault: FaultSink,
}

impl FlashKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { flash_kernels::load(ctx)? };
        Ok(FlashKernels { module })
    }

    /// Enqueue the f32 -> f16 append of `m` new rows of `cache.cols()` f32
    /// (`src.len() >= m * width`, token-major) at rows `pos..pos + m` of the
    /// `[cache.rows() x width]` u16 cache. `pos` is a launch scalar and is
    /// frozen inside a captured graph — for the captured decode step use
    /// [`FlashKernels::enqueue_kv_append_pos_buf`]. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_kv_append(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        check_append("enqueue_kv_append", src.len(), width, m)?;
        if pos as usize + m > rows {
            return Err(GpuError::shape(
                "enqueue_kv_append",
                format!(
                    "rows {}..{} land past the cache's {rows} rows",
                    pos,
                    pos as usize + m
                ),
            ));
        }
        let what = "enqueue_kv_append";
        let grid = launch_u32(what, "grid", (m * width).div_ceil(256))?;
        let m = launch_u32(what, "m", m)?;
        let width = launch_u32(what, "width", width)?;
        let prep = self
            .module
            .prepare_kv_append(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .kv_append(stream, &prep, m, width, pos, src, cache.buf_mut())?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_kv_append`] with `pos` read from
    /// `pos_buf[0]` on the device at run time — the captured-graph form: the
    /// graph replays unchanged while the position advances through the
    /// buffer. A row past the cache's height is refused in-kernel: stored
    /// nowhere, it raises [`FaultSite::CachePos`] on `fault`. `pos_buf`
    /// must stay allocated and in place for as long as a captured graph
    /// replaying this launch lives. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_kv_append_pos_buf(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        pos_buf: &DeviceBuffer<u32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        if pos_buf.is_empty() {
            return Err(GpuError::shape(
                "enqueue_kv_append_pos_buf",
                "pos_buf must hold 1 u32",
            ));
        }
        check_append("enqueue_kv_append_pos_buf", src.len(), width, m)?;
        let what = "enqueue_kv_append_pos_buf";
        let grid = launch_u32(what, "grid", (m * width).div_ceil(256))?;
        let m = launch_u32(what, "m", m)?;
        let width = launch_u32(what, "width", width)?;
        let rows = launch_u32(what, "rows", rows)?;
        let prep = self
            .module
            .prepare_kv_append_pos_buf(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.kv_append_pos_buf(
            stream,
            &prep,
            m,
            width,
            rows,
            pos_buf,
            src,
            cache.buf_mut(),
            fault,
        )?;
        Ok(())
    }

    /// Enqueue the latent flash attention: `q` holds `tokens * heads` rows
    /// of `rope_dims + latent_dims` f32 (row `t * heads + h`, content
    /// `[q_rope | q_nope2]`), `kv` is the `[rows x width]` u16 cache, and
    /// `n_keys_buf[0]` names the live key count (`>= tokens`: the batch's
    /// own rows are appended first) — query `t` attends to keys
    /// `0..n_keys − tokens + t`. `y` holds `tokens * heads` rows of
    /// `latent_dims` f32, the `kqv_compressed` order. `kq_scale` is
    /// `MlaParams::kq_scale` (the YaRN mscale is inside it — not `1/√d`).
    /// `latent_dims` must be [`LATENT`], this family's block width;
    /// `rope_dims` is free as long as `rope_dims + latent_dims` is a
    /// multiple of [`DIM_SPLIT`] and at most [`MAX_WIDTH`] (the shared
    /// staging row). Asynchronous, allocation-free, capturable.
    pub fn enqueue_flash_latent(
        &self,
        stream: &CudaStream,
        a: FlashLatentArgs<'_>,
    ) -> Result<(), GpuError> {
        let FlashLatentArgs { inputs, y } = a;
        let FlashInputs {
            q,
            kv,
            n_keys_buf,
            kq_scale: scale,
            geom,
        } = inputs;
        let FlashGeom {
            tokens: m,
            heads: n_heads,
            rope_dims,
            latent_dims: latent,
        } = geom;
        let what = "enqueue_flash_latent";
        let q_rows = check_flash(what, q.len(), kv, n_keys_buf.len(), geom, Some(y.len()))?;
        let m = launch_u32(what, "tokens", m)?;
        let n_heads = launch_u32(what, "heads", n_heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let rope_dims = launch_u32(what, "rope_dims", rope_dims)?;
        let latent = launch_u32(what, "latent_dims", latent)?;
        let dst_rows = launch_u32(what, "kv.rows()", kv.rows())?;
        let prep = self
            .module
            .prepare_flash_latent(LaunchConfig1D::new(q_rows, LATENT_U32, 0))?;
        self.module.flash_latent(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m,
            n_heads,
            q_rows,
            rope_dims,
            latent,
            dst_rows,
            y,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_latent`] with the key range spread over
    /// [`segments_for`]`(kv.rows())` segments: the tensor-core segment pass
    /// ([`FlashKernels::enqueue_flash_latent_mma`]) writes partials into
    /// `part_v`/`part_ms` and the merge pass folds them into `y`. Returns the
    /// number of launches enqueued — one when the cache is short enough to
    /// hold a single segment (the single-block kernel serves it, no partials
    /// touched), two otherwise. That choice comes from the cache height alone,
    /// so it is fixed for the life of a captured graph, and the grid does not
    /// move with `n_keys_buf`. The two-launch path takes only the segment
    /// pass's [`MMA_WIDTH`] row.
    ///
    /// `part_v` must hold [`partials_v_len`] and `part_ms`
    /// [`partials_ms_len`] for the same `tokens * heads` rows and cache
    /// height.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_flash_latent_split(
        &self,
        stream: &CudaStream,
        a: FlashSplitArgs<'_>,
    ) -> Result<usize, GpuError> {
        let FlashSplitArgs {
            inputs,
            part_v,
            part_ms,
            y,
        } = a;
        if segments_for(inputs.kv.rows()) == 1 {
            self.enqueue_flash_latent(stream, FlashLatentArgs { inputs, y })?;
            return Ok(1);
        }
        self.enqueue_flash_latent_mma(
            stream,
            FlashSegArgs {
                inputs,
                part_v: &mut *part_v,
                part_ms: &mut *part_ms,
            },
        )?;
        self.enqueue_flash_merge(
            stream,
            FlashMergeArgs {
                n_keys_buf: inputs.n_keys_buf,
                cache_rows: inputs.kv.rows(),
                tokens: inputs.geom.tokens,
                heads: inputs.geom.heads,
                latent_dims: inputs.geom.latent_dims,
                part_v,
                part_ms,
                y,
            },
        )?;
        Ok(2)
    }

    /// The tensor-core segment pass alone — the first of the two launches
    /// [`FlashKernels::enqueue_flash_latent_split`] makes: a grid of
    /// [`mma_groups`]`(q_rows) * `[`segments_for`]`(kv.rows())` blocks of
    /// [`MMA_BLOCK`] threads, a block carrying [`MMA_ROWS`] query rows and
    /// putting their `Q·Kᵀ` on `mma.sync`, each writing its rows' partials
    /// into `part_v`/`part_ms`. A caller that wants the two launches timed or
    /// observed separately enqueues this and [`FlashKernels::enqueue_flash_merge`]
    /// (or its q8 twin) itself, after asking [`segments_for`] whether the
    /// cache is tall enough to need them at all.
    ///
    /// The kernel's shared tiles are sized for a [`MMA_WIDTH`] row and its
    /// `k` walk assumes the width divides by `2 * `[`MMA_K`], so any other
    /// width is refused here rather than read past a tile.
    pub fn enqueue_flash_latent_mma(
        &self,
        stream: &CudaStream,
        a: FlashSegArgs<'_>,
    ) -> Result<(), GpuError> {
        let FlashSegArgs {
            inputs,
            part_v,
            part_ms,
        } = a;
        let FlashInputs {
            q,
            kv,
            n_keys_buf,
            kq_scale: scale,
            geom,
        } = inputs;
        let FlashGeom {
            tokens: m,
            heads: n_heads,
            rope_dims,
            latent_dims: latent,
        } = geom;
        let what = "enqueue_flash_latent_mma";
        let (q_rows, segs) = check_seg(
            what,
            q.len(),
            kv,
            n_keys_buf.len(),
            geom,
            (part_v.len(), part_ms.len()),
        )?;
        if rope_dims + latent != MMA_WIDTH {
            return Err(GpuError::shape(
                "enqueue_flash_latent_mma",
                format!(
                    "rope_dims + latent = {} — this kernel's staged \
                 tiles are sized for {MMA_WIDTH}",
                    rope_dims + latent
                ),
            ));
        }
        if !seg_keys().is_multiple_of(MMA_KEYS) {
            return Err(GpuError::shape(
                "enqueue_flash_latent_mma",
                format!(
                    "seg_keys {} is not a multiple of the {MMA_KEYS}-key \
                 tile",
                    seg_keys()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", mma_groups(q_rows) * segs)?;
        let m = launch_u32(what, "tokens", m)?;
        let n_heads = launch_u32(what, "heads", n_heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let rope_dims = launch_u32(what, "rope_dims", rope_dims)?;
        let latent = launch_u32(what, "latent_dims", latent)?;
        let dst_rows = launch_u32(what, "kv.rows()", kv.rows())?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", seg_keys())?;
        let prep = self.module.prepare_flash_latent_mma(LaunchConfig1D::new(
            grid,
            MMA_BLOCK_U32,
            MMA_DYN_BYTES_U32,
        ))?;
        self.module.flash_latent_mma(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m,
            n_heads,
            q_rows,
            rope_dims,
            latent,
            dst_rows,
            segs,
            seg_keys,
            part_v,
            part_ms,
        )?;
        Ok(())
    }

    /// The merge pass alone — the second of the two launches. `cache_rows`
    /// is the height the partials were written for, so the segment count
    /// matches the segment pass's grid exactly.
    pub fn enqueue_flash_merge(
        &self,
        stream: &CudaStream,
        a: FlashMergeArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_flash_merge";
        let (q_rows, segs) = check_merge(what, &a)?;
        let FlashMergeArgs {
            n_keys_buf,
            cache_rows,
            tokens: m,
            heads: n_heads,
            latent_dims: latent,
            part_v,
            part_ms,
            y,
        } = a;
        let m = launch_u32(what, "tokens", m)?;
        let n_heads = launch_u32(what, "heads", n_heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let latent = launch_u32(what, "latent_dims", latent)?;
        let cache_rows = launch_u32(what, "cache_rows", cache_rows)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", seg_keys())?;
        let prep = self
            .module
            .prepare_flash_merge(LaunchConfig1D::new(q_rows, LATENT_U32, 0))?;
        self.module.flash_merge(
            stream, &prep, n_keys_buf, m, n_heads, q_rows, latent, cache_rows, segs, seg_keys,
            part_v, part_ms, y,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_merge`] that also writes the q8_1 form
    /// of the merged rows into `lo` and `hi`: the rows below `lo.m()` fill
    /// `lo`'s columns in order, the rest `hi`'s. The bytes are the ones the
    /// q8_1 quantizer (`q3k_quantize_q8_1`) writes for each half's columns
    /// of `y`, so no quantize launch follows the merge; a refused block
    /// raises [`FaultSite::QuantColumn`] on `a.fault`.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_flash_merge_q8(
        &self,
        stream: &CudaStream,
        a: FlashMergeQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let FlashMergeQ8Args {
            merge,
            lo,
            hi,
            fault,
        } = a;
        let what = "enqueue_flash_merge_q8";
        let (q_rows, segs) = check_merge(what, &merge)?;
        let FlashMergeArgs {
            n_keys_buf,
            cache_rows,
            tokens: m,
            heads: n_heads,
            latent_dims: latent,
            part_v,
            part_ms,
            y,
        } = merge;
        let (m_lo, n_sb) = check_side_quant(what, lo, hi, q_rows, latent)?;
        let m = launch_u32(what, "tokens", m)?;
        let n_heads = launch_u32(what, "heads", n_heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let latent = launch_u32(what, "latent_dims", latent)?;
        let cache_rows = launch_u32(what, "cache_rows", cache_rows)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", seg_keys())?;
        let m_lo = launch_u32(what, "lo.m()", m_lo)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_flash_merge_q8(LaunchConfig1D::new(q_rows, LATENT_U32, 0))?;
        self.module.flash_merge_q8(
            stream,
            &prep,
            n_keys_buf,
            m,
            n_heads,
            q_rows,
            latent,
            cache_rows,
            segs,
            seg_keys,
            m_lo,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            part_v,
            part_ms,
            y,
            &mut lo.q3,
            &mut lo.q4,
            &mut lo.q6,
            &mut lo.s8,
            &mut lo.d8,
            &mut hi.q3,
            &mut hi.q4,
            &mut hi.q6,
            &mut hi.s8,
            &mut hi.d8,
            fault,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_latent`] with the same q8_1 side output
    /// as [`FlashKernels::enqueue_flash_merge_q8`] — the single-segment path,
    /// so both paths of the folded step quantize, and refuse alike.
    pub fn enqueue_flash_latent_q8(
        &self,
        stream: &CudaStream,
        a: FlashLatentQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let FlashLatentQ8Args {
            inputs,
            y,
            lo,
            hi,
            fault,
        } = a;
        let FlashInputs {
            q,
            kv,
            n_keys_buf,
            kq_scale: scale,
            geom,
        } = inputs;
        let FlashGeom {
            tokens: m,
            heads: n_heads,
            rope_dims,
            latent_dims: latent,
        } = geom;
        let what = "enqueue_flash_latent_q8";
        let q_rows = check_flash(what, q.len(), kv, n_keys_buf.len(), geom, Some(y.len()))?;
        let (m_lo, n_sb) = check_side_quant(what, lo, hi, q_rows, latent)?;
        let m = launch_u32(what, "tokens", m)?;
        let n_heads = launch_u32(what, "heads", n_heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let rope_dims = launch_u32(what, "rope_dims", rope_dims)?;
        let latent = launch_u32(what, "latent_dims", latent)?;
        let dst_rows = launch_u32(what, "kv.rows()", kv.rows())?;
        let m_lo = launch_u32(what, "lo.m()", m_lo)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_flash_latent_q8(LaunchConfig1D::new(q_rows, LATENT_U32, 0))?;
        self.module.flash_latent_q8(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m,
            n_heads,
            q_rows,
            rope_dims,
            latent,
            dst_rows,
            m_lo,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            y,
            &mut lo.q3,
            &mut lo.q4,
            &mut lo.q6,
            &mut lo.s8,
            &mut lo.d8,
            &mut hi.q3,
            &mut hi.q4,
            &mut hi.q6,
            &mut hi.s8,
            &mut hi.d8,
            fault,
        )?;
        Ok(())
    }
}

/// The side output's shape check, shared by the two q8 twins: the two
/// scratches must together cover the launch's query rows one column each,
/// at the latent width, and `lat / 128` must be the block count per column
/// a 512-thread block can quantize in one pass. Returns `(lo.m(), n_sb)`.
fn check_side_quant(
    what: &'static str,
    lo: &Q8Act,
    hi: &Q8Act,
    q_rows: usize,
    latent: usize,
) -> Result<(usize, usize), GpuError> {
    if lo.k() != latent || hi.k() != latent {
        return Err(GpuError::shape(
            what,
            format!(
                "the side scratches' k must be the latent width {latent}, got {} and {}",
                lo.k(),
                hi.k()
            ),
        ));
    }
    if lo.m() + hi.m() != q_rows {
        return Err(GpuError::shape(
            what,
            format!(
                "the side scratches hold {} + {} columns, the launch has {q_rows} query rows",
                lo.m(),
                hi.m()
            ),
        ));
    }
    Ok((lo.m(), lo.n_sb()))
}

/// The partials both split launches share.
fn check_partials(
    what: &'static str,
    v_len: usize,
    ms_len: usize,
    q_rows: usize,
    segs: usize,
    latent: usize,
) -> Result<(), GpuError> {
    let (want_v, want_ms) = (q_rows * segs * latent, q_rows * segs * 2);
    if v_len < want_v || ms_len < want_ms {
        return Err(GpuError::shape(
            what,
            format!(
                "partials are {v_len}/{ms_len} f32, want {want_v}/{want_ms} for \
             {q_rows} rows x {segs} segments"
            ),
        ));
    }
    Ok(())
}

/// The merge pass's inputs, shared by its two entries: the live-count
/// buffer, the latent tail, the partials for `tokens * heads` rows over
/// [`segments_for`]`(cache_rows)` segments, and `y`. Returns the query rows
/// and the segment count.
fn check_merge(what: &'static str, a: &FlashMergeArgs<'_>) -> Result<(usize, usize), GpuError> {
    let latent = a.latent_dims;
    let q_rows = a.tokens * a.heads;
    let segs = segments_for(a.cache_rows);
    let y_len = a.y.len();
    if a.n_keys_buf.is_empty() {
        return Err(GpuError::shape(what, "n_keys_buf must hold 1 u32"));
    }
    if latent != LATENT {
        return Err(GpuError::shape(
            what,
            format!("this family's latent tail is {LATENT}, got {latent}"),
        ));
    }
    check_partials(what, a.part_v.len(), a.part_ms.len(), q_rows, segs, latent)?;
    if y_len < q_rows * latent {
        return Err(GpuError::shape(
            what,
            format!("y.len() {y_len} < m*n_heads*latent = {}", q_rows * latent),
        ));
    }
    Ok((q_rows, segs))
}

/// The segment pass's inputs: [`check_flash`] without a `y`, then the
/// partials at [`segments_for`]`(kv.rows())`.
/// `parts` is `(part_v.len(), part_ms.len())`. Returns the query rows and
/// the segment count, the launch grid's two factors.
fn check_seg(
    what: &'static str,
    q_len: usize,
    kv: &DeviceTensor<u16>,
    n_keys_len: usize,
    geom: FlashGeom,
    parts: (usize, usize),
) -> Result<(usize, usize), GpuError> {
    let q_rows = check_flash(what, q_len, kv, n_keys_len, geom, None)?;
    let segs = segments_for(kv.rows());
    check_partials(what, parts.0, parts.1, q_rows, segs, geom.latent_dims)?;
    Ok((q_rows, segs))
}

/// The geometry every flash entry shares: the latent tail is [`LATENT`], the
/// query row fits the single-block kernel's shared staging row and its QK dot
/// splits a row across [`DIM_SPLIT`] threads; `y_len` is `None` for the
/// segment pass, which writes partials instead. Returns `tokens * heads`, the
/// query rows.
fn check_flash(
    what: &'static str,
    q_len: usize,
    kv: &DeviceTensor<u16>,
    n_keys_len: usize,
    geom: FlashGeom,
    y_len: Option<usize>,
) -> Result<usize, GpuError> {
    let FlashGeom {
        tokens: m,
        heads: n_heads,
        rope_dims,
        latent_dims: latent,
    } = geom;
    if !(1..=8).contains(&m) {
        return Err(GpuError::shape(what, format!("1 <= m <= 8, got {m}")));
    }
    if n_heads == 0 {
        return Err(GpuError::shape(what, "n_heads >= 1"));
    }
    if latent != LATENT {
        return Err(GpuError::shape(
            what,
            format!(
                "this family's latent tail is {LATENT} (one block thread per dim), \
             got {latent}"
            ),
        ));
    }
    let width = rope_dims + latent;
    if !width.is_multiple_of(DIM_SPLIT) {
        return Err(GpuError::shape(
            what,
            format!(
                "the QK dot splits a row across {DIM_SPLIT} threads, need rope_dims + \
             latent = {width} a multiple of {DIM_SPLIT}"
            ),
        ));
    }
    if width > MAX_WIDTH {
        return Err(GpuError::shape(
            what,
            format!(
                "the query row is staged in {MAX_WIDTH} shared f32, got rope_dims + \
             latent = {width}"
            ),
        ));
    }
    if kv.cols() != width {
        return Err(GpuError::shape(
            what,
            format!(
                "kv is {}-wide, want rope_dims + latent = {width}",
                kv.cols()
            ),
        ));
    }
    if n_keys_len == 0 {
        return Err(GpuError::shape(what, "n_keys_buf must hold 1 u32"));
    }
    let q_rows = m * n_heads;
    if q_len < q_rows * width {
        return Err(GpuError::shape(
            what,
            format!("q.len() {q_len} < m*n_heads*width = {}", q_rows * width),
        ));
    }
    // The segment pass has no `y` of its own — it writes partials.
    if let Some(y_len) = y_len
        && y_len < q_rows * latent
    {
        return Err(GpuError::shape(
            what,
            format!("y.len() {y_len} < m*n_heads*latent = {}", q_rows * latent),
        ));
    }
    Ok(q_rows)
}

/// Reject geometry the append kernels' launch contracts do not cover:
/// positive width, `m` rows, and `src` holding all of them. The landing-row
/// bound is checked only where `pos` is a host scalar.
fn check_append(
    what: &'static str,
    src_len: usize,
    width: usize,
    m: usize,
) -> Result<(), GpuError> {
    if width == 0 || m == 0 {
        return Err(GpuError::shape(
            what,
            format!("need width >= 1 and m >= 1, got {width}/{m}"),
        ));
    }
    if src_len < m * width {
        return Err(GpuError::shape(
            what,
            format!("src.len() {src_len} < m*width = {}", m * width),
        ));
    }
    Ok(())
}
