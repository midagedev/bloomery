//! MiniMax-M3's sparse attention (MSA) on the card: the indexer's head kernel and key cache, the
//! block-max selector, and the flash over the blocks it picks.
//!
//! A sparse layer projects four index query heads and one index key per token from the layer's
//! normed input (the caller's gemvs). After them the chain is five launches:
//! - [`enqueue_index`] — the common head kernel (`rope_neox::head_norm_neox_append_128_partial`) at
//!   4 heads over 1 key head: each index head normed by `indexer.q_norm`, the key by `indexer.k_norm`,
//!   the first [`ROT`] values of both turned by the main attention's rope table, the key appended to
//!   the layer's f16 index-key plane. The head kernel appends a value row too; the index has none, so
//!   it goes to a plane nothing reads.
//! - [`MsaKernels::enqueue_score`] — [`msa_kernels::msa_score`]: block `(t, b)` scores row `t`'s
//!   block `b` for every index head and writes the block's largest score.
//! - [`MsaKernels::enqueue_pick`] — [`msa_kernels::msa_select`]: one block per row, one warp per
//!   index head, the `top_k` best blocks of the head ascending, and the row's key count.
//! - [`enqueue_attention`] — the common selected flash (`flash_gqa::enqueue_pass_p16_blocks`) over
//!   each key head's own blocks.
//!
//! The rule is `runtime::msa`'s, mirrored here. A row of live count `c` sees `⌈c / 128⌉` blocks. A
//! row that sees at most `top_k` of them attends all of them and computes no score; any other row
//! keeps, for each index head, the `top_k` best blocks, the `local` blocks nearest the row forced by
//! a bias of [`runtime::msa::FORCE`], a tie to the lower block. Scores are unscaled.
//!
//! Numeric contract (reruns are bit-identical):
//! - a score: the head's query values rounded once to f16 (the index cache's own format), lane `j`'s
//!   dot with key `j` over [`ILP`] rotating f32 partials
//!   (partial `p` takes the value pairs `p, p + ILP, …`, each value one fused multiply-add), combined
//!   `(a0 + a1) + (a2 + a3)` — every product is exact in f32, so the order is the whole contract;
//! - a block's score: the largest of its visible positions' scores by [`order_key`], returned as the
//!   value that key stands for (`−0.0` and `+0.0` are one key, read back `+0.0`);
//! - a selection: the key of `score + bias` (an f32 add), rounds of "the best key below the previous
//!   round's", a tie to the lower block.
//!
//! No silent failure: a score that is not finite raises [`FaultSite::PoolSelect`] (the value is
//! written as computed), a row whose own block the selection lost raises it too and gets length 0,
//! and a count of 0 or past the cache gets length 0, which the flash's merge refuses by name.

use bloomery_gpu::fault::{FaultSink, FaultSite};
use bloomery_gpu::flash::{f32_to_f16_bits, half_bits_to_f32};
use bloomery_gpu::flash_gqa::{
    FlashGqaKernels, GqaBlocksArgs, LISTED_BLOCK, PACK_16, listed_partials_ms_len,
    listed_partials_v_len,
};
use bloomery_gpu::qsa::order_key;
use bloomery_gpu::rope_neox::{PartialNeoxArgs, ROT_PARTIAL, RopeNeoxKernels};
use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use model::arch::minimaxm3::hparams::{Hparams, Msa as HparamsMsa};
use runtime::msa::{FORCE, Msa};
use std::sync::Arc;

/// Index heads (`attention.indexer.head_count`), one for each key head.
pub const HEADS: usize = 4;
/// Values of an index head and of an index key (`attention.indexer.key_length`).
pub const DIM: usize = 128;
/// Positions per block (`attention.indexer.block_size`): the flash's block.
pub const BLOCK: usize = LISTED_BLOCK;
/// Values at the head of a key or query that turn (`rope.dimension_count`).
pub const ROT: usize = ROT_PARTIAL;
/// Query heads of the attention: [`PACK_16`] for each key head.
pub const N_HEAD: usize = HEADS * PACK_16;
/// The most blocks a row keeps: a warp holds one kept block a lane.
pub const MAX_TOP_K: usize = 32;

/// Keys one score tile holds: one a lane.
const TILE: usize = 32;
/// Tiles of a block.
const TILES: usize = BLOCK / TILE;
/// Threads of a score or select block: one warp an index head.
const THREADS: u32 = 128;
const WARPS: usize = THREADS as usize / 32;
/// u32 words of a key row, and the staged tile's stride: lane `j` reads word `w` of key `j` at
/// `j·65 + w`, a distinct bank per lane.
const ROW_WORDS: usize = DIM / 2;
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of a key row.
const ROW_QWORDS: usize = DIM / 4;
/// Rotating partials of the score dot.
const ILP: usize = 4;
/// u64 words of the key tile one score thread stages.
const STAGE_PER: usize = TILE * ROW_QWORDS / THREADS as usize;

const _: () = assert!(HEADS == 4 && DIM == 128 && BLOCK == 128 && ROT == 64 && N_HEAD == 64);
const _: () = assert!(WARPS == HEADS && TILES * TILE == BLOCK && ROW_WORDS.is_multiple_of(ILP));
const _: () = assert!(STAGE_PER * THREADS as usize == TILE * ROW_QWORDS);

// ------------------------------------------------------------------ cores

/// The value an [`order_key`] stands for: `−0.0` was made `+0.0` first, so it reads back `+0.0`.
#[inline(always)]
#[must_use]
pub fn order_value(key: u32) -> f32 {
    if key & 0x8000_0000 != 0 {
        f32::from_bits(key ^ 0x8000_0000)
    } else {
        f32::from_bits(!key)
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod msa_kernels {
    use super::*;

    /// The score pass: block `b` serves row `t = b % m` and block `blk = b / m` of its positions. A
    /// row whose live count `c` is accepted (`1..=ctx`) and sees more than `top_k` blocks scores
    /// block `blk` when the row sees it: warp `w` is index head `w`, it rounds the head's query
    /// `iq[(t·4 + w)·128 ..]` to f16, then walks the block's visible positions in tiles of 32 keys,
    /// lane `j` the dot of the head with key `j` (module doc), and the lanes' largest score by
    /// [`order_key`] is written to `scores[(t·4 + w)·nblk + blk]`. Any other block writes nothing. A
    /// score that is not finite raises [`FaultSite::PoolSelect`] on `fault`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys.len() >= m,
            iq.len() >= m * 512,
            ik.len() >= ctx * 128,
            nblk * 128 >= ctx,
            scores.len() >= m * 4 * nblk
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub fn msa_score(
        iq: &[f32],
        ik: &[u16],
        n_keys: &[u32],
        ctx: u32,
        nblk: u32,
        top_k: u32,
        m: u32,
        fault: FaultSink,
        mut scores: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { HEADS * DIM }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { TILE * K_STRIDE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let rows = m as usize;
        let t = b % rows;
        let blk = b / rows;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) } as usize;
        // Block-uniform: c and blk are the block's.
        if c == 0 || c > ctx as usize || blk >= nblk as usize {
            return;
        }
        let visible = c.div_ceil(BLOCK);
        if visible <= top_k as usize || blk >= visible {
            return;
        }
        // The block's positions below the row's count.
        let n_valid = (c - blk * BLOCK).min(BLOCK);
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        // SAFETY: each `static mut` above is this block's own shared allocation; the raw form
        // reaches it without a reference. Every write is ordered before its reads by a barrier
        // (the tile) or by the warp's own sync (its head's query).
        let (qs, ks) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
            )
        };

        // ---- the query head: warp w rounds head w of row t, lane l values l, l + 32, ….
        let qb = (t * HEADS + w) * DIM;
        for i in 0..DIM / 32 {
            thread::__unroll_config::<0>();
            let d = lane + 32 * i;
            // SAFETY: qb + d < (t·4 + w + 1)·128 <= m·512 <= iq.len(); w·128 + d < HEADS·DIM.
            unsafe {
                let v = *iq.get_unchecked(qb + d);
                *qs.add(w * DIM + d) = half_bits_to_f32(f32_to_f16_bits(v));
            }
        }
        warp::sync_mask(u32::MAX);

        let ik64 = ik.as_ptr() as *const u64;
        let mut best = 0u32;
        let mut tile = 0usize;
        // Block-uniform loop: n_valid is the block's.
        while tile * TILE < n_valid {
            thread::sync_threads();
            for i in 0..STAGE_PER {
                thread::__unroll_config::<0>();
                let e = tid + THREADS as usize * i;
                let key = e / ROW_QWORDS;
                let word = e - key * ROW_QWORDS;
                let kw = if tile * TILE + key < n_valid {
                    // SAFETY: the position blk·128 + tile·32 + key < blk·128 + n_valid <= c <= ctx,
                    // so the row is inside ik's ctx·128 u16 = ctx·32 u64; the buffer starts
                    // 8-byte aligned (a device allocation).
                    unsafe { *ik64.add((blk * BLOCK + tile * TILE + key) * ROW_QWORDS + word) }
                } else {
                    0u64
                };
                // SAFETY: key < TILE and 2·word + 1 < K_STRIDE.
                unsafe {
                    *ks.add(key * K_STRIDE + 2 * word) = kw as u32;
                    *ks.add(key * K_STRIDE + 2 * word + 1) = (kw >> 32) as u32;
                }
            }
            thread::sync_threads();

            let mut a = [0.0f32; ILP];
            let mut wd = 0usize;
            while wd < ROW_WORDS {
                for p in 0..ILP {
                    thread::__unroll_config::<0>();
                    // SAFETY: lane < TILE and wd + p < ROW_WORDS: inside KS; w·128 + 2(wd + p) + 1
                    // < HEADS·DIM: inside QS.
                    let (kw, q0, q1) = unsafe {
                        (
                            *ks.add(lane * K_STRIDE + wd + p),
                            *qs.add(w * DIM + 2 * (wd + p)),
                            *qs.add(w * DIM + 2 * (wd + p) + 1),
                        )
                    };
                    a[p] = f32::mul_add(q0, half_bits_to_f32(kw as u16), a[p]);
                    a[p] = f32::mul_add(q1, half_bits_to_f32((kw >> 16) as u16), a[p]);
                }
                wd += ILP;
            }
            let dot = (a[0] + a[1]) + (a[2] + a[3]);
            if tile * TILE + lane < n_valid {
                if !dot.is_finite() {
                    fault.raise(FaultSite::PoolSelect);
                }
                best = best.max(order_key(dot));
            }
            tile += 1;
        }

        let mut off = 16u32;
        while off > 0 {
            best = best.max(warp::shuffle_xor(best, off));
            off >>= 1;
        }
        if lane == 0 {
            // SAFETY: (t·4 + w)·nblk + blk < m·4·nblk <= scores.len() (blk < nblk); one lane
            // writes each score.
            unsafe {
                *scores.get_unchecked_mut((t * HEADS + w) * nblk as usize + blk) =
                    order_value(best);
            }
        }
    }

    /// The select pass: block `t` serves row `t`, warp `w` index head `w`. A row whose live count
    /// `c` is accepted writes, for each head, its kept blocks ascending to
    /// `ids[(t·4 + w)·top_k ..]` and its key count to `n_sel[t]`: every visible block when the row
    /// sees at most `top_k`, else the `top_k` best of `scores` after the bias (module doc), by
    /// `top_k` rounds of the best key below the previous round's, then ranked by id. A count of 0 or
    /// past `ctx` gets `n_sel[t] = 0`. A head that lost the row's own block raises
    /// [`FaultSite::PoolSelect`] and the row gets length 0.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys.len() >= m,
            nblk * 128 >= ctx,
            scores.len() >= m * 4 * nblk,
            top_k >= 1,
            top_k <= 32,
            local >= 1,
            local <= top_k,
            ids.len() >= m * 4 * top_k,
            n_sel.len() >= m
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn msa_select(
        n_keys: &[u32],
        scores: &[f32],
        ctx: u32,
        nblk: u32,
        top_k: u32,
        local: u32,
        m: u32,
        fault: FaultSink,
        mut ids: DisjointSlice<u32>,
        mut n_sel: DisjointSlice<u32>,
    ) {
        static mut KEPT_OWN: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) } as usize;
        if c == 0 || c > ctx as usize {
            if tid == 0 {
                // SAFETY: t < m <= n_sel.len(); thread 0 alone writes it.
                unsafe { *n_sel.get_unchecked_mut(t) = 0 };
            }
            return; // block-uniform: c is the row's
        }
        let k = top_k as usize;
        let visible = c.div_ceil(BLOCK);
        let row = (t * HEADS + w) * k;
        if visible <= k {
            if lane < visible {
                // SAFETY: row + lane < (t·4 + w + 1)·top_k <= ids.len(); one lane writes each.
                unsafe { *ids.get_unchecked_mut(row + lane) = lane as u32 };
            }
            if tid == 0 {
                // SAFETY: t < n_sel.len(); thread 0 alone writes it.
                unsafe { *n_sel.get_unchecked_mut(t) = c as u32 };
            }
            return; // block-uniform: visible is the row's
        }
        let own = visible - 1;
        let sc = (t * HEADS + w) * nblk as usize;
        let mut mine = 0u32;
        let mut prev = 0u64;
        let mut first = true;
        for r in 0..k {
            // The best (key, then lower block) below the previous round's, as one u64: the key
            // above, the complement of the block below.
            let mut best = 0u64;
            let mut b = lane;
            while b < visible {
                let bias = if own - b < local as usize { FORCE } else { 0.0 };
                // SAFETY: b < visible <= nblk, so sc + b < (t·4 + w + 1)·nblk <= scores.len().
                let s = unsafe { *scores.get_unchecked(sc + b) } + bias;
                let comp = (u64::from(order_key(s)) << 32) | u64::from(u32::MAX - b as u32);
                if (first || comp < prev) && comp > best {
                    best = comp;
                }
                b += 32;
            }
            let mut hi = (best >> 32) as u32;
            let mut lo = best as u32;
            let mut off = 16u32;
            while off > 0 {
                let oh = warp::shuffle_xor(hi, off);
                let ol = warp::shuffle_xor(lo, off);
                if oh > hi || (oh == hi && ol > lo) {
                    hi = oh;
                    lo = ol;
                }
                off >>= 1;
            }
            if lane == r {
                mine = u32::MAX - lo;
            }
            prev = (u64::from(hi) << 32) | u64::from(lo);
            first = false;
        }

        // Lane r holds round r's block: rank it among the others and keep the own block's count.
        let mut rank = 0usize;
        let mut has_own = false;
        for j in 0..k {
            let other = warp::shuffle_f32(f32::from_bits(mine), j as u32).to_bits();
            if other < mine {
                rank += 1;
            }
            if other as usize == own {
                has_own = true;
            }
        }
        if lane < k {
            // SAFETY: row + rank < row + top_k <= ids.len(); the kept blocks are distinct, so each
            // rank has one writer.
            unsafe { *ids.get_unchecked_mut(row + rank) = mine };
        }
        // SAFETY: KEPT_OWN is this block's own shared array; warp w writes slot w, thread 0 reads
        // all four after the barrier.
        let kept_own = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEPT_OWN) };
        if lane == 0 {
            // SAFETY: w < WARPS.
            unsafe { *kept_own.add(w) = u32::from(has_own) };
        }
        thread::sync_threads();
        if tid == 0 {
            let mut all = true;
            for i in 0..WARPS {
                // SAFETY: i < WARPS, written before the barrier.
                all &= unsafe { *kept_own.add(i) } == 1;
            }
            if !all {
                fault.raise(FaultSite::PoolSelect);
            }
            let len = if all {
                (k - 1) * BLOCK + (c - 1) % BLOCK + 1
            } else {
                0
            };
            // SAFETY: t < n_sel.len(); thread 0 alone writes it.
            unsafe { *n_sel.get_unchecked_mut(t) = len as u32 };
        }
    }
}

// ------------------------------------------------------------------ shape

/// The selector's shape, held to the facts the kernels are built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsaShape {
    rule: Msa,
}

impl MsaShape {
    /// The shape keeping `top_k` blocks of [`BLOCK`] positions, `local` forced.
    ///
    /// # Errors
    /// [`GpuError::Shape`] for `top_k` outside `1..=`[`MAX_TOP_K`] or `local` outside `1..=top_k`.
    pub fn new(top_k: usize, local: usize) -> Result<MsaShape, GpuError> {
        let what = "minimaxm3::attn::shape";
        let as_u32 = |name: &str, v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(what, format!("{name} {v} past u32")))
        };
        if top_k == 0 || top_k > MAX_TOP_K {
            return Err(GpuError::shape(
                what,
                format!("top_k {top_k}: a warp keeps 1..={MAX_TOP_K} blocks"),
            ));
        }
        let rule = Msa::new(
            as_u32("top_k", top_k)?,
            as_u32("block", BLOCK)?,
            as_u32("local", local)?,
        )
        .map_err(|e| GpuError::shape(what, e.to_string()))?;
        Ok(MsaShape { rule })
    }

    /// The shape of a file's hyperparameters ([`MsaShape::check`]).
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming the first fact that differs.
    pub fn of(hp: &Hparams) -> Result<MsaShape, GpuError> {
        MsaShape::check(hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope.dims, hp.msa)
    }

    /// The shape of a file whose attention has `n_head` query heads over `n_kv` key heads of
    /// `head_dim` values, a rope over the first `rope_dims`, and the selector `msa`, refused by name
    /// when it is not the one the kernels are built for: 64 query heads over 4 key heads of 128
    /// values, a rope over the first 64, 4 index heads of 128 values, blocks of [`BLOCK`] positions.
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming the first fact that differs.
    pub fn check(
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        rope_dims: usize,
        msa: HparamsMsa,
    ) -> Result<MsaShape, GpuError> {
        let facts = [
            ("attention.head_count", n_head, N_HEAD),
            ("attention.head_count_kv", n_kv, HEADS),
            ("attention.key_length", head_dim, DIM),
            ("rope.dimension_count", rope_dims, ROT),
            ("attention.indexer.head_count", msa.heads, HEADS),
            ("attention.indexer.key_length", msa.d, DIM),
            ("attention.indexer.block_size", msa.block, BLOCK),
        ];
        if let Some((key, got, want)) = facts.iter().find(|(_, got, want)| got != want) {
            return Err(GpuError::shape(
                "minimaxm3::attn::shape",
                format!("{key} is {got}; the kernels are built for {want}"),
            ));
        }
        MsaShape::new(msa.top_k, msa.local)
    }

    /// The selection rule the kernels and their gate follow.
    #[must_use]
    pub fn rule(&self) -> Msa {
        self.rule
    }

    /// Blocks a row keeps when it sees more: the lists' stride.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.rule.top_k()
    }
}

/// The attention's score scale, `1/√128` as the reference rounds it (`1.0f/sqrtf(float(128))`).
#[must_use]
pub fn attn_scale() -> f32 {
    1.0 / (DIM as f32).sqrt()
}

/// Blocks a `ctx`-position cache holds: the scores' row length.
#[must_use]
pub fn blocks_for(ctx: usize) -> usize {
    ctx.div_ceil(BLOCK)
}

// ---------------------------------------------------------------- scratch

/// What one selection leaves: the block scores of the rows that were scored, the kept blocks and
/// the key counts the flash reads. Allocated once for up to `rows` rows over a `ctx`-position
/// cache and reused by every launch.
pub struct MsaScratch {
    /// `[rows][HEADS][blocks_for(ctx)]`: each scored row's block scores.
    pub scores: DeviceBuffer<f32>,
    /// `[rows][HEADS][top_k]`: each row's kept blocks per index head, ascending.
    pub ids: DeviceBuffer<u32>,
    /// `[rows]`: the keys each row's blocks stand for.
    pub n_sel: DeviceBuffer<u32>,
    rows: usize,
    nblk: usize,
    top_k: usize,
}

impl MsaScratch {
    /// Allocate for `rows` rows over a `ctx`-position cache keeping `shape`'s blocks. Load-time
    /// only.
    ///
    /// # Errors
    /// [`GpuError::Shape`] for no row or no position; an allocation error.
    pub fn new(
        stream: &CudaStream,
        shape: &MsaShape,
        rows: usize,
        ctx: usize,
    ) -> Result<MsaScratch, GpuError> {
        if rows == 0 || ctx == 0 {
            return Err(GpuError::shape(
                "minimaxm3::attn::scratch",
                format!("need rows and ctx >= 1, got rows={rows} ctx={ctx}"),
            ));
        }
        let nblk = blocks_for(ctx);
        let top_k = shape.top_k();
        Ok(MsaScratch {
            scores: DeviceBuffer::zeroed(stream, rows * HEADS * nblk)?,
            ids: DeviceBuffer::zeroed(stream, rows * HEADS * top_k)?,
            n_sel: DeviceBuffer::zeroed(stream, rows)?,
            rows,
            nblk,
            top_k,
        })
    }

    /// The most rows a launch may serve.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

// --------------------------------------------------------------- launchers

/// [`MsaKernels::enqueue_score`]'s arguments: the rows' index query heads normed and turned
/// (`[m][HEADS][DIM]` f32, [`IndexArgs::iq`]'s output), the layer's index-key plane (`[ctx][DIM]`
/// f16, the rows the call wrote included), the rows' live counts on the device (`m`), and `blocks`,
/// the most blocks any of the rows sees (`⌈max count / BLOCK⌉`, or the cache's for a captured step):
/// the launch is `m · blocks` blocks of 128.
pub struct ScoreArgs<'a> {
    pub iq: &'a DeviceBuffer<f32>,
    pub ik: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub ctx: usize,
    pub m: usize,
    pub blocks: usize,
    pub fault: FaultSink,
}

/// [`MsaKernels::enqueue_pick`]'s arguments: the rows' live counts on the device and the cache's
/// height.
pub struct PickArgs<'a> {
    pub n_keys: &'a DeviceBuffer<u32>,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
}

/// The two selection kernels of a device.
pub struct MsaKernels {
    module: msa_kernels::LoadedModule,
}

impl MsaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MsaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launchers check its launch contracts.
        let module = unsafe { bloomery_gpu::shared_module!(msa_kernels, ctx)? };
        Ok(MsaKernels { module })
    }

    /// Enqueue the score pass for `args.m` rows: every block of every row that selects gets its
    /// scores per index head. One launch of `m · blocks` blocks of 128; it follows the call's
    /// [`enqueue_index`] and precedes [`MsaKernels::enqueue_pick`]. Refused by name: no row, more
    /// rows or blocks than the scratch holds, buffers shorter than the contract needs.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_score(
        &self,
        stream: &CudaStream,
        shape: &MsaShape,
        args: ScoreArgs<'_>,
        scratch: &mut MsaScratch,
    ) -> Result<(), GpuError> {
        let what = "minimaxm3::attn::enqueue_score";
        let ScoreArgs {
            iq,
            ik,
            n_keys,
            ctx,
            m,
            blocks,
            fault,
        } = args;
        check_scratch(what, shape, scratch, ctx, m)?;
        if blocks == 0 || blocks > scratch.nblk {
            return Err(GpuError::shape(
                what,
                format!("need 1..={} blocks, got {blocks}", scratch.nblk),
            ));
        }
        let lens = [
            ("iq", iq.len(), m * HEADS * DIM),
            ("ik", ik.len(), ctx * DIM),
            ("n_keys", n_keys.len(), m),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * blocks)?;
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let nblk = launch_u32(what, "nblk", scratch.nblk)?;
        let top_k = launch_u32(what, "top_k", shape.top_k())?;
        let m_u = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_msa_score(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.msa_score(
            stream,
            &prep,
            iq,
            ik,
            n_keys,
            ctx_u,
            nblk,
            top_k,
            m_u,
            fault,
            &mut scratch.scores,
        )?;
        Ok(())
    }

    /// Enqueue the select pass for `args.m` rows: every row's kept blocks per index head and its
    /// key count. One launch of `m` blocks of 128. Refused by name as
    /// [`MsaKernels::enqueue_score`]. Asynchronous, allocation-free, capturable.
    pub fn enqueue_pick(
        &self,
        stream: &CudaStream,
        shape: &MsaShape,
        args: PickArgs<'_>,
        scratch: &mut MsaScratch,
    ) -> Result<(), GpuError> {
        let what = "minimaxm3::attn::enqueue_pick";
        let PickArgs {
            n_keys,
            ctx,
            m,
            fault,
        } = args;
        check_scratch(what, shape, scratch, ctx, m)?;
        if n_keys.len() < m {
            return Err(GpuError::shape(
                what,
                format!("n_keys.len() {} < {m}", n_keys.len()),
            ));
        }
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let nblk = launch_u32(what, "nblk", scratch.nblk)?;
        let top_k = launch_u32(what, "top_k", shape.top_k())?;
        let local = launch_u32(what, "local", shape.rule().local())?;
        let m_u = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_msa_select(LaunchConfig1D::new(m_u, THREADS, 0))?;
        self.module.msa_select(
            stream,
            &prep,
            n_keys,
            &scratch.scores,
            ctx_u,
            nblk,
            top_k,
            local,
            m_u,
            fault,
            &mut scratch.ids,
            &mut scratch.n_sel,
        )?;
        Ok(())
    }
}

/// The checks the two passes share: a row count within the scratch's, the scratch built for this
/// shape and a cache of this height.
fn check_scratch(
    what: &'static str,
    shape: &MsaShape,
    scratch: &MsaScratch,
    ctx: usize,
    m: usize,
) -> Result<(), GpuError> {
    if m == 0
        || m > scratch.rows
        || scratch.top_k != shape.top_k()
        || scratch.nblk != blocks_for(ctx)
    {
        return Err(GpuError::shape(
            what,
            format!(
                "need 1..={} rows, the scratch's top_k {} and {} blocks; got m={m} top_k={} \
                 blocks={}",
                scratch.rows,
                scratch.top_k,
                scratch.nblk,
                shape.top_k(),
                blocks_for(ctx)
            ),
        ));
    }
    Ok(())
}

/// [`enqueue_index`]'s arguments: the head kernel's, at 4 index heads over 1 key head. `proj_q` is
/// the query projection (`[m][HEADS][DIM]` f32, raw) and `iq` its normed and turned heads; `proj_k`
/// the key projection (`[m][DIM]`, raw), turned in place; `zeros` a value row for each token
/// (`[m][DIM]`) and `sink` the plane the head kernel writes it to (`[ctx][DIM]` f16), which nothing
/// reads and every layer may share; `keys` the layer's index-key plane (`[ctx][DIM]` f16); `table`
/// the main attention's rope table (`ROT` f32 for each of `ctx` positions) and `pos` the tokens'
/// positions.
pub struct IndexArgs<'a> {
    pub proj_q: &'a DeviceBuffer<f32>,
    pub iq: &'a mut DeviceBuffer<f32>,
    pub proj_k: &'a mut DeviceBuffer<f32>,
    pub zeros: &'a DeviceBuffer<f32>,
    pub gain_q: &'a DeviceBuffer<f32>,
    pub gain_k: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub keys: &'a mut DeviceBuffer<u16>,
    pub sink: &'a mut DeviceBuffer<u16>,
}

/// Enqueue the index heads of `args.m` tokens: the common head kernel at [`HEADS`] heads over one
/// key head — each head normed by its gain, the first [`ROT`] values turned, the key appended to
/// `args.keys`. Refusals and faults as the head kernel's.
///
/// # Safety
/// The `m` positions of `args.pos` are distinct (the head kernel's contract).
pub unsafe fn enqueue_index(
    rope: &RopeNeoxKernels,
    stream: &CudaStream,
    args: IndexArgs<'_>,
) -> Result<(), GpuError> {
    let IndexArgs {
        proj_q,
        iq,
        proj_k,
        zeros,
        gain_q,
        gain_k,
        table,
        pos,
        eps,
        ctx,
        m,
        fault,
        keys,
        sink,
    } = args;
    // SAFETY: the caller's contract is the head kernel's.
    unsafe {
        rope.enqueue_head_norm_neox_append_128_partial(
            stream,
            PartialNeoxArgs {
                qg: proj_q,
                q: iq,
                k: proj_k,
                v: zeros,
                gq: gain_q,
                gk: gain_k,
                table,
                pos,
                eps,
                n_head: HEADS,
                n_kv: 1,
                ctx,
                m,
                fault,
                cache_k: keys,
                cache_v: sink,
            },
        )
    }
}

/// [`enqueue_attention`]'s arguments: the rows' query heads normed and turned (`[m][N_HEAD][DIM]`
/// f32, the main head kernel's output), the layer's key and value planes (`[4][ctx][DIM]` f16,
/// the rows of the call written), the flash's partials ([`partials_lens`]) and the output
/// (`[m][N_HEAD][DIM]`). The blocks and key counts are the scratch's, the selection's.
pub struct AttnArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub ctx: usize,
    pub m: usize,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The partials' lengths ([`AttnArgs::part_v`], [`AttnArgs::part_ms`]) for `m` rows.
#[must_use]
pub fn partials_lens(shape: &MsaShape, m: usize) -> (usize, usize) {
    let width = shape.rule().width();
    (
        listed_partials_v_len(m, N_HEAD, width),
        listed_partials_ms_len(m, N_HEAD, width),
    )
}

/// Enqueue the flash over the blocks `scratch` holds for `args.m` rows ([`MsaKernels::enqueue_pick`]
/// ran for them): each key head's sixteen query heads attend the key head's own kept blocks, at the
/// scale [`attn_scale`]. Two launches. A row of length 0 (a refused count, or a selection that lost
/// the own block) is NaN with [`FaultSite::KeyCount`] raised.
pub fn enqueue_attention(
    flash: &FlashGqaKernels,
    stream: &CudaStream,
    shape: &MsaShape,
    scratch: &MsaScratch,
    args: AttnArgs<'_>,
) -> Result<(), GpuError> {
    let AttnArgs {
        q,
        kc,
        vc,
        ctx,
        m,
        part_v,
        part_ms,
        fault,
        y,
    } = args;
    flash.enqueue_pass_p16_blocks(
        stream,
        GqaBlocksArgs {
            q,
            kc,
            vc,
            ids: &scratch.ids,
            n_keys: &scratch.n_sel,
            blocks: shape.top_k(),
            scale: attn_scale(),
            n_kv: HEADS,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        },
        N_HEAD,
    )
}
