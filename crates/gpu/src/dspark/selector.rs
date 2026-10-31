//! DFlash2's selector head on the card: from a block's normed rows and the
//! head's logits, each block row after the anchor's [`TOP_K`] best ids, and
//! every candidate scored against every id the row before kept, the
//! lattice [`runtime::selector`] walks on the host.
//!
//! Per block of `m` rows (rows `1..m`, the anchor's row 0 drafts nothing):
//!
//! - the gate, `selector_hidden` over the normed rows: the head's Q8_0 gemv
//!   over `m` columns, `gate[c · RANK + r]`;
//! - [`selector_kernels::sel_topk_part`] — each block of 256 threads takes
//!   [`TILE`] ids of every row (the tile in shared memory, the logits' own
//!   layout `x[v · m + c]` read coalesced), and warp `c` finds row `c`'s 16
//!   best of them by sixteen rounds of a warp tournament
//!   ([`crate::best::warp_best`]);
//! - [`selector_kernels::sel_topk_merge`] — one block a row merges its
//!   parts' lists the same way (a warp's 512 entries to 16, the eight warps'
//!   128 to 16): the row's candidates, ids into the lattice and their logits
//!   into `unary`;
//! - [`selector_kernels::sel_score`] — one block per (row, predecessor): the
//!   predecessor's `selector_predecessor` row dequantized and multiplied by
//!   the row's gate, then each of the row's candidates' `selector_successor`
//!   rows dotted with it, plus the candidate's logit. Row 1's one
//!   predecessor is the anchor `first[0]`; every later row's are the 16
//!   candidates of the row before.
//!
//! Selection is exact and takes the walk's rule: the highest value first,
//! the lower id among equals — the host's [`runtime::selector::top_k`] bit
//! for bit.
//!
//! The lattice buffer holds, per scored row, [`ROW_WORDS`] words: the 16 ids,
//! then the scores `scores[j · 16 + i]` of candidate `i` after predecessor
//! `j` as f32 bits (row 1 writes only its first 16); two words after the last
//! row hold the card's fault word and its layer's site mask, copied by the
//! last block of the scoring launch.
//!
//! Numeric contract of a score: a Q8_0 row dequantizes exactly (the code
//! times the block's f16 scale, one f32 product); the predecessor times the
//! gate is one f32 product per value; the dot over the [`RANK`] values is 32
//! lane partials, lane `l` the fused multiply-add chain from 0 over values
//! `8l .. 8l + 8` in order, combined by the xor butterfly (16, 8, 4, 2, 1);
//! the candidate's logit is added once. [`runtime::selector::lane_dot`] is
//! its host twin.
//!
//! No silent failure: a logit that is not finite raises [`FaultSite::Logit`]
//! (the part launch), an id at or past a table's rows [`FaultSite::TokenId`]
//! (the scoring launch; the score is NaN and no table word is read).

use crate::best::warp_best;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::half_bits_to_f32;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

use super::Q8Rows;

const WHAT: &str = "dspark::selector";

/// The factors' width: the row length of the three tables, the width the
/// score kernel is built for.
pub const RANK: usize = 256;
/// Candidates a row keeps: the width the kernels are built for.
pub const TOP_K: usize = 16;
/// The most block rows: the head's projection kernels take eight columns.
pub const MAX_ROWS: usize = 8;
/// Words of one scored row of the lattice buffer: the ids, then the scores
/// of every candidate after each of [`TOP_K`] predecessors.
pub const ROW_WORDS: usize = TOP_K + TOP_K * TOP_K;
/// The most ids a head may score: [`MAX_PARTS`] tiles.
pub const MAX_VOCAB: usize = TILE * MAX_PARTS;

/// Threads of every launch, and its warps.
const THREADS: u32 = 256;
const WARPS: usize = THREADS as usize / 32;
/// Ids one part launch block takes of each row, and the entries one lane of
/// a warp holds of them.
const TILE: usize = 1024;
const TILE_LANE: usize = TILE / 32;
/// Floats of a part launch block's shared tile: [`TILE`] ids of every row.
const TILE_FLOATS: usize = TILE * MAX_ROWS;
/// Parts the merge holds: its threads' 16 entries each cover `MAX_PARTS ·
/// TOP_K` candidates.
const MAX_PARTS: usize = 256;
const MERGE_LANE: usize = MAX_PARTS * TOP_K / THREADS as usize;
/// The merge's second level: the warps' lists, and one lane's share of them.
const LISTS: usize = WARPS * TOP_K;
const LISTS_LANE: usize = LISTS / 32;
/// Words of a Q8_0 row of [`RANK`] values, and its scales.
const QWORDS: usize = RANK / 4;
const SCALES: usize = RANK / 32;
/// Predecessors one row is scored against, and the values a lane of the
/// score kernel owns.
const PREDS: usize = TOP_K;
const SCORE_LANE: usize = RANK / 32;

const _: () = assert!(MAX_ROWS == 8 && TOP_K == 16 && RANK == 256 && THREADS == 256);
const _: () = assert!(TILE.is_multiple_of(32) && MERGE_LANE == 16 && LISTS_LANE == 4);
const _: () = assert!(SCORE_LANE * 32 == RANK && SCORE_LANE == 8 && ROW_WORDS == 272);

/// The device ABI's launches of one block: the gate's gemv, the two top-k
/// launches and the scoring launch.
pub const LAUNCHES: usize = 4;

// ------------------------------------------------------------------ cores

/// Value `q` (0..4) of a word of four Q8_0 codes.
#[inline(always)]
fn code(word: u32, q: usize) -> f32 {
    (word >> (8 * q)) as u8 as i8 as f32
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod selector_kernels {
    use super::*;

    /// Part `blockIdx.x` of every row `1..m`: the 16 best of the [`TILE`] ids
    /// `part · TILE ..`, written to slot `((c − 1) · n_parts + part) · 16` of
    /// `cv` (their logits) and `ci` (their ids). The tile of all `m` rows is
    /// staged in shared memory in the logits' layout `x[v · m + c]`, so the
    /// block's loads are contiguous; warp `c` then reads its row's column
    /// and runs the tournament. Warp 0 (the anchor's row) and the warps past
    /// `m` take no part. A logit that is not finite raises
    /// [`FaultSite::Logit`]; ids past `n` are −∞ entries.
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
            m >= 2,
            m <= 8,
            n >= 16,
            n_parts >= 1,
            n_parts * 1024 >= n,
            x.len() >= n * m,
            cv.len() >= (m - 1) * n_parts * 16,
            ci.len() >= (m - 1) * n_parts * 16
        )
    )]
    pub fn sel_topk_part(
        x: &[f32],
        n: u32,
        m: u32,
        n_parts: u32,
        fault: FaultSink,
        mut cv: DisjointSlice<f32>,
        mut ci: DisjointSlice<u32>,
    ) {
        static mut TILE_V: SharedArray<f32, TILE_FLOATS> = SharedArray::UNINIT;

        let part = thread::blockIdx_x();
        if part >= n_parts {
            return; // block-uniform, before any barrier
        }
        let tid = thread::threadIdx_x() as usize;
        let (n, m) = (n as usize, m as usize);
        let base = part as usize * TILE;
        // SAFETY: TILE_V is this block's own shared allocation; the raw form
        // is the only way to reach it without a reference to a `static mut`.
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE_V) };
        let mut i = tid;
        while i < TILE * m {
            let valid = base + i / m < n;
            let v = if valid {
                // SAFETY: id = base + i / m < n, so id · m + i % m = base · m
                // + i < n · m <= x.len() by the launch contract.
                unsafe { *x.get_unchecked(base * m + i) }
            } else {
                f32::NEG_INFINITY
            };
            if valid && !v.is_finite() {
                fault.raise(FaultSite::Logit);
            }
            // SAFETY: i < TILE · m <= TILE_FLOATS; slot i is this thread's
            // alone (i is tid mod THREADS).
            unsafe { *tile.add(i) = v };
            i += THREADS as usize;
        }
        thread::sync_threads();
        let w = tid / 32;
        if w == 0 || w >= m {
            return; // warp-uniform; no barrier follows
        }
        let lane = tid % 32;
        let mut v = [0.0f32; TILE_LANE];
        let mut id = [0u32; TILE_LANE];
        for j in 0..TILE_LANE {
            thread::__unroll_config::<0>();
            let local = lane + 32 * j;
            // SAFETY: local < TILE and w < m, so local · m + w < TILE · m
            // <= TILE_FLOATS, a slot written above and visible past the
            // barrier.
            v[j] = unsafe { *tile.add(local * m + w) };
            id[j] = (base + local) as u32;
        }
        let at = ((w - 1) * n_parts as usize + part as usize) * TOP_K;
        // SAFETY: at + TOP_K <= (m - 1) · n_parts · TOP_K <= cv.len(), ci.len()
        // by the launch contract; the slots of (row w, this part) are this
        // warp's alone.
        unsafe {
            warp_best::<TILE_LANE, TOP_K>(v, id, cv.as_mut_ptr().add(at), ci.as_mut_ptr().add(at))
        };
    }

    /// Row `blockIdx.x + 1`'s merge: its `n_parts` lists of 16 into the 16
    /// best. Each of the block's eight warps takes 512 of the candidates (16
    /// a lane) down to its own 16, then warp 0 takes the eight lists' 128
    /// down to the row's: the logits to `unary[ρ · 16 ..]` and the ids to
    /// the lattice row `ρ` (`lat[ρ · ROW_WORDS ..]`), `ρ = blockIdx.x`, both
    /// highest first.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m >= 2,
            m <= 8,
            n_parts >= 1,
            n_parts <= 256,
            cv.len() >= (m - 1) * n_parts * 16,
            ci.len() >= (m - 1) * n_parts * 16,
            lat.len() >= (m - 1) * 272 + 2,
            unary.len() >= (m - 1) * 16
        )
    )]
    pub fn sel_topk_merge(
        cv: &[f32],
        ci: &[u32],
        n_parts: u32,
        m: u32,
        mut lat: DisjointSlice<u32>,
        mut unary: DisjointSlice<f32>,
    ) {
        static mut LIST_V: SharedArray<f32, LISTS> = SharedArray::UNINIT;
        static mut LIST_I: SharedArray<u32, LISTS> = SharedArray::UNINIT;

        let rho = thread::blockIdx_x() as usize;
        if rho + 1 >= m as usize {
            return; // block-uniform, before any barrier
        }
        let tid = thread::threadIdx_x() as usize;
        let n_c = n_parts as usize * TOP_K;
        // SAFETY: both arrays are this block's own shared allocations.
        let (lv, li) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut LIST_V),
                SharedArray::as_raw_mut_ptr(&raw mut LIST_I),
            )
        };
        let mut v = [f32::NEG_INFINITY; MERGE_LANE];
        let mut id = [u32::MAX; MERGE_LANE];
        for j in 0..MERGE_LANE {
            thread::__unroll_config::<0>();
            let e = tid + THREADS as usize * j;
            if e < n_c {
                // SAFETY: rho < m - 1 and e < n_c, so rho · n_c + e <
                // (m - 1) · n_parts · 16 <= cv.len(), ci.len().
                unsafe {
                    v[j] = *cv.get_unchecked(rho * n_c + e);
                    id[j] = *ci.get_unchecked(rho * n_c + e);
                }
            }
        }
        let w = tid / 32;
        // SAFETY: the warp's 16 slots of the lists are its own.
        unsafe { warp_best::<MERGE_LANE, TOP_K>(v, id, lv.add(w * TOP_K), li.add(w * TOP_K)) };
        thread::sync_threads();
        if w != 0 {
            return;
        }
        let mut v2 = [f32::NEG_INFINITY; LISTS_LANE];
        let mut id2 = [u32::MAX; LISTS_LANE];
        for j in 0..LISTS_LANE {
            thread::__unroll_config::<0>();
            let e = tid + 32 * j;
            // SAFETY: e < LISTS, written by the warps above and visible past
            // the barrier.
            unsafe {
                v2[j] = *lv.add(e);
                id2[j] = *li.add(e);
            }
        }
        // SAFETY: rho < m - 1, so the row's 16 slots of unary and of the
        // lattice's ids are inside the launch contract's bounds and this
        // warp's alone.
        unsafe {
            warp_best::<LISTS_LANE, TOP_K>(
                v2,
                id2,
                unary.as_mut_ptr().add(rho * TOP_K),
                lat.as_mut_ptr().add(rho * ROW_WORDS),
            );
        }
    }

    /// Block `b` of `(m − 1) · 16`: lattice row `ρ = b / 16` (block row
    /// `ρ + 1`), predecessor `j = b % 16`. The predecessor `p` is the anchor
    /// `first[0]` for row 1 (which has only `j = 0`) and candidate `j` of the
    /// row before otherwise; the block builds `cond = pred[p] · gate[ρ + 1]`
    /// in shared memory, then warp `w` scores the row's candidates `w` and
    /// `w + 8`: `scores[j · 16 + i] = unary[ρ][i] + succ[id_i] · cond`
    /// (module doc), written to `lat[ρ · ROW_WORDS + 16 + j · 16 + i]`. An id
    /// at or past its table's rows raises [`FaultSite::TokenId`], reads no
    /// word of the table and scores NaN. Every block draws a ticket; the last
    /// copies the card's fault word and its site mask to `lat[(m − 1) ·
    /// ROW_WORDS ..]` and puts the count back to zero.
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
            m >= 2,
            m <= 8,
            gate.len() >= m * 256,
            unary.len() >= (m - 1) * 16,
            first.len() >= 1,
            pq.len() >= 64 * n_pred,
            pd.len() >= 8 * n_pred,
            sq.len() >= 64 * n_succ,
            sd.len() >= 8 * n_succ,
            lat.len() >= (m - 1) * 272 + 2,
            done.len() >= 1
        )
    )]
    pub fn sel_score(
        gate: &[f32],
        unary: &[f32],
        first: &[u32],
        pq: &[u32],
        pd: &[u16],
        sq: &[u32],
        sd: &[u16],
        n_pred: u32,
        n_succ: u32,
        m: u32,
        fault: FaultSink,
        mut lat: DisjointSlice<u32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut COND: SharedArray<f32, RANK> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let total = (m as usize - 1) * PREDS;
        if b >= total {
            return; // block-uniform, and no ticket owed
        }
        let (rho, j) = (b / PREDS, b % PREDS);
        let tid = thread::threadIdx_x() as usize;
        let latp = lat.as_mut_ptr();
        if rho > 0 || j == 0 {
            // SAFETY: COND is this block's own shared allocation.
            let cond = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COND) };
            // SAFETY: first.len() >= 1; for rho > 0, (rho - 1) · ROW_WORDS +
            // j < (m - 1) · ROW_WORDS <= lat.len(), a word the merge launch
            // wrote.
            let p = unsafe {
                if rho == 0 {
                    *first.get_unchecked(0)
                } else {
                    *latp.add((rho - 1) * ROW_WORDS + j)
                }
            };
            let p_ok = p < n_pred;
            if tid == 0 && !p_ok {
                fault.raise(FaultSite::TokenId);
            }
            let value = if p_ok {
                let pu = p as usize;
                // SAFETY: pu < n_pred, so word pu · 64 + tid / 4 < 64 · n_pred
                // <= pq.len() and scale pu · 8 + tid / 32 < 8 · n_pred <=
                // pd.len(); gate row rho + 1 < m, value tid < 256, inside
                // gate.len() >= m · 256.
                unsafe {
                    let word = *pq.get_unchecked(pu * QWORDS + tid / 4);
                    let scale = half_bits_to_f32(*pd.get_unchecked(pu * SCALES + tid / 32));
                    (code(word, tid % 4) * scale) * *gate.get_unchecked((rho + 1) * RANK + tid)
                }
            } else {
                f32::NAN
            };
            // SAFETY: tid < RANK slots of COND, one per thread.
            unsafe { *cond.add(tid) = value };
            thread::sync_threads();
            let lane = tid % 32;
            for pass in 0..2 {
                let i = tid / 32 + WARPS * pass;
                // SAFETY: rho < m - 1 and i < 16: the ids' words and the
                // unary slots are inside the launch contract's bounds, and
                // the merge launch wrote them.
                let (s, u) = unsafe {
                    (
                        *latp.add(rho * ROW_WORDS + i),
                        *unary.get_unchecked(rho * TOP_K + i),
                    )
                };
                let s_ok = s < n_succ;
                if lane == 0 && !s_ok {
                    fault.raise(FaultSite::TokenId);
                }
                let mut acc = 0.0f32;
                if s_ok {
                    let su = s as usize;
                    // SAFETY: su < n_succ, so words su · 64 + 2 lane and + 1
                    // lie below 64 · n_succ <= sq.len() and the scale su · 8
                    // + lane / 4 below 8 · n_succ <= sd.len(); the lane's
                    // eight COND values 8 lane .. 8 lane + 8 are below RANK,
                    // written before the barrier.
                    unsafe {
                        let w0 = *sq.get_unchecked(su * QWORDS + 2 * lane);
                        let w1 = *sq.get_unchecked(su * QWORDS + 2 * lane + 1);
                        let scale = half_bits_to_f32(*sd.get_unchecked(su * SCALES + lane / 4));
                        for q in 0..SCORE_LANE {
                            thread::__unroll_config::<0>();
                            let c = if q < 4 { code(w0, q) } else { code(w1, q - 4) };
                            acc = (c * scale).mul_add(*cond.add(SCORE_LANE * lane + q), acc);
                        }
                    }
                } else {
                    acc = f32::NAN;
                }
                let dot = warp::reduce_sum_f32(acc);
                if lane == 0 {
                    // SAFETY: rho · ROW_WORDS + 16 + j · 16 + i < (rho + 1) ·
                    // ROW_WORDS <= lat.len(); the (row, predecessor,
                    // candidate) slot is this lane's alone.
                    unsafe {
                        *latp.add(rho * ROW_WORDS + TOP_K + j * TOP_K + i) = (dot + u).to_bits();
                    }
                }
            }
        }
        if tid != 0 {
            return;
        }
        // SAFETY: done.len() >= 1 by the launch contract; every access to
        // done[0] on the card is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(done.as_mut_ptr()) };
        // The ticket's release orders this block's raises before it; the last
        // block's acquire sees every block's.
        if count.fetch_add(1, AtomicOrdering::AcqRel) + 1 != total as u32 {
            return;
        }
        count.store(0, AtomicOrdering::Relaxed);
        let word = fault.read();
        let sites = fault.read_sites(word);
        // SAFETY: (m - 1) · ROW_WORDS + 1 < lat.len() by the launch contract;
        // the last block's thread 0 alone writes both.
        unsafe {
            *latp.add((m as usize - 1) * ROW_WORDS) = word;
            *latp.add((m as usize - 1) * ROW_WORDS + 1) = sites;
        }
    }
}

// ------------------------------------------------------------------- host

/// The three tables of a selector head, Q8_0 rows of [`RANK`] values.
pub struct SelectorTail {
    /// `selector_hidden`: [`RANK`] rows over the draft's hidden width, the
    /// gate's projection of a normed row.
    pub hidden: Q8Rows,
    /// `selector_predecessor`: one row per id.
    pub pred: Q8Rows,
    /// `selector_successor`: one row per id.
    pub succ: Q8Rows,
    /// Candidates a row keeps; the kernels are built for [`TOP_K`].
    pub top_k: usize,
}

/// The loaded module. Owns no stream.
pub struct SelectorKernels {
    module: selector_kernels::LoadedModule,
}

impl SelectorKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<SelectorKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(selector_kernels, ctx)? };
        Ok(SelectorKernels { module })
    }

    /// Parts of the part launch for a head of `n` ids.
    #[must_use]
    pub fn parts(n: usize) -> usize {
        n.div_ceil(TILE)
    }
}

/// [`Selector::enqueue`]'s arguments beside the head's own buffers.
pub struct SelectArgs<'a> {
    /// The head's normed rows, `m · hidden`, token-major.
    pub normed: &'a DeviceBuffer<f32>,
    /// The head's logits, `logits[v · m + c]`.
    pub logits: &'a DeviceBuffer<f32>,
    /// The anchor's id at word 0.
    pub first: &'a DeviceBuffer<u32>,
    /// One zero word, the scoring launch's ticket count.
    pub done: &'a mut DeviceBuffer<u32>,
    /// Rows in the block.
    pub m: usize,
    /// Where a launch's fault is raised.
    pub fault: FaultSink,
}

/// The selector's buffers and launches for blocks of up to `rows` rows over
/// a head of `n` ids.
pub struct Selector {
    kernels: SelectorKernels,
    n: usize,
    parts: usize,
    rows: usize,
    /// The gate, `rows · RANK`, token-major.
    gate: DeviceBuffer<f32>,
    /// Each part's 16 best logits and ids, per scored row.
    cand_v: DeviceBuffer<f32>,
    cand_i: DeviceBuffer<u32>,
    /// Each scored row's 16 candidates' logits.
    unary: DeviceBuffer<f32>,
    /// The lattice: [`ROW_WORDS`] a scored row, then two fault words.
    lattice: DeviceBuffer<u32>,
}

impl Selector {
    /// Buffers for blocks of `2..=rows` rows over `hidden`-wide normed rows
    /// and a head of `n` ids, refusing by name a table the launches have no
    /// shape for. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        tail: &SelectorTail,
        n: usize,
        hidden: usize,
        rows: usize,
    ) -> Result<Selector, GpuError> {
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        if !(2..=MAX_ROWS).contains(&rows) {
            return Err(refuse(format!(
                "blocks of up to {rows} rows; the selector takes 2..={MAX_ROWS}"
            )));
        }
        if tail.top_k != TOP_K {
            return Err(refuse(format!(
                "a top-k of {}; the kernels take {TOP_K}",
                tail.top_k
            )));
        }
        if !(TOP_K..=MAX_VOCAB).contains(&n) {
            return Err(refuse(format!(
                "a head of {n} ids; the selector takes {TOP_K}..={MAX_VOCAB}"
            )));
        }
        let rows_ok = |t: &Q8Rows, want_rows: usize, k: usize| {
            t.qs.rows() == want_rows
                && t.d.rows() == want_rows
                && t.qs.cols() == k / 4
                && t.d.cols() == k / 32
        };
        if !rows_ok(&tail.hidden, RANK, hidden) {
            return Err(refuse(format!(
                "selector_hidden of {}x{} words and {}x{} scales; {RANK} rows over {hidden} \
                 values need {RANK}x{} words and {RANK}x{} scales",
                tail.hidden.qs.rows(),
                tail.hidden.qs.cols(),
                tail.hidden.d.rows(),
                tail.hidden.d.cols(),
                hidden / 4,
                hidden / 32
            )));
        }
        for (name, t) in [
            ("selector_predecessor", &tail.pred),
            ("selector_successor", &tail.succ),
        ] {
            if !rows_ok(t, n, RANK) {
                return Err(refuse(format!(
                    "{name} of {}x{} words and {}x{} scales; {n} ids of {RANK} values need \
                     {n}x{QWORDS} words and {n}x{SCALES} scales",
                    t.qs.rows(),
                    t.qs.cols(),
                    t.d.rows(),
                    t.d.cols()
                )));
            }
        }
        let parts = SelectorKernels::parts(n);
        let scored = rows - 1;
        Ok(Selector {
            kernels: SelectorKernels::load(ctx)?,
            n,
            parts,
            rows,
            gate: DeviceBuffer::zeroed(stream, rows * RANK)?,
            cand_v: DeviceBuffer::zeroed(stream, scored * parts * TOP_K)?,
            cand_i: DeviceBuffer::zeroed(stream, scored * parts * TOP_K)?,
            unary: DeviceBuffer::zeroed(stream, scored * TOP_K)?,
            lattice: DeviceBuffer::zeroed(stream, scored * ROW_WORDS + 2)?,
        })
    }

    /// Device bytes of the scratch, as allocated.
    #[must_use]
    pub fn scratch_bytes(&self) -> u64 {
        (self.gate.num_bytes()
            + self.cand_v.num_bytes()
            + self.cand_i.num_bytes()
            + self.unary.num_bytes()
            + self.lattice.num_bytes()) as u64
    }

    /// Device bytes of each table buffer, as allocated: the three tables'
    /// quants and scales.
    #[must_use]
    pub fn weight_buffers(tail: &SelectorTail) -> Vec<u64> {
        [&tail.hidden, &tail.pred, &tail.succ]
            .iter()
            .flat_map(|t| [t.qs.buf().num_bytes() as u64, t.d.buf().num_bytes() as u64])
            .collect()
    }

    /// Enqueue the block's selection over `a.m` rows (module doc): the gate's
    /// gemv, the two top-k launches and the scoring launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &mut self,
        gpu: &Gpu,
        tail: &SelectorTail,
        a: SelectArgs<'_>,
    ) -> Result<(), GpuError> {
        let SelectArgs {
            normed,
            logits,
            first,
            done,
            m,
            fault,
        } = a;
        if !(2..=self.rows).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a block of {m} rows; the selector holds 2..={}", self.rows),
            ));
        }
        let s = gpu.stream();
        gpu.q8f32().enqueue_q8_0_gemv_mcol(
            s,
            crate::q8f32::Q8_0GemvMcolArgs {
                qs: &tail.hidden.qs,
                d: &tail.hidden.d,
                x: normed,
                m,
                out: crate::q8f32::GemvOut::TokenMajor,
                y: &mut self.gate,
            },
        )?;
        let (n, mm, parts) = (
            launch_u32(WHAT, "n", self.n)?,
            launch_u32(WHAT, "m", m)?,
            launch_u32(WHAT, "n_parts", self.parts)?,
        );
        let prep = self
            .kernels
            .module
            .prepare_sel_topk_part(LaunchConfig1D::new(parts, THREADS, 0))?;
        self.kernels.module.sel_topk_part(
            s,
            &prep,
            logits,
            n,
            mm,
            parts,
            fault,
            &mut self.cand_v,
            &mut self.cand_i,
        )?;
        let prep = self
            .kernels
            .module
            .prepare_sel_topk_merge(LaunchConfig1D::new(mm - 1, THREADS, 0))?;
        self.kernels.module.sel_topk_merge(
            s,
            &prep,
            &self.cand_v,
            &self.cand_i,
            parts,
            mm,
            &mut self.lattice,
            &mut self.unary,
        )?;
        let blocks = launch_u32(WHAT, "blocks", (m - 1) * PREDS)?;
        let prep = self
            .kernels
            .module
            .prepare_sel_score(LaunchConfig1D::new(blocks, THREADS, 0))?;
        self.kernels.module.sel_score(
            s,
            &prep,
            &self.gate,
            &self.unary,
            first,
            tail.pred.qs.buf(),
            tail.pred.d.buf(),
            tail.succ.qs.buf(),
            tail.succ.d.buf(),
            launch_u32(WHAT, "n_pred", tail.pred.qs.rows())?,
            launch_u32(WHAT, "n_succ", tail.succ.qs.rows())?,
            mm,
            fault,
            &mut self.lattice,
            done,
        )?;
        Ok(())
    }

    /// The lattice words of the last block of `m` rows, [`ROW_WORDS`] a
    /// scored row (`m - 1` of them); a fault any launch of the pass raised is
    /// [`GpuError::Fault`], not a lattice. Blocking.
    pub fn lattice(&self, stream: &CudaStream, m: usize) -> Result<Vec<u32>, GpuError> {
        self.check(m)?;
        let words = self.lattice.to_host_vec(stream)?;
        let at = (m - 1) * ROW_WORDS;
        if let Some(fault) = crate::Fault::from_words(words[at], words[at + 1]) {
            return Err(GpuError::fault("dspark::selector::lattice", fault));
        }
        Ok(words[..at].to_vec())
    }

    /// The gate of the last block's `m` rows, token-major, `m · RANK` values.
    /// Blocking; gate use.
    pub fn gate(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        self.check(m)?;
        Ok(self.gate.to_host_vec(stream)?[..m * RANK].to_vec())
    }

    /// The candidates' logits of the last block's `m - 1` scored rows,
    /// [`TOP_K`] a row, highest first. Blocking; gate use.
    pub fn unary(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        self.check(m)?;
        Ok(self.unary.to_host_vec(stream)?[..(m - 1) * TOP_K].to_vec())
    }

    /// Refuse a row count the scratch holds no room for.
    fn check(&self, m: usize) -> Result<(), GpuError> {
        if (2..=self.rows).contains(&m) {
            Ok(())
        } else {
            Err(GpuError::shape(
                WHAT,
                format!("a block of {m} rows; the selector holds 2..={}", self.rows),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceTensor;
    use crate::dspark::HeadSource;
    use crate::dspark::head::{Head, HeadTail, HeadWeights};
    use crate::{FaultSite, Gpu};
    use runtime::dspark::{DraftVocab, HeadKind};
    use runtime::selector::{Lattice, lane_dot, top_k, walk_p};

    /// The hidden width of the normed rows.
    const HIDDEN: usize = 64;
    /// Block scales of the synthetic tables: f16 bits and the value, each
    /// exact, so a host dequantization is the card's.
    const SCALES: [(u16, f32); 4] = [
        (0x1C00, 1.0 / 256.0),
        (0x2000, 2.0 / 256.0),
        (0x2200, 3.0 / 256.0),
        (0x2400, 4.0 / 256.0),
    ];

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A Q8_0 table of `rows` rows of `k` values: the card's planes and the
    /// host's exact dequantization.
    struct Table {
        qs: Vec<u32>,
        d: Vec<u16>,
        deq: Vec<f32>,
        rows: usize,
        k: usize,
    }

    impl Table {
        fn new(rows: usize, k: usize, seed: u64) -> Table {
            let mut st = seed;
            let (mut qs, mut d, mut deq) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..rows * (k / 32) {
                let (bits, scale) = SCALES[(next(&mut st) % 4) as usize];
                d.push(bits);
                let codes: Vec<i8> = (0..32)
                    .map(|_| ((next(&mut st) % 255) as i32 - 127) as i8)
                    .collect();
                deq.extend(codes.iter().map(|&c| f32::from(c) * scale));
                for w in codes.as_chunks::<4>().0 {
                    qs.push(u32::from_le_bytes([
                        w[0] as u8, w[1] as u8, w[2] as u8, w[3] as u8,
                    ]));
                }
            }
            Table {
                qs,
                d,
                deq,
                rows,
                k,
            }
        }

        fn rows(&self, gpu: &Gpu) -> Q8Rows {
            let s = gpu.stream();
            Q8Rows {
                qs: DeviceTensor::upload(s, &self.qs, self.rows, self.k / 4).expect("quants"),
                d: DeviceTensor::upload(s, &self.d, self.rows, self.k / 32).expect("scales"),
            }
        }

        fn row(&self, id: usize) -> &[f32] {
            &self.deq[id * self.k..(id + 1) * self.k]
        }
    }

    /// The three tables of a selector over `n` ids, on the host and the card.
    struct Tables {
        hidden: Table,
        pred: Table,
        succ: Table,
    }

    impl Tables {
        fn new(n: usize) -> Tables {
            Tables {
                hidden: Table::new(RANK, HIDDEN, 11),
                pred: Table::new(n, RANK, 22),
                succ: Table::new(n, RANK, 33),
            }
        }

        fn tail(&self, gpu: &Gpu) -> SelectorTail {
            SelectorTail {
                hidden: self.hidden.rows(gpu),
                pred: self.pred.rows(gpu),
                succ: self.succ.rows(gpu),
                top_k: TOP_K,
            }
        }
    }

    /// Logits `x[v · m + c]` over `n` ids that tie a lot: values on a grid of
    /// quarters, and row `c` carries `c`-dependent outliers, some rows so few
    /// that the top 16 ends inside a tie.
    fn logits(n: usize, m: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        let mut x: Vec<f32> = (0..n * m)
            .map(|_| (next(&mut st) % 64) as f32 / 4.0 - 8.0)
            .collect();
        for c in 0..m {
            let outliers = if c % 2 == 0 { 10 } else { 20 };
            for q in 0..outliers.min(n) {
                let v = (next(&mut st) as usize) % n;
                x[v * m + c] = 20.0 + q as f32;
            }
        }
        x
    }

    fn normed(m: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        (0..m * HIDDEN)
            .map(|_| (next(&mut st) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// The host's lattice of the selector's last block, the card's order: the
    /// candidates of each row from `logits` by the host's `top_k`, each scored
    /// with [`lane_dot`] over the card's gate.
    fn host_lattice(
        t: &Tables,
        x: &[f32],
        gate: &[f32],
        n: usize,
        m: usize,
        anchor: u32,
    ) -> (Vec<Vec<(u32, f32)>>, Lattice) {
        let cands: Vec<Vec<(u32, f32)>> = (1..m)
            .map(|r| {
                let row: Vec<f32> = (0..n).map(|v| x[v * m + r]).collect();
                top_k(&row, TOP_K, r).expect("finite logits")
            })
            .collect();
        let mut rows = Vec::new();
        for (rho, cs) in cands.iter().enumerate() {
            let r = rho + 1;
            let preds: Vec<u32> = if rho == 0 {
                vec![anchor]
            } else {
                cands[rho - 1].iter().map(|c| c.0).collect()
            };
            let mut scores = Vec::new();
            for &p in &preds {
                let cond: Vec<f32> = (0..RANK)
                    .map(|e| t.pred.row(p as usize)[e] * gate[r * RANK + e])
                    .collect();
                for &(id, unary) in cs {
                    scores.push(lane_dot(t.succ.row(id as usize), &cond).expect("rank") + unary);
                }
            }
            rows.push(runtime::selector::Scored {
                ids: cs.iter().map(|c| c.0).collect(),
                scores,
            });
        }
        (cands, Lattice { rows })
    }

    /// The candidates are the host's `top_k` (the highest first, the lower id
    /// among equals) and every score is the host twin's, bit for bit, over
    /// ids that end a tile short, fill tiles exactly, tie across the cut and
    /// need one part only.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_selector_candidates_and_scores_are_the_host_twins() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        for (n, m) in [
            (3037, 8),
            (3037, 2),
            (40, 5),
            (1024, 8),
            (1025, 3),
            (2049, 7),
        ] {
            let case = format!("{n} ids, {m} rows");
            let t = Tables::new(n);
            let tail = t.tail(&gpu);
            let mut sel = Selector::new(gpu.context(), stream, &tail, n, HIDDEN, MAX_ROWS)
                .expect("the selector");
            let x = logits(n, m, 0x5eed + n as u64);
            let anchor = (n / 3) as u32;
            let (xb, nb) = (
                DeviceBuffer::from_host(stream, &x).expect("logits"),
                DeviceBuffer::from_host(stream, &normed(m, 7)).expect("normed rows"),
            );
            let first = DeviceBuffer::from_host(stream, &[anchor]).expect("anchor");
            let mut done = DeviceBuffer::zeroed(stream, 1).expect("ticket");
            sel.enqueue(
                &gpu,
                &tail,
                SelectArgs {
                    normed: &nb,
                    logits: &xb,
                    first: &first,
                    done: &mut done,
                    m,
                    fault: gpu.unlabelled_sink(),
                },
            )
            .expect("the launches");
            let words = sel.lattice(stream, m).expect("no fault");
            let (gate, unary) = (
                sel.gate(stream, m).expect("gate"),
                sel.unary(stream, m).expect("unary"),
            );
            assert_eq!(
                done.to_host_vec(stream).expect("ticket"),
                [0],
                "{case}: ticket"
            );
            let (cands, host) = host_lattice(&t, &x, &gate, n, m, anchor);
            for (rho, cs) in cands.iter().enumerate() {
                let row = &words[rho * ROW_WORDS..(rho + 1) * ROW_WORDS];
                let want_ids: Vec<u32> = cs.iter().map(|c| c.0).collect();
                assert_eq!(row[..TOP_K], want_ids[..], "{case}: row {} ids", rho + 1);
                for (i, c) in cs.iter().enumerate() {
                    assert_eq!(
                        unary[rho * TOP_K + i].to_bits(),
                        c.1.to_bits(),
                        "{case}: row {} logit {i}",
                        rho + 1
                    );
                }
                let preds = if rho == 0 { 1 } else { TOP_K };
                let want: Vec<u32> = host.rows[rho].scores.iter().map(|s| s.to_bits()).collect();
                assert_eq!(
                    row[TOP_K..TOP_K + preds * TOP_K],
                    want[..],
                    "{case}: row {} scores",
                    rho + 1
                );
            }
        }
    }

    /// A logit that is not finite raises `Logit` and an id past a table's
    /// rows `TokenId`; each is a fault in the readback, not a lattice.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_selector_faults_by_name() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let (n, m) = (3037, 4);
        let t = Tables::new(n);
        let tail = t.tail(&gpu);
        let mut sel =
            Selector::new(gpu.context(), stream, &tail, n, HIDDEN, MAX_ROWS).expect("selector");
        let nb = DeviceBuffer::from_host(stream, &normed(m, 7)).expect("normed rows");
        let mut done = DeviceBuffer::zeroed(stream, 1).expect("ticket");
        for (what, site) in [("logit", FaultSite::Logit), ("anchor", FaultSite::TokenId)] {
            let mut x = logits(n, m, 9);
            let mut anchor = 5u32;
            if what == "logit" {
                x[17 * m + 2] = f32::NAN;
            } else {
                anchor = n as u32;
            }
            let xb = DeviceBuffer::from_host(stream, &x).expect("logits");
            let first = DeviceBuffer::from_host(stream, &[anchor]).expect("anchor");
            sel.enqueue(
                &gpu,
                &tail,
                SelectArgs {
                    normed: &nb,
                    logits: &xb,
                    first: &first,
                    done: &mut done,
                    m,
                    fault: gpu.unlabelled_sink(),
                },
            )
            .expect("the launches");
            let Err(GpuError::Fault { fault, .. }) = sel.lattice(stream, m) else {
                panic!("{what}: no fault in the readback");
            };
            assert_eq!(fault.site(), Some(site), "{what}");
            gpu.clear_fault().expect("clear");
        }
    }

    /// A selector head runs end to end: its tokens are the host's walk over
    /// the card's own logits and gate, the probabilities the walk's softmax,
    /// row 0 the anchor with no id.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_selector_head_returns_the_host_walk() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let (n, m) = (3037, 8);
        let t = Tables::new(n);
        let own = Table::new(n, HIDDEN, 44);
        let gain = vec![1.0f32; HIDDEN];
        let weights = || HeadWeights {
            norm: DeviceTensor::upload(stream, &gain, 1, HIDDEN).expect("norm"),
            eps: 1e-6,
            source: HeadSource::Own(own.rows(&gpu)),
            tail: HeadTail::Selector(t.tail(&gpu)),
        };
        let full = DraftVocab::Full { n };
        let want = HeadKind::Selector {
            rank: RANK,
            top_k: TOP_K,
        };
        let w = weights();
        assert_eq!(w.kind(), want);
        w.check_against(&gpu, &want, &full)
            .expect("the selector's own shape");
        for (kind, vocab) in [
            (
                HeadKind::Markov {
                    rank: 256,
                    confidence: false,
                },
                full.clone(),
            ),
            (
                HeadKind::Selector {
                    rank: RANK,
                    top_k: 8,
                },
                full,
            ),
            (
                want,
                DraftVocab::Reduced {
                    to_target: vec![0],
                    n_target: n,
                },
            ),
        ] {
            assert!(
                w.check_against(&gpu, &kind, &vocab).is_err(),
                "{kind:?} over {vocab:?}"
            );
        }
        let mut head = Head::new(&gpu, w, HIDDEN, MAX_ROWS).expect("the head");
        assert_eq!(head.kind(), want);
        assert_eq!(head.launches_of(m), Head::LOGIT_LAUNCHES + LAUNCHES);
        let x: Vec<f32> = normed(m, 3);
        let xb = DeviceBuffer::from_host(stream, &x).expect("rows");
        let anchor = 123u32;
        let first = DeviceBuffer::from_host(stream, &[anchor]).expect("anchor");
        head.enqueue_logits(&gpu, &xb, m).expect("projection");
        head.enqueue_tail(&gpu, m, &first).expect("selector");
        head.enqueue_p(&gpu, m).expect("no launch");
        let (ids, p) = head.tokens_p(stream, m).expect("tokens");
        assert_eq!(head.tokens(stream, m).expect("tokens"), ids);
        let logits = head.logits(stream, m).expect("logits");
        let gate = head
            .selector()
            .expect("a selector")
            .gate(stream, m)
            .expect("gate");
        let (_, host) = host_lattice(&t, &logits, &gate, n, m, anchor);
        let walk = walk_p(&host, 0.0).expect("walk");
        assert_eq!(walk.len(), m - 1);
        assert_eq!((ids[0], p[0].is_nan()), (u32::MAX, true));
        for (r, step) in walk.iter().enumerate() {
            assert_eq!(ids[r + 1], step.id, "row {}", r + 1);
            assert_eq!(p[r + 1].to_bits(), step.p.to_bits(), "row {}", r + 1);
        }
    }
}
