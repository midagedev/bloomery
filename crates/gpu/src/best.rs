//! A pass's rows read as their best logits: for each of a head's `m` rows
//! (`x[v · m + r]`, the head's own layout), the [`BEST`] largest logits and
//! their ids, so a pick that reads only a row's best (a sampler's `top_k`
//! after its penalties) reads [`ROW_WORDS`] words a row instead of the
//! vocabulary's f32 row.
//!
//! Two launches:
//!
//! - [`best_kernels::best_part`] — each block of 256 threads takes [`TILE`]
//!   ids of every row (the tile in shared memory, read coalesced), and warp
//!   `r` finds row `r`'s [`BEST`] best of them by [`BEST`] rounds of a warp
//!   tournament ([`warp_best`]) and the tile's first NaN id;
//! - [`best_kernels::best_merge`] — one block a row: each of its eight warps
//!   takes its share of the parts' lists down to [`BEST`] in batches, a
//!   running list carried from batch to batch, then warp 0 takes the eight
//!   lists down to the row's; the row's first NaN id is the least of its
//!   parts'.
//!
//! The order is the engine's greedy rule ([`argmax_take`]): the larger
//! logit first, the lower id among equals (`−0` and `+0` tie), so the list
//! is the host's best-first order of the row and its first entry the row's
//! argmax. A NaN never wins: it reads as `−∞` in the list, and the row
//! carries the first NaN id for the reader to refuse by name. A row of fewer
//! than [`BEST`] ids ends its list with ids `u32::MAX`, which name no id.
//!
//! Row `r`'s words in the readback: the [`BEST`] logits' bits, then their
//! ids, then the first NaN id (`u32::MAX` for none).

use crate::elem::argmax_take;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;

const WHAT: &str = "best";

/// The logits a row's list holds.
pub const BEST: usize = 64;
/// The words of a row in the readback: the logits' bits, the ids, the first
/// NaN id.
pub const ROW_WORDS: usize = 2 * BEST + 1;
/// The most rows one launch reads: a pass's rows.
pub(crate) const MAX_ROWS: usize = crate::model::MAX_PASS_ROWS;

const THREADS: u32 = 256;
const WARPS: usize = THREADS as usize / 32;
/// Ids of every row a part block takes.
const TILE: usize = 1024;
const TILE_LANE: usize = TILE / 32;
const TILE_FLOATS: usize = TILE * MAX_ROWS;
/// New entries a lane takes into each merge batch, beside the running list's
/// [`RUN_LANE`].
const BATCH_LANE: usize = 16;
const RUN_LANE: usize = BEST / 32;
const MERGE_LANE: usize = BATCH_LANE + RUN_LANE;
/// The eight warps' lists, which warp 0 takes down to the row's.
const LISTS: usize = WARPS * BEST;
const LISTS_LANE: usize = LISTS / 32;

const _: () = assert!(TILE.is_multiple_of(32) && BEST.is_multiple_of(32));
const _: () = assert!(MAX_ROWS <= WARPS);

// ------------------------------------------------------------------ cores

/// The `K` best of the 32 lanes' `PER` entries each, `(v[j], id[j])` of lane
/// `l`, by `K` rounds of a tournament: every lane's best by [`argmax_take`],
/// the xor butterfly merging them to every lane, lane 0 writing the round's
/// pair to slot `t` of `out_v` and `out_i`, and the owner retiring its entry
/// as `(−∞, u32::MAX)`, below every entry that names an id. A warp with fewer
/// than `K` finite entries ends its list with −∞ slots. All 32 lanes call it
/// together; the selector head's top-k ([`crate::dspark::selector`]) takes
/// its lists by it too.
///
/// # Safety
///
/// `out_v` and `out_i` address `K` writable slots that no other thread writes
/// during the call.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
pub(crate) unsafe fn warp_best<const PER: usize, const K: usize>(
    mut v: [f32; PER],
    mut id: [u32; PER],
    out_v: *mut f32,
    out_i: *mut u32,
) {
    let lane = warp::lane_id();
    let mut t = 0usize;
    while t < K {
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = u32::MAX;
        for j in 0..PER {
            thread::__unroll_config::<0>();
            if argmax_take(v[j], id[j], best_v, best_i) {
                best_v = v[j];
                best_i = id[j];
            }
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (
                warp::shuffle_xor_f32(best_v, off),
                warp::shuffle_xor(best_i, off),
            );
            if argmax_take(ov, oi, best_v, best_i) {
                best_v = ov;
                best_i = oi;
            }
            off >>= 1;
        }
        if lane == 0 {
            // SAFETY: t < K slots of out_v, lane 0 alone writes.
            unsafe { *out_v.add(t) = best_v };
            // SAFETY: t < K slots of out_i, lane 0 alone writes.
            unsafe { *out_i.add(t) = best_i };
        }
        for j in 0..PER {
            thread::__unroll_config::<0>();
            if id[j] == best_i {
                v[j] = f32::NEG_INFINITY;
                id[j] = u32::MAX;
            }
        }
        t += 1;
    }
}

/// The least of the 32 lanes' `x`, on every lane.
#[inline(always)]
fn warp_min(mut x: u32) -> u32 {
    let mut off = 16u32;
    while off > 0 {
        x = x.min(warp::shuffle_xor(x, off));
        off >>= 1;
    }
    x
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod best_kernels {
    use super::*;

    /// Part `blockIdx.x` of every row `r < m`: the [`BEST`] best of the
    /// [`TILE`] ids `part · TILE ..`, written to slot `(r · n_parts + part) ·
    /// BEST` of `cv` and `ci`, and the part's first NaN id to `nan[r ·
    /// n_parts + part]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m >= 1,
            m <= 8,
            n >= 1,
            n_parts >= 1,
            n_parts * 1024 >= n,
            x.len() >= n * m,
            cv.len() >= m * n_parts * 64,
            ci.len() >= m * n_parts * 64,
            nan.len() >= m * n_parts
        )
    )]
    pub fn best_part(
        x: &[f32],
        n: u32,
        m: u32,
        n_parts: u32,
        mut cv: DisjointSlice<f32>,
        mut ci: DisjointSlice<u32>,
        mut nan: DisjointSlice<u32>,
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
            let v = if base + i / m < n {
                // SAFETY: id = base + i / m < n, so id · m + i % m = base · m
                // + i < n · m <= x.len() by the launch contract.
                unsafe { *x.get_unchecked(base * m + i) }
            } else {
                f32::NEG_INFINITY
            };
            // SAFETY: i < TILE · m <= TILE_FLOATS; slot i is this thread's
            // alone (i is tid mod THREADS).
            unsafe { *tile.add(i) = v };
            i += THREADS as usize;
        }
        thread::sync_threads();
        let w = tid / 32;
        if w >= m {
            return; // warp-uniform; no barrier follows
        }
        let lane = tid % 32;
        let mut v = [0.0f32; TILE_LANE];
        let mut id = [0u32; TILE_LANE];
        let mut first_nan = u32::MAX;
        for j in 0..TILE_LANE {
            thread::__unroll_config::<0>();
            let at = base + lane + 32 * j;
            // SAFETY: (lane + 32j) · m + w < TILE · m <= TILE_FLOATS, a slot
            // written above and visible past the barrier.
            let x = unsafe { *tile.add((lane + 32 * j) * m + w) };
            let named = at < n;
            if named && x.is_nan() {
                first_nan = first_nan.min(at as u32);
            }
            v[j] = if x.is_nan() { f32::NEG_INFINITY } else { x };
            id[j] = if named { at as u32 } else { u32::MAX };
        }
        let first_nan = warp_min(first_nan);
        let slot = w * n_parts as usize + part as usize;
        if lane == 0 {
            // SAFETY: slot < m · n_parts <= nan.len() by the launch contract;
            // the slot of (row w, this part) is this lane's alone.
            unsafe { *nan.get_unchecked_mut(slot) = first_nan };
        }
        let (out_v, out_i) = (
            cv.as_mut_ptr().wrapping_add(slot * BEST),
            ci.as_mut_ptr().wrapping_add(slot * BEST),
        );
        // SAFETY: slot · BEST + BEST <= m · n_parts · BEST <= cv.len(),
        // ci.len() by the launch contract; the slots of (row w, this part)
        // are this warp's alone.
        unsafe { warp_best::<TILE_LANE, BEST>(v, id, out_v, out_i) };
    }

    /// Row `blockIdx.x`'s merge: its `n_parts` lists of [`BEST`] into the
    /// row's [`BEST`], written as the module doc lays a row at `out[r ·
    /// ROW_WORDS ..]`, and the least of its parts' first NaN ids after them.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m >= 1,
            m <= 8,
            n_parts >= 1,
            cv.len() >= m * n_parts * 64,
            ci.len() >= m * n_parts * 64,
            nan.len() >= m * n_parts,
            out.len() >= m * 129
        )
    )]
    pub fn best_merge(
        cv: &[f32],
        ci: &[u32],
        nan: &[u32],
        n_parts: u32,
        m: u32,
        mut out: DisjointSlice<u32>,
    ) {
        static mut LIST_V: SharedArray<f32, LISTS> = SharedArray::UNINIT;
        static mut LIST_I: SharedArray<u32, LISTS> = SharedArray::UNINIT;
        static mut NAN: SharedArray<u32, WARPS> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        if row >= m as usize {
            return; // block-uniform, before any barrier
        }
        let tid = thread::threadIdx_x() as usize;
        let (w, lane) = (tid / 32, tid % 32);
        let parts = n_parts as usize;
        let n_c = parts * BEST;
        // SAFETY: LIST_V is this block's own shared allocation.
        let lv = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LIST_V) };
        // SAFETY: LIST_I is this block's own shared allocation.
        let li = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LIST_I) };
        // SAFETY: NAN is this block's own shared allocation.
        let ln = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NAN) };
        let mut first_nan = u32::MAX;
        let mut p = tid;
        while p < parts {
            // SAFETY: row < m and p < n_parts, inside nan.len() >= m · n_parts.
            first_nan = first_nan.min(unsafe { *nan.get_unchecked(row * parts + p) });
            p += THREADS as usize;
        }
        let first_nan = warp_min(first_nan);
        // The warp's running list: its BEST slots of the lists (w < WARPS, so
        // w · BEST + BEST <= LISTS), no entry yet.
        let (run_v, run_i) = (lv.wrapping_add(w * BEST), li.wrapping_add(w * BEST));
        for j in 0..RUN_LANE {
            thread::__unroll_config::<0>();
            // SAFETY: lane + 32j < BEST slots of this warp's list of logits.
            unsafe { *run_v.add(lane + 32 * j) = f32::NEG_INFINITY };
            // SAFETY: lane + 32j < BEST slots of this warp's list of ids.
            unsafe { *run_i.add(lane + 32 * j) = u32::MAX };
        }
        if lane == 0 {
            // SAFETY: w < WARPS slots of NAN, one a warp.
            unsafe { *ln.add(w) = first_nan };
        }
        let share = n_c.div_ceil(WARPS);
        let (lo, hi) = (w * share, ((w + 1) * share).min(n_c));
        let mut at = lo;
        while at < hi {
            warp::sync_mask(u32::MAX);
            let mut v = [f32::NEG_INFINITY; MERGE_LANE];
            let mut id = [u32::MAX; MERGE_LANE];
            for j in 0..BATCH_LANE {
                thread::__unroll_config::<0>();
                let e = at + lane + 32 * j;
                if e < hi {
                    // SAFETY: row < m and e < n_c, so row · n_c + e < m ·
                    // n_parts · BEST <= cv.len().
                    v[j] = unsafe { *cv.get_unchecked(row * n_c + e) };
                    // SAFETY: the same index, inside ci.len() alike.
                    id[j] = unsafe { *ci.get_unchecked(row * n_c + e) };
                }
            }
            for j in 0..RUN_LANE {
                thread::__unroll_config::<0>();
                // SAFETY: lane + 32j < BEST slots of this warp's list of
                // logits, written by lane 0 before the sync above.
                v[BATCH_LANE + j] = unsafe { *run_v.add(lane + 32 * j) };
                // SAFETY: the same slot of the list of ids.
                id[BATCH_LANE + j] = unsafe { *run_i.add(lane + 32 * j) };
            }
            // Every lane holds its entries before lane 0's first write: the
            // tournament's first shuffle waits for all 32.
            // SAFETY: the warp's BEST slots of the lists are its own.
            unsafe { warp_best::<MERGE_LANE, BEST>(v, id, run_v, run_i) };
            at += 32 * BATCH_LANE;
        }
        thread::sync_threads();
        if w != 0 {
            return;
        }
        let mut v = [f32::NEG_INFINITY; LISTS_LANE];
        let mut id = [u32::MAX; LISTS_LANE];
        for j in 0..LISTS_LANE {
            thread::__unroll_config::<0>();
            let e = lane + 32 * j;
            // SAFETY: e < LISTS slots of the logits, written by the warps
            // above and visible past the barrier.
            v[j] = unsafe { *lv.add(e) };
            // SAFETY: the same slot of the ids.
            id[j] = unsafe { *li.add(e) };
        }
        let warp_nan = if lane < WARPS {
            // SAFETY: lane < WARPS slots of NAN, written before the barrier.
            unsafe { *ln.add(lane) }
        } else {
            u32::MAX
        };
        let row_nan = warp_min(warp_nan);
        let words = out.as_mut_ptr().wrapping_add(row * ROW_WORDS);
        // SAFETY: row < m, so the row's ROW_WORDS words lie inside out.len()
        // >= m · ROW_WORDS and are this block's alone; the logits' slots take
        // f32 bits.
        unsafe {
            warp_best::<LISTS_LANE, BEST>(v, id, words.cast::<f32>(), words.wrapping_add(BEST))
        };
        if lane == 0 {
            // SAFETY: word 2 · BEST < ROW_WORDS of the row, lane 0 alone writes.
            unsafe { *words.add(2 * BEST) = row_nan };
        }
    }
}

// ------------------------------------------------------------------- host

/// The loaded launches and their scratch for one vocabulary: [`MAX_ROWS`]
/// rows' parts and the readback.
pub(crate) struct Best {
    module: best_kernels::LoadedModule,
    n: usize,
    parts: usize,
    cand_v: DeviceBuffer<f32>,
    cand_i: DeviceBuffer<u32>,
    nan: DeviceBuffer<u32>,
    out: DeviceBuffer<u32>,
}

impl Best {
    /// The launches over rows of `n_vocab` logits, their scratch allocated
    /// once. Made by the first best read of a model
    /// ([`crate::model::GpuModel::pass_best`]),
    /// not at load: that read loads the module and allocates the scratch.
    pub(crate) fn new(gpu: &Gpu, n_vocab: usize) -> Result<Best, GpuError> {
        if n_vocab == 0 {
            return Err(GpuError::shape(WHAT, "a vocabulary of no ids".to_owned()));
        }
        let parts = n_vocab.div_ceil(TILE);
        let stream = gpu.stream();
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(best_kernels, gpu.context())? };
        Ok(Best {
            module,
            n: n_vocab,
            parts,
            cand_v: DeviceBuffer::zeroed(stream, MAX_ROWS * parts * BEST)?,
            cand_i: DeviceBuffer::zeroed(stream, MAX_ROWS * parts * BEST)?,
            nan: DeviceBuffer::zeroed(stream, MAX_ROWS * parts)?,
            out: DeviceBuffer::zeroed(stream, MAX_ROWS * ROW_WORDS)?,
        })
    }

    /// Enqueue the best of `m` rows of `logits` (`[v · m + r]`, `n_vocab · m`
    /// f32): the two launches, on `gpu`'s stream. Refused by name for `m`
    /// outside `1..=MAX_ROWS` and for logits of another length.
    pub(crate) fn enqueue(
        &mut self,
        gpu: &Gpu,
        logits: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        if !(1..=MAX_ROWS).contains(&m) || logits.len() != self.n * m {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{m} rows of {} logits for rows of {} (1..={MAX_ROWS} rows)",
                    logits.len(),
                    self.n
                ),
            ));
        }
        let s: &CudaStream = gpu.stream();
        let (n, mm, parts) = (
            launch_u32(WHAT, "n", self.n)?,
            launch_u32(WHAT, "m", m)?,
            launch_u32(WHAT, "n_parts", self.parts)?,
        );
        let prep = self
            .module
            .prepare_best_part(LaunchConfig1D::new(parts, THREADS, 0))?;
        self.module.best_part(
            s,
            &prep,
            logits,
            n,
            mm,
            parts,
            &mut self.cand_v,
            &mut self.cand_i,
            &mut self.nan,
        )?;
        let prep = self
            .module
            .prepare_best_merge(LaunchConfig1D::new(mm, THREADS, 0))?;
        self.module.best_merge(
            s,
            &prep,
            &self.cand_v,
            &self.cand_i,
            &self.nan,
            parts,
            mm,
            &mut self.out,
        )?;
        Ok(())
    }

    /// The readback of the last [`Best::enqueue`] into `words`, which holds
    /// [`MAX_ROWS`] · [`ROW_WORDS`] words, row `r` at `r · ROW_WORDS`.
    /// Blocking read.
    pub(crate) fn read(&self, gpu: &Gpu, words: &mut [u32]) -> Result<(), GpuError> {
        if words.len() != self.out.len() {
            return Err(GpuError::shape(
                WHAT,
                format!("a readback of {} words for {}", words.len(), self.out.len()),
            ));
        }
        self.out.copy_to_host(gpu.stream(), words)?;
        Ok(())
    }
}

/// Row `r`'s list in a readback (`r · ROW_WORDS ..`): its `(id, logit)`
/// pairs best first, without the ids `u32::MAX` that end a short row's list.
pub fn row_pairs(row: &[u32]) -> impl Iterator<Item = (u32, f32)> + '_ {
    row[BEST..2 * BEST]
        .iter()
        .zip(&row[..BEST])
        .filter(|(id, _)| **id != u32::MAX)
        .map(|(&id, &bits)| (id, f32::from_bits(bits)))
}

/// Row `r`'s first NaN id in a readback, `None` for a row with none.
#[must_use]
pub fn row_nan(row: &[u32]) -> Option<u32> {
    let id = row[2 * BEST];
    (id != u32::MAX).then_some(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of `n` logits a seed draws, with runs of equal values (ties
    /// broken by id), `−0` beside `+0`, and an infinity or two.
    fn row(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let k = (s >> 33) as u32;
                match k % 97 {
                    0 => 7.25,
                    1 => -0.0,
                    2 => 0.0,
                    3 if i % 5000 == 3 => f32::INFINITY,
                    4 if i % 7000 == 4 => f32::NEG_INFINITY,
                    _ => (k % 20_000) as f32 / 1000.0 - 10.0,
                }
            })
            .collect()
    }

    /// The host's list of a row: NaN as `−∞`, the larger first, the lower id
    /// among equals, at most [`BEST`].
    fn host(row: &[f32]) -> Vec<(u32, u32)> {
        let mut all: Vec<(u32, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, if v.is_nan() { f32::NEG_INFINITY } else { v }))
            .collect();
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("no NaN left")
                .then(a.0.cmp(&b.0))
        });
        all.truncate(BEST);
        all.into_iter().map(|(i, v)| (i, v.to_bits())).collect()
    }

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_best_lists_are_the_host_lists() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        for (n, m) in [
            (248_320, 8),
            (248_320, 1),
            (151_936, 5),
            (3_037, 3),
            (1_024, 2),
            (1_025, 7),
            (40, 4),
        ] {
            let case = format!("{n} ids, {m} rows");
            let rows: Vec<Vec<f32>> = (0..m).map(|r| row(n, 0x5eed + r as u64)).collect();
            let mut x = vec![0.0f32; n * m];
            for (r, row) in rows.iter().enumerate() {
                for (v, &l) in row.iter().enumerate() {
                    x[v * m + r] = l;
                }
            }
            let nan_at = (n > 100).then_some((n / 3, n / 2));
            if let Some((a, b)) = nan_at {
                x[b * m + m - 1] = f32::NAN;
                x[a * m + m - 1] = f32::NAN;
            }
            let mut best = Best::new(&gpu, n).expect("the launches");
            let xb = DeviceBuffer::from_host(stream, &x).expect("logits");
            best.enqueue(&gpu, &xb, m).expect("enqueued");
            let mut words = vec![0u32; MAX_ROWS * ROW_WORDS];
            best.read(&gpu, &mut words).expect("read back");
            for (r, row) in rows.iter().enumerate() {
                let mut row = row.clone();
                let mut nan = None;
                if r == m - 1
                    && let Some((a, b)) = nan_at
                {
                    (row[a], row[b], nan) = (f32::NAN, f32::NAN, Some(a as u32));
                }
                let got = &words[r * ROW_WORDS..(r + 1) * ROW_WORDS];
                let pairs: Vec<(u32, u32)> =
                    row_pairs(got).map(|(id, v)| (id, v.to_bits())).collect();
                assert_eq!(pairs, host(&row), "{case}: row {r}");
                assert_eq!(row_nan(got), nan, "{case}: row {r}'s NaN");
            }
        }
        let mut best = Best::new(&gpu, 40).expect("the launches");
        let xb = DeviceBuffer::from_host(stream, &[0.0f32; 40 * 9]).expect("logits");
        assert!(best.enqueue(&gpu, &xb, 9).is_err(), "nine rows are refused");
    }
}
