//! The DSpark draft's Markov head: after the draft head's logits, position
//! `i` of a block adds the bigram correction of its previous token `p`,
//! `logits_i[v] += markov_w2[v] · markov_w1[p]`, and its argmax is position
//! `i + 1`'s previous token. Serial over `i` by construction.
//!
//! Layouts. Both weights are the file's bf16 `[rank, n_vocab]` as u32 words:
//! token `t`'s rank-long row starts at word `t * rank / 2`, value `r` in word
//! `r / 2`, the low half for even `r` (little-endian). The logits are the
//! head gemv's `m`-row layout, `logits[v * m + c]` for row `c` (one row is
//! `m = 1`). A position's previous token is read on the device: row 0's from
//! `first[0]` (the block's last accepted token), row `c > 0`'s from `tok[c -
//! 1]`, the argmax the step wrote for row `c - 1`. An id at or past
//! `n_vocab` reads row 0 — in bounds, deterministic, and never the argmax's.
//!
//! The numeric rule `ds41_markov` holds, and the gate transcribes on the host:
//! bf16 widens to f32 as `bits << 16` (exact), and the dot for entry `v` is
//! `markov-accept`'s order: eight accumulators, `a[l] = fma(w2[v][8j + l],
//! e[8j + l], a[l])` from 0 over the octets `j` ascending, then
//! `((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))`; the result is added to
//! the logit once, `logit + delta`. One thread per `v`.
//!
//! A step is `ds41_markov` then `bloomery_gpu`'s `argmax_rows_fault` over
//! all `m` rows ([`MarkovKernels::enqueue_step`]): rows below `c` are
//! unchanged since their own step and give the same tokens again; rows above
//! `c` write a token their own step overwrites. The argmax copies the card's
//! fault word to `tok[m]` after the tokens, so the last step's readback of
//! `tok[..=m]` carries every fault the pass raised.

use bloomery_gpu::elem::ElemKernels;
use bloomery_gpu::{DeviceTensor, FaultSink, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The Markov rank this kernel is built for (`markov_w1`'s row length).
pub const MARKOV_RANK: usize = 256;
/// u32 words of one rank-long bf16 row.
pub const MARKOV_ROW_WORDS: usize = MARKOV_RANK / 2;
const MARKOV_THREADS: u32 = 256;
const _: () = assert!(MARKOV_ROW_WORDS <= MARKOV_THREADS as usize && MARKOV_RANK.is_multiple_of(8));

/// The two bf16 values of word `w`, widened: value `2i` then `2i + 1`.
#[inline(always)]
fn bf16_pair(w: u32) -> (f32, f32) {
    (f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000))
}

/// The dot of `w2` row `v` with `e` in the module doc's order.
///
/// SAFETY: `w2.len() >= (v + 1) * MARKOV_ROW_WORDS` and `e` points at
/// [`MARKOV_RANK`] readable f32.
#[inline(always)]
unsafe fn markov_dot(w2: &[u32], v: usize, e: *const f32) -> f32 {
    let base = v * MARKOV_ROW_WORDS;
    let mut a = [0.0f32; 8];
    let mut j = 0usize;
    while j < MARKOV_RANK / 8 {
        // SAFETY: base + 4j + 3 < base + 128 <= w2.len(), and 8j + 7 < 256,
        // by this fn's contract.
        let (w0, w1, w2v, w3, e8) = unsafe {
            let o = base + 4 * j;
            (
                *w2.get_unchecked(o),
                *w2.get_unchecked(o + 1),
                *w2.get_unchecked(o + 2),
                *w2.get_unchecked(o + 3),
                e.add(8 * j),
            )
        };
        let (x0, x1) = bf16_pair(w0);
        let (x2, x3) = bf16_pair(w1);
        let (x4, x5) = bf16_pair(w2v);
        let (x6, x7) = bf16_pair(w3);
        // SAFETY: e8 .. e8 + 8 is inside e's MARKOV_RANK values.
        unsafe {
            a[0] = x0.mul_add(*e8, a[0]);
            a[1] = x1.mul_add(*e8.add(1), a[1]);
            a[2] = x2.mul_add(*e8.add(2), a[2]);
            a[3] = x3.mul_add(*e8.add(3), a[3]);
            a[4] = x4.mul_add(*e8.add(4), a[4]);
            a[5] = x5.mul_add(*e8.add(5), a[5]);
            a[6] = x6.mul_add(*e8.add(6), a[6]);
            a[7] = x7.mul_add(*e8.add(7), a[7]);
        }
        j += 1;
    }
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}

#[cuda_module]
mod markov_kernels {
    use super::*;

    /// Row `row`'s Markov correction (the module doc's rule): every block
    /// gathers the previous token's `w1` row into shared memory, widened, and
    /// thread `v` adds `w2[v] · e` into `logits[v * m + row]`.
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
            w1.len() >= 128 * n_vocab,
            w2.len() >= 128 * n_vocab,
            first.len() >= 1,
            tok.len() >= m,
            logits.len() >= n_vocab * m,
            row < m,
            n_vocab >= 1
        )
    )]
    pub fn ds41_markov(
        w1: &[u32],
        w2: &[u32],
        first: &[u32],
        tok: &[u32],
        n_vocab: u32,
        m: u32,
        row: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        static mut E: SharedArray<f32, MARKOV_RANK> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: E is this block's own shared allocation; thread tid <
        // 128 alone writes slots 2tid and 2tid + 1 before the barrier.
        let e = unsafe { SharedArray::as_raw_mut_ptr(&raw mut E) };
        if tid < MARKOV_ROW_WORDS {
            // SAFETY: first.len() >= 1 and row - 1 < m <= tok.len() by the
            // launch contract.
            let p = unsafe {
                if row == 0 {
                    *first.get_unchecked(0)
                } else {
                    *tok.get_unchecked(row as usize - 1)
                }
            };
            let p = if p < n_vocab { p as usize } else { 0 };
            // SAFETY: p < n_vocab, so word p*128 + tid < 128*n_vocab <=
            // w1.len(); the two slots are this thread's.
            unsafe {
                let (lo, hi) = bf16_pair(*w1.get_unchecked(p * MARKOV_ROW_WORDS + tid));
                *e.add(2 * tid) = lo;
                *e.add(2 * tid + 1) = hi;
            }
        }
        thread::sync_threads();
        let v = thread::index_1d().get();
        if v >= n_vocab as usize {
            return;
        }
        // SAFETY: v < n_vocab, w2.len() >= 128*n_vocab; e holds 256 values
        // written before the barrier.
        let delta = unsafe { markov_dot(w2, v, e) };
        let at = v * m as usize + row as usize;
        // SAFETY: at < n_vocab*m <= logits.len(); one thread per v.
        unsafe {
            let l = logits.get_unchecked_mut(at);
            *l += delta;
        }
    }
}

/// The Markov head's weights on the card, the file's bf16 bits as words.
pub struct MarkovWeights<'a> {
    /// `markov_w1`: `n_vocab` rows of [`MARKOV_ROW_WORDS`] words.
    pub w1: &'a DeviceTensor<u32>,
    /// `markov_w2`: the same shape.
    pub w2: &'a DeviceTensor<u32>,
}

/// The loaded Markov module. Owns no stream.
pub struct MarkovKernels {
    module: markov_kernels::LoadedModule,
}

impl MarkovKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MarkovKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { markov_kernels::load(ctx)? };
        Ok(MarkovKernels { module })
    }

    /// Enqueue row `row`'s correction (`ds41_markov`) into `logits` (`m`
    /// rows, `logits[v * m + c]`), its previous token `first[0]` for row 0
    /// and `tok[row - 1]` otherwise. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's buffers and its row, all distinct roles"
    )]
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        w: &MarkovWeights<'_>,
        first: &DeviceBuffer<u32>,
        tok: &DeviceBuffer<u32>,
        m: usize,
        row: usize,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "MarkovKernels::enqueue_add";
        let n = w.w2.rows();
        if w.w1.rows() != n
            || w.w1.cols() != MARKOV_ROW_WORDS
            || w.w2.cols() != MARKOV_ROW_WORDS
            || n == 0
            || first.is_empty()
            || m == 0
            || row >= m
            || tok.len() < m
            || logits.len() < n * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "w1 {}x{} w2 {}x{} (need n_vocab x {MARKOV_ROW_WORDS}), first {}, m {m} row {row}, \
                     tok {}, logits {} (need {})",
                    w.w1.rows(),
                    w.w1.cols(),
                    w.w2.rows(),
                    w.w2.cols(),
                    first.len(),
                    tok.len(),
                    logits.len(),
                    n * m
                ),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(MARKOV_THREADS as usize))?;
        let (nv, mm, r) = (
            launch_u32(what, "n_vocab", n)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "row", row)?,
        );
        let prep = self
            .module
            .prepare_ds41_markov(LaunchConfig1D::new(grid, MARKOV_THREADS, 0))?;
        self.module.ds41_markov(
            stream,
            &prep,
            w.w1.buf(),
            w.w2.buf(),
            first,
            tok,
            nv,
            mm,
            r,
            logits,
        )?;
        Ok(())
    }

    /// One Markov step: [`Self::enqueue_add`] for row `row`, then
    /// `argmax_rows_fault` over the `m` rows into `tok[..m]` and `fault`'s
    /// word into `tok[m]` (`tok` holds `m + 1`) — `tok[row]` is then row
    /// `row + 1`'s previous token. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one step's buffers, its row and the fault word, all distinct roles"
    )]
    pub fn enqueue_step(
        &self,
        stream: &CudaStream,
        elem: &ElemKernels,
        w: &MarkovWeights<'_>,
        first: &DeviceBuffer<u32>,
        tok: &mut DeviceBuffer<u32>,
        m: usize,
        row: usize,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_add(stream, w, first, tok, m, row, logits)?;
        elem.enqueue_argmax_rows_fault(stream, logits, w.w2.rows(), m, fault, tok)
    }
}
