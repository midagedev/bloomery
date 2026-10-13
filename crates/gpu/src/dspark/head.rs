//! The draft's head chain: the final norm, the projection
//! ([`super::HeadSource`]), the Markov loop over the block's positions and
//! the draft-to-target id map.
//!
//! [`Head`] launches the chain over a block's `m` folded rows: `rms_norm`
//! (the head's own gain and epsilon), then the projection, one launch each
//! over the `m` rows. The target's Q6_K matrix takes the shared q8_1
//! quantizer and its row kernel after the norm; the draft's own Q8_0 rows
//! take the norm's f32 rows straight into `q8f32`'s m-column row kernel
//! (`enqueue_q8_0_gemv_mcol`), so that head runs one launch fewer. Each row
//! is bit for bit its `m = 1` launch in both. The logits land in the gemv's
//! layout, `logits[v · m + c]`, which `ds41_markov`'s is: the Q8_0 kernel
//! writes row `v`'s column `c` at `v · y_row + c · y_col`, and
//! `GemvOut::RowMajor` is `(m, 1)`. The Markov loop is `m` serial steps
//! ([`MarkovKernels::enqueue_step`], or [`MarkovKernels::enqueue_step_reduced`]
//! for a reduced head): row 0's previous token is the block's first id, read
//! from the device where the pass's staging put it, row `c`'s the argmax the
//! step before wrote. The last step's argmax buffer holds the block's
//! proposal and, after it, the card's fault word, so [`Head::tokens`]' one
//! readback turns a fault any launch of the pass raised into
//! [`GpuError::Fault`], as [`crate::head::Head`] does for the target. The
//! probabilities of a proposal ([`Head::enqueue_p`]) are one launch of their
//! own, enqueued only for a caller that asks.
//!
//! Three heads run: the target's Q6_K matrix over the target's vocabulary
//! (V4.1's draft), the draft's own Q8_0 rows over the same vocabulary, and
//! the draft's own rows over a reduced one. A reduced head scores `n_head`
//! ids, `markov_w2` and the logits have `n_head` rows, `markov_w1` has the
//! target's `n_prev`, and [`HeadWeights::to_target`] gives each head id its
//! target id. The map is read only inside the Markov loop, where the rows
//! after the first index `markov_w1` by their previous token's target id. The
//! head returns head ids ([`Head::tokens`], [`Head::tokens_p`]); the owner
//! maps them to target ids. [`Head::n_vocab`] counts the logits' rows,
//! `n_head`. The target's Q8_0 matrix, and any map over the target's matrix
//! (no file needs one), are refused by name.
//!
//! # The Markov loop
//!
//! After the head's logits, position `i` of a block adds the bigram
//! correction of its previous token `p`, `logits_i[v] += markov_w2[v] ·
//! markov_w1[p]`, and its argmax is position `i + 1`'s previous token.
//! Serial over `i` by construction.
//!
//! Layouts. Both weights are the file's bf16 `[rank, rows]` as u32 words:
//! row `t` is `rank / 2` words long and starts at word `t * rank / 2`, value
//! `r` in word `r / 2`, the low half for even `r` (little-endian).
//! `markov_w1` has `n_prev` rows, one per previous token, in the target's
//! vocabulary; `markov_w2` has `n_head` rows, one per logit, in the head's.
//! V4.1's draft has one vocabulary (`n_prev = n_head`, [`MarkovWeights`]); a
//! reduced head has `n_head < n_prev` and a map `to_target` from each head id
//! to its target id ([`ReducedMarkovWeights`]). The logits are the head
//! gemv's `m`-row layout, `logits[v * m + c]` for row `c` and `v < n_head`
//! (one row is `m = 1`). A position's previous token is read on the device:
//! row 0's from `first[0]` (the block's last accepted token, a target id
//! already), row `c > 0`'s from `tok[c - 1]`, the argmax the step wrote for
//! row `c - 1`, a head id. With the map off that id is the `w1` row; with it
//! on the row is `to_target[tok[c - 1]]`, and row 0 never maps. A previous
//! token outside its table — a `w1` row at or past `n_prev`, and with the map
//! on a `tok[c - 1]` at or past `n_head` — raises `FaultSite::TokenId`,
//! reads no `w1` word, and makes the row's correction NaN, so every logit of
//! that row is NaN — no value the pass could take for a draft; the pass's
//! readback carries the fault. In a step only `first[0]` or a map entry can
//! be one: the argmax's are below `n_head`.
//!
//! The numeric rule `ds41_markov` holds, and the gate transcribes on the host:
//! bf16 widens to f32 as `bits << 16` (exact), and the dot for entry `v` is
//! `markov-accept`'s order: eight accumulators, `a[l] = fma(w2[v][8j + l],
//! e[8j + l], a[l])` from 0 over the octets `j` ascending, then
//! `((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))`; the result is added to
//! the logit once, `logit + delta`. One thread per `v`.
//!
//! A step is `ds41_markov` then the `argmax_rows_fault` of [`crate::elem`] over
//! all `m` rows of `n_head` ([`MarkovKernels::enqueue_step`],
//! [`MarkovKernels::enqueue_step_reduced`]): rows below `c` are
//! unchanged since their own step and give the same tokens again; rows above
//! `c` write a token their own step overwrites. The argmax copies the card's
//! fault to `tok[m]` (the first-layer word) and `tok[m + 1]` (its layer's
//! site mask) after the tokens, so the last step's readback of `tok[..m +
//! 2]` carries every fault the pass raised.

use crate::elem::ElemKernels;
use crate::mtp::{ArgmaxPArgs, MtpKernels};
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

use super::{HeadSource, Q8Rows, TargetHead};

const WHAT: &str = "dspark::head";

/// The head's weights on the card.
pub struct HeadWeights {
    /// The final norm's gain.
    pub norm: DeviceTensor<f32>,
    /// The final norm's epsilon.
    pub eps: f32,
    /// The projection: the target's matrix or the draft's own Q8_0 rows.
    pub source: HeadSource,
    /// The Markov term's input table, one row per target id (the previous
    /// token): bf16 values two to a word.
    pub markov_w1: DeviceTensor<u32>,
    /// The Markov term's output table, one row per head id: bf16 values two
    /// to a word.
    pub markov_w2: DeviceTensor<u32>,
    /// A reduced head's draft-to-target map on the card, `n_head` words: the
    /// Markov loop's rows after the first index `markov_w1` with it.
    pub to_target: Option<DeviceBuffer<u32>>,
    /// The confidence head's projection (hidden plus rank) and bias. Carried
    /// with the weights; no launch reads it.
    pub conf: Option<(DeviceTensor<f32>, f32)>,
}

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
    /// thread `v < n_head` adds `w2[v] · e` into `logits[v * m + row]`. With
    /// `map` on and `row > 0` the previous token `tok[row - 1]` is a head id
    /// and `to_target` gives its `w1` row; `map` off reads no `to_target`
    /// word, and any buffer stands in for it. A previous token outside its
    /// table raises [`FaultSite::TokenId`] on `fault` (thread 0 of the grid)
    /// and gathers NaN instead of a row.
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
            w1.len() >= 128 * n_prev,
            w2.len() >= 128 * n_head,
            to_target.len() >= n_head * map,
            first.len() >= 1,
            tok.len() >= m,
            logits.len() >= n_head * m,
            row < m,
            n_prev >= 1,
            n_head >= 1,
            map <= 1
        )
    )]
    pub fn ds41_markov(
        w1: &[u32],
        w2: &[u32],
        to_target: &[u32],
        first: &[u32],
        tok: &[u32],
        n_prev: u32,
        n_head: u32,
        m: u32,
        row: u32,
        map: u32,
        fault: FaultSink,
        mut logits: DisjointSlice<f32>,
    ) {
        static mut E: SharedArray<f32, MARKOV_RANK> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: E is this block's own shared allocation; thread tid <
        // 128 alone writes slots 2tid and 2tid + 1 before the barrier.
        let e = unsafe { SharedArray::as_raw_mut_ptr(&raw mut E) };
        if tid < MARKOV_ROW_WORDS {
            // The `w1` row: the previous token itself, or with the map on a
            // head id's target; `listed` is false for a head id past the map.
            //
            // SAFETY: first.len() >= 1 and row - 1 < m <= tok.len() by the
            // launch contract; to_target.len() >= n_head * map = n_head when
            // the map is on, and the read is of q < n_head.
            let (listed, p) = unsafe {
                let q = if row == 0 {
                    *first.get_unchecked(0)
                } else {
                    *tok.get_unchecked(row as usize - 1)
                };
                if map == 0 || row == 0 {
                    (true, q)
                } else if q < n_head {
                    (true, *to_target.get_unchecked(q as usize))
                } else {
                    (false, q)
                }
            };
            let (lo, hi) = if listed && p < n_prev {
                // SAFETY: p < n_prev, so word p*128 + tid < 128*n_prev <=
                // w1.len().
                bf16_pair(unsafe { *w1.get_unchecked(p as usize * MARKOV_ROW_WORDS + tid) })
            } else {
                if thread::index_1d().get() == 0 {
                    fault.raise(FaultSite::TokenId);
                }
                (f32::NAN, f32::NAN)
            };
            // SAFETY: the two slots are this thread's.
            unsafe {
                *e.add(2 * tid) = lo;
                *e.add(2 * tid + 1) = hi;
            }
        }
        thread::sync_threads();
        let v = thread::index_1d().get();
        if v >= n_head as usize {
            return;
        }
        // SAFETY: v < n_head, w2.len() >= 128*n_head; e holds 256 values
        // written before the barrier.
        let delta = unsafe { markov_dot(w2, v, e) };
        let at = v * m as usize + row as usize;
        // SAFETY: at < n_head*m <= logits.len(); one thread per v.
        unsafe {
            let l = logits.get_unchecked_mut(at);
            *l += delta;
        }
    }
}

/// The Markov head's weights on the card, the file's bf16 bits as words,
/// for a draft with one vocabulary (V4.1's): no map, `n_prev = n_head`.
pub struct MarkovWeights<'a> {
    /// `markov_w1`: `n_vocab` rows of [`MARKOV_ROW_WORDS`] words.
    pub w1: &'a DeviceTensor<u32>,
    /// `markov_w2`: the same shape.
    pub w2: &'a DeviceTensor<u32>,
}

/// The Markov head's weights of a reduced head, whose logits cover the head's
/// vocabulary but whose previous token is a target id.
pub struct ReducedMarkovWeights<'a> {
    /// `markov_w1`: `n_prev` rows of [`MARKOV_ROW_WORDS`] words, one per
    /// target id.
    pub w1: &'a DeviceTensor<u32>,
    /// `markov_w2`: `n_head` rows of [`MARKOV_ROW_WORDS`] words, one per head
    /// id (the logits' rows).
    pub w2: &'a DeviceTensor<u32>,
    /// Each head id's target id, `d + d2t[d]` in the file's terms: at least
    /// `n_head` words, the first `n_head` each below `n_prev`.
    pub to_target: &'a DeviceBuffer<u32>,
}

/// The tables of one launch; `to_target` is `None` for one vocabulary.
struct Tables<'a> {
    w1: &'a DeviceTensor<u32>,
    w2: &'a DeviceTensor<u32>,
    to_target: Option<&'a DeviceBuffer<u32>>,
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
        let module = unsafe { crate::shared_module!(markov_kernels, ctx)? };
        Ok(MarkovKernels { module })
    }

    /// Enqueue row `row`'s correction (`ds41_markov`) into `logits` (`m`
    /// rows, `logits[v * m + c]`), its previous token `first[0]` for row 0
    /// and `tok[row - 1]` otherwise; a previous token past the vocabulary
    /// raises [`FaultSite::TokenId`] on `fault`. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's buffers, its row and the fault word, all distinct roles"
    )]
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        w: &MarkovWeights<'_>,
        first: &DeviceBuffer<u32>,
        tok: &DeviceBuffer<u32>,
        m: usize,
        row: usize,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let t = Tables {
            w1: w.w1,
            w2: w.w2,
            to_target: None,
        };
        self.launch_add(
            "MarkovKernels::enqueue_add",
            stream,
            &t,
            first,
            tok,
            m,
            row,
            fault,
            logits,
        )
    }

    /// [`Self::enqueue_add`] for a reduced head: `w1` has `n_prev` rows and
    /// `w2` and the logits `n_head`, row `c > 0`'s previous token `tok[c - 1]`
    /// is a head id and indexes `w1` through `to_target`, and row 0's
    /// `first[0]` is a target id. A head id at or past `n_head`, or a target
    /// id at or past `n_prev`, raises [`FaultSite::TokenId`] on `fault`.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's buffers, its row and the fault word, all distinct roles"
    )]
    pub fn enqueue_add_reduced(
        &self,
        stream: &CudaStream,
        w: &ReducedMarkovWeights<'_>,
        first: &DeviceBuffer<u32>,
        tok: &DeviceBuffer<u32>,
        m: usize,
        row: usize,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let t = Tables {
            w1: w.w1,
            w2: w.w2,
            to_target: Some(w.to_target),
        };
        self.launch_add(
            "MarkovKernels::enqueue_add_reduced",
            stream,
            &t,
            first,
            tok,
            m,
            row,
            fault,
            logits,
        )
    }

    /// One launch of `ds41_markov`: `n_prev` is `w1`'s rows and `n_head`
    /// `w2`'s; without a map they are one vocabulary.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's buffers, its row and the fault word, all distinct roles"
    )]
    fn launch_add(
        &self,
        what: &'static str,
        stream: &CudaStream,
        t: &Tables<'_>,
        first: &DeviceBuffer<u32>,
        tok: &DeviceBuffer<u32>,
        m: usize,
        row: usize,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_prev, n_head) = (t.w1.rows(), t.w2.rows());
        let map_len = t.to_target.map(DeviceBuffer::len);
        if (map_len.is_none() && n_prev != n_head)
            || map_len.is_some_and(|len| len < n_head)
            || t.w1.cols() != MARKOV_ROW_WORDS
            || t.w2.cols() != MARKOV_ROW_WORDS
            || n_prev == 0
            || n_head == 0
            || first.is_empty()
            || m == 0
            || row >= m
            || tok.len() < m
            || logits.len() < n_head * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "w1 {}x{} w2 {}x{} (need n_prev x {MARKOV_ROW_WORDS}, n_head x {MARKOV_ROW_WORDS}; \
                     one vocabulary without a map), map {map_len:?} (need {n_head}), first {}, \
                     m {m} row {row}, tok {}, logits {} (need {})",
                    t.w1.rows(),
                    t.w1.cols(),
                    t.w2.rows(),
                    t.w2.cols(),
                    first.len(),
                    tok.len(),
                    logits.len(),
                    n_head * m
                ),
            });
        }
        let grid = launch_u32(what, "grid", n_head.div_ceil(MARKOV_THREADS as usize))?;
        let (np, nh, mm, r) = (
            launch_u32(what, "n_prev", n_prev)?,
            launch_u32(what, "n_head", n_head)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "row", row)?,
        );
        // Map off: the kernel reads no `to_target` word, so `first` stands in.
        let (map, to_target) = t.to_target.map_or((0, first), |d| (1, d));
        let prep = self
            .module
            .prepare_ds41_markov(LaunchConfig1D::new(grid, MARKOV_THREADS, 0))?;
        self.module.ds41_markov(
            stream,
            &prep,
            t.w1.buf(),
            t.w2.buf(),
            to_target,
            first,
            tok,
            np,
            nh,
            mm,
            r,
            map,
            fault,
            logits,
        )?;
        Ok(())
    }

    /// One Markov step: [`Self::enqueue_add`] for row `row`, then
    /// `argmax_rows_fault` over the `m` rows into `tok[..m]` and `fault`'s
    /// two words into `tok[m..m + 2]` (`tok` holds `m + 2`) — `tok[row]` is
    /// then row `row + 1`'s previous token. `fault` is also what the correction's
    /// out-of-vocabulary previous token raises into. Asynchronous,
    /// allocation-free, capturable.
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
        self.enqueue_add(stream, w, first, tok, m, row, fault, logits)?;
        elem.enqueue_argmax_rows_fault(stream, logits, w.w2.rows(), m, fault, tok)
    }

    /// [`Self::enqueue_step`] for a reduced head: [`Self::enqueue_add_reduced`]
    /// for row `row`, then the argmax over the `n_head` rows of each of the
    /// `m` columns — `tok[row]` is a head id, the next row's previous token
    /// through `to_target`. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one step's buffers, its row and the fault word, all distinct roles"
    )]
    pub fn enqueue_step_reduced(
        &self,
        stream: &CudaStream,
        elem: &ElemKernels,
        w: &ReducedMarkovWeights<'_>,
        first: &DeviceBuffer<u32>,
        tok: &mut DeviceBuffer<u32>,
        m: usize,
        row: usize,
        fault: FaultSink,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_add_reduced(stream, w, first, tok, m, row, fault, logits)?;
        elem.enqueue_argmax_rows_fault(stream, logits, w.w2.rows(), m, fault, tok)
    }
}

/// The head chain's buffers and weights, for blocks of up to
/// [`Head::MAX_ROWS`] rows.
pub struct Head {
    w: HeadWeights,
    markov: MarkovKernels,
    mtp: MtpKernels,
    normed: DeviceBuffer<f32>,
    /// The quantized rows, one scratch per row count (`Q8Act` fixes its
    /// columns). Empty for a head on the draft's own Q8_0 rows, whose row
    /// kernel takes the f32 rows.
    acts: Vec<Q8Act>,
    logits: DeviceBuffer<f32>,
    /// Each row's argmax, a head id, then the fault's first-layer word at
    /// `tok[m]` and its layer's site mask at `tok[m + 1]`; after the loop, the
    /// block's proposal. `rows + 2` words.
    tok: DeviceBuffer<u32>,
    /// [`Head::enqueue_p`]'s readback: each row's token, its probability's
    /// bits, then the fault's two words. `2 · rows + 2` words.
    probs: DeviceBuffer<u32>,
    /// That launch's ticket count: one word, zero between launches.
    done: DeviceBuffer<u32>,
    /// The stand-in for the row map the probabilities' argmax takes none of.
    no_map: DeviceBuffer<u32>,
    hidden: usize,
    rows: usize,
    /// Ids the head scores: the logits' rows and `markov_w2`'s.
    n_head: usize,
}

impl Head {
    /// The most rows a block holds: the projection kernels take eight
    /// columns.
    pub const MAX_ROWS: usize = 8;

    /// Launches of a block of any row count before the Markov loop, on the
    /// target's Q6_K matrix (norm, quantizer, projection); the draft's own
    /// Q8_0 rows take one fewer, no quantizer.
    pub const LOGIT_LAUNCHES: usize = 3;

    /// Buffers for blocks of `1..=rows` rows over `hidden`-wide folded rows,
    /// refusing by name a weight set the chain has no launch for. Load-time
    /// only.
    pub fn new(gpu: &Gpu, w: HeadWeights, hidden: usize, rows: usize) -> Result<Head, GpuError> {
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        if !(1..=Head::MAX_ROWS).contains(&rows) {
            return Err(refuse(format!(
                "blocks of up to {rows} rows; the head's projection kernels take 1..={}",
                Head::MAX_ROWS
            )));
        }
        let n_head = match &w.source {
            HeadSource::Target(target) => {
                if w.to_target.is_some() {
                    return Err(refuse(
                        "a draft-to-target map over the target's matrix: no draft needs one; \
                         a reduced head scores its own rows"
                            .into(),
                    ));
                }
                let TargetHead::Q6K(proj) = target else {
                    return Err(refuse(
                        "a Q8_0 target matrix: the head chain has a row kernel for Q6_K".into(),
                    ));
                };
                proj.rows()
            }
            HeadSource::Own(q) => {
                let (qs, d) = (&q.qs, &q.d);
                if d.rows() == 0
                    || qs.rows() != d.rows()
                    || d.cols() * 32 != hidden
                    || qs.cols() != d.cols() * 8
                {
                    return Err(refuse(format!(
                        "own rows of {}x{} words and {}x{} scales; {hidden}-wide Q8_0 rows need \
                         n_head x {} words and n_head x {} scales",
                        qs.rows(),
                        qs.cols(),
                        d.rows(),
                        d.cols(),
                        hidden / 4,
                        hidden / 32
                    )));
                }
                d.rows()
            }
        };
        if w.norm.buf().len() != hidden {
            return Err(refuse(format!(
                "a norm gain of {} values for rows of {hidden}",
                w.norm.buf().len()
            )));
        }
        let n_prev = if w.to_target.is_some() {
            w.markov_w1.rows()
        } else {
            n_head
        };
        for (name, t, need, ids) in [
            ("markov_w1", &w.markov_w1, n_prev, "previous-token"),
            ("markov_w2", &w.markov_w2, n_head, "head"),
        ] {
            if t.rows() != need || t.cols() != MARKOV_ROW_WORDS || need == 0 {
                return Err(refuse(format!(
                    "{name} is {} x {} words; the {need} {ids} ids need {need} x \
                     {MARKOV_ROW_WORDS}",
                    t.rows(),
                    t.cols()
                )));
            }
        }
        let s = gpu.stream();
        if let Some(map) = &w.to_target {
            if map.len() != n_head {
                return Err(refuse(format!(
                    "a draft-to-target map of {} words for a head of {n_head} ids",
                    map.len()
                )));
            }
            let host = map.to_host_vec(s)?;
            if let Some((d, &t)) = host
                .iter()
                .enumerate()
                .find(|&(_, &t)| t as usize >= n_prev)
            {
                return Err(refuse(format!(
                    "head id {d} maps to target id {t}, past the {n_prev} ids of markov_w1"
                )));
            }
        }
        let acts = match &w.source {
            HeadSource::Target(_) => (1..=rows)
                .map(|m| Q8Act::with_k(s, m, hidden))
                .collect::<Result<Vec<_>, _>>()?,
            HeadSource::Own(_) => Vec::new(),
        };
        Ok(Head {
            markov: MarkovKernels::load(gpu.context())?,
            mtp: MtpKernels::load(gpu.context())?,
            normed: DeviceBuffer::zeroed(s, rows * hidden)?,
            acts,
            logits: DeviceBuffer::zeroed(s, rows * n_head)?,
            tok: DeviceBuffer::zeroed(s, rows + 2)?,
            probs: DeviceBuffer::zeroed(s, 2 * rows + 2)?,
            done: DeviceBuffer::zeroed(s, 1)?,
            no_map: DeviceBuffer::zeroed(s, 1)?,
            hidden,
            rows,
            n_head,
            w,
        })
    }

    /// Rows of the logits: the ids the head scores. A reduced head's is its
    /// own `n_head`, not the target's vocabulary.
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.n_head
    }

    /// The most rows a block of this head holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Launches of a block of `m` rows: the norm, the quantizer and the
    /// projection, then the Markov loop's two a row. A head on the draft's
    /// own Q8_0 rows runs one fewer (no quantizer).
    #[must_use]
    pub fn launches(m: usize) -> usize {
        Head::LOGIT_LAUNCHES + 2 * m
    }

    /// Device bytes of the weights and the buffers.
    #[must_use]
    pub fn card_bytes(&self) -> u64 {
        let tensor = |t: &DeviceTensor<u32>| t.buf().num_bytes();
        let rows = |q: &Q8Rows| q.qs.buf().num_bytes() + q.d.buf().num_bytes();
        let source = match &self.w.source {
            HeadSource::Target(TargetHead::Q6K(w)) => tensor(w),
            HeadSource::Target(TargetHead::Q8_0(q)) | HeadSource::Own(q) => rows(q),
        };
        let conf = self.w.conf.as_ref().map_or(0, |(c, _)| c.buf().num_bytes());
        let to_target = self.w.to_target.as_ref().map_or(0, DeviceBuffer::num_bytes);
        let own = self.normed.num_bytes()
            + self.logits.num_bytes()
            + self.tok.num_bytes()
            + self.probs.num_bytes()
            + self.done.num_bytes()
            + self.no_map.num_bytes();
        (self.w.norm.buf().num_bytes()
            + source
            + tensor(&self.w.markov_w1)
            + tensor(&self.w.markov_w2)
            + conf
            + to_target
            + own) as u64
    }

    /// Refuse a row count the head holds no scratch for.
    fn check(&self, m: usize) -> Result<(), GpuError> {
        if (1..=self.rows).contains(&m) {
            Ok(())
        } else {
            Err(GpuError::shape(
                WHAT,
                format!("a block of {m} rows; the head holds 1..={}", self.rows),
            ))
        }
    }

    /// Enqueue the norm and the projection over `m` folded rows `x` (`m ·
    /// hidden`, token-major): the quantizer between them for the target's
    /// matrix, so three launches there and two on the draft's own rows.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_logits(
        &mut self,
        gpu: &Gpu,
        x: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.check(m)?;
        let s = gpu.stream();
        gpu.elem().enqueue_rms_norm(
            s,
            x,
            self.w.norm.buf(),
            self.w.eps,
            self.hidden,
            m,
            &mut self.normed,
        )?;
        match &self.w.source {
            HeadSource::Target(TargetHead::Q6K(proj)) => {
                let act = &mut self.acts[m - 1];
                gpu.enqueue_quantize_q8_1(&self.normed, act)?;
                gpu.enqueue_gemv_q6k(proj, act, &mut self.logits)
            }
            HeadSource::Own(q) => gpu.q8f32().enqueue_q8_0_gemv_mcol(
                s,
                Q8_0GemvMcolArgs {
                    qs: &q.qs,
                    d: &q.d,
                    x: &self.normed,
                    m,
                    out: GemvOut::RowMajor,
                    y: &mut self.logits,
                },
            ),
            HeadSource::Target(TargetHead::Q8_0(_)) => Err(GpuError::State {
                what: WHAT,
                missing: "a projection with a row kernel",
            }),
        }
    }

    /// Enqueue the Markov loop over the `m` rows' logits, row 0's previous
    /// token `first[0]` (a target id): `m` steps of two launches. A reduced
    /// head's steps carry the map ([`MarkovKernels::enqueue_step_reduced`]);
    /// the others are V4.1's ([`MarkovKernels::enqueue_step`]), unchanged.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_markov(
        &mut self,
        gpu: &Gpu,
        m: usize,
        first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.check(m)?;
        match &self.w.to_target {
            None => {
                let mw = MarkovWeights {
                    w1: &self.w.markov_w1,
                    w2: &self.w.markov_w2,
                };
                for row in 0..m {
                    self.markov.enqueue_step(
                        gpu.stream(),
                        gpu.elem(),
                        &mw,
                        first,
                        &mut self.tok,
                        m,
                        row,
                        gpu.unlabelled_sink(),
                        &mut self.logits,
                    )?;
                }
            }
            Some(to_target) => {
                let mw = ReducedMarkovWeights {
                    w1: &self.w.markov_w1,
                    w2: &self.w.markov_w2,
                    to_target,
                };
                for row in 0..m {
                    self.markov.enqueue_step_reduced(
                        gpu.stream(),
                        gpu.elem(),
                        &mw,
                        first,
                        &mut self.tok,
                        m,
                        row,
                        gpu.unlabelled_sink(),
                        &mut self.logits,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Enqueue each of the `m` rows' argmax with its probability: one
    /// launch over the logits the Markov loop left, after the loop. The
    /// probability is the largest of the row's softmax, `1 / Σ exp(x − max)`;
    /// a logit that is not finite raises [`FaultSite::Logit`]. The ids are
    /// head ids, as [`Head::tokens`]'. Asynchronous, allocation-free,
    /// capturable; the engine enqueues it only for a caller that asks, so a
    /// block's graph does not hold it.
    pub fn enqueue_p(&mut self, gpu: &Gpu, m: usize) -> Result<(), GpuError> {
        self.check(m)?;
        self.mtp.enqueue_argmax_p_rows_fault(
            gpu.stream(),
            ArgmaxPArgs {
                x: &self.logits,
                n: self.n_head,
                m,
                map: None,
                no_map: &self.no_map,
                vocab: self.n_head,
                fault: gpu.unlabelled_sink(),
                out: &mut self.probs,
                done: &mut self.done,
            },
        )
    }

    /// The first `m` columns of the logits as the last launch left them:
    /// `logits[v · m + c]` over [`Head::n_vocab`] rows `v`. After
    /// [`Head::enqueue_logits`] the projection's, after
    /// [`Head::enqueue_markov`] the corrected ones. Blocking.
    pub fn logits(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        self.check(m)?;
        let mut out = self.logits.to_host_vec(stream)?;
        out.truncate(self.n_head * m);
        Ok(out)
    }

    /// The `m` rows' proposal, head ids: a reduced head's are its own, not
    /// the target's. Blocking. A fault the last step copied is
    /// [`GpuError::Fault`], not a proposal; it stays raised on the card until
    /// the `Gpu`'s owner clears it.
    pub fn tokens(&self, stream: &CudaStream, m: usize) -> Result<Vec<u32>, GpuError> {
        self.check(m)?;
        let out = self.tok.to_host_vec(stream)?;
        if let Some(fault) = Fault::from_words(out[m], out[m + 1]) {
            return Err(GpuError::fault("dspark::head::tokens", fault));
        }
        Ok(out[..m].to_vec())
    }

    /// The `m` rows' tokens (head ids) and probabilities after
    /// [`Head::enqueue_p`]. Blocking. A fault raised on the card is
    /// [`GpuError::Fault`].
    pub fn tokens_p(
        &self,
        stream: &CudaStream,
        m: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        self.check(m)?;
        let out = self.probs.to_host_vec(stream)?;
        if let Some(fault) = Fault::from_words(out[2 * m], out[2 * m + 1]) {
            return Err(GpuError::fault("dspark::head::tokens_p", fault));
        }
        let p = out[m..2 * m].iter().map(|&b| f32::from_bits(b)).collect();
        Ok((out[..m].to_vec(), p))
    }
}
