//! The draft's head chain: the final norm, the projection
//! ([`super::HeadSource`]), the Markov loop over the block's positions and
//! the draft-to-target id map.
//!
//! [`Head`] launches the chain over a block's `m` folded rows: `rms_norm`
//! (the head's own gain and epsilon) → the shared q8_1 quantizer → the
//! projection's row kernel, one launch each over the `m` rows (each row bit
//! for bit its `m = 1` launch). The logits land in the gemv's layout,
//! `logits[v · m + c]`, which `ds41_markov`'s is. The Markov loop is `m`
//! serial steps ([`MarkovKernels::enqueue_step`]): row 0's previous token is
//! the block's first id, read from the device where the pass's staging put
//! it, row `c`'s the argmax the step before wrote. The last step's argmax
//! buffer holds the block's proposal and, after it, the card's fault word,
//! so [`Head::tokens`]' one readback turns a fault any launch of the pass
//! raised into [`GpuError::Fault`], as [`crate::head::Head`] does for the
//! target. The probabilities of a proposal ([`Head::enqueue_p`]) are one
//! launch of their own, enqueued only for a caller that asks.
//!
//! The projection runs the target's Q6_K matrix. Any other type, the
//! draft's own rows and a reduced head are refused by name until their row
//! kernel and Markov entry exist.
//!
//! # The Markov loop
//!
//! After the head's logits, position `i` of a block adds the bigram
//! correction of its previous token `p`, `logits_i[v] += markov_w2[v] ·
//! markov_w1[p]`, and its argmax is position `i + 1`'s previous token.
//! Serial over `i` by construction.
//!
//! Layouts. Both weights are the file's bf16 `[rank, n_vocab]` as u32 words:
//! token `t`'s rank-long row starts at word `t * rank / 2`, value `r` in word
//! `r / 2`, the low half for even `r` (little-endian). A position's previous
//! token is read on the device: row 0's from `first[0]` (the block's last
//! accepted token), row `c > 0`'s from `tok[c - 1]`, the argmax the step wrote
//! for row `c - 1`. An id at or past `n_vocab` (only `first[0]` can hold one:
//! the argmax's are below the vocabulary) raises `FaultSite::TokenId`, reads
//! no `w1` word, and makes the row's correction NaN, so every logit of that
//! row is NaN — no value the pass could take for a draft; the pass's readback
//! carries the fault.
//!
//! The numeric rule `ds41_markov` holds, and the gate transcribes on the host:
//! bf16 widens to f32 as `bits << 16` (exact), and the dot for entry `v` is
//! `markov-accept`'s order: eight accumulators, `a[l] = fma(w2[v][8j + l],
//! e[8j + l], a[l])` from 0 over the octets `j` ascending, then
//! `((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))`; the result is added to
//! the logit once, `logit + delta`. One thread per `v`.
//!
//! A step is `ds41_markov` then the `argmax_rows_fault` of [`crate::elem`] over
//! all `m` rows ([`MarkovKernels::enqueue_step`]): rows below `c` are
//! unchanged since their own step and give the same tokens again; rows above
//! `c` write a token their own step overwrites. The argmax copies the card's
//! fault to `tok[m]` (the first-layer word) and `tok[m + 1]` (its layer's
//! site mask) after the tokens, so the last step's readback of `tok[..m +
//! 2]` carries every fault the pass raised.

use crate::elem::ElemKernels;
use crate::mtp::{ArgmaxPArgs, MtpKernels};
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
    /// A reduced head's draft-to-target map on the card.
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
    /// thread `v` adds `w2[v] · e` into `logits[v * m + row]`. A previous
    /// token past the vocabulary raises [`FaultSite::TokenId`] on `fault`
    /// (thread 0 of the grid) and gathers NaN instead of a row.
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
        fault: FaultSink,
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
            let (lo, hi) = if p < n_vocab {
                // SAFETY: p < n_vocab, so word p*128 + tid < 128*n_vocab <=
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
}

/// The head chain's buffers and weights, for blocks of up to
/// [`Head::MAX_ROWS`] rows.
pub struct Head {
    w: HeadWeights,
    markov: MarkovKernels,
    mtp: MtpKernels,
    normed: DeviceBuffer<f32>,
    /// The quantized rows, one scratch per row count (`Q8Act` fixes its
    /// columns).
    acts: Vec<Q8Act>,
    logits: DeviceBuffer<f32>,
    /// Each row's argmax, then the fault's first-layer word at `tok[m]` and
    /// its layer's site mask at `tok[m + 1]`; after the loop, the block's
    /// proposal. `rows + 2` words.
    tok: DeviceBuffer<u32>,
    /// [`Head::enqueue_p`]'s readback: each row's token, its probability's
    /// bits, then the fault's two words. `2 · rows + 2` words.
    probs: DeviceBuffer<u32>,
    /// That launch's ticket count: one word, zero between launches.
    done: DeviceBuffer<u32>,
    /// The stand-in for the row map a full head has none of.
    no_map: DeviceBuffer<u32>,
    hidden: usize,
    n_vocab: usize,
}

impl Head {
    /// The most rows a block holds: the q8_1 scratch takes eight columns.
    pub const MAX_ROWS: usize = 8;

    /// Launches one block of any row count makes before the Markov loop.
    pub const LOGIT_LAUNCHES: usize = 3;

    /// Buffers for blocks of `1..=rows` rows over `hidden`-wide folded rows,
    /// refusing by name a weight set the chain has no launch for. Load-time
    /// only.
    pub fn new(gpu: &Gpu, w: HeadWeights, hidden: usize, rows: usize) -> Result<Head, GpuError> {
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        if !(1..=Head::MAX_ROWS).contains(&rows) {
            return Err(refuse(format!(
                "blocks of up to {rows} rows; the head's q8_1 scratch takes 1..={}",
                Head::MAX_ROWS
            )));
        }
        if w.to_target.is_some() {
            return Err(refuse(
                "a reduced head's draft-to-target map: the Markov entry for a head of other rows \
                 than the target's is not written"
                    .into(),
            ));
        }
        let HeadSource::Target(target) = &w.source else {
            return Err(refuse(
                "the draft's own projection: the head runs the target's matrix only".into(),
            ));
        };
        let TargetHead::Q6K(proj) = target else {
            return Err(refuse(
                "a Q8_0 target matrix: the head chain has a row kernel for Q6_K".into(),
            ));
        };
        let n_vocab = proj.rows();
        if w.norm.buf().len() != hidden {
            return Err(refuse(format!(
                "a norm gain of {} values for rows of {hidden}",
                w.norm.buf().len()
            )));
        }
        for (name, t) in [("markov_w1", &w.markov_w1), ("markov_w2", &w.markov_w2)] {
            if t.rows() != n_vocab || t.cols() != MARKOV_ROW_WORDS {
                return Err(refuse(format!(
                    "{name} is {} x {} words; the head's {n_vocab} ids need {n_vocab} x \
                     {MARKOV_ROW_WORDS}",
                    t.rows(),
                    t.cols()
                )));
            }
        }
        let s = gpu.stream();
        let acts = (1..=rows)
            .map(|m| Q8Act::with_k(s, m, hidden))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Head {
            markov: MarkovKernels::load(gpu.context())?,
            mtp: MtpKernels::load(gpu.context())?,
            normed: DeviceBuffer::zeroed(s, rows * hidden)?,
            acts,
            logits: DeviceBuffer::zeroed(s, rows * n_vocab)?,
            tok: DeviceBuffer::zeroed(s, rows + 2)?,
            probs: DeviceBuffer::zeroed(s, 2 * rows + 2)?,
            done: DeviceBuffer::zeroed(s, 1)?,
            no_map: DeviceBuffer::zeroed(s, 1)?,
            hidden,
            n_vocab,
            w,
        })
    }

    /// Rows of the projection: the vocabulary.
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The most rows a block of this head holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.acts.len()
    }

    /// Launches of a block of `m` rows: the norm, the quantizer and the
    /// projection, then the Markov loop's two a row.
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
        if (1..=self.acts.len()).contains(&m) {
            Ok(())
        } else {
            Err(GpuError::shape(
                WHAT,
                format!(
                    "a block of {m} rows; the head holds 1..={}",
                    self.acts.len()
                ),
            ))
        }
    }

    /// Enqueue the norm, the quantizer and the projection over `m` folded
    /// rows `x` (`m · hidden`, token-major). Three launches. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_logits(
        &mut self,
        gpu: &Gpu,
        x: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.check(m)?;
        let HeadSource::Target(TargetHead::Q6K(proj)) = &self.w.source else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the target's projection",
            });
        };
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
        let act = &mut self.acts[m - 1];
        gpu.enqueue_quantize_q8_1(&self.normed, act)?;
        gpu.enqueue_gemv_q6k(proj, act, &mut self.logits)
    }

    /// Enqueue the Markov loop over the `m` rows' logits, row 0's previous
    /// token `first[0]`: `m` steps of two launches. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_markov(
        &mut self,
        gpu: &Gpu,
        m: usize,
        first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.check(m)?;
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
        Ok(())
    }

    /// Enqueue each of the `m` rows' argmax with its probability: one
    /// launch over the logits the Markov loop left, after the loop. The
    /// probability is the largest of the row's softmax, `1 / Σ exp(x − max)`;
    /// a logit that is not finite raises [`FaultSite::Logit`]. Asynchronous,
    /// allocation-free, capturable; the engine enqueues it only for a caller
    /// that asks, so a block's graph does not hold it.
    pub fn enqueue_p(&mut self, gpu: &Gpu, m: usize) -> Result<(), GpuError> {
        self.check(m)?;
        self.mtp.enqueue_argmax_p_rows_fault(
            gpu.stream(),
            ArgmaxPArgs {
                x: &self.logits,
                n: self.n_vocab,
                m,
                map: None,
                no_map: &self.no_map,
                vocab: self.n_vocab,
                fault: gpu.unlabelled_sink(),
                out: &mut self.probs,
                done: &mut self.done,
            },
        )
    }

    /// The `m` rows' proposal. Blocking. A fault the last step copied is
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

    /// The `m` rows' tokens and probabilities after [`Head::enqueue_p`].
    /// Blocking. A fault raised on the card is [`GpuError::Fault`].
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
