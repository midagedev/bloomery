//! The recurrent Markov head's loop ([`super::head::HeadTail::MarkovRnn`]):
//! a state of `rank` values carried down a block's rows, and a correction
//! of each row's logits from it. Mistral Large 4's draft has it; the
//! reference is llama.cpp's `build_dspark_markov_head` with `joint_proj` set.
//!
//! Row `i` of a block of `m` rows, its previous token `p` (row 0's the
//! block's first id `first[0]`, row `i > 0`'s the argmax row `i − 1` left in
//! `tok[i − 1]`):
//!
//! 1. `x = [s | markov_w1[p] | h_i]` ([`RnnKernels::enqueue_input`]): `s` the
//!    state, zero at row 0; `markov_w1[p]` the table's Q8_0 row dequantized
//!    (`q · d`, ggml's `dequantize_q8_0`); `h_i` the head's normed row `i`;
//! 2. `j = joint_proj · x` (Q8_0, `2·rank + hidden → 3·rank`, the q8f32 row
//!    kernel at one column);
//! 3. ([`RnnKernels::enqueue_gate`]) with `a_k = j[k·rank + r] + b[k·rank +
//!    r]`: `g = sigmoid(a_0)`, `c = tanh(a_1)`, the state `c + g · (s − c)`
//!    (a subtraction, a product and a sum, each rounded on its own, as the
//!    reference's three ops), and `e = tanh(a_2)`;
//! 4. `δ = markov_w2 · e` (Q8_0, `rank → n_head`, one column);
//! 5. `logits[v · m + i] += δ[v]` ([`RnnKernels::enqueue_add`]);
//! 6. the argmax over the `m` rows into `tok` (`elem::argmax_rows_fault`):
//!    `tok[i]` is row `i + 1`'s previous token.
//!
//! The logits are the head's gemv layout, `logits[v · m + c]`. Rows below
//! `i` are unchanged since their own step and give the same tokens again;
//! rows above `i` write a token their own step overwrites. A previous token
//! at or past `markov_w1`'s rows raises [`FaultSite::TokenId`] and puts NaN
//! where its row would stand, so every logit of the row turns NaN; the
//! pass's readback carries the fault.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::cores::half_to_f32;
use crate::route_core::sigmoid;
use crate::{FaultSink, FaultSite, GpuError, launch_u32};

const THREADS: u32 = 256;

/// Launches of one row of the loop: the input, the joint gemv, the gate,
/// the correction's gemv, its add and the argmax.
pub const ROW_LAUNCHES: usize = 6;

#[cuda_module]
mod rnn_kernels {
    use super::*;

    /// Row `row`'s joint input `x = [s | markov_w1[p] | normed row]` (module
    /// doc step 1). `rank` is `32 · row_scales`; `markov_w1` is `n_prev`
    /// Q8_0 rows of `rank` values, its quants four to a word
    /// (`8 · row_scales` words a row) and one f16 scale a 32 values. The
    /// state is read from `state` on every row but the first, where it is
    /// zero. A previous token at or past `n_prev` raises
    /// [`FaultSite::TokenId`] on `fault` (thread 0) and fills the row's
    /// table part with NaN.
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
            w1_qs.len() >= n_prev * 8 * row_scales,
            w1_d.len() >= n_prev * row_scales,
            first.len() >= 1,
            tok.len() >= m,
            state.len() >= 32 * row_scales,
            normed.len() >= m * hidden,
            x.len() >= 64 * row_scales + hidden,
            row < m,
            n_prev >= 1,
            row_scales >= 1,
            hidden >= 1
        )
    )]
    pub fn dspark_rnn_input(
        w1_qs: &[u32],
        w1_d: &[u16],
        first: &[u32],
        tok: &[u32],
        state: &[f32],
        normed: &[f32],
        n_prev: u32,
        row_scales: u32,
        hidden: u32,
        m: u32,
        row: u32,
        fault: FaultSink,
        mut x: DisjointSlice<f32>,
    ) {
        // `m` is read by the launch contract only (`row < m`).
        let _ = m;
        let i = thread::index_1d().get();
        let rank = 32 * row_scales as usize;
        // SAFETY: first.len() >= 1, and row - 1 < m <= tok.len() for a row
        // past the first, by the launch contract.
        let p = unsafe {
            if row == 0 {
                *first.get_unchecked(0)
            } else {
                *tok.get_unchecked(row as usize - 1)
            }
        };
        if i == 0 && p >= n_prev {
            fault.raise(FaultSite::TokenId);
        }
        if i >= 2 * rank + hidden as usize {
            return;
        }
        let v = if i < rank {
            if row == 0 {
                0.0
            } else {
                // SAFETY: i < rank = 32 * row_scales <= state.len().
                unsafe { *state.get_unchecked(i) }
            }
        } else if i < 2 * rank {
            let k = i - rank;
            if p < n_prev {
                let p = p as usize;
                // SAFETY: p < n_prev and k < rank, so the word p * 8 *
                // row_scales + k / 4 < n_prev * 8 * row_scales and the scale
                // p * row_scales + k / 32 < n_prev * row_scales: both inside
                // the tables by the launch contract.
                let (word, d) = unsafe {
                    (
                        *w1_qs.get_unchecked(p * 8 * row_scales as usize + k / 4),
                        *w1_d.get_unchecked(p * row_scales as usize + k / 32),
                    )
                };
                let q = ((word >> (8 * (k % 4))) & 0xff) as u8 as i8;
                mul_rn_f32(f32::from(q), half_to_f32(d))
            } else {
                f32::NAN
            }
        } else {
            let at = row as usize * hidden as usize + (i - 2 * rank);
            // SAFETY: row < m and i - 2 * rank < hidden, so at < m * hidden
            // <= normed.len().
            unsafe { *normed.get_unchecked(at) }
        };
        // SAFETY: i < 2 * rank + hidden <= x.len(); thread i alone writes it.
        unsafe {
            *x.get_unchecked_mut(i) = v;
        }
    }

    /// The gate (module doc step 3): thread `r < rank` reads the joint
    /// output `joint[k · rank + r]` and bias `b[k · rank + r]` of the three
    /// chunks and the old state `x[r]` (the input kernel's copy), and writes
    /// the new state `state[r]` and the correction vector `e[r]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            joint.len() >= 3 * rank,
            b.len() >= 3 * rank,
            x.len() >= rank,
            state.len() >= rank,
            e.len() >= rank
        )
    )]
    pub fn dspark_rnn_gate(
        joint: &[f32],
        b: &[f32],
        x: &[f32],
        rank: u32,
        mut state: DisjointSlice<f32>,
        mut e: DisjointSlice<f32>,
    ) {
        let r = thread::index_1d().get();
        let rank = rank as usize;
        if r >= rank {
            return;
        }
        // SAFETY: r < rank, so r + 2 * rank < 3 * rank <= joint.len() and
        // b.len(), and r < x.len() (launch contract).
        let (a0, a1, a2, s) = unsafe {
            (
                add_rn_f32(*joint.get_unchecked(r), *b.get_unchecked(r)),
                add_rn_f32(*joint.get_unchecked(rank + r), *b.get_unchecked(rank + r)),
                add_rn_f32(
                    *joint.get_unchecked(2 * rank + r),
                    *b.get_unchecked(2 * rank + r),
                ),
                *x.get_unchecked(r),
            )
        };
        let g = sigmoid(a0);
        let c = a1.tanh();
        let next = add_rn_f32(c, mul_rn_f32(g, add_rn_f32(s, -c)));
        // SAFETY: r < rank <= state.len() and e.len(); thread r alone writes
        // both.
        unsafe {
            *state.get_unchecked_mut(r) = next;
            *e.get_unchecked_mut(r) = a2.tanh();
        }
    }

    /// Row `row`'s correction added (module doc step 5): thread `v < n_head`
    /// adds `delta[v]` into `logits[v · m + row]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            delta.len() >= n_head,
            logits.len() >= n_head * m,
            row < m
        )
    )]
    pub fn dspark_rnn_add(
        delta: &[f32],
        n_head: u32,
        m: u32,
        row: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        let v = thread::index_1d().get();
        if v >= n_head as usize {
            return;
        }
        let at = v * m as usize + row as usize;
        // SAFETY: v < n_head <= delta.len(), at < n_head * m <=
        // logits.len(); one thread per v.
        unsafe {
            let l = logits.get_unchecked_mut(at);
            *l = add_rn_f32(*l, *delta.get_unchecked(v));
        }
    }
}

/// [`RnnKernels::enqueue_input`]'s arguments.
pub struct RnnInputArgs<'a> {
    /// `markov_w1`'s quants: `n_prev` rows of `rank / 4` words.
    pub w1_qs: &'a DeviceBuffer<u32>,
    /// `markov_w1`'s f16 scales: `n_prev` rows of `rank / 32`.
    pub w1_d: &'a DeviceBuffer<u16>,
    /// `markov_w1`'s rows: the ids a previous token may be.
    pub n_prev: usize,
    /// The state's values, a multiple of 32.
    pub rank: usize,
    /// The block's first id, word 0.
    pub first: &'a DeviceBuffer<u32>,
    /// The argmax of every row so far.
    pub tok: &'a DeviceBuffer<u32>,
    /// The state the row before left.
    pub state: &'a DeviceBuffer<f32>,
    /// The head's normed rows, `m · hidden`, token-major.
    pub normed: &'a DeviceBuffer<f32>,
    pub hidden: usize,
    pub m: usize,
    pub row: usize,
    pub fault: FaultSink,
    /// The joint input, `2 · rank + hidden` values.
    pub x: &'a mut DeviceBuffer<f32>,
}

/// The loaded recurrent-head module. Owns no stream.
pub struct RnnKernels {
    module: rnn_kernels::LoadedModule,
}

impl RnnKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RnnKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(rnn_kernels, ctx)? };
        Ok(RnnKernels { module })
    }

    /// Enqueue row `a.row`'s joint input (`dspark_rnn_input`). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_input(&self, stream: &CudaStream, a: RnnInputArgs<'_>) -> Result<(), GpuError> {
        let what = "RnnKernels::enqueue_input";
        let RnnInputArgs {
            w1_qs,
            w1_d,
            n_prev,
            rank,
            first,
            tok,
            state,
            normed,
            hidden,
            m,
            row,
            fault,
            x,
        } = a;
        let width = 2 * rank + hidden;
        if rank == 0
            || !rank.is_multiple_of(32)
            || n_prev == 0
            || hidden == 0
            || w1_qs.len() < n_prev * rank / 4
            || w1_d.len() < n_prev * rank / 32
            || first.is_empty()
            || m == 0
            || row >= m
            || tok.len() < m
            || state.len() < rank
            || normed.len() < m * hidden
            || x.len() < width
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "rank {rank} (a positive multiple of 32), {n_prev} table rows over {} words \
                     and {} scales, first {}, row {row} of m {m}, tok {}, state {}, normed {} \
                     (need {}), x {} (need {width})",
                    w1_qs.len(),
                    w1_d.len(),
                    first.len(),
                    tok.len(),
                    state.len(),
                    normed.len(),
                    m * hidden,
                    x.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", width.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_dspark_rnn_input(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.dspark_rnn_input(
            stream,
            &prep,
            w1_qs,
            w1_d,
            first,
            tok,
            state,
            normed,
            launch_u32(what, "n_prev", n_prev)?,
            launch_u32(what, "row_scales", rank / 32)?,
            launch_u32(what, "hidden", hidden)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "row", row)?,
            fault,
            x,
        )?;
        Ok(())
    }

    /// Enqueue the gate over `rank` values (`dspark_rnn_gate`): `joint` and
    /// `b` hold the three chunks, `x` the old state at its head. Writes the
    /// new state and the correction vector. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's inputs and its two outputs, all distinct roles"
    )]
    pub fn enqueue_gate(
        &self,
        stream: &CudaStream,
        joint: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        rank: usize,
        state: &mut DeviceBuffer<f32>,
        e: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "RnnKernels::enqueue_gate";
        if rank == 0
            || joint.len() < 3 * rank
            || b.len() < 3 * rank
            || x.len() < rank
            || state.len() < rank
            || e.len() < rank
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "rank {rank}: joint {} and bias {} (need {}), x {}, state {}, e {}",
                    joint.len(),
                    b.len(),
                    3 * rank,
                    x.len(),
                    state.len(),
                    e.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", rank.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_dspark_rnn_gate(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.dspark_rnn_gate(
            stream,
            &prep,
            joint,
            b,
            x,
            launch_u32(what, "rank", rank)?,
            state,
            e,
        )?;
        Ok(())
    }

    /// Enqueue row `row`'s correction `delta` (`n_head` values) into `logits`
    /// (`m` rows, `logits[v · m + c]`) (`dspark_rnn_add`). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        delta: &DeviceBuffer<f32>,
        n_head: usize,
        m: usize,
        row: usize,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "RnnKernels::enqueue_add";
        if n_head == 0 || m == 0 || row >= m || delta.len() < n_head || logits.len() < n_head * m {
            return Err(GpuError::shape(
                what,
                format!(
                    "n_head {n_head}, row {row} of m {m}, delta {}, logits {} (need {})",
                    delta.len(),
                    logits.len(),
                    n_head * m
                ),
            ));
        }
        let grid = launch_u32(what, "grid", n_head.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_dspark_rnn_add(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.dspark_rnn_add(
            stream,
            &prep,
            delta,
            launch_u32(what, "n_head", n_head)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "row", row)?,
            logits,
        )?;
        Ok(())
    }
}
