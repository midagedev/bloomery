//! The MoE sub-layer over a prompt batch: [`FfnBatch`], the buffers a batch
//! of up to its column count keeps between a layer's phases, and the piece's
//! batch enqueues. A layer runs, over the chunks of at most [`HC_MAX_TOKENS`]
//! consecutive tokens that hold its block — a suffix of the batch's tokens,
//! `at .. u`:
//!
//! 1. the route ([`FfnPiece::enqueue_batch_route`]), per chunk: the norm with
//!    its q8_1 form — the step's `norm_quant`, whose f32 output is the
//!    router's input and the host's activation — the router one token at a
//!    time into the batch's ids and weights, and each slot's place in the
//!    card's routed stacks (`ds41_ffn_places`, the handoff's rule);
//! 2. the exchange ([`FfnBatch::enqueue_download`], [`FfnBatch::serve`],
//!    [`FfnBatch::enqueue_upload`]): the block's activations and routing to
//!    the host, one union call over its tokens for the layer's host experts
//!    ([`Hybrid::serve_batch`]), the sums back;
//! 3. the shadow ([`FfnPiece::enqueue_batch_shadow`]), per chunk, enqueued
//!    before the host computes: HC_PRE, the norm's q8_1 form again, the card's
//!    routed experts over the chunk's slots (`ds41_expert_gate_up_tok`, the
//!    step's dot on column `slot / 6`; the q8_1 of `h`; `q4k_sel` over the
//!    slots) and their sum (`ds41_ffn_card_acc`), the shared expert;
//! 4. the join ([`FfnPiece::enqueue_batch_join`]), one launch over the block:
//!    the card sum, the host sum and the shared expert combined, then HC_POST
//!    with the next fold where the layer folds.
//!
//! The feature tap of a batch's kept tokens is one launch here too
//! ([`FfnBatch::enqueue_tap_means`], `ds41_tap_means`).
//!
//! Every token's values are the step's bit for bit: each launch writes, per
//! token, what its one-token launch writes — the m-column kernels carry that
//! contract, the routed dot reads its token's column through the step's
//! `q3k_row_dot`, and the combine is [`combine_elem`] cut at its one seam
//! ([`card_sum_elem`], then [`join_elem`]).

use std::mem::size_of;

use bloomery_gpu::cores::q3k_row_dot;
use cuda_core::{CudaEvent, IntoResult, PinnedHostBuffer, sys};
use cuda_device::warp;

use super::*;
use crate::experts::swiglu_clamp;
use crate::hc::hc_mean_elem;
use crate::span::{span, span_mut};

const WHAT: &str = "FfnBatch";

/// Threads per block of every kernel here.
const THREADS: u32 = 256;

#[cuda_module]
mod ffn_batch_kernels {
    use super::*;

    /// Each slot's place, the handoff's rule: thread `i < n` writes `sel[i] =
    /// map[row_off + ids[i]]` — the id's slot in the card's routed stacks, or
    /// [`HOST`] — and [`HOST`] for an id not below `n_expert`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (ids.len() >= n, map.len() >= row_off + n_expert, sel.len() >= n)
    )]
    pub fn ds41_ffn_places(
        ids: &[u32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        n: u32,
        mut sel: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(i) };
        let place = if id < n_expert {
            // SAFETY: id < n_expert, so row_off + id < map.len() by the launch
            // contract.
            unsafe { *map.get_unchecked(row_off as usize + id as usize) }
        } else {
            HOST
        };
        // SAFETY: i < n <= sel.len(); thread i is sel[i]'s only writer.
        unsafe { *sel.get_unchecked_mut(i) = place };
    }

    /// The routed experts' gate·up·SwiGLU over the slots of `cols` tokens,
    /// six a token: `ds41_expert_gate_up` with slot `s` dotting column `s /
    /// 6` of the q8_1 activation — thread row `n = s · rows_per_expert + r`
    /// (a warp per row, 8 rows per block) reads weight row `sel[s] ·
    /// rows_per_expert + r` of both stacks, `cores::q3k_row_dot` on that one
    /// column, the warp tree, then `swiglu_clamp` into `h[n]`. A slot whose
    /// place is not below `n_experts` returns before any load and leaves its
    /// rows of `h` as they were.
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
            4 * wg.len() >= n_experts * rows_per_expert * 110 * n_sb,
            4 * wu.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters * cols,
            d8.len() >= 2 * n_sb * cols,
            n_slots <= 6 * cols,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn ds41_expert_gate_up_tok(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract. All 32
        // lanes of the warp share `row`, hence `slot`, `id` and `col`: the
        // returns below are warp-uniform.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let col = slot / N_USED;
        if col >= cols as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_slots * rows_per_expert <= h.len() by the
            // launch contract; lane 0 of the row's warp is its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// The card slots' sum of `m` tokens: thread `i < m·n` is token `t = i /
    /// n`, value `d = i % n`, and writes [`card_sum_elem`] of its six slots'
    /// down outputs `down[(6t + j)·n + d]`, weights `w[6t + j]` and places
    /// `sel[6t + j]` (the card's below `n_card`; no other slot's rows are
    /// read) to `acc[i]`.
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
            down.len() >= 6 * n * m,
            w.len() >= 6 * m,
            sel.len() >= 6 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let mut dv = [0.0f32; N_USED];
        let mut wv = [0.0f32; N_USED];
        let mut card = [false; N_USED];
        let mut j = 0usize;
        while j < N_USED {
            let s = t * N_USED + j;
            // SAFETY: s < 6m <= sel.len() and w.len() by the launch contract.
            let (place, ws) = unsafe { (*sel.get_unchecked(s), *w.get_unchecked(s)) };
            if place < n_card {
                card[j] = true;
                wv[j] = ws;
                // SAFETY: s < 6m and d < n, so s·n + d < 6nm <= down.len().
                dv[j] = unsafe { *down.get_unchecked(s * n + d) };
            }
            j += 1;
        }
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = card_sum_elem(dv, wv, card) };
    }

    /// The join of `m` tokens: thread `i < m·n` (token `t = i / n`, value `d
    /// = i % n`) combines `y = join_elem(acc[i], hsum[i], shexp[i])`, then
    /// HC_POST of it (`hc_post_elem`) by token `t`'s HC_PRE result
    /// `hc[24t ..]` and its four streams `res[4nt + kn + d]`, into `out` in
    /// the same layout, and their fold by the result's `pre` into `fold[i]`.
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
            acc.len() >= n * m,
            hsum.len() >= n * m,
            shexp.len() >= n * m,
            res.len() >= 4 * n * m,
            hc.len() >= 24 * m,
            out.len() >= 4 * n * m,
            fold.len() >= n * m
        )
    )]
    pub fn ds41_ffn_post_batch(
        acc: &[f32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let a = JoinIn {
            acc,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for.
        let (o, pre) = unsafe { join_post_at(&a, n, i) };
        let b = 4 * (i / n) * n + i % n;
        // SAFETY: b + 3n < 4n(t + 1) <= 4nm <= out.len(), i < nm <= fold.len(),
        // by the launch contract; thread i is the only writer of the four
        // stream values at b and of fold[i].
        unsafe {
            store4(&mut out, n, b, o);
            *fold.get_unchecked_mut(i) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post_batch`] without the fold: into an engram layer and
    /// after the last layer.
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
            acc.len() >= n * m,
            hsum.len() >= n * m,
            shexp.len() >= n * m,
            res.len() >= 4 * n * m,
            hc.len() >= 24 * m,
            out.len() >= 4 * n * m
        )
    )]
    pub fn ds41_ffn_post_batch_streams(
        acc: &[f32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let a = JoinIn {
            acc,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for.
        let (o, _) = unsafe { join_post_at(&a, n, i) };
        let b = 4 * (i / n) * n + i % n;
        // SAFETY: b + 3n < 4nm <= out.len() by the launch contract; thread i
        // is the only writer of the four stream values at b.
        unsafe { store4(&mut out, n, b, o) };
    }

    /// The feature tap over `m` tokens: thread `i < m·n` is token `t = i /
    /// n`, value `d = i % n`, and writes the mean of token `t`'s four streams
    /// at `d` ([`hc_mean_elem`], the one-token tap's value) to `y[t · width +
    /// off + d]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (s.len() >= 4 * n * m, width >= off + n, y.len() >= width * m)
    )]
    pub fn ds41_tap_means(
        s: &[f32],
        n: u32,
        m: u32,
        width: u32,
        off: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let b = 4 * t * n + d;
        // SAFETY: b + 3n < 4n(t + 1) <= 4nm <= s.len() by the launch contract.
        let o = unsafe {
            [
                *s.get_unchecked(b),
                *s.get_unchecked(b + n),
                *s.get_unchecked(b + 2 * n),
                *s.get_unchecked(b + 3 * n),
            ]
        };
        let at = t * width as usize + off as usize + d;
        // SAFETY: at < t·width + width <= m·width <= y.len() (off + n <=
        // width, launch contract); thread i is at's only writer.
        unsafe { *y.get_unchecked_mut(at) = hc_mean_elem(o) };
    }
}

/// What the batch join reads, `m` tokens token-major: `acc` the card sums,
/// `hsum` the host sums and `shexp` the shared expert's outputs (`n` a
/// token), `res` the streams the sub-layer read (`4n` a token) and `hc` its
/// HC_PRE results (24 a token).
struct JoinIn<'a> {
    acc: &'a [f32],
    hsum: &'a [f32],
    shexp: &'a [f32],
    res: &'a [f32],
    hc: &'a [f32],
}

/// Value `i` of the batch (token `i / n`): its combine ([`join_elem`]), then
/// HC_POST of it (`hc_post_elem`): the four new stream values and `pre`.
///
/// SAFETY: every buffer of `a` holds the token `i / n`: `acc`, `hsum`,
/// `shexp` more than `i` values, `res` at least `4n(i / n + 1)`, `hc` at
/// least `24(i / n + 1)`.
#[inline(always)]
unsafe fn join_post_at(a: &JoinIn<'_>, n: usize, i: usize) -> ([f32; 4], [f32; 4]) {
    let (t, d) = (i / n, i % n);
    let (b, h) = (4 * t * n + d, t * HC_MIX);
    // SAFETY: i < acc/hsum/shexp lengths; b + 3n < 4n(t + 1) <= res.len();
    // h + 23 < 24(t + 1) <= hc.len() — all by this fn's contract.
    let (ac, hs, sh, r, hc) = unsafe {
        let mut hc = [0.0f32; HC_MIX];
        let mut k = 0usize;
        while k < HC_MIX {
            hc[k] = *a.hc.get_unchecked(h + k);
            k += 1;
        }
        (
            *a.acc.get_unchecked(i),
            *a.hsum.get_unchecked(i),
            *a.shexp.get_unchecked(i),
            [
                *a.res.get_unchecked(b),
                *a.res.get_unchecked(b + n),
                *a.res.get_unchecked(b + 2 * n),
                *a.res.get_unchecked(b + 3 * n),
            ],
            hc,
        )
    };
    let pre = [hc[0], hc[1], hc[2], hc[3]];
    let post = [hc[4], hc[5], hc[6], hc[7]];
    let comb = [
        hc[8], hc[9], hc[10], hc[11], hc[12], hc[13], hc[14], hc[15], hc[16], hc[17], hc[18],
        hc[19], hc[20], hc[21], hc[22], hc[23],
    ];
    let y = join_elem(ac, hs, sh);
    (hc_post_elem(y, r, post, &comb), pre)
}

/// The buffers a prompt batch keeps between a layer's phases, for up to
/// `cap` tokens a batch and chunks of up to [`HC_MAX_TOKENS`]: allocated
/// once, when the first batch runs — a decode that never prefills a batch
/// never holds them.
pub struct FfnBatch {
    module: ffn_batch_kernels::LoadedModule,
    cap: usize,
    n_embd: usize,
    ff: usize,
    /// Per token: the norm's f32 output (the router's input and the host's
    /// activation, `n_embd`), the routing (six ids and weights), each slot's
    /// place, the HC_PRE result, the card sum, the shared expert's output and
    /// the host sum (`n_embd` each).
    x: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    hc: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    shexp: DeviceBuffer<f32>,
    hsum: DeviceBuffer<f32>,
    /// A chunk's scratch: the router's per-expert rows and ticket; the norm's
    /// f32 output a second time (discarded) and, per token count `m` (index
    /// `m − 1`), its q8_1 form; the HC_PRE mixes; the routed SwiGLU outputs
    /// (six slots a token), their q8_1 form per token count, the routed down
    /// outputs; the shared expert's SwiGLU output (token-major), its q8_1 form
    /// per token count and its down output row-major.
    rout: RouterOut,
    normed: DeviceBuffer<f32>,
    act_x: Vec<Q8Act>,
    mixes: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    act_h: Vec<Q8Act>,
    down: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    act_sh: Vec<Q8Act>,
    sh_raw: DeviceBuffer<f32>,
    /// The exchange: page-locked copies of the activations, the routing and
    /// the host sums; the host's view of the activations the union reads;
    /// the event the route's copies complete at.
    host_x: PinnedHostBuffer<f32>,
    host_ids: PinnedHostBuffer<u32>,
    host_w: PinnedHostBuffer<f32>,
    host_sum: PinnedHostBuffer<f32>,
    x_host: Tensor2,
    routed: CudaEvent,
}

impl FfnBatch {
    /// The buffers for batches of up to `cap` tokens (at most
    /// [`UNION_MAX_COLS`], the host union's columns) of rows of `n_embd`
    /// through experts of `ff`, on `gpu`.
    pub fn new(gpu: &Gpu, n_embd: usize, ff: usize, cap: usize) -> Result<FfnBatch, GpuError> {
        if cap == 0 || cap > UNION_MAX_COLS {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a batch of {cap} tokens: 1..={UNION_MAX_COLS}, the host union's"),
            });
        }
        let (ctx, stream) = (gpu.context(), gpu.stream());
        let c = HC_MAX_TOKENS;
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let pinned = |what: &'static str, n: usize| {
            PinnedHostBuffer::<f32>::zeroed(ctx, n).map_err(|source| GpuError::Driver {
                op: Some(what),
                source,
            })
        };
        let counts = 1..=c;
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { ffn_batch_kernels::load(ctx)? };
        Ok(FfnBatch {
            module,
            cap,
            n_embd,
            ff,
            x: z(cap * n_embd)?,
            ids: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            weights: z(cap * N_USED)?,
            sel: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            hc: z(cap * HC_MIX)?,
            acc: z(cap * n_embd)?,
            shexp: z(cap * n_embd)?,
            hsum: z(cap * n_embd)?,
            rout: RouterOut::new(stream)?,
            normed: z(c * n_embd)?,
            act_x: counts
                .clone()
                .map(|m| Q8Act::with_k(stream, m, n_embd))
                .collect::<Result<_, _>>()?,
            mixes: z(c * HC_MIX)?,
            h: z(c * N_USED * ff)?,
            act_h: counts
                .clone()
                .map(|m| {
                    let slots = m * N_USED;
                    if slots <= 8 {
                        Q8Act::with_k(stream, slots, ff)
                    } else {
                        Q8Act::with_slots(stream, slots, ff)
                    }
                })
                .collect::<Result<_, _>>()?,
            down: z(c * N_USED * n_embd)?,
            sh_h: z(c * ff)?,
            act_sh: counts
                .map(|m| Q8Act::with_k(stream, m, ff))
                .collect::<Result<_, _>>()?,
            sh_raw: z(c * n_embd)?,
            host_x: pinned("cuMemAllocHost (the batch's activations)", cap * n_embd)?,
            host_ids: PinnedHostBuffer::<u32>::zeroed(ctx, cap * N_USED).map_err(|source| {
                GpuError::Driver {
                    op: Some("cuMemAllocHost (the batch's routing)"),
                    source,
                }
            })?,
            host_w: pinned("cuMemAllocHost (the batch's routing)", cap * N_USED)?,
            host_sum: pinned("cuMemAllocHost (the batch's host sums)", cap * n_embd)?,
            x_host: Tensor2 {
                ne0: n_embd,
                ne1: cap,
                data: vec![0.0; cap * n_embd],
            },
            routed: ctx.new_event(None)?,
        })
    }

    /// Tokens a batch takes at most.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The HC_PRE results of the last layer's shadow, [`HC_MIX`] a token: the
    /// `pre` that folds the streams into an engram layer's attention and into
    /// the head.
    #[must_use]
    pub fn hc(&self) -> &DeviceBuffer<f32> {
        &self.hc
    }

    /// Device bytes of the buffers, the chunk scratch's included; the
    /// page-locked host copies are not counted.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let f32s = [
            &self.x,
            &self.weights,
            &self.hc,
            &self.acc,
            &self.shexp,
            &self.hsum,
            &self.rout.logits,
            &self.rout.probs,
            &self.rout.weights,
            &self.normed,
            &self.mixes,
            &self.h,
            &self.down,
            &self.sh_h,
            &self.sh_raw,
        ];
        let acts = self.act_x.iter().chain(&self.act_h).chain(&self.act_sh);
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.ids.num_bytes()
            + self.sel.num_bytes()
            + self.rout.ids.num_bytes()
            + ROUTER_TICKET_BYTES
            + acts.map(q8act_bytes).sum::<usize>()
    }

    /// Enqueue the feature tap of `m` tokens: the mean of each one's four
    /// streams (`streams`, `4 · n_embd` a token) into its row of `rows`
    /// (`width` values a row, the mean at `off`), the value the step's tap
    /// writes. One launch. Asynchronous, allocation-free.
    pub fn enqueue_tap_means(
        &self,
        gpu: &Gpu,
        streams: &DeviceBuffer<f32>,
        m: usize,
        width: usize,
        off: usize,
        rows: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n, what) = (self.n_embd, "ds41_tap_means");
        if m == 0 || off + n > width || streams.len() < HC_STREAMS * n * m || rows.len() < width * m
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a tap of {m} tokens at {off} of rows of {width}: streams {}, rows {}",
                    streams.len(),
                    rows.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_tap_means(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_tap_means(
            gpu.stream(),
            &prep,
            streams,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "width", width)?,
            launch_u32(what, "off", off)?,
            rows,
        )?;
        Ok(())
    }

    /// Enqueue the copies of tokens `at .. u`'s activations and routing to
    /// the host, and the event they complete at. Asynchronous.
    pub fn enqueue_download(&mut self, gpu: &Gpu, at: usize, u: usize) -> Result<(), GpuError> {
        self.check_tokens(at, u)?;
        let stream = gpu.stream();
        let (n, s) = (self.n_embd, N_USED);
        // SAFETY: each copy reads the first values of a device buffer this
        // value owns (u ≤ cap, checked above) and writes as many into a
        // page-locked buffer it owns, which the host reads only after
        // `serve`'s wait on the event recorded below, and which no earlier
        // copy still writes (the previous batch layer's serve waited on its
        // own event).
        unsafe {
            dtoh(stream, &mut self.host_x, &self.x, at * n..u * n)?;
            dtoh(stream, &mut self.host_ids, &self.ids, at * s..u * s)?;
            dtoh(stream, &mut self.host_w, &self.weights, at * s..u * s)?;
        }
        self.routed.record(stream)?;
        Ok(())
    }

    /// Wait for the route's copies, then serve layer `layer`'s host experts
    /// for tokens `at .. u` in one union call: the sums into the host copy
    /// [`FfnBatch::enqueue_upload`] sends back.
    pub fn serve<H: HostExperts>(
        &mut self,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        at: usize,
        u: usize,
    ) -> Result<(), GpuError> {
        self.check_tokens(at, u)?;
        self.routed.synchronize()?;
        let n = self.n_embd;
        self.x_host.ne1 = u - at;
        self.x_host.data.clear();
        self.x_host
            .data
            .extend_from_slice(&self.host_x[at * n..u * n]);
        hybrid.serve_batch(
            layer,
            &self.x_host,
            &self.host_ids[at * N_USED..u * N_USED],
            &self.host_w[at * N_USED..u * N_USED],
            &mut self.host_sum[at * n..u * n],
        )
    }

    /// Enqueue the copy of tokens `at .. u`'s host sums to the card.
    /// Asynchronous: the host writes them again only after the next layer's
    /// route event, which this copy precedes on the stream.
    pub fn enqueue_upload(&mut self, gpu: &Gpu, at: usize, u: usize) -> Result<(), GpuError> {
        self.check_tokens(at, u)?;
        let n = self.n_embd;
        // SAFETY: the copy writes values at·n .. u·n of a device buffer this
        // value owns from a page-locked buffer it owns, which the host writes
        // again only in the next serve, after a wait on an event recorded
        // behind this copy.
        unsafe { htod(gpu.stream(), &mut self.hsum, &self.host_sum, at * n..u * n) }
    }

    /// Tokens `at .. u` of a batch: at least one, at most the cap.
    fn check_tokens(&self, at: usize, u: usize) -> Result<(), GpuError> {
        if at >= u || u > self.cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("tokens {at}..{u} of a batch of at most {}", self.cap),
            });
        }
        Ok(())
    }
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `dst` is not read, written or freed until the copy completes.
unsafe fn dtoh<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut PinnedHostBuffer<T>,
    src: &DeviceBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `dst`
    // stays untouched until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().add(at.start).cast(),
            src.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemcpyDtoHAsync_v2 (the batch's handoffs)"),
        source,
    })
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `src` is not written or freed until the copy completes.
unsafe fn htod<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<T>,
    src: &PinnedHostBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `src`
    // stays unwritten until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            src.as_ptr().add(at.start).cast(),
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemcpyHtoDAsync_v2 (the batch's host sums)"),
        source,
    })
}

/// The buffers one chunk of a batch shares with the rest of it: tokens `at
/// .. at + m` of the batch, their streams and folds.
pub struct ChunkIo<'a> {
    /// The chunk's first token in the batch, and its tokens.
    pub at: usize,
    pub m: usize,
    /// The streams the sub-layer reads, `4 · n_embd` a token.
    pub streams: &'a DeviceBuffer<f32>,
    /// The norm's input, `n_embd` a token.
    pub fold_in: &'a DeviceBuffer<f32>,
}

/// The buffers a batch join shares with the rest of the batch: its streams
/// and folds, the batch's tokens from 0 on — the join reads and writes the
/// tokens it is given.
pub struct JoinIo<'a> {
    /// The streams the sub-layer read, `4 · n_embd` a token.
    pub streams: &'a DeviceBuffer<f32>,
    /// The new streams.
    pub streams_out: &'a mut DeviceBuffer<f32>,
    /// The next fold, `n_embd` a token: `Some` exactly where the layer folds.
    pub fold_out: Option<&'a mut DeviceBuffer<f32>>,
}

impl FfnPiece {
    /// Layer `layer`'s route for the chunk `io` of a batch: the norm with its
    /// q8_1 form, the router one token at a time, each slot's place from
    /// `slots`, the map's card copy. Asynchronous, allocation-free.
    pub fn enqueue_batch_route(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        b: &mut FfnBatch,
        layer: usize,
        io: &ChunkIo<'_>,
        slots: &DeviceTensor<u32>,
    ) -> Result<(), GpuError> {
        let i = self.batch_layer(layer, b, io)?;
        let (stream, n, m, at) = (gpu.stream(), self.n_embd, io.m, io.at);
        let c = &self.cfg[i];
        let lw = LayerWeights::resolve(c, w, [n, self.ff], self.hc_eps, self.hc_iters)?;
        let fault = gpu.layer_sink(layer)?;
        let mut x = span_mut(WHAT, &mut b.x, at * n, m * n)?;
        self.fused.enqueue_norm_quant(
            stream,
            io.fold_in,
            lw.gain,
            self.rms_eps,
            count_of(&mut b.act_x, m)?,
            &mut x,
            fault,
        )?;
        drop(x);
        for t in at..at + m {
            let x = span(WHAT, &b.x, t * n, n)?;
            let mut ids = span_mut(WHAT, &mut b.ids, t * N_USED, N_USED)?;
            let mut wts = span_mut(WHAT, &mut b.weights, t * N_USED, N_USED)?;
            self.router.enqueue_router_into(
                stream,
                lw.router,
                &x,
                lw.bias,
                self.scale,
                &mut b.rout,
                &mut ids,
                &mut wts,
                fault,
            )?;
        }
        let what = "ds41_ffn_places";
        let slots_n = m * N_USED;
        let ids = span(WHAT, &b.ids, at * N_USED, slots_n)?;
        let mut sel = span_mut(WHAT, &mut b.sel, at * N_USED, slots_n)?;
        let grid = launch_u32(what, "grid", slots_n.div_ceil(THREADS as usize))?;
        let prep = b
            .module
            .prepare_ds41_ffn_places(LaunchConfig1D::new(grid, THREADS, 0))?;
        b.module.ds41_ffn_places(
            stream,
            &prep,
            &ids,
            slots.buf(),
            launch_u32(what, "row_off", c.row_off)?,
            launch_u32(what, "n_expert", self.n_expert)?,
            launch_u32(what, "n", slots_n)?,
            &mut sel,
        )?;
        Ok(())
    }

    /// Layer `layer`'s shadow work for the chunk `io`, after the batch's
    /// route: HC_PRE into the batch's results, the norm's q8_1 form again,
    /// the card's routed experts over the chunk's slots and their sum, the
    /// shared expert into the batch's outputs. Asynchronous, allocation-free.
    pub fn enqueue_batch_shadow(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        layer: usize,
        io: &ChunkIo<'_>,
    ) -> Result<(), GpuError> {
        let i = self.batch_layer(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        let (stream, n, ff, m, at) = (gpu.stream(), self.n_embd, self.ff, io.m, io.at);
        let c = &self.cfg[i];
        let lw = LayerWeights::resolve(c, w, [n, ff], self.hc_eps, self.hc_iters)?;
        let fault = gpu.layer_sink(layer)?;
        let pre = HcPreArgs {
            params: &lw.hc,
            x: io.streams,
            tokens: m,
            rms_eps: self.rms_eps,
            fault,
        };
        let mut hc = span_mut(WHAT, &mut b.hc, at * HC_MIX, m * HC_MIX)?;
        self.hc
            .enqueue_pre(stream, &pre, &mut self.hc_scratch, &mut b.mixes, &mut hc)?;
        drop(hc);
        let act_x = count_of(&mut b.act_x, m)?;
        self.fused.enqueue_norm_quant(
            stream,
            io.fold_in,
            lw.gain,
            self.rms_eps,
            act_x,
            &mut b.normed,
            fault,
        )?;
        let act_x = &b.act_x[m - 1];
        let slots_n = m * N_USED;
        let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
        if let Some(s) = card {
            let what = "ds41_expert_gate_up_tok";
            let n_sb = act_x.n_sb();
            let grid = launch_u32(what, "grid", (slots_n * ff).div_ceil(8))?;
            let prep = b
                .module
                .prepare_ds41_expert_gate_up_tok(LaunchConfig1D::new(grid, THREADS, 0))?;
            b.module.ds41_expert_gate_up_tok(
                stream,
                &prep,
                s.gate.buf(),
                s.up.buf(),
                act_x.q3(),
                act_x.d8(),
                &sel,
                launch_u32(what, "n_experts", s.gate.rows() / ff)?,
                launch_u32(what, "rows_per_expert", ff)?,
                launch_u32(what, "n_slots", slots_n)?,
                launch_u32(what, "cols", m)?,
                launch_u32(what, "n_sb", n_sb)?,
                launch_u32(what, "iters", n_sb.div_ceil(2))?,
                c.limit,
                &mut b.h,
            )?;
            let act_h = count_of(&mut b.act_h, m)?;
            gpu.enqueue_quantize_q8_1_layer(&b.h, act_h, layer)?;
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                s.down,
                act_h,
                &sel,
                slots_n,
                n,
                &mut b.down,
            )?;
        }
        {
            let what = "ds41_ffn_card_acc";
            let wts = span(WHAT, &b.weights, at * N_USED, slots_n)?;
            let mut acc = span_mut(WHAT, &mut b.acc, at * n, m * n)?;
            let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
            let prep = b
                .module
                .prepare_ds41_ffn_card_acc(LaunchConfig1D::new(grid, THREADS, 0))?;
            b.module.ds41_ffn_card_acc(
                stream,
                &prep,
                &b.down,
                &wts,
                &sel,
                launch_u32(what, "n", n)?,
                launch_u32(what, "m", m)?,
                launch_u32(what, "n_card", c.n_card)?,
                &mut acc,
            )?;
        }
        drop(sel);
        match (lw.sh_gate, lw.sh_up) {
            (
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: g,
                    ..
                },
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: u,
                    ..
                },
            ) => self.dense.enqueue_shexp_gate_up_q3k(
                stream,
                g,
                u,
                &b.act_x[m - 1],
                c.limit_shared,
                &mut b.sh_h,
            )?,
            (gate, up) => {
                for t in 0..m {
                    let x = span(WHAT, &b.x, (at + t) * n, n)?;
                    let mut h = span_mut(WHAT, &mut b.sh_h, t * ff, ff)?;
                    self.experts.enqueue_shexp_gate_up(
                        stream,
                        gate,
                        up,
                        &x,
                        c.limit_shared,
                        &mut h,
                    )?;
                }
            }
        }
        if lw.sh_down.reads_q8_1() != c.sh_down_q8_1 {
            return Err(tensor_err(
                &c.sh_down,
                "of the format the file's header names for it",
            ));
        }
        let act = if lw.sh_down.reads_q8_1() {
            let act_sh = count_of(&mut b.act_sh, m)?;
            gpu.enqueue_quantize_q8_1_layer(&b.sh_h, act_sh, layer)?;
            Some(&b.act_sh[m - 1])
        } else {
            None
        };
        let mut shexp = span_mut(WHAT, &mut b.shexp, at * n, m * n)?;
        if m == 1 {
            self.dense
                .enqueue_m(gpu, lw.sh_down, &b.sh_h, act, 1, &mut shexp)?;
        } else {
            self.dense
                .enqueue_m(gpu, lw.sh_down, &b.sh_h, act, m, &mut b.sh_raw)?;
            self.transpose
                .enqueue(stream, &b.sh_raw, n, m, &mut shexp)?;
        }
        Ok(())
    }

    /// Layer `layer`'s join over the batch's tokens `at .. u`, after the host
    /// sums' upload ([`JoinIo`]). One launch. Asynchronous, allocation-free.
    pub fn enqueue_batch_join(
        &mut self,
        gpu: &Gpu,
        b: &FfnBatch,
        layer: usize,
        at: usize,
        u: usize,
        io: JoinIo<'_>,
    ) -> Result<(), GpuError> {
        let JoinIo {
            streams,
            streams_out,
            fold_out,
        } = io;
        let i = layer
            .checked_sub(self.layers.start)
            .filter(|&i| i < self.cfg.len())
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer} is outside {:?}", self.layers),
            })?;
        let n = self.n_embd;
        if at >= u
            || u > b.cap
            || fold_out.is_some() != self.cfg[i].fold
            || streams.len() < HC_STREAMS * n * u
            || streams_out.len() < HC_STREAMS * n * u
            || fold_out.as_ref().is_some_and(|f| f.len() < n * u)
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer}'s join of tokens {at}..{u} (batches of {}): streams {} and {}, \
                     a fold out {:?} (the layer folds: {})",
                    b.cap,
                    streams.len(),
                    streams_out.len(),
                    fold_out.as_ref().map(|f| f.len()),
                    self.cfg[i].fold
                ),
            });
        }
        let what = "ds41_ffn_post_batch";
        let m = u - at;
        let s4 = HC_STREAMS * n;
        let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let (nn, mm) = (launch_u32(what, "n", n)?, launch_u32(what, "m", m)?);
        let stream = gpu.stream();
        let acc = span(WHAT, &b.acc, at * n, m * n)?;
        let hsum = span(WHAT, &b.hsum, at * n, m * n)?;
        let shexp = span(WHAT, &b.shexp, at * n, m * n)?;
        let hc = span(WHAT, &b.hc, at * HC_MIX, m * HC_MIX)?;
        let res = span(WHAT, streams, at * s4, m * s4)?;
        let mut out = span_mut(WHAT, streams_out, at * s4, m * s4)?;
        match fold_out {
            Some(fold) => {
                let mut fold = span_mut(WHAT, fold, at * n, m * n)?;
                let prep = b.module.prepare_ds41_ffn_post_batch(cfg)?;
                b.module.ds41_ffn_post_batch(
                    stream, &prep, &acc, &hsum, &shexp, &res, &hc, nn, mm, &mut out, &mut fold,
                )?;
            }
            None => {
                let prep = b.module.prepare_ds41_ffn_post_batch_streams(cfg)?;
                b.module.ds41_ffn_post_batch_streams(
                    stream, &prep, &acc, &hsum, &shexp, &res, &hc, nn, mm, &mut out,
                )?;
            }
        }
        Ok(())
    }

    /// Layer `layer`'s index, once the chunk `io` is checked against the
    /// batch's buffers.
    fn batch_layer(&self, layer: usize, b: &FfnBatch, io: &ChunkIo<'_>) -> Result<usize, GpuError> {
        let n = self.n_embd;
        let i = layer
            .checked_sub(self.layers.start)
            .filter(|&i| i < self.cfg.len())
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer} is outside {:?}", self.layers),
            })?;
        if !(1..=HC_MAX_TOKENS).contains(&io.m)
            || io.at + io.m > b.cap
            || b.n_embd != n
            || b.ff != self.ff
            || io.streams.len() < HC_STREAMS * n * io.m
            || io.fold_in.len() < n * io.m
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a chunk of {} tokens from {} in a batch of {} (rows of {} through {}): streams \
                     {}, fold {}; the piece's rows are {n} through {}",
                    io.m,
                    io.at,
                    b.cap,
                    b.n_embd,
                    b.ff,
                    io.streams.len(),
                    io.fold_in.len(),
                    self.ff
                ),
            });
        }
        Ok(i)
    }
}

/// The scratch of token count `m` in `by_count` (index `m − 1`).
fn count_of<T>(by_count: &mut [T], m: usize) -> Result<&mut T, GpuError> {
    let n = by_count.len();
    m.checked_sub(1)
        .and_then(|i| by_count.get_mut(i))
        .ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("a chunk of {m} tokens; the batch holds scratch for 1..={n}"),
        })
}
