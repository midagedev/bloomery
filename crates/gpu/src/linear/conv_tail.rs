//! The residual short conv of Inkling 2's post convs ([`super::conv_res`]'s rule at one token, a
//! static conv of [`CONV_TAPS`] taps) run in the tail of the launch that produces its input, so a
//! visit's two post convs and the stream's joins are not launches of their own.
//!
//! Two entries, each channel's value bit for bit what the launches it replaces write:
//!
//! - `q8_0_gemv_conv_join`: the output projection (`q8_0_gemv` at one column) with the attention
//!   post conv in lane 0 of each row's warp, which then adds the stream to the conv's output
//!   where the visit joins by addition (a block's first visit banks the stream instead and
//!   joins nothing here). The conv's loads (the ring's slots, the filter, the stream's value) are
//!   issued before the row's weight walk, so they complete under it.
//! - `card_acc_8_conv_add`: the card slots' sum of eight slots a token
//!   ([`crate::card_slots::card_acc_at`], the sum's one owner), the feed-forward post conv of the
//!   sum and the stream's add, one thread a channel; the conv's loads are issued before the sum's.
//!
//! Every value the replaced launches leave in a buffer is written here too: the projection's
//! output is the conv's input and goes only into the ring, the conv's output, the sum and the
//! join are each stored as they were.
//!
//! Faults as [`super::conv_res`]: a conv output that is not finite raises
//! [`FaultSite::LinearConv`] and is written NaN; the sum's own sites are `card_acc_at`'s.

use super::ring_input;
use crate::card_slots::{SLOTS_8, card_acc_at};
use crate::fault::{FaultSink, FaultSite};
use crate::q8f32::q8_0_lane_partial_1col;
use crate::qk_clip::CONV_TAPS;
use crate::{DeviceTensor, GpuError, launch_u32};
use chain::linear::Ring;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a block of both entries.
const THREADS: u32 = 256;
/// Rows of the projection's block: one a warp.
const ROWS: usize = 8;
// The entries' `launch_bounds` and `launch_contract` blocks are literals, and the unrolled taps
// below are written for four.
const _: () = assert!(THREADS == 256 && ROWS == 8 && CONV_TAPS == 4 && SLOTS_8 == 8);

/// The conv's filters, position and ring as a launch holds them, all for the one token of the
/// call: channel `c` of the conv is channel `base + c` of the ring's rows of `ch`.
#[derive(Clone, Copy)]
struct TailConv<'a> {
    /// The static filters `[d][CONV_TAPS]`, the oldest tap first.
    w: &'a [f32],
    /// The token's position at word 0.
    pos: &'a [u32],
    ring: *mut f32,
    rows: usize,
    ch: usize,
    base: usize,
    scale: f32,
}

/// What a channel's conv reads besides its input: the inputs of the three positions before the
/// token and the four filter taps.
#[derive(Clone, Copy)]
struct Gathered {
    p: usize,
    past: [f32; 3],
    tap: [f32; CONV_TAPS],
}

impl TailConv<'_> {
    /// Channel `c`'s reads, all issued together and none depending on the channel's input.
    ///
    /// # Safety
    ///
    /// `pos.len() >= 1`, `w.len() >= (c + 1) · CONV_TAPS`, `base + c < ch`, `rows > 3`, and `ring`
    /// addresses `rows · ch` readable f32s that no other thread writes while this one runs.
    #[inline(always)]
    unsafe fn gather(&self, c: usize) -> Gathered {
        // SAFETY: pos.len() >= 1 by the fn's contract.
        let p = unsafe { *self.pos.get_unchecked(0) } as usize;
        let at = self.base + c;
        let ring = self.ring.cast_const();
        // SAFETY: the fn's contract: the ring's rows, `at < ch`, `rows > back >= 1`; the filter's
        // four taps of channel c.
        unsafe {
            Gathered {
                p,
                past: [
                    ring_input(ring, self.rows, p, 1, self.ch, at),
                    ring_input(ring, self.rows, p, 2, self.ch, at),
                    ring_input(ring, self.rows, p, 3, self.ch, at),
                ],
                tap: [
                    *self.w.get_unchecked(c * CONV_TAPS),
                    *self.w.get_unchecked(c * CONV_TAPS + 1),
                    *self.w.get_unchecked(c * CONV_TAPS + 2),
                    *self.w.get_unchecked(c * CONV_TAPS + 3),
                ],
            }
        }
    }

    /// Channel `c`'s conv output for the input `cur` ([`super::conv_res`]'s rule: the first tap's
    /// product rounds alone, the rest fuse ascending, the residual joins cur-first, the scale
    /// last), the input stored into its ring slot after the reads, and [`FaultSite::LinearConv`]
    /// raised, the value NaN, for an output that is not finite.
    ///
    /// # Safety
    ///
    /// `g` is `gather(c)`'s, and this thread alone touches channel `c` of the ring.
    #[inline(always)]
    unsafe fn finish(&self, g: &Gathered, c: usize, cur: f32, fault: FaultSink) -> f32 {
        let mut acc = mul_rn_f32(g.past[2], g.tap[0]);
        acc = fma_rn_f32(g.tap[1], g.past[1], acc);
        acc = fma_rn_f32(g.tap[2], g.past[0], acc);
        acc = fma_rn_f32(g.tap[3], cur, acc);
        let out = mul_rn_f32(add_rn_f32(cur, acc), self.scale);
        // SAFETY: slot < rows and base + c < ch put the index inside the ring; the channel is
        // this thread's alone and its reads are done.
        unsafe { *self.ring.add(g.p % self.rows * self.ch + self.base + c) = cur };
        if out.is_finite() {
            out
        } else {
            fault.raise(FaultSite::LinearConv);
            f32::NAN
        }
    }
}

#[cuda_module]
mod conv_tail_kernels {
    use super::*;

    /// `q8_0_gemv` at one column over `n_rows` rows of `k` values (one warp a row, eight rows a
    /// block) whose row sum is the post conv's input (module doc): row `r`'s lane 0 writes the
    /// conv's output to `attn_out[r]` and, when `join` is 1, `stream[r] + attn_out[r]` to
    /// `joined[r]`. `w` the conv's filters, `pos` its token's position, `ring` the conv ring (`rows`
    /// slots of `ch` channels, this conv's from `base`), read and written in place.
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
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= k,
            w.len() >= n_rows * 4,
            pos.len() >= 1,
            stream.len() >= n_rows,
            n_rows >= 1,
            rows >= 4,
            base + n_rows <= ch,
            attn_out.len() >= n_rows,
            joined.len() >= n_rows,
            ring.len() >= rows * ch
        )
    )]
    pub fn q8_0_gemv_conv_join(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        w: &[f32],
        pos: &[u32],
        stream: &[f32],
        n_rows: u32,
        k: u32,
        rows: u32,
        ch: u32,
        base: u32,
        join: u32,
        scale: f32,
        fault: FaultSink,
        mut attn_out: DisjointSlice<f32>,
        mut joined: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * ROWS + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let conv = TailConv {
            w,
            pos,
            ring: ring.as_mut_ptr(),
            rows: rows as usize,
            ch: ch as usize,
            base: base as usize,
            scale,
        };
        // Lane 0's reads go out before the row's weight walk.
        let (g, s_in) = if lane == 0 {
            // SAFETY: row < n_rows, so the filter and the ring's channel base + row are inside
            // what the launch contract bounds; stream holds n_rows values; only this warp's lane 0
            // touches the ring's channel.
            unsafe {
                (
                    conv.gather(row),
                    if join != 0 {
                        *stream.get_unchecked(row)
                    } else {
                        0.0
                    },
                )
            }
        } else {
            (
                Gathered {
                    p: 0,
                    past: [0.0; 3],
                    tap: [0.0; CONV_TAPS],
                },
                0.0,
            )
        };
        // SAFETY: row < n_rows puts the row's words and scales inside qs and d (the contract's
        // bounds), and x.len() >= k is the one column the body reads from x0 = 0.
        let f = unsafe { q8_0_lane_partial_1col(qs, d, x, k, row, 0, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: `g` is gather(row)'s; row < n_rows <= attn_out.len(), joined.len(), and
            // only lane 0 of the row's warp writes slot row.
            unsafe {
                let out = conv.finish(&g, row, s, fault);
                *attn_out.get_unchecked_mut(row) = out;
                if join != 0 {
                    *joined.get_unchecked_mut(row) = add_rn_f32(s_in, out);
                }
            }
        }
    }

    /// The card slots' sum of one token's eight slots over `n` values ([`card_acc_at`]), then
    /// the post conv of the sum and the stream's add (module doc): thread
    /// `i < n` writes the sum to `acc[i]`, the conv's output to `conv_out[i]` and
    /// `inp[i] + conv_out[i]` to `y[i]`. `down`, `wts`, `sel`, `n_card` and `w_pitch` are the sum's;
    /// the conv's `cw`, `pos`, `ring`, `rows`, `ch` and `base` are the projection entry's.
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
            w_pitch >= 8,
            down.len() >= 8 * n,
            wts.len() >= w_pitch,
            sel.len() >= 8,
            cw.len() >= n * 4,
            pos.len() >= 1,
            inp.len() >= n,
            rows >= 4,
            base + n <= ch,
            acc.len() >= n,
            conv_out.len() >= n,
            y.len() >= n,
            ring.len() >= rows * ch
        )
    )]
    pub fn card_acc_8_conv_add(
        down: &[f32],
        wts: &[f32],
        sel: &[u32],
        cw: &[f32],
        pos: &[u32],
        inp: &[f32],
        n: u32,
        n_card: u32,
        w_pitch: u32,
        rows: u32,
        ch: u32,
        base: u32,
        scale: f32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
        mut conv_out: DisjointSlice<f32>,
        mut y: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let conv = TailConv {
            w: cw,
            pos,
            ring: ring.as_mut_ptr(),
            rows: rows as usize,
            ch: ch as usize,
            base: base as usize,
            scale,
        };
        // SAFETY: i < n, so the filter, the ring's channel base + i and inp[i] are inside what the
        // launch contract bounds; thread i alone touches the ring's channel.
        let (g, h) = unsafe { (conv.gather(i), *inp.get_unchecked(i)) };
        // SAFETY: i < n, and the launch contract gives every length the helper's contract asks
        // for at SLOTS_8 slots of one token.
        let a = unsafe {
            card_acc_at::<SLOTS_8>(
                down,
                wts,
                sel,
                n as usize,
                n_card,
                w_pitch as usize,
                fault,
                i,
            )
        };
        // SAFETY: `g` is gather(i)'s; i < n bounds acc, conv_out and y, and thread i is their
        // slot i's only writer.
        unsafe {
            let out = conv.finish(&g, i, a, fault);
            *acc.get_unchecked_mut(i) = a;
            *conv_out.get_unchecked_mut(i) = out;
            *y.get_unchecked_mut(i) = add_rn_f32(h, out);
        }
    }
}

/// The conv a tail runs: its filters, the ring it reads and writes in place (this conv's channels
/// from `base`, `ring.rows()` slots of `ch`), the token's position and the output scale.
pub struct TailConvArgs<'a> {
    pub ring: Ring,
    /// The ring's row width: the visit's whole channel width.
    pub ch: usize,
    /// The conv's first channel in the ring's rows.
    pub base: usize,
    /// The output scale: `1 / √(taps + 1)`.
    pub scale: f32,
    /// The filters `[d][CONV_TAPS]`, the oldest tap first.
    pub weights: &'a DeviceBuffer<f32>,
    /// The token's position: the word the embedding launch writes.
    pub pos: &'a DeviceBuffer<u32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// [`ConvTailKernels::enqueue_gemv_conv`]'s arguments: the projection `qs`, `d` over `x`, the
/// conv, the stream row `stream` and where the join goes. `out` takes the conv's output; with
/// `join` the join `stream + out` goes to `joined`, without it `stream` and `joined` are not read
/// or written.
pub struct GemvConvArgs<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub x: &'a DeviceBuffer<f32>,
    pub conv: TailConvArgs<'a>,
    pub join: bool,
    pub stream: &'a DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub out: &'a mut DeviceBuffer<f32>,
    pub joined: &'a mut DeviceBuffer<f32>,
}

/// [`ConvTailKernels::enqueue_card_acc_conv`]'s arguments: the card slots' sum of one token's
/// eight slots (the down rows `down` slot-major `[8][n]`, the routing weights `w`, the places
/// `sel`, the card's `n_card` places), the conv, and the stream row `inp` the conv's output joins.
/// `acc` takes the sum, `conv_out` the conv's output and `y` `inp + conv_out`.
pub struct CardAccConvArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
    pub n: usize,
    pub conv: TailConvArgs<'a>,
    pub inp: &'a DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub acc: &'a mut DeviceBuffer<f32>,
    pub conv_out: &'a mut DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ConvTailKernels {
    module: conv_tail_kernels::LoadedModule,
}

impl ConvTailKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ConvTailKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(conv_tail_kernels, ctx)? };
        Ok(ConvTailKernels { module })
    }

    /// The conv's host checks for a launch of `d` channels: a ring of [`CONV_TAPS`] taps, a
    /// channel range inside the ring's rows, a finite scale, and buffers long enough. Returns
    /// `(rows, ch, base)` as the launch's `u32`s.
    fn conv_dims(
        what: &'static str,
        d: usize,
        c: &TailConvArgs<'_>,
    ) -> Result<(u32, u32, u32), GpuError> {
        if c.ring.taps() != CONV_TAPS {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} conv taps: the tail forms are built for {CONV_TAPS}",
                    c.ring.taps()
                ),
            ));
        }
        if c.base + d > c.ch {
            return Err(GpuError::shape(
                what,
                format!(
                    "channels {}..{} of a ring row of {}: the conv's range is inside it",
                    c.base,
                    c.base + d,
                    c.ch
                ),
            ));
        }
        if !c.scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("scale {}: a finite one", c.scale),
            ));
        }
        let rows = c.ring.rows();
        crate::short(
            what,
            &[
                ("weights", c.weights.len(), d * CONV_TAPS),
                ("pos", c.pos.len(), 1),
                ("state", c.state.len(), rows * c.ch),
            ],
        )?;
        Ok((
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "ch", c.ch)?,
            launch_u32(what, "base", c.base)?,
        ))
    }

    /// Enqueue the output projection of one column with its post conv and join in the tail
    /// (`q8_0_gemv_conv_join`): `⌈rows / 8⌉` blocks of 256 threads. Refuses by name a projection
    /// of no row or whose width is not a multiple of 32, planes of another shape, a ring that is
    /// not [`CONV_TAPS`] taps, a channel range outside the ring's rows and a buffer shorter than
    /// the launch needs. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_conv(
        &self,
        stream: &CudaStream,
        args: GemvConvArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "conv_tail::enqueue_gemv_conv";
        let GemvConvArgs {
            qs,
            d,
            x,
            conv,
            join,
            stream: row,
            fault,
            out,
            joined,
        } = args;
        let (n_rows, k) = (d.rows(), d.cols() * 32);
        if n_rows == 0 || k == 0 || qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "planes of {n_rows} rows: qs is {}x{}, d is {}x{} (k/4 words and k/32 scales a \
                     row, k a positive multiple of 32)",
                    qs.rows(),
                    qs.cols(),
                    d.rows(),
                    d.cols()
                ),
            ));
        }
        let (rows, ch, base) = Self::conv_dims(WHAT, n_rows, &conv)?;
        crate::short(
            WHAT,
            &[
                ("x", x.len(), k),
                ("stream", row.len(), n_rows),
                ("out", out.len(), n_rows),
                ("joined", joined.len(), n_rows),
            ],
        )?;
        let grid = launch_u32(WHAT, "grid", n_rows.div_ceil(ROWS))?;
        let (n_rows, k) = (
            launch_u32(WHAT, "n_rows", n_rows)?,
            launch_u32(WHAT, "k", k)?,
        );
        let prep = self
            .module
            .prepare_q8_0_gemv_conv_join(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.q8_0_gemv_conv_join(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            conv.weights,
            conv.pos,
            row,
            n_rows,
            k,
            rows,
            ch,
            base,
            u32::from(join),
            conv.scale,
            fault,
            out,
            joined,
            conv.state,
        )?;
        Ok(())
    }

    /// Enqueue the card slots' sum of eight slots with the post conv and the stream's add in its
    /// tail (`card_acc_8_conv_add`): `⌈n / 256⌉` blocks of 256 threads. Refuses by name a call of
    /// no value, a ring that is not [`CONV_TAPS`] taps, a channel range outside the ring's rows
    /// and a buffer shorter than the launch needs. Asynchronous, allocation-free, capturable.
    pub fn enqueue_card_acc_conv(
        &self,
        stream: &CudaStream,
        args: CardAccConvArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "conv_tail::enqueue_card_acc_conv";
        const SLOTS: usize = SLOTS_8;
        let CardAccConvArgs {
            down,
            w,
            sel,
            n_card,
            n,
            conv,
            inp,
            fault,
            acc,
            conv_out,
            y,
        } = args;
        if n == 0 {
            return Err(GpuError::shape(WHAT, "need n >= 1".to_owned()));
        }
        let (rows, ch, base) = Self::conv_dims(WHAT, n, &conv)?;
        crate::short(
            WHAT,
            &[
                ("down", down.len(), SLOTS * n),
                ("w", w.len(), SLOTS),
                ("sel", sel.len(), SLOTS),
                ("inp", inp.len(), n),
                ("acc", acc.len(), n),
                ("conv_out", conv_out.len(), n),
                ("y", y.len(), n),
            ],
        )?;
        let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS as usize))?;
        let n = launch_u32(WHAT, "n", n)?;
        let n_card = launch_u32(WHAT, "n_card", n_card)?;
        let prep = self
            .module
            .prepare_card_acc_8_conv_add(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.card_acc_8_conv_add(
            stream,
            &prep,
            down,
            w,
            sel,
            conv.weights,
            conv.pos,
            inp,
            n,
            n_card,
            SLOTS as u32,
            rows,
            ch,
            base,
            conv.scale,
            fault,
            acc,
            conv_out,
            y,
            conv.state,
        )?;
        Ok(())
    }
}
