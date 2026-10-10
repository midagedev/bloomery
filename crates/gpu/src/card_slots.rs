//! The card slots' places and their weighted sum: the launch that resolves a
//! batch's routed ids to places in the card's stacks, and the launch that
//! sums the card slots' down rows. Both are common to every program that runs
//! experts on the card ([`crate::card_leg`] runs the launches between them):
//! each reads `m` tokens of `n_used` routed slots, slot-major.
//!
//! - **Places** (`ds41_ffn_places`): the handoff's rule over a batch's ids,
//!   `sel[i] = map[row_off + ids[i]]`, the id's slot in the card's routed
//!   stacks, or [`HOST`] for an expert the card does not hold; an id not below
//!   the layer's expert count has no place and raises
//!   [`FaultSite::ExpertId`].
//! - **Sum** (`ds41_ffn_card_acc` for six slots a token, `ds41_ffn_card_acc_8`
//!   for eight, any other count refused by name): value `d` of token `t` is
//!   `acc = fma(down[(Nt + j)·n + d], w[Nt + j], acc)` from zero for each slot
//!   `j` whose place is below `n_card`, in ascending `j` (this order is the
//!   gate, `runtime::combine::card_sum`). No other slot's rows are read.
//!
//! The entries keep the names the DeepSeek-V4.1 family gave them; a family
//! whose sum is another rule (a shared expert, the expert tier's rows, a fault
//! on a place no side serves) keeps its own.

use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use runtime::combine::Slots;
use std::sync::Arc;

/// Threads per block of every kernel here.
const THREADS: u32 = 256;
// The kernels' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Slots a token of the six-slot sum, whose contract spells it as a literal.
pub const SLOTS_6: usize = 6;
/// Slots a token of the `_8` sum, whose contract spells it as a literal.
pub const SLOTS_8: usize = 8;
// The instance table's counts are the entries' own.
const _: () = assert!(Slots::Six.n() == SLOTS_6 && Slots::Eight.n() == SLOTS_8);

/// The card half of one output value's combine: `acc = fma(down_j, w_j, acc)`
/// from zero for each slot `j` whose place is on the card, in slot order.
#[inline(always)]
pub fn card_sum_elem(down: [f32; SLOTS_6], w: [f32; SLOTS_6], card: [bool; SLOTS_6]) -> f32 {
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < SLOTS_6 {
        if card[j] {
            acc = down[j].mul_add(w[j], acc);
        }
        j += 1;
    }
    acc
}

#[cuda_module]
mod card_slots_kernels {
    use super::*;

    /// Each slot's place, the handoff's rule: thread `i < n` writes `sel[i] =
    /// map[row_off + ids[i]]` — the id's slot in the card's routed stacks, or
    /// [`HOST`]. An id not below `n_expert` has no place: it raises
    /// [`FaultSite::ExpertId`] on `fault` and its place is [`HOST`], so no
    /// card kernel reads a row for it.
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
        fault: FaultSink,
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
            fault.raise(FaultSite::ExpertId);
            HOST
        };
        // SAFETY: i < n <= sel.len(); thread i is sel[i]'s only writer.
        unsafe { *sel.get_unchecked_mut(i) = place };
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
        let mut dv = [0.0f32; SLOTS_6];
        let mut wv = [0.0f32; SLOTS_6];
        let mut card = [false; SLOTS_6];
        let mut j = 0usize;
        while j < SLOTS_6 {
            let s = t * SLOTS_6 + j;
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

    /// [`ds41_ffn_card_acc`] of eight slots a token ([`card_acc_at`]).
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
            down.len() >= 8 * n * m,
            w.len() >= 8 * m,
            sel.len() >= 8 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_8(
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
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_8 slots.
        let v = unsafe { card_acc_at::<SLOTS_8>(down, w, sel, n, n_card, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }
}

/// The card slots' sum of value `i` (token `t = i / n`, value `d = i % n`)
/// of `N` slots a token: `acc = fma(down[(Nt + j)·n + d], w[Nt + j], acc)`
/// from zero for each slot `j` below `N` whose place `sel[Nt + j]` is below
/// `n_card`, in ascending `j` — the order `runtime::combine::card_sum` pins
/// (this order is the gate); no other slot's rows are read.
///
/// SAFETY: `i < n · m` for an `m` with `down.len() >= N · n · m`, `w.len()`
/// and `sel.len() >= N · m`.
#[inline(always)]
unsafe fn card_acc_at<const N: usize>(
    down: &[f32],
    w: &[f32],
    sel: &[u32],
    n: usize,
    n_card: u32,
    i: usize,
) -> f32 {
    let (t, d) = (i / n, i % n);
    let mut acc = 0.0f32;
    for j in 0..N {
        cuda_device::thread::__unroll_config::<0>();
        let s = t * N + j;
        // SAFETY: s < N·m <= sel.len() and w.len() by this fn's contract.
        let (place, ws) = unsafe { (*sel.get_unchecked(s), *w.get_unchecked(s)) };
        if place < n_card {
            // SAFETY: s < N·m and d < n, so s·n + d < N·n·m <= down.len().
            let ds = unsafe { *down.get_unchecked(s * n + d) };
            acc = ds.mul_add(ws, acc);
        }
    }
    acc
}

/// What [`CardSlotKernels::enqueue_places`] reads: `n` routed ids, and the
/// slot map's card copy with the layer's row at `row_off` (`n_expert` places
/// a row).
pub struct Places<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}

/// What [`CardSlotKernels::enqueue_card_acc`] reads, `m` tokens of `n_used`
/// slots: `down` the slots' down outputs slot-major (`n` each, token `t`'s
/// slots `n_used · t ..`), `w` their routing weights and `sel` their places
/// (the card's below `n_card`; no other slot's rows are read).
pub struct CardAcc<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_used: usize,
}

/// The launches' module: loaded once per context by its owner, or alone by a
/// gate that drives them.
pub struct CardSlotKernels {
    module: card_slots_kernels::LoadedModule,
}

impl CardSlotKernels {
    /// Load the entries' device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<CardSlotKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; each launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(card_slots_kernels, ctx)? };
        Ok(CardSlotKernels { module })
    }

    /// Enqueue `ds41_ffn_places`: `sel[i]` the place of `p.ids[i]` for `i <
    /// p.n`, [`HOST`] and [`FaultSite::ExpertId`] on `fault` for an id past
    /// the stack. One launch. Asynchronous, allocation-free.
    pub fn enqueue_places(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_ffn_places";
        let grid = launch_u32(what, "grid", p.n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_places(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_ffn_places(
            stream,
            &prep,
            p.ids,
            p.map,
            launch_u32(what, "row_off", p.row_off)?,
            launch_u32(what, "n_expert", p.n_expert)?,
            launch_u32(what, "n", p.n)?,
            fault,
            sel,
        )?;
        Ok(())
    }

    /// Enqueue the card sums of `a.m` tokens of `a.n_used` slots
    /// (`ds41_ffn_card_acc` for 6, `ds41_ffn_card_acc_8` for 8; any other
    /// count is refused by name): value `i` of token `t = i / a.n` into
    /// `acc[i]`, the card half of the combine. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_card_acc(
        &self,
        stream: &CudaStream,
        a: &CardAcc<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let slots = Slots::of(a.n_used).map_err(|e| GpuError::Shape {
            what: "ds41_ffn_card_acc",
            detail: e.to_string(),
        })?;
        let what = match slots {
            Slots::Six => "ds41_ffn_card_acc",
            Slots::Eight => "ds41_ffn_card_acc_8",
        };
        let grid = launch_u32(what, "grid", (a.m * a.n).div_ceil(THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let (n, m, n_card) = (
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
        );
        match slots {
            Slots::Six => {
                let prep = self.module.prepare_ds41_ffn_card_acc(cfg)?;
                self.module
                    .ds41_ffn_card_acc(stream, &prep, a.down, a.w, a.sel, n, m, n_card, acc)?;
            }
            Slots::Eight => {
                let prep = self.module.prepare_ds41_ffn_card_acc_8(cfg)?;
                self.module
                    .ds41_ffn_card_acc_8(stream, &prep, a.down, a.w, a.sel, n, m, n_card, acc)?;
            }
        }
        Ok(())
    }
}
