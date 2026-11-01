//! The card slots' places and their weighted sum: the launch that resolves a
//! batch's routed ids to places in the card's stacks, and the launch that
//! sums the card slots' down rows. Both are common to every program that runs
//! experts on the card ([`crate::card_leg`] runs the launches between them):
//! each reads `m` tokens of `n_used` routed slots, slot-major.
//!
//! - **Places** (`ds41_ffn_places`): the handoff's rule over `m` columns of
//!   routed ids, `sel[c·sel_pitch + e] = map[row_off + ids[c·ids_pitch +
//!   e]]` for the routed slots `e < used`, the id's slot in the card's routed
//!   stacks, or [`HOST`] for an expert the card does not hold; an id not
//!   below the layer's expert count has no place and raises
//!   [`FaultSite::ExpertId`]. A lane `e >= used` of a `sel_pitch` wider than
//!   `used` is [`HOST`] and reads no id (the router's shared slot); an
//!   `ids_pitch` wider than `used` leaves the router's extra words unread.
//! - **Sum** (`ds41_ffn_card_acc` for six slots a token, `_8` for eight,
//!   `_10` for ten, any other count refused by name): value `d` of token `t`
//!   is `acc = fma(down[(Nt + j)·n + d], w[Nt·p/N + j], acc)` from zero for
//!   each slot `j` whose place is below `n_card`, in ascending `j` (this
//!   order is the gate, `runtime::combine::card_sum`), `p` the weights' pitch
//!   a token (`N` or the router's `N + 1`). No other slot's rows or weight are
//!   read. A place in `[n_card, HOST)` names neither a card slot nor the
//!   host: it raises [`FaultSite::ExpertId`] and its token's values are NaN.
//!
//! - **Join** (`card_sum_join`, `_bf16` over BF16 down rows): the sum above
//!   at its slot counts (the count as data), then the shared expert's output added in the same
//!   launch, `(base + acc) + sh · g` with each step rounded — `g` a router
//!   weight or 1, `base` the zeros of an empty host list or nothing — so a
//!   layer whose host part is added later, or that has none, writes no sum
//!   and reads it back. A non-finite input or result of the join raises
//!   [`FaultSite::F32Product`].
//!
//! The entries keep the names the DeepSeek-V4.1 family gave them; no family
//! keeps its own sum for the fault, the weights' pitch or the shared
//! expert's join. A family whose sum is another rule (the expert tier's
//! rows) keeps its own.

use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::q8f32::RowValues;
use crate::{GpuError, launch_u32, short};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of every kernel here.
const THREADS: u32 = 256;
// The kernels' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Slots a token of the `_4` sum, whose contract spells it as a literal.
pub const SLOTS_4: usize = 4;
/// Slots a token of the six-slot sum, whose contract spells it as a literal.
pub const SLOTS_6: usize = 6;
/// Slots a token of the `_8` sum, whose contract spells it as a literal.
pub const SLOTS_8: usize = 8;
/// Slots a token of the `_9` sum, whose contract spells it as a literal.
pub const SLOTS_9: usize = 9;
/// Slots a token of the `_10` sum, whose contract spells it as a literal.
pub const SLOTS_10: usize = 10;

/// The routed slot counts the sum has an entry for: one entry per count,
/// picked at launch, and every other count refused by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardSlots {
    /// Four slots a token (MiniMax-M3).
    Four,
    /// Six slots a token (DeepSeek-V4.1).
    Six,
    /// Eight slots a token (GLM-5.3-Flash).
    Eight,
    /// Nine slots a token (Bolt).
    Nine,
    /// Ten slots a token, the router's weights at a pitch of eleven
    /// (Qwen3.8).
    Ten,
}

impl CardSlots {
    /// The instance for `n_used` slots a token, or `None`.
    #[must_use]
    pub const fn of(n_used: usize) -> Option<CardSlots> {
        match n_used {
            SLOTS_4 => Some(CardSlots::Four),
            SLOTS_6 => Some(CardSlots::Six),
            SLOTS_8 => Some(CardSlots::Eight),
            SLOTS_9 => Some(CardSlots::Nine),
            SLOTS_10 => Some(CardSlots::Ten),
            _ => None,
        }
    }

    /// The instance's slot count.
    #[must_use]
    pub const fn n(self) -> usize {
        match self {
            CardSlots::Four => SLOTS_4,
            CardSlots::Six => SLOTS_6,
            CardSlots::Eight => SLOTS_8,
            CardSlots::Nine => SLOTS_9,
            CardSlots::Ten => SLOTS_10,
        }
    }

    /// The entry's name.
    #[must_use]
    const fn entry(self) -> &'static str {
        match self {
            CardSlots::Four => "ds41_ffn_card_acc_4",
            CardSlots::Six => "ds41_ffn_card_acc",
            CardSlots::Eight => "ds41_ffn_card_acc_8",
            CardSlots::Nine => "ds41_ffn_card_acc_9",
            CardSlots::Ten => "ds41_ffn_card_acc_10",
        }
    }

    /// The count `n_used` refused by name, as the launchers report it.
    pub(crate) fn refusal(n_used: usize) -> GpuError {
        GpuError::Shape {
            what: "ds41_ffn_card_acc",
            detail: format!(
                "{n_used} routed slots a token: the card sum's entries are built for 4, 6, 8, 9 and 10"
            ),
        }
    }
}

/// The card half of one output value's combine: `acc = fma(down_j, w_j, acc)`
/// from zero for each slot `j` whose place is on the card (`card[j]`), in slot
/// order.
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

    /// Each slot's place, the handoff's rule: thread `k < m·sel_pitch` is
    /// column `c = k / sel_pitch`, lane `e = k % sel_pitch`, and writes
    /// `sel[k] = map[row_off + ids[c·ids_pitch + e]]` — the id's slot in the
    /// card's routed stacks, or [`HOST`] — for a routed lane `e < used`; a
    /// lane `e >= used` is [`HOST`] and reads no id. An id not below
    /// `n_expert` has no place: it raises [`FaultSite::ExpertId`] on `fault`
    /// and its place is [`HOST`], so no card kernel reads a row for it.
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
            used >= 1,
            used <= ids_pitch,
            used <= sel_pitch,
            ids.len() >= m * ids_pitch,
            map.len() >= row_off + n_expert,
            sel.len() >= m * sel_pitch
        )
    )]
    pub fn ds41_ffn_places(
        ids: &[u32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        ids_pitch: u32,
        used: u32,
        sel_pitch: u32,
        m: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= m as usize * sel_pitch as usize {
            return;
        }
        let (c, e) = (k / sel_pitch as usize, k % sel_pitch as usize);
        let place = if e >= used as usize {
            HOST
        } else {
            // SAFETY: c < m and e < used <= ids_pitch put c·ids_pitch + e
            // below m·ids_pitch <= ids.len() by the launch contract.
            let id = unsafe { *ids.get_unchecked(c * ids_pitch as usize + e) };
            if id < n_expert {
                // SAFETY: id < n_expert, so row_off + id < map.len() by the
                // launch contract.
                unsafe { *map.get_unchecked(row_off as usize + id as usize) }
            } else {
                fault.raise(FaultSite::ExpertId);
                HOST
            }
        };
        // SAFETY: k < m·sel_pitch <= sel.len(); thread k is sel[k]'s only
        // writer.
        unsafe { *sel.get_unchecked_mut(k) = place };
    }

    /// The card slots' sum of `m` tokens: thread `i < m·n` is token `t = i /
    /// n`, value `d = i % n`, and writes [`card_acc_at`] of its six slots to
    /// `acc[i]`.
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
            w_pitch >= 6,
            down.len() >= 6 * n * m,
            w.len() >= w_pitch * m,
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
        w_pitch: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_6 slots.
        let v =
            unsafe { card_acc_at::<SLOTS_6>(down, w, sel, n, n_card, w_pitch as usize, fault, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
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
            w_pitch >= 8,
            down.len() >= 8 * n * m,
            w.len() >= w_pitch * m,
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
        w_pitch: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_8 slots.
        let v =
            unsafe { card_acc_at::<SLOTS_8>(down, w, sel, n, n_card, w_pitch as usize, fault, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }

    /// [`ds41_ffn_card_acc`] of ten slots a token ([`card_acc_at`]).
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
            w_pitch >= 10,
            down.len() >= 10 * n * m,
            w.len() >= w_pitch * m,
            sel.len() >= 10 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_10(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        w_pitch: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_10 slots.
        let v =
            unsafe { card_acc_at::<SLOTS_10>(down, w, sel, n, n_card, w_pitch as usize, fault, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }

    /// [`ds41_ffn_card_acc`] of four slots a token ([`card_acc_at`]).
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
            w_pitch >= 4,
            down.len() >= 4 * n * m,
            w.len() >= w_pitch * m,
            sel.len() >= 4 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_4(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        w_pitch: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_4 slots.
        let v =
            unsafe { card_acc_at::<SLOTS_4>(down, w, sel, n, n_card, w_pitch as usize, fault, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }

    /// [`ds41_ffn_card_acc`] of nine slots a token ([`card_acc_at`]).
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
            w_pitch >= 9,
            down.len() >= 9 * n * m,
            w.len() >= w_pitch * m,
            sel.len() >= 9 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_9(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        w_pitch: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_9 slots.
        let v =
            unsafe { card_acc_at::<SLOTS_9>(down, w, sel, n, n_card, w_pitch as usize, fault, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }

    /// The block output of `m` tokens of `n` values, one thread a value
    /// ([`card_join_at`]): the card sum of `slots` slots a token, then
    /// `(base + acc) + sh · g` — `g` the token's router weight `gate` of
    /// `w_pitch`, or 1 for a `gate` at or past it — each step rounded. A
    /// place in `[n_card, HOST)` raises [`FaultSite::ExpertId`] and its
    /// token's values are NaN; a non-finite input or result of the join
    /// raises [`FaultSite::F32Product`].
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
            w_pitch >= slots,
            down.len() >= slots * n * m,
            w.len() >= w_pitch * m,
            sel.len() >= slots * m,
            sh.len() >= n * m,
            y.len() >= n * m
        )
    )]
    pub fn card_sum_join(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        sh: &[f32],
        n: u32,
        m: u32,
        n_card: u32,
        slots: u32,
        w_pitch: u32,
        gate: u32,
        base: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract's lengths, read as the body's.
        unsafe {
            card_join_at(
                i,
                down,
                (w, sel, sh),
                (n, m, n_card),
                (slots, w_pitch, gate, base),
                fault,
                &mut y,
            )
        }
    }

    /// [`card_sum_join`] over BF16 down rows (each cell widened exactly to
    /// F32): the same sum, join and faults.
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
            w_pitch >= slots,
            down.len() >= slots * n * m,
            w.len() >= w_pitch * m,
            sel.len() >= slots * m,
            sh.len() >= n * m,
            y.len() >= n * m
        )
    )]
    pub fn card_sum_join_bf16(
        down: &[u16],
        w: &[f32],
        sel: &[u32],
        sh: &[f32],
        n: u32,
        m: u32,
        n_card: u32,
        slots: u32,
        w_pitch: u32,
        gate: u32,
        base: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract's lengths, read as the body's.
        unsafe {
            card_join_at(
                i,
                down,
                (w, sel, sh),
                (n, m, n_card),
                (slots, w_pitch, gate, base),
                fault,
                &mut y,
            )
        }
    }
}

/// BF16 bits, each widened exactly to F32: the slot-major down rows a card
/// sum reads in BF16.
impl RowValues for &[u16] {
    #[inline(always)]
    unsafe fn at(self, i: usize) -> f32 {
        // SAFETY: i is below the slice's length by this fn's contract.
        cuda_device::convert::bf16_to_f32(unsafe { *self.get_unchecked(i) })
    }
}

/// The card slots' sum of value `i` (token `t = i / n`, value `d = i % n`)
/// of `N` slots a token: `acc = fma(down[(Nt + j)·n + d], w[t·w_pitch + j],
/// acc)` from zero for each slot `j` below `N` whose place `sel[Nt + j]` is
/// below `n_card`, in ascending `j` — the order `runtime::combine::card_sum`
/// pins (this order is the gate); no other slot's rows or weight are read. A
/// place in `[n_card, HOST)` raises [`FaultSite::ExpertId`] (from the token's
/// value 0) and the value is NaN.
///
/// SAFETY: `i < n · m` for an `m` with `down` holding `N · n · m` cells,
/// `sel.len() >= N · m`, and `w.len() >= w_pitch · m` with `w_pitch >= N`.
#[allow(
    clippy::too_many_arguments,
    reason = "device helper: the entries' arguments flat (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn card_acc_at<const N: usize>(
    down: impl RowValues,
    w: &[f32],
    sel: &[u32],
    n: usize,
    n_card: u32,
    w_pitch: usize,
    fault: FaultSink,
    i: usize,
) -> f32 {
    let (t, d) = (i / n, i % n);
    let mut acc = 0.0f32;
    for j in 0..N {
        cuda_device::thread::__unroll_config::<0>();
        let s = t * N + j;
        // SAFETY: s < N·m <= sel.len() by this fn's contract.
        let place = unsafe { *sel.get_unchecked(s) };
        if place < n_card {
            // SAFETY: t < m and j < N <= w_pitch put t·w_pitch + j below
            // w_pitch·m <= w.len(), and s < N·m and d < n put s·n + d below
            // the N·n·m cells of down, by this fn's contract.
            let (ws, ds) = unsafe { (*w.get_unchecked(t * w_pitch + j), down.at(s * n + d)) };
            acc = ds.mul_add(ws, acc);
        } else if place != HOST {
            if d == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            acc = f32::NAN;
        }
    }
    acc
}

/// One value of a block output with card experts: `(h + a) + s · g`, the
/// host's routed sum `h`, then the card's `a`, then the shared expert's
/// output `s` times its gate `g`, the product and each sum rounded — this
/// order is the gate. A non-finite input or result raises
/// [`FaultSite::F32Product`].
#[inline(always)]
pub(crate) fn join_value(h: f32, a: f32, s: f32, g: f32, fault: FaultSink) -> f32 {
    let out = add_rn_f32(add_rn_f32(h, a), mul_rn_f32(s, g));
    if !(h.is_finite() & a.is_finite() & s.is_finite() & g.is_finite() & out.is_finite()) {
        fault.raise(FaultSite::F32Product);
    }
    out
}

/// The join entries' value `i` (the thread's): the card sum of its `slots`
/// slots ([`card_acc_at`]), then [`join_value`] from `base` with the shared
/// expert's output `sh[i]` times the token's router weight `gate` — or 1
/// for a `gate` at or past `w_pitch`, an ungated shared expert.
///
/// # Safety
/// [`card_acc_at`]'s contract at `slots` slots, and `sh` and `y` hold
/// `n·m` values.
#[inline(always)]
unsafe fn card_join_at(
    i: usize,
    down: impl RowValues,
    (w, sel, sh): (&[f32], &[u32], &[f32]),
    (n, m, n_card): (u32, u32, u32),
    (slots, w_pitch, gate, base): (u32, u32, u32, f32),
    fault: FaultSink,
    y: &mut DisjointSlice<f32>,
) {
    let (n, m, w_pitch) = (n as usize, m as usize, w_pitch as usize);
    if i >= n * m {
        return;
    }
    let at = (n, n_card, w_pitch, fault, i);
    // SAFETY: i < n·m, and this fn's contract is the helper's at `slots`
    // slots. Another count than the launcher takes (`CardSlots`) reads
    // nothing: its NaN raises `F32Product` in the join.
    let a = unsafe {
        match slots as usize {
            SLOTS_4 => card_acc_at::<SLOTS_4>(down, w, sel, at.0, at.1, at.2, at.3, at.4),
            SLOTS_6 => card_acc_at::<SLOTS_6>(down, w, sel, at.0, at.1, at.2, at.3, at.4),
            SLOTS_8 => card_acc_at::<SLOTS_8>(down, w, sel, at.0, at.1, at.2, at.3, at.4),
            SLOTS_9 => card_acc_at::<SLOTS_9>(down, w, sel, at.0, at.1, at.2, at.3, at.4),
            SLOTS_10 => card_acc_at::<SLOTS_10>(down, w, sel, at.0, at.1, at.2, at.3, at.4),
            _ => f32::NAN,
        }
    };
    // SAFETY: i < n·m <= sh.len() by this fn's contract.
    let s = unsafe { *sh.get_unchecked(i) };
    let g = if (gate as usize) < w_pitch {
        // SAFETY: i / n < m and gate < w_pitch put the index below
        // w_pitch·m <= w.len() by this fn's contract.
        unsafe { *w.get_unchecked(i / n * w_pitch + gate as usize) }
    } else {
        1.0
    };
    let out = join_value(base, a, s, g, fault);
    // SAFETY: i < n·m <= y.len() by this fn's contract; thread i is y[i]'s
    // only writer.
    unsafe { *y.get_unchecked_mut(i) = out };
}

/// The launch of `n · m` threads over `slots` slots a token at a weights
/// pitch of `w_pitch` (`card_sum_join`'s), refused by name as `what`'s for
/// no value, token or slot, a pitch under the slot count, a count past one
/// launch's grid, and a shared `gate` among the routed slots or past the
/// pitch. The launch and `n · m`.
fn slots_launch(
    what: &'static str,
    (n, m): (usize, usize),
    (slots, w_pitch): (usize, usize),
    gate: Option<usize>,
) -> Result<(LaunchConfig1D, usize), GpuError> {
    let gate_ok = gate.is_none_or(|g| g >= slots && g < w_pitch);
    let nm = n.checked_mul(m).filter(|&nm| nm != 0);
    let (Some(nm), true, true) = (nm, slots != 0 && w_pitch >= slots, gate_ok) else {
        return Err(GpuError::shape(
            what,
            format!(
                "{n} values of {m} tokens, {slots} slots at a weights pitch of {w_pitch}, the \
                 shared gate {gate:?}: a value, a token and a slot, a pitch of the slot count or \
                 more, and a gate past the routed slots within the pitch"
            ),
        ));
    };
    let grid = launch_u32(what, "grid", nm.div_ceil(THREADS as usize))?;
    Ok((LaunchConfig1D::new(grid, THREADS, 0), nm))
}

/// `k · x` cells, or more than any buffer holds when the product wraps.
fn cells(k: usize, x: usize) -> usize {
    k.saturating_mul(x)
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

/// What [`CardSlotKernels::enqueue_places_cols`] reads: `m` columns of the
/// router's words at `ids_pitch` a column, the routed `used` first, and the
/// slot map's card copy as in [`Places`]; the places go out at `sel_pitch` a
/// column.
pub struct ColPlaces<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub ids_pitch: usize,
    pub used: usize,
    pub sel_pitch: usize,
    pub m: usize,
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}

/// A card route's slot-major down rows as a card sum reads them: F32, or
/// BF16 bits (a down product's BF16 store, `enqueue_gemm32_bf16`).
#[derive(Clone, Copy)]
pub enum DownRows<'a> {
    F32(&'a DeviceBuffer<f32>),
    Bf16(&'a DeviceBuffer<u16>),
}

impl DownRows<'_> {
    /// The rows' cell count.
    #[must_use]
    pub fn cells(&self) -> usize {
        match self {
            Self::F32(d) => d.len(),
            Self::Bf16(d) => d.len(),
        }
    }
}

impl<'a> DownRows<'a> {
    /// The F32 rows of a launch that reads F32 rows alone (`what`'s): BF16
    /// rows are refused by name.
    pub fn f32(self, what: &'static str) -> Result<&'a DeviceBuffer<f32>, GpuError> {
        match self {
            Self::F32(d) => Ok(d),
            Self::Bf16(_) => Err(GpuError::state(
                what,
                "F32 down rows (BF16 rows feed the card sum's join alone)",
            )),
        }
    }
}

/// What [`CardSlotKernels::enqueue_card_acc`] reads, `m` tokens of `n_used`
/// slots: `down` the slots' down outputs slot-major (`n` each, token `t`'s
/// slots `n_used · t ..`), `w` their routing weights (`w_pitch` a token, the
/// routed slots first: `n_used`, or `n_used + 1` for a router whose last
/// weight is the shared expert's) and `sel` their places (the card's below
/// `n_card`, the host's [`HOST`]; a place between raises
/// [`FaultSite::ExpertId`] on `fault` and its token's sum is NaN).
pub struct CardAcc<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_used: usize,
    pub w_pitch: usize,
    pub fault: FaultSink,
}

/// The shared expert's output a card sum joins
/// ([`CardSlotKernels::enqueue_card_join`]): `sh` (`[m][n]`) times its gate —
/// the router's weight `gate` of a token's `w_pitch` (Qwen3.8's last), or 1
/// for an ungated shared expert (`None`, GLM-5.3) — added after the card sum.
/// A gated join starts from the zeros of an empty host list, `+0.0 + acc`
/// (a layer run without its host leg, whose sum a two-launch join added to
/// those zeros: a `-0.0` sum turns `+0.0`); an ungated one from the card sum
/// itself (`-0.0 + acc`, exact).
#[derive(Clone, Copy)]
pub struct SharedJoin<'a> {
    pub sh: &'a DeviceBuffer<f32>,
    pub gate: Option<usize>,
}

/// What [`CardSlotKernels::enqueue_card_join`] reads: [`CardAcc`]'s slots
/// with the down rows in either form ([`DownRows`]) and the shared expert's
/// output it joins.
pub struct CardJoin<'a> {
    pub down: DownRows<'a>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_used: usize,
    pub w_pitch: usize,
    pub shared: SharedJoin<'a>,
    pub fault: FaultSink,
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

    /// Enqueue `ds41_ffn_places` over `p.n` flat ids: `sel[i]` the place of
    /// `p.ids[i]` for `i < p.n`, [`HOST`] and [`FaultSite::ExpertId`] on
    /// `fault` for an id past the stack. One launch. Asynchronous,
    /// allocation-free.
    pub fn enqueue_places(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let cols = ColPlaces {
            ids: p.ids,
            ids_pitch: 1,
            used: 1,
            sel_pitch: 1,
            m: p.n,
            map: p.map,
            row_off: p.row_off,
            n_expert: p.n_expert,
        };
        self.enqueue_places_cols(stream, &cols, fault, sel)
    }

    /// Enqueue `ds41_ffn_places` over `p.m` columns (module doc): lane `e <
    /// p.used` of column `c` is the place of `p.ids[c·ids_pitch + e]`, a
    /// lane up to `sel_pitch` past the routed ones [`HOST`], and
    /// [`FaultSite::ExpertId`] on `fault` for an id past the stack. No
    /// column, no routed slot, `used` past either pitch, and buffers short
    /// of `m` columns are refused by name. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_places_cols(
        &self,
        stream: &CudaStream,
        p: &ColPlaces<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ds41_ffn_places";
        if p.m == 0
            || p.used == 0
            || p.used > p.ids_pitch
            || p.used > p.sel_pitch
            || p.ids.len() < p.m * p.ids_pitch
            || sel.len() < p.m * p.sel_pitch
            || p.map.len() < p.row_off + p.n_expert
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} columns of {} routed slots, ids at a pitch of {} over {} words, places at \
                     a pitch of {} into {}, a map of {} words read from {} for {} experts: a \
                     column or more, the routed slots within both pitches, and every buffer that \
                     long",
                    p.m,
                    p.used,
                    p.ids_pitch,
                    p.ids.len(),
                    p.sel_pitch,
                    sel.len(),
                    p.map.len(),
                    p.row_off,
                    p.n_expert
                ),
            ));
        }
        let grid = launch_u32(WHAT, "grid", (p.m * p.sel_pitch).div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_places(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_ffn_places(
            stream,
            &prep,
            p.ids,
            p.map,
            launch_u32(WHAT, "row_off", p.row_off)?,
            launch_u32(WHAT, "n_expert", p.n_expert)?,
            launch_u32(WHAT, "ids_pitch", p.ids_pitch)?,
            launch_u32(WHAT, "used", p.used)?,
            launch_u32(WHAT, "sel_pitch", p.sel_pitch)?,
            launch_u32(WHAT, "m", p.m)?,
            fault,
            sel,
        )?;
        Ok(())
    }

    /// Enqueue the card sums of `a.m` tokens of `a.n_used` slots
    /// (`ds41_ffn_card_acc` for 6, `_8` for 8, `_10` for 10, `_4` and `_9`;
    /// any other count is refused by name): value `i` of token `t = i / a.n` into `acc[i]`,
    /// the card half of the combine; a card of no expert (`a.n_card` 0, every
    /// place [`HOST`]) sums to zeros. No value or token, a weights pitch
    /// under the slot count, and buffers short of `a.m` tokens are refused by
    /// name, nothing launched. A place in `[a.n_card, HOST)` raises
    /// [`FaultSite::ExpertId`] on `a.fault`. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_card_acc(
        &self,
        stream: &CudaStream,
        a: &CardAcc<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let down = a.down;
        let slots = CardSlots::of(a.n_used).ok_or_else(|| CardSlots::refusal(a.n_used))?;
        let what = slots.entry();
        let k = slots.n();
        let (cfg, nm) = slots_launch(what, (a.n, a.m), (k, a.w_pitch), None)?;
        short(
            what,
            &[
                ("down", down.len(), cells(k, nm)),
                ("w", a.w.len(), cells(a.w_pitch, a.m)),
                ("sel", a.sel.len(), cells(k, a.m)),
                ("acc", acc.len(), nm),
            ],
        )?;
        let (n, m, n_card, w_pitch) = (
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            launch_u32(what, "w_pitch", a.w_pitch)?,
        );
        let mo = &self.module;
        match slots {
            CardSlots::Six => {
                let prep = mo.prepare_ds41_ffn_card_acc(cfg)?;
                mo.ds41_ffn_card_acc(
                    stream, &prep, down, a.w, a.sel, n, m, n_card, w_pitch, a.fault, acc,
                )?;
            }
            CardSlots::Eight => {
                let prep = mo.prepare_ds41_ffn_card_acc_8(cfg)?;
                mo.ds41_ffn_card_acc_8(
                    stream, &prep, down, a.w, a.sel, n, m, n_card, w_pitch, a.fault, acc,
                )?;
            }
            CardSlots::Four => {
                let prep = mo.prepare_ds41_ffn_card_acc_4(cfg)?;
                mo.ds41_ffn_card_acc_4(
                    stream, &prep, down, a.w, a.sel, n, m, n_card, w_pitch, a.fault, acc,
                )?;
            }
            CardSlots::Nine => {
                let prep = mo.prepare_ds41_ffn_card_acc_9(cfg)?;
                mo.ds41_ffn_card_acc_9(
                    stream, &prep, down, a.w, a.sel, n, m, n_card, w_pitch, a.fault, acc,
                )?;
            }
            CardSlots::Ten => {
                let prep = mo.prepare_ds41_ffn_card_acc_10(cfg)?;
                mo.ds41_ffn_card_acc_10(
                    stream, &prep, down, a.w, a.sel, n, m, n_card, w_pitch, a.fault, acc,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue the block output of `a.m` tokens of `a.n` values
    /// ([`CardJoin`], `card_sum_join` or `_bf16` by the rows' form): the
    /// card sum of `a.n_used` slots ([`CardSlotKernels::enqueue_card_acc`]'s
    /// slot counts, order and faults) and the shared expert's gated output,
    /// `(base + acc) + sh · g` ([`SharedJoin`]), into `y`, without writing the
    /// sum. No value, token or slot, a weights pitch under the slot count, a gate
    /// among the routed slots or past the pitch, more than one launch's grid
    /// and a buffer short of them are refused by name, nothing launched. One
    /// launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_card_join(
        &self,
        stream: &CudaStream,
        a: &CardJoin<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "card_sum_join";
        CardSlots::of(a.n_used).ok_or_else(|| CardSlots::refusal(a.n_used))?;
        let shape = (a.n_used, a.w_pitch);
        let (cfg, nm) = slots_launch(WHAT, (a.n, a.m), shape, a.shared.gate)?;
        short(
            WHAT,
            &[
                ("down", a.down.cells(), cells(a.n_used, nm)),
                ("w", a.w.len(), cells(a.w_pitch, a.m)),
                ("sel", a.sel.len(), cells(a.n_used, a.m)),
                ("sh", a.shared.sh.len(), nm),
                ("y", y.len(), nm),
            ],
        )?;
        let w_pitch = launch_u32(WHAT, "w_pitch", a.w_pitch)?;
        let (n, m, n_card, slots, gate) = (
            launch_u32(WHAT, "n", a.n)?,
            launch_u32(WHAT, "m", a.m)?,
            launch_u32(WHAT, "n_card", a.n_card)?,
            launch_u32(WHAT, "slots", a.n_used)?,
            a.shared
                .gate
                .map_or(Ok(w_pitch), |g| launch_u32(WHAT, "gate", g))?,
        );
        let base = if a.shared.gate.is_some() { 0.0 } else { -0.0 };
        let (mo, sh) = (&self.module, a.shared.sh);
        match a.down {
            DownRows::F32(down) => {
                let prep = mo.prepare_card_sum_join(cfg)?;
                mo.card_sum_join(
                    stream, &prep, down, a.w, a.sel, sh, n, m, n_card, slots, w_pitch, gate, base,
                    a.fault, y,
                )?;
            }
            DownRows::Bf16(down) => {
                let prep = mo.prepare_card_sum_join_bf16(cfg)?;
                mo.card_sum_join_bf16(
                    stream, &prep, down, a.w, a.sel, sh, n, m, n_card, slots, w_pitch, gate, base,
                    a.fault, y,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use runtime::combine::card_sum;

    /// The host rule of value `d` of token `t` at `N` slots.
    fn rule<const N: usize>(
        (down, w, sel): (&[f32], &[f32], &[u32]),
        (n, n_card, w_pitch): (usize, u32, usize),
        (t, d): (usize, usize),
    ) -> f32 {
        card_sum::<N>(
            std::array::from_fn(|j| down[(t * N + j) * n + d]),
            std::array::from_fn(|j| w[t * w_pitch + j]),
            std::array::from_fn(|j| sel[t * N + j] < n_card),
        )
    }

    /// Both card sum launchers refuse empty or overflowing geometry and
    /// short buffers through their common checks, naming the same input.
    #[test]
    #[ignore = "needs CUDA buffers; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_card_sum_launchers_refuse_invalid_shapes() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let kernels = CardSlotKernels::load(gpu.context()).expect("the card slots' module");
        let mut red = Vec::new();
        for (name, n, m, pitch, lens, needle) in [
            ("minimal", 1, 1, 4, [4, 4, 4, 1], None),
            (
                "no_value",
                0,
                2,
                4,
                [16, 8, 8, 4],
                Some("4 slots at a weights pitch of 4"),
            ),
            (
                "no_token",
                2,
                0,
                4,
                [16, 8, 8, 4],
                Some("4 slots at a weights pitch of 4"),
            ),
            (
                "short_pitch",
                2,
                2,
                3,
                [16, 8, 8, 4],
                Some("4 slots at a weights pitch of 3"),
            ),
            (
                "shape_overflow",
                usize::MAX,
                2,
                4,
                [16, 8, 8, 4],
                Some("a value, a token and a slot"),
            ),
            (
                "weight_overflow",
                1,
                2,
                usize::MAX / 2 + 1,
                [16, 8, 8, 4],
                Some("w.len()"),
            ),
            ("short_down", 2, 2, 4, [15, 8, 8, 4], Some("down.len()")),
            ("short_weights", 2, 2, 4, [16, 7, 8, 4], Some("w.len()")),
            ("short_places", 2, 2, 4, [16, 8, 7, 4], Some("sel.len()")),
            ("short_output", 2, 2, 4, [16, 8, 8, 3], Some(".len() 3 < 4")),
        ] {
            let down = DeviceBuffer::from_host(stream, &vec![0.0f32; lens[0]]).expect("down");
            let w = DeviceBuffer::from_host(stream, &vec![0.0f32; lens[1]]).expect("weights");
            let sel = DeviceBuffer::from_host(stream, &vec![HOST; lens[2]]).expect("places");
            let sh = DeviceBuffer::from_host(stream, &[0.0f32; 4]).expect("shared output");
            let mut out = DeviceBuffer::from_host(stream, &vec![1.0f32; lens[3]]).expect("output");
            let fault = gpu.unlabelled_sink();
            let a = CardAcc {
                down: &down,
                w: &w,
                sel: &sel,
                n,
                m,
                n_card: 0,
                n_used: SLOTS_4,
                w_pitch: pitch,
                fault,
            };
            let j = CardJoin {
                down: DownRows::F32(&down),
                w: &w,
                sel: &sel,
                n,
                m,
                n_card: 0,
                n_used: SLOTS_4,
                w_pitch: pitch,
                shared: SharedJoin {
                    sh: &sh,
                    gate: None,
                },
                fault,
            };
            for (what, result) in [
                (
                    CardSlots::Four.entry(),
                    kernels.enqueue_card_acc(stream, &a, &mut out),
                ),
                (
                    "card_sum_join",
                    kernels.enqueue_card_join(stream, &j, &mut out),
                ),
            ] {
                let pass = match (needle, &result) {
                    (None, Ok(())) => true,
                    (Some(needle), Err(GpuError::Shape { what: got, detail })) => {
                        *got == what && detail.contains(needle)
                    }
                    _ => false,
                };
                println!(
                    "card_slots_refuse[{what}:{name}] {result:?} {}",
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    red.push(format!("{what}:{name}"));
                }
            }
            let want = if needle.is_some() { 1.0 } else { 0.0 };
            let unchanged = out
                .to_host_vec(stream)
                .expect("output")
                .iter()
                .all(|&v| v == want);
            if !unchanged {
                red.push(format!("{name}:output"));
            }
        }
        stream.synchronize().expect("sync");
        assert_eq!(gpu.take_fault().expect("the fault word"), None);
        assert!(red.is_empty(), "red refusals: {red:?}");
    }

    /// The sum at 4, 6, 8, 9 and 10 slots (the weights at the router's pitch of
    /// eleven at ten): a token whose slots are on the card or [`HOST`] is
    /// `runtime::combine::card_sum`'s and raises nothing; a place in
    /// `[n_card, HOST)` — the first past the card and the last before the
    /// host — raises [`FaultSite::ExpertId`] alone, at the launch's layer,
    /// and its token's values are NaN, the other tokens' the rule's. A card
    /// of no expert sums to zeros with nothing raised. Every case is judged
    /// and printed before the test fails.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_place_between_the_card_and_the_host_faults_the_sum() {
        const N: usize = 300;
        const M: usize = 3;
        const N_CARD: u32 = 5;
        const LAYER: usize = 7;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let kernels = CardSlotKernels::load(gpu.context()).expect("the card slots' module");
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let mut red = Vec::new();
        for n_used in [SLOTS_4, SLOTS_6, SLOTS_8, SLOTS_9, SLOTS_10] {
            let w_pitch = if n_used == SLOTS_10 {
                n_used + 1
            } else {
                n_used
            };
            // Slots on the card or the host's; token 1's slot 3 takes the
            // stray's place.
            let base: Vec<u32> = (0..M * n_used)
                .map(|s| {
                    if (s * 7 + s / n_used).is_multiple_of(3) {
                        HOST
                    } else {
                        (s % 5) as u32
                    }
                })
                .collect();
            for stray in [None, Some(N_CARD), Some(HOST - 1)] {
                let mut sel = base.clone();
                if let Some(p) = stray {
                    sel[n_used + 3] = p;
                }
                let on = |s: usize| sel[s] < N_CARD;
                // A slot the sum must not read holds NaN, its weight too.
                let down: Vec<f32> = (0..M * n_used * N)
                    .map(|i| if on(i / N) { next() } else { f32::NAN })
                    .collect();
                let w: Vec<f32> = (0..M * w_pitch)
                    .map(|i| {
                        let (t, j) = (i / w_pitch, i % w_pitch);
                        if j < n_used && on(t * n_used + j) {
                            next()
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                let (sel_d, down_d, w_d) = (
                    DeviceBuffer::from_host(stream, &sel).expect("sel"),
                    DeviceBuffer::from_host(stream, &down).expect("down"),
                    DeviceBuffer::from_host(stream, &w).expect("w"),
                );
                let mut acc = DeviceBuffer::from_host(stream, &[0.0f32; N * M]).expect("acc");
                let a = CardAcc {
                    down: &down_d,
                    w: &w_d,
                    sel: &sel_d,
                    n: N,
                    m: M,
                    n_card: N_CARD as usize,
                    n_used,
                    w_pitch,
                    fault: gpu.layer_sink(LAYER).expect("a sink"),
                };
                kernels
                    .enqueue_card_acc(stream, &a, &mut acc)
                    .expect("the card sum");
                let got = acc.to_host_vec(stream).expect("sums");
                stream.synchronize().expect("sync");
                let fault = gpu.take_fault().expect("the fault word");
                let off = got
                    .iter()
                    .enumerate()
                    .filter(|&(i, g)| {
                        let (t, d) = (i / N, i % N);
                        if stray.is_some() && t == 1 {
                            return !g.is_nan();
                        }
                        let args = ((&down[..], &w[..], &sel[..]), (N, N_CARD, w_pitch), (t, d));
                        let want = match n_used {
                            SLOTS_4 => rule::<SLOTS_4>(args.0, args.1, args.2),
                            SLOTS_6 => rule::<SLOTS_6>(args.0, args.1, args.2),
                            SLOTS_8 => rule::<SLOTS_8>(args.0, args.1, args.2),
                            SLOTS_9 => rule::<SLOTS_9>(args.0, args.1, args.2),
                            _ => rule::<SLOTS_10>(args.0, args.1, args.2),
                        };
                        g.to_bits() != want.to_bits()
                    })
                    .count();
                let fault_ok = match (stray, fault) {
                    (None, None) => true,
                    (Some(_), Some(f)) => {
                        f.layer as usize == LAYER
                            && f.site() == Some(FaultSite::ExpertId)
                            && f.sites == 1 << FaultSite::ExpertId as u32
                    }
                    _ => false,
                };
                let pass = off == 0 && fault_ok;
                println!(
                    "card_acc of {n_used} slots, weights at {w_pitch}, stray {stray:?}: {off} \
                     values off the rule, fault {fault:?}: {}",
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    red.push(format!("{n_used} slots, stray {stray:?}"));
                }
            }
        }
        // A card of no expert: every place is the host's, and the sum is
        // zeros with nothing raised.
        for n_used in [SLOTS_4, SLOTS_6, SLOTS_8, SLOTS_9, SLOTS_10] {
            let sel = DeviceBuffer::from_host(stream, &vec![HOST; M * n_used]).expect("sel");
            let z = DeviceBuffer::from_host(stream, &vec![f32::NAN; M * n_used * N]).expect("z");
            let mut acc = DeviceBuffer::from_host(stream, &[1.0f32; N * M]).expect("acc");
            let a = CardAcc {
                down: &z,
                w: &z,
                sel: &sel,
                n: N,
                m: M,
                n_card: 0,
                n_used,
                w_pitch: n_used,
                fault: gpu.layer_sink(LAYER).expect("a sink"),
            };
            kernels
                .enqueue_card_acc(stream, &a, &mut acc)
                .expect("the card sum of a card of no expert");
            let got = acc.to_host_vec(stream).expect("sums");
            stream.synchronize().expect("sync");
            let fault = gpu.take_fault().expect("the fault word");
            let pass = fault.is_none() && got.iter().all(|v| v.to_bits() == 0);
            println!(
                "card_acc of {n_used} slots over a card of no expert: zeros, fault {fault:?}: {}",
                if pass { "PASS" } else { "FAIL" }
            );
            if !pass {
                red.push(format!("{n_used} slots, no card"));
            }
        }
        assert!(red.is_empty(), "red cases: {red:?}");
    }

    /// The places' lanes past `used` at a `sel_pitch` of the router's width:
    /// [`HOST`], and the id words there — the shared expert's, past the
    /// experts — never read, so nothing is raised.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_places_lanes_past_the_routed_slots_are_host_and_read_no_id() {
        const N_EXPERT: usize = 40;
        const USED: usize = 10;
        const PITCH: usize = USED + 1;
        const M: usize = 4;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let kernels = CardSlotKernels::load(gpu.context()).expect("the card slots' module");
        let place_of = |e: usize| {
            if e.is_multiple_of(3) {
                HOST
            } else {
                (e * 7 % 20) as u32
            }
        };
        let map: Vec<u32> = (0..N_EXPERT).map(place_of).collect();
        let ids: Vec<u32> = (0..M * PITCH)
            .map(|k| {
                if k % PITCH == USED {
                    N_EXPERT as u32
                } else {
                    ((k * 37 + 11) % N_EXPERT) as u32
                }
            })
            .collect();
        let (ids_d, map_d) = (
            DeviceBuffer::from_host(stream, &ids).expect("ids"),
            DeviceBuffer::from_host(stream, &map).expect("map"),
        );
        let mut sel = DeviceBuffer::from_host(stream, &[0xdead_beefu32; M * PITCH]).expect("sel");
        let p = ColPlaces {
            ids: &ids_d,
            ids_pitch: PITCH,
            used: USED,
            sel_pitch: PITCH,
            m: M,
            map: &map_d,
            row_off: 0,
            n_expert: N_EXPERT,
        };
        kernels
            .enqueue_places_cols(stream, &p, gpu.layer_sink(9).expect("a sink"), &mut sel)
            .expect("the places");
        let got = sel.to_host_vec(stream).expect("places");
        stream.synchronize().expect("sync");
        let want: Vec<u32> = (0..M * PITCH)
            .map(|k| match k % PITCH {
                USED => HOST,
                _ => place_of(ids[k] as usize),
            })
            .collect();
        assert_eq!(got, want);
        assert_eq!(gpu.take_fault().expect("the fault word"), None);
    }
}
