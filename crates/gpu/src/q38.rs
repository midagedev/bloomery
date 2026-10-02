//! Qwen3.8's own element kernels, the ones no other body launches: the Q8_0
//! embedding row, the attention output under its sigmoid gate in f32, the
//! selector's raw key append, and the shared expert's gated sum, alone or
//! with the card's routed slots' sum.
//!
//! - [`q38_kernels::embed_rows_q8_0`] — one thread per value: row `ids[t]`
//!   of the Q8_0 table in the q8f32 planes (`crate::weights::q8_0_planes`),
//!   each value `f32(code) · f32(d)` as ggml's `dequantize_row_q8_0` makes it
//!   (one product of an 8-bit and an 11-bit significand: exact), and row
//!   `t`'s position `pos0[0] + first + t` and live key count one more.
//! - [`q38_kernels::gqa_out_gate_f32`] — one thread per value: `attn ·
//!   σ(gate)`, the gate at `[q | gate]`'s head stride, `σ` the Qwen3.6 gated
//!   quantizer's (`route_core::sigmoid`); the output projection reads the f32
//!   product (its weights are Q8_0 with f32 activations).
//! - [`q38_kernels::qsa_key_append`] — one thread per value: the indexer's
//!   raw key, `DIM` values a row of the `f32_gemv` output (row-major `[DIM]
//!   [m]`), rounded once to f16 into the raw plane at the row's position.
//! - [`q38_kernels::q38_shared_add`] — one thread per value: `hsum + sh · w`,
//!   the shared expert's output times its gate weight (the router's last slot)
//!   rounded, then added to the host tier's routed sum and rounded — ggml's
//!   MUL then ADD, each an explicit round-to-nearest intrinsic so the device
//!   build does not contract them into one fused multiply-add.
//! - [`q38_kernels::q38_card_acc`] — one thread per value: the card's routed
//!   slots' sum, each of a token's [`SLOTS`] slots whose place is on the card
//!   (below the layer's card expert count) by a fused multiply-add from zero,
//!   in slot order (`runtime::combine::card_sum`), its weight the router's at
//!   pitch [`W_PITCH`] (the routed slots, then the shared expert's gate); a
//!   slot the host serves is never read.
//! - [`q38_kernels::q38_card_shared_add`] — `q38_shared_add` with the card
//!   sum: `(hsum + acc) + sh · w`, the host's sum, then the card's, then the
//!   shared expert's rounded product, each step rounded.
//! - [`q38_kernels::q38_card_tier_shared_add`] — the join of a layer an
//!   expert tier holds experts of, after the wait: `q38_card_acc`'s sum over
//!   the slots on the card or on the tier, in slot order, each slot's down
//!   output read from the card's rows or, for a tier slot, from the tier's
//!   rows through the mapping, then `q38_card_shared_add`'s combine of it —
//!   with the union of both cards' experts on one card the two launches
//!   read the same values in the same order, so the two agree bit for bit.
//! - [`q38_kernels::q38_card_tier_acc`] — that join's sum alone, into the
//!   card sum `q38_card_shared_add` combines: a ubatch's tier layer, whose
//!   host sums are uploaded only after the tier's rows are joined; a tier
//!   slot's row is read at its rank (`q38_tier_rank`).
//! - [`q38_kernels::q38_tier_ids`] — one thread per slot, on the tier card:
//!   a tier place as the remapped route's expert id, [`HOST`] as the one id
//!   past the tier's experts, which the route's map sends to the host.
//! - [`q38_kernels::q38_tier_rank`] — one block, on either card: each tier
//!   slot's rank among a block's tier slots in slot order, the row its down
//!   output takes packed; the same places give the same ranks on both cards.
//! - [`q38_kernels::q38_tier_rows_pack`] — one thread per value, on the tier
//!   card: a run's tier slots' down rows into the block's down outputs at
//!   their ranks, packed, which the batch service copies to the set's rows
//!   alone (`crate::host::tier::BlockRows::Packed`), so only the tier's rows
//!   cross the bus, by the copy engine.
//!
//! The card sum runs in the layer's host-leg shadow and the combine after the
//! wait in `q38_shared_add`'s place, so the card leg adds no launch after the
//! wait; on a tier layer the tier's rows land only with the wait, so the sum
//! moves into the combine and the leg still adds none. A layer without card experts keeps `q38_shared_add`: `hsum + 0.0`
//! would turn a `-0.0` sum into `+0.0`.
//!
//! No silent failure: an embedding id past the table raises
//! [`FaultSite::TokenId`] and writes a NaN row; a position at or past the
//! raw plane raises [`FaultSite::CachePos`] and writes nothing; a card slot's
//! place that is neither a card expert nor [`HOST`] raises
//! [`FaultSite::ExpertId`] and writes its token's card sum NaN; a raw key that
//! is not finite after its f16 rounding raises [`FaultSite::PoolSelect`]; an
//! input or a result of the two f32 products, or of the combine with the card
//! sum, that is not finite raises [`FaultSite::F32Product`] (a card slot's
//! non-finite down or weight reaches it through the sum); a tier slot's place
//! that is neither a tier expert nor [`HOST`], on a slot the card does not
//! hold, raises [`FaultSite::ExpertId`] as a card slot's does. Every value is
//! written as computed.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f32_to_f16_bits, half_bits_to_f32};
use crate::hybrid::HOST;
use crate::route_core::sigmoid;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values of an attention head: the gate kernel's head, whose `[q | gate]`
/// rows are twice as wide.
pub const HEAD: usize = 256;
/// Values of an indexer key.
pub const DIM: usize = crate::qsa::DIM;
/// Threads per block of every kernel here but [`q38_kernels::q38_tier_rank`].
const THREADS: u32 = 256;
/// Warps of [`q38_kernels::q38_tier_rank`]'s one block, whose launch contract
/// spells its 1,024 threads as a literal.
const RANK_WARPS: usize = 32;
const _: () = assert!(RANK_WARPS * 32 == 1024);
const _: () = assert!(HEAD == 256 && DIM == 128);
/// Routed slots a token: the card sum's slot count, which its launch
/// contract spells as a literal.
pub const SLOTS: usize = 10;
/// Router weights a token: the routed slots, then the shared expert's gate.
pub const W_PITCH: usize = SLOTS + 1;
const _: () = assert!(SLOTS == 10 && W_PITCH == 11);

/// `v` as a position or live key count word: `v` while it fits a u32, else
/// `u32::MAX` — a value past every cache, which the rope and the flash
/// refuse, where a wrapped one would name a plausible row.
#[inline(always)]
fn position_word(v: usize) -> u32 {
    if v > u32::MAX as usize {
        u32::MAX
    } else {
        v as u32
    }
}

#[cuda_module]
mod q38_kernels {
    use super::*;

    /// Dequantize `ids.len()` rows of the Q8_0 table (`qs` `8·k32` words and
    /// `d` `k32` f16 scales a row, `32·k32` values) into `y`, token-major,
    /// one thread per value, and give each row its position and live key
    /// count (module doc). An id past the table's `n_rows` rows raises
    /// [`FaultSite::TokenId`], reads no table word and writes a NaN row; the
    /// positions do not depend on the ids.
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
            qs.len() >= 8 * k32 * n_rows,
            d.len() >= k32 * n_rows,
            y.len() >= 32 * k32 * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q8_0(
        qs: &[u32],
        d: &[u16],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        k32: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut pos: DisjointSlice<u32>,
        mut n_keys: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i < ids.len() {
            // SAFETY: pos0 holds a word by the launch contract.
            let p = unsafe { *pos0.get_unchecked(0) } as usize + first as usize + i;
            // SAFETY: i < ids.len() <= pos.len(), n_keys.len() by the launch
            // contract; thread i owns entry i of both.
            unsafe {
                *pos.get_unchecked_mut(i) = position_word(p);
                *n_keys.get_unchecked_mut(i) = position_word(p + 1);
            }
        }
        let width = 32 * k32 as usize;
        if i >= ids.len() * width {
            return;
        }
        let t = i / width;
        let v = i % width;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let val = if id < n_rows {
            let row = id as usize;
            // SAFETY: id < n_rows, so word row·8·k32 + v/4 and scale
            // row·k32 + v/32 lie inside qs and d by the launch contract.
            let (word, bits) = unsafe {
                (
                    *qs.get_unchecked(row * 8 * k32 as usize + v / 4),
                    *d.get_unchecked(row * k32 as usize + v / 32),
                )
            };
            let code = (word >> (8 * (v % 4))) as u8 as i8;
            code as f32 * half_bits_to_f32(bits)
        } else {
            if v == 0 {
                fault.raise(FaultSite::TokenId);
            }
            f32::NAN
        };
        // SAFETY: i < ids.len()·width <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = val;
        }
    }

    /// `y = attn · σ(gate)` over `m` tokens of `n_head` heads of [`HEAD`]
    /// values: value `v` of head `h` of token `t` at `(t·n_head + h)·HEAD +
    /// v`, its gate at `(t·n_head + h)·2·HEAD + HEAD + v` of the query
    /// projection's `[q | gate]` rows. A non-finite input or product raises
    /// [`FaultSite::F32Product`].
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            attn.len() >= 256 * n_head * m,
            qg.len() >= 512 * n_head * m,
            y.len() >= 256 * n_head * m
        )
    )]
    pub fn gqa_out_gate_f32(
        attn: &[f32],
        qg: &[f32],
        n_head: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= HEAD * n_head as usize * m as usize {
            return;
        }
        let (hv, v) = (i / HEAD, i % HEAD);
        // SAFETY: i < HEAD·n_head·m <= attn.len(), and hv < n_head·m puts
        // hv·2·HEAD + HEAD + v below 2·HEAD·n_head·m <= qg.len(), by the
        // launch contract.
        let (a, g) = unsafe {
            (
                *attn.get_unchecked(i),
                *qg.get_unchecked(hv * 2 * HEAD + HEAD + v),
            )
        };
        let out = a * sigmoid(g);
        if !(a.is_finite() & g.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < HEAD·n_head·m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// Row `c` of `m` indexer keys — value `r` at `kr[r·m + c]`, the f32
    /// gemv's row-major output — rounded once to f16 into the raw plane's
    /// row `pos[c]`, [`DIM`] values a row. A position at or past `ctx`
    /// raises [`FaultSite::CachePos`] and writes nothing; a value that is not
    /// finite after its rounding raises [`FaultSite::PoolSelect`] and is
    /// written.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (kr.len() >= 128 * m, pos.len() >= m, raw.len() >= 128 * ctx)
    )]
    pub fn qsa_key_append(
        kr: &[f32],
        pos: &[u32],
        m: u32,
        ctx: u32,
        fault: FaultSink,
        mut raw: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i >= DIM * m as usize {
            return;
        }
        let (c, r) = (i / DIM, i % DIM);
        // SAFETY: c < m, so c < pos.len() and r·m + c < DIM·m <= kr.len(), by
        // the launch contract.
        let (p, x) = unsafe { (*pos.get_unchecked(c), *kr.get_unchecked(r * m as usize + c)) };
        if p >= ctx {
            if r == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return;
        }
        let bits = f32_to_f16_bits(x);
        if !half_bits_to_f32(bits).is_finite() {
            fault.raise(FaultSite::PoolSelect);
        }
        // SAFETY: p < ctx, so p·DIM + r < DIM·ctx <= raw.len() by the launch
        // contract; positions of one launch are distinct (`pos[0] + c`), so
        // thread (c, r) is the word's only writer.
        unsafe {
            *raw.get_unchecked_mut(p as usize * DIM + r) = bits;
        }
    }

    /// `y = hsum + sh · w[t·slots + slot]` over `m` tokens of `n` values,
    /// token `t`'s weight the router's slot `slot`: the product rounded, then
    /// the sum rounded (module doc). A non-finite input or result raises
    /// [`FaultSite::F32Product`].
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
            slots >= slot + 1,
            hsum.len() >= n * m,
            sh.len() >= n * m,
            w.len() >= slots * m,
            y.len() >= n * m
        )
    )]
    pub fn q38_shared_add(
        hsum: &[f32],
        sh: &[f32],
        w: &[f32],
        slot: u32,
        slots: u32,
        n: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize * m as usize {
            return;
        }
        let t = i / n as usize;
        // SAFETY: i < n·m bounds hsum and sh; t < m and slot < slots put
        // t·slots + slot below slots·m <= w.len(); by the launch contract.
        let (h, s, wt) = unsafe {
            (
                *hsum.get_unchecked(i),
                *sh.get_unchecked(i),
                *w.get_unchecked(t * slots as usize + slot as usize),
            )
        };
        let out = add_rn_f32(h, mul_rn_f32(s, wt));
        if !(h.is_finite() & s.is_finite() & wt.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < n·m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// The card slots' sum of `m` tokens: thread `i < n·m`, token `t = i /
    /// n`, value `d = i % n`, writes to `acc[i]` the fused multiply-adds from
    /// zero of `down[(10t + j)·n + d]` by `w[11t + j]`, in ascending `j <
    /// 10`, over the slots whose place `sel[10t + j]` is below `n_card` — this
    /// order is the gate (`runtime::combine::card_sum`). No other slot's
    /// down row or weight is read. A [`HOST`] place is the host's slot and
    /// raises nothing; a place in `[n_card, HOST)` is no expert either side
    /// serves: it raises [`FaultSite::ExpertId`] on `fault` and its token's
    /// values are NaN.
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
            down.len() >= 10 * n * m,
            w.len() >= 11 * m,
            sel.len() >= 10 * m,
            acc.len() >= n * m
        )
    )]
    pub fn q38_card_acc(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let mut a = 0.0f32;
        for j in 0..SLOTS {
            cuda_device::thread::__unroll_config::<0>();
            let s = t * SLOTS + j;
            // SAFETY: t < m and j < 10 put s below 10m <= sel.len() by the
            // launch contract.
            let place = unsafe { *sel.get_unchecked(s) };
            if place < n_card {
                // SAFETY: t < m and j < 10 put 11t + j below 11m <= w.len(),
                // and s·n + d below 10nm <= down.len(), by the launch
                // contract.
                let (ws, ds) = unsafe {
                    (
                        *w.get_unchecked(t * W_PITCH + j),
                        *down.get_unchecked(s * n + d),
                    )
                };
                a = ds.mul_add(ws, a);
            } else if place != HOST {
                if d == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
                a = f32::NAN;
            }
        }
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe {
            *acc.get_unchecked_mut(i) = a;
        }
    }

    /// `y = (hsum + acc) + sh · w[t·slots + slot]` over `m` tokens of `n`
    /// values: the host tier's routed sum, then the card's
    /// ([`q38_card_acc`]), then the shared expert's output times its gate
    /// weight, the product and each sum rounded — this order is the gate. A
    /// non-finite input or result raises [`FaultSite::F32Product`].
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
            slots >= slot + 1,
            hsum.len() >= n * m,
            acc.len() >= n * m,
            sh.len() >= n * m,
            w.len() >= slots * m,
            y.len() >= n * m
        )
    )]
    pub fn q38_card_shared_add(
        hsum: &[f32],
        acc: &[f32],
        sh: &[f32],
        w: &[f32],
        slot: u32,
        slots: u32,
        n: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize * m as usize {
            return;
        }
        let t = i / n as usize;
        // SAFETY: i < n·m bounds hsum, acc and sh; t < m and slot < slots put
        // t·slots + slot below slots·m <= w.len(); by the launch contract.
        let (h, a, s, wt) = unsafe {
            (
                *hsum.get_unchecked(i),
                *acc.get_unchecked(i),
                *sh.get_unchecked(i),
                *w.get_unchecked(t * slots as usize + slot as usize),
            )
        };
        let out = add_rn_f32(add_rn_f32(h, a), mul_rn_f32(s, wt));
        if !(h.is_finite() & a.is_finite() & s.is_finite() & wt.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < n·m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// The join of a tier layer over `m` tokens of `n` values, one thread a
    /// value: [`q38_card_acc`]'s sum from zero over the slots on the card —
    /// place `sel[10t + j]` below `n_card`, its row `down`'s — or, on a slot
    /// the card does not hold ([`HOST`] there), on the tier — tier place
    /// `tsel[10t + j]` below `n_tier`, its row `trows`' (slot-major as
    /// `down`) — in ascending `j < 10`, this order the gate; then
    /// [`q38_card_shared_add`]'s `(hsum + acc) + sh · w[11t + 10]`. A slot
    /// [`HOST`] in both places is the host's and raises nothing; a place in
    /// `[n_card, HOST)`, or a tier place in `[n_tier, HOST)` on a host slot of
    /// the card, raises [`FaultSite::ExpertId`] and its token's values are
    /// NaN; a non-finite input or result of the combine raises
    /// [`FaultSite::F32Product`].
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
            down.len() >= 10 * n * m,
            trows.len() >= 10 * n * m,
            w.len() >= 11 * m,
            sel.len() >= 10 * m,
            tsel.len() >= 10 * m,
            hsum.len() >= n * m,
            sh.len() >= n * m,
            y.len() >= n * m
        )
    )]
    pub fn q38_card_tier_shared_add(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        hsum: &[f32],
        sh: &[f32],
        n: u32,
        m: u32,
        n_card: u32,
        n_tier: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let mut a = 0.0f32;
        for j in 0..SLOTS {
            cuda_device::thread::__unroll_config::<0>();
            let s = t * SLOTS + j;
            // SAFETY: t < m and j < 10 put s below 10m <= sel.len() and
            // tsel.len() by the launch contract.
            let (place, tplace) = unsafe { (*sel.get_unchecked(s), *tsel.get_unchecked(s)) };
            // SAFETY: t < m and j < 10 put 11t + j below 11m <= w.len() by
            // the launch contract.
            let ws = unsafe { *w.get_unchecked(t * W_PITCH + j) };
            if place < n_card {
                // SAFETY: s·n + d < 10nm <= down.len() by the launch contract.
                let ds = unsafe { *down.get_unchecked(s * n + d) };
                a = ds.mul_add(ws, a);
            } else if place != HOST {
                if d == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
                a = f32::NAN;
            } else if tplace < n_tier {
                // SAFETY: s·n + d < 10nm <= trows.len() by the launch
                // contract.
                let ds = unsafe { *trows.get_unchecked(s * n + d) };
                a = ds.mul_add(ws, a);
            } else if tplace != HOST {
                if d == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
                a = f32::NAN;
            }
        }
        // SAFETY: i < n·m bounds hsum and sh; t < m puts 11t + 10 below 11m
        // <= w.len(); by the launch contract.
        let (h, s, wt) = unsafe {
            (
                *hsum.get_unchecked(i),
                *sh.get_unchecked(i),
                *w.get_unchecked(t * W_PITCH + SLOTS),
            )
        };
        let out = add_rn_f32(add_rn_f32(h, a), mul_rn_f32(s, wt));
        if !(h.is_finite() & a.is_finite() & s.is_finite() & wt.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < n·m <= y.len() by the launch contract; thread i is
        // y[i]'s only writer.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// [`q38_card_tier_shared_add`]'s sum alone over `m` tokens of `n`
    /// values, one thread a value, into `acc`: from zero over the slots on
    /// the card — place `sel[10t + j]` below `n_card`, its row `down`'s — or,
    /// on a slot the card does not hold ([`HOST`] there), on the tier — tier
    /// place `tsel[10t + j]` below `n_tier`, its row `trows`' at the slot's
    /// rank `trank[10t + j]` ([`q38_tier_rank`]: the tier's rows come packed)
    /// — in ascending `j < 10`, each by a fused multiply-add of the router's
    /// weight, this order the gate. A ubatch's join, after which
    /// [`q38_card_shared_add`] combines it as it combines [`q38_card_acc`]'s
    /// sum. A slot [`HOST`] in both places is the host's; a place in
    /// `[n_card, HOST)`, a tier place in `[n_tier, HOST)` on a host slot of
    /// the card, or a tier slot's rank at or past `10m`, raises
    /// [`FaultSite::ExpertId`] and its token's sum is NaN.
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
            down.len() >= 10 * n * m,
            trows.len() >= 10 * n * m,
            w.len() >= 11 * m,
            sel.len() >= 10 * m,
            tsel.len() >= 10 * m,
            trank.len() >= 10 * m,
            acc.len() >= n * m
        )
    )]
    pub fn q38_card_tier_acc(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        trank: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        n_tier: u32,
        fault: FaultSink,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let mut a = 0.0f32;
        for j in 0..SLOTS {
            cuda_device::thread::__unroll_config::<0>();
            let s = t * SLOTS + j;
            // SAFETY: t < m and j < 10 put s below 10m <= sel.len() and
            // tsel.len() by the launch contract.
            let (place, tplace) = unsafe { (*sel.get_unchecked(s), *tsel.get_unchecked(s)) };
            // SAFETY: t < m and j < 10 put 11t + j below 11m <= w.len() by
            // the launch contract.
            let ws = unsafe { *w.get_unchecked(t * W_PITCH + j) };
            if place < n_card {
                // SAFETY: s·n + d < 10nm <= down.len() by the launch contract.
                let ds = unsafe { *down.get_unchecked(s * n + d) };
                a = ds.mul_add(ws, a);
            } else if place != HOST {
                if d == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
                a = f32::NAN;
            } else if tplace < n_tier {
                // SAFETY: s < 10m <= trank.len() by the launch contract.
                let r = unsafe { *trank.get_unchecked(s) } as usize;
                if r < SLOTS * m {
                    // SAFETY: r < 10m puts r·n + d below 10nm <= trows.len()
                    // by the launch contract.
                    let ds = unsafe { *trows.get_unchecked(r * n + d) };
                    a = ds.mul_add(ws, a);
                } else {
                    if d == 0 {
                        fault.raise(FaultSite::ExpertId);
                    }
                    a = f32::NAN;
                }
            } else if tplace != HOST {
                if d == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
                a = f32::NAN;
            }
        }
        // SAFETY: i < n·m <= acc.len() by the launch contract; thread i is
        // acc[i]'s only writer.
        unsafe {
            *acc.get_unchecked_mut(i) = a;
        }
    }

    /// The expert ids of a tier's remapped route from its places, one thread
    /// a slot of `slots`: a tier place `sel[i]` as it is, [`HOST`] as
    /// `n_tier` — the one id past the tier's experts, which the route's map
    /// (`0 .. n_tier`, then [`HOST`]) sends to the host. A place in
    /// `[n_tier, HOST)` stays itself, past the map, so the route raises
    /// [`FaultSite::ExpertId`] for it and its slot's outputs are NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (sel.len() >= slots, ids.len() >= slots)
    )]
    pub fn q38_tier_ids(sel: &[u32], slots: u32, n_tier: u32, mut ids: DisjointSlice<u32>) {
        let i = thread::index_1d().get();
        if i >= slots as usize {
            return;
        }
        // SAFETY: i < slots <= sel.len() by the launch contract.
        let p = unsafe { *sel.get_unchecked(i) };
        // SAFETY: i < slots <= ids.len() by the launch contract; thread i is
        // ids[i]'s only writer.
        unsafe {
            *ids.get_unchecked_mut(i) = if p == HOST { n_tier } else { p };
        }
    }

    /// Each tier slot's rank among the first `slots` places — the count of
    /// the slots before it whose place `sel[i]` is below `n_tier` — into
    /// `rank`, and [`HOST`] for every other slot: one block of
    /// [`RANK_WARPS`] warps, each warp a run of the slots in order, counted,
    /// then the warps' counts summed in warp order, then ranked; integer
    /// sums, so the same places give the same ranks on any card. A block
    /// past the first does nothing.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (sel.len() >= slots, rank.len() >= slots)
    )]
    pub fn q38_tier_rank(sel: &[u32], slots: u32, n_tier: u32, mut rank: DisjointSlice<u32>) {
        static mut WTOT: SharedArray<u32, RANK_WARPS> = SharedArray::UNINIT;
        if thread::blockIdx_x() != 0 {
            return; // block-uniform
        }
        let slots = slots as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        // SAFETY: a block-shared static; the raw form reaches it without a
        // reference, and every access below is bounded and barrier-ordered.
        let wtot = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WTOT) };
        let chunk = slots.div_ceil(32 * RANK_WARPS) * 32;
        let c0 = (wid * chunk).min(slots);
        let c1 = (c0 + chunk).min(slots);
        let l = lane as usize;
        let mut tot = 0u32;
        let mut base = c0;
        while base < c1 {
            let i = base + l;
            // SAFETY: i < c1 <= slots <= sel.len() by the launch contract.
            let mine = i < c1 && unsafe { *sel.get_unchecked(i) } < n_tier;
            tot += warp::ballot(mine).count_ones();
            base += 32;
        }
        if lane == 0 {
            // SAFETY: wid < RANK_WARPS bounds the slot; lane 0 of warp wid is
            // its only writer, before the barrier.
            unsafe { *wtot.add(wid) = tot };
        }
        thread::sync_threads();
        let mut off = 0u32;
        for w in 0..RANK_WARPS {
            thread::__unroll_config::<0>();
            if w < wid {
                // SAFETY: w < RANK_WARPS; published by the barrier above.
                off += unsafe { *wtot.add(w) };
            }
        }
        let lt = warp::lanemask_lt();
        let mut base = c0;
        while base < c1 {
            let i = base + l;
            // SAFETY: i < c1 <= slots <= sel.len() by the launch contract.
            let mine = i < c1 && unsafe { *sel.get_unchecked(i) } < n_tier;
            let mask = warp::ballot(mine);
            if i < c1 {
                let r = if mine {
                    off + (mask & lt).count_ones()
                } else {
                    HOST
                };
                // SAFETY: i < c1 <= slots <= rank.len() by the launch
                // contract; the lane at slot i is rank[i]'s only writer.
                unsafe { *rank.get_unchecked_mut(i) = r };
            }
            off += mask.count_ones();
            base += 32;
        }
    }

    /// A run of a tier block's rows packed into the block's down outputs,
    /// one thread a value of the run's `slots` slots of `n`: slot `s`'s row
    /// of `down` (the run's, slot-major) into row `rank[s]` of `rows` when its
    /// tier place `sel[s]` is below `n_tier`; no other value is written. A
    /// tier slot's rank at or past `rows_cap` raises [`FaultSite::ExpertId`]
    /// and writes nothing.
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
            down.len() >= n * slots,
            sel.len() >= slots,
            rank.len() >= slots,
            rows.len() >= n * rows_cap
        )
    )]
    pub fn q38_tier_rows_pack(
        down: &[f32],
        sel: &[u32],
        rank: &[u32],
        n: u32,
        slots: u32,
        n_tier: u32,
        rows_cap: u32,
        fault: FaultSink,
        mut rows: DisjointSlice<f32>,
    ) {
        let (n, slots) = (n as usize, slots as usize);
        let i = thread::index_1d().get();
        if i >= n * slots {
            return;
        }
        let (s, d) = (i / n, i % n);
        // SAFETY: s < slots <= sel.len() and rank.len() by the launch
        // contract.
        let (p, r) = unsafe { (*sel.get_unchecked(s), *rank.get_unchecked(s) as usize) };
        if p >= n_tier {
            return;
        }
        if r >= rows_cap as usize {
            if d == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        // SAFETY: i < n·slots <= down.len(), and r < rows_cap puts r·n + d
        // below n·rows_cap <= rows.len(), by the launch contract; a rank names
        // one tier slot, so thread i is rows[r·n + d]'s only writer.
        unsafe {
            *rows.get_unchecked_mut(r * n + d) = *down.get_unchecked(i);
        }
    }
}

/// [`Q38Kernels::enqueue_embed_rows`]'s arguments: the Q8_0 table's planes,
/// one unit's ids, the word holding the first position of the input the ids
/// are a window of and the window's offset in it, and the three outputs —
/// the rows, each row's position and its live key count.
pub struct EmbedQ8Args<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub ids: &'a DeviceBuffer<u32>,
    pub pos0: &'a DeviceBuffer<u32>,
    pub first: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub pos: &'a mut DeviceBuffer<u32>,
    pub n_keys: &'a mut DeviceBuffer<u32>,
}

/// [`Q38Kernels::enqueue_shared_add`]'s arguments: the host tier's routed
/// sums and the shared expert's outputs (`[m][n]` each), the router's
/// weights (`slots` a token) and the shared expert's slot among them, and
/// the output.
pub struct SharedAddArgs<'a> {
    pub hsum: &'a DeviceBuffer<f32>,
    pub sh: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub slot: usize,
    pub slots: usize,
    pub n: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_card_acc`]'s arguments: `m` tokens' [`SLOTS`]
/// slots — their down outputs slot-major (`n` values a slot, token `t`'s
/// slots from `SLOTS · t`), the router's weights ([`W_PITCH`] a token) and
/// the slots' places (the card's below `n_card`, the layer's card expert
/// count; the host's [`HOST`]) — the sink a place that is neither raises
/// on, and the sum's output, `n` values a token.
pub struct CardAccArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub fault: FaultSink,
    pub acc: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_card_shared_add`]'s arguments: [`SharedAddArgs`]'s
/// and the card slots' sum `acc` (`[m][n]`, [`Q38Kernels::enqueue_card_acc`]).
pub struct CardSharedAddArgs<'a> {
    pub hsum: &'a DeviceBuffer<f32>,
    pub acc: &'a DeviceBuffer<f32>,
    pub sh: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub slot: usize,
    pub slots: usize,
    pub n: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_card_tier_shared_add`]'s arguments: `m` tokens'
/// [`SLOTS`] slots — the card's down outputs and the tier's rows, slot-major
/// (`n` values a slot), the router's weights ([`W_PITCH`] a token, the
/// shared expert's gate last), the card places (below `n_card`, else
/// [`HOST`]) and the tier places (below `n_tier`, else [`HOST`]) — the host
/// tier's routed sums and the shared expert's outputs (`[m][n]` each), the
/// sink and the output.
pub struct CardTierSharedAddArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub trows: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub tsel: &'a DeviceBuffer<u32>,
    pub hsum: &'a DeviceBuffer<f32>,
    pub sh: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_tier: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_card_tier_acc`]'s arguments: [`CardTierSharedAddArgs`]'s
/// slots — the card's down outputs and the tier's rows, slot-major, the
/// router's weights, the card and the tier places — over `m` tokens of `n`
/// values, the sink and the sum.
pub struct CardTierAccArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub trows: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub tsel: &'a DeviceBuffer<u32>,
    pub trank: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_tier: usize,
    pub fault: FaultSink,
    pub acc: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_tier_rows_pack`]'s arguments: a run's down outputs
/// (slot-major), tier places and ranks over `slots` slots of `n` values, the
/// tier's expert count, and the block's down outputs the run packs into, of
/// `rows_cap` rows.
pub struct TierRowsPackArgs<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub rank: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub slots: usize,
    pub n_tier: usize,
    pub rows_cap: usize,
    pub fault: FaultSink,
    pub rows: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_out_gate`]'s arguments: the attention output and
/// the `[q | gate]` rows it is gated by, `m` tokens of `n_head` heads, and
/// the output.
pub struct OutGateArgs<'a> {
    pub attn: &'a DeviceBuffer<f32>,
    pub qg: &'a DeviceBuffer<f32>,
    pub n_head: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_key_append`]'s arguments: the f32 gemv's `[DIM][m]`
/// raw keys and the `m` rows' positions, the plane's `ctx` rows, and the
/// layer's raw plane.
pub struct KeyAppendArgs<'a> {
    pub kr: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub m: usize,
    pub ctx: usize,
    pub fault: FaultSink,
    pub raw: &'a mut DeviceBuffer<u16>,
}

/// A buffer shorter than a launch reads or writes, named.
fn short(what: &'static str, lens: &[(&str, usize, usize)]) -> Result<(), GpuError> {
    match lens.iter().find(|(_, got, need)| got < need) {
        Some((name, got, need)) => Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        )),
        None => Ok(()),
    }
}

/// A launch of `n` threads in blocks of [`THREADS`].
fn grid(what: &'static str, n: usize) -> Result<LaunchConfig1D, GpuError> {
    let blocks = launch_u32(what, "grid", n.div_ceil(THREADS as usize))?;
    Ok(LaunchConfig1D::new(blocks, THREADS, 0))
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct Q38Kernels {
    module: q38_kernels::LoadedModule,
}

impl Q38Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q38Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(q38_kernels, ctx)? };
        Ok(Q38Kernels { module })
    }

    /// Enqueue the Q8_0 embedding lookup of one unit of rows ([`EmbedQ8Args`],
    /// module doc): `k = 32 · d.cols()` values a row, `qs.cols()` its
    /// `8 · d.cols()` words. The rows are `gguf::quant::dequant_row`'s bits.
    /// One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows(
        &self,
        stream: &CudaStream,
        a: EmbedQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_embed_rows";
        let (qs, d) = (a.qs, a.d);
        let k32 = d.cols();
        if qs.rows() == 0 || qs.rows() != d.rows() || k32 == 0 || qs.cols() != 8 * k32 {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q8_0 planes qs {}x{} and d {}x{}: equal rows, 8 words a scale",
                    qs.rows(),
                    qs.cols(),
                    d.rows(),
                    d.cols()
                ),
            ));
        }
        let n = a.ids.len();
        if n == 0 {
            return Err(GpuError::shape(what, "empty ids"));
        }
        short(
            what,
            &[
                ("y", a.y.len(), 32 * k32 * n),
                ("pos0", a.pos0.len(), 1),
                ("pos", a.pos.len(), n),
                ("n_keys", a.n_keys.len(), n),
            ],
        )?;
        let cfg = grid(what, 32 * k32 * n)?;
        let first = launch_u32(what, "first", a.first)?;
        let n_rows = launch_u32(what, "rows", qs.rows())?;
        let k32 = launch_u32(what, "k32", k32)?;
        let prep = self.module.prepare_embed_rows_q8_0(cfg)?;
        self.module.embed_rows_q8_0(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            a.ids,
            a.pos0,
            first,
            n_rows,
            k32,
            a.fault,
            a.y,
            a.pos,
            a.n_keys,
        )?;
        Ok(())
    }

    /// Enqueue `y = attn · σ(gate)` over `m` tokens of `n_head` heads of
    /// [`HEAD`] ([`OutGateArgs`], module doc): `attn` and `y`
    /// `m·n_head·HEAD`, `qg` the `[q | gate]` rows, `m·n_head·2·HEAD`. One
    /// launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_out_gate(
        &self,
        stream: &CudaStream,
        a: OutGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let OutGateArgs {
            attn,
            qg,
            n_head,
            m,
            fault,
            y,
        } = a;
        let what = "q38::enqueue_out_gate";
        let n = HEAD * n_head * m;
        if n == 0 {
            return Err(GpuError::shape(what, "no head or no token"));
        }
        short(
            what,
            &[
                ("attn", attn.len(), n),
                ("qg", qg.len(), 2 * n),
                ("y", y.len(), n),
            ],
        )?;
        let cfg = grid(what, n)?;
        let prep = self.module.prepare_gqa_out_gate_f32(cfg)?;
        self.module.gqa_out_gate_f32(
            stream,
            &prep,
            attn,
            qg,
            launch_u32(what, "n_head", n_head)?,
            launch_u32(what, "m", m)?,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the raw key append of `m` rows ([`KeyAppendArgs`], module
    /// doc): `kr` the f32 gemv's `[DIM][m]` output, `pos` the rows'
    /// positions, `raw` the layer's `[ctx][DIM]` f16 plane. One launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_key_append(
        &self,
        stream: &CudaStream,
        a: KeyAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let KeyAppendArgs {
            kr,
            pos,
            m,
            ctx,
            fault,
            raw,
        } = a;
        let what = "q38::enqueue_key_append";
        if m == 0 || ctx == 0 {
            return Err(GpuError::shape(
                what,
                format!("need m and ctx >= 1, got m={m} ctx={ctx}"),
            ));
        }
        short(
            what,
            &[
                ("kr", kr.len(), DIM * m),
                ("pos", pos.len(), m),
                ("raw", raw.len(), DIM * ctx),
            ],
        )?;
        let cfg = grid(what, DIM * m)?;
        let prep = self.module.prepare_qsa_key_append(cfg)?;
        self.module.qsa_key_append(
            stream,
            &prep,
            kr,
            pos,
            launch_u32(what, "m", m)?,
            launch_u32(what, "ctx", ctx)?,
            fault,
            raw,
        )?;
        Ok(())
    }

    /// Enqueue the shared expert's gated sum ([`SharedAddArgs`], module
    /// doc). One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_shared_add(
        &self,
        stream: &CudaStream,
        a: SharedAddArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_shared_add";
        let nm = a.n * a.m;
        if nm == 0 || a.slot >= a.slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens, slot {} of {}",
                    a.n, a.m, a.slot, a.slots
                ),
            ));
        }
        short(
            what,
            &[
                ("hsum", a.hsum.len(), nm),
                ("sh", a.sh.len(), nm),
                ("w", a.w.len(), a.slots * a.m),
                ("y", a.y.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_shared_add(cfg)?;
        self.module.q38_shared_add(
            stream,
            &prep,
            a.hsum,
            a.sh,
            a.w,
            launch_u32(what, "slot", a.slot)?,
            launch_u32(what, "slots", a.slots)?,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            a.fault,
            a.y,
        )?;
        Ok(())
    }

    /// Enqueue the card slots' sum ([`CardAccArgs`], module doc): `down`
    /// `SLOTS·n·m` values, `w` `W_PITCH·m`, `sel` `SLOTS·m`, `acc` `n·m`. A
    /// card of no expert is refused by name: a layer without card experts has
    /// no card sum and keeps [`Q38Kernels::enqueue_shared_add`]. A place in
    /// `[n_card, HOST)` raises [`FaultSite::ExpertId`] on `a.fault`. One launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_card_acc(
        &self,
        stream: &CudaStream,
        a: CardAccArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_card_acc";
        let nm = a.n * a.m;
        if nm == 0 || a.n_card == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens over a card of {} experts: at least one of each",
                    a.n, a.m, a.n_card
                ),
            ));
        }
        short(
            what,
            &[
                ("down", a.down.len(), SLOTS * nm),
                ("w", a.w.len(), W_PITCH * a.m),
                ("sel", a.sel.len(), SLOTS * a.m),
                ("acc", a.acc.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_card_acc(cfg)?;
        self.module.q38_card_acc(
            stream,
            &prep,
            a.down,
            a.w,
            a.sel,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            a.fault,
            a.acc,
        )?;
        Ok(())
    }

    /// Enqueue the combine with the card sum ([`CardSharedAddArgs`], module
    /// doc): the host's sum, then the card's, then the shared expert's gated
    /// output. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_card_shared_add(
        &self,
        stream: &CudaStream,
        a: CardSharedAddArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_card_shared_add";
        let nm = a.n * a.m;
        if nm == 0 || a.slot >= a.slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens, slot {} of {}",
                    a.n, a.m, a.slot, a.slots
                ),
            ));
        }
        short(
            what,
            &[
                ("hsum", a.hsum.len(), nm),
                ("acc", a.acc.len(), nm),
                ("sh", a.sh.len(), nm),
                ("w", a.w.len(), a.slots * a.m),
                ("y", a.y.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_card_shared_add(cfg)?;
        self.module.q38_card_shared_add(
            stream,
            &prep,
            a.hsum,
            a.acc,
            a.sh,
            a.w,
            launch_u32(what, "slot", a.slot)?,
            launch_u32(what, "slots", a.slots)?,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            a.fault,
            a.y,
        )?;
        Ok(())
    }

    /// Enqueue the join of a tier layer ([`CardTierSharedAddArgs`], module
    /// doc): the card's and the tier's slots' sum in slot order, then the
    /// combine with the host's sum and the shared expert's gated output.
    /// `down` and `trows` `SLOTS·n·m` values, `w` `W_PITCH·m`, `sel` and
    /// `tsel` `SLOTS·m`, `hsum`, `sh` and `y` `n·m`. A layer with no card or
    /// no tier expert is refused by name: the card alone joins by
    /// [`Q38Kernels::enqueue_card_shared_add`]. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_card_tier_shared_add(
        &self,
        stream: &CudaStream,
        a: CardTierSharedAddArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_card_tier_shared_add";
        let nm = a.n * a.m;
        if nm == 0 || a.n_card == 0 || a.n_tier == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens over a card of {} experts and a tier of {}: at least \
                     one of each",
                    a.n, a.m, a.n_card, a.n_tier
                ),
            ));
        }
        short(
            what,
            &[
                ("down", a.down.len(), SLOTS * nm),
                ("trows", a.trows.len(), SLOTS * nm),
                ("w", a.w.len(), W_PITCH * a.m),
                ("sel", a.sel.len(), SLOTS * a.m),
                ("tsel", a.tsel.len(), SLOTS * a.m),
                ("hsum", a.hsum.len(), nm),
                ("sh", a.sh.len(), nm),
                ("y", a.y.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_card_tier_shared_add(cfg)?;
        self.module.q38_card_tier_shared_add(
            stream,
            &prep,
            a.down,
            a.trows,
            a.w,
            a.sel,
            a.tsel,
            a.hsum,
            a.sh,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            launch_u32(what, "n_tier", a.n_tier)?,
            a.fault,
            a.y,
        )?;
        Ok(())
    }

    /// Enqueue a ubatch tier layer's join sum ([`CardTierAccArgs`],
    /// `q38_card_tier_acc`): the card's and the tier's slots in slot order
    /// into `acc`, which [`Q38Kernels::enqueue_card_shared_add`] then
    /// combines. `down` and `trows` `SLOTS·n·m` values, `w` `W_PITCH·m`,
    /// `sel`, `tsel` and `trank` `SLOTS·m`, `acc` `n·m`. A layer with no card or no
    /// tier expert is refused by name. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_card_tier_acc(
        &self,
        stream: &CudaStream,
        a: CardTierAccArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_card_tier_acc";
        let nm = a.n * a.m;
        if nm == 0 || a.n_card == 0 || a.n_tier == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens over a card of {} experts and a tier of {}: at least \
                     one of each",
                    a.n, a.m, a.n_card, a.n_tier
                ),
            ));
        }
        short(
            what,
            &[
                ("down", a.down.len(), SLOTS * nm),
                ("trows", a.trows.len(), SLOTS * nm),
                ("w", a.w.len(), W_PITCH * a.m),
                ("sel", a.sel.len(), SLOTS * a.m),
                ("tsel", a.tsel.len(), SLOTS * a.m),
                ("trank", a.trank.len(), SLOTS * a.m),
                ("acc", a.acc.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_card_tier_acc(cfg)?;
        self.module.q38_card_tier_acc(
            stream,
            &prep,
            a.down,
            a.trows,
            a.w,
            a.sel,
            a.tsel,
            a.trank,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            launch_u32(what, "n_tier", a.n_tier)?,
            a.fault,
            a.acc,
        )?;
        Ok(())
    }

    /// Enqueue a tier route's expert ids from its first `slots` places
    /// (`q38_tier_ids`): a tier place as it is, [`HOST`] as `n_tier`. A
    /// tier of no expert, no slot, and buffers short of `slots` are refused
    /// by name. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_tier_ids(
        &self,
        stream: &CudaStream,
        sel: &DeviceBuffer<u32>,
        (slots, n_tier): (usize, usize),
        ids: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_tier_ids";
        if slots == 0 || n_tier == 0 {
            return Err(GpuError::shape(
                what,
                format!("{slots} slots over a tier of {n_tier} experts: at least one of each"),
            ));
        }
        short(
            what,
            &[("sel", sel.len(), slots), ("ids", ids.len(), slots)],
        )?;
        let cfg = grid(what, slots)?;
        let prep = self.module.prepare_q38_tier_ids(cfg)?;
        self.module.q38_tier_ids(
            stream,
            &prep,
            sel,
            launch_u32(what, "slots", slots)?,
            launch_u32(what, "n_tier", n_tier)?,
            ids,
        )?;
        Ok(())
    }

    /// Enqueue the ranks of the first `slots` places' tier slots
    /// (`q38_tier_rank`): a tier slot's rank among them, [`HOST`] for every
    /// other slot. A tier of no expert, no slot, and buffers short of `slots`
    /// are refused by name. One launch of one block. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_tier_rank(
        &self,
        stream: &CudaStream,
        sel: &DeviceBuffer<u32>,
        (slots, n_tier): (usize, usize),
        rank: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_tier_rank";
        if slots == 0 || n_tier == 0 {
            return Err(GpuError::shape(
                what,
                format!("{slots} slots over a tier of {n_tier} experts: at least one of each"),
            ));
        }
        short(
            what,
            &[("sel", sel.len(), slots), ("rank", rank.len(), slots)],
        )?;
        let threads = launch_u32(what, "threads", 32 * RANK_WARPS)?;
        let prep = self
            .module
            .prepare_q38_tier_rank(LaunchConfig1D::new(1, threads, 0))?;
        self.module.q38_tier_rank(
            stream,
            &prep,
            sel,
            launch_u32(what, "slots", slots)?,
            launch_u32(what, "n_tier", n_tier)?,
            rank,
        )?;
        Ok(())
    }

    /// Enqueue a run's tier slots' rows packed into the block's down outputs
    /// at their ranks ([`TierRowsPackArgs`], `q38_tier_rows_pack`): `down`
    /// `n·slots` values, `sel` and `rank` `slots`, `rows` `n·rows_cap`. A
    /// tier of no expert, no value, and buffers short of them are refused by
    /// name. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_tier_rows_pack(
        &self,
        stream: &CudaStream,
        a: TierRowsPackArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_tier_rows_pack";
        let ns = a.n * a.slots;
        if ns == 0 || a.n_tier == 0 || a.rows_cap == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} slots of {} values over a tier of {} experts into {} rows: at least one \
                     of each",
                    a.slots, a.n, a.n_tier, a.rows_cap
                ),
            ));
        }
        short(
            what,
            &[
                ("down", a.down.len(), ns),
                ("sel", a.sel.len(), a.slots),
                ("rank", a.rank.len(), a.slots),
                ("rows", a.rows.len(), a.n * a.rows_cap),
            ],
        )?;
        let cfg = grid(what, ns)?;
        let prep = self.module.prepare_q38_tier_rows_pack(cfg)?;
        self.module.q38_tier_rows_pack(
            stream,
            &prep,
            a.down,
            a.sel,
            a.rank,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "slots", a.slots)?,
            launch_u32(what, "n_tier", a.n_tier)?,
            launch_u32(what, "rows_cap", a.rows_cap)?,
            a.fault,
            a.rows,
        )?;
        Ok(())
    }
}
