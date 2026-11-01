//! The SwiGLU epilogue of an up GEMM into q8 blocks of 32: the up GEMM's
//! products never stored, each tile's up values `u` and the gate GEMM's
//! rows go through `silu(gate) · u` into the down GEMM's activations (a
//! `GemmAct32` of K = the up's rows), each column the bytes and faults
//! `swiglu_quant32` (every slot, [`SwigluSlots::All`]) or
//! `swiglu_quant32_sel` (the card slots of `sel`, [`SwigluSlots::Card`])
//! gives it from the stored `u`. One device body, [`swiglu32`], serves the
//! block of both GEMM families, the grouped K-quant walk (`grouped.rs`) and
//! the 32-value walk (`gemm32.rs`): their warps hold the same fragment cells
//! (sixteen rows a warp over a [`GEMM_BM`]-row slab, n-tiles of eight
//! columns), so a slab's 128 rows are one quantizer group, staged in a free
//! shared stage a round of [`SW_NT`] n-tiles at a time once the walk is done.
//!
//! [`SwigluSlots`] names the columns the epilogue writes; the launchers
//! (`GemmKernels::enqueue_up_swiglu32` and
//! `Gemm32Kernels::enqueue_up_swiglu32`) take the entry of the up's type and
//! slot set where one is declared and otherwise run the up GEMM and the
//! split quantizer, the same bytes and faults.

use super::grouped::GEMM_BM;
use cuda_core::DeviceBuffer;

/// f32 words between two columns of a round the SwiGLU epilogue stages:
/// the slab's rows and four of pad, so the eight columns one store phase
/// writes land in distinct banks and each lane's four rows of a block are one
/// aligned 16-byte read.
pub(super) const SW_PITCH: usize = GEMM_BM + 4;
/// n-tiles of one such round.
pub(super) const SW_NT: usize = 4;
/// f32 words a round stages: a block's free stage must hold them.
pub(super) const SW_STAGE: usize = SW_NT * 8 * SW_PITCH;

/// The slots an up GEMM's SwiGLU epilogue writes into the down's
/// activations: every slot the route lists or refuses (a dense block's
/// one-expert table, `swiglu_quant32`'s columns), or the card slots of a
/// remapped route, `sel` each slot's place and a place below `n_card` a card
/// expert's (`swiglu_quant32_sel`'s).
#[derive(Clone, Copy)]
pub enum SwigluSlots<'a> {
    All,
    Card {
        sel: &'a DeviceBuffer<u32>,
        n_card: usize,
    },
}

/// The SwiGLU store's place of n-tile `nt` in its round of [`SW_NT`], and
/// whether the round ends there: a `fast` walk stores every live n-tile in
/// rounds, an `exact` one its one n-tile as a round.
macro_rules! sw_round {
    (fast, $nt:expr, $nt_live:expr) => {
        (
            $nt % $crate::gemm::swiglu32::SW_NT,
            $nt % $crate::gemm::swiglu32::SW_NT == $crate::gemm::swiglu32::SW_NT - 1
                || $nt + 1 == $nt_live,
        )
    };
    (exact, $nt:expr, $nt_live:expr) => {
        (0usize, true)
    };
}

/// The SwiGLU epilogue's device body (module doc). The output tuple is
/// `swiglu(gate, oblocks, osteps, oq, od, os, fault; slots)`, `slots` either
/// `all` or `card(sel, n_card)` ([`SwigluSlots`]); `oblocks` and `osteps` are
/// the down activation's blocks and steps a column, `rows == 32 · oblocks`.
///
/// - `@places`: block 0's check of every slot's place before the table's
///   walk, a place in `[n_card, HOST)` raising `ExpertId` even if the route
///   omits it (`card` only).
/// - `@store`: a walk's cells `acc` (mode, n-tile and cells as
///   `range_walks` gives them), each scaled by `scale` (`|v| expr`), into the
///   SwiGLU rows staged in `stage` (a free shared stage of [`SW_STAGE`] f32,
///   16-byte aligned), then quantized a round at a time. Every thread of the
///   block enters: it holds the block's barriers.
/// - `@refused`: a refused slot's quantizer group of slab `r / GEMM_BM` as
///   the quantizer's refused blocks (its `u` is NaN); the thread of slab row
///   `r % GEMM_BM` enters, and the first 32 rows of a slab are one warp's.
macro_rules! swiglu32 {
    (@on_card all, $slot:expr) => {
        true
    };
    (@on_card card($sel:ident, $n_card:ident), $slot:expr) => {{
        // SAFETY: the slot is below n_slots <= sel.len() (launch contract).
        let place = unsafe { *$sel.get_unchecked($slot) };
        place < $n_card
    }};
    (@places [$gate:ident, $ob:ident, $osteps:ident, $oq:ident, $od:ident, $os:ident,
        $fault:ident; card($sel:ident, $n_card:ident)], $n_slots:ident) => {{
        if thread::blockIdx_x() == 0 {
            let mut slot = thread::threadIdx_x() as usize;
            while slot < $n_slots as usize {
                // SAFETY: slot < n_slots <= sel.len() by the launch contract.
                let place = unsafe { *$sel.get_unchecked(slot) };
                if place != $crate::hybrid::HOST && place >= $n_card {
                    $fault.raise($crate::fault::FaultSite::ExpertId);
                }
                slot += $crate::gemm::grouped::GEMM_THREADS;
            }
        }
    }};
    (
        @store $mode:ident, $nx:expr, $acc:ident, |$v:ident| $scale:expr,
        [$gate:ident, $ob:ident, $osteps:ident, $oq:ident, $od:ident, $os:ident,
            $fault:ident; $($slots:tt)+], (
            $t:ident, $active:ident, $nt_live:ident, $len:ident, $row0:ident, $g:ident,
            $rows_n:ident, $slot_sh:ident, $stage:ident, $rt:ident
        )
    ) => {{
        const SW_PITCH: usize = $crate::gemm::swiglu32::SW_PITCH;
        const BM: usize = $crate::gemm::grouped::GEMM_BM;
        let lane = warp::lane_id() as usize;
        let wid = thread::threadIdx_x() as usize / 32;
        let (ob, osteps) = ($ob as usize, $osteps as usize);
        // Every warp is done with the walk, so the stage holds a round of
        // n-tiles' SwiGLU values: round column `c` at word `c · SW_PITCH`,
        // the slab's rows in order.
        thread::sync_threads();
        let hs = $stage.cast::<f32>();
        gemm_nts!($mode, $nx, false, $nt_live, |nt, j0| {
            let (k, last) = sw_round!($mode, nt, $nt_live);
            if $active {
                // The n-tile's gate rows, read together just before their
                // use: a walk's worth held beside its cells would spill.
                let mut gv = [0.0f32; 4];
                for i in 0..4usize {
                    ::cuda_device::thread::__unroll_config::<0>();
                    let n = 8 * nt + 2 * $t + (i & 1);
                    if n < $len {
                        let r = $row0 + $g + 8 * (i / 2);
                        // SAFETY: n < GEMM_BN bounds the shared read.
                        let slot = unsafe { *$slot_sh.add(n) } as usize;
                        // SAFETY: the slot is below n_slots and r below rows,
                        // so the read is inside gate (launch contract).
                        gv[i] = unsafe { *$gate.get_unchecked(slot * $rows_n + r) };
                    }
                }
                for i in 0..4usize {
                    ::cuda_device::thread::__unroll_config::<0>();
                    let c = 2 * $t + (i & 1);
                    if 8 * nt + c < $len {
                        let $v = $acc[j0 + i];
                        let h = $crate::elem::silu_mul(gv[i], $scale);
                        // SAFETY: round column 8k + c < 8 · SW_NT and slab row
                        // 16 · wid + g + 8(i / 2) < GEMM_BM put the word inside
                        // the stage (SW_STAGE); one lane writes it.
                        unsafe {
                            *hs.add((8 * k + c) * SW_PITCH + 16 * wid + $g + 8 * (i / 2)) = h;
                        }
                    }
                }
            }
            if last {
                thread::sync_threads();
                // Warp `wid` quantizes the round's columns `wid + 8m`, each as
                // `swiglu_quant32_sel`'s warp does its (column, group): the
                // group is the slab's four blocks.
                let mut m = 0usize;
                while m <= k {
                    let c = wid + 8 * m;
                    let n = 8 * (nt - k) + c;
                    if n < $len {
                        // SAFETY: n < GEMM_BN bounds the shared read.
                        let slot = unsafe { *$slot_sh.add(n) } as usize;
                        if swiglu32!(@on_card $($slots)+, slot) {
                            // A lane past the column's blocks holds the last
                            // block's values (`quant32_group`'s rule); every
                            // block of the group lies in this slab.
                            let b = (4 * $rt + (lane >> 3)).min(ob - 1);
                            // SAFETY: block b's 32 rows are slab rows below
                            // GEMM_BM of round column c < 8 · SW_NT, written
                            // above and published by the barrier; the word is
                            // 16-byte aligned (the stage is, and SW_PITCH and
                            // 4(lane % 8) are multiples of four).
                            let w = unsafe {
                                *hs.add(c * SW_PITCH + 32 * b - BM * $rt + 4 * (lane & 7))
                                    .cast::<::cuda_device::vector::U32x4>()
                            }
                            .0;
                            let v = [
                                f32::from_bits(w[0]),
                                f32::from_bits(w[1]),
                                f32::from_bits(w[2]),
                                f32::from_bits(w[3]),
                            ];
                            // SAFETY: the warp enters with one (slot, rt), the
                            // slot below n_slots and rt below ceil(ob / 4); the
                            // planes hold every slot's column (launch
                            // contract).
                            let refused = unsafe {
                                $crate::gemm::act32::quant32_group(
                                    v, slot, $rt, ob, osteps, lane, &mut $oq, &mut $od, &mut $os,
                                )
                            };
                            if refused && lane & 7 == 0 {
                                $fault.raise($crate::fault::FaultSite::QuantColumn);
                            }
                        }
                    }
                    m += 1;
                }
                // Every warp is done with the round before the next one's
                // stores.
                thread::sync_threads();
            }
        });
    }};
    (
        @refused [$gate:ident, $ob:ident, $osteps:ident, $oq:ident, $od:ident, $os:ident,
            $fault:ident; $($slots:tt)+], $slot:expr, $r:expr
    ) => {{
        const BM: usize = $crate::gemm::grouped::GEMM_BM;
        let (slot, r, osteps): (usize, usize, usize) = ($slot, $r, $osteps as usize);
        if r % BM < 32 && swiglu32!(@on_card $($slots)+, slot) {
            // SAFETY: one whole warp handles this (slot, slab); rows are a
            // multiple of 32, so its first warp is live even in the last slab.
            // slot < n_slots, r / GEMM_BM < ceil(ob / 4), and the launch
            // contract bounds all three planes. Each block has one writer.
            let refused = unsafe {
                $crate::gemm::act32::quant32_group(
                    [f32::NAN; 4], slot, r / BM, $ob as usize,
                    osteps, r % 32, &mut $oq, &mut $od, &mut $os,
                )
            };
            if refused && r % 32 == 0 {
                $fault.raise($crate::fault::FaultSite::QuantColumn);
            }
        }
    }};
}
