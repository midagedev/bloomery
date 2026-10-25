//! The card's expert-select entries over a quant type's file blocks, one
//! `#[cuda_module]` for every type of the two families that already have a
//! generic body, and the one struct that holds it ([`QuantSelKernels`]).
//!
//! - Raw blocks ([`crate::iq_sel`]'s `RawSb`): the format's own sub-block
//!   decode feeds the plane rule's `dp4a` chains. A type takes a `gate_up`
//!   entry ([`gate_up_sel_body`], one stack pair and a rule as a launch
//!   argument), a `gate_up_tab` entry (the gate·up of a format with a shared
//!   table), a `down` entry ([`down_sel_body`], slot `s` against column
//!   `s`) or a `down_tab` entry (the down of a format with a shared table,
//!   slot `s` against column `s / slots_per_col`).
//! - Walk A ([`crate::kquant::walk::SbDecode`]): the decoder feeds
//!   [`row_dot_1col`](crate::kquant::walk::row_dot_1col). A type takes a
//!   `gemv` (down), `gate_up`, `gemv_at` or `gate_up_at` entry (the `_at`
//!   forms read each slot's expert from an arena by offset).
//!
//! [`quant_sel_module!`] writes the entries from one line a type. Each entry
//! is a thin `#[kernel]` over the generic body: the thread-row split and the
//! early return are the entry's, the body is the family's, so the entry's PTX
//! is the hand-written wrapper's. A body moved into a helper changes the PTX
//! (thread-id ranges, block order), so the entries keep their own thread-row
//! lines, and `thread::index_1d` stays in the entry's own tokens, where the
//! kernel attribute rewrites it (a call inside a nested macro is not
//! rewritten). The macro emits the whole `#[cuda_module]`, not kernels inside
//! a hand-written one: a kernel a macro emits into a hand-written module gets
//! no launcher (docs/upstream/nvlabs-ledger.md). Entry names are the PTX
//! entry names, and each `prepare` ident names the `prepare_*` launcher
//! `#[cuda_module]` derives from its entry: a macro cannot join identifiers.
//!
//! What a new type of either family costs here is its line in the invocation
//! below and its decoder (`impl RawSb` in `iq_sel.rs`, `impl SbDecode` beside
//! `kquant::walk`). The raw family's launch dispatch ([`RawType`], the
//! `launch_raw_*` matches) and the Walk A family's launchers
//! (`enqueue_gemv_q5k_sel` and its kin) come from the same line, so a caller
//! of a new type calls the method the line names.
//!
//! The launch contracts spell the stacks in the format's file bytes
//! (`4 · w.len() >= .. · bb · n_sb`, `w` in whole words) for the raw family
//! and in super-block words for Walk A; `bb` and `words` are literals, and a
//! `const` assert per line holds each to its decoder's constant.
//!
//! The types are the card columns of each type's `gguf::QRow`, read through
//! `kernels::gate_up` and `kernels::down`: a const assert per line refuses a
//! line that has a gate·up or down entry where the row names none, or lacks
//! one where the row names one. The lines stay here because they name this
//! module's PTX entries, which the pure row cannot.

use crate::GpuError;
use crate::fault::FaultSink;
use crate::iq::{IQ2XXS_TAB_WORDS, Iq2XxsTab};
use crate::iq_sel::{
    Iq1M, Iq1S, Iq2S, Iq2Xs, Iq2Xxs, Iq3S, Iq3Xxs, Iq4Xs, RawDownLaunch, RawGateUpLaunch, RawSb,
    blk32_kernels, down_sel_body, gate_up_sel_body,
};
use crate::kquant::sel::{
    ARENA_PARTS, Fmt, GateUpAct, GateUpAt, SelDown, SelDownAt, down_at_dims, down_dims,
    gate_up_act_at_body, gate_up_act_body, gate_up_at_dims, gate_up_dims, gemv_at_body,
    gemv_sel_body, walk_a_planes,
};
use crate::kquant::walk::{Q4k, SbDecode};
use crate::kquant::{Q5k, Q8_0};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::quant::GgmlType;
use model::placement::kernels;
use std::sync::Arc;

/// Threads per block of every entry, one warp per output row.
pub const THREADS: u32 = 256;
/// Output rows a block computes: one per warp.
pub const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The entries' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// A raw-block type's row in the invocation: what its launchers check a stack
/// against and which entries it has.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawType {
    pub(crate) ty: GgmlType,
    /// The type's ggml name, as the launchers' errors spell it.
    pub(crate) name: &'static str,
    /// File bytes per 256-value super-block.
    pub(crate) bb: usize,
    pub(crate) gate_up: bool,
    /// The down reads a column a run of `slots_per_col` slots (`down_tab`).
    pub(crate) down_cols: bool,
}

/// The entry-writing macro: one `#[cuda_module]` named `$m` holding every
/// entry of the lines, `$S` the struct that owns its launchers, `$hm` the
/// hand-written module that is bound with it (the 32-value-block entry,
/// whose body is not generic over either decoder trait).
macro_rules! quant_sel_module {
    (
        module: $m:ident,
        owner: $S:ident,
        hand: $hm:ident,
        raw {
            $(
                $RTy:ident $rggml:ident bb $bb:tt {
                    $(
                        gate_up: $gu:ident, $gup:ident,
                    )?
                    $(
                        gate_up_tab: $gut:ident, $gutp:ident, $GTab:ident, $gtabw:ident,
                    )?
                    $(
                        down: $dn:ident, $dnp:ident,
                    )?
                    $(
                        down_tab: $dt:ident, $dtp:ident, $Tab:ident, $tabw:ident,
                    )?
                }
            )*
        }
        walk {
            $(
                $WTy:ident $wggml:ident words $ww:tt {
                    $(
                        walk_gemv: $gv:ident, $gvp:ident, $gve:ident,
                    )?
                    $(
                        walk_gate_up: $wgu:ident, $wgup:ident, $wgue:ident,
                    )?
                    $(
                        walk_gemv_at: $gva:ident, $gvap:ident, $gvae:ident,
                    )?
                    $(
                        walk_gate_up_at: $wgua:ident, $wguap:ident, $wguae:ident,
                    )?
                }
            )*
        }
    ) => {
        $( const _: () = assert!(<$RTy as RawSb>::BB == $bb); )*
        $( const _: () = assert!(<$WTy as SbDecode>::WORDS == $ww); )*
        // A raw line has a `gate_up` or `gate_up_tab` entry exactly where
        // `kernels::gate_up` has a row for its type and a `down` or `down_tab`
        // entry exactly where `kernels::down` has one; a Walk A line has a
        // `walk_gate_up` entry exactly where `kernels::gate_up` has a row.
        $(
            const _: () = assert!(
                kernels::gate_up(GgmlType::$rggml).is_some()
                    == (<[&str]>::len(&[$( stringify!($gu) ),*])
                        + <[&str]>::len(&[$( stringify!($gut) ),*])
                        == 1)
            );
            const _: () = assert!(
                kernels::down(GgmlType::$rggml).is_some()
                    == (<[&str]>::len(&[$( stringify!($dn) ),*])
                        + <[&str]>::len(&[$( stringify!($dt) ),*])
                        == 1)
            );
        )*
        $(
            const _: () = assert!(
                kernels::gate_up(GgmlType::$wggml).is_some()
                    == (<[&str]>::len(&[$( stringify!($wgu) ),*]) == 1)
            );
        )*

        #[allow(
            dead_code,
            reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
        )]
        #[cuda_module]
        pub(crate) mod $m {
            use super::*;

            $(
                $(
                    /// The gate·up `_sel` over raw blocks of the line's type
                    /// ([`gate_up_sel_body`]): the rule is a launch argument,
                    /// and `m_cols` only bounds the activation in the launch
                    /// contract.
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
                            4 * wg.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            4 * wu.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            slots_per_col >= 1,
                            n_slots <= m_cols * slots_per_col,
                            q.len() >= m_cols * 256 * n_grp,
                            d8.len() >= m_cols * 2 * n_sb,
                            sel.len() >= n_slots,
                            h.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $gu(
                        wg: &[u32],
                        wu: &[u32],
                        q: &[u32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        m_cols: u32,
                        slots_per_col: u32,
                        n_sb: u32,
                        n_grp: u32,
                        act: u32,
                        limit: f32,
                        fault: FaultSink,
                        mut h: DisjointSlice<f32>,
                    ) {
                        // m_cols only bounds the activation in the launch contract.
                        let _ = m_cols;
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // The launch contract is the body's (`bb` bytes a
                        // super-block); the host passes n_grp = ceil(n_sb/4)
                        // and one of Act::code's codes; the return above is
                        // warp-uniform (a warp's lanes share `row`) and keeps
                        // row in range.
                        gate_up_sel_body::<$RTy>(
                            wg,
                            wu,
                            q,
                            d8,
                            sel,
                            n_experts,
                            rows_per_expert,
                            slots_per_col,
                            n_sb,
                            n_grp,
                            act,
                            limit,
                            fault,
                            row,
                            &mut h,
                            (),
                        );
                    }
                )?

                $(
                    /// The gate·up `_sel` over raw blocks of a format with a
                    /// shared table ([`gate_up_sel_body`]): the table is
                    /// filled into shared memory ahead of any return, the
                    /// rule is a launch argument, and `m_cols` only bounds
                    /// the activation in the launch contract.
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
                            4 * wg.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            4 * wu.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            slots_per_col >= 1,
                            n_slots <= m_cols * slots_per_col,
                            q.len() >= m_cols * 256 * n_grp,
                            d8.len() >= m_cols * 2 * n_sb,
                            sel.len() >= n_slots,
                            h.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $gut(
                        wg: &[u32],
                        wu: &[u32],
                        q: &[u32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        m_cols: u32,
                        slots_per_col: u32,
                        n_sb: u32,
                        n_grp: u32,
                        act: u32,
                        limit: f32,
                        fault: FaultSink,
                        mut h: DisjointSlice<f32>,
                    ) {
                        static mut TAB: SharedArray<u32, $gtabw> = SharedArray::UNINIT;
                        // SAFETY: TAB is this block's alone; every thread of the
                        // 256-thread block fills it here, before any return.
                        let tab = unsafe { $GTab::fill(SharedArray::as_raw_mut_ptr(&raw mut TAB)) };
                        // m_cols only bounds the activation in the launch contract.
                        let _ = m_cols;
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // The launch contract is the body's (`bb` bytes a
                        // super-block); the host passes n_grp = ceil(n_sb/4)
                        // and one of Act::code's codes; the return above is
                        // warp-uniform (a warp's lanes share `row`) and keeps
                        // row in range.
                        gate_up_sel_body::<$RTy>(
                            wg,
                            wu,
                            q,
                            d8,
                            sel,
                            n_experts,
                            rows_per_expert,
                            slots_per_col,
                            n_sb,
                            n_grp,
                            act,
                            limit,
                            fault,
                            row,
                            &mut h,
                            tab,
                        );
                    }
                )?

                $(
                    /// The down `_sel` over raw blocks of the line's type
                    /// ([`down_sel_body`]): one warp per output row, eight rows
                    /// per 256-thread block; thread row `n = slot ·
                    /// rows_per_expert + r` stores `y[n]`, the dot of weight row
                    /// `sel[slot] · rows_per_expert + r` with column `slot` of
                    /// the [`Q8Act`](crate::tensor::Q8Act).
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
                            4 * w.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            q.len() >= n_slots * 256 * n_grp,
                            d8.len() >= n_slots * 2 * n_sb,
                            sel.len() >= n_slots,
                            y.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $dn(
                        w: &[u32],
                        q: &[u32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        n_sb: u32,
                        n_grp: u32,
                        fault: FaultSink,
                        mut y: DisjointSlice<f32>,
                    ) {
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // The launch contract is the body's; the host passes
                        // n_grp = ceil(n_sb/4); the return above is
                        // warp-uniform and keeps row in range.
                        down_sel_body::<$RTy>(
                            w,
                            q,
                            d8,
                            sel,
                            n_experts,
                            rows_per_expert,
                            1,
                            n_sb,
                            n_grp,
                            fault,
                            row,
                            &mut y,
                            (),
                        );
                    }
                )?

                $(
                    /// The down `_sel` over raw blocks of a format with a
                    /// shared table ([`down_sel_body`]): the table is filled
                    /// into shared memory ahead of any return, and slot `s`
                    /// reads column `s / slots_per_col` of the activation (a
                    /// routed up shares a token's column; a down is
                    /// `slots_per_col` 1). `m_cols` only bounds the activation
                    /// in the launch contract.
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
                            4 * w.len() >= n_experts * rows_per_expert * $bb * n_sb,
                            slots_per_col >= 1,
                            n_slots <= m_cols * slots_per_col,
                            q.len() >= m_cols * 256 * n_grp,
                            d8.len() >= m_cols * 2 * n_sb,
                            sel.len() >= n_slots,
                            y.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $dt(
                        w: &[u32],
                        q: &[u32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        m_cols: u32,
                        slots_per_col: u32,
                        n_sb: u32,
                        n_grp: u32,
                        fault: FaultSink,
                        mut y: DisjointSlice<f32>,
                    ) {
                        static mut TAB: SharedArray<u32, $tabw> = SharedArray::UNINIT;
                        // SAFETY: TAB is this block's alone; every thread of the
                        // 256-thread block fills it here, before any return.
                        let tab = unsafe { $Tab::fill(SharedArray::as_raw_mut_ptr(&raw mut TAB)) };
                        // m_cols only bounds the activation in the launch contract.
                        let _ = m_cols;
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // The launch contract is the body's (n_slots <= m_cols ·
                        // slots_per_col, so every slot's column is below
                        // m_cols); the host passes n_grp = ceil(n_sb/4); the
                        // return above is warp-uniform and keeps row in range.
                        down_sel_body::<$RTy>(
                            w,
                            q,
                            d8,
                            sel,
                            n_experts,
                            rows_per_expert,
                            slots_per_col,
                            n_sb,
                            n_grp,
                            fault,
                            row,
                            &mut y,
                            tab,
                        );
                    }
                )?
            )*

            $(
                $(
                    /// The Walk A down `_sel` of the line's type
                    /// ([`gemv_sel_body`]), four blocks an SM pinned: a
                    /// register count past 64 spills (the spill ratchet's red)
                    /// instead of dropping a resident block.
                    #[allow(
                        clippy::too_many_arguments,
                        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
                    )]
                    #[kernel]
                    #[launch_bounds(256, 4)]
                    #[launch_contract(
                        domain = 1,
                        block = (256, 1, 1),
                        requires = (
                            w.len() >= n_experts * rows_per_expert * $ww * n_sb,
                            q.len() >= n_slots * 256 * iters,
                            s8.len() >= n_slots * 8 * n_sb,
                            d8.len() >= n_slots * 2 * n_sb,
                            sel.len() >= n_slots,
                            y.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $gv(
                        w: &[u32],
                        q: &[u32],
                        s8: &[i32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        n_sb: u32,
                        iters: u32,
                        fault: FaultSink,
                        mut y: DisjointSlice<f32>,
                    ) {
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // SAFETY: the launch contract is the body's (`words` =
                        // the decoder's WORDS); the host passes iters =
                        // ceil(n_sb/4); the return above is warp-uniform (a
                        // warp's lanes share `row`) and keeps row in range.
                        unsafe {
                            gemv_sel_body::<$WTy>(
                                w,
                                q,
                                s8,
                                d8,
                                sel,
                                n_experts,
                                rows_per_expert,
                                n_sb,
                                iters,
                                fault,
                                row,
                                &mut y,
                            );
                        }
                    }
                )?

                $(
                    /// The Walk A gate·up of the line's type with its rule as
                    /// an argument ([`gate_up_act_body`]), four blocks an SM
                    /// pinned as the down.
                    #[allow(
                        clippy::too_many_arguments,
                        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
                    )]
                    #[kernel]
                    #[launch_bounds(256, 4)]
                    #[launch_contract(
                        domain = 1,
                        block = (256, 1, 1),
                        requires = (
                            wg.len() >= n_experts * rows_per_expert * $ww * n_sb,
                            wu.len() >= n_experts * rows_per_expert * $ww * n_sb,
                            slots_per_col >= 1,
                            n_slots <= m_cols * slots_per_col,
                            q.len() >= m_cols * 256 * iters,
                            s8.len() >= m_cols * 8 * n_sb,
                            d8.len() >= m_cols * 2 * n_sb,
                            sel.len() >= n_slots,
                            h.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $wgu(
                        wg: &[u32],
                        wu: &[u32],
                        q: &[u32],
                        s8: &[i32],
                        d8: &[f32],
                        sel: &[u32],
                        n_experts: u32,
                        rows_per_expert: u32,
                        n_slots: u32,
                        m_cols: u32,
                        slots_per_col: u32,
                        n_sb: u32,
                        iters: u32,
                        act: u32,
                        limit: f32,
                        fault: FaultSink,
                        mut h: DisjointSlice<f32>,
                    ) {
                        // m_cols only bounds the activation in the launch contract.
                        let _ = m_cols;
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // SAFETY: the launch contract is the body's (`words` =
                        // the decoder's WORDS); the host passes iters =
                        // ceil(n_sb/4) and one of Act::code's codes; the return
                        // above is warp-uniform and keeps row in range.
                        unsafe {
                            gate_up_act_body::<$WTy>(
                                wg,
                                wu,
                                q,
                                s8,
                                d8,
                                sel,
                                n_experts,
                                rows_per_expert,
                                slots_per_col,
                                n_sb,
                                iters,
                                act,
                                limit,
                                fault,
                                row,
                                &mut h,
                            );
                        }
                    }
                )?

                $(
                    /// The Walk A down arena entry of the line's type
                    /// ([`gemv_at_body`]), four blocks an SM pinned as the
                    /// stack down.
                    #[allow(
                        clippy::too_many_arguments,
                        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
                    )]
                    #[kernel]
                    #[launch_bounds(256, 4)]
                    #[launch_contract(
                        domain = 1,
                        block = (256, 1, 1),
                        requires = (
                            offs.len() >= n_slots * 3,
                            q.len() >= n_slots * 256 * iters,
                            s8.len() >= n_slots * 8 * n_sb,
                            d8.len() >= n_slots * 2 * n_sb,
                            y.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $gva(
                        arena: &[u32],
                        offs: &[u64],
                        q: &[u32],
                        s8: &[i32],
                        d8: &[f32],
                        rows_per_expert: u32,
                        n_slots: u32,
                        n_sb: u32,
                        iters: u32,
                        fault: FaultSink,
                        mut y: DisjointSlice<f32>,
                    ) {
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // SAFETY: the launch contract is the body's (3 =
                        // ARENA_PARTS); the host passes iters = ceil(n_sb/4);
                        // the return above is warp-uniform and keeps row in
                        // range.
                        unsafe {
                            gemv_at_body::<$WTy>(
                                arena,
                                offs,
                                q,
                                s8,
                                d8,
                                rows_per_expert,
                                n_sb,
                                iters,
                                fault,
                                row,
                                &mut y,
                            );
                        }
                    }
                )?

                $(
                    /// The Walk A gate·up arena entry of the line's type
                    /// ([`gate_up_act_at_body`]), four blocks an SM pinned as
                    /// the stack down.
                    #[allow(
                        clippy::too_many_arguments,
                        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
                    )]
                    #[kernel]
                    #[launch_bounds(256, 4)]
                    #[launch_contract(
                        domain = 1,
                        block = (256, 1, 1),
                        requires = (
                            offs.len() >= n_slots * 3,
                            slots_per_col >= 1,
                            n_slots <= m_cols * slots_per_col,
                            q.len() >= m_cols * 256 * iters,
                            s8.len() >= m_cols * 8 * n_sb,
                            d8.len() >= m_cols * 2 * n_sb,
                            h.len() >= n_slots * rows_per_expert
                        )
                    )]
                    pub fn $wgua(
                        arena: &[u32],
                        offs: &[u64],
                        q: &[u32],
                        s8: &[i32],
                        d8: &[f32],
                        rows_per_expert: u32,
                        n_slots: u32,
                        m_cols: u32,
                        slots_per_col: u32,
                        n_sb: u32,
                        iters: u32,
                        act: u32,
                        limit: f32,
                        fault: FaultSink,
                        mut h: DisjointSlice<f32>,
                    ) {
                        // m_cols only bounds the activation in the launch contract.
                        let _ = m_cols;
                        let t = thread::index_1d().get() % THREADS as usize;
                        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
                        if row >= n_slots as usize * rows_per_expert as usize {
                            return;
                        }
                        // SAFETY: the launch contract is the body's (3 =
                        // ARENA_PARTS); the host passes iters = ceil(n_sb/4)
                        // and one of Act::code's codes; the return above is
                        // warp-uniform and keeps row in range.
                        unsafe {
                            gate_up_act_at_body::<$WTy>(
                                arena,
                                offs,
                                q,
                                s8,
                                d8,
                                rows_per_expert,
                                slots_per_col,
                                n_sb,
                                iters,
                                act,
                                limit,
                                fault,
                                row,
                                &mut h,
                            );
                        }
                    }
                )?
            )*
        }

        /// The loaded expert-select module and its launchers. Owns no stream:
        /// every enqueue takes the engine stream, so launches order with the
        /// step and are capturable.
        pub struct $S {
            pub(crate) module: $m::LoadedModule,
            pub(crate) hand: $hm::LoadedModule,
            /// The fault word of the `Gpu` that owns the context: the launches'
            /// sinks point into it, and the modules keep it alive.
            _fault: Arc<DeviceBuffer<u32>>,
        }

        impl $S {
            /// Load this file's device bundle into `ctx`, whose launches raise
            /// into `word`, the fault word of the `Gpu` that owns `ctx`
            /// ([`crate::Gpu::fault_word`]); a word of another context is
            /// refused. Load-time only.
            pub fn load(
                ctx: &Arc<CudaContext>,
                word: &Arc<DeviceBuffer<u32>>,
            ) -> Result<$S, GpuError> {
                let fault =
                    crate::module_fault_word(ctx, word, concat!(stringify!($S), "::load"))?;
                // SAFETY: this package owns $m's embedded bundle; its
                // launchers check their launch contracts.
                let module = unsafe { crate::shared_module!($m, ctx)? };
                // SAFETY: this package owns $hm's embedded bundle; its
                // launchers check their launch contracts.
                let hand = unsafe { crate::shared_module!($hm, ctx)? };
                Ok($S {
                    module,
                    hand,
                    _fault: fault,
                })
            }

            /// The raw-block types the invocation lists, in its order.
            pub(crate) const RAW_TYPES: &'static [RawType] = &[
                $(
                    RawType {
                        ty: GgmlType::$rggml,
                        name: stringify!($rggml),
                        bb: $bb,
                        gate_up: <[&str]>::len(&[$( stringify!($gu) ),*])
                            + <[&str]>::len(&[$( stringify!($gut) ),*])
                            == 1,
                        down_cols: <[&str]>::len(&[$( stringify!($dt) ),*]) == 1,
                    },
                )*
            ];

            /// Launch the raw-block gate·up entry of `ty` over `grid` blocks
            /// of [`THREADS`]; an `Err` names a type with no such entry.
            pub(crate) fn launch_raw_gate_up(
                &self,
                stream: &CudaStream,
                ty: GgmlType,
                grid: u32,
                a: &RawGateUpLaunch<'_>,
                fault: FaultSink,
                h: &mut DeviceBuffer<f32>,
            ) -> Result<(), GpuError> {
                let cfg = LaunchConfig1D::new(grid, THREADS, 0);
                match ty {
                    $( $(
                        GgmlType::$rggml => {
                            let prep = self.module.$gup(cfg)?;
                            self.module.$gu(
                                stream,
                                &prep,
                                a.wg,
                                a.wu,
                                a.q,
                                a.d8,
                                a.sel,
                                a.n_experts,
                                a.rows_per_expert,
                                a.n_slots,
                                a.m_cols,
                                a.slots_per_col,
                                a.n_sb,
                                a.n_grp,
                                a.act,
                                a.limit,
                                fault,
                                h,
                            )?;
                        }
                    )? $(
                        GgmlType::$rggml => {
                            let prep = self.module.$gutp(cfg)?;
                            self.module.$gut(
                                stream,
                                &prep,
                                a.wg,
                                a.wu,
                                a.q,
                                a.d8,
                                a.sel,
                                a.n_experts,
                                a.rows_per_expert,
                                a.n_slots,
                                a.m_cols,
                                a.slots_per_col,
                                a.n_sb,
                                a.n_grp,
                                a.act,
                                a.limit,
                                fault,
                                h,
                            )?;
                        }
                    )? )*
                    other => {
                        return Err(GpuError::shape(
                            "launch_raw_gate_up",
                            format!("no raw-block gate·up entry for {other:?}"),
                        ));
                    }
                }
                Ok(())
            }

            /// Launch the raw-block down entry of `ty` over `grid` blocks of
            /// [`THREADS`]; an `Err` names a type with no such entry.
            pub(crate) fn launch_raw_down(
                &self,
                stream: &CudaStream,
                ty: GgmlType,
                grid: u32,
                a: &RawDownLaunch<'_>,
                fault: FaultSink,
                y: &mut DeviceBuffer<f32>,
            ) -> Result<(), GpuError> {
                let cfg = LaunchConfig1D::new(grid, THREADS, 0);
                match ty {
                    $( $(
                        GgmlType::$rggml => {
                            let prep = self.module.$dnp(cfg)?;
                            self.module.$dn(
                                stream,
                                &prep,
                                a.w,
                                a.q,
                                a.d8,
                                a.sel,
                                a.n_experts,
                                a.rows_per_expert,
                                a.n_slots,
                                a.n_sb,
                                a.n_grp,
                                fault,
                                y,
                            )?;
                        }
                    )? $(
                        GgmlType::$rggml => {
                            let prep = self.module.$dtp(cfg)?;
                            self.module.$dt(
                                stream,
                                &prep,
                                a.w,
                                a.q,
                                a.d8,
                                a.sel,
                                a.n_experts,
                                a.rows_per_expert,
                                a.n_slots,
                                a.m_cols,
                                a.slots_per_col,
                                a.n_sb,
                                a.n_grp,
                                fault,
                                y,
                            )?;
                        }
                    )? )*
                    other => {
                        return Err(GpuError::shape(
                            "launch_raw_down",
                            format!("no raw-block down entry for {other:?}"),
                        ));
                    }
                }
                Ok(())
            }

            $( $(
                /// Enqueue the Walk A down `_sel` of the line's type over
                /// rows in the file's block layout: slot `s` writes `y[s ·
                /// rows_per_expert ..][..rows_per_expert]` as the rows of
                /// expert `sel[s]` dotted with column `s` of `a.act`
                /// ([`crate::kquant::sel`] for the host tier's slot and ids past the
                /// stack, which raise on `fault`). A non-routed down is a
                /// one-expert stack with every id 0. Asynchronous,
                /// allocation-free, capturable.
                pub fn $gve(
                    &self,
                    stream: &CudaStream,
                    a: &SelDown<'_>,
                    fault: FaultSink,
                    y: &mut DeviceBuffer<f32>,
                ) -> Result<(), GpuError> {
                    let fmt = Fmt { name: stringify!($wggml), words: $ww };
                    let d = down_dims(stringify!($gve), fmt, a, y.len())?;
                    let prep = self
                        .module
                        .$gvp(LaunchConfig1D::new(d.grid, THREADS, 0))?;
                    let (q, s8, d8) = walk_a_planes(a.act);
                    self.module.$gv(
                        stream,
                        &prep,
                        a.w.buf(),
                        q,
                        s8,
                        d8,
                        a.sel,
                        d.n_experts,
                        d.rows_per_expert,
                        d.n_slots,
                        d.n_sb,
                        d.iters,
                        fault,
                        y,
                    )?;
                    Ok(())
                }
            )? )*

            $( $(
                /// Enqueue the Walk A gate·up of the line's type over rows in
                /// the file's block layout: slot `s` writes `h[s ·
                /// rows_per_expert ..][..rows_per_expert]` as `a.rule` over
                /// the rows of expert `sel[s]` of both stacks dotted with
                /// column `s / slots_per_col` of `a.act`
                /// ([`crate::kquant::sel`] for the host tier's slot and ids past the
                /// stack, which raise on `fault` and write NaN). A
                /// non-routed gate·up is a one-expert stack with every id 0.
                /// Asynchronous, allocation-free, capturable.
                pub fn $wgue(
                    &self,
                    stream: &CudaStream,
                    a: &GateUpAct<'_>,
                    fault: FaultSink,
                    h: &mut DeviceBuffer<f32>,
                ) -> Result<(), GpuError> {
                    let fmt = Fmt { name: stringify!($wggml), words: $ww };
                    let d = gate_up_dims(stringify!($wgue), fmt, a, h.len())?;
                    let prep = self
                        .module
                        .$wgup(LaunchConfig1D::new(d.grid, THREADS, 0))?;
                    let (q, s8, d8) = walk_a_planes(a.act);
                    self.module.$wgu(
                        stream,
                        &prep,
                        a.wg.buf(),
                        a.wu.buf(),
                        q,
                        s8,
                        d8,
                        a.sel,
                        d.n_experts,
                        d.rows_per_expert,
                        d.n_slots,
                        d.m_cols,
                        d.slots_per_col,
                        d.n_sb,
                        d.iters,
                        d.act,
                        d.limit,
                        fault,
                        h,
                    )?;
                    Ok(())
                }
            )? )*

            $( $(
                /// Enqueue the Walk A down arena entry of the line's type: as
                /// the stack down, slot `s`'s expert the down part at its
                /// offset in `a.arena` (`ARENA_SKIP` skips the slot, a part
                /// past the arena raises on `fault`). Asynchronous,
                /// allocation-free, capturable.
                pub fn $gvae(
                    &self,
                    stream: &CudaStream,
                    a: &SelDownAt<'_>,
                    fault: FaultSink,
                    y: &mut DeviceBuffer<f32>,
                ) -> Result<(), GpuError> {
                    let d = down_at_dims(stringify!($gvae), a, y.len())?;
                    let prep = self
                        .module
                        .$gvap(LaunchConfig1D::new(d.grid, THREADS, 0))?;
                    let (q, s8, d8) = walk_a_planes(a.act);
                    self.module.$gva(
                        stream,
                        &prep,
                        a.arena,
                        a.offs,
                        q,
                        s8,
                        d8,
                        d.rows_per_expert,
                        d.n_slots,
                        d.n_sb,
                        d.iters,
                        fault,
                        y,
                    )?;
                    Ok(())
                }
            )? )*

            $( $(
                /// Enqueue the Walk A gate·up arena entry of the line's type:
                /// as the stack gate·up, slot `s`'s expert the gate and up
                /// parts at its offsets in `a.arena` (a part past the arena
                /// raises on `fault` and writes NaN). Asynchronous,
                /// allocation-free, capturable.
                pub fn $wguae(
                    &self,
                    stream: &CudaStream,
                    a: &GateUpAt<'_>,
                    fault: FaultSink,
                    h: &mut DeviceBuffer<f32>,
                ) -> Result<(), GpuError> {
                    let d = gate_up_at_dims(stringify!($wguae), a, h.len())?;
                    let prep = self
                        .module
                        .$wguap(LaunchConfig1D::new(d.grid, THREADS, 0))?;
                    let (q, s8, d8) = walk_a_planes(a.act);
                    self.module.$wgua(
                        stream,
                        &prep,
                        a.arena,
                        a.offs,
                        q,
                        s8,
                        d8,
                        d.rows_per_expert,
                        d.n_slots,
                        d.m_cols,
                        d.slots_per_col,
                        d.n_sb,
                        d.iters,
                        d.act,
                        d.limit,
                        fault,
                        h,
                    )?;
                    Ok(())
                }
            )? )*
        }
    };
}

quant_sel_module! {
    module: quant_sel,
    owner: QuantSelKernels,
    hand: blk32_kernels,
    raw {
        Iq2Xxs IQ2_XXS bb 66 {
            gate_up_tab: iq2_xxs_gate_up_sel, prepare_iq2_xxs_gate_up_sel, Iq2XxsTab, IQ2XXS_TAB_WORDS,
            down_tab: iq2_xxs_gemv_sel, prepare_iq2_xxs_gemv_sel, Iq2XxsTab, IQ2XXS_TAB_WORDS,
        }
        Iq2Xs IQ2_XS bb 74 {
            gate_up: iq2_xs_gate_up_sel, prepare_iq2_xs_gate_up_sel,
            down: iq2_xs_gemv_sel, prepare_iq2_xs_gemv_sel,
        }
        Iq2S IQ2_S bb 82 {
            gate_up: iq2_s_gate_up_sel, prepare_iq2_s_gate_up_sel,
            down: iq2_s_gemv_sel, prepare_iq2_s_gemv_sel,
        }
        Iq1S IQ1_S bb 50 {
            gate_up: iq1_s_gate_up_sel, prepare_iq1_s_gate_up_sel,
            down: iq1_s_gemv_sel, prepare_iq1_s_gemv_sel,
        }
        Iq1M IQ1_M bb 56 {
            gate_up: iq1_m_gate_up_sel, prepare_iq1_m_gate_up_sel,
            down: iq1_m_gemv_sel, prepare_iq1_m_gemv_sel,
        }
        Iq3Xxs IQ3_XXS bb 98 {
            gate_up: iq3_xxs_gate_up_sel, prepare_iq3_xxs_gate_up_sel,
            down: iq3_xxs_gemv_sel, prepare_iq3_xxs_gemv_sel,
        }
        Iq3S IQ3_S bb 110 {
            gate_up: iq3_s_gate_up_sel, prepare_iq3_s_gate_up_sel,
            down: iq3_s_gemv_sel, prepare_iq3_s_gemv_sel,
        }
        Iq4Xs IQ4_XS bb 136 {
            gate_up: iq4_xs_gate_up_sel, prepare_iq4_xs_gate_up_sel,
            down: iq4_xs_gemv_sel, prepare_iq4_xs_gemv_sel,
        }
    }
    walk {
        Q4k Q4_K words 36 {
            walk_gate_up: kq_gate_up_act_q4k, prepare_kq_gate_up_act_q4k, enqueue_gate_up_q4k,
            walk_gate_up_at: kq_gate_up_act_q4k_at, prepare_kq_gate_up_act_q4k_at, enqueue_gate_up_q4k_at,
        }
        Q5k Q5_K words 44 {
            walk_gemv: q5k_gemv_sel, prepare_q5k_gemv_sel, enqueue_gemv_q5k_sel,
            walk_gate_up: kq_gate_up_act_q5k, prepare_kq_gate_up_act_q5k, enqueue_gate_up_q5k,
            walk_gemv_at: q5k_gemv_sel_at, prepare_q5k_gemv_sel_at, enqueue_gemv_q5k_sel_at,
            walk_gate_up_at: kq_gate_up_act_q5k_at, prepare_kq_gate_up_act_q5k_at, enqueue_gate_up_q5k_at,
        }
        Q8_0 Q8_0 words 68 {
            walk_gemv: q8_0_gemv_sel, prepare_q8_0_gemv_sel, enqueue_gemv_q8_0_sel,
            walk_gate_up: kq_gate_up_act_q8_0, prepare_kq_gate_up_act_q8_0, enqueue_gate_up_q8_0,
        }
    }
}

// The arena entries' contracts spell the table's parts a slot as 3.
const _: () = assert!(ARENA_PARTS == 3);
