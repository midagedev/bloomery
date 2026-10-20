//! The MoE router bodies the sigmoid and √softplus instances share: the three
//! kernel bodies of a router over `E` experts, `U` slots and a [`Score`], and
//! the two stages they are built from. A device crate's `#[kernel]` entry is a
//! thin instance of one: it declares the shared arrays at its size (a `static`
//! inside a generic function cannot take its size from the function's
//! parameters), hands them in, and keeps its own launch contract.
//!
//! - [`router_body`]: the logit gemv, the score, the block ticket and, in the
//!   block that draws the last one, the selection; one token a launch.
//! - [`pick_body`]: a batch's selection, one block a token over scores the
//!   pass's tile GEMM and its sigmoid wrote; bit for bit what
//!   [`router_body`] writes a token.
//!
//! [`sigmoid_router_instance!`](crate::sigmoid_router_instance) is the one
//! form of a sigmoid instance of the three: its entries, its output buffers
//! and its launchers at one expert and slot count.
//!
//! The bodies are `#[inline(always)]` and each is whole: on this
//! toolchain a function boundary around the selection loop changes the
//! instruction stream (predicated selects become branches), so the loop, its
//! butterfly and the weight stage stay inside their bodies. The score and the
//! tie rule come from [`crate::route_core`]. Every shape an instance asserts
//! holds when `E` is a multiple of [`ROWS_PER_BLOCK`] and `⌈E / 32⌉ <= 32`. Where
//! `E` is not a multiple of 32 the selecting lanes own `⌈E / 32⌉` or one fewer
//! experts; the guard that skips the missing ones is a compile-time `true`
//! at a multiple of 32, so those instances' instructions are the guard-free
//! ones.

use crate::q8f32::f32_lane_partial_1col_w32;
use crate::route_core::{Score, renorm_divisor, take};
use crate::{FaultSink, FaultSite};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, thread, threadfence, warp};

/// Threads per router block, four warps, one expert row per warp: 96 blocks
/// at 384 experts, so every SM of the card holds a row's walk.
pub const ROUTER_THREADS: usize = 128;
/// [`ROUTER_THREADS`] as a launch argument.
pub const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
/// Expert rows a block walks, one a warp.
pub const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
// The entries' `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(ROUTER_THREADS == 128);

/// V4.1's `publish` over `E` experts: thread `tid` parks `score[e] = probs[at +
/// e]` (a volatile load) and `sel_v[e] = score[e] + bias[e]` for `e = tid +
/// ROUTER_THREADS · i`, raising [`FaultSite::Router`] on a selection value
/// that is not finite.
///
/// # Safety
///
/// `probs` addresses at least `at + E` values, `bias` holds `E`,
/// `score` and `sel_v` are the block's shared arrays of `E`, and the caller's
/// barrier follows before any read of them.
#[inline(always)]
pub unsafe fn publish_n<const E: usize>(
    probs: *const f32,
    bias: &[f32],
    at: usize,
    score: *mut f32,
    sel_v: *mut f32,
    tid: usize,
    fault: FaultSink,
) {
    let mut e = tid;
    while e < E {
        // SAFETY: at + e < at + E, inside probs by this fn's contract.
        let p = unsafe { core::ptr::read_volatile(probs.add(at + e)) };
        // SAFETY: e < E <= bias.len() by this fn's contract.
        let v = p + unsafe { *bias.get_unchecked(e) };
        if !v.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: e < E; thread `tid` writes entries tid + ROUTER_THREADS·i
        // only, before the caller's barrier.
        unsafe {
            *score.add(e) = p;
            *sel_v.add(e) = v;
        }
        e += ROUTER_THREADS;
    }
}

/// V4.1's `select` over `E` experts and `U` slots: `U` rounds of the butterfly
/// argmax under [`take`]`::<true>`, lane `L` scanning experts `L + 32 j < E`
/// (`j < ⌈E / 32⌉`) no earlier round took; lane 0 writes each round's id to
/// `ids[at + s]`, then sums the `U` scores in f64 in slot order, takes
/// [`renorm_divisor`] of the sum narrowed to f32, and writes each weight
/// `score / sum · scale` to `weights[at + s]`, raising [`FaultSite::Router`]
/// on one that is not finite.
///
/// # Safety
///
/// all 32 lanes of warp 0 call it; `⌈E / 32⌉ <= 32`; `score` and
/// `sel_v` are shared arrays of `E` values published by a barrier, `slot_p`
/// one of `U` that lane 0 alone uses; `ids` and `weights` hold `at + U`
/// values, of which the calling block is the only writer of `at ..`.
#[allow(
    clippy::too_many_arguments,
    reason = "the selecting block's shared arrays, its lane and scale, the fault and the outputs"
)]
#[inline(always)]
pub unsafe fn select_n<const E: usize, const U: usize>(
    score: *const f32,
    sel_v: *const f32,
    slot_p: *mut f32,
    lane: usize,
    scale: f32,
    fault: FaultSink,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    at: usize,
) {
    let mut taken = 0u32;
    let mut s = 0usize;
    while s < U {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < const { E.div_ceil(32) } {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // At a multiple of 32 every `ej` is below `E`: the guard is a
                // compile-time `true` and leaves no instruction.
                if const { E.is_multiple_of(32) } || ej < E {
                    // SAFETY: ej < E (lane < 32 and j < E / 32 at a multiple
                    // of 32, the guard otherwise), published by the caller's
                    // barrier.
                    let v = unsafe { *sel_v.add(ej) };
                    if take::<true>(v, ej as u32, bv, bi) {
                        bv = v;
                        bi = ej as u32;
                    }
                }
            }
            j += 1;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
            if take::<true>(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
            off >>= 1;
        }
        // Every lane leaves the butterfly with the same winner; its owner
        // marks it taken.
        if bi as usize % 32 == lane {
            taken |= 1 << (bi as usize / 32);
        }
        if lane == 0 {
            // SAFETY: bi < E (every candidate is one of a lane's own
            // entries); s < U; lane 0 alone writes and reads slot_p; at + s <
            // at + U is inside ids by this fn's contract.
            unsafe {
                *slot_p.add(s) = *score.add(bi as usize);
                *ids.get_unchecked_mut(at + s) = bi;
            }
        }
        s += 1;
    }
    if lane != 0 {
        return;
    }
    let mut sum = 0.0f64;
    let mut s = 0usize;
    while s < U {
        // SAFETY: s < U, this lane's own write.
        sum += f64::from(unsafe { *slot_p.add(s) });
        s += 1;
    }
    let sum = renorm_divisor(sum as f32);
    let mut s = 0usize;
    while s < U {
        // SAFETY: s < U, this lane's own write.
        let wt = unsafe { *slot_p.add(s) } / sum * scale;
        if !wt.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: at + s < at + U is inside weights by this fn's contract;
        // lane 0 is the only writer.
        unsafe { *weights.get_unchecked_mut(at + s) = wt };
        s += 1;
    }
}

/// `ds41_router`'s body (`gpu-deepseek41`) over `E` experts, `U` slots and the score `S`: the
/// row walk (`f32_gemv`'s m = 1 row, the logit into `logits`, `S::score` of
/// it into `probs`), the fence and the block's ticket, and in the block that
/// draws the last ticket [`publish_n`] and [`select_n`] into `ids` and
/// `weights` from 0, then the ticket count back to zero.
///
/// # Safety
///
/// the grid is `E / ROWS_PER_BLOCK` blocks of [`ROUTER_THREADS`]
/// (the launcher's), every thread arrives converged; `n_expert == E`, `w`
/// holds `n_expert · k` values, `x` `k`, `bias`, `logits` and `probs`
/// `n_expert`, `ids` and `weights` `U`, `done` one, zero before the launch;
/// `k` is a positive multiple of 32; `last`, `score`, `sel_v` and `slot_p`
/// are this block's own shared arrays of 1, `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn router_body<S: Score, const E: usize, const U: usize>(
    w: &[f32],
    x: &[f32],
    bias: &[f32],
    n_expert: u32,
    k: u32,
    scale: f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
    last: *mut u32,
    score: *mut f32,
    sel_v: *mut f32,
    slot_p: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
    if row < n_expert as usize {
        // `f32_gemv`'s m = 1 row: the same sum order and the same tree.
        // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
        // k, and x.len() >= k, by this fn's contract; k is a positive
        // multiple of 32; lane < 32.
        let partial = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
        let logit = warp::reduce_sum_f32(partial);
        if lane == 0 {
            // SAFETY: row < n_expert <= logits.len() by this fn's contract;
            // lane 0 of the row's warp is the slot's only writer.
            unsafe { *logits.get_unchecked_mut(row) = logit };
            // SAFETY: row < n_expert <= probs.len() by this fn's contract;
            // lane 0 of the row's warp is the slot's only writer.
            unsafe { *probs.get_unchecked_mut(row) = S::score(logit) };
        }
    }
    threadfence();
    thread::sync_threads();

    let ticket = done.as_mut_ptr();
    if tid == 0 {
        // SAFETY: `ticket` is done[0], inside `done` by this fn's contract;
        // every access to it is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
        let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
        // SAFETY: block-shared, one element, written before the barrier that
        // publishes it.
        unsafe { *last = u32::from(is_last) };
    }
    thread::sync_threads();
    // SAFETY: block-shared, one element, written by thread 0 before the
    // barrier above.
    if unsafe { *last } == 0 {
        return;
    }

    // The last block: every block's rows are published.
    // SAFETY: probs holds n_expert = E scores by this fn's contract; score,
    // sel_v and slot_p are this block's shared arrays of E, E and U entries.
    unsafe { publish_n::<E>(probs.as_mut_ptr(), bias, 0, score, sel_v, tid, fault) };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; ids and
        // weights hold U entries from 0 by this fn's contract.
        unsafe { select_n::<E, U>(score, sel_v, slot_p, lane, scale, fault, ids, weights, 0) };
        if lane == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by this fn's
            // contract; every block has drawn its ticket by now.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            count.store(0, AtomicOrdering::Relaxed);
        }
    }
}

/// `ds41_router_pick`'s body over `E` experts and `U` slots: token
/// `blockIdx.x`'s scores (`probs[token · E ..]`) through [`publish_n`] and
/// [`select_n`] into `ids` and `weights` at `U · token` — [`router_body`]'s
/// last block on those scores.
///
/// # Safety
///
/// the block is [`ROUTER_THREADS`] threads, converged; `n_expert ==
/// E` (else the block returns), `probs` holds `n_expert · t` values, `bias`
/// `n_expert`, `ids` and `weights` `U · t`; `score`, `sel_v` and `slot_p` are
/// this block's own shared arrays of `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn pick_body<const E: usize, const U: usize>(
    probs: &[f32],
    bias: &[f32],
    n_expert: u32,
    t: u32,
    scale: f32,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
    score: *mut f32,
    sel_v: *mut f32,
    slot_p: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let token = thread::blockIdx_x() as usize;
    if token >= t as usize || n_expert as usize != E {
        return;
    }
    // SAFETY: token < t, so the token's E scores are inside probs
    // (probs.len() >= n_expert * t by this fn's contract).
    unsafe { publish_n::<E>(probs.as_ptr(), bias, token * E, score, sel_v, tid, fault) };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; token < t, so
        // ids and weights hold U entries from U · token by this fn's
        // contract.
        unsafe {
            select_n::<E, U>(
                score,
                sel_v,
                slot_p,
                lane,
                scale,
                fault,
                ids,
                weights,
                token * U,
            )
        };
    }
}

/// One sigmoid router instance, from the generic bodies above: a module
/// holding `N_EXPERT` and `N_USED`, the three `#[cuda_module]` entries
/// (`$router`, `$scores`, `$pick`), `RouterOut` and `RouterKernels` at those
/// counts, compiled into the crate that invokes it (`gpu-deepseek41`'s
/// `glm5next` and `mimo2`; Bolt's five, `arch/nemotron_h_moe/router.rs`). The expert and slot counts are tokens, not constants, so the
/// entries' `launch_contract` literals and the module's constants are one
/// source. The entry names are the PTX entry names, and `$p_router`,
/// `$p_scores` and `$p_pick` the `prepare_*` launchers `#[cuda_module]`
/// derives from them: a macro cannot join identifiers.
#[macro_export]
macro_rules! sigmoid_router_instance {
    (
        $(#[$meta:meta])*
        pub mod $m:ident {
            experts: $e:tt,
            used: $u:tt,
            kernels: $kernels:ident,
            entries: $router:ident, $pick:ident,
            prepare: $p_router:ident, $p_pick:ident $(,)?
        }
    ) => {
        $(#[$meta])*
        pub mod $m {

        use $crate::gemm::Gemm32Kernels;
        use $crate::route_body::{
            ROUTER_THREADS, ROUTER_THREADS_U32, ROWS_PER_BLOCK, pick_body, router_body,
        };
        use $crate::route_core::Sigmoid;
        use $crate::{DeviceTensor, FaultSink, Gpu, GpuError, launch_u32};
        use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
        use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
        use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract};
        use cuda_host::cuda_module;
        use std::sync::Arc;

        /// Experts the router scores (`expert_count`).
        pub const N_EXPERT: usize = $e;
        /// Experts each token routes to (`expert_used_count`).
        pub const N_USED: usize = $u;

        /// The instance's row in `models::shape::ROUTERS`: a shape read from
        /// a file selects it through `select_router`.
        pub const ROW: RouterInst =
            shape::router_row(RouterBody::BiasedSigmoid, N_EXPERT as u32);
        const _: () = assert!(shape::router_row_is(
            ROW,
            rules::BIASED_SIGMOID,
            false,
            (N_USED as u32, N_USED as u32)
        ));

        const _: () = assert!(N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
        // The selecting lane keeps its taken entries as bits of one u32.
        const _: () = assert!(N_EXPERT.div_ceil(32) <= 32);
        // The entries' `launch_bounds` and `launch_contract` block are literals; the expert and slot counts in the contracts are the instance's own tokens.
        const _: () = assert!(ROUTER_THREADS == 128);

        #[allow(
            dead_code,
            reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
        )]
        #[cuda_module]
        mod $kernels {
            use super::*;

            /// The router for one token (`router_body` at [`N_EXPERT`],
            /// [`N_USED`], sigmoid): `n_expert` rows of `k` f32 in `w`, the
            /// token's `k` activations in `x`, the selection bias in `bias`.
            /// Writes `logits[e]` and `probs[e]` (the score) for every expert and
            /// `ids[s]`/`weights[s]` for the slots in rank order. `done[0]` is
            /// the block ticket count: zero before the launch, zero again after
            /// it. Grid `n_expert / 4` blocks of 128, warp `r % 4` of block `r /
            /// 4` owning row `r`.
            #[allow(
                clippy::too_many_arguments,
                reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
            )]
            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(
                domain = 1,
                block = (128, 1, 1),
                requires = (
                    n_expert == $e,
                    w.len() >= n_expert * k,
                    x.len() >= k,
                    bias.len() >= n_expert,
                    logits.len() >= n_expert,
                    probs.len() >= n_expert,
                    ids.len() >= $u,
                    weights.len() >= $u,
                    done.len() >= 1
                )
            )]
            pub fn $router(
                w: &[f32],
                x: &[f32],
                bias: &[f32],
                n_expert: u32,
                k: u32,
                scale: f32,
                mut logits: DisjointSlice<f32>,
                mut probs: DisjointSlice<f32>,
                mut ids: DisjointSlice<u32>,
                mut weights: DisjointSlice<f32>,
                mut done: DisjointSlice<u32>,
                fault: FaultSink,
            ) {
                static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
                static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
                static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
                static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;
                // SAFETY: SCORE, SEL_V, LAST and SLOT_P are this block's own
                // shared arrays of N_EXPERT, N_EXPERT, 1 and N_USED entries (the
                // raw form reaches each `static mut` without a reference); the
                // grid is N_EXPERT / 4 blocks of 128 (the launcher's), every
                // thread arrives converged; the lengths are the launch
                // contract's, the launcher passes k a positive multiple of 32,
                // and `done[0]` is zero before the launch.
                unsafe {
                    router_body::<Sigmoid, N_EXPERT, N_USED>(
                        w,
                        x,
                        bias,
                        n_expert,
                        k,
                        scale,
                        &mut logits,
                        &mut probs,
                        &mut ids,
                        &mut weights,
                        &mut done,
                        fault,
                        SharedArray::as_raw_mut_ptr(&raw mut LAST),
                        SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                        SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                        SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                    );
                }
            }

            /// The selection of token `blockIdx.x` of a batch from its scores
            /// (`probs`, [`N_EXPERT`] a token, as the scores
            /// entry writes them) and `bias` into `ids` and `weights` at `N_USED ·
            /// token` (`pick_body`): the router entry's last block on those
            /// scores.
            #[allow(
                clippy::too_many_arguments,
                reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
            )]
            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(
                domain = 1,
                block = (128, 1, 1),
                requires = (
                    n_expert == $e,
                    probs.len() >= n_expert * t,
                    bias.len() >= n_expert,
                    ids.len() >= $u * t,
                    weights.len() >= $u * t
                )
            )]
            pub fn $pick(
                probs: &[f32],
                bias: &[f32],
                n_expert: u32,
                t: u32,
                scale: f32,
                mut ids: DisjointSlice<u32>,
                mut weights: DisjointSlice<f32>,
                fault: FaultSink,
            ) {
                static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
                static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
                static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;
                // SAFETY: SCORE, SEL_V and SLOT_P are this block's own shared
                // arrays of N_EXPERT, N_EXPERT and N_USED entries (the raw form
                // reaches each `static mut` without a reference); the block is
                // 128 threads, converged; the lengths are the launch contract's.
                unsafe {
                    pick_body::<N_EXPERT, N_USED>(
                        probs,
                        bias,
                        n_expert,
                        t,
                        scale,
                        &mut ids,
                        &mut weights,
                        fault,
                        SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                        SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                        SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                    );
                }
            }
        }

        /// Where one [`RouterKernels::enqueue_router`] launch leaves its results:
        /// per expert the logit and the score, per slot the expert id and the
        /// weight, and the block ticket count the launch returns to zero. The
        /// count serves one launch at a time, so launches that share a
        /// `RouterOut` must be ordered on one stream.
        pub struct RouterOut {
            pub logits: DeviceBuffer<f32>,
            pub probs: DeviceBuffer<f32>,
            pub ids: DeviceBuffer<u32>,
            pub weights: DeviceBuffer<f32>,
            done: DeviceBuffer<u32>,
        }

        impl RouterOut {
            /// Allocate the buffers, the ticket count zeroed. Load-time only.
            pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
                Ok(RouterOut {
                    logits: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                    probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                    ids: DeviceBuffer::zeroed(stream, N_USED)?,
                    weights: DeviceBuffer::zeroed(stream, N_USED)?,
                    done: DeviceBuffer::zeroed(stream, 1)?,
                })
            }

            /// The ticket count as it stands: zero between launches.
            pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
                Ok(self.done.to_host_vec(stream)?[0])
            }
        }

        /// The loaded router module. Owns no context and no stream — every
        /// enqueue takes the engine stream, so launches order with the rest of
        /// the step and are capturable.
        pub struct RouterKernels {
            module: $kernels::LoadedModule,
        }

        /// The router weight's shape: [`N_EXPERT`] rows of a positive multiple
        /// of 32.
        fn weight_k(what: &'static str, w: &DeviceTensor<f32>) -> Result<usize, GpuError> {
            let k = w.cols();
            if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
                return Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                        w.rows()
                    ),
                });
            }
            Ok(k)
        }

        impl RouterKernels {
            /// Load this module's device bundle into `ctx`. Load-time only.
            pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
                // SAFETY: this package owns the embedded device bundle produced
                // for the module above; the launcher checks its launch contract.
                let module = unsafe { $crate::shared_module!($kernels, ctx)? };
                Ok(RouterKernels { module })
            }

            /// Enqueue the router for one token: `w` the router weight as f32
            /// ([`N_EXPERT`] rows of `k`, `k` a positive multiple of 32), `x` the
            /// token's `k` activations, `bias` the [`N_EXPERT`] selection biases,
            /// `scale` the file's `expert_weights_scale`; results into `out`. A
            /// selection it cannot make raises `fault`. Asynchronous,
            /// allocation-free, capturable.
            #[allow(
                clippy::too_many_arguments,
                reason = "the launcher's inputs are the kernel's (rust-quality R8)"
            )]
            pub fn enqueue_router(
                &self,
                stream: &CudaStream,
                w: &DeviceTensor<f32>,
                x: &DeviceBuffer<f32>,
                bias: &DeviceBuffer<f32>,
                scale: f32,
                out: &mut RouterOut,
                fault: FaultSink,
            ) -> Result<(), GpuError> {
                let what = concat!(stringify!($m), "::enqueue_router");
                let k = weight_k(what, w)?;
                if x.len() < k || bias.len() < N_EXPERT {
                    return Err(GpuError::Shape {
                        what,
                        detail: format!(
                            "x.len() {} < k {k} or bias.len() {} < {N_EXPERT}",
                            x.len(),
                            bias.len()
                        ),
                    });
                }
                let k = launch_u32(what, "k", k)?;
                let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
                let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
                let prep = self.module.$p_router(LaunchConfig1D::new(
                    grid,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
                let RouterOut {
                    logits,
                    probs,
                    ids,
                    weights,
                    done,
                } = out;
                self.module.$router(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    bias,
                    n_expert,
                    k,
                    scale,
                    logits,
                    probs,
                    ids,
                    weights,
                    done,
                    fault,
                )?;
                Ok(())
            }

            /// Enqueue the router for `t` tokens of a batch in two launches: `x`
            /// their activations, `k` a token and token-major, the per-token
            /// scores into `probs` ([`N_EXPERT`] a token), then each token's ids
            /// and weights into `ids` and `weights` ([`N_USED`] a token) — what
            /// `t` [`RouterKernels::enqueue_router`] launches write, bit for bit.
            /// A selection value or a weight that is not finite raises `fault`.
            /// Asynchronous, allocation-free.
            #[allow(
                clippy::too_many_arguments,
                reason = "enqueue_router's inputs, the token count and the batch's buffers (rust-quality R8)"
            )]
            pub fn enqueue_router_rows(
                &self,
                stream: &CudaStream,
                gpu: &Gpu,
                wide: &Gemm32Kernels,
                w: &DeviceTensor<f32>,
                x: &DeviceBuffer<f32>,
                bias: &DeviceBuffer<f32>,
                scale: f32,
                t: usize,
                probs: &mut DeviceBuffer<f32>,
                ids: &mut DeviceBuffer<u32>,
                weights: &mut DeviceBuffer<f32>,
                fault: FaultSink,
            ) -> Result<(), GpuError> {
                let what = concat!(stringify!($m), "::enqueue_router_rows");
                let k = weight_k(what, w)?;
                if t == 0
                    || x.len() < k * t
                    || bias.len() < N_EXPERT
                    || probs.len() < N_EXPERT * t
                    || ids.len() < N_USED * t
                    || weights.len() < N_USED * t
                {
                    return Err(GpuError::Shape {
                        what,
                        detail: format!(
                            "{t} tokens of {k} (at least one): x {}, bias {}, probs {}, ids {}, weights {}",
                            x.len(),
                            bias.len(),
                            probs.len(),
                            ids.len(),
                            weights.len()
                        ),
                    });
                }
                let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
                let tt = launch_u32(what, "t", t)?;
                // The scores as one F32 tile GEMM over every token — the
                // common `f32_tile_gemm`, bit for bit this body's
                // per-(token, expert) gemv (the same lane walk,
                // `f32_lane_partials`, and the same warp tree,
                // `gemv_lane_sums`) — then the score's scalar over its
                // logits, in place. The pick launch is unchanged.
                wide.enqueue_f32_tile(stream, w, x, t, probs)?;
                gpu.elem()
                    .enqueue_sigmoid_rows(stream, probs, N_EXPERT * t)?;
                let prep = self
                    .module
                    .$p_pick(LaunchConfig1D::new(tt, ROUTER_THREADS_U32, 0))?;
                self.module.$pick(
                    stream, &prep, probs, bias, n_expert, tt, scale, ids, weights, fault,
                )?;
                Ok(())
            }
        }

        /// The instance as the common routed block (`routed_block`) takes it:
        /// its counts, its results' buffers and the two launches.
        impl $crate::routed_block::RouterEntry for RouterKernels {
            const N_EXPERT: usize = N_EXPERT;
            const N_USED: usize = N_USED;
            type Out = RouterOut;

            fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
                RouterKernels::load(ctx)
            }

            fn new_out(stream: &CudaStream) -> Result<RouterOut, GpuError> {
                RouterOut::new(stream)
            }

            fn out_ids(out: &RouterOut) -> &DeviceBuffer<u32> {
                &out.ids
            }

            fn out_weights(out: &RouterOut) -> &DeviceBuffer<f32> {
                &out.weights
            }

            fn out_bytes(out: &RouterOut) -> usize {
                out.logits.num_bytes()
                    + out.probs.num_bytes()
                    + out.ids.num_bytes()
                    + out.weights.num_bytes()
                    + out.done.num_bytes()
            }

            fn route(
                &self,
                stream: &CudaStream,
                x: &DeviceBuffer<f32>,
                r: &$crate::routed_block::RouteIn<'_>,
                out: &mut RouterOut,
            ) -> Result<(), GpuError> {
                self.enqueue_router(stream, r.w, x, r.bias, r.scale, out, r.fault)
            }

            fn route_rows(
                &self,
                (stream, gpu, wide): (&CudaStream, &Gpu, &Gemm32Kernels),
                x: &DeviceBuffer<f32>,
                r: &$crate::routed_block::RouteIn<'_>,
                t: usize,
                rows: $crate::routed_block::RoutedRows<'_>,
            ) -> Result<(), GpuError> {
                self.enqueue_router_rows(
                    stream,
                    gpu,
                    wide,
                    r.w,
                    x,
                    r.bias,
                    r.scale,
                    t,
                    rows.probs,
                    rows.ids,
                    rows.weights,
                    r.fault,
                )
            }
        }
        }
    };
}
