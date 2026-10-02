//! The recurrent delta step of a linear-attention layer over `m` tokens,
//! one launch per call: decode is `m = 1`, a prompt runs its tokens through
//! the same loop with the state held in registers, read once and written
//! once, in place.
//!
//! The state has `lanes` lanes (a load parameter); the lane a call reads is
//! the device word `lane[lane_at]` (the body's parameter image), never a
//! launch argument, so one captured graph serves every step. With one lane
//! the call writes the state after its last token back into the lane it
//! read. A word at or past `lanes` raises [`FaultSite::DeltaLane`] and the
//! call reads and writes lane `word mod lanes`. `gdn_delta` and `kda_delta`
//! take one lane; the launcher refuses more by name.
//!
//! `gdn_delta_lanes` is `gdn_delta` over `lanes` lanes with a stamp a lane —
//! the position the lane's state stands at, the count of tokens it has
//! absorbed. It reads lane `c = word mod lanes`; block 0's thread 0 raises
//! [`FaultSite::DeltaStamp`] when `stamp[c]` is not the call's first
//! position `pos[0]`, so a lane never written (its stamp [`NEVER`]) or one a
//! commit did not leave there is never read as a state. In place mode it
//! writes the state after its last token back into lane `c`, stamped
//! `pos[0] + m`; in row mode (a verify, `m <= lanes`) it writes the state
//! after token `j` into lane `(c + j) mod lanes`, stamped `pos[0] + j + 1`,
//! so a host that keeps `k` of the rows moves the word to `(c + k − 1) mod
//! lanes` and copies nothing. Each token's `o` and each state are the bits
//! `gdn_delta` writes: the loop is its loop, the stores placed per row.
//!
//! `kda_delta_lanes` is `kda_delta` over stamped lanes by the same rule, with
//! one more launch value, the row base `row`: the call's first token is row
//! `row` of a verify whose rows before it an earlier launch of the same verify
//! ran, one launch a row. It reads the lane the row before it wrote — lane
//! [`read_lane`]`(c, row)`, `c` itself for row 0 — checks that lane's stamp
//! against `pos[0]`, and in row mode writes the state after token `j` into
//! lane [`row_lane`]`(c, row + j)`: the lanes a one-launch verify of the same
//! rows writes, so the commit rule is the one above. A verify of one token a
//! row writes row 0 in place and row `r` into lane `c + r`; the one-token step
//! and a prompt batch are row 0 in place. In place mode takes row 0 only, and
//! row mode `row + m <= lanes`; the launcher refuses anything else by name.
//!
//! Geometry: a warp owns four value columns of one head, eight lanes per
//! column; lane `j` of a column holds its [`KEYS_PER_LANE`] keys
//! `32·r + 4·j + c` (`r, c = 0..4`, in `(r, c)` order). Four warps per block,
//! eight blocks per head, `8·n_v` blocks. There is no shared memory and no
//! barrier: a column's two dot products are a lane's own sums, then a
//! butterfly over its eight lanes.
//!
//! Two entries share each body: `gdn_delta` and `gdn_delta_lanes`, one decay
//! per value head (Gated DeltaNet), and `kda_delta` and `kda_delta_lanes`,
//! one per (value head, key channel) (Kimi Delta Attention), read
//! `[m][n_v][HEAD]` by the lane's own sixteen keys.
//!
//! Numeric contract (the host rules [`delta_host`] and [`kda_delta_host`]
//! are this list), per token and column:
//! - `S'ᵢ = decayᵢ·Sᵢ`, each rounded (`decayᵢ` the head's one decay, or key
//!   `i`'s);
//! - a dot `Σᵢ Aᵢ·Bᵢ` (first `S'ᵀk`, then `Sᵀq`): lane `j` forms four
//!   partials `p_c = A(0,c)·B(0,c)` rounded, then `p_c = fma(A(r,c), B(r,c),
//!   p_c)` for `r = 1, 2, 3`, and its sum `(p₀ + p₁) + (p₂ + p₃)`; the eight
//!   lane sums by the xor butterfly (4, 2, 1);
//! - `u = (v − S'ᵀk)·β`, each rounded;
//! - `Sᵢ = fma(kᵢ, u, S'ᵢ)`;
//! - `o = Sᵀq` with the conv's `q` (already scaled by `1/√HEAD`).
//!
//! No silent failure: the kernel checks one value per token and column, `o`.
//! A non-finite input reaches it (a NaN or infinite k, v, β or decay makes
//! `u` or `S'` non-finite, and a non-finite q makes the dot so), and so does
//! any state element that stops being finite: `Sᵢ·qᵢ` is infinite or NaN
//! whatever `qᵢ` is (`∞·0` is NaN). A non-finite `o` raises
//! [`FaultSite::LinearDelta`] once per lane and is written as it is.

use super::{BLOCK, DECAY_HEAD, DECAY_KEY, HEAD, LinearShape};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, ptx_asm, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Lanes that share one value column.
pub const LANES_PER_COLUMN: usize = 8;

/// Keys each lane holds: `HEAD / LANES_PER_COLUMN`.
pub const KEYS_PER_LANE: usize = HEAD / LANES_PER_COLUMN;

/// Value columns per block: four warps of four columns.
const COLUMNS_PER_BLOCK: usize = 16;

const _: () =
    assert!(HEAD == 128 && KEYS_PER_LANE == 16 && BLOCK as usize == 32 * COLUMNS_PER_BLOCK / 4);

/// Key index of lane `j`'s value `4·r + c` of a column: `32·r + 4·j + c`.
#[inline(always)]
#[must_use]
pub fn key_of(j: usize, i: usize) -> usize {
    32 * (i / 4) + 4 * j + (i % 4)
}

/// One lane's sum of `Σ a[i]·b[i]` over its sixteen values (module doc).
#[inline(always)]
fn lane_dot(a: [f32; KEYS_PER_LANE], b: [f32; KEYS_PER_LANE]) -> f32 {
    let mut p = [
        mul_rn_f32(a[0], b[0]),
        mul_rn_f32(a[1], b[1]),
        mul_rn_f32(a[2], b[2]),
        mul_rn_f32(a[3], b[3]),
    ];
    for i in 4..KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        p[i % 4] = fma_rn_f32(a[i], b[i], p[i % 4]);
    }
    (p[0] + p[1]) + (p[2] + p[3])
}

/// The global-space address of `p`, for [`ld4`].
#[inline(always)]
fn global_addr(p: *const f32) -> u64 {
    let g: u64;
    // SAFETY: a register-only address conversion; nothing is dereferenced.
    unsafe {
        ptx_asm!(
            "cvta.to.global.u64 %0, %1;",
            out("=l") g,
            in("l") p,
            options(register_only),
        );
    }
    g
}

/// Four f32 at global address `g` in one 128-bit load: the load width the
/// lane's four consecutive keys allow, where four scalar reads cost four
/// instructions. `NC` takes the read-only data path, for memory no thread
/// writes while the kernel runs; the state, which the kernel writes back in
/// place, takes the coherent one.
///
/// # Safety
///
/// `g` is 16-byte aligned and addresses four f32 of global memory; with
/// `NC`, memory no thread writes while the kernel runs.
#[inline(always)]
unsafe fn ld4<const NC: bool>(g: u64) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    // SAFETY: the caller's contract.
    unsafe {
        if NC {
            ptx_asm!(
                "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];",
                out("=f") a,
                out("=f") b,
                out("=f") c,
                out("=f") d,
                in("l") g,
            );
        } else {
            ptx_asm!(
                "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];",
                out("=f") a,
                out("=f") b,
                out("=f") c,
                out("=f") d,
                in("l") g,
            );
        }
    }
    [a, b, c, d]
}

/// Lane `j`'s sixteen keys of the `HEAD`-value vector at global address
/// `g`: four 128-bit loads at `32·r + 4·j`, through [`ld4::<NC>`].
///
/// # Safety
///
/// `g` is 16-byte aligned and addresses `HEAD` f32 of global memory, with
/// `NC` memory no thread writes while the kernel runs; `j <
/// LANES_PER_COLUMN`.
#[inline(always)]
unsafe fn lane_keys<const NC: bool>(g: u64, j: usize) -> [f32; KEYS_PER_LANE] {
    let mut v = [0.0f32; KEYS_PER_LANE];
    for r in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: 32·r + 4·j + 3 < HEAD and a multiple of four floats from
        // an aligned base, by the fn's contract.
        let w = unsafe { ld4::<NC>(g + 4 * (32 * r + 4 * j) as u64) };
        v[4 * r] = w[0];
        v[4 * r + 1] = w[1];
        v[4 * r + 2] = w[2];
        v[4 * r + 3] = w[3];
    }
    v
}

/// The eight lanes of a column summed by the xor butterfly; every lane of
/// the column ends with the same bits.
#[inline(always)]
fn column_sum(x: f32) -> f32 {
    let mut s = x;
    s += warp::shuffle_xor_f32(s, 4);
    s += warp::shuffle_xor_f32(s, 2);
    s += warp::shuffle_xor_f32(s, 1);
    s
}

/// The loop of one block, for either decay granularity (`DECAY` is
/// [`DECAY_HEAD`], `gdn_delta`, or [`DECAY_KEY`], `kda_delta`).
///
/// # Safety
///
/// The slices hold the lengths `gdn_delta`'s launch contract names, with
/// `decay` `m·n_v` long for [`DECAY_HEAD`] and `m·n_v·HEAD` for
/// [`DECAY_KEY`]; `o` and `state` address `m·n_v·HEAD` and
/// `lanes·n_v·HEAD·HEAD` writable f32s that no other launch touches; `lane`
/// holds word `lane_at`; `lanes >= 1`; `n_k >= 1` divides `n_v`; the block is
/// 128 threads and the grid `8·n_v` blocks.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn delta_body<const DECAY: u32>(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    lane_word: &[u32],
    lane_at: u32,
    lanes: u32,
    n_k: u32,
    n_v: u32,
    grouped: u32,
    m: u32,
    fault: FaultSink,
    o: *mut f32,
    state: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let j = lane % LANES_PER_COLUMN;
    let b = thread::blockIdx_x() as usize;
    let blocks_per_head = HEAD / COLUMNS_PER_BLOCK;
    let h = b / blocks_per_head;
    let (n_k, n_v, m) = (n_k as usize, n_v as usize, m as usize);
    if h >= n_v {
        return; // block-uniform
    }
    let col = (b % blocks_per_head) * COLUMNS_PER_BLOCK + (tid / 32) * 4 + lane / LANES_PER_COLUMN;
    let kh = if grouped != 0 {
        h / (n_v / n_k)
    } else {
        h % n_k
    };
    let ch = (2 * n_k + n_v) * HEAD;
    let q_at = kh * HEAD;
    let k_at = (n_k + kh) * HEAD;
    let v_at = 2 * n_k * HEAD + h * HEAD + col;
    // SAFETY: lane_at < lane_word.len() by the contract.
    let word = unsafe { *lane_word.get_unchecked(lane_at as usize) };
    if word >= lanes && tid == 0 {
        fault.raise(FaultSite::DeltaLane);
    }
    let s_at = (((word % lanes) as usize * n_v + h) * HEAD + col) * HEAD;

    // SAFETY: the rows read below are inside qkv (m·ch values, which no
    // launch writes while this one runs) and state (lanes·n_v·HEAD·HEAD, as
    // `word % lanes < lanes`) by the contract, 16-byte aligned (device
    // allocations are, and every offset is a multiple of four floats). The
    // state is read before this thread writes the same sixteen keys back
    // after the loop, and no other thread touches them, so the coherent
    // load reads what the previous launch wrote.
    let (qkv_g, state_g) = (global_addr(qkv.as_ptr()), global_addr(state.cast_const()));
    // SAFETY: s_at + HEAD <= lanes·n_v·HEAD·HEAD, the column inside state.
    let mut s = unsafe { lane_keys::<false>(state_g + 4 * s_at as u64, j) };
    let mut raised = false;
    let mut t = 0usize;
    while t < m {
        let row = t * ch;
        // SAFETY: row + k_at + HEAD <= t·ch + 2·n_k·HEAD <= m·ch and row +
        // q_at + HEAD <= row + n_k·HEAD: whole heads inside qkv.
        let (k, q) = unsafe {
            (
                lane_keys::<true>(qkv_g + 4 * (row + k_at) as u64, j),
                lane_keys::<true>(qkv_g + 4 * (row + q_at) as u64, j),
            )
        };
        let dk = if DECAY == DECAY_HEAD {
            [0.0f32; KEYS_PER_LANE]
        } else {
            // SAFETY: the per-key decay row (t·n_v + h)·HEAD .. + HEAD is
            // inside decay's m·n_v·HEAD values for DECAY_KEY, a multiple of
            // four floats from an aligned base, and no launch writes it
            // while this one runs.
            unsafe {
                lane_keys::<true>(
                    global_addr(decay.as_ptr()) + 4 * ((t * n_v + h) * HEAD) as u64,
                    j,
                )
            }
        };
        // SAFETY: row + v_at < row + ch <= m·ch; t·n_v + h < m·n_v <= the
        // lengths of beta and (per head) decay.
        let (vt, bt, dh) = unsafe {
            (
                *qkv.get_unchecked(row + v_at),
                *beta.get_unchecked(t * n_v + h),
                if DECAY == DECAY_HEAD {
                    *decay.get_unchecked(t * n_v + h)
                } else {
                    0.0
                },
            )
        };
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = mul_rn_f32(s[i], if DECAY == DECAY_HEAD { dh } else { dk[i] });
        }
        let kv = column_sum(lane_dot(s, k));
        let u = mul_rn_f32(vt - kv, bt);
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = fma_rn_f32(k[i], u, s[i]);
        }
        let y = column_sum(lane_dot(s, q));
        if !y.is_finite() && !raised {
            fault.raise(FaultSite::LinearDelta);
            raised = true;
        }
        if j == 0 {
            // SAFETY: (t·n_v + h)·HEAD + col < m·n_v·HEAD, inside `o` by the
            // fn's contract; one lane per (token, column).
            unsafe { *o.add((t * n_v + h) * HEAD + col) = y };
        }
        t += 1;
    }
    for i in 0..KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: the sixteen keys this lane read from `state` above, inside
        // it by the fn's contract; each read and written by this one lane.
        unsafe { *state.add(s_at + key_of(j, i)) = s[i] };
    }
}

#[cuda_module]
mod delta_kernels {
    use super::*;

    /// The delta step of `m` tokens with one decay per value head (module
    /// doc), on lane `lane[lane_at] mod lanes` of `state`, in place. Block
    /// `b` is head `b / 8`, columns `16·(b % 8) ..`.
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
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            lane.len() >= lane_at + 1,
            lanes >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384
        )
    )]
    pub fn gdn_delta(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
    ) {
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        // SAFETY: the launch contract holds the lengths the body names for
        // DECAY_HEAD, the lane word and `lanes >= 1`; `o` and `state` are
        // this launch's own outputs, `state` read and written in place by
        // the thread that owns each key (the body's doc); the launcher
        // checks n_k divides n_v, the block width and the grid.
        unsafe {
            delta_body::<DECAY_HEAD>(
                qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, grouped, m, fault, o, state,
            );
        }
    }

    /// The delta step of `m` tokens with one decay per (value head, key
    /// channel), `decay` `[m][n_v][HEAD]` (module doc); otherwise
    /// [`gdn_delta`].
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
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v * 128,
            lane.len() >= lane_at + 1,
            lanes >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384
        )
    )]
    pub fn kda_delta(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
    ) {
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        // SAFETY: as gdn_delta's, with the contract's m·n_v·HEAD decay
        // values the body reads for DECAY_KEY.
        unsafe {
            delta_body::<DECAY_KEY>(
                qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, grouped, m, fault, o, state,
            );
        }
    }

    /// [`gdn_delta`] over `lanes` lanes with their stamps (module doc): lane
    /// `lane[lane_at] mod lanes` read, checked against `pos[0]`, and written
    /// in place (`each == 0`) or token `j` into lane `(c + j) mod lanes`
    /// (`each != 0`, `m <= lanes`).
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
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            lane.len() >= lane_at + 1,
            lanes >= 1,
            pos.len() >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384,
            stamp.len() >= lanes
        )
    )]
    pub fn gdn_delta_lanes(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        each: u32,
        pos: &[u32],
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
        mut stamp: DisjointSlice<u32>,
    ) {
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        let stamp = stamp.as_mut_ptr();
        // SAFETY: the launch contract holds the lengths the body names, the
        // lane word, `lanes >= 1`, `pos[0]` and `lanes` stamps; `o`, `state`
        // and `stamp` are this launch's own outputs, the state read and
        // written by the thread that owns each key and the stamps by block
        // 0's thread 0 alone (the body's doc); the launcher checks n_k
        // divides n_v, `m <= lanes` in row mode, the block width and the grid.
        unsafe {
            delta_lanes_body::<DECAY_HEAD>(
                qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, grouped, m, each, 0, pos, fault,
                o, state, stamp,
            );
        }
    }

    /// [`kda_delta`] over `lanes` lanes with their stamps, from row `row` of
    /// a verify (module doc): lane [`read_lane`]`(c, row)` read, checked
    /// against `pos[0]`, and written in place (`each == 0`, `row == 0`) or
    /// token `j` into lane [`row_lane`]`(c, row + j)` (`each != 0`, `row + m
    /// <= lanes`), `c = lane[lane_at] mod lanes`.
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
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v * 128,
            lane.len() >= lane_at + 1,
            lanes >= 1,
            pos.len() >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384,
            stamp.len() >= lanes
        )
    )]
    pub fn kda_delta_lanes(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        each: u32,
        row: u32,
        pos: &[u32],
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
        mut stamp: DisjointSlice<u32>,
    ) {
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        let stamp = stamp.as_mut_ptr();
        // SAFETY: the launch contract holds the lengths the body names, the
        // lane word, `lanes >= 1`, `pos[0]`, `lanes` stamps and the m·n_v·HEAD
        // decay values the body reads for DECAY_KEY; `o`, `state` and `stamp`
        // are this launch's own outputs, the state read and written by the
        // thread that owns each key and the stamps by block 0's thread 0 alone
        // (the body's doc); the launcher checks n_k divides n_v, `row <
        // lanes`, row 0 in place mode and `row + m <= lanes` in row mode.
        unsafe {
            delta_lanes_body::<DECAY_KEY>(
                qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, grouped, m, each, row, pos,
                fault, o, state, stamp,
            );
        }
    }
}

/// [`DeltaKernels::enqueue_delta`]'s and [`DeltaKernels::enqueue_kda_delta`]'s
/// arguments: `m` tokens of the conv's output `qkv` (`[m][C]`), β
/// (`[m][n_v]`) and decay (`[m][n_v]`, or `[m][n_v][HEAD]` per key), the lane word
/// `lane[lane_at]` (a word of the body's parameter image), the output `o`
/// (`[m][n_v][HEAD]`), and the state (`[lanes][n_v][HEAD][HEAD]`,
/// value-major), whose lane the call reads and overwrites with the state
/// after its last token.
pub struct DeltaArgs<'a> {
    pub qkv: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub decay: &'a DeviceBuffer<f32>,
    pub lane: &'a DeviceBuffer<u32>,
    pub lane_at: usize,
    /// Lanes of `state`: 1 for [`DeltaKernels::enqueue_delta`] and
    /// [`DeltaKernels::enqueue_kda_delta`], any for the stamped entries.
    pub lanes: usize,
    pub shape: LinearShape,
    pub m: usize,
    pub fault: FaultSink,
    pub o: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct DeltaKernels {
    module: delta_kernels::LoadedModule,
}

impl DeltaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<DeltaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(delta_kernels, ctx)? };
        Ok(DeltaKernels { module })
    }

    /// Enqueue the delta step of `args.m` tokens with one decay per value
    /// head (`decay` `[m][n_v]`): `8·n_v` blocks of 128 threads, the state
    /// read and written in place. Refuses `lanes` other than 1 by name.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_delta(&self, stream: &CudaStream, args: DeltaArgs<'_>) -> Result<(), GpuError> {
        self.enqueue::<DECAY_HEAD>(stream, args, "enqueue_delta")
    }

    /// [`enqueue_delta`](Self::enqueue_delta) with one decay per (value
    /// head, key channel): `decay` `[m][n_v][HEAD]`, refused by name when
    /// shorter.
    pub fn enqueue_kda_delta(
        &self,
        stream: &CudaStream,
        args: DeltaArgs<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue::<DECAY_KEY>(stream, args, "enqueue_kda_delta")
    }

    fn enqueue<const DECAY: u32>(
        &self,
        stream: &CudaStream,
        args: DeltaArgs<'_>,
        what: &'static str,
    ) -> Result<(), GpuError> {
        let DeltaArgs {
            qkv,
            beta,
            decay,
            lane,
            lane_at,
            lanes,
            shape,
            m,
            fault,
            o,
            state,
        } = args;
        shape.check(what)?;
        if m == 0 {
            return Err(GpuError::shape(what, "need m >= 1, got m=0".to_owned()));
        }
        if lanes != 1 {
            return Err(GpuError::shape(
                what,
                format!(
                    "lanes={lanes}: only one state lane is built; more lanes need the \
                     per-token lane store (token j into lane (c + j) mod lanes), which is not"
                ),
            ));
        }
        let nv = shape.n_v;
        let decay_len = if DECAY == DECAY_HEAD {
            m * nv
        } else {
            m * nv * HEAD
        };
        let lens = [
            ("qkv", qkv.len(), m * shape.channels()),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), decay_len),
            ("lane", lane.len(), lane_at + 1),
            ("o", o.len(), m * nv * HEAD),
            ("state", state.len(), lanes * shape.state_len()),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", nv * (HEAD / COLUMNS_PER_BLOCK))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", nv)?;
        let m = launch_u32(what, "m", m)?;
        let lane_at = launch_u32(what, "lane_at", lane_at)?;
        let lanes = launch_u32(what, "lanes", lanes)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let map = shape.map.code();
        if DECAY == DECAY_HEAD {
            let prep = self.module.prepare_gdn_delta(cfg)?;
            self.module.gdn_delta(
                stream, &prep, qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, map, m, fault, o,
                state,
            )?;
        } else {
            let prep = self.module.prepare_kda_delta(cfg)?;
            self.module.kda_delta(
                stream, &prep, qkv, beta, decay, lane, lane_at, lanes, n_k, n_v, map, m, fault, o,
                state,
            )?;
        }
        Ok(())
    }
}

/// The host twin of [`lane_dot`] and [`column_sum`] over one column's
/// `HEAD` values `a`, `b` (key order): the eight lanes' sums, then the
/// butterfly.
#[must_use]
pub fn column_dot_host(a: &[f32], b: &[f32]) -> f32 {
    let lanes: Vec<f32> = (0..LANES_PER_COLUMN)
        .map(|j| {
            let at = |i: usize| key_of(j, i);
            let mut p: [f32; 4] = std::array::from_fn(|c| a[at(c)] * b[at(c)]);
            for r in 1..4 {
                for (c, pc) in p.iter_mut().enumerate() {
                    *pc = a[at(4 * r + c)].mul_add(b[at(4 * r + c)], *pc);
                }
            }
            (p[0] + p[1]) + (p[2] + p[3])
        })
        .collect();
    super::butterfly_f32(&lanes)
}

/// What [`delta_host`] computes: `o` (`[m][n_v][HEAD]`) and the whole state
/// buffer after the call (`[lanes][n_v][HEAD][HEAD]`).
pub struct DeltaOut {
    pub o: Vec<f32>,
    pub state: Vec<f32>,
}

/// The host rule of [`DeltaKernels::enqueue_delta`]: the module doc's
/// numeric contract, op for op, one thread per value head, on lane `lane
/// mod lanes` of `state` (`lanes` lanes), in place; `decay` `[m][n_v]`.
///
/// # Panics
///
/// When a slice is shorter than [`DeltaArgs`]'s lengths for `shape`, `m`
/// and `lanes`, or `lanes` is not 1 (the launcher's refusal).
#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
#[must_use]
pub fn delta_host(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    state: &[f32],
    lanes: usize,
    lane: u32,
    shape: LinearShape,
    m: usize,
) -> DeltaOut {
    delta_rule_host::<DECAY_HEAD>(qkv, beta, decay, state, lanes, lane, shape, m)
}

/// The host rule of [`DeltaKernels::enqueue_kda_delta`]: [`delta_host`]
/// with `decay` `[m][n_v][HEAD]`, key `i`'s decay scaling state key `i`.
///
/// # Panics
///
/// As [`delta_host`], `decay` shorter than `m·n_v·HEAD` included.
#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
#[must_use]
pub fn kda_delta_host(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    state: &[f32],
    lanes: usize,
    lane: u32,
    shape: LinearShape,
    m: usize,
) -> DeltaOut {
    delta_rule_host::<DECAY_KEY>(qkv, beta, decay, state, lanes, lane, shape, m)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
fn delta_rule_host<const DECAY: u32>(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    state: &[f32],
    lanes: usize,
    lane: u32,
    shape: LinearShape,
    m: usize,
) -> DeltaOut {
    assert!(
        lanes == 1,
        "delta_host: lanes={lanes}, only one state lane is built"
    );
    let (n_k, n_v) = (shape.n_k, shape.n_v);
    let ch = shape.channels();
    let mut o = vec![0.0f32; m * n_v * HEAD];
    let mut state = state[..lanes * shape.state_len()].to_vec();
    let at = (lane as usize % lanes) * shape.state_len();
    std::thread::scope(|sc| {
        let mut o_heads: Vec<Vec<f32>> = Vec::with_capacity(n_v);
        let handles: Vec<_> = state[at..at + shape.state_len()]
            .chunks_mut(HEAD * HEAD)
            .enumerate()
            .map(|(h, sh)| {
                sc.spawn(move || {
                    let kh = shape.map.k_head(h, n_k, n_v);
                    let mut oh = vec![0.0f32; m * HEAD];
                    for (col, s) in sh.chunks_mut(HEAD).enumerate() {
                        for t in 0..m {
                            let row = &qkv[t * ch..(t + 1) * ch];
                            let q = &row[kh * HEAD..(kh + 1) * HEAD];
                            let k = &row[(n_k + kh) * HEAD..(n_k + kh + 1) * HEAD];
                            let v = row[2 * n_k * HEAD + h * HEAD + col];
                            let bt = beta[t * n_v + h];
                            if DECAY == DECAY_HEAD {
                                let dc = decay[t * n_v + h];
                                for si in s.iter_mut() {
                                    *si *= dc;
                                }
                            } else {
                                let dk = &decay[(t * n_v + h) * HEAD..(t * n_v + h + 1) * HEAD];
                                for (si, &di) in s.iter_mut().zip(dk) {
                                    *si *= di;
                                }
                            }
                            let u = (v - column_dot_host(s, k)) * bt;
                            for (si, &ki) in s.iter_mut().zip(k) {
                                *si = ki.mul_add(u, *si);
                            }
                            oh[t * HEAD + col] = column_dot_host(s, q);
                        }
                    }
                    oh
                })
            })
            .collect();
        for hd in handles {
            o_heads.push(hd.join().expect("a delta_host head thread panicked"));
        }
        for (h, oh) in o_heads.iter().enumerate() {
            for t in 0..m {
                o[(t * n_v + h) * HEAD..(t * n_v + h + 1) * HEAD]
                    .copy_from_slice(&oh[t * HEAD..(t + 1) * HEAD]);
            }
        }
    });
    DeltaOut { o, state }
}

/// The stamp of a lane no call has written: no position a call starts at,
/// so reading the lane raises [`FaultSite::DeltaStamp`].
pub const NEVER: u32 = u32::MAX;

/// The lane row `j` of a verify writes when the verify's lanes start from
/// the committed lane `c`: `(c + j) mod lanes`. A host that keeps the first
/// `k` rows moves the lane word to `row_lane(c, k − 1)`.
#[must_use]
pub const fn row_lane(c: u32, j: u32, lanes: u32) -> u32 {
    (c + j) % lanes
}

/// The lane row `row` of a verify reads when the verify's lanes start from
/// the committed lane `c`: the lane the row before it wrote, `c` itself for
/// row 0.
#[must_use]
pub const fn read_lane(c: u32, row: u32, lanes: u32) -> u32 {
    if row == 0 { c } else { (c + row - 1) % lanes }
}

/// The loop of one block of `gdn_delta_lanes` and `kda_delta_lanes` (module
/// doc): [`delta_body`]'s loop for `DECAY` over lane [`read_lane`]`(c, row)`,
/// `c = lane[lane_at] mod lanes`, with the stamp check and writes on block
/// 0's thread 0 and the state stored per row (`each != 0`) or once after the
/// last token.
///
/// # Safety
///
/// [`delta_body`]'s contract for `DECAY`, and `pos` holds a word, `stamp`
/// addresses `lanes` writable u32s no other launch touches, `row == 0` when
/// `each == 0`, and `row + m <= lanes` when `each != 0`.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn delta_lanes_body<const DECAY: u32>(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    lane_word: &[u32],
    lane_at: u32,
    lanes: u32,
    n_k: u32,
    n_v: u32,
    grouped: u32,
    m: u32,
    each: u32,
    row: u32,
    pos: &[u32],
    fault: FaultSink,
    o: *mut f32,
    state: *mut f32,
    stamp: *mut u32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let j = lane % LANES_PER_COLUMN;
    let b = thread::blockIdx_x() as usize;
    let blocks_per_head = HEAD / COLUMNS_PER_BLOCK;
    let h = b / blocks_per_head;
    let (n_k, n_v, m) = (n_k as usize, n_v as usize, m as usize);
    // SAFETY: lane_at < lane_word.len() and pos holds a word, by the
    // contract.
    let (word, p0) = unsafe {
        (
            *lane_word.get_unchecked(lane_at as usize),
            *pos.get_unchecked(0),
        )
    };
    if word >= lanes && tid == 0 {
        fault.raise(FaultSite::DeltaLane);
    }
    let c = word % lanes;
    let rin = if row == 0 { c } else { (c + row - 1) % lanes };
    if b == 0 && tid == 0 {
        // SAFETY: rin < lanes and every (c + row + t) mod lanes < lanes:
        // inside the `lanes` stamps; block 0's thread 0 is their only reader
        // and writer in the launch, and it reads stamp rin before it writes
        // any.
        unsafe {
            if *stamp.add(rin as usize) != p0 {
                fault.raise(FaultSite::DeltaStamp);
            }
            if each != 0 {
                let mut t = 0u32;
                while (t as usize) < m {
                    *stamp.add(((c + row + t) % lanes) as usize) = p0.wrapping_add(t + 1);
                    t += 1;
                }
            } else {
                *stamp.add(c as usize) = p0.wrapping_add(m as u32);
            }
        }
    }
    if h >= n_v {
        return; // block-uniform
    }
    let col = (b % blocks_per_head) * COLUMNS_PER_BLOCK + (tid / 32) * 4 + lane / LANES_PER_COLUMN;
    let kh = if grouped != 0 {
        h / (n_v / n_k)
    } else {
        h % n_k
    };
    let ch = (2 * n_k + n_v) * HEAD;
    let q_at = kh * HEAD;
    let k_at = (n_k + kh) * HEAD;
    let v_at = 2 * n_k * HEAD + h * HEAD + col;
    let lane_len = n_v * HEAD * HEAD;
    let col_at = (h * HEAD + col) * HEAD;
    let s_at = rin as usize * lane_len + col_at;

    // SAFETY: as in `delta_body`: the rows read are inside qkv and the
    // column inside lane rin of the state (`rin < lanes`), 16-byte aligned;
    // the state is read before this thread writes its keys of any lane, and
    // no other thread touches them.
    let (qkv_g, state_g) = (global_addr(qkv.as_ptr()), global_addr(state.cast_const()));
    // SAFETY: s_at + HEAD <= lanes·n_v·HEAD·HEAD, the column inside state.
    let mut s = unsafe { lane_keys::<false>(state_g + 4 * s_at as u64, j) };
    let mut raised = false;
    let mut t = 0usize;
    while t < m {
        let row_at = t * ch;
        // SAFETY: whole heads inside qkv, as in `delta_body`.
        let (k, q) = unsafe {
            (
                lane_keys::<true>(qkv_g + 4 * (row_at + k_at) as u64, j),
                lane_keys::<true>(qkv_g + 4 * (row_at + q_at) as u64, j),
            )
        };
        let dk = if DECAY == DECAY_HEAD {
            [0.0f32; KEYS_PER_LANE]
        } else {
            // SAFETY: the per-key decay row (t·n_v + h)·HEAD .. + HEAD is
            // inside decay's m·n_v·HEAD values for DECAY_KEY, a multiple of
            // four floats from an aligned base, and no launch writes it
            // while this one runs.
            unsafe {
                lane_keys::<true>(
                    global_addr(decay.as_ptr()) + 4 * ((t * n_v + h) * HEAD) as u64,
                    j,
                )
            }
        };
        // SAFETY: row_at + v_at < m·ch; t·n_v + h < m·n_v <= the lengths of
        // beta and (per head) decay.
        let (vt, bt, dh) = unsafe {
            (
                *qkv.get_unchecked(row_at + v_at),
                *beta.get_unchecked(t * n_v + h),
                if DECAY == DECAY_HEAD {
                    *decay.get_unchecked(t * n_v + h)
                } else {
                    0.0
                },
            )
        };
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = mul_rn_f32(s[i], if DECAY == DECAY_HEAD { dh } else { dk[i] });
        }
        let kv = column_sum(lane_dot(s, k));
        let u = mul_rn_f32(vt - kv, bt);
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = fma_rn_f32(k[i], u, s[i]);
        }
        let y = column_sum(lane_dot(s, q));
        if !y.is_finite() && !raised {
            fault.raise(FaultSite::LinearDelta);
            raised = true;
        }
        if j == 0 {
            // SAFETY: (t·n_v + h)·HEAD + col < m·n_v·HEAD, inside `o`; one
            // lane per (token, column).
            unsafe { *o.add((t * n_v + h) * HEAD + col) = y };
        }
        if each != 0 {
            let at = ((c as usize + row as usize + t) % lanes as usize) * lane_len + col_at;
            for i in 0..KEYS_PER_LANE {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: lane (c + row + t) mod lanes < lanes, so this
                // lane's sixteen keys of the column are inside state; this
                // thread is their only writer.
                unsafe { *state.add(at + key_of(j, i)) = s[i] };
            }
        }
        t += 1;
    }
    if each == 0 {
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: the sixteen keys this lane read from lane rin above
            // (lane c: in place mode is row 0).
            unsafe { *state.add(s_at + key_of(j, i)) = s[i] };
        }
    }
}

/// [`DeltaKernels::enqueue_delta_lanes`]'s arguments: [`DeltaArgs`] over
/// `lanes` lanes (one decay per value head), the row mode, the call's
/// positions (`pos[0]` its first) and the store's `lanes` stamps.
pub struct DeltaLanesArgs<'a> {
    pub delta: DeltaArgs<'a>,
    /// Row mode: the state after token `j` into lane `(c + j) mod lanes`;
    /// else the last token's back into lane `c`.
    pub each: bool,
    pub pos: &'a DeviceBuffer<u32>,
    pub stamp: &'a mut DeviceBuffer<u32>,
}

/// [`DeltaKernels::enqueue_kda_delta_lanes`]'s arguments: [`DeltaLanesArgs`]
/// with the decay per key (`[m][n_v][HEAD]`), and the verify row the call's
/// first token is.
pub struct KdaLanesArgs<'a> {
    pub lanes: DeltaLanesArgs<'a>,
    /// Row 0 for the one-token step and a prompt batch (in place), the
    /// verify's row for a launch of one of its rows.
    pub row: usize,
}

impl DeltaKernels {
    /// Enqueue `gdn_delta_lanes` (module doc): `8·n_v` blocks of 128
    /// threads. Refused by name: no lane, row mode over more tokens than
    /// lanes, a slice shorter than the launch reads. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_delta_lanes(
        &self,
        stream: &CudaStream,
        args: DeltaLanesArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_delta_lanes";
        let DeltaLanesArgs {
            delta:
                DeltaArgs {
                    qkv,
                    beta,
                    decay,
                    lane,
                    lane_at,
                    lanes,
                    shape,
                    m,
                    fault,
                    o,
                    state,
                },
            each,
            pos,
            stamp,
        } = args;
        shape.check(WHAT)?;
        if m == 0 || lanes == 0 || (each && m > lanes) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "m={m} over {lanes} lanes{}: a call takes one token or more and a lane or \
                     more, and row mode at most a token a lane",
                    if each { " in row mode" } else { "" }
                ),
            ));
        }
        let nv = shape.n_v;
        let lens = [
            ("qkv", qkv.len(), m * shape.channels()),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv),
            ("lane", lane.len(), lane_at + 1),
            ("pos", pos.len(), 1),
            ("o", o.len(), m * nv * HEAD),
            ("state", state.len(), lanes * shape.state_len()),
            ("stamp", stamp.len(), lanes),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                WHAT,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(WHAT, "grid", nv * (HEAD / COLUMNS_PER_BLOCK))?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let prep = self.module.prepare_gdn_delta_lanes(cfg)?;
        self.module.gdn_delta_lanes(
            stream,
            &prep,
            qkv,
            beta,
            decay,
            lane,
            launch_u32(WHAT, "lane_at", lane_at)?,
            launch_u32(WHAT, "lanes", lanes)?,
            launch_u32(WHAT, "n_k", shape.n_k)?,
            launch_u32(WHAT, "n_v", nv)?,
            shape.map.code(),
            launch_u32(WHAT, "m", m)?,
            u32::from(each),
            pos,
            fault,
            o,
            state,
            stamp,
        )?;
        Ok(())
    }

    /// Enqueue `kda_delta_lanes` (module doc): `8·n_v` blocks of 128
    /// threads. Refused by name: no token or no lane, a row base at or past
    /// the lanes, in place mode past row 0, row mode past the lanes (`row +
    /// m > lanes`), a slice shorter than the launch reads. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_kda_delta_lanes(
        &self,
        stream: &CudaStream,
        args: KdaLanesArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_kda_delta_lanes";
        let KdaLanesArgs {
            lanes:
                DeltaLanesArgs {
                    delta:
                        DeltaArgs {
                            qkv,
                            beta,
                            decay,
                            lane,
                            lane_at,
                            lanes,
                            shape,
                            m,
                            fault,
                            o,
                            state,
                        },
                    each,
                    pos,
                    stamp,
                },
            row,
        } = args;
        shape.check(WHAT)?;
        if let Some(why) = lanes_refusal(m, lanes, each, row) {
            return Err(GpuError::shape(WHAT, why));
        }
        let nv = shape.n_v;
        let lens = [
            ("qkv", qkv.len(), m * shape.channels()),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv * HEAD),
            ("lane", lane.len(), lane_at + 1),
            ("pos", pos.len(), 1),
            ("o", o.len(), m * nv * HEAD),
            ("state", state.len(), lanes * shape.state_len()),
            ("stamp", stamp.len(), lanes),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                WHAT,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(WHAT, "grid", nv * (HEAD / COLUMNS_PER_BLOCK))?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let prep = self.module.prepare_kda_delta_lanes(cfg)?;
        self.module.kda_delta_lanes(
            stream,
            &prep,
            qkv,
            beta,
            decay,
            lane,
            launch_u32(WHAT, "lane_at", lane_at)?,
            launch_u32(WHAT, "lanes", lanes)?,
            launch_u32(WHAT, "n_k", shape.n_k)?,
            launch_u32(WHAT, "n_v", nv)?,
            shape.map.code(),
            launch_u32(WHAT, "m", m)?,
            u32::from(each),
            launch_u32(WHAT, "row", row)?,
            pos,
            fault,
            o,
            state,
            stamp,
        )?;
        Ok(())
    }
}

/// Why a stamped call of `m` tokens over `lanes` lanes from verify row `row`,
/// in row mode when `each`, is refused ([`DeltaKernels::enqueue_kda_delta_lanes`]'s
/// rule); `None` for a call the kernel takes.
#[must_use]
pub fn lanes_refusal(m: usize, lanes: usize, each: bool, row: usize) -> Option<String> {
    if m == 0 || lanes == 0 {
        return Some(format!(
            "m={m} over {lanes} lanes: a call takes one token or more and a lane or more"
        ));
    }
    if row >= lanes {
        return Some(format!(
            "row {row} over {lanes} lanes: a verify's rows each write a lane of their own"
        ));
    }
    if !each && row != 0 {
        return Some(format!(
            "row {row} in place: only row 0 writes back into the lane it read"
        ));
    }
    if each && row + m > lanes {
        return Some(format!(
            "rows {row}..{} over {lanes} lanes in row mode: at most a token a lane",
            row + m
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{delta_host, kda_delta_host, lanes_refusal, read_lane, row_lane};
    use crate::linear::{HEAD, KHeadMap, LinearShape};

    /// Knuth's MMIX LCG, the high 24 bits as a float in `[lo, hi)`.
    fn fill(seed: &mut u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                lo + (hi - lo) * ((*seed >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    /// The per-key rule with every key of a head given the head's decay is
    /// the per-head rule, bit for bit: the body is one rule at two
    /// granularities.
    #[test]
    fn per_key_decay_equal_per_head_is_the_head_rule() {
        let shape = LinearShape {
            n_k: 2,
            n_v: 4,
            map: KHeadMap::Tiled,
        };
        let m = 5;
        let mut r = 0x6b64_6131;
        let qkv = fill(&mut r, m * shape.channels(), -1.0, 1.0);
        let beta = fill(&mut r, m * shape.n_v, 0.0, 1.0);
        let decay = fill(&mut r, m * shape.n_v, 0.5, 1.0);
        let state = fill(&mut r, shape.state_len(), -0.1, 0.1);
        let per_key: Vec<f32> = decay
            .iter()
            .flat_map(|&d| std::iter::repeat_n(d, HEAD))
            .collect();
        let h = delta_host(&qkv, &beta, &decay, &state, 1, 0, shape, m);
        let k = kda_delta_host(&qkv, &beta, &per_key, &state, 1, 0, shape, m);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&h.o), bits(&k.o));
        assert_eq!(bits(&h.state), bits(&k.state));
    }

    /// A per-key decay moves only its own key of the state: a key whose
    /// decay differs from the head's changes state values of that key alone.
    #[test]
    fn a_key_decay_scales_its_own_key() {
        let shape = LinearShape {
            n_k: 1,
            n_v: 1,
            map: KHeadMap::Tiled,
        };
        let mut r = 0x6b64_6132;
        let qkv = fill(&mut r, shape.channels(), -1.0, 1.0);
        let state = fill(&mut r, shape.state_len(), -0.1, 0.1);
        // β = 0: the step is S = decay·S, the key's own column scaling.
        let beta = vec![0.0f32];
        let flat = vec![0.75f32; HEAD];
        let mut moved = flat.clone();
        moved[37] = 0.25;
        let a = kda_delta_host(&qkv, &beta, &flat, &state, 1, 0, shape, 1);
        let b = kda_delta_host(&qkv, &beta, &moved, &state, 1, 0, shape, 1);
        for (i, (x, y)) in a.state.iter().zip(&b.state).enumerate() {
            let key = i % HEAD;
            assert_eq!(x.to_bits() == y.to_bits(), key != 37, "state value {i}");
        }
    }

    /// A verify launched one row at a time reads and writes the lanes a
    /// one-launch verify of its rows does: row 0 reads the committed lane
    /// `c`, row `r` the lane row `r − 1` wrote, and row `r` writes lane `c +
    /// r`; every row's lane is its own, none is the lane row 0 read unless it
    /// is row 0's, and keeping `k` rows leaves the word at the lane row `k −
    /// 1` wrote.
    #[test]
    fn a_row_reads_the_lane_the_row_before_wrote() {
        for lanes in 1..=4u32 {
            for c in 0..lanes {
                let wrote: Vec<u32> = (0..lanes).map(|r| row_lane(c, r, lanes)).collect();
                let mut sorted = wrote.clone();
                sorted.sort_unstable();
                assert_eq!(
                    sorted,
                    (0..lanes).collect::<Vec<_>>(),
                    "lanes {lanes} c {c}"
                );
                assert_eq!(read_lane(c, 0, lanes), c);
                for r in 1..lanes {
                    assert_eq!(read_lane(c, r, lanes), wrote[r as usize - 1], "row {r}");
                }
            }
        }
        // Two lanes, committed lane 1: row 0 in place on 1, row 1 reads 1
        // and writes 0; keeping both moves the word to 0.
        assert_eq!((read_lane(1, 1, 2), row_lane(1, 1, 2)), (1, 0));
    }

    /// The launcher's refusals: no token or lane, a row past the lanes, a
    /// later row in place, row mode past the lanes; the step (row 0 in
    /// place, any m) and each row of a two-row verify (row mode, m = 1) are
    /// taken, and on a state of one lane the step alone.
    #[test]
    fn the_lanes_launch_takes_the_step_and_each_verify_row() {
        assert!(lanes_refusal(1, 1, false, 0).is_none());
        assert!(lanes_refusal(512, 1, false, 0).is_none());
        assert!(lanes_refusal(1, 2, false, 0).is_none());
        assert!(lanes_refusal(512, 2, false, 0).is_none());
        assert!(lanes_refusal(1, 2, true, 0).is_none());
        assert!(lanes_refusal(1, 2, true, 1).is_none());
        assert!(lanes_refusal(2, 2, true, 0).is_none());
        for (m, lanes, each, row, says) in [
            (0, 2, false, 0, "one token or more"),
            (1, 0, false, 0, "a lane or more"),
            (1, 2, true, 2, "row 2 over 2 lanes"),
            (1, 2, false, 1, "only row 0"),
            (2, 2, true, 1, "rows 1..3 over 2 lanes"),
            (1, 1, true, 1, "row 1 over 1 lanes"),
            (2, 1, true, 0, "rows 0..2 over 1 lanes"),
        ] {
            let why = lanes_refusal(m, lanes, each, row).expect("refused");
            assert!(why.contains(says), "{m} {lanes} {each} {row}: {why}");
        }
    }
}
