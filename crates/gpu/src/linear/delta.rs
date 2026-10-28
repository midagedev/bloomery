//! The recurrent delta step of a linear-attention layer over `m` tokens,
//! one launch per call: decode is `m = 1`, a prompt runs its tokens through
//! the same loop with the state held in registers, read once and written
//! once, in place. A GDN prompt can take the prompt form instead, three
//! launches over chunks of 16 tokens with a carry beside the state (the
//! last paragraphs).
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
//! A GDN verify can instead use [`DeltaKernels::enqueue_delta_recovery`].
//! Row mode leaves the committed state and stamps untouched and records each
//! row's key, rounded update scalar and head decay in two spare state lanes.
//! An accepted prefix is replayed from the committed state using the same
//! rounded decay and fused update, without recomputing its dot products or
//! outputs. Its last row's lane receives the only full state store and its
//! stamp. Two record banks let replay read outside that destination lane,
//! even when it coincides with a spare lane. A discarded verify leaves its
//! committed state intact. Both calls first fold the same prompt carry.
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
//!
//! The prompt form ([`DeltaKernels::enqueue_delta_prompt`]) runs an
//! in-place call as the chunked delta rule over chunks of
//! [`CHUNK_ROWS`](super::CHUNK_ROWS) tokens, its products on BF16 tensor
//! cores with f32 accumulation. In a chunk that starts from the state `S`,
//! with `Dₜ` the product of the chunk's decays up to row `t` and `A[t][j] =
//! βₜ·(Dₜ/Dⱼ)·kₜ·kⱼ` for `j < t`:
//! - the solved rows `u = (I + A)⁻¹·r`, `rₜ = βₜ·(vₜ − Dₜ·Sᵀkₜ)`;
//! - `oₜ = Dₜ·Sᵀqₜ + Σ_{j ≤ t} (Dₜ/Dⱼ)·(qₜ·kⱼ)·uⱼ`;
//! - the state after it `D·S + Σⱼ (D/Dⱼ)·kⱼ·uⱼᵀ`, `D` the last row's.
//!
//! `gdn_chunk_prepare`, one block a (chunk, value head), all chunks at once,
//! writes a record of `work` from the chunk's own rows: the inverse (by
//! forward substitution in f32), the causal product
//! `(Dₜ/Dⱼ)·qₜ·kⱼ` (`j ≤ t`) and the decays. `gdn_delta_prepared`, two blocks
//! a value head of 64 columns each, walks the chunks in order with its
//! columns' state in f32 registers, read once and written once, under
//! `gdn_delta_lanes`'s lane and stamp rules. Each F32 operand is represented
//! by a BF16 high part and BF16 residual. Products accumulate high-high and
//! both cross terms in F32, retaining low bits in the solved updates and in
//! `v − Dₜ·Sᵀkₜ`. The sum order differs from `gdn_delta_lanes`:
//! `gate_linear`'s `prompt` clause holds `o` and the state within a band of
//! [`delta_host`]. A non-finite `o`
//! or state value raises [`FaultSite::LinearDelta`] once per thread; a
//! non-finite `v` reaches its column of `o` from its chunk's first row on
//! (the tensor cores' `0·∞`) and that column of the state.
//!
//! Carry. A token's bits do not depend on the calls a prompt is cut into: a
//! prompt's chunks start at the multiples of
//! [`CHUNK_ROWS`](super::CHUNK_ROWS) from the position its rows began at,
//! and row `t` of a chunk reads only the state
//! the chunk starts from and the chunk's rows up to `t` (the masked terms
//! and the padding are exact zeros, and every product keeps its shape). A
//! call that ends inside a chunk leaves the state the chunk starts from in
//! the lane, stamped at the call's end, and the chunk's rows in the store's
//! carry ([`LinearShape::carry_len`]): [`CARRY_HEAD`] words (the rows'
//! count and the position after them), then up to [`CARRY_ROWS`] rows of
//! the conv's key and value heads, then their β and decays. A call from
//! that position takes the carried rows first. The prompt form runs them
//! through the chunk again, writing the outputs of its own rows only, and
//! completes the chunk; its third launch, `gdn_carry_keep`, then writes the
//! carry the call leaves. `gdn_delta_lanes` (a step, a pass, a verify's row
//! 0) runs them through its loop first, writing no output for them. A
//! carry that stands at another position holds no rows for the call, so a
//! call after any other writer of the lane starts a chunk of its own. The
//! carry moves with the lane wherever a state is copied.

use super::{
    BLOCK, CARRY_HEAD, CARRY_ROWS, CHUNK_ROWS as CHUNK, DECAY_HEAD, DECAY_KEY, HEAD, KHeadMap,
    LinearShape,
};
use crate::GpuError;
use crate::cores::global_addr;
use crate::fault::{FaultSink, FaultSite, quad_finite};
use crate::launch_u32;
use crate::tensor::check_alignment;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::cvt_bf16x2_f32 as chunk_pair;
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::vector::U32x4;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, ptx_asm, thread, warp,
};
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

/// One token's update of a lane's sixteen keys (module doc, the first four
/// items): each key decayed (by `dh`, the head's decay, for
/// `DECAY_HEAD`, else by its own `dk`), then `u = (v − S'ᵀk)·β` and `S =
/// fma(k, u, S')`.
#[inline(always)]
fn rule_row<const DECAY: u32>(
    s: [f32; KEYS_PER_LANE],
    k: [f32; KEYS_PER_LANE],
    dk: [f32; KEYS_PER_LANE],
    dh: f32,
    (v, b): (f32, f32),
) -> [f32; KEYS_PER_LANE] {
    rule_update::<DECAY>(s, k, dk, dh, (v, b)).0
}

#[inline(always)]
fn rule_update<const DECAY: u32>(
    s: [f32; KEYS_PER_LANE],
    k: [f32; KEYS_PER_LANE],
    dk: [f32; KEYS_PER_LANE],
    dh: f32,
    (v, b): (f32, f32),
) -> ([f32; KEYS_PER_LANE], f32) {
    let s = decay_row::<DECAY>(s, dk, dh);
    let u = mul_rn_f32(v - column_sum(lane_dot(s, k)), b);
    (update_row(s, k, u), u)
}

#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
fn decay_row<const DECAY: u32>(
    mut s: [f32; KEYS_PER_LANE],
    dk: [f32; KEYS_PER_LANE],
    dh: f32,
) -> [f32; KEYS_PER_LANE] {
    for i in 0..KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        s[i] = mul_rn_f32(s[i], if DECAY == DECAY_HEAD { dh } else { dk[i] });
    }
    s
}

#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
fn update_row(
    mut s: [f32; KEYS_PER_LANE],
    k: [f32; KEYS_PER_LANE],
    u: f32,
) -> [f32; KEYS_PER_LANE] {
    for i in 0..KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        s[i] = fma_rn_f32(k[i], u, s[i]);
    }
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
        s = rule_row::<DECAY>(s, k, dk, dh, (vt, bt));
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

/// u32 words of one (chunk, value head) record `gdn_chunk_prepare` writes
/// and `gdn_delta_prepared` reads: the BF16 high pairs of the inverse and of
/// the causal product, then their BF16 residual pairs, each row-major with
/// eight words a row; then the cumulative decays
/// `Dₜ`, the β values and the factors `D/Dₜ` (`D` the chunk's last row's),
/// sixteen f32 bits each.
const CHUNK_WORDS: usize = 4 * CHUNK * CHUNK / 2 + 3 * CHUNK;

/// Value columns of one `gdn_delta_prepared` block: four warps of two
/// eight-column tiles.
const PREPARED_COLUMNS: usize = 64;

/// Row stride, in BF16 pairs, of a packed tile of [`HEAD`]-wide rows; the
/// four spare words keep a warp's fragment loads off one bank.
const PSTRIDE: usize = HEAD / 2 + 4;

/// Words of a packed Q or K tile of one chunk.
const PTILE: usize = CHUNK * PSTRIDE;

/// Word of carried row `t`'s key heads in a carry of `kv = (n_k + n_v)·HEAD`
/// words a row (module doc, "Carry"): its value heads follow them.
#[inline(always)]
const fn carry_row(t: usize, kv: usize) -> usize {
    CARRY_HEAD + t * kv
}

/// Word of carried row `t`'s β of value head `h` in a carry of `kv` words a
/// row over `nv` value heads; its decay lies [`CARRY_ROWS`]` · nv` words on.
#[inline(always)]
const fn carry_beta(t: usize, h: usize, kv: usize, nv: usize) -> usize {
    carry_row(CARRY_ROWS, kv) + t * nv + h
}

/// The rows a call from position `p0` takes from a carry whose header holds
/// `n` rows standing at position `at` (module doc, "Carry"): `n` when the
/// carry stands at `p0`, else zero. A count past [`CARRY_ROWS`] is invalid.
#[inline(always)]
const fn held_rows(n: usize, at: u32, p0: u32) -> Option<usize> {
    if n > CARRY_ROWS {
        None
    } else {
        Some(if at == p0 { n } else { 0 })
    }
}

/// [`held_rows`] of `carry`'s header.
///
/// # Safety
///
/// `carry` holds its [`CARRY_HEAD`] words.
#[inline(always)]
unsafe fn carried_rows(carry: &[f32], p0: u32) -> Option<usize> {
    // SAFETY: word 0 of the header, by the contract.
    let n = unsafe { carry.get_unchecked(0) }.to_bits() as usize;
    // SAFETY: word 1 of the header, by the contract.
    let at = unsafe { carry.get_unchecked(1) }.to_bits();
    held_rows(n, at, p0)
}

/// A call's rows as the prompt form reads them (module doc, "Carry"), as
/// global addresses: row `r` below `lead` is carried row `r` of `old` (a
/// carry's layout, `kv` words a row: the key heads, then the value heads),
/// any other row `r − lead` of `qkv` (`ch` words a row: the query, key and
/// value heads).
#[derive(Clone, Copy)]
struct CallRows {
    qkv: u64,
    old: u64,
    lead: usize,
    ch: usize,
    kv: usize,
    nk: usize,
}

impl CallRows {
    /// The first key of row `r`, carried or current.
    #[inline(always)]
    fn kv_row(self, r: usize) -> u64 {
        if r < self.lead {
            self.old + 4 * carry_row(r, self.kv) as u64
        } else {
            self.qkv + 4 * ((r - self.lead) * self.ch + self.nk * HEAD) as u64
        }
    }

    /// Key head `kh` of row `r`.
    #[inline(always)]
    fn key(self, r: usize, kh: usize) -> u64 {
        self.kv_row(r) + 4 * (kh * HEAD) as u64
    }

    /// Value word `at` (`h·HEAD` plus a column) of row `r`.
    #[inline(always)]
    fn value(self, r: usize, at: usize) -> u64 {
        self.kv_row(r) + 4 * (self.nk * HEAD + at) as u64
    }

    /// Query head `kh` of row `r >= lead`: a carried row keeps no query, its
    /// output written by the call that carried it.
    #[inline(always)]
    fn query(self, r: usize, kh: usize) -> u64 {
        self.qkv + 4 * ((r - self.lead) * self.ch + kh * HEAD) as u64
    }
}

/// A warp's 16 × 8 product `A·Bᵀ + c` over `K` columns: `a` sixteen packed
/// rows, `b` eight, each `K / 2` pairs at stride [`PSTRIDE`] when `K` is
/// [`HEAD`] and `K / 2` otherwise. `c` and the result are the MMA's
/// accumulator fragment: lane `l` holds rows `l / 4` and `l / 4 + 8`,
/// columns `2·(l % 4)` and the next.
///
/// # Safety
///
/// `a` and `b` address initialized shared words of those shapes, and every
/// lane of the warp executes the call.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn chunk_mma<const K: usize>(a: *const u32, b: *const u32, mut c: [f32; 4]) -> [f32; 4] {
    let lane = thread::threadIdx_x() as usize % 32;
    let r = lane / 4;
    let k = lane % 4;
    let stride = if K == HEAD { PSTRIDE } else { K / 2 };
    let mut base = 0;
    while base < K / 2 {
        let mut af = [0; 4];
        for i in 0..4 {
            let at = (r + (i % 2) * 8) * stride + base + k + (i / 2) * 4;
            // SAFETY: rows r and r + 8 of A's sixteen, pair k or k + 4 of
            // this step's eight.
            af[i] = unsafe { *a.add(at) };
        }
        let mut bf = [0; 2];
        for i in 0..2 {
            let at = r * stride + base + k + i * 4;
            // SAFETY: row r of B's eight, pair k or k + 4 of this step's.
            bf[i] = unsafe { *b.add(at) };
        }
        // SAFETY: every lane runs this step with its documented fragments.
        c = unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c, af, bf) };
        base += 8;
    }
    c
}

/// BF16 residual pairs after rounding two F32 operands to BF16.
#[inline(always)]
fn chunk_pair_low(a: f32, b: f32) -> u32 {
    let high = chunk_pair(a, b);
    chunk_pair(
        a - f32::from_bits(high << 16),
        b - f32::from_bits(high & 0xffff_0000),
    )
}

/// The high-high and two cross products of split BF16 operands.
///
/// # Safety
///
/// Each pointer addresses initialized shared memory: A is sixteen rows of
/// K BF16 values, B eight columns of K values, with stride PSTRIDE pairs
/// for K = HEAD and K / 2 otherwise. Every warp lane executes the call.
#[inline(always)]
unsafe fn chunk_product<const K: usize>(
    a: (*const u32, *const u32),
    b: (*const u32, *const u32),
    c: [f32; 4],
) -> [f32; 4] {
    // SAFETY: a.1 (A's residual) and b.0 (B's high) are initialized sixteen-row and eight-column
    // tiles of K BF16 values at the contract's stride; every lane executes the call.
    let c = unsafe { chunk_mma::<K>(a.1, b.0, c) };
    // SAFETY: a.0 (A's high) and b.1 (B's residual) are initialized tiles of the same shapes and
    // stride; every lane executes the call.
    let c = unsafe { chunk_mma::<K>(a.0, b.1, c) };
    // SAFETY: a.0 and b.0 are the initialized high tiles of the same shapes and stride; every lane
    // executes all three products.
    unsafe { chunk_mma::<K>(a.0, b.0, c) }
}

/// One thread's eight 4-key pieces of a chunk's query rows (the first four)
/// and key rows of key head `kh`, the chunk's rows `start ..` of `rows`, all
/// loads issued before any is used; rows at or past `count`, and a carried
/// row's query, read as zero.
///
/// # Safety
///
/// Rows `start .. start + count` of `rows` lie inside its buffers, 16-byte
/// aligned, and no thread writes them while the kernel runs; the block is
/// 128 threads.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn chunk_qk_regs(
    rows: CallRows,
    (start, count): (usize, usize),
    kh: usize,
) -> [[f32; 4]; 8] {
    let tid = thread::threadIdx_x() as usize;
    let mut v = [[0.0f32; 4]; 8];
    for i in 0usize..8 {
        cuda_device::thread::__unroll_config::<0>();
        let r = start + tid / 32 + 4 * (i % 4);
        if r < start + count && (i >= 4 || r >= rows.lead) {
            let head = if i < 4 {
                rows.query(r, kh)
            } else {
                rows.key(r, kh)
            };
            // SAFETY: four keys of a whole head of row r, at a multiple of
            // four floats from an aligned base, by the contract.
            v[i] = unsafe { ld4::<true>(head + 16 * (tid % 32) as u64) };
        }
    }
    v
}

/// Store [`chunk_qk_regs`]'s pieces as BF16 pairs (the residual when `low`):
/// the Q tile at `packed`, the K tile [`PTILE`] words on.
///
/// # Safety
///
/// `packed` addresses two tiles of shared memory the block owns that no
/// thread reads until the next barrier.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn chunk_qk_store(packed: *mut u32, v: [[f32; 4]; 8], low: bool) {
    let tid = thread::threadIdx_x() as usize;
    for i in 0usize..8 {
        cuda_device::thread::__unroll_config::<0>();
        let t = tid / 32 + 4 * (i % 4);
        let at = if i < 4 { 0 } else { PTILE } + t * PSTRIDE + 2 * (tid % 32);
        let p0 = if low {
            chunk_pair_low(v[i][0], v[i][1])
        } else {
            chunk_pair(v[i][0], v[i][1])
        };
        let p1 = if low {
            chunk_pair_low(v[i][2], v[i][3])
        } else {
            chunk_pair(v[i][2], v[i][3])
        };
        // SAFETY: tid < 128 gives t < CHUNK and 2*(tid%32) < PSTRIDE; this thread owns this pair
        // inside the two tiles.
        unsafe { *packed.add(at) = p0 };
        // SAFETY: tid < 128 gives t < CHUNK and 2*(tid%32)+1 < PSTRIDE; this thread owns this pair
        // inside the two tiles.
        unsafe { *packed.add(at + 1) = p1 };
    }
}

/// Transpose the complete K tile into pairs of consecutive time rows.
///
/// # Safety
///
/// `packed` holds two complete PTILE-word tiles with K at offset PTILE.
/// `dst .. dst + HEAD * 8` is writable shared memory disjoint from K, and
/// no thread reads the destination until the next block barrier.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn chunk_keys_by_time(packed: *mut u32, dst: usize) {
    let tid = thread::threadIdx_x() as usize;
    for i in 0usize..4 {
        cuda_device::thread::__unroll_config::<0>();
        let f = tid + 128 * i;
        let (kp, p) = (f % 64, f / 64);
        // SAFETY: f < 512 gives kp < 64 < PSTRIDE and p < 8, so row 2p < CHUNK: key pair kp of
        // that row of the complete K tile at PTILE.
        let w0 = unsafe { *packed.add(PTILE + 2 * p * PSTRIDE + kp) };
        // SAFETY: p < 8 gives row 2p + 1 < CHUNK and kp < 64 < PSTRIDE: key pair kp of that row of
        // the complete K tile at PTILE.
        let w1 = unsafe { *packed.add(PTILE + (2 * p + 1) * PSTRIDE + kp) };
        // SAFETY: kp < 64 and p < 8 bound this word of key 2kp within dst .. dst + HEAD*8, disjoint
        // from K; this thread alone writes it.
        unsafe { *packed.add(dst + 2 * kp * 8 + p) = (w0 & 0xffff) | (w1 << 16) };
        // SAFETY: kp < 64 and p < 8 bound this word of key 2kp + 1 within dst .. dst + HEAD*8,
        // disjoint from K; this thread alone writes it.
        unsafe { *packed.add(dst + (2 * kp + 1) * 8 + p) = (w0 >> 16) | (w1 & 0xffff_0000) };
    }
}

/// Shared words of `gdn_delta_prepared`: Q and K, the high keys by time,
/// the state tile, and high/residual planes of solved rows and both matrices.
/// The Q tile holds the residual keys by time after the state projections.
const PREPARED_WORDS: usize =
    2 * PTILE + HEAD * 8 + PREPARED_COLUMNS * PSTRIDE + 2 * PREPARED_COLUMNS * 8 + 4 * 128;

/// Shared f32s of `gdn_delta_prepared`: the key and query projections and the
/// values, `[CHUNK][PREPARED_COLUMNS]` each, then D, β and the final factors.
const PREPARED_FLOATS: usize = 3 * CHUNK * PREPARED_COLUMNS + 3 * CHUNK;

/// Words of `work` the prompt form takes for a call of `m` rows at `shape`
/// ([`DeltaKernels::enqueue_delta_prompt`]): the records of the most chunks
/// a call of `m` rows after a carry spans, then the lead word.
#[must_use]
pub fn prompt_work_len(shape: LinearShape, m: usize) -> usize {
    (m + CARRY_ROWS).div_ceil(CHUNK) * shape.n_v * CHUNK_WORDS + 1
}

/// Refuse by name a `qkv`, `state` or `carry` the kernels' 128-bit loads
/// ([`ld4`]) cannot read: each base must be 16-byte aligned. A window into a
/// larger buffer can start anywhere a whole f32 does.
fn check_vector_bases(
    what: &'static str,
    qkv: &DeviceBuffer<f32>,
    state: &DeviceBuffer<f32>,
    carry: Option<&DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    check_alignment::<U32x4, _>(what, "qkv", qkv)?;
    check_alignment::<U32x4, _>(what, "state", state)?;
    carry.map_or(Ok(()), |c| check_alignment::<U32x4, _>(what, "carry", c))
}

/// Refuse by name what `gdn_delta_lanes` and the prompt form cannot launch
/// with `carry` ([`DeltaKernels::enqueue_delta_lanes`]'s list).
fn check_gdn_lanes(
    what: &'static str,
    args: &DeltaLanesArgs<'_>,
    carry: Option<&DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let DeltaLanesArgs {
        delta: d,
        each,
        pos,
        stamp,
    } = args;
    d.shape.check(what)?;
    let (m, lanes) = (d.m, d.lanes);
    if m == 0 || lanes == 0 || (*each && m > lanes) {
        return Err(GpuError::shape(
            what,
            format!(
                "m={m} over {lanes} lanes{}: a call takes one token or more and a lane or \
                 more, and row mode at most a token a lane",
                if *each { " in row mode" } else { "" }
            ),
        ));
    }
    let nv = d.shape.n_v;
    let overflow = || GpuError::shape(what, "GDN launch size overflows usize");
    let channels = d
        .shape
        .n_k
        .checked_mul(2)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_mul(HEAD))
        .ok_or_else(overflow)?;
    let qkv = m.checked_mul(channels).ok_or_else(overflow)?;
    let heads = m.checked_mul(nv).ok_or_else(overflow)?;
    let output = heads.checked_mul(HEAD).ok_or_else(overflow)?;
    let state = lanes
        .checked_mul(nv)
        .and_then(|n| n.checked_mul(HEAD * HEAD))
        .ok_or_else(overflow)?;
    let lane = d.lane_at.checked_add(1).ok_or_else(overflow)?;
    let lens = [
        ("qkv", d.qkv.len(), qkv),
        ("beta", d.beta.len(), heads),
        ("decay", d.decay.len(), heads),
        ("lane", d.lane.len(), lane),
        ("pos", pos.len(), 1),
        ("o", d.o.len(), output),
        ("state", d.state.len(), state),
        ("stamp", stamp.len(), lanes),
    ];
    let carried = carry.map(|c| ("carry", c.len(), d.shape.carry_len()));
    if let Some((name, got, need)) = lens
        .into_iter()
        .chain(carried)
        .find(|(_, got, need)| got < need)
    {
        return Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        ));
    }
    check_vector_bases(what, d.qkv, d.state, carry)
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
    /// (`each != 0`, `m <= lanes`); with `carried`, the rows `carry` holds
    /// for `pos[0]` first (module doc, "Carry"), else `carry` is never read.
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
            stamp.len() >= lanes,
            carry.len() >= 4
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
        carry: &[f32],
        carried: u32,
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
        // divides n_v, `m <= lanes` in row mode, a carried launch's whole
        // carry, the block width and the grid.
        unsafe {
            delta_lanes_body::<DECAY_HEAD, false>(
                qkv,
                beta,
                decay,
                lane,
                lane_at,
                lanes,
                n_k,
                n_v,
                grouped,
                (m, each, 0),
                thread::blockIdx_x() as usize,
                pos,
                (carry, carried),
                fault,
                o,
                state,
                stamp,
            );
        }
    }

    /// A verify's compact records, or their accepted prefix replay.
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
            lanes >= 3,
            m * 3 <= 128,
            pos.len() >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384,
            stamp.len() >= lanes,
            carry.len() >= 4
        )
    )]
    pub fn gdn_delta_recovery(
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
        carry: &[f32],
        carried: u32,
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
        mut stamp: DisjointSlice<u32>,
    ) {
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        let stamp = stamp.as_mut_ptr();
        // SAFETY: the launcher checks three or more lanes, m <= lanes and
        // m * 3 <= HEAD, so both record banks fit outside the input state.
        // Each record word has one writer; replay reads a bank outside its
        // destination. The launch contract bounds every input and output;
        // block zero's thread zero alone reads and writes the stamps.
        unsafe {
            delta_lanes_body::<DECAY_HEAD, true>(
                qkv,
                beta,
                decay,
                lane,
                lane_at,
                lanes,
                n_k,
                n_v,
                grouped,
                (m, each, 0),
                thread::blockIdx_x() as usize,
                pos,
                (carry, carried),
                fault,
                o,
                state,
                stamp,
            );
        }
    }

    /// [`gdn_delta_recovery`]'s replay (`each == 0`) of `n_stores` stores in
    /// one launch ([`DeltaKernels::enqueue_delta_recovery_layers`]): blocks
    /// `s·per .. (s + 1)·per`, `per = 8·n_v`, are store `s`'s launch, block
    /// `b - s·per` of it, on the state, stamps, carry and fault layer of row
    /// `s` of `table` ([`RecoveryTable`]). A replay reads no `qkv`, `beta` or
    /// `decay` and writes no `o`, so the stores share none of them.
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
            lane.len() >= lane_at + 1,
            lanes >= 3,
            m * 3 <= 128,
            table.len() >= 4 * n_stores
        )
    )]
    pub fn gdn_delta_recovery_layers(
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        table: &[u64],
        n_stores: u32,
        carry_len: u32,
        fault: FaultSink,
    ) {
        let per = n_v as usize * (HEAD / COLUMNS_PER_BLOCK);
        let b = thread::blockIdx_x() as usize;
        let s = b / per;
        if s >= n_stores as usize {
            return; // block-uniform
        }
        // SAFETY: s < n_stores, so row s's RECOVERY_ROW words are inside the
        // table by the launch contract.
        let row = unsafe { table.get_unchecked(RECOVERY_ROW * s..RECOVERY_ROW * (s + 1)) };
        let &[state, stamp, carry, layer] = row else {
            return; // a row is RECOVERY_ROW words
        };
        let carried = u32::from(carry != 0);
        let carry = if carried != 0 {
            // SAFETY: RecoveryTable::new checked that a nonzero carry address
            // starts carry_len f32s, 16-byte aligned, that no state or stamp
            // of the table overlaps; its caller keeps them live and unwritten
            // by any launch while a replay runs (its Safety section).
            unsafe { core::slice::from_raw_parts(carry as usize as *const f32, carry_len as usize) }
        } else {
            &[]
        };
        // SAFETY: RecoveryTable::new checked the layer below LAYER_HEAD.
        let store_fault = unsafe { fault.at_layer(layer as u32) };
        // SAFETY: RecoveryTable::new checked row s's state (`lanes` lanes of
        // the shape, 16-byte aligned) and stamps (`lanes` words), and that
        // their byte ranges overlap no other state, stamp or carry of the
        // table; its caller keeps them live and touched by no other launch
        // while a replay runs (its Safety section). So each block's state
        // keys and its store's stamps have this launch's one writer as in
        // `gdn_delta_recovery`; the launcher checks three or more lanes and
        // m * 3 <= HEAD. The replay reads no qkv, beta or decay and writes no
        // o, so empty slices and a null o stand for them.
        unsafe {
            delta_lanes_body::<DECAY_HEAD, true>(
                &[],
                &[],
                &[],
                lane,
                lane_at,
                lanes,
                n_k,
                n_v,
                grouped,
                (m, 0, 0),
                b - s * per,
                &[],
                (carry, carried),
                store_fault,
                core::ptr::null_mut(),
                state as usize as *mut f32,
                stamp as usize as *mut u32,
            );
        }
    }

    /// The prompt form's first launch (module doc): block `b` reads chunk
    /// `b / n_v` of the call's rows (module doc, "Carry": the rows `carry`
    /// holds for `pos[0]`, then the call's `m`) for value head `b % n_v` and
    /// writes its [`CHUNK_WORDS`]-word record, record `b` of `work`, from
    /// those rows alone; a block past the call's chunks returns. Block 0's
    /// thread 0 writes the count of carried rows into the word past the
    /// `chunks · n_v` records, the lead word the next two launches read.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(128, 3)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            carry.len() >= 4 + 15 * ((n_k + n_v) * 128 + 2 * n_v),
            pos.len() >= 1,
            work.len() >= chunks * n_v * 560 + 1,
            chunks * 16 >= m + 15
        )
    )]
    pub fn gdn_chunk_prepare(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        carry: &[f32],
        pos: &[u32],
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        chunks: u32,
        fault: FaultSink,
        mut work: DisjointSlice<u32>,
    ) {
        // The f32 scratch: the key products (then the strictly lower matrix)
        // and the inverse, rows of 20; the query products and
        // the factors `G[t][j] = Dₜ/Dⱼ`, rows of 16; then D, β and the decays.
        const KK: usize = 0;
        const QK: usize = KK + CHUNK * 20;
        const INV: usize = QK + CHUNK * CHUNK;
        const G: usize = INV + CHUNK * 20;
        const D: usize = G + CHUNK * CHUNK;
        const B: usize = D + CHUNK;
        const DR: usize = B + CHUNK;
        static mut PACKED: SharedArray<u32, { 4 * PTILE }> = SharedArray::UNINIT;
        static mut SCRATCH: SharedArray<f32, { DR + CHUNK }> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let r = lane / 4;
        let c = lane % 4 * 2;
        let block = thread::blockIdx_x() as usize;
        let (nk, nv, m) = (n_k as usize, n_v as usize, m as usize);
        // SAFETY: the carry's header and pos[0], by the contract.
        let lead = match unsafe { carried_rows(carry, *pos.get_unchecked(0)) } {
            Some(n) => n,
            None => {
                // Fault the count and complete the records for the queued
                // walk.
                fault.raise(FaultSite::LinearDelta);
                0
            }
        };
        let all = lead + m;
        if block >= nv * all.div_ceil(CHUNK) {
            return;
        }
        let h = block % nv;
        let start = block / nv * CHUNK;
        let count = (all - start).min(CHUNK);
        let kh = if grouped != 0 { h / (nv / nk) } else { h % nk };
        let kv = (nk + nv) * HEAD;
        let rows = CallRows {
            qkv: global_addr(qkv.as_ptr()),
            old: global_addr(carry.as_ptr()),
            lead,
            ch: (2 * nk + nv) * HEAD,
            kv,
            nk,
        };
        let work = work.as_mut_ptr();
        if block == 0 && tid == 0 {
            // SAFETY: the lead word, inside work by the contract; this
            // thread alone writes it.
            unsafe { *work.add(chunks as usize * nv * CHUNK_WORDS) = lead as u32 };
        }
        // SAFETY: PACKED is this block's own shared allocation.
        let packed = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PACKED) };
        // SAFETY: SCRATCH is this block's own shared allocation.
        let sm = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        // SAFETY: the chunk's rows of the carry and qkv, by the contract.
        let qk = unsafe { chunk_qk_regs(rows, (start, count), kh) };
        // SAFETY: the high Q and K tiles at 0 .. 2*PTILE of PACKED's 4*PTILE words, read by no
        // thread before the following barrier.
        unsafe { chunk_qk_store(packed, qk, false) };
        // SAFETY: the residual Q and K tiles at 2*PTILE .. 4*PTILE of PACKED's 4*PTILE words, read
        // by no thread before the following barrier.
        unsafe { chunk_qk_store(packed.add(2 * PTILE), qk, true) };
        if tid < CHUNK {
            let r = start + tid;
            let (d, b) = if tid >= count {
                (1.0, 0.0)
            } else if r < lead {
                let at = carry_beta(r, h, kv, nv);
                // SAFETY: carried row r's β and decay of head h, inside
                // the carry by the contract.
                unsafe {
                    (
                        *carry.get_unchecked(at + CARRY_ROWS * nv),
                        *carry.get_unchecked(at),
                    )
                }
            } else {
                (decay[(r - lead) * nv + h], beta[(r - lead) * nv + h])
            };
            // SAFETY: row tid of the decays; one writer a row.
            unsafe { *sm.add(DR + tid) = d };
            // SAFETY: row tid of β; one writer a row.
            unsafe { *sm.add(B + tid) = b };
        }
        thread::sync_threads();
        if tid < CHUNK {
            let mut product = 1.0;
            let mut between = 1.0;
            for t in 0..CHUNK {
                // SAFETY: row t of the decays, complete past the barrier.
                let d = unsafe { *sm.add(DR + t) };
                product *= d;
                if tid == 0 {
                    // SAFETY: thread 0 alone writes D.
                    unsafe { *sm.add(D + t) = product };
                }
                if t > tid {
                    between *= d;
                }
                // SAFETY: thread tid alone writes column tid of G.
                unsafe { *sm.add(G + t * CHUNK + tid) = between };
            }
        }
        thread::sync_threads();
        let a = if warp < 2 { PTILE } else { 0 };
        let half = warp % 2 * 8;
        // SAFETY: the complete Q and K tiles.
        let dots = unsafe {
            chunk_product::<HEAD>(
                (packed.add(a), packed.add(2 * PTILE + a)),
                (
                    packed.add(PTILE + half * PSTRIDE),
                    packed.add(3 * PTILE + half * PSTRIDE),
                ),
                [0.0; 4],
            )
        };
        for i in 0..4 {
            let row = r + i / 2 * 8;
            let col = half + c + i % 2;
            let (dst, stride) = if warp < 2 { (KK, 20) } else { (QK, CHUNK) };
            // SAFETY: two warps a product, each its own half of the tile.
            unsafe { *sm.add(dst + row * stride + col) = dots[i] };
        }
        thread::sync_threads();
        // The strictly lower L[t][j] = −βₜ·G[t][j]·kₜ·kⱼ (j < t) over the key
        // products.
        let mut at = tid;
        while at < CHUNK * CHUNK {
            let t = at / CHUNK;
            let j = at % CHUNK;
            // SAFETY: the complete key products.
            let dot = unsafe { *sm.add(KK + t * 20 + j) };
            // SAFETY: the complete decay factors.
            let g = unsafe { *sm.add(G + at) };
            // SAFETY: row t of β.
            let b = unsafe { *sm.add(B + t) };
            let lower = if j < t { -b * g * dot } else { 0.0 };
            // SAFETY: each thread replaces the cells it read.
            unsafe { *sm.add(KK + t * 20 + j) = lower };
            at += 128;
        }
        let t = tid / 8;
        let j = tid % 8 * 2;
        // SAFETY: t < CHUNK and j < CHUNK bound this QK cell; the preceding barrier published the
        // query products.
        let p0 = unsafe { *sm.add(QK + t * CHUNK + j) };
        // SAFETY: t < CHUNK and j+1 < CHUNK bound this QK cell; the preceding barrier published
        // the query products.
        let p1 = unsafe { *sm.add(QK + t * CHUNK + j + 1) };
        // SAFETY: t < CHUNK and j < CHUNK bound this G cell; the preceding barrier published the
        // decay factors.
        let g0 = unsafe { *sm.add(G + t * CHUNK + j) };
        // SAFETY: t < CHUNK and j+1 < CHUNK bound this G cell; the preceding barrier published the
        // decay factors.
        let g1 = unsafe { *sm.add(G + t * CHUNK + j + 1) };
        let p = [
            if j <= t { p0 * g0 } else { 0.0 },
            if j < t { p1 * g1 } else { 0.0 },
        ];
        thread::sync_threads();
        // (I − L)⁻¹ by forward substitution in f32, column tid by thread tid.
        if tid < CHUNK {
            let mut x = [0.0f32; CHUNK];
            for i in 0..CHUNK {
                cuda_device::thread::__unroll_config::<0>();
                let mut acc = if i == tid { 1.0 } else { 0.0 };
                for q in 0..CHUNK {
                    cuda_device::thread::__unroll_config::<0>();
                    if q < i {
                        // SAFETY: the complete strictly lower matrix.
                        let l = unsafe { *sm.add(KK + i * 20 + q) };
                        acc = fma_rn_f32(l, x[q], acc);
                    }
                }
                x[i] = acc;
                // SAFETY: thread tid alone writes column tid of the inverse.
                unsafe { *sm.add(INV + i * 20 + tid) = acc };
            }
        }
        thread::sync_threads();
        // SAFETY: t < CHUNK and j < CHUNK bound this INV cell at stride 20; the preceding barrier
        // published the inverse.
        let i0 = unsafe { *sm.add(INV + t * 20 + j) };
        // SAFETY: t < CHUNK and j+1 < CHUNK bound this INV cell at stride 20; the preceding
        // barrier published the inverse.
        let i1 = unsafe { *sm.add(INV + t * 20 + j + 1) };
        let base = block * CHUNK_WORDS;
        // SAFETY: block < chunks·nv owns this record of work and tid < 128 < CHUNK_WORDS; thread
        // tid alone writes this inverse word.
        unsafe { *work.add(base + tid) = chunk_pair(i0, i1) };
        // SAFETY: block owns this record and 128+tid < CHUNK_WORDS; thread tid alone writes this
        // causal product word.
        unsafe { *work.add(base + 128 + tid) = chunk_pair(p[0], p[1]) };
        // SAFETY: block owns this record and 256+tid < CHUNK_WORDS; thread tid alone writes this
        // inverse residual word.
        unsafe { *work.add(base + 256 + tid) = chunk_pair_low(i0, i1) };
        // SAFETY: block owns this record and 384+tid < CHUNK_WORDS; thread tid alone writes this
        // causal residual word.
        unsafe { *work.add(base + 384 + tid) = chunk_pair_low(p[0], p[1]) };
        if tid < CHUNK {
            // SAFETY: tid < CHUNK bounds this D row; the preceding barriers published D.
            let d = unsafe { *sm.add(D + tid) };
            // SAFETY: tid < CHUNK bounds this β row; the preceding barriers published β.
            let b = unsafe { *sm.add(B + tid) };
            // SAFETY: 1 <= count <= CHUNK and tid < CHUNK bound this G cell of row count − 1; the
            // preceding barriers published G.
            let end = unsafe { *sm.add(G + (count - 1) * CHUNK + tid) };
            // SAFETY: block owns this record and 512+tid < CHUNK_WORDS; thread tid alone writes
            // this decay word.
            unsafe { *work.add(base + 512 + tid) = d.to_bits() };
            // SAFETY: block owns this record and 528+tid < CHUNK_WORDS; thread tid alone writes
            // this beta word.
            unsafe { *work.add(base + 528 + tid) = b.to_bits() };
            // SAFETY: block owns this record and 544+tid < CHUNK_WORDS; thread tid alone writes
            // this factor word.
            unsafe { *work.add(base + 544 + tid) = end.to_bits() };
        }
    }

    /// The prompt form's second launch (module doc): block `b` walks value
    /// head `b / 2`'s 64 columns `64·(b % 2) ..` through the call's chunks
    /// (module doc, "Carry"), the records `gdn_chunk_prepare` wrote into
    /// `work` for the same rows, in order, warp `w` the eight-column tiles
    /// `2·w` and `2·w + 1`, their state in registers from the first chunk to
    /// the last, a last chunk the call leaves incomplete left out of it.
    /// The lane and stamp rules are `gdn_delta_lanes`'s in place mode (block
    /// 0's thread 0 alone touches the stamps).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(128, 2)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            qkv.len() >= m * (2 * n_k + n_v) * 128,
            carry.len() >= 4 + 15 * ((n_k + n_v) * 128 + 2 * n_v),
            work.len() >= chunks * n_v * 560 + 1,
            chunks * 16 >= m + 15,
            lane.len() >= lane_at + 1,
            lanes >= 1,
            pos.len() >= 1,
            o.len() >= m * n_v * 128,
            state.len() >= lanes * n_v * 16384,
            stamp.len() >= lanes
        )
    )]
    pub fn gdn_delta_prepared(
        qkv: &[f32],
        carry: &[f32],
        work: &[u32],
        lane: &[u32],
        lane_at: u32,
        lanes: u32,
        n_k: u32,
        n_v: u32,
        grouped: u32,
        m: u32,
        chunks: u32,
        pos: &[u32],
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
        mut stamp: DisjointSlice<u32>,
    ) {
        static mut PACKED: SharedArray<u32, PREPARED_WORDS> = SharedArray::UNINIT;
        static mut SCRATCH: SharedArray<f32, PREPARED_FLOATS> = SharedArray::UNINIT;
        // Offsets into PACKED and SCRATCH, in the order PREPARED_WORDS and
        // PREPARED_FLOATS count them.
        const COLS: usize = PREPARED_COLUMNS;
        const KT: usize = 2 * PTILE;
        const S_TILE: usize = KT + HEAD * 8;
        const UB: usize = S_TILE + COLS * PSTRIDE;
        const IP: usize = UB + COLS * 8;
        const UB_LOW: usize = IP + 256;
        const IP_LOW: usize = UB_LOW + COLS * 8;
        const KS: usize = 0;
        const QS: usize = CHUNK * COLS;
        const U: usize = 2 * CHUNK * COLS;
        const D: usize = 3 * CHUNK * COLS;
        const B: usize = D + CHUNK;
        const END: usize = B + CHUNK;
        // SAFETY: PACKED is this block's own shared allocation.
        let packed = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PACKED) };
        // SAFETY: SCRATCH is this block's own shared allocation.
        let sm = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        let o = o.as_mut_ptr();
        let state = state.as_mut_ptr();
        let stamp = stamp.as_mut_ptr();
        let tid = thread::threadIdx_x() as usize;
        let lane_id = tid % 32;
        let r = lane_id / 4;
        let c = lane_id % 4 * 2;
        let wc = tid / 32 * 16;
        let block = thread::blockIdx_x() as usize;
        let h = block / 2;
        let col0 = block % 2 * COLS;
        let (nk, nv, m) = (n_k as usize, n_v as usize, m as usize);
        let kh = if grouped != 0 { h / (nv / nk) } else { h % nk };
        // SAFETY: the lead word, inside work by the contract.
        let lead = unsafe { *work.get_unchecked(chunks as usize * nv * CHUNK_WORDS) } as usize;
        let all = lead + m;
        let rows = CallRows {
            qkv: global_addr(qkv.as_ptr()),
            old: global_addr(carry.as_ptr()),
            lead,
            ch: (2 * nk + nv) * HEAD,
            kv: (nk + nv) * HEAD,
            nk,
        };
        let word = lane[lane_at as usize];
        let slot = (word % lanes) as usize;
        if block == 0 && tid == 0 {
            if word >= lanes {
                fault.raise(FaultSite::DeltaLane);
            }
            // SAFETY: slot < lanes, inside the stamps; block 0's thread 0 is
            // their only reader and writer in the launch.
            let at = unsafe { *stamp.add(slot) };
            if at != pos[0] {
                fault.raise(FaultSite::DeltaStamp);
            }
            // SAFETY: slot < lanes bounds this stamp; block 0's thread 0 alone writes it after
            // reading its old value.
            unsafe { *stamp.add(slot) = pos[0].wrapping_add(m as u32) };
        }
        let s_base = slot * nv * HEAD * HEAD + (h * HEAD + col0) * HEAD;
        let mut st = [[[0.0f32; 4]; 2]; 8];
        for tile in 0usize..8 {
            cuda_device::thread::__unroll_config::<0>();
            for n in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                for i in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = tile * 16 + r + i / 2 * 8;
                    let col = wc + n * 8 + c + i % 2;
                    // SAFETY: the grid gives each (key, column) of lane slot one
                    // thread, which reads it here and writes it after the walk.
                    st[tile][n][i] = unsafe { *state.add(s_base + col * HEAD + key) };
                }
            }
        }
        let mut raised = false;
        for chunk in 0..all.div_ceil(CHUNK) {
            let start = chunk * CHUNK;
            let count = (all - start).min(CHUNK);
            let rec = (chunk * nv + h) * CHUNK_WORDS;
            // Every global read of the chunk is issued before the first use.
            // SAFETY: the chunk's rows of the carry and qkv, by the contract.
            let qk = unsafe { chunk_qk_regs(rows, (start, count), kh) };
            let mut vr = [[0.0f32; 4]; 2];
            for i in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                let f = tid + 128 * i;
                if f / 16 < count {
                    let at = rows.value(start + f / 16, h * HEAD + col0 + 4 * (f % 16));
                    // SAFETY: four of the block's columns of the row, at a
                    // multiple of four floats from an aligned base.
                    vr[i] = unsafe { ld4::<true>(at) };
                }
            }
            let inv = work[rec + tid];
            let prod = work[rec + 128 + tid];
            let inv_low = work[rec + 256 + tid];
            let prod_low = work[rec + 384 + tid];
            let meta = if tid < 3 * CHUNK {
                work[rec + 512 + tid]
            } else {
                0
            };
            thread::sync_threads();
            // SAFETY: past the barrier no thread reads the previous chunk's tiles.
            unsafe { chunk_qk_store(packed, qk, false) };
            // SAFETY: IP+tid < PREPARED_WORDS; thread tid owns this inverse word, read only after
            // the following barrier.
            unsafe { *packed.add(IP + tid) = inv };
            // SAFETY: IP+128+tid < PREPARED_WORDS; thread tid owns this product word, read only
            // after the following barrier.
            unsafe { *packed.add(IP + 128 + tid) = prod };
            // SAFETY: IP_LOW+tid < PREPARED_WORDS; thread tid owns this inverse residual word, read
            // only after the following barrier.
            unsafe { *packed.add(IP_LOW + tid) = inv_low };
            // SAFETY: IP_LOW+128+tid < PREPARED_WORDS; thread tid owns this causal residual word,
            // read only after the following barrier.
            unsafe { *packed.add(IP_LOW + 128 + tid) = prod_low };
            if tid < 3 * CHUNK {
                // SAFETY: D, β and the final factors follow one another in sm.
                unsafe { *sm.add(D + tid) = f32::from_bits(meta) };
            }
            for i in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                let f = tid + 128 * i;
                for e in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: row f / 16 of the value tile; each thread owns four cells.
                    unsafe { *sm.add(U + f / 16 * COLS + 4 * (f % 16) + e) = vr[i][e] };
                }
            }
            // The three products share the Q/K and state staging tiles.
            for part in 0usize..3 {
                cuda_device::thread::__unroll_config::<0>();
                if part != 0 {
                    thread::sync_threads();
                }
                if part == 2 {
                    // SAFETY: the barrier above ended the earlier parts' reads of the Q and K
                    // tiles at 0 .. 2*PTILE; this thread owns its pairs, read only after the next
                    // barrier.
                    unsafe { chunk_qk_store(packed, qk, true) };
                }
                for tile in 0usize..8 {
                    cuda_device::thread::__unroll_config::<0>();
                    for n in 0usize..2 {
                        cuda_device::thread::__unroll_config::<0>();
                        for i in 0usize..4 {
                            cuda_device::thread::__unroll_config::<0>();
                            let key = tile * 16 + r + i / 2 * 8;
                            let col = wc + n * 8 + c + i % 2;
                            let v = st[tile][n][i];
                            let next = warp::shuffle_xor_f32(v, 4);
                            let pair = if part == 1 {
                                chunk_pair_low(v, next)
                            } else {
                                chunk_pair(v, next)
                            };
                            if r.is_multiple_of(2) {
                                // SAFETY: the even key row owns the pair (key, key + 1).
                                unsafe {
                                    *packed.add(S_TILE + col * PSTRIDE + key / 2) = pair;
                                }
                            }
                        }
                    }
                }
                thread::sync_threads();
                if part == 0 {
                    // SAFETY: the preceding barrier completed the K tile, and KT .. KT+HEAD*8
                    // lies past both PTILE-word tiles; no warp reads KT before the block's final
                    // barrier.
                    unsafe { chunk_keys_by_time(packed, KT) };
                }
                for n in 0usize..2 {
                    cuda_device::thread::__unroll_config::<0>();
                    let sb = S_TILE + (wc + n * 8) * PSTRIDE;
                    let mut kp = [0.0; 4];
                    let mut qp = [0.0; 4];
                    if part != 0 {
                        for i in 0usize..4 {
                            cuda_device::thread::__unroll_config::<0>();
                            let at = (r + i / 2 * 8) * COLS + wc + n * 8 + c + i % 2;
                            // SAFETY: at < CHUNK*COLS bounds this KS cell; this lane alone wrote
                            // it in the preceding part.
                            kp[i] = unsafe { *sm.add(KS + at) };
                            // SAFETY: at < CHUNK*COLS bounds this QS cell; this lane alone wrote
                            // it in the preceding part.
                            qp[i] = unsafe { *sm.add(QS + at) };
                        }
                    }
                    // SAFETY: the complete K and state tiles.
                    let kp = unsafe { chunk_mma::<HEAD>(packed.add(PTILE), packed.add(sb), kp) };
                    // SAFETY: the complete Q and state tiles.
                    let qp = unsafe { chunk_mma::<HEAD>(packed, packed.add(sb), qp) };
                    for i in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        let at = (r + i / 2 * 8) * COLS + wc + n * 8 + c + i % 2;
                        // SAFETY: at < CHUNK*COLS bounds this KS cell; this lane alone writes it
                        // before the following barrier.
                        unsafe { *sm.add(KS + at) = kp[i] };
                        // SAFETY: at < CHUNK*COLS bounds this QS cell; this lane alone writes it
                        // before the following barrier.
                        unsafe { *sm.add(QS + at) = qp[i] };
                    }
                }
            }
            thread::sync_threads();
            // SAFETY: the barrier above ended the state projections; K at PTILE holds its
            // residual, and 0 .. HEAD*8 lies inside the Q tile (HEAD*8 <= PTILE), which no thread
            // reads before the next barrier.
            unsafe { chunk_keys_by_time(packed, 0) };
            thread::sync_threads();
            // From here a warp reads and writes only its own sixteen columns.
            for i in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let f = lane_id + 32 * i;
                let (cb, p) = (wc + f / 8, f % 8);
                let mut pair = [0.0f32; 2];
                for e in 0usize..2 {
                    cuda_device::thread::__unroll_config::<0>();
                    let t = 2 * p + e;
                    // SAFETY: row t of β, complete past the barrier.
                    let b = unsafe { *sm.add(B + t) };
                    // SAFETY: row t of the warp's own value column.
                    let v = unsafe { *sm.add(U + t * COLS + cb) };
                    // SAFETY: row t of D, complete past the barrier.
                    let d = unsafe { *sm.add(D + t) };
                    // SAFETY: row t of the warp's own key projection column.
                    let kp = unsafe { *sm.add(KS + t * COLS + cb) };
                    pair[e] = b * (v - d * kp);
                }
                // SAFETY: the warp's own packed column.
                unsafe { *packed.add(UB + cb * 8 + p) = chunk_pair(pair[0], pair[1]) };
                // SAFETY: cb < COLS and p < 8 bound this UB_LOW word within COLS*8; the warp owns
                // column cb, read only after the following warp sync.
                unsafe { *packed.add(UB_LOW + cb * 8 + p) = chunk_pair_low(pair[0], pair[1]) };
            }
            warp::sync_mask(u32::MAX);
            for n in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: the inverse and the warp's packed residuals.
                let u = unsafe {
                    chunk_product::<16>(
                        (packed.add(IP), packed.add(IP_LOW)),
                        (
                            packed.add(UB + (wc + n * 8) * 8),
                            packed.add(UB_LOW + (wc + n * 8) * 8),
                        ),
                        [0.0; 4],
                    )
                };
                for i in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: row r + 8·(i/2) < CHUNK and column wc + n·8 + c + i%2 < COLS: one of
                    // the warp's own U cells, whose old value the warp read before the warp sync.
                    unsafe { *sm.add(U + (r + i / 2 * 8) * COLS + wc + n * 8 + c + i % 2) = u[i] };
                }
            }
            warp::sync_mask(u32::MAX);
            for i in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let f = lane_id + 32 * i;
                let (cb, p) = (wc + f / 8, f % 8);
                // SAFETY: p < 8 gives row 2p < CHUNK and cb < COLS is one of the warp's own
                // columns, written by this warp before the preceding warp sync.
                let u0 = unsafe { *sm.add(U + 2 * p * COLS + cb) };
                // SAFETY: p < 8 gives row 2p + 1 < CHUNK and cb < COLS is one of the warp's own
                // columns, written by this warp before the preceding warp sync.
                let u1 = unsafe { *sm.add(U + (2 * p + 1) * COLS + cb) };
                // SAFETY: the warp's own packed column.
                unsafe { *packed.add(UB + cb * 8 + p) = chunk_pair(u0, u1) };
                // SAFETY: cb < COLS and p < 8 bound this UB_LOW word within COLS*8; the warp owns
                // column cb, read only after the following warp sync.
                unsafe { *packed.add(UB_LOW + cb * 8 + p) = chunk_pair_low(u0, u1) };
            }
            warp::sync_mask(u32::MAX);
            for n in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                let mut acc = [0.0f32; 4];
                for i in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let t = r + i / 2 * 8;
                    // SAFETY: row t of D, complete past the barrier.
                    let d = unsafe { *sm.add(D + t) };
                    // SAFETY: a cell of the warp's own query projection columns.
                    let qp = unsafe { *sm.add(QS + t * COLS + wc + n * 8 + c + i % 2) };
                    acc[i] = d * qp;
                }
                // SAFETY: the causal product and the warp's packed solved rows.
                let y = unsafe {
                    chunk_product::<16>(
                        (packed.add(IP + 128), packed.add(IP_LOW + 128)),
                        (
                            packed.add(UB + (wc + n * 8) * 8),
                            packed.add(UB_LOW + (wc + n * 8) * 8),
                        ),
                        acc,
                    )
                };
                for i in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let t = r + i / 2 * 8;
                    // A carried row's output is the carrying call's.
                    if t < count && start + t >= lead {
                        if !y[i].is_finite() && !raised {
                            fault.raise(FaultSite::LinearDelta);
                            raised = true;
                        }
                        let row = start + t - lead;
                        // SAFETY: row < m, a column of head h: inside o, one
                        // lane a cell.
                        unsafe {
                            *o.add((row * nv + h) * HEAD + col0 + wc + n * 8 + c + i % 2) = y[i]
                        };
                    }
                }
            }
            // A chunk the call leaves incomplete stays out of the state: the
            // carry holds its rows for the call that completes it.
            if count < CHUNK {
                continue;
            }
            warp::sync_mask(u32::MAX);
            for i in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let f = lane_id + 32 * i;
                let (cb, p) = (wc + f / 8, f % 8);
                // SAFETY: p < 8 gives row 2p < CHUNK and cb < COLS is one of the warp's own
                // columns, written by this warp before the preceding warp sync.
                let u0 = unsafe { *sm.add(U + 2 * p * COLS + cb) };
                // SAFETY: p < 8 gives row 2p + 1 < CHUNK and cb < COLS is one of the warp's own
                // columns, written by this warp before the preceding warp sync.
                let u1 = unsafe { *sm.add(U + (2 * p + 1) * COLS + cb) };
                // SAFETY: p < 8 gives row 2p < CHUNK of the final factors, published by the
                // chunk's first barriers.
                let e0 = unsafe { *sm.add(END + 2 * p) };
                // SAFETY: p < 8 gives row 2p + 1 < CHUNK of the final factors, published by the
                // chunk's first barriers.
                let e1 = unsafe { *sm.add(END + 2 * p + 1) };
                let (u0, u1) = (u0 * e0, u1 * e1);
                // SAFETY: the warp's own packed column, read by no MMA in flight.
                unsafe { *packed.add(UB + cb * 8 + p) = chunk_pair(u0, u1) };
                // SAFETY: cb < COLS and p < 8 bound this UB_LOW word within COLS*8; the warp owns
                // column cb, and no MMA in flight reads it.
                unsafe { *packed.add(UB_LOW + cb * 8 + p) = chunk_pair_low(u0, u1) };
            }
            warp::sync_mask(u32::MAX);
            // SAFETY: the chunk's last row of D.
            let d = unsafe { *sm.add(D + CHUNK - 1) };
            for tile in 0usize..8 {
                cuda_device::thread::__unroll_config::<0>();
                for n in 0usize..2 {
                    cuda_device::thread::__unroll_config::<0>();
                    for i in 0usize..4 {
                        cuda_device::thread::__unroll_config::<0>();
                        st[tile][n][i] *= d;
                    }
                    // SAFETY: the keys by time and the warp's decayed rows.
                    st[tile][n] = unsafe {
                        chunk_product::<16>(
                            (packed.add(KT + tile * 16 * 8), packed.add(tile * 16 * 8)),
                            (
                                packed.add(UB + (wc + n * 8) * 8),
                                packed.add(UB_LOW + (wc + n * 8) * 8),
                            ),
                            st[tile][n],
                        )
                    };
                    if !quad_finite(st[tile][n]) && !raised {
                        fault.raise(FaultSite::LinearDelta);
                        raised = true;
                    }
                }
            }
        }
        for tile in 0usize..8 {
            cuda_device::thread::__unroll_config::<0>();
            for n in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                for i in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = tile * 16 + r + i / 2 * 8;
                    let col = wc + n * 8 + c + i % 2;
                    // SAFETY: the cell this thread read before the walk.
                    unsafe { *state.add(s_base + col * HEAD + key) = st[tile][n][i] };
                }
            }
        }
    }

    /// The prompt form's last launch (module doc, "Carry"): the rows of the
    /// last chunk of the call's rows into `carry` when the call leaves it
    /// incomplete. Slot `s` holds the chunk's row `s`; a slot below the
    /// carried rows the chunk starts with keeps its row, a slot past the
    /// chunk's rows is zero. Block `b < n_k + n_v` writes head `b` of the
    /// rows (the key heads, then the value heads), block `n_k + n_v` their β
    /// and decays and, on thread 0, the header: the count and `pos[0] + m`.
    /// It reads the lead word `gdn_chunk_prepare` wrote, never the carry, so
    /// no block reads what another writes.
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
            work.len() >= chunks * n_v * 560 + 1,
            chunks * 16 >= m + 15,
            pos.len() >= 1,
            carry.len() >= 4 + 15 * ((n_k + n_v) * 128 + 2 * n_v)
        )
    )]
    pub fn gdn_carry_keep(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        work: &[u32],
        pos: &[u32],
        n_k: u32,
        n_v: u32,
        m: u32,
        chunks: u32,
        mut carry: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let b = thread::blockIdx_x() as usize;
        let (nk, nv, m) = (n_k as usize, n_v as usize, m as usize);
        let (ch, kv) = ((2 * nk + nv) * HEAD, (nk + nv) * HEAD);
        // SAFETY: the lead word, inside work by the contract.
        let lead = unsafe { *work.get_unchecked(chunks as usize * nv * CHUNK_WORDS) } as usize;
        let all = lead + m;
        // The last chunk starts at call row `last`; it holds `held` rows when
        // incomplete, and slots below `from` the rows the call carried in.
        let last = (all - 1) / CHUNK * CHUNK;
        let held = if all - last < CHUNK { all - last } else { 0 };
        let from = lead.saturating_sub(last);
        let carry = carry.as_mut_ptr();
        if b < nk + nv {
            for s in 0..CARRY_ROWS {
                if s >= from || s >= held {
                    let v = if s < held {
                        // SAFETY: word tid of head b of qkv row last + s −
                        // lead < m: the key heads (nk + b < 2·nk), then the
                        // value heads past them.
                        unsafe {
                            *qkv.get_unchecked((last + s - lead) * ch + (nk + b) * HEAD + tid)
                        }
                    } else {
                        0.0
                    };
                    // SAFETY: slot s < CARRY_ROWS, head b: inside the carry,
                    // this thread's word alone.
                    unsafe { *carry.add(carry_row(s, kv) + b * HEAD + tid) = v };
                }
            }
        } else if b == nk + nv {
            let mut f = tid;
            while f < CARRY_ROWS * nv {
                let (s, h) = (f / nv, f % nv);
                if s >= from || s >= held {
                    let (bt, dt) = if s < held {
                        let at = (last + s - lead) * nv + h;
                        (beta[at], decay[at])
                    } else {
                        (0.0, 0.0)
                    };
                    let at = carry_beta(s, h, kv, nv);
                    // SAFETY: slot s's β of head h, inside the carry; this
                    // thread's alone.
                    unsafe { *carry.add(at) = bt };
                    // SAFETY: slot s < CARRY_ROWS and h < nv put this decay word CARRY_ROWS·nv
                    // past the β word, inside the carry; this thread's alone.
                    unsafe { *carry.add(at + CARRY_ROWS * nv) = dt };
                }
                f += 128;
            }
            if tid == 0 {
                // SAFETY: word 0 is inside CARRY_HEAD; thread 0 of block nk+nv alone writes this
                // header word.
                unsafe { *carry = f32::from_bits(held as u32) };
                // SAFETY: word 1 is inside CARRY_HEAD; thread 0 of block nk+nv alone writes this
                // header word.
                unsafe { *carry.add(1) = f32::from_bits(pos[0].wrapping_add(m as u32)) };
            }
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
            delta_lanes_body::<DECAY_KEY, false>(
                qkv,
                beta,
                decay,
                lane,
                lane_at,
                lanes,
                n_k,
                n_v,
                grouped,
                (m, each, row),
                thread::blockIdx_x() as usize,
                pos,
                (qkv, 0),
                fault,
                o,
                state,
                stamp,
            );
        }
    }
}

/// The kernels' `grouped` argument for `map`: 1 for grouped value heads, 0 for tiled ones.
fn grouped_arg(map: KHeadMap) -> u32 {
    u32::from(map == KHeadMap::Grouped)
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
        check_vector_bases(what, qkv, state, None)?;
        let grid = launch_u32(what, "grid", nv * (HEAD / COLUMNS_PER_BLOCK))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", nv)?;
        let m = launch_u32(what, "m", m)?;
        let lane_at = launch_u32(what, "lane_at", lane_at)?;
        let lanes = launch_u32(what, "lanes", lanes)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let map = grouped_arg(shape.map);
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

/// The host rule of the rows `carry` holds for position `p0` (module doc,
/// "Carry") taken into `lane`, one lane of `shape`'s state: what
/// `gdn_delta_lanes` from `p0` does to its lane before its first token,
/// [`delta_host`] over the carried rows; `lane` itself when the carry holds
/// none for `p0`.
///
/// # Panics
///
/// When `carry` is shorter than `shape`'s carry, its count exceeds
/// [`CARRY_ROWS`], or `lane` is shorter than a lane.
#[must_use]
pub fn carry_fold_host(carry: &[f32], lane: &[f32], p0: u32, shape: LinearShape) -> Vec<f32> {
    let Some(held) = held_rows(carry[0].to_bits() as usize, carry[1].to_bits(), p0) else {
        panic!("carry_fold_host: carry count exceeds CARRY_ROWS")
    };
    if held == 0 {
        return lane.to_vec();
    }
    let (n_k, n_v) = (shape.n_k, shape.n_v);
    let kv = (n_k + n_v) * HEAD;
    let mut qkv = Vec::with_capacity(held * shape.channels());
    let (mut beta, mut decay) = (Vec::new(), Vec::new());
    for t in 0..held {
        // A carried row keeps no query: its output is not this rule's.
        qkv.extend(std::iter::repeat_n(0.0, n_k * HEAD));
        qkv.extend_from_slice(&carry[carry_row(t, kv)..carry_row(t + 1, kv)]);
        let b = carry_beta(t, 0, kv, n_v);
        beta.extend_from_slice(&carry[b..b + n_v]);
        let d = b + CARRY_ROWS * n_v;
        decay.extend_from_slice(&carry[d..d + n_v]);
    }
    delta_host(&qkv, &beta, &decay, lane, 1, 0, shape, held).state
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

/// The lane word after a verify from the committed lane `c` keeps its first
/// `k` rows (`k >= 1`): the lane row `k − 1` wrote, `row_lane(c, k − 1)`.
/// A kept count of 0 has no lane (the verify's row 0 always stays), and is
/// refused by name.
///
/// # Panics
///
/// On `k == 0`, by name (in a const context, a compile error).
#[must_use]
pub const fn kept_lane(c: u32, k: u32, lanes: u32) -> u32 {
    assert!(k >= 1, "kept_lane: a verify keeps at least its first row");
    row_lane(c, k - 1, lanes)
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
/// `c = lane[lane_at] mod lanes`, as the call's block `b` (head `b / 8`),
/// with the stamp check and writes and the carry's count check on block 0's
/// thread 0 and the state stored per row
/// (`each != 0`) or once after the last token. For `DECAY_HEAD` with `carried`, the rows `carry` holds for
/// `pos[0]` come first, their outputs not written (module doc, "Carry").
///
/// # Safety
///
/// [`delta_body`]'s contract for `DECAY`, and `pos` holds a word unless
/// `RECOVERY && each == 0`. `stamp` addresses `lanes` writable u32s no
/// other launch touches, `row == 0` when `each == 0` or `carried != 0`,
/// `row + m <= lanes` when `each != 0`, and with `carried` `carry` is a
/// whole carry of the shape no launch writes while this one runs.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn delta_lanes_body<const DECAY: u32, const RECOVERY: bool>(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    lane_word: &[u32],
    lane_at: u32,
    lanes: u32,
    n_k: u32,
    n_v: u32,
    grouped: u32,
    (m, each, row): (u32, u32, u32),
    b: usize,
    pos: &[u32],
    (carry, carried): (&[f32], u32),
    fault: FaultSink,
    o: *mut f32,
    state: *mut f32,
    stamp: *mut u32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let j = lane % LANES_PER_COLUMN;
    let blocks_per_head = HEAD / COLUMNS_PER_BLOCK;
    let h = b / blocks_per_head;
    let (n_k, n_v, m) = (n_k as usize, n_v as usize, m as usize);
    // SAFETY: lane_at < lane_word.len(), checked by the launcher.
    let word = unsafe { *lane_word.get_unchecked(lane_at as usize) };
    if word >= lanes && tid == 0 {
        fault.raise(FaultSite::DeltaLane);
    }
    let c = word % lanes;
    let target = if RECOVERY && each == 0 {
        row_lane(c, m as u32 - 1, lanes)
    } else {
        c
    };
    let record_lane = if RECOVERY && each == 0 && target == row_lane(c, 1, lanes) {
        row_lane(c, 2, lanes)
    } else {
        row_lane(c, 1, lanes)
    };
    let lane_len = n_v * HEAD * HEAD;
    let recovery_at = record_lane as usize * lane_len;
    let p0 = if RECOVERY && each == 0 {
        // SAFETY: the first record's unused decay words hold its position
        // and row count, written once before replay on this stream.
        let p = unsafe { *state.add(recovery_at + 2 * HEAD + 1).cast::<u32>() };
        // SAFETY: record_lane < lanes and 2·HEAD + 2 < lane_len put the first record's count word
        // inside the state; it was written once before replay on this stream.
        let rows = unsafe { *state.add(recovery_at + 2 * HEAD + 2).cast::<u32>() };
        if rows == 0 || rows as usize > HEAD / 3 || m > rows as usize {
            if tid == 0 {
                fault.raise(FaultSite::DeltaStamp);
            }
            return;
        }
        p
    } else {
        // SAFETY: pos holds at least one word, checked by the launcher.
        unsafe { *pos.get_unchecked(0) }
    };
    if RECOVERY && each != 0 && b == 0 && tid == 0 {
        for bank in 1..=2 {
            let at = row_lane(c, bank, lanes) as usize * lane_len + 2 * HEAD;
            // SAFETY: the first record's position word, outside its key,
            // update and head decay words; only this thread writes it.
            unsafe { *state.add(at + 1).cast::<u32>() = p0 };
            // SAFETY: the bank's lane < lanes and 2·HEAD + 2 < lane_len put the first record's
            // count word inside the state; only this thread writes it.
            unsafe { *state.add(at + 2).cast::<u32>() = m as u32 };
        }
    }
    let rin = if row == 0 { c } else { (c + row - 1) % lanes };
    if b == 0 && tid == 0 {
        // SAFETY: rin < lanes, inside the stamps; only this thread reads
        // or writes stamps, and it reads the input before writing.
        let stamped = unsafe { *stamp.add(rin as usize) };
        if stamped != p0 {
            fault.raise(FaultSite::DeltaStamp);
        }
        if !RECOVERY || each == 0 {
            if each != 0 {
                let mut t = 0u32;
                while (t as usize) < m {
                    let at = (c + row + t) % lanes;
                    // SAFETY: at < lanes, and this thread owns the stamps.
                    unsafe { *stamp.add(at as usize) = p0.wrapping_add(t + 1) };
                    t += 1;
                }
            } else {
                // SAFETY: target < lanes, and this thread owns the stamps.
                unsafe { *stamp.add(target as usize) = p0.wrapping_add(m as u32) };
            }
        }
        if DECAY == DECAY_HEAD && carried != 0 {
            // SAFETY: the carry holds the launch-wide header checked by the launcher.
            if unsafe { carried_rows(carry, p0) }.is_none() {
                fault.raise(FaultSite::LinearDelta);
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
    let col_at = (b * COLUMNS_PER_BLOCK + tid / LANES_PER_COLUMN) * HEAD;
    let s_at = rin as usize * lane_len + col_at;

    // SAFETY: as in `delta_body`: the rows read are inside qkv and the
    // column inside lane rin of the state (`rin < lanes`), 16-byte aligned;
    // the state is read before this thread writes its keys of any lane, and
    // no other thread touches them.
    let (qkv_g, state_g) = (global_addr(qkv.as_ptr()), global_addr(state.cast_const()));
    // SAFETY: s_at + HEAD <= lanes·n_v·HEAD·HEAD, the column inside state.
    let mut s = unsafe { lane_keys::<false>(state_g + 4 * s_at as u64, j) };
    if DECAY == DECAY_HEAD && carried != 0 {
        let kv = (n_k + n_v) * HEAD;
        let carry_g = global_addr(carry.as_ptr());
        // SAFETY: the carry's header, by the contract.
        let held = unsafe { carried_rows(carry, p0) }.unwrap_or(0);
        let mut t = 0usize;
        while t < held {
            let at = carry_row(t, kv);
            // SAFETY: row t < CARRY_ROWS of the carry: this lane's sixteen
            // keys of key head kh, 16-byte aligned.
            let k = unsafe { lane_keys::<true>(carry_g + 4 * (at + kh * HEAD) as u64, j) };
            let b_at = carry_beta(t, h, kv, n_v);
            // SAFETY: this row's value column is inside the carry.
            let vt = unsafe { *carry.get_unchecked(at + n_k * HEAD + h * HEAD + col) };
            // SAFETY: b_at addresses this head's beta inside the carry.
            let bt = unsafe { *carry.get_unchecked(b_at) };
            // SAFETY: this head's decay lies after the carry's beta plane.
            let dh = unsafe { *carry.get_unchecked(b_at + CARRY_ROWS * n_v) };
            s = rule_row::<DECAY_HEAD>(s, k, [0.0; KEYS_PER_LANE], dh, (vt, bt));
            t += 1;
        }
    }
    let mut raised = false;
    let mut t = 0usize;
    while t < m {
        let rec = recovery_at + (t * n_v + h) * 3 * HEAD;
        if RECOVERY && each == 0 {
            // SAFETY: the aligned key head is inside the spare lane; the
            // recording launch finished before this launch.
            let k = unsafe { lane_keys::<false>(state_g + 4 * rec as u64, j) };
            // SAFETY: col < HEAD, inside this record's update vector.
            let u = unsafe { *state.add(rec + HEAD + col) };
            // SAFETY: the head decay is the first word of the record's tail.
            let dh = unsafe { *state.add(rec + 2 * HEAD) };
            s = update_row(decay_row::<DECAY_HEAD>(s, [0.0; KEYS_PER_LANE], dh), k, u);
            t += 1;
            continue;
        }
        let row_at = t * ch;
        // SAFETY: this aligned key head lies inside qkv's row t < m.
        let k = unsafe { lane_keys::<true>(qkv_g + 4 * (row_at + k_at) as u64, j) };
        // SAFETY: this aligned query head lies inside qkv's row t < m.
        let q = unsafe { lane_keys::<true>(qkv_g + 4 * (row_at + q_at) as u64, j) };
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
        // SAFETY: row_at + v_at < m * ch, inside qkv.
        let vt = unsafe { *qkv.get_unchecked(row_at + v_at) };
        // SAFETY: t * n_v + h < m * n_v, inside beta.
        let bt = unsafe { *beta.get_unchecked(t * n_v + h) };
        let dh = if DECAY == DECAY_HEAD {
            // SAFETY: t * n_v + h < m * n_v, inside per-head decay.
            unsafe { *decay.get_unchecked(t * n_v + h) }
        } else {
            0.0
        };
        let (next, u) = rule_update::<DECAY>(s, k, dk, dh, (vt, bt));
        s = next;
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
        if RECOVERY {
            for bank in 1..=2 {
                let rec = row_lane(c, bank, lanes) as usize * lane_len + (t * n_v + h) * 3 * HEAD;
                if col == 0 {
                    for i in 0..KEYS_PER_LANE {
                        cuda_device::thread::__unroll_config::<0>();
                        // SAFETY: column zero's lanes own the aligned record's
                        // keys, each written once inside the spare lane.
                        unsafe { *state.add(rec + key_of(j, i)) = k[i] };
                    }
                    if j == 0 {
                        // SAFETY: one thread writes the record's head decay.
                        unsafe { *state.add(rec + 2 * HEAD) = dh };
                    }
                }
                if j == 0 {
                    // SAFETY: each value column writes its own update scalar.
                    unsafe { *state.add(rec + HEAD + col) = u };
                }
            }
        } else if each != 0 {
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
        let out_at = target as usize * lane_len + col_at;
        for i in 0..KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: target < lanes; this thread owns these keys of its
            // column, and the record bank lies outside the target lane.
            unsafe { *state.add(out_at + key_of(j, i)) = s[i] };
        }
    }
}

/// Words of a [`RecoveryTable`] row: the store's state, stamp and carry
/// device addresses (0 for a store with no carry) and its fault layer.
const RECOVERY_ROW: usize = runtime::stores::DELTA_REPLAY_WORDS;

/// One store a [`RecoveryTable`] names: the buffers of its
/// [`DeltaKernels::enqueue_delta_recovery`] call and the layer its faults
/// carry.
pub struct RecoveryStore<'a> {
    pub layer: usize,
    pub state: &'a DeviceBuffer<f32>,
    pub stamp: &'a DeviceBuffer<u32>,
    pub carry: Option<&'a DeviceBuffer<f32>>,
}

/// The stores one [`DeltaKernels::enqueue_delta_recovery_layers`] launch
/// replays, a row of [`RECOVERY_ROW`] words each, uploaded once.
pub struct RecoveryTable {
    words: DeviceBuffer<u64>,
    writes: Vec<(u64, u64)>,
    stores: usize,
    lanes: usize,
    shape: LinearShape,
}

impl RecoveryTable {
    /// The table of `stores`, each of `lanes` lanes of `shape`. Refused by
    /// name: no store, fewer than three lanes, a state, stamp or carry
    /// shorter than a replay reads, a state or carry not 16-byte aligned,
    /// a state or stamp whose bytes a replay touches overlap another
    /// state's, stamp's or carry's (its own store's included), and a layer
    /// a fault word cannot carry ([`crate::LAYER_HEAD`] or past it). Buffer
    /// handles may be non-owning windows; distinct base addresses alone do
    /// not prove that their memory ranges are disjoint, so the check is on
    /// byte ranges. Load-time only.
    ///
    /// # Safety
    ///
    /// The table holds the buffers' addresses, not borrows: every buffer
    /// stays allocated at its address while any launch or captured graph
    /// using the table can run. No other launch reads or writes a store's
    /// state or stamps while a replay runs, and no launch writes its carry
    /// while a replay runs.
    pub unsafe fn new(
        stream: &CudaStream,
        shape: LinearShape,
        lanes: usize,
        stores: &[RecoveryStore<'_>],
    ) -> Result<RecoveryTable, GpuError> {
        const WHAT: &str = "RecoveryTable::new";
        shape.check(WHAT)?;
        let refuse = |detail: String| Err(GpuError::shape(WHAT, detail));
        if stores.is_empty() || lanes < 3 {
            return refuse(format!(
                "{} stores of {lanes} lanes: need a store and 3 or more lanes",
                stores.len()
            ));
        }
        let state = lanes
            .checked_mul(shape.state_len())
            .ok_or_else(|| GpuError::shape(WHAT, "state size overflows usize"))?;
        let mut words = Vec::with_capacity(RECOVERY_ROW * stores.len());
        // Byte ranges a replay touches: (store, name, start, end, written).
        let mut spans: Vec<(usize, &str, u64, u64, bool)> = Vec::with_capacity(3 * stores.len());
        for (i, st) in stores.iter().enumerate() {
            let lens = [
                ("state", st.state.len(), state),
                ("stamp", st.stamp.len(), lanes),
            ];
            let carried = st.carry.map(|c| ("carry", c.len(), shape.carry_len()));
            if let Some((name, got, need)) = lens
                .into_iter()
                .chain(carried)
                .find(|(_, got, need)| got < need)
            {
                return refuse(format!("store {i}: {name}.len() {got} < {need}"));
            }
            check_alignment::<U32x4, _>(WHAT, "state", st.state)?;
            if let Some(c) = st.carry {
                check_alignment::<U32x4, _>(WHAT, "carry", c)?;
            }
            if st.layer >= crate::LAYER_HEAD as usize {
                return refuse(format!(
                    "store {i}: layer {} at or past the head's fault layer {}",
                    st.layer,
                    crate::LAYER_HEAD
                ));
            }
            // Each count is at most its buffer's len (checked above), so each
            // range lies inside its buffer and cannot wrap.
            let at = |p: u64, n: usize| (p, p + 4 * n as u64);
            let mine = [
                ("state", at(st.state.cu_deviceptr(), state), true),
                ("stamp", at(st.stamp.cu_deviceptr(), lanes), true),
            ]
            .into_iter()
            .chain(
                st.carry
                    .map(|c| ("carry", at(c.cu_deviceptr(), shape.carry_len()), false)),
            );
            for (name, (start, end), written) in mine {
                if let Some(&(j, other, ..)) = spans
                    .iter()
                    .find(|&&(_, _, s, e, w)| (written || w) && start < e && s < end)
                {
                    return refuse(format!(
                        "store {i}: its {name} overlaps store {j}'s {other}"
                    ));
                }
                spans.push((i, name, start, end, written));
            }
            words.extend([
                st.state.cu_deviceptr(),
                st.stamp.cu_deviceptr(),
                st.carry.map_or(0, DeviceBuffer::cu_deviceptr),
                st.layer as u64,
            ]);
        }
        Ok(RecoveryTable {
            words: DeviceBuffer::from_host(stream, &words)?,
            writes: spans
                .into_iter()
                .filter_map(|(_, _, start, end, written)| written.then_some((start, end)))
                .collect(),
            stores: stores.len(),
            lanes,
            shape,
        })
    }

    /// Device bytes: [`RECOVERY_ROW`] u64 words a store.
    pub(crate) fn bytes(&self) -> usize {
        self.words.num_bytes()
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
    /// Enqueue the in-place call `args` in the prompt form (module doc):
    /// `gdn_chunk_prepare` over `⌈(m + 15) / 16⌉·n_v` blocks writing their
    /// records and the lead word into `work`, `gdn_delta_prepared` over
    /// `2·n_v` blocks, which takes the rows `carry` holds for `pos[0]` first,
    /// then `gdn_carry_keep` over `n_k + n_v + 1` blocks, which leaves in
    /// `carry` the rows of a chunk the call does not complete (module doc,
    /// "Carry"). Refused by name: what `enqueue_delta_lanes` refuses, row
    /// mode, a carry shorter than the shape's and a `work` shorter than
    /// [`prompt_work_len`]. Asynchronous, allocation-free.
    pub fn enqueue_delta_prompt(
        &self,
        stream: &CudaStream,
        args: DeltaLanesArgs<'_>,
        carry: &mut DeviceBuffer<f32>,
        work: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_delta_prompt";
        if args.each {
            return Err(GpuError::shape(
                WHAT,
                "row mode: the prompt form runs a call in place",
            ));
        }
        check_gdn_lanes(WHAT, &args, Some(&*carry))?;
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
            each: _,
            pos,
            stamp,
        } = args;
        let need = prompt_work_len(shape, m);
        if work.len() < need {
            return Err(GpuError::shape(
                WHAT,
                format!("work.len() {} < {need}", work.len()),
            ));
        }
        let chunks = (m + CARRY_ROWS).div_ceil(CHUNK);
        let nk = launch_u32(WHAT, "n_k", shape.n_k)?;
        let nv = launch_u32(WHAT, "n_v", shape.n_v)?;
        let map = grouped_arg(shape.map);
        let rows = launch_u32(WHAT, "m", m)?;
        let count = launch_u32(WHAT, "chunks", chunks)?;
        let grid = launch_u32(WHAT, "prepare grid", chunks * shape.n_v)?;
        let prep = self
            .module
            .prepare_gdn_chunk_prepare(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_chunk_prepare(
            stream, &prep, qkv, beta, decay, carry, pos, nk, nv, map, rows, count, fault, work,
        )?;
        let grid = launch_u32(WHAT, "walk grid", shape.n_v * (HEAD / PREPARED_COLUMNS))?;
        let prep = self
            .module
            .prepare_gdn_delta_prepared(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_delta_prepared(
            stream,
            &prep,
            qkv,
            carry,
            work,
            lane,
            launch_u32(WHAT, "lane_at", lane_at)?,
            launch_u32(WHAT, "lanes", lanes)?,
            nk,
            nv,
            map,
            rows,
            count,
            pos,
            fault,
            o,
            state,
            stamp,
        )?;
        let grid = launch_u32(WHAT, "carry grid", shape.n_k + shape.n_v + 1)?;
        let prep = self
            .module
            .prepare_gdn_carry_keep(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_carry_keep(
            stream, &prep, qkv, beta, decay, work, pos, nk, nv, rows, count, carry,
        )?;
        Ok(())
    }

    /// Enqueue `gdn_delta_lanes` (module doc): `8·n_v` blocks of 128
    /// threads, taking first the rows `carry` holds for `pos[0]` (module
    /// doc, "Carry"; `None` on a store whose prompt does not run the chunked
    /// rule). Refused by name: no lane, row mode over more tokens than
    /// lanes, a slice shorter than the launch reads. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_delta_lanes(
        &self,
        stream: &CudaStream,
        args: DeltaLanesArgs<'_>,
        carry: Option<&DeviceBuffer<f32>>,
    ) -> Result<(), GpuError> {
        self.enqueue_lanes(stream, args, carry, false)
    }

    /// Record a verify (`each`) without changing its committed lane, or
    /// replay its first `m` records into the last kept row's lane (`!each`).
    /// Records occupy the next two state lanes, with aligned key, update
    /// and decay planes per (row, value head). Replay reads the bank outside its destination lane,
    /// so no block can overwrite another block's unread records. The caller
    /// moves the lane word to the last kept row after replay. Replay reads
    /// the recorded position, not another slot's current parameter word;
    /// both calls take the same carry. Allocation-free and capturable; refuses
    /// fewer than three lanes or records past a lane.
    pub fn enqueue_delta_recovery(
        &self,
        stream: &CudaStream,
        args: DeltaLanesArgs<'_>,
        carry: Option<&DeviceBuffer<f32>>,
    ) -> Result<(), GpuError> {
        let d = &args.delta;
        if d.lanes < 3 || d.m > d.lanes || d.m > HEAD / 3 {
            return Err(GpuError::shape(
                "enqueue_delta_recovery",
                format!(
                    "{} rows over {} lanes: need 3 or more lanes and rows <= min(lanes, {})",
                    d.m,
                    d.lanes,
                    HEAD / 3
                ),
            ));
        }
        self.enqueue_lanes(stream, args, carry, true)
    }

    /// The replay of [`DeltaKernels::enqueue_delta_recovery`] (`each ==
    /// false`) for every store of `table` in one launch: store `s` takes the
    /// launch's blocks `8·n_v·s ..` and leaves its state, stamps and faults
    /// as its own replay call would, the lane word `lane[lane_at]` and `m`
    /// kept rows shared by all. Refused by name: `m` of 0 or past the lanes
    /// or `HEAD / 3`, a lane word shorter than read or overlapping any
    /// replay state or stamp. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_delta_recovery_layers(
        &self,
        stream: &CudaStream,
        (lane, lane_at): (&DeviceBuffer<u32>, usize),
        m: usize,
        table: &RecoveryTable,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_delta_recovery_layers";
        let RecoveryTable {
            words,
            writes,
            stores,
            lanes,
            shape,
        } = table;
        if m == 0 || m > *lanes || m > HEAD / 3 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{m} rows over {lanes} lanes: need rows in 1..=min(lanes, {})",
                    HEAD / 3
                ),
            ));
        }
        if lane.len() <= lane_at {
            return Err(GpuError::shape(
                WHAT,
                format!("lane.len() {} with lane_at {lane_at}", lane.len()),
            ));
        }
        let at = lane.cu_deviceptr() + 4 * lane_at as u64;
        if writes.iter().any(|&(start, end)| start <= at && at < end) {
            return Err(GpuError::shape(
                WHAT,
                "lane word overlaps replay state or stamps",
            ));
        }
        let per = shape.n_v * (HEAD / COLUMNS_PER_BLOCK);
        let blocks = stores
            .checked_mul(per)
            .ok_or_else(|| GpuError::shape(WHAT, "grid overflows usize"))?;
        let cfg = LaunchConfig1D::new(launch_u32(WHAT, "grid", blocks)?, BLOCK, 0);
        let prep = self.module.prepare_gdn_delta_recovery_layers(cfg)?;
        self.module.gdn_delta_recovery_layers(
            stream,
            &prep,
            lane,
            launch_u32(WHAT, "lane_at", lane_at)?,
            launch_u32(WHAT, "lanes", *lanes)?,
            launch_u32(WHAT, "n_k", shape.n_k)?,
            launch_u32(WHAT, "n_v", shape.n_v)?,
            grouped_arg(shape.map),
            launch_u32(WHAT, "m", m)?,
            words,
            launch_u32(WHAT, "stores", *stores)?,
            launch_u32(WHAT, "carry_len", shape.carry_len())?,
            fault,
        )?;
        Ok(())
    }

    fn enqueue_lanes(
        &self,
        stream: &CudaStream,
        args: DeltaLanesArgs<'_>,
        carry: Option<&DeviceBuffer<f32>>,
        recovery: bool,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_delta_lanes";
        check_gdn_lanes(WHAT, &args, carry)?;
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
        let nv = shape.n_v;
        let grid = launch_u32(WHAT, "grid", nv * (HEAD / COLUMNS_PER_BLOCK))?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        if recovery {
            let prep = self.module.prepare_gdn_delta_recovery(cfg)?;
            self.module.gdn_delta_recovery(
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
                grouped_arg(shape.map),
                launch_u32(WHAT, "m", m)?,
                u32::from(each),
                pos,
                carry.unwrap_or(qkv),
                u32::from(carry.is_some()),
                fault,
                o,
                state,
                stamp,
            )?;
        } else {
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
                grouped_arg(shape.map),
                launch_u32(WHAT, "m", m)?,
                u32::from(each),
                pos,
                carry.unwrap_or(qkv),
                u32::from(carry.is_some()),
                fault,
                o,
                state,
                stamp,
            )?;
        }
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
            grouped_arg(shape.map),
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
    use super::{delta_host, kda_delta_host, kept_lane, lanes_refusal, read_lane, row_lane};
    use crate::linear::{HEAD, KHeadMap, LinearShape};

    #[test]
    #[should_panic(expected = "carry_fold_host: carry count exceeds CARRY_ROWS")]
    fn a_carry_count_past_the_chunk_is_refused() {
        let shape = LinearShape {
            n_k: 1,
            n_v: 1,
            map: KHeadMap::Tiled,
        };
        let mut carry = vec![0.0; shape.carry_len()];
        carry[0] = f32::from_bits((super::CARRY_ROWS + 1) as u32);
        drop(super::carry_fold_host(
            &carry,
            &vec![0.0; shape.state_len()],
            0,
            shape,
        ));
    }

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

    /// Keeping `k` of a verify's rows leaves the word at the lane row `k − 1`
    /// wrote, for every committed lane and every kept count, at the lane
    /// counts of the bodies that verify (1 for a store that cannot, 4 for
    /// Qwen3.8's, 8 for Qwen3.6's): `k = 1` is the committed lane's row 0 in
    /// place, `k = lanes` the lane before the committed one, and the word
    /// stays below `lanes`.
    #[test]
    fn keeping_k_rows_leaves_the_word_at_the_last_kept_rows_lane() {
        for lanes in [1u32, 4, 8] {
            for c in 0..lanes {
                assert_eq!(kept_lane(c, 1, lanes), c, "lanes {lanes} c {c}");
                for k in 1..=lanes {
                    let w = kept_lane(c, k, lanes);
                    assert_eq!(w, row_lane(c, k - 1, lanes), "lanes {lanes} c {c} k {k}");
                    assert_eq!(w, (c + k - 1) % lanes);
                    assert!(w < lanes);
                }
                assert_eq!(kept_lane(c, lanes, lanes), (c + lanes - 1) % lanes);
            }
        }
        // Eight lanes, committed lane 5, all eight kept: row 7 wrote lane 4.
        assert_eq!(kept_lane(5, 8, 8), 4);
        // Four lanes, committed lane 3, two kept: row 1 wrote lane 0.
        assert_eq!(kept_lane(3, 2, 4), 0);
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
