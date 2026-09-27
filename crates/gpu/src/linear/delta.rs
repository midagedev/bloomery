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
//! call reads and writes lane `word mod lanes`. More than one lane — a
//! state per verified token — is not built: the launcher refuses it by name.
//!
//! Geometry: a warp owns four value columns of one head, eight lanes per
//! column; lane `j` of a column holds its [`KEYS_PER_LANE`] keys
//! `32·r + 4·j + c` (`r, c = 0..4`, in `(r, c)` order). Four warps per block,
//! eight blocks per head, `8·n_v` blocks. There is no shared memory and no
//! barrier: a column's two dot products are a lane's own sums, then a
//! butterfly over its eight lanes.
//!
//! Numeric contract (the host rule [`delta_host`] is this list), per token
//! and column:
//! - `S'ᵢ = decay·Sᵢ`, each rounded;
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

use super::{BLOCK, DECAY_HEAD, HEAD, LinearShape};
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
    let mut i = 4usize;
    while i < KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        p[i % 4] = fma_rn_f32(a[i], b[i], p[i % 4]);
        i += 1;
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
    let mut r = 0usize;
    while r < 4 {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: 32·r + 4·j + 3 < HEAD and a multiple of four floats from
        // an aligned base, by the fn's contract.
        let w = unsafe { ld4::<NC>(g + 4 * (32 * r + 4 * j) as u64) };
        v[4 * r] = w[0];
        v[4 * r + 1] = w[1];
        v[4 * r + 2] = w[2];
        v[4 * r + 3] = w[3];
        r += 1;
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
/// [`DECAY_HEAD`] or [`super::DECAY_KEY`]; only the first has an entry).
///
/// # Safety
///
/// The slices hold the lengths `gdn_delta`'s launch contract names, with
/// `decay` `m·n_v` long for [`DECAY_HEAD`] and `m·n_v·HEAD` for
/// [`super::DECAY_KEY`]; `o` and `state` address `m·n_v·HEAD` and
/// `lanes·n_v·HEAD·HEAD` writable f32s that no other launch touches; `lane`
/// holds word `lane_at`; `lanes >= 1`; `n_k >= 1` divides `n_v`; the block is
/// 128 threads and the grid `8·n_v` blocks.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
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
        let mut dk = [0.0f32; KEYS_PER_LANE];
        if DECAY != DECAY_HEAD {
            let mut i = 0usize;
            while i < KEYS_PER_LANE {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: the per-key decay row (t·n_v + h)·HEAD + key is
                // inside decay's m·n_v·HEAD values for DECAY_KEY.
                dk[i] = unsafe { *decay.get_unchecked((t * n_v + h) * HEAD + key_of(j, i)) };
                i += 1;
            }
        }
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
        let mut i = 0usize;
        while i < KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = mul_rn_f32(s[i], if DECAY == DECAY_HEAD { dh } else { dk[i] });
            i += 1;
        }
        let kv = column_sum(lane_dot(s, k));
        let u = mul_rn_f32(vt - kv, bt);
        let mut i = 0usize;
        while i < KEYS_PER_LANE {
            cuda_device::thread::__unroll_config::<0>();
            s[i] = fma_rn_f32(k[i], u, s[i]);
            i += 1;
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
    let mut i = 0usize;
    while i < KEYS_PER_LANE {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: the sixteen keys this lane read from `state` above, inside
        // it by the fn's contract; each read and written by this one lane.
        unsafe { *state.add(s_at + key_of(j, i)) = s[i] };
        i += 1;
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
}

/// [`DeltaKernels::enqueue_delta`]'s arguments: `m` tokens of the conv's
/// output `qkv` (`[m][C]`), β and decay (`[m][n_v]`), the lane word
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
    /// Lanes of `state`; only 1 is built.
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
        let module = unsafe { delta_kernels::load(ctx)? };
        Ok(DeltaKernels { module })
    }

    /// Enqueue the delta step of `args.m` tokens: `8·n_v` blocks of 128
    /// threads, the state read and written in place. Refuses `lanes` other
    /// than 1 by name. Asynchronous, allocation-free, capturable.
    pub fn enqueue_delta(&self, stream: &CudaStream, args: DeltaArgs<'_>) -> Result<(), GpuError> {
        let what = "enqueue_delta";
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
        let lens = [
            ("qkv", qkv.len(), m * shape.channels()),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv),
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
        let prep = self
            .module
            .prepare_gdn_delta(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.gdn_delta(
            stream,
            &prep,
            qkv,
            beta,
            decay,
            lane,
            lane_at,
            lanes,
            n_k,
            n_v,
            shape.map.code(),
            m,
            fault,
            o,
            state,
        )?;
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
/// mod lanes` of `state` (`lanes` lanes), in place.
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
                            let (bt, dc) = (beta[t * n_v + h], decay[t * n_v + h]);
                            for si in s.iter_mut() {
                                *si *= dc;
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
