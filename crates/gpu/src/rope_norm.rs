//! One launch per layer for the query and key heads of a grouped-query
//! attention layer with a NORM rope over the whole head and no per-head norm
//! (a Llama-style layer): the turn of each head's adjacent pairs `(2i, 2i + 1)`
//! by pair `i` of the rope table's row at the token's position, and for the
//! key heads the append of the turned key and of the value head to the layer's
//! cache planes at that position. The table holds a row for every cache
//! position, so the one check a position gets — below the planes' rows —
//! covers the table read and the append.
//!
//! The planes are head-major, `[n_kv][ctx][HEAD]`, and the launch contract is
//! [`crate::rope_neox`]'s head-128 append without the gains and the epsilon:
//! block `b = t·(n_head + n_kv) + h` of [`HEAD`]`/2` threads, the query and
//! key rows turned in place, the append through [`KvSink`] — the f16 planes
//! ([`RopeNormKernels::enqueue_rope_norm_append`]) or the Q8_0 planes
//! ([`RopeNormKernels::enqueue_rope_norm_append_q8`]). One launch serves a
//! decode row and the rows of a prompt chunk.
//!
//! Numeric contract, per head of [`HEAD`] values:
//! - the turn is [`rope_pair_rn`], the shared NORM core: `y0 = x0·c − x1·s`,
//!   `y1 = x0·s + x1·c`, each product and each sum rounded on its own, the
//!   rounding ik's compiled CPU NORM loop produces;
//! - thread `t` owns values `t` and `t + HEAD/2` (the layout the Q8_0 blocks
//!   close over, [`KvSink`]) and takes each value's pair partner, the value at
//!   the index xor 1, from the adjacent lane by a shuffle, so every output is
//!   the pair's turn computed from the pair's own two inputs;
//! - the cache rows rounded once to f16, to nearest even, or quantized by
//!   [`KvSink`]'s Q8_0 rule.
//!
//! A position at or past `ctx` has no row in the planes or the table: every
//! block of the token raises [`FaultSite::CachePos`], writes NaN over its head
//! in place and appends nothing. A Q8_0 block holding a non-finite value is
//! refused as [`crate::rope_neox`]'s q8 append refuses it.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::rope_neox::{F16Sink, HEAD, KvSink, Q8Sink, q8_plane_lens};
use crate::rope_table::rope_pair_rn;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block: one per two values of a head.
const THREADS: usize = HEAD / 2;
const THREADS_U32: u32 = THREADS as u32;
// The entries' launch contracts spell HEAD and the block out as 128 and 64.
const _: () = assert!(THREADS_U32 as usize == THREADS && HEAD == 128 && THREADS == 64);

/// The turn and append of [`rope_norm_kernels::rope_norm_append`] and
/// [`rope_norm_kernels::rope_norm_append_q8`], written once over the cache
/// format `S`. Block `b = t·(n_head + n_kv) + h`: query head `h` of token `t`
/// is turned in place in `q[(t·n_head + h)·HEAD ..]`; key head `j = h − n_head`
/// is turned in place in `k[(t·n_kv + j)·HEAD ..]` and appended, with value
/// head `j` of `v`, at plane row `j·ctx + pos[t]`.
///
/// SAFETY: the entry's launch contract at [`HEAD`] over `sink`'s planes of
/// `n_kv · ctx` rows, a block of `HEAD/2` threads, every thread of the block
/// running the body, and no other block writing this block's rows (the tokens
/// of one launch hold distinct positions).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn norm_append<S: KvSink>(
    table: &[f32],
    pos: &[u32],
    v: &[f32],
    n_head: u32,
    n_kv: u32,
    ctx: u32,
    m: u32,
    fault: FaultSink,
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut sink: S,
) {
    let heads = (n_head + n_kv) as usize;
    let b = thread::blockIdx_x() as usize;
    if b >= m as usize * heads {
        return; // block-uniform
    }
    let t = b / heads;
    let h = b - t * heads;
    let tid = thread::threadIdx_x() as usize;
    let is_q = h < n_head as usize;
    let kh = h.wrapping_sub(n_head as usize);
    let base = if is_q {
        (t * n_head as usize + h) * HEAD
    } else {
        (t * n_kv as usize + kh) * HEAD
    };
    // SAFETY: t < m <= pos.len() by the launch contract.
    let p = unsafe { *pos.get_unchecked(t) } as usize;

    if p >= ctx as usize {
        if tid == 0 {
            fault.raise(FaultSite::CachePos);
        }
        // SAFETY: base + tid + 64 < (t·n + h + 1)·128 <= m·n·128, inside the
        // head's buffer by the launch contract; this thread owns the two
        // positions.
        unsafe {
            if is_q {
                *q.get_unchecked_mut(base + tid) = f32::NAN;
                *q.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
            } else {
                *k.get_unchecked_mut(base + tid) = f32::NAN;
                *k.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
            }
        }
        return; // block-uniform: p is the token's
    }

    // SAFETY: base + tid + 64 < (t·n + h + 1)·128 <= m·n·128, inside the
    // head's buffer by the launch contract; the thread owns the two positions.
    let (x0, x1) = unsafe {
        if is_q {
            (
                *q.get_unchecked_mut(base + tid),
                *q.get_unchecked_mut(base + tid + THREADS),
            )
        } else {
            (
                *k.get_unchecked_mut(base + tid),
                *k.get_unchecked_mut(base + tid + THREADS),
            )
        }
    };
    // The value at the index xor 1 is the adjacent lane's: tid and tid + 64
    // keep their parity, so the partner of each is the other lane's same slot.
    let (m0, m1) = (warp::shuffle_xor_f32(x0, 1), warp::shuffle_xor_f32(x1, 1));
    // Pair `tid / 2` of the table's row p turns values `tid` and `tid ^ 1`,
    // pair `tid / 2 + 32` values `tid + 64` and `(tid + 64) ^ 1`.
    let at = p * HEAD + (tid - tid % 2);
    // SAFETY: p < ctx and tid - tid % 2 + 65 < 128, so both pairs are inside
    // row p of the table's ctx rows.
    let (c0, s0, c1, s1) = unsafe {
        (
            *table.get_unchecked(at),
            *table.get_unchecked(at + 1),
            *table.get_unchecked(at + THREADS),
            *table.get_unchecked(at + THREADS + 1),
        )
    };
    let odd = tid % 2 == 1;
    let (a0, b0) = if odd { (m0, x0) } else { (x0, m0) };
    let (a1, b1) = if odd { (m1, x1) } else { (x1, m1) };
    let (r00, r01) = rope_pair_rn(a0, b0, c0, s0);
    let (r10, r11) = rope_pair_rn(a1, b1, c1, s1);
    let y0 = if odd { r01 } else { r00 };
    let y1 = if odd { r11 } else { r10 };

    if is_q {
        // SAFETY: the positions read above; this thread owns them.
        unsafe {
            *q.get_unchecked_mut(base + tid) = y0;
            *q.get_unchecked_mut(base + tid + THREADS) = y1;
        }
        return;
    }
    // SAFETY: the positions read above; this thread owns them.
    unsafe {
        *k.get_unchecked_mut(base + tid) = y0;
        *k.get_unchecked_mut(base + tid + THREADS) = y1;
    }
    // SAFETY: kh < n_kv and p < ctx, so the row is inside the planes; base +
    // tid + 64 < m·n_kv·128 <= v.len(). One thread per two values of the row,
    // each warp's values `32·w + lane` and `32·w + 64 + lane`, whole 32-value
    // blocks; no other block writes the row (this fn's contract).
    unsafe {
        sink.append::<HEAD>(
            kh * ctx as usize + p,
            (tid, tid + THREADS),
            (y0, y1),
            v,
            (base + tid, base + tid + THREADS),
            fault,
        )
    };
}

#[cuda_module]
mod rope_norm_kernels {
    use super::*;

    /// The NORM turn and f16 append of `m` tokens. Block `b = t·(n_head +
    /// n_kv) + h`: for `h < n_head` query head `h` of token `t` is turned and
    /// written back in place in `q`; for `h = n_head + j` key head `j` is
    /// turned, written back in place in `k` and rounded to f16 into `cache_k`
    /// row `j·ctx + pos[t]`, and value head `j` of `v` is rounded into
    /// `cache_v` at the same row. Token `t` turns by the table's row
    /// `pos[t]`, `table[pos[t]·HEAD ..]`, `[cos_0, sin_0, …]` for the 64
    /// pairs. The branch is on the block index, so no warp splits across it.
    /// A position at or past `ctx` raises [`FaultSite::CachePos`] on `fault`,
    /// writes NaN over the head in place and appends nothing.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            table.len() >= ctx * 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * ctx * 128,
            cache_v.len() >= n_kv * ctx * 128
        )
    )]
    pub fn rope_norm_append(
        table: &[f32],
        pos: &[u32],
        v: &[f32],
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        cache_k: DisjointSlice<u16>,
        cache_v: DisjointSlice<u16>,
    ) {
        // SAFETY: the launch contract is `norm_append`'s over the f16 planes,
        // and the block is THREADS threads.
        unsafe {
            norm_append(
                table,
                pos,
                v,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                q,
                k,
                F16Sink {
                    k: cache_k,
                    v: cache_v,
                },
            )
        };
    }

    /// [`rope_norm_append`] with the append quantized to Q8_0 (`norm_append`
    /// over [`Q8Sink`]): the turn and the in-place writes are the f16 entry's
    /// bit for bit, and each key and value row is stored as four 32-value
    /// blocks in the two-plane layout `crate::rope_neox` documents — per side
    /// a codes plane of `n_kv·ctx·32` words and a scales plane of
    /// `n_kv·ctx·4` f16 scales.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            table.len() >= ctx * 128,
            pos.len() >= m,
            kq.len() >= n_kv * ctx * 32,
            kd.len() >= n_kv * ctx * 4,
            vq.len() >= n_kv * ctx * 32,
            vd.len() >= n_kv * ctx * 4
        )
    )]
    pub fn rope_norm_append_q8(
        table: &[f32],
        pos: &[u32],
        v: &[f32],
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        m: u32,
        fault: FaultSink,
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        kq: DisjointSlice<u32>,
        kd: DisjointSlice<u16>,
        vq: DisjointSlice<u32>,
        vd: DisjointSlice<u16>,
    ) {
        // SAFETY: the launch contract is `norm_append`'s over the four Q8_0
        // planes, and the block is THREADS threads.
        unsafe {
            norm_append(
                table,
                pos,
                v,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                q,
                k,
                Q8Sink { kq, kd, vq, vd },
            )
        };
    }
}

/// [`RopeNormKernels::enqueue_rope_norm_append`]'s arguments: `m` tokens of
/// `n_head` query heads in `q` and `n_kv` key and value heads in `k` and `v`
/// (token-major, head after head, [`HEAD`] values each; `q` and `k` are turned
/// in place), the rope table on the device — a row of [`HEAD`] f32 for each of
/// at least `ctx` positions, `RopeTable::push` over a [`HEAD`]-wide spec — and
/// the `m` tokens' positions, the sink a position past the planes raises on,
/// and the layer's two cache planes of `n_kv · ctx` rows of [`HEAD`] f16.
pub struct NormArgs<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub cache_k: &'a mut DeviceBuffer<u16>,
    pub cache_v: &'a mut DeviceBuffer<u16>,
}

/// [`NormArgs`]'s q8_0 form: the same inputs, the cache the four planes of
/// [`RopeNormKernels::enqueue_rope_norm_append_q8`]'s append — per side the
/// codes and the scales of `n_kv · ctx` rows of [`HEAD`] values
/// ([`q8_plane_lens`]).
pub struct NormQ8Args<'a> {
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub kq: &'a mut DeviceBuffer<u32>,
    pub kd: &'a mut DeviceBuffer<u16>,
    pub vq: &'a mut DeviceBuffer<u32>,
    pub vd: &'a mut DeviceBuffer<u16>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RopeNormKernels {
    module: rope_norm_kernels::LoadedModule,
}

impl RopeNormKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeNormKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(rope_norm_kernels, ctx)? };
        Ok(RopeNormKernels { module })
    }

    /// Enqueue the NORM turn and f16 append of `args.m` tokens: one block per
    /// (token, head), `m·(n_head + n_kv)` blocks of 64 threads, token `t`
    /// turned by the table's row `pos[t]`. The tokens of one launch must hold
    /// distinct positions below `ctx` (the caller's positions are
    /// consecutive); a position `>= ctx` raises [`FaultSite::CachePos`] on
    /// `args.fault`, leaves the token's heads NaN and is not appended. A shape
    /// or a buffer the launch is not built for is refused by name before any
    /// launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_rope_norm_append(
        &self,
        stream: &CudaStream,
        args: NormArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_rope_norm_append";
        let NormArgs {
            q,
            k,
            v,
            table,
            pos,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("table", table.len(), ctx * HEAD),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), n_kv * ctx * HEAD),
            ("cache_v", cache_v.len(), n_kv * ctx * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep =
            self.module
                .prepare_rope_norm_append(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.rope_norm_append(
            stream, &prep, table, pos, v, n_head, n_kv, ctx, m, fault, q, k, cache_k, cache_v,
        )?;
        Ok(())
    }

    /// [`RopeNormKernels::enqueue_rope_norm_append`] with the append
    /// quantized to Q8_0: the same turn and in-place writes, and each key and
    /// value row stored as four 32-value blocks in `args`' four planes — per
    /// side the codes and the scales ([`q8_plane_lens`]), the quantization
    /// rule and refusal [`crate::rope_neox`]'s q8 append's. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_rope_norm_append_q8(
        &self,
        stream: &CudaStream,
        args: NormQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_rope_norm_append_q8";
        let NormQ8Args {
            q,
            k,
            v,
            table,
            pos,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            kq,
            kd,
            vq,
            vd,
        } = args;
        if n_head == 0 || n_kv == 0 || m == 0 || ctx == 0 || m > ctx {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_head, n_kv, ctx >= 1 and 1 <= m <= ctx, got n_head={n_head} \
                     n_kv={n_kv} ctx={ctx} m={m}"
                ),
            ));
        }
        let (words, scales) = q8_plane_lens(HEAD, n_kv, ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("k", k.len(), m * n_kv * HEAD),
            ("v", v.len(), m * n_kv * HEAD),
            ("table", table.len(), ctx * HEAD),
            ("pos", pos.len(), m),
            ("kq", kq.len(), words),
            ("kd", kd.len(), scales),
            ("vq", vq.len(), words),
            ("vd", vd.len(), scales),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * (n_head + n_kv))?;
        let n_head = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_rope_norm_append_q8(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.rope_norm_append_q8(
            stream, &prep, table, pos, v, n_head, n_kv, ctx, m, fault, q, k, kq, kd, vq, vd,
        )?;
        Ok(())
    }
}
