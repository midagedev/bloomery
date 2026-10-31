//! The conv and prep of a linear-attention layer, one launch per call over
//! `m` tokens: the depthwise causal conv of every channel over its last
//! [`CONV_TAPS`] inputs, SiLU, the L2 norm of each query and key head (the
//! query also scaled by [`Q_SCALE`]), β and the decay, and the conv ring's
//! new slots. Three entries run the same conv, norm and ring rule: the decay
//! of each value head (`gdn_conv_prep`, Gated DeltaNet), or of each (value
//! head, key channel) (`kda_conv_prep`, Kimi Delta Attention as GLM-5.3-Flash
//! runs it, and `kda_conv_prep_sp2`, as Solar-Open2 runs it).
//! `gdn_conv_prep_bf16` is `gdn_conv_prep` over channels held as BF16 words
//! (a ubatch walk's projection stores them so), each value widened as it is
//! read: its outputs are the F32 entry's rule over the widened values.
//! `gdn_conv_prep_seqs` is `gdn_conv_prep` over the rows of several
//! sequences in one launch, each sequence's rows over its own ring
//! (`crate::seq_rows`).
//!
//! Positions: token `t` is position `pos[t]`, the words the embedding launch
//! writes, `pos[t] = pos[0] + t`. Its predecessor at distance `d` (1 to
//! [`CONV_ROWS`]) is the call's row `t − d` when `d <= t`; otherwise it lies
//! before the call: zero when `pos[t] < d` (the sequence start, whatever the
//! ring holds), else ring slot `(pos[t] − d) mod RING_ROWS`. The call writes
//! the input of each of its last `min(m, RING_ROWS)` tokens into slot
//! `pos[t] mod RING_ROWS`, so the ring holds the inputs of the last
//! [`RING_ROWS`] positions and a later call from any of the last
//! [`super::PASS_ROWS`] + 1 positions reads its predecessors without a copy.
//! A token whose position is not `pos[0] + t` has no defined predecessors: it
//! raises [`FaultSite::LinearConv`] and its output is NaN.
//!
//! Geometry: one warp per (unit, conv head), four warps per block, no shared
//! memory and no barrier. Lane `l` owns channels `l + 32·i` (`i = 0..4`) of
//! its head; in `gdn_conv_prep_bf16` the two channel pairs `2l, 2l + 1` and
//! `64 + 2l, 65 + 2l` ([`pair_channel`]), each pair one word it loads once
//! and one 8-byte store of `y`, so the warp's loads and stores stay whole
//! 128- and 256-byte lines. Unit 0 is tokens `0 .. min(m, 3)` — every
//! token that reads the ring — and afterwards writes the ring; unit
//! `u >= 1` is token `u + 2`,
//! whose predecessors are all the call's rows. The ring is touched by unit 0
//! alone, each channel by one lane, its reads before its writes. A value
//! head's lane 0 computes that head's β for its token, and its decay
//! (`gdn_conv_prep`); in the KDA entries lane `l` computes the decay of key
//! channels `l + 32·i` of the head.
//!
//! Numeric contract (the host rules [`conv_prep_host`] and
//! [`kda_conv_prep_host`] are this list):
//! - conv: `acc = x₀·w₀` rounded, then `acc = fma(xⱼ, wⱼ, acc)` for taps
//!   1, 2, 3 — oldest input first, tap j of channel c the weight at `4c + j`;
//! - [`silu`] of `acc`;
//! - L2 norm of a query or key head: each value squared in f32, lane `l`'s
//!   four squares summed in f64 in `i` order, the 32 lanes by the xor
//!   butterfly (16, 8, 4, 2, 1) in f64, the sum rounded to f32; `scale =
//!   1 / max(√sum, eps)` in f32 (ggml's `l2_norm`), then `value·scale`, and
//!   for a query that times [`Q_SCALE`];
//! - β = [`sigmoid`]`(b)`, times 2 in `kda_conv_prep_sp2` (exact in f32);
//! - `gdn_conv_prep`: decay = [`expf_ik`]`(`[`softplus`]`(a + dt)·ssm_a)` per
//!   value head, `a` and `dt` of the head;
//! - `kda_conv_prep`: per value head `h` and key channel `k`, `r = (f + dt)·
//!   ssm_a_h` and decay = [`expf_ik`]`(lb·`[`sigmoid`]`(−r))`, `f` and `dt` of
//!   channel `h·HEAD + k`, each product rounded — ik's `build_kda_beta_gate`
//!   (`ssm_a = −e^A_log`, so `−r = e^A_log·(f + dt)`) with its delta op's exp.
//!   `lb < 0` bounds the decay to `[e^lb, 1]`;
//! - `kda_conv_prep_sp2`: decay = [`expf_ik`]`(`[`softplus`]`(f + dt)·ssm_a)`
//!   per (value head, key channel), `gdn_conv_prep`'s operations at key
//!   granularity, unbounded (`ssm_a = −e^A_log`, so the decay is
//!   `exp(−e^A_log·softplus(f + dt))`).
//!
//! No silent failure: a non-finite input, a conv sum that overflows, a norm
//! sum that is not finite as f32 (the square of a finite value can be), a β
//! or decay that is not finite, or a position that is not `pos[0] + t`
//! raises [`FaultSite::LinearConv`], and the value written there is not
//! finite. `max(NaN, eps)` would otherwise turn an overflowed norm into a
//! zero head.

use super::{BLOCK, CONV_ROWS, CONV_TAPS, HEAD, LinearShape, Q_SCALE, RING_ROWS, ring_input};
use super::{DECAY_HEAD, DECAY_KEY, KDA_BETA_X1, KDA_BETA_X2, KDA_GATE_SIGMOID, KDA_GATE_SOFTPLUS};
use super::{expf_ik, sigmoid, silu, softplus};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use crate::q8f32::Bf16Words;
use crate::seq_rows::{SEQ_WORDS, SeqRows};
use crate::tensor::check_alignment;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::vector::F32x2;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::ops::Range;
use std::sync::Arc;

const _: () = assert!(HEAD == 128 && CONV_TAPS == 4 && BLOCK == 128 && RING_ROWS == 11);

/// The warps of a call of `m` tokens per conv head: unit 0 takes the first
/// `min(m, CONV_ROWS)` tokens, every later token a unit of its own.
#[inline(always)]
#[must_use]
pub fn units(m: usize) -> usize {
    m.max(CONV_ROWS) - (CONV_ROWS - 1)
}

/// Where one warp writes: the call's outputs.
#[derive(Clone, Copy)]
struct Outputs {
    y: *mut f32,
    beta: *mut f32,
    decay: *mut f32,
}

/// The conv of channel `c` over its four inputs `xs`, oldest first, and the
/// SiLU of the sum (module doc); `ok` drops when an input or the sum is not
/// finite.
///
/// # Safety
///
/// `w` holds `4·(c + 1)` values.
#[inline(always)]
unsafe fn conv_channel(xs: [f32; 4], w: &[f32], c: usize, ok: &mut bool) -> f32 {
    // SAFETY: 4c + 3 < w.len() by the contract.
    let wv = unsafe {
        [
            *w.get_unchecked(4 * c),
            *w.get_unchecked(4 * c + 1),
            *w.get_unchecked(4 * c + 2),
            *w.get_unchecked(4 * c + 3),
        ]
    };
    let mut acc = mul_rn_f32(xs[0], wv[0]);
    acc = fma_rn_f32(xs[1], wv[1], acc);
    acc = fma_rn_f32(xs[2], wv[2], acc);
    acc = fma_rn_f32(xs[3], wv[3], acc);
    *ok &= crate::fault::quad_finite(xs) & acc.is_finite();
    silu(acc)
}

/// The L2 norm of a query or key head `hp` (module doc) over this lane's four
/// values and the f64 sum of their squares; a value head's values pass as
/// they are. `ok` drops when the head's sum is not finite. Called by all 32
/// lanes of the warp together.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
fn head_norm(vals: &mut [f32; 4], sumsq: f64, hp: usize, n_k: usize, eps: f32, ok: &mut bool) {
    if hp < 2 * n_k {
        let mut acc = sumsq;
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        let sum = acc as f32;
        let fin = sum.is_finite();
        *ok &= fin;
        let scale = if fin {
            1.0 / sum.sqrt().max(eps)
        } else {
            f32::NAN
        };
        for i in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            vals[i] = mul_rn_f32(vals[i], scale);
            if hp < n_k {
                vals[i] = mul_rn_f32(vals[i], Q_SCALE);
            }
        }
    }
}

/// Channel `i` (0 to 3) of a head that lane `lane` owns when the channels
/// are BF16 words: the pair `2·lane`, `2·lane + 1` of each 64-channel half,
/// one word of each half.
#[inline(always)]
const fn pair_channel(lane: usize, i: usize) -> usize {
    64 * (i / 2) + 2 * lane + i % 2
}

/// The channels a warp's conv reads, as the entry holds them: F32 values
/// (`&[f32]`, channels `lane + 32*i`) or BF16 words ([`Bf16Words`], the
/// channels [`pair_channel`] gives the lane, read and stored as pairs).
trait ConvIn: Copy {
    /// The conv, norm and `y` of token `t` of head `hp` (module rule).
    ///
    /// # Safety
    ///
    /// `t < m`, `hp < 2·n_k + n_v`, `lane < 32`, all 32 lanes together;
    /// `C = (2·n_k + n_v)·HEAD`. The input holds `m·C` values (F32) or
    /// `m·C / 2` words (BF16), `w` holds `4·C` taps, and `y` addresses `m·C`
    /// writable f32s, 8-byte aligned for BF16 input. This warp owns its
    /// `(t, hp)` outputs and head `hp` of the `RING_ROWS·C` ring values;
    /// the ring is read only when `t < CONV_ROWS`.
    #[allow(
        clippy::too_many_arguments,
        reason = "one token's coordinates and the warp's conv inputs and output, forwarded from the entry"
    )]
    unsafe fn head(
        self,
        t: usize,
        p: usize,
        contiguous: bool,
        hp: usize,
        lane: usize,
        w: &[f32],
        eps: f32,
        n_k: usize,
        n_v: usize,
        ring: *const f32,
        y: *mut f32,
    ) -> bool;

    /// The ring's new slots of head `hp` on this lane's channels.
    ///
    /// # Safety
    ///
    /// `ch = (2·n_k + n_v)·HEAD`, `hp < 2·n_k + n_v`, `lane < 32`;
    /// `pos` holds `m` positions and the input holds `m·ch` values (F32)
    /// or `m·ch / 2` words (BF16). `ring` addresses `RING_ROWS·ch` writable
    /// f32s whose head `hp` this warp owns, after all ring readers finish.
    unsafe fn store_ring(
        self,
        pos: &[u32],
        m: usize,
        ch: usize,
        hp: usize,
        lane: usize,
        ring: *mut f32,
    );
}

impl ConvIn for &[f32] {
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[inline(always)]
    unsafe fn head(
        self,
        t: usize,
        p: usize,
        contiguous: bool,
        hp: usize,
        lane: usize,
        w: &[f32],
        eps: f32,
        n_k: usize,
        n_v: usize,
        ring: *const f32,
        y: *mut f32,
    ) -> bool {
        let heads = 2 * n_k + n_v;
        let ch = heads * HEAD;
        let mut vals = [0.0f32; 4];
        let mut sumsq = 0.0f64;
        let mut ok = contiguous;
        for i in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            let c = hp * HEAD + lane + 32 * i;
            // Tap j reads the predecessor at distance d = CONV_ROWS − j.
            let mut xs = [0.0f32; 4];
            for j in 0..CONV_TAPS {
                cuda_device::thread::__unroll_config::<0>();
                let d = CONV_ROWS - j;
                // SAFETY: c < ch; for d <= t the row t − d < m, inside self (m·ch
                // values by the contract); else the slot (p − d) mod RING_ROWS
                // (p >= d) is a row of the ring (RING_ROWS·ch values).
                xs[j] = unsafe {
                    if d <= t {
                        *self.get_unchecked((t - d) * ch + c)
                    } else {
                        ring_input(ring, RING_ROWS, p, d, ch, c)
                    }
                };
            }
            // SAFETY: c < ch, so 4c + 3 < 4·ch <= w.len() by the contract.
            let s = unsafe { conv_channel(xs, w, c, &mut ok) };
            vals[i] = s;
            sumsq += f64::from(mul_rn_f32(s, s));
        }
        head_norm(&mut vals, sumsq, hp, n_k, eps, &mut ok);
        for i in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            let v = if contiguous { vals[i] } else { f32::NAN };
            // SAFETY: t < m and c < ch, inside y (m·ch values by the contract);
            // one lane per channel.
            unsafe { *y.add(t * ch + hp * HEAD + lane + 32 * i) = v };
        }
        ok
    }

    #[inline(always)]
    unsafe fn store_ring(
        self,
        pos: &[u32],
        m: usize,
        ch: usize,
        hp: usize,
        lane: usize,
        ring: *mut f32,
    ) {
        let mut t = m.saturating_sub(RING_ROWS);
        while t < m {
            // SAFETY: t < m <= pos.len().
            let slot = unsafe { *pos.get_unchecked(t) } as usize % RING_ROWS;
            for i in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let c = hp * HEAD + lane + 32 * i;
                // SAFETY: slot < RING_ROWS and c < ch: inside the ring
                // (RING_ROWS·ch values) and row t of self (m·ch values);
                // channel c of the ring is this lane's alone.
                unsafe { *ring.add(slot * ch + c) = *self.get_unchecked(t * ch + c) };
            }
            t += 1;
        }
    }
}

impl ConvIn for Bf16Words<'_> {
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[inline(always)]
    unsafe fn head(
        self,
        t: usize,
        p: usize,
        contiguous: bool,
        hp: usize,
        lane: usize,
        w: &[f32],
        eps: f32,
        n_k: usize,
        n_v: usize,
        ring: *const f32,
        y: *mut f32,
    ) -> bool {
        let heads = 2 * n_k + n_v;
        let ch = heads * HEAD;
        let words = ch / 2;
        let mut vals = [0.0f32; 4];
        let mut sumsq = 0.0f64;
        let mut ok = contiguous;
        for half in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let c = hp * HEAD + pair_channel(lane, 2 * half);
            // Tap j of the pair's two channels, the predecessor at distance
            // d = CONV_ROWS − j.
            let mut lo = [0.0f32; 4];
            let mut hi = [0.0f32; 4];
            for j in 0..CONV_TAPS {
                cuda_device::thread::__unroll_config::<0>();
                let d = CONV_ROWS - j;
                // SAFETY: c + 1 < ch and c is even; for d <= t the row t − d < m
                // and its word c / 2 lie inside self.0 (m·ch / 2 words by the
                // contract); else the slot (p − d) mod RING_ROWS (p >= d) is a
                // row of the ring (RING_ROWS·ch values).
                unsafe {
                    if d <= t {
                        let word = *self.0.get_unchecked((t - d) * words + c / 2);
                        lo[j] = Bf16Words::widen(word, false);
                        hi[j] = Bf16Words::widen(word, true);
                    } else {
                        lo[j] = ring_input(ring, RING_ROWS, p, d, ch, c);
                        hi[j] = ring_input(ring, RING_ROWS, p, d, ch, c + 1);
                    }
                }
            }
            // SAFETY: c + 1 < ch, so 4c + 3 < 4·ch <= w.len() by the contract.
            let s0 = unsafe { conv_channel(lo, w, c, &mut ok) };
            // SAFETY: c + 1 < ch, so 4(c + 1) + 3 < 4·ch <= w.len() by the
            // contract.
            let s1 = unsafe { conv_channel(hi, w, c + 1, &mut ok) };
            vals[2 * half] = s0;
            vals[2 * half + 1] = s1;
            sumsq += f64::from(mul_rn_f32(s0, s0));
            sumsq += f64::from(mul_rn_f32(s1, s1));
        }
        head_norm(&mut vals, sumsq, hp, n_k, eps, &mut ok);
        for half in 0usize..2 {
            cuda_device::thread::__unroll_config::<0>();
            let pair = if contiguous {
                [vals[2 * half], vals[2 * half + 1]]
            } else {
                [f32::NAN; 2]
            };
            let at = t * ch + hp * HEAD + pair_channel(lane, 2 * half);
            // SAFETY: t < m and at + 1 < m·ch, inside y (by the contract); at is
            // even and y 8-byte aligned, so the pair is one aligned F32x2; one
            // lane per pair.
            unsafe { *y.add(at).cast::<F32x2>() = F32x2(pair) };
        }
        ok
    }

    #[inline(always)]
    unsafe fn store_ring(
        self,
        pos: &[u32],
        m: usize,
        ch: usize,
        hp: usize,
        lane: usize,
        ring: *mut f32,
    ) {
        let words = ch / 2;
        let mut t = m.saturating_sub(RING_ROWS);
        while t < m {
            // SAFETY: t < m <= pos.len().
            let slot = unsafe { *pos.get_unchecked(t) } as usize % RING_ROWS;
            for half in 0usize..2 {
                cuda_device::thread::__unroll_config::<0>();
                let c = hp * HEAD + pair_channel(lane, 2 * half);
                // SAFETY: slot < RING_ROWS and c + 1 < ch, c even: inside the
                // ring (RING_ROWS·ch values), and word c / 2 of row t inside self.0
                // (m·ch / 2 words); channels c and c + 1 of the ring are this
                // lane's alone.
                unsafe {
                    let word = *self.0.get_unchecked(t * words + c / 2);
                    *ring.add(slot * ch + c) = Bf16Words::widen(word, false);
                    *ring.add(slot * ch + c + 1) = Bf16Words::widen(word, true);
                }
            }
            t += 1;
        }
    }
}

/// The inputs one warp reads: the conv's (`x` as the entry holds the
/// channels, [`ConvIn`]), and the decay's raw values `decay_raw` — `a`
/// (`[m][n_v]`) with bias `dt_bias` (`[n_v]`) under
/// [`DECAY_HEAD`], `f` (`[m][n_v·HEAD]`) with `dt_bias` (`[n_v·HEAD]`) under
/// [`DECAY_KEY`] — `ssm_a` (`[n_v]`) and bound `lb`, which only
/// [`KDA_GATE_SIGMOID`] reads. Under [`DECAY_HEAD`] token `t`'s β and α
/// are `ba_pitch·t` values into `b_raw` and `decay_raw`; under [`DECAY_KEY`]
/// its β is `n_v·t` values into `b_raw`.
#[derive(Clone, Copy)]
struct PrepInputs<'a, X> {
    x: X,
    w: &'a [f32],
    b_raw: &'a [f32],
    decay_raw: &'a [f32],
    dt_bias: &'a [f32],
    ssm_a: &'a [f32],
    lb: f32,
    eps: f32,
    n_k: usize,
    n_v: usize,
    ba_pitch: usize,
}

/// One decay (module doc) under the gate `GATE`, each product rounded on its
/// own; [`kda_decay_host`] is the same operations.
#[inline(always)]
fn kda_decay<const GATE: u32>(f: f32, dt: f32, sa: f32, lb: f32) -> f32 {
    const { assert!(GATE == KDA_GATE_SIGMOID || GATE == KDA_GATE_SOFTPLUS) };
    if GATE == KDA_GATE_SIGMOID {
        let r = mul_rn_f32(f + dt, sa);
        expf_ik(mul_rn_f32(lb, sigmoid(-r)))
    } else {
        expf_ik(mul_rn_f32(softplus(f + dt), sa))
    }
}

/// One head's β (module doc) with the scale `BETA`.
#[inline(always)]
fn kda_beta<const BETA: u32>(b: f32) -> f32 {
    const { assert!(BETA == KDA_BETA_X1 || BETA == KDA_BETA_X2) };
    if BETA == KDA_BETA_X1 {
        sigmoid(b)
    } else {
        mul_rn_f32(sigmoid(b), BETA as f32)
    }
}

/// Token `t` of conv head `hp`: the conv of `inp.x` ([`ConvIn::head`]),
/// then on a value head the
/// decay under `GATE` and β scaled by `BETA`. Under [`DECAY_HEAD`] lane 0
/// writes both of the head; under [`DECAY_KEY`] every lane writes the decay of
/// its four key channels `lane + 32·i` of the head and lane 0 the β. Whether
/// every value stayed finite.
///
/// # Safety
///
/// [`ConvIn::head`]'s contract; the slices and `out` hold the lengths the
/// entries' launch contract names.
#[allow(
    clippy::too_many_arguments,
    reason = "one token's coordinates and the warp's inputs and outputs, forwarded from the entry"
)]
#[inline(always)]
unsafe fn prep_token<const DECAY: u32, const GATE: u32, const BETA: u32, X: ConvIn>(
    t: usize,
    p: usize,
    contiguous: bool,
    hp: usize,
    lane: usize,
    inp: PrepInputs<'_, X>,
    ring: *const f32,
    out: Outputs,
) -> bool {
    const { assert!(DECAY == DECAY_HEAD || DECAY == DECAY_KEY) };
    // SAFETY: the fn's contract.
    let mut ok = unsafe {
        inp.x.head(
            t, p, contiguous, hp, lane, inp.w, inp.eps, inp.n_k, inp.n_v, ring, out.y,
        )
    };
    if DECAY == DECAY_HEAD {
        // One condition, not nested `if`s: the nested form changes the entry's
        // register allocation.
        if hp >= 2 * inp.n_k && lane == 0 {
            let h = hp - 2 * inp.n_k;
            let (at, raw) = (t * inp.n_v + h, t * inp.ba_pitch + h);
            // SAFETY: h < n_v and t < m: `at` < m·n_v <= beta, decay
            // lengths, `raw` < (m − 1)·ba_pitch + n_v <= b_raw, decay_raw
            // lengths; h < n_v <= dt_bias, ssm_a lengths.
            let (b, a, dt, sa) = unsafe {
                (
                    *inp.b_raw.get_unchecked(raw),
                    *inp.decay_raw.get_unchecked(raw),
                    *inp.dt_bias.get_unchecked(h),
                    *inp.ssm_a.get_unchecked(h),
                )
            };
            let bt = kda_beta::<BETA>(b);
            let dc = kda_decay::<GATE>(a, dt, sa, inp.lb);
            let fin = crate::fault::quad_finite([b, a, dt, sa]) & bt.is_finite() & dc.is_finite();
            ok &= fin;
            // SAFETY: as the reads above; one lane per (token, head).
            unsafe {
                *out.beta.add(at) = if fin { bt } else { f32::NAN };
                *out.decay.add(at) = if fin { dc } else { f32::NAN };
            }
        }
    } else if hp >= 2 * inp.n_k {
        let h = hp - 2 * inp.n_k;
        // SAFETY: h < n_v <= ssm_a.len().
        let sa = unsafe { *inp.ssm_a.get_unchecked(h) };
        let mut fin = sa.is_finite();
        for i in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            let k = h * HEAD + lane + 32 * i;
            let at = t * inp.n_v * HEAD + k;
            // SAFETY: k < n_v·HEAD <= dt_bias.len() and t < m: `at` <
            // m·n_v·HEAD, inside decay_raw and decay by the contract; one
            // lane per (token, head, key channel).
            unsafe {
                let (f, dt) = (
                    *inp.decay_raw.get_unchecked(at),
                    *inp.dt_bias.get_unchecked(k),
                );
                let dc = kda_decay::<GATE>(f, dt, sa, inp.lb);
                let ok_k = f.is_finite() & dt.is_finite() & dc.is_finite();
                fin &= ok_k;
                *out.decay.add(at) = if ok_k & sa.is_finite() { dc } else { f32::NAN };
            }
        }
        if lane == 0 {
            let at = t * inp.n_v + h;
            // SAFETY: `at` < m·n_v, inside b_raw and beta; one lane per
            // (token, head).
            unsafe {
                let b = *inp.b_raw.get_unchecked(at);
                let bt = kda_beta::<BETA>(b);
                let ok_b = b.is_finite() & bt.is_finite();
                fin &= ok_b;
                *out.beta.add(at) = if ok_b { bt } else { f32::NAN };
            }
        }
        ok &= fin;
    }
    ok
}

/// The entries' body under the decay granularity `DECAY`, the decay gate
/// `GATE` and the β scale `BETA`, the entry's arguments forwarded flat: warp
/// `4·block + warp` is unit `w / H` ([`units`]) and conv head `w % H`, `H =
/// 2·n_k + n_v`; [`prep_token`] for each token of the unit, then unit 0 stores
/// the ring. A value that was not finite raises [`FaultSite::LinearConv`].
/// `gdn_conv_prep` and `gdn_conv_prep_bf16` are [`DECAY_HEAD`] under
/// [`KDA_GATE_SOFTPLUS`] and [`KDA_BETA_X1`], the KDA entries [`DECAY_KEY`].
/// `lb` is read by [`KDA_GATE_SIGMOID`] alone, `ba_pitch` by [`DECAY_HEAD`]
/// alone ([`PrepInputs`]); `x` is read through [`ConvIn`].
///
/// # Safety
///
/// The entries' launch contract: the slices hold the lengths it names, `m >=
/// 1`, `y`, `beta`, `decay` and `ring` hold the values it names and no other
/// launch touches them; the block is 128 threads and the grid `⌈units(m)·H /
/// 4⌉` blocks.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
unsafe fn conv_body<const DECAY: u32, const GATE: u32, const BETA: u32, X: ConvIn>(
    x: X,
    w: &[f32],
    b_raw: &[f32],
    decay_raw: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    pos: &[u32],
    lb: f32,
    eps: f32,
    n_k: u32,
    n_v: u32,
    ba_pitch: u32,
    m: u32,
    fault: FaultSink,
    y: &mut DisjointSlice<f32>,
    beta: &mut DisjointSlice<f32>,
    decay: &mut DisjointSlice<f32>,
    ring: &mut DisjointSlice<f32>,
) {
    let out = Outputs {
        y: y.as_mut_ptr(),
        beta: beta.as_mut_ptr(),
        decay: decay.as_mut_ptr(),
    };
    let wi = thread::blockIdx_x() as usize * 4 + thread::threadIdx_x() as usize / 32;
    // SAFETY: the contract's lengths and grid, forwarded.
    unsafe {
        conv_warp::<DECAY, GATE, BETA, X>(
            wi,
            x,
            w,
            (b_raw, decay_raw, dt_bias, ssm_a),
            pos,
            (lb, eps),
            (n_k, n_v, ba_pitch, m),
            fault,
            out,
            ring.as_mut_ptr(),
        );
    }
}

/// Warp `wi` of [`conv_body`]: unit `wi / H` and conv head `wi % H` over
/// the call's `m` tokens, its outputs written through `out` and the ring
/// `ring`; a warp past the call's `units(m)·H` writes nothing.
///
/// # Safety
///
/// [`conv_body`]'s contract, with `out` and `ring` addressing the values it
/// names for `y`, `beta`, `decay` and `ring`, which this warp alone touches
/// at its `(token, head)` outputs and head of the ring.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel entry's arguments, forwarded flat"
)]
#[inline(always)]
unsafe fn conv_warp<const DECAY: u32, const GATE: u32, const BETA: u32, X: ConvIn>(
    wi: usize,
    x: X,
    w: &[f32],
    (b_raw, decay_raw, dt_bias, ssm_a): (&[f32], &[f32], &[f32], &[f32]),
    pos: &[u32],
    (lb, eps): (f32, f32),
    (n_k, n_v, ba_pitch, m): (u32, u32, u32, u32),
    fault: FaultSink,
    out: Outputs,
    ring: *mut f32,
) {
    let heads = (2 * n_k + n_v) as usize;
    let ch = heads * HEAD;
    let tid = thread::threadIdx_x() as usize;
    let m = m as usize;
    if wi >= units(m) * heads {
        return; // warp-uniform
    }
    let lane = tid % 32;
    let u = wi / heads;
    let hp = wi - u * heads;
    let inp = PrepInputs {
        x,
        w,
        b_raw,
        decay_raw,
        dt_bias,
        ssm_a,
        lb,
        eps,
        n_k: n_k as usize,
        n_v: n_v as usize,
        ba_pitch: ba_pitch as usize,
    };
    let (t0, t1) = if u == 0 {
        (0, m.min(CONV_ROWS))
    } else {
        (u + CONV_ROWS - 1, u + CONV_ROWS)
    };
    // SAFETY: pos holds m >= 1 words by the contract.
    let p0 = unsafe { *pos.get_unchecked(0) } as u64;
    let mut ok = true;
    let mut t = t0;
    while t < t1 {
        // SAFETY: t < m <= pos.len().
        let p = unsafe { *pos.get_unchecked(t) };
        // SAFETY: t < m and hp < heads by the guard, lane < 32; the
        // contract's lengths, per key under DECAY_KEY for decay_raw,
        // dt_bias and decay; the ring is read only by unit 0, whose tokens
        // these are when t < CONV_ROWS, and written below by the same lanes
        // after these reads; (t, hp) is this warp's own.
        ok &= unsafe {
            prep_token::<DECAY, GATE, BETA, X>(
                t,
                p as usize,
                u64::from(p) == p0 + t as u64,
                hp,
                lane,
                inp,
                ring,
                out,
            )
        };
        t += 1;
    }
    if u == 0 {
        // SAFETY: hp < heads and lane < 32; the contract's lengths; the
        // ring's channels of head hp are this warp's, read above.
        unsafe { x.store_ring(pos, m, ch, hp, lane, ring) };
    }
    if !ok {
        fault.raise(FaultSite::LinearConv);
    }
}

#[cuda_module]
mod conv_kernels {
    use super::*;

    /// The conv and prep of `m` tokens (module doc). Warp `w = 4·block +
    /// warp` is unit `w / H` ([`units`]) and conv head `w % H`, `H = 2·n_k +
    /// n_v`: heads below `n_k` are queries, below `2·n_k` keys, the rest
    /// values. Token `t`'s β and α are `ba_pitch·t` values into `b_raw` and
    /// `a_raw`.
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
            x.len() >= m * (2 * n_k + n_v) * 128,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            ba_pitch >= n_v,
            b_raw.len() >= (m - 1) * ba_pitch + n_v,
            a_raw.len() >= (m - 1) * ba_pitch + n_v,
            dt_bias.len() >= n_v,
            ssm_a.len() >= n_v,
            pos.len() >= m,
            m >= 1,
            y.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            ring.len() >= 11 * (2 * n_k + n_v) * 128
        )
    )]
    pub fn gdn_conv_prep(
        x: &[f32],
        w: &[f32],
        b_raw: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        ba_pitch: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract's lengths and block. The bound is 0.0: the
        // sigmoid gate alone reads it, and this entry's gate is softplus.
        unsafe {
            conv_body::<DECAY_HEAD, KDA_GATE_SOFTPLUS, KDA_BETA_X1, _>(
                x, w, b_raw, a_raw, dt_bias, ssm_a, pos, 0.0, eps, n_k, n_v, ba_pitch, m, fault,
                &mut y, &mut beta, &mut decay, &mut ring,
            );
        }
    }

    /// [`gdn_conv_prep`] over several sequences' rows (`crate::seq_rows`):
    /// sequence `s` of `seqs` holds rows `lead .. lead + m_s` of every input
    /// and output and its own ring at word 0, and its warps are the `per`
    /// from `s·per` (`per = units(most)·H` for its longest sequence's
    /// `most` rows), each [`gdn_conv_prep`]'s warp over the sequence's rows
    /// alone; a warp past its sequence's units writes nothing.
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
            x.len() >= rows * (2 * n_k + n_v) * 128,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            ba_pitch >= n_v,
            b_raw.len() >= (rows - 1) * ba_pitch + n_v,
            a_raw.len() >= (rows - 1) * ba_pitch + n_v,
            dt_bias.len() >= n_v,
            ssm_a.len() >= n_v,
            pos.len() >= rows,
            rows >= 1,
            y.len() >= rows * (2 * n_k + n_v) * 128,
            beta.len() >= rows * n_v,
            decay.len() >= rows * n_v
        )
    )]
    pub fn gdn_conv_prep_seqs(
        x: &[f32],
        w: &[f32],
        b_raw: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        ba_pitch: u32,
        rows: u32,
        per: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        #[grid_constant] seqs: &SeqRows,
    ) {
        // rows only bounds the lengths in the launch contract.
        let _ = rows;
        let wi = thread::blockIdx_x() as usize * 4 + thread::threadIdx_x() as usize / 32;
        let s = wi / per as usize;
        if s >= seqs.len() {
            return; // warp-uniform
        }
        // SAFETY: s < the table's sequences.
        let ((lead, m), ring) = unsafe { (seqs.span(s), seqs.word(s, 0)) };
        let ch = (2 * n_k + n_v) as usize * HEAD;
        let (nv, pitch) = (n_v as usize, ba_pitch as usize);
        // SAFETY: the sequence's rows lead .. lead + m lie inside the launch's
        // `rows` (the table tiles them), so each window starts inside its
        // input and holds the m rows the contract names, the outputs' too;
        // the launcher put a ring of `RING_ROWS·C` f32 at word 0, the
        // sequence's own; this warp is the one-sequence entry's warp
        // `wi − s·per` over them.
        unsafe {
            let out = Outputs {
                y: y.as_mut_ptr().add(lead * ch),
                beta: beta.as_mut_ptr().add(lead * nv),
                decay: decay.as_mut_ptr().add(lead * nv),
            };
            conv_warp::<DECAY_HEAD, KDA_GATE_SOFTPLUS, KDA_BETA_X1, _>(
                wi - s * per as usize,
                x.get_unchecked(lead * ch..),
                w,
                (
                    b_raw.get_unchecked(lead * pitch..),
                    a_raw.get_unchecked(lead * pitch..),
                    dt_bias,
                    ssm_a,
                ),
                pos.get_unchecked(lead..),
                (0.0, eps),
                (n_k, n_v, ba_pitch, m as u32),
                fault,
                out,
                ring as usize as *mut f32,
            );
        }
    }

    /// [`gdn_conv_prep`] over channels held as BF16 words (`x`, value `2j`
    /// in the low half of word `j`), each value widened as it is read, the
    /// ring included, on the lanes the module doc's geometry names; `y` is
    /// 8-byte aligned. The launch bound budgets registers for six blocks
    /// of 128 threads per SM; the F32 entry has no minimum-block bound.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128, 6)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            x.len() >= m * (2 * n_k + n_v) * 64,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            ba_pitch >= n_v,
            b_raw.len() >= (m - 1) * ba_pitch + n_v,
            a_raw.len() >= (m - 1) * ba_pitch + n_v,
            dt_bias.len() >= n_v,
            ssm_a.len() >= n_v,
            pos.len() >= m,
            m >= 1,
            y.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v,
            ring.len() >= 11 * (2 * n_k + n_v) * 128
        )
    )]
    pub fn gdn_conv_prep_bf16(
        x: &[u32],
        w: &[f32],
        b_raw: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        ba_pitch: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        // SAFETY: 128 threads and the contract's checked input/output spans;
        // the host checks y's 8-byte alignment for the paired F32x2 stores.
        unsafe {
            conv_body::<DECAY_HEAD, KDA_GATE_SOFTPLUS, KDA_BETA_X1, _>(
                Bf16Words(x),
                w,
                b_raw,
                a_raw,
                dt_bias,
                ssm_a,
                pos,
                0.0,
                eps,
                n_k,
                n_v,
                ba_pitch,
                m,
                fault,
                &mut y,
                &mut beta,
                &mut decay,
                &mut ring,
            );
        }
    }

    /// The conv and prep of `m` tokens with one decay per (value head, key
    /// channel) (module doc): [`gdn_conv_prep`]'s warps and conv, then each
    /// value head's decay from `f`, `dt_bias`, `ssm_a` and `lb`.
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
            x.len() >= m * (2 * n_k + n_v) * 128,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            b_raw.len() >= m * n_v,
            f.len() >= m * n_v * 128,
            dt_bias.len() >= n_v * 128,
            ssm_a.len() >= n_v,
            pos.len() >= m,
            m >= 1,
            y.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v * 128,
            ring.len() >= 11 * (2 * n_k + n_v) * 128
        )
    )]
    pub fn kda_conv_prep(
        x: &[f32],
        w: &[f32],
        b_raw: &[f32],
        f: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        lb: f32,
        eps: f32,
        n_k: u32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract's lengths and block.
        unsafe {
            conv_body::<DECAY_KEY, KDA_GATE_SIGMOID, KDA_BETA_X1, _>(
                x, w, b_raw, f, dt_bias, ssm_a, pos, lb, eps, n_k, n_v, n_v, m, fault, &mut y,
                &mut beta, &mut decay, &mut ring,
            );
        }
    }

    /// [`kda_conv_prep`] under Solar-Open2's rule (module doc): the decay
    /// `exp(ssm_a · softplus(f + dt))`, no bound and so no `lb`, and `β =
    /// 2·sigmoid(b)`.
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
            x.len() >= m * (2 * n_k + n_v) * 128,
            w.len() >= 4 * (2 * n_k + n_v) * 128,
            b_raw.len() >= m * n_v,
            f.len() >= m * n_v * 128,
            dt_bias.len() >= n_v * 128,
            ssm_a.len() >= n_v,
            pos.len() >= m,
            m >= 1,
            y.len() >= m * (2 * n_k + n_v) * 128,
            beta.len() >= m * n_v,
            decay.len() >= m * n_v * 128,
            ring.len() >= 11 * (2 * n_k + n_v) * 128
        )
    )]
    pub fn kda_conv_prep_sp2(
        x: &[f32],
        w: &[f32],
        b_raw: &[f32],
        f: &[f32],
        dt_bias: &[f32],
        ssm_a: &[f32],
        pos: &[u32],
        eps: f32,
        n_k: u32,
        n_v: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut beta: DisjointSlice<f32>,
        mut decay: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract's lengths and block. The bound is 0.0: the
        // sigmoid gate alone reads it.
        unsafe {
            conv_body::<DECAY_KEY, KDA_GATE_SOFTPLUS, KDA_BETA_X2, _>(
                x, w, b_raw, f, dt_bias, ssm_a, pos, 0.0, eps, n_k, n_v, n_v, m, fault, &mut y,
                &mut beta, &mut decay, &mut ring,
            );
        }
    }
}

/// The projected channels a conv reads: f32 values, or BF16 words, two values
/// a word (value `2j` in the low half of word `j`), which the conv widens.
#[derive(Clone, Copy)]
pub enum ConvRows<'a> {
    F32(&'a DeviceBuffer<f32>),
    Bf16(&'a DeviceBuffer<u32>),
}

/// [`ConvKernels::enqueue_conv_prep`]'s arguments: `m` tokens of the
/// projected channels `x` (`[m][C]`), `b_raw` and `a_raw` (the β and α
/// projections, `n_v` values a token, each token's `ba_pitch` values past
/// the one before: `n_v` for their own rows, `2·n_v` for two windows of one
/// joined projection's rows, α's `n_v` values in), the layer's conv taps `w`
/// (`[C][CONV_TAPS]`, oldest first), `dt_bias` and `ssm_a` (`[n_v]`), the tokens' positions `pos`
/// (`[m]`, the words the rope reads), the buffers the call writes: the conv
/// output `y` (`[m][C]`), β and decay (`[m][n_v]`), and the layer's conv ring
/// (`[RING_ROWS][C]`), read and written in place.
pub struct ConvArgs<'a> {
    pub x: ConvRows<'a>,
    pub b_raw: &'a DeviceBuffer<f32>,
    pub a_raw: &'a DeviceBuffer<f32>,
    pub ba_pitch: usize,
    pub w: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub ssm_a: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub shape: LinearShape,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
}

/// One sequence of [`ConvKernels::enqueue_conv_prep_seqs`]: its rows of the
/// launch and its conv ring (`[RING_ROWS][C]`), read and written in place.
pub(crate) struct ConvSeq<'a> {
    pub(crate) rows: Range<usize>,
    pub(crate) ring: &'a mut DeviceBuffer<f32>,
}

/// [`ConvKernels::enqueue_conv_prep_seqs`]'s arguments: [`ConvArgs`]' over
/// the rows of every sequence, the channels F32, each sequence's rows and
/// ring in `seqs`.
pub(crate) struct ConvSeqsArgs<'a, 's> {
    pub(crate) x: &'a DeviceBuffer<f32>,
    pub(crate) b_raw: &'a DeviceBuffer<f32>,
    pub(crate) a_raw: &'a DeviceBuffer<f32>,
    pub(crate) ba_pitch: usize,
    pub(crate) w: &'a DeviceBuffer<f32>,
    pub(crate) dt_bias: &'a DeviceBuffer<f32>,
    pub(crate) ssm_a: &'a DeviceBuffer<f32>,
    pub(crate) pos: &'a DeviceBuffer<u32>,
    pub(crate) shape: LinearShape,
    pub(crate) eps: f32,
    pub(crate) fault: FaultSink,
    pub(crate) y: &'a mut DeviceBuffer<f32>,
    pub(crate) beta: &'a mut DeviceBuffer<f32>,
    pub(crate) decay: &'a mut DeviceBuffer<f32>,
    pub(crate) seqs: &'s [ConvSeq<'a>],
}

/// The refusals the conv's launches share over `m` rows of `shape` at
/// `ba_pitch`: the shape's, no row, a pitch below `n_v`, and a buffer
/// shorter than the rows read or written. `got` holds the lengths of `x`
/// (in values), `b_raw`, `a_raw`, `w`, `dt_bias`, `ssm_a`, `pos`, `y`, `beta`
/// and `decay`, in that order.
fn check_rows(
    what: &'static str,
    (shape, m, ba_pitch): (LinearShape, usize, usize),
    got: [usize; 10],
) -> Result<(), GpuError> {
    shape.check(what)?;
    if m == 0 {
        return Err(GpuError::shape(what, "need m >= 1, got m=0".to_owned()));
    }
    let ch = shape.channels();
    let nv = shape.n_v;
    if ba_pitch < nv {
        return Err(GpuError::shape(
            what,
            format!("ba_pitch {ba_pitch} < n_v {nv}: a token's β and α rows overlap the next's"),
        ));
    }
    let ba = (m - 1)
        .checked_mul(ba_pitch)
        .and_then(|v| v.checked_add(nv))
        .ok_or_else(|| GpuError::shape(what, format!("m={m} rows of pitch {ba_pitch}")))?;
    let need = [
        ("x", m * ch),
        ("b_raw", ba),
        ("a_raw", ba),
        ("w", CONV_TAPS * ch),
        ("dt_bias", nv),
        ("ssm_a", nv),
        ("pos", m),
        ("y", m * ch),
        ("beta", m * nv),
        ("decay", m * nv),
    ];
    match need.iter().zip(got).find(|&(&(_, need), got)| got < need) {
        Some((&(name, need), got)) => Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        )),
        None => Ok(()),
    }
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ConvKernels {
    module: conv_kernels::LoadedModule,
}

impl ConvKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ConvKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(conv_kernels, ctx)? };
        Ok(ConvKernels { module })
    }

    /// Enqueue the conv and prep of `args.m` tokens: `⌈units(m)·H / 4⌉`
    /// blocks of 128 threads, `H = 2·n_k + n_v`, the ring read and written
    /// in place; BF16 channels by `gdn_conv_prep_bf16`, bit for bit the F32
    /// rule over their widened values on that entry's lanes, refused by name
    /// when `y` is not 8-byte aligned. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_conv_prep(
        &self,
        stream: &CudaStream,
        args: ConvArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_conv_prep";
        let ConvArgs {
            x,
            b_raw,
            a_raw,
            ba_pitch,
            w,
            dt_bias,
            ssm_a,
            pos,
            shape,
            eps,
            m,
            fault,
            y,
            beta,
            decay,
            ring,
        } = args;
        let x_values = match x {
            ConvRows::F32(x) => x.len(),
            ConvRows::Bf16(x) => {
                check_alignment::<F32x2, _>(what, "y", y)?;
                2 * x.len()
            }
        };
        check_rows(
            what,
            (shape, m, ba_pitch),
            [
                x_values,
                b_raw.len(),
                a_raw.len(),
                w.len(),
                dt_bias.len(),
                ssm_a.len(),
                pos.len(),
                y.len(),
                beta.len(),
                decay.len(),
            ],
        )?;
        if ring.len() < shape.ring_len() {
            return Err(GpuError::shape(
                what,
                format!("ring.len() {} < {}", ring.len(), shape.ring_len()),
            ));
        }
        let nv = shape.n_v;
        let grid = launch_u32(what, "grid", (units(m) * shape.conv_heads()).div_ceil(4))?;
        let n_k = launch_u32(what, "n_k", shape.n_k)?;
        let n_v = launch_u32(what, "n_v", nv)?;
        let ba_pitch = launch_u32(what, "ba_pitch", ba_pitch)?;
        let m = launch_u32(what, "m", m)?;
        let (k, cfg) = (&self.module, LaunchConfig1D::new(grid, BLOCK, 0));
        match x {
            ConvRows::F32(x) => k.gdn_conv_prep(
                stream,
                &k.prepare_gdn_conv_prep(cfg)?,
                x,
                w,
                b_raw,
                a_raw,
                dt_bias,
                ssm_a,
                pos,
                eps,
                n_k,
                n_v,
                ba_pitch,
                m,
                fault,
                y,
                beta,
                decay,
                ring,
            )?,
            ConvRows::Bf16(x) => k.gdn_conv_prep_bf16(
                stream,
                &k.prepare_gdn_conv_prep_bf16(cfg)?,
                x,
                w,
                b_raw,
                a_raw,
                dt_bias,
                ssm_a,
                pos,
                eps,
                n_k,
                n_v,
                ba_pitch,
                m,
                fault,
                y,
                beta,
                decay,
                ring,
            )?,
        }
        Ok(())
    }

    /// Enqueue [`ConvKernels::enqueue_conv_prep`]'s F32 conv and prep of
    /// every sequence of `args` in one launch (`crate::seq_rows`): each
    /// sequence's rows over its own ring, bit for bit the launch of its rows
    /// alone, `⌈units(most)·H·n / 4⌉` blocks for `n` sequences of at most
    /// `most` rows. Refused by name: what that launch refuses over the
    /// sequences' rows, a ring shorter than the shape's, and rows that do not
    /// tile the launch from row 0 ([`SeqRows::new`]). Asynchronous and
    /// capturable; builds its sequence table on the host.
    pub(crate) fn enqueue_conv_prep_seqs(
        &self,
        stream: &CudaStream,
        args: ConvSeqsArgs<'_, '_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_conv_prep_seqs";
        let ConvSeqsArgs {
            x,
            b_raw,
            a_raw,
            ba_pitch,
            w,
            dt_bias,
            ssm_a,
            pos,
            shape,
            eps,
            fault,
            y,
            beta,
            decay,
            seqs,
        } = args;
        let mut words = Vec::with_capacity(seqs.len());
        for (s, q) in seqs.iter().enumerate() {
            if q.ring.len() < shape.ring_len() {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "sequence {s}'s ring.len() {} < {}",
                        q.ring.len(),
                        shape.ring_len()
                    ),
                ));
            }
            let mut word = [0; SEQ_WORDS];
            word[0] = q.ring.cu_deviceptr();
            words.push((q.rows.clone(), word));
        }
        let table = SeqRows::new(what, &words)?;
        let m = table.rows();
        check_rows(
            what,
            (shape, m, ba_pitch),
            [
                x.len(),
                b_raw.len(),
                a_raw.len(),
                w.len(),
                dt_bias.len(),
                ssm_a.len(),
                pos.len(),
                y.len(),
                beta.len(),
                decay.len(),
            ],
        )?;
        let per = units(table.most()) * shape.conv_heads();
        let grid = launch_u32(what, "grid", (per * table.len()).div_ceil(4))?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        let k = &self.module;
        // SAFETY: each sequence's word 0 is its ring of the shape's
        // `ring_len` f32s (checked above), borrowed mutably while this
        // enqueue runs, so no other sequence's names it; a graph that
        // captures the launch replays it over the same rings, which the
        // caller keeps alive and in place while the graph lives, as for any
        // buffer it captured.
        unsafe {
            k.gdn_conv_prep_seqs(
                stream,
                &k.prepare_gdn_conv_prep_seqs(cfg)?,
                x,
                w,
                b_raw,
                a_raw,
                dt_bias,
                ssm_a,
                pos,
                eps,
                launch_u32(what, "n_k", shape.n_k)?,
                launch_u32(what, "n_v", shape.n_v)?,
                launch_u32(what, "ba_pitch", ba_pitch)?,
                launch_u32(what, "rows", m)?,
                launch_u32(what, "per", per)?,
                fault,
                y,
                beta,
                decay,
                table,
            )?;
        }
        Ok(())
    }

    /// Enqueue the KDA conv and prep of `args.m` tokens: [`enqueue_conv_prep`]'s
    /// grid, with one decay per (value head, key channel) under `args.gate` and
    /// β scaled by `args.beta_scale`. Refuses by name a gate and scale that no
    /// entry runs together and a `Sigmoid` bound that is not below 0.
    /// Asynchronous, allocation-free, capturable.
    ///
    /// [`enqueue_conv_prep`]: ConvKernels::enqueue_conv_prep
    pub fn enqueue_kda_conv_prep(
        &self,
        stream: &CudaStream,
        args: KdaConvArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_kda_conv_prep";
        let KdaConvArgs {
            x,
            b_raw,
            f,
            w,
            dt_bias,
            ssm_a,
            pos,
            shape,
            gate,
            beta_scale,
            eps,
            m,
            fault,
            y,
            beta,
            decay,
            ring,
        } = args;
        shape.check(WHAT)?;
        if m == 0 {
            return Err(GpuError::shape(WHAT, "need m >= 1, got m=0".to_owned()));
        }
        if let KdaGateRule::Sigmoid { lower_bound } = gate
            && (lower_bound.is_nan() || lower_bound >= 0.0)
        {
            return Err(GpuError::shape(
                WHAT,
                format!("lb={lower_bound}: the decay's bound must be below 0"),
            ));
        }
        let ch = shape.channels();
        let nv = shape.n_v;
        let lens = [
            ("x", x.len(), m * ch),
            ("b_raw", b_raw.len(), m * nv),
            ("f", f.len(), m * nv * HEAD),
            ("w", w.len(), CONV_TAPS * ch),
            ("dt_bias", dt_bias.len(), nv * HEAD),
            ("ssm_a", ssm_a.len(), nv),
            ("pos", pos.len(), m),
            ("y", y.len(), m * ch),
            ("beta", beta.len(), m * nv),
            ("decay", decay.len(), m * nv * HEAD),
            ("ring", ring.len(), shape.ring_len()),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                WHAT,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(WHAT, "grid", (units(m) * shape.conv_heads()).div_ceil(4))?;
        let n_k = launch_u32(WHAT, "n_k", shape.n_k)?;
        let n_v = launch_u32(WHAT, "n_v", nv)?;
        let m = launch_u32(WHAT, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        match (gate, beta_scale) {
            (KdaGateRule::Sigmoid { lower_bound }, KdaBetaScale::X1) => {
                let prep = self.module.prepare_kda_conv_prep(cfg)?;
                self.module.kda_conv_prep(
                    stream,
                    &prep,
                    x,
                    w,
                    b_raw,
                    f,
                    dt_bias,
                    ssm_a,
                    pos,
                    lower_bound,
                    eps,
                    n_k,
                    n_v,
                    m,
                    fault,
                    y,
                    beta,
                    decay,
                    ring,
                )?;
            }
            (KdaGateRule::Softplus, KdaBetaScale::X2) => {
                let prep = self.module.prepare_kda_conv_prep_sp2(cfg)?;
                self.module.kda_conv_prep_sp2(
                    stream, &prep, x, w, b_raw, f, dt_bias, ssm_a, pos, eps, n_k, n_v, m, fault, y,
                    beta, decay, ring,
                )?;
            }
            _ => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("no KDA entry runs gate {gate:?} with beta scale {beta_scale:?}"),
                ));
            }
        }
        Ok(())
    }
}

/// The KDA decay's rule: what each model's `A_log` and forget projection mean
/// to the prep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KdaGateRule {
    /// Bounded sigmoid, `exp(lower_bound · sigmoid(−ssm_a·(f + dt)))` with
    /// `lower_bound` below 0, bounding the decay to `[e^lower_bound, 1]`
    /// (GLM-5.3-Flash; `kda_conv_prep`).
    Sigmoid { lower_bound: f32 },
    /// `exp(ssm_a · softplus(f + dt))`, no bound (Solar-Open2;
    /// `kda_conv_prep_sp2`).
    Softplus,
}

/// The scale on the KDA β = `sigmoid(b)`; `X2` is exact in f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdaBetaScale {
    /// `sigmoid(b)` (`kda_conv_prep`).
    X1,
    /// `2 · sigmoid(b)` (`kda_conv_prep_sp2`).
    X2,
}

/// [`ConvKernels::enqueue_kda_conv_prep`]'s arguments: [`ConvArgs`] with the
/// per-key decay's inputs in place of `a_raw` — `f` (`[m][n_v·HEAD]`, the
/// low-rank forget projection's output), `dt_bias` (`[n_v·HEAD]`), `ssm_a`
/// (`[n_v]`, `−e^A_log`) and the rule `gate` — and the decay written
/// `[m][n_v][HEAD]`, β `[m][n_v]` scaled by `beta_scale`.
pub struct KdaConvArgs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub b_raw: &'a DeviceBuffer<f32>,
    pub f: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    pub ssm_a: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub shape: LinearShape,
    pub gate: KdaGateRule,
    pub beta_scale: KdaBetaScale,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub ring: &'a mut DeviceBuffer<f32>,
}

/// What [`conv_prep_host`] and [`kda_conv_prep_host`] compute: the conv output, β, decay and the ring after the call, in the kernel's layouts.
pub struct ConvOut {
    pub y: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub ring: Vec<f32>,
}

/// The shared conv part of the host rules: `y` and the ring after the call,
/// the module doc's conv, norm and position rule op for op, on the lanes of
/// `gdn_conv_prep_bf16` when `bf16`.
#[allow(
    clippy::too_many_arguments,
    reason = "the conv inputs of one launch and its lane layout"
)]
fn conv_host_part(
    x: &[f32],
    w: &[f32],
    pos: &[u32],
    ring: &[f32],
    shape: LinearShape,
    eps: f32,
    m: usize,
    bf16: bool,
) -> (Vec<f32>, Vec<f32>) {
    let ch = shape.channels();
    let heads = shape.conv_heads();
    let mut y = vec![0.0f32; m * ch];
    for t in 0..m {
        let p = pos[t] as usize;
        let contiguous = u64::from(pos[t]) == u64::from(pos[0]) + t as u64;
        for hp in 0..heads {
            let mut vals = [0.0f32; HEAD];
            let mut lanes = [0.0f64; 32];
            for (lane, sum) in lanes.iter_mut().enumerate() {
                for i in 0..4 {
                    let k = if bf16 {
                        pair_channel(lane, i)
                    } else {
                        lane + 32 * i
                    };
                    let c = hp * HEAD + k;
                    let xs: [f32; 4] = std::array::from_fn(|j| {
                        let d = CONV_ROWS - j;
                        if d <= t {
                            x[(t - d) * ch + c]
                        } else if p < d {
                            0.0
                        } else {
                            ring[(p + RING_ROWS - d) % RING_ROWS * ch + c]
                        }
                    });
                    let mut acc = xs[0] * w[4 * c];
                    for j in 1..CONV_TAPS {
                        acc = xs[j].mul_add(w[4 * c + j], acc);
                    }
                    let s = silu(acc);
                    vals[k] = s;
                    *sum += f64::from(s * s);
                }
            }
            if hp < 2 * shape.n_k {
                let sum = super::butterfly_f64(&lanes) as f32;
                let scale = if sum.is_finite() {
                    1.0 / sum.sqrt().max(eps)
                } else {
                    f32::NAN
                };
                for v in &mut vals {
                    *v *= scale;
                    if hp < shape.n_k {
                        *v *= Q_SCALE;
                    }
                }
            }
            if !contiguous {
                vals.fill(f32::NAN);
            }
            y[t * ch + hp * HEAD..t * ch + (hp + 1) * HEAD].copy_from_slice(&vals);
        }
    }
    let mut ring = ring[..shape.ring_len()].to_vec();
    for t in m.saturating_sub(RING_ROWS)..m {
        let slot = pos[t] as usize % RING_ROWS;
        ring[slot * ch..(slot + 1) * ch].copy_from_slice(&x[t * ch..(t + 1) * ch]);
    }
    (y, ring)
}

/// The host rule of [`ConvKernels::enqueue_conv_prep`]: the module doc's
/// numeric contract and position rule, op for op. Inputs as [`ConvArgs`]
/// names them, host slices of the same lengths; `ring` is the ring before
/// the call. `bf16`: `x` holds the widened values of BF16 channels
/// ([`ConvRows::Bf16`]), on that entry's lanes.
///
/// # Panics
///
/// When a slice is shorter than [`ConvArgs`]'s lengths for `shape` and `m`.
#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
#[must_use]
pub fn conv_prep_host(
    x: &[f32],
    b_raw: &[f32],
    a_raw: &[f32],
    w: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    pos: &[u32],
    ring: &[f32],
    shape: LinearShape,
    eps: f32,
    m: usize,
    bf16: bool,
) -> ConvOut {
    let nv = shape.n_v;
    let (y, ring) = conv_host_part(x, w, pos, ring, shape, eps, m, bf16);
    let mut beta = vec![0.0f32; m * nv];
    let mut decay = vec![0.0f32; m * nv];
    for t in 0..m {
        for h in 0..nv {
            let at = t * nv + h;
            let (b, a, dt, sa) = (b_raw[at], a_raw[at], dt_bias[h], ssm_a[h]);
            let bt = sigmoid(b);
            let dc = expf_ik(softplus(a + dt) * sa);
            let fin = [b, a, dt, sa, bt, dc].iter().all(|v| v.is_finite());
            beta[at] = if fin { bt } else { f32::NAN };
            decay[at] = if fin { dc } else { f32::NAN };
        }
    }
    ConvOut {
        y,
        beta,
        decay,
        ring,
    }
}

/// The host twin of the KDA prep's per-key decay under `gate`. `Sigmoid`:
/// `r = (f + dt)·ssm_a`, then `expf_ik(lower_bound·sigmoid(−r))`, each product
/// rounded, in ik's op order (`build_kda_beta_gate`: add, mul, scale by −1,
/// sigmoid, scale by the bound; the delta op's exp). `Softplus`:
/// `expf_ik(softplus(f + dt)·ssm_a)`, the product rounded, the operations of
/// `gdn_conv_prep`'s decay.
#[must_use]
pub fn kda_decay_host(f: f32, dt: f32, sa: f32, gate: KdaGateRule) -> f32 {
    match gate {
        KdaGateRule::Sigmoid { lower_bound } => {
            let r = (f + dt) * sa;
            expf_ik(lower_bound * sigmoid(-r))
        }
        KdaGateRule::Softplus => expf_ik(softplus(f + dt) * sa),
    }
}

/// The host twin of the KDA prep's β: [`sigmoid`]`(b)` times the scale.
#[must_use]
pub fn kda_beta_host(b: f32, scale: KdaBetaScale) -> f32 {
    match scale {
        KdaBetaScale::X1 => sigmoid(b),
        KdaBetaScale::X2 => sigmoid(b) * 2.0,
    }
}

/// The host rule of [`ConvKernels::enqueue_kda_conv_prep`]: the conv part of
/// [`conv_prep_host`], β by [`kda_beta_host`] and the decay of every (token,
/// value head, key channel) by [`kda_decay_host`]. Inputs as [`KdaConvArgs`]
/// names them, host slices of the same lengths.
///
/// # Panics
///
/// When a slice is shorter than [`KdaConvArgs`]'s lengths for `shape` and
/// `m`.
#[allow(
    clippy::too_many_arguments,
    reason = "the host twin of one launch: its inputs, each named as the launch names them"
)]
#[must_use]
pub fn kda_conv_prep_host(
    x: &[f32],
    b_raw: &[f32],
    f: &[f32],
    w: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    pos: &[u32],
    ring: &[f32],
    shape: LinearShape,
    gate: KdaGateRule,
    beta_scale: KdaBetaScale,
    eps: f32,
    m: usize,
) -> ConvOut {
    let nv = shape.n_v;
    let (y, ring) = conv_host_part(x, w, pos, ring, shape, eps, m, false);
    let mut beta = vec![0.0f32; m * nv];
    let mut decay = vec![0.0f32; m * nv * HEAD];
    for t in 0..m {
        for h in 0..nv {
            let at = t * nv + h;
            let b = b_raw[at];
            let bt = kda_beta_host(b, beta_scale);
            beta[at] = if b.is_finite() && bt.is_finite() {
                bt
            } else {
                f32::NAN
            };
            let sa = ssm_a[h];
            for k in 0..HEAD {
                let (fk, dt) = (f[at * HEAD + k], dt_bias[h * HEAD + k]);
                let dc = kda_decay_host(fk, dt, sa, gate);
                let fin = [fk, dt, sa, dc].iter().all(|v| v.is_finite());
                decay[at * HEAD + k] = if fin { dc } else { f32::NAN };
            }
        }
    }
    ConvOut {
        y,
        beta,
        decay,
        ring,
    }
}

#[cfg(test)]
mod tests {
    use super::{KdaBetaScale, KdaGateRule, conv_prep_host, kda_conv_prep_host, kda_decay_host};
    use crate::linear::{CONV_TAPS, HEAD, KHeadMap, LinearShape};

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

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The two preps share the conv, the norm, β and the ring: on the same
    /// channels, taps, β projection, ring and positions both write the same
    /// `y`, β and ring bit for bit.
    #[test]
    fn kda_prep_shares_the_conv_norm_beta_and_ring() {
        let shape = LinearShape {
            n_k: 2,
            n_v: 2,
            map: KHeadMap::Tiled,
        };
        let (m, ch, nv) = (6, shape.channels(), shape.n_v);
        let mut r = 0x636f_6e31;
        let x = fill(&mut r, m * ch, -1.0, 1.0);
        let w = fill(&mut r, CONV_TAPS * ch, -0.5, 0.5);
        let b = fill(&mut r, m * nv, -3.0, 3.0);
        let a = fill(&mut r, m * nv, -3.0, 3.0);
        let f = fill(&mut r, m * nv * HEAD, -3.0, 3.0);
        let dt_h = fill(&mut r, nv, -1.0, 1.0);
        let dt_k = fill(&mut r, nv * HEAD, -1.0, 1.0);
        let sa = fill(&mut r, nv, -5.0, -0.1);
        let ring = fill(&mut r, shape.ring_len(), -1.0, 1.0);
        let pos: Vec<u32> = (40..40 + m as u32).collect();
        let g = conv_prep_host(
            &x, &b, &a, &w, &dt_h, &sa, &pos, &ring, shape, 1e-5, m, false,
        );
        let k = kda_conv_prep_host(
            &x,
            &b,
            &f,
            &w,
            &dt_k,
            &sa,
            &pos,
            &ring,
            shape,
            KdaGateRule::Sigmoid { lower_bound: -5.0 },
            KdaBetaScale::X1,
            1e-5,
            m,
        );
        assert_eq!(bits(&g.y), bits(&k.y));
        assert_eq!(bits(&g.beta), bits(&k.beta));
        assert_eq!(bits(&g.ring), bits(&k.ring));
        assert_eq!(k.decay.len(), m * nv * HEAD);
    }

    /// The per-key decay against `e^(lb·σ(e^A_log·(f + dt)))` in f64 on the
    /// same f32 inputs: within 4e-6 relative (the f32 chain's roundings,
    /// amplified by |g| <= 5 through the exponential), and inside `[e^lb, 1]`.
    #[test]
    fn kda_decay_is_the_bounded_sigmoid_rule() {
        let mut r = 0x6463_6179;
        let lb = -5.0f32;
        let gate = KdaGateRule::Sigmoid { lower_bound: lb };
        let f = fill(&mut r, 1 << 16, -8.0, 8.0);
        let dt = fill(&mut r, 1 << 16, -1.0, 1.0);
        let sa: Vec<f32> = fill(&mut r, 1 << 16, -3.0, 3.0)
            .iter()
            .map(|&u| -u.exp())
            .collect();
        let lo = f64::from(lb).exp();
        for i in 0..f.len() {
            let d = kda_decay_host(f[i], dt[i], sa[i], gate);
            let e = -f64::from(sa[i]) * (f64::from(f[i]) + f64::from(dt[i]));
            let want = (f64::from(lb) / (1.0 + (-e).exp())).exp();
            let rel = (f64::from(d) - want).abs() / want;
            assert!(rel <= 4e-6, "i={i}: {d} vs {want} rel {rel:e}");
            assert!(
                f64::from(d) >= lo * (1.0 - 4e-6) && d <= 1.0,
                "i={i}: {d} outside [e^lb, 1]"
            );
        }
    }

    /// The softplus prep shares the conv, the norm and the ring with the
    /// bounded one and writes β doubled: on the same inputs `y` and the ring
    /// are equal bit for bit, and each β is exactly twice the bounded prep's.
    #[test]
    fn kda_sp2_prep_is_the_bounded_conv_with_twice_beta() {
        let shape = LinearShape {
            n_k: 2,
            n_v: 2,
            map: KHeadMap::Tiled,
        };
        let (m, ch, nv) = (6, shape.channels(), shape.n_v);
        let mut r = 0x7370_3201;
        let x = fill(&mut r, m * ch, -1.0, 1.0);
        let w = fill(&mut r, CONV_TAPS * ch, -0.5, 0.5);
        let b = fill(&mut r, m * nv, -3.0, 3.0);
        let f = fill(&mut r, m * nv * HEAD, -3.0, 3.0);
        let dt = fill(&mut r, nv * HEAD, -1.0, 1.0);
        let sa = fill(&mut r, nv, -5.0, -0.1);
        let ring = fill(&mut r, shape.ring_len(), -1.0, 1.0);
        let pos: Vec<u32> = (40..40 + m as u32).collect();
        let run = |gate, beta_scale| {
            kda_conv_prep_host(
                &x, &b, &f, &w, &dt, &sa, &pos, &ring, shape, gate, beta_scale, 1e-5, m,
            )
        };
        let g = run(KdaGateRule::Sigmoid { lower_bound: -5.0 }, KdaBetaScale::X1);
        let s = run(KdaGateRule::Softplus, KdaBetaScale::X2);
        assert_eq!(bits(&g.y), bits(&s.y));
        assert_eq!(bits(&g.ring), bits(&s.ring));
        let twice: Vec<f32> = g.beta.iter().map(|v| 2.0 * v).collect();
        assert_eq!(bits(&twice), bits(&s.beta));
        assert_eq!(s.decay.len(), m * nv * HEAD);
    }

    /// The per-key decay against `exp(ssm_a·ln(1 + e^(f + dt)))` in f64 on the
    /// same f32 inputs, with `|ssm_a| <= e` and `|f + dt| <= 9` so the decay
    /// stays a normal f32, `g` the exponent: the band is 4e-7 for the exp
    /// polynomial and the final rounding, `1e-7·|ssm_a|` for the rounded
    /// `1 + e^x` before the log (half an ulp) and ggml's zero below `x ≈
    /// −17`, and `|g|·8e-7` for the rounded sum, the log polynomial and the
    /// product; and the decay stays inside `[0, 1]`.
    #[test]
    fn kda_decay_is_the_softplus_rule() {
        let mut r = 0x7370_6463;
        let f = fill(&mut r, 1 << 16, -8.0, 8.0);
        let dt = fill(&mut r, 1 << 16, -1.0, 1.0);
        let sa: Vec<f32> = fill(&mut r, 1 << 16, -3.0, 1.0)
            .iter()
            .map(|&u| -u.exp())
            .collect();
        for (i, ((&f, &dt), &sa)) in f.iter().zip(&dt).zip(&sa).enumerate() {
            let d = kda_decay_host(f, dt, sa, KdaGateRule::Softplus);
            let x = f64::from(f) + f64::from(dt);
            let g = f64::from(sa) * x.exp().ln_1p();
            let want = g.exp();
            let rel = (f64::from(d) - want).abs() / want;
            let band = 4e-7 + 1e-7 * f64::from(sa).abs() + 8e-7 * g.abs();
            assert!(
                rel <= band,
                "i={i}: {d} vs {want} rel {rel:e} band {band:e}"
            );
            assert!((0.0..=1.0).contains(&d), "i={i}: {d} outside [0, 1]");
        }
    }
}
