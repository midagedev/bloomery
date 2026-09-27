//! The card kernels of a Gated DeltaNet layer (linear attention): the causal
//! conv with SiLU, the q/k L2 norm and the β/decay prep ([`conv`]), the
//! recurrent delta step over a run of tokens ([`delta`]), and the gated
//! per-head RMS norm of its output ([`norm_gate`]). The projections around
//! them (`W_qkv`, `w_β`, `w_α`, `W_gate`, `W_out`) are the tree's dense
//! gemv/GEMM and are not here.
//!
//! The rule, per token t and value head h (`n_v` value heads, `n_k` query/key
//! heads, every head [`HEAD`] wide, `C = (2·n_k + n_v)·HEAD` channels):
//!
//! ```text
//! [q̃; k̃; ṽ] = SiLU(conv4(x))              depthwise causal conv over the last CONV_TAPS inputs
//! q, k      = l2norm(q̃_kh), l2norm(k̃_kh)   kh = KHeadMap::k_head(h)
//! β         = sigmoid(b)                   decay = exp(ssm_a · softplus(a + dt_bias))
//! S'        = decay · S                    u = β · (v − S'ᵀk)        S = S' + k uᵀ
//! o         = Sᵀ (q / √HEAD)               y = RMSNorm_w(o_h) ⊙ act(z_h)
//! ```
//!
//! Kimi Delta Attention (GLM-5.3-Flash) takes the same steps with one decay
//! per key channel, `decay_hk = exp(lb · sigmoid(−ssm_a_h · (f_hk + dt_hk)))`
//! from the low-rank forget projection `f`, scaling state key `k`, and the
//! sigmoid gate.
//!
//! Layouts, all f32 and token-major:
//! - channels `[m][C]`: q heads, then k heads, then v heads, [`HEAD`] each —
//!   the projection's output order, and the conv's output in the same order
//!   (q already L2-normed and scaled by [`Q_SCALE`], k L2-normed, v as the
//!   conv left it);
//! - conv ring `[RING_ROWS][C]`: the conv input of position `p` in slot
//!   `p mod RING_ROWS`, so the ring holds the inputs of the last
//!   [`RING_ROWS`] positions a call wrote;
//! - recurrent state `[lanes][n_v][HEAD v][HEAD k]`: value column v of head
//!   h in lane `l` is the HEAD keys at `((l·n_v + h)·HEAD + v)·HEAD`,
//!   contiguous. ik stores the transpose (`[k][v]`, v contiguous);
//! - β `[m][n_v]`; decay `[m][n_v]`, or `[m][n_v][HEAD]` per key channel
//!   (KDA, with its `f` and `dt_bias` `[m][n_v·HEAD]` and `[n_v·HEAD]`); the
//!   delta output `o` and the gate's `z` and `y` `[m][n_v][HEAD]`.
//!
//! The state is read and written in place: the conv finds a token's
//! predecessors by its position (`pos[t]`, the words the embedding launch
//! writes), the delta step its lane through a device word, so one captured
//! graph serves every step. Every exponential is [`expf_ik`] and the one logarithm is
//! [`logf_poly`]: both are fused multiply-adds and bit operations only, so a
//! host transcription rounds as the card does. Each kernel raises a site of
//! its own on the fault word for a non-finite input or a result that stops
//! being finite, and writes a NaN there, never a plausible value.

pub mod conv;
pub mod delta;
pub mod norm_gate;

use crate::GpuError;
use cuda_core::CudaContext;
use std::sync::Arc;

/// Width of every query, key and value head (d_k = d_v).
pub const HEAD: usize = 128;

/// Taps of the causal conv.
pub const CONV_TAPS: usize = 4;

/// A token's predecessors the conv reads: the inputs of its last
/// `CONV_TAPS − 1` positions.
pub const CONV_ROWS: usize = CONV_TAPS - 1;

/// The widest call a later call may roll back into: the pass of up to eight
/// positions (a verify of up to seven drafted tokens).
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring. A rollback to position `n` inside a call of `m <=
/// PASS_ROWS` positions from `p` reads the inputs of `n − 3 .. n − 1 >= p −
/// 2`, which the ring holds while it keeps `m + 2` positions; with `m + 3`
/// the call's writes also never land on the slots of `p − 3 .. p − 1`.
pub const RING_ROWS: usize = CONV_ROWS + PASS_ROWS;

/// `1 / √HEAD` rounded to f32: the query scale the conv applies after the
/// L2 norm.
pub const Q_SCALE: f32 = f32::from_bits(0x3db5_04f3);

/// Threads per block of every kernel here: four warps.
pub(crate) const BLOCK: u32 = 128;

/// The query/key head value head `h` reads when `n_v > n_k`: a property of
/// the file's converter, not of the architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KHeadMap {
    /// `h % n_k`: the converter tiled the value heads (Qwen3.5 and later).
    Tiled,
    /// `h / (n_v / n_k)`: grouped value heads (Qwen3-Next).
    Grouped,
}

impl KHeadMap {
    /// The key head of value head `h`.
    #[must_use]
    pub fn k_head(self, h: usize, n_k: usize, n_v: usize) -> usize {
        match self {
            KHeadMap::Tiled => h % n_k,
            KHeadMap::Grouped => h / (n_v / n_k),
        }
    }

    /// The launch argument the delta kernel reads.
    pub(crate) fn code(self) -> u32 {
        match self {
            KHeadMap::Tiled => 0,
            KHeadMap::Grouped => 1,
        }
    }
}

/// The decay's granularity, the delta body's const parameter: one scalar
/// per value head (Gated DeltaNet, `gdn_conv_prep` and `gdn_delta`).
pub const DECAY_HEAD: u32 = 0;
/// One decay per (value head, key channel) (Kimi Delta Attention,
/// `kda_conv_prep` and `kda_delta`).
pub const DECAY_KEY: u32 = 1;

/// The output gate's activation, the norm body's const parameter: SiLU
/// (Qwen3.5/3.6, Qwen3-Next; `gdn_norm_gate`).
pub const GATE_SILU: u32 = 0;
/// `sigmoid(z)` (Qwen3.8-Flash-Next, Kimi Delta Attention;
/// `gdn_norm_gate_sigmoid`).
pub const GATE_SIGMOID: u32 = 1;

/// The head counts of one layer. `HEAD`, the conv width and the state layout
/// are fixed; these are the launch arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearShape {
    /// Query/key heads.
    pub n_k: usize,
    /// Value heads, a multiple of `n_k`.
    pub n_v: usize,
    /// How a value head finds its key head.
    pub map: KHeadMap,
}

impl LinearShape {
    /// Channels of the conv: `(2·n_k + n_v)·HEAD`.
    #[must_use]
    pub fn channels(&self) -> usize {
        (2 * self.n_k + self.n_v) * HEAD
    }

    /// Heads of the conv, each [`HEAD`] channels: `2·n_k + n_v`.
    #[must_use]
    pub fn conv_heads(&self) -> usize {
        2 * self.n_k + self.n_v
    }

    /// f32s of one lane of the recurrent state: `n_v·HEAD·HEAD`.
    #[must_use]
    pub fn state_len(&self) -> usize {
        self.n_v * HEAD * HEAD
    }

    /// f32s of the conv ring: `RING_ROWS·C`.
    #[must_use]
    pub fn ring_len(&self) -> usize {
        RING_ROWS * self.channels()
    }

    /// Refuse by name a shape the kernels have no geometry for.
    pub(crate) fn check(&self, what: &'static str) -> Result<(), GpuError> {
        if self.n_k == 0 || self.n_v == 0 || !self.n_v.is_multiple_of(self.n_k) {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_k >= 1 and n_v a positive multiple of n_k, got n_k={} n_v={}",
                    self.n_k, self.n_v
                ),
            ));
        }
        Ok(())
    }
}

/// The three modules, loaded once.
pub struct LinearKernels {
    pub conv: conv::ConvKernels,
    pub delta: delta::DeltaKernels,
    pub norm_gate: norm_gate::NormGateKernels,
}

impl LinearKernels {
    /// Load the three device bundles into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<LinearKernels, GpuError> {
        Ok(LinearKernels {
            conv: conv::ConvKernels::load(ctx)?,
            delta: delta::DeltaKernels::load(ctx)?,
            norm_gate: norm_gate::NormGateKernels::load(ctx)?,
        })
    }
}

/// ik's AVX2 `v_expf` (`iqk_utils.h`), one lane, op for op: `x` split as
/// `n·ln2 + b` by the `0x1.8p23` shift (two fused multiply-adds for `b`), a
/// degree-5 polynomial in `b` by fused multiply-adds, and the scale `2^n` put
/// into the exponent bits — with ik's two escape paths for `|n| > 126` (split
/// scale) and `|n| > 192` (overflow to infinity, underflow to zero). Every
/// product that meets an add is an explicit `mul_add`, so the device and the
/// host round the same operations; the two plain products of the escape paths
/// scale by a power of two, exact unless the result leaves the normal range,
/// where adding 1 rounds to the same value fused or not.
#[inline(always)]
#[must_use]
pub fn expf_ik(x: f32) -> f32 {
    const SHIFT: f32 = f32::from_bits(0x4b40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C0: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    const C1: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const C2: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const C3: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const C4: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    let z = x.mul_add(LOG2E, SHIFT);
    let n = z - SHIFT;
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = C4
        .mul_add(b, C3)
        .mul_add(u, C2.mul_add(b, C1))
        .mul_add(u, C0 * b);
    // An ordered compare, as ik's: a NaN `n` takes the main path.
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(g));
        return if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(j, s2) * s1
        };
    }
    j.mul_add(k, k)
}

/// The natural logarithm of a positive normal `x`, Cephes' `logf` with its
/// products spelled as fused multiply-adds: `x = f·2^e` with `f` in
/// `[√½, √2)`, a degree-8 polynomial in `f − 1`, then `e·ln2` in two parts.
/// The subtractions that form `f − 1` are exact. Not for zero, subnormals,
/// infinities or NaN (its one caller, [`softplus`], passes `1 + e^x`).
#[inline(always)]
#[must_use]
pub fn logf_poly(x: f32) -> f32 {
    const SQRTHF: f32 = f32::from_bits(0x3f35_04f3);
    const P0: f32 = f32::from_bits(0x3d90_21bb); // 7.0376836292e-2
    const P1: f32 = f32::from_bits(0xbdeb_d1b8); // -1.1514610310e-1
    const P2: f32 = f32::from_bits(0x3def_251a); // 1.1676998740e-1
    const P3: f32 = f32::from_bits(0xbdfe_5d4f); // -1.2420140846e-1
    const P4: f32 = f32::from_bits(0x3e11_e9bf); // 1.4249322787e-1
    const P5: f32 = f32::from_bits(0xbe2a_ae50); // -1.6668057665e-1
    const P6: f32 = f32::from_bits(0x3e4c_ceac); // 2.0000714765e-1
    const P7: f32 = f32::from_bits(0xbe7f_fffc); // -2.4999993993e-1
    const P8: f32 = f32::from_bits(0x3eaa_aaaa); // 3.3333331174e-1
    const LN2_LO: f32 = f32::from_bits(0xb95e_8083); // -2.12194440e-4
    const LN2_HI: f32 = f32::from_bits(0x3f31_8000); // 0.693359375
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff).cast_signed() - 126;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let low = m < SQRTHF;
    e -= i32::from(low);
    let f = if low { m + m } else { m } - 1.0;
    let z = f * f;
    let p = P0
        .mul_add(f, P1)
        .mul_add(f, P2)
        .mul_add(f, P3)
        .mul_add(f, P4)
        .mul_add(f, P5)
        .mul_add(f, P6)
        .mul_add(f, P7)
        .mul_add(f, P8);
    let fe = e as f32;
    let y = z.mul_add(-0.5, fe.mul_add(LN2_LO, p * f * z));
    fe.mul_add(LN2_HI, f + y)
}

/// `x / (1 + e^(0 − x))`: ik's `v_silu` with [`expf_ik`].
#[inline(always)]
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf_ik(0.0 - x))
}

/// `1 / (1 + e^(0 − x))` with [`expf_ik`].
#[inline(always)]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf_ik(0.0 - x))
}

/// ggml's softplus: `x` above 20, else `ln(1 + e^x)` with [`expf_ik`] and
/// [`logf_poly`]. The sum `1 + e^x` is rounded before the logarithm, as
/// ggml's `logf(1.0f + expf(x))` does, so for `x` below about −17 the result
/// is 0 rather than `e^x`.
#[inline(always)]
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        logf_poly(1.0 + expf_ik(x))
    }
}

/// The host transcription of a warp's xor butterfly over `lanes` (32 or 8
/// partials, one per lane, lane order): at each distance `d` from
/// `lanes/2` down to 1, lane `l` adds lane `l ^ d`'s value to its own. Every
/// lane ends with the same bits; this returns lane 0's.
#[must_use]
pub fn butterfly_f32(lanes: &[f32]) -> f32 {
    let mut cur = lanes.to_vec();
    let mut d = cur.len() / 2;
    while d > 0 {
        cur = (0..cur.len()).map(|l| cur[l] + cur[l ^ d]).collect();
        d /= 2;
    }
    cur[0]
}

/// [`butterfly_f32`] in f64.
#[must_use]
pub fn butterfly_f64(lanes: &[f64]) -> f64 {
    let mut cur = lanes.to_vec();
    let mut d = cur.len() / 2;
    while d > 0 {
        cur = (0..cur.len()).map(|l| cur[l] + cur[l ^ d]).collect();
        d /= 2;
    }
    cur[0]
}
