//! The KDA mixer of one layer over `m` rows: the one launch sequence every
//! KDA family runs (GLM-5.3-Flash, Solar-Open2), over the family's own
//! projections ([`KdaProj`]) and the buffers its body lends ([`KdaBufs`]).
//! The step is `m` = 1; a prompt batch is its `m` rows in position order,
//! bit for bit the steps (the conv and the delta step walk the rows in order).
//!
//! The launches, in order (ik `src/llama-kda.cpp`):
//! 1. `attn_norm` RMS of `x` into `xn`;
//! 2. [`KdaProj::input`]: the joined q·k·v projection, the forget and gate
//!    low-rank halves (`ssm_f_a`, `ssm_g_a`), β's raw projection of `xn`,
//!    then `ssm_f_b` and `ssm_g_b` on the halves' outputs, in the family's
//!    order (a GEMM front may run `g_b` beside the first four);
//! 3. the conv and prep with one decay per key channel (`kda_conv_prep` at
//!    the layer's [`KdaRule`]: the conv over the ring by each row's position,
//!    SiLU, the q and k L2 norms, `beta_scale · σ(β)`, the decay), and on a
//!    rule with a write gate the values scaled by `σ(b + w) / σ(b)`
//!    ([`crate::linear::write_gate`], Raon Delta Attention);
//! 4. the delta step (`kda_delta_lanes`) over the state's lane, its stamp
//!    checked against the position;
//! 5. the gated per-head RMS norm with the sigmoid gate;
//! 6. [`KdaProj::output`]: the output projection into `out`.
//!
//! The projections are the family's: a weight's format and how a body groups
//! its quantizations (one q8_1 of `xn` for four K-quant sites, a GEMM front
//! past a batch width) are not the mixer's. Launches 1 and 3-5 are this
//! module's and the same for every family; the launch count is
//! [`CORE_LAUNCHES`] (one more with a write gate, [`KdaRule::launches`]) plus
//! what the family's projections launch.
//!
//! The mixer never clamps the state; the delta step raises its fault site
//! when the state stops being finite.

use cuda_core::{CudaStream, DeviceBuffer};

use crate::linear::conv::{KdaBetaScale, KdaConvArgs, KdaGateRule};
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs, KdaLanesArgs};
use crate::linear::norm_gate::{GdnGate, NormGateArgs};
use crate::linear::write_gate::WriteGateArgs;
use crate::linear::{LinearKernels, LinearShape};
use crate::{Gpu, GpuError};

/// The launches the mixer makes itself (1 and 3-5 of the module doc).
pub const CORE_LAUNCHES: usize = 4;

/// How a layer's decay is made, the file's fact (the description's
/// `models::KdaGate`, mapped by the body at load).
///
/// so3's conv-prep launcher takes the variant; this is the value the body
/// hands the mixer, and the mixer passes it through unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KdaDecay {
    /// `exp(lb · σ(−ssm_a · (f + dt)))`, `lb` < 0 (GLM-5.3-Flash).
    Sigmoid { lower_bound: f32 },
    /// `exp(ssm_a · softplus(f + dt))`, unbounded (Solar-Open2).
    Softplus,
}

/// A KDA layer's rule: the decay, the scale on `σ(b_proj x)` (1 for GLM,
/// 2 for Solar's `allow_neg_eigval`) and whether a write gate apart from that
/// erase gate scales the values (Raon Delta Attention: `σ(b + w) / σ(b)`, `w`
/// the family's write projection, [`KdaIn::write_raw`]). Only the pairs so3's
/// conv-prep entries instantiate are accepted, and a write gate only over the
/// erase gate `σ(b)` ([`KdaRule::check`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdaRule {
    pub decay: KdaDecay,
    pub beta_scale: f32,
    pub write_gate: bool,
}

impl KdaRule {
    /// The rule, refused by name when no conv-prep entry runs it
    /// (`(Sigmoid` with a bound below 0`, 1)` and `(Softplus, 2)` today).
    /// Load-time only.
    pub fn check(self, what: &'static str) -> Result<KdaRule, GpuError> {
        self.entry(what).map(|_| self)
    }

    /// The launches the mixer makes itself at this rule: [`CORE_LAUNCHES`],
    /// and the write gate's.
    #[must_use]
    pub fn launches(self) -> usize {
        CORE_LAUNCHES + usize::from(self.write_gate)
    }

    /// The conv-prep launch's gate rule and β scale: the one table of the
    /// pairs.
    fn entry(self, what: &'static str) -> Result<(KdaGateRule, KdaBetaScale), GpuError> {
        if self.write_gate && self.beta_scale != 1.0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "a write gate scales the values by sigmoid(b + w) / sigmoid(b), over the \
                     erase gate sigmoid(b); beta_scale {} is not 1",
                    self.beta_scale
                ),
            ));
        }
        match (self.decay, self.beta_scale) {
            (KdaDecay::Sigmoid { lower_bound }, scale) if scale == 1.0 && lower_bound < 0.0 => {
                Ok((KdaGateRule::Sigmoid { lower_bound }, KdaBetaScale::X1))
            }
            (KdaDecay::Softplus, 2.0) => Ok((KdaGateRule::Softplus, KdaBetaScale::X2)),
            (decay, scale) => Err(GpuError::shape(
                what,
                format!(
                    "no conv-prep entry runs {decay:?} with beta_scale {scale}; the entries \
                     are (Sigmoid with lower_bound < 0, 1) and (Softplus, 2)"
                ),
            )),
        }
    }
}

/// The widths and constants every launch of the mixer takes, read once.
#[derive(Clone, Copy, Debug)]
pub struct KdaDims {
    pub embd: usize,
    pub shape: LinearShape,
    pub rms_eps: f32,
    pub rule: KdaRule,
}

/// A layer's resident f32 vectors, resolved once at load by the family.
pub struct KdaVecs<'a> {
    pub norm: &'a DeviceBuffer<f32>,
    /// The q|k|v conv weights, concatenated at load.
    pub conv: &'a DeviceBuffer<f32>,
    pub dt_bias: &'a DeviceBuffer<f32>,
    /// `ssm_a` = −exp(A_log), as the file bakes it.
    pub a: &'a DeviceBuffer<f32>,
    pub gate_norm: &'a DeviceBuffer<f32>,
}

/// The layer's recurrent store and the lane the launch reads and writes: the
/// committed lane word, how many lanes the store holds, whether each row
/// writes its own lane (a verify's rows) and the row base (`KdaLanesArgs`).
pub struct KdaState<'a> {
    pub state: &'a mut DeviceBuffer<f32>,
    pub stamp: &'a mut DeviceBuffer<u32>,
    pub ring: &'a mut DeviceBuffer<f32>,
    pub lane: &'a DeviceBuffer<u32>,
    pub lanes: usize,
    pub each: bool,
    pub row: usize,
}

/// The buffers the body lends the mixer, each at least `m` rows of its
/// width: `x` the layer input in, `out` the update out, the rest
/// intermediates the mixer writes before it reads them.
pub struct KdaBufs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub xn: &'a mut DeviceBuffer<f32>,
    pub qkv: &'a mut DeviceBuffer<f32>,
    pub fa: &'a mut DeviceBuffer<f32>,
    pub ga: &'a mut DeviceBuffer<f32>,
    pub beta_raw: &'a mut DeviceBuffer<f32>,
    /// The write projection's raw rows (`[m][n_v]`), lent exactly when the
    /// rule has a write gate.
    pub write_raw: Option<&'a mut DeviceBuffer<f32>>,
    pub f: &'a mut DeviceBuffer<f32>,
    pub z: &'a mut DeviceBuffer<f32>,
    pub conv: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub o: &'a mut DeviceBuffer<f32>,
    pub gated: &'a mut DeviceBuffer<f32>,
    pub out: &'a mut DeviceBuffer<f32>,
}

/// The rows [`KdaProj::input`] writes: the four projections of `xn`, the
/// low-rank halves' intermediates `fa` and `ga`, `f`, `z` from them, and on a
/// rule with a write gate the write projection's raw rows `write_raw`.
pub struct KdaIn<'a> {
    pub qkv: &'a mut DeviceBuffer<f32>,
    pub fa: &'a mut DeviceBuffer<f32>,
    pub ga: &'a mut DeviceBuffer<f32>,
    pub beta_raw: &'a mut DeviceBuffer<f32>,
    pub write_raw: Option<&'a mut DeviceBuffer<f32>>,
    pub f: &'a mut DeviceBuffer<f32>,
    pub z: &'a mut DeviceBuffer<f32>,
}

/// A family's projections of layer `l` over `m` rows, token-major. glm5next
/// implements it over its q8_0 weights (the gemv at a step, the chunked
/// gemvs or the GEMM front over a batch); Solar over typed sites
/// (`crate::site`: Q4_K, Q6_K, Q5_0) with one quantization of `xn` for the
/// four input sites.
pub trait KdaProj {
    /// q·k·v, `f_a`, `g_a`, β's raw rows of the normed rows `xn`, then
    /// `f = f_b · fa` and `z = g_b · ga`; the order is the family's.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        y: KdaIn<'_>,
    ) -> Result<(), GpuError>;

    /// `out = o_proj · gated`.
    fn output(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError>;
}

/// Enqueue layer `l`'s KDA mixer over `m` rows on `stream` (module doc):
/// the norm, the family's input projections, the conv-prep at `d.rule`, the delta
/// step over `st`'s lane, the gated norm, the output projection. A rule no
/// entry runs is refused by name ([`KdaRule::check`] at load).
#[allow(
    clippy::too_many_arguments,
    reason = "the launch context, the family's projections and the layer's four parts, each named as the module doc names it (rust-quality R8)"
)]
pub fn enqueue(
    gpu: &Gpu,
    stream: &CudaStream,
    k: &LinearKernels,
    proj: &mut impl KdaProj,
    (l, m, d): (usize, usize, &KdaDims),
    v: &KdaVecs<'_>,
    st: KdaState<'_>,
    b: KdaBufs<'_>,
) -> Result<(), GpuError> {
    const WHAT: &str = "mixer_kda::enqueue";
    let entry = d.rule.entry(WHAT)?;
    let fault = gpu.layer_sink(l)?;
    let KdaBufs {
        x,
        pos,
        xn,
        qkv,
        fa,
        ga,
        beta_raw,
        mut write_raw,
        f,
        z,
        conv,
        beta,
        decay,
        o,
        gated,
        out,
    } = b;
    if d.rule.write_gate != write_raw.is_some() {
        return Err(GpuError::state(
            WHAT,
            "the write projection's rows exactly when the rule has a write gate",
        ));
    }
    gpu.elem()
        .enqueue_rms_norm(stream, x, v.norm, d.rms_eps, d.embd, m, xn)?;
    proj.input(
        gpu,
        l,
        m,
        xn,
        KdaIn {
            qkv: &mut *qkv,
            fa,
            ga,
            beta_raw: &mut *beta_raw,
            write_raw: write_raw.as_deref_mut(),
            f: &mut *f,
            z: &mut *z,
        },
    )?;
    let (gate, beta_scale) = entry;
    k.conv.enqueue_kda_conv_prep(
        stream,
        KdaConvArgs {
            x: qkv,
            b_raw: beta_raw,
            f,
            w: v.conv,
            dt_bias: v.dt_bias,
            ssm_a: v.a,
            pos,
            shape: d.shape,
            gate,
            beta_scale,
            eps: d.rms_eps,
            m,
            fault,
            y: &mut *conv,
            beta: &mut *beta,
            decay: &mut *decay,
            ring: &mut *st.ring,
        },
    )?;
    if let Some(w) = write_raw {
        k.write_gate.enqueue_write_gate(
            stream,
            WriteGateArgs {
                b: beta_raw,
                w,
                shape: d.shape,
                m,
                fault,
                y: &mut *conv,
            },
        )?;
    }
    k.delta.enqueue_kda_delta_lanes(
        stream,
        KdaLanesArgs {
            lanes: DeltaLanesArgs {
                delta: DeltaArgs {
                    qkv: conv,
                    beta,
                    decay,
                    lane: st.lane,
                    lane_at: 0,
                    lanes: st.lanes,
                    shape: d.shape,
                    m,
                    fault,
                    o: &mut *o,
                    state: st.state,
                },
                each: st.each,
                pos,
                stamp: st.stamp,
            },
            row: st.row,
        },
    )?;
    k.norm_gate.enqueue_norm_gate(
        stream,
        NormGateArgs {
            o,
            z,
            w: v.gate_norm,
            gate: GdnGate::Sigmoid,
            eps: d.rms_eps,
            n_v: d.shape.n_v,
            m,
            fault,
            y: &mut *gated,
        },
    )?;
    proj.output(gpu, l, m, gated, out)
}

#[cfg(test)]
mod tests {
    use super::{KdaDecay, KdaRule};

    /// The pairs a conv-prep entry runs pass `check` unchanged; every other
    /// pair, and a sigmoid bound that is not below 0, is refused by name.
    #[test]
    fn check_accepts_only_the_pairs_an_entry_runs() {
        let rule = |decay, beta_scale| KdaRule {
            decay,
            beta_scale,
            write_gate: false,
        };
        let sigmoid = |lower_bound| KdaDecay::Sigmoid { lower_bound };
        let write = |r: KdaRule| KdaRule {
            write_gate: true,
            ..r
        };
        for ok in [
            rule(sigmoid(-5.0), 1.0),
            rule(KdaDecay::Softplus, 2.0),
            write(rule(sigmoid(-5.0), 1.0)),
        ] {
            assert_eq!(ok.check("test").unwrap(), ok, "{ok:?}");
        }
        let e = write(rule(KdaDecay::Softplus, 2.0))
            .check("test")
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("test: a write gate scales"), "{e}");
        for bad in [
            rule(sigmoid(-5.0), 2.0),
            rule(KdaDecay::Softplus, 1.0),
            rule(sigmoid(0.0), 1.0),
            rule(sigmoid(5.0), 1.0),
            rule(sigmoid(f32::NAN), 1.0),
            rule(KdaDecay::Softplus, f32::NAN),
        ] {
            let e = bad.check("test").unwrap_err().to_string();
            assert!(
                e.starts_with("test: no conv-prep entry runs "),
                "{bad:?}: {e}"
            );
        }
    }
}
