//! A Kimi K3 delta layer on the card: the KDA mixer ([`crate::linear::mixer_kda`], the bounded
//! sigmoid decay and the `σ(b)` erase gate) over K3's typed projections ([`KdaSites`]), from
//! llama.cpp `src/models/kimi-k3.cpp` (`build_kda_layer`).
//!
//! What the file forces here, each read from its header at load:
//! - q, k, v and the output gate `g` are full-rank projections of the normed row (`attn_q`,
//!   `attn_k`, `attn_v`, `ssm_g`; Q6_K), read through one q8_1 form of it; `g` writes the gate
//!   rows `z` directly, so the mixer's `ga` rows are never written;
//! - β's raw projection `ssm_beta` (F32), the forget projection's halves `ssm_f_a` (hidden to
//!   one head width, Q6_K) and `ssm_f_b` (one head width to the inner width, Q8_0);
//! - the output projection `attn_output` (Q6_K);
//! - no write gate: the rule is the erase gate alone, so the mixer takes no write rows.
//!
//! [`step`] runs one row of the layer, [`rows`] a pass of up to [`PASS_ROWS`]: each row the
//! step's bits, the K-quant sites through one q8_1 form of the rows (q, k and v stacked
//! row-major in the pass's stage and laid into the joined rows by one copy), a site with no
//! m-column launch on each row's window. The three conv tap sets are joined in q, k, v order at
//! load ([`derive`]).

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::rows_kit::{RowsKit, collect_rows, has_columns, kgemv_rows, order, project_rows};
use crate::linear::mixer_kda::{
    self, KdaBufs, KdaDecay, KdaDims, KdaFile, KdaIn, KdaProj, KdaRule, KdaState, KdaVecs,
};
use crate::linear::{CONV_TAPS, HEAD, LinearKernels, LinearShape, PASS_ROWS};
use crate::site::{self, Form, Order, SiteTy};
use crate::tensor::WindowMut;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::kimi_k3::hparams::Hparams;
use model::arch::kimi_k3::names;
use model::arch::models::KdaGate;

/// What the refusals of this file name.
const WHAT: &str = "kimi_k3 kda";

/// The types a projection of a delta layer runs: the trunk's Q4_K, Q5_K, Q6_K and Q8_0 and the
/// F32 of the gate's beta, each a launch of the common dense site. The table's BF16
/// (`gguf::arch::kimi_k3`) is not here: the load widens a BF16 weight to f32, so no resident
/// weight is a BF16 site and the file's type is refused by name.
pub(super) const TYPES: &[SiteTy] = &[
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The joined conv taps of delta layer `l`'s q, k and v, in the conv's channel order: the one
/// weight the mixer's conv reads, made at load ([`derive`]).
fn conv_joined(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

/// Join layer `l`'s three conv tap sets into the one vector the mixer reads. Load-time only; the
/// body calls it for every delta layer before [`KdaSites::load`].
pub(super) fn derive(stream: &CudaStream, w: &mut Weights, l: usize) -> Result<(), GpuError> {
    let parts = ['q', 'k', 'v'].map(|p| names::ssm_conv1d(l, p));
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    w.join_rows(stream, &parts, conv_joined(l))
}

/// The mixer's widths and rule for K3's delta layers, from the file's header: `heads` query/key
/// and value heads of [`HEAD`], the RMS epsilon and the decay's lower bound. A softplus decay, a
/// head width or a conv the kernels have no geometry for are refused by name.
pub(super) fn dims(hp: &Hparams) -> Result<KdaDims, GpuError> {
    let KdaGate::Sigmoid { lower_bound } = hp.gate else {
        return Err(GpuError::shape(
            WHAT,
            "the delta layers' decay is the softplus form; the card runs the bounded sigmoid \
             (kda.gate_lower_bound)",
        ));
    };
    KdaDims::of_file(
        WHAT,
        KdaFile {
            embd: hp.n_embd,
            heads: hp.n_head,
            head_dim: hp.kda.head_dim,
            conv_taps: hp.kda.conv,
            rms_eps: hp.rms_eps,
        },
        KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound },
            beta_scale: 1.0,
            write_gate: false,
        },
    )
}

/// One delta layer's typed projections and the names of its f32 vectors.
pub(super) struct KdaSites {
    /// q, k and v, written into the joined rows' three windows.
    qkv: [Site; 3],
    /// The output gate's full-rank projection, written into `z`.
    g: Site,
    beta: Site,
    f_a: Site,
    f_b: Site,
    out: Site,
    /// attn_norm, the joined conv taps, dt bias, `ssm_a`, the gate norm.
    vecs: [String; 5],
}

impl KdaSites {
    /// Layer `l`'s sites at `d`'s widths, each type read from `file`'s header and checked against
    /// the resident weight, and its vectors checked by length. Load-time only, after [`derive`].
    pub(super) fn load(
        file: &Split,
        w: &Weights,
        d: &KdaDims,
        l: usize,
    ) -> Result<KdaSites, GpuError> {
        let embd = d.embd;
        let lin = d.shape;
        let (qk, v) = (lin.n_k * HEAD, lin.n_v * HEAD);
        let typed = |name: String, (rows, k): (usize, usize)| {
            Site::of_file((file, w), WHAT, name, (rows, k), TYPES)
        };
        let vector = |name: String, len: usize| site::vector(w, WHAT, name, len);
        Ok(KdaSites {
            qkv: [
                typed(names::attn_q(l), (qk, embd))?,
                typed(names::attn_k(l), (qk, embd))?,
                typed(names::attn_v(l), (v, embd))?,
            ],
            g: typed(names::ssm_g(l), (v, embd))?,
            beta: typed(names::ssm_beta(l), (lin.n_v, embd))?,
            f_a: typed(names::ssm_f_a(l), (HEAD, embd))?,
            f_b: typed(names::ssm_f_b(l), (v, HEAD))?,
            out: typed(names::attn_output(l), (embd, v))?,
            vecs: [
                vector(names::attn_norm(l), embd)?,
                vector(conv_joined(l), lin.channels() * CONV_TAPS)?,
                vector(names::ssm_dt_bias(l), v)?,
                vector(names::ssm_a(l), lin.n_v)?,
                vector(names::ssm_norm(l), HEAD)?,
            ],
        })
    }

    /// Every site the layer launches, for the load's [`Acts::new`].
    pub(super) fn sites(&self) -> [&Site; 8] {
        let [q, k, v] = &self.qkv;
        [
            q, k, v, &self.g, &self.beta, &self.f_a, &self.f_b, &self.out,
        ]
    }

    /// The layer's resident f32 vectors, as the mixer reads them.
    fn vecs<'w>(&self, w: &'w Weights) -> Result<KdaVecs<'w>, GpuError> {
        let [norm, conv, dt_bias, a, gate_norm] = &self.vecs;
        Ok(KdaVecs {
            norm: w.f32_buf(WHAT, norm)?,
            conv: w.f32_buf(WHAT, conv)?,
            dt_bias: w.f32_buf(WHAT, dt_bias)?,
            a: w.f32_buf(WHAT, a)?,
            gate_norm: w.f32_buf(WHAT, gate_norm)?,
        })
    }
}

/// The step's projections of one delta layer: its sites, the activation forms they read and the
/// launch context.
struct Step<'a> {
    cx: &'a GqaCx<'a>,
    sites: &'a KdaSites,
    shape: LinearShape,
    acts: &'a mut Acts,
}

/// The window of `y` input site `i` writes (in [`KdaSites::sites`] order): q, k, v in the joined
/// rows, then `z`, β's raw rows and the forget projection's first half.
fn window<'b>(
    y: &'b mut KdaIn<'_>,
    i: usize,
    shape: LinearShape,
) -> Result<WindowMut<'b, f32>, GpuError> {
    let (qk, v) = (shape.n_k * HEAD, shape.n_v * HEAD);
    let whole = |b: &'b mut DeviceBuffer<f32>| {
        let len = b.len();
        WindowMut::of_mut(b, 0, len)
    };
    match i {
        0 => WindowMut::of_mut(&mut *y.qkv, 0, qk),
        1 => WindowMut::of_mut(&mut *y.qkv, 4 * qk, qk),
        2 => WindowMut::of_mut(&mut *y.qkv, 8 * qk, v),
        3 => whole(&mut *y.z),
        4 => whole(&mut *y.beta_raw),
        5 => whole(&mut *y.fa),
        _ => Err(GpuError::state(WHAT, "an input site past the sixth")),
    }
}

impl KdaProj for Step<'_> {
    /// q, k, v, the gate rows `z`, β's raw rows and the forget projection's first half from the
    /// normed row, the K-quant sites through one q8_1 form of it and the others each through its
    /// own, then `f = f_b · fa`. `ga` is not written: K3's gate is full-rank.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        mut y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        if m != 1 {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l}: {m} rows through the step's gemvs"),
            ));
        }
        let (cx, s, shape) = (self.cx, self.sites, self.shape);
        let all = s.sites();
        let input = &all[..6];
        let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
        if input.iter().any(|t| q8(t)) {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, 1)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            for (i, t) in input.iter().enumerate().filter(|(_, t)| q8(t)) {
                let mut win = window(&mut y, i, shape)?;
                collect_rows(cx, t, xn, 1)?;
                t.kgemv(cx, act, l, &mut win)?;
            }
        }
        for (i, t) in input.iter().enumerate().filter(|(_, t)| !q8(t)) {
            let mut win = window(&mut y, i, shape)?;
            t.project(cx, self.acts, l, xn, &mut win)?;
        }
        s.f_b.project(cx, self.acts, l, y.fa, y.f)
    }

    /// `out = o_proj · gated`: the site quantizes `gated` into its own form.
    fn output(
        &mut self,
        _gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if m != 1 {
            return Err(GpuError::shape(WHAT, format!("layer {l}: {m} rows")));
        }
        self.sites.out.project(self.cx, self.acts, l, gated, out)
    }
}

/// Enqueue delta layer `l` at one row: `bufs.x` the layer's input (the mixed row, before the
/// layer's norm), `bufs.out` the update the layer hands back, the store `st` read and written in
/// place. `d` is [`dims`]'s.
pub(super) fn step(
    cx: &GqaCx<'_>,
    k: &LinearKernels,
    acts: &mut Acts,
    sites: &KdaSites,
    (l, d): (usize, &KdaDims),
    st: KdaState<'_>,
    bufs: KdaBufs<'_>,
) -> Result<(), GpuError> {
    let gpu = cx.gpu;
    let vecs = sites.vecs(cx.w)?;
    let mut proj = Step {
        cx,
        sites,
        shape: d.shape,
        acts,
    };
    mixer_kda::enqueue(gpu, gpu.stream(), k, &mut proj, (l, 1, d), &vecs, st, bufs)
}

/// A pass's projections of one delta layer over 2 to [`PASS_ROWS`] rows: the step's sites, forms
/// and context, and the pass's kit.
struct Rows<'a, 'k> {
    cx: &'a GqaCx<'a>,
    sites: &'a KdaSites,
    shape: LinearShape,
    acts: &'a mut Acts,
    kit: &'a mut RowsKit<'k>,
}

impl KdaProj for Rows<'_, '_> {
    /// [`Step::input`]'s projections over the pass's `m` rows: the K-quant sites through one
    /// q8_1 form of the normed rows, q, k and v stacked row-major in the stage and laid into the
    /// joined rows by one copy, `z`, β's raw rows and `fa` each token-major ([`project_rows`]),
    /// then `f = f_b · fa`. A q, k or v that is not an m-column row-major K-quant is refused by
    /// name.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        mut y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        let (cx, s, shape) = (self.cx, self.sites, self.shape);
        let (qk, v) = (shape.n_k * HEAD, shape.n_v * HEAD);
        let joined = [(&s.qkv[0], qk), (&s.qkv[1], qk), (&s.qkv[2], v)];
        if let Some((t, _)) = joined
            .iter()
            .find(|(t, _)| order(t) != Some(Order::RowMajor) || !has_columns(t))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a pass of {m} rows stacks q, k and v as row-major K-quants with an \
                     m-column launch; {} is {} at {} values a row",
                    t.name(),
                    t.ty(),
                    t.k()
                ),
            ));
        }
        let all = s.sites();
        let rest = &all[3..6];
        let rows_of = [v, shape.n_v, HEAD];
        let shared = |t: &Site| t.ty().reads() == Form::Q8x128 && has_columns(t);
        {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, m)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            let mut at = 0;
            for (t, rows) in joined {
                collect_rows(cx, t, xn, m)?;
                let mut win = WindowMut::<f32>::of_mut(&mut *self.kit.stage, 4 * at, rows * m)?;
                t.kgemv(cx, act, l, &mut win)?;
                at += rows * m;
            }
            self.kit
                .rows
                .tm
                .enqueue(gpu.stream(), self.kit.stage, shape.channels(), m, y.qkv)?;
            for (i, t) in rest.iter().enumerate().filter(|(_, t)| shared(t)) {
                let mut win = window(&mut y, 3 + i, shape)?;
                collect_rows(cx, t, xn, m)?;
                kgemv_rows(cx, self.kit, (t, rows_of[i]), act, l, &mut win)?;
            }
        }
        for (i, t) in rest.iter().enumerate().filter(|(_, t)| !shared(t)) {
            let mut win = window(&mut y, 3 + i, shape)?;
            let at = (l, m);
            project_rows(cx, self.kit, self.acts, at, (t, rows_of[i]), xn, &mut win)?;
        }
        project_rows(cx, self.kit, self.acts, (l, m), (&s.f_b, v), y.fa, y.f)
    }

    /// `out = o_proj · gated` over the pass's rows ([`project_rows`]).
    fn output(
        &mut self,
        _gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let t = (&self.sites.out, self.cx.hidden);
        project_rows(self.cx, self.kit, self.acts, (l, m), t, gated, out)
    }
}

/// f32s of the stage a delta layer's pass of up to [`PASS_ROWS`] rows writes: the joined
/// channels (or the model width, if wider) of every row.
#[must_use]
pub(super) fn stage_len(d: &KdaDims) -> usize {
    PASS_ROWS * d.shape.channels().max(d.embd)
}

/// Enqueue delta layer `l` over a pass of `m` rows, 1 to [`PASS_ROWS`], in position order:
/// `bufs.x` the rows' inputs token-major, `bufs.pos` their positions (`pos[t] = pos[0] + t`),
/// `bufs.out` the rows' updates, every buffer of `bufs` at least `m` rows of its width; the store
/// `st` is read and written in place, each row the step's bits. One row is [`step`]; a count past
/// [`PASS_ROWS`] is refused by name. `d` is [`dims`]'s.
pub(super) fn rows(
    cx: &GqaCx<'_>,
    mut kit: RowsKit<'_>,
    acts: &mut Acts,
    sites: &KdaSites,
    (l, d, m): (usize, &KdaDims, usize),
    st: KdaState<'_>,
    bufs: KdaBufs<'_>,
) -> Result<(), GpuError> {
    let k = kit.linear;
    if m == 1 {
        return step(cx, k, acts, sites, (l, d), st, bufs);
    }
    if !(2..=PASS_ROWS).contains(&m) {
        return Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a pass of {m} rows; a pass runs 1 to {PASS_ROWS}"),
        ));
    }
    let gpu = cx.gpu;
    let vecs = sites.vecs(cx.w)?;
    let mut proj = Rows {
        cx,
        sites,
        shape: d.shape,
        acts,
        kit: &mut kit,
    };
    mixer_kda::enqueue(gpu, gpu.stream(), k, &mut proj, (l, m, d), &vecs, st, bufs)
}
