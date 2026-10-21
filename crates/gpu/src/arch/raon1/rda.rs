//! A Raon Delta Attention layer on the card: the KDA mixer
//! ([`crate::linear::mixer_kda`], GLM-5.3-Flash's bounded-sigmoid decay and
//! `σ(b)` erase gate) with the one fact Raon adds, a write gate apart from
//! the erase gate (`σ(b + w) / σ(b)` on the values, [`crate::linear::write_gate`]),
//! over Raon's typed projections ([`RdaSites`]).
//!
//! What the file forces here, each read from its header at load:
//! - q, k, v and the output gate `g` are full-rank projections of the normed
//!   input (`attn_q`, `attn_k`, `attn_v`, `ssm_g`; Q6_K in the Q4_K_M file),
//!   read through one q8_1 form of it when they are K-quants; `g` writes the
//!   gate rows `z` directly (Raon has no low-rank gate);
//! - β's raw projection `ssm_beta`, the write projection `ssm_rda` and the
//!   forget projection's halves `ssm_f_a` (2560 → 128) and `ssm_f_b` (128 →
//!   heads·128), F16 in the file and widened to f32 at load;
//! - the output projection `attn_output` (Q8_0).
//!
//! [`step`] runs one row. [`rows`] runs a pass of 1 to [`PASS_ROWS`] rows in
//! position order — a verify's rows or a prompt chunk — as one launch a site
//! and one mixer a layer: each site's m-column launch (a K-quant's gemv over
//! the rows' q8_1 form, the gemv arm's m-column entry), whose column `c` is
//! the site's one-column launch on row `c` bit for bit, and the mixer at `m`
//! rows, which walks the conv and the delta step in position order. So a
//! pass leaves every output, conv ring slot and state lane a run of its rows
//! as steps leaves; at one row it is [`step`]. A K-quant gemv writes its
//! columns row-major: q, k and v land stacked in the stage and one copy lays
//! the joined rows token-major ([`RowKernels`]); every other row-major site
//! takes the stage and the copy alone ([`kgemv_rows`]).

use super::state::PASS_ROWS;
use crate::Gpu;
use crate::GpuError;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::linear::mixer_kda::{
    self, KdaBufs, KdaDecay, KdaDims, KdaIn, KdaProj, KdaRule, KdaState, KdaVecs,
};
use crate::linear::{CONV_TAPS, HEAD, KHeadMap, LinearKernels, LinearShape};
use crate::site::{self, Form, Order, SiteTy};
use crate::tensor::{Q8Act, WindowMut};
use crate::token_major::TokenMajor;
use crate::weights::Weights;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::raon1::names;
use std::sync::Arc;

/// What the refusals of this file name.
const WHAT: &str = "raon1 rda";

/// Types a full-rank projection of the normed input may have.
const DENSE: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The joined conv taps of RDA layer `l`'s q, k and v, in the conv's channel
/// order: the one weight the mixer's conv reads, made at load ([`derive`]).
fn conv_joined(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

/// Join layer `l`'s three conv tap sets into the one vector the mixer reads.
/// Load-time only; the body calls it for every RDA layer before
/// [`RdaSites::load`].
pub(crate) fn derive(stream: &CudaStream, w: &mut Weights, l: usize) -> Result<(), GpuError> {
    let parts = ['q', 'k', 'v'].map(|p| names::ssm_conv1d(l, p));
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    w.join_rows(stream, &parts, conv_joined(l))
}

/// The mixer's widths and rule for Raon's RDA layers: `heads` query/key and
/// value heads of [`HEAD`], the RMS epsilon, and the decay's lower bound
/// (`kda.gate_lower_bound`, below 0).
pub(crate) fn dims(embd: usize, heads: usize, rms_eps: f32, lower_bound: f32) -> KdaDims {
    KdaDims {
        embd,
        shape: LinearShape {
            n_k: heads,
            n_v: heads,
            map: KHeadMap::Tiled,
        },
        rms_eps,
        rule: KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound },
            beta_scale: 1.0,
            write_gate: true,
        },
    }
}

/// One RDA layer's typed projections and the names of its f32 vectors.
pub(crate) struct RdaSites {
    /// q, k and v, written into the joined rows' three windows.
    pub qkv: [Site; 3],
    /// The output gate's full-rank projection, written into `z`.
    pub g: Site,
    pub beta: Site,
    pub write: Site,
    pub f_a: Site,
    pub f_b: Site,
    pub out: Site,
    /// attn_norm, the joined conv taps, dt bias, `ssm_a`, the gate norm.
    pub vecs: [String; 5],
}

impl RdaSites {
    /// Layer `l`'s sites at `d`'s widths, each type read from `file`'s header
    /// and checked against the resident weight, and its vectors checked by
    /// length. Load-time only, after [`derive`].
    pub(crate) fn load(
        file: &Split,
        w: &Weights,
        d: &KdaDims,
        l: usize,
    ) -> Result<RdaSites, GpuError> {
        let embd = d.embd;
        let lin = d.shape;
        let (qk, v) = (lin.n_k * HEAD, lin.n_v * HEAD);
        let typed = |name: String, (rows, k): (usize, usize)| -> Result<Site, GpuError> {
            let ty = site::file_site(file, WHAT, &name, (rows, k), DENSE)?;
            site::site(w, WHAT, &name, (rows, k), ty)?;
            Ok(Site::new(name, ty, k))
        };
        let vector = |name: String, len: usize| site::vector(w, WHAT, name, len);
        let f_a = names::ssm_f_a(l);
        let low_rank = w.get(&f_a).map_or(0, |_| {
            file.find(&f_a)
                .and_then(|(_, t)| t.dims.get(1).copied())
                .and_then(|r| usize::try_from(r).ok())
                .unwrap_or(0)
        });
        if low_rank == 0 {
            return Err(GpuError::tensor(
                WHAT,
                f_a,
                "resident, with a positive rank",
            ));
        }
        Ok(RdaSites {
            qkv: [
                typed(names::attn_q(l), (qk, embd))?,
                typed(names::attn_k(l), (qk, embd))?,
                typed(names::attn_v(l), (v, embd))?,
            ],
            g: typed(names::ssm_g(l), (v, embd))?,
            beta: typed(names::ssm_beta(l), (lin.n_v, embd))?,
            write: typed(names::ssm_rda(l), (lin.n_v, embd))?,
            f_a: typed(names::ssm_f_a(l), (low_rank, embd))?,
            f_b: typed(names::ssm_f_b(l), (v, low_rank))?,
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
    pub(crate) fn sites(&self) -> [&Site; 9] {
        let [q, k, v] = &self.qkv;
        [
            q,
            k,
            v,
            &self.g,
            &self.beta,
            &self.write,
            &self.f_a,
            &self.f_b,
            &self.out,
        ]
    }

    /// The layer's resident f32 vectors, as the mixer reads them.
    pub(crate) fn vecs<'w>(&self, w: &'w Weights) -> Result<KdaVecs<'w>, GpuError> {
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

/// The kernels a pass of more than one row launches beyond the step's: the
/// copy of a row-major K-quant output into token-major rows
/// ([`TokenMajor`], the one owner of that copy).
pub(crate) struct RowKernels {
    tm: TokenMajor,
}

impl RowKernels {
    /// Load the copy's module into `ctx`. Load-time only.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<RowKernels, GpuError> {
        Ok(RowKernels {
            tm: TokenMajor::load(ctx)?,
        })
    }
}

/// What a pass of rows lends a layer beyond its buffers: the linear
/// families' kernels, the row kernels and the stage a row-major site's
/// columns land in before their copy, at least the [`stage_len`] of every
/// layer the pass runs.
pub(crate) struct RowsKit<'a> {
    pub linear: &'a LinearKernels,
    pub rows: &'a RowKernels,
    pub stage: &'a mut DeviceBuffer<f32>,
}

/// f32s of the stage an RDA layer's pass of up to [`PASS_ROWS`] rows writes:
/// the joined channels (or the model width, if wider) of every row.
#[must_use]
pub(crate) fn stage_len(d: &KdaDims) -> usize {
    PASS_ROWS * d.shape.channels().max(d.embd)
}

/// The q8_1 forms of 2 to [`PASS_ROWS`] columns a pass of rows reads, at the
/// width of each K-quant site of `sites`, beside the one-column forms
/// [`Acts::new`] makes. Load-time only.
pub(crate) fn add_row_forms<'s>(
    stream: &CudaStream,
    acts: &mut Acts,
    sites: impl Iterator<Item = &'s Site>,
) -> Result<(), GpuError> {
    for s in sites.filter(|s| s.ty().reads() == Form::Q8x128) {
        for m in 2..=PASS_ROWS {
            acts.add_q8_1(stream, s.k(), m)?;
        }
    }
    Ok(())
}

/// Where site `t`'s m-column launch lands column `c`'s outputs: a K-quant's
/// by its gemv ([`SiteTy::kgemv_order`]), token-major on the gemv arm;
/// `None` for the Q5_0 form, which no row launch here reads.
fn order(t: &Site) -> Option<Order> {
    match t.ty().reads() {
        Form::Q8x128 => t.ty().kgemv_order(),
        Form::F32 | Form::Q8x32 => Some(Order::TokenMajor),
        Form::Q8Blocks32 => None,
    }
}

/// The refusal of site `t` at a pass of `m` rows, naming `what`.
fn no_rows(what: &'static str, t: &Site, m: usize) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "{} is {}: a pass of {m} rows runs no m-column launch of it",
            t.name(),
            t.ty()
        ),
    )
}

/// K-quant site `t` over the columns of the q8_1 form `act`, its `rows`
/// outputs of each column token-major into `y`: a row-major gemv through the
/// stage and the copy, a token-major one straight in. Refusals name
/// `cx.what`, as [`Site::project`]'s do.
pub(super) fn kgemv_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    (t, rows): (&Site, usize),
    act: &Q8Act,
    l: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let m = act.m();
    match t.ty().kgemv_order() {
        Some(Order::TokenMajor) => t.kgemv(cx, act, l, y),
        Some(Order::RowMajor) => {
            {
                let mut stage = WindowMut::<f32>::of_mut(&mut *kit.stage, 0, rows * m)?;
                t.kgemv(cx, act, l, &mut stage)?;
            }
            kit.rows.tm.enqueue(cx.gpu.stream(), kit.stage, rows, m, y)
        }
        None => Err(no_rows(cx.what, t, m)),
    }
}

/// `y = W · x` for site `t` over the `m` token-major rows of `x`, its `rows`
/// outputs of each row token-major into `y`: the gemv arm's m-column launch,
/// or a K-quant's gemv over `x`'s q8_1 form of `m` columns ([`kgemv_rows`]).
/// Each column is the site's one-column launch on that row bit for bit.
pub(super) fn project_rows(
    cx: &GqaCx<'_>,
    kit: &mut RowsKit<'_>,
    acts: &mut Acts,
    (l, m): (usize, usize),
    (t, rows): (&Site, usize),
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    match t.ty().reads() {
        Form::F32 | Form::Q8x32 => site::gemv(cx.gpu, cx.g32, (t.ty(), cx.w, t.name()), x, m, y),
        Form::Q8x128 => {
            let act = acts.q8_1_at(cx.what, t.k(), m)?;
            cx.gpu.enqueue_quantize_q8_1_layer(x, act, l)?;
            kgemv_rows(cx, kit, (t, rows), act, l, y)
        }
        Form::Q8Blocks32 => Err(no_rows(cx.what, t, m)),
    }
}

/// The step's projections of one RDA layer: its sites, the activation forms
/// they read and the launch context.
struct Step<'a> {
    cx: &'a GqaCx<'a>,
    sites: &'a RdaSites,
    shape: LinearShape,
    acts: &'a mut Acts,
}

/// The window of `y` input site `i` writes (in [`RdaSites::sites`] order):
/// q, k, v in the joined rows, then `z`, β's raw rows, the write gate's raw
/// rows and the forget projection's first half.
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
        5 => whole(y.write_raw.as_deref_mut().ok_or(GpuError::state(
            WHAT,
            "the write projection's rows (the rule has a write gate)",
        ))?),
        6 => whole(&mut *y.fa),
        _ => Err(GpuError::state(WHAT, "an input site past the seventh")),
    }
}

impl KdaProj for Step<'_> {
    /// q, k, v, the gate rows `z`, β's and the write gate's raw rows and the
    /// forget projection's first half from the normed row — the K-quant sites
    /// through one q8_1 form of it, the others each through its own — then
    /// `f = f_b · fa`. `ga` is not written: Raon's gate is full-rank.
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
                format!("layer {l}: {m} rows through the step's gemvs; a pass runs `rows`"),
            ));
        }
        let (cx, s, shape) = (self.cx, self.sites, self.shape);
        let all = s.sites();
        let input = &all[..7];
        let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
        if input.iter().any(|t| q8(t)) {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, 1)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            for (i, t) in input.iter().enumerate().filter(|(_, t)| q8(t)) {
                let mut win = window(&mut y, i, shape)?;
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

/// A pass's projections of one RDA layer over 2 to [`PASS_ROWS`] rows: the
/// step's sites, forms and context, and the pass's kit.
struct Rows<'a, 'k> {
    cx: &'a GqaCx<'a>,
    sites: &'a RdaSites,
    shape: LinearShape,
    acts: &'a mut Acts,
    kit: &'a mut RowsKit<'k>,
}

impl KdaProj for Rows<'_, '_> {
    /// [`Step::input`]'s projections over the pass's `m` rows: the K-quant
    /// sites through one q8_1 form of the normed rows, q, k and v stacked
    /// row-major in the stage and laid into the joined rows by one copy, `z`,
    /// β's and the write gate's raw rows and `fa` each token-major
    /// ([`project_rows`]), then `f = f_b · fa`. A q, k or v that is not a
    /// row-major K-quant is refused by name.
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
            .find(|(t, _)| order(t) != Some(Order::RowMajor))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a pass of {m} rows stacks q, k and v as row-major K-quants; {} \
                     is {}",
                    t.name(),
                    t.ty()
                ),
            ));
        }
        let all = s.sites();
        let rest = &all[3..7];
        let rows_of = [v, shape.n_v, shape.n_v, s.f_b.k()];
        let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
        {
            let act = self.acts.q8_1_at(WHAT, cx.hidden, m)?;
            gpu.enqueue_quantize_q8_1_layer(xn, act, l)?;
            let mut at = 0;
            for (t, rows) in joined {
                let mut win = WindowMut::<f32>::of_mut(&mut *self.kit.stage, 4 * at, rows * m)?;
                t.kgemv(cx, act, l, &mut win)?;
                at += rows * m;
            }
            self.kit
                .rows
                .tm
                .enqueue(gpu.stream(), self.kit.stage, shape.channels(), m, y.qkv)?;
            for (i, t) in rest.iter().enumerate().filter(|(_, t)| q8(t)) {
                let mut win = window(&mut y, 3 + i, shape)?;
                kgemv_rows(cx, self.kit, (t, rows_of[i]), act, l, &mut win)?;
            }
        }
        for (i, t) in rest.iter().enumerate().filter(|(_, t)| !q8(t)) {
            let mut win = window(&mut y, 3 + i, shape)?;
            project_rows(
                cx,
                self.kit,
                self.acts,
                (l, m),
                (t, rows_of[i]),
                xn,
                &mut win,
            )?;
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

/// Enqueue RDA layer `l` at one row (module doc): `bufs.x` the layer's input
/// (the hyper-connections' fold), `bufs.out` the update the layer hands
/// back, the store `st` read and written in place. `d` is [`dims`]'s,
/// `bufs.write_raw` lent.
pub(crate) fn step(
    cx: &GqaCx<'_>,
    k: &LinearKernels,
    acts: &mut Acts,
    sites: &RdaSites,
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

/// Enqueue RDA layer `l` over a pass of `m` rows, 1 to [`PASS_ROWS`], in
/// position order (module doc): `bufs.x` the rows' inputs token-major,
/// `bufs.pos` their positions (`pos[t] = pos[0] + t`), `bufs.out` the rows'
/// updates, every buffer of `bufs` at least `m` rows of its width; the store
/// `st` as the pass's lanes name it (`state::Lanes::state`: in place for a
/// prompt chunk, row mode for a verify). One row is [`step`]; a count past
/// [`PASS_ROWS`] is refused by name.
pub(crate) fn rows(
    cx: &GqaCx<'_>,
    mut kit: RowsKit<'_>,
    acts: &mut Acts,
    sites: &RdaSites,
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

#[cfg(test)]
pub(super) mod tests {
    use super::{
        RdaSites, RowKernels, RowsKit, add_row_forms, conv_joined, derive, dims, rows, stage_len,
    };
    use crate::Gpu;
    use crate::arch::hybrid::{Acts, GqaCx, Site};
    use crate::arch::raon1::state::{Lanes, PASS_ROWS, Pass, rda_store};
    use crate::flash_gqa::FlashGqaKernels;
    use crate::gemm::Gemm32Kernels;
    use crate::linear::mixer_kda::{KdaBufs, KdaDims};
    use crate::linear::{CONV_TAPS, HEAD, LinearKernels};
    use crate::recur::RecStore;
    use crate::site::{KGemvKernels, SiteTy};
    use crate::weights::{TensorPart, Weights};
    use cuda_core::DeviceBuffer;
    use gguf::quant::Q8Block;
    use gguf::{GgmlType, TensorInfo};
    use model::arch::raon1::names;

    /// The layer the names carry.
    const L: usize = 2;
    /// Two Q6_K super-blocks a row: the K-quant gemv's even count.
    const EMBD: usize = 512;
    const HEADS: usize = 2;
    const V: usize = HEADS * HEAD;
    const C: usize = 3 * V;
    const RANK: usize = 128;
    /// Positions before the verify, run as one prompt chunk.
    const PRE: usize = 5;
    /// The verify's rows, and the rows its commit keeps.
    const M: usize = 7;
    const KEEP: usize = 3;

    /// A 64-bit LCG (Knuth's MMIX constants).
    pub(crate) struct Lcg(pub(crate) u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 40
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 24) as f32)
        }

        pub(crate) fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| self.uniform(lo, hi)).collect()
        }
    }

    /// A file tensor of the test's layers: its name, type, ggml dims (the
    /// contiguous axis first) and bytes.
    pub(crate) struct Part(String, GgmlType, Vec<usize>, Vec<u8>);

    /// A file tensor of `ty` over `dims` whose bytes are `bytes` as they stand.
    pub(crate) fn raw(name: String, ty: GgmlType, dims: Vec<usize>, bytes: Vec<u8>) -> Part {
        Part(name, ty, dims, bytes)
    }

    /// Q6_K rows over `dims`: random codes, sub-block scales in −4..=4 and a
    /// small `d` (high byte 0x14), so every weight is finite and small.
    pub(crate) fn q6k(r: &mut Lcg, name: String, dims: Vec<usize>) -> Part {
        let n: usize = dims.iter().product();
        let bytes = (0..n / 256 * 210)
            .map(|i| match i % 210 {
                192..=207 => ((r.next() % 9) as i8 - 4).to_le_bytes()[0],
                209 => 0x14,
                _ => r.next() as u8,
            })
            .collect();
        Part(name, GgmlType::Q6_K, dims, bytes)
    }

    /// Q8_0 rows over `dims`, every weight in ±0.05.
    pub(crate) fn q8_0(r: &mut Lcg, name: String, dims: Vec<usize>) -> Part {
        let n: usize = dims.iter().product();
        let mut bytes = Vec::with_capacity(n / 32 * 34);
        for _ in 0..n / 32 {
            let x: [f32; 32] = std::array::from_fn(|_| r.uniform(-0.05, 0.05));
            let q = Q8Block::quantize_ref(&x).expect("a finite block");
            bytes.extend(q.d.to_le_bytes());
            bytes.extend(q.q.map(|c| c.to_le_bytes()[0]));
        }
        Part(name, GgmlType::Q8_0, dims, bytes)
    }

    /// F32 values over `dims`, uniform in `lo..hi`.
    pub(crate) fn f32s(r: &mut Lcg, name: String, dims: Vec<usize>, (lo, hi): (f32, f32)) -> Part {
        let v = r.fill(dims.iter().product(), lo, hi);
        let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Part(name, GgmlType::F32, dims, bytes)
    }

    /// `parts` resident on `gpu`.
    pub(crate) fn resident(gpu: &Gpu, parts: &[Part]) -> Weights {
        let parts: Vec<TensorPart<'_>> = parts
            .iter()
            .map(|Part(name, ty, dims, bytes)| TensorPart {
                info: TensorInfo {
                    name: name.clone(),
                    dims: dims.iter().map(|&n| n as u64).collect(),
                    ty: *ty,
                    offset: 0,
                    nbytes: bytes.len() as u64,
                },
                spans: vec![bytes.as_slice()],
                scale: None,
            })
            .collect();
        let mut w = Weights::default();
        w.add_parts(gpu.stream(), &parts)
            .expect("the layer's tensors");
        w
    }

    /// Bit for bit.
    pub(crate) fn same<T: Copy + PartialEq>(a: &[T], b: &[T], bits: impl Fn(T) -> u32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| bits(x) == bits(y))
    }

    /// One RDA layer's file tensors at the test's widths, resident with its
    /// conv taps joined, and its sites.
    fn layer(gpu: &Gpu) -> (Weights, RdaSites) {
        let mut r = Lcg(0x0072_6461);
        let mut parts = vec![
            q6k(&mut r, names::attn_q(L), vec![EMBD, V]),
            q6k(&mut r, names::attn_k(L), vec![EMBD, V]),
            q6k(&mut r, names::attn_v(L), vec![EMBD, V]),
            q6k(&mut r, names::ssm_g(L), vec![EMBD, V]),
            f32s(&mut r, names::ssm_beta(L), vec![EMBD, HEADS], (-0.05, 0.05)),
            f32s(&mut r, names::ssm_rda(L), vec![EMBD, HEADS], (-0.05, 0.05)),
            f32s(&mut r, names::ssm_f_a(L), vec![EMBD, RANK], (-0.05, 0.05)),
            f32s(&mut r, names::ssm_f_b(L), vec![RANK, V], (-0.1, 0.1)),
            q8_0(&mut r, names::attn_output(L), vec![V, EMBD]),
            f32s(&mut r, names::attn_norm(L), vec![EMBD], (0.5, 1.5)),
            f32s(&mut r, names::ssm_dt_bias(L), vec![V], (-0.5, 0.5)),
            f32s(&mut r, names::ssm_a(L), vec![HEADS], (-2.0, -0.5)),
            f32s(&mut r, names::ssm_norm(L), vec![HEAD], (0.5, 1.5)),
        ];
        for p in ['q', 'k', 'v'] {
            let taps = vec![CONV_TAPS, V];
            parts.push(f32s(&mut r, names::ssm_conv1d(L, p), taps, (-0.5, 0.5)));
        }
        let mut w = resident(gpu, &parts);
        derive(gpu.stream(), &mut w, L).expect("the joined conv taps");
        let site = |name: String, ty: SiteTy, k: usize| Site::new(name, ty, k);
        let sites = RdaSites {
            qkv: [
                site(names::attn_q(L), SiteTy::Q6K, EMBD),
                site(names::attn_k(L), SiteTy::Q6K, EMBD),
                site(names::attn_v(L), SiteTy::Q6K, EMBD),
            ],
            g: site(names::ssm_g(L), SiteTy::Q6K, EMBD),
            beta: site(names::ssm_beta(L), SiteTy::F32, EMBD),
            write: site(names::ssm_rda(L), SiteTy::F32, EMBD),
            f_a: site(names::ssm_f_a(L), SiteTy::F32, EMBD),
            f_b: site(names::ssm_f_b(L), SiteTy::F32, RANK),
            out: site(names::attn_output(L), SiteTy::Q8_0, V),
            vecs: [
                names::attn_norm(L),
                conv_joined(L),
                names::ssm_dt_bias(L),
                names::ssm_a(L),
                names::ssm_norm(L),
            ],
        };
        (w, sites)
    }

    /// The buffers of a pass of up to [`PASS_ROWS`] rows, each its width
    /// times [`PASS_ROWS`]: the input, the positions, the mixer's buffers in
    /// [`KdaBufs`] order from `xn` to `out`, and the stage.
    struct Bufs {
        x: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        f: [DeviceBuffer<f32>; 14],
        stage: DeviceBuffer<f32>,
    }

    impl Bufs {
        fn new(gpu: &Gpu, d: &KdaDims) -> Bufs {
            let s = gpu.stream();
            let z = |n: usize| DeviceBuffer::<f32>::zeroed(s, PASS_ROWS * n).expect("a buffer");
            Bufs {
                x: z(EMBD),
                pos: DeviceBuffer::zeroed(s, PASS_ROWS).expect("the positions"),
                f: [
                    EMBD, C, RANK, RANK, HEADS, HEADS, V, V, C, HEADS, V, V, V, EMBD,
                ]
                .map(z),
                stage: DeviceBuffer::zeroed(s, stage_len(d)).expect("the stage"),
            }
        }
    }

    /// One RDA layer and everything a pass of it reads.
    struct Rig<'g> {
        gpu: &'g Gpu,
        w: Weights,
        sites: RdaSites,
        d: KdaDims,
        lin: LinearKernels,
        rk: RowKernels,
        kgemv: KGemvKernels,
        g32: Gemm32Kernels,
        flash: FlashGqaKernels,
        acts: Acts,
        b: Bufs,
    }

    impl Rig<'_> {
        /// The first `n` rows of `xs` from position `p` as one pass of kind
        /// `pass` on `store` under `lanes`: the rows' updates.
        fn pass(
            &mut self,
            (store, lanes): (&mut RecStore, &mut Lanes),
            (pass, p, n): (Pass, usize, usize),
            xs: &[f32],
        ) -> Vec<f32> {
            let s = self.gpu.stream();
            let at = u32::try_from(p).expect("a u32 position");
            lanes.open(pass, at, n).expect("the pass opens");
            let mut x = vec![0.0f32; PASS_ROWS * EMBD];
            x[..n * EMBD].copy_from_slice(&xs[..n * EMBD]);
            let pos: Vec<u32> = (at..).take(PASS_ROWS).collect();
            let b = &mut self.b;
            b.x.copy_from_host(s, &x).expect("the rows up");
            b.pos.copy_from_host(s, &pos).expect("the positions up");
            let cx = GqaCx {
                what: "raon1 rda test",
                gpu: self.gpu,
                w: &self.w,
                kgemv: &self.kgemv,
                g32: &self.g32,
                flash: &self.flash,
                mma: false,
                out_gate: None,
                hidden: EMBD,
                eps: self.d.rms_eps,
                n_kv: 1,
                ctx: 64,
            };
            let [
                xn,
                qkv,
                fa,
                ga,
                beta_raw,
                write_raw,
                f,
                z,
                conv,
                beta,
                decay,
                o,
                gated,
                out,
            ] = &mut b.f;
            let bufs = KdaBufs {
                x: &b.x,
                pos: &b.pos,
                xn,
                qkv,
                fa,
                ga,
                beta_raw,
                write_raw: Some(write_raw),
                f,
                z,
                conv,
                beta,
                decay,
                o,
                gated,
                out: &mut *out,
            };
            let kit = RowsKit {
                linear: &self.lin,
                rows: &self.rk,
                stage: &mut b.stage,
            };
            let st = lanes.state(store).expect("the store under its lanes");
            let lm = (L, &self.d, n);
            rows(&cx, kit, &mut self.acts, &self.sites, lm, st, bufs).expect("the pass");
            assert!(
                self.gpu.fault().expect("the fault word").is_none(),
                "a clean pass raises no fault"
            );
            out.to_host_vec(s).expect("the updates back")[..n * EMBD].to_vec()
        }
    }

    /// A pass of rows is its rows run as steps, bit for bit: a prompt chunk
    /// in place (its updates); a verify in row mode (its updates, and lane
    /// `c + j` holding the state the steps leave after row `j`); a commit of
    /// the verify's first rows, then a step at the first position it did
    /// not keep, beside steps that never ran the rows it took back (the
    /// update reads the kept lane and the conv ring the verify left).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_rda_rows_are_steps() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (w, sites) = layer(&gpu);
        let d = dims(EMBD, HEADS, 1e-5, -5.0);
        let (s, ctx) = (gpu.stream(), gpu.context());
        let mut acts = Acts::new(s, sites.sites().into_iter()).expect("the one-row forms");
        add_row_forms(s, &mut acts, sites.sites().into_iter()).expect("the row forms");
        let mut rig = Rig {
            gpu: &gpu,
            lin: LinearKernels::load(ctx).expect("the linear kernels"),
            rk: RowKernels::load(ctx).expect("the row kernels"),
            kgemv: KGemvKernels::load(&gpu).expect("the K-quant gemvs"),
            g32: Gemm32Kernels::load(ctx).expect("the F32 tile"),
            flash: FlashGqaKernels::load(ctx).expect("the flash kernels"),
            acts,
            b: Bufs::new(&gpu, &d),
            w,
            sites,
            d,
        };
        let xs = Lcg(0x0078_7300).fill((PRE + M + 1) * EMBD, -1.0, 1.0);
        let row = |t: usize| &xs[t * EMBD..];
        let store = |lanes| rda_store(s, d.shape, lanes).expect("an RDA store");
        let word = |n| Lanes::new(s, n).expect("a lane word");
        let bits = f32::to_bits;

        let (mut st_s, mut ln_s) = (store(1), word(1));
        let (mut steps, mut states) = (Vec::new(), Vec::new());
        for t in 0..PRE + M {
            steps.extend(rig.pass((&mut st_s, &mut ln_s), (Pass::InPlace, t, 1), row(t)));
            states.push(st_s.state.to_host_vec(s).expect("the state back"));
        }

        let (mut st_p, mut ln_p) = (store(PASS_ROWS), word(PASS_ROWS));
        let chunk = rig.pass((&mut st_p, &mut ln_p), (Pass::InPlace, 0, PRE), row(0));
        let pre = PRE * EMBD;
        assert!(
            same(&chunk, &steps[..pre], bits),
            "a prompt chunk is its rows as steps"
        );
        let verify = rig.pass((&mut st_p, &mut ln_p), (Pass::Verify, PRE, M), row(PRE));
        assert!(
            same(&verify, &steps[pre..], bits),
            "a verify's rows are their steps"
        );
        let held = st_p.state.to_host_vec(s).expect("the lanes back");
        let (lane, c) = (d.shape.state_len(), ln_p.committed() as usize);
        for (j, want) in states[PRE..].iter().enumerate() {
            let at = (c + j) % PASS_ROWS * lane;
            assert!(
                same(&held[at..at + lane], want, bits),
                "lane {} holds the state row {j}'s step leaves",
                (c + j) % PASS_ROWS
            );
        }

        let at = ln_p.commit(s, KEEP).expect("the commit");
        assert_eq!(
            at as usize,
            PRE + KEEP,
            "the commit stands after the kept rows"
        );
        let next = (Pass::InPlace, PRE + KEEP, 1);
        let after = rig.pass((&mut st_p, &mut ln_p), next, row(PRE + M));
        let (mut st_k, mut ln_k) = (store(1), word(1));
        for t in 0..PRE + KEEP {
            rig.pass((&mut st_k, &mut ln_k), (Pass::InPlace, t, 1), row(t));
        }
        let kept = rig.pass((&mut st_k, &mut ln_k), next, row(PRE + M));
        assert!(
            same(&after, &kept, bits),
            "after a commit the next step is the kept steps' next"
        );
    }
}
