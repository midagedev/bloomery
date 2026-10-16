//! The prompt path: a prompt runs layer-major over passes of up to
//! [`PASS_ROWS`] ids, each launch over the pass's rows.
//!
//! - a projection is its site's gemv over column runs of at most
//!   [`COLS`] rows ([`project_rows`]): Q8_0 `q8_0_gemv_mcol` on the f32 rows,
//!   NVFP4 `nvfp4_gemv` on the run's q8_1 columns, each weight read once a
//!   run; past [`COLS`] rows a Q8_0 site is one 32-value GEMM over the pass
//!   and an NVFP4 site the same GEMM over its resident NVFP4 planes
//!   ([`Gemm32Weight::Nvfp4Planes`]);
//! - a Mamba-2 layer is one pass over its tokens
//!   ([`Mamba2Kernels::enqueue_steps`], bit for bit the steps' scan);
//! - an attention layer appends the pass's K/V rows to its planes and runs
//!   the group-16 flash in runs of [`MAX_PASS_ROWS`] rows, each row over the
//!   keys up to its own position;
//! - a latent MoE layer routes the pass's rows in two launches
//!   ([`super::router::Routers::enqueue_rows`]), runs its kept experts as
//!   one `iq2_xxs_gemv_sel` over the pass's `TOP_K · m` slots each for the
//!   up and the down, sums each row's slots by its weights, and runs the
//!   shared expert as a one-expert `_sel` over the pass's `m` slots — the
//!   step's launches at `m` rows, in the step's order.
//!
//! The head reads the last row of the prompt's last pass only.
//!
//! After [`GpuModel::prefill_with`] the stores (the scan states, the conv
//! rings and the K/V planes) and the position are those of the prompt's ids
//! run as steps; the next step continues from them.
//!
//! The pass's rows live in [`Prompt`], cut at load for the most rows a pass
//! takes, so a pass allocates nothing. A run's multi-column gemv sums each
//! output as the one-column gemv of its family does, so how far a pass's
//! rows agree with the steps' is the multi-column launches' own contract.

use super::body::{Body, Layer};
use super::moe::{MoeLayer, MoeShape};
use super::router::Routers;
use super::router::{COUNTS, TOP_K};
use crate::arch::hybrid::{RowBuffers, Shape, Site, norm_rows, residual_rows};
use crate::fault::FaultSink;
use crate::flash::PlaneAppend;
use crate::flash_gqa::{GROUP_16, GqaArgs, HEAD};
use crate::gemm::Gemm32Kernels;
use crate::gemm::{Gemm32Args, Gemm32Weight, GemmAct, GemmArgs, GemmInput, GemmRoute, GemmWeight};
use crate::head::Head;
use crate::iq_sel::IqSelKernels;
use crate::kquant::SelDown;
use crate::model::{GpuModel, MAX_PASS_ROWS};
use crate::q38::EmbedQ8Args;
use crate::site::{self, SiteTy};
use crate::site::{KGemvKernels, SiteWeight, WideActs, WideKernels};
use crate::tensor::{Pitched, WindowMut, dtod_2d};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act, Window, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;

const WHAT: &str = "nemotron_h_moe::prefill";

/// The most ids one pass of [`GpuModel::prefill`] takes: the most whose
/// `TOP_K` slots a q8_1 slot form holds.
pub const PASS_ROWS: usize = 448;
const _: () = assert!(TOP_K * PASS_ROWS <= crate::tensor::Q8ACT_MAX_SLOTS);

/// The columns one gemv run takes: the multi-column gemvs' most.
const COLS: usize = 8;

/// The pass's word record: the first position, then one row word a key head
/// (head `h` at `P_ROWS + h`, the plane row `h · ctx + pos` the pass's first
/// K/V row lands at), then the pass's ids from [`Prompt::ids_at`].
const P_POS: usize = 0;
const P_ROWS: usize = 1;

/// The buffers one pass runs over: `rows` rows each, every row buffer cut at
/// `rows` rounded up to [`COLS`], so a run of [`COLS`] columns past the
/// pass's last row writes inside its buffer.
pub(super) struct Prompt {
    rows: usize,
    /// The query heads' width a row, `n_head · HEAD`.
    q_width: usize,
    /// The pass's two residual streams, each layer reading one and writing
    /// the other.
    x: [DeviceBuffer<f32>; 2],
    /// Every other buffer of the pass.
    b: PassBufs,
}

/// [`Prompt`]'s buffers beside the residual streams.
struct PassBufs {
    /// The trunk's row buffers.
    buf: RowBuffers,
    /// The pass's K and V rows regrouped head-major: head `h`'s rows at
    /// `h · rows · HEAD`, what the plane append reads.
    kh: DeviceBuffer<f32>,
    vh: DeviceBuffer<f32>,
    /// The word record ([`P_POS`], [`P_ROWS`], the ids) and its host copy.
    words: DeviceBuffer<u32>,
    host_words: Vec<u32>,
    /// The router's outputs: every expert's score (the widest layer's count
    /// a row), then `TOP_K` ids and weights a row.
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The latent down's rows; the slots' up rows and their squared ReLU
    /// (the widest expert); the slots' down rows; the rows' weighted sums.
    lat: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    /// The shared expert's up rows, their squared ReLU and its down rows,
    /// then `moe_out + shared`.
    sh_up: DeviceBuffer<f32>,
    sh_act: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    sum: DeviceBuffer<f32>,
    /// One id 0 a row: the shared expert's slots.
    zero_ids: DeviceBuffer<u32>,
    /// The q8_1 slot forms the `_sel` launches read, one a width:
    /// `(k, columns, form)`.
    slots: Vec<(usize, usize, Q8Act)>,
    /// The projections' forms ([`project_rows`]).
    arms: Arms,
    /// The experts' GEMM forms ([`experts_gemm`]).
    ex: ExpertArms,
}

/// What the experts' GEMMs read past [`COLS`] rows.
struct ExpertArms {
    /// One routed table a layer expert count, `TOP_K` slots a row.
    routes: Vec<(usize, GemmRoute)>,
    /// The q8_1 blocks of 128 the GEMMs read: `(k, columns, form)`.
    acts: Vec<(usize, usize, GemmAct)>,
}

/// What a projection over the pass's rows reads beside its f32 rows.
struct Arms {
    /// The q8_1 forms of one run of [`COLS`] columns the NVFP4 gemv reads,
    /// one a width.
    runs: Vec<Q8Act>,
    /// The q8 blocks of 32 a Q8_0 GEMM reads, the pass's rows, one a width.
    wide: Vec<(usize, WideActs)>,
    /// The pass's one-expert table: slot `s` reads column `s`, filled once a
    /// pass.
    route: GemmRoute,
}

impl Prompt {
    /// The buffers for passes of up to `rows` rows over `sh`, `n_head` query
    /// heads, the MoE shape `moe` and `layers`' widths. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        (sh, moe, n_head): (&Shape, MoeShape, usize),
        layers: &[Layer],
    ) -> Result<Prompt, GpuError> {
        if rows == 0 || rows > PASS_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {rows} rows; a pass holds 1..={PASS_ROWS}"),
            ));
        }
        let cut = rows.next_multiple_of(COLS);
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, cut * n.max(1))?)
        };
        let kv_width = sh.n_kv * HEAD;
        let moes = || {
            layers.iter().filter_map(|l| match l {
                Layer::Moe(m) => Some(m),
                _ => None,
            })
        };
        let ff = moes().map(|m| m.ff).max().unwrap_or(0);
        let n_expert = COUNTS.iter().copied().max().unwrap_or(0);
        let mut slots: Vec<(usize, usize, Q8Act)> = Vec::new();
        let mut want = |k: usize, cols: usize| -> Result<(), GpuError> {
            if !slots.iter().any(|s| s.0 == k && s.1 == cols) {
                slots.push((k, cols, Q8Act::with_slots(stream, cols, k)?));
            }
            Ok(())
        };
        for m in moes() {
            want(moe.latent, rows)?;
            want(m.ff, TOP_K * rows)?;
            want(sh.hidden, rows)?;
            want(moe.shared_ff, rows)?;
        }
        let mut ex = ExpertArms {
            routes: Vec::new(),
            acts: Vec::new(),
        };
        for m in moes() {
            let n = m.route.n_expert();
            if !ex.routes.iter().any(|r| r.0 == n) {
                ex.routes.push((n, GemmRoute::new(stream, TOP_K * cut, n)?));
            }
            for (k, cols) in [
                (moe.latent, cut),
                (m.ff, TOP_K * cut),
                (sh.hidden, cut),
                (moe.shared_ff, cut),
            ] {
                if !ex.acts.iter().any(|a| a.0 == k && a.1 == cols) {
                    ex.acts.push((k, cols, GemmAct::new(stream, cols, k)?));
                }
            }
        }
        let mut runs: Vec<Q8Act> = Vec::new();
        let mut wide: Vec<(usize, WideActs)> = Vec::new();
        for s in layers.iter().flat_map(Layer::sites) {
            if s.ty == SiteTy::Nvfp4 && !runs.iter().any(|a| a.k() == s.k) {
                runs.push(Q8Act::with_k(stream, COLS, s.k)?);
            }
            if matches!(s.ty, SiteTy::Q8_0 | SiteTy::Nvfp4) && !wide.iter().any(|a| a.0 == s.k) {
                wide.push((s.k, WideActs::new(stream, (false, true), cut, s.k)?));
            }
        }
        let n_words = P_ROWS + sh.n_kv + rows;
        Ok(Prompt {
            rows,
            q_width: n_head * HEAD,
            x: [z(sh.hidden)?, z(sh.hidden)?],
            b: PassBufs {
                buf: RowBuffers::new(stream, cut, sh, (n_head, 0))?,
                kh: z(kv_width)?,
                vh: z(kv_width)?,
                words: DeviceBuffer::zeroed(stream, n_words)?,
                host_words: vec![0; n_words],
                probs: z(n_expert)?,
                ids: DeviceBuffer::zeroed(stream, cut * TOP_K)?,
                weights: z(TOP_K)?,
                lat: z(moe.latent)?,
                up: z(TOP_K * ff)?,
                act: z(TOP_K * ff)?,
                down: z(TOP_K * moe.latent)?,
                moe: z(moe.latent)?,
                sh_up: z(moe.shared_ff)?,
                sh_act: z(moe.shared_ff)?,
                sh: z(sh.hidden)?,
                sum: z(sh.hidden)?,
                zero_ids: DeviceBuffer::zeroed(stream, cut)?,
                slots,
                ex,
                arms: Arms {
                    runs,
                    wide,
                    route: GemmRoute::new(stream, cut, 1)?,
                },
            },
        })
    }

    /// The most ids a pass takes.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// The device bytes the pass's buffers hold.
    pub(super) fn bytes(&self) -> usize {
        let b = &self.b;
        let f = [
            &self.x[0], &self.x[1], &b.kh, &b.vh, &b.probs, &b.weights, &b.lat, &b.up, &b.act,
            &b.down, &b.moe, &b.sh_up, &b.sh_act, &b.sh, &b.sum,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        f + b.buf.bytes()
            + b.words.num_bytes()
            + b.ids.num_bytes()
            + b.zero_ids.num_bytes()
            + b.slots.iter().map(|s| s.2.device_bytes()).sum::<usize>()
            + b.arms.runs.iter().map(Q8Act::device_bytes).sum::<usize>()
            + b.arms.wide.iter().map(|a| a.1.bytes()).sum::<usize>()
            + b.arms.route.bytes()
            + b.ex.routes.iter().map(|r| r.1.bytes()).sum::<usize>()
            + b.ex.acts.iter().map(|a| a.2.bytes()).sum::<usize>()
    }

    /// The word index the pass's ids start at.
    fn ids_at(n_kv: usize) -> usize {
        P_ROWS + n_kv
    }
}

/// The q8_1 slot form of `cols` columns of `k` values in `slots`.
fn slot_form(
    slots: &mut [(usize, usize, Q8Act)],
    k: usize,
    cols: usize,
) -> Result<&mut Q8Act, GpuError> {
    slots
        .iter_mut()
        .find(|s| s.0 == k && s.1 >= cols)
        .map(|s| &mut s.2)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no q8_1 slot form of {cols} columns of {k}")))
}

/// What every launch of a pass reads.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    wk: WideKernels<'a>,
    g32: &'a Gemm32Kernels,
    kgemv: &'a KGemvKernels,
    routers: &'a Routers,
    iq: &'a IqSelKernels,
    m: usize,
}

/// `y = W · x` for site `s` over the pass's `m` rows of `x` (`s.k` values a
/// row), `rows` output values a row, token-major. Up to [`COLS`] rows, and
/// for an NVFP4 site at any count, runs of up to [`COLS`] columns, each its
/// family's multi-column gemv (the step's sums); a Q8_0 site past [`COLS`]
/// rows is one 32-value GEMM over the pass's table, on the rows' q8 blocks of
/// 32. A type the pass does not run is refused by name.
fn project_rows(
    cx: &Cx<'_>,
    arms: &mut Arms,
    layer: usize,
    (s, rows): (&Site, usize),
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (gpu, f) = (cx.gpu, size_of::<f32>());
    let key = (s.ty, cx.w, s.name.as_str());
    if s.ty == SiteTy::Q8_0 && cx.m > COLS {
        let a =
            arms.wide.iter_mut().find(|a| a.0 == s.k).ok_or_else(|| {
                GpuError::shape(WHAT, format!("no q8 blocks of 32 of width {}", s.k))
            })?;
        a.1.quantize(gpu, &cx.wk, gpu.layer_sink(layer)?, x, cx.m, &[s.ty])?;
        return site::gemm(
            gpu,
            &cx.wk,
            (s.ty, cx.w, SiteWeight::Resident(&s.name)),
            rows,
            &a.1.input(x),
            (&arms.route, GemmInput::PerSlot),
            cx.m,
            y,
        );
    }
    if s.ty == SiteTy::Nvfp4 && cx.m > COLS {
        let a =
            arms.wide.iter_mut().find(|a| a.0 == s.k).ok_or_else(|| {
                GpuError::shape(WHAT, format!("no q8 blocks of 32 of width {}", s.k))
            })?;
        a.1.quantize(
            gpu,
            &cx.wk,
            gpu.layer_sink(layer)?,
            x,
            cx.m,
            &[SiteTy::Q8_0],
        )?;
        let (qs, e, scale) = match cx.w.resident(WHAT, &s.name)? {
            DevWeight::Nvfp4 { qs, e, scale, .. } => (qs, e, *scale),
            _ => return Err(GpuError::tensor(WHAT, &s.name, "an NVFP4 weight")),
        };
        return cx.g32.enqueue_gemm32(
            gpu.stream(),
            Gemm32Args {
                w: Gemm32Weight::Nvfp4Planes { qs, e, scale },
                rows_per_expert: rows,
                act: a
                    .1
                    .input(x)
                    .q32
                    .ok_or(GpuError::state(WHAT, "the q8 blocks of 32"))?,
                route: &arms.route,
                input: GemmInput::PerSlot,
                y,
            },
        );
    }
    let mut c0 = 0;
    while c0 < cx.m {
        let c = (cx.m - c0).min(COLS);
        let xw = Window::<f32>::of(x, c0 * s.k * f, c * s.k)?;
        match s.ty {
            SiteTy::Q8_0 => {
                let mut yw = WindowMut::<f32>::of_mut(y, c0 * rows * f, c * rows)?;
                site::gemv(gpu, cx.g32, key, &xw, c, &mut yw)?;
            }
            SiteTy::Nvfp4 => {
                let act = arms.runs.iter_mut().find(|a| a.k() == s.k).ok_or_else(|| {
                    GpuError::shape(WHAT, format!("no q8_1 run form of width {}", s.k))
                })?;
                gpu.quantize_q8_1_cols_at(x, c0 * s.k, act, c, gpu.layer_sink(layer)?)?;
                // The gemv runs the form's COLS columns: the run's output
                // window is COLS rows, inside the buffer's cut.
                let mut yw = WindowMut::<f32>::of_mut(y, c0 * rows * f, COLS * rows)?;
                site::kgemv(gpu, cx.kgemv, key, act, gpu.layer_sink(layer)?, &mut yw)?;
            }
            other => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{} is {other}: the pass runs Q8_0 and NVFP4 sites", s.name),
                ));
            }
        }
        c0 += c;
    }
    Ok(())
}

/// The resident stack `name` a `_sel` launch reads.
fn stack<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(WHAT, name)? {
        DevWeight::KQuant { w, .. } => Ok(w),
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a weight in the file's blocks",
        )),
    }
}

/// One latent MoE layer over the pass's `m` rows of `xn` (the layer's normed
/// rows), the layer's delta into `p.sum`: the step's order (`moe.rs`'s module
/// doc) at `m` rows.
fn moe_rows(
    cx: &Cx<'_>,
    p: &mut PassBufs,
    (l, ml, sh): (usize, &mut MoeLayer, MoeShape),
    hidden: usize,
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let iq = cx.iq;
    cx.routers.enqueue_rows(
        stream,
        ml.route.n_expert(),
        w.f32_tensor(WHAT, &ml.router)?,
        &p.buf.xn,
        w.f32_buf(WHAT, &ml.bias)?,
        sh.scale,
        m,
        (&mut p.probs, &mut p.ids, &mut p.weights),
        fault,
    )?;
    ml.route
        .enqueue_row_from(stream, (&p.probs, &p.ids, &p.weights), m - 1)?;
    project_rows(
        cx,
        &mut p.arms,
        l,
        (&ml.latent_down, sh.latent),
        &p.buf.xn,
        &mut p.lat,
    )?;
    let slots = TOP_K * m;
    // The grouped GEMM's IQ2_XXS entry, where the build has one.
    let gemm = GemmWeight::from_ggml(GgmlType::IQ2_XXS)
        .ok()
        .filter(|_| m > COLS);
    if let Some(ty) = gemm {
        experts_gemm(
            cx,
            &mut p.ex,
            (fault, ml, sh.latent, ty),
            &p.ids,
            (&p.lat, &mut p.up, &mut p.act, &mut p.down),
        )?;
    } else {
        {
            let act = slot_form(&mut p.slots, sh.latent, m)?;
            gpu.quantize_q8_1_cols_at(&p.lat, 0, act, m, fault)?;
            let up = SelDown {
                w: stack(w, &ml.up_exps)?,
                act,
                sel: &p.ids,
                n_slots: slots,
                rows_per_expert: ml.ff,
            };
            iq.enqueue_gemv_sel_cols(stream, GgmlType::IQ2_XXS, &up, TOP_K, fault, &mut p.up)?;
        }
        gpu.elem()
            .enqueue_relu_sqr(stream, &p.up, slots * ml.ff, &mut p.act)?;
        {
            let act = slot_form(&mut p.slots, ml.ff, slots)?;
            gpu.quantize_q8_1_cols_at(&p.act, 0, act, slots, fault)?;
            let down = SelDown {
                w: stack(w, &ml.down_exps)?,
                act,
                sel: &p.ids,
                n_slots: slots,
                rows_per_expert: sh.latent,
            };
            iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut p.down)?;
        }
    }
    gpu.elem().enqueue_weighted_sum(
        stream,
        &p.down,
        &p.weights,
        sh.latent,
        TOP_K as u32,
        m,
        &mut p.moe,
    )?;
    project_rows(
        cx,
        &mut p.arms,
        l,
        (&ml.latent_up, hidden),
        &p.moe,
        &mut p.buf.delta,
    )?;
    if let Some(ty) = gemm {
        shared_gemm(
            cx,
            (&mut p.ex, &p.arms.route),
            (fault, ml, sh.shared_ff, hidden, ty),
            &p.buf.xn,
            (&mut p.sh_up, &mut p.sh_act, &mut p.sh),
        )?;
    } else {
        {
            let act = slot_form(&mut p.slots, hidden, m)?;
            gpu.quantize_q8_1_cols_at(&p.buf.xn, 0, act, m, fault)?;
            let up = SelDown {
                w: stack(w, &ml.up_shexp)?,
                act,
                sel: &p.zero_ids,
                n_slots: m,
                rows_per_expert: sh.shared_ff,
            };
            iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &up, fault, &mut p.sh_up)?;
        }
        gpu.elem()
            .enqueue_relu_sqr(stream, &p.sh_up, m * sh.shared_ff, &mut p.sh_act)?;
        {
            let act = slot_form(&mut p.slots, sh.shared_ff, m)?;
            gpu.quantize_q8_1_cols_at(&p.sh_act, 0, act, m, fault)?;
            let down = SelDown {
                w: stack(w, &ml.down_shexp)?,
                act,
                sel: &p.zero_ids,
                n_slots: m,
                rows_per_expert: hidden,
            };
            iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut p.sh)?;
        }
    }
    gpu.elem()
        .enqueue_add(stream, &p.buf.delta, &p.sh, m * hidden, &mut p.sum)
}

/// The GEMM form of `cols` columns of `k` values in `acts`.
fn gemm_form(
    acts: &mut [(usize, usize, GemmAct)],
    k: usize,
    cols: usize,
) -> Result<&mut GemmAct, GpuError> {
    acts.iter_mut()
        .find(|a| a.0 == k && a.1 >= cols)
        .map(|a| &mut a.2)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no GEMM form of {cols} columns of {k}")))
}

/// The routed experts over the pass's `TOP_K · m` slots past [`COLS`] rows:
/// the slots' table from `ids`, the up as one grouped IQ2_XXS GEMM on the
/// latent rows `lat` (each row's `TOP_K` slots reading its column), the
/// squared ReLU, the down as one GEMM on each slot's own row.
fn experts_gemm(
    cx: &Cx<'_>,
    ex: &mut ExpertArms,
    (fault, ml, latent, ty): (FaultSink, &MoeLayer, usize, GemmWeight),
    ids: &DeviceBuffer<u32>,
    (lat, up, act, down): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let slots = TOP_K * m;
    let n = ml.route.n_expert();
    let route = ex
        .routes
        .iter_mut()
        .find(|r| r.0 == n)
        .map(|r| &mut r.1)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no routed table of {n} experts")))?;
    cx.wk.gemm.enqueue_route(stream, ids, slots, route, fault)?;
    let route = &*route;
    {
        let a = gemm_form(&mut ex.acts, latent, m)?;
        gpu.enqueue_quantize_gemm(lat, m, a, fault)?;
        cx.wk.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w: stack(w, &ml.up_exps)?,
                rows_per_expert: ml.ff,
                act: a,
                route,
                input: GemmInput::Shared { top_k: TOP_K },
                y: up,
            },
        )?;
    }
    gpu.elem()
        .enqueue_relu_sqr(stream, up, slots * ml.ff, act)?;
    let a = gemm_form(&mut ex.acts, ml.ff, slots)?;
    gpu.enqueue_quantize_gemm(act, slots, a, fault)?;
    cx.wk.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty,
            w: stack(w, &ml.down_exps)?,
            rows_per_expert: latent,
            act: a,
            route,
            input: GemmInput::PerSlot,
            y: down,
        },
    )
}

/// The shared expert over the pass's `m` rows of `xn` past [`COLS`] rows:
/// its up and down each one IQ2_XXS GEMM over the pass's one-expert table
/// `dense`, the squared ReLU between.
fn shared_gemm(
    cx: &Cx<'_>,
    (ex, dense): (&mut ExpertArms, &GemmRoute),
    (fault, ml, shared_ff, hidden, ty): (FaultSink, &MoeLayer, usize, usize, GemmWeight),
    xn: &DeviceBuffer<f32>,
    (up, act, y): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    {
        let a = gemm_form(&mut ex.acts, hidden, m)?;
        gpu.enqueue_quantize_gemm(xn, m, a, fault)?;
        cx.wk.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w: stack(w, &ml.up_shexp)?,
                rows_per_expert: shared_ff,
                act: a,
                route: dense,
                input: GemmInput::PerSlot,
                y: up,
            },
        )?;
    }
    gpu.elem()
        .enqueue_relu_sqr(stream, up, m * shared_ff, act)?;
    let a = gemm_form(&mut ex.acts, shared_ff, m)?;
    gpu.enqueue_quantize_gemm(act, m, a, fault)?;
    cx.wk.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty,
            w: stack(w, &ml.down_shexp)?,
            rows_per_expert: hidden,
            act: a,
            route: dense,
            input: GemmInput::PerSlot,
            y,
        },
    )
}

impl Body {
    /// One pass of `ids` (at most the prompt buffers' rows) from position
    /// `pos`, layer-major; with `head`, the head over the pass's last row.
    /// Each MoE layer's results ([`Body::routing`]) are its last row's.
    pub(super) fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (ids, pos): (&[u32], u32),
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let (sh, moe) = (self.sh, self.moe);
        let n_kv = sh.n_kv;
        let stream = gpu.stream();
        let p = &mut self.prompt;
        let q_width = p.q_width;
        if m == 0 || m > p.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} ids; the prompt buffers hold 1..={}", p.rows),
            ));
        }
        let b = &mut p.b;
        b.host_words[P_POS] = pos;
        for h in 0..n_kv {
            b.host_words[P_ROWS + h] = h
                .checked_mul(sh.ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("row {h} · {} + {pos} does not fit a u32 row word", sh.ctx),
                    )
                })?;
        }
        let at = Prompt::ids_at(n_kv);
        b.host_words[at..at + m].copy_from_slice(ids);
        b.words.copy_from_host(stream, &b.host_words)?;
        if m > COLS {
            self.k
                .gemm
                .enqueue_route_dense(stream, m, &mut b.arms.route, gpu.unlabelled_sink())?;
        }
        {
            let (qs, d) = w.q8_planes(WHAT, &self.embed)?;
            let ids_w = Window::<u32>::of(&b.words, at * size_of::<u32>(), m)?;
            let pos0 = Window::<u32>::of(&b.words, P_POS * size_of::<u32>(), 1)?;
            self.q38.enqueue_embed_rows(
                stream,
                EmbedQ8Args {
                    qs,
                    d,
                    ids: &ids_w,
                    pos0: &pos0,
                    first: 0,
                    fault: gpu.unlabelled_sink(),
                    y: &mut p.x[0],
                    pos: &mut b.buf.pos,
                    n_keys: &mut b.buf.n_keys,
                },
            )?;
        }
        let cx = Cx {
            gpu,
            w,
            wk: WideKernels {
                gemm: &self.k.gemm,
                g32: Some(&self.k.g32),
            },
            g32: &self.k.g32,
            kgemv: &self.k.kgemv,
            routers: &self.mk.routers,
            iq: &self.mk.iq,
            m,
        };
        let f = size_of::<f32>();
        let mut cur = 0;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let [x0, x1] = &mut p.x;
            let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) =
                if cur == 0 { (x0, x1) } else { (x1, x0) };
            let fault = gpu.layer_sink(l)?;
            let delta = match layer {
                Layer::Mamba(mm) => {
                    let g = sh.mamba.geom();
                    let (inner, conv_dim) = (g.d_inner, sh.mamba.conv_dim());
                    let ld = sh.mamba.in_proj_width();
                    let buf = &mut b.buf;
                    norm_rows((gpu, w), (sh.eps, sh.hidden, m), &mm.norm, src, &mut buf.xn)?;
                    project_rows(
                        &cx,
                        &mut b.arms,
                        l,
                        (&mm.ssm_in, ld),
                        &buf.xn,
                        &mut buf.zxbcdt,
                    )?;
                    self.k.mamba.enqueue_steps(
                        stream,
                        crate::mamba2::StepsArgs {
                            shape: sh.mamba,
                            u: m,
                            rows: &buf.zxbcdt,
                            ld,
                            z0: 0,
                            x0: inner,
                            dt0: inner + conv_dim,
                            dt_bias: w.f32_buf(WHAT, &mm.dt_bias)?,
                            a: w.f32_buf(WHAT, &mm.a)?,
                            d: w.f32_buf(WHAT, &mm.d)?,
                            conv_w: w.f32_buf(WHAT, &mm.conv_w)?,
                            conv_b: w.f32_buf(WHAT, &mm.conv_b)?,
                            norm: Some(w.f32_buf(WHAT, &mm.ssm_norm)?),
                            eps: sh.eps,
                            pos: &buf.pos,
                            pos_at: 0,
                            fault,
                            y: &mut buf.y,
                            ring: &mut mm.store.ring,
                            state: &mut mm.store.state,
                        },
                    )?;
                    project_rows(
                        &cx,
                        &mut b.arms,
                        l,
                        (&mm.ssm_out, sh.hidden),
                        &buf.y,
                        &mut buf.delta,
                    )?;
                    &buf.delta
                }
                Layer::Attn(a) => {
                    let kv_width = n_kv * HEAD;
                    let buf = &mut b.buf;
                    norm_rows((gpu, w), (sh.eps, sh.hidden, m), &a.norm, src, &mut buf.xn)?;
                    project_rows(&cx, &mut b.arms, l, (&a.q, q_width), &buf.xn, &mut buf.q)?;
                    project_rows(&cx, &mut b.arms, l, (&a.k, kv_width), &buf.xn, &mut buf.k)?;
                    project_rows(&cx, &mut b.arms, l, (&a.v, kv_width), &buf.xn, &mut buf.v)?;
                    for h in 0..n_kv {
                        let shape = Pitched {
                            width: HEAD * f,
                            rows: m,
                            dev_pitch: kv_width * f,
                            host_pitch: HEAD * f,
                        };
                        dtod_2d(
                            stream,
                            (&buf.k, h * HEAD * f),
                            (&mut b.kh, h * m * HEAD * f),
                            shape,
                        )?;
                        dtod_2d(
                            stream,
                            (&buf.v, h * HEAD * f),
                            (&mut b.vh, h * m * HEAD * f),
                            shape,
                        )?;
                    }
                    let append = PlaneAppend {
                        words: &b.words,
                        first_word: P_ROWS,
                        heads: n_kv,
                        m,
                        fault,
                    };
                    let flash = gpu.flash();
                    let (kh, vh) = (&b.kh, &b.vh);
                    flash.enqueue_kv_append_planes(
                        stream,
                        &append,
                        |h| Window::<f32>::of(kh, h * m * HEAD * f, m * HEAD),
                        &mut a.kc,
                    )?;
                    flash.enqueue_kv_append_planes(
                        stream,
                        &append,
                        |h| Window::<f32>::of(vh, h * m * HEAD * f, m * HEAD),
                        &mut a.vc,
                    )?;
                    let mut r0 = 0;
                    while r0 < m {
                        let mr = (m - r0).min(MAX_PASS_ROWS);
                        let qw = Window::<f32>::of(&buf.q, r0 * q_width * f, mr * q_width)?;
                        let nk = Window::<u32>::of(&buf.n_keys, r0 * size_of::<u32>(), mr)?;
                        let mut yw = WindowMut::<f32>::of_mut(
                            &mut buf.attn,
                            r0 * q_width * f,
                            mr * q_width,
                        )?;
                        self.k.flash.enqueue_pass::<GROUP_16>(
                            stream,
                            GqaArgs {
                                q: &qw,
                                kc: a.kc.buf(),
                                vc: a.vc.buf(),
                                n_keys: &nk,
                                scale: 1.0 / (HEAD as f32).sqrt(),
                                n_kv,
                                ctx: sh.ctx,
                                m: mr,
                                part_v: &mut buf.part_v,
                                part_ms: &mut buf.part_ms,
                                fault,
                                y: &mut yw,
                            },
                            false,
                        )?;
                        r0 += mr;
                    }
                    project_rows(
                        &cx,
                        &mut b.arms,
                        l,
                        (&a.o, sh.hidden),
                        &buf.attn,
                        &mut buf.delta,
                    )?;
                    &b.buf.delta
                }
                Layer::Moe(ml) => {
                    norm_rows(
                        (gpu, w),
                        (sh.eps, sh.hidden, m),
                        &ml.norm,
                        src,
                        &mut b.buf.xn,
                    )?;
                    moe_rows(&cx, b, (l, ml, moe), sh.hidden)?;
                    &b.sum
                }
            };
            residual_rows(gpu, m * sh.hidden, src, delta, dst)?;
            cur = 1 - cur;
        }
        if let Some(head) = head {
            let row = Window::<f32>::of(&p.x[cur], (m - 1) * sh.hidden * f, sh.hidden)?;
            head.input_mut().copy_from_device_async(&row, stream)?;
            head.enqueue(gpu, w)?;
        }
        Ok(())
    }
}

impl GpuModel<Body> {
    /// [`GpuModel::prefill_with`] in passes of the most ids the prompt
    /// buffers hold: [`PASS_ROWS`], or the context where that is less.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        let rows = self.body(WHAT)?.prompt.rows();
        self.prefill_with(ids, rows)
    }

    /// Feed `ids` through the chain in passes of up to `rows` ids (module
    /// doc) and return the greedy next token after the last one. Positions
    /// continue from wherever the model stands, and the stores after it are
    /// those of the ids run as steps. `rows` past the prompt buffers' rows,
    /// or 0, and an empty `ids` are refused by name. The passes run eager in
    /// either step mode; a fault any pass raised is read behind the last
    /// pass's head (`GpuModel::run_rows`, the call-end read).
    pub fn prefill_with(&mut self, ids: &[u32], rows: usize) -> Result<u32, GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let cap = self.body(WHAT)?.prompt.rows();
        if rows == 0 || rows > cap {
            return Err(GpuError::shape(
                WHAT,
                format!("passes of {rows} ids; the prompt buffers hold 1..={cap}"),
            ));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "ids", ids.len())? - 1, WHAT)?;
        let n = ids.len().div_ceil(rows);
        let mut next = None;
        for (i, pass) in ids.chunks(rows).enumerate() {
            let last = i + 1 == n;
            next = self.run_rows(pass.len(), WHAT, |gpu, w, body, head, pos| {
                body.enqueue_pass(gpu, w, (pass, pos), last.then_some(head))?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
    }
}
