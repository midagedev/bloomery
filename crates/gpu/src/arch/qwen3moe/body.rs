//! The qwen3moe `ChainBody`: the file's shape read and checked once at load,
//! each layer's weight names and quantization types resolved once, the arena
//! and the K/V planes allocated once, and the per-step parameters refreshed
//! before every enqueue or replay (docs/arch-split.md).

use super::dispatch;
use super::experts::ExpertKernels;
use super::head_argmax::{HeadArgmaxKernels, HeadArgmaxState};
use super::plan::{GqaKind, GqaPlan, Kq, LayerPlan, MixerPlan, MoePlan};
use super::prefill::Prefill;
use super::proj::ProjKernels;
use super::router::{RouterDims, RouterKernels, gated};
use super::scratch::{Arena, Dims, KvPlanes, RopeRows, StepParams, f32_view};
use super::ubatch::{Ubatch, ubatch_size};
use crate::flash_gqa::{FlashGqaKernels, GROUP, HEAD};
use crate::gated_quant::GatedQuantKernels;
use crate::gemm::GemmKernels;
use crate::head::Head;
use crate::linear::LinearKernels;
use crate::model::{ChainBody, GpuModel, Instrumented, NoHost, block_count};
use crate::q6k_sel::Q6kSelKernels;
use crate::rope_neox::RopeNeoxKernels;
use crate::rope_table::{RopeSpec, RopeTable};
use crate::tensor::window;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::Arch;
use model::arch::models::shape::{MoeShape, rules};
use model::arch::qwen3moe::hparams::Hparams;
use model::arch::qwen3moe::names;
use std::mem::ManuallyDrop;
use std::ops::Range;

/// The attention's score scale `1/√HEAD` in f32: the bits `1.0 / (HEAD as
/// f32).sqrt()` rounds to at `HEAD` 128, which every flash launch of every
/// path takes.
pub(super) const ATTN_SCALE: f32 = 0.088_388_346;
const _: () = assert!(HEAD == 128 && ATTN_SCALE.to_bits() == 0x3db5_04f3);

/// The attention's score scale `1/√256` at Qwen3.6's head of 256: exact.
pub(super) const ATTN_SCALE_256: f32 = 0.0625;
const _: () = assert!(crate::flash_gqa::HEAD_256 == 256);

/// The kernels only Qwen3.6's layers launch: the delta rule's three, the
/// gated router and the gated output projection's quantizer.
pub(super) struct Q35Kernels {
    pub(super) linear: LinearKernels,
    pub(super) router: gated::RouterKernels,
    pub(super) gated: GatedQuantKernels,
}

/// The kernels the chain launches beyond the crate's shared modules.
pub(super) struct Kernels {
    pub(super) proj: ProjKernels,
    pub(super) neox: RopeNeoxKernels,
    pub(super) flash: FlashGqaKernels,
    pub(super) router: RouterKernels,
    pub(super) experts: ExpertKernels,
    pub(super) q6_sel: Q6kSelKernels,
    /// The head's Q6_K projection with the argmax folded in.
    pub(super) head: HeadArgmaxKernels,
    /// The grouped GEMM of the ubatch prefill.
    pub(super) gemm: GemmKernels,
    /// Qwen3.6's kernels; `None` on a qwen3moe load.
    pub(super) q35: Option<Q35Kernels>,
}

impl Kernels {
    /// Every module a chain of this family launches, loaded into `gpu`'s
    /// context; Qwen3.6's own when `q35`. Load-time only.
    pub(super) fn load(gpu: &Gpu, q35: bool) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            proj: ProjKernels::load(ctx)?,
            neox: RopeNeoxKernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            q6_sel: Q6kSelKernels::load(ctx, gpu.fault_word())?,
            head: HeadArgmaxKernels::load(ctx)?,
            gemm: GemmKernels::load(ctx)?,
            q35: if q35 {
                Some(Q35Kernels {
                    linear: LinearKernels::load(ctx)?,
                    router: gated::RouterKernels::load(ctx)?,
                    gated: GatedQuantKernels::load(ctx)?,
                })
            } else {
                None
            },
        })
    }

    /// Qwen3.6's kernels, or a named refusal on a qwen3moe load.
    pub(super) fn q35(&self, what: &'static str) -> Result<&Q35Kernels, GpuError> {
        self.q35.as_ref().ok_or(GpuError::state(
            what,
            "Qwen3.6's kernels (a qwen3moe load has none)",
        ))
    }
}

/// qwen3moe's per-replay host values: the token the chain embeds and the
/// cache row it lands in — the step's input record. The row's live key count
/// follows on the card, and its rope row is the table's.
pub struct DecodeInput {
    token: u32,
    pos: u32,
}

/// Everything qwen3moe's chain owns on the device.
pub struct Body {
    /// The prompt prefill's arena, image, slot and captured passes. Declared
    /// first: fields drop in declaration order, and its graphs address the
    /// cache planes and the rope table below.
    pub(super) prefill: Prefill,
    /// The GEMM prefill's arena and prompt image.
    pub(super) ub: Ubatch,
    pub(super) hp: Hparams,
    pub(super) plans: Vec<LayerPlan>,
    pub(super) kv: Vec<KvPlanes>,
    /// The rope table every path reads by position.
    pub(super) rope: RopeRows,
    /// The decode step's one-row arena and its input record.
    pub(super) s: Arena,
    pub(super) sp: StepParams,
    pub(super) k: Kernels,
    /// The fused head argmax's key and ticket, back at their seeds after
    /// every launch.
    pub(super) head_state: HeadArgmaxState,
    /// The flash pass the chain runs: the tensor-core pass from load on,
    /// the scalar one after `GpuModel::set_flash_mma(false)`.
    pub(super) mma: bool,
    /// The per-layer output copies an instrument asked for
    /// ([`Body::set_taps`]).
    pub(super) taps: Option<TapRows>,
}

/// One `hidden` row per layer and a window onto each, the copy target of
/// that layer's output residual.
pub(super) struct TapRows {
    pub(super) buf: DeviceBuffer<f32>,
    pub(super) rows: Vec<ManuallyDrop<DeviceBuffer<f32>>>,
}

/// `(what, want, got)` for every hyperparameter the kernels were built for
/// that the file does not carry: the flash head and group, the rope width;
/// and the router's instance, which the file's routed shape selects.
fn pins(hp: &Hparams) -> Result<RouterDims, GpuError> {
    let e = &hp.experts;
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    let mut rule = rules::SOFTMAX_NORM;
    rule.norm = e.weights_norm;
    let router = RouterDims::of(MoeShape {
        rule,
        experts: count(e.n_expert),
        top_k: count(e.n_used),
    })?;
    let checks = [
        ("attention.key_length (the flash head)", HEAD, hp.head_dim),
        (
            "query heads per key head (the flash group)",
            GROUP,
            hp.group(),
        ),
        ("rope.dimension_count (the NEOX turn)", HEAD, hp.rope.dims),
    ];
    let bad: Vec<String> = checks
        .iter()
        .filter(|(_, want, got)| want != got)
        .map(|(what, want, got)| format!("{what} is {got}, the kernels take {want}"))
        .collect();
    if !bad.is_empty() {
        return Err(GpuError::shape("qwen3moe::Body::load", bad.join("; ")));
    }
    if !hp.n_embd.is_multiple_of(256) || !hp.experts.ff.is_multiple_of(256) {
        return Err(GpuError::shape(
            "qwen3moe::Body::load",
            format!(
                "hidden {} and expert width {} must be whole K-quant super-blocks",
                hp.n_embd, hp.experts.ff
            ),
        ));
    }
    Ok(router)
}

/// Weight `name` as a K-quant word plane of `rows` rows of `k` values, and
/// which of the two K-quants it is; `allowed` says which it may be.
pub(super) fn kq_site(
    w: &Weights,
    name: &str,
    rows: usize,
    k: usize,
    allowed: &[Kq],
) -> Result<Kq, GpuError> {
    let what = "qwen3moe::Body::load";
    let Some(DevWeight::KQuant { ty, w: t, k: wk }) = w.get(name) else {
        return Err(GpuError::tensor(
            what,
            name,
            "a resident K-quant word plane",
        ));
    };
    let kq = match ty {
        GgmlType::Q4_K => Kq::Q4K,
        GgmlType::Q6_K => Kq::Q6K,
        _ => {
            return Err(GpuError::shape(
                what,
                format!("{name} is {ty}; the chain runs Q4_K and Q6_K"),
            ));
        }
    };
    if !allowed.contains(&kq) || t.rows() != rows || *wk != k {
        return Err(GpuError::shape(
            what,
            format!(
                "{name} is {ty} {} rows x {wk}; the chain takes {allowed:?} {rows} rows x {k}",
                t.rows()
            ),
        ));
    }
    Ok(kq)
}

/// Weight `name` as a resident F32 plane of `rows` rows of `k`.
pub(super) fn f32_site(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    let what = "qwen3moe::Body::load";
    match w.get(name) {
        Some(DevWeight::F32 { w: t, k: wk }) if t.rows() == rows && *wk == k => Ok(()),
        Some(DevWeight::F32 { w: t, k: wk }) => Err(GpuError::shape(
            what,
            format!(
                "{name} is F32 {} x {wk}, the chain takes {rows} x {k}",
                t.rows()
            ),
        )),
        _ => Err(GpuError::tensor(what, name, "a resident F32 plane")),
    }
}

/// Layer `l`'s plan: attention at head 128, eight routed experts, every
/// weight checked against the shape and type its launch takes.
fn resolve(w: &Weights, hp: &Hparams, l: usize) -> Result<LayerPlan, GpuError> {
    let (h, q, kv, ff) = (
        hp.n_embd,
        hp.n_head * hp.head_dim,
        hp.n_head_kv * hp.head_dim,
        hp.experts.ff,
    );
    let e = hp.experts.n_expert;
    let mut g = GqaPlan {
        kind: GqaKind::Neox128,
        attn_norm: names::attn_norm(l),
        attn_q: names::attn_q(l),
        attn_k: names::attn_k(l),
        attn_v: names::attn_v(l),
        attn_q_norm: names::attn_q_norm(l),
        attn_k_norm: names::attn_k_norm(l),
        attn_output: names::attn_output(l),
        v_ty: Kq::Q4K,
    };
    let mut f = MoePlan {
        ffn_norm: names::ffn_norm(l),
        ffn_gate_inp: names::ffn_gate_inp(l),
        ffn_gate_exps: names::ffn_gate_exps(l),
        ffn_up_exps: names::ffn_up_exps(l),
        ffn_down_exps: names::ffn_down_exps(l),
        down_ty: Kq::Q4K,
        shared: None,
    };
    for (name, len) in [
        (&g.attn_norm, h),
        (&f.ffn_norm, h),
        (&g.attn_q_norm, hp.head_dim),
        (&g.attn_k_norm, hp.head_dim),
    ] {
        f32_site(w, name, 1, len)?;
    }
    f32_site(w, &f.ffn_gate_inp, e, h)?;
    kq_site(w, &g.attn_q, q, h, &[Kq::Q4K])?;
    kq_site(w, &g.attn_k, kv, h, &[Kq::Q4K])?;
    g.v_ty = kq_site(w, &g.attn_v, kv, h, &[Kq::Q4K, Kq::Q6K])?;
    kq_site(w, &g.attn_output, h, q, &[Kq::Q4K])?;
    kq_site(w, &f.ffn_gate_exps, e * ff, h, &[Kq::Q4K])?;
    kq_site(w, &f.ffn_up_exps, e * ff, h, &[Kq::Q4K])?;
    f.down_ty = kq_site(w, &f.ffn_down_exps, e * h, ff, &[Kq::Q4K, Kq::Q6K])?;
    Ok(LayerPlan {
        mixer: MixerPlan::Gqa(g),
        ffn: f,
    })
}

impl Body {
    /// Keep a copy of every layer's output residual after each layer of the
    /// chain (`on`), or stop. The copies are one memcpy node per layer in a
    /// capture; the gates read them in eager mode. Load-time allocation.
    pub(super) fn set_taps(&mut self, stream: &CudaStream, on: bool) -> Result<(), GpuError> {
        self.taps = None;
        if on {
            let hidden = self.s.dims.hidden;
            let buf = DeviceBuffer::<f32>::zeroed(stream, self.plans.len() * hidden)?;
            let rows = (0..self.plans.len())
                .map(|l| {
                    let ptr = buf.cu_deviceptr() + (l * hidden * size_of::<f32>()) as u64;
                    // SAFETY: row `l` spans elements `l·hidden ..
                    // (l+1)·hidden` of `buf`, which moves into the same
                    // `TapRows` beside its windows and outlives them.
                    unsafe { window(ptr, hidden, buf.context()) }
                })
                .collect();
            self.taps = Some(TapRows { buf, rows });
        }
        Ok(())
    }

    /// The flash pass this body runs: `true` for the tensor-core pass.
    #[must_use]
    pub fn flash_mma(&self) -> bool {
        self.mma
    }

    /// The file's hyperparameters, as the body read them.
    #[must_use]
    pub fn hparams(&self) -> &Hparams {
        &self.hp
    }
}

/// What [`GpuModel::open`] takes besides the card and the file: the cache's
/// rows and the two load-time choices a binary reads at its edge
/// ([`GpuModel::lever_opts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenOpts {
    /// KV rows the caches hold.
    pub ctx: usize,
    /// Tokens per ubatch of the GEMM prefill ([`Ubatch`]); a size outside
    /// the arena's range is refused at load.
    pub ubatch: usize,
    /// The decode flash pass.
    pub flash: FlashKind,
}

/// The decode flash pass a qwen3moe body runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashKind {
    /// The tensor-core pass.
    Mma,
    /// The scalar segment pass, the tensor-core pass's banded twin.
    Scalar,
}

impl FlashKind {
    /// `true` is the tensor-core pass.
    #[must_use]
    pub fn from_mma(mma: bool) -> FlashKind {
        if mma {
            FlashKind::Mma
        } else {
            FlashKind::Scalar
        }
    }
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format, the caches of `opts.ctx`
    /// rows, the arenas and the step module. The model takes the file and
    /// closes it once the weights are resident.
    pub fn open(gpu: Gpu, file: Split, opts: OpenOpts) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "qwen3moe GpuModel::open")?;
        let mma = opts.flash == FlashKind::Mma;
        GpuModel::load_blocks(gpu, &file, opts.ctx, 0..n_layers, true, |gpu, w| {
            Body::load(gpu, &file, w, 0..n_layers, opts.ctx, opts.ubatch, mma)
        })
    }

    /// `ctx` cache rows, the ubatch size this process's lever names
    /// ([`ubatch_size`]) and the tensor-core flash, the engine's only decode
    /// pass: the options a binary's edge passes to [`GpuModel::open`] when it
    /// takes the levers as they are.
    pub fn lever_opts(ctx: usize) -> Result<OpenOpts, GpuError> {
        Ok(OpenOpts {
            ctx,
            ubatch: ubatch_size()?,
            flash: FlashKind::Mma,
        })
    }
}

impl Body {
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
        ubatch: usize,
        mma: bool,
    ) -> Result<Body, GpuError> {
        let what = "qwen3moe::Body::load";
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(what, e))?;
        let router = pins(&hp)?;
        if layers != (0..hp.n_layer) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the chain runs the whole model, layers 0..{}; asked for {layers:?}",
                    hp.n_layer
                ),
            ));
        }
        if ctx_max == 0 {
            return Err(GpuError::shape(what, "a cache of 0 rows"));
        }
        let plans = layers
            .clone()
            .map(|l| resolve(w, &hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        let dims = Dims::qwen3(
            hp.n_embd,
            hp.n_head,
            hp.n_head_kv,
            hp.experts.ff,
            router,
            ctx_max,
        );
        let stream = gpu.stream();
        let kv = layers
            .map(|_| KvPlanes::new(stream, &dims))
            .collect::<Result<Vec<_>, _>>()?;
        let k = Kernels::load(gpu, false)?;
        let rope = RopeTable::new(&RopeSpec::window(hp.rope.base, hp.rope.dims))?;
        Ok(Body {
            prefill: Prefill::new(stream, dims)?,
            ub: Ubatch::new(stream, dims, ubatch)?,
            rope: RopeRows::new(stream, &rope, HEAD, dims.ctx)?,
            s: Arena::new(stream, dims, 1)?,
            sp: StepParams::new(stream, false)?,
            hp,
            plans,
            kv,
            k,
            head_state: HeadArgmaxState::new(stream)?,
            mma,
            taps: None,
        })
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = NoHost;

    fn arch() -> Arch {
        Arch::Qwen3moe
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step's input record — its position and its token — in one
    /// asynchronous copy ahead of the step's launches.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        let DecodeInput { token, pos } = *input;
        self.sp.write(stream, token, pos)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        dispatch::enqueue_chain(gpu, w, self, head)
    }

    /// Nothing to clear: the flash never loads a key row at or past the live
    /// count, and every row below it is written by its own step first.
    fn reset(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.kv.iter().map(KvPlanes::bytes).sum::<usize>()
            + self.rope.table.num_bytes()
            + self.s.bytes()
            + self.sp.bytes()
            + self.head_state.bytes()
            + self.prefill.bytes()
            + self.ub.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}

impl Instrumented for Body {
    /// Rows `0..rows` of every layer's K and V planes filled with a
    /// deterministic pattern of finite, nonzero f16 values that differ from
    /// row to row — a step shape, not a model state.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let d = self.s.dims;
        let mut plane = vec![0u16; d.n_kv * d.ctx * HEAD];
        let mut state = 0x9e37_79b9u32 ^ rows as u32;
        for h in 0..d.n_kv {
            for r in 0..rows.min(d.ctx) {
                for c in 0..HEAD {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let v = ((state >> 9) as f32 / (1u32 << 23) as f32 - 0.5) + 1.0 / 64.0;
                    plane[(h * d.ctx + r) * HEAD + c] = gguf::quant::f32_to_f16_bits(v);
                }
            }
        }
        let stream = gpu.stream();
        for p in &mut self.kv {
            p.k.copy_from_host(stream, &plane)?;
            p.v.copy_from_host(stream, &plane)?;
        }
        stream.synchronize()?;
        Ok(())
    }
}

impl GpuModel<Body> {
    /// Rows `positions` of the rope table every path reads, [`HEAD`] f32 a
    /// row (the table's width). Synchronizes; gate use.
    pub fn rope_rows(&self, positions: Range<usize>) -> Result<Vec<f32>, GpuError> {
        const WHAT: &str = "qwen3moe::rope_rows";
        let rope = &self.body(WHAT)?.rope;
        if positions.is_empty() || positions.end > rope.rows() {
            return Err(GpuError::shape(
                WHAT,
                format!("rope rows {positions:?} of a {}-row table", rope.rows()),
            ));
        }
        // SAFETY: rows `positions` end at or below the table's rows, inside
        // it; the window lives for this copy alone.
        let rows = unsafe {
            f32_view(
                &rope.table,
                positions.start * rope.width,
                positions.len() * rope.width,
            )
        };
        Ok(rows.to_host_vec(self.stage_stream()?)?)
    }
}
