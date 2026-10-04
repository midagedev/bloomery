//! The qwen3moe `ChainBody`: the file's shape read and checked once at load,
//! each layer's weight names and quantization types resolved once, the arena
//! and the K/V planes allocated once, and the per-step parameters refreshed
//! before every enqueue or replay (docs/arch-split.md).

use super::dispatch::{self, PassCtx};
use super::experts::ExpertKernels;
use super::head_argmax::{HeadArgmaxKernels, HeadArgmaxState};
use super::placed::{Placed, PlacedOpen, StepWalk, WalkParts, placed_bytes};
use super::plan::{FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, LayerPlan, MixerPlan, SiteTy};
use super::prefill::Prefill;
use super::program::Tail;
use super::proj::ProjKernels;
use super::router::{RouterDims, RouterKernels, gated};
use super::scratch::{Arena, Dims, KvPlanes, RopeRows, StepParams, f32_view};
use super::ubatch::{Ubatch, ubatch_size};
use crate::flash_gqa::{FlashGqaKernels, GROUP, HEAD};
use crate::flash_gqa_prefill::FlashGqaPrefill;
use crate::gated_quant::GatedQuantKernels;
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::head::Head;
use crate::host::StepLeg;
use crate::linear::LinearKernels;
use crate::model::{ChainBody, GpuModel, Instrumented, Rollback, Slots, block_count};
use crate::q6k_sel::Q6kSelKernels;
use crate::q38::Q38Kernels;
use crate::rope_neox::RopeNeoxKernels;
use crate::rope_table::{RopeSpec, RopeTable};
use crate::site::KGemvKernels;
use crate::tensor::window;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Graph};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::Arch;
use model::arch::models::shape::{MoeShape, rules};
use model::arch::qwen3moe::hparams::Hparams;
use model::arch::qwen3moe::names;
use model::placement::Plan;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;

/// The attention's score scale `1/√HEAD` in f32: the bits `1.0 / (HEAD as
/// f32).sqrt()` rounds to at `HEAD` 128, which every flash launch of every
/// path takes.
pub(super) const ATTN_SCALE: f32 = 0.088_388_346;
const _: () = assert!(HEAD == 128 && ATTN_SCALE.to_bits() == 0x3db5_04f3);

/// The attention's score scale `1/√256` at Qwen3.6's head of 256: exact.
pub(super) const ATTN_SCALE_256: f32 = 0.0625;
const _: () = assert!(crate::flash_gqa::HEAD_256 == 256);

/// The kernels only Qwen3.6's layers launch: the delta rule's three, the
/// gated router and the gated output projection's quantizer; and what a
/// site of a type other than the K-quants launches — the 32-value GEMM
/// family with its quantizers and the F32 tile, and the Q8_0 embedding and
/// the f32 out gate (a Q8_0 or F32 `attn_output` reads the gated rows as
/// f32).
pub(super) struct Q35Kernels {
    pub(super) linear: LinearKernels,
    pub(super) router: gated::RouterKernels,
    pub(super) gated: GatedQuantKernels,
    pub(super) g32: Gemm32Kernels,
    pub(super) q38: Q38Kernels,
    /// A K-quant site's gemv beyond the `Gpu`'s own (the Q5_K one).
    pub(super) kgemv: KGemvKernels,
}

/// The kernels the chain launches beyond the crate's shared modules.
pub(super) struct Kernels {
    pub(super) proj: ProjKernels,
    pub(super) neox: RopeNeoxKernels,
    pub(super) flash: FlashGqaKernels,
    /// The prefill flash: each of a prompt unit's rows over its own live key
    /// count, the cache's key tiles staged once per block.
    pub(super) prefill: FlashGqaPrefill,
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
            prefill: FlashGqaPrefill::load(ctx)?,
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
                    g32: Gemm32Kernels::load(ctx)?,
                    q38: Q38Kernels::load(ctx)?,
                    kgemv: KGemvKernels::load(gpu)?,
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
    /// A placed load's placed side ([`Placed`]); `None` on a whole-card
    /// load.
    pub(super) placed: Option<Placed>,
}

/// One `hidden` row per layer and a window onto each, the copy target of
/// that layer's output residual.
pub(super) struct TapRows {
    pub(super) buf: DeviceBuffer<f32>,
    pub(super) rows: Vec<ManuallyDrop<DeviceBuffer<f32>>>,
}

/// One qwen3moe sequence's own state ([`Slots`]): its per-layer K/V planes,
/// and the prefill passes captured over them — a captured pass replays the
/// recorded plane addresses, so the captures follow the planes. The captures
/// are declared before the planes: a parked sequence drops them first, while
/// the buffers they address are alive.
pub struct Seq {
    pass_graphs: Vec<Option<Graph>>,
    kv: Vec<KvPlanes>,
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
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    let what = "qwen3moe::Body::load";
    let Some(DevWeight::KQuant { ty, w: t, k: wk }) = w.get(name) else {
        return Err(GpuError::tensor(
            what,
            name,
            "a resident K-quant word plane",
        ));
    };
    let kq = match ty {
        GgmlType::Q4_K => SiteTy::Q4K,
        GgmlType::Q6_K => SiteTy::Q6K,
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
/// weight checked against the shape and type its launch takes — the routed
/// stacks only when `stacks` (a placed load's are its placed side's to
/// check, `placed`).
fn resolve(w: &Weights, hp: &Hparams, l: usize, stacks: bool) -> Result<LayerPlan, GpuError> {
    let (h, q, kv, ff) = (
        hp.n_embd,
        hp.n_head * hp.head_dim,
        hp.n_head_kv * hp.head_dim,
        hp.experts.ff,
    );
    let e = hp.experts.n_expert;
    let mut g = GqaPlan {
        kind: GqaKind::Neox128,
        flash: Flash::Group,
        attn_norm: names::attn_norm(l),
        attn_q: names::attn_q(l),
        attn_k: names::attn_k(l),
        attn_v: names::attn_v(l),
        attn_q_norm: names::attn_q_norm(l),
        attn_k_norm: names::attn_k_norm(l),
        attn_output: names::attn_output(l),
        q_ty: SiteTy::Q4K,
        k_ty: SiteTy::Q4K,
        v_ty: SiteTy::Q4K,
        o_ty: SiteTy::Q4K,
    };
    let router = names::ffn_gate_inp(l);
    let mut f = FfnPlan {
        ffn_norm: names::ffn_norm(l),
        route: FfnRoute::Router {
            gate_inp: router.clone(),
            shared: None,
        },
        gate: names::ffn_gate_exps(l),
        up: names::ffn_up_exps(l),
        down: names::ffn_down_exps(l),
        gate_ty: SiteTy::Q4K,
        up_ty: SiteTy::Q4K,
        down_ty: SiteTy::Q4K,
    };
    for (name, len) in [
        (&g.attn_norm, h),
        (&f.ffn_norm, h),
        (&g.attn_q_norm, hp.head_dim),
        (&g.attn_k_norm, hp.head_dim),
    ] {
        f32_site(w, name, 1, len)?;
    }
    f32_site(w, &router, e, h)?;
    kq_site(w, &g.attn_q, q, h, &[SiteTy::Q4K])?;
    kq_site(w, &g.attn_k, kv, h, &[SiteTy::Q4K])?;
    g.v_ty = kq_site(w, &g.attn_v, kv, h, &[SiteTy::Q4K, SiteTy::Q6K])?;
    kq_site(w, &g.attn_output, h, q, &[SiteTy::Q4K])?;
    if !stacks {
        return Ok(LayerPlan {
            mixer: MixerPlan::Gqa(g),
            ffn: f,
        });
    }
    kq_site(w, &f.gate, e * ff, h, &[SiteTy::Q4K])?;
    kq_site(w, &f.up, e * ff, h, &[SiteTy::Q4K])?;
    f.down_ty = kq_site(w, &f.down, e * h, ff, &[SiteTy::Q4K, SiteTy::Q6K])?;
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

    /// The placed side of a placed load ([`Placed`]); `None` on a
    /// whole-card load.
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
    }
}

/// The format a qwen3-family load holds its KV cache planes in: f16 (the
/// format before the lever), or q8_0's two-plane layout — 17/16 bytes a
/// value against f16's 2, both planes (K and V) quantized together by the
/// one choice. `q8_0` is the opt-in arm ([`OpenOpts::kv`], the seat's
/// `--cache-type-k` spelling): a cache half the f16 bytes wide, its rows the
/// quantizing appends' and the q8 read paths'.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQ8 {
    /// `[n_kv][ctx][head]` f16 a side — the only format until the lever.
    #[default]
    F16,
    /// Per side a codes plane of `head/4` u32 a row and a scales plane of
    /// `head/32` u16 ([`rope_neox::q8_plane_lens`](crate::rope_neox)).
    Q8,
}

impl KvQ8 {
    /// The word the lever and `--cache-type-k` take (llama-server's
    /// spelling), or `None` on any other.
    #[must_use]
    pub fn parse(word: &str) -> Option<KvQ8> {
        match word {
            "f16" => Some(KvQ8::F16),
            "q8_0" => Some(KvQ8::Q8),
            _ => None,
        }
    }

    /// The word [`KvQ8::parse`] took.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            KvQ8::F16 => "f16",
            KvQ8::Q8 => "q8_0",
        }
    }
}

/// What [`GpuModel::open`] takes besides the card and the file: the cache's
/// rows and the load-time choices a binary reads at its edge
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
    /// The KV planes' format ([`KvQ8`]).
    pub kv: KvQ8,
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
            Body::load(
                gpu,
                &file,
                w,
                0..n_layers,
                opts.ctx,
                opts.ubatch,
                mma,
                opts.kv,
                None,
            )
        })
    }

    /// The Qwen3-30B model of `file` placed by `plan` (made by
    /// `model::arch::qwen3moe::place::PlanInputs::plan` on the machine
    /// `model::arch::qwen3moe::place::machine` lays out): the plan's card
    /// segments resident ([`GpuModel::load_placed`]: the trunk and each
    /// layer's card experts), the host set read in and locked as `host`
    /// asks, and the body over them with its placed side ([`Placed`]) and
    /// caches of `opts.ctx` rows, the plan's context. Its prompt runs as
    /// eager passes through the host tier's batch port, so its ubatch arena
    /// holds one row. Refused by name as [`GpuModel::open`] refuses, for a
    /// plan of another context, and as the placed side refuses
    /// ([`Placed::new`]).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        opts: OpenOpts,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        const WHAT_P: &str = "qwen3moe GpuModel::open_placed";
        let n_layers = block_count(&file, WHAT_P)?;
        if u64::try_from(opts.ctx).ok() != Some(plan.ctx_max) {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "caches of {} rows on a plan of {} positions",
                    opts.ctx, plan.ctx_max
                ),
            ));
        }
        let card = plan
            .machine
            .cards
            .first()
            .ok_or(GpuError::shape(WHAT_P, "a plan of no card"))?;
        let counted = model::arch::qwen3moe::place::counted_arena_bytes(card);
        let mma = opts.flash == FlashKind::Mma;
        GpuModel::load_placed(
            file,
            plan,
            0,
            host,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set| {
                let file = Arc::new(file);
                let open = PlacedOpen {
                    plan,
                    file: Arc::clone(&file),
                    host,
                    set,
                    arch: "qwen3moe",
                };
                Body::load(
                    gpu,
                    &file,
                    w,
                    0..n_layers,
                    opts.ctx,
                    1,
                    mma,
                    opts.kv,
                    Some((open, counted)),
                )
            },
        )
    }

    /// The card bytes a placed load of `file` holds past the m = 1 scratch
    /// for its placed side ([`placed_bytes`]), at caches of `ctx` rows: what
    /// the plan's machine counts in the card's scratch
    /// (`model::arch::qwen3moe::place::machine`), and what
    /// [`GpuModel::open_placed`] refuses to pass.
    pub fn placed_arena_bytes(file: &Split, ctx: usize) -> Result<u64, GpuError> {
        let what = "qwen3moe GpuModel::placed_arena_bytes";
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(what, e))?;
        let router = pins(&hp)?;
        let dims = Dims::qwen3(
            hp.n_embd,
            hp.n_head,
            hp.n_head_kv,
            hp.experts.ff,
            router,
            ctx,
        );
        placed_bytes(&dims, hp.n_layer)
    }

    /// `ctx` cache rows, the ubatch size this process's lever names
    /// ([`ubatch_size`]), the tensor-core flash (the engine's only decode
    /// pass) and the KV planes' format `kv` ([`KvQ8`]): the options a
    /// binary's edge passes to [`GpuModel::open`] when it takes the levers
    /// as they are.
    pub fn lever_opts(ctx: usize, kv: KvQ8) -> Result<OpenOpts, GpuError> {
        Ok(OpenOpts {
            ctx,
            ubatch: ubatch_size()?,
            flash: FlashKind::Mma,
            kv,
        })
    }
}

impl Body {
    /// The body over the resident weights `w` of `file`'s `layers` (every
    /// layer), with caches of `ctx_max` rows in the `kv` format, ubatches of
    /// `ubatch` tokens and the flash `mma`; on a placed load (`placed`: its
    /// placed side's inputs and the bytes its plan counts for that side) the
    /// routed stacks are the placed side's to check, and the side is built
    /// ([`Placed::new`]), its bytes held to the plan's count first.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card, file, weights, layers, cache and format, ubatch and flash, and the placed side (rust-quality R8)"
    )]
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
        ubatch: usize,
        mma: bool,
        kv: KvQ8,
        placed: Option<(PlacedOpen<'_>, u64)>,
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
            .map(|l| resolve(w, &hp, l, placed.is_none()))
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
            .map(|_| KvPlanes::new(stream, &dims, kv))
            .collect::<Result<Vec<_>, _>>()?;
        let k = Kernels::load(gpu, false)?;
        let rope = RopeTable::new(&RopeSpec::window(hp.rope.base, hp.rope.dims))?;
        let placed = match placed {
            None => None,
            Some((open, counted)) => {
                let side = placed_bytes(&dims, plans.len())?;
                if side > counted {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "the placed side needs {side} card bytes; the plan counts {counted}"
                        ),
                    ));
                }
                Some(Placed::new(gpu, w, &plans, &dims, open)?)
            }
        };
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
            placed,
        })
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Placed;

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
        if self.placed.is_some() {
            return enqueue_placed_chain(gpu, w, self, head);
        }
        dispatch::enqueue_chain(gpu, w, self, head)
    }

    /// The caches need nothing: the flash never loads a key row at or past
    /// the live count, and every row below it is written by its own step
    /// first. A placed load's host tier is reset ([`Placed`]).
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(p) => p.reset(gpu.stream()),
            None => Ok(()),
        }
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
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}

impl Slots for Body {
    type Seq = Seq;

    /// A sequence of the load's shape and cache format in the state the load
    /// leaves: zeroed planes, no captured pass. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Seq, GpuError> {
        const WHAT: &str = "qwen3moe::Body::new_seq";
        let Some(first) = self.kv.first() else {
            return Err(GpuError::state(
                WHAT,
                "a layer's planes (the chain has none)",
            ));
        };
        // The load's one cache-format choice, read back from the live planes:
        // the variant the load allocated is the variant a new one takes.
        let kv = match first {
            KvPlanes::F16 { .. } => KvQ8::F16,
            KvPlanes::Q8 { .. } => KvQ8::Q8,
        };
        let stream = gpu.stream();
        let dims = self.prefill.a.dims;
        let planes = (0..self.kv.len())
            .map(|_| KvPlanes::new(stream, &dims, kv))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Seq {
            pass_graphs: Prefill::no_passes(),
            kv: planes,
        })
    }

    /// Exchange the live sequence's planes and captured passes with `seq`'s:
    /// pointer moves, no device work.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Seq) -> Result<(), GpuError> {
        std::mem::swap(&mut self.kv, &mut seq.kv);
        std::mem::swap(&mut self.prefill.graphs, &mut seq.pass_graphs);
        Ok(())
    }

    /// Device bytes one sequence holds: its layers' K/V planes.
    fn seq_bytes(&self) -> usize {
        self.kv.iter().map(KvPlanes::bytes).sum()
    }
}

/// Enqueue a placed load's decode chain at one row through the host tier's
/// step port (`placed`'s step walk), then the head.
fn enqueue_placed_chain(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let Body {
        hp,
        plans,
        kv,
        rope,
        s,
        sp,
        k,
        head_state,
        mma,
        taps,
        placed,
        ..
    } = b;
    let Some(Placed {
        hybrid, side, step, ..
    }) = placed
    else {
        return Err(GpuError::state(
            "qwen3moe::enqueue_placed_chain",
            "a placed side (a placed load)",
        ));
    };
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: hp.rms_eps,
        table: &rope.table,
    };
    let mut leg = StepLeg::new(gpu.stream(), hybrid);
    StepWalk {
        p: WalkParts {
            c: &c,
            stores: kv.as_mut_slice(),
            s,
            io: &sp.io(),
            m: 1,
            side,
            rows: step,
        },
        tail: Tail::Step {
            head,
            state: head_state,
            taps: taps.as_mut(),
        },
    }
    .walk(&mut leg)
}

impl Instrumented for Body {
    /// Rows `0..rows` of every layer's K and V planes filled with a
    /// deterministic pattern of finite, nonzero values that differ from row
    /// to row — f16 bits on an f16 cache, the q8_0 form of the same values
    /// (the engine's one quantizer's rule) on a q8_0 one — a step shape, not
    /// a model state.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let d = self.s.dims;
        let mut vals = vec![0f32; d.n_kv * d.ctx * HEAD];
        let mut state = 0x9e37_79b9u32 ^ rows as u32;
        for h in 0..d.n_kv {
            for r in 0..rows.min(d.ctx) {
                for c in 0..HEAD {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let v = ((state >> 9) as f32 / (1u32 << 23) as f32 - 0.5) + 1.0 / 64.0;
                    vals[(h * d.ctx + r) * HEAD + c] = v;
                }
            }
        }
        let stream = gpu.stream();
        for p in &mut self.kv {
            p.fill(stream, &vals, &d)?;
        }
        stream.synchronize()?;
        Ok(())
    }
}

impl Rollback for Body {
    /// Nothing to take back: the caches are per-position, the flash never
    /// loads a key row at or past the live count, and every row below it is
    /// written by its own step first ([`ChainBody::reset`]'s rule), so the
    /// rows past `pos` are dead the moment the model stands at `pos`.
    fn rollback(&mut self, _pos: u32) -> Result<(), GpuError> {
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
