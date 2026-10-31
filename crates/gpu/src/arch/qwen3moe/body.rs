//! The qwen3moe `ChainBody`: the file's shape read and checked once at load,
//! each layer's weight names and quantization types resolved once, the arena
//! and the K/V planes allocated once, and the per-step parameters refreshed
//! before every enqueue or replay (docs/arch-split.md).

use super::dispatch::{self, Ctx, PassCtx};
use super::experts::ExpertKernels;
use super::head_argmax::{HeadArgmaxKernels, HeadArgmaxState};
use super::placed::{
    ColsMix, ColsWalk, CommonForms, PLACED_ROWS, Placed, PlacedOpen, StepWalk, WalkParts,
    common_forms, counted_arena, placed_bytes, placed_scratch, slot_chain,
};
use super::plan::{
    FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, LayerPlan, MixerPlan, ResidualPlan, SiteTy,
    down_entry, gate_up_entry, kq_sel, o_fused, qkv_fused,
};
use super::prefill::Prefill;
use super::program::Tail;
use super::proj::ProjKernels;
use super::router::gated;
use super::scratch::{
    Append128, Arena, Dims, FlashPass, Forms, KvPlanes, RopeRows, StepParams, cells_view,
};
use super::slot_pass::{SlotIn, SlotPass};
use super::ubatch::{Q32, Ubatch, ubatch_size};
use super::wide::GEMV_COLS;
use crate::flash_gqa::{FlashGqaKernels, GROUP, HEAD, RowTable, partials_ms_len, partials_v_len};
use crate::flash_gqa_prefill::FlashGqaPrefill;
use crate::gated_quant::GatedQuantKernels;
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::head::Head;
use crate::host::StepLeg;
use crate::host::swap_source::ResidencyGlue;
use crate::linear::LinearKernels;
use crate::model::lookup::f32_gain;
use crate::model::{
    ChainBody, GpuModel, Instrumented, Rollback, RowHeads, SlotRange, SlotRows, Slots, block_count,
    refuse_stage_rows_past,
};
use crate::q6k_sel::Q6kSelKernels;
use crate::q38::Q38Kernels;
use crate::rope_neox::RopeNeoxKernels;
use crate::rope_norm::RopeNormKernels;
use crate::rope_table::{RopeSpec, RopeTable};
use crate::route_softmax::{RouterDims, RouterKernels};
use crate::site::{self, KGemvKernels};
use crate::tensor::{Window, WindowMut};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Graph};
use bloomery_levers::HostCfg;
use chain::ChainError;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::Arch;
use model::arch::models::RouterNorm;
use model::arch::models::shape::{MoeShape, rules};
use model::arch::qwen3moe::hparams::Hparams;
use model::arch::qwen3moe::names::{self, token_embd};
use model::placement::accepts::qwen3moe_dense;
use model::placement::kernels;
use model::placement::{Plan, WholeLoad};
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
/// gated router and the gated output projection's quantizer, the NORM rope
/// append of a head-128 layer with no head norm; and what a
/// site that is not one of the fused Q4_K groups launches — the 32-value
/// GEMM family with its quantizers and the F32 tile, the K-quant gemv beyond
/// the `Gpu`'s own (Q5_K), the Q8_0 embedding and the f32 out gate (a Q8_0
/// or F32 `attn_output` reads the gated rows as f32). A qwen3moe load holds
/// them only when one of its sites or its embedding takes such a launch
/// ([`needs_site_kernels`]).
pub(super) struct Q35Kernels {
    pub(super) linear: LinearKernels,
    pub(super) router: gated::RouterKernels,
    pub(super) gated: GatedQuantKernels,
    pub(super) g32: Gemm32Kernels,
    pub(super) q38: Q38Kernels,
    /// The NORM turn and cache append of a head-128 layer with no head norm.
    pub(super) norm: RopeNormKernels,
    /// A K-quant site's gemv beyond the `Gpu`'s own (the Q5_K one).
    pub(super) kgemv: KGemvKernels,
    /// The FFN's K-quant `_sel` family and its combine with a host sum.
    pub(super) ffn: dispatch::FfnKernels,
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
                    norm: RopeNormKernels::load(ctx)?,
                    kgemv: KGemvKernels::load(gpu)?,
                    ffn: dispatch::FfnKernels::load(gpu)?,
                })
            } else {
                None
            },
        })
    }

    /// Qwen3.6's kernels, or a named refusal on a load that holds none (a
    /// qwen3moe load whose sites are all fused Q4_K groups).
    pub(super) fn q35(&self, what: &'static str) -> Result<&Q35Kernels, GpuError> {
        self.q35.as_ref().ok_or(GpuError::state(
            what,
            "Qwen3.6's kernels (a qwen3moe load whose sites and embedding take none holds none)",
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
    /// A pass of several slots' input records ([`SlotRows`]); the pass
    /// runs on the prompt arena.
    pub(super) slot_in: SlotIn,
    pub(super) k: Kernels,
    /// The fused head argmax's key and ticket, back at their seeds after
    /// every launch.
    pub(super) head_state: HeadArgmaxState,
    /// The flash pass the chain runs: the tensor-core pass from load on,
    /// the scalar one after `GpuModel::set_flash_mma(false)`.
    pub(super) mma: bool,
    /// The per-layer output copies an instrument asked for
    /// ([`TapRows::set`]).
    pub(super) taps: Option<TapRows>,
    /// A placed load's placed side ([`Placed`]); `None` on a whole-card
    /// load.
    pub(super) placed: Option<Placed>,
}

/// One `hidden` row per layer and a window onto each, the copy target of
/// that layer's output residual.
pub(super) struct TapRows {
    pub(super) buf: DeviceBuffer<f32>,
    pub(super) rows: Vec<WindowMut<'static, f32>>,
}

impl TapRows {
    /// Keep a copy of every layer's output residual after each layer of the
    /// chain (`on`), or stop: `layers` rows of `hidden`. The copies are one
    /// memcpy node per layer in a capture; the gates read them in eager
    /// mode. Load-time allocation.
    pub(super) fn set(
        taps: &mut Option<Self>,
        stream: &CudaStream,
        layers: usize,
        hidden: usize,
        on: bool,
    ) -> Result<(), GpuError> {
        *taps = None;
        if on {
            let buf = DeviceBuffer::<f32>::zeroed(stream, layers * hidden)?;
            let rows = (0..layers)
                // SAFETY: row `l` spans elements `l·hidden .. (l+1)·hidden` of
                // `buf`, which moves into the same `TapRows` beside its windows
                // and outlives them.
                .map(|l| unsafe { cells_view(&buf, l * hidden, hidden) })
                .collect();
            *taps = Some(Self { buf, rows });
        }
        Ok(())
    }
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
    rule.norm = RouterNorm::from_legacy_bool(e.weights_norm);
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

/// Layer `l`'s routed stacks `[gate, up, down]` of `n` experts (or a stack of
/// one expert: a placed layer's shared expert) of `ff` rows of `h` values:
/// each a resident K-quant word plane of its shape, its type the file's (the
/// layout checked by [`super::placed::stack_files`]), then the types the card
/// kernel table's entries admit for the family's FFN dispatch
/// ([`gate_up_entry`]: one launch reads the gate and up of one type;
/// [`down_entry`]), a type with no entry refused by name with the layer.
/// Returns the gate, up and down types.
pub(super) fn stack_tys(
    w: &Weights,
    names: [&str; 3],
    (n, ff, h): (usize, usize, usize),
    l: usize,
) -> Result<[SiteTy; 3], GpuError> {
    let what = "qwen3moe::Body::load";
    let raw = super::placed::stack_files(w, names, (n, ff, h), l, what)?;
    let site_ty = |i: usize| {
        let (name, ty) = (names[i], raw[i]);
        SiteTy::of_ggml(ty).ok_or_else(|| {
            GpuError::shape(what, format!("{name} is {ty}; no launch family reads it"))
        })
    };
    let tys = [site_ty(0)?, site_ty(1)?, site_ty(2)?];
    stack_entries(tys, l)?;
    Ok(tys)
}

/// The card kernel table's entries of layer `l`'s routed stacks `tys`
/// ([`gate_up_entry`], [`down_entry`]), refused by name with the layer: the
/// one routed-type verdict, run by the load ([`stack_tys`]) on the resident
/// weights and by the whole-fit check ([`GpuModel::<Body>::whole_load`]) on
/// the file's header, so a type the load refuses never fits the plan.
fn stack_entries([gate, up, down]: [SiteTy; 3], l: usize) -> Result<(), GpuError> {
    let what = "qwen3moe::Body::load";
    let at = |e: ChainError| GpuError::shape(what, format!("layer {l}: {e}"));
    gate_up_entry(gate, up, what).map_err(at)?;
    down_entry(down, what).map_err(at)?;
    Ok(())
}

/// Weight `name` as a dense site of `rows` rows of `k` values and the launch
/// family the qwen3moe body's dense owner ([`qwen3moe_dense`]) picks for the
/// site type of its resident form ([`site::of_resident`]). A form no site
/// reads, a type the table names no dense site for, a site whose activation
/// the arena does not carry, or a weight of another shape, is refused by
/// name.
pub(super) fn dense_site(
    w: &Weights,
    name: &str,
    rows: usize,
    k: usize,
) -> Result<SiteTy, GpuError> {
    let what = "qwen3moe::Body::load";
    let Some(ty) = site::of_resident(w.resident(what, name)?) else {
        return Err(GpuError::shape(
            what,
            format!("{name} is resident in a form no site launch reads"),
        ));
    };
    let family = admit_dense(what, name, ty.ggml())?;
    site::site(w, what, name, (rows, k), family)?;
    Ok(family)
}

/// The body's dense family, or a named refusal identifying the missing arena form.
fn admit_dense(what: &'static str, name: &str, ty: GgmlType) -> Result<SiteTy, GpuError> {
    if let Some(site) = qwen3moe_dense(ty).and_then(SiteTy::of_entry) {
        return Ok(site);
    }
    let msg = match kernels::dense(ty) {
        Some(entry) => format!(
            "{name} is {ty}; its dense site reads {:?}, which this body's arena does not carry",
            entry.reads()
        ),
        None => format!("{name} is {ty}; no dense site launches it (kernels::dense)"),
    };
    Err(GpuError::shape(what, msg))
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

/// Layer `l`'s attention at head 128 and its FFN of eight routed experts,
/// every site's type the Q4_K placeholder until the load or the header sets
/// it.
fn parts_of(l: usize) -> (GqaPlan, FfnPlan) {
    let g = GqaPlan {
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
        window: None,
        gate: None,
        rope_freqs: None,
    };
    let f = FfnPlan {
        ffn_norm: names::ffn_norm(l),
        route: FfnRoute::Router {
            gate_inp: names::ffn_gate_inp(l),
            shared: None,
        },
        gate: names::ffn_gate_exps(l),
        up: names::ffn_up_exps(l),
        down: names::ffn_down_exps(l),
        gate_ty: SiteTy::Q4K,
        up_ty: SiteTy::Q4K,
        down_ty: SiteTy::Q4K,
    };
    (g, f)
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
    let (mut g, mut f) = parts_of(l);
    let router = names::ffn_gate_inp(l);
    for (name, len) in [
        (&g.attn_norm, h),
        (&f.ffn_norm, h),
        (&g.attn_q_norm, hp.head_dim),
        (&g.attn_k_norm, hp.head_dim),
    ] {
        f32_site(w, name, 1, len)?;
    }
    f32_site(w, &router, e, h)?;
    g.q_ty = dense_site(w, &g.attn_q, q, h)?;
    g.k_ty = dense_site(w, &g.attn_k, kv, h)?;
    g.v_ty = dense_site(w, &g.attn_v, kv, h)?;
    g.o_ty = dense_site(w, &g.attn_output, h, q)?;
    if !stacks {
        return Ok(LayerPlan {
            mixer: MixerPlan::Gqa(g),
            ffn: f,
            residual: ResidualPlan::Plain,
        });
    }
    [f.gate_ty, f.up_ty, f.down_ty] = stack_tys(w, [&f.gate, &f.up, &f.down], (e, ff, h), l)?;
    Ok(LayerPlan {
        mixer: MixerPlan::Gqa(g),
        ffn: f,
        residual: ResidualPlan::Plain,
    })
}

/// Layer `l`'s routed stacks' types, read from `file`'s header before any
/// upload and held to the card kernel table's entries ([`stack_entries`]).
fn header_stacks(file: &Split, hp: &Hparams, l: usize) -> Result<[SiteTy; 3], GpuError> {
    const WHAT_H: &str = "qwen3moe::Body::load";
    let (h, ff, e) = (hp.n_embd, hp.experts.ff, hp.experts.n_expert);
    let site = |name: String, shape| site::file_site(file, WHAT_H, &name, shape, SiteTy::ALL);
    let tys = [
        site(names::ffn_gate_exps(l), (e * ff, h))?,
        site(names::ffn_up_exps(l), (e * ff, h))?,
        site(names::ffn_down_exps(l), (e * h, ff))?,
    ];
    stack_entries(tys, l)?;
    Ok(tys)
}

/// Layer `l`'s routed stacks' types as a placed load's dispatch runs them,
/// read from `file`'s header before any upload: `None` when the dispatch's
/// entries run none for them (a type no site launches, or one
/// [`stack_entries`] refuses). Such a layer is the common card leg's
/// ([`model::placement::accepts::qwen35_leg`]) or the host tier's, which the
/// placed side classifies from the resident stacks (`placed`'s `n_card > 0`
/// arm), so the plan does not refuse it. A missing tensor, or a site type of another
/// shape, is refused by name.
fn card_stacks(file: &Split, hp: &Hparams, l: usize) -> Result<Option<[SiteTy; 3]>, GpuError> {
    const WHAT_H: &str = "qwen3moe::Body::load";
    let (h, ff, e) = (hp.n_embd, hp.experts.ff, hp.experts.n_expert);
    let site = |name: String, shape| -> Result<Option<SiteTy>, GpuError> {
        let (_, t) = file
            .find(&name)
            .ok_or_else(|| GpuError::tensor(WHAT_H, &name, "in the file"))?;
        if SiteTy::of_ggml(t.ty).is_none() {
            return Ok(None);
        }
        let ty = site::file_site(file, WHAT_H, &name, shape, SiteTy::ALL)?;
        Ok(Some(ty))
    };
    let (Some(gate), Some(up), Some(down)) = (
        site(names::ffn_gate_exps(l), (e * ff, h))?,
        site(names::ffn_up_exps(l), (e * ff, h))?,
        site(names::ffn_down_exps(l), (e * h, ff))?,
    ) else {
        return Ok(None);
    };
    let tys = [gate, up, down];
    Ok(stack_entries(tys, l).is_ok().then_some(tys))
}

/// Layer `l`'s plan read from `file`'s header alone: the four attention
/// sites' types and the three routed stacks' as the card part runs them
/// ([`card_stacks`]), which are the types [`resolve`] finds resident; a layer
/// whose stacks the dispatch runs none of keeps [`parts_of`]'s, which no
/// dispatch launch of its experts reads (the common leg's buffers are
/// [`CommonForms`]'s). An attention site [`qwen3moe_dense`] does not run is
/// refused by name, the missing activation form named. A placed load plans its card
/// bytes from it before any weight is uploaded
/// ([`GpuModel::placed_arena_bytes`]) and its wide arm reads the stacks'
/// types from it, the resident stacks being the card's part of them.
fn header_plan(file: &Split, hp: &Hparams, l: usize) -> Result<LayerPlan, GpuError> {
    const WHAT_H: &str = "qwen3moe::Body::load";
    let (h, q, kv) = (
        hp.n_embd,
        hp.n_head * hp.head_dim,
        hp.n_head_kv * hp.head_dim,
    );
    let site = |name: &str, shape| -> Result<SiteTy, GpuError> {
        let got = site::file_site(file, WHAT_H, name, shape, SiteTy::ALL)?;
        admit_dense(WHAT_H, name, got.ggml())
    };
    let (mut g, mut f) = parts_of(l);
    g.q_ty = site(&g.attn_q, (q, h))?;
    g.k_ty = site(&g.attn_k, (kv, h))?;
    g.v_ty = site(&g.attn_v, (kv, h))?;
    g.o_ty = site(&g.attn_output, (h, q))?;
    if let Some(tys) = card_stacks(file, hp, l)? {
        [f.gate_ty, f.up_ty, f.down_ty] = tys;
    }
    Ok(LayerPlan {
        mixer: MixerPlan::Gqa(g),
        ffn: f,
        residual: ResidualPlan::Plain,
    })
}

/// The forms the layers the common card leg runs hold buffers of
/// ([`CommonForms`]), from `file`'s header: every layer routes the family's
/// experts, so each is read for its stacks' types ([`common_forms`]), `what`
/// naming the entry point.
fn header_common_forms(
    file: &Split,
    hp: &Hparams,
    what: &'static str,
) -> Result<CommonForms, GpuError> {
    common_forms(
        file,
        what,
        (0..hp.n_layer).map(|l| (l, hp.experts.n_expert)),
        (hp.experts.ff, hp.n_embd),
    )
}

/// A placed load's plans: [`resolve`]d against the resident weights, with the
/// routed stacks' types the header's ([`header_plan`]); a layer whose
/// resident attention sites are not the header's is refused by name, so the
/// plan the card bytes were counted from is the plan the load runs.
fn placed_plans(
    file: &Split,
    w: &Weights,
    hp: &Hparams,
    layers: Range<usize>,
) -> Result<Vec<LayerPlan>, GpuError> {
    layers
        .map(|l| {
            let mut plan = resolve(w, hp, l, false)?;
            let header = header_plan(file, hp, l)?;
            let (got, want) = (
                plan.gqa("qwen3moe::Body::load", l)?,
                header.gqa("qwen3moe::Body::load", l)?,
            );
            let tys = |g: &GqaPlan| [g.q_ty, g.k_ty, g.v_ty, g.o_ty];
            if tys(got) != tys(want) {
                return Err(GpuError::shape(
                    "qwen3moe::Body::load",
                    format!(
                        "layer {l}'s attention sites are resident as {:?}, the file's header \
                         holds {:?}",
                        tys(got),
                        tys(want)
                    ),
                ));
            }
            [plan.ffn.gate_ty, plan.ffn.up_ty, plan.ffn.down_ty] =
                [header.ffn.gate_ty, header.ffn.up_ty, header.ffn.down_ty];
            Ok(plan)
        })
        .collect()
}

/// Whether a launch of the chain over `plans` and the token embedding
/// resident in `w` reaches [`Kernels::q35`]: an attention site that is not
/// one of the fused Q4_K groups (`dispatch::site_gemv`: its gemv, and on the
/// prompt's wide arm its GEMM and quantizers), or a Q8_0 embedding
/// (`dispatch::embed_into`), or a routed stack whose launch is the K-quant
/// `_sel` family's ([`FfnPlan::kq_sel`]: [`Q35Kernels::ffn`]; a placed load's
/// routed launches are its placed side's). A chain of fused Q4_K and Q6_K
/// sites over K-quant rows and Q4_K or Q6_K stacks launches none of them.
fn needs_site_kernels(w: &Weights, plans: &[LayerPlan], placed: bool) -> bool {
    let fused = plans
        .iter()
        .all(|p| matches!(&p.mixer, MixerPlan::Gqa(g) if qkv_fused(g) && o_fused(g)));
    !fused
        || (!placed && plans.iter().any(|p| kq_sel(&p.ffn)))
        || matches!(w.get(&token_embd()), Some(DevWeight::Q8_0 { .. }))
}

impl Body {
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
    /// The cache format selected by a layout's q8_0 bit.
    #[must_use]
    pub const fn of(q8: bool) -> KvQ8 {
        if q8 { KvQ8::Q8 } else { KvQ8::F16 }
    }

    /// The flag's word, else the lever's, else the caller's default; any other word is refused
    /// by name before the load.
    pub fn choose(
        flag: Option<&str>,
        lever: Option<&str>,
        default: KvQ8,
    ) -> Result<KvQ8, GpuError> {
        let word = flag.or(lever).unwrap_or(default.name());
        KvQ8::parse(word).ok_or_else(|| {
            GpuError::shape(
                "KvQ8::choose",
                format!("--cache-type-k takes f16 or q8_0, not {word:?}"),
            )
        })
    }

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
    /// the arena's range is refused at load. A placed load's ubatch is the
    /// placed units' rows, whatever this asks ([`GpuModel::open_placed`]).
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
    /// `model::placement::workstation::placed_with_arena` lays out): the plan's card
    /// segments resident ([`GpuModel::load_placed`]: the trunk and each
    /// layer's card experts), the host set read in and locked as `host`
    /// asks, and the body over them with its placed side ([`Placed`]) and
    /// caches of `opts.ctx` rows, the plan's context. Its prompt runs as
    /// eager units through the host tier's batch port — passes of up to
    /// [`crate::model::MAX_PASS_ROWS`] rows, or wide units of up to [`PLACED_ROWS`] — so its
    /// ubatch arena holds that many rows whatever `opts.ubatch` asks, held to
    /// the bytes the plan counts ([`GpuModel::placed_arena_bytes`]). Refused
    /// by name as [`GpuModel::open`] refuses, for a plan of another context,
    /// and as the placed side refuses ([`Placed::new`]).
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
        let counted = model::placement::workstation::counted_arena_bytes(card);
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
                    glue: ResidencyGlue::off(),
                    arch: "qwen3moe",
                    family: &super::tier_leg::QWEN3_FAMILY,
                    rows: PLACED_ROWS,
                };
                Body::load(
                    gpu,
                    &file,
                    w,
                    0..n_layers,
                    opts.ctx,
                    PLACED_ROWS,
                    mma,
                    opts.kv,
                    Some((open, counted)),
                )
            },
        )
    }

    /// The card bytes a placed load of `file` holds past the m = 1 scratch,
    /// at caches of `ctx` rows: its ubatch arena at the placed units' rows
    /// and its placed side ([`placed_scratch`]) — what the plan's machine
    /// counts in the card's scratch (`model::placement::workstation::placed_with_arena`),
    /// and what [`GpuModel::open_placed`] refuses to pass. The arena's forms
    /// are read from the file's header plan ([`header_plan`]), the plan the
    /// load runs, and the common leg's from the stacks' header types
    /// ([`header_common_forms`]).
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
        let plans = (0..hp.n_layer)
            .map(|l| header_plan(file, &hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        let common = header_common_forms(file, &hp, what)?;
        placed_scratch(&dims, plans.len(), (Forms::of(&plans, &dims), common))
    }

    /// What a whole load of `file` ([`GpuModel::open`]) holds past its
    /// weights, stores and step arenas: `WholeLoad::Unchecked` — the body
    /// allocates its ubatch arena with no fit check, and the arena's bytes
    /// have no formula outside the allocation, so the card keeps the expert
    /// rule's margin past the step arenas in its place, as the placed
    /// machine does. Past the load's own per-layer routed-type check
    /// ([`stack_entries`] on the stacks' header types): a file the load
    /// refuses is refused here by the same name.
    pub fn whole_load(file: &Split) -> Result<WholeLoad, GpuError> {
        const WHAT: &str = "qwen3moe::Body::load";
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        for l in 0..hp.n_layer {
            header_stacks(file, &hp, l)?;
        }
        Ok(WholeLoad::Unchecked)
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
    /// placed side's inputs and the bytes its plan counts for the ubatch arena
    /// and that side) the routed stacks are the placed side's to check, their
    /// types in the plans the header's ([`placed_plans`]), the arena held to
    /// the plan's count with the side's bytes ([`counted_arena`]) and the side
    /// built ([`Placed::new`]) and held to those bytes ([`Placed::within`]).
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
        let plans = if placed.is_some() {
            placed_plans(file, w, &hp, layers.clone())?
        } else {
            layers
                .clone()
                .map(|l| resolve(w, &hp, l, true))
                .collect::<Result<Vec<_>, _>>()?
        };
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
        let k = Kernels::load(gpu, needs_site_kernels(w, &plans, placed.is_some()))?;
        let table = RopeTable::new(&RopeSpec::window(hp.rope.base, hp.rope.dims))?;
        let rope = RopeRows::new(stream, &table, HEAD, dims.ctx)?;
        let forms = Forms::of(&plans, &dims);
        let (ub, placed) = match placed {
            None => (Ubatch::new(stream, dims, ubatch, Q32::of(&plans))?, None),
            Some((open, counted)) => {
                refuse_stage_rows_past(what, open.plan, 0, rope.table.num_bytes() as u64)?;
                let common = header_common_forms(file, &hp, what)?;
                let side = placed_bytes(&dims, plans.len(), open.rows, common)?;
                let arena = counted_arena(stream, (dims, forms), ubatch, counted, side)?;
                let ub = Ubatch::placed(stream, arena, ubatch)?;
                let built = Placed::new(gpu, w, &plans, &dims, open)?;
                built.within(side)?;
                (ub, Some(built))
            }
        };
        Ok(Body {
            prefill: Prefill::new(stream, dims, forms)?,
            ub,
            rope,
            s: Arena::new(stream, dims, 1)?,
            sp: StepParams::new(stream, false)?,
            slot_in: SlotIn::new(stream)?,
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
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

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
            + self.slot_in.bytes()
            + self.head_state.bytes()
            + self.prefill.bytes()
            + self.ub.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
            + self.k.q35.as_ref().map_or(0, |k| k.kgemv.device_bytes())
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
        let kv = first.format();
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

impl SlotRows for Body {
    /// The pass runs on the prompt arena and its input record holds this
    /// many rows ([`SlotIn`]).
    const MAX_ROWS: usize = crate::model::MAX_PASS_ROWS;

    /// One head of every row: the pass's rows read the lm_head once
    /// ([`RowHeads::One`]).
    const HEADS: RowHeads = RowHeads::One;

    /// The chain the placed replay serves ([`slot_chain`]). A whole-card
    /// load has no host service of the pass.
    fn chain_of(rows: &[SlotRange]) -> crate::hybrid::Chain {
        slot_chain(rows)
    }

    /// Each busy slot's input record, in one copy ([`SlotIn`]).
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        _parked: &mut [&mut Seq],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        self.slot_in.write(stream, rows, ids)
    }

    /// The pass on the prompt arena. A whole-card load is [`SlotPass`]. A
    /// placed load is the same attention, then the placed FFN over the
    /// rows ([`enqueue_placed_slots`]). The live planes are slot 0's, each
    /// parked sequence's its own.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Seq],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        if self.placed.is_some() {
            return enqueue_placed_slots(gpu, w, self, heads, parked, rows);
        }
        let Body {
            hp,
            plans,
            kv,
            rope,
            prefill,
            slot_in,
            k,
            head_state,
            mma,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: hp.rms_eps,
            table: &rope.table,
        };
        let mut planes: Vec<&mut [KvPlanes]> =
            parked.iter_mut().map(|seq| seq.kv.as_mut_slice()).collect();
        SlotPass {
            c: &c,
            live: kv,
            parked: &mut planes,
            rows,
            ins: slot_in,
            s: &mut prefill.a,
            heads,
            state: head_state,
        }
        .enqueue()
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

/// What a placed pass of several slots' refusals name.
const WHAT_PASS: &str = "qwen3moe placed slots";

/// The K/V planes a placed pass's sequence-bound launches read and write:
/// one busy slot's, or every busy slot's through the per-row table.
enum PassPlanes<'k, 't> {
    Slot(&'k mut KvPlanes),
    Rows(&'k RowTable<'t>),
}

/// Layer `layer`'s attention of a placed pass: the whole-card pass's
/// launches ([`SlotPass`]), over the live planes and the parked sequences'
/// planes, then the placed FFN ([`ColsWalk`]).
struct Mix30<'a, 'p> {
    live: &'a mut [KvPlanes],
    parked: &'a mut [&'p mut [KvPlanes]],
    rows: &'a [SlotRange],
    per_row: bool,
    total: usize,
}

impl ColsMix for Mix30<'_, '_> {
    fn mix_layer(&mut self, layer: usize, c: &PassCtx<'_>, s: &mut Arena) -> Result<(), GpuError> {
        let p = c
            .plans
            .get(layer)
            .ok_or(GpuError::state(WHAT_PASS, "a plan for every layer"))?;
        let lc = Ctx::new(c.gpu, c.w, (p, layer), c.k, c.mma, c.eps, c.table)?;
        let g = match &p.mixer {
            MixerPlan::Gqa(g) if g.kind == GqaKind::Neox128 => g,
            _ => {
                return Err(GpuError::shape(
                    WHAT_PASS,
                    format!(
                        "layer {layer} is not head-128 attention over K/V planes, the one mixer \
                         a pass of slots binds per slot"
                    ),
                ));
            }
        };
        dispatch::attn_in(&lc, g, s, self.total)?;
        if self.per_row {
            let mut table = RowTable::default();
            for slot in pass_planes(self.live, self.parked, self.rows, layer) {
                let (r, kv) = slot?;
                kv.put_rows(&mut table, r.rows.clone())?;
            }
            pass_attention(&lc, g, PassPlanes::Rows(&table), s, &(0..self.total))?;
        } else {
            for slot in pass_planes(self.live, self.parked, self.rows, layer) {
                let (r, kv) = slot?;
                pass_attention(&lc, g, PassPlanes::Slot(kv), s, &r.rows)?;
            }
        }
        let i = s.col(self.total)?;
        c.gpu
            .enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], layer)?;
        dispatch::attn_out(&lc, g, s, self.total)
    }
}

/// Enqueue a placed load's pass of `rows` on the prompt arena: each slot's
/// embedding, then [`ColsWalk`] over the pass rows' places, into one head
/// of every row.
fn enqueue_placed_slots(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    heads: &mut [Head],
    parked: &mut [&mut Seq],
    rows: &[SlotRange],
) -> Result<(), GpuError> {
    let [head] = heads else {
        return Err(GpuError::shape(
            WHAT_PASS,
            format!(
                "{} heads; a pass of several slots runs into one head of every row",
                heads.len()
            ),
        ));
    };
    let Body {
        hp,
        plans,
        kv,
        rope,
        prefill,
        slot_in,
        k,
        head_state,
        mma,
        placed,
        ..
    } = b;
    let Some(Placed {
        hybrid, side, pass, ..
    }) = placed
    else {
        return Err(GpuError::state(WHAT_PASS, "a placed side (a placed load)"));
    };
    let mut planes: Vec<&mut [KvPlanes]> =
        parked.iter_mut().map(|seq| seq.kv.as_mut_slice()).collect();
    let total = pass_check(rows, prefill.a.rows.min(GEMV_COLS), 1, planes.len())?;
    let per_row = rows.len() > 1 && kv.first().is_some_and(|p| p.rows_serve(*mma));
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: hp.rms_eps,
        table: &rope.table,
    };
    let hidden = prefill.a.dims.hidden;
    for (j, r) in rows.iter().enumerate() {
        let rec = slot_in.io(j, r)?;
        let Arena { x, pos, n_keys, .. } = &mut prefill.a;
        let mut x = pass_rows_mut(x, &r.rows, hidden)?;
        let mut pos = pass_rows_mut(pos, &r.rows, 1)?;
        let mut n_keys = pass_rows_mut(n_keys, &r.rows, 1)?;
        dispatch::embed_into(c.gpu, c.w, c.k, &rec.io(), (&mut x, &mut pos, &mut n_keys))?;
    }
    let mut leg = StepLeg::new(gpu.stream(), hybrid);
    ColsWalk {
        c: &c,
        s: &mut prefill.a,
        m: total,
        side,
        rows: pass,
        mix: Mix30 {
            live: kv,
            parked: &mut planes,
            rows,
            per_row,
            total,
        },
        tail: Tail::One {
            head,
            state: head_state,
        },
    }
    .walk(&mut leg)
}

/// The pass's rows `R`, once `rows` tile `0..R` in order with no empty
/// range, `R` is at most `most`, and the head is one with the parked
/// sequences as many as the ranges past slot 0; else refused by name.
fn pass_check(
    rows: &[SlotRange],
    most: usize,
    heads: usize,
    parked: usize,
) -> Result<usize, GpuError> {
    let mut at = 0;
    for r in rows {
        if r.rows.start != at || r.rows.is_empty() {
            return Err(GpuError::shape(
                WHAT_PASS,
                format!(
                    "slot {}'s rows {:?}; the ranges tile the pass in order from row 0",
                    r.slot, r.rows
                ),
            ));
        }
        at = r.rows.end;
    }
    let others = rows.iter().filter(|r| r.slot != 0).count();
    if at == 0 || at > most || heads != 1 || parked != others {
        return Err(GpuError::shape(
            WHAT_PASS,
            format!(
                "a pass of {at} rows (1..={most}) with {heads} heads and {parked} parked \
                 sequences for {others} slots past slot 0; the pass runs into one head of \
                 every row"
            ),
        ));
    }
    Ok(at)
}

/// Each busy slot of `rows` with its planes of layer `l`, in pass order:
/// slot 0's are the live ones, every other slot's the next parked
/// sequence's; a slot whose planes are missing is refused by name.
fn pass_planes<'a>(
    live: &'a mut [KvPlanes],
    parked: &'a mut [&mut [KvPlanes]],
    rows: &'a [SlotRange],
    l: usize,
) -> impl Iterator<Item = Result<(&'a SlotRange, &'a mut KvPlanes), GpuError>> {
    let mut live = Some(live);
    let mut others = parked.iter_mut();
    rows.iter().map(move |r| {
        let planes = if r.slot == 0 {
            live.take().and_then(|p| p.get_mut(l))
        } else {
            others.next().and_then(|p| p.get_mut(l))
        };
        planes.map(|kv| (r, kv)).ok_or_else(|| {
            GpuError::shape(WHAT_PASS, format!("slot {}'s planes of layer {l}", r.slot))
        })
    })
}

/// Rows `r` of `parent`, `width` values a row, shared.
fn pass_rows<'a, T>(
    parent: &'a DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<Window<'a, T>, GpuError> {
    Window::of(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// Rows `r` of `parent`, `width` values a row, to write through.
fn pass_rows_mut<'a, T>(
    parent: &'a mut DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<WindowMut<'a, T>, GpuError> {
    WindowMut::of_mut(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// The sequence-bound attention of rows `r` of the arena over `planes`: the
/// head norm and rope with the cache append, then the flash. One slot's
/// launches over its row window, or the per-row launches over all the
/// pass's rows.
fn pass_attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    mut planes: PassPlanes<'_, '_>,
    s: &mut Arena,
    r: &Range<usize>,
) -> Result<(), GpuError> {
    let d = s.dims;
    let m = r.len();
    let stream = c.gpu.stream();
    let Arena {
        q,
        k,
        v,
        pos,
        n_keys,
        part_v,
        part_ms,
        attn,
        ..
    } = s;
    let mut q = pass_rows_mut(q, r, d.q_rows)?;
    let mut k = pass_rows_mut(k, r, d.kv_len())?;
    let v = pass_rows(v, r, d.kv_len())?;
    let pos = pass_rows(pos, r, 1)?;
    let append = Append128 {
        q: &mut q,
        k: &mut k,
        v: &v,
        gq: f32_gain(c.w, &n.attn_q_norm)?,
        gk: f32_gain(c.w, &n.attn_k_norm)?,
        table: c.table,
        pos: &pos,
        eps: c.eps,
        n_head: d.n_head,
        n_kv: d.n_kv,
        ctx: d.ctx,
        m,
        fault: c.sink,
    };
    match &mut planes {
        // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint planes.
        PassPlanes::Slot(kv) => unsafe { kv.append_128(&c.k.neox, stream, append) }?,
        PassPlanes::Rows(table) => {
            // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint
            // planes.
            unsafe { KvPlanes::append_128_rows(&c.k.neox, stream, table, append) }?
        }
    }
    let n_keys = pass_rows(n_keys, r, 1)?;
    let mut part_v = pass_rows_mut(part_v, r, partials_v_len(1, d.n_head))?;
    let mut part_ms = pass_rows_mut(part_ms, r, partials_ms_len(1, d.n_head))?;
    let mut y = pass_rows_mut(attn, r, d.attn_len())?;
    let flash = FlashPass {
        q: &q,
        n_keys: &n_keys,
        scale: ATTN_SCALE,
        n_kv: d.n_kv,
        ctx: d.ctx,
        m,
        part_v: &mut part_v,
        part_ms: &mut part_ms,
        fault: c.sink,
        y: &mut y,
    };
    match planes {
        PassPlanes::Slot(kv) => kv.flash_128(&c.k.flash, stream, flash, c.mma),
        PassPlanes::Rows(table) => {
            KvPlanes::flash_128_rows(&c.k.flash, stream, table, flash, c.mma)
        }
    }
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
            cells_view(
                &rope.table,
                positions.start * rope.width,
                positions.len() * rope.width,
            )
        };
        Ok(rows.to_host_vec(self.stage_stream()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::KvQ8;
    use crate::GpuError;
    use crate::site::SiteTy;
    use gguf::quant::GgmlType;
    use model::placement::accepts::qwen3moe_dense;
    use model::placement::kernels;

    /// The KV cache's format: `--cache-type-k`'s word wins over the lever's, the lever's over the
    /// default, and unset a checkpoint folder runs q8_0 and a GGUF file f16; any other word is
    /// refused by name, from the flag or the lever.
    #[test]
    fn the_cache_format_is_the_flags_then_the_levers_then_the_folders() {
        let k = |flag, lever, folder| {
            KvQ8::choose(flag, lever, KvQ8::of(folder)).expect("a cache word")
        };
        assert_eq!(k(None, None, true), KvQ8::Q8);
        assert_eq!(k(None, None, false), KvQ8::F16);
        assert_eq!(k(Some("f16"), None, true), KvQ8::F16);
        assert_eq!(k(Some("q8_0"), None, false), KvQ8::Q8);
        assert_eq!(k(None, Some("f16"), true), KvQ8::F16);
        assert_eq!(k(Some("q8_0"), Some("f16"), false), KvQ8::Q8);
        assert_eq!(k(Some("f16"), Some("fp8"), true), KvQ8::F16);
        for (flag, lever) in [
            (Some("q8"), None),
            (None, Some("fp8")),
            (Some("q4"), Some("f16")),
            (Some(""), None),
        ] {
            let e: GpuError =
                KvQ8::choose(flag, lever, KvQ8::Q8).expect_err("a word the flag does not take");
            assert!(matches!(
                &e,
                GpuError::Shape {
                    what: "KvQ8::choose",
                    ..
                }
            ));
            assert!(
                e.to_string().contains("--cache-type-k takes f16 or q8_0"),
                "{e}"
            );
        }
    }

    /// A dense entry the qwen3moe body runs is the launch family of its file
    /// type: [`qwen3moe_dense`] maps the type to the entry,
    /// `SiteTy::of_entry` to the family `site.rs` names, and `SiteTy::ggml`
    /// back. A type the owner does not return has no site launch (Q4_0 has no
    /// table row), unless the table row reads an activation the arena does not
    /// carry (Q5_0).
    #[test]
    fn a_dense_entry_is_the_launch_family_of_its_file_type() {
        let mut named = 0;
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::BF16,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::MXFP8,
            GgmlType::FP8_B128,
            GgmlType::IQ3_XXS,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ] {
            let Some(entry) = qwen3moe_dense(ty) else {
                if kernels::dense(ty).is_none() {
                    assert_eq!(
                        SiteTy::of_ggml(ty),
                        None,
                        "{ty}: a launch with no table row"
                    );
                }
                continue;
            };
            named += 1;
            let site = SiteTy::of_entry(entry).expect("a family launches the entry");
            assert_eq!(site.ggml(), ty, "{ty}");
            assert_eq!(SiteTy::of_ggml(ty), Some(site), "{ty}");
        }
        assert!(named >= 6, "the body runs {named} of the listed types");
    }
}
