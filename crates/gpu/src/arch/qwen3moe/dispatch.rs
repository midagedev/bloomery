//! The launch groups of a layer's ops, each enqueued asynchronously against a
//! resident arena. No allocation, no synchronization, no host round trip —
//! so the same enqueues are the eager body and what the capture records. The
//! chain crate's walk orders the ops (its doc names what it does not lay
//! out); [`Program`] and the passes of several slots run them. One op serves
//! every pass and both bodies of the family: the decode step is its one-row
//! instance, a pass its `m`-row instance, and a layer's [`LayerPlan`] picks
//! its mixer — attention over K/V planes, or the delta
//! rule over a recurrent store ([`super::delta`]) — and whether its experts
//! carry a folded shared expert. Which kernel an op launches follows from
//! `m` and the plan inside the op: past [`GEMV_COLS`] rows its first line
//! hands the unit to its wide arm (`wide`: the grouped GEMM, the prefill
//! flash, the router's two launches), and a unit's embedding is followed by
//! its dense route table. At one row the FFN norm is folded into
//! the router's launch, which writes the bytes of the pair it replaces, so a
//! one-row pass and a multi-row one leave the same state. The q8_1 of the
//! attention rows and of the SwiGLU rows stay launches of their own: folded
//! into the projection beside them, the work lands in every block of a grid
//! that already streams its weights at the card's bandwidth.

use super::body::{ATTN_SCALE, ATTN_SCALE_256, Body, Kernels};
use super::delta;
use super::experts::{CombineArgs, GateUpArgs};
use super::head_argmax::HeadArgmaxState;
use super::hostleg::{HostCombineArgs, HostLegKernels};
use super::plan::{
    DownEntry, FfnPlan, FfnRoute, GateUpEntry, GqaKind, GqaPlan, LayerPlan, MixerPlan, Pairs,
    SharedPlan, SharedSites, SiteTy, down_entry, down_sel, gate_up_entry, gate_up_fused, o_fused,
    qkv_fused, reads_f32,
};
use super::program::{Program, Tail};
use super::proj::{OResidArgs, QkvArgs};
use super::scratch::{Append128, Append256, AppendNorm, Arena, FlashPass, Io, KvPlanes};
use super::wide::{self, GEMV_COLS};
use crate::elem::EmbedRowsArgs;
use crate::gated_quant::GateLayout;
use crate::head::Head;
use crate::kquant::{Act, GateUpAct, SelDown};
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::q4k_sel::QuantSel;
use crate::q38::{EmbedArgs, EmbedRows, OutGateArgs};
use crate::quant_sel::QuantSelKernels;
use crate::site::{self, Order};
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, HEAD_TENSOR, Weights};
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::quant::GgmlType;
use model::arch::qwen3moe::names::token_embd;

/// What every launch of one layer reads besides the arena and the store:
/// the engine, the weights, the layer's plan, the kernels, the flash pass,
/// the norm epsilon, the rope table, and the layer's index with its fault
/// sink — every launch of the layer that can refuse its input raises with
/// that index.
pub(super) struct Ctx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) p: &'a LayerPlan,
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) layer: usize,
    pub(super) sink: FaultSink,
}

impl<'a> Ctx<'a> {
    /// Layer `layer`'s context, its sink made from its index.
    pub(super) fn new(
        gpu: &'a Gpu,
        w: &'a Weights,
        (p, layer): (&'a LayerPlan, usize),
        k: &'a Kernels,
        mma: bool,
        eps: f32,
        table: &'a DeviceBuffer<f32>,
    ) -> Result<Ctx<'a>, GpuError> {
        Ok(Ctx {
            gpu,
            w,
            p,
            k,
            mma,
            eps,
            table,
            layer,
            sink: gpu.layer_sink(layer)?,
        })
    }
}

/// Enqueue the whole decode chain at m = 1: layer 0 with its embedding,
/// every later layer reading the previous layer's output, then the head —
/// the walk `(1, 1, Step)` ([`Program`]). The combine writes a layer's
/// output where the next reader takes it — the arena's `x` for the next
/// layer, the head's input after the last — so no copy crosses a layer
/// boundary.
pub(super) fn enqueue_chain(
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
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: hp.rms_eps,
        table: &rope.table,
    };
    Program {
        c: &c,
        stores: kv.as_mut_slice(),
        s,
        io: &sp.io(),
        m: 1,
        tail: Tail::Step {
            head,
            state: head_state,
            taps: taps.as_mut(),
        },
    }
    .walk()
}

/// Enqueue the head: a head of one row under a Q6_K `output.weight` runs its
/// norm and quantization, then the Q6_K projection with the argmax folded in
/// (`head_argmax`) in place of the shared head's gemv and `argmax_fault` —
/// the same logits and the same (token, fault word) readback. Every other
/// head — another weight type (the type the load read from the file), or
/// more rows than one (the fused projection reads a single activation
/// column, `head_argmax::enqueue` refuses it) — runs the shared head's own
/// arm ([`Head::enqueue`]): a Q8_0 one-row head the norm, the q8f32 gemv and
/// the argmax, three launches as the fused head's; a Q4_K one-row one the
/// norm, the quantizer, the Q4_K gemv and the argmax; a head of `m` rows
/// those launches over its rows, its argmax `argmax_rows_fault`.
pub(super) fn enqueue_head(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    head: &mut Head,
) -> Result<(), GpuError> {
    if head.m() > 1
        || !matches!(
            w.get(HEAD_TENSOR),
            Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                ..
            })
        )
    {
        return head.enqueue(gpu, w);
    }
    let stream = gpu.stream();
    head.enqueue_with_tail(gpu, w, |act, out_w, fault, logits, out| {
        k.head
            .enqueue(stream, out_w, act, fault, logits, state, out)
    })
}

/// Enqueue the step's front alone: the embedding row of the step's token
/// into the one-row arena's `x`, and its position and live key count. A
/// layer run on its own from another input (`taps::step_layer`) needs the
/// position the front writes; the caller then replaces `x`.
pub(super) fn enqueue_front(gpu: &Gpu, w: &Weights, b: &mut Body) -> Result<(), GpuError> {
    let Body { s, sp, k, .. } = b;
    embed_rows(gpu, w, k, &sp.io(), s)
}

/// What every launch of a prefill pass reads besides its arena, its inputs
/// and the cache: the engine, the weights, every layer's plan, the kernels,
/// the flash pass, the norm epsilon and the rope table.
pub(super) struct PassCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) plans: &'a [LayerPlan],
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
}

/// Enqueue one prefill pass of `m` tokens over arena `s` from the input
/// record windows in `io`: the embedding rows with the rows' positions and
/// live key counts, then every layer at `m` rows, the combine leaving each
/// layer's output in `x` — the walk `(1, m, Step)` ([`Program`]). The same
/// enqueues run eager and under a capture.
pub(super) fn enqueue_pass(
    c: &PassCtx<'_>,
    kv: &mut [KvPlanes],
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    Program {
        c,
        stores: kv,
        s,
        io,
        m,
        tail: Tail::Pass,
    }
    .walk()
}

/// The launches a pass of `m` rows makes over layers `plans`, counted from
/// the ops' enqueues in the walk's order: the embedding, then per layer the
/// mixer and the FFN half.
/// - Attention: norm+quant, q·k·v, QK-norm+rope+append, the flash's segment
///   pass and merge, the attention rows' quantizer (with the output gate at
///   head 256) and the output projection — 7 — plus a Q6_K value
///   projection's gemv, and at more than one row its token-major copy. With
///   q·k·v of other types each projection launches alone
///   ([`site_launches`]); an output projection not of Q4_K is its
///   quantizer (a K-quant) or the f32 out gate (any other), the projection
///   and the residual add. At head 128 a Q8_0 or F32 output projection has
///   neither: the flash's f32 rows are its input.
/// - Delta rule ([`delta::launches`]): norm+quant, the two projection
///   launches, conv, delta, gated norm, quantizer and output projection —
///   8 — plus a Q6_K q·k·v projection's token-major copy at more than one
///   row.
/// - FFN: the norm with the router (one launch), gate·up, one quantizer
///   over every token's slots, the down `_sel` over every token's slots,
///   and the combine — 5.
///
/// - A dense FFN: `norm_quant` and the four launches after the router — 5;
///   an unfused gate·up is the gate, the up and the SwiGLU; a Q8_0 down is
///   one launch with no quantizer, a Q3_K, Q5_K or NVFP4 one the quantizer
///   and its projection ([`reads_q8_1`], [`site_launches`]).
/// - A routed FFN whose shared expert is kept apart ([`apart_down`]): the
///   norm with the router (`norm_quant` and the router, two, when the
///   shared gate or up reads f32 rows), the card list, the routed gate·up,
///   its quantizer and down, the shared gate and up, the SwiGLU, a K-quant
///   shared down's quantizer, the shared down, and the combine — 11 of
///   Q8_0 or of K-quant shared sites at one row.
///
/// So a qwen3moe layer is 12 or 13 launches at one row and 12 or 14 at
/// every `m` from two to [`GEMV_COLS`]. Past it (the wide arm, `wide`) the
/// embedding is followed by the unit's dense route table, and a layer is
/// its mixer — the gated attention's norm, quantizer, three GEMMs, rope,
/// prefill flash, gated quantizer, output GEMM and residual add, 10; the
/// delta rule's 12 ([`delta::launches`]) — and the FFN's norm, quantizer,
/// router logits and routing, route table, gate, up, SwiGLU quantizer, down
/// and combine, 10; with the shared expert kept apart, the norm, the
/// router's two, the card list, the routed table, the routed four, the
/// shared gate, up, SwiGLU (with its quantizer) and down, and the combine,
/// 14, beside the input's quantizers; a dense FFN's norm, quantizer, gate,
/// up, SwiGLU quantizer, down and combine, 7. A wide input is quantized once
/// for each form its sites read ([`wide_quants`]: none for F32 sites alone, two for
/// a K-quant beside a Q8_0 or an NVFP4), and a gated output projection not of a K-quant
/// takes the f32 out gate and its own quantizer in place of the gated one.
pub(super) fn pass_launches(plans: &[LayerPlan], m: usize) -> usize {
    let wide = m > GEMV_COLS;
    let many = usize::from(m > 1);
    let site = |t: SiteTy| site_launches(t, m);
    let per_layer = |p: &LayerPlan| {
        let mixer = match &p.mixer {
            MixerPlan::Gqa(g) if wide => {
                let out = if g.o_ty.kquant() {
                    1
                } else {
                    1 + wide_quants(&[g.o_ty])
                };
                1 + wide_quants(&[g.q_ty, g.k_ty, g.v_ty]) + 3 + 2 + out + 2
            }
            MixerPlan::Gqa(g) => {
                let qkv = if qkv_fused(g) {
                    1 + usize::from(g.v_ty == SiteTy::Q6K) * (1 + many)
                } else {
                    site(g.q_ty) + site(g.k_ty) + site(g.v_ty)
                };
                let out = if o_fused(g) {
                    2
                } else {
                    // The quantizer of the attention rows (a K-quant) or the
                    // f32 out gate; head 128 has no out gate.
                    let rows = usize::from(g.kind == GqaKind::Gated256 || g.o_ty.kquant());
                    rows + out_resid_launches(g.o_ty, m)
                };
                1 + qkv + 3 + out
            }
            MixerPlan::Delta(d) => delta::launches(d, m),
            MixerPlan::ShortConv(_) => {
                panic!(
                    "qwen3moe::pass_launches: a short-conv layer, which the card's plans never hold"
                )
            }
        };
        let f = &p.ffn;
        let ffn = match (&f.route, wide) {
            (
                FfnRoute::Router {
                    shared: Some(SharedPlan::Apart(sh)),
                    ..
                },
                true,
            ) => 1 + wide_quants(&[f.gate_ty, f.up_ty, sh.gate_ty, sh.up_ty]) + 2 + 2 + 4 + 4 + 1,
            (
                FfnRoute::Router {
                    shared: Some(SharedPlan::Apart(sh)),
                    ..
                },
                false,
            ) => {
                1 + usize::from(reads_f32(sh))
                    + 4
                    + site(sh.gate_ty)
                    + site(sh.up_ty)
                    + 1
                    + usize::from(reads_q8_1(sh.down_ty))
                    + site(sh.down_ty)
                    + 1
            }
            (FfnRoute::Router { .. }, true) => 10,
            (FfnRoute::Router { .. }, false) => 5,
            (FfnRoute::Dense, true) => 1 + wide_quants(&[f.gate_ty, f.up_ty]) + 5,
            (FfnRoute::Dense, false) => {
                let gate_up = if gate_up_fused(f) {
                    1
                } else {
                    site(f.gate_ty) + site(f.up_ty) + 1
                };
                let down = if down_sel(f) {
                    2
                } else {
                    usize::from(reads_q8_1(f.down_ty)) + site(f.down_ty)
                };
                2 + gate_up + down
            }
        };
        mixer + ffn
    };
    1 + usize::from(wide) + plans.iter().map(per_layer).sum::<usize>()
}

/// The quantizer launches a wide input takes for sites of types `tys`: one
/// for each of the q8_1-of-128 and q8-of-32 forms one of them reads.
pub(super) fn wide_quants(tys: &[SiteTy]) -> usize {
    let want = crate::site::WideActs::wants_of(tys);
    usize::from(want.q128) + usize::from(want.q32)
}

/// A unit's front: its embedding rows with their positions and live key
/// counts ([`embed_rows`]), a media call's rows over the embeddings of their
/// positions, and past [`GEMV_COLS`] rows the unit's dense route table.
pub(super) fn unit_front(
    c: &Ctx<'_>,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    embed_rows(c.gpu, c.w, c.k, io, s)?;
    s.write_media(c.gpu.stream())?;
    if m > GEMV_COLS {
        wide::route_dense(c, s, m)?;
    }
    Ok(())
}

/// The refusal of layer `layer`'s plan and store being of two kinds.
pub(super) fn two_kinds(layer: usize) -> GpuError {
    GpuError::shape(
        "qwen3moe::layer",
        format!(
            "layer {layer}'s plan and store are of two kinds (attention over K/V planes, the \
             delta rule over a recurrent store)"
        ),
    )
}

/// A unit's front, one launch: the embedding rows of `io`'s ids into `s.x`
/// and each row's position and live key count into `s.pos` and `s.n_keys` —
/// the rows every later launch of the unit reads. The table's resident type
/// picks the lookup: Q3_K, Q4_K, Q5_K or Q6_K rows, or Q8_0 planes (the load
/// admits these). Q5_0 packed rows are served here for a body that admits them.
pub(super) fn embed_rows(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    io: &Io<'_>,
    s: &mut Arena,
) -> Result<(), GpuError> {
    embed_into(gpu, w, k, io, (&mut s.x, &mut s.pos, &mut s.n_keys))
}

/// [`embed_rows`] into `(y, pos, n_keys)`, a unit's embedding rows and
/// their positions and live key counts: the arena's own, or a window of
/// them where a pass of several slots embeds each slot's rows apart. Q5_0
/// packed rows take [`crate::elem::ElemKernels::enqueue_embed_rows_q5_0`]
/// for a body that admits them.
pub(super) fn embed_into(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    io: &Io<'_>,
    (y, pos, n_keys): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<u32>,
        &mut DeviceBuffer<u32>,
    ),
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::embed_rows";
    let name = token_embd();
    let table = match w.get(&name) {
        Some(DevWeight::Q8_0 { qs, d, .. }) => Some(EmbedRows::Q8_0 { qs, d }),
        Some(DevWeight::Bf16 { w, .. }) => Some(EmbedRows::Bf16(w)),
        _ => None,
    };
    if let Some(table) = table {
        return k.q35(WHAT)?.q38.enqueue_embed_rows(
            gpu.stream(),
            EmbedArgs {
                table,
                ids: io.ids,
                pos0: io.pos0,
                first: io.first,
                fault: gpu.unlabelled_sink(),
                y,
                pos,
                n_keys,
            },
        );
    }
    if let Some(DevWeight::Q5_0 { w, k }) = w.get(&name) {
        return gpu.elem().enqueue_embed_rows_q5_0(
            gpu.stream(),
            EmbedRowsArgs {
                w,
                ids: io.ids,
                pos0: io.pos0,
                first: io.first,
                y,
                pos,
                n_keys,
            },
            *k,
        );
    }
    let ty = match w.get(&name) {
        Some(DevWeight::KQuant { ty, .. }) => *ty,
        _ => {
            return Err(GpuError::tensor(
                WHAT,
                &name,
                "K-quant rows, Q8_0 planes or BF16 words; Q5_0 packed rows when the body admits them",
            ));
        }
    };
    gpu.elem().enqueue_embed_rows_kquant(
        gpu.stream(),
        ty,
        EmbedRowsArgs {
            w: kq_weight(w, &name)?,
            ids: io.ids,
            pos0: io.pos0,
            first: io.first,
            y,
            pos,
            n_keys,
        },
    )
}

/// The launches [`site_gemv`] makes for a site of type `ty` at `m` rows: a
/// K-quant's gemv, and past one row the token-major copy of a row-major one;
/// any other type's one launch.
pub(super) fn site_launches(ty: SiteTy, m: usize) -> usize {
    1 + usize::from(site::kgemv_order(ty) == Some(Order::RowMajor) && m > 1)
}

/// Whether a site of type `ty` reads the q8_1 rows on the gemv arm, so a quantizer runs into
/// them before its launch: the form it reads ([`site::reads_at`]), which [`site_gemv`] dispatches
/// by ([`site::kgemv_order`]). Not [`SiteTy::kquant`]: NVFP4 and IQ4_XS read them and are no
/// K-quant. Every width of this arena's q8_1 rows is whole super-blocks of 256, where `reads_at`
/// is [`SiteTy::reads`]. The launch and the node count ([`pass_launches`]) both ask this one
/// function, so they cannot disagree.
pub(super) fn reads_q8_1(ty: SiteTy) -> bool {
    ty.reads() == site::Form::Q8x128
}

/// `y = W · x` for site `name` of type `ty`, `rows` rows, at `m <=
/// GEMV_COLS` rows, token-major: a K-quant's `site::kgemv` on the q8_1 rows
/// `act` (a row-major one past one row into `cols`, then copied
/// token-major), any other type's `site::gemv` on the f32 rows `x`.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's context, weight, rows, two input forms, width, scratch and output (rust-quality R8)"
)]
pub(super) fn site_gemv(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    rows: usize,
    (act, x): (&Q8Act, &DeviceBuffer<f32>),
    m: usize,
    cols: Option<&mut DeviceBuffer<f32>>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::site_gemv";
    let gpu = c.gpu;
    let q35 = c.k.q35(WHAT)?;
    let Some(order) = site::kgemv_order(ty) else {
        return site::gemv(gpu, &q35.g32, (ty, c.w, name), x, m, y);
    };
    let gemv = |out: &mut DeviceBuffer<f32>| {
        site::kgemv(gpu, &q35.kgemv, (ty, c.w, name), act, c.sink, out)
    };
    if m == 1 || order == Order::TokenMajor {
        return gemv(y);
    }
    let cols = cols.ok_or(GpuError::state(
        WHAT,
        "the arena's row-major scratch (a K-quant site launched alone past one row)",
    ))?;
    gemv(cols)?;
    c.k.proj.token_major.enqueue(gpu.stream(), cols, rows, m, y)
}

/// The buffers a shared expert run as its own dense FFN ([`shared_dense`])
/// writes: the gate and up rows (`ff` values a token), the SwiGLU rows `h`
/// with their q8_1 form `act` (the activation of the unit's `m` columns), and
/// the output `y` (`hidden` a token).
pub(super) struct SharedDenseBufs<'a> {
    pub(super) g: &'a mut DeviceBuffer<f32>,
    pub(super) u: &'a mut DeviceBuffer<f32>,
    pub(super) h: &'a mut DeviceBuffer<f32>,
    pub(super) act: &'a mut Q8Act,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

/// A shared expert as its own dense FFN at its site types, at `m <=
/// GEMV_COLS` rows: the owner of the run for the whole-card load's apart arm
/// ([`apart_down`]) and for a placed layer whose shared expert the `_sel`
/// table does not run (`placed::SharedFfn::Sites`). Gate and up
/// ([`site_gemv`]) read the normed rows in the form their type reads: `xin`
/// is `(q8_1 of the normed rows, the f32 normed rows)`, a site that reads the
/// q8_1 rows ([`reads_q8_1`]) the first, a Q8_0 or F32 site the second. Then
/// the SwiGLU of the `ff`-wide rows into `b.h` (f32), its q8_1 form `b.act`
/// only when the down reads those (any other down reads `b.h`), and the down
/// into `b.y` (`hidden` wide). `cols` is the row-major scratch of a K-quant
/// site launched alone past one row.
pub(super) fn shared_dense(
    c: &Ctx<'_>,
    sh: &SharedSites,
    (ff, hidden): (usize, usize),
    xin: (&Q8Act, &DeviceBuffer<f32>),
    m: usize,
    mut cols: Option<&mut DeviceBuffer<f32>>,
    b: SharedDenseBufs<'_>,
) -> Result<(), GpuError> {
    let SharedDenseBufs { g, u, h, act, y } = b;
    site_gemv(
        c,
        (sh.gate_ty, &sh.gate),
        ff,
        xin,
        m,
        cols.as_deref_mut(),
        g,
    )?;
    site_gemv(c, (sh.up_ty, &sh.up), ff, xin, m, cols.as_deref_mut(), u)?;
    c.gpu
        .elem()
        .enqueue_swiglu(c.gpu.stream(), g, u, m * ff, h)?;
    if reads_q8_1(sh.down_ty) {
        c.gpu.enqueue_quantize_q8_1_layer(h, act, c.layer)?;
    }
    site_gemv(c, (sh.down_ty, &sh.down), hidden, (act, h), m, cols, y)
}

/// The attention half's middle at `m` rows, after [`attn_in`]: the head
/// norm and the rope by the rows' positions with the cache append
/// ([`attn_append`]), the flash over the rows' live key counts, and the
/// output projection's input — the attention rows' q8_1 for a K-quant one
/// ([`attn_in`]'s doc); past [`GEMV_COLS`] rows the wide arm's
/// ([`wide::attn_core`]).
pub(super) fn attn_core(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::attn_core(c, n, kv, s, m);
    }
    attn_append(c, n, kv, s, m)?;
    let gpu = c.gpu;
    let d = s.dims;
    let i = s.col(m)?;
    match n.kind {
        GqaKind::Neox128 => {
            kv.flash_128(
                &c.k.flash,
                gpu.stream(),
                FlashPass {
                    q: &s.q,
                    n_keys: &s.n_keys,
                    scale: ATTN_SCALE,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                    fault: c.sink,
                    y: &mut s.attn,
                },
                c.mma,
            )?;
            if n.o_ty.kquant() {
                gpu.enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], c.layer)?;
            }
            Ok(())
        }
        GqaKind::Gated256 => gated_256(c, n, kv, s, m),
        // The head-128 NORM turn: `attn_append` refused every other kind.
        GqaKind::Head128 { .. } | GqaKind::HeadGate128 => norm_128(c, n, kv, s, m),
    }
}

/// The attention's q/k norm and rope by the rows' positions with the cache
/// append at `m` rows, the arena's `q`, `k` and `v` rows into `kv` at the
/// rows' positions: what [`attn_core`] runs first, and all a store walk
/// runs after [`attn_in`] (`walk::store`). The launch is the plan's kind's
/// (the head-128 NEOX turn with the head norm, the head-256 partial turn
/// with the queries read out of the `[q | gate]` rows into `q_out`, the
/// head-128 NORM turn), the same at every `m`; any other kind is refused by
/// name before it.
pub(super) fn attn_append(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::attn_append";
    let (stream, w, k) = (c.gpu.stream(), c.w, c.k);
    let d = s.dims;
    match n.kind {
        GqaKind::Neox128 => {
            // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint planes.
            unsafe {
                kv.append_128(
                    &k.neox,
                    stream,
                    Append128 {
                        q: &mut s.q,
                        k: &mut s.k,
                        v: &s.v,
                        gq: f32_gain(w, &n.attn_q_norm)?,
                        gk: f32_gain(w, &n.attn_k_norm)?,
                        table: c.table,
                        pos: &s.pos,
                        eps: c.eps,
                        n_head: d.n_head,
                        n_kv: d.n_kv,
                        ctx: d.ctx,
                        m,
                        fault: c.sink,
                    },
                )
            }
        }
        GqaKind::Gated256 => {
            let q_out = s
                .q_out
                .as_mut()
                .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
            // SAFETY: the layer program embeds consecutive positions before attention;
            // saturation is outside the cache. Each call uses one sequence's row window
            // and its own cache planes, not the combined positions of different slots.
            unsafe {
                kv.append_256(
                    &k.neox,
                    stream,
                    Append256 {
                        qg: &s.q,
                        q: q_out,
                        k: &mut s.k,
                        v: &s.v,
                        gq: f32_gain(w, &n.attn_q_norm)?,
                        gk: f32_gain(w, &n.attn_k_norm)?,
                        table: c.table,
                        pos: &s.pos,
                        eps: c.eps,
                        n_head: d.n_head,
                        n_kv: d.n_kv,
                        ctx: d.ctx,
                        m,
                        fault: c.sink,
                    },
                )
            }
        }
        GqaKind::Head128 {
            pairs: Pairs::Norm,
            head_norm: false,
        } => kv.append_norm(
            &k.q35(WHAT)?.norm,
            stream,
            AppendNorm {
                q: &mut s.q,
                k: &mut s.k,
                v: &s.v,
                table: c.table,
                pos: &s.pos,
                n_head: d.n_head,
                n_kv: d.n_kv,
                ctx: d.ctx,
                m,
                fault: c.sink,
            },
        ),
        GqaKind::Head128 { pairs, head_norm } => Err(GpuError::shape(
            "qwen3moe::attention",
            format!(
                "a head-128 attention of {pairs:?} pairs with head norm {head_norm}, which no \
                 launch here serves"
            ),
        )),
        GqaKind::HeadGate128 => Err(GpuError::shape(
            "qwen3moe::attention",
            "a head-128 attention with an output gate of its own, which no launch here serves",
        )),
    }
}

/// The attention half's input side at `m` rows: the attention norm with its
/// quantizer, then q, k and v — in one launch (a Q6_K v in its own) when
/// Q4_K, else each by its type ([`site_gemv`]) — into the arena's `q`, `k`
/// and `v` rows; past [`GEMV_COLS`] rows the wide arm's
/// ([`wide::attn_in`]).
///
/// The half, [`attn_in`], [`attn_core`] and [`attn_out`] in the walk's order:
/// `x` in, `ffn_inp = x + attn_output(attn(x))` out —
/// q, k and v in one launch (a Q6_K v in its own) when Q4_K, else each by
/// its type ([`site_gemv`]), the head norm and the rope by the rows'
/// positions with the cache append, the flash over the rows' live key
/// counts, the output projection — a Q4_K one with the residual add in its
/// store, any other into `normed` and an add. At head 256 the query
/// projection writes each head's `[q | gate]`, the rope reads the queries
/// out of those rows and turns the first 64 values, and the output
/// projection's input is the flash output multiplied by `σ(gate)`: in its
/// quantizer for a K-quant, as f32 rows for any other type
/// ([`GqaKind::Gated256`]). A head-128 layer with a NORM rope and no head
/// norm ([`GqaKind::Head128`]) turns all 128 values by neighbouring pairs
/// with the cache append, and its flash output is the output projection's
/// input as it is.
pub(super) fn attn_in(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::attn_in(c, n, s, m);
    }
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        &mut s.act_x[i],
        &mut s.normed,
        c.sink,
    )?;
    if qkv_fused(n) {
        let wv = kq_weight(w, &n.attn_v)?;
        let (off_k, off_v) = s.qkv_offsets();
        k.proj.enqueue_qkv(
            stream,
            QkvArgs {
                wq: kq_weight(w, &n.attn_q)?,
                wk: kq_weight(w, &n.attn_k)?,
                wv: (n.v_ty == SiteTy::Q4K).then_some(wv),
                act: &s.act_x[i],
                y: &mut s.qkv,
                off_k,
                off_v,
            },
        )?;
        if n.v_ty == SiteTy::Q6K {
            match s.v_cols.as_mut().filter(|_| m > 1) {
                Some(cols) => {
                    gpu.enqueue_gemv_q6k(wv, &s.act_x[i], cols)?;
                    k.proj
                        .token_major
                        .enqueue(stream, cols, d.kv_len(), m, &mut s.v)?;
                }
                None => gpu.enqueue_gemv_q6k(wv, &s.act_x[i], &mut s.v)?,
            }
        }
    } else {
        let Arena {
            normed,
            act_x,
            q,
            k: kr,
            v,
            cols,
            ..
        } = s;
        let x = (&act_x[i], &*normed);
        site_gemv(c, (n.q_ty, &n.attn_q), d.q_rows, x, m, cols.as_mut(), q)?;
        site_gemv(c, (n.k_ty, &n.attn_k), d.kv_len(), x, m, cols.as_mut(), kr)?;
        site_gemv(c, (n.v_ty, &n.attn_v), d.kv_len(), x, m, cols.as_mut(), v)?;
    }
    Ok(())
}

/// The attention half's output side at `m` rows, once the attention rows
/// are ready for the output projection — their q8_1 in `act_attn` for a
/// K-quant one, f32 rows for any other: the output projection, a Q4_K one
/// with the residual add in its store, any other into `normed` and an add,
/// into `ffn_inp`; past [`GEMV_COLS`] rows the wide arm's
/// ([`wide::attn_out`]).
pub(super) fn attn_out(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::attn_out(c, n, s, m);
    }
    let (w, k) = (c.w, c.k);
    let stream = c.gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    if o_fused(n) {
        let wo = kq_weight(w, &n.attn_output)?;
        return k.proj.enqueue_o_resid(
            stream,
            OResidArgs {
                w: wo,
                act: &s.act_attn[i],
                x: &s.x,
                y: &mut s.ffn_inp,
            },
        );
    }
    let Arena {
        x,
        normed,
        gated_rows,
        attn,
        act_attn,
        ffn_inp,
        cols,
        ..
    } = s;
    // An output projection of another type reads f32 rows: at head 128 the
    // flash's own, at head 256 the products `gated_256` wrote into the gated
    // rows. A K-quant one reads `act_attn`, so its f32 operand is unused.
    let rows = match n.kind {
        GqaKind::Neox128 | GqaKind::Head128 { .. } => &*attn,
        GqaKind::Gated256 if n.o_ty.kquant() => &*attn,
        GqaKind::Gated256 => gated_rows.as_ref().ok_or(GpuError::state(
            "qwen3moe::attention",
            "the arena's gated rows",
        ))?,
        GqaKind::HeadGate128 => {
            return Err(GpuError::shape(
                "qwen3moe::attention",
                "a head-128 attention with an output gate of its own, which no launch here serves",
            ));
        }
    };
    out_resid(
        c,
        (n.o_ty, &n.attn_output),
        d.hidden,
        (&act_attn[i], rows),
        m,
        (x, normed, cols.as_mut()),
        ffn_inp,
    )
}

/// The launches [`out_resid`] makes for an output projection of type `ty`
/// at `m` rows: the projection into `normed`, then the residual add.
pub(super) fn out_resid_launches(ty: SiteTy, m: usize) -> usize {
    site_launches(ty, m) + 1
}

/// `y = x + W · rows` for an output projection `name` of type `ty` not
/// folded into its residual add: the projection into `normed` (free once
/// the layer's input projections ran), then the add.
pub(super) fn out_resid(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    hidden: usize,
    input: (&Q8Act, &DeviceBuffer<f32>),
    m: usize,
    (x, normed, cols): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        Option<&mut DeviceBuffer<f32>>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    site_gemv(c, (ty, name), hidden, input, m, cols, normed)?;
    c.gpu
        .elem()
        .enqueue_add(c.gpu.stream(), x, normed, m * hidden, y)
}

/// The attention's head-128 middle with a NORM rope and no head norm
/// ([`GqaKind::Head128`]) after its append ([`attn_append`]): the flash over
/// the rows' live key counts at the layer's layout (`n.flash`); and, for a
/// K-quant `attn_output`, the flash rows' q8_1.
fn norm_128(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, k) = (c.gpu, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    kv.flash_head128(
        &k.flash,
        stream,
        FlashPass {
            q: &s.q,
            n_keys: &s.n_keys,
            scale: ATTN_SCALE,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            fault: c.sink,
            y: &mut s.attn,
        },
        n.flash,
        c.mma,
    )?;
    if n.o_ty.kquant() {
        gpu.enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], c.layer)?;
    }
    Ok(())
}

/// The attention's head-256 middle ([`GqaKind::Gated256`]) after its append
/// ([`attn_append`], which read the queries out of the `[q | gate]` rows into
/// `q_out`): the flash over the rows' live key counts; the output
/// projection's input — for a K-quant `attn_output` quantized with each
/// value multiplied by the sigmoid of its gate, for any other type those
/// products as f32 rows into the arena's gated rows. The queries stay in
/// `q_out`.
fn gated_256(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::gated_256";
    let (gpu, k) = (c.gpu, c.k);
    let q35 = k.q35(WHAT)?;
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    let q_out = s
        .q_out
        .as_ref()
        .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
    kv.flash_256(
        &k.flash,
        stream,
        FlashPass {
            q: q_out,
            n_keys: &s.n_keys,
            scale: ATTN_SCALE_256,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            fault: c.sink,
            y: &mut s.attn,
        },
        d.n_head,
        n.flash,
        c.mma,
    )?;
    if !n.o_ty.kquant() {
        let gated = s
            .gated_rows
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the arena's gated rows"))?;
        return q35.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn: &s.attn,
                qg: &s.q,
                n_head: d.n_head,
                m,
                fault: c.sink,
                y: gated,
            },
        );
    }
    q35.gated.enqueue_q8act(
        stream,
        (&s.attn, &s.q),
        GateLayout {
            head: d.head,
            head_stride: 2 * d.head,
            offset: d.head,
            col_stride: d.q_rows,
        },
        &mut s.act_attn[i],
        m,
        c.sink,
    )
}

/// The FFN half's front at `m` rows: the norm with its quantizer and the
/// router's pick of every token's slots; a dense FFN's norm alone; past
/// [`GEMV_COLS`] rows the wide arm's ([`wide::route`]).
///
/// The half, [`route`] then [`experts`] in the walk's order: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). A plan without a shared expert routes `k` slots a
/// token (the file's `top_k`); with one ([`FfnRoute::Router`]'s `shared`)
/// the gated router adds one more, the shared expert's id in the joined
/// stacks weighted by the sigmoid of the router's last row, and every launch
/// after it runs `k + 1` slots a token; a shared expert kept apart from the
/// stacks runs as its own FFN ([`apart_down`]). Each routed stack launches
/// at the card kernel table's entry for its type (`plan::gate_up_entry`,
/// `plan::down_entry`), the family's own Q4_K gate·up on joined stacks. A
/// dense FFN ([`FfnRoute::Dense`]) launches no router: its norm is
/// `norm_quant` at every `m`, and the launches after it run its one slot a
/// token from the arena's fixed route.
/// The down `_sel` runs every token's slots in one launch: slot `t · slots
/// + j` is token `t`'s slot `j`, its id, its q8_1 column and its down rows
/// all at that index, so each slot's row is the row a one-token launch
/// writes. Its input is one quantizer launch over every token's slots (each
/// 128-value block is quantized on its own, so a column's bytes do not
/// depend on the columns beside it). The norm runs inside the router's
/// launch, `norm_quant`'s bytes: `enqueue_norm_fused` at one token,
/// `enqueue_norm_fused_m` at more; a shared expert kept apart whose gate or
/// up reads f32 rows takes `norm_quant` and the router on those rows.
pub(super) fn route(c: &Ctx<'_>, n: &FfnPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::route(c, n, s, m);
    }
    const WHAT: &str = "qwen3moe::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let i = s.col(m)?;
    let gain = f32_gain(w, &n.ffn_norm)?;
    let (gate_inp, shared) = match &n.route {
        FfnRoute::Router { gate_inp, shared } => (gate_inp, shared),
        FfnRoute::Dense => {
            return gpu.fused().enqueue_norm_quant(
                stream,
                &s.ffn_inp,
                gain,
                c.eps,
                &mut s.act_ffn[i],
                &mut s.normed,
                c.sink,
            );
        }
    };
    let router = f32_tensor(w, gate_inp)?;
    match shared {
        // A shared expert kept apart whose gate or up reads the f32 normed
        // rows: `norm_quant` writes them beside their q8_1, then the router
        // reads them — the bytes the fused norm and router write.
        Some(SharedPlan::Apart(sh)) if reads_f32(sh) => {
            gpu.fused().enqueue_norm_quant(
                stream,
                &s.ffn_inp,
                gain,
                c.eps,
                &mut s.act_ffn[i],
                &mut s.normed,
                c.sink,
            )?;
            k.q35(WHAT)?.router.enqueue_fused(
                stream,
                router,
                &s.normed,
                m,
                c.sink,
                s.route.gated(WHAT)?,
            )?;
        }
        None if m == 1 => k.router.enqueue_norm_fused(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.plain(WHAT)?,
        )?,
        None => k.router.enqueue_norm_fused_m(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.plain(WHAT)?,
        )?,
        Some(_) if m == 1 => k.q35(WHAT)?.router.enqueue_norm_fused(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.gated(WHAT)?,
        )?,
        Some(_) => k.q35(WHAT)?.router.enqueue_norm_fused_m(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.gated(WHAT)?,
        )?,
    }
    Ok(())
}

/// The FFN half after its slots are picked, at `m` rows ([`route`]'s doc):
/// the routed slots' gate·up·SwiGLU, quantizer and down, the shared expert
/// kept apart as its own FFN, and the combine into `out` (`None`: into `x`);
/// past [`GEMV_COLS`] rows the wide arm's ([`wide::experts`]).
pub(super) fn experts(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::experts(c, n, s, m, out);
    }
    match n.apart() {
        Some(sh) => apart_down(c, n, sh, s, m, out),
        None => slots_down(c, n, s, m, out),
    }
}

/// The FFN half after its slots are picked: the gate·up·SwiGLU of every
/// token's slots, one quantizer over them, the down `_sel` and the combine
/// ([`route`]'s doc).
fn slots_down(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    let slots = d.slots();
    let used = d.router.map_or(0, |r| r.used());
    if slots != c.p.slots(used) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(used)
            ),
        ));
    }
    if gate_up_fused(n) {
        match gate_up_entry(n.gate_ty, n.up_ty, WHAT)? {
            // The family's own Q4_K gate·up·SwiGLU over the slots' ids.
            GateUpEntry::Q4k => k.experts.enqueue_gate_up(
                stream,
                GateUpArgs {
                    wg: kq_weight(w, &n.gate)?,
                    wu: kq_weight(w, &n.up)?,
                    act: &s.act_ffn[i],
                    sel: s.route.ids(),
                    n_slots: m * slots,
                    rows_per_expert: d.ff,
                    fault: c.sink,
                    h: &mut s.h,
                },
            )?,
            GateUpEntry::Q5k => k.q35(WHAT)?.ffn.kq.enqueue_gate_up_q5k(
                stream,
                &GateUpAct {
                    wg: kq_weight(w, &n.gate)?,
                    wu: kq_weight(w, &n.up)?,
                    act: &s.act_ffn[i],
                    sel: s.route.ids(),
                    n_slots: m * slots,
                    rows_per_expert: d.ff,
                    slots_per_col: slots,
                    rule: Act::SiluMul,
                },
                c.sink,
                &mut s.h,
            )?,
            GateUpEntry::Q8_0
            | GateUpEntry::Q6k
            | GateUpEntry::Iq2xxs
            | GateUpEntry::Iq2xs
            | GateUpEntry::Iq2s
            | GateUpEntry::Iq1s
            | GateUpEntry::Iq1m
            | GateUpEntry::Iq3xxs
            | GateUpEntry::Iq3s
            | GateUpEntry::Iq4xs
            | GateUpEntry::Mxfp4
            | GateUpEntry::Nvfp4 => {
                return Err(GpuError::state(
                    WHAT,
                    "a gate·up entry the plan admits for no stack here",
                ));
            }
        }
    } else {
        // A dense FFN's one slot a token (the load refuses an unfused
        // routed gate·up): the token's gate and up rows apart, then the
        // SwiGLU into `h`.
        let Arena {
            normed,
            act_ffn,
            glu,
            cols,
            h,
            ..
        } = s;
        let g = glu.as_mut().ok_or(GpuError::state(
            WHAT,
            "the arena's gate and up rows (an unfused gate·up)",
        ))?;
        let x = (&act_ffn[i], &*normed);
        site_gemv(c, (n.gate_ty, &n.gate), d.ff, x, m, cols.as_mut(), &mut g.g)?;
        site_gemv(c, (n.up_ty, &n.up), d.ff, x, m, cols.as_mut(), &mut g.u)?;
        gpu.elem()
            .enqueue_swiglu(stream, &g.g, &g.u, m * slots * d.ff, h)?;
    }
    if down_sel(n) {
        let entry = down_entry(n.down_ty, WHAT)?;
        let wd = kq_weight(w, &n.down)?;
        gpu.enqueue_quantize_q8_1_layer(&s.h, &mut s.act_h[i], c.layer)?;
        let act_h = &s.act_h[i];
        match entry {
            DownEntry::Q4k => gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                wd,
                act_h,
                s.route.ids(),
                m * slots,
                d.hidden,
                &mut s.down,
            )?,
            DownEntry::Q6k => k.q6_sel.enqueue_gemv_q6k_sel(
                stream,
                wd,
                act_h,
                s.route.ids(),
                m * slots,
                d.hidden,
                &mut s.down,
            )?,
            DownEntry::Q5k => k.q35(WHAT)?.ffn.kq.enqueue_gemv_q5k_sel(
                stream,
                &SelDown {
                    w: wd,
                    act: act_h,
                    sel: s.route.ids(),
                    n_slots: m * slots,
                    rows_per_expert: d.hidden,
                },
                c.sink,
                &mut s.down,
            )?,
            DownEntry::Q5_0
            | DownEntry::Q5_1
            | DownEntry::Q8_0
            | DownEntry::Iq2xxs
            | DownEntry::Iq2xs
            | DownEntry::Iq2s
            | DownEntry::Iq1s
            | DownEntry::Iq1m
            | DownEntry::Iq3xxs
            | DownEntry::Iq3s
            | DownEntry::Iq4nl
            | DownEntry::Iq4xs
            | DownEntry::Mxfp4
            | DownEntry::Nvfp4 => {
                return Err(GpuError::state(
                    WHAT,
                    "a down entry the plan admits for no stack here",
                ));
            }
        }
    } else {
        // A dense FFN's one slot a token: the down over the token's SwiGLU
        // row is the plain projection, token-major.
        if reads_q8_1(n.down_ty) {
            gpu.enqueue_quantize_q8_1_layer(&s.h, &mut s.act_h[i], c.layer)?;
        }
        let Arena {
            h,
            act_h,
            cols,
            down,
            ..
        } = s;
        site_gemv(
            c,
            (n.down_ty, &n.down),
            d.hidden,
            (&act_h[i], h),
            m * slots,
            cols.as_mut(),
            down,
        )?;
    }
    let y = match out {
        Some(y) => y,
        None => &mut s.x,
    };
    k.experts.enqueue_combine_tokens(
        stream,
        CombineArgs {
            down: &s.down,
            w: s.route.weights(),
            resid: &s.ffn_inp,
            rows: d.hidden,
            n_slots: slots,
            m,
            y,
        },
    )
}

/// The FFN's kernels beyond the family's own: the K-quant `_sel` family (the
/// table's Q4_K and Q5_K gate·up, its Q5_K down, each skipping a
/// [`crate::hybrid::HOST`] slot) and the combine with a host sum
/// (`hostleg`), which adds a shared expert kept apart.
pub(super) struct FfnKernels {
    pub(super) kq: QuantSelKernels,
    pub(super) leg: HostLegKernels,
}

impl FfnKernels {
    /// Load both into `gpu`'s context, the `_sel` family raising into its
    /// fault word. Load-time only.
    pub(super) fn load(gpu: &Gpu) -> Result<FfnKernels, GpuError> {
        Ok(FfnKernels {
            kq: QuantSelKernels::load(gpu.context(), gpu.fault_word())?,
            leg: HostLegKernels::load(gpu.context())?,
        })
    }
}

/// The FFN half after its slots are picked on a layer that keeps its shared
/// expert apart ([`FfnPlan::apart`]), at `m <= GEMV_COLS` rows: the slots'
/// card list (`elem::card_sel`: the routed ids, the shared slot's id — the
/// routed count — [`crate::hybrid::HOST`]); the routed gate·up·SwiGLU over
/// it at the table's entry (the K-quant family's, which skips the shared
/// slot), the q8_1 of the routed slots' columns alone, the routed down
/// `_sel`; the shared expert as its own dense FFN at its types
/// ([`shared_dense`]); and the combine with a host sum (`hostleg`): `((Σ
/// routed w·d + 0) + resid) + σ · shared`, ik's grouping, the host's routed
/// sum zero because a whole-card load serves every routed slot on the card.
fn apart_down(
    c: &Ctx<'_>,
    n: &FfnPlan,
    sh: &SharedSites,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    let slots = d.slots();
    let router = d.routed(WHAT)?;
    let (used, n_routed) = (router.used(), router.experts());
    if slots != c.p.slots(used) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(used)
            ),
        ));
    }
    let n_slots = m * slots;
    let kk = &k.q35(WHAT)?.ffn;
    let Arena {
        normed,
        act_ffn,
        route,
        h,
        act_h,
        down,
        cols,
        apart,
        ffn_inp,
        x,
        ..
    } = s;
    let a = apart.as_mut().ok_or(GpuError::state(
        WHAT,
        "the arena's buffers of a shared expert kept apart",
    ))?;
    gpu.elem().enqueue_card_sel(
        stream,
        route.ids(),
        n_slots,
        n_routed,
        n_routed + 1,
        c.sink,
        &mut a.sel,
    )?;
    let gate_up = GateUpAct {
        wg: kq_weight(w, &n.gate)?,
        wu: kq_weight(w, &n.up)?,
        act: &act_ffn[i],
        sel: &a.sel,
        n_slots,
        rows_per_expert: d.ff,
        slots_per_col: slots,
        rule: Act::SiluMul,
    };
    match gate_up_entry(n.gate_ty, n.up_ty, WHAT)? {
        GateUpEntry::Q4k => kk.kq.enqueue_gate_up_q4k(stream, &gate_up, c.sink, h)?,
        GateUpEntry::Q5k => kk.kq.enqueue_gate_up_q5k(stream, &gate_up, c.sink, h)?,
        GateUpEntry::Q8_0
        | GateUpEntry::Q6k
        | GateUpEntry::Iq2xxs
        | GateUpEntry::Iq2xs
        | GateUpEntry::Iq2s
        | GateUpEntry::Iq1s
        | GateUpEntry::Iq1m
        | GateUpEntry::Iq3xxs
        | GateUpEntry::Iq3s
        | GateUpEntry::Iq4xs
        | GateUpEntry::Mxfp4
        | GateUpEntry::Nvfp4 => {
            return Err(GpuError::state(
                WHAT,
                "a gate·up entry the plan admits for no stack here",
            ));
        }
    }
    gpu.q4k_sel().enqueue_quantize_sel(
        stream,
        &QuantSel {
            x: h,
            cols: 0..n_slots,
            sel: &a.sel,
            n_card: n_routed,
        },
        c.sink,
        &mut act_h[i],
    )?;
    let wd = kq_weight(w, &n.down)?;
    let act = &act_h[i];
    match down_entry(n.down_ty, WHAT)? {
        DownEntry::Q4k => gpu
            .q4k_sel()
            .enqueue_gemv_q4k_sel(stream, wd, act, &a.sel, n_slots, d.hidden, down)?,
        DownEntry::Q6k => {
            k.q6_sel
                .enqueue_gemv_q6k_sel(stream, wd, act, &a.sel, n_slots, d.hidden, down)?;
        }
        DownEntry::Q5k => kk.kq.enqueue_gemv_q5k_sel(
            stream,
            &SelDown {
                w: wd,
                act,
                sel: &a.sel,
                n_slots,
                rows_per_expert: d.hidden,
            },
            c.sink,
            down,
        )?,
        DownEntry::Q5_0
        | DownEntry::Q5_1
        | DownEntry::Q8_0
        | DownEntry::Iq2xxs
        | DownEntry::Iq2xs
        | DownEntry::Iq2s
        | DownEntry::Iq1s
        | DownEntry::Iq1m
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4nl
        | DownEntry::Iq4xs
        | DownEntry::Mxfp4
        | DownEntry::Nvfp4 => {
            return Err(GpuError::state(
                WHAT,
                "a down entry the plan admits for no stack here",
            ));
        }
    }
    let sh_act = a.act.get_mut(i).ok_or(GpuError::state(
        WHAT,
        "the shared expert's SwiGLU activations of m columns",
    ))?;
    shared_dense(
        c,
        sh,
        (d.ff, d.hidden),
        (&act_ffn[i], &*normed),
        m,
        cols.as_mut(),
        SharedDenseBufs {
            g: &mut a.g,
            u: &mut a.u,
            h: &mut a.h,
            act: sh_act,
            y: &mut a.y,
        },
    )?;
    kk.leg.enqueue_combine(
        stream,
        HostCombineArgs {
            down,
            w: route.weights(),
            sel: &a.sel,
            hsum: &a.zero,
            resid: ffn_inp,
            shared: Some((&a.y, used)),
            rows: d.hidden,
            pitch: slots,
            n_card: n_routed,
            m,
            fault: c.sink,
            y: out.unwrap_or(x),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{pass_launches, reads_q8_1, wide_quants};
    use crate::arch::qwen3moe::plan::fixtures::layer;
    use crate::arch::qwen3moe::plan::{FfnRoute, GqaKind, Pairs, SiteTy};
    use crate::site::{Form, kgemv_order, reads_at};

    /// The launches of a one-layer pass at `m` rows, less the embedding.
    fn one(kind: GqaKind, tys: [SiteTy; 4], m: usize) -> usize {
        pass_launches(&[layer(kind, tys)], m) - 1
    }

    /// A layer's attention launches follow its sites' types: the fused Q4_K
    /// groups are 7 (12 with the FFN's 5, the pin of the Q4_K_M file), a site
    /// that leaves a group launches alone (+3 for four Q5_K sites at one row,
    /// +2 for a Q5_K value beside Q4_K q and k, which keep their gemvs), and
    /// a Q8_0 or F32 output projection at head 128 is its gemv and the add:
    /// no quantizer, as many launches as the fused one.
    #[test]
    fn an_attention_layer_launches_by_its_site_types() {
        use SiteTy::{F32, Q3K, Q4K, Q5K, Q6K, Q8_0};
        let h = GqaKind::Neox128;
        let ffn = 5;
        assert_eq!(one(h, [Q4K; 4], 1), 7 + ffn);
        assert_eq!(one(h, [Q4K, Q4K, Q6K, Q4K], 1), 8 + ffn);
        assert_eq!(one(h, [Q4K, Q4K, Q6K, Q4K], 2), 9 + ffn);
        assert_eq!(one(h, [Q5K; 4], 1), 10 + ffn);
        assert_eq!(one(h, [Q4K, Q4K, Q5K, Q4K], 1), 9 + ffn);
        // Past one row a row-major site alone adds its token-major copy; a
        // token-major Q5_K site does not.
        assert_eq!(one(h, [Q4K, Q4K, Q5K, Q4K], 2), 11 + ffn);
        assert_eq!(one(h, [Q5K; 4], 2), 10 + ffn);
        assert_eq!(one(h, [Q6K; 4], 2), 1 + 6 + 3 + (1 + 2 + 1) + ffn);
        assert_eq!(one(h, [Q3K, Q3K, Q3K, Q4K], 1), 1 + 3 + 3 + 2 + ffn);
        // Head 128 and a Q8_0 or F32 output projection: gemv and add, no
        // quantizer; the same count as the fused projection's two.
        assert_eq!(one(h, [Q4K, Q4K, Q4K, Q8_0], 1), 7 + ffn);
        assert_eq!(one(h, [Q4K, Q4K, Q4K, F32], 1), 7 + ffn);
        // Head 256 keeps its out gate in front of the projection.
        assert_eq!(one(GqaKind::Gated256, [Q4K, Q4K, Q4K, Q8_0], 1), 8 + ffn);
        // A NORM head of 128 with no head norm launches as the NEOX one does:
        // the append one launch, the flash two, no gate.
        let norm = GqaKind::Head128 {
            pairs: Pairs::Norm,
            head_norm: false,
        };
        assert_eq!(one(norm, [Q4K; 4], 1), 7 + ffn);
        assert_eq!(one(norm, [Q4K, Q4K, Q4K, Q8_0], 1), 7 + ffn);
    }

    /// The quantizer runs exactly where a site's gemv reads the q8_1 rows: the form it reads at the
    /// whole-super-block widths this arena holds ([`reads_at`]), the split `site_gemv` dispatches
    /// by ([`kgemv_order`]). NVFP4 and IQ4_XS read them and are no K-quant.
    #[test]
    fn the_quantizer_runs_where_the_gemv_reads_q8_1() {
        for &ty in SiteTy::ALL {
            assert_eq!(reads_q8_1(ty), reads_at(ty, 5120) == Form::Q8x128, "{ty}");
            assert_eq!(reads_q8_1(ty), kgemv_order(ty).is_some(), "{ty}");
        }
        use SiteTy::{Bf16, F32, Fp8B128, Iq4Xs, Mxfp8, Nvfp4, Q3K, Q4K, Q5_0, Q5K, Q6K, Q8_0};
        for ty in [Q3K, Q4K, Q5K, Q6K, Iq4Xs, Nvfp4] {
            assert!(reads_q8_1(ty), "{ty}");
        }
        for ty in [Q5_0, Q8_0, F32, Mxfp8, Fp8B128, Bf16] {
            assert!(!reads_q8_1(ty), "{ty}");
        }
    }

    /// A dense FFN's launches follow its sites' forms. At one row: `norm_quant` and the combine
    /// beside the gate, the up and the SwiGLU, then the down — alone for a Q8_0 one, behind its
    /// q8_1 quantizer for a Q5_K or NVFP4 one, one `_sel` launch for a fused Q4_K one. Past
    /// [`super::GEMV_COLS`] rows: the norm, one quantizer a form the gate and up read, the gate, the
    /// up, the SwiGLU's quantizer, the down and the combine, whatever the types; an NVFP4 site
    /// reads the q8 blocks of 32 there, so it takes a second quantizer beside a K-quant.
    #[test]
    fn a_dense_ffn_launches_by_its_site_types() {
        use SiteTy::{Nvfp4, Q4K, Q5K, Q8_0};
        let launches = |gate_up: SiteTy, down: SiteTy, m: usize| {
            let mut p = layer(GqaKind::Neox128, [Q4K; 4]);
            p.ffn.route = FfnRoute::Dense;
            [p.ffn.gate_ty, p.ffn.up_ty, p.ffn.down_ty] = [gate_up, gate_up, down];
            pass_launches(&[p], m)
        };
        // The embedding, the fused attention's 7 and the FFN.
        assert_eq!(launches(Q4K, Q4K, 1), 1 + 7 + 5);
        assert_eq!(launches(Nvfp4, Nvfp4, 1), 1 + 7 + 7);
        assert_eq!(launches(Q5K, Q5K, 1), launches(Nvfp4, Nvfp4, 1));
        assert_eq!(launches(Q8_0, Q8_0, 1) + 1, launches(Nvfp4, Nvfp4, 1));
        // A Q4_K gate·up beside an NVFP4 down: the fused gate·up, then the quantizer and the
        // projection the Q4_K `_sel` down also takes.
        assert_eq!(launches(Q4K, Nvfp4, 1), launches(Q4K, Q4K, 1));
        // The embedding and the route table, the wide attention's 10 and the FFN's 7.
        for ty in [Q4K, Q8_0, Nvfp4] {
            assert_eq!(launches(ty, ty, 9), 1 + 1 + 10 + 7, "{ty}");
        }
        assert_eq!(wide_quants(&[Q4K, Nvfp4]), 2);
        assert_eq!(wide_quants(&[Q8_0, Nvfp4]), 1);
    }
}
