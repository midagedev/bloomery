//! The draft's block pass: `w` rows `[id_last, mask × (w − 1)]` through the
//! three draft layers and the head, attending each layer's window ring and
//! the block's own rows.
//!
//! Input. Each row's embedding is the target's `token_embd` row (bf16,
//! widened exactly), copied into all four hyper-connection streams; layer
//! 0's folded input is stream 0 (the one-hot initial `pre`). Rows `1..` are
//! the mask token's, the same every block: they are written once, at load.
//! Row 0 is `id_last`'s: [`BlockPass::stage`] writes its bf16 words, each
//! row's rope tables, the visible counts and the head's first id into one
//! pinned image and copies it in one transfer ([`Inbox`]), and the pass's
//! first launch widens the row on the card (`ds41_glue_embed`, the target
//! step's own broadcast). The pass itself reads only device memory, so a
//! captured pass replays against the same image every block.
//!
//! Per layer, `m = w` rows per launch:
//!
//! 1. HC_PRE with F32 weights (`ds41_hc_pre_f32`) of the streams;
//! 2. the attention norm of the fold the previous sub-layer left;
//! 3. `q = q_b · rms(q_a · x)` (Q8_0, token-major) and its tail rope;
//! 4. the block's latent rows: `attn_kv`, then norm, tail rope and f16 row
//!    `t` of the block's own `[max_width × head_dim]` buffer;
//! 5. the attention (`ds41_attn_seg` + merge) over the window ring as the
//!    first source and the block's rows as the second: row `t` sees
//!    `vis[2t]` ring rows and `vis[2t + 1]` block rows ([`Rule`]);
//! 6. the inverse rope of the output, `wo_a` per group (one
//!    `q8_0_gemv_heads_mcol`), `wo_b`;
//! 7. HC_POST and the ffn's fold (`ds41_hc_post`);
//! 8. HC_PRE of the new streams, the ffn norm;
//! 9. the router, one launch per row; the q8_1 of the normed rows; the
//!    MXFP4 gate·up·SwiGLU and down with its combine over the concat plan
//!    (`sel` = the router's ids, `n_slots = 3m`, no dedup);
//! 10. the shared expert, gate·up·SwiGLU one launch per row, down over all
//!     rows; its sum with the routed output;
//! 11. HC_POST and the next fold.
//!
//! After the last layer its fold is the head's input ([`DraftHead`]).
//! Launches per pass: the widening, `(14 + 9 + 2m)` per layer, the head's
//! three and its Markov loop's `2m` ([`BlockPass::launches`]).
//!
//! Every buffer is per layer and allocated at load, so a pass leaves each
//! layer's intermediates in place for the gate to read, and a captured pass
//! replays against fixed addresses.

use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Graph, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use gguf::Split;
use model::arch::deepseek41::names as target_names;
use model::arch::dspark::{DraftHparams, names};

use super::head::DraftHead;
use super::kv::DraftRings;
use super::load::DraftWeights;
use super::stage::{Inbox, put_f32, view, view_mut};
use crate::attn::{self as attn_op, AttnArgs, AttnKernels};
use crate::chain::glue::glue_kernels;
use crate::experts::ExpertKernels;
use crate::experts_mxfp4::{
    DownArgs, DraftExpertKernels, DraftRouterOut, GateUpArgs, MxAct, N_EXPERT, N_USED, RouterArgs,
    concat_route,
};
use crate::hc::{HC_MIX, HC_STREAMS, HcKernels, HcPostArgs};
use crate::hc_f32::{HcF32Args, HcF32Kernels, HcF32Params};
use crate::rope::{Direction, KvAppendArgs, RopeKernels, RopeSpec, RopeTable, TailShape};

const WHAT: &str = "draft::block";

/// The widest block the pass holds scratch for.
pub const MAX_WIDTH: usize = 5;

/// The rule a pass follows where the reference and ik's draft differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// The reference `model.py`, which the engine runs: every block row
    /// attends every row of the block (`get_dspark_topk_idxs`), and every
    /// SwiGLU, routed and shared, clamps at its layer's limit (`Expert`,
    /// which caps `g` where `swiglu_clamp` caps `silu(g)`).
    Reference,
    /// ik's draft, which the oracle sets were dumped under: row `t` attends
    /// block rows `0..=t` (its block-causal mask) and no SwiGLU clamps,
    /// routed or shared (both read off the sets). The gate runs it to compare
    /// with a set like for like; the engine never does. Both differences are
    /// data — the attention's counts and the limits — not another path.
    Ik,
}

/// One block's inputs ([`BlockPass::stage`]).
pub struct BlockInput<'a> {
    /// The block's first id: the last accepted token.
    pub id_last: u32,
    /// `id_last`'s row of the target's `token_embd`, its bf16 bytes.
    pub row: &'a [u8],
    /// Rows, `1..=MAX_WIDTH`.
    pub width: usize,
    /// Row 0's rope position; row `t` turns at `first_pos + t`.
    pub first_pos: u32,
    /// Ring rows every row attends: a prefix of the ring (all of it once
    /// the positions have wrapped it).
    pub ring_rows: usize,
    pub rule: Rule,
}

/// The target's `token_embd` row `id`: its bf16 bytes.
pub fn embedding_row(target: &Split, n_embd: usize, id: u32) -> Result<&[u8], GpuError> {
    super::load::embedding_row(target, &target_names::token_embd(), n_embd as u64, id)
}

/// bf16 bytes widened to f32, exactly (`bits << 16`).
fn widen(row: &[u8]) -> Vec<f32> {
    row.as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
        .collect()
}

/// One layer's buffers, every one for [`MAX_WIDTH`] rows.
pub struct LayerBufs {
    /// The attention's HC_PRE: scaled mixes and result, [`HC_MIX`] a row.
    pub mix_a: DeviceBuffer<f32>,
    pub hc_a: DeviceBuffer<f32>,
    /// The attention norm of the incoming fold.
    pub normed: DeviceBuffer<f32>,
    pub q_a: DeviceBuffer<f32>,
    pub q_a_n: DeviceBuffer<f32>,
    /// The query rows after their tail rope.
    pub q: DeviceBuffer<f32>,
    /// `attn_kv · normed`, then the normed and turned rows in f32.
    pub kv: DeviceBuffer<f32>,
    pub kv_row: DeviceBuffer<f32>,
    /// The block's own latent rows in f16, row `t` for block row `t`.
    pub block_kv: DeviceTensor<u16>,
    /// The block append's shadow: no rows, so nothing is written beside `block_kv`.
    no_shadow: DeviceTensor<u16>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// The attention output after its inverse rope.
    pub y: DeviceBuffer<f32>,
    /// `wo_a` per group, `o_groups · o_lora_rank` a row.
    pub wo_a: DeviceBuffer<f32>,
    /// `wo_b`: the attention sub-layer's output.
    pub out_a: DeviceBuffer<f32>,
    /// The streams after the attention's HC_POST and the ffn's fold.
    pub streams_a: DeviceBuffer<f32>,
    pub fold_a: DeviceBuffer<f32>,
    pub mix_f: DeviceBuffer<f32>,
    pub hc_f: DeviceBuffer<f32>,
    /// The ffn norm of the fold.
    pub normed_f: DeviceBuffer<f32>,
    pub router: DraftRouterOut,
    /// The q8_1 of the normed rows, one scratch per width.
    act_x: Vec<MxAct>,
    /// gate·up·SwiGLU, slot-major then token-major.
    pub h: DeviceBuffer<f32>,
    /// The q8_1 of `h`, one scratch per width (`3m · m` columns).
    act_h: Vec<MxAct>,
    /// The routed output, combined.
    pub moe: DeviceBuffer<f32>,
    /// The shared expert's gate·up·SwiGLU and its output.
    pub sh_h: DeviceBuffer<f32>,
    pub sh_y: DeviceBuffer<f32>,
    /// The ffn sub-layer's output: routed plus shared.
    pub ffn_out: DeviceBuffer<f32>,
    /// The streams after the ffn's HC_POST and the next fold.
    pub streams_f: DeviceBuffer<f32>,
    pub fold_f: DeviceBuffer<f32>,
}

/// The widths every launch takes, from the draft's hyperparameters.
#[derive(Clone, Copy)]
struct Dims {
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    q_lora: usize,
    o_groups: usize,
    o_lora: usize,
    rope_dims: usize,
    ff: usize,
    ff_shared: usize,
    eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    routed_scale: f32,
    weights_norm: bool,
    window: usize,
}

impl Dims {
    fn of(hp: &DraftHparams) -> Result<Dims, GpuError> {
        if hp.hc.streams != HC_STREAMS || hp.hc_mix() != HC_MIX || hp.experts.n_used != N_USED {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} streams, {} mixes, {} experts a token; the kernels take {HC_STREAMS}, \
                     {HC_MIX}, {N_USED}",
                    hp.hc.streams,
                    hp.hc_mix(),
                    hp.experts.n_used
                ),
            });
        }
        Ok(Dims {
            n_embd: hp.n_embd,
            n_head: hp.n_head,
            head_dim: hp.head_dim,
            q_lora: hp.q_lora_rank,
            o_groups: hp.o_groups,
            o_lora: hp.o_lora_rank,
            rope_dims: hp.rope_dims,
            ff: hp.experts.ff,
            ff_shared: hp.experts.ff * hp.experts.n_shared,
            eps: hp.rms_eps,
            hc_eps: hp.hc.eps,
            hc_iters: u32::try_from(hp.hc.sinkhorn_iters).map_err(|_| GpuError::Shape {
                what: WHAT,
                detail: format!("{} Sinkhorn iterations", hp.hc.sinkhorn_iters),
            })?,
            routed_scale: hp.experts.routed_scale,
            weights_norm: hp.experts.weights_norm,
            window: hp.window,
        })
    }

    /// A query or output row's heads, `m` rows.
    fn heads(&self, m: usize) -> TailShape {
        TailShape {
            width: self.head_dim,
            n_dims: self.rope_dims,
            n_vec: self.n_head,
            m,
        }
    }
}

impl LayerBufs {
    fn new(stream: &CudaStream, d: &Dims) -> Result<LayerBufs, GpuError> {
        let b = MAX_WIDTH;
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let q_rows = b * d.n_head;
        let segs = attn_op::segments(d.window, b);
        Ok(LayerBufs {
            mix_a: z(b * HC_MIX)?,
            hc_a: z(b * HC_MIX)?,
            normed: z(b * d.n_embd)?,
            q_a: z(b * d.q_lora)?,
            q_a_n: z(b * d.q_lora)?,
            q: z(q_rows * d.head_dim)?,
            kv: z(b * d.head_dim)?,
            kv_row: z(b * d.head_dim)?,
            block_kv: DeviceTensor::zeroed(stream, b, d.head_dim)?,
            no_shadow: DeviceTensor::zeroed(stream, 0, d.head_dim)?,
            part_v: z(attn_op::partials_v_len(q_rows, segs))?,
            part_ms: z(attn_op::partials_ms_len(q_rows, segs))?,
            y: z(q_rows * d.head_dim)?,
            wo_a: z(b * d.o_groups * d.o_lora)?,
            out_a: z(b * d.n_embd)?,
            streams_a: z(b * HC_STREAMS * d.n_embd)?,
            fold_a: z(b * d.n_embd)?,
            mix_f: z(b * HC_MIX)?,
            hc_f: z(b * HC_MIX)?,
            normed_f: z(b * d.n_embd)?,
            router: DraftRouterOut::new(stream, b)?,
            act_x: (1..=b)
                .map(|m| MxAct::new(stream, m, d.n_embd))
                .collect::<Result<_, _>>()?,
            h: z(N_USED * b * b * d.ff)?,
            act_h: (1..=b)
                .map(|m| MxAct::new(stream, N_USED * m * m, d.ff))
                .collect::<Result<_, _>>()?,
            moe: z(b * d.n_embd)?,
            sh_h: z(b * d.ff_shared)?,
            sh_y: z(b * d.n_embd)?,
            ffn_out: z(b * d.n_embd)?,
            streams_f: z(b * HC_STREAMS * d.n_embd)?,
            fold_f: z(b * d.n_embd)?,
        })
    }
}

/// The loaded modules the pass launches.
struct Kernels {
    embed: glue_kernels::LoadedModule,
    attn: AttnKernels,
    rope: RopeKernels,
    hc: HcKernels,
    hc_f32: HcF32Kernels,
    dflash: DraftExpertKernels,
    experts: ExpertKernels,
}

/// What every launch of one layer reads besides its own buffers.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a DraftWeights,
    k: &'a Kernels,
    d: &'a Dims,
    /// The rows of this pass.
    m: usize,
    /// Each row's rope table, forward and back.
    cs_fwd: &'a DeviceBuffer<f32>,
    cs_back: &'a DeviceBuffer<f32>,
    vis: &'a DeviceBuffer<u32>,
    slots: &'a DeviceBuffer<u32>,
    route: &'a DeviceBuffer<u32>,
    /// Whether the SwiGLUs clamp ([`Rule`]).
    clamp: bool,
}

impl Cx<'_> {
    /// `y = W · x` over the pass's rows for Q8_0 weight `name` (`rows` rows
    /// of `k`), token-major.
    fn gemv(
        &self,
        name: &str,
        k: usize,
        rows: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (qs, d) = self.w.q8(name, k, rows)?;
        self.gpu.q8f32().enqueue_q8_0_gemv_mcol(
            self.gpu.stream(),
            Q8_0GemvMcolArgs {
                qs,
                d,
                x,
                m: self.m,
                out: GemvOut::TokenMajor,
                y,
            },
        )
    }

    /// A sub-layer's HC_PRE with its F32 weights `[fn, scale, base]`.
    fn hc_pre(
        &self,
        fns: [String; 3],
        x: &DeviceBuffer<f32>,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = self.d;
        let [f, scale, base] = fns;
        let params = HcF32Params {
            w: self.w.f32(&f, HC_MIX * HC_STREAMS * d.n_embd)?,
            scale: self.w.gain(&scale, 3)?,
            base: self.w.gain(&base, HC_MIX)?,
            eps: d.hc_eps,
            iters: d.hc_iters,
        };
        self.k.hc_f32.enqueue_pre(
            self.gpu.stream(),
            &HcF32Args {
                params: &params,
                x,
                tokens: self.m,
                rms_eps: d.eps,
            },
            mixes,
            hc,
        )
    }
}

/// Layer `l`'s attention sub-layer: steps 1–7 of the module comment.
fn enqueue_attn(
    cx: &Cx<'_>,
    l: usize,
    streams: &DeviceBuffer<f32>,
    fold: &DeviceBuffer<f32>,
    ring: &DeviceTensor<u16>,
    b: &mut LayerBufs,
) -> Result<(), GpuError> {
    let (d, w, s, m) = (cx.d, cx.w, cx.gpu.stream(), cx.m);
    let elem = cx.gpu.elem();
    cx.hc_pre(
        [
            names::hc_attn_fn(l),
            names::hc_attn_scale(l),
            names::hc_attn_base(l),
        ],
        streams,
        &mut b.mix_a,
        &mut b.hc_a,
    )?;
    let gain = w.gain(&names::attn_norm(l), d.n_embd)?;
    elem.enqueue_rms_norm(s, fold, gain, d.eps, d.n_embd, m, &mut b.normed)?;

    let heads_latent = d.n_head * d.head_dim;
    cx.gemv(
        &names::attn_q_a(l),
        d.n_embd,
        d.q_lora,
        &b.normed,
        &mut b.q_a,
    )?;
    let gain = w.gain(&names::attn_q_a_norm(l), d.q_lora)?;
    elem.enqueue_rms_norm(s, &b.q_a, gain, d.eps, d.q_lora, m, &mut b.q_a_n)?;
    cx.gemv(
        &names::attn_q_b(l),
        d.q_lora,
        heads_latent,
        &b.q_a_n,
        &mut b.q,
    )?;
    cx.k.rope
        .enqueue_rope_tail(s, &mut b.q, cx.cs_fwd, d.heads(m))?;

    cx.gemv(
        &names::attn_kv(l),
        d.n_embd,
        d.head_dim,
        &b.normed,
        &mut b.kv,
    )?;
    cx.k.rope.enqueue_kv_norm_rope_append(
        s,
        KvAppendArgs {
            kv: &b.kv,
            gain: w.gain(&names::attn_kv_a_norm(l), d.head_dim)?,
            cs: cx.cs_fwd,
            pos: cx.slots,
            eps: d.eps,
            n_dims: d.rope_dims,
            m,
            out: &mut b.kv_row,
            cache: &mut b.block_kv,
            shadow: &mut b.no_shadow,
        },
    )?;

    cx.k.attn.enqueue(
        s,
        AttnArgs {
            q: &b.q,
            window: ring,
            compressed: Some(&b.block_kv),
            selected: None,
            vis: cx.vis,
            sinks: w.gain(&names::attn_sinks(l), d.n_head)?,
            scale: 1.0 / (d.head_dim as f32).sqrt(),
            tokens: m,
            heads: d.n_head,
            part_v: &mut b.part_v,
            part_ms: &mut b.part_ms,
            y: &mut b.y,
        },
    )?;
    cx.k.rope
        .enqueue_rope_tail(s, &mut b.y, cx.cs_back, d.heads(m))?;
    let (group_k, groups_out) = (heads_latent / d.o_groups, d.o_groups * d.o_lora);
    let (qs, qd) = w.q8(&names::attn_output_a(l), group_k, groups_out)?;
    cx.gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
        s,
        Q8_0GemvHeadsMcolArgs {
            qs,
            d: qd,
            x: &b.y,
            rows_per_head: d.o_lora,
            x_head_stride: group_k,
            y_head_stride: d.o_lora,
            y_off: 0,
            m,
            x_col_stride: heads_latent,
            y_col_stride: groups_out,
            y: &mut b.wo_a,
        },
    )?;
    cx.gemv(
        &names::attn_output_b(l),
        groups_out,
        d.n_embd,
        &b.wo_a,
        &mut b.out_a,
    )?;
    cx.k.hc.enqueue_post(
        s,
        &HcPostArgs {
            x: &b.out_a,
            res: streams,
            hc: &b.hc_a,
            n_embd: d.n_embd,
            tokens: m,
        },
        &mut b.streams_a,
        &mut b.fold_a,
    )
}

/// Layer `l`'s ffn sub-layer: steps 8–11 of the module comment.
fn enqueue_ffn(cx: &Cx<'_>, l: usize, b: &mut LayerBufs) -> Result<(), GpuError> {
    let (d, w, s, m) = (cx.d, cx.w, cx.gpu.stream(), cx.m);
    let elem = cx.gpu.elem();
    cx.hc_pre(
        [
            names::hc_ffn_fn(l),
            names::hc_ffn_scale(l),
            names::hc_ffn_base(l),
        ],
        &b.streams_a,
        &mut b.mix_f,
        &mut b.hc_f,
    )?;
    let gain = w.gain(&names::ffn_norm(l), d.n_embd)?;
    elem.enqueue_rms_norm(s, &b.fold_a, gain, d.eps, d.n_embd, m, &mut b.normed_f)?;

    let gate_inp = w.f32(&names::ffn_gate_inp(l), N_EXPERT * d.n_embd)?;
    let bias = w.gain(&names::exp_probs_b(l), N_EXPERT)?;
    for tok in 0..m {
        cx.k.dflash.enqueue_router(
            s,
            &RouterArgs {
                w: gate_inp,
                x: &b.normed_f,
                bias,
                scale: d.routed_scale,
                norm: d.weights_norm,
                tok,
                fault: cx.gpu.unlabelled_sink(),
            },
            &mut b.router,
        )?;
    }
    let ex = w.experts(l).ok_or_else(|| GpuError::State {
        what: WHAT,
        missing: "a layer's routed experts",
    })?;
    let limit = if cx.clamp {
        *w.hp().swiglu_limit.get(l).ok_or(GpuError::State {
            what: WHAT,
            missing: "a layer's SwiGLU limit",
        })?
    } else {
        // `swiglu_clamp`'s no-clamp value.
        0.0
    };
    let n_slots = N_USED * m;
    let act_x = &mut b.act_x[m - 1];
    cx.k.dflash
        .enqueue_quantize(s, &b.normed_f, act_x, cx.gpu.unlabelled_sink())?;
    cx.k.dflash.enqueue_gate_up(
        s,
        &GateUpArgs {
            gate: &ex.gate,
            up: &ex.up,
            act: act_x,
            sel: &b.router.ids,
            route: cx.route,
            n_slots,
            limit,
        },
        &mut b.h,
    )?;
    let act_h = &mut b.act_h[m - 1];
    cx.k.dflash
        .enqueue_quantize(s, &b.h, act_h, cx.gpu.unlabelled_sink())?;
    cx.k.dflash.enqueue_down(
        s,
        &DownArgs {
            down: &ex.down,
            act: act_h,
            sel: &b.router.ids,
            route: cx.route,
            wts: &b.router.weights,
            n_slots,
            m,
        },
        &mut b.moe,
    )?;

    let dense = |name: String| {
        w.dense(&name).ok_or(GpuError::Tensor {
            what: WHAT,
            name,
            need: "a resident Q8_0 shared expert",
        })
    };
    let (gate, up) = (
        dense(names::ffn_gate_shexp(l))?,
        dense(names::ffn_up_shexp(l))?,
    );
    let limit_sh = if cx.clamp {
        *w.hp().swiglu_limit_shared.get(l).ok_or(GpuError::State {
            what: WHAT,
            missing: "a layer's shared SwiGLU limit",
        })?
    } else {
        // `swiglu_clamp`'s no-clamp value.
        0.0
    };
    for t in 0..m {
        let x = view::<f32, _>(&b.normed_f, t * d.n_embd, d.n_embd)?;
        let mut h = view_mut::<f32, _>(&mut b.sh_h, t * d.ff_shared, d.ff_shared)?;
        cx.k.experts
            .enqueue_shexp_gate_up(s, gate, up, &x, limit_sh, &mut h)?;
    }
    cx.gemv(
        &names::ffn_down_shexp(l),
        d.ff_shared,
        d.n_embd,
        &b.sh_h,
        &mut b.sh_y,
    )?;
    elem.enqueue_add(s, &b.moe, &b.sh_y, m * d.n_embd, &mut b.ffn_out)?;
    cx.k.hc.enqueue_post(
        s,
        &HcPostArgs {
            x: &b.ffn_out,
            res: &b.streams_a,
            hc: &b.hc_f,
            n_embd: d.n_embd,
            tokens: m,
        },
        &mut b.streams_f,
        &mut b.fold_f,
    )
}

/// Where [`enqueue_embed`] reads the row: word `at` of `params`, `n_embd`
/// bf16.
struct Embed<'a> {
    params: &'a DeviceBuffer<u32>,
    at: usize,
    n_embd: usize,
}

/// Row 0 of layer 0's streams and fold from the image's bf16 row: one
/// launch.
fn enqueue_embed(
    gpu: &Gpu,
    module: &glue_kernels::LoadedModule,
    e: &Embed<'_>,
    streams0: &mut DeviceBuffer<f32>,
    fold0: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let n = e.n_embd;
    let half = launch_u32(WHAT, "half a row", n / 2)?;
    let mut streams = view_mut::<f32, _>(streams0, 0, HC_STREAMS * n)?;
    let mut input = view_mut::<f32, _>(fold0, 0, n)?;
    let prep = module.prepare_ds41_glue_embed(LaunchConfig1D::new(half.div_ceil(256), 256, 0))?;
    module.ds41_glue_embed(
        gpu.stream(),
        &prep,
        e.params,
        launch_u32(WHAT, "the row's word", e.at)?,
        half,
        launch_u32(WHAT, "streams", HC_STREAMS)?,
        &mut streams,
        &mut input,
    )?;
    Ok(())
}

/// Where each per-block input sits in the pass's [`Inbox`], in words.
#[derive(Clone, Copy)]
struct Image {
    /// `id_last`'s embedding row, two bf16 a word, the low half first.
    row: usize,
    /// Each row's rope table, forward then back, [`MAX_WIDTH`] rows of
    /// `rope_dims` f32 each.
    fwd: usize,
    back: usize,
    /// Per row `t`: `vis[2t]` ring rows, `vis[2t + 1]` block rows.
    vis: usize,
    /// The head's first id.
    first: usize,
    words: usize,
}

impl Image {
    fn of(d: &Dims) -> Image {
        let row = 0;
        let fwd = row + d.n_embd / 2;
        let back = fwd + MAX_WIDTH * d.rope_dims;
        let vis = back + MAX_WIDTH * d.rope_dims;
        let first = vis + 2 * MAX_WIDTH;
        Image {
            row,
            fwd,
            back,
            vis,
            first,
            words: first + 1,
        }
    }
}

/// The block pass's buffers and constants. See the module comment.
pub struct BlockPass {
    layers: Vec<LayerBufs>,
    head: DraftHead,
    k: Kernels,
    d: Dims,
    table: RopeTable,
    /// Layer 0's streams and fold: row 0 widened by each pass, the mask
    /// token's rows after it written at load.
    streams0: DeviceBuffer<f32>,
    fold0: DeviceBuffer<f32>,
    /// Every per-block input, one copy a block ([`Image`]).
    inbox: Inbox,
    image: Image,
    /// Row `t`'s slot in the block buffer: `t`.
    slots: DeviceBuffer<u32>,
    /// The concat plan's route, `0 .. 3 · MAX_WIDTH`; a pass of `m` rows
    /// reads its first `3m`.
    route: DeviceBuffer<u32>,
    /// The rope tables of the block being staged; kept to stage without
    /// allocating.
    fwd: Vec<f32>,
    back: Vec<f32>,
    /// Rows the last [`BlockPass::stage`] wrote; 0 before the first.
    width: usize,
    /// The rule the last stage wrote.
    rule: Rule,
}

impl BlockPass {
    /// Buffers for blocks of up to [`MAX_WIDTH`] rows over `w`'s draft.
    /// Load-time only.
    pub fn new(gpu: &Gpu, w: &DraftWeights) -> Result<BlockPass, GpuError> {
        let s = gpu.stream();
        let hp = w.hp();
        let d = Dims::of(hp)?;
        let ctx = gpu.context();
        let mask_words = w.mask_row().buf().to_host_vec(s)?;
        let mask_bytes: Vec<u8> = mask_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mask = widen(&mask_bytes);
        if mask.len() != d.n_embd {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a mask row of {} values; rows of {}", mask.len(), d.n_embd),
            });
        }
        let b = MAX_WIDTH;
        let n = d.n_embd;
        let mut streams0 = vec![0.0f32; HC_STREAMS * n];
        let mut fold0 = vec![0.0f32; n];
        for _ in 1..b {
            for _ in 0..HC_STREAMS {
                streams0.extend_from_slice(&mask);
            }
            fold0.extend_from_slice(&mask);
        }
        let image = Image::of(&d);
        // SAFETY: this crate owns the embedded device bundle produced for
        // `glue_kernels`; the launcher checks its launch contract.
        let embed = unsafe { glue_kernels::load(ctx)? };
        Ok(BlockPass {
            layers: (0..hp.n_layer)
                .map(|_| LayerBufs::new(s, &d))
                .collect::<Result<_, _>>()?,
            head: DraftHead::new(gpu, w, b)?,
            k: Kernels {
                embed,
                attn: AttnKernels::load(ctx)?,
                rope: RopeKernels::load(ctx)?,
                hc: HcKernels::load(ctx)?,
                hc_f32: HcF32Kernels::load(ctx)?,
                dflash: DraftExpertKernels::load(ctx)?,
                experts: ExpertKernels::load(ctx)?,
            },
            table: RopeTable::new(&RopeSpec::window(hp.rope_base, hp.rope_dims))?,
            streams0: DeviceBuffer::from_host(s, &streams0)?,
            fold0: DeviceBuffer::from_host(s, &fold0)?,
            inbox: Inbox::new(gpu, image.words)?,
            image,
            slots: DeviceBuffer::from_host(s, &(0u32..).take(b).collect::<Vec<_>>())?,
            route: DeviceBuffer::from_host(s, &concat_route(b))?,
            fwd: Vec::with_capacity(b * d.rope_dims),
            back: Vec::with_capacity(b * d.rope_dims),
            width: 0,
            rule: Rule::Reference,
            d,
        })
    }

    /// Write one block's inputs — `id_last`'s row, the rope tables, the
    /// visible counts, the head's first id — into the pinned image and
    /// enqueue its one copy to the card on `stream`. Asynchronous, except
    /// that it waits for the previous block's copy to have read the image;
    /// never inside a capture.
    pub fn stage(&mut self, stream: &CudaStream, input: &BlockInput<'_>) -> Result<(), GpuError> {
        let (m, n) = (input.width, self.d.n_embd);
        let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
        if !(1..=MAX_WIDTH).contains(&m) {
            return Err(shape(format!("a block of {m} rows; 1..={MAX_WIDTH}")));
        }
        if input.row.len() != 2 * n {
            return Err(shape(format!(
                "an embedding row of {} bytes; rows of {n} bf16",
                input.row.len()
            )));
        }
        if input.ring_rows == 0 || input.ring_rows > self.d.window {
            return Err(shape(format!(
                "{} ring rows; 1..={}",
                input.ring_rows, self.d.window
            )));
        }
        let count = |v: usize| u32::try_from(v).map_err(|_| shape(format!("a count of {v}")));
        let (rows, ring_rows) = (count(m)?, count(input.ring_rows)?);
        self.fwd.clear();
        self.back.clear();
        for t in 0..rows {
            let pos = input
                .first_pos
                .checked_add(t)
                .ok_or_else(|| shape(format!("position {} + {t}", input.first_pos)))?;
            self.table.push(pos, Direction::Forward, &mut self.fwd);
            self.table.push(pos, Direction::Back, &mut self.back);
        }
        let im = self.image;
        let host = self.inbox.host_mut()?;
        for (d, b) in host[im.row..im.fwd]
            .iter_mut()
            .zip(input.row.as_chunks::<4>().0)
        {
            *d = u32::from_le_bytes(*b);
        }
        put_f32(&mut host[im.fwd..im.back], &self.fwd);
        put_f32(&mut host[im.back..im.vis], &self.back);
        for (t, v) in (0..MAX_WIDTH).zip(host[im.vis..im.first].as_chunks_mut::<2>().0) {
            *v = if t < m {
                let block = match input.rule {
                    Rule::Reference => rows,
                    Rule::Ik => count(t + 1)?,
                };
                [ring_rows, block]
            } else {
                [0, 0]
            };
        }
        host[im.first] = input.id_last;
        self.inbox.upload(stream, im.words)?;
        self.width = m;
        self.rule = input.rule;
        Ok(())
    }

    /// Enqueue the widening of row 0, every layer and the head's logits
    /// over the staged block, reading `rings`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_body(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &DraftRings,
    ) -> Result<(), GpuError> {
        let m = self.width;
        if m == 0 {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a staged block",
            });
        }
        let (d, im) = (self.d, self.image);
        let params = self.inbox.u32s(0, im.words)?;
        enqueue_embed(
            gpu,
            &self.k.embed,
            &Embed {
                params: &params,
                at: im.row,
                n_embd: d.n_embd,
            },
            &mut self.streams0,
            &mut self.fold0,
        )?;
        let cs_fwd = self.inbox.f32s(im.fwd, MAX_WIDTH * d.rope_dims)?;
        let cs_back = self.inbox.f32s(im.back, MAX_WIDTH * d.rope_dims)?;
        let vis = self.inbox.u32s(im.vis, 2 * MAX_WIDTH)?;
        let cx = Cx {
            gpu,
            w,
            k: &self.k,
            d: &self.d,
            m,
            cs_fwd: &cs_fwd,
            cs_back: &cs_back,
            vis: &vis,
            slots: &self.slots,
            route: &self.route,
            clamp: self.rule == Rule::Reference,
        };
        for l in 0..self.layers.len() {
            let (done, rest) = self.layers.split_at_mut(l);
            let b = &mut rest[0];
            let (streams, fold) = match done.last() {
                None => (&self.streams0, &self.fold0),
                Some(prev) => (&prev.streams_f, &prev.fold_f),
            };
            let ring = rings.ring(l).ok_or(GpuError::State {
                what: WHAT,
                missing: "a ring per layer",
            })?;
            enqueue_attn(&cx, l, streams, fold, ring, b)?;
            enqueue_ffn(&cx, l, b)?;
        }
        let last = self.layers.last().ok_or(GpuError::State {
            what: WHAT,
            missing: "a layer",
        })?;
        self.head.enqueue_logits(gpu, w, &last.fold_f, m)
    }

    /// Enqueue the head's Markov loop over the logits the body left.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_markov(&mut self, gpu: &Gpu, w: &DraftWeights) -> Result<(), GpuError> {
        let first = self.inbox.u32s(self.image.first, 1)?;
        self.head.enqueue_markov(gpu, w, self.width, &first)
    }

    /// [`BlockPass::enqueue_body`] then [`BlockPass::enqueue_markov`].
    pub fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &DraftRings,
    ) -> Result<(), GpuError> {
        self.enqueue_body(gpu, w, rings)?;
        self.enqueue_markov(gpu, w)
    }

    /// Capture the whole pass of `m` rows under [`Rule::Reference`] over
    /// `rings`: [`BlockPass::enqueue`] with the staging left out, so a
    /// replay after any [`BlockPass::stage`] of `m` rows under that rule is
    /// that block's pass. The graph names this pass's buffers, `w` and
    /// `rings`: the caller keeps all three alive and in place while it holds
    /// the graph. Leaves the pass staged as `m` rows of whatever the image
    /// holds. Load-time only.
    pub fn capture(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &DraftRings,
        m: usize,
    ) -> Result<Graph, GpuError> {
        if !(1..=MAX_WIDTH).contains(&m) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a capture of {m} rows; 1..={MAX_WIDTH}"),
            });
        }
        self.width = m;
        self.rule = Rule::Reference;
        gpu.capture(|_| self.enqueue(gpu, w, rings))
    }

    /// Kernel launches one pass of `m` rows makes: the widening, per layer
    /// the attention sub-layer's 14 and the ffn's `9 + 2m`, then the head's.
    #[must_use]
    pub fn launches(&self, m: usize) -> usize {
        1 + self.layers.len() * (14 + 9 + 2 * m)
            + DraftHead::LOGIT_LAUNCHES
            + DraftHead::markov_launches(m)
    }

    /// Rows the last stage wrote.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The rule the last stage wrote.
    #[must_use]
    pub fn rule(&self) -> Rule {
        self.rule
    }

    /// Layer 0's streams as the last pass left them.
    #[must_use]
    pub fn streams0(&self) -> &DeviceBuffer<f32> {
        &self.streams0
    }

    /// Layer `l`'s buffers as the last pass left them.
    #[must_use]
    pub fn layer(&self, l: usize) -> Option<&LayerBufs> {
        self.layers.get(l)
    }

    /// The head.
    #[must_use]
    pub fn head(&self) -> &DraftHead {
        &self.head
    }

    /// The block's proposal after the Markov loop. Blocking.
    pub fn tokens(&self, stream: &CudaStream) -> Result<Vec<u32>, GpuError> {
        self.head.tokens(stream, self.width)
    }
}
