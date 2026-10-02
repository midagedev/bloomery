//! The wide arm of each layer op: a unit of more than [`GEMV_COLS`] rows.
//! Each op of `dispatch` (and the delta mixer's input and output halves)
//! takes its first line from here when `m > GEMV_COLS` and keeps its gemv
//! body below it — one layer walk, the op choosing its kernel family from
//! `m`, as ik's `mul_mat` picks its matrix-vector kernel for at most eight
//! columns.
//!
//! A wide op reads each weight once for the unit: every projection runs
//! through the GEMM its file type picks (`crate::site::gemm`: a K-quant the
//! grouped int8 GEMM over q8_1 blocks of 128, a Q8_0 the 32-value GEMM over
//! q8 blocks of 32, an F32 the F32 tile) over a one-expert table
//! ([`route_dense`], filled once per unit after its embedding), the routed
//! experts through one table per layer (gate and up reading each token's
//! column, down each slot's own), a dense FFN's three matrices through the
//! one-expert table (its one slot a token is the token's column). A unit's
//! rows are quantized into exactly the forms the sites reading them take;
//! the norms, the quantizers, the rope with the cache append,
//! the prefill flash, the router's two launches, the SwiGLU quantizer and
//! the combine run over the unit's rows. The conv, the delta step and the
//! gated norm are the gemv arm's own launches: they already take any row
//! count.
//!
//! Numeric class. A wide op computes a token's values from that token's
//! inputs alone — the GEMM accumulates each (slot, row) on its own, the
//! flash walks a row's keys in fixed tiles whatever the rows beside it — so
//! a token's bits do not depend on the unit it lands in. Against the gemv arm
//! the products sum in another order (128-value integer blocks into one f32
//! accumulator, where the gemv sums lane partials through a warp tree) and
//! the attention weighs the values with f16 weights, so the two arms agree
//! to the error of those sums and are not bit-equal.

use super::body::ATTN_SCALE_256;
use super::dispatch::Ctx;
use super::experts::CombineArgs;
use super::plan::{DeltaPlan, FfnPlan, FfnRoute, Flash, Form, GqaKind, GqaPlan, SiteTy};
use super::router::MAX_TOKENS;
use super::scratch::{Arena, Dims, Forms, GdnArena, KvPlanes, Wants};
use crate::GpuError;
use crate::flash_gqa::{HEAD_256, partials_ms_len, partials_v_len, partials_v_len_256};
use crate::flash_gqa_prefill::GqaPrefillArgs;
use crate::gated_quant::GateLayout;
use crate::gemm::{GEMM_BN, GEMM_MAX_SLOTS, GEMM32_STEP, GemmAct, GemmAct32, GemmInput, GemmRoute};
use crate::linear::{self, LinearShape};
use crate::model::MAX_PASS_ROWS;
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::q38::OutGateArgs;
use crate::rope_neox::PartialNeoxArgs;
use crate::site::{self, WideIn, WideKernels};
use cuda_core::{CudaStream, DeviceBuffer};

/// The most rows an op runs through its gemv arm — ik's matrix-vector cut
/// (`ne[1] <= 8`), and the rows of a captured pass. Past it, the wide arm.
pub(super) const GEMV_COLS: usize = MAX_PASS_ROWS;
const _: () = assert!(GEMV_COLS == MAX_TOKENS);

const WHAT: &str = "qwen3moe::wide";

/// One wide input's quantized forms, each held when a site reads it
/// ([`Wants`]): q8_1 blocks of 128 for a K-quant site, q8 blocks of 32 for a
/// Q8_0 site.
struct Acts {
    q128: Option<GemmAct>,
    q32: Option<GemmAct32>,
}

impl Acts {
    fn new(stream: &CudaStream, want: Wants, cols: usize, k: usize) -> Result<Acts, GpuError> {
        Ok(Acts {
            q128: want
                .q128
                .then(|| GemmAct::new(stream, cols, k))
                .transpose()?,
            q32: want
                .q32
                .then(|| GemmAct32::new(stream, cols, k))
                .transpose()?,
        })
    }

    fn bytes(&self) -> usize {
        self.q128.as_ref().map_or(0, GemmAct::bytes) + self.q32.as_ref().map_or(0, GemmAct32::bytes)
    }

    /// The forms `x`'s first `m` columns are quantized into for sites of
    /// types `tys`: each form one of them reads, and nothing else.
    fn quantize(
        &mut self,
        c: &Ctx<'_>,
        x: &DeviceBuffer<f32>,
        m: usize,
        tys: &[SiteTy],
    ) -> Result<(), GpuError> {
        const WHAT_Q: &str = "qwen3moe::wide::quantize";
        let reads = |f: Form| tys.iter().any(|t| t.reads() == f);
        if reads(Form::Q8x128) {
            let a = self.q128.as_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the wide part's q8_1 blocks of 128 (a K-quant site reads them)",
            ))?;
            c.gpu.enqueue_quantize_gemm(x, m, a, c.sink)?;
        }
        if reads(Form::Q8x32) {
            let a = self.q32.as_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the wide part's q8 blocks of 32 (a Q8_0 site reads them)",
            ))?;
            c.k.q35(WHAT_Q)?
                .g32
                .enqueue_quantize_gemm32(c.gpu.stream(), x, m, a, c.sink)?;
        }
        Ok(())
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_ref(),
            q32: self.q32.as_ref(),
            f32: Some(f32),
        }
    }
}

/// The wide part of an arena of `rows > GEMV_COLS` rows: the GEMMs'
/// activations, the second SwiGLU operand, the output projection's rows and
/// the two route tables.
pub(super) struct Wide {
    /// The normed rows (`hidden` a token): the mixer's projections' input,
    /// then the experts' gate·up input.
    act_hid: Acts,
    /// The attention rows (or a delta layer's gated norm): the output
    /// projection's input.
    act_attn: Acts,
    /// The slots' SwiGLU, one column per slot: the down's input.
    act_h: Acts,
    /// The up rows, per slot `ff` values; the gate rows are the arena's `h`.
    up: DeviceBuffer<f32>,
    /// The output projection's rows; the residual add reads them.
    attn_o: DeviceBuffer<f32>,
    /// The one-expert table every projection of a unit reads.
    dense: GemmRoute,
    /// The layer's expert table over `t · slots` slots, `slots` the
    /// router's per token (a folded shared expert's among them), over the
    /// joined stacks' `logits()` experts; `None` for a dense chain, whose
    /// FFN reads `dense`.
    moe: Option<GemmRoute>,
}

impl Wide {
    /// The wide part for `rows` tokens of `d` and the sites' forms `forms`,
    /// or a named refusal when their slots pass the GEMM's route table.
    /// Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        d: &Dims,
        rows: usize,
        forms: Forms,
    ) -> Result<Wide, GpuError> {
        let slots = rows * d.slots();
        if slots > GEMM_MAX_SLOTS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{rows} rows at {} slots a token are {slots} slots; the grouped GEMM's route \
                     table holds {GEMM_MAX_SLOTS}",
                    d.slots()
                ),
            ));
        }
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(Wide {
            act_hid: Acts::new(stream, forms.hid, rows, d.hidden)?,
            act_attn: Acts::new(stream, forms.attn, rows, d.attn_len())?,
            act_h: Acts::new(stream, forms.h, slots, d.ff)?,
            up: f(slots * d.ff)?,
            attn_o: f(rows * d.hidden)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            moe: d
                .router
                .map(|r| GemmRoute::new(stream, slots, r.logits()))
                .transpose()?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.act_hid.bytes()
            + self.act_attn.bytes()
            + self.act_h.bytes()
            + self.up.num_bytes()
            + self.attn_o.num_bytes()
            + self.dense.bytes()
            + self.moe.as_ref().map_or(0, GemmRoute::bytes)
    }
}

/// Bytes of one q8_1 activation column of `k` values, a `Q8Act`'s or a
/// `GemmAct`'s: the q3 pairs (u64), the q4 and q6 permutations, the 32-value
/// code sums and the 128-value scales.
fn act_col_bytes(k: usize) -> usize {
    let n_sb = k / 256;
    64 * n_sb.div_ceil(2) * 8
        + (256 * n_sb.div_ceil(4) + 128 * n_sb.div_ceil(2)) * 4
        + 10 * n_sb * 4
}

/// Bytes of one `GemmAct32` column of `k` values: per 64-value step sixteen
/// code words, two scales and two sums.
fn act32_col_bytes(k: usize) -> usize {
    k.div_ceil(GEMM32_STEP) * (16 + 2 + 2) * 4
}

/// Bytes of one wide input's forms `want` over `cols` columns of `k`.
fn wants_bytes(want: Wants, cols: usize, k: usize) -> usize {
    cols * (usize::from(want.q128) * act_col_bytes(k) + usize::from(want.q32) * act32_col_bytes(k))
}

/// Bytes of a route table for `slots` slots over `experts` experts.
fn route_bytes(slots: usize, experts: usize) -> usize {
    let tiles = slots / GEMM_BN + experts.min(slots);
    let zeros = if experts == 1 { slots } else { 0 };
    (slots + 2 * tiles + 2 + zeros) * 4
}

/// Bytes of a delta layer's intermediates for `rows` tokens
/// (`GdnArena::new`): the four projection blocks, the conv, β, the decay,
/// the delta output, and past one row the Q6_K projection's row-major copy
/// of at most [`GEMV_COLS`] tokens.
fn gdn_bytes(s: LinearShape, rows: usize) -> usize {
    let (c, zl, nv) = (s.channels(), s.n_v * linear::HEAD, s.n_v);
    let cols = if rows > 1 { rows.min(GEMV_COLS) * c } else { 0 };
    (rows * (c + zl + 2 * nv) + rows * c + 2 * rows * nv + rows * zl + cols) * 4
}

/// The device bytes `Arena::with(d, rows, forms)` allocates, from `d`, `rows`
/// and `forms` alone: what a load checks against the card's free bytes
/// before it allocates the arena, and what the arena it then allocates must
/// hold.
pub(super) fn arena_bytes(d: &Dims, rows: usize, forms: Forms) -> usize {
    let n = rows.min(GEMV_COLS);
    let (q_len, kv_len, att, slots) = (d.q_rows, d.kv_len(), d.attn_len(), d.slots());
    let part_v = if d.head == HEAD_256 {
        partials_v_len_256(n, d.n_head, d.ctx)
    } else {
        partials_v_len(n, d.n_head, d.ctx)
    };
    let f32s = 3 * rows * d.hidden
        + rows * (q_len + 2 * kv_len)
        + if q_len == att { 0 } else { rows * att }
        + if rows > 1 { n * kv_len } else { 0 }
        + part_v
        + partials_ms_len(n, d.n_head, d.ctx)
        + rows * att
        + rows * slots * (d.ff + d.hidden)
        + if forms.glu { 2 * n * slots * d.ff } else { 0 }
        + if rows > 1 { n * forms.cols } else { 0 };
    let u32s = 2 * rows;
    let acts: usize = (1..=n)
        .map(|m| {
            m * (2 * act_col_bytes(d.hidden) + act_col_bytes(att)) + m * slots * act_col_bytes(d.ff)
        })
        .sum();
    let route = match d.router {
        Some(r) => (rows * (r.logits() + r.experts() + 2 * slots) + 1) * 4,
        None => 2 * rows * 4,
    };
    let wide = if rows > GEMV_COLS {
        let s = rows * slots;
        wants_bytes(forms.hid, rows, d.hidden)
            + wants_bytes(forms.attn, rows, att)
            + wants_bytes(forms.h, s, d.ff)
            + (s * d.ff + rows * d.hidden) * 4
            + route_bytes(rows, 1)
            + d.router.map_or(0, |r| route_bytes(s, r.logits()))
    } else {
        0
    };
    (f32s + u32s) * 4 + acts + route + d.lin.map_or(0, |s| gdn_bytes(s, rows)) + wide
}

/// The arena's wide part, or a named refusal (an arena of at most
/// [`GEMV_COLS`] rows has none).
fn wide_of<'a>(w: &'a mut Option<Wide>, what: &'static str) -> Result<&'a mut Wide, GpuError> {
    w.as_mut().ok_or(GpuError::state(
        what,
        "the arena's wide part (an arena of more than GEMV_COLS rows)",
    ))
}

/// Enqueue `y = W · x` for `rows` rows of weight `name` (of type `ty`) over
/// the columns `route` holds, `input` picking each slot's column of the
/// forms `x` holds, `m` the unit's rows.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's context, weight, rows, inputs, table, width and output (rust-quality R8)"
)]
fn gemm(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let k = WideKernels {
        gemm: &c.k.gemm,
        g32: &c.k.q35(WHAT)?.g32,
    };
    site::gemm(c.gpu, &k, (ty, c.w, name), rows, x, (route, input), m, y)
}

/// The one-expert table of a unit of `m` rows: `m` slots on expert 0, slot
/// `s` reading column `s`, which every projection of the unit reads. Once per
/// unit, behind its embedding.
pub(super) fn route_dense(c: &Ctx<'_>, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    let w = wide_of(&mut s.wide, "qwen3moe::wide::route_dense")?;
    c.k.gemm
        .enqueue_route_dense(c.gpu.stream(), m, &mut w.dense, c.gpu.unlabelled_sink())
}

/// The attention half at `m` rows: `x` in, `ffn_inp = x + W_o · (flash ⊙
/// σ(gate))` out, the unit's K/V rows appended to the layer's planes at the
/// rows' positions. A K-quant `attn_output` reads the gated rows quantized
/// in one launch; any other type reads them as f32 rows (the out gate into
/// the free query buffer), quantized as it takes them. Qwen3's head-128
/// attention takes its wide rows through `ubatch.rs` and is refused here by
/// name.
pub(super) fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_ATTN: &str = "qwen3moe::wide::attention";
    if n.kind != GqaKind::Gated256 {
        return Err(GpuError::shape(
            WHAT_ATTN,
            format!(
                "layer {}: head-128 attention at {m} rows (Qwen3 runs more than GEMV_COLS rows \
                 through ubatch.rs)",
                c.layer
            ),
        ));
    }
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT_ATTN)?;
    let stream = gpu.stream();
    let d = s.dims;
    let Arena {
        x,
        normed,
        pos,
        n_keys,
        q,
        k: kr,
        v,
        q_out,
        attn,
        ffn_inp,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_ATTN)?;
    let q_out = q_out.as_mut().ok_or(GpuError::state(
        WHAT_ATTN,
        "the arena's buffer of gated queries",
    ))?;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    wd.act_hid
        .quantize(c, normed, m, &[n.q_ty, n.k_ty, n.v_ty])?;
    let table = (&wd.dense, GemmInput::PerSlot);
    let hid = wd.act_hid.input(normed);
    gemm(c, (n.q_ty, &n.attn_q), d.q_rows, &hid, table, m, q)?;
    gemm(c, (n.k_ty, &n.attn_k), d.kv_len(), &hid, table, m, kr)?;
    gemm(c, (n.v_ty, &n.attn_v), d.kv_len(), &hid, table, m, v)?;
    k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
            qg: q,
            q: q_out,
            k: kr,
            v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            table: c.table,
            pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            fault: c.sink,
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    let args = GqaPrefillArgs {
        q: q_out,
        kc: &kv.k,
        vc: &kv.v,
        n_keys,
        scale: ATTN_SCALE_256,
        n_head: d.n_head,
        n_kv: d.n_kv,
        ctx: d.ctx,
        t: m,
        fault: c.sink,
        y: attn,
    };
    match n.flash {
        Flash::Group => k.prefill.enqueue_256(stream, args)?,
        Flash::Quads => k.prefill.enqueue_256_p4(stream, args)?,
        Flash::Pairs => k.prefill.enqueue_256_p2(stream, args)?,
    }
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wd;
    let gated: &DeviceBuffer<f32> = if n.o_ty.kquant() {
        let a = act_attn.q128.as_mut().ok_or(GpuError::state(
            WHAT_ATTN,
            "the wide part's q8_1 blocks of the attention rows",
        ))?;
        q35.gated.enqueue_gemm(
            stream,
            (attn, q),
            GateLayout {
                head: d.head,
                head_stride: 2 * d.head,
                offset: d.head,
                col_stride: d.q_rows,
            },
            a,
            m,
            c.sink,
        )?;
        attn
    } else {
        // The flash has read the queries: their buffer takes the gated rows.
        q35.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn,
                qg: q,
                n_head: d.n_head,
                m,
                fault: c.sink,
                y: q_out,
            },
        )?;
        act_attn.quantize(c, q_out, m, &[n.o_ty])?;
        q_out
    };
    let input = act_attn.input(gated);
    gemm(
        c,
        (n.o_ty, &n.attn_output),
        d.hidden,
        &input,
        (dense, GemmInput::PerSlot),
        m,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * d.hidden, ffn_inp)
}

/// A delta layer's input half at `m` rows: `x` normed and quantized into the
/// forms its projections read, then its four projections — `attn_qkv`,
/// `attn_gate`, β and α — into the arena's `[x | z | b | a]` blocks, each a
/// GEMM of its type.
pub(super) fn delta_in(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_IN: &str = "qwen3moe::wide::delta_in";
    let (gpu, w) = (c.gpu, c.w);
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let Arena {
        x,
        normed,
        gdn,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_IN)?;
    let g: &mut GdnArena = gdn
        .as_mut()
        .ok_or(GpuError::state(WHAT_IN, "the arena's delta intermediates"))?;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        f32_gain(w, &d.attn_norm)?,
        c.eps,
        hidden,
        m,
        normed,
    )?;
    wd.act_hid
        .quantize(c, normed, m, &[d.qkv_ty, d.gate_ty, d.beta_ty, d.alpha_ty])?;
    let (a, t) = (wd.act_hid.input(normed), (&wd.dense, GemmInput::PerSlot));
    let (ch, zl, nv) = (d.shape.channels(), d.shape.n_v * linear::HEAD, d.shape.n_v);
    gemm(c, (d.qkv_ty, &d.qkv), ch, &a, t, m, &mut g.x)?;
    gemm(c, (d.gate_ty, &d.gate), zl, &a, t, m, &mut g.z)?;
    gemm(c, (d.beta_ty, &d.beta), nv, &a, t, m, &mut g.b)?;
    gemm(c, (d.alpha_ty, &d.alpha), nv, &a, t, m, &mut g.a)
}

/// A delta layer's output half at `m` rows: the gated norm's rows (the
/// arena's `attn`) quantized into the form `ssm_out` reads, `ssm_out`, then
/// `ffn_inp = x + ssm_out(·)`.
pub(super) fn delta_out(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_OUT: &str = "qwen3moe::wide::delta_out";
    let gpu = c.gpu;
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let Arena {
        x,
        attn,
        ffn_inp,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_OUT)?;
    wd.act_attn.quantize(c, attn, m, &[d.out_ty])?;
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wd;
    gemm(
        c,
        (d.out_ty, &d.ssm_out),
        hidden,
        &act_attn.input(attn),
        (dense, GemmInput::PerSlot),
        m,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * hidden, ffn_inp)
}

/// The FFN half at `m` rows: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). The gated router's slots (the shared expert the last
/// of each token's, its id the joined stacks' last expert) through the
/// layer's expert table, or a dense FFN's one slot a token through the
/// unit's one-expert table at the arena's fixed weight 1; the SwiGLU's
/// quantizer is the one the down's type reads. Qwen3's plain router takes
/// its wide rows through `ubatch.rs` and is refused here by name.
pub(super) fn ffn(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    if let FfnRoute::Router { shared: None, .. } = n.route {
        return Err(GpuError::shape(
            WHAT_FFN,
            format!(
                "layer {}: the plain router at {m} rows (Qwen3 runs more than GEMV_COLS rows \
                 through ubatch.rs)",
                c.layer
            ),
        ));
    }
    let used = d.router.map_or(0, |r| r.used());
    if slots != c.p.slots(used) {
        return Err(GpuError::shape(
            WHAT_FFN,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(used)
            ),
        ));
    }
    let Arena {
        x,
        normed,
        ffn_inp,
        route,
        h,
        down,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_FFN)?;
    gpu.elem().enqueue_rms_norm(
        stream,
        ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    wd.act_hid.quantize(c, normed, m, &[n.gate_ty, n.up_ty])?;
    let Wide {
        act_hid,
        act_h,
        up,
        dense,
        moe,
        ..
    } = wd;
    let (table, input): (&GemmRoute, GemmInput) = match &n.route {
        FfnRoute::Router { gate_inp, .. } => {
            k.q35(WHAT_FFN)?.router.enqueue_ubatch(
                stream,
                f32_tensor(w, gate_inp)?,
                normed,
                m,
                c.sink,
                route.gated(WHAT_FFN)?,
            )?;
            let moe = moe.as_mut().ok_or(GpuError::state(
                WHAT_FFN,
                "the wide part's expert table (a routed chain's)",
            ))?;
            k.gemm
                .enqueue_route(stream, route.ids(), m * slots, moe, c.sink)?;
            (moe, GemmInput::Shared { top_k: slots })
        }
        FfnRoute::Dense => (dense, GemmInput::PerSlot),
    };
    let hid = act_hid.input(normed);
    gemm(c, (n.gate_ty, &n.gate), d.ff, &hid, (table, input), m, h)?;
    gemm(c, (n.up_ty, &n.up), d.ff, &hid, (table, input), m, up)?;
    let n_slots = m * slots;
    match n.down_ty.reads() {
        Form::Q8x128 => {
            let a = act_h.q128.as_mut().ok_or(GpuError::state(
                WHAT_FFN,
                "the wide part's q8_1 blocks of the SwiGLU rows",
            ))?;
            k.gemm
                .enqueue_swiglu_quant(stream, h, up, n_slots, a, c.sink)?;
        }
        Form::Q8x32 => {
            let a = act_h.q32.as_mut().ok_or(GpuError::state(
                WHAT_FFN,
                "the wide part's q8 blocks of the SwiGLU rows",
            ))?;
            k.q35(WHAT_FFN)?
                .g32
                .enqueue_swiglu_quant32(stream, h, up, n_slots, a, c.sink)?;
        }
        Form::F32 => {
            return Err(GpuError::shape(
                WHAT_FFN,
                format!(
                    "layer {}: an F32 down, which the load refuses (no SwiGLU writes its rows)",
                    c.layer
                ),
            ));
        }
    }
    gemm(
        c,
        (n.down_ty, &n.down),
        d.hidden,
        &act_h.input(h),
        (table, GemmInput::PerSlot),
        n_slots,
        down,
    )?;
    let y = match out {
        Some(y) => y,
        None => x,
    };
    k.experts.enqueue_combine_tokens(
        stream,
        CombineArgs {
            down,
            w: route.weights(),
            resid: ffn_inp,
            rows: d.hidden,
            n_slots: slots,
            m,
            y,
        },
    )
}
