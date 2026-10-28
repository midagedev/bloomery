//! A Raon gated MLA layer on the card: absorbed latent attention
//! with no positional encoding, its `rope.dimension_count` channels an
//! unrotated extra key part shared by the heads, and a sigmoid output gate
//! (llama.cpp `src/models/raon1.cpp` `build_mla_layer`, the HF
//! `Raon1MiniMLA`).
//!
//! The launches, in order, over common owners:
//! 1. `attn_norm` RMS of `x`;
//! 2. `attn_q_a` and `attn_kv_a_mqa` of the normed row (through one q8_1
//!    form when they are K-quants) and `attn_gate` (its own form);
//! 3. `attn_q_a_norm` RMS of the q_a rows (the latent ε);
//! 4. `attn_q_b`: each head's `[nope | extra]` query;
//!
//! then [`attend`], the core the kernel gate runs:
//! 5. the cache row at the position: the raw extra part, then the latent's
//!    RMS with `attn_kv_a_norm` (`latent::enqueue_latent_extra_append`), f16;
//! 6. `attn_k_b` per head: each head's nope query absorbed into the latent
//!    width, behind its extra part (`enqueue_q8_0_gemv_heads`);
//! 7. each head's extra query part copied in front (`enqueue_gather` over
//!    load-time tables);
//! 8. the latent attention over the cache rows `0 ..= pos`
//!    (`flash::enqueue_flash_latent_split`, one or two launches), the row
//!    `[extra | latent]` the key and its latent the value, at the scale
//!    `1/√(nope + extra)`;
//! 9. `attn_v_b` per head: each head's latent output to its values
//!    (`q6k_heads` for Q6_K, `enqueue_q8_0_gemv_heads` for Q8_0);
//! 10. `attn · σ(gate)` (`linear::out_gate`);
//!
//! and last, 11. `attn_output` into `out`.
//!
//! [`step`] runs one row. [`rows_wide`] runs a prompt unit of up to
//! [`WIDE_ROWS`] rows: the projections as GEMMs, the cache append, the gather,
//! the absorb ([`absorb_heads`], one 32-value GEMM over every (row, head)),
//! the value projection and the output gate once over the unit, and the
//! latent attention in chunks of [`ATTN_CHUNK`] query tokens. [`rows`] runs a
//! pass of 1 to [`PASS_ROWS`] rows in position order — a verify's rows or a prompt chunk — through the same
//! launches, each once over the pass's rows: the projections' m-column
//! launches ([`super::rda::project_rows`]), the cache rows of every row
//! appended before any attends, the absorb and a Q8_0 value projection over
//! the rows' windows (`q8_0_gemv_heads_mcol`; a Q6_K one is `q6k_heads` a row
//! at a time), the gather over tables built for [`PASS_ROWS`] rows, and the
//! latent attention over the `m` query tokens, token `t` over keys
//! `0 ..= pos + t`. Row `t` of each launch is the step's launch on row `t`
//! bit for bit, and so is the attention's while the pass's live count stays
//! at or under `SEGMENTS · seg_keys` (`crate::flash`): the split cuts every
//! group at the fixed `seg_keys`-key span there. Past it a token whose group
//! reaches a wider cut than its own step's sums its keys in another order: a
//! float-order band, not a bit pin.
//!
//! What the file forces here: the NoPE extra part and the output gate; the
//! widths are the header's ([`MlaDims::of`]).

use super::rda::{RowsKit, kgemv_rows, project_rows};
use super::state::PASS_ROWS;
use super::wide::{
    AbsorbShape, Input, WIDE_ROWS, WideKit, absorb_heads, input, out_runs, qb_runs, quantize_hid,
    quantize_out, quantize_qb, route_heads, value_heads,
};
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::fault::FaultSink;
use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs, partials_ms_len, partials_v_len};
use crate::latent::{LATENT, LatentExtraAppendArgs, LatentKernels, Rows};
use crate::linear::out_gate::{OutGateArgs, OutGateKernels};
use crate::model::{Q8_0GemvHeadsArgs, StepKernels};
use crate::q6k_heads::{Q6kHeadsArgs, Q6kHeadsKernels};
use crate::q8f32::Q8_0GemvHeadsMcolArgs;
use crate::site::{self, Form, SiteTy, WideActs};
use crate::tensor::{DeviceTensor, Window, WindowMut};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::OutGate;
use model::arch::raon1::names;

/// What the refusals of this file name.
const WHAT: &str = "raon1 mla";

/// The query tokens a wide unit's latent flash takes a launch: the partials
/// ([`partials_lens_rows`]) hold this many tokens' rows a segment, and an
/// even count keeps each launch's groups of sixteen rows in the same token
/// pairs whatever the chunk (`heads` is 40 rows a token).
pub(super) const ATTN_CHUNK: usize = 32;

const _: () = assert!(ATTN_CHUNK >= PASS_ROWS && ATTN_CHUNK.is_multiple_of(2));
const _: () = assert!(WIDE_ROWS.is_multiple_of(ATTN_CHUNK));

/// Types a dense projection of the layer may have.
const DENSE: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The layer's widths, from the file's header.
#[derive(Clone, Copy, Debug)]
pub struct MlaDims {
    pub embd: usize,
    pub heads: usize,
    pub q_lora: usize,
    /// A query head's values before the absorption.
    pub nope: usize,
    /// The unrotated key values beside the latent, shared by the heads.
    pub extra: usize,
    pub v_head: usize,
    pub rms_eps: f32,
    /// The latent norms' ε (`attention.latent_norm_rms_epsilon`).
    pub latent_eps: f32,
}

impl MlaDims {
    /// The key width a score's scale is `1/√` of.
    #[must_use]
    pub fn head_k(&self) -> usize {
        self.nope + self.extra
    }

    /// The cache row: the extra part, then the latent.
    #[must_use]
    pub fn row(&self) -> usize {
        self.extra + LATENT
    }

    /// Refuse by name a shape no launch here runs.
    pub fn check(&self) -> Result<(), GpuError> {
        if self.v_head != crate::linear::HEAD {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the output gate's entry is 128 a head, got v_head {}",
                    self.v_head
                ),
            ));
        }
        if self.row() != 576 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the latent attention's tensor-core pass reads 576-wide rows, got extra {} + \
                     latent {LATENT}",
                    self.extra
                ),
            ));
        }
        Ok(())
    }
}

/// How a layer's per-head value projection runs, by the file type of
/// `attn_v_b`.
enum ValueHeads {
    Q6K(String),
    Q8_0(String),
}

/// A resident per-head value projection, as [`attend`] reads it.
pub enum ValueW<'a> {
    /// Q6_K words, `heads·v_head` rows of `LATENT`.
    Q6K(&'a DeviceTensor<u32>),
    /// Q8_0 planes (`qs`, `d`).
    Q8_0(&'a DeviceTensor<u32>, &'a DeviceTensor<u16>),
}

/// One MLA layer's widths, projections and the names of its vectors.
pub(crate) struct MlaSites {
    q_a: Site,
    kv_a: Site,
    gate: Site,
    q_b: Site,
    out: Site,
    /// `attn_k_b`, Q8_0: each head's `LATENT` rows over its nope values.
    k_b: String,
    v_b: ValueHeads,
    /// attn_norm, attn_q_a_norm, attn_kv_a_norm.
    vecs: [String; 3],
}

impl MlaSites {
    /// Layer `l`'s sites at `d`'s widths, each type read from `file`'s
    /// header and checked against the resident weight. Load-time only.
    pub(crate) fn load(
        file: &Split,
        w: &Weights,
        d: &MlaDims,
        l: usize,
    ) -> Result<MlaSites, GpuError> {
        d.check()?;
        let typed = |name: String, (rows, k): (usize, usize)| -> Result<Site, GpuError> {
            let ty = site::file_site(file, WHAT, &name, (rows, k), DENSE)?;
            site::site(w, WHAT, &name, (rows, k), ty)?;
            Ok(Site::new(name, ty, k))
        };
        let vector = |name: String, len: usize| site::vector(w, WHAT, name, len);
        let ty_of = |name: &str| file.find(name).map(|(_, t)| t.ty);
        let k_b = names::attn_k_b(l);
        if ty_of(&k_b) != Some(GgmlType::Q8_0) {
            return Err(GpuError::shape(
                WHAT,
                format!("{k_b} is {:?}; the per-head absorb runs Q8_0", ty_of(&k_b)),
            ));
        }
        let v_name = names::attn_v_b(l);
        let v_b = match ty_of(&v_name) {
            Some(GgmlType::Q6_K) => ValueHeads::Q6K(v_name),
            Some(GgmlType::Q8_0) => ValueHeads::Q8_0(v_name),
            other => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{v_name} is {other:?}; the per-head value runs Q6_K or Q8_0"),
                ));
            }
        };
        let (embd, hv) = (d.embd, d.heads * d.v_head);
        Ok(MlaSites {
            q_a: typed(names::attn_q_a(l), (d.q_lora, embd))?,
            kv_a: typed(names::attn_kv_a_mqa(l), (d.row(), embd))?,
            gate: typed(names::attn_gate(l), (hv, embd))?,
            q_b: typed(names::attn_q_b(l), (d.heads * d.head_k(), d.q_lora))?,
            out: typed(names::attn_output(l), (embd, hv))?,
            k_b,
            v_b,
            vecs: [
                vector(names::attn_norm(l), embd)?,
                vector(names::attn_q_a_norm(l), d.q_lora)?,
                vector(names::attn_kv_a_norm(l), LATENT)?,
            ],
        })
    }

    /// Every site the layer launches, for the load's [`Acts::new`].
    pub(crate) fn sites(&self) -> [&Site; 5] {
        [&self.q_a, &self.kv_a, &self.gate, &self.q_b, &self.out]
    }

    /// `Ok` when the wide arm runs the layer's projections ([`rows_wide`]), else
    /// the refusal it raises: `attn_q_b` and the output projection read q8
    /// blocks of 32.
    pub(super) fn wide_runs(&self) -> Result<(), GpuError> {
        qb_runs(self.q_b.ty())?;
        out_runs(self.out.ty())
    }
}

/// The kernels every MLA layer launches beyond the `Gpu`'s own, and the
/// extra part's gather tables for up to [`WIDE_ROWS`] rows
/// (`[(t·heads + h)·head_k + nope + j] → [(t·heads + h)·row + j]`; a call of
/// `m` rows reads the first `m·heads·extra` pairs).
pub struct MlaKernels {
    step: StepKernels,
    latent: LatentKernels,
    q6k: Q6kHeadsKernels,
    out_gate: OutGateKernels,
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
}

impl MlaKernels {
    /// Load the modules and build the gather tables for `d`. Load-time only.
    pub fn load(gpu: &Gpu, d: &MlaDims) -> Result<MlaKernels, GpuError> {
        d.check()?;
        let at = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(WHAT, "a gather index past u32"))
        };
        let pairs = WIDE_ROWS * d.heads * d.extra;
        let (mut src, mut dst) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
        for head in 0..WIDE_ROWS * d.heads {
            for j in 0..d.extra {
                src.push(at(head * d.head_k() + d.nope + j)?);
                dst.push(at(head * d.row() + j)?);
            }
        }
        let (ctx, stream) = (gpu.context(), gpu.stream());
        Ok(MlaKernels {
            step: StepKernels::load(ctx)?,
            latent: LatentKernels::load(ctx)?,
            q6k: Q6kHeadsKernels::load(ctx)?,
            out_gate: OutGateKernels::load(ctx)?,
            src: DeviceBuffer::from_host(stream, &src)?,
            dst: DeviceBuffer::from_host(stream, &dst)?,
        })
    }
}

/// The lengths of the attention's partials (`part_v`, `part_ms`) for one row
/// of `d.heads` heads over a cache of `ctx` rows.
#[must_use]
pub fn partials_lens(d: &MlaDims, ctx: usize) -> (usize, usize) {
    partials_lens_rows(d, ctx, 1)
}

/// [`partials_lens`] for a pass of `m` rows.
#[must_use]
pub fn partials_lens_rows(d: &MlaDims, ctx: usize, m: usize) -> (usize, usize) {
    let q_rows = m * d.heads;
    (partials_v_len(q_rows, ctx), partials_ms_len(q_rows, ctx))
}

/// f32s of the stage an MLA layer's pass of up to [`PASS_ROWS`] rows writes
/// (`rda::RowsKit`): the widest projection's outputs of every row.
#[must_use]
pub(crate) fn stage_len(d: &MlaDims) -> usize {
    let widest = [
        d.q_lora,
        d.row(),
        d.heads * d.v_head,
        d.heads * d.head_k(),
        d.embd,
    ];
    PASS_ROWS * widest.into_iter().max().unwrap_or(0)
}

/// The buffers a pass of `m` rows of an MLA layer reads and writes (one row
/// at a step), each at least `m` rows of its width, token-major: `x` the
/// layer's inputs, `pos` the rows' positions and `n_keys` their live key
/// counts (`pos + 1` a row; the attention reads row `m − 1`'s), the
/// intermediates, the attention's partials ([`partials_lens_rows`]), `out`
/// the updates, and the layer's cache (`[ctx][row]` f16).
pub(crate) struct MlaBufs<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    /// `[embd]`.
    pub xn: &'a mut DeviceBuffer<f32>,
    /// `[q_lora]` each.
    pub qa: &'a mut DeviceBuffer<f32>,
    pub qr: &'a mut DeviceBuffer<f32>,
    /// `[row]`: the latent, then the extra part.
    pub kv: &'a mut DeviceBuffer<f32>,
    /// `[heads·v_head]`.
    pub gate: &'a mut DeviceBuffer<f32>,
    /// `[heads·head_k]`.
    pub q: &'a mut DeviceBuffer<f32>,
    /// `[heads·row]`.
    pub qabs: &'a mut DeviceBuffer<f32>,
    /// `[heads·LATENT]`.
    pub att: &'a mut DeviceBuffer<f32>,
    /// `[heads·v_head]` each.
    pub av: &'a mut DeviceBuffer<f32>,
    pub gated: &'a mut DeviceBuffer<f32>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    /// `[embd]`.
    pub out: &'a mut DeviceBuffer<f32>,
    pub cache: &'a mut DeviceTensor<u16>,
}

/// [`attend`]'s own buffers: the absorbed queries (`[heads·row]`), the
/// latent outputs (`[heads·LATENT]`), the value heads' outputs
/// (`[heads·v_head]`) and the attention's partials ([`partials_lens`]).
pub struct AttnBufs<'a> {
    pub qabs: &'a mut DeviceBuffer<f32>,
    pub att: &'a mut DeviceBuffer<f32>,
    pub av: &'a mut DeviceBuffer<f32>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
}

/// The Q8_0 planes of `name`, or a named refusal.
fn q8_planes<'w>(
    w: &'w Weights,
    name: &str,
) -> Result<(&'w DeviceTensor<u32>, &'w DeviceTensor<u16>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(GpuError::tensor(WHAT, name, "Q8_0 planes")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}

/// The resident value projection `v`, or a named refusal.
fn value_w<'w>(w: &'w Weights, v: &ValueHeads) -> Result<ValueW<'w>, GpuError> {
    match v {
        ValueHeads::Q6K(name) => match w.get(name) {
            Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w: wv,
                ..
            }) => Ok(ValueW::Q6K(wv)),
            _ => Err(GpuError::tensor(WHAT, name, "resident Q6_K words")),
        },
        ValueHeads::Q8_0(name) => {
            let (qs, d) = q8_planes(w, name)?;
            Ok(ValueW::Q8_0(qs, d))
        }
    }
}

/// [`attend`]'s inputs: one row's projections, the layer's weights and the
/// row's position and live key count (`pos + 1`).
pub struct AttendIn<'a> {
    /// `[row]`: `attn_kv_a_mqa`'s output, the latent then the extra part.
    pub kv: &'a DeviceBuffer<f32>,
    /// `[heads·head_k]`: `attn_q_b`'s output, `[nope | extra]` a head.
    pub q: &'a DeviceBuffer<f32>,
    /// `[heads·v_head]`: `attn_gate`'s output.
    pub gate: &'a DeviceBuffer<f32>,
    /// `attn_kv_a_norm`, `[LATENT]`.
    pub kv_norm: &'a DeviceBuffer<f32>,
    /// `attn_k_b`'s Q8_0 planes.
    pub k_b: (&'a DeviceTensor<u32>, &'a DeviceTensor<u16>),
    pub v_b: ValueW<'a>,
    pub pos: &'a DeviceBuffer<u32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub fault: FaultSink,
}

/// Steps 5 to 10 of the module doc for one row: the cache row at `pos`
/// written into `cache`, the gated heads into `y` (`[heads·v_head]`).
pub fn attend(
    gpu: &Gpu,
    k: &MlaKernels,
    d: &MlaDims,
    a: AttendIn<'_>,
    s: AttnBufs<'_>,
    cache: &mut DeviceTensor<u16>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    k.latent.enqueue_latent_extra_append(
        stream,
        LatentExtraAppendArgs {
            rows: Rows {
                x: a.kv,
                stride: d.row(),
                m: 1,
            },
            off: 0,
            e_off: LATENT,
            extra: d.extra,
            gain: a.kv_norm,
            pos: a.pos,
            eps: d.latent_eps,
            fault: a.fault,
            cache: &mut *cache,
        },
    )?;
    let (qs, dd) = a.k_b;
    k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: a.q,
            rows_per_head: LATENT,
            x_head_stride: d.head_k(),
            y_head_stride: d.row(),
            y_off: d.extra,
            y: &mut *s.qabs,
        },
    )?;
    k.step
        .enqueue_gather(stream, a.q, &k.src, &k.dst, d.heads * d.extra, s.qabs)?;
    gpu.flash().enqueue_flash_latent_split(
        stream,
        FlashSplitArgs {
            inputs: FlashInputs {
                q: s.qabs,
                kv: cache,
                n_keys_buf: a.n_keys,
                kq_scale: 1.0 / (d.head_k() as f32).sqrt(),
                geom: FlashGeom {
                    tokens: 1,
                    heads: d.heads,
                    rope_dims: d.extra,
                    latent_dims: LATENT,
                },
            },
            part_v: s.part_v,
            part_ms: s.part_ms,
            y: &mut *s.att,
        },
    )?;
    match a.v_b {
        ValueW::Q6K(wv) => k.q6k.enqueue_q6k_heads(
            stream,
            Q6kHeadsArgs {
                w: wv,
                x: s.att,
                k: LATENT,
                rows_per_head: d.v_head,
                x_head_stride: LATENT,
                y_head_stride: d.v_head,
                y_off: 0,
                y: &mut *s.av,
            },
        )?,
        ValueW::Q8_0(qs, dd) => k.step.enqueue_q8_0_gemv_heads(
            stream,
            Q8_0GemvHeadsArgs {
                qs,
                d: dd,
                x: s.att,
                rows_per_head: d.v_head,
                x_head_stride: LATENT,
                y_head_stride: d.v_head,
                y_off: 0,
                y: &mut *s.av,
            },
        )?,
    }
    k.out_gate.enqueue_out_gate(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: s.av,
            gate: a.gate,
            n_head: d.heads,
            m: 1,
            fault: a.fault,
            y,
        },
    )
}

/// [`attend`] over a pass of `m` rows, 1 to [`PASS_ROWS`] (module doc): `a`'s
/// rows and `s`'s buffers each at least `m` rows of their widths, `a.pos`
/// the rows' positions and `a.n_keys` their live counts; the cache rows at
/// the positions written into `cache`, the gated heads of every row into `y`
/// (`[m][heads·v_head]`). One row is [`attend`]; a count past [`PASS_ROWS`],
/// or rows the gather's tables would read or write past, is refused by name.
#[allow(
    clippy::too_many_arguments,
    reason = "attend's six and the pass's row count, each named as attend names it (rust-quality R8)"
)]
pub fn attend_rows(
    gpu: &Gpu,
    k: &MlaKernels,
    d: &MlaDims,
    m: usize,
    a: AttendIn<'_>,
    s: AttnBufs<'_>,
    cache: &mut DeviceTensor<u16>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if m == 1 {
        return attend(gpu, k, d, a, s, cache, y);
    }
    let (hk, row, hl, hv) = (
        d.heads * d.head_k(),
        d.heads * d.row(),
        d.heads * LATENT,
        d.heads * d.v_head,
    );
    if !(2..=PASS_ROWS).contains(&m) || a.q.len() < m * hk || s.qabs.len() < m * row {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a pass of {m} rows (1 to {PASS_ROWS}) over q of {} and qabs of {}: the gather \
                 reads {} and writes {}",
                a.q.len(),
                s.qabs.len(),
                m * hk,
                m * row
            ),
        ));
    }
    let stream = gpu.stream();
    k.latent.enqueue_latent_extra_append(
        stream,
        LatentExtraAppendArgs {
            rows: Rows {
                x: a.kv,
                stride: d.row(),
                m,
            },
            off: 0,
            e_off: LATENT,
            extra: d.extra,
            gain: a.kv_norm,
            pos: a.pos,
            eps: d.latent_eps,
            fault: a.fault,
            cache: &mut *cache,
        },
    )?;
    let q8 = gpu.q8f32();
    let (qs, dd) = a.k_b;
    q8.enqueue_q8_0_gemv_heads_mcol(
        stream,
        Q8_0GemvHeadsMcolArgs {
            qs,
            d: dd,
            x: a.q,
            rows_per_head: LATENT,
            x_head_stride: d.head_k(),
            y_head_stride: d.row(),
            y_off: d.extra,
            m,
            x_col_stride: hk,
            y_col_stride: row,
            y: &mut *s.qabs,
        },
    )?;
    let pairs = m * d.heads * d.extra;
    k.step
        .enqueue_gather(stream, a.q, &k.src, &k.dst, pairs, s.qabs)?;
    let n_keys = Window::<u32>::of(a.n_keys, 4 * (m - 1), 1)?;
    gpu.flash().enqueue_flash_latent_split(
        stream,
        FlashSplitArgs {
            inputs: FlashInputs {
                q: s.qabs,
                kv: cache,
                n_keys_buf: &n_keys,
                kq_scale: 1.0 / (d.head_k() as f32).sqrt(),
                geom: FlashGeom {
                    tokens: m,
                    heads: d.heads,
                    rope_dims: d.extra,
                    latent_dims: LATENT,
                },
            },
            part_v: s.part_v,
            part_ms: s.part_ms,
            y: &mut *s.att,
        },
    )?;
    match a.v_b {
        ValueW::Q6K(wv) => {
            for t in 0..m {
                let x = Window::<f32>::of(&*s.att, 4 * t * hl, hl)?;
                let mut yv = WindowMut::<f32>::of_mut(&mut *s.av, 4 * t * hv, hv)?;
                k.q6k.enqueue_q6k_heads(
                    stream,
                    Q6kHeadsArgs {
                        w: wv,
                        x: &x,
                        k: LATENT,
                        rows_per_head: d.v_head,
                        x_head_stride: LATENT,
                        y_head_stride: d.v_head,
                        y_off: 0,
                        y: &mut yv,
                    },
                )?;
            }
        }
        ValueW::Q8_0(qs, dd) => q8.enqueue_q8_0_gemv_heads_mcol(
            stream,
            Q8_0GemvHeadsMcolArgs {
                qs,
                d: dd,
                x: s.att,
                rows_per_head: d.v_head,
                x_head_stride: LATENT,
                y_head_stride: d.v_head,
                y_off: 0,
                m,
                x_col_stride: hl,
                y_col_stride: hv,
                y: &mut *s.av,
            },
        )?,
    }
    k.out_gate.enqueue_out_gate(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: s.av,
            gate: a.gate,
            n_head: d.heads,
            m,
            fault: a.fault,
            y,
        },
    )
}

/// Enqueue MLA layer `l` at one row on `b`'s buffers (module doc).
pub(crate) fn step(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    l: usize,
    b: MlaBufs<'_>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let [norm, q_a_norm, kv_a_norm] = &sites.vecs;
    gpu.elem().enqueue_rms_norm(
        stream,
        b.x,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        1,
        b.xn,
    )?;
    let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
    let input: [(&Site, &mut DeviceBuffer<f32>); 3] = [
        (&sites.q_a, &mut *b.qa),
        (&sites.kv_a, &mut *b.kv),
        (&sites.gate, &mut *b.gate),
    ];
    let mut rest = Vec::with_capacity(3);
    if input.iter().any(|(t, _)| q8(t)) {
        let act = acts.q8_1_at(WHAT, d.embd, 1)?;
        gpu.enqueue_quantize_q8_1_layer(b.xn, act, l)?;
        for (t, y) in input {
            if q8(t) {
                t.kgemv(cx, act, l, y)?;
            } else {
                rest.push((t, y));
            }
        }
    } else {
        rest.extend(input);
    }
    for (t, y) in rest {
        t.project(cx, acts, l, b.xn, y)?;
    }
    gpu.elem().enqueue_rms_norm(
        stream,
        b.qa,
        w.f32_buf(WHAT, q_a_norm)?,
        d.latent_eps,
        d.q_lora,
        1,
        b.qr,
    )?;
    sites.q_b.project(cx, acts, l, b.qr, b.q)?;
    attend(
        gpu,
        k,
        d,
        AttendIn {
            kv: b.kv,
            q: b.q,
            gate: b.gate,
            kv_norm: w.f32_buf(WHAT, kv_a_norm)?,
            k_b: q8_planes(w, &sites.k_b)?,
            v_b: value_w(w, &sites.v_b)?,
            pos: b.pos,
            n_keys: b.n_keys,
            fault,
        },
        AttnBufs {
            qabs: b.qabs,
            att: b.att,
            av: b.av,
            part_v: b.part_v,
            part_ms: b.part_ms,
        },
        b.cache,
        b.gated,
    )?;
    sites.out.project(cx, acts, l, b.gated, b.out)
}

/// Enqueue MLA layer `l` over a pass of `m` rows, 1 to [`PASS_ROWS`], in
/// position order (module doc) on `b`'s buffers, each at least `m` rows of
/// its width ([`MlaBufs`]), the projections' row-major outputs staged in
/// `kit`. One row is [`step`]; a count past [`PASS_ROWS`] is refused by name.
pub(crate) fn rows(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    (l, m): (usize, usize),
    b: MlaBufs<'_>,
    mut kit: RowsKit<'_>,
) -> Result<(), GpuError> {
    if m == 1 {
        return step(cx, k, acts, (sites, d), l, b);
    }
    if !(2..=PASS_ROWS).contains(&m) {
        return Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a pass of {m} rows; a pass runs 1 to {PASS_ROWS}"),
        ));
    }
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let [norm, q_a_norm, kv_a_norm] = &sites.vecs;
    gpu.elem().enqueue_rms_norm(
        stream,
        b.x,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        m,
        b.xn,
    )?;
    let q8 = |t: &Site| t.ty().reads() == Form::Q8x128;
    let input: [(&Site, usize, &mut DeviceBuffer<f32>); 3] = [
        (&sites.q_a, d.q_lora, &mut *b.qa),
        (&sites.kv_a, d.row(), &mut *b.kv),
        (&sites.gate, d.heads * d.v_head, &mut *b.gate),
    ];
    let mut rest = Vec::with_capacity(3);
    if input.iter().any(|(t, ..)| q8(t)) {
        let act = acts.q8_1_at(WHAT, d.embd, m)?;
        gpu.enqueue_quantize_q8_1_layer(b.xn, act, l)?;
        for (t, n, y) in input {
            if q8(t) {
                kgemv_rows(cx, &mut kit, (t, n), act, l, y)?;
            } else {
                rest.push((t, n, y));
            }
        }
    } else {
        rest.extend(input);
    }
    for (t, n, y) in rest {
        project_rows(cx, &mut kit, acts, (l, m), (t, n), b.xn, y)?;
    }
    gpu.elem().enqueue_rms_norm(
        stream,
        b.qa,
        w.f32_buf(WHAT, q_a_norm)?,
        d.latent_eps,
        d.q_lora,
        m,
        b.qr,
    )?;
    let q_b = (&sites.q_b, d.heads * d.head_k());
    project_rows(cx, &mut kit, acts, (l, m), q_b, b.qr, b.q)?;
    attend_rows(
        gpu,
        k,
        d,
        m,
        AttendIn {
            kv: b.kv,
            q: b.q,
            gate: b.gate,
            kv_norm: w.f32_buf(WHAT, kv_a_norm)?,
            k_b: q8_planes(w, &sites.k_b)?,
            v_b: value_w(w, &sites.v_b)?,
            pos: b.pos,
            n_keys: b.n_keys,
            fault,
        },
        AttnBufs {
            qabs: b.qabs,
            att: b.att,
            av: b.av,
            part_v: b.part_v,
            part_ms: b.part_ms,
        },
        b.cache,
        b.gated,
    )?;
    let out = (&sites.out, d.embd);
    project_rows(cx, &mut kit, acts, (l, m), out, b.gated, b.out)
}

/// Enqueue MLA layer `l` over a wide unit of `m` rows (above [`PASS_ROWS`]),
/// in position order: the projections as GEMMs over the unit's rows
/// ([`super::wide`]); the cache append, the gather of the extra part, the
/// key absorption ([`absorb_heads`]: one GEMM over every (row, head) slot of
/// the unit), a Q6_K per-head value projection ([`value_heads`]: one grouped
/// GEMM over the same slots) and the output gate once over the unit; and the
/// latent flash in chunks of [`ATTN_CHUNK`] query tokens on windows of the
/// unit's buffers, each chunk over the cache rows the unit's append wrote,
/// bounded by its own live counts (a Q8_0 value projection runs per chunk, in
/// windows of [`PASS_ROWS`] rows). `b`'s buffers hold at least `m` rows of
/// their widths, the partials [`ATTN_CHUNK`] rows.
pub(super) fn rows_wide(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    kit: &mut WideKit<'_>,
    (sites, d): (&MlaSites, &MlaDims),
    (l, m): (usize, usize),
    b: MlaBufs<'_>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let [norm, q_a_norm, kv_a_norm] = &sites.vecs;
    gpu.elem().enqueue_rms_norm(
        stream,
        b.x,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        m,
        b.xn,
    )?;
    let (wk, wb) = (&kit.k, &mut kit.b);
    let tys = [sites.q_a.ty(), sites.kv_a.ty(), sites.gate.ty()];
    quantize_hid(
        gpu,
        wk.g32,
        wb.forms,
        (&*b.xn, m),
        (fault, WideActs::wants_of(&tys)),
    )?;
    {
        let x_in = input(wb.forms, Input::Hid, &*b.xn);
        wk.project(cx, (&sites.q_a, d.q_lora), &x_in, m, b.qa)?;
        wk.project(cx, (&sites.kv_a, d.row()), &x_in, m, b.kv)?;
        wk.project(cx, (&sites.gate, d.heads * d.v_head), &x_in, m, b.gate)?;
    }
    gpu.elem().enqueue_rms_norm(
        stream,
        b.qa,
        w.f32_buf(WHAT, q_a_norm)?,
        d.latent_eps,
        d.q_lora,
        m,
        b.qr,
    )?;
    quantize_qb(gpu, wk.g32, wb.forms, (&*b.qr, m), (fault, sites.q_b.ty()))?;
    {
        let x_in = input(wb.forms, Input::Qb, &*b.qr);
        wk.project(cx, (&sites.q_b, d.heads * d.head_k()), &x_in, m, b.q)?;
    }
    let (row, hl, hv) = (d.heads * d.row(), d.heads * LATENT, d.heads * d.v_head);
    let (f, u) = (size_of::<f32>(), size_of::<u32>());
    let kv_norm = w.f32_buf(WHAT, kv_a_norm)?;
    k.latent.enqueue_latent_extra_append(
        stream,
        LatentExtraAppendArgs {
            rows: Rows {
                x: &*b.kv,
                stride: d.row(),
                m,
            },
            off: 0,
            e_off: LATENT,
            extra: d.extra,
            gain: kv_norm,
            pos: b.pos,
            eps: d.latent_eps,
            fault,
            cache: &mut *b.cache,
        },
    )?;
    k.step.enqueue_gather(
        stream,
        &*b.q,
        &k.src,
        &k.dst,
        m * d.heads * d.extra,
        &mut *b.qabs,
    )?;
    route_heads(gpu, (wk, wb), (m, fault))?;
    let shape = AbsorbShape {
        head_k: d.head_k(),
        nope: d.nope,
        row: d.row(),
    };
    absorb_heads(
        gpu,
        (wk, wb),
        (w, &sites.k_b),
        (&*b.q, m),
        (shape, fault, &mut *b.qabs),
    )?;
    let value = value_w(w, &sites.v_b)?;
    let q8 = gpu.q8f32();
    for c0 in (0..m).step_by(ATTN_CHUNK) {
        let mc = ATTN_CHUNK.min(m - c0);
        let n_keys = Window::<u32>::of(b.n_keys, (c0 + mc - 1) * u, 1)?;
        let qabs = Window::<f32>::of(&*b.qabs, c0 * row * f, mc * row)?;
        let mut att = WindowMut::<f32>::of_mut(&mut *b.att, c0 * hl * f, mc * hl)?;
        gpu.flash().enqueue_flash_latent_split(
            stream,
            FlashSplitArgs {
                inputs: FlashInputs {
                    q: &qabs,
                    kv: &*b.cache,
                    n_keys_buf: &n_keys,
                    kq_scale: 1.0 / (d.head_k() as f32).sqrt(),
                    geom: FlashGeom {
                        tokens: mc,
                        heads: d.heads,
                        rope_dims: d.extra,
                        latent_dims: LATENT,
                    },
                },
                part_v: &mut *b.part_v,
                part_ms: &mut *b.part_ms,
                y: &mut att,
            },
        )?;
        drop(att);
        if let ValueW::Q8_0(vqs, vd) = value {
            for c1 in (0..mc).step_by(PASS_ROWS) {
                let mm = PASS_ROWS.min(mc - c1);
                let at = c0 + c1;
                let x = Window::<f32>::of(&*b.att, at * hl * f, mm * hl)?;
                let mut av = WindowMut::<f32>::of_mut(&mut *b.av, at * hv * f, mm * hv)?;
                q8.enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: vqs,
                        d: vd,
                        x: &x,
                        rows_per_head: d.v_head,
                        x_head_stride: LATENT,
                        y_head_stride: d.v_head,
                        y_off: 0,
                        m: mm,
                        x_col_stride: hl,
                        y_col_stride: hv,
                        y: &mut av,
                    },
                )?;
            }
        }
    }
    if let ValueW::Q6K(wv) = value {
        value_heads(
            gpu,
            (wk, wb),
            wv,
            (&*b.att, m),
            (d.v_head, fault, &mut *b.av),
        )?;
    }
    k.out_gate.enqueue_out_gate(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: &*b.av,
            gate: &*b.gate,
            n_head: d.heads,
            m,
            fault,
            y: &mut *b.gated,
        },
    )?;
    quantize_out(
        gpu,
        wk.g32,
        wb.forms,
        (&*b.gated, m),
        (fault, sites.out.ty()),
    )?;
    let x_in = input(wb.forms, Input::Out, &*b.gated);
    wk.project(cx, (&sites.out, d.embd), &x_in, m, b.out)
}

#[cfg(test)]
pub(super) mod tests {
    use super::{MlaBufs, MlaDims, MlaKernels, MlaSites, ValueHeads, partials_lens_rows, rows};
    use crate::Gpu;
    use crate::arch::hybrid::{Acts, GqaCx, Site};
    use crate::arch::raon1::rda::tests::{Lcg, f32s, q6k, q8_0, resident, same};
    use crate::arch::raon1::rda::{RowKernels, RowsKit, add_row_forms};
    use crate::arch::raon1::state::PASS_ROWS;
    use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs};
    use crate::flash_gqa::FlashGqaKernels;
    use crate::gemm::Gemm32Kernels;
    use crate::latent::LATENT;
    use crate::linear::LinearKernels;
    use crate::site::{KGemvKernels, SiteTy};
    use crate::tensor::{DeviceTensor, Window, WindowMut};
    use crate::weights::Weights;
    use cuda_core::DeviceBuffer;
    use gguf::quant::f32_to_f16_bits;
    use model::arch::raon1::names;

    /// The layer the names carry.
    const L: usize = 3;
    /// Raon's per-head widths at four heads, the model width two Q6_K
    /// super-blocks.
    const D: MlaDims = MlaDims {
        embd: 512,
        heads: 4,
        q_lora: 576,
        nope: 128,
        extra: 64,
        v_head: 128,
        rms_eps: 1e-5,
        latent_eps: 1e-6,
    };
    /// The pass's rows.
    const M: usize = 7;
    /// (cache rows, the pass's first position): the single-launch attention
    /// and the split one.
    const RUNS: [(usize, usize); 2] = [(64, 40), (1024, 300)];

    /// One MLA layer's file tensors at [`D`]'s widths, resident, and its
    /// sites: Raon's types (Q6_K `q_a`, `kv_a` and `v_b`, Q8_0 the rest).
    fn layer(gpu: &Gpu) -> (Weights, MlaSites) {
        let mut r = Lcg(0x006d_6c61);
        let (e, h, hv) = (D.embd, D.heads, D.heads * D.v_head);
        let gains = (0.5, 1.5);
        let parts = [
            f32s(&mut r, names::attn_norm(L), vec![e], gains),
            q6k(&mut r, names::attn_q_a(L), vec![e, D.q_lora]),
            f32s(&mut r, names::attn_q_a_norm(L), vec![D.q_lora], gains),
            q6k(&mut r, names::attn_kv_a_mqa(L), vec![e, D.row()]),
            f32s(&mut r, names::attn_kv_a_norm(L), vec![LATENT], gains),
            q8_0(&mut r, names::attn_gate(L), vec![e, hv]),
            q8_0(&mut r, names::attn_q_b(L), vec![D.q_lora, h * D.head_k()]),
            q8_0(&mut r, names::attn_k_b(L), vec![D.nope, LATENT, h]),
            q6k(&mut r, names::attn_v_b(L), vec![LATENT, D.v_head, h]),
            q8_0(&mut r, names::attn_output(L), vec![hv, e]),
        ];
        let w = resident(gpu, &parts);
        let site = |name: String, ty: SiteTy, k: usize| Site::new(name, ty, k);
        let sites = MlaSites {
            q_a: site(names::attn_q_a(L), SiteTy::Q6K, e),
            kv_a: site(names::attn_kv_a_mqa(L), SiteTy::Q6K, e),
            gate: site(names::attn_gate(L), SiteTy::Q8_0, e),
            q_b: site(names::attn_q_b(L), SiteTy::Q8_0, D.q_lora),
            out: site(names::attn_output(L), SiteTy::Q8_0, hv),
            k_b: names::attn_k_b(L),
            v_b: ValueHeads::Q6K(names::attn_v_b(L)),
            vecs: [
                names::attn_norm(L),
                names::attn_q_a_norm(L),
                names::attn_kv_a_norm(L),
            ],
        };
        (w, sites)
    }

    /// A layer's sites with `q_b` and `out` of the given types, the others Raon's.
    pub(crate) fn typed_sites(q_b: SiteTy, out: SiteTy) -> MlaSites {
        let site = |name: String, ty: SiteTy, k: usize| Site::new(name, ty, k);
        let (e, hv) = (D.embd, D.heads * D.v_head);
        MlaSites {
            q_a: site(names::attn_q_a(L), SiteTy::Q6K, e),
            kv_a: site(names::attn_kv_a_mqa(L), SiteTy::Q6K, e),
            gate: site(names::attn_gate(L), SiteTy::Q8_0, e),
            q_b: site(names::attn_q_b(L), q_b, D.q_lora),
            out: site(names::attn_output(L), out, hv),
            k_b: names::attn_k_b(L),
            v_b: ValueHeads::Q6K(names::attn_v_b(L)),
            vecs: [
                names::attn_norm(L),
                names::attn_q_a_norm(L),
                names::attn_kv_a_norm(L),
            ],
        }
    }

    /// The wide arm runs a latent layer whose `attn_q_b` and output projection
    /// are Q8_0 (Raon's), and refuses by name one whose either is another type:
    /// a Q6_K output projection reads q8_1 blocks of 128, and the wide arm
    /// quantizes the gated rows in blocks of 32 alone.
    #[test]
    fn a_latent_layer_runs_wide_only_with_q8_0_query_up_and_output() {
        assert!(typed_sites(SiteTy::Q8_0, SiteTy::Q8_0).wide_runs().is_ok());
        for ty in [SiteTy::Q6K, SiteTy::Q4K, SiteTy::F32] {
            let e = typed_sites(SiteTy::Q8_0, ty).wide_runs().unwrap_err();
            assert!(e.to_string().contains("an output projection"), "{e}");
            let e = typed_sites(ty, SiteTy::Q8_0).wide_runs().unwrap_err();
            assert!(e.to_string().contains("a query up projection"), "{e}");
        }
    }

    /// The buffers of a pass of up to [`PASS_ROWS`] rows over a cache of
    /// `ctx` rows, each its width times [`PASS_ROWS`]: the input, the
    /// positions and live counts, [`MlaBufs`]' f32 buffers in its order from
    /// `xn` to `out`, and the stage.
    struct Bufs {
        x: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        n_keys: DeviceBuffer<u32>,
        f: [DeviceBuffer<f32>; 13],
        stage: DeviceBuffer<f32>,
    }

    impl Bufs {
        fn new(gpu: &Gpu, ctx: usize) -> Bufs {
            let s = gpu.stream();
            let z = |n: usize| DeviceBuffer::<f32>::zeroed(s, n).expect("a buffer");
            let u = || DeviceBuffer::<u32>::zeroed(s, PASS_ROWS).expect("a word row");
            let (h, hv) = (D.heads, D.heads * D.v_head);
            let (pv, pm) = partials_lens_rows(&D, ctx, PASS_ROWS);
            let rows = [
                D.embd,
                D.q_lora,
                D.q_lora,
                D.row(),
                hv,
                h * D.head_k(),
                h * D.row(),
                h * LATENT,
                hv,
                hv,
            ];
            let [xn, qa, qr, kv, gate, q, qabs, att, av, gated] = rows.map(|n| z(PASS_ROWS * n));
            Bufs {
                x: z(PASS_ROWS * D.embd),
                pos: u(),
                n_keys: u(),
                f: [
                    xn,
                    qa,
                    qr,
                    kv,
                    gate,
                    q,
                    qabs,
                    att,
                    av,
                    gated,
                    z(pv),
                    z(pm),
                    z(PASS_ROWS * D.embd),
                ],
                stage: z(super::stage_len(&D)),
            }
        }
    }

    /// One MLA layer and everything a pass of it reads.
    struct Rig<'g> {
        gpu: &'g Gpu,
        w: Weights,
        sites: MlaSites,
        mk: MlaKernels,
        lin: LinearKernels,
        rk: RowKernels,
        kgemv: KGemvKernels,
        g32: Gemm32Kernels,
        flash: FlashGqaKernels,
        acts: Acts,
    }

    impl Rig<'_> {
        /// The first `n` rows of `xs` from position `p` as one pass on
        /// `cache`: the rows' updates.
        fn pass(
            &mut self,
            (b, cache): (&mut Bufs, &mut DeviceTensor<u16>),
            (p, n): (usize, usize),
            xs: &[f32],
        ) -> Vec<f32> {
            let s = self.gpu.stream();
            let at = u32::try_from(p).expect("a u32 position");
            let mut x = vec![0.0f32; PASS_ROWS * D.embd];
            x[..n * D.embd].copy_from_slice(&xs[..n * D.embd]);
            let pos: Vec<u32> = (at..).take(PASS_ROWS).collect();
            let live: Vec<u32> = (at + 1..).take(PASS_ROWS).collect();
            b.x.copy_from_host(s, &x).expect("the rows up");
            b.pos.copy_from_host(s, &pos).expect("the positions up");
            b.n_keys
                .copy_from_host(s, &live)
                .expect("the live counts up");
            let cx = GqaCx {
                what: "raon1 mla test",
                gpu: self.gpu,
                w: &self.w,
                kgemv: &self.kgemv,
                g32: &self.g32,
                flash: &self.flash,
                mma: false,
                out_gate: None,
                hidden: D.embd,
                eps: D.rms_eps,
                n_kv: 1,
                ctx: cache.rows(),
            };
            let [
                xn,
                qa,
                qr,
                kv,
                gate,
                q,
                qabs,
                att,
                av,
                gated,
                part_v,
                part_ms,
                out,
            ] = &mut b.f;
            let bufs = MlaBufs {
                x: &b.x,
                pos: &b.pos,
                n_keys: &b.n_keys,
                xn,
                qa,
                qr,
                kv,
                gate,
                q,
                qabs,
                att,
                av,
                gated,
                part_v,
                part_ms,
                out: &mut *out,
                cache,
            };
            let kit = RowsKit {
                linear: &self.lin,
                rows: &self.rk,
                stage: &mut b.stage,
            };
            let sd = (&self.sites, &D);
            rows(&cx, &self.mk, &mut self.acts, sd, (L, n), bufs, kit).expect("the pass");
            assert!(
                self.gpu.fault().expect("the fault word").is_none(),
                "a clean pass raises no fault"
            );
            out.to_host_vec(s).expect("the updates back")[..n * D.embd].to_vec()
        }
    }

    /// A pass of rows is its rows run as steps, bit for bit — the updates
    /// and every cache row — from a position past a filled cache prefix, on
    /// a cache short enough for the single-launch attention and on one the
    /// split attention cuts.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_mla_rows_are_steps() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (w, sites) = layer(&gpu);
        let (s, ctx) = (gpu.stream(), gpu.context());
        let mut acts = Acts::new(s, sites.sites().into_iter()).expect("the one-row forms");
        add_row_forms(s, &mut acts, sites.sites().into_iter()).expect("the row forms");
        let mut rig = Rig {
            gpu: &gpu,
            mk: MlaKernels::load(&gpu, &D).expect("the MLA kernels"),
            lin: LinearKernels::load(ctx).expect("the linear kernels"),
            rk: RowKernels::load(ctx).expect("the row kernels"),
            kgemv: KGemvKernels::load(&gpu).expect("the K-quant gemvs"),
            g32: Gemm32Kernels::load(ctx).expect("the F32 tile"),
            flash: FlashGqaKernels::load(ctx).expect("the flash kernels"),
            acts,
            w,
            sites,
        };
        let e = D.embd;
        for (height, p0) in RUNS {
            let mut r = Lcg(0x0070_6173 + p0 as u64);
            let xs = r.fill(M * e, -1.0, 1.0);
            let mut prefix: Vec<u16> = r
                .fill(p0 * D.row(), -1.0, 1.0)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            prefix.resize(height * D.row(), 0);
            let cache = || DeviceTensor::upload(s, &prefix, height, D.row()).expect("the cache");
            let (mut c_steps, mut c_pass) = (cache(), cache());
            let mut b = Bufs::new(&gpu, height);
            let mut steps = Vec::with_capacity(M * e);
            for t in 0..M {
                steps.extend(rig.pass((&mut b, &mut c_steps), (p0 + t, 1), &xs[t * e..]));
            }
            let pass = rig.pass((&mut b, &mut c_pass), (p0, M), &xs);
            assert!(
                same(&pass, &steps, f32::to_bits),
                "cache of {height} rows from {p0}: a pass of {M} rows is its rows as steps"
            );
            let held = |c: &DeviceTensor<u16>| c.buf().to_host_vec(s).expect("the cache back");
            assert!(
                same(&held(&c_pass), &held(&c_steps), u32::from),
                "cache of {height} rows from {p0}: the pass writes the steps' cache rows"
            );
        }
    }

    /// The latent flash over a unit's query tokens is the same bits however
    /// the tokens are cut into launches: a row's cut of the keys is its
    /// group's, and under the single-span regime (a live count within
    /// `SEGMENTS` segments of the floor) every group cuts alike. The split
    /// attention (a cache past one segment), chunks of the groups' token
    /// period and a cut that falls inside a group.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_flash_tokens_per_launch_are_bits() {
        const TOKENS: usize = 32;
        const HEIGHT: usize = 1024;
        // The last token's live key count.
        const LIVE: usize = 600;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let s = gpu.stream();
        let mut r = Lcg(0x0066_6c61);
        let mut cache_rows: Vec<u16> = r
            .fill(LIVE * D.row(), -1.0, 1.0)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        cache_rows.resize(HEIGHT * D.row(), 0);
        let cache = DeviceTensor::upload(s, &cache_rows, HEIGHT, D.row()).expect("the cache");
        let (row, hl) = (D.heads * D.row(), D.heads * LATENT);
        let q = r.fill(TOKENS * row, -1.0, 1.0);
        let q = DeviceBuffer::from_host(s, &q).expect("the queries up");
        // The tokens cut into launches of `chunks` tokens each, in order.
        let run = |chunks: &[usize]| -> Vec<f32> {
            assert_eq!(
                chunks.iter().sum::<usize>(),
                TOKENS,
                "the cut covers the tokens"
            );
            let widest = chunks.iter().copied().max().expect("a chunk");
            let (pv, pm) = partials_lens_rows(&D, HEIGHT, widest);
            let mut part_v = DeviceBuffer::<f32>::zeroed(s, pv).expect("the partials");
            let mut part_ms = DeviceBuffer::<f32>::zeroed(s, pm).expect("the partials");
            let mut y = DeviceBuffer::<f32>::zeroed(s, TOKENS * hl).expect("the output");
            let mut c0 = 0;
            for &chunk in chunks {
                let last = u32::try_from(LIVE - (TOKENS - c0 - chunk)).expect("a u32 count");
                let n_keys = DeviceBuffer::from_host(s, &[last]).expect("the live count up");
                let qw = Window::<f32>::of(&q, c0 * row * 4, chunk * row).expect("the queries");
                let mut yw = WindowMut::<f32>::of_mut(&mut y, c0 * hl * 4, chunk * hl)
                    .expect("the output window");
                gpu.flash()
                    .enqueue_flash_latent_split(
                        s,
                        FlashSplitArgs {
                            inputs: FlashInputs {
                                q: &qw,
                                kv: &cache,
                                n_keys_buf: &n_keys,
                                kq_scale: 1.0 / (D.head_k() as f32).sqrt(),
                                geom: FlashGeom {
                                    tokens: chunk,
                                    heads: D.heads,
                                    rope_dims: D.extra,
                                    latent_dims: LATENT,
                                },
                            },
                            part_v: &mut part_v,
                            part_ms: &mut part_ms,
                            y: &mut yw,
                        },
                    )
                    .expect("the latent flash");
                c0 += chunk;
            }
            y.to_host_vec(s).expect("the output back")
        };
        let whole = run(&[TOKENS]);
        for cut in [
            &[16, 16][..],
            &[8, 8, 8, 8],
            &[4; 8],
            &[6, 26],
            &[1, 31],
            &[9, 9, 14],
        ] {
            assert!(
                same(&whole, &run(cut), f32::to_bits),
                "{TOKENS} tokens in one launch are the same bits cut as {cut:?}"
            );
        }
    }
}
