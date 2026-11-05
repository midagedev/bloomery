//! The gated no-position latent attention core of an absorbed MLA layer on the card: the
//! `rope.dimension_count` channels beside the latent are an unrotated extra key part shared by the
//! heads, the cache row is `[extra | latent]`, and a sigmoid projection gates the heads' values
//! (llama.cpp `src/models/raon1.cpp` and `src/models/kimi-k3.cpp`, `build_mla_layer`). Raon-1 and
//! Kimi K3 run these launches alike; the norms and projections around them are each family's own
//! layer walk.
//!
//! [`attend`] runs one row, its launches in order, over common owners:
//! 1. the cache row at the position: the raw extra part, then the latent's
//!    RMS with `attn_kv_a_norm` (`latent::enqueue_latent_extra_append`), f16;
//! 2. `attn_k_b` per head: each head's nope query absorbed into the latent
//!    width, behind its extra part (`enqueue_q8_0_gemv_heads`);
//! 3. each head's extra query part copied in front (`enqueue_gather` over
//!    load-time tables);
//! 4. the latent attention over the cache rows `0 ..= pos`
//!    (`flash::enqueue_flash_latent_split`, one or two launches), the row
//!    `[extra | latent]` the key and its latent the value, at the scale
//!    `1/√(nope + extra)`;
//! 5. `attn_v_b` per head: each head's latent output to its values
//!    (`q6k_heads` for Q6_K, `enqueue_q8_0_gemv_heads` for Q8_0);
//! 6. `attn · σ(gate)` (`linear::out_gate`).
//!
//! [`attend_rows`] runs a pass of 1 to [`PASS_ROWS`] rows in position order — a verify's rows or a
//! prompt chunk — through the same launches, each once over the pass's rows: the cache rows of
//! every row appended before any attends, the absorb and a Q8_0 value projection over the rows'
//! windows (`q8_0_gemv_heads_mcol`; a Q6_K one is `q6k_heads` a row at a time), the gather over the
//! tables [`MlaKernels::load`] built, and the latent attention over the `m` query tokens, token
//! `t` over keys `0 ..= pos + t`. Row `t` of each launch is the single row's launch on row `t` bit
//! for bit, and so is the attention's while the pass's live count stays at or under
//! `SEGMENTS · seg_keys` (`crate::flash`): the split cuts every group at the fixed `seg_keys`-key
//! span there. Past it a token whose group reaches a wider cut than its own row's sums its keys in
//! another order: a float-order band, not a bit pin.
//!
//! What the file forces here: the NoPE extra part and the output gate; the widths are the
//! caller's ([`MlaDims`]).

use crate::fault::FaultSink;
use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs, partials_ms_len, partials_v_len};
use crate::latent::{LATENT, LatentExtraAppendArgs, LatentKernels, Rows};
use crate::linear::PASS_ROWS;
use crate::linear::out_gate::{OutGateArgs, OutGateKernels};
use crate::model::{Q8_0GemvHeadsArgs, StepKernels};
use crate::q6k_heads::{Q6kHeadsArgs, Q6kHeadsKernels};
use crate::q8f32::Q8_0GemvHeadsMcolArgs;
use crate::tensor::{DeviceTensor, Window, WindowMut};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use model::arch::models::OutGate;

/// What the refusals of this file name.
const WHAT: &str = "gated mla";

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

    /// Refuse by name per-head weights that are not this layer's: `attn_k_b` is
    /// `heads · LATENT` rows of `nope` values and `attn_v_b` is `heads ·
    /// v_head` rows, so planes that disagree with the dims would otherwise run
    /// on other widths or on fewer heads without a word.
    pub fn check_heads(
        &self,
        k_b: (&DeviceTensor<u32>, &DeviceTensor<u16>),
        v_b: &ValueW<'_>,
    ) -> Result<(), GpuError> {
        let (qs, d) = k_b;
        let k_rows = self.heads * LATENT;
        if qs.rows() != k_rows || d.rows() != k_rows || d.cols() * 32 != self.nope {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "attn_k_b holds {} rows of {} values; {} heads of nope {} are {k_rows} rows of {}",
                    qs.rows(),
                    d.cols() * 32,
                    self.heads,
                    self.nope,
                    self.nope
                ),
            ));
        }
        let v_rows = self.heads * self.v_head;
        let have = match v_b {
            ValueW::Q6K(w) => w.rows(),
            ValueW::Q8_0(qs, _) => qs.rows(),
        };
        if have != v_rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "attn_v_b holds {have} rows; {} heads of {} values are {v_rows}",
                    self.heads, self.v_head
                ),
            ));
        }
        Ok(())
    }
}

/// A resident per-head value projection, as [`attend`] reads it.
pub enum ValueW<'a> {
    /// Q6_K words, `heads·v_head` rows of `LATENT`.
    Q6K(&'a DeviceTensor<u32>),
    /// Q8_0 planes (`qs`, `d`).
    Q8_0(&'a DeviceTensor<u32>, &'a DeviceTensor<u16>),
}

/// The kernels every MLA layer launches beyond the `Gpu`'s own, and the
/// extra part's gather tables for up to `rows` rows
/// (`[(t·heads + h)·head_k + nope + j] → [(t·heads + h)·row + j]`; a call of
/// `m` rows reads the first `m·heads·extra` pairs).
pub struct MlaKernels {
    pub(crate) step: StepKernels,
    pub(crate) latent: LatentKernels,
    pub(crate) q6k: Q6kHeadsKernels,
    pub(crate) out_gate: OutGateKernels,
    pub(crate) src: DeviceBuffer<u32>,
    pub(crate) dst: DeviceBuffer<u32>,
}

impl MlaKernels {
    /// Load the modules and build the gather tables for `d`, for the most rows
    /// one call gathers: a family passes its widest unit. Load-time only.
    pub fn load(gpu: &Gpu, d: &MlaDims, rows: usize) -> Result<MlaKernels, GpuError> {
        d.check()?;
        if rows == 0 {
            return Err(GpuError::shape(WHAT, "gather tables for 0 rows"));
        }
        let at = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(WHAT, "a gather index past u32"))
        };
        let pairs = rows * d.heads * d.extra;
        let (mut src, mut dst) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
        for head in 0..rows * d.heads {
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

/// The module doc's launches for one row: the cache row at `pos`
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
    d.check_heads(a.k_b, &a.v_b)?;
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
    d.check_heads(a.k_b, &a.v_b)?;
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
