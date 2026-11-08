//! The gated MLA layer on the card: the layer walk around the gated latent attention core
//! ([`crate::latent_gated`]), which both Raon-1 and Kimi K3 run (llama.cpp `src/models/raon1.cpp`
//! `build_mla_layer`, the HF `Raon1MiniMLA`).
//!
//! The launches, in order, over common owners:
//! 1. `attn_norm` RMS of `x`;
//! 2. `attn_q_a` and `attn_kv_a_mqa` of the normed row (through one q8_1
//!    form when they are K-quants) and `attn_gate` (its own form);
//! 3. `attn_q_a_norm` RMS of the q_a rows (the latent ε);
//! 4. `attn_q_b`: each head's `[nope | extra]` query;
//!
//! then [`attend`], the core the kernel gate runs: the cache row, the absorb, the gather of the
//! extra part, the latent attention, the value projection and the output gate;
//!
//! and last, 5. `attn_output` into `out`.
//!
//! [`step`] runs one row: [`front`] (1 to 4), then [`back`] (the core and 5). A family whose
//! layer selects its keys (Kimi K3.1's indexer) runs [`front`], its own selection, then [`back`] with
//! the core over the selected rows ([`attend_sel`]). [`rows`] runs a pass of 1 to [`PASS_ROWS`] rows in position order — a
//! verify's rows or a prompt chunk — through the core's [`attend_rows`] with the projections'
//! m-column launches ([`project_rows`]) before and after it. A family whose prompt runs wider
//! units reads the same [`MlaSites`] and [`MlaBufs`] from its own wide walk.
//!
//! The tensor names are the model crate's common spellings ([`model::arch::names`]), the ones
//! every family re-exports. What the file forces here: the layer's projections and their types
//! (Q3_K to Q8_0 and F32 for the dense ones, Q6_K or Q8_0 per-head value projection); the widths
//! are the header's.

use crate::GpuError;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::rows_kit::{RowsKit, kgemv_rows, project_rows};
use crate::latent::LATENT;
use crate::latent_gated::{
    AttendIn, AttnBufs, MlaDims, MlaKernels, Selection, ValueW, attend, attend_rows, attend_sel,
};
use crate::linear::PASS_ROWS;
use crate::site::{self, Form, SiteTy};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use cuda_core::DeviceBuffer;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::names;

/// What the refusals of this file name.
const WHAT: &str = "gated mla";

/// Types a dense projection of the layer may have.
const DENSE: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// How a layer's per-head value projection runs, by the file type of
/// `attn_v_b`.
pub(crate) enum ValueHeads {
    Q6K(String),
    Q8_0(String),
}

/// One MLA layer's widths, projections and the names of its vectors.
pub(crate) struct MlaSites {
    pub(crate) q_a: Site,
    pub(crate) kv_a: Site,
    pub(crate) gate: Site,
    pub(crate) q_b: Site,
    pub(crate) out: Site,
    /// `attn_k_b`, Q8_0: each head's `LATENT` rows over its nope values.
    pub(crate) k_b: String,
    pub(crate) v_b: ValueHeads,
    /// attn_norm, attn_q_a_norm, attn_kv_a_norm.
    pub(crate) vecs: [String; 3],
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
}

/// f32s of the stage an MLA layer's pass of up to [`PASS_ROWS`] rows writes
/// ([`RowsKit`]): the widest projection's outputs of every row.
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
/// intermediates, the attention's partials
/// ([`partials_lens_rows`](crate::latent_gated::partials_lens_rows)), `out`
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
pub(crate) fn value_w<'w>(w: &'w Weights, v: &ValueHeads) -> Result<ValueW<'w>, GpuError> {
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

/// Enqueue MLA layer `l` at one row on `b`'s buffers (module doc): [`front`], then [`back`] with no
/// selection.
pub(crate) fn step(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    l: usize,
    mut b: MlaBufs<'_>,
) -> Result<(), GpuError> {
    front(cx, acts, (sites, d), l, &mut b)?;
    back(cx, k, acts, (sites, d), l, b, None)
}

/// The module doc's launches 1 to 4 of layer `l` at one row: `b.xn` the normed input, `b.qa` and
/// `b.qr` the query latent before and after its norm, `b.kv`, `b.gate` and `b.q` the projections
/// [`back`] reads.
pub(crate) fn front(
    cx: &GqaCx<'_>,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    l: usize,
    b: &mut MlaBufs<'_>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let [norm, q_a_norm, _] = &sites.vecs;
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
    sites.q_b.project(cx, acts, l, b.qr, b.q)
}

/// The core over [`front`]'s projections in `b` (the cache row, the absorb, the gather, the latent
/// attention over the cache rows `0 ..= pos`, or over `sel`'s rows when there is one, the value
/// projection, the output gate), then launch 5, `attn_output` into `b.out`.
pub(crate) fn back(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    acts: &mut Acts,
    (sites, d): (&MlaSites, &MlaDims),
    l: usize,
    b: MlaBufs<'_>,
    sel: Option<Selection<'_>>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let fault = gpu.layer_sink(l)?;
    let [_, _, kv_a_norm] = &sites.vecs;
    let a = AttendIn {
        kv: b.kv,
        q: b.q,
        gate: b.gate,
        kv_norm: w.f32_buf(WHAT, kv_a_norm)?,
        k_b: q8_planes(w, &sites.k_b)?,
        v_b: value_w(w, &sites.v_b)?,
        pos: b.pos,
        n_keys: b.n_keys,
        fault,
    };
    let s = AttnBufs {
        qabs: b.qabs,
        att: b.att,
        av: b.av,
        part_v: b.part_v,
        part_ms: b.part_ms,
    };
    match sel {
        None => attend(gpu, k, d, a, s, b.cache, b.gated)?,
        Some(sel) => attend_sel(gpu, k, d, a, s, b.cache, sel, b.gated)?,
    }
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
