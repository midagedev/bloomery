//! The stage's load-time arena: the layer scratch buffers, the per-step
//! parameter image, and the weight names and MoE shapes they are sized from.

use super::lookup::{f32_gain, kq_weight, q8_derived};
use super::probe::StepProbe;
use crate::GpuError;
use crate::q5::Q8Blocks32;
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, Weights};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use model::attn::MlaParams;
use std::mem::ManuallyDrop;
use std::sync::Arc;

// ------------------------------------------------------------------ scratch

/// The shapes the block-0 scratch is sized for, derived once from the
/// resident weights and `MlaParams` — never literals.
pub(super) struct Dims {
    pub(super) hidden: usize,
    /// `q_rows / rope_dims` — the 64-value columns `enqueue_rope` walks over
    /// the q projection.
    pub(super) q_cols: usize,
}

/// One device index table for the gather (source and destination indices of
/// one permutation).
pub(super) struct Gather {
    pub(super) src: DeviceBuffer<u32>,
    pub(super) dst: DeviceBuffer<u32>,
    pub(super) n: usize,
}

impl Gather {
    /// Build the table for `pairs: (src, dst)`. Load-time only.
    fn new(
        stream: &CudaStream,
        pairs: impl Iterator<Item = (usize, usize)>,
        what: &'static str,
    ) -> Result<Gather, GpuError> {
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for (s, d) in pairs {
            if s > u32::MAX as usize || d > u32::MAX as usize {
                return Err(GpuError::shape(what, "index overflows u32"));
            }
            src.push(s as u32);
            dst.push(d as u32);
        }
        if src.is_empty() {
            return Err(GpuError::shape(what, "empty table"));
        }
        Ok(Gather {
            n: src.len(),
            src: DeviceBuffer::from_host(stream, &src)?,
            dst: DeviceBuffer::from_host(stream, &dst)?,
        })
    }
}

/// The weight names one layer's chain looks up, built once at load so the
/// step never formats a string (decision 4). `routed` is read from the
/// file, not from the layer index: a layer is MoE exactly when its router
/// weight is resident.
pub(super) struct LayerNames {
    pub(super) layer: usize,
    pub(super) attn_norm: String,
    pub(super) attn_q: String,
    pub(super) attn_kv_a_mqa: String,
    pub(super) attn_kv_a_norm: String,
    pub(super) attn_kv_b: String,
    pub(super) attn_output: String,
    pub(super) ffn_norm: String,
    pub(super) ffn_gate: String,
    pub(super) ffn_up: String,
    pub(super) ffn_down: String,
    pub(super) ffn_gate_inp: String,
    pub(super) ffn_gate_exps: String,
    pub(super) ffn_up_exps: String,
    pub(super) ffn_down_exps: String,
    pub(super) ffn_gate_shexp: String,
    pub(super) ffn_up_shexp: String,
    pub(super) ffn_down_shexp: String,
    /// The derived q_nope2 planes' name — a `format!` of the layer index, so
    /// it is resolved here and never on the step's path (decision 4).
    pub(super) derived: String,
    pub(super) routed: bool,
}

impl LayerNames {
    /// Every name of block `l`, and whether that block routes.
    pub(super) fn new(w: &Weights, l: usize) -> LayerNames {
        let n = |stem: &str| format!("blk.{l}.{stem}.weight");
        let ffn_gate_inp = n("ffn_gate_inp");
        LayerNames {
            layer: l,
            routed: w.get(&ffn_gate_inp).is_some(),
            attn_norm: n("attn_norm"),
            attn_q: n("attn_q"),
            attn_kv_a_mqa: n("attn_kv_a_mqa"),
            attn_kv_a_norm: n("attn_kv_a_norm"),
            attn_kv_b: n("attn_kv_b"),
            attn_output: n("attn_output"),
            ffn_norm: n("ffn_norm"),
            ffn_gate: n("ffn_gate"),
            ffn_up: n("ffn_up"),
            ffn_down: n("ffn_down"),
            ffn_gate_inp,
            ffn_gate_exps: n("ffn_gate_exps"),
            ffn_up_exps: n("ffn_up_exps"),
            ffn_down_exps: n("ffn_down_exps"),
            ffn_gate_shexp: n("ffn_gate_shexp"),
            ffn_up_shexp: n("ffn_up_shexp"),
            ffn_down_shexp: n("ffn_down_shexp"),
            derived: crate::weights::derived_name(l),
        }
    }
}

/// The MoE shapes of this model, read from the file's metadata and from the
/// resident expert stacks — never literals. Present only when the stage
/// holds a routed layer.
#[derive(Clone)]
pub(super) struct MoeDims {
    pub(super) n_expert: usize,
    pub(super) n_used: usize,
    /// `expert_feed_forward_length` — one routed expert's hidden width.
    pub(super) ff: usize,
    /// `expert_weights_scale`, applied to every router weight.
    pub(super) scale: f32,
    /// The shared expert's hidden width (its gate/up row count).
    pub(super) shexp_ff: usize,
}

/// The routed FFN half's arena, sized at load from [`MoeDims`].
pub(super) struct MoeScratch {
    /// `ffn_norm(ffn_inp)` as f32 — the router's input and the block's
    /// `ffn_norm` tap. The experts read the q8_1 form in `act_ffn`.
    pub(super) normed: DeviceBuffer<f32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) probs: DeviceBuffer<f32>,
    /// The router's expert ids — at m = 1 this buffer IS the `sel` the
    /// expert kernels read, so a captured graph follows the routing.
    pub(super) ids: DeviceBuffer<u32>,
    pub(super) weights: DeviceBuffer<f32>,
    pub(super) h_exp: DeviceBuffer<f32>,
    pub(super) act32_exp: Q8Blocks32,
    pub(super) down: DeviceBuffer<f32>,
    pub(super) h_sh: DeviceBuffer<f32>,
    pub(super) act_sh: Q8Act,
    pub(super) shexp: DeviceBuffer<f32>,
}

impl MoeScratch {
    fn bytes(&self) -> usize {
        let mut total = [
            &self.normed,
            &self.logits,
            &self.probs,
            &self.weights,
            &self.h_exp,
            &self.down,
            &self.h_sh,
            &self.shexp,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        total += self.ids.num_bytes();
        total += self.act32_exp.q.num_bytes()
            + self.act32_exp.s8.num_bytes()
            + self.act32_exp.d8.num_bytes();
        total += self.act_sh.q3.num_bytes()
            + self.act_sh.q4.num_bytes()
            + self.act_sh.q6.num_bytes()
            + self.act_sh.s8.num_bytes()
            + self.act_sh.d8.num_bytes();
        total
    }
}

/// The dense FFN half's arena: the swiglu intermediate and its 32-value
/// quantization.
pub(super) struct DenseScratch {
    pub(super) h: DeviceBuffer<f32>,
    pub(super) act32: Q8Blocks32,
}

impl DenseScratch {
    fn bytes(&self) -> usize {
        self.h.num_bytes()
            + self.act32.q.num_bytes()
            + self.act32.s8.num_bytes()
            + self.act32.d8.num_bytes()
    }
}

/// Element offsets into `LayerScratch::step_params`, the single buffer the
/// per-step parameters share: the three u32 first, then the rope cos/sin
/// table as f32 bits. The table is read element by element on the device, so
/// four-byte alignment is all its window needs.
const SP_TOKEN: usize = 0;
pub(super) const SP_POS: usize = 1;
pub(super) const SP_N_KEYS: usize = 2;
const SP_CS: usize = 3;

/// A non-owning window of `len` `T` over `parent`, starting at element `off`
/// of the parent's u32 grid. The launches read it exactly as they read a
/// buffer of its own.
///
/// # Safety
///
/// - `off * 4 + len * size_of::<T>()` must be within `parent`'s allocation,
///   and `off * 4` must be a multiple of `align_of::<T>()`.
/// - `parent` must outlive the window and must not be reallocated: a captured
///   graph bakes the address in.
unsafe fn param_view<T>(
    parent: &DeviceBuffer<u32>,
    off: usize,
    len: usize,
) -> ManuallyDrop<DeviceBuffer<T>> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<u32>()) as u64;
    // SAFETY: the range is the caller's contract above; `parent` was allocated
    // by `DeviceBuffer::from_host`, the synchronous allocator `from_raw_parts`
    // assumes, and in the context cloned here. `ManuallyDrop` is what keeps the
    // window from ever freeing an allocation it does not own.
    ManuallyDrop::new(unsafe { DeviceBuffer::from_raw_parts(ptr, len, parent.context().clone()) })
}

/// The layer scratch arena and the device-side step parameters, sized at
/// load for m = 1 (decision 4: nothing here is allocated per step). One
/// arena serves every layer of a stage — layers run sequentially, so the
/// intermediates are reused; only the KV caches are per layer.
pub(super) struct LayerScratch {
    pub(super) dims: Dims,
    pub(super) x: DeviceBuffer<f32>,
    pub(super) normed: DeviceBuffer<f32>,
    pub(super) act_q: Q8Act,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) q_rope_all: DeviceBuffer<f32>,
    pub(super) kv_a: DeviceBuffer<f32>,
    /// `[kv_compressed | k_rope]` — the fused key-path launch writes both
    /// spans.
    pub(super) kv_s: DeviceBuffer<f32>,
    /// `[k_rope | kv_compressed]` — the appended cache row, kept as f32 for
    /// the `k_rope` tap (the append itself writes from registers).
    pub(super) kvr: DeviceBuffer<f32>,
    /// Flash query rows, `[q_rope | q_nope2]` per head.
    pub(super) f_rows: DeviceBuffer<f32>,
    pub(super) kqvc: DeviceBuffer<f32>,
    pub(super) act_kv_lo: Q8Act,
    pub(super) act_kv_hi: Q8Act,
    pub(super) kqv_2d: DeviceBuffer<f32>,
    pub(super) act_ao: Q8Act,
    pub(super) attn_out: DeviceBuffer<f32>,
    pub(super) ffn_inp: DeviceBuffer<f32>,
    /// The q8_1 form of the FFN-normed vector — read by the dense gate/up,
    /// by the routed experts and by the shared expert alike.
    pub(super) act_ffn: Q8Act,
    /// The dense half's arena — present iff the stage holds a dense layer.
    pub(super) dense: Option<DenseScratch>,
    /// The routed half's arena — present iff the stage holds a routed layer.
    pub(super) moe: Option<MoeScratch>,
    pub(super) l_out: DeviceBuffer<f32>,
    /// Flash split partials: `Σ exp·V` per (query row, key segment), and
    /// the `(running max, Σ exp)` pair beside it. Sized at load from the
    /// cache height, so the split launch allocates nothing per step.
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    pub(super) g_f_rope_lo: Gather,
    pub(super) g_f_rope_hi: Gather,
    /// Every per-step parameter in one allocation, laid out by the `SP_*`
    /// constants. The four quantities used to be four buffers and four
    /// `copy_from_host` calls per step, and each of those synchronizes the
    /// stream; one image means one copy and one synchronization.
    pub(super) step_params: DeviceBuffer<u32>,
    /// Host image of `step_params`, refilled in place each step so this image
    /// adds no allocation of its own.
    pub(super) params_host: Vec<u32>,
    /// Non-owning windows into `step_params`, one per kernel argument. The
    /// launches take them exactly as they took the separate buffers; the
    /// parent owns the allocation and these never free it.
    pub(super) pos_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) n_keys_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) token_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) cs_buf: ManuallyDrop<DeviceBuffer<f32>>,
    /// Host mirror of `pos_buf`, written by the same `refresh_params` that
    /// fills the device buffers. The launches never read it: it is what the
    /// byte accounting counts the live key rows from, since the count the
    /// kernels use lives on the device.
    pub(super) pos_host: u32,
    /// The node-price probe's empty kernel and the 32 f32 it stores into.
    /// Resident always (128 B); launched only when `probe_cfg` asks.
    pub(super) probe: crate::probe::Probe,
    pub(super) probe_buf: DeviceBuffer<f32>,
    /// Off in every normal step — see [`StepProbe`].
    pub(super) probe_cfg: StepProbe,
}

impl LayerScratch {
    /// Device bytes of the arena and the parameter buffers (weights and KV
    /// are counted by their owners).
    pub(super) fn bytes(&self) -> usize {
        let mut total = [
            &self.x,
            &self.normed,
            &self.q,
            &self.q_rope_all,
            &self.kv_a,
            &self.kv_s,
            &self.kvr,
            &self.f_rows,
            &self.kqvc,
            &self.kqv_2d,
            &self.attn_out,
            &self.ffn_inp,
            &self.l_out,
            &self.part_v,
            &self.part_ms,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        // The four parameter windows are counted once, through their parent.
        total += self.step_params.num_bytes() + self.probe_buf.num_bytes();
        let act = |a: &Q8Act| {
            a.q3.num_bytes()
                + a.q4.num_bytes()
                + a.q6.num_bytes()
                + a.s8.num_bytes()
                + a.d8.num_bytes()
        };
        total += act(&self.act_q)
            + act(&self.act_kv_lo)
            + act(&self.act_kv_hi)
            + act(&self.act_ao)
            + act(&self.act_ffn);
        total += [&self.g_f_rope_lo, &self.g_f_rope_hi]
            .iter()
            .map(|g| g.src.num_bytes() + g.dst.num_bytes())
            .sum::<usize>();
        total += self.dense.as_ref().map_or(0, DenseScratch::bytes);
        total += self.moe.as_ref().map_or(0, MoeScratch::bytes);
        total
    }
}

impl MoeDims {
    /// Read the MoE shapes from the file's metadata and cross-check them
    /// against the resident expert stacks of `names` — the quantization
    /// types and row counts the enqueue leans on are proven here, at load.
    pub(super) fn read(
        gguf: &gguf::Gguf,
        w: &Weights,
        names: &LayerNames,
    ) -> Result<MoeDims, GpuError> {
        let meta = model::moe::Meta::read(gguf)?;
        let hidden = f32_gain(w, &names.attn_norm)?.len();
        // The router kernel ranks a fixed 64 experts into a fixed 6 slots.
        if meta.n_expert != 64 || meta.n_used != 6 {
            return Err(GpuError::shape(
                "MoeDims::read",
                format!(
                    "the router kernel is 64 experts into 6 slots, the file says \
                 {} into {}",
                    meta.n_expert, meta.n_used
                ),
            ));
        }
        let kq_ty = |name: &str, want: GgmlType| -> Result<(), GpuError> {
            match w.get(name) {
                Some(DevWeight::KQuant { ty, .. }) if *ty == want => Ok(()),
                Some(other) => Err(GpuError::shape(
                    "MoeDims::read",
                    format!(
                        "{name} is resident as {} rows of k={}, want {want}",
                        other.rows(),
                        other.k()
                    ),
                )),
                None => Err(GpuError::tensor("MoeDims::read", name, "resident")),
            }
        };
        kq_ty(&names.ffn_gate_exps, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_up_exps, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_gate_shexp, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_up_shexp, GgmlType::Q3_K)?;
        kq_ty(&names.ffn_down_shexp, GgmlType::Q4_K)?;
        // The routed down stack's kernel is `q5_0_gemv_sel`: a Q5_1 or a
        // K-quant here is a different kernel, not a different constant.
        match w.get(&names.ffn_down_exps) {
            Some(DevWeight::Q5_0 { .. }) => {}
            Some(_) => {
                return Err(GpuError::tensor(
                    "MoeDims::read",
                    &names.ffn_down_exps,
                    "resident as Q5_0 — the routed down projection runs q5_0_gemv_sel",
                ));
            }
            None => {
                return Err(GpuError::tensor(
                    "MoeDims::read",
                    &names.ffn_down_exps,
                    "resident",
                ));
            }
        }
        match w.get(&names.ffn_gate_inp) {
            Some(DevWeight::F32 { .. }) => {}
            _ => {
                return Err(GpuError::tensor(
                    "MoeDims::read",
                    &names.ffn_gate_inp,
                    "resident as F32 — the router is an f32 gemv",
                ));
            }
        }
        let gate_exps = kq_weight(w, &names.ffn_gate_exps)?;
        let down_exps = kq_weight(w, &names.ffn_down_exps)?;
        let shexp_ff = kq_weight(w, &names.ffn_gate_shexp)?.rows();
        let check = |what: &'static str, want: usize, got: usize| -> Result<(), GpuError> {
            if want == got {
                Ok(())
            } else {
                Err(GpuError::shape(
                    "MoeDims::read",
                    format!("{what}: {want} != {got}"),
                ))
            }
        };
        check(
            "ffn_gate_exps rows vs n_expert*expert_ff",
            gate_exps.rows(),
            meta.n_expert * meta.ff,
        )?;
        check(
            "ffn_up_exps rows vs n_expert*expert_ff",
            kq_weight(w, &names.ffn_up_exps)?.rows(),
            meta.n_expert * meta.ff,
        )?;
        check(
            "ffn_down_exps rows vs n_expert*hidden",
            down_exps.rows(),
            meta.n_expert * hidden,
        )?;
        check(
            "ffn_up_shexp rows vs ffn_gate_shexp rows",
            kq_weight(w, &names.ffn_up_shexp)?.rows(),
            shexp_ff,
        )?;
        check(
            "ffn_down_shexp rows vs hidden",
            kq_weight(w, &names.ffn_down_shexp)?.rows(),
            hidden,
        )?;
        Ok(MoeDims {
            n_expert: meta.n_expert,
            n_used: meta.n_used,
            ff: meta.ff,
            scale: meta.scale,
            shexp_ff,
        })
    }
}

impl LayerScratch {
    /// Derive every size from the resident weights and the MLA metadata,
    /// cross-check the geometry the enqueue leans on, and allocate the arena
    /// (plus the pair tables the gathers run). The dense FFN buffers are
    /// sized from the stage's first dense layer, the routed ones from
    /// `moe`. Load-time only.
    pub(super) fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        w: &Weights,
        mla: &MlaParams,
        names: &[LayerNames],
        moe: Option<&MoeDims>,
        ctx_max: usize,
    ) -> Result<LayerScratch, GpuError> {
        let first = names.first().ok_or(GpuError::state(
            "LayerScratch::new",
            "the stage holds no layer",
        ))?;
        let hidden = f32_gain(w, &first.attn_norm)?.len();
        let q_rows = kq_weight(w, &first.attn_q)?.rows();
        let kv_width = kq_weight(w, &first.attn_kv_a_mqa)?.rows();
        // The dense arena is sized from the stage's first dense layer; a
        // stage with none carries no dense arena.
        let dense_ff = match names.iter().find(|n| !n.routed) {
            Some(n) => Some(kq_weight(w, &n.ffn_gate)?.rows()),
            None => None,
        };
        check_geometry(w, mla, first, q_rows, kv_width)?;
        let (rope, latent) = (mla.rope_dims, mla.latent);
        let half = mla.n_head / 2;
        let dims = Dims {
            hidden,
            q_cols: q_rows / rope,
        };
        let f32n = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let dense = match dense_ff {
            Some(ff) => Some(DenseScratch::new(stream, ff)?),
            None => None,
        };
        let moe = match moe {
            Some(m) => Some(MoeScratch::new(stream, m, hidden)?),
            None => None,
        };
        let ParamImage {
            step_params,
            params_host,
            pos_buf,
            n_keys_buf,
            token_buf,
            cs_buf,
        } = ParamImage::new(stream, rope)?;
        Ok(LayerScratch {
            x: f32n(hidden)?,
            normed: f32n(hidden)?,
            act_q: Q8Act::with_k(stream, 1, hidden)?,
            q: f32n(q_rows)?,
            q_rope_all: f32n(q_rows)?,
            kv_a: f32n(kv_width)?,
            kv_s: f32n(kv_width)?,
            kvr: f32n(kv_width)?,
            f_rows: f32n(mla.n_head * kv_width)?,
            kqvc: f32n(mla.n_head * latent)?,
            act_kv_lo: Q8Act::with_k(stream, 8, latent)?,
            act_kv_hi: Q8Act::with_k(stream, 8, latent)?,
            kqv_2d: f32n(mla.n_head * mla.v_head)?,
            act_ao: Q8Act::with_k(stream, 1, hidden)?,
            attn_out: f32n(hidden)?,
            ffn_inp: f32n(hidden)?,
            act_ffn: Q8Act::with_k(stream, 1, hidden)?,
            dense,
            moe,
            l_out: f32n(hidden)?,
            g_f_rope_lo: rope_gather(stream, mla, kv_width, 0..half, "Gather::new f_rope_lo")?,
            g_f_rope_hi: rope_gather(
                stream,
                mla,
                kv_width,
                half..mla.n_head,
                "Gather::new f_rope_hi",
            )?,
            part_v: f32n(crate::flash::partials_v_len(mla.n_head, ctx_max))?,
            part_ms: f32n(crate::flash::partials_ms_len(mla.n_head, ctx_max))?,
            step_params,
            params_host,
            pos_buf,
            n_keys_buf,
            token_buf,
            cs_buf,
            pos_host: 0,
            probe: crate::probe::Probe::load(ctx)?,
            probe_buf: f32n(32)?,
            probe_cfg: StepProbe::default(),
            dims,
        })
    }
}

/// Cross-check the geometry the enqueue leans on: the projection and
/// derived-plane row counts against the MLA metadata, and the rope and
/// half-split shapes the launches assume. `first` is the stage's first
/// layer; `q_rows` and `kv_width` are its projections' row counts.
fn check_geometry(
    w: &Weights,
    mla: &MlaParams,
    first: &LayerNames,
    q_rows: usize,
    kv_width: usize,
) -> Result<(), GpuError> {
    let (_, qn2_d) = q8_derived(w, &first.derived)?;
    let (derived, derived_k) = (qn2_d.rows(), qn2_d.cols() * 32);
    let kv_b_rows = kq_weight(w, &first.attn_kv_b)?.rows();
    let check = |what: &'static str, want: usize, got: usize| -> Result<(), GpuError> {
        if want == got {
            Ok(())
        } else {
            Err(GpuError::shape(
                "LayerScratch",
                format!("{what}: {want} != {got}"),
            ))
        }
    };
    check(
        "attn_q rows vs n_head*kq_head",
        q_rows,
        mla.n_head * mla.kq_head,
    )?;
    check(
        "kv_a rows vs latent+rope",
        kv_width,
        mla.latent + mla.rope_dims,
    )?;
    check(
        "derived rows vs n_head*latent",
        derived,
        mla.n_head * mla.latent,
    )?;
    check("derived k vs nope", derived_k, mla.nope)?;
    check(
        "kv_b rows vs n_head*(nope+v_head)",
        kv_b_rows,
        mla.n_head * (mla.nope + mla.v_head),
    )?;
    if mla.rope_dims == 0 {
        return Err(GpuError::shape(
            "LayerScratch",
            "rope_dims must be at least 1",
        ));
    }
    if !q_rows.is_multiple_of(mla.rope_dims) {
        return Err(GpuError::shape(
            "LayerScratch",
            format!(
                "rope walks 64-value columns from the buffer start; q rows \
                 {q_rows} are not {}-aligned",
                mla.rope_dims
            ),
        ));
    }
    let half = mla.n_head / 2;
    if !mla.n_head.is_multiple_of(2) || half > 8 {
        return Err(GpuError::shape(
            "LayerScratch",
            format!(
                "the half-split m = 8 quantize at the wv_b site needs an even \
                 n_head <= 16, got {}",
                mla.n_head
            ),
        ));
    }
    Ok(())
}

impl DenseScratch {
    /// The dense arena for a gate/up width of `ff`. Load-time only.
    fn new(stream: &CudaStream, ff: usize) -> Result<DenseScratch, GpuError> {
        Ok(DenseScratch {
            h: DeviceBuffer::<f32>::zeroed(stream, ff)?,
            act32: Q8Blocks32::new(stream, ff, 1)?,
        })
    }
}

impl MoeScratch {
    /// The routed arena for the shapes `m` at `hidden`. Load-time only.
    fn new(stream: &CudaStream, m: &MoeDims, hidden: usize) -> Result<MoeScratch, GpuError> {
        let f32n = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(MoeScratch {
            normed: f32n(hidden)?,
            logits: f32n(m.n_expert)?,
            probs: f32n(m.n_expert)?,
            ids: DeviceBuffer::<u32>::zeroed(stream, m.n_used)?,
            weights: f32n(m.n_used)?,
            h_exp: f32n(m.n_used * m.ff)?,
            act32_exp: Q8Blocks32::new(stream, m.ff, m.n_used)?,
            down: f32n(m.n_used * hidden)?,
            h_sh: f32n(m.shexp_ff)?,
            act_sh: Q8Act::with_k(stream, 1, m.shexp_ff)?,
            shexp: f32n(hidden)?,
        })
    }
}

/// The per-step parameter image and its four windows, as
/// `LayerScratch::new` moves them into the arena.
struct ParamImage {
    step_params: DeviceBuffer<u32>,
    params_host: Vec<u32>,
    pos_buf: ManuallyDrop<DeviceBuffer<u32>>,
    n_keys_buf: ManuallyDrop<DeviceBuffer<u32>>,
    token_buf: ManuallyDrop<DeviceBuffer<u32>>,
    cs_buf: ManuallyDrop<DeviceBuffer<f32>>,
}

impl ParamImage {
    /// Allocate the image for a rope table of `rope` values and cut the
    /// windows out of it. Load-time only.
    fn new(stream: &CudaStream, rope: usize) -> Result<ParamImage, GpuError> {
        // The step image starts where `refresh_params` would leave position 0:
        // token 0, pos 0, one live key, a zeroed rope table.
        let mut params_host = vec![0u32; SP_CS + rope];
        params_host[SP_N_KEYS] = 1;
        let step_params = DeviceBuffer::from_host(stream, &params_host)?;
        // SAFETY: each window is inside `step_params`'s extent by the `SP_*`
        // layout, every offset is a u32 multiple and so four-byte aligned, and
        // `step_params` moves into the arena beside them (a move of the handle,
        // not of the allocation), where it outlives them and is never
        // reallocated.
        let (cs_buf, token_buf, pos_buf, n_keys_buf) = unsafe {
            (
                param_view::<f32>(&step_params, SP_CS, rope),
                param_view::<u32>(&step_params, SP_TOKEN, 1),
                param_view::<u32>(&step_params, SP_POS, 1),
                param_view::<u32>(&step_params, SP_N_KEYS, 1),
            )
        };
        Ok(ParamImage {
            step_params,
            params_host,
            pos_buf,
            n_keys_buf,
            token_buf,
            cs_buf,
        })
    }
}

/// The gather table that copies each rope slice of q (`q_rope_all`'s
/// 64-value column holding head `h`'s rope span) into the rope span of
/// flash row `h`, for the heads in `heads`. Load-time only.
fn rope_gather(
    stream: &CudaStream,
    mla: &MlaParams,
    kv_width: usize,
    heads: std::ops::Range<usize>,
    what: &'static str,
) -> Result<Gather, GpuError> {
    let (rope, nope, kq_head) = (mla.rope_dims, mla.nope, mla.kq_head);
    Gather::new(
        stream,
        heads.flat_map(|h| {
            let col = (h * kq_head + nope) / rope;
            (0..rope).map(move |d| (col * rope + d, h * kv_width + d))
        }),
        what,
    )
}
