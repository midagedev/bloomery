//! The qwen3moe `ChainBody`: the file's shape read and checked once at load,
//! each layer's weight names and quantization types resolved once, the arena
//! and the K/V planes allocated once, and the per-step parameters refreshed
//! before every enqueue or replay (docs/arch-split.md).

use super::dispatch;
use super::experts::ExpertKernels;
use super::head_argmax::{HeadArgmaxKernels, HeadArgmaxState};
use super::prefill::Prefill;
use super::proj::ProjKernels;
use super::router::{N_EXPERT, N_USED, RouterKernels};
use super::scratch::{Arena, Dims, KvPlanes, SP_CS, SP_N_KEYS, SP_POS, SP_TOKEN, StepParams};
use crate::flash_gqa::{FlashGqaKernels, GROUP, HEAD, gqa_mma};
use crate::head::Head;
use crate::model::{ChainBody, StepProbe};
use crate::q6k_sel::Q6kSelKernels;
use crate::rope_neox::RopeNeoxKernels;
use crate::rope_table::{Direction, RopeSpec, RopeTable};
use crate::tensor::window;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::Arch;
use model::arch::qwen3moe::hparams::Hparams;
use model::arch::qwen3moe::names;
use std::mem::ManuallyDrop;
use std::ops::Range;

/// The two K-quants a qwen3moe projection comes in: the attention value
/// projection and the experts' down stack are Q4_K on some layers and Q6_K
/// on the rest; every other projection is Q4_K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kq {
    Q4K,
    Q6K,
}

/// One layer's weight names and the quantization of its two mixed sites,
/// resolved at load.
pub(super) struct LayerNames {
    pub(super) attn_norm: String,
    pub(super) attn_q: String,
    pub(super) attn_k: String,
    pub(super) attn_v: String,
    pub(super) attn_q_norm: String,
    pub(super) attn_k_norm: String,
    pub(super) attn_output: String,
    pub(super) ffn_norm: String,
    pub(super) ffn_gate_inp: String,
    pub(super) ffn_gate_exps: String,
    pub(super) ffn_up_exps: String,
    pub(super) ffn_down_exps: String,
    pub(super) v_ty: Kq,
    pub(super) down_ty: Kq,
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
}

/// qwen3moe's per-replay host values: the token the chain embeds and the
/// cache row it lands in. The live key count and the rope table follow.
pub struct DecodeInput {
    token: u32,
    pos: u32,
}

/// Everything qwen3moe's chain owns on the device.
pub struct Body {
    /// The prompt prefill's arena, image, slot and captured passes. Declared
    /// first: fields drop in declaration order, and its graphs address the
    /// cache planes below.
    pub(super) prefill: Prefill,
    pub(super) hp: Hparams,
    pub(super) names: Vec<LayerNames>,
    pub(super) kv: Vec<KvPlanes>,
    /// The decode step's one-row arena and its parameter image.
    pub(super) s: Arena,
    pub(super) sp: StepParams,
    pub(super) k: Kernels,
    /// The fused head argmax's key and ticket, back at their seeds after
    /// every launch.
    pub(super) head_state: HeadArgmaxState,
    pub(super) rope: RopeTable,
    /// The host rope row `refresh` fills, reused every step.
    cs_host: Vec<f32>,
    /// The flash pass this process runs ([`gqa_mma`]), read once at load.
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
/// that the file does not carry: the flash head and group, the router's
/// expert counts, the rope width.
fn pins(hp: &Hparams) -> Result<(), GpuError> {
    let checks = [
        ("attention.key_length (the flash head)", HEAD, hp.head_dim),
        (
            "query heads per key head (the flash group)",
            GROUP,
            hp.group(),
        ),
        ("expert_count (the router)", N_EXPERT, hp.experts.n_expert),
        ("expert_used_count (the router)", N_USED, hp.experts.n_used),
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
    if !hp.experts.weights_norm {
        return Err(GpuError::shape(
            "qwen3moe::Body::load",
            "the router kernel renormalizes the chosen weights; this file does not",
        ));
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
    Ok(())
}

/// Weight `name` as a K-quant word plane of `rows` rows of `k` values, and
/// which of the two K-quants it is; `allowed` says which it may be.
fn kq_site(w: &Weights, name: &str, rows: usize, k: usize, allowed: &[Kq]) -> Result<Kq, GpuError> {
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
fn f32_site(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
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

impl LayerNames {
    /// Layer `l`'s names, every weight checked against the shape and type
    /// its launch takes.
    fn resolve(w: &Weights, hp: &Hparams, l: usize) -> Result<LayerNames, GpuError> {
        let (h, q, kv, ff) = (
            hp.n_embd,
            hp.n_head * hp.head_dim,
            hp.n_head_kv * hp.head_dim,
            hp.experts.ff,
        );
        let e = hp.experts.n_expert;
        let n = LayerNames {
            attn_norm: names::attn_norm(l),
            attn_q: names::attn_q(l),
            attn_k: names::attn_k(l),
            attn_v: names::attn_v(l),
            attn_q_norm: names::attn_q_norm(l),
            attn_k_norm: names::attn_k_norm(l),
            attn_output: names::attn_output(l),
            ffn_norm: names::ffn_norm(l),
            ffn_gate_inp: names::ffn_gate_inp(l),
            ffn_gate_exps: names::ffn_gate_exps(l),
            ffn_up_exps: names::ffn_up_exps(l),
            ffn_down_exps: names::ffn_down_exps(l),
            v_ty: Kq::Q4K,
            down_ty: Kq::Q4K,
        };
        for (name, len) in [
            (&n.attn_norm, h),
            (&n.ffn_norm, h),
            (&n.attn_q_norm, hp.head_dim),
            (&n.attn_k_norm, hp.head_dim),
        ] {
            f32_site(w, name, 1, len)?;
        }
        f32_site(w, &n.ffn_gate_inp, e, h)?;
        kq_site(w, &n.attn_q, q, h, &[Kq::Q4K])?;
        kq_site(w, &n.attn_k, kv, h, &[Kq::Q4K])?;
        let v_ty = kq_site(w, &n.attn_v, kv, h, &[Kq::Q4K, Kq::Q6K])?;
        kq_site(w, &n.attn_output, h, q, &[Kq::Q4K])?;
        kq_site(w, &n.ffn_gate_exps, e * ff, h, &[Kq::Q4K])?;
        kq_site(w, &n.ffn_up_exps, e * ff, h, &[Kq::Q4K])?;
        let down_ty = kq_site(w, &n.ffn_down_exps, e * h, ff, &[Kq::Q4K, Kq::Q6K])?;
        Ok(LayerNames { v_ty, down_ty, ..n })
    }
}

impl Body {
    /// Keep a copy of every layer's output residual after each layer of the
    /// chain (`on`), or stop. The copies are one memcpy node per layer in a
    /// capture; the gates read them in eager mode. Load-time allocation.
    pub(super) fn set_taps(&mut self, stream: &CudaStream, on: bool) -> Result<(), GpuError> {
        self.taps = None;
        if on {
            let hidden = self.s.dims.hidden;
            let buf = DeviceBuffer::<f32>::zeroed(stream, self.names.len() * hidden)?;
            let rows = (0..self.names.len())
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

impl ChainBody for Body {
    type Input = DecodeInput;
    type Meta = ();

    fn arch() -> Arch {
        Arch::Qwen3moe
    }

    /// Nothing is derived: every weight runs from the file's bytes.
    fn derive(
        _stream: &CudaStream,
        _file: &Split,
        _layers: Range<usize>,
        _w: &mut Weights,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
    ) -> Result<Body, GpuError> {
        let what = "qwen3moe::Body::load";
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(what, e))?;
        pins(&hp)?;
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
        let names = layers
            .clone()
            .map(|l| LayerNames::resolve(w, &hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        let dims = Dims {
            hidden: hp.n_embd,
            n_head: hp.n_head,
            n_kv: hp.n_head_kv,
            ff: hp.experts.ff,
            ctx: ctx_max,
        };
        let stream = gpu.stream();
        let kv = layers
            .map(|_| KvPlanes::new(stream, &dims))
            .collect::<Result<Vec<_>, _>>()?;
        let ctx = gpu.context();
        let k = Kernels {
            proj: ProjKernels::load(ctx)?,
            neox: RopeNeoxKernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            q6_sel: Q6kSelKernels::load(ctx)?,
            head: HeadArgmaxKernels::load(ctx)?,
        };
        let rope = RopeTable::new(&RopeSpec::window(hp.rope.base, hp.rope.dims))?;
        Ok(Body {
            prefill: Prefill::new(stream, dims, ctx_max)?,
            s: Arena::new(stream, dims, 1)?,
            sp: StepParams::new(stream)?,
            hp,
            names,
            kv,
            k,
            head_state: HeadArgmaxState::new(stream)?,
            rope,
            cs_host: Vec::with_capacity(HEAD),
            mma: gqa_mma(),
            taps: None,
        })
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// Token, landing row, live key count and the position's rope table in
    /// one host-to-device copy.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        let DecodeInput { token, pos } = *input;
        let sp = &mut self.sp;
        sp.host.truncate(SP_CS);
        sp.host[SP_TOKEN] = token;
        sp.host[SP_POS] = pos;
        sp.host[SP_N_KEYS] = pos + 1;
        self.cs_host.clear();
        self.rope.push(pos, Direction::Forward, &mut self.cs_host);
        sp.host.extend(self.cs_host.iter().map(|v| v.to_bits()));
        sp.buf.copy_from_host(stream, &sp.host)?;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        dispatch::enqueue_chain(gpu, w, self, head)
    }

    /// Nothing to clear: the flash never loads a key row at or past the live
    /// count, and every row below it is written by its own step first.
    fn reset(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        Ok(())
    }

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

    /// The chain carries no probe arms: only the off state is accepted.
    fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError> {
        if probe == StepProbe::default() {
            Ok(())
        } else {
            Err(GpuError::state(
                "qwen3moe::Body::set_probe",
                "a probe arm — this chain has none",
            ))
        }
    }

    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        self.kv.iter().map(KvPlanes::bytes).sum::<usize>()
            + self.s.bytes()
            + self.sp.buf.num_bytes()
            + self.head_state.bytes()
            + self.prefill.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
    }
}
