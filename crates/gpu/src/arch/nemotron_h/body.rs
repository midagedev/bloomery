//! The nemotron_h `ChainBody`: the file's shape read and checked once at
//! load, each layer's weight names and site types resolved once, the caches
//! and the scan stores allocated once, and the per-step record refreshed
//! before every enqueue or replay.

use super::prefill::{PASS_ROWS, Prompt};
use super::step::{Layer, MlpLayer, SITE_TYPES};
use crate::arch::hybrid::{
    self, Acts, DecodeInput, Kernels, Names, Scratch, Words, attention_heads, mamba_shape,
};
use crate::flash_gqa::GROUP_5;
use crate::head::Head;
use crate::model::{ChainBody, GpuModel, NoHost, block_count};
use crate::site::SiteTy;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use chain::arch::hybrid::plan::LayerPlan;
use chain::arch::nemotron_h::plan::Plan;
use chain::arch::nemotron_h::read::layer_plans;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::nemotron_h::hparams::{Hparams, Kind};
use model::arch::nemotron_h::{names, spec};
use std::ops::Range;

const WHAT: &str = "nemotron_h::Body::load";

/// The nemotron_h chain over one sequence on one card.
pub struct Body {
    /// The Q5_0 token table's name.
    pub(super) embed: String,
    pub(super) sh: hybrid::Shape,
    /// The chain's plans, one a layer: the walk's order, from the file's header.
    pub(super) plans: Vec<Plan>,
    pub(super) layers: Vec<Layer>,
    pub(super) k: Kernels,
    pub(super) acts: Acts,
    pub(super) s: Scratch,
    /// The residual stream's second buffer ([`Body::enqueue`]).
    pub(super) r: DeviceBuffer<f32>,
    /// A prompt pass's buffers ([`super::prefill`]).
    pub(super) prompt: Prompt,
    /// The step record's host words, rewritten by every refresh.
    words: Words,
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format, K/V planes of `ctx` rows
    /// a key head, the scan stores and the step scratch. The model takes the
    /// file and closes it once the weights are resident.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "nemotron_h GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            Body::load(gpu, &file, w, ctx)
        })
    }
}

impl Body {
    fn load(gpu: &Gpu, file: &Split, w: &mut Weights, ctx: usize) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let ssm = hp.ssm;
        let mamba = mamba_shape(WHAT, &ssm)?;
        let n_kv = attention_heads(
            WHAT,
            hp.kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| **k == Kind::Attention)
                .map(|(l, _)| (l, hp.kv[l])),
            (hp.n_head, hp.key_len, hp.value_len),
            GROUP_5,
        )?;
        let hidden = hp.n_embd;
        let stream = gpu.stream();
        hybrid::derive_zero_conv_bias(
            w,
            file,
            hp.kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| **k == Kind::Mamba)
                .map(|(l, _)| l),
            ssm.conv_dim,
            stream,
        )?;
        let read = spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let plans = layer_plans(file, WHAT, &read.spec.layers, hidden, SITE_TYPES)?;
        let names = Names {
            what: WHAT,
            file,
            w,
            types: SITE_TYPES,
        };
        let embed = names::token_embd();
        names.site(&embed, (hp.n_vocab, hidden), &[SiteTy::Q5_0])?;
        let layers = plans
            .iter()
            .map(|p| layer(&names, p, hidden, ctx, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let sh = hybrid::Shape {
            hidden,
            eps: hp.rms_eps,
            mamba,
            n_kv,
            ctx,
        };
        let ff = hp.ff.iter().copied().max().unwrap_or(0);
        Ok(Body {
            k: Kernels::load(gpu)?,
            prompt: Prompt::new(stream, PASS_ROWS.min(ctx), &sh, (hp.n_head, ff), &layers)?,
            acts: Acts::new(stream, layers.iter().flat_map(Layer::sites))?,
            s: Scratch::new(stream, &sh, hp.n_head, ff)?,
            r: DeviceBuffer::zeroed(stream, hidden)?,
            words: Words::new(n_kv),
            sh,
            plans,
            layers,
            embed,
        })
    }

    /// Layer `layer`'s scan state read back (a gate's instrument; it
    /// synchronizes `stream`). A layer that is not a Mamba-2 layer is refused
    /// by name.
    pub fn scan_state(&self, stream: &CudaStream, layer: usize) -> Result<Vec<f32>, GpuError> {
        match self.layers.get(layer) {
            Some(Layer::Mamba(m)) => Ok(m.store.state.to_host_vec(stream)?),
            _ => Err(GpuError::shape(
                "nemotron_h::Body::scan_state",
                format!("layer {layer} holds no scan state"),
            )),
        }
    }
}

/// A layer of the kind its plan names, resolved against `names`.
fn layer(
    names: &Names<'_>,
    plan: &Plan,
    hidden: usize,
    ctx: usize,
    stream: &CudaStream,
) -> Result<Layer, GpuError> {
    Ok(match plan {
        LayerPlan::Mamba(p) => Layer::Mamba(names.mamba_of(p, hidden, stream)?),
        LayerPlan::Attn(p) => Layer::Attn(names.attn_of(p, (hidden, ctx), stream)?),
        LayerPlan::Ffn(p) => Layer::Mlp(MlpLayer {
            norm: names.planned_vector(&p.norm, hidden)?,
            up: names.planned(&p.up, p.up_ty, (p.ff, hidden))?,
            down: names.planned(&p.down, p.down_ty, (hidden, p.ff))?,
            ff: p.ff,
        }),
    })
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = NoHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::NemotronH
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step record in one copy ([`Words::refresh`]).
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        self.words.refresh(
            "nemotron_h::Body::refresh",
            stream,
            &self.sh,
            input,
            &mut self.s.words,
        )
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue(gpu, w, head)
    }

    /// Every scan state and conv ring back to zero. The K/V planes need
    /// nothing: the flash never reads a row at or past the live count, and
    /// every row below it is written by its own step first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            if let Layer::Mamba(m) = layer {
                m.store.clear(gpu.stream())?;
            }
        }
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn resident_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Mamba(m) => m.store.bytes(),
                Layer::Attn(a) => a.bytes(),
                Layer::Mlp(_) => 0,
            })
            .sum();
        layers
            + self.s.bytes()
            + self.acts.bytes()
            + self.r.num_bytes()
            + self.k.kgemv.device_bytes()
            + self.prompt.bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}
