//! The nemotron_h `ChainBody`: the file's shape read and checked once at
//! load, each layer's weight names and site types resolved once, the caches
//! and the scan stores allocated once, and the per-step record refreshed
//! before every enqueue or replay.

use super::step::{
    Acts, AttnLayer, Kernels, Layer, MambaLayer, MlpLayer, ROW_WORDS, SITE_TYPES, Scratch, Shape,
    Site, WORD_POS, WORD_TOKEN,
};
use crate::flash_gqa::{FlashGqaKernels, GROUP_5, HEAD};
use crate::gemm::Gemm32Kernels;
use crate::head::Head;
use crate::mamba2::{self, Mamba2Kernels};
use crate::model::{ChainBody, GpuModel, NoHost, block_count};
use crate::recur::RecStore;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::nemotron_h::hparams::{Hparams, Kind};
use model::arch::nemotron_h::names;
use model::mamba2::Geom;
use std::ops::Range;

const WHAT: &str = "nemotron_h::Body::load";

/// The input of one decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeInput {
    pub token: u32,
    pub pos: u32,
}

/// The nemotron_h chain over one sequence on one card.
pub struct Body {
    pub(super) hp: Hparams,
    /// The Q5_0 token table's name.
    pub(super) embed: String,
    pub(super) sh: Shape,
    pub(super) layers: Vec<Layer>,
    pub(super) k: Kernels,
    pub(super) acts: Acts,
    pub(super) s: Scratch,
    /// The residual stream's second buffer ([`Body::enqueue`]).
    pub(super) r: DeviceBuffer<f32>,
    /// Host zeros as long as the longest recurrent buffer, what a reset
    /// copies over every store.
    zeros: Vec<f32>,
    /// The step record's host words, rewritten by every refresh.
    host_words: Vec<u32>,
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
    fn load(gpu: &Gpu, file: &Split, w: &Weights, ctx: usize) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        if ctx == 0 {
            return Err(GpuError::shape(WHAT, "a cache of 0 rows"));
        }
        let ssm = hp.ssm;
        let mamba = mamba2::check(
            WHAT,
            Geom {
                d_inner: ssm.d_inner,
                n_heads: ssm.n_heads,
                n_groups: ssm.n_groups,
                d_state: ssm.d_state,
                d_conv: ssm.d_conv,
            },
        )?;
        let n_kv = attention_heads(&hp)?;
        let hidden = hp.n_embd;
        let kinds = Names { file, w };
        let embed = names::token_embd();
        kinds.site(&embed, (hp.n_vocab, hidden), &[SiteTy::Q5_0])?;
        let stream = gpu.stream();
        let layers = (0..hp.n_layer)
            .map(|l| kinds.layer(&hp, l, ctx, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let sh = Shape {
            hidden,
            eps: hp.rms_eps,
            mamba,
            n_kv,
            ctx,
        };
        let ff = hp.ff.iter().copied().max().unwrap_or(0);
        let longest = layers
            .iter()
            .filter_map(|l| match l {
                Layer::Mamba(m) => Some(m.store.state.len().max(m.store.ring.len())),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let ctx_ = gpu.context();
        Ok(Body {
            k: Kernels {
                mamba: Mamba2Kernels::load(ctx_)?,
                flash: FlashGqaKernels::load(ctx_)?,
                g32: Gemm32Kernels::load(ctx_)?,
                kgemv: KGemvKernels::load(gpu)?,
            },
            acts: Acts::new(stream, layers.iter().flat_map(Layer::sites))?,
            s: Scratch::new(stream, &sh, hp.n_head, ff)?,
            r: DeviceBuffer::zeroed(stream, hidden)?,
            zeros: vec![0.0; longest],
            host_words: vec![0; ROW_WORDS + n_kv],
            sh,
            layers,
            embed,
            hp,
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

/// The key heads of the attention layers, which the body takes equal across
/// layers: the planes, the row words and the flash group are cut for one.
/// Heads of 128 values and five query heads a key head are what the flash
/// entry runs; any other is refused by name.
fn attention_heads(hp: &Hparams) -> Result<usize, GpuError> {
    let mut found = None;
    for (l, &kind) in hp.kinds.iter().enumerate() {
        if kind != Kind::Attention {
            continue;
        }
        let kv = hp.kv[l];
        if *found.get_or_insert(kv) != kv {
            return Err(GpuError::shape(
                WHAT,
                format!("attention layer {l} has {kv} key heads, an earlier one {found:?}"),
            ));
        }
        if hp.key_len != HEAD || hp.value_len != HEAD || hp.n_head != kv * GROUP_5 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "attention of {} heads over {kv} key heads of {}/{} values: the flash runs \
                     {GROUP_5} query heads a key head of {HEAD} values",
                    hp.n_head, hp.key_len, hp.value_len
                ),
            ));
        }
    }
    Ok(found.unwrap_or(0))
}

/// The file and its resident weights, what a load resolves names against.
struct Names<'a> {
    file: &'a Split,
    w: &'a Weights,
}

impl Names<'_> {
    /// Site `name` of `rows` rows of `k` values: its type from the file's
    /// header (one of `allowed`), checked against the resident weight.
    fn site(
        &self,
        name: &str,
        (rows, k): (usize, usize),
        allowed: &[SiteTy],
    ) -> Result<Site, GpuError> {
        let ty = site::file_site(self.file, WHAT, name, (rows, k), allowed)?;
        site::site(self.w, WHAT, name, (rows, k), ty)?;
        Ok(Site {
            name: name.to_string(),
            ty,
            k,
        })
    }

    /// The projection `stem` of layer `l`.
    fn proj(&self, l: usize, stem: &str, shape: (usize, usize)) -> Result<Site, GpuError> {
        self.site(&layer_name(l, stem), shape, SITE_TYPES)
    }

    /// The f32 vector `stem` of layer `l`, `len` values: its name.
    fn vector(&self, l: usize, stem: &str, len: usize) -> Result<String, GpuError> {
        let name = layer_name(l, stem);
        let got = self.w.f32_buf(WHAT, &name)?.len();
        if got != len {
            return Err(GpuError::shape(
                WHAT,
                format!("{name} holds {got} values, the layer reads {len}"),
            ));
        }
        Ok(name)
    }

    fn layer(
        &self,
        hp: &Hparams,
        l: usize,
        ctx: usize,
        stream: &CudaStream,
    ) -> Result<Layer, GpuError> {
        let hidden = hp.n_embd;
        let norm = self.vector(l, names::ATTN_NORM, hidden)?;
        Ok(match hp.kinds[l] {
            Kind::Mamba => {
                let s = hp.ssm;
                let lens = [s.d_inner * s.d_state, crate::linear::RING_ROWS * s.conv_dim];
                Layer::Mamba(MambaLayer {
                    ssm_in: self.proj(l, names::SSM_IN, (s.d_in_proj, hidden))?,
                    ssm_out: self.proj(l, names::SSM_OUT, (hidden, s.d_inner))?,
                    a: self.vector(l, names::SSM_A, s.n_heads)?,
                    d: self.vector(l, names::SSM_D, s.n_heads)?,
                    dt_bias: self.vector(l, names::SSM_DT_BIAS, s.n_heads)?,
                    conv_w: self.vector(l, names::SSM_CONV, s.d_conv * s.conv_dim)?,
                    conv_b: self.vector(l, names::SSM_CONV_BIAS, s.conv_dim)?,
                    ssm_norm: self.vector(l, names::SSM_NORM, s.d_inner)?,
                    store: RecStore::new(stream, lens, 1)?,
                    norm,
                })
            }
            Kind::Attention => {
                let kv = hp.kv[l] * HEAD;
                Layer::Attn(AttnLayer {
                    q: self.proj(l, names::ATTN_Q, (hp.n_head * HEAD, hidden))?,
                    k: self.proj(l, names::ATTN_K, (kv, hidden))?,
                    v: self.proj(l, names::ATTN_V, (kv, hidden))?,
                    o: self.proj(l, names::ATTN_OUTPUT, (hidden, hp.n_head * HEAD))?,
                    kc: DeviceTensor::zeroed(stream, hp.kv[l] * ctx, HEAD)?,
                    vc: DeviceTensor::zeroed(stream, hp.kv[l] * ctx, HEAD)?,
                    norm,
                })
            }
            Kind::Mlp => {
                let ff = hp.ff[l];
                Layer::Mlp(MlpLayer {
                    up: self.proj(l, names::FFN_UP, (ff, hidden))?,
                    down: self.proj(l, names::FFN_DOWN, (hidden, ff))?,
                    ff,
                    norm,
                })
            }
        })
    }
}

/// `blk.{l}.{stem}`.
fn layer_name(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
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

    /// The step record in one copy: the position, the token and, for each key
    /// head, the row `h · ctx + pos` its plane's append lands at.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        let DecodeInput { token, pos } = *input;
        let ctx = self.sh.ctx;
        self.host_words[WORD_POS] = pos;
        self.host_words[WORD_TOKEN] = token;
        for h in 0..self.sh.n_kv {
            let row = h
                .checked_mul(ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        "nemotron_h::Body::refresh",
                        format!("row {h} · {ctx} + {pos} does not fit a u32 row word"),
                    )
                })?;
            self.host_words[ROW_WORDS + h] = row;
        }
        Ok(self.s.words.copy_from_host(stream, &self.host_words)?)
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
                m.store.clear(gpu.stream(), &self.zeros)?;
            }
        }
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.hp.rms_eps
    }

    fn resident_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Mamba(m) => m.store.bytes(),
                Layer::Attn(a) => a.kc.buf().num_bytes() + a.vc.buf().num_bytes(),
                Layer::Mlp(_) => 0,
            })
            .sum();
        layers + self.s.bytes() + self.acts.bytes() + self.r.num_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}
