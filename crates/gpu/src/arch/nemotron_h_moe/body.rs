//! The nemotron_h_moe `ChainBody`: the file's shape read and checked once at
//! load, each layer's weight names, site types and router instance resolved
//! once, the caches, the scan stores and the router results allocated once,
//! and the per-step record refreshed before every enqueue or replay.

use super::moe::{MoeKernels, MoeLayer, MoeScratch, MoeShape, Routing, moe_layer, routed};
use crate::arch::hybrid::{
    self, Acts, AttnLayer, Cx, DecodeInput, Kernels, MambaLayer, Names, Site, WORD_POS, WORD_TOKEN,
    Words, attention_heads, attn_layer, mamba_layer, mamba_shape,
};
use crate::flash_gqa::GROUP_16;
use crate::head::Head;
use crate::iq_sel::IqSelKernels;
use crate::model::{ChainBody, GpuModel, MAX_PASS_ROWS, NoHost, block_count};
use crate::q38::{EmbedQ8Args, Q38Kernels};
use crate::site::SiteTy;
use crate::weights::Weights;
use crate::{Gpu, GpuError, Window};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Split};
use model::arch::Arch;
use model::arch::names::blk;
use model::arch::nemotron_h_moe::hparams::{Hparams, Kind};
use model::arch::nemotron_h_moe::names;
use std::ops::Range;

use super::prefill::{PASS_ROWS, Prompt, derive_q8};
use super::router::Routers;
use super::verify::Verify;

const WHAT: &str = "nemotron_h_moe::Body::load";
const STEP: &str = "nemotron_h_moe::Body::enqueue_chain";

/// The file types a Mamba-2 or attention projection launches: NVFP4 through
/// its gemv on the q8_1 rows, Q8_0 through the f32 one.
const SITE_TYPES: &[SiteTy] = &[SiteTy::Nvfp4, SiteTy::Q8_0];

/// A layer, by the kind the file's per-layer arrays give it.
pub(super) enum Layer {
    Mamba(MambaLayer),
    Attn(AttnLayer),
    Moe(MoeLayer),
}

impl Layer {
    pub(super) fn sites(&self) -> Vec<&Site> {
        match self {
            Layer::Mamba(m) => m.sites().to_vec(),
            Layer::Attn(a) => a.sites().to_vec(),
            Layer::Moe(m) => m.sites().to_vec(),
        }
    }
}

/// The nemotron_h_moe chain over one sequence on one card.
pub struct Body {
    /// The Q8_0 token table's name.
    pub(super) embed: String,
    pub(super) sh: hybrid::Shape,
    pub(super) moe: MoeShape,
    pub(super) layers: Vec<Layer>,
    pub(super) k: Kernels,
    pub(super) mk: MoeKernels,
    pub(super) q38: Q38Kernels,
    acts: Acts,
    s: hybrid::Scratch,
    ms: MoeScratch,
    /// The residual stream's second buffer ([`Body::enqueue`]).
    r: DeviceBuffer<f32>,
    /// Host zeros as long as the longest recurrent buffer, what a reset
    /// copies over every store.
    zeros: Vec<f32>,
    /// The step record's host words, rewritten by every refresh.
    words: Words,
    /// The prompt pass's buffers ([`Body::enqueue_pass`]).
    pub(super) prompt: Prompt,
    /// The Mamba-2 layers' verify lanes and the verify they wait on
    /// ([`GpuModel::open_verifying`]); `None` on a plain load, which holds
    /// no bytes and verifies nothing.
    pub(super) verify: Option<Verify>,
    /// The positions the recurrent stores hold: a call's end once its
    /// launches are enqueued. The next position is the model's
    /// ([`GpuModel::pos`]); a rollback with no verify waiting is a no-op
    /// only at it.
    pub(super) held: u32,
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format (the IQ2_XXS stacks in the
    /// file's blocks), K/V planes of `ctx` rows a key head, the scan stores,
    /// the routers' results and the step scratch. The model takes the file
    /// and closes it once the weights are resident.
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "nemotron_h_moe GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            Body::load(gpu, &file, w, ctx, None)
        })
    }

    /// [`GpuModel::open`] for a load that verifies drafted rows ([`Rows`]):
    /// beside the plain load's buffers, each Mamba-2 layer keeps `rows − 1`
    /// stamped states — the state after each row but the last of a pass — so
    /// a verify of up to `rows` rows keeps any `1..=m` of them at its commit
    /// by one device copy a Mamba-2 layer. A plain load is byte for byte the
    /// plain [`GpuModel::open`]. `rows` outside `2..=`[`MAX_PASS_ROWS`] is
    /// refused by name, before anything uploads.
    pub fn open_verifying(
        gpu: Gpu,
        file: Split,
        ctx: usize,
        rows: usize,
    ) -> Result<GpuModel<Body>, GpuError> {
        const WHAT: &str = "nemotron_h_moe GpuModel::open_verifying";
        if !(2..=MAX_PASS_ROWS).contains(&rows) {
            return Err(GpuError::shape(
                WHAT,
                format!("a verify of {rows} rows; a verifying load holds 2..={MAX_PASS_ROWS}"),
            ));
        }
        let n_layers = block_count(&file, WHAT)?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            Body::load(gpu, &file, w, ctx, Some(rows))
        })
    }
}

impl Body {
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &mut Weights,
        ctx: usize,
        verify_rows: Option<usize>,
    ) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        // A companion `.scale` is read for NVFP4 projections only
        // (weights::tensor_scale); a scaled Q8_0 ssm_in/ssm_out would enter
        // the scan unscaled, so the load refuses it by name.
        for (l, kind) in hp.kinds.iter().enumerate() {
            if *kind != Kind::Mamba {
                continue;
            }
            for (stem, scale) in [
                (names::SSM_IN, names::SSM_IN_SCALE),
                (names::SSM_OUT, names::SSM_OUT_SCALE),
            ] {
                let name = blk(l, stem);
                if let Some((_, t)) = file.find(&name)
                    && t.ty != GgmlType::NVFP4
                    && file.find(&blk(l, scale)).is_some()
                {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "{name}: a companion {scale} is read for NVFP4 only, this \
                             projection is {}",
                            t.ty
                        ),
                    ));
                }
            }
        }
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
            GROUP_16,
        )?;
        let hidden = hp.n_embd;
        let stream = gpu.stream();
        let mut names = Names {
            what: WHAT,
            file,
            w,
            types: SITE_TYPES,
        };
        let embed = names::token_embd();
        names.site(&embed, (hp.n_vocab, hidden), &[SiteTy::Q8_0])?;
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let norm = names.vector(l, names::ATTN_NORM, hidden)?;
            layers.push(match hp.kinds[l] {
                Kind::Mamba => Layer::Mamba(names.mamba((l, norm), &ssm, hidden, stream)?),
                Kind::Attention => Layer::Attn(names.attn(
                    (l, norm),
                    (hp.n_head, hp.kv[l]),
                    (hidden, ctx),
                    stream,
                )?),
                Kind::Moe => Layer::Moe(MoeLayer::load(&names, (l, norm), &hp, stream)?),
            });
        }
        let mut acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        for l in &layers {
            if let Layer::Moe(m) = l {
                for (k, cols) in m.forms(&hp) {
                    acts.add_q8_1(stream, k, cols)?;
                }
            }
        }
        let sh = hybrid::Shape {
            hidden,
            eps: hp.rms_eps,
            mamba,
            n_kv,
            ctx,
        };
        let moe = MoeShape {
            latent: hp.latent,
            shared_ff: hp.shared_ff,
            scale: hp.weights_scale,
        };
        let ff = hp.experts.iter().flatten().map(|x| x.ff).max().unwrap_or(0);
        let longest = layers
            .iter()
            .filter_map(|l| match l {
                Layer::Mamba(m) => Some(m.store.state.len().max(m.store.ring.len())),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let prompt = Prompt::new(stream, PASS_ROWS.min(ctx), (&sh, moe, hp.n_head), &layers)?;
        derive_q8(stream, file, w, &layers)?;
        let verify = verify_rows
            .map(|rows| Verify::new(stream, WHAT, &layers, rows, mamba.geom().d_inner))
            .transpose()?;
        let ctx_ = gpu.context();
        Ok(Body {
            prompt,
            verify,
            k: Kernels::load(gpu)?,
            mk: MoeKernels {
                routers: Routers::load(ctx_)?,
                iq: IqSelKernels::load(ctx_, gpu.fault_word())?,
            },
            q38: Q38Kernels::load(ctx_)?,
            acts,
            s: hybrid::Scratch::new(stream, &sh, hp.n_head, 0)?,
            ms: MoeScratch::new(stream, moe, hidden, ff)?,
            r: DeviceBuffer::zeroed(stream, hidden)?,
            zeros: vec![0.0; longest],
            words: Words::new(n_kv),
            sh,
            moe,
            layers,
            embed,
            held: 0,
        })
    }

    /// Layer `layer`'s scan state read back (a gate's instrument; it
    /// synchronizes `stream`). A layer that is not a Mamba-2 layer is refused
    /// by name.
    pub fn scan_state(&self, stream: &CudaStream, layer: usize) -> Result<Vec<f32>, GpuError> {
        match self.layers.get(layer) {
            Some(Layer::Mamba(m)) => Ok(m.store.state.to_host_vec(stream)?),
            _ => Err(GpuError::shape(
                "nemotron_h_moe::Body::scan_state",
                format!("layer {layer} holds no scan state"),
            )),
        }
    }

    /// MoE layer `layer`'s last routing read back ([`Routing`]: the kept
    /// ids in rank order, their weights and every expert's unbiased score; a
    /// gate's instrument, it synchronizes `stream`). A layer that is not a
    /// MoE layer is refused by name.
    pub fn routing(&self, stream: &CudaStream, layer: usize) -> Result<Routing, GpuError> {
        match self.layers.get(layer) {
            Some(Layer::Moe(m)) => routed(m, stream),
            _ => Err(GpuError::shape(
                "nemotron_h_moe::Body::routing",
                format!("layer {layer} routes no experts"),
            )),
        }
    }

    /// The whole chain: the embedding, every layer, then `head`. The residual
    /// stream alternates between the head's input and the body's own buffer,
    /// each layer writing the one it did not read, so the last layer's sum is
    /// the head's input whatever the layer count's parity.
    fn enqueue(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let cx = Cx {
            what: STEP,
            gpu,
            w,
            k: &self.k,
            sh: self.sh,
        };
        let (qs, d) = w.q8_planes(STEP, &self.embed)?;
        let hin = head.input_mut();
        let mut in_head = self.layers.len().is_multiple_of(2);
        {
            let t = &mut self.s;
            let ids = Window::<u32>::of(&t.words, WORD_TOKEN * size_of::<u32>(), 1)?;
            let pos0 = Window::<u32>::of(&t.words, WORD_POS * size_of::<u32>(), 1)?;
            let first = if in_head { &mut *hin } else { &mut self.r };
            self.q38.enqueue_embed_rows(
                gpu.stream(),
                EmbedQ8Args {
                    qs,
                    d,
                    ids: &ids,
                    pos0: &pos0,
                    first: 0,
                    fault: gpu.unlabelled_sink(),
                    y: first,
                    pos: &mut t.buf.pos,
                    n_keys: &mut t.buf.n_keys,
                },
            )?;
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let (src, dst) = if in_head {
                (&*hin, &mut self.r)
            } else {
                (&self.r as &DeviceBuffer<f32>, &mut *hin)
            };
            match layer {
                Layer::Mamba(m) => {
                    mamba_layer(&cx, (l, m), (&mut self.s, &mut self.acts), (src, dst))?;
                }
                Layer::Attn(a) => {
                    attn_layer::<GROUP_16>(&cx, (l, a), (&mut self.s, &mut self.acts), (src, dst))?;
                }
                Layer::Moe(m) => moe_layer(
                    &cx,
                    (&self.mk, self.moe),
                    (l, m),
                    (&mut self.s, &mut self.ms, &mut self.acts),
                    (src, dst),
                )?,
            }
            in_head = !in_head;
        }
        head.enqueue(gpu, w)
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = NoHost;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::NemotronHMoe
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step record in one copy ([`Words::refresh`]), refused while a
    /// verify waits for its commit.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        const WHAT: &str = "nemotron_h_moe::Body::step";
        let DecodeInput { token, pos } = *input;
        self.refuse_waiting(WHAT, &format!("a step at position {pos}"))?;
        self.held = pos + 1;
        let input = DecodeInput { token, pos };
        self.words
            .refresh(WHAT, stream, &self.sh, &input, &mut self.s.words)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue(gpu, w, head)
    }

    /// Every scan state and conv ring back to zero, and a waiting verify
    /// dropped at 0 rows kept. The K/V planes need nothing: the flash never
    /// reads a row at or past the live count, and every row below it is
    /// written by its own step first. The router's results are written whole
    /// by every step before any read.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            if let Layer::Mamba(m) = layer {
                m.store.clear(gpu.stream(), &self.zeros)?;
            }
        }
        if let Some(v) = self.verify.as_mut() {
            v.pending = None;
        }
        self.held = 0;
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
                Layer::Moe(m) => m.route.bytes(),
            })
            .sum();
        layers
            + self.prompt.bytes()
            + self.verify.as_ref().map_or(0, Verify::bytes)
            + self.s.bytes()
            + self.ms.bytes()
            + self.acts.bytes()
            + self.r.num_bytes()
            + self.k.kgemv.device_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}
