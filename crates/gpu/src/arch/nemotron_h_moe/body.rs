//! The nemotron_h_moe `ChainBody`: the file's shape read and checked once at
//! load, each layer's weight names, site types and router instance resolved
//! once, the caches, the scan stores and the router results allocated once,
//! and the per-step record refreshed before every enqueue or replay.

use super::moe::{MoeKernels, MoeLayer, MoeScratch, MoeShape, Routing, moe_layer, routed};
use super::placed::{PLACED_ROWS, Placed, PlacedOpen, StepWalk, plan_map};
use crate::arch::hybrid::{
    self, Acts, AttnLayer, Cx, DecodeInput, Kernels, MambaLayer, Names, Site, WORD_POS, WORD_TOKEN,
    Words, attention_heads, attn_layer, derive_zero_conv_bias, mamba_layer, mamba_shape,
};
use crate::flash_gqa::GROUP_16;
use crate::flash_gqa_prefill::FlashGqaPrefill;
use crate::head::Head;
use crate::host::StepLeg;
use crate::model::{ChainBody, GpuModel, block_count, refuse_scratch_past};
use crate::q38::{EmbedQ8Args, Q38Kernels};
use crate::quant_sel::QuantSelKernels;
use crate::site::SiteTy;
use crate::weights::Weights;
use crate::{Gpu, GpuError, Window};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Split};
use model::arch::Arch;
use model::arch::names::blk;
use model::arch::nemotron_h_moe::hparams::{Hparams, Kind};
use model::arch::nemotron_h_moe::names;
use model::placement::Plan;
use std::ops::Range;
use std::sync::Arc;

use super::prefill::{PASS_ROWS_MAX, PassSizes, Prompt, choose_pass_rows};
use super::router::Routers;

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
    /// The prompt pass's attention: the prefill flash's split entry over the
    /// pass's rows (the step path keeps the trunk's group-16 flash).
    pub(super) prefill: FlashGqaPrefill,
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
    /// The host tier and the card's side of it; `None` on a whole-card load.
    pub(super) placed: Option<Placed>,
}

/// How a load holds the routed experts.
enum Open<'a> {
    /// Every expert resident, the prompt pass's rows the user's or the
    /// chooser's.
    Whole { pass_rows: Option<usize> },
    /// The plan's card segment resident and the rest on the host tier.
    Placed(Box<PlacedOpen<'a>>),
}

impl GpuModel<Body> {
    /// The whole model of `file` resident on `gpu`, plus the output head:
    /// every tensor in its kernels' device format (the IQ2_XXS stacks in the
    /// file's blocks), K/V planes of `ctx` rows a key head, the scan stores,
    /// the routers' results and the step scratch. The model takes the file
    /// and closes it once the weights are resident. The prompt pass's rows
    /// are chosen at load from the card's room
    /// ([`choose_pass_rows`](super::prefill)): the largest that fits, at
    /// most [`PASS_ROWS_MAX`](super::prefill::PASS_ROWS_MAX).
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        Self::open_with_pass_rows(gpu, file, ctx, None)
    }

    /// [`GpuModel::open`](GpuModel::<Body>::open) with the prompt pass's rows
    /// as the user named them: `None` is the chooser's, `Some(rows)` exactly
    /// those or a named refusal (the free bytes, the need, the largest size
    /// that fits), never shrunk.
    pub fn open_with_pass_rows(
        gpu: Gpu,
        file: Split,
        ctx: usize,
        pass_rows: Option<usize>,
    ) -> Result<GpuModel<Body>, GpuError> {
        let n_layers = block_count(&file, "nemotron_h_moe GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, ctx, 0..n_layers, true, |gpu, w| {
            Body::derive(gpu.stream(), &file, w)?;
            Body::load(gpu, &file, w, ctx, Open::Whole { pass_rows })
        })
    }

    /// Card `card` of `plan` resident: the weights the plan puts on the card
    /// (each routed stack's card experts, in the plan's order), the stores at
    /// the plan's `ctx_max` and the step's buffers, and the host tier over the
    /// layers whose experts the plan leaves on the host, in relu² through the
    /// common host run ([`crate::host`]). The prompt runs as passes of up to
    /// [`PLACED_ROWS`] rows through the host tier's batch port
    /// ([`GpuModel::prefill`](GpuModel::<Body>::prefill)), the pass's buffers
    /// cut from the card's room as a whole-card load's are.
    /// Refused by name: a plan of other than one card holding every layer,
    /// with an expert tier, a layer whose card count the file's does not
    /// admit, and scratch past the plan's term for it.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan's ctx_max {}", plan.ctx_max)))?;
        GpuModel::load_placed(
            file,
            plan,
            card,
            host,
            |stream, file, _, w| Body::derive(stream, file, w),
            |gpu, file, w, set| {
                let file = Arc::new(file);
                let open = PlacedOpen {
                    plan,
                    file: Arc::clone(&file),
                    host,
                    set,
                };
                let body = Body::load(gpu, &file, w, ctx, Open::Placed(Box::new(open)))?;
                refuse_scratch_past(
                    WHAT,
                    plan,
                    card,
                    body.scratch_bytes(),
                    "the step's buffers, the routers' results, the quantized forms, the host \
                     boundary, the slot map's copy",
                )?;
                Ok(body)
            },
        )
    }
}

impl Body {
    /// The weights a load derives before any name is resolved: the zero conv
    /// bias of a file whose Mamba-2 layers hold none.
    fn derive(stream: &CudaStream, file: &Split, w: &mut Weights) -> Result<(), GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        derive_zero_conv_bias(
            w,
            file,
            hp.kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| **k == Kind::Mamba)
                .map(|(l, _)| l),
            hp.ssm.conv_dim,
            stream,
        )
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        ctx: usize,
        open: Open<'_>,
    ) -> Result<Body, GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let map = match &open {
            Open::Whole { .. } => None,
            Open::Placed(o) => Some(plan_map(o.plan, &hp)?),
        };
        // A companion `.scale` is read for NVFP4 projections only
        // (weights::tensor_scales); a scaled Q8_0 ssm_in/ssm_out would enter
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
            let n_card = |n_expert: usize| match &map {
                Some((m, _)) => m.on_card(l),
                None => Ok(n_expert),
            };
            let norm = names.vector(l, names::ATTN_NORM, hidden)?;
            layers.push(match hp.kinds[l] {
                Kind::Mamba => Layer::Mamba(names.mamba((l, norm), &ssm, hidden, stream)?),
                Kind::Attention => Layer::Attn(names.attn(
                    (l, norm),
                    (hp.n_head, hp.kv[l]),
                    (hidden, ctx),
                    stream,
                )?),
                Kind::Moe => {
                    let n_expert = hp.experts[l].map_or(0, |x| x.n_expert);
                    Layer::Moe(MoeLayer::load(
                        &names,
                        (l, norm),
                        &hp,
                        n_card(n_expert)?,
                        stream,
                    )?)
                }
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
        // A whole-card load's pass rows come from the card's room after the
        // weights, the stores and the acts are resident: the largest pass
        // whose buffers fit what is left, before anything after them
        // allocates. A placed load's passes are at most the host union's
        // columns ([`PLACED_ROWS`]); its host tier is made for them.
        let (prompt, placed) = match open {
            Open::Whole { pass_rows } => {
                let sizes = PassSizes::of((&sh, moe, hp.n_head), &layers, false);
                let free = gpu.alloc_room()?;
                let rows = choose_pass_rows(&sizes, (ctx, pass_rows), free, PASS_ROWS_MAX)?;
                (Prompt::new(stream, &sizes, rows)?, None)
            }
            Open::Placed(o) => {
                let map = map.ok_or(GpuError::state(WHAT, "a slot map for a placed load"))?;
                let sizes = PassSizes::of((&sh, moe, hp.n_head), &layers, true);
                let free = gpu.alloc_room()?;
                let rows = choose_pass_rows(&sizes, (ctx, None), free, PLACED_ROWS)?;
                let prompt = Prompt::new(stream, &sizes, rows)?;
                let placed = Placed::new(gpu, &hp, &layers, map, (*o, rows))?;
                (prompt, Some(placed))
            }
        };
        let ctx_ = gpu.context();
        Ok(Body {
            prompt,
            placed,
            k: Kernels::load(gpu)?,
            mk: MoeKernels {
                routers: Routers::load(ctx_)?,
                iq: QuantSelKernels::load(ctx_, gpu.fault_word())?,
            },
            q38: Q38Kernels::load(ctx_)?,
            prefill: FlashGqaPrefill::load(ctx_)?,
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
        })
    }

    /// Device bytes the body holds past the weights and the stores: what the
    /// plan sets aside as scratch, the one owner of the term
    /// [`Body::resident_bytes`] adds the stores and the prompt pass's buffers
    /// to (those are cut from the card's room, not from the plan's scratch).
    fn scratch_bytes(&self) -> usize {
        let routes: usize = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Moe(m) => m.route.bytes(),
                _ => 0,
            })
            .sum();
        routes
            + self.s.bytes()
            + self.ms.bytes()
            + self.acts.bytes()
            + self.r.num_bytes()
            + self.k.kgemv.device_bytes()
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
    }

    /// The placed side of a placed load (the host tier and the card's side of
    /// it), `None` on a whole-card load: a gate's instrument.
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
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
    /// the head's input whatever the layer count's parity. A placed load walks
    /// its layers through the host tier's step port instead
    /// ([`StepWalk`]).
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
        if let Some(Placed { hybrid, side }) = self.placed.as_mut() {
            let mut leg = StepLeg::new(gpu.stream(), hybrid);
            StepWalk {
                cx,
                mk: &self.mk,
                moe: self.moe,
                layers: &mut self.layers,
                t: &mut self.s,
                ms: &mut self.ms,
                acts: &mut self.acts,
                side,
                hin,
                r: &mut self.r,
            }
            .walk(&mut leg)?;
            return head.enqueue(gpu, w);
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
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::NemotronHMoe
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The step record in one copy ([`Words::refresh`]).
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        self.words.refresh(
            "nemotron_h_moe::Body::refresh",
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
    /// every row below it is written by its own step first. The router's
    /// results are written whole by every step before any read. A placed
    /// load's expert up rows are cleared too: the launch that fills them skips
    /// the slots the host serves and the quantize after it reads every slot,
    /// so the non-finite rows a faulted call left would fault the next valid
    /// one.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        for layer in &mut self.layers {
            if let Layer::Mamba(m) = layer {
                m.store.clear(gpu.stream(), &self.zeros)?;
            }
        }
        if let Some(p) = self.placed.as_mut() {
            self.ms.up.zero_async(gpu.stream())?;
            self.prompt.clear_expert_up(gpu.stream())?;
            p.reset(gpu.stream())?;
        }
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn resident_bytes(&self) -> usize {
        let stores: usize = self
            .layers
            .iter()
            .map(|l| match l {
                Layer::Mamba(m) => m.store.bytes(),
                Layer::Attn(a) => a.bytes(),
                Layer::Moe(_) => 0,
            })
            .sum();
        stores + self.prompt.bytes() + self.scratch_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}
