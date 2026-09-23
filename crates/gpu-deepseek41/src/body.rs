//! The DeepSeek-V4.1 chain body ([`Body`], the [`ChainBody`] of
//! [`Deepseek41Model`]): every buffer one decode step reads or writes,
//! allocated once when the body loads by its placement plan and none of them
//! per step, each sized from the file's hyperparameters, never from layer
//! numbers:
//!
//! - per layer, the raw window ring: `min(ctx_max, window)` latent rows in
//!   f16, the row of cell `c` in slot `c % window`;
//! - per layer that owns a compressor, its compressed rows (`⌈ctx_max /
//!   ratio⌉` latent rows in f16) and, above ratio 1, its state: the `ratio`
//!   latest latent projections in f32, values and scores (a ratio-1 group is
//!   its one row: nothing pools); per layer that owns index keys, as many
//!   rows of index key in f16;
//! - the hyper-connection streams as a ping-pong pair, so each sub-layer reads
//!   one and writes the other and no copy node sits between layers;
//! - the device copy of the step image ([`crate::params`]);
//! - the slot map: per layer and expert id, the slot of the card's routed
//!   stack that holds the expert, or [`HOST`] — filled from the plan's
//!   segments, so moving an expert between the card and the host changes
//!   this map and not the code; the chain reads its card copy and the host
//!   tier its host copy ([`SlotMap`]);
//! - the join buffers of the host tier ([`Boundary`]).
//!
//! The chain is not assembled: [`ChainBody::enqueue_chain`] refuses, and so
//! does [`ChainBody::seed_depth`] — a synthetic depth would have to fill the
//! rings, the compressed rows, the index keys and the states consistently
//! with each other. The step's host input needs the engram row lookup, which
//! the chain's assembly brings, so [`ChainBody::decode_input`] refuses too;
//! [`Body::image_mut`] builds an image from a plan and rows a caller holds.

use std::ops::Range;

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{Boundary, BoundaryShape, HOST, SlotMap};
use bloomery_gpu::model::{ChainBody, StepProbe};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::kv::KvLayout;
use model::arch::deepseek41::plan::Planner;
use model::arch::deepseek41::{names, roles};
use model::placement::{self, Device, Machine, Plan, Role};

use crate::params::{ImageDims, ImageLayout, StepImage, rope_specs};

/// The V4.1 engine: the shared skeleton over this body.
pub type Deepseek41Model = GpuModel<Body>;

/// Tokens one decode step runs.
pub const STEP_TOKENS: usize = 1;

/// The whole V4.1 model on this machine's placement `machine`: the file's
/// headers read once ([`Hparams`]), its tensors classified and planned at
/// `ctx_max` positions, and the plan's card loaded with its layers and the
/// head ([`GpuModel::load_placed`]). A plan that breaks its invariants, or
/// that spreads the layers over more than one card, is refused before
/// anything is uploaded.
pub fn open(
    file: Split,
    machine: fn(usize) -> Machine,
    ctx_max: usize,
) -> Result<Deepseek41Model, GpuError> {
    const WHAT: &str = "deepseek41 body::open";
    let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
    let hp = Hparams::read(&file).map_err(|e| refuse(e.to_string()))?;
    let model = roles::classify(&file, &hp).map_err(|e| refuse(e.to_string()))?;
    let kv = KvLayout::from_file(&file, &hp).map_err(|e| refuse(e.to_string()))?;
    let machine = machine(model.layers);
    if machine.cards.len() != 1 {
        return Err(refuse(format!(
            "the placement puts the layers on {} cards; the chain runs on one",
            machine.cards.len()
        )));
    }
    let plan = placement::plan(&model, &machine, ctx_max as u64, &kv)
        .map_err(|e| refuse(e.to_string()))?;
    let broken = plan.violations();
    if !broken.is_empty() {
        let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
        return Err(refuse(format!(
            "the plan breaks its invariants: {}",
            list.join("; ")
        )));
    }
    GpuModel::load_placed(file, &plan, 0, &hp)
}

/// One layer's cache and compressor state.
struct LayerKv {
    /// The raw window ring.
    ring: DeviceTensor<u16>,
    /// The compressed rows of the stream the layer's compressor writes.
    rows: Option<DeviceTensor<u16>>,
    /// The index keys the layer owns.
    keys: Option<DeviceTensor<u16>>,
    /// The compressor's state above ratio 1: its stream's latest
    /// projections, values and scores.
    values: Option<DeviceTensor<f32>>,
    scores: Option<DeviceTensor<f32>>,
}

/// Device bytes of a buffer the layer may not hold.
fn bytes_of<T: DeviceCopy>(t: Option<&DeviceTensor<T>>) -> usize {
    t.map_or(0, |t| t.buf().num_bytes())
}

impl LayerKv {
    /// Every buffer of the layer by name, with its device bytes (0 for one
    /// the layer does not hold), measured from the buffers.
    fn buffers(&self) -> [(&'static str, usize); 5] {
        [
            ("window ring", self.ring.buf().num_bytes()),
            ("compressed rows", bytes_of(self.rows.as_ref())),
            ("index keys", bytes_of(self.keys.as_ref())),
            ("state values", bytes_of(self.values.as_ref())),
            ("state scores", bytes_of(self.scores.as_ref())),
        ]
    }

    fn zero(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.ring.buf_mut().zero_async(stream)?;
        for t in [&mut self.rows, &mut self.keys].into_iter().flatten() {
            t.buf_mut().zero_async(stream)?;
        }
        for t in [&mut self.values, &mut self.scores].into_iter().flatten() {
            t.buf_mut().zero_async(stream)?;
        }
        Ok(())
    }
}

/// The per-step input the body refreshes the card with: the step whose
/// image was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInput {
    /// The step's first position.
    pub pos: u32,
}

/// The V4.1 chain body: see the module comment.
pub struct Body {
    /// The layers this card runs.
    layers: Range<usize>,
    /// Per layer of `layers`, in order.
    kv: Vec<LayerKv>,
    /// The hyper-connection streams, ping and pong: `streams` × `n_embd` f32
    /// each.
    hc: [DeviceTensor<f32>; 2],
    image: StepImage,
    /// The image's device copy, which the captured chain reads.
    params: DeviceBuffer<u32>,
    /// The slot map's card copy: `layers.len()` rows of `n_expert` slots.
    /// The boundary holds the host copy.
    slots: DeviceTensor<u32>,
    boundary: Boundary,
    /// The file, for the tensors the plan leaves on the host.
    file: Split,
    eps: f32,
}

impl Body {
    /// The layers this body's card runs.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Layer `layer`'s cache and compressor buffers by name, with their device
    /// bytes; `None` for a layer this card does not run.
    #[must_use]
    pub fn state_buffers(&self, layer: usize) -> Option<[(&'static str, usize); 5]> {
        let i = layer.checked_sub(self.layers.start)?;
        self.kv.get(i).map(LayerKv::buffers)
    }

    /// Layer `layer`'s cache and compressor bytes — the figure `KvLayout`
    /// plans for the layer; `None` for a layer this card does not run.
    #[must_use]
    pub fn state_bytes(&self, layer: usize) -> Option<usize> {
        self.state_buffers(layer)
            .map(|b| b.iter().map(|&(_, n)| n).sum())
    }

    /// The buffers besides the layers' caches, by name, with their device
    /// bytes.
    #[must_use]
    pub fn step_buffers(&self) -> [(&'static str, usize); 4] {
        [
            (
                "hyper-connection streams",
                self.hc.iter().map(|t| t.buf().num_bytes()).sum(),
            ),
            ("step image", self.params.num_bytes()),
            ("slot map", self.slots.buf().num_bytes()),
            ("join buffers", self.boundary.device_bytes()),
        ]
    }

    /// The step image's host side, which builds the image from a plan.
    #[must_use]
    pub fn image(&self) -> &StepImage {
        &self.image
    }

    /// The step image's host side, for a caller that builds a step's image
    /// from a plan and rows it holds; [`ChainBody::refresh`] uploads it.
    pub fn image_mut(&mut self) -> &mut StepImage {
        &mut self.image
    }

    /// The image's device copy, as the captured chain reads it.
    #[must_use]
    pub fn params(&self) -> &DeviceBuffer<u32> {
        &self.params
    }

    /// The slot map's card copy: `layers().len()` rows of the file's
    /// `n_expert` slots, [`HOST`] where the plan leaves the expert off the
    /// card.
    #[must_use]
    pub fn slots(&self) -> &DeviceTensor<u32> {
        &self.slots
    }

    /// The slot map's host copy, which the host tier serves by.
    #[must_use]
    pub fn slot_map(&self) -> &SlotMap {
        self.boundary.slots()
    }

    /// The file the body keeps for the tensors the plan leaves on the host.
    #[must_use]
    pub fn file(&self) -> &Split {
        &self.file
    }
}

impl ChainBody for Body {
    type Input = StepInput;
    type Meta = Hparams;

    fn arch() -> Arch {
        Arch::Deepseek41
    }

    /// V4.1 derives no weights at load: every tensor its chain reads is a
    /// file tensor in its card format.
    fn derive(
        _stream: &CudaStream,
        _file: &Split,
        _layers: Range<usize>,
        _w: &mut Weights,
    ) -> Result<(), GpuError> {
        Ok(())
    }

    /// V4.1 loads by its placement plan ([`ChainBody::load_placed`]); a load
    /// of a layer range with every weight on the card is refused.
    fn load(
        _gpu: &Gpu,
        _file: &Split,
        _w: &Weights,
        _layers: Range<usize>,
        _ctx_max: usize,
    ) -> Result<Body, GpuError> {
        Err(GpuError::State {
            what: "deepseek41 Body::load",
            missing: "a placement plan: V4.1 loads by one (GpuModel::load_placed)",
        })
    }

    fn decode_input(&mut self, _token: u32, _pos: u32) -> Result<StepInput, GpuError> {
        Err(GpuError::State {
            what: "deepseek41 Body::decode_input",
            missing: "the engram row lookup: the chain is not assembled",
        })
    }

    /// One host-to-device copy of the whole image, which must hold the step
    /// `input` names.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        if self.image.pos() != Some(input.pos) {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::refresh",
                detail: format!(
                    "the image holds the step at {:?}, the input names {}",
                    self.image.pos(),
                    input.pos
                ),
            });
        }
        self.params.copy_from_host(stream, self.image.words())?;
        Ok(())
    }

    fn enqueue_chain(
        &mut self,
        _gpu: &Gpu,
        _w: &Weights,
        _head: &mut Head,
    ) -> Result<(), GpuError> {
        Err(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the chain is not assembled",
        })
    }

    /// Every ring, compressed row, index key, compressor state and stream is
    /// zeroed in place: a captured chain keeps their addresses.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.kv {
            layer.zero(stream)?;
        }
        for t in &mut self.hc {
            t.buf_mut().zero_async(stream)?;
        }
        Ok(())
    }

    fn seed_depth(&mut self, _gpu: &Gpu, _rows: usize) -> Result<(), GpuError> {
        Err(GpuError::State {
            what: "deepseek41 Body::seed_depth",
            missing: "a synthetic depth: the rings, compressed rows, index keys and states \
                      would have to agree with each other",
        })
    }

    fn set_probe(&mut self, _probe: StepProbe) -> Result<(), GpuError> {
        Err(GpuError::State {
            what: "deepseek41 Body::set_probe",
            missing: "the node-price probe: the chain is not assembled",
        })
    }

    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.eps
    }

    fn resident_bytes(&self) -> usize {
        let caches: usize = self
            .kv
            .iter()
            .flat_map(|l| l.buffers())
            .map(|(_, n)| n)
            .sum();
        caches + self.step_buffers().iter().map(|&(_, n)| n).sum::<usize>()
    }

    /// The body of card `card`: its buffers sized from `hp` — the
    /// hyperparameters the plan was made from — at the plan's `ctx_max`, and
    /// its slot map from the plan's routed segments on the card.
    fn load_placed(
        gpu: &Gpu,
        file: Split,
        _w: &Weights,
        plan: &Plan<'_>,
        card: usize,
        hp: &Hparams,
    ) -> Result<Body, GpuError> {
        const WHAT: &str = "deepseek41 Body::load_placed";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let spec = plan
            .machine
            .cards
            .get(card)
            .ok_or_else(|| refuse(format!("the plan has no card {card}")))?;
        let layers = spec.layers.clone();
        if hp.n_layer != plan.model.layers || layers.end > hp.n_layer {
            return Err(refuse(format!(
                "the card runs layers {layers:?} of a plan of {} layers; the file has {}",
                plan.model.layers, hp.n_layer
            )));
        }
        let ctx_max = usize::try_from(plan.ctx_max)
            .map_err(|_| refuse(format!("ctx_max {} passes usize", plan.ctx_max)))?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();

        let kv = layers
            .clone()
            .map(|l| layer_kv(stream, hp, l, ctx_max))
            .collect::<Result<Vec<_>, _>>()?;
        let hc = [
            DeviceTensor::zeroed(stream, hp.hc.streams, hp.n_embd)?,
            DeviceTensor::zeroed(stream, hp.hc.streams, hp.n_embd)?,
        ];

        let planner =
            Planner::from_file(&file, hp, plan.ctx_max).map_err(|e| refuse(e.to_string()))?;
        let dims = ImageDims::of(hp, &planner, STEP_TOKENS, engram_row_bytes(&file, hp)?);
        let (window, yarn) = rope_specs(hp)?;
        let image = StepImage::new(ImageLayout::new(dims)?, &window, &yarn)?;
        let params = DeviceBuffer::<u32>::zeroed(stream, image.layout().words())?;

        let n_expert = hp.experts.n_expert;
        let map = SlotMap::from_rows(
            layers.clone(),
            n_expert,
            plan_slots(plan, card, &layers, n_expert)?,
        )?;
        let slots = DeviceTensor::upload(stream, map.as_slice(), layers.len(), n_expert)?;

        // The boundary keeps the map's host copy; the wait sits before the
        // combine.
        let boundary = Boundary::new(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: hp.n_embd,
                n_used: hp.experts.n_used,
            },
            map,
            true,
        )?;

        Ok(Body {
            layers,
            kv,
            hc,
            image,
            params,
            slots,
            boundary,
            file,
            eps: hp.rms_eps,
        })
    }
}

/// Layer `l`'s buffers at `ctx_max` positions: its window ring; its
/// compressed rows when it owns a compressor, and its state when that
/// compressor's ratio is above 1; its index keys when it owns them. A
/// compressor or index keys on a layer that attends no stream have no ratio
/// to size them by, and are refused. A latent row is `head_dim` wide (the
/// key length, which the value length equals) and an index key
/// `indexer.head_dim`; `KvLayout` reads both widths off the tensors instead,
/// and the load gate holds the two readings to one byte count.
fn layer_kv(
    stream: &CudaStream,
    hp: &Hparams,
    l: usize,
    ctx_max: usize,
) -> Result<LayerKv, GpuError> {
    let kind = &hp.layers[l];
    let latent = hp.head_dim;
    let ratio = || {
        kind.stream
            .map(|s| s.ratio as usize)
            .ok_or_else(|| GpuError::Shape {
                what: "deepseek41 Body::load_placed",
                detail: format!("layer {l} owns a compressor or index keys and attends no stream"),
            })
    };
    let ring = DeviceTensor::zeroed(stream, ctx_max.min(hp.window), latent)?;
    let (rows, values, scores) = match kind.compressor {
        Some(_) => {
            let r = ratio()?;
            let state = || {
                (r > 1)
                    .then(|| DeviceTensor::zeroed(stream, r, latent))
                    .transpose()
            };
            (
                Some(DeviceTensor::zeroed(stream, ctx_max.div_ceil(r), latent)?),
                state()?,
                state()?,
            )
        }
        None => (None, None, None),
    };
    let keys = if kind.index_keys {
        Some(DeviceTensor::zeroed(
            stream,
            ctx_max.div_ceil(ratio()?),
            hp.indexer.head_dim,
        )?)
    } else {
        None
    };
    Ok(LayerKv {
        ring,
        rows,
        keys,
        values,
        scores,
    })
}

/// The slot of card `card`'s stack that holds each expert of each of
/// `layers` — `layers.len()` rows of `n_expert` — or [`HOST`] where the plan
/// leaves the expert off the card. The card's segments of a routed stack, in
/// plan order, fill its slots from 0; every routed stack of a layer must put
/// the same experts in the same slots.
fn plan_slots(
    plan: &Plan<'_>,
    card: usize,
    layers: &Range<usize>,
    n_expert: usize,
) -> Result<Vec<u32>, GpuError> {
    let refuse = |detail: String| GpuError::Shape {
        what: "deepseek41 plan_slots",
        detail,
    };
    let mut map = vec![HOST; layers.len() * n_expert];
    let mut first: Vec<Option<&str>> = vec![None; layers.len()];
    let mut stack = vec![HOST; n_expert];
    for row in &plan.rows {
        let t = plan
            .model
            .tensors
            .get(row.tensor)
            .ok_or_else(|| refuse(format!("a plan row names tensor {}", row.tensor)))?;
        let Some(l) = t.layer.filter(|l| layers.contains(l)) else {
            continue;
        };
        if t.role != Role::RoutedExperts {
            continue;
        }
        stack.fill(HOST);
        let mut slot = 0u32;
        for seg in row
            .segments
            .iter()
            .filter(|s| s.device == Device::Card(card))
        {
            let experts = seg.experts.clone().ok_or_else(|| {
                refuse(format!(
                    "{}: a card segment without an expert range",
                    t.name
                ))
            })?;
            for e in experts {
                let entry = usize::try_from(e)
                    .ok()
                    .and_then(|e| stack.get_mut(e))
                    .ok_or_else(|| refuse(format!("{}: expert {e} of {n_expert}", t.name)))?;
                *entry = slot;
                slot += 1;
            }
        }
        let i = l - layers.start;
        let dst = &mut map[i * n_expert..(i + 1) * n_expert];
        let seen = first[i];
        match seen {
            None => {
                dst.copy_from_slice(&stack);
                first[i] = Some(&t.name);
            }
            Some(other) if *dst != *stack => {
                return Err(refuse(format!(
                    "{} puts other experts on the card than {other}",
                    t.name
                )));
            }
            Some(_) => {}
        }
    }
    Ok(map)
}

/// Bytes of one engram table row: every site's table holds rows of
/// `engram.key_length` values, all in one format.
fn engram_row_bytes(file: &Split, hp: &Hparams) -> Result<usize, GpuError> {
    let refuse = |detail: String| GpuError::Shape {
        what: "deepseek41 engram_row_bytes",
        detail,
    };
    let mut bytes = None;
    for &l in &hp.engram.layer_ids {
        let name = names::engram_embd(l);
        let (_, t) = file
            .find(&name)
            .ok_or_else(|| refuse(format!("{name} is not in the file")))?;
        let rows: u64 = t.dims.iter().skip(1).product();
        if t.dims.first().copied() != Some(hp.engram.key_length as u64)
            || rows == 0
            || !t.nbytes.is_multiple_of(rows)
        {
            return Err(refuse(format!(
                "{name} has dims {:?} and {} bytes: not rows of {} values",
                t.dims, t.nbytes, hp.engram.key_length
            )));
        }
        let row = t.nbytes / rows;
        match bytes {
            Some(b) if b != row => {
                return Err(refuse(format!(
                    "{name} has rows of {row} bytes, another site's are {b}"
                )));
            }
            _ => bytes = Some(row),
        }
    }
    let row = bytes.ok_or_else(|| refuse("the model has no engram site".to_string()))?;
    usize::try_from(row).map_err(|_| refuse(format!("rows of {row} bytes")))
}
