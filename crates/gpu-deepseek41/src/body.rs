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
//! - the hyper-connection streams and the folded input as ping-pong pairs, so
//!   each sub-layer reads one and writes the other and no copy node sits
//!   between sub-layers;
//! - the device copy of the step image ([`crate::params`]);
//! - the slot map: per layer and expert id, the slot of the card's routed
//!   stack that holds the expert, or [`HOST`] — filled from the plan's
//!   segments, so moving an expert between the card and the host changes
//!   this map and not the code; the chain reads its card copy and the host
//!   tier its host copy ([`SlotMap`]);
//! - per layer that runs the indexer, its list: the compressed rows its
//!   stream's layers attend this step, [`AttnChain::list_len`] entries;
//! - the host tier ([`Hybrid`]): the join buffers and the host experts;
//! - the three chain pieces ([`crate::chain`]) with their scratch.
//!
//! One step ([`ChainBody::enqueue_chain`]), in the order the dump's nodes
//! fix: the attention piece's gather of the step words; the embedding
//! broadcast into the streams and layer 0's input; each engram site's
//! token-only work; then per layer the engram step where the layer carries a
//! site, the attention sub-layer and the MoE sub-layer, whose HC_POST folds
//! the next sub-layer's input except before an engram layer and after the
//! last layer; last the streams' collapse into the head. The host half of a
//! step ([`ChainBody::decode_input`]) plans it, reads its embedding and
//! engram rows from the file and builds its image.
//!
//! [`ChainBody::seed_depth`] refuses: a synthetic depth would have to fill
//! the rings, the compressed rows, the index keys and the states
//! consistently with each other.

use std::ops::Range;

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{Boundary, BoundaryShape, HOST, Hybrid, SlotMap, levers};
use bloomery_gpu::model::{ChainBody, StepProbe};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::names;
use model::arch::deepseek41::place::PlanInputs;
use model::arch::deepseek41::plan::{Planner, StepPlan};
use model::placement::{Device, Machine, Plan, Role};

use crate::chain::attn::{AttnChain, AttnIo, AttnTaps, Compressed, Selection, SourceIo};
use crate::chain::ffn::{CardStacks, Ds41Host, FfnIo, FfnPiece, FfnTaps};
use crate::chain::glue::{EngramStep, Glue, StepRows};
use crate::hc::HC_STREAMS;
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
    let inputs = PlanInputs::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
    let machine = machine(inputs.model.layers);
    if machine.cards.len() != 1 {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the placement puts the layers on {} cards; the chain runs on one",
                machine.cards.len()
            ),
        });
    }
    let plan = inputs
        .plan(&machine, ctx_max as u64)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    GpuModel::load_placed(file, &plan, 0, &inputs.hp)
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

/// One layer's cache and compressor buffers, for a caller that writes a
/// state into them itself ([`Body::state_mut`]).
pub struct StateMut<'a> {
    pub ring: &'a mut DeviceTensor<u16>,
    pub rows: Option<&'a mut DeviceTensor<u16>>,
    pub keys: Option<&'a mut DeviceTensor<u16>>,
    pub values: Option<&'a mut DeviceTensor<f32>>,
    pub scores: Option<&'a mut DeviceTensor<f32>>,
}

/// The compressed rows a layer's attention reads.
#[derive(Clone, Copy, Debug)]
enum RowsOf {
    /// A window-only layer.
    None,
    /// The rows of the layer at this index of the body's layers, which runs
    /// earlier in the step.
    Reads(usize),
    /// Its own compressor's.
    Source,
}

/// The list a layer's attention reads its compressed rows through.
#[derive(Clone, Copy, Debug)]
enum ListOf {
    /// A window-only layer.
    None,
    /// The list at this index of the body's lists, which a layer earlier in
    /// the step wrote.
    Reads(usize),
    /// The layer runs the indexer into the list at this index, scoring the
    /// index keys of the layer at `keys` of the body's layers — `None` when
    /// it owns them.
    Writes { list: usize, keys: Option<usize> },
}

/// What one layer of the step runs besides its two sub-layers' launches,
/// resolved at load.
#[derive(Clone, Copy, Debug)]
struct LayerStep {
    rows: RowsOf,
    list: ListOf,
    /// The layer carries an engram site: the glue's engram step precedes its
    /// attention.
    engram: bool,
    /// Its MoE sub-layer folds the next sub-layer's input.
    folds: bool,
}

/// A point of the step, shown to an observer of
/// [`Body::enqueue_observed`] once the launches before it are enqueued:
/// the buffers one piece just wrote. The engine's own step observes nothing.
pub enum Seam<'a> {
    /// Layer `layer`'s attention sub-layer: the new streams, the MoE
    /// sub-layer's folded input, the piece's own buffers, and the list its
    /// attention read the compressed rows through.
    Attn {
        layer: usize,
        streams: &'a DeviceBuffer<f32>,
        fold: &'a DeviceBuffer<f32>,
        taps: AttnTaps<'a>,
        list: Option<&'a DeviceBuffer<u32>>,
    },
    /// The engram step before layer `layer`'s attention: the gated streams
    /// and the attention's folded input.
    Engram {
        layer: usize,
        streams: &'a DeviceBuffer<f32>,
        fold: &'a DeviceBuffer<f32>,
    },
    /// Layer `layer`'s MoE sub-layer: the new streams, the next sub-layer's
    /// folded input where the layer folds, and the piece's own buffers.
    Ffn {
        layer: usize,
        streams: &'a DeviceBuffer<f32>,
        fold: Option<&'a DeviceBuffer<f32>>,
        taps: FfnTaps<'a>,
    },
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
    steps: Vec<LayerStep>,
    /// The hyper-connection streams, ping and pong: `streams` × `n_embd` f32
    /// each.
    hc: [DeviceTensor<f32>; 2],
    /// The folded input a sub-layer reads, ping and pong: `n_embd` f32 each.
    folds: [DeviceBuffer<f32>; 2],
    /// Per indexer layer of the card, in layer order: its list.
    lists: Vec<DeviceBuffer<u32>>,
    image: StepImage,
    /// The image's device copy, which the captured chain reads.
    params: DeviceBuffer<u32>,
    /// The slot map's card copy: `layers.len()` rows of `n_expert` slots.
    /// The host tier holds the host copy.
    slots: DeviceTensor<u32>,
    hybrid: Hybrid<Ds41Host>,
    attn: AttnChain,
    ffn: FfnPiece,
    glue: Glue,
    /// The step's host half: its plan, its rows and the tokens before it.
    planner: Planner,
    plan: StepPlan,
    rows: StepRows,
    /// The tokens decoded so far, one per position: `ctx_max` reserved.
    history: Vec<u32>,
    /// The file, for the tensors the plan leaves on the host.
    file: Split,
    eps: f32,
}

/// `pair[read]` to read and the other to write.
fn ping<T>(pair: &mut [T; 2], read: usize) -> (&T, &mut T) {
    let [a, b] = pair;
    if read == 0 { (&*a, b) } else { (&*b, a) }
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

    /// Layer `layer`'s cache and compressor buffers, for a caller that writes
    /// a state into them itself; `None` for a layer this card does not run.
    pub fn state_mut(&mut self, layer: usize) -> Option<StateMut<'_>> {
        let i = layer.checked_sub(self.layers.start)?;
        self.kv.get_mut(i).map(|k| StateMut {
            ring: &mut k.ring,
            rows: k.rows.as_mut(),
            keys: k.keys.as_mut(),
            values: k.values.as_mut(),
            scores: k.scores.as_mut(),
        })
    }

    /// The buffers besides the layers' caches, by name, with their device
    /// bytes: the pieces' scratch among them.
    #[must_use]
    pub fn step_buffers(&self) -> [(&'static str, usize); 9] {
        [
            (
                "hyper-connection streams",
                self.hc.iter().map(|t| t.buf().num_bytes()).sum(),
            ),
            (
                "folded inputs",
                self.folds.iter().map(DeviceBuffer::num_bytes).sum(),
            ),
            ("step image", self.params.num_bytes()),
            (
                "selection lists",
                self.lists.iter().map(DeviceBuffer::num_bytes).sum(),
            ),
            ("slot map", self.slots.buf().num_bytes()),
            ("join buffers", self.hybrid.boundary().device_bytes()),
            ("attention piece", self.attn.device_bytes()),
            ("ffn piece", self.ffn.device_bytes()),
            ("glue piece", self.glue.device_bytes()),
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
        self.hybrid.boundary().slots()
    }

    /// The host tier: the boundary and what it has served.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<Ds41Host> {
        &self.hybrid
    }

    /// The rows the last [`ChainBody::decode_input`] read for its step.
    #[must_use]
    pub fn step_rows(&self) -> &StepRows {
        &self.rows
    }

    /// The step plan the last [`ChainBody::decode_input`] made.
    #[must_use]
    pub fn step_plan(&self) -> &StepPlan {
        &self.plan
    }

    /// The layers whose attention an engram step precedes, in order.
    pub fn engram_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.glue.engram_layers()
    }

    /// Kernel launches layer `layer`'s MoE sub-layer enqueues, besides its
    /// two memory-operation batches ([`FfnPiece::launches`]).
    #[must_use]
    pub fn ffn_launches(&self, layer: usize) -> Option<usize> {
        self.ffn.launches(layer)
    }

    /// Device bytes of the three pieces' own scratch: attention, MoE, glue.
    #[must_use]
    pub fn piece_bytes(&self) -> [usize; 3] {
        [
            self.attn.device_bytes(),
            self.ffn.device_bytes(),
            self.glue.device_bytes(),
        ]
    }

    /// The file the body keeps for the tensors the plan leaves on the host.
    #[must_use]
    pub fn file(&self) -> &Split {
        &self.file
    }

    /// The tokens before the next step, for a caller that has written the
    /// state they leave into the buffers itself ([`Body::state_mut`]): the
    /// next [`ChainBody::decode_input`] runs at position `history.len()`.
    pub fn set_history(&mut self, history: &[u32]) -> Result<(), GpuError> {
        if history.len() >= self.history.capacity() {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::set_history",
                detail: format!(
                    "{} tokens leave no position of the {} the caches hold",
                    history.len(),
                    self.history.capacity()
                ),
            });
        }
        self.history.clear();
        self.history.extend_from_slice(history);
        Ok(())
    }

    /// The indexer's `top_k` the step selects with.
    #[must_use]
    pub fn indexer_top_k(&self) -> usize {
        self.attn.top_k()
    }

    /// Select `top_k` compressed rows per stream instead of the file's
    /// `attention.indexer.top_k` — ik's `--override-kv` of that key: at least
    /// 1, at most the file's. Load-time only; a step captured before it must
    /// be captured again ([`AttnChain::set_top_k`]).
    pub fn set_indexer_top_k(&mut self, gpu: &Gpu, top_k: usize) -> Result<(), GpuError> {
        self.attn.set_top_k(gpu, top_k)
    }

    /// Enqueue the step as [`ChainBody::enqueue_chain`] does, showing
    /// `observe` each [`Seam`] as its launches are enqueued: a gate reads the
    /// streams there after it synchronizes. Asynchronous apart from what the
    /// observer does and the host tier's service of an eager chain.
    pub fn enqueue_observed(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        observe: &mut dyn FnMut(&Gpu, Seam<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let Body {
            layers,
            kv,
            steps,
            hc,
            folds,
            lists,
            params,
            slots,
            hybrid,
            attn,
            ffn,
            glue,
            ..
        } = self;
        hybrid.begin_chain(gpu.stream())?;
        attn.enqueue_step(gpu, params)?;
        // `s` is the streams the next sub-layer reads, `f` its folded input.
        let (mut s, mut f) = (0usize, 0usize);
        {
            let [s0, _] = &mut *hc;
            let [f0, _] = &mut *folds;
            glue.enqueue_embed(gpu, params, s0.buf_mut(), f0)?;
        }
        glue.enqueue_engram_kv(gpu, w, params)?;
        for (i, (l, step)) in layers.clone().zip(steps.iter()).enumerate() {
            if step.engram {
                let (streams, out) = ping(hc, s);
                let [f0, f1] = &mut *folds;
                glue.enqueue_engram(
                    gpu,
                    w,
                    l,
                    EngramStep {
                        streams: streams.buf(),
                        pre: ffn.taps().hc,
                        out: out.buf_mut(),
                        input: if f == 0 { f0 } else { f1 },
                    },
                )?;
                s ^= 1;
                observe(
                    gpu,
                    Seam::Engram {
                        layer: l,
                        streams: hc[s].buf(),
                        fold: &folds[f],
                    },
                )?;
            }
            {
                let (streams_in, streams_out) = ping(hc, s);
                let (fold_in, fold_out) = ping(folds, f);
                let (ring, compressed, selection) = layer_io(kv, lists, i, step)?;
                attn.enqueue_layer(
                    gpu,
                    w,
                    l,
                    AttnIo {
                        streams_in: streams_in.buf(),
                        fold_in,
                        streams_out: streams_out.buf_mut(),
                        fold_out,
                        ring,
                        compressed,
                        selection,
                    },
                )?;
            }
            s ^= 1;
            f ^= 1;
            observe(
                gpu,
                Seam::Attn {
                    layer: l,
                    streams: hc[s].buf(),
                    fold: &folds[f],
                    taps: attn.taps(),
                    list: match step.list {
                        ListOf::None => None,
                        ListOf::Reads(j) | ListOf::Writes { list: j, .. } => lists.get(j),
                    },
                },
            )?;
            {
                let (streams_in, streams_out) = ping(hc, s);
                let (fold_in, fold_out) = ping(folds, f);
                ffn.enqueue(
                    gpu,
                    w,
                    CardStacks::of(w, l)?,
                    FfnIo {
                        streams: streams_in.buf(),
                        fold_in,
                        streams_out: streams_out.buf_mut(),
                        fold_out: step.folds.then_some(fold_out),
                        slots: &*slots,
                    },
                    &mut *hybrid,
                    l,
                )?;
            }
            s ^= 1;
            if step.folds {
                f ^= 1;
            }
            observe(
                gpu,
                Seam::Ffn {
                    layer: l,
                    streams: hc[s].buf(),
                    fold: step.folds.then_some(&folds[f]),
                    taps: ffn.taps(),
                },
            )?;
        }
        glue.enqueue_head(gpu, w, hc[s].buf(), ffn.taps().hc, head)
    }
}

/// Layer `i`'s window ring, the compressed rows its attention reads and the
/// list it reads them through, out of the body's layers `kv` and `lists`.
fn layer_io<'a>(
    kv: &'a mut [LayerKv],
    lists: &'a mut [DeviceBuffer<u32>],
    i: usize,
    step: &LayerStep,
) -> Result<(&'a mut DeviceTensor<u16>, Compressed<'a>, Selection<'a>), GpuError> {
    let refuse = |detail: &'static str| GpuError::State {
        what: "deepseek41 Body::enqueue_chain",
        missing: detail,
    };
    let (before, rest) = kv.split_at_mut(i);
    let selection = match step.list {
        ListOf::None => Selection::None,
        ListOf::Reads(j) => Selection::Read(lists.get(j).ok_or(refuse("the layer's list"))?),
        ListOf::Writes { list, keys } => Selection::Run {
            keys: keys
                .map(|k| {
                    before
                        .get(k)
                        .and_then(|b| b.keys.as_ref())
                        .ok_or(refuse("the key source layer's index keys"))
                })
                .transpose()?,
            list: lists.get_mut(list).ok_or(refuse("the layer's list"))?,
        },
    };
    let (ring, compressed) = match step.rows {
        RowsOf::None => (&mut rest[0].ring, Compressed::None),
        RowsOf::Reads(src) => {
            let rows = before
                .get(src)
                .and_then(|k| k.rows.as_ref())
                .ok_or(refuse("the source layer's compressed rows"))?;
            (&mut rest[0].ring, Compressed::Read(rows))
        }
        RowsOf::Source => {
            let LayerKv {
                ring,
                rows,
                keys,
                values,
                scores,
            } = &mut rest[0];
            let rows = rows
                .as_mut()
                .ok_or(refuse("the layer's own compressed rows"))?;
            let state = match (values.as_mut(), scores.as_mut()) {
                (Some(v), Some(sc)) => Some((v, sc)),
                _ => None,
            };
            (
                ring,
                Compressed::Source(SourceIo {
                    rows,
                    keys: keys.as_mut(),
                    ring: state,
                }),
            )
        }
    };
    Ok((ring, compressed, selection))
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

    /// The step at `pos` after the tokens decoded so far: its plan, its
    /// embedding row and engram rows read from the file on this thread, and
    /// its image built. A position other than the next one is refused.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        const WHAT: &str = "deepseek41 Body::decode_input";
        if self.history.len() != pos as usize || self.history.len() == self.history.capacity() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a step at position {pos} after {} tokens, in caches of {} positions",
                    self.history.len(),
                    self.history.capacity()
                ),
            });
        }
        self.planner
            .plan_into(&[token], pos, &self.history, &mut self.plan)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        self.rows.fill(&self.file, &self.plan)?;
        self.image
            .build(&self.plan, self.rows.embd(), self.rows.engram())?;
        self.history.push(token);
        Ok(StepInput { pos })
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

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue_observed(gpu, w, head, &mut |_, _| Ok(()))
    }

    /// Every ring, compressed row, index key, compressor state, stream, fold
    /// and list is zeroed in place — a captured chain keeps their addresses —
    /// and the token history is emptied.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.kv {
            layer.zero(stream)?;
        }
        for t in &mut self.hc {
            t.buf_mut().zero_async(stream)?;
        }
        for f in &mut self.folds {
            f.zero_async(stream)?;
        }
        for l in &mut self.lists {
            l.zero_async(stream)?;
        }
        self.history.clear();
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
            missing: "the node-price probe: V4.1 has none",
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

    /// The host tier's share of the chain a replay just submitted.
    fn serve_replay(&mut self) -> Result<(), GpuError> {
        self.hybrid.serve_captured()
    }

    /// The body of card `card`: its buffers sized from `hp` — the
    /// hyperparameters the plan was made from — at the plan's `ctx_max`, its
    /// slot map from the plan's routed segments on the card, the host tier
    /// over the file, and the three pieces.
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
        let folds = [
            DeviceBuffer::zeroed(stream, hp.n_embd)?,
            DeviceBuffer::zeroed(stream, hp.n_embd)?,
        ];

        let planner =
            Planner::from_file(&file, hp, plan.ctx_max).map_err(|e| GpuError::plan(WHAT, e))?;
        let row_bytes = engram_row_bytes(&file, hp)?;
        let rows = StepRows::open(&file, hp, STEP_TOKENS)?;
        if rows.row_bytes() != row_bytes {
            return Err(refuse(format!(
                "the engram tables' rows are {row_bytes} bytes, the step's host half reads {}",
                rows.row_bytes()
            )));
        }
        let dims = ImageDims::of(hp, &planner, STEP_TOKENS, row_bytes);
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

        let attn = AttnChain::new(gpu, hp, layers.clone(), image.layout(), &planner)?;
        let ffn = FfnPiece::new(gpu, hp, &map)?;
        let glue = Glue::new(gpu, hp, image.layout())?;
        let steps = layer_steps(hp, &layers, &kv, &ffn, &glue)?;
        let lists = (0..steps
            .iter()
            .filter(|s| matches!(s.list, ListOf::Writes { .. }))
            .count())
            .map(|_| DeviceBuffer::zeroed(stream, STEP_TOKENS * attn.list_len()))
            .collect::<Result<Vec<_>, _>>()?;

        let boundary = Boundary::new(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: hp.n_embd,
                n_used: hp.experts.n_used,
            },
            map,
            levers()?.overlap,
        )?;
        let first = file
            .shard_path(0)
            .ok_or_else(|| refuse("the file has no shard 0".into()))?;
        let host = Ds41Host::build(Split::open(first)?, hp, layers.clone())?;
        let hybrid = Hybrid::new(boundary, host, layers.len())?;

        Ok(Body {
            layers,
            kv,
            steps,
            hc,
            folds,
            lists,
            image,
            params,
            slots,
            hybrid,
            attn,
            ffn,
            glue,
            planner,
            plan: StepPlan::default(),
            rows,
            history: Vec::with_capacity(ctx_max),
            file,
            eps: hp.rms_eps,
        })
    }
}

/// What each of `layers` runs besides its two sub-layers, checked against the
/// pieces: a reading layer's source runs before it on this card and holds
/// compressed rows; a layer of a stream reads the list of its top-k source,
/// an indexer layer that runs before it (or is it) on this card, and an
/// indexer layer scores the keys of a layer that runs before it (or is it)
/// and holds them; an engram step precedes exactly the layers the glue has
/// sites at, none of them the first, whose attention input the embedding
/// broadcast writes; and the MoE sub-layer folds the next input everywhere
/// except before an engram layer and after the last layer, where the glue
/// folds instead.
fn layer_steps(
    hp: &Hparams,
    layers: &Range<usize>,
    kv: &[LayerKv],
    ffn: &FfnPiece,
    glue: &Glue,
) -> Result<Vec<LayerStep>, GpuError> {
    let refuse = |l: usize, detail: String| GpuError::Shape {
        what: "deepseek41 Body::load_placed",
        detail: format!("layer {l}: {detail}"),
    };
    let sites: Vec<usize> = glue.engram_layers().collect();
    let mut steps = Vec::with_capacity(layers.len());
    // The list index of each indexer layer on the card, in layer order.
    let mut writers: Vec<usize> = Vec::new();
    for (i, l) in layers.clone().enumerate() {
        let kind = &hp.layers[l];
        let list = match kind.stream {
            None => ListOf::None,
            Some(st) if kind.indexer => {
                let keys = if st.index_key_source == l {
                    kv[i].keys.is_some().then_some(None)
                } else {
                    st.index_key_source
                        .checked_sub(layers.start)
                        .filter(|&k| k < i && kv[k].keys.is_some())
                        .map(Some)
                }
                .ok_or_else(|| {
                    refuse(
                        l,
                        format!(
                            "it scores the index keys of layer {}, which does not hold them on \
                             this card by then",
                            st.index_key_source
                        ),
                    )
                })?;
                writers.push(l);
                ListOf::Writes {
                    list: writers.len() - 1,
                    keys,
                }
            }
            Some(st) => {
                let j = writers
                    .iter()
                    .position(|&w| w == st.topk_source)
                    .ok_or_else(|| {
                        refuse(
                            l,
                            format!(
                                "its list comes from layer {}, which runs no indexer before it \
                                 on this card",
                                st.topk_source
                            ),
                        )
                    })?;
                ListOf::Reads(j)
            }
        };
        let rows = match (kind.stream, kind.compressor) {
            (None, _) => RowsOf::None,
            (Some(st), Some(_)) if st.kv_source == l => RowsOf::Source,
            (Some(st), None) => {
                let src = st
                    .kv_source
                    .checked_sub(layers.start)
                    .filter(|&s| s < i && kv[s].rows.is_some())
                    .ok_or_else(|| {
                        refuse(
                            l,
                            format!(
                                "its rows come from layer {}, which does not run before it on \
                                 this card with a compressor",
                                st.kv_source
                            ),
                        )
                    })?;
                RowsOf::Reads(src)
            }
            (Some(st), Some(_)) => {
                return Err(refuse(
                    l,
                    format!("owns a compressor and reads layer {}'s rows", st.kv_source),
                ));
            }
        };
        let engram = sites.contains(&l);
        if engram && i == 0 {
            return Err(refuse(
                l,
                "an engram site on the first layer: its input is the embedding row".to_string(),
            ));
        }
        let folds = ffn
            .folds(l)
            .ok_or_else(|| refuse(l, "the ffn piece does not run it".to_string()))?;
        let into_glue = l + 1 == layers.end || sites.contains(&(l + 1));
        if folds == into_glue {
            return Err(refuse(
                l,
                format!(
                    "the ffn piece folds: {folds}; the next sub-layer is {}",
                    if into_glue {
                        "the glue's"
                    } else {
                        "an attention with no engram step"
                    }
                ),
            ));
        }
        steps.push(LayerStep {
            rows,
            list,
            engram,
            folds,
        });
    }
    if hp.hc.streams != HC_STREAMS {
        return Err(GpuError::Shape {
            what: "deepseek41 Body::load_placed",
            detail: format!("{} streams; the chain runs {HC_STREAMS}", hp.hc.streams),
        });
    }
    Ok(steps)
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
/// `engram.key_length` values, all in one format. The one owner of the
/// figure: the step image is laid out by it, and the step's host half must
/// read rows of it.
pub fn engram_row_bytes(file: &Split, hp: &Hparams) -> Result<usize, GpuError> {
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
