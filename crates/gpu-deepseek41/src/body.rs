//! The DeepSeek-V4.1 chain body ([`Body`], the [`ChainBody`] of
//! [`Deepseek41Model`]): every buffer one decode step reads or writes,
//! allocated once when the body loads by its placement plan and none of them
//! per step, each sized from the file's hyperparameters, never from layer
//! numbers:
//!
//! - per layer, the raw window ring: `min(ctx_max, window)` latent rows in
//!   f16, the row of cell `c` in slot `c % window`;
//! - the rings' shadows, in page-locked host memory the card writes at its
//!   host address ([`Shadows`]): per layer `ctx_max` rows of the same f16
//!   bits, the row of cell `c` in row `c`, written by the same append and
//!   read only when a cut restores the ring ([`Body::rollback`]);
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
//! broadcast into the streams and layer 0's input; then per layer the engram
//! step where the layer carries a site, the attention sub-layer and the MoE
//! sub-layer, whose HC_POST folds the next sub-layer's input except before an
//! engram layer and after the last layer; last the streams' collapse into the
//! head. Each engram site's token-only work runs in the host-leg shadow of the
//! MoE sub-layer before the site's layer: it reads the step image alone, and
//! the site's engram step is its first reader. The host half of a
//! step ([`ChainBody::decode_input`]) plans it, reads its embedding and
//! engram rows from the file (the engram rows on a helper thread,
//! [`StepRows`]) and builds its image.
//!
//! The pair pass ([`Body::enqueue_pair`]) runs two tokens as rows one layer
//! apart on the same launches, so the host tier serves one row's layer while
//! the card runs the other's: each row has its own streams, folds, image
//! copy and lists, and each piece its own row of the buffers a layer leaves
//! for later; the caches are shared, written by row 0 before row 1 reads
//! them. [`Body::rollback`] cuts the positions back to any point
//! [`Body::keep_point`] grants.
//!
//! A draft's feature tap ([`attach_features`]) adds, per row, one launch
//! after the MoE sub-layer of each layer before a tapped layer: the mean of
//! the four streams that layer leaves, into the row's feature buffer, which
//! [`Body::read_features`] brings to the host. Without it the step is the
//! same launches as before, and none of its buffers exist.
//!
//! [`ChainBody::seed_depth`] refuses: a synthetic depth would have to fill
//! the rings, the compressed rows, the index keys and the states
//! consistently with each other.

use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{Boundary, BoundaryShape, Chain, HOST, Hybrid, SlotMap, levers};
use bloomery_gpu::model::{ChainBody, StepProbe};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel, PartedBuffer, window};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, IntoResult, PinnedHostBuffer, sys};
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::names;
use model::arch::deepseek41::place::PlanInputs;
use model::arch::deepseek41::plan::{Planner, StepPlan};
use model::placement::{Device, Machine, Plan, Role};

use crate::chain::attn::{
    AttnChain, AttnIo, AttnTaps, Compressed, Selection, SourceIo, join_projections,
};
use crate::chain::ffn::{CardStacks, Ds41Host, FfnIo, FfnPiece, FfnTaps, ShadowWork};
use crate::chain::glue::{EngramKv, EngramStep, Glue, StepRows};
use crate::hc::{HC_STREAMS, HcKernels};
use crate::params::{ImageDims, ImageLayout, StepImage, rope_specs};

/// The V4.1 engine: the shared skeleton over this body.
pub type Deepseek41Model = GpuModel<Body>;

/// Tokens one decode step runs.
pub const STEP_TOKENS: usize = 1;

/// Rows of the pair pass ([`Body::enqueue_pair`]): tokens `t` and `t + 1`
/// one layer apart. The one-token step runs on row 0.
pub const PAIR_ROWS: usize = 2;

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

/// Build the feature tap a draft reads ([`Body::read_features`]): for every
/// layer of `layers` — a draft's `target_layers`, strictly increasing, each
/// at least 1 and at most the layer count — the mean of the four streams
/// entering it, which the layer before it leaves, per row of the step and of
/// the pair pass. Before any step and any capture: a graph captured without
/// the tap would never write the features its readback returns, so a model
/// that holds one is refused. Load-time only.
pub fn attach_features(m: &mut Deepseek41Model, layers: &[usize]) -> Result<(), GpuError> {
    const WHAT: &str = "deepseek41 attach_features";
    if m.step_graph_nodes().is_ok() || m.pair_graph_nodes().is_ok() {
        return Err(GpuError::State {
            what: WHAT,
            missing: "a model with no captured graph: attach the tap before the first capture",
        });
    }
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.attach_features(gpu, layers)
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

/// The ring shadows of the card's layers, in one page-locked host allocation
/// (`cuMemAllocHost`): layer `i` of the card holds `rows` rows of the ring's
/// width from row `i · rows` on, the row of position `p` in its row `p`. The
/// card reaches page-locked host memory at its host address under unified
/// addressing, which the load checks and without which it refuses — the
/// shadows never move to card memory instead — so the append writes a
/// layer's rows through a window over them ([`Shadows::layer_mut`]) and no
/// card memory holds them. A cut copies the rows it needs back into the ring,
/// host to device ([`Shadows::restore`]).
struct Shadows {
    /// Per layer, a window over its rows of `host`: they free nothing, and
    /// the drop gives them back.
    windows: Vec<ManuallyDrop<DeviceTensor<u16>>>,
    host: PinnedHostBuffer<u16>,
    rows: usize,
    width: usize,
    /// The card's `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`, as the load read it.
    unified_addressing: i32,
}

/// Where the ring shadows live ([`Body::shadow_host`]): one page-locked host
/// allocation the card reaches at its host address.
#[derive(Clone, Copy, Debug)]
pub struct ShadowHost {
    /// Every layer's shadow, in bytes.
    pub bytes: usize,
    /// The card's `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING` at load: 1, or the
    /// load was refused.
    pub unified_addressing: i32,
}

impl Shadows {
    /// `layers` zeroed shadows of `rows` rows of `width` for `gpu`'s card, or
    /// the refusal: a card without unified addressing, or an allocation the
    /// card reaches at an address other than its host address.
    fn new(gpu: &Gpu, layers: usize, rows: usize, width: usize) -> Result<Shadows, GpuError> {
        const WHAT: &str = "deepseek41 Body::load_placed";
        let ctx = gpu.context();
        let mut unified_addressing = 0;
        // SAFETY: the call writes the live local `unified_addressing`; the
        // device is the context's own.
        let rc = unsafe {
            sys::cuDeviceGetAttribute(
                &mut unified_addressing,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                ctx.cu_device(),
            )
        };
        rc.result().map_err(|source| GpuError::Driver {
            op: Some("cuDeviceGetAttribute(UNIFIED_ADDRESSING)"),
            source,
        })?;
        if unified_addressing != 1 {
            return Err(GpuError::State {
                what: WHAT,
                missing: "unified addressing on the card: the ring shadows are page-locked host \
                          memory the append writes at its host address",
            });
        }
        let len = layers
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("{layers} ring shadows of {rows} rows of {width} pass usize"),
            })?;
        let host =
            PinnedHostBuffer::<u16>::zeroed(ctx, len).map_err(|source| GpuError::Driver {
                op: Some("cuMemAllocHost (the ring shadows)"),
                source,
            })?;
        let base = if host.is_empty() {
            0
        } else {
            host_address(&host)?
        };
        let windows = (0..layers)
            .map(|i| {
                let at = u64::try_from(i * rows * width * size_of::<u16>()).map_err(|_| {
                    GpuError::Shape {
                        what: WHAT,
                        detail: format!("ring shadow {i} starts past u64 bytes"),
                    }
                })?;
                // SAFETY: the window spans rows i·rows .. (i + 1)·rows of the
                // allocation's layers·rows (i < layers), which the card reaches
                // from `base` on; the allocation lives in the same value as the
                // window, and the drop gives the window back before it frees
                // the allocation.
                Ok(unsafe { DeviceTensor::window(base + at, rows, width, ctx) })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Shadows {
            windows,
            host,
            rows,
            width,
            unified_addressing,
        })
    }

    /// Layer `i`'s shadow, the tensor the append writes.
    fn layer_mut(&mut self, i: usize) -> Option<&mut DeviceTensor<u16>> {
        self.windows.get_mut(i).map(|w| &mut **w)
    }

    /// Bytes of one layer's shadow.
    fn layer_bytes(&self) -> usize {
        self.rows * self.width * size_of::<u16>()
    }

    /// Copy layer `i`'s shadow rows of `positions` into their slots of
    /// `ring`, host to device on `stream`: at most two copies (one where the
    /// slots wrap). `positions` ends at most at the shadow's rows and spans at
    /// most the ring's. The rows are the append's stores to host memory; a
    /// kernel's stores are complete when it completes, and the stream starts
    /// the copy only after every launch enqueued before it, so the copy reads
    /// what every earlier step wrote.
    fn restore(
        &self,
        i: usize,
        ring: &mut DeviceTensor<u16>,
        stream: &CudaStream,
        positions: Range<usize>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "deepseek41 Body::rollback";
        let (slots, width) = (ring.rows(), ring.cols());
        let layer = self.rows * self.width;
        let shadow = self
            .host
            .get(i * layer..(i + 1) * layer)
            .filter(|_| {
                width == self.width
                    && width > 0
                    && positions.end <= self.rows
                    && positions.len() <= slots
            })
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a restore of positions {positions:?} into shadow {i}: shadows of {} rows of \
                     {}, {} of them; a ring of {slots} rows of {width}",
                    self.rows,
                    self.width,
                    self.windows.len()
                ),
            })?;
        let bytes = |rows: usize| -> Result<u64, GpuError> {
            u64::try_from(rows * width * size_of::<u16>()).map_err(|_| GpuError::Shape {
                what: WHAT,
                detail: format!("{rows} rows pass u64 bytes"),
            })
        };
        let mut q = positions.start;
        while q < positions.end {
            let slot = q % slots;
            let n = (positions.end - q).min(slots - slot);
            let at = ring.buf().cu_deviceptr() + bytes(slot)?;
            // SAFETY: slots slot .. slot + n lie in the ring (n <= slots − slot),
            // the layer's own allocation, borrowed mutably for the copy; the
            // window is given back below.
            let mut dst = unsafe { window::<u16>(at, n * width, ring.buf().context()) };
            let src = &shadow[q * width..(q + n) * width];
            // SAFETY: `src` is page-locked memory `self` owns, which the host
            // never writes; the refresh that enqueues the copy synchronizes the
            // stream before it returns (the image upload), and the drop waits
            // for the context before it frees the allocation, so the copy
            // completes while `src` lives.
            let copied = unsafe { dst.copy_from_host_async_unchecked(stream, src) };
            drop(ManuallyDrop::into_inner(dst).into_raw_parts());
            copied?;
            q += n;
        }
        Ok(())
    }
}

impl Drop for Shadows {
    fn drop(&mut self) {
        // The appends write the allocation and a restore reads it, both in
        // stream order: the context finishes that work before `host` frees
        // it. A failure on the drop path is unreportable; the context records
        // it.
        let ctx = self.host.context();
        ctx.record_err(ctx.synchronize());
        for w in self.windows.drain(..) {
            DeviceTensor::release(w);
        }
    }
}

/// The address the card reaches the page-locked allocation `host` at: its
/// host address, which unified addressing makes it; any other is refused.
fn host_address(host: &PinnedHostBuffer<u16>) -> Result<sys::CUdeviceptr, GpuError> {
    let at = host.as_ptr();
    let mut dev: sys::CUdeviceptr = 0;
    // SAFETY: `at` is a live page-locked allocation of the context its
    // allocation bound to this thread; the call writes the live local `dev`,
    // and the flags must be 0.
    let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, at.cast_mut().cast(), 0) };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemHostGetDevicePointer_v2 (the ring shadows)"),
        source,
    })?;
    if u64::try_from(at.addr()).ok() != Some(dev) {
        return Err(GpuError::Shape {
            what: "deepseek41 Body::load_placed",
            detail: format!(
                "the card reaches the ring shadows' page-locked allocation at {dev:#x}, not at its \
                 host address {:#x}",
                at.addr()
            ),
        });
    }
    Ok(dev)
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
    /// The engram site whose token-only work the layer's MoE sub-layer runs
    /// in its host-leg shadow: the site at the next layer.
    shadow_site: Option<usize>,
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

/// One row's buffers besides the layers' caches and the pieces' own: what
/// a token's step writes and reads again later in the step.
struct Lane {
    /// The hyper-connection streams, ping and pong: `streams` × `n_embd` f32
    /// each.
    hc: [DeviceTensor<f32>; 2],
    /// The folded input a sub-layer reads, ping and pong: `n_embd` f32 each.
    folds: [DeviceBuffer<f32>; 2],
    /// The row's image's device copy, which the captured chain reads.
    params: DeviceBuffer<u32>,
}

impl Lane {
    fn new(stream: &CudaStream, hp: &Hparams, words: usize) -> Result<Lane, GpuError> {
        Ok(Lane {
            hc: [
                DeviceTensor::zeroed(stream, hp.hc.streams, hp.n_embd)?,
                DeviceTensor::zeroed(stream, hp.hc.streams, hp.n_embd)?,
            ],
            folds: [
                DeviceBuffer::zeroed(stream, hp.n_embd)?,
                DeviceBuffer::zeroed(stream, hp.n_embd)?,
            ],
            params: DeviceBuffer::<u32>::zeroed(stream, words)?,
        })
    }
}

/// Which halves of a row's ping-pong pairs the row's next sub-layer reads.
#[derive(Clone, Copy, Debug, Default)]
struct Cursor {
    /// The streams.
    s: usize,
    /// The folded input.
    f: usize,
}

/// The V4.1 chain body: see the module comment.
pub struct Body {
    /// The layers this card runs.
    layers: Range<usize>,
    /// Per layer of `layers`, in order.
    kv: Vec<LayerKv>,
    /// Every layer's ring shadow, in page-locked host memory.
    shadows: Shadows,
    steps: Vec<LayerStep>,
    /// Per row of the pair pass: its streams, folds and image copy.
    lanes: Vec<Lane>,
    /// Per row, per indexer layer of the card, in layer order: its list. A
    /// row's later layers read the list its indexer layer wrote, and the
    /// pair pass runs the other row's indexer layer between them.
    lists: Vec<Vec<DeviceBuffer<u32>>>,
    image: StepImage,
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
    /// Which position's row each ring slot and each compressor state slot
    /// holds, as the steps refreshed since the last known state left them.
    holds: Holds,
    /// The first position whose shadow row a step of the current history
    /// wrote: rows below it hold nothing a cut may restore (a caller wrote
    /// the caches itself, [`Body::set_history`]).
    shadow_from: usize,
    /// A cut left ring slots the next step reads holding other positions'
    /// rows: the next refresh restores them before the step runs.
    restore: bool,
    /// The file, for the tensors the plan leaves on the host: the one
    /// mapping the host tier reads and the load populated.
    file: Arc<Split>,
    eps: f32,
    /// A draft's feature tap, once [`attach_features`] built it.
    tap: Option<FeatureTap>,
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

    /// Layer `layer`'s ring shadow bytes, in page-locked host memory — the
    /// figure `KvLayout::shadow_bytes` plans for the layer; `None` for a
    /// layer this card does not run.
    #[must_use]
    pub fn shadow_bytes(&self, layer: usize) -> Option<usize> {
        let i = layer.checked_sub(self.layers.start)?;
        self.kv.get(i).map(|_| self.shadows.layer_bytes())
    }

    /// Where the ring shadows live: the page-locked host allocation's bytes
    /// and the card's unified addressing, as the load checked it.
    #[must_use]
    pub fn shadow_host(&self) -> ShadowHost {
        ShadowHost {
            bytes: self.shadows.host.num_bytes(),
            unified_addressing: self.shadows.unified_addressing,
        }
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
    /// bytes, every row's: the pieces' scratch among them.
    #[must_use]
    pub fn step_buffers(&self) -> [(&'static str, usize); 9] {
        let lanes = |f: fn(&Lane) -> usize| self.lanes.iter().map(f).sum::<usize>();
        [
            (
                "hyper-connection streams",
                lanes(|l| l.hc.iter().map(|t| t.buf().num_bytes()).sum()),
            ),
            (
                "folded inputs",
                lanes(|l| l.folds.iter().map(DeviceBuffer::num_bytes).sum()),
            ),
            ("step image", lanes(|l| l.params.num_bytes())),
            (
                "selection lists",
                self.lists
                    .iter()
                    .flatten()
                    .map(DeviceBuffer::num_bytes)
                    .sum(),
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

    /// Row 0's image's device copy, as the captured chain reads it.
    #[must_use]
    pub fn params(&self) -> &DeviceBuffer<u32> {
        &self.lanes[0].params
    }

    /// Row `row`'s buffers as its last step left them: the streams' and the
    /// folds' ping-pong pairs, and its lists in indexer-layer order; `None`
    /// for a row the body does not hold.
    #[must_use]
    pub fn row_buffers(&self, row: usize) -> Option<RowBuffers<'_>> {
        let lane = self.lanes.get(row)?;
        Some(RowBuffers {
            streams: [lane.hc[0].buf(), lane.hc[1].buf()],
            folds: [&lane.folds[0], &lane.folds[1]],
            lists: self.lists.get(row)?,
        })
    }

    /// The tokens decoded so far, one per position.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Device bytes of one row's own buffers — its streams, folds, image
    /// copy and lists — and of one row of each piece's: what a second row
    /// adds. By name.
    #[must_use]
    pub fn row_bytes(&self) -> [(&'static str, usize); 7] {
        let lane = &self.lanes[0];
        [
            (
                "hyper-connection streams",
                lane.hc.iter().map(|t| t.buf().num_bytes()).sum(),
            ),
            (
                "folded inputs",
                lane.folds.iter().map(DeviceBuffer::num_bytes).sum(),
            ),
            ("step image", lane.params.num_bytes()),
            (
                "selection lists",
                self.lists[0].iter().map(DeviceBuffer::num_bytes).sum(),
            ),
            ("attention step words", self.attn.row_bytes()),
            ("ffn row", self.ffn.row_bytes()),
            ("engram site rows", self.glue.row_bytes()),
        ]
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

    /// The positions the caches hold: the plan's `ctx_max`, as the planner
    /// stores it.
    fn positions(&self) -> usize {
        self.planner.ctx_max() as usize
    }

    /// The file the body keeps for the tensors the plan leaves on the host.
    #[must_use]
    pub fn file(&self) -> &Split {
        &self.file
    }

    /// The tokens before the next step, for a caller that has written the
    /// state they leave into the buffers itself ([`Body::state_mut`]): the
    /// next [`ChainBody::decode_input`] runs at position `history.len()`. The
    /// ring shadow holds none of those positions, so no cut reaches a ring
    /// row below them.
    pub fn set_history(&mut self, history: &[u32]) -> Result<(), GpuError> {
        if history.len() >= self.positions() {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::set_history",
                detail: format!(
                    "{} tokens leave no position of the {} the caches hold",
                    history.len(),
                    self.positions()
                ),
            });
        }
        self.history.clear();
        self.history.extend_from_slice(history);
        self.holds.known(history.len());
        self.shadow_from = history.len();
        self.restore = false;
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

    /// [`attach_features`] on the body: refused once a step has run, and
    /// when a tap exists already.
    fn attach_features(&mut self, gpu: &Gpu, layers: &[usize]) -> Result<(), GpuError> {
        const WHAT: &str = "deepseek41 Body::attach_features";
        if self.tap.is_some() || !self.history.is_empty() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a fresh body: one tap, attached before the first step",
            });
        }
        self.tap = Some(FeatureTap::new(gpu, &self.layers, layers, self.n_embd())?);
        Ok(())
    }

    /// The layers the feature tap reads the input of, in its order; `None`
    /// without a tap.
    #[must_use]
    pub fn feature_layers(&self) -> Option<&[usize]> {
        self.tap.as_ref().map(|t| t.layers.as_slice())
    }

    /// f32 values of one row's features: tapped layers × `n_embd`; 0 without
    /// a tap.
    #[must_use]
    pub fn feature_width(&self) -> usize {
        self.tap.as_ref().map_or(0, FeatureTap::width)
    }

    /// Device bytes of the feature tap's buffer, every row's; 0 without one.
    #[must_use]
    pub fn feature_bytes(&self) -> usize {
        self.tap.as_ref().map_or(0, |t| t.dev.whole().num_bytes())
    }

    /// Rows `0 .. rows` of the feature tap as the last step or pair pass
    /// left them, copied to the host in one transfer (a blocking read on the
    /// engine stream): row `r` holds the features of position `pos + r`.
    /// Refused without a tap, for a row whose last refresh is not the
    /// position after the row before it (row 1 after a one-row step), and
    /// for a position the history no longer holds (taken back).
    pub fn read_features(&mut self, gpu: &Gpu, rows: usize) -> Result<Features<'_>, GpuError> {
        const WHAT: &str = "deepseek41 Body::read_features";
        let len = self.history.len();
        let tap = self.tap.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "a feature tap (attach_features)",
        })?;
        let first = match tap.pos.first().copied().flatten() {
            Some(p) if (1..=PAIR_ROWS).contains(&rows) => p,
            p => {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("{rows} rows of {PAIR_ROWS}; row 0 holds position {p:?}"),
                });
            }
        };
        for (r, p) in tap.pos.iter().take(rows).enumerate() {
            let want = first as usize + r;
            if *p != u32::try_from(want).ok() || want >= len {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "row {r} holds position {p:?}, not {want} of a history of {len}: a row \
                         the last pass did not run, or a position taken back"
                    ),
                });
            }
        }
        let w = tap.width();
        let host = &mut tap.host[..rows * w];
        let stream = gpu.stream();
        if rows == 1 {
            tap.dev.part(0).copy_to_host(stream, host)?;
        } else {
            tap.dev.whole().copy_to_host(stream, host)?;
        }
        Ok(Features {
            pos: first,
            width: w,
            values: host,
        })
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
        let mut p = self.parts();
        p.hybrid.begin_chain(gpu.stream())?;
        let mut cur = Cursor::default();
        p.begin_row(gpu, 0, &mut cur)?;
        for (i, l) in p.layers.clone().enumerate() {
            let step = p.steps[i];
            if p.engram(gpu, w, i, l, 0, &mut cur)? {
                let lane = &p.lanes[0];
                observe(
                    gpu,
                    Seam::Engram {
                        layer: l,
                        streams: lane.hc[cur.s].buf(),
                        fold: &lane.folds[cur.f],
                    },
                )?;
            }
            p.attn(gpu, w, i, l, 0, &mut cur)?;
            {
                let lane = &p.lanes[0];
                observe(
                    gpu,
                    Seam::Attn {
                        layer: l,
                        streams: lane.hc[cur.s].buf(),
                        fold: &lane.folds[cur.f],
                        taps: p.attn.taps(),
                        list: match step.list {
                            ListOf::None => None,
                            ListOf::Reads(j) | ListOf::Writes { list: j, .. } => p.lists[0].get(j),
                        },
                    },
                )?;
            }
            p.ffn_go(gpu, w, i, l, 0, cur)?;
            p.ffn_join(gpu, i, l, 0, &mut cur)?;
            let lane = &p.lanes[0];
            observe(
                gpu,
                Seam::Ffn {
                    layer: l,
                    streams: lane.hc[cur.s].buf(),
                    fold: step.folds.then_some(&lane.folds[cur.f]),
                    taps: p.ffn.taps(),
                },
            )?;
        }
        p.head(gpu, w, 0, cur, head)
    }

    /// Enqueue the pair pass: rows 0 and 1 — the two tokens
    /// [`Body::decode_pair`] planned, at `pos` and `pos + 1` — one layer
    /// apart on the one stream, `heads[r]` taking row `r`'s logits. Per
    /// layer `l`, each row in turn joins its layer `l − 1`, runs the engram
    /// step where `l` carries one, its attention and its MoE sub-layer up to
    /// the host leg's shadow; so the host serves row 0's layer `l` while the
    /// card runs row 1's layer `l` up to its go, and row 1's while the card
    /// runs row 0's layer `l + 1`. Row 1's attention at layer `l` reads the
    /// cache row row 0 wrote at `l` just before it; nothing row 1 writes is
    /// read by row 0. Every launch is the one-token step's, in the same
    /// order within its row, so the two rows' logits and every cache, row,
    /// key and state are bit for bit those of two steps in turn.
    /// Asynchronous apart from the host tier's service of an eager chain.
    pub fn enqueue_pair(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: [&mut Head; PAIR_ROWS],
    ) -> Result<(), GpuError> {
        let mut p = self.parts();
        if p.lanes.len() < PAIR_ROWS {
            return Err(GpuError::State {
                what: "deepseek41 Body::enqueue_pair",
                missing: "a second row of buffers",
            });
        }
        p.hybrid.begin_chain_of(gpu.stream(), Chain::Pair)?;
        let mut cur = [Cursor::default(); PAIR_ROWS];
        for (row, c) in cur.iter_mut().enumerate() {
            p.begin_row(gpu, row, c)?;
        }
        let mut last = None;
        for (i, l) in p.layers.clone().enumerate() {
            for (row, c) in cur.iter_mut().enumerate() {
                if let Some((j, k)) = last {
                    p.ffn_join(gpu, j, k, row, c)?;
                }
                p.engram(gpu, w, i, l, row, c)?;
                p.attn(gpu, w, i, l, row, c)?;
                p.ffn_go(gpu, w, i, l, row, *c)?;
            }
            last = Some((i, l));
        }
        let (j, k) = last.ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_pair",
            missing: "a layer",
        })?;
        for ((row, head), c) in heads.into_iter().enumerate().zip(cur.iter_mut()) {
            p.ffn_join(gpu, j, k, row, c)?;
            p.head(gpu, w, row, *c, head)?;
        }
        Ok(())
    }

    /// The pair pass's host half: the step at `pos` for `tokens[0]` into
    /// row 0's image copy, then the step at `pos + 1` for `tokens[1]` into
    /// row 1's — each planned after the tokens before it, the first of the
    /// two included — with both tokens appended to the history.
    pub fn decode_pair(
        &mut self,
        stream: &CudaStream,
        tokens: [u32; PAIR_ROWS],
        pos: u32,
    ) -> Result<(), GpuError> {
        if self.lanes.len() < PAIR_ROWS {
            return Err(GpuError::State {
                what: "deepseek41 Body::decode_pair",
                missing: "a second row of buffers",
            });
        }
        for (row, (token, pos)) in tokens.into_iter().zip(pos..).enumerate() {
            let input = self.decode_input(token, pos)?;
            self.refresh_row(stream, &input, row)?;
        }
        Ok(())
    }

    /// The longest prefix of at most `n` positions the caches can be cut back
    /// to ([`Body::rollback`]): every position when `n` reaches them all, 0
    /// when nothing shorter can be kept.
    ///
    /// - A compressed stream of ratio `r` keeps its state ring in slot `p %
    ///   r`, and the group of positions `g .. g + r` (`g` a multiple of `r`)
    ///   reads the ring's rows of the positions before the step that
    ///   completes it. A cut to `n` inside a group (`g < n`) needs those of
    ///   `g .. n`, and a later step of the same residue may have overwritten
    ///   them. So `n` is rounded down until, for every ratio, it starts a
    ///   group or the state ring still holds `g .. n` ([`Holds`]). With
    ///   V4.1's ratios 2 and 1 after a plain run of `len` positions: `n` is
    ///   kept when it is even or `len − 1`, else `n − 1`.
    /// - The raw window ring: the step at `n` reads the rows of `n + 1 −
    ///   slots ..= n − 1`; the slots of those that hold another position's
    ///   row ([`Holds::stale`]) are restored from the shadow, which holds the
    ///   rows of positions from `shadow_from` on. A cut that needs a row below
    ///   that keeps nothing.
    /// - Compressed rows, index keys and lists are indexed by position and
    ///   written by the step that completes them, and a step reads only the
    ///   ones below its own position, so whatever stands at or past the cut
    ///   is never read before it is written again. The engram history is the
    ///   token history, truncated.
    #[must_use]
    pub fn keep_point(&self, n: usize) -> usize {
        let len = self.history.len();
        if n >= len {
            return len;
        }
        let mut k = n;
        while k > 0 && !self.holds.state_keeps(k) {
            k -= 1;
        }
        if self.holds.stale(k).any(|q| q < self.shadow_from) {
            return 0;
        }
        k
    }

    /// `n_embd`, as the planner's image holds it.
    fn n_embd(&self) -> usize {
        self.image.layout().dims().n_embd
    }

    /// Take back the tokens from position `pos` on: the next step runs at
    /// `pos`, and the step there and every step after it write what they
    /// did before, bit for bit. `pos` must be a point [`Body::keep_point`]
    /// grants. The ring slots the step at `pos` reads that hold later rows
    /// are restored from the shadow by the next refresh, host to device on
    /// the engine stream and outside any graph (this call has no stream);
    /// nothing else on the device is touched.
    pub fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        let (to, len) = (pos as usize, self.history.len());
        let kept = self.keep_point(to);
        if to > len || kept != to {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::rollback",
                detail: format!(
                    "back to position {pos} after {len} tokens: the caches can be cut back to \
                     {kept} at most there (see Body::keep_point)"
                ),
            });
        }
        if to < len {
            self.history.truncate(to);
            self.shadow_from = self.shadow_from.min(to);
            self.restore = self.holds.stale(to).next().is_some();
        }
        Ok(())
    }

    /// One host-to-device copy of the whole image into row `row`'s copy,
    /// which must hold the step `input` names.
    fn refresh_row(
        &mut self,
        stream: &CudaStream,
        input: &StepInput,
        row: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "deepseek41 Body::refresh";
        if self.image.pos() != Some(input.pos) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the image holds the step at {:?}, the input names {}",
                    self.image.pos(),
                    input.pos
                ),
            });
        }
        let pos = input.pos as usize;
        if self.restore {
            for run in self.holds.stale_runs(pos) {
                for (i, layer) in self.kv.iter_mut().enumerate() {
                    self.shadows
                        .restore(i, &mut layer.ring, stream, run.clone())?;
                }
                run.for_each(|q| self.holds.ring_wrote(q));
            }
            self.restore = false;
        }
        self.holds.wrote(pos);
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: WHAT,
            missing: "the row's buffers",
        })?;
        lane.params.copy_from_host(stream, self.image.words())?;
        if let Some(tap) = self.tap.as_mut() {
            tap.pos[row] = Some(input.pos);
        }
        Ok(())
    }

    /// The body's buffers a row's launches borrow, apart from the host half.
    fn parts(&mut self) -> Parts<'_> {
        let Body {
            layers,
            kv,
            shadows,
            steps,
            lanes,
            lists,
            slots,
            hybrid,
            attn,
            ffn,
            glue,
            tap,
            ..
        } = self;
        Parts {
            layers,
            kv,
            shadows,
            steps,
            lanes,
            lists,
            slots,
            hybrid,
            attn,
            ffn,
            glue,
            tap: tap.as_mut(),
        }
    }
}

/// Row `row`'s buffers as [`Body::row_buffers`] shows them.
pub struct RowBuffers<'a> {
    pub streams: [&'a DeviceBuffer<f32>; 2],
    pub folds: [&'a DeviceBuffer<f32>; 2],
    pub lists: &'a [DeviceBuffer<u32>],
}

/// The body's step buffers and pieces, borrowed apart from its host half:
/// what one row's launches take.
struct Parts<'a> {
    layers: &'a Range<usize>,
    kv: &'a mut [LayerKv],
    shadows: &'a mut Shadows,
    steps: &'a [LayerStep],
    lanes: &'a mut [Lane],
    lists: &'a mut [Vec<DeviceBuffer<u32>>],
    slots: &'a DeviceTensor<u32>,
    hybrid: &'a mut Hybrid<Ds41Host>,
    attn: &'a mut AttnChain,
    ffn: &'a mut FfnPiece,
    glue: &'a mut Glue,
    tap: Option<&'a mut FeatureTap>,
}

impl Parts<'_> {
    /// Row `row`'s start: the attention piece's gather of the row's step
    /// words, then the embedding broadcast into the row's streams and layer
    /// 0's input.
    fn begin_row(&mut self, gpu: &Gpu, row: usize, cur: &mut Cursor) -> Result<(), GpuError> {
        let (attn, glue) = (&mut *self.attn, &*self.glue);
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        attn.enqueue_step_of(gpu, &lane.params, row)?;
        let Lane { hc, folds, params } = lane;
        let [s0, _] = hc;
        let [f0, _] = folds;
        glue.enqueue_embed(gpu, params, s0.buf_mut(), f0)?;
        *cur = Cursor::default();
        Ok(())
    }

    /// Row `row`'s engram step before layer `l` (index `i`), where the layer
    /// carries one; whether it does.
    fn engram(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        i: usize,
        l: usize,
        row: usize,
        cur: &mut Cursor,
    ) -> Result<bool, GpuError> {
        if !self.steps[i].engram {
            return Ok(false);
        }
        let (glue, pre) = (&mut *self.glue, self.ffn.taps_of(row)?.hc);
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        let (streams, out) = ping(&mut lane.hc, cur.s);
        let [f0, f1] = &mut lane.folds;
        glue.enqueue_engram_of(
            gpu,
            w,
            l,
            row,
            EngramStep {
                streams: streams.buf(),
                pre,
                out: out.buf_mut(),
                input: if cur.f == 0 { f0 } else { f1 },
            },
        )?;
        cur.s ^= 1;
        Ok(true)
    }

    /// Row `row`'s attention sub-layer of layer `l` (index `i`).
    fn attn(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        i: usize,
        l: usize,
        row: usize,
        cur: &mut Cursor,
    ) -> Result<(), GpuError> {
        let step = self.steps[i];
        let lists = self.lists.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's lists",
        })?;
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        let (streams_in, streams_out) = ping(&mut lane.hc, cur.s);
        let (fold_in, fold_out) = ping(&mut lane.folds, cur.f);
        let LayerIo {
            ring,
            shadow,
            compressed,
            selection,
        } = layer_io(self.kv, self.shadows, lists, i, &step)?;
        self.attn.enqueue_layer_of(
            gpu,
            w,
            l,
            row,
            AttnIo {
                streams_in: streams_in.buf(),
                fold_in,
                streams_out: streams_out.buf_mut(),
                fold_out,
                ring,
                shadow,
                compressed,
                selection,
            },
        )?;
        cur.s ^= 1;
        cur.f ^= 1;
        Ok(())
    }

    /// Row `row`'s MoE sub-layer of layer `l` (index `i`) up to its host
    /// leg's shadow, the token-only work of the engram site at the next
    /// layer inside it; reads the halves `cur` names.
    fn ffn_go(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        i: usize,
        l: usize,
        row: usize,
        cur: Cursor,
    ) -> Result<(), GpuError> {
        let step = self.steps[i];
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        let Lane { hc, folds, params } = lane;
        let (streams_in, streams_out) = ping(hc, cur.s);
        let (fold_in, fold_out) = ping(folds, cur.f);
        let mut engram_kv = step.shadow_site.map(|layer| EngramKv {
            glue: &mut *self.glue,
            w,
            params: &*params,
            layer,
            row,
        });
        let mut one: [&mut dyn ShadowWork; 1];
        let shadow: &mut [&mut dyn ShadowWork] = match engram_kv.as_mut() {
            Some(job) => {
                one = [job as &mut dyn ShadowWork];
                &mut one
            }
            None => &mut [],
        };
        self.ffn.enqueue_go_half(
            gpu,
            w,
            CardStacks::of(w, l)?,
            &FfnIo {
                streams: streams_in.buf(),
                fold_in,
                streams_out: streams_out.buf_mut(),
                fold_out: step.folds.then_some(fold_out),
                slots: self.slots,
            },
            &mut *self.hybrid,
            l,
            row,
            shadow,
        )
    }

    /// Row `row`'s MoE sub-layer of layer `l` (index `i`) from its wait on,
    /// after its [`Parts::ffn_go`] with the same cursor.
    fn ffn_join(
        &mut self,
        gpu: &Gpu,
        i: usize,
        l: usize,
        row: usize,
        cur: &mut Cursor,
    ) -> Result<(), GpuError> {
        let step = self.steps[i];
        let lane = self.lanes.get_mut(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        let (streams_in, streams_out) = ping(&mut lane.hc, cur.s);
        let (fold_in, fold_out) = ping(&mut lane.folds, cur.f);
        self.ffn.enqueue_join_half(
            gpu,
            FfnIo {
                streams: streams_in.buf(),
                fold_in,
                streams_out: streams_out.buf_mut(),
                fold_out: step.folds.then_some(fold_out),
                slots: self.slots,
            },
            &mut *self.hybrid,
            l,
            row,
        )?;
        cur.s ^= 1;
        if step.folds {
            cur.f ^= 1;
        }
        if let Some(tap) = self.tap.as_deref_mut() {
            tap.enqueue(gpu, i, row, lane.hc[cur.s].buf())?;
        }
        Ok(())
    }

    /// Row `row`'s end: its streams' collapse into `head`, and the head.
    fn head(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        row: usize,
        cur: Cursor,
        head: &mut Head,
    ) -> Result<(), GpuError> {
        let pre = self.ffn.taps_of(row)?.hc;
        let lane = self.lanes.get(row).ok_or(GpuError::State {
            what: "deepseek41 Body::enqueue_chain",
            missing: "the row's buffers",
        })?;
        self.glue
            .enqueue_head(gpu, w, lane.hc[cur.s].buf(), pre, head)
    }
}

/// Layer `i`'s window ring and its shadow, the compressed rows its attention
/// reads and the list it reads them through, out of the body's layers `kv`,
/// `shadows` and `lists`.
fn layer_io<'a>(
    kv: &'a mut [LayerKv],
    shadows: &'a mut Shadows,
    lists: &'a mut [DeviceBuffer<u32>],
    i: usize,
    step: &LayerStep,
) -> Result<LayerIo<'a>, GpuError> {
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
    let LayerKv {
        ring,
        rows,
        keys,
        values,
        scores,
    } = rest.first_mut().ok_or(refuse("the layer's cache"))?;
    let shadow = shadows
        .layer_mut(i)
        .ok_or(refuse("the layer's ring shadow"))?;
    let compressed = match step.rows {
        RowsOf::None => Compressed::None,
        RowsOf::Reads(src) => {
            let rows = before
                .get(src)
                .and_then(|k| k.rows.as_ref())
                .ok_or(refuse("the source layer's compressed rows"))?;
            Compressed::Read(rows)
        }
        RowsOf::Source => {
            let rows = rows
                .as_mut()
                .ok_or(refuse("the layer's own compressed rows"))?;
            let state = match (values.as_mut(), scores.as_mut()) {
                (Some(v), Some(sc)) => Some((v, sc)),
                _ => None,
            };
            Compressed::Source(SourceIo {
                rows,
                keys: keys.as_mut(),
                ring: state,
            })
        }
    };
    Ok(LayerIo {
        ring,
        shadow,
        compressed,
        selection,
    })
}

/// The host's record of which position's row each cache slot that a later
/// position overwrites holds — the raw window ring's `slots` and each
/// compressed stream's state ring of `ratio` — every layer alike, since every
/// step writes every layer's slots. [`Body::keep_point`] reads it.
struct Holds {
    ring: Vec<Option<usize>>,
    /// Per stream of the plan, above ratio 1: its ratio and its slots.
    states: Vec<(usize, Vec<Option<usize>>)>,
}

impl Holds {
    fn new(slots: usize, ratios: &[u32]) -> Holds {
        Holds {
            ring: vec![None; slots],
            states: ratios
                .iter()
                .map(|&r| r as usize)
                .filter(|&r| r > 1)
                .map(|r| (r, vec![None; r]))
                .collect(),
        }
    }

    /// The caches as a run of `len` positions from a reset leaves them.
    fn known(&mut self, len: usize) {
        self.ring.fill(None);
        for (_, s) in &mut self.states {
            s.fill(None);
        }
        let back = self
            .states
            .iter()
            .map(|&(r, _)| r)
            .fold(self.ring.len(), usize::max);
        for q in len.saturating_sub(back)..len {
            self.wrote(q);
        }
    }

    /// The step at `p` writes its ring slot and its state slots.
    fn wrote(&mut self, p: usize) {
        self.ring_wrote(p);
        for (r, s) in &mut self.states {
            s[p % *r] = Some(p);
        }
    }

    fn ring_wrote(&mut self, p: usize) {
        let n = self.ring.len();
        if n > 0 {
            self.ring[p % n] = Some(p);
        }
    }

    /// Whether every state ring still holds the positions the group of `n`
    /// reads from it: those of `g .. n`, `g` the group's start.
    fn state_keeps(&self, n: usize) -> bool {
        self.states.iter().all(|(r, s)| {
            let g = n - n % r;
            (g..n).all(|q| s[q % r] == Some(q))
        })
    }

    /// The positions the step at `n` reads from the ring, `n + 1 − slots ..=
    /// n − 1`, whose slot holds another position's row.
    fn stale(&self, n: usize) -> impl Iterator<Item = usize> + '_ {
        let slots = self.ring.len();
        ((n + 1).saturating_sub(slots)..n).filter(move |&q| self.ring[q % slots] != Some(q))
    }

    /// [`Holds::stale`] as runs of consecutive positions.
    fn stale_runs(&self, n: usize) -> Vec<Range<usize>> {
        let mut runs: Vec<Range<usize>> = Vec::new();
        for q in self.stale(n) {
            match runs.last_mut() {
                Some(r) if r.end == q => r.end = q + 1,
                _ => runs.push(q..q + 1),
            }
        }
        runs
    }
}

/// A layer's caches as [`layer_io`] lends them to its attention.
struct LayerIo<'a> {
    ring: &'a mut DeviceTensor<u16>,
    shadow: &'a mut DeviceTensor<u16>,
    compressed: Compressed<'a>,
    selection: Selection<'a>,
}

/// Rows of features [`Body::read_features`] brought to the host: row `r`
/// is `values[r * width ..][.. width]`, the features of position `pos + r`.
pub struct Features<'a> {
    pub pos: u32,
    pub width: usize,
    pub values: &'a [f32],
}

impl Features<'_> {
    /// Row `r`'s features.
    #[must_use]
    pub fn row(&self, r: usize) -> Option<&[f32]> {
        self.values.get(r * self.width..(r + 1) * self.width)
    }
}

/// The feature tap ([`attach_features`]): per row, the stream means of the
/// tapped layers' inputs, concatenated in the tap's order, on the card and
/// their host copy.
struct FeatureTap {
    hc: HcKernels,
    /// The tapped layers, in order.
    layers: Vec<usize>,
    /// Per layer of the card, by index: the tap slot its streams fill — the
    /// slot of the layer after it.
    after: Vec<Option<usize>>,
    n_embd: usize,
    /// Row `r`'s features: part `r`.
    dev: PartedBuffer<f32, PAIR_ROWS>,
    host: Vec<f32>,
    /// The position each row's last refresh named: the one its features hold
    /// once that step has run.
    pos: [Option<u32>; PAIR_ROWS],
}

impl FeatureTap {
    fn new(
        gpu: &Gpu,
        card: &Range<usize>,
        layers: &[usize],
        n_embd: usize,
    ) -> Result<FeatureTap, GpuError> {
        const WHAT: &str = "deepseek41 FeatureTap::new";
        let increasing = layers.windows(2).all(|w| w[0] < w[1]);
        let mut after = vec![None; card.len()];
        for (k, &l) in layers.iter().enumerate() {
            let i = l
                .checked_sub(1)
                .and_then(|b| b.checked_sub(card.start))
                .filter(|&i| i < card.len() && increasing);
            match i {
                Some(i) => after[i] = Some(k),
                None => {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "tapped layers {layers:?}: each strictly after the one before, and the \
                             layer before each one on this card's {card:?}"
                        ),
                    });
                }
            }
        }
        let width = layers.len() * n_embd;
        if width == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("tapped layers {layers:?} of {n_embd} values"),
            });
        }
        Ok(FeatureTap {
            hc: HcKernels::load(gpu.context())?,
            layers: layers.to_vec(),
            after,
            n_embd,
            dev: PartedBuffer::zeroed(gpu.stream(), [width; PAIR_ROWS])?,
            host: vec![0.0; PAIR_ROWS * width],
            pos: [None; PAIR_ROWS],
        })
    }

    fn width(&self) -> usize {
        self.layers.len() * self.n_embd
    }

    /// Row `row`'s mean of `streams`, the streams layer index `i` of the card
    /// left, where the layer after it is tapped. Asynchronous, capturable.
    fn enqueue(
        &mut self,
        gpu: &Gpu,
        i: usize,
        row: usize,
        streams: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Some(k) = self.after.get(i).copied().flatten() else {
            return Ok(());
        };
        let n = self.n_embd;
        self.hc
            .enqueue_mean(gpu.stream(), streams, n, k * n, self.dev.part_mut(row))
    }
}

impl ChainBody for Body {
    type Input = StepInput;
    type Meta = Hparams;

    fn arch() -> Arch {
        Arch::Deepseek41
    }

    /// V4.1 derives no new values at load: it moves the attention's Q3_K
    /// projections that read one activation into row joins
    /// ([`join_projections`]), one launch per join.
    fn derive(
        stream: &CudaStream,
        file: &Split,
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        let hp = Hparams::read(file).map_err(|e| GpuError::plan("deepseek41 Body::derive", e))?;
        join_projections(stream, &hp, layers, w)
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
    /// embedding row read from the file on this thread and its engram rows by
    /// the rows' helper meanwhile ([`StepRows::fill`]), and its image built.
    /// A position other than the next one is refused.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        const WHAT: &str = "deepseek41 Body::decode_input";
        if self.history.len() != pos as usize || self.history.len() >= self.positions() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a step at position {pos} after {} tokens, in caches of {} positions",
                    self.history.len(),
                    self.positions()
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

    /// One host-to-device copy of the whole image into row 0's copy; the
    /// image must hold the step `input` names.
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        self.refresh_row(stream, input, 0)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        self.enqueue_observed(gpu, w, head, &mut |_, _| Ok(()))
    }

    /// Every ring, compressed row, index key, compressor state, and every
    /// row's streams, folds and lists are zeroed in place — a captured chain
    /// keeps their addresses — and the token history is emptied. The ring
    /// shadows are not: a cut reads only shadow rows a step since wrote.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for layer in &mut self.kv {
            layer.zero(stream)?;
        }
        for lane in &mut self.lanes {
            for t in &mut lane.hc {
                t.buf_mut().zero_async(stream)?;
            }
            for f in &mut lane.folds {
                f.zero_async(stream)?;
            }
        }
        for l in self.lists.iter_mut().flatten() {
            l.zero_async(stream)?;
        }
        if let Some(tap) = self.tap.as_mut() {
            tap.pos = [None; PAIR_ROWS];
        }
        self.history.clear();
        self.holds.known(0);
        self.shadow_from = 0;
        self.restore = false;
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

    /// The ring shadows are page-locked host memory, not counted here
    /// ([`Body::shadow_host`]); a feature tap's device buffer is.
    fn resident_bytes(&self) -> usize {
        let caches: usize = self
            .kv
            .iter()
            .map(|l| l.buffers().iter().map(|&(_, n)| n).sum::<usize>())
            .sum();
        caches + self.step_buffers().iter().map(|&(_, n)| n).sum::<usize>() + self.feature_bytes()
    }

    /// The host tier's share of the chain a replay just submitted.
    fn serve_replay(&mut self) -> Result<(), GpuError> {
        self.hybrid.serve_captured()
    }

    fn decode_pair(
        &mut self,
        stream: &CudaStream,
        tokens: [u32; 2],
        pos: u32,
    ) -> Result<(), GpuError> {
        Body::decode_pair(self, stream, tokens, pos)
    }

    fn enqueue_pair(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: [&mut Head; 2],
    ) -> Result<(), GpuError> {
        Body::enqueue_pair(self, gpu, w, heads)
    }

    /// The host tier's share of the pair pass a replay just submitted.
    fn serve_replay_pair(&mut self) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(Chain::Pair)
    }

    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Body::rollback(self, pos)
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
        let shadows = Shadows::new(gpu, layers.len(), ctx_max, hp.head_dim)?;

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
        let lanes = (0..PAIR_ROWS)
            .map(|_| Lane::new(stream, hp, image.layout().words()))
            .collect::<Result<Vec<_>, _>>()?;

        let n_expert = hp.experts.n_expert;
        let map = SlotMap::from_rows(
            layers.clone(),
            n_expert,
            plan_slots(plan, card, &layers, n_expert)?,
        )?;
        let slots = DeviceTensor::upload(stream, map.as_slice(), layers.len(), n_expert)?;

        let attn =
            AttnChain::with_rows(gpu, hp, layers.clone(), image.layout(), &planner, PAIR_ROWS)?;
        let ffn = FfnPiece::with_rows(gpu, hp, &map, PAIR_ROWS)?;
        let glue = Glue::with_rows(gpu, hp, image.layout(), PAIR_ROWS)?;
        let steps = layer_steps(hp, &layers, &kv, &ffn, &glue)?;
        let writers = steps
            .iter()
            .filter(|s| matches!(s.list, ListOf::Writes { .. }))
            .count();
        let lists = (0..PAIR_ROWS)
            .map(|_| {
                (0..writers)
                    .map(|_| DeviceBuffer::zeroed(stream, STEP_TOKENS * attn.list_len()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let boundary = Boundary::with_rows(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: hp.n_embd,
                n_used: hp.experts.n_used,
            },
            map,
            levers()?.overlap,
            PAIR_ROWS,
        )?;
        let file = Arc::new(file);
        let host = Ds41Host::build(Arc::clone(&file), hp, layers.clone())?;
        let hybrid = Hybrid::new(boundary, host, layers.len())?;
        let holds = Holds::new(
            kv.first().map_or(0, |k| k.ring.rows()),
            planner.stream_ratios(),
        );

        Ok(Body {
            layers,
            kv,
            shadows,
            steps,
            lanes,
            lists,
            image,
            slots,
            hybrid,
            attn,
            ffn,
            glue,
            planner,
            plan: StepPlan::default(),
            rows,
            history: Vec::with_capacity(ctx_max),
            holds,
            shadow_from: 0,
            restore: false,
            file,
            eps: hp.rms_eps,
            tap: None,
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
/// broadcast writes, and the layer before each such site runs the site's
/// token-only work in its MoE sub-layer's shadow; and the MoE sub-layer folds
/// the next input everywhere except before an engram layer and after the last
/// layer, where the glue folds instead.
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
            shadow_site: (l + 1 < layers.end && sites.contains(&(l + 1))).then_some(l + 1),
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
