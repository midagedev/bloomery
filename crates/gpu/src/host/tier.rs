//! The expert tier: a second card hung under the host tier ([`TierCard`]).
//! The stage card keeps its plan byte for byte; per hybrid layer the tier
//! card computes the routed experts its [`TierSet`] names, in the host leg's
//! shadow, and the stage card's join reads their rows through the mapping.
//!
//! Per tier layer, beside the host tier's protocol ([`super`]):
//!
//! ```text
//! stage: router → handoff (+ q8_1 x, tier places) → go (+ tier go) → [shadow] → wait (host, tier) → join
//! tier:  wait tier go, −1 → copy x, places in → [gate·up → h q8_1 → down → rows] → barrier → cnt +1, prog +1
//! ```
//!
//! The tier's words live in its own host-mapped page ([`TierLayout`]): per
//! row a go counter the stage card adds one to and the tier takes back, and
//! a counter the tier adds one to and the stage card's wait takes back — the
//! host tier's counter rule, one writer at a time — then the progress word,
//! which the tier adds one to per layer served and the host reads, and the
//! tier's fault copy. Per row the page holds the image the stage card's
//! handoff writes — the tier's places of the routed slots, the stage card's
//! q8_1 activation — and the rows the tier's down writes, one per routed
//! slot, which the stage card's join reads in place.
//!
//! The host thread never drives the tier per layer: a captured chain's tier
//! work is one graph per chain kind ([`Chain::Step`], [`Chain::Pair`]),
//! captured from the stage capture's go order at its first replay and
//! launched on the tier's stream before the host serves the replay. Its
//! last layer copies the tier's fault word into the page before its signal.
//! After the host has served the replay's last layer it waits for the tier's
//! progress under the go deadline and reads the copy: a raised word is the
//! step's fault, merged with the stage card's ([`crate::fault::read_cards`],
//! the first layer wins). An eager chain enqueues each tier layer as the
//! host serves it and settles it after, the copy taken each layer.
//!
//! A tier that stops signalling is a lost card: the host's go deadline, or
//! the settle's, names the tier when its progress is the one behind, the
//! host tier is poisoned as [`super::PoisonKind::CardLost`], which a reset
//! does not lift, and the release writes [`RELEASE`] to the tier's counters
//! and go counters too, so both streams drain.

use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use super::page::MAX_ROWS;
use super::slots::{Slot, SlotMap};
use super::step::{Boundary, Chain, RELEASE};
use crate::fault::{FAULT_NONE, Fault};
use crate::graph::{Graph, MappedHost, cu, mem_batch, op_add, op_barrier_sys, op_wait_geq};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError, Q8Act};

/// What the tier's errors name.
const WHAT: &str = "TierCard";

/// Spins between two clock reads while the host waits for the tier.
const DEADLINE_POLL: u32 = 1 << 12;

/// How long a live tier takes at most to finish the layer it is on: a host
/// service that failed while the tier was behind waits this long before it
/// names the tier lost.
pub(super) const TIER_GRACE: Duration = Duration::from_millis(100);

// ------------------------------------------------------------------ set

/// Which experts of each hybrid layer the tier card holds: per layer of
/// `layers`, its expert ids in tier-slot order (ascending, each once, each
/// below `n_expert`) — slot `s` of the tier's routed stacks holds `ids[s]`.
/// A layer may hold none; the tier then does nothing for it. A copy of the
/// slot map's tier rows ([`TierSet::of_map`]), made before the tier card
/// loads its stacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierSet {
    layers: Range<usize>,
    n_expert: usize,
    ids: Vec<Box<[u32]>>,
}

impl TierSet {
    /// The set from its rows: `layers.len()` lists of ids, each ascending,
    /// each id once and below `n_expert`; any other row is refused by name.
    pub fn new(
        layers: Range<usize>,
        n_expert: usize,
        ids: Vec<Vec<u32>>,
    ) -> Result<TierSet, GpuError> {
        const WHAT: &str = "TierSet::new";
        if ids.len() != layers.len() || n_expert == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} rows for layers {layers:?} of {n_expert} experts",
                    ids.len()
                ),
            ));
        }
        for (row, l) in ids.iter().zip(layers.clone()) {
            if let Some(w) = row.windows(2).find(|w| w[0] >= w[1]) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {l}: ids {} then {} are not ascending", w[0], w[1]),
                ));
            }
            if let Some(&e) = row.last().filter(|&&e| e as usize >= n_expert) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {l}: expert {e} of {n_expert}"),
                ));
            }
        }
        Ok(TierSet {
            layers,
            n_expert,
            ids: ids.into_iter().map(Vec::into_boxed_slice).collect(),
        })
    }

    /// The tier's set as `map` holds it: per layer of the map, the ids of
    /// its tier entries (`TIER | s`) in tier-slot order. The map is the one
    /// owner of where an expert runs; the set is a copy of its tier rows the
    /// tier card keeps.
    pub fn of_map(map: &SlotMap) -> Result<TierSet, GpuError> {
        const WHAT: &str = "TierSet::of_map";
        let layers = map.layers();
        let n = u32::try_from(map.n_expert())
            .map_err(|_| GpuError::shape(WHAT, "the map's experts pass u32"))?;
        let mut rows = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let k = map.on_tier(l)?;
            let mut ids = vec![u32::MAX; k];
            for id in 0..n {
                if let Some(Slot::Tier(s)) = map.slot(l, id) {
                    let at = ids.get_mut(s as usize).ok_or_else(|| {
                        GpuError::shape(WHAT, format!("layer {l}: tier slot {s} of {k}"))
                    })?;
                    *at = id;
                }
            }
            rows.push(ids);
        }
        TierSet::new(layers, map.n_expert(), rows)
    }

    /// The layers the set has rows for.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Experts per layer of the file.
    #[must_use]
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// Layer `layer`'s ids in tier-slot order; `None` for a layer the set
    /// has no row for.
    #[must_use]
    pub fn ids(&self, layer: usize) -> Option<&[u32]> {
        let i = layer.checked_sub(self.layers.start)?;
        self.ids.get(i).map(|r| &r[..])
    }

    /// The experts of layer `layer` the tier holds; a layer the set has no
    /// row for is refused by name.
    pub fn on_tier(&self, layer: usize) -> Result<usize, GpuError> {
        self.ids(layer).map(<[u32]>::len).ok_or_else(|| {
            GpuError::shape(
                "TierSet::on_tier",
                format!(
                    "layer {layer} is outside the set's layers {:?}",
                    self.layers
                ),
            )
        })
    }

    /// Experts the tier holds, summed over its layers.
    #[must_use]
    pub fn experts(&self) -> usize {
        self.ids.iter().map(|r| r.len()).sum()
    }
}

// ------------------------------------------------------------------ page

/// The tier page's flag words, one 64-byte line each.
#[derive(Clone, Copy, Debug)]
enum TWord {
    /// Row `.0`'s go counter: the stage card adds one per go of a tier
    /// layer; the tier waits for it and takes it back.
    Go(usize),
    /// Row `.0`'s counter: the tier adds one per layer served; the stage
    /// card's wait takes it back.
    Cnt(usize),
    /// Layers the tier has served since load (wrapping); the host reads it.
    Prog,
    /// The tier's fault copy: the first-layer word, then that layer's mask.
    Fault,
}

impl TWord {
    const fn offset(self) -> usize {
        match self {
            TWord::Go(r) => 128 * r,
            TWord::Cnt(r) => 64 + 128 * r,
            TWord::Prog => 128 * MAX_ROWS,
            TWord::Fault => 128 * MAX_ROWS + 64,
        }
    }
}

/// Byte offset of row 0's image: past every flag word.
const TIER_PAYLOAD_OFF: usize = (128 * MAX_ROWS + 128).next_multiple_of(256);

const _: () = assert!(TWord::Fault.offset() + 8 <= TIER_PAYLOAD_OFF);

/// One row's tier image, in words from its start: the tier's places of the
/// routed slots (`n_used`), the stage card's q8_1 activation codes
/// (`q3_words` u32: the u64 codes as word pairs) and its block scales
/// (`d8` f32), each field at a 256-byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierImageLayout {
    pub sel: usize,
    pub q3: usize,
    pub d8: usize,
    pub n_used: usize,
    /// u32 words of the q8_1 codes: twice the activation's u64 codes.
    pub q3_words: usize,
    pub d8_len: usize,
    pub hidden: usize,
}

impl TierImageLayout {
    /// Words of the image.
    #[must_use]
    pub fn words(&self) -> usize {
        self.d8 + self.d8_len
    }
}

/// The tier page's layout: the flag words, then per row its image, then per
/// row its routed rows (`n_used · hidden` f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierLayout {
    rows: usize,
    image: TierImageLayout,
    image_stride: usize,
    rows_stride: usize,
}

impl TierLayout {
    /// The layout of `shape`'s rows over an activation whose q8_1 codes are
    /// `q3_u64` u64 and its scales `d8_len` f32; a size past `usize` or a
    /// row count outside `1..=MAX_ROWS` is refused by name.
    pub fn new(shape: TierShape, q3_u64: usize, d8_len: usize) -> Result<TierLayout, GpuError> {
        const WHAT: &str = "TierLayout::new";
        let TierShape {
            hidden,
            n_used,
            rows,
        } = shape;
        if !(1..=MAX_ROWS).contains(&rows) || hidden == 0 || n_used == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("{rows} rows of {hidden} values, {n_used} slots"),
            ));
        }
        let o = || GpuError::shape(WHAT, "the page's size passes usize");
        let q3_words = q3_u64.checked_mul(2).ok_or_else(o)?;
        let q3 = n_used.next_multiple_of(64);
        let d8 = q3
            .checked_add(q3_words)
            .map(|w| w.next_multiple_of(64))
            .ok_or_else(o)?;
        let image = TierImageLayout {
            sel: 0,
            q3,
            d8,
            n_used,
            q3_words,
            d8_len,
            hidden,
        };
        let image_stride = image
            .words()
            .checked_mul(4)
            .map(|b| b.next_multiple_of(256))
            .ok_or_else(o)?;
        let rows_stride = n_used
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(4))
            .map(|b| b.next_multiple_of(256))
            .ok_or_else(o)?;
        let l = TierLayout {
            rows,
            image,
            image_stride,
            rows_stride,
        };
        l.bytes().ok_or_else(o)?;
        Ok(l)
    }

    /// One row's image layout.
    #[must_use]
    pub fn image(&self) -> TierImageLayout {
        self.image
    }

    /// Rows the page carries.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Byte offset of row `row`'s image.
    fn image_off(&self, row: usize) -> usize {
        TIER_PAYLOAD_OFF + row * self.image_stride
    }

    /// Byte offset of row `row`'s routed rows.
    fn rows_off(&self, row: usize) -> usize {
        TIER_PAYLOAD_OFF + self.rows * self.image_stride + row * self.rows_stride
    }

    /// Bytes the page holds.
    fn bytes(&self) -> Option<usize> {
        self.rows
            .checked_mul(self.image_stride + self.rows_stride)?
            .checked_add(TIER_PAYLOAD_OFF)
    }
}

/// The shapes a tier is cut for: the model width, the routed slots a token,
/// the rows (tokens) in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierShape {
    pub hidden: usize,
    pub n_used: usize,
    pub rows: usize,
}

// ------------------------------------------------------------ the kernels

#[cuda_module]
mod tier_kernels {
    use super::*;

    /// The tier's fault word as it stands, and the site mask of the layer
    /// it names, into `out[0]` and `out[1]`: one thread, after every kernel
    /// of the tier's layer before it.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), requires = (out.len() >= 2))]
    pub fn tier_fault_copy(fault: FaultSink, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            let word = fault.read();
            let sites = fault.read_sites(word);
            // SAFETY: out.len() >= 2 by the launch contract; only thread 0
            // writes.
            unsafe {
                *out.get_unchecked_mut(0) = word;
                *out.get_unchecked_mut(1) = sites;
            }
        }
    }
}

// ------------------------------------------------------------- the card

/// What the tier's layer computes, supplied by the architecture: over the
/// staged activation `act` (the stage card's q8_1 codes and scales of the
/// normed activation, one column), each routed slot whose place in `sel`
/// (`n_used` places, a tier slot or [`super::slots::HOST`]) is a tier slot
/// runs its expert and writes its down output over `rows[j · hidden ..]`
/// for its routed slot `j`; no other row is written.
pub struct TierIo<'a> {
    pub act: &'a Q8Act,
    pub sel: &'a DeviceBuffer<u32>,
    pub rows: &'a mut DeviceBuffer<f32>,
}

/// The architecture's tier computation: layer `layer`'s tier experts over
/// `io` from `weights`, the tier's resident stacks, enqueued on `gpu`'s
/// stream. Asynchronous, allocation-free, capturable; a layer the tier
/// holds no expert of is refused by name. The protocol around the call —
/// the go, the staging, the fault copy, the signal — is [`TierCard`]'s.
pub trait TierExperts {
    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError>;

    /// Layer `layer`'s tier experts over the block `io` of a prompt batch,
    /// from `weights`, on `gpu`'s stream, by the same tile path the stage
    /// card runs for its card experts over a block. Eager,
    /// allocation-free; a layer the tier holds no expert of, and a block
    /// past the scratch the architecture made at load, are refused by name.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError>;

    /// Device bytes of the scratch [`TierExperts::enqueue_block`] made at
    /// load: part of the tier card's prompt-batch reserve.
    fn block_bytes(&self) -> usize;
}

/// A block of a prompt batch as the tier's batch service hands it to the
/// architecture ([`TierExperts::enqueue_block`]): the q8_1 form of its
/// `cols` token columns from column 0, each of its `n_used · cols` slots'
/// tier place (a tier slot, or [`super::slots::HOST`] for a slot the tier
/// does not compute), and the down outputs, slot-major (`hidden` a slot), of
/// which the tier's slots' rows are written and no other.
pub struct TierBlock<'a> {
    pub act: &'a Q8Act,
    pub sel: &'a DeviceBuffer<u32>,
    pub cols: usize,
    pub down: &'a mut DeviceBuffer<f32>,
}

/// What the stage card's handoff writes into a row's tier image and its
/// join reads back ([`TierCard::target_of`]): the image, the routed rows the
/// tier writes, as windows of the stage card's context, and the layout.
pub struct TierTarget<'a> {
    pub image: &'a mut DeviceBuffer<u32>,
    pub rows: &'a DeviceBuffer<f32>,
    pub layout: TierImageLayout,
}

/// What the tier has done since load, counted by the decode thread.
#[derive(Clone, Debug, Default)]
pub struct TierStats {
    /// Tier layers the host asked of the tier (replayed or enqueued).
    pub issued: u64,
    /// Routed slots the host saw go to the tier, per layer of the set, in
    /// layer order: the tier's hits.
    pub layer_hits: Vec<u64>,
    /// Settles: the host's waits for the tier after its last service of a
    /// replay, or after each eager layer, and their wall time (ns), summed.
    pub settles: u64,
    pub settle_ns: u64,
    /// Settles whose tier had already signalled when the host began to wait.
    pub settle_early: u64,
}

/// One row's windows over the tier page.
struct TierRow {
    /// The image and the routed rows, as the stage card's launches address
    /// them (the stage card's context).
    image: ManuallyDrop<DeviceBuffer<u32>>,
    rows_stage: ManuallyDrop<DeviceBuffer<f32>>,
    /// The routed rows as the tier's down writes them (the tier's context).
    rows_tier: ManuallyDrop<DeviceBuffer<f32>>,
}

impl Drop for TierRow {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — the tier's page frees its allocation after the rows.
        unsafe {
            drop(ManuallyDrop::take(&mut self.image).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.rows_stage).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.rows_tier).into_raw_parts());
        }
    }
}

/// The stage card as the tier reaches it: its stream (a fault is merged
/// once it drained) and its fault word.
struct Stage {
    stream: Arc<CudaStream>,
    fault: Arc<DeviceBuffer<u32>>,
}

/// The expert tier's card: its `Gpu` (its own stream and fault word), its
/// resident stacks, its page, one captured graph per chain kind, and the
/// architecture's computation. Owned by the host tier
/// ([`super::HostTier::attach_tier`]).
///
/// Field order is drop order: the graphs before the buffers they address,
/// the windows before the page, the page and the weights before the card.
pub struct TierCard {
    graphs: [Option<Graph>; 2],
    /// Per chain kind, the (layer, row) services its graph was captured
    /// for, in go order.
    captured: [Vec<(usize, usize)>; 2],
    experts: Box<dyn TierExperts>,
    module: tier_kernels::LoadedModule,
    /// The staged activation and places a layer's kernels read, copied in
    /// from the row's image.
    act: Q8Act,
    sel: DeviceBuffer<u32>,
    rows: Vec<TierRow>,
    page: MappedHost,
    layout: TierLayout,
    set: TierSet,
    weights: Weights,
    stage: Option<Stage>,
    stats: TierStats,
    /// Layers the host asked of the tier since load (wrapping): what the
    /// progress word reaches once the tier has served them.
    issued: u32,
    name: String,
    gpu: Gpu,
}

impl TierCard {
    /// The tier on `gpu` (the card named `name`), with `weights` its
    /// resident routed stacks for `set`, `experts` the architecture's
    /// computation, cut for `shape`. Load-time only. Its stage-side windows
    /// are made when the host tier takes it ([`super::HostTier::attach_tier`]).
    pub fn open(
        gpu: Gpu,
        name: String,
        weights: Weights,
        set: TierSet,
        experts: Box<dyn TierExperts>,
        shape: TierShape,
    ) -> Result<TierCard, GpuError> {
        let ctx = Arc::clone(gpu.context());
        ctx.bind_to_thread()?;
        let stream = gpu.stream();
        let act = Q8Act::with_k(stream, 1, shape.hidden)?;
        let layout = TierLayout::new(shape, act.q3.len(), act.d8.len())?;
        let bytes = layout
            .bytes()
            .ok_or_else(|| GpuError::shape(WHAT, "the page's size passes usize"))?;
        let page = MappedHost::new(&ctx, bytes, "cuMemHostAlloc (tier page)")?;
        // SAFETY: the fault copy's two words lie in the page's flag lines.
        unsafe {
            page.host_at(TWord::Fault.offset())
                .cast::<u32>()
                .write_volatile(FAULT_NONE);
        }
        let img = layout.image();
        let rows = (0..shape.rows)
            .map(|r| {
                // SAFETY: row r's image (`img.words()` words) and its routed
                // rows (`n_used · hidden` f32) lie inside the page, apart from
                // each other and from every other row's (the layout's
                // strides); the page moves into the card beside the windows
                // and is freed only after them.
                unsafe {
                    TierRow {
                        image: window::<u32>(page.dev_at(layout.image_off(r)), img.words(), &ctx),
                        rows_stage: window::<f32>(
                            page.dev_at(layout.rows_off(r)),
                            img.n_used * img.hidden,
                            &ctx,
                        ),
                        rows_tier: window::<f32>(
                            page.dev_at(layout.rows_off(r)),
                            img.n_used * img.hidden,
                            &ctx,
                        ),
                    }
                }
            })
            .collect();
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; its launcher checks the launch contract.
        let module = unsafe { tier_kernels::load(&ctx)? };
        let layer_hits = vec![0; set.layers().len()];
        Ok(TierCard {
            graphs: [None, None],
            captured: [Vec::new(), Vec::new()],
            experts,
            module,
            sel: DeviceBuffer::zeroed(stream, shape.n_used)?,
            act,
            rows,
            page,
            layout,
            set,
            weights,
            stage: None,
            stats: TierStats {
                layer_hits,
                ..TierStats::default()
            },
            issued: 0,
            name,
            gpu,
        })
    }

    /// The tier's card.
    #[must_use]
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The card's name, as the plan gives it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Device bytes of the architecture's prompt-batch block scratch on the
    /// tier card ([`TierExperts::block_bytes`]).
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        self.experts.block_bytes()
    }

    /// Which experts the tier holds.
    #[must_use]
    pub fn set(&self) -> &TierSet {
        &self.set
    }

    /// The tier's resident stacks.
    #[must_use]
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The page's layout.
    #[must_use]
    pub fn layout(&self) -> TierLayout {
        self.layout
    }

    /// What the tier has done since load.
    #[must_use]
    pub fn stats(&self) -> TierStats {
        self.stats.clone()
    }

    /// Device bytes the tier holds besides its weights: the staging.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.act.device_bytes() + self.sel.num_bytes()
    }

    /// Nodes per tier layer of a captured tier graph: the go wait, three
    /// copies in, the architecture's launches (`launches`) and the signal;
    /// the graph's last layer adds the fault copy.
    #[must_use]
    pub const fn nodes_per_layer(launches: usize) -> usize {
        1 + 3 + launches + 1
    }

    /// Row `row`'s image and routed rows for the stage card's handoff and
    /// join; a row the page does not carry is refused by name.
    pub fn target_of(&mut self, row: usize) -> Result<TierTarget<'_>, GpuError> {
        let layout = self.layout.image();
        let n = self.rows.len();
        let r = self.rows.get_mut(row).ok_or_else(|| {
            GpuError::shape(WHAT, format!("row {row} of a tier page of {n} rows"))
        })?;
        Ok(TierTarget {
            image: &mut r.image,
            rows: &r.rows_stage,
            layout,
        })
    }

    /// Nodes of `chain`'s captured tier graph; `None` before its first
    /// replay captured it, or for a chain the tier does not serve.
    #[must_use]
    pub fn graph_nodes(&self, chain: Chain) -> Option<usize> {
        let i = chain_slot(chain).ok()?;
        self.graphs[i].as_ref().map(Graph::node_count)
    }

    /// Row `row`'s routed rows as the stage card's join reads them; a row
    /// the page does not carry is refused by name.
    pub fn rows_of(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        let n = self.rows.len();
        self.rows
            .get(row)
            .map(|r| &*r.rows_stage)
            .ok_or_else(|| GpuError::shape(WHAT, format!("row {row} of a tier page of {n} rows")))
    }

    /// The tier's fault copy as the page holds it: what the tier's last
    /// fault copy saw.
    #[must_use]
    pub fn fault_copy(&self) -> Option<Fault> {
        let (w, s) = (
            self.word(TWord::Fault).load(Ordering::Acquire),
            self.word_at(TWord::Fault.offset() + 4)
                .load(Ordering::Acquire),
        );
        Fault::from_words(w, s)
    }

    /// Enqueue layer `layer`'s tier experts over the block `io` of a prompt
    /// batch on the tier's stream ([`TierExperts::enqueue_block`]), the
    /// tier's context bound; a layer the set holds no expert of is refused
    /// by name. The caller binds its own context again afterwards.
    pub(super) fn enqueue_block(
        &mut self,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        if self.set.on_tier(layer)? == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer}: a tier block of a layer the tier holds no expert of"),
            ));
        }
        self.gpu.context().bind_to_thread()?;
        self.experts
            .enqueue_block(&self.gpu, &self.weights, layer, io)
    }

    /// Make the stage-side windows in `stage`'s context and keep its
    /// stream and fault word. The windows and the stage card's waits on the
    /// tier's counters take the page's device address from the tier's
    /// context, so the page must sit at that address in `stage`'s context
    /// too (unified addressing), else the call is refused by name. The
    /// host tier's load-time call; `stage`'s context is current on return.
    pub(super) fn bind_stage(&mut self, stage: &Gpu) -> Result<(), GpuError> {
        const BIND: &str = "TierCard::bind_stage";
        let ctx = stage.context();
        ctx.bind_to_thread()?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `stage`'s context is current on this thread (bound above),
        // `dev` is a live local the call writes, and the page's first byte is
        // the start of its live mapped allocation; the flags must be 0.
        let rc =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, self.page.host_at(0).cast(), 0) };
        cu(
            rc,
            "cuMemHostGetDevicePointer_v2 (tier page, stage context)",
        )?;
        if dev != self.page.dev_at(0) {
            return Err(GpuError::shape(
                BIND,
                format!(
                    "the tier page is at {:#x} in the tier's context and at {dev:#x} in the stage \
                     card's: its windows and the stage's waits need one address",
                    self.page.dev_at(0)
                ),
            ));
        }
        let img = self.layout.image();
        for (r, row) in self.rows.iter_mut().enumerate() {
            // SAFETY: as in `open`: row r's image and routed rows lie inside
            // the page, which outlives the windows; the windows they replace
            // are given back first, their raw parts dropped and nothing freed.
            unsafe {
                drop(ManuallyDrop::take(&mut row.image).into_raw_parts());
                drop(ManuallyDrop::take(&mut row.rows_stage).into_raw_parts());
                row.image =
                    window::<u32>(self.page.dev_at(self.layout.image_off(r)), img.words(), ctx);
                row.rows_stage = window::<f32>(
                    self.page.dev_at(self.layout.rows_off(r)),
                    img.n_used * img.hidden,
                    ctx,
                );
            }
        }
        self.stage = Some(Stage {
            stream: stage.stream_handle(),
            fault: Arc::clone(stage.fault_word()),
        });
        Ok(())
    }

    /// Enqueue on the stage card's `stream` the go of layer `layer` of row
    /// `row` for a tier layer: the host tier's go batch
    /// ([`Boundary::enqueue_go_of`]) with one added to the row's tier go
    /// counter after the first system barrier, so the tier, like the host,
    /// starts only once the image has landed. One batch.
    pub(super) fn enqueue_go_of(
        &self,
        stream: &CudaStream,
        boundary: &Boundary,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let what = "TierCard::enqueue_go";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        self.check_row(row, what)?;
        let (lyr, gen_, seq) = boundary.go_words(row, what)?;
        mem_batch(
            stream,
            &mut [
                op_barrier_sys(),
                crate::graph::op_write(lyr, layer),
                op_add(self.page.dev_at(TWord::Go(row).offset()), 1),
                op_barrier_sys(),
                op_add(gen_, 1),
                op_add(seq, 1),
            ],
            "cuStreamBatchMemOp_v2 (hybrid go, tier)",
        )
    }

    /// Enqueue on the stage card's `stream` the wait of row `row` for a
    /// tier layer: until the host's counter and the tier's are each at
    /// least one, then minus one from each. One batch.
    pub(super) fn enqueue_back_of(
        &self,
        stream: &CudaStream,
        boundary: &Boundary,
        row: usize,
    ) -> Result<(), GpuError> {
        let what = "TierCard::enqueue_back";
        self.check_row(row, what)?;
        let host = boundary.cnt_word(row, what)?;
        let tier = self.page.dev_at(TWord::Cnt(row).offset());
        mem_batch(
            stream,
            &mut [
                op_wait_geq(host, 1),
                op_wait_geq(tier, 1),
                op_add(host, u32::MAX),
                op_add(tier, u32::MAX),
            ],
            "cuStreamBatchMemOp_v2 (hybrid wait, tier)",
        )
    }

    fn check_row(&self, row: usize, what: &'static str) -> Result<(), GpuError> {
        if row < self.rows.len() {
            Ok(())
        } else {
            Err(GpuError::shape(
                what,
                format!("row {row} of a tier page of {} rows", self.rows.len()),
            ))
        }
    }

    /// Launch `chain`'s tier graph for the services `list` (the stage
    /// capture's tier layers, in go order) on the tier's stream when one is
    /// held for that list ([`Pass::Graph`]); the host then expects the tier to
    /// serve `list.len()` layers more. With none held nothing is enqueued and
    /// the host feeds the pass a layer at a time ([`Pass::Feed`]), then
    /// captures the graph once it has settled ([`TierCard::capture`]): an
    /// instantiation may wait for the card, and the stage card's launched
    /// chain waits for the host, so none is made while that chain runs.
    pub(super) fn replay(
        &mut self,
        chain: Chain,
        list: &[(usize, usize)],
    ) -> Result<Pass, GpuError> {
        let i = chain_slot(chain)?;
        if list.is_empty() {
            return Ok(Pass::Graph);
        }
        let Some(graph) = self.graphs[i].as_ref().filter(|_| self.captured[i] == list) else {
            return Ok(Pass::Feed);
        };
        self.gpu.context().bind_to_thread()?;
        graph.launch(self.gpu.stream())?;
        self.issue(list.len());
        self.rebind_stage()?;
        Ok(Pass::Graph)
    }

    /// Capture `chain`'s tier graph for `list`, replacing the one held, after
    /// a fed pass has settled: the stage card's chain waits on neither the
    /// host nor the tier any more. Nothing is launched.
    pub(super) fn capture(
        &mut self,
        chain: Chain,
        list: &[(usize, usize)],
    ) -> Result<(), GpuError> {
        let i = chain_slot(chain)?;
        self.graphs[i] = None;
        self.captured[i].clear();
        let Some(last) = list.len().checked_sub(1) else {
            return Ok(());
        };
        self.gpu.context().bind_to_thread()?;
        let TierCard {
            experts,
            module,
            act,
            sel,
            rows,
            page,
            layout,
            weights,
            gpu,
            ..
        } = self;
        let mut parts = Parts {
            experts: &mut **experts,
            module,
            act,
            sel,
            rows,
            page,
            layout: *layout,
            weights,
            gpu,
        };
        let graph = Graph::capture(parts.gpu.stream(), |_| {
            for (k, &(layer, row)) in list.iter().enumerate() {
                parts.enqueue_layer(layer, row, k == last)?;
            }
            Ok(())
        })?;
        self.graphs[i] = Some(graph);
        self.captured[i] = list.to_vec();
        self.rebind_stage()
    }

    /// Enqueue layer `layer` of row `row` on the tier's stream now, for an
    /// eager chain or a fed pass ([`Pass::Feed`]), its fault copy with it;
    /// the host then expects one layer more.
    pub(super) fn enqueue_eager(&mut self, layer: usize, row: usize) -> Result<(), GpuError> {
        self.gpu.context().bind_to_thread()?;
        let TierCard {
            experts,
            module,
            act,
            sel,
            rows,
            page,
            layout,
            weights,
            gpu,
            ..
        } = self;
        Parts {
            experts: &mut **experts,
            module,
            act,
            sel,
            rows,
            page,
            layout: *layout,
            weights,
            gpu,
        }
        .enqueue_layer(layer, row, true)?;
        self.issue(1);
        self.rebind_stage()
    }

    /// Make the stage card's context current again after tier work, so the
    /// stage card's next launch is enqueued in its own.
    fn rebind_stage(&self) -> Result<(), GpuError> {
        if let Some(s) = &self.stage {
            s.stream.context().bind_to_thread()?;
        }
        Ok(())
    }

    fn issue(&mut self, n: usize) {
        let n32 =
            u32::try_from(n).expect("a pass lists at most two rows a layer, far below u32::MAX");
        self.issued = self.issued.wrapping_add(n32);
        self.stats.issued += n as u64;
    }

    /// Count a routed slot of layer `layer` the host saw go to the tier.
    pub(super) fn hit(&mut self, layer: usize, slots: u64) {
        if let Some(h) = layer
            .checked_sub(self.set.layers().start)
            .and_then(|i| self.stats.layer_hits.get_mut(i))
        {
            *h += slots;
        }
    }

    /// The layers the tier has served since load, as its progress word says.
    #[must_use]
    pub fn progress(&self) -> u32 {
        self.word(TWord::Prog).load(Ordering::Acquire)
    }

    /// Layers the host asked of the tier since load.
    #[must_use]
    pub fn issued(&self) -> u32 {
        self.issued
    }

    /// Whether the tier has served fewer than `want` layers.
    pub(super) fn behind_of(&self, want: u32) -> bool {
        let p = self.progress();
        p != want && want.wrapping_sub(p) < 1 << 31
    }

    /// Whether the tier has served fewer layers than the host asked of it.
    #[must_use]
    pub fn behind(&self) -> bool {
        self.behind_of(self.issued)
    }

    /// Wait until the tier has served every layer the host asked of it, or
    /// `deadline` has passed; `true` when it has.
    pub(super) fn wait_caught_up(&mut self, deadline: Instant) -> bool {
        let t0 = Instant::now();
        let early = !self.behind();
        if !self.wait_for(self.issued, deadline) {
            return false;
        }
        let s = &mut self.stats;
        s.settles += 1;
        s.settle_early += u64::from(early);
        s.settle_ns += u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        true
    }

    /// Wait until the tier has served `want` layers since load, or
    /// `deadline` has passed; `true` when it has.
    pub(super) fn wait_for(&self, want: u32, deadline: Instant) -> bool {
        let mut spins = 0u32;
        while self.behind_of(want) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(DEADLINE_POLL) && Instant::now() > deadline {
                return false;
            }
            std::hint::spin_loop();
        }
        true
    }

    /// The loss of the tier as an error's detail: how far it got, and how
    /// long the host `waited` for it.
    pub(super) fn lost_detail(&self, waited: Duration) -> String {
        format!(
            "the expert tier on {} served {} of the {} layers asked of it and signalled nothing more in {waited:.1?}: the card is lost",
            self.name,
            self.progress(),
            self.issued
        )
    }

    /// The step's fault once the tier raised one: its copy merged with the
    /// stage card's word, read once the stage card's stream drained — the
    /// first layer wins. `None` while the copy is clean.
    pub(super) fn merged_fault(&self) -> Result<Option<Fault>, GpuError> {
        let Some(tier) = self.fault_copy() else {
            return Ok(None);
        };
        let stage = match &self.stage {
            Some(s) => {
                s.stream.synchronize()?;
                crate::fault::read(&s.fault)?
            }
            None => None,
        };
        Ok(crate::fault::read_cards(&[stage, Some(tier)]))
    }

    /// Release every wait of both cards that is still pending, for good:
    /// the tier's counters and go counters go far past anything a step
    /// subtracts.
    pub(super) fn release(&self) {
        for r in 0..self.rows.len() {
            self.word(TWord::Go(r)).store(RELEASE, Ordering::Release);
            self.word(TWord::Cnt(r)).store(RELEASE, Ordering::Release);
        }
    }

    /// Whether the tier's words hold what a drained step leaves: no go and
    /// no signal pending, every layer asked of it served.
    #[must_use]
    pub fn at_rest(&self) -> bool {
        (0..self.rows.len()).all(|r| {
            self.word(TWord::Go(r)).load(Ordering::Acquire) == 0
                && self.word(TWord::Cnt(r)).load(Ordering::Acquire) == 0
        }) && !self.behind()
    }

    /// Back to a fresh tier's words once the tier's stream has drained:
    /// counters and go counters at 0, the host's count at the tier's
    /// progress, the tier's fault word and its copy clean. The host tier's
    /// reset calls it.
    pub(super) fn reset(&mut self) -> Result<(), GpuError> {
        self.gpu.stream().synchronize()?;
        for r in 0..self.rows.len() {
            self.word(TWord::Go(r)).store(0, Ordering::Release);
            self.word(TWord::Cnt(r)).store(0, Ordering::Release);
        }
        self.issued = self.progress();
        self.clear_fault()
    }

    /// The tier's fault word and its copy back to clean.
    pub(super) fn clear_fault(&mut self) -> Result<(), GpuError> {
        self.gpu.clear_fault()?;
        self.gpu.stream().synchronize()?;
        self.word(TWord::Fault).store(FAULT_NONE, Ordering::Release);
        self.word_at(TWord::Fault.offset() + 4)
            .store(0, Ordering::Release);
        self.rebind_stage()
    }

    /// Forget the go order the graph of `chain` was captured for: its stage
    /// capture is being made again, so the next pass is fed and the tier's
    /// graph captured anew after it. The graph itself goes at that capture,
    /// outside the stage's: destroying one while this thread captures
    /// invalidates that capture.
    pub(super) fn forget_order(&mut self, chain: Chain) {
        if let Ok(i) = chain_slot(chain) {
            self.captured[i].clear();
        }
    }

    fn word(&self, w: TWord) -> &AtomicU32 {
        self.word_at(w.offset())
    }

    fn word_at(&self, off: usize) -> &AtomicU32 {
        self.page
            .atomic_u32(off)
            .expect("every tier flag word lies in the page's first TIER_PAYLOAD_OFF bytes")
    }
}

/// How a replay of a captured chain reaches the tier
/// ([`TierCard::replay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pass {
    /// The held graph was launched for the whole pass, or it has no tier
    /// layer.
    Graph,
    /// No graph is held for the pass: the host enqueues each tier layer as
    /// it serves it ([`TierCard::enqueue_eager`]) and captures the graph once
    /// the pass has settled ([`TierCard::capture`]).
    Feed,
}

/// The tier graph's slot of `chain`: the step and the pair; a chain of
/// several columns is refused by name (the tier serves one column a go).
fn chain_slot(chain: Chain) -> Result<usize, GpuError> {
    match chain {
        Chain::Step => Ok(0),
        Chain::Pair => Ok(1),
        Chain::Cols(m) => Err(GpuError::shape(
            WHAT,
            format!("a chain of {m} columns: the tier serves the step and the pair"),
        )),
    }
}

/// The card's parts one tier layer's enqueue reads and writes.
struct Parts<'a> {
    experts: &'a mut dyn TierExperts,
    module: &'a tier_kernels::LoadedModule,
    act: &'a mut Q8Act,
    sel: &'a mut DeviceBuffer<u32>,
    rows: &'a mut [TierRow],
    page: &'a MappedHost,
    layout: TierLayout,
    weights: &'a Weights,
    gpu: &'a Gpu,
}

impl Parts<'_> {
    /// Layer `layer` of row `row` on the tier's stream: wait for the row's
    /// go and take it back, copy the row's image — the q8_1 codes, their
    /// scales, the places — into the staging, the architecture's launches
    /// into the row's routed rows, then (when `copy_fault`) the fault copy,
    /// a system barrier and the row's counter and the progress word each
    /// plus one.
    fn enqueue_layer(
        &mut self,
        layer: usize,
        row: usize,
        copy_fault: bool,
    ) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        let page = self.page;
        let img = self.layout.image();
        if row >= self.rows.len() {
            return Err(GpuError::shape(
                WHAT,
                format!("row {row} of a tier page of {} rows", self.rows.len()),
            ));
        }
        let go = page.dev_at(TWord::Go(row).offset());
        mem_batch(
            stream,
            &mut [op_wait_geq(go, 1), op_add(go, u32::MAX)],
            "cuStreamBatchMemOp_v2 (tier go)",
        )?;
        let at = self.layout.image_off(row);
        if self.act.q3.len() * 2 != img.q3_words
            || self.act.d8.len() != img.d8_len
            || self.sel.len() != img.n_used
        {
            return Err(GpuError::shape(
                WHAT,
                "the staging is not the image's shape".to_string(),
            ));
        }
        copy_in(
            stream,
            self.act.q3.cu_deviceptr(),
            page.dev_at(at + 4 * img.q3),
            4 * img.q3_words,
        )?;
        copy_in(
            stream,
            self.act.d8.cu_deviceptr(),
            page.dev_at(at + 4 * img.d8),
            4 * img.d8_len,
        )?;
        copy_in(
            stream,
            self.sel.cu_deviceptr(),
            page.dev_at(at + 4 * img.sel),
            4 * img.n_used,
        )?;
        let r = &mut self.rows[row];
        self.experts.enqueue_layer(
            self.gpu,
            self.weights,
            layer,
            TierIo {
                act: self.act,
                sel: self.sel,
                rows: &mut r.rows_tier,
            },
        )?;
        if copy_fault {
            let ctx = self.gpu.context();
            // SAFETY: the fault copy's two words lie in the page's flag
            // lines; the window is used for this launch only and never
            // dropped as a buffer (its raw parts are, below).
            let mut out = unsafe { window::<u32>(page.dev_at(TWord::Fault.offset()), 2, ctx) };
            let prep = self
                .module
                .prepare_tier_fault_copy(LaunchConfig1D::new(1, 32, 0))?;
            let r =
                self.module
                    .tier_fault_copy(stream, &prep, self.gpu.unlabelled_sink(), &mut out);
            // SAFETY: the window is taken once, here; its raw parts are
            // dropped and nothing is freed.
            unsafe { drop(ManuallyDrop::take(&mut out).into_raw_parts()) };
            r?;
        }
        mem_batch(
            stream,
            &mut [
                op_barrier_sys(),
                op_add(page.dev_at(TWord::Cnt(row).offset()), 1),
                op_add(page.dev_at(TWord::Prog.offset()), 1),
            ],
            "cuStreamBatchMemOp_v2 (tier signal)",
        )
    }
}

/// Copy `bytes` from `src` to `dst` on `stream`: one copy node when
/// captured.
fn copy_in(
    stream: &CudaStream,
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    bytes: usize,
) -> Result<(), GpuError> {
    // SAFETY: both spans are the caller's: `dst` a device buffer of the
    // tier's context of at least `bytes`, `src` a span of the tier's page
    // inside its allocation; both outlive every graph that captured the copy.
    let rc = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
    cu(rc, "cuMemcpyDtoDAsync_v2 (tier stage-in)")
}

#[cfg(test)]
mod tests {
    use super::{TIER_PAYLOAD_OFF, TWord, TierLayout, TierSet, TierShape};
    use crate::host::slots::{HOST, SlotMap, TIER};

    /// A set's rows are ascending ids below the stack, each once; a map's
    /// set is its tier entries in tier-slot order, and a layer outside it is
    /// refused.
    #[test]
    fn a_set_is_the_maps_tier_rows() {
        let rows = vec![0, HOST, TIER, HOST, HOST, TIER, 0, TIER | 1];
        let map = SlotMap::from_rows(3..5, 4, rows).expect("two rows of 4");
        let set = TierSet::of_map(&map).expect("the map's set");
        assert_eq!(set.ids(3), Some(&[2][..]));
        assert_eq!(set.ids(4), Some(&[1, 3][..]));
        assert_eq!(set.on_tier(3).expect("layer 3"), 1);
        assert_eq!(set.on_tier(4).expect("layer 4"), 2);
        assert!(set.on_tier(5).is_err());
        assert_eq!(set.experts(), 3);
        let set = TierSet::new(3..5, 8, vec![vec![2, 5], vec![]]).expect("a set");
        assert_eq!(set.on_tier(4).expect("layer 4"), 0);
        assert!(TierSet::new(0..1, 8, vec![vec![3, 3]]).is_err());
        assert!(TierSet::new(0..1, 8, vec![vec![4, 2]]).is_err());
        assert!(TierSet::new(0..1, 8, vec![vec![8]]).is_err());
        assert!(TierSet::new(0..2, 8, vec![vec![1]]).is_err());
    }

    /// V4.1's tier page: two rows of six slots over 4096 values, the q8_1
    /// codes of 16 super-blocks (512 u64) and their 32 scales; every field
    /// apart and inside the page, the flag words each on a line of its own.
    #[test]
    fn v41_tier_page_keeps_its_fields_apart() {
        let shape = TierShape {
            hidden: 4096,
            n_used: 6,
            rows: 2,
        };
        let l = TierLayout::new(shape, 512, 32).expect("V4.1's tier page");
        let i = l.image();
        assert_eq!((i.sel, i.q3, i.d8), (0, 64, 64 + 1024));
        assert!(i.sel + i.n_used <= i.q3 && i.q3 + i.q3_words <= i.d8);
        assert!(l.image_off(1) >= l.image_off(0) + 4 * i.words());
        assert!(l.rows_off(0) >= l.image_off(1) + 4 * i.words());
        assert!(l.rows_off(1) >= l.rows_off(0) + 4 * 6 * 4096);
        assert!(l.bytes().unwrap() >= l.rows_off(1) + 4 * 6 * 4096);
        let mut offs = vec![TWord::Prog.offset(), TWord::Fault.offset()];
        for r in 0..2 {
            offs.extend([TWord::Go(r).offset(), TWord::Cnt(r).offset()]);
        }
        offs.sort_unstable();
        assert!(offs.windows(2).all(|w| w[1] - w[0] >= 64));
        assert!(offs.iter().all(|&o| o + 64 <= TIER_PAYLOAD_OFF));
        assert!(TierLayout::new(TierShape { rows: 3, ..shape }, 512, 32).is_err());
    }
}
