//! The expert tiers: further cards hung under the host tier ([`TierCard`]).
//! The stage card keeps its plan byte for byte; per hybrid layer each tier
//! card computes the routed experts its [`TierSet`] names, in the host leg's
//! shadow, and the stage card's join reads their rows through the mapping.
//!
//! Per tier layer, beside the host tier's protocol ([`super`]):
//!
//! ```text
//! stage: router → handoff (+ q8_1 x, tier places) → go (+ tier go) → [shadow] → wait (host, tier) → join
//! tier:  wait tier go, −1 → copy x, places in → [gate·up → h q8_1 → down] → copy rows out → barrier → cnt +1, prog +1
//! ```
//!
//! The tiers' words live in one page ([`TierPage`], laid out by
//! [`TierLayout`]) the host tier owns, which the stage card, each tier card
//! and the host reach over its link (the `link` module). Per tier it holds
//! that tier's flag lines: per row a go counter the stage card adds one to
//! and the tier takes back, and a counter the tier adds one to and the stage
//! card's wait takes back — the host tier's counter rule, one writer at a
//! time — then the progress word, which the tier adds one to per layer
//! served and the host reads, and the tier's fault copy. Per row it holds
//! the image the stage card's handoff writes — each tier's places of the
//! routed slots, then the stage card's q8_1 activation, which every tier
//! reads — and the rows the tiers' downs computed, one per routed slot, each
//! tier its own slots, which the stage card's join reads in place.
//!
//! A tier's launches write their rows into the tier card's own memory
//! ([`TierIo::rows`]), and one copy a layer moves the go's columns of them
//! to the page: the downs store a value a row, and a store of four bytes
//! into host-mapped memory crosses the bus as a transaction of its own,
//! which the copy engine's bulk write does not.
//!
//! A row may carry several columns ([`TierCard::open_cols`]): a
//! [`Chain::Cols`] go hands the tier `m` consecutive positions in one image,
//! `n_used` places and the activation a column, and the tier's rows hold
//! `n_used` slots a column; a tier opened for one column serves the step and
//! the pair alone.
//!
//! The host thread never drives a tier per layer: a captured chain's tier
//! work is one graph per tier and chain ([`Chain::Step`], [`Chain::Pair`],
//! each [`Chain::Cols`] width), captured from the stage capture's go order
//! at its first replay and launched on the tier's stream before the host
//! serves the replay. Its last layer copies the tier's fault word into the page before
//! its signal. After the host has served the replay's last layer it waits for
//! each tier's progress under the go deadline and reads the copies: a raised
//! word is the step's fault, merged with the stage card's
//! ([`crate::fault::read_cards`], the first layer wins). An eager chain
//! enqueues each tier layer as the host serves it and settles it after, the
//! copy taken each layer.
//!
//! A tier that stops signalling is a lost card: the host's go deadline, or
//! the settle's, names each tier whose progress is behind, the host tier is
//! poisoned as [`super::PoisonKind::CardLost`], which a reset does not lift,
//! and the release writes [`RELEASE`] to every tier's counters and go
//! counters too, so every stream drains, unless an end of mirrored pages
//! cannot show its posts over ([`Link::release`]): its cards keep waiting,
//! the host's error names that end, and a synchronize of their streams does
//! not return.
//!
//! A tier's work on its card ([`TierWork`]: its stacks, staging, windows and
//! graphs) is apart from the stage side that drives it ([`TierCard`]: the
//! replay or feed decision, the books, the settle, the release and the
//! reset). The work runs in this process, or in a tier node's
//! ([`super::node`]), which runs the same steps when the stage side asks.

use std::mem::ManuallyDrop;
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use model::placement::Machine;
use model::placement::devices::DeviceInfo;

use super::link::{Crossing, Link, LinkEnd, Reader};
use super::node::TierNode;
use super::page::MAX_ROWS;
use super::slots::{MAX_TIERS, Slot, SlotMap};
use super::step::{Boundary, Chain, RELEASE};
use crate::fault::{FAULT_NONE, Fault};
use crate::graph::{Graph, cu, mem_batch, op_add, op_barrier_sys, op_wait_geq};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError, Q8Act};
use bloomery_levers::TierLink;
use model::ops::DEFER_MAX_COLS;

/// What the tier's errors name.
const WHAT: &str = "TierCard";

/// Spins between two clock reads while the host waits for the tier.
const DEADLINE_POLL: u32 = 1 << 12;

/// How long a live tier takes at most to finish the layer it is on: a host
/// service that failed while the tier was behind waits this long before it
/// names the tier lost.
pub(super) const TIER_GRACE: Duration = Duration::from_millis(100);

// ------------------------------------------------------------------ set

/// Which experts of each hybrid layer a tier card holds: per layer of
/// `layers`, its expert ids in tier-slot order (ascending, each once, each
/// below `n_expert`) — slot `s` of the tier's routed stacks holds `ids[s]`.
/// A layer may hold none; the tier then does nothing for it. A copy of the
/// slot map's rows of one tier ([`TierSet::of_map`]), made before the tier
/// card loads its stacks.
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

    /// Tier `tier`'s set as `map` holds it: per layer of the map, the ids of
    /// that tier's entries in its slot order. The map is the one owner of
    /// where an expert runs; the set is a copy of one tier's rows the tier
    /// card keeps. A tier the map does not name is refused by name.
    pub fn of_map(map: &SlotMap, tier: usize) -> Result<TierSet, GpuError> {
        const WHAT: &str = "TierSet::of_map";
        let layers = map.layers();
        let n = u32::try_from(map.n_expert())
            .map_err(|_| GpuError::shape(WHAT, "the map's experts pass u32"))?;
        let mut rows = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let k = map.on_tier_of(tier, l)?;
            let mut ids = vec![u32::MAX; k];
            for id in 0..n {
                if let Some(Slot::Tier { tier: t, slot: s }) = map.slot(l, id)
                    && t == tier
                {
                    let at = ids.get_mut(s as usize).ok_or_else(|| {
                        GpuError::shape(WHAT, format!("layer {l}: tier {tier} slot {s} of {k}"))
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

/// A tier card a placed load hangs under the host tier: the plan's device
/// index its routed segments sit on (`Device::Card(card)`), the card's name,
/// the device the plan resolved it to, by which it opens
/// ([`Gpu::open_card`]), and the node that serves it, when it runs on one
/// ([`super::node`]).
#[derive(Clone, Debug)]
pub struct TierOpen {
    pub card: usize,
    pub name: String,
    pub device: Option<model::placement::workstation::DeviceId>,
    pub node: Option<SocketAddr>,
}

impl TierOpen {
    /// The expert tier cards `machine` names, in tier order: each card's
    /// index in [`Machine::all_cards`] (after the stage cards), its name, its
    /// device, and the node `BLOOMERY_TIER_NODES` names for it, the lever's
    /// `t`-th address for tier `t`. Empty without a tier. A lever that names
    /// another count of nodes than the machine's tiers is refused by name.
    pub fn of_machine(machine: &Machine) -> Result<Vec<TierOpen>, GpuError> {
        const WHAT: &str = "TierOpen::of_machine";
        let nodes = bloomery_levers::tier_nodes().map_err(|e| GpuError::plan(WHAT, e))?;
        if !nodes.is_empty() && nodes.len() != machine.tiers.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} names {} tier nodes for a plan of {} expert tiers: one a tier",
                    bloomery_levers::TIER_NODES,
                    nodes.len(),
                    machine.tiers.len()
                ),
            ));
        }
        Ok(machine
            .tiers
            .iter()
            .enumerate()
            .map(|(i, t)| TierOpen {
                card: machine.cards.len() + i,
                name: t.name.clone(),
                device: t.device,
                node: nodes.get(i).copied(),
            })
            .collect())
    }

    /// The tier's card, opened in this process ([`Gpu::open_card`]), for a
    /// family whose tier opens only on a card of the loading process: a tier
    /// that runs on a node is refused by name, as `what`, before any card
    /// opens.
    pub fn open_here(&self, what: &'static str) -> Result<Gpu, GpuError> {
        if let Some(addr) = self.node {
            return Err(GpuError::shape(
                what,
                format!(
                    "the expert tier {} on the node at {addr}: this family's tier opens on a \
                     card of the loading process only",
                    self.name
                ),
            ));
        }
        Gpu::open_card(&self.name, self.device)
    }
}

// ------------------------------------------------------------------ page

/// One tier's flag words on the page, one 64-byte line each, from the
/// tier's first line ([`TierLayout::word_off`]).
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

/// Bytes of one tier's flag lines: tier `t`'s start at `t · TIER_FLAGS`.
const TIER_FLAGS: usize = 128 * MAX_ROWS + 128;

const _: () = assert!(TWord::Fault.offset() + 8 <= TIER_FLAGS);

/// One row's tier image as one tier reads it, in words from its start: that
/// tier's places of the routed slots (`n_used` a column for `cols` columns,
/// column-major, at `sel`), then the
/// activation in the form the architecture's [`TierAct`] names — the stage
/// card's q8_1 codes (`q3_words` u32 at `q3`: the u64 codes as word pairs)
/// and its block scales (`d8_len` f32 at `d8`), or the normed activation
/// itself (`x_len` f32 at `x`) — each field at a 256-byte boundary; the
/// fields of the other form are empty; each field holds the `cols` columns
/// one after another. Every tier's image of a row has the same activation
/// and its own `sel` ([`TierLayout::image`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierImageLayout {
    pub sel: usize,
    pub q3: usize,
    pub d8: usize,
    pub x: usize,
    pub n_used: usize,
    /// u32 words of the q8_1 codes: twice the activation's u64 codes.
    pub q3_words: usize,
    pub d8_len: usize,
    pub x_len: usize,
    pub hidden: usize,
    /// Columns the image carries: one but on a page of several columns a
    /// row ([`TierLayout::with_cols`]).
    pub cols: usize,
}

impl TierImageLayout {
    /// Words of the row's image, every tier's places with it.
    #[must_use]
    pub fn words(&self) -> usize {
        self.x + self.x_len
    }
}

/// The activation a row's tier image carries, the architecture's choice
/// ([`TierShape::act`]): what its tier kernels read decides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierAct {
    /// The stage card's q8_1 codes and block scales of the normed
    /// activation, which the tier's kernels read as staged.
    Q8,
    /// The normed activation in f32, which the architecture quantizes on
    /// the tier card itself, with the stage card's quantizer.
    F32,
}

impl TierAct {
    /// Copies from the page a tier layer stages: the activation's fields
    /// and the places.
    #[must_use]
    pub const fn copies(self) -> usize {
        match self {
            TierAct::Q8 => 3,
            TierAct::F32 => 2,
        }
    }
}

/// The tier page's layout over `tiers` tiers: each tier's flag lines, then
/// per row its image (each tier's places, the shared q8_1 activation), then
/// per row its routed rows (`n_used · cols · hidden` f32, slot-major), which
/// every tier writes its own slots of. With one tier every offset is the
/// one-tier page's; with one column a row every offset is the one-column
/// page's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierLayout {
    tiers: usize,
    rows: usize,
    /// Tier 0's image of a row; tier `t`'s has `sel` at `t · sel_stride`.
    image: TierImageLayout,
    sel_stride: usize,
    image_stride: usize,
    rows_stride: usize,
}

impl TierLayout {
    /// The layout of `shape`'s rows for `tiers` tiers over an activation
    /// whose q8_1 codes are `q3_u64` u64 and its scales `d8_len` f32 when
    /// `shape.act` is [`TierAct::Q8`], or its `hidden` f32 values when it is
    /// [`TierAct::F32`] (codes and scales then empty); a size past `usize`, a
    /// row count outside `1..=MAX_ROWS`, a tier count outside
    /// `1..=MAX_TIERS` and an f32 image given codes or scales are refused by
    /// name. One column a row.
    pub fn new(
        shape: TierShape,
        q3_u64: usize,
        d8_len: usize,
        tiers: usize,
    ) -> Result<TierLayout, GpuError> {
        TierLayout::with_cols(shape, 1, q3_u64, d8_len, tiers)
    }

    /// [`TierLayout::new`] of rows of `cols` columns (`1..=DEFER_MAX_COLS`,
    /// else refused by name): the places `n_used · cols`, the activation's
    /// codes and scales — `q3_u64` and `d8_len` are the whole staging's, its
    /// `cols` columns — or its `cols · hidden` f32, and the routed rows
    /// `n_used · cols · hidden` f32.
    pub fn with_cols(
        shape: TierShape,
        cols: usize,
        q3_u64: usize,
        d8_len: usize,
        tiers: usize,
    ) -> Result<TierLayout, GpuError> {
        const WHAT: &str = "TierLayout::new";
        let TierShape {
            hidden,
            n_used,
            rows,
            act,
        } = shape;
        if !(1..=MAX_ROWS).contains(&rows)
            || !(1..=MAX_TIERS).contains(&tiers)
            || !(1..=DEFER_MAX_COLS).contains(&cols)
            || hidden == 0
            || n_used == 0
            || (act == TierAct::F32 && (q3_u64, d8_len) != (0, 0))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{rows} rows of {cols} columns of {hidden} values, {n_used} slots, {tiers} \
                     tiers, an {act:?} image of {q3_u64} codes and {d8_len} scales"
                ),
            ));
        }
        let o = || GpuError::shape(WHAT, "the page's size passes usize");
        let q3_words = q3_u64.checked_mul(2).ok_or_else(o)?;
        let places = n_used.checked_mul(cols).ok_or_else(o)?;
        let sel_stride = places.next_multiple_of(64);
        let q3 = sel_stride.checked_mul(tiers).ok_or_else(o)?;
        let d8 = q3
            .checked_add(q3_words)
            .map(|w| w.next_multiple_of(64))
            .ok_or_else(o)?;
        let (x, x_len) = match act {
            TierAct::Q8 => (d8.checked_add(d8_len).ok_or_else(o)?, 0),
            TierAct::F32 => (d8, hidden.checked_mul(cols).ok_or_else(o)?),
        };
        let image = TierImageLayout {
            sel: 0,
            q3,
            d8,
            x,
            n_used,
            q3_words,
            d8_len,
            x_len,
            hidden,
            cols,
        };
        let image_stride = image
            .words()
            .checked_mul(4)
            .map(|b| b.next_multiple_of(256))
            .ok_or_else(o)?;
        let rows_stride = places
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(4))
            .map(|b| b.next_multiple_of(256))
            .ok_or_else(o)?;
        let l = TierLayout {
            tiers,
            rows,
            image,
            sel_stride,
            image_stride,
            rows_stride,
        };
        l.bytes().ok_or_else(o)?;
        Ok(l)
    }

    /// Tier `tier`'s image of a row: its own places, the shared activation.
    /// A tier the page does not carry is refused by name.
    pub fn image(&self, tier: usize) -> Result<TierImageLayout, GpuError> {
        self.check_tier(tier)?;
        Ok(TierImageLayout {
            sel: tier * self.sel_stride,
            ..self.image
        })
    }

    /// Bytes in one row's go and done posts, including their generation words.
    #[must_use]
    pub fn wire_bytes(&self) -> [usize; 2] {
        let crossings = self.crossings();
        std::array::from_fn(|i| {
            crossings[i].spans.iter().map(|s| s.len()).sum::<usize>() + std::mem::size_of::<u32>()
        })
    }

    /// Tiers the page carries.
    #[must_use]
    pub fn tiers(&self) -> usize {
        self.tiers
    }

    /// Rows the page carries.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    fn check_tier(&self, tier: usize) -> Result<(), GpuError> {
        if tier < self.tiers {
            Ok(())
        } else {
            Err(GpuError::shape(
                "TierLayout",
                format!("tier {tier} of a tier page of {} tiers", self.tiers),
            ))
        }
    }

    /// Byte offset of tier `tier`'s word `w`.
    const fn word_off(&self, tier: usize, w: TWord) -> usize {
        tier * TIER_FLAGS + w.offset()
    }

    /// Byte offset of every counter of tier `tier`, each row's go paired
    /// with its done: what a release, a rest check and a reset of the tier take.
    fn counters(self, tier: usize) -> impl Iterator<Item = usize> {
        (0..self.rows)
            .flat_map(|r| [TWord::Cnt(r), TWord::Go(r)])
            .map(move |w| self.word_off(tier, w))
    }

    /// Byte offset of row 0's image: past every tier's flag lines.
    const fn payload_off(&self) -> usize {
        (self.tiers * TIER_FLAGS).next_multiple_of(256)
    }

    /// Byte offset of row `row`'s image.
    fn image_off(&self, row: usize) -> usize {
        self.payload_off() + row * self.image_stride
    }

    /// Byte offset of row `row`'s routed rows.
    fn rows_off(&self, row: usize) -> usize {
        self.payload_off() + self.rows * self.image_stride + row * self.rows_stride
    }

    /// Every counter of the page and the bytes it publishes, as a link that
    /// mirrors the page carries them: per tier and row the go, which
    /// publishes the row's image, and the done, which publishes the row's
    /// routed rows and the tier's fault copy and counts in its progress word.
    fn crossings(&self) -> Vec<Crossing> {
        let image = 4 * self.image.words();
        let rows = 4 * self.image.n_used * self.image.cols * self.image.hidden;
        (0..self.tiers)
            .flat_map(|t| (0..self.rows).map(move |r| (t, r)))
            .flat_map(|(t, r)| {
                let fault = self.word_off(t, TWord::Fault);
                let go = self.image_off(r)..self.image_off(r) + image;
                [
                    Crossing {
                        counter: self.word_off(t, TWord::Go(r)),
                        reader: Reader::Tier,
                        spans: vec![go],
                        prog: None,
                    },
                    Crossing {
                        counter: self.word_off(t, TWord::Cnt(r)),
                        reader: Reader::Stage,
                        spans: vec![self.rows_off(r)..self.rows_off(r) + rows, fault..fault + 8],
                        prog: Some(self.word_off(t, TWord::Prog)),
                    },
                ]
            })
            .collect()
    }

    /// One tier's end, with its relay tail apart from every other tier's.
    fn remote_end(
        &self,
        link: &Link,
        index: usize,
        side: Reader,
        ip: std::net::IpAddr,
    ) -> Result<super::link::RdmaEnd, GpuError> {
        self.check_tier(index)?;
        let n = 2 * self.rows;
        let bytes = TierPage::bytes_of(*self, "the remote tier page")?;
        let (_, tail) = super::link::link_tail(bytes, index * n);
        let crossings = self
            .crossings()
            .into_iter()
            .skip(index * n)
            .take(n)
            .collect();
        link.rdma_end(tail, crossings, side, ip)
    }

    /// Bytes the page holds.
    fn bytes(&self) -> Option<usize> {
        self.rows
            .checked_mul(self.image_stride + self.rows_stride)?
            .checked_add(self.payload_off())
    }
}

/// The shapes a tier is cut for: the model width, the routed slots a token,
/// the rows in flight, and the activation its image carries; the columns a
/// row carries are [`TierCard::open_cols`]'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierShape {
    pub hidden: usize,
    pub n_used: usize,
    pub rows: usize,
    pub act: TierAct,
}

/// One row's windows over the page as the stage card's launches address
/// them (the stage card's context): the image its handoff writes, the routed
/// rows its join reads.
struct StageRow {
    image: ManuallyDrop<DeviceBuffer<u32>>,
    rows: ManuallyDrop<DeviceBuffer<f32>>,
}

impl Drop for StageRow {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — the page frees its allocation after the rows.
        unsafe {
            drop(ManuallyDrop::take(&mut self.image).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.rows).into_raw_parts());
        }
    }
}

/// The tier page: the bytes every tier card and the stage card reach over
/// the link ([`TierLayout`] lays them out), allocated in the stage card's
/// context, and the stage card's windows over it. The host tier owns it and
/// drops it after its tier cards, whose graphs and windows address it.
///
/// Field order is drop order: the windows before the link.
pub struct TierPage {
    stage: Vec<StageRow>,
    link: Link,
    layout: TierLayout,
}

impl TierPage {
    /// The page laid out by `layout` over a link of kind `kind`, in
    /// `stage`'s context (and on the loopback, the tier cards' side in
    /// `tier`'s), every tier's fault copy clean, with the stage card's
    /// windows. Load-time only; `stage`'s context is current on return.
    pub(super) fn new(
        stage: &Gpu,
        tier: &Gpu,
        layout: TierLayout,
        kind: TierLink,
    ) -> Result<TierPage, GpuError> {
        const NEW: &str = "TierPage::new";
        let bytes = Self::bytes_of(layout, NEW)?;
        let link = Link::new(
            kind,
            stage.context(),
            tier.context(),
            bytes,
            layout.crossings(),
            NEW,
        )?;
        Self::over(stage, layout, link)
    }

    /// The page laid out by `layout` for a load with a tier on a node: a
    /// shared page in a file the node's process opens ([`Link::across`]),
    /// every tier's fault copy clean, with the stage card's windows.
    /// Load-time only; `stage`'s context is current on return.
    pub(super) fn across(stage: &Gpu, layout: TierLayout) -> Result<TierPage, GpuError> {
        let bytes = Self::bytes_of(layout, "TierPage::across")?;
        let link = Link::across(
            stage.context(),
            bytes,
            layout.crossings().iter().map(|c| c.counter).collect(),
        )?;
        Self::over(stage, layout, link)
    }

    /// A page of this process alone, with one RDMA end per remote tier.
    pub(super) fn remote(stage: &Gpu, layout: TierLayout) -> Result<TierPage, GpuError> {
        let bytes = Self::bytes_of(layout, "TierPage::remote")?;
        let link = Link::remote(stage.context(), bytes, layout.crossings().len())?;
        Self::over(stage, layout, link)
    }

    /// The bytes a page laid out by `layout` holds; past `usize` refused by
    /// name as `what`.
    fn bytes_of(layout: TierLayout, what: &'static str) -> Result<usize, GpuError> {
        layout
            .bytes()
            .ok_or_else(|| GpuError::shape(what, "the page's size passes usize"))
    }

    /// The page over `link`, laid out by `layout`, every tier's fault copy
    /// clean, with the stage card's windows in `stage`'s context.
    fn over(stage: &Gpu, layout: TierLayout, link: Link) -> Result<TierPage, GpuError> {
        let ctx = stage.context();
        for t in 0..layout.tiers {
            link.store(layout.word_off(t, TWord::Fault), FAULT_NONE);
        }
        let img = layout.image;
        let end = link.own_end();
        let stage_rows = (0..layout.rows)
            .map(|r| {
                // SAFETY: row r's image (`img.words()` words, every tier's
                // places with it) and its routed rows (`n_used · cols · hidden`
                // f32)
                // lie inside the page, apart from each other and from every
                // other row's (the layout's strides); the windows drop before
                // the page (field order).
                unsafe {
                    StageRow {
                        image: window::<u32>(end.dev(layout.image_off(r)), img.words(), ctx),
                        rows: window::<f32>(
                            end.dev(layout.rows_off(r)),
                            img.n_used * img.cols * img.hidden,
                            ctx,
                        ),
                    }
                }
            })
            .collect();
        Ok(TierPage {
            stage: stage_rows,
            link,
            layout,
        })
    }

    /// The page's layout.
    #[must_use]
    pub fn layout(&self) -> TierLayout {
        self.layout
    }

    /// Remove the name of the file the page is shared through, once every
    /// tier node has opened it ([`Link::unlink`]); nothing for a page of this
    /// process alone.
    pub(super) fn unlink(&mut self) -> Result<(), GpuError> {
        self.link.unlink()
    }

    /// Row `row`'s image as tier `tier` reads it and the routed rows, for
    /// the stage card's handoff and join; a row or a tier the page does not
    /// carry is refused by name.
    pub(super) fn target_of(
        &mut self,
        row: usize,
        tier: usize,
    ) -> Result<TierTarget<'_>, GpuError> {
        let layout = self.layout.image(tier)?;
        let n = self.stage.len();
        let r = self.stage.get_mut(row).ok_or_else(|| {
            GpuError::shape(WHAT, format!("row {row} of a tier page of {n} rows"))
        })?;
        Ok(TierTarget {
            image: &mut r.image,
            rows: &r.rows,
            layout,
        })
    }

    /// Row `row`'s routed rows as the stage card's join reads them; a row
    /// the page does not carry is refused by name.
    pub(super) fn rows_of(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        let n = self.stage.len();
        self.stage
            .get(row)
            .map(|r| &*r.rows)
            .ok_or_else(|| GpuError::shape(WHAT, format!("row {row} of a tier page of {n} rows")))
    }

    fn check_row(&self, row: usize, what: &'static str) -> Result<(), GpuError> {
        if row < self.layout.rows {
            Ok(())
        } else {
            Err(GpuError::shape(
                what,
                format!("row {row} of a tier page of {} rows", self.layout.rows),
            ))
        }
    }

    /// Enqueue on the stage card's `stream` the go of layer `layer` of row
    /// `row` for a tier layer: the host tier's go batch
    /// ([`Boundary::enqueue_go_of`]) with one added to the row's go counter
    /// of each tier of `tiers` — the tiers that hold experts of the layer, in
    /// tier order — after the first system barrier, so each tier, like the
    /// host, starts only once the image has landed. One batch, whatever the
    /// tier count; an empty `tiers` is refused by name.
    pub(super) fn enqueue_go_of(
        &self,
        stream: &CudaStream,
        boundary: &Boundary,
        layer: usize,
        row: usize,
        tiers: &[usize],
    ) -> Result<(), GpuError> {
        let what = "TierCard::enqueue_go";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        self.check_row(row, what)?;
        let k = self.check_tiers(tiers, layer as usize, what)?;
        let (lyr, gen_, seq) = boundary.go_words(row, what)?;
        let end = self.link.own_end();
        let mut ops = [op_barrier_sys(); 5 + MAX_TIERS];
        ops[1] = crate::graph::op_write(lyr, layer);
        for (op, &t) in ops[2..].iter_mut().zip(tiers) {
            *op = op_add(end.dev(self.layout.word_off(t, TWord::Go(row))), 1);
        }
        ops[3 + k] = op_add(gen_, 1);
        ops[4 + k] = op_add(seq, 1);
        // SAFETY: the tiers' go words lie in this page (row and tiers checked
        // above) and the rest in the boundary's page and region (`go_words`
        // checked the row); each outlives every graph that captured this
        // batch, as for the boundary's own go.
        unsafe {
            mem_batch(
                stream,
                &mut ops[..5 + k],
                "cuStreamBatchMemOp_v2 (hybrid go, tier)",
            )
        }
    }

    /// Enqueue on the stage card's `stream` the wait of row `row` for a
    /// tier layer whose experts the tiers `tiers` hold: until the host's
    /// counter and each of those tiers' counters are at least one, then minus
    /// one from each. The settle rule: a tier graph serves each of its layers
    /// every step, its go count fixed when it was captured, so layer `l`'s
    /// wait waits for every tier `t` with `kₜ(l) > 0` and for no other. One
    /// batch, whatever the tier count; an empty `tiers` is refused by name.
    pub(super) fn enqueue_back_of(
        &self,
        stream: &CudaStream,
        boundary: &Boundary,
        layer: usize,
        row: usize,
        tiers: &[usize],
    ) -> Result<(), GpuError> {
        let what = "TierCard::enqueue_back";
        self.check_row(row, what)?;
        let k = self.check_tiers(tiers, layer, what)?;
        let host = boundary.cnt_word(row, what)?;
        let end = self.link.own_end();
        let mut ops = [op_wait_geq(host, 1); 2 + 2 * MAX_TIERS];
        ops[1 + k] = op_add(host, u32::MAX);
        for (i, &t) in tiers.iter().enumerate() {
            let cnt = end.dev(self.layout.word_off(t, TWord::Cnt(row)));
            ops[1 + i] = op_wait_geq(cnt, 1);
            ops[2 + k + i] = op_add(cnt, u32::MAX);
        }
        // SAFETY: the tiers' counters lie in this page (row and tiers checked
        // above) and the host's in the boundary's page (`cnt_word` checked
        // the row); each outlives every graph that captured this batch, as
        // for the boundary's own wait.
        unsafe {
            mem_batch(
                stream,
                &mut ops[..2 + 2 * k],
                "cuStreamBatchMemOp_v2 (hybrid wait, tier)",
            )
        }
    }

    /// The count of `tiers`, each a tier of the page, ascending, at least
    /// one; refused by name as `what` at `layer` otherwise.
    fn check_tiers(
        &self,
        tiers: &[usize],
        layer: usize,
        what: &'static str,
    ) -> Result<usize, GpuError> {
        let ascending = tiers.windows(2).all(|w| w[0] < w[1]);
        match tiers.last() {
            Some(&t) if ascending && t < self.layout.tiers => Ok(tiers.len()),
            _ => Err(GpuError::shape(
                what,
                format!(
                    "layer {layer}: the tiers {tiers:?} of a tier page of {} tiers",
                    self.layout.tiers
                ),
            )),
        }
    }

    /// Byte offset of tier `tier`'s word `w`.
    fn off(&self, tier: usize, w: TWord) -> usize {
        self.layout.word_off(tier, w)
    }

    fn counters(&self, tier: usize) -> impl Iterator<Item = usize> + '_ {
        self.layout.counters(tier)
    }

    /// Release every wait of both cards on the page's words that is still
    /// pending, for good: every tier's counters and go counters go far past
    /// anything a step subtracts, in one link call ([`Link::release`]), which
    /// names an end that leaves its cards' waits standing.
    pub(super) fn release(&self) -> Result<(), String> {
        self.link.release(RELEASE)
    }

    /// Why the link stopped carrying for good ([`Link::broken`]), as the
    /// note an error ends with; empty while it carries.
    pub(super) fn broken_note(&self) -> String {
        self.link
            .broken()
            .map_or_else(String::new, |b| format!(" ({b})"))
    }
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
/// staged activation `act` (its first `cols` columns, in the form
/// [`TierShape::act`] names), each routed slot whose place in `sel`
/// (`n_used` places a column, a tier slot or [`super::slots::HOST`]) is a
/// tier slot runs its expert and writes its down output over
/// `rows[j · hidden ..]` for its routed slot `j` (column `j / n_used`); no
/// other row is written. `rows` is the tier card's own buffer
/// (`n_used · cols · hidden` of the tier's columns), which the protocol
/// copies to the row's routed rows on the page after the launches. The
/// staging holds the tier's columns
/// ([`TierCard::open_cols`]), of which a go of the step or the pair fills
/// one and a [`Chain::Cols`] go its width.
pub struct TierIo<'a> {
    pub act: TierInput<'a>,
    pub sel: &'a DeviceBuffer<u32>,
    pub rows: &'a mut DeviceBuffer<f32>,
    pub cols: usize,
}

/// A tier layer's staged activation: the stage card's q8_1 codes and scales
/// of the normed activation ([`TierAct::Q8`]), or the normed activation in
/// f32 ([`TierAct::F32`]).
#[derive(Clone, Copy)]
pub enum TierInput<'a> {
    Q8(&'a Q8Act),
    F32(&'a DeviceBuffer<f32>),
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

    /// Where a block's tier rows reach the stage card ([`BlockRows`]): by
    /// default the batch service's copy of the block's down outputs.
    fn block_rows(&self) -> BlockRows {
        BlockRows::Staged
    }
}

/// Where the rows of a prompt batch's block that the tier computed reach the
/// set's host-mapped rows the stage card's join reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRows {
    /// The batch service copies the block's down outputs ([`TierBlock::down`])
    /// to the set's rows whole, every slot's row, after the block.
    Staged,
    /// The block packs its tier slots' rows at the front of the down outputs
    /// in slot order — the tier slot with `k` tier slots before it at row
    /// `k` — and the batch service copies those rows alone, a row for each
    /// slot the block's places do not leave to the host: a block whose tier
    /// slots are a small share of its slots moves only theirs over the bus.
    Packed,
}

/// A block of a prompt batch as the tier's batch service hands it to the
/// architecture ([`TierExperts::enqueue_block`]): its `cols` token columns
/// from column 0 in f32 (`x`, `hidden` a column) and in q8_1 form (`act`,
/// the quantizer's over `x`), each of its `n_used · cols` slots' tier place
/// (a tier slot, or [`super::slots::HOST`] for a slot the tier does not
/// compute), and the down outputs (`hidden` a slot) — slot-major, of which
/// the tier's slots' rows are written and no other, or packed
/// ([`BlockRows::Packed`]).
pub struct TierBlock<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub act: &'a Q8Act,
    pub sel: &'a DeviceBuffer<u32>,
    pub cols: usize,
    pub down: &'a mut DeviceBuffer<f32>,
}

/// What the stage card's handoff writes into a row's tier image and its
/// join reads back ([`TierPage::target_of`]): the image, the routed rows the
/// tiers write, as windows of the stage card's context, and the layout of
/// one tier's image.
pub struct TierTarget<'a> {
    pub image: &'a mut DeviceBuffer<u32>,
    pub rows: &'a DeviceBuffer<f32>,
    pub layout: TierImageLayout,
}

/// What the tier has done since load, counted by the decode thread.
#[derive(Clone, Debug, Default)]
pub struct TierStats {
    /// Completed control request/reply pairs on a node tier, zero for a local tier.
    pub control_round_trips: u64,
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
    /// Passes of a captured chain that launched the tier's graph held for
    /// it ([`TierCard::replay`]), passes fed a layer at a time for want of
    /// one, and the graphs captured after those: a chain is fed and captured
    /// once, at its first pass, and replayed by every later one.
    pub replays: u64,
    pub feeds: u64,
    pub captures: u64,
}

/// One row's routed rows on the page as the tier's copy out writes them (the
/// tier's context).
struct TierRows(ManuallyDrop<DeviceBuffer<f32>>);

impl Drop for TierRows {
    fn drop(&mut self) {
        // SAFETY: the window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — the host tier frees the page after its tier cards.
        unsafe { drop(ManuallyDrop::take(&mut self.0).into_raw_parts()) };
    }
}

/// The stage card as the tier reaches it: its stream (a fault is merged
/// once it drained) and its fault word.
struct Stage {
    stream: Arc<CudaStream>,
    fault: Arc<DeviceBuffer<u32>>,
}

/// The tier graphs' slots: the step, the pair, the quad, and one per
/// [`Chain::Cols`] width `2..=DEFER_MAX_COLS` ([`chain_slot`]).
pub(super) const TIER_CHAINS: usize = 2 + DEFER_MAX_COLS;

/// An expert tier's work on its card: its `Gpu` (its own stream and fault
/// word), its resident stacks, the architecture's computation, the staging
/// a layer's kernels read, its windows over the page and one captured graph
/// per chain. A tier card in the stage card's process holds it
/// ([`TierCard`]); a tier node's server holds it in another
/// ([`super::node`]). Each method is one step the stage side's protocol asks
/// of the tier, and the stage side alone decides which.
///
/// Field order is drop order: the graphs before the buffers they address,
/// the windows before the link, the weights and the link before the card.
pub(crate) struct TierWork {
    graphs: [Option<Graph>; TIER_CHAINS],
    experts: Box<dyn TierExperts>,
    module: tier_kernels::LoadedModule,
    /// The staged activation and places a layer's kernels read, copied in
    /// from the row's image.
    act: Staged,
    sel: DeviceBuffer<u32>,
    /// The architecture's routed rows of the layer being served, on this
    /// card ([`TierIo::rows`]): `n_used · cols · hidden`, copied to the
    /// row's routed rows on the page after the launches.
    out: DeviceBuffer<f32>,
    /// Per row, the page's routed rows in this card's context: made when the
    /// work is bound to the page ([`TierWork::bind`]).
    rows: Vec<TierRows>,
    /// The card's end of the page's link, in this card's context, and the
    /// page's layout: set when the work is bound to the page.
    end: Option<(LinkEnd, TierLayout)>,
    /// The card's tier on the page: its index in the host tier's tiers.
    index: usize,
    shape: TierShape,
    /// Columns a row's image carries ([`TierWork::open`]).
    cols: usize,
    weights: Weights,
    /// The page as a tier node opened it ([`TierWork::bind_opened`]); `None`
    /// beside the stage card, whose host tier owns the page.
    link: Option<Link>,
    gpu: Gpu,
}

impl TierWork {
    /// The work on `gpu` with `weights` its resident routed stacks,
    /// `experts` the architecture's computation, cut for `shape`, rows of
    /// `cols` columns (`1..=DEFER_MAX_COLS`, else refused by name): the
    /// staging holds `cols` columns' activation and places, the page's rows
    /// `cols` columns' ([`TierLayout::with_cols`]). Load-time only. Its
    /// windows over the page are made when it is bound ([`TierWork::bind`]).
    pub(super) fn open(
        gpu: Gpu,
        weights: Weights,
        experts: Box<dyn TierExperts>,
        shape: TierShape,
        cols: usize,
    ) -> Result<TierWork, GpuError> {
        if !(1..=DEFER_MAX_COLS).contains(&cols) {
            return Err(GpuError::shape(
                WHAT,
                format!("a tier of {cols} columns a row: 1..={DEFER_MAX_COLS}"),
            ));
        }
        let ctx = Arc::clone(gpu.context());
        ctx.bind_to_thread()?;
        let stream = gpu.stream();
        let act = match shape.act {
            TierAct::Q8 => Staged::Q8(Q8Act::with_k(stream, cols, shape.hidden)?),
            TierAct::F32 => Staged::F32(DeviceBuffer::zeroed(stream, cols * shape.hidden)?),
        };
        // The shape is checked once here, as the page the work binds to
        // will be laid out.
        let (q3, d8) = act.planes();
        TierLayout::with_cols(shape, cols, q3, d8, 1)?;
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; its launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(tier_kernels, &ctx)? };
        Ok(TierWork {
            graphs: std::array::from_fn(|_| None),
            experts,
            module,
            sel: DeviceBuffer::zeroed(stream, cols * shape.n_used)?,
            out: DeviceBuffer::zeroed(stream, cols * shape.n_used * shape.hidden)?,
            act,
            rows: Vec::new(),
            end: None,
            index: 0,
            shape,
            cols,
            weights,
            link: None,
            gpu,
        })
    }

    /// The tier's card.
    pub(super) fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The tier's resident stacks.
    pub(super) fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The staging's q8_1 codes and scales ([`Staged::planes`]).
    pub(super) fn planes(&self) -> (usize, usize) {
        self.act.planes()
    }

    /// The layout of a page of `tiers` tiers the work's staging reads.
    pub(super) fn page_layout(&self, tiers: usize) -> Result<TierLayout, GpuError> {
        let (q3, d8) = self.act.planes();
        TierLayout::with_cols(self.shape, self.cols, q3, d8, tiers)
    }

    /// Device bytes the work holds besides its weights: the staging and the
    /// routed rows.
    pub(super) fn device_bytes(&self) -> usize {
        self.act.device_bytes() + self.sel.num_bytes() + self.out.num_bytes()
    }

    /// Nodes of the graph held in slot `i`; `None` when none is held.
    fn graph_nodes(&self, i: usize) -> Option<usize> {
        self.graphs[i].as_ref().map(Graph::node_count)
    }

    /// Bind the work to a page laid out by `layout` as tier `index`, `end`
    /// its end of the page's link in its card's context: its windows over
    /// the page's routed rows, in that context.
    pub(super) fn bind(&mut self, layout: TierLayout, index: usize, end: LinkEnd) {
        let ctx = self.gpu.context();
        let img = layout.image;
        self.rows = (0..layout.rows)
            .map(|r| {
                // SAFETY: row r's routed rows (`n_used · cols · hidden` f32)
                // lie inside the page, apart from every other row's (the
                // layout's strides); the host tier frees the page only after
                // its cards, and a node's link is a field dropped after the
                // windows, bound once ([`TierWork::bind_opened`]).
                TierRows(unsafe {
                    window::<f32>(
                        end.dev(layout.rows_off(r)),
                        img.n_used * img.cols * img.hidden,
                        ctx,
                    )
                })
            })
            .collect();
        self.end = Some((end, layout));
        self.index = index;
    }

    /// [`TierWork::bind`] on a tier node, to the page the stage card's
    /// process made, the file at `path`, `bytes` long, laid out for `tiers`
    /// tiers: opened in the work's context ([`Link::open_across`]), its end
    /// that context's address of the page, the link kept while the work
    /// lives. A second bind, a tier the page does not carry and a page
    /// smaller than the layout are refused by name before the page is
    /// opened: the graphs captured hold the first page's addresses.
    pub(super) fn bind_opened(
        &mut self,
        path: &std::path::Path,
        bytes: usize,
        tiers: usize,
        index: usize,
    ) -> Result<(), GpuError> {
        const BIND: &str = "TierWork::bind_opened";
        let layout = self.binding_layout(tiers, index)?;
        let need = TierPage::bytes_of(layout, BIND)?;
        if bytes < need {
            return Err(GpuError::shape(
                BIND,
                format!(
                    "tier {index} of a page of {bytes} bytes laid out for {tiers} tiers of \
                     {need}: the stage card's process laid out another page"
                ),
            ));
        }
        let counters = layout.counters(index).collect();
        let link = Link::open_across(self.gpu.context(), path, bytes, counters)?;
        self.bind(layout, index, link.own_end());
        self.link = Some(link);
        Ok(())
    }

    fn binding_layout(&self, tiers: usize, index: usize) -> Result<TierLayout, GpuError> {
        const BIND: &str = "TierWork::bind";
        if self.link.is_some() {
            return Err(GpuError::shape(
                BIND,
                "a second bind: the tier's graphs hold its page's addresses",
            ));
        }
        let layout = self.page_layout(tiers)?;
        layout.check_tier(index)?;
        Ok(layout)
    }

    /// Bind this node's own page and connect its RDMA end to the stage's endpoint.
    pub(super) fn bind_remote(
        &mut self,
        tiers: usize,
        index: usize,
        peer: &bloomery_verbs::Endpoint,
        ip: std::net::IpAddr,
    ) -> Result<bloomery_verbs::Endpoint, GpuError> {
        let layout = self.binding_layout(tiers, index)?;
        let bytes = TierPage::bytes_of(layout, "TierWork::bind_remote")?;
        let mut link = Link::remote(self.gpu.context(), bytes, layout.crossings().len())?;
        link.store(layout.word_off(index, TWord::Fault), FAULT_NONE);
        let mut end = layout.remote_end(&link, index, Reader::Tier, ip)?;
        let endpoint = end.endpoint(0x200);
        end.connect(endpoint.psn, peer)?;
        link.keep_remote(end);
        self.bind(layout, index, link.own_end());
        self.link = Some(link);
        Ok(endpoint)
    }

    pub(super) fn reset_remote(&self, phase: u8) -> Result<(), GpuError> {
        let (_, layout) = self.end.ok_or(GpuError::State {
            what: "TierWork::reset_remote",
            missing: "a bound page",
        })?;
        let link = self.link.as_ref().ok_or(GpuError::State {
            what: "TierWork::reset_remote",
            missing: "the node's link",
        })?;
        if phase == 0 {
            self.release_opened()?;
            self.drain(super::DRAIN_DEADLINE, "the remote reset's drain")?;
        }
        link.reset_remote(phase, &layout.counters(self.index).collect::<Vec<_>>())
            .map_err(|e| GpuError::protocol("TierWork::reset_remote", e))
    }

    /// Rows of the page the work is bound to; `None` before it is bound.
    pub(super) fn page_rows(&self) -> Option<usize> {
        self.end.map(|(_, l)| l.rows)
    }

    /// Launch the graph held in `chain`'s slot on the tier's stream; a chain
    /// with no graph held is refused by name.
    pub(super) fn launch(&mut self, chain: Chain) -> Result<(), GpuError> {
        let i = chain_slot(chain)?;
        let graph = self.graphs[i].as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "a tier graph of the chain (TierWork::capture)",
        })?;
        self.gpu.context().bind_to_thread()?;
        graph.launch(self.gpu.stream())
    }

    /// Capture `chain`'s graph for the services `list` (in go order), over
    /// the first `cols` columns of each row's image, replacing the one held:
    /// its node count, 0 for an empty list, which holds none. Nothing is
    /// launched.
    pub(super) fn capture(
        &mut self,
        chain: Chain,
        cols: usize,
        list: &[(usize, usize)],
    ) -> Result<usize, GpuError> {
        let i = chain_slot(chain)?;
        self.graphs[i] = None;
        let Some(last) = list.len().checked_sub(1) else {
            return Ok(0);
        };
        self.gpu.context().bind_to_thread()?;
        let mut parts = self.parts()?;
        let graph = Graph::capture(parts.gpu.stream(), |_| {
            for (k, &(layer, row)) in list.iter().enumerate() {
                parts.enqueue_layer(layer, row, cols, k == last)?;
            }
            Ok(())
        })?;
        let n = graph.node_count();
        self.graphs[i] = Some(graph);
        Ok(n)
    }

    /// Enqueue layer `layer` of row `row` on the tier's stream now, over the
    /// image's first `cols` columns, its fault copy with it.
    pub(super) fn enqueue(
        &mut self,
        layer: usize,
        row: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        self.gpu.context().bind_to_thread()?;
        self.parts()?.enqueue_layer(layer, row, cols, true)
    }

    /// Enqueue layer `layer`'s tier experts over the block `io` of a prompt
    /// batch on the tier's stream ([`TierExperts::enqueue_block`]), the
    /// tier's context bound.
    pub(super) fn enqueue_block(
        &mut self,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        self.gpu.context().bind_to_thread()?;
        self.experts
            .enqueue_block(&self.gpu, &self.weights, layer, io)
    }

    /// Wait until the tier's stream has drained.
    pub(super) fn sync(&self) -> Result<(), GpuError> {
        self.gpu.stream().synchronize()?;
        Ok(())
    }

    /// The tier card's fault word back to clean, once its stream drained.
    pub(super) fn clear_fault(&self) -> Result<(), GpuError> {
        self.gpu.clear_fault()?;
        self.gpu.stream().synchronize()?;
        Ok(())
    }

    /// [`TierWork::clear_fault`] with every wait for the stream polled, at
    /// most `deadline` each; past it the call fails by name as `what`. A
    /// tier node's server clears this way, so a stream that never drains
    /// leaves it free to see its stage side lost.
    pub(super) fn clear_fault_within(
        &self,
        deadline: Duration,
        what: &'static str,
    ) -> Result<(), GpuError> {
        self.drain(deadline, what)?;
        self.gpu.clear_fault()?;
        self.drain(deadline, what)
    }

    /// Wait, polling, until the tier's stream has drained, at most
    /// `deadline`; past it the wait fails by name as `what`.
    pub(super) fn drain(&self, deadline: Duration, what: &'static str) -> Result<(), GpuError> {
        super::drain_within(self.gpu.stream(), deadline, what)
    }

    /// The tier card's fault word, once its stream drained ([`Gpu::fault`]).
    pub(super) fn fault(&self) -> Result<Option<Fault>, GpuError> {
        self.gpu.fault()
    }

    /// On a tier node, the layers the tier has served since load, as its
    /// progress word on the page says; `None` before it is bound to a page
    /// it opened.
    pub(super) fn opened_progress(&self) -> Option<u32> {
        let (link, (_, layout)) = (self.link.as_ref()?, self.end?);
        Some(link.load(layout.word_off(self.index, TWord::Prog)))
    }

    /// On a tier node whose stage card's process is lost, release the
    /// tier's words for good, as [`TierPage::release`] does in one process:
    /// each row's go counter and counter, far past anything a layer
    /// takes back. Every go wait the tier's stream holds passes and the
    /// stream drains; a stage side that runs again finds the counters
    /// released and refuses the step by name ([`TierCard::released`]) rather
    /// than read rows the tier served from images it never wrote. Nothing
    /// before it is bound to a page it opened. Returns whether it held a page
    /// whose words it released.
    pub(super) fn release_opened(&self) -> Result<bool, GpuError> {
        match &self.link {
            Some(link) => {
                link.release(RELEASE)
                    .map_err(|e| GpuError::protocol("TierWork::release_opened", e))?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// The parts one tier layer's enqueue reads and writes; refused by name
    /// before the work is bound to a page.
    fn parts(&mut self) -> Result<Parts<'_>, GpuError> {
        let TierWork {
            experts,
            module,
            act,
            sel,
            out,
            rows,
            end,
            index,
            weights,
            gpu,
            ..
        } = self;
        let (end, layout) = end.ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier page (TierWork::bind)",
        })?;
        Ok(Parts {
            experts: &mut **experts,
            module,
            act,
            sel,
            out,
            rows,
            layout,
            end,
            index: *index,
            weights,
            gpu,
        })
    }
}

/// Where a tier card's work runs: on a card of this process, or on a tier
/// node's card ([`TierNode`]), which runs the same [`TierWork`] steps as
/// requests over its channel.
enum TierSide {
    Here(Box<TierWork>),
    Node(Box<TierNode>),
}

/// The error of a call that needs the tier's card in this process, on the
/// tier `name` that runs on the node at `addr`.
fn not_here(name: &str, addr: SocketAddr) -> GpuError {
    GpuError::shape(
        WHAT,
        format!("the tier {name} runs on the node at {addr}: its card is not in this process"),
    )
}

/// An expert tier card as the stage side drives it: where its work runs
/// ([`TierSide`]), its tier on the host tier's page, the go order each held
/// graph was captured for, which experts it holds, and what the host asked
/// of it. Owned by the host tier ([`super::HostTier::attach_tiers`]), which
/// binds it to the page; every call that reads or writes the page takes it.
///
/// Field order is drop order: the work before the stage card's stream.
pub struct TierCard {
    side: TierSide,
    /// Per chain, the (layer, row) services its graph was captured for, in
    /// go order.
    captured: [Vec<(usize, usize)>; TIER_CHAINS],
    /// The card's tier on the page: its index in the host tier's tiers.
    index: usize,
    set: TierSet,
    stage: Option<Stage>,
    stats: TierStats,
    /// Layers the host asked of the tier since load (wrapping): what the
    /// progress word reaches once the tier has served them.
    issued: u32,
    name: String,
}

impl TierCard {
    /// The tier on `gpu` (the card named `name`), with `weights` its
    /// resident routed stacks for `set`, `experts` the architecture's
    /// computation, cut for `shape`, one column a row. Load-time only. Its
    /// windows over the page are made when the host tier takes it
    /// ([`super::HostTier::attach_tiers`]).
    pub fn open(
        gpu: Gpu,
        name: String,
        weights: Weights,
        set: TierSet,
        experts: Box<dyn TierExperts>,
        shape: TierShape,
    ) -> Result<TierCard, GpuError> {
        TierCard::open_cols(gpu, name, weights, set, experts, shape, 1)
    }

    /// [`TierCard::open`] of rows of `cols` columns ([`TierWork::open`]):
    /// the card serves [`Chain::Cols`] up to `cols` wide beside the step and
    /// the pair.
    pub fn open_cols(
        gpu: Gpu,
        name: String,
        weights: Weights,
        set: TierSet,
        experts: Box<dyn TierExperts>,
        shape: TierShape,
        cols: usize,
    ) -> Result<TierCard, GpuError> {
        let work = TierWork::open(gpu, weights, experts, shape, cols)?;
        Ok(TierCard::of(TierSide::Here(Box::new(work)), name, set))
    }

    /// The tier the node `node` serves (the card named `name`, holding
    /// `set`), as its server opened it ([`TierNode::open`]).
    pub(crate) fn on_node(node: TierNode, name: String, set: TierSet) -> TierCard {
        TierCard::of(TierSide::Node(Box::new(node)), name, set)
    }

    fn of(side: TierSide, name: String, set: TierSet) -> TierCard {
        let layer_hits = vec![0; set.layers().len()];
        TierCard {
            side,
            captured: std::array::from_fn(|_| Vec::new()),
            index: 0,
            set,
            stage: None,
            stats: TierStats {
                layer_hits,
                ..TierStats::default()
            },
            issued: 0,
            name,
        }
    }

    fn work(&self) -> Result<&TierWork, GpuError> {
        match &self.side {
            TierSide::Here(w) => Ok(w),
            TierSide::Node(n) => Err(not_here(&self.name, n.addr())),
        }
    }

    fn work_mut(&mut self) -> Result<&mut TierWork, GpuError> {
        match &mut self.side {
            TierSide::Here(w) => Ok(w),
            TierSide::Node(n) => Err(not_here(&self.name, n.addr())),
        }
    }

    /// The tier's card, when it is in this process; a tier on a node is
    /// refused by name.
    pub fn gpu(&self) -> Result<&Gpu, GpuError> {
        self.work().map(TierWork::gpu)
    }

    /// Resident bytes of the tier's stacks, wherever its card is.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.side {
            TierSide::Here(w) => w.weights().resident_bytes(),
            TierSide::Node(n) => n.resident_bytes(),
        }
    }

    /// The address of the node the tier runs on; `None` for a card of this
    /// process.
    #[must_use]
    pub fn node(&self) -> Option<SocketAddr> {
        match &self.side {
            TierSide::Here(_) => None,
            TierSide::Node(n) => Some(n.addr()),
        }
    }

    /// The device of its node's census the tier opened on; `None` for a
    /// card of this process.
    #[must_use]
    pub fn node_device(&self) -> Option<&DeviceInfo> {
        match &self.side {
            TierSide::Here(_) => None,
            TierSide::Node(n) => Some(n.device()),
        }
    }

    /// Why the tier's node is lost, once its channel failed; `None` for a
    /// card of this process and a node still reached.
    #[must_use]
    pub fn node_lost(&self) -> Option<&str> {
        match &self.side {
            TierSide::Here(_) => None,
            TierSide::Node(n) => n.lost(),
        }
    }

    /// The card's name, as the plan gives it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Device bytes of the architecture's prompt-batch block scratch on the
    /// tier card ([`TierExperts::block_bytes`]), when it is in this process.
    pub fn block_bytes(&self) -> Result<usize, GpuError> {
        Ok(self.work()?.experts.block_bytes())
    }

    /// Where the architecture's block rows reach the set's rows
    /// ([`TierExperts::block_rows`]), when the card is in this process.
    pub fn block_rows(&self) -> Result<BlockRows, GpuError> {
        Ok(self.work()?.experts.block_rows())
    }

    /// Which experts the tier holds.
    #[must_use]
    pub fn set(&self) -> &TierSet {
        &self.set
    }

    /// The layout of a page of `tiers` tiers this card's staging reads.
    pub fn page_layout(&self, tiers: usize) -> Result<TierLayout, GpuError> {
        match &self.side {
            TierSide::Here(w) => w.page_layout(tiers),
            TierSide::Node(n) => n.page_layout(tiers),
        }
    }

    /// Columns a row's image carries.
    #[must_use]
    pub fn cols(&self) -> usize {
        match &self.side {
            TierSide::Here(w) => w.cols,
            TierSide::Node(n) => n.cols(),
        }
    }

    /// `chain`'s columns, refused by name past the tier's own.
    fn chain_cols(&self, chain: Chain) -> Result<usize, GpuError> {
        let m = chain.cols();
        if m <= self.cols() {
            Ok(m)
        } else {
            Err(GpuError::shape(
                WHAT,
                format!(
                    "a {chain:?} go of {m} columns to the tier {}, opened for {}",
                    self.name,
                    self.cols()
                ),
            ))
        }
    }

    /// What the tier has done since load.
    #[must_use]
    pub fn stats(&self) -> TierStats {
        let mut stats = self.stats.clone();
        if let TierSide::Node(n) = &self.side {
            stats.control_round_trips = n.control_round_trips();
        }
        stats
    }

    /// Device bytes the tier holds besides its weights: the staging and the
    /// routed rows.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        match &self.side {
            TierSide::Here(w) => w.device_bytes(),
            TierSide::Node(n) => n.device_bytes(),
        }
    }

    /// Nodes per tier layer of a captured tier graph whose image carries
    /// `act`: the go wait, the copies in ([`TierAct::copies`]), the
    /// architecture's launches (`launches`), the routed rows' copy out and
    /// the signal; the graph's last layer adds the fault copy.
    #[must_use]
    pub const fn nodes_per_layer(act: TierAct, launches: usize) -> usize {
        1 + act.copies() + launches + 1 + 1
    }

    /// Nodes of `chain`'s captured tier graph; `None` before its first
    /// replay captured it, or for a chain the tier does not serve.
    #[must_use]
    pub fn graph_nodes(&self, chain: Chain) -> Option<usize> {
        self.graph_nodes_at(chain_slot(chain).ok()?)
    }

    /// Nodes of the graph held in slot `i`; `None` when none is held.
    fn graph_nodes_at(&self, i: usize) -> Option<usize> {
        match &self.side {
            TierSide::Here(w) => w.graph_nodes(i),
            TierSide::Node(n) => n.graph_nodes(i),
        }
    }

    /// The tier's fault copy as `page` holds it: what the tier's last fault
    /// copy saw.
    pub(super) fn fault_copy(&self, page: &TierPage) -> Option<Fault> {
        let at = page.off(self.index, TWord::Fault);
        let (w, s) = (page.link.load(at), page.link.load(at + 4));
        Fault::from_words(w, s)
    }

    /// Enqueue layer `layer`'s tier experts over the block `io` of a prompt
    /// batch on the tier's stream ([`TierExperts::enqueue_block`]), the
    /// tier's context bound; a layer the set holds no expert of, and a tier
    /// on a node, are refused by name. The caller binds its own context
    /// again afterwards.
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
        self.work_mut()?.enqueue_block(layer, io)
    }

    /// Bind the card to `page` as tier `index`: its end of the page's link
    /// and its windows over the page's routed rows, in its own context, or
    /// on a node, the page's file opened there ([`TierNode::bind`]); and
    /// `stage`'s stream and fault word kept. The card's staging copies and
    /// its waits take their addresses from that end; an end the link refuses
    /// ([`Link::tier_end`]) refuses the call by name, and so do a page laid
    /// out for another shape and, for a tier on a node, a page no other
    /// process can open. The host tier's load-time call; `stage`'s context
    /// is current on return.
    pub(super) fn bind(
        &mut self,
        page: &mut TierPage,
        index: usize,
        stage: &Gpu,
    ) -> Result<(), GpuError> {
        const BIND: &str = "TierCard::bind";
        let layout = page.layout();
        if layout != self.page_layout(layout.tiers)? || index >= layout.tiers {
            return Err(GpuError::shape(
                BIND,
                format!(
                    "the tier {} as tier {index} of a page of {} tiers laid out for another shape",
                    self.name, layout.tiers
                ),
            ));
        }
        match &mut self.side {
            TierSide::Here(w) => {
                let end = page.link.tier_end(w.gpu().context(), BIND, &self.name)?;
                w.bind(layout, index, end);
            }
            TierSide::Node(n) if page.link.is_remote() => {
                let mut end = layout.remote_end(&page.link, index, Reader::Stage, n.local_ip()?)?;
                let endpoint = end.endpoint(0x100);
                let peer = n.bind_remote(layout.tiers, index, &endpoint)?;
                end.connect(endpoint.psn, &peer)?;
                page.link.keep_remote(end);
            }
            TierSide::Node(n) => {
                let Some(path) = page.link.path() else {
                    return Err(GpuError::shape(
                        BIND,
                        format!(
                            "the tier {} on the node at {}: the page is not shared through a \
                             file (TierPage::new)",
                            self.name,
                            n.addr()
                        ),
                    ));
                };
                n.bind(path, page.link.bytes(), layout.tiers, index)?;
            }
        }
        self.index = index;
        self.stage = Some(Stage {
            stream: stage.stream_handle(),
            fault: Arc::clone(stage.fault_word()),
        });
        stage.context().bind_to_thread()?;
        Ok(())
    }

    /// Launch `chain`'s tier graph for the services `list` (the stage
    /// capture's tier layers of this tier, in go order) on the tier's stream
    /// when one is held for that list ([`Pass::Graph`]); the host then
    /// expects the tier to serve `list.len()` layers more. With none held
    /// nothing is enqueued and the host feeds the pass a layer at a time
    /// ([`Pass::Feed`]), then captures the graph once it has settled
    /// ([`TierCard::capture`]): an instantiation may wait for the card, and
    /// the stage card's launched chain waits for the host, so none is made
    /// while that chain runs.
    pub(super) fn replay(
        &mut self,
        chain: Chain,
        list: &[(usize, usize)],
    ) -> Result<Pass, GpuError> {
        let i = chain_slot(chain)?;
        self.chain_cols(chain)?;
        if list.is_empty() {
            return Ok(Pass::Graph);
        }
        if self.graph_nodes_at(i).is_none() || self.captured[i] != list {
            self.stats.feeds += 1;
            return Ok(Pass::Feed);
        }
        match &mut self.side {
            TierSide::Here(w) => w.launch(chain)?,
            TierSide::Node(n) => n.launch(chain)?,
        }
        self.issue(list.len());
        self.stats.replays += 1;
        self.rebind_stage()?;
        Ok(Pass::Graph)
    }

    /// Capture `chain`'s tier graph for `list`, replacing the one held,
    /// after a fed pass has settled: the stage card's chain waits on neither
    /// the host nor the tier any more. Nothing is launched.
    pub(super) fn capture(
        &mut self,
        chain: Chain,
        list: &[(usize, usize)],
    ) -> Result<(), GpuError> {
        let i = chain_slot(chain)?;
        let cols = self.chain_cols(chain)?;
        self.captured[i].clear();
        match &mut self.side {
            TierSide::Here(w) => w.capture(chain, cols, list)?,
            TierSide::Node(n) => n.capture(chain, cols, list)?,
        };
        if list.is_empty() {
            return Ok(());
        }
        self.captured[i] = list.to_vec();
        self.stats.captures += 1;
        self.rebind_stage()
    }

    /// Enqueue layer `layer` of row `row` of `chain` on the tier's stream
    /// now, for an eager chain or a fed pass ([`Pass::Feed`]), over the
    /// chain's columns, its fault copy with it; the host then expects one
    /// layer more. A chain wider than the tier's columns is refused by name.
    pub(super) fn enqueue_eager(
        &mut self,
        chain: Chain,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let cols = self.chain_cols(chain)?;
        match &mut self.side {
            TierSide::Here(w) => w.enqueue(layer, row, cols)?,
            TierSide::Node(n) => n.enqueue(layer, row, cols)?,
        }
        self.issue(1);
        self.rebind_stage()
    }

    /// Wait, polling, until the tier's stream has drained, at most
    /// `deadline`; past it the wait fails by name as `what`. The stage
    /// card's context is current on return.
    pub(super) fn drain(&mut self, deadline: Duration, what: &'static str) -> Result<(), GpuError> {
        match &mut self.side {
            TierSide::Here(w) => w.drain(deadline, what)?,
            TierSide::Node(n) => n.drain(deadline)?,
        }
        self.rebind_stage()
    }

    /// The tier card's fault word, once its stream drained
    /// ([`Gpu::fault`]); the caller binds its own context again afterwards.
    pub(super) fn card_fault(&mut self) -> Result<Option<Fault>, GpuError> {
        match &mut self.side {
            TierSide::Here(w) => w.fault(),
            TierSide::Node(n) => n.fault(),
        }
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
            u32::try_from(n).expect("a pass lists at most four rows a layer, far below u32::MAX");
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

    /// The layers the tier has served since load, as its progress word on
    /// `page` says.
    pub(super) fn progress(&self, page: &TierPage) -> u32 {
        page.link.load(page.off(self.index, TWord::Prog))
    }

    /// Layers the host asked of the tier since load.
    #[must_use]
    pub fn issued(&self) -> u32 {
        self.issued
    }

    /// Whether the tier has served fewer than `want` layers.
    pub(super) fn behind_of(&self, page: &TierPage, want: u32) -> bool {
        let p = self.progress(page);
        p != want && want.wrapping_sub(p) < 1 << 31
    }

    /// Whether the tier has served fewer layers than the host asked of it.
    pub(super) fn behind(&self, page: &TierPage) -> bool {
        self.behind_of(page, self.issued)
    }

    /// Wait until the tier has served every layer the host asked of it, or
    /// `deadline` has passed; `true` when it has.
    pub(super) fn wait_caught_up(&mut self, page: &TierPage, deadline: Instant) -> bool {
        let t0 = Instant::now();
        let early = !self.behind(page);
        if !self.wait_for(page, self.issued, deadline) {
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
    pub(super) fn wait_for(&self, page: &TierPage, want: u32, deadline: Instant) -> bool {
        let mut spins = 0u32;
        while self.behind_of(page, want) {
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
    pub(super) fn lost_detail(&self, page: &TierPage, waited: Duration) -> String {
        format!(
            "the expert tier on {} served {} of the {} layers asked of it and signalled nothing more in {waited:.1?}: the card is lost{}",
            self.name,
            self.progress(page),
            self.issued,
            page.broken_note()
        )
    }

    /// Whether a row's counter of the tier holds a release
    /// ([`TierPage::release`]): a tier node that lost this process
    /// releases its own words ([`TierWork::release_opened`]), and every row
    /// the tier served after that is stale.
    pub(super) fn released(&self, page: &TierPage) -> bool {
        (0..page.layout.rows)
            .any(|r| page.link.load(page.off(self.index, TWord::Cnt(r))) >= RELEASE >> 1)
    }

    /// Whether the tier's words hold what a drained step leaves: no go and
    /// no signal pending, every layer asked of it served.
    pub(super) fn at_rest(&self, page: &TierPage) -> bool {
        page.counters(self.index).all(|off| page.link.at_rest(off)) && !self.behind(page)
    }

    /// Release the remote reader's waits and drain its stream, keeping its relay stopped.
    pub(super) fn release_remote(&mut self, page: &TierPage) -> Result<(), GpuError> {
        if page.link.is_remote()
            && let TierSide::Node(n) = &mut self.side
        {
            n.release_remote()?;
        }
        Ok(())
    }

    /// Back to a fresh tier's words once the tier's stream has drained:
    /// counters and go counters at 0, the host's count at the tier's
    /// progress, the tier's fault word and its copy clean. The host tier's
    /// reset calls it. A link that stopped carrying for good fails it by
    /// why.
    pub(super) fn reset(&mut self, page: &TierPage) -> Result<(), GpuError> {
        if page.link.is_remote() {
            match &mut self.side {
                TierSide::Node(n) => {
                    n.reset_remote(&page.link, &page.counters(self.index).collect::<Vec<_>>())?
                }
                TierSide::Here(_) => {
                    return Err(GpuError::shape(
                        "TierCard::reset",
                        "a local tier on a remote page",
                    ));
                }
            }
        } else {
            match &mut self.side {
                TierSide::Here(w) => w.sync()?,
                TierSide::Node(n) => n.drain(super::DRAIN_DEADLINE)?,
            }
            for off in page.counters(self.index) {
                page.link.reset(off).map_err(|b| {
                    GpuError::protocol(
                        "TierCard::reset",
                        format!(
                            "the tier link of the expert tier on {} stopped carrying ({b}): reload the \
                             model",
                            self.name
                        ),
                    )
                })?;
            }
        }
        self.issued = self.progress(page);
        self.clear_fault(page)
    }

    /// The tier's fault word and its copy back to clean.
    pub(super) fn clear_fault(&mut self, page: &TierPage) -> Result<(), GpuError> {
        match &mut self.side {
            TierSide::Here(w) => w.clear_fault()?,
            TierSide::Node(n) => n.clear_fault()?,
        }
        let at = page.off(self.index, TWord::Fault);
        page.link.store(at, FAULT_NONE);
        page.link.store(at + 4, 0);
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
}

/// The step's fault once a tier of `tiers` whose bit is set in `mask` raised
/// one, over `page`: those tiers' copies merged with the stage card's word,
/// read once the stage card's stream drained — the first layer wins. `None`
/// while every copy is clean.
pub(super) fn merged_fault(
    page: &TierPage,
    tiers: &[TierCard],
    mask: u32,
) -> Result<Option<Fault>, GpuError> {
    let mut cards = [None; 1 + MAX_TIERS];
    let mut n = 1;
    for t in tiers.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1) {
        cards[n] = t.1.fault_copy(page);
        n += 1;
    }
    if cards[1..n].iter().all(Option::is_none) {
        return Ok(None);
    }
    cards[0] = match tiers.iter().find_map(|t| t.stage.as_ref()) {
        Some(s) => {
            s.stream.synchronize()?;
            crate::fault::read(&s.fault)?
        }
        None => None,
    };
    Ok(crate::fault::read_cards(&cards[..n]))
}

/// How a replay of a captured chain reaches a tier
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

/// The tier graph's slot of `chain`: the step 0, the pair 1, the quad the
/// last slot, a [`Chain::Cols`] of `m` columns `m` for `2..=DEFER_MAX_COLS`;
/// any other width is refused by name.
pub(super) fn chain_slot(chain: Chain) -> Result<usize, GpuError> {
    match chain {
        Chain::Step => Ok(0),
        Chain::Pair => Ok(1),
        Chain::Quad => Ok(1 + DEFER_MAX_COLS),
        Chain::Cols(m) if (2..=DEFER_MAX_COLS).contains(&m) => Ok(m),
        Chain::Cols(m) => Err(GpuError::shape(
            WHAT,
            format!("a chain of {m} columns: the tier serves 2..={DEFER_MAX_COLS} a row"),
        )),
    }
}

const _: () = assert!(TIER_CHAINS == DEFER_MAX_COLS + 2);

/// A tier card's staging of a row's activation, in its image's form
/// ([`TierAct`]).
enum Staged {
    Q8(Q8Act),
    F32(DeviceBuffer<f32>),
}

impl Staged {
    /// The q8_1 codes (u64) and scales the staging holds: none for f32.
    fn planes(&self) -> (usize, usize) {
        match self {
            Staged::Q8(a) => (a.q3.len(), a.d8.len()),
            Staged::F32(_) => (0, 0),
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Staged::Q8(a) => a.device_bytes(),
            Staged::F32(x) => x.num_bytes(),
        }
    }

    /// The staging as the architecture's launches read it.
    fn input(&self) -> TierInput<'_> {
        match self {
            Staged::Q8(a) => TierInput::Q8(a),
            Staged::F32(x) => TierInput::F32(x),
        }
    }
}

/// The work's parts one tier layer's enqueue reads and writes.
struct Parts<'a> {
    experts: &'a mut dyn TierExperts,
    module: &'a tier_kernels::LoadedModule,
    act: &'a mut Staged,
    sel: &'a mut DeviceBuffer<u32>,
    out: &'a mut DeviceBuffer<f32>,
    rows: &'a mut [TierRows],
    layout: TierLayout,
    /// The work's end of the page's link: every device address the layer's
    /// copies and stream memory operations take.
    end: LinkEnd,
    index: usize,
    weights: &'a Weights,
    gpu: &'a Gpu,
}

impl Parts<'_> {
    /// Layer `layer` of row `row` on the tier's stream over the image's
    /// first `cols` columns: wait for the row's go and take it back, copy
    /// those columns of the row's image — the activation (the q8_1 codes and
    /// their scales, or the f32 values), this tier's places — into the
    /// staging, the architecture's launches into the card's routed rows, one
    /// copy of those columns' rows to the row's routed rows on the page, then
    /// (when `copy_fault`) the fault copy, a system barrier and the row's
    /// counter and the progress word each plus one.
    fn enqueue_layer(
        &mut self,
        layer: usize,
        row: usize,
        cols: usize,
        copy_fault: bool,
    ) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        let (layout, t, end) = (self.layout, self.index, self.end);
        let img = layout.image(t)?;
        if row >= self.rows.len() || !(1..=img.cols).contains(&cols) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "row {row} of {cols} columns on a tier page of {} rows of {} columns",
                    self.rows.len(),
                    img.cols
                ),
            ));
        }
        let go = end.dev(layout.word_off(t, TWord::Go(row)));
        // SAFETY: this tier's go word of row `row` lies in the page (row
        // checked above), which the host tier drops after its tier cards and
        // so after every graph that captured this batch.
        unsafe {
            mem_batch(
                stream,
                &mut [op_wait_geq(go, 1), op_add(go, u32::MAX)],
                "cuStreamBatchMemOp_v2 (tier go)",
            )?;
        }
        let at = layout.image_off(row);
        let staged = match &*self.act {
            Staged::Q8(a) => a.q3.len() * 2 == img.q3_words && a.d8.len() == img.d8_len,
            Staged::F32(x) => x.len() == img.x_len && img.q3_words + img.d8_len == 0,
        };
        let routed = img.n_used * cols * img.hidden;
        if !staged
            || self.sel.len() != img.n_used * img.cols
            || self.out.len() < routed
            || self.rows[row].0.len() < routed
        {
            return Err(GpuError::shape(
                WHAT,
                "the staging is not the image's shape".to_string(),
            ));
        }
        // Each field holds its columns one after another: the first `cols`
        // columns are its first `cols / img.cols` part.
        let part = |len: usize| len / img.cols * cols;
        match &*self.act {
            Staged::Q8(a) => {
                copy_page(
                    stream,
                    a.q3.cu_deviceptr(),
                    end.dev(at + 4 * img.q3),
                    4 * part(img.q3_words),
                )?;
                copy_page(
                    stream,
                    a.d8.cu_deviceptr(),
                    end.dev(at + 4 * img.d8),
                    4 * part(img.d8_len),
                )?;
            }
            Staged::F32(x) => copy_page(
                stream,
                x.cu_deviceptr(),
                end.dev(at + 4 * img.x),
                4 * part(img.x_len),
            )?,
        }
        copy_page(
            stream,
            self.sel.cu_deviceptr(),
            end.dev(at + 4 * img.sel),
            4 * img.n_used * cols,
        )?;
        self.experts.enqueue_layer(
            self.gpu,
            self.weights,
            layer,
            TierIo {
                act: self.act.input(),
                sel: self.sel,
                rows: self.out,
                cols,
            },
        )?;
        copy_page(
            stream,
            self.rows[row].0.cu_deviceptr(),
            self.out.cu_deviceptr(),
            4 * routed,
        )?;
        if copy_fault {
            let ctx = self.gpu.context();
            // SAFETY: the fault copy's two words lie in this tier's flag
            // lines; the window is used for this launch only and never
            // dropped as a buffer (its raw parts are, below).
            let mut out =
                unsafe { window::<u32>(end.dev(layout.word_off(t, TWord::Fault)), 2, ctx) };
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
        // SAFETY: this tier's counter of row `row` and its progress word lie
        // in the page (row checked above), which the host tier drops after
        // its tier cards and so after every graph that captured this batch.
        unsafe {
            mem_batch(
                stream,
                &mut [
                    op_barrier_sys(),
                    op_add(end.dev(layout.word_off(t, TWord::Cnt(row))), 1),
                    op_add(end.dev(layout.word_off(t, TWord::Prog)), 1),
                ],
                "cuStreamBatchMemOp_v2 (tier signal)",
            )
        }
    }
}

/// Copy `bytes` from `src` to `dst` on `stream`, one span a device buffer of
/// the tier's context and the other a span of the tier page (a field of a
/// row's image in, a row's routed rows out): one copy node when captured.
fn copy_page(
    stream: &CudaStream,
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    bytes: usize,
) -> Result<(), GpuError> {
    // SAFETY: both spans are the caller's, each at least `bytes`: one a
    // device buffer of the tier's context, the other a span of the tier page
    // inside its allocation; both outlive every graph that captured the copy.
    let rc = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
    cu(rc, "cuMemcpyDtoDAsync_v2 (tier page copy)")
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use super::{
        TIER_CHAINS, TIER_FLAGS, TWord, TierAct, TierLayout, TierOpen, TierPage, TierSet,
        TierShape, chain_slot,
    };
    use crate::Gpu;
    use crate::host::slots::{HOST, Slot, SlotMap, TIER};
    use crate::host::step::{Chain, RELEASE};
    use bloomery_levers::TierLink;
    use model::ops::DEFER_MAX_COLS;

    pub(in crate::host) fn remote_work(
        file: &gguf::Split,
        dropped: Arc<AtomicBool>,
        stalled: bool,
    ) -> super::TierWork {
        struct Empty(Arc<AtomicBool>);
        impl Drop for Empty {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        impl super::TierExperts for Empty {
            fn enqueue_layer(
                &mut self,
                _: &Gpu,
                _: &super::Weights,
                _: usize,
                _: super::TierIo<'_>,
            ) -> Result<(), crate::GpuError> {
                Ok(())
            }
            fn enqueue_block(
                &mut self,
                _: &Gpu,
                _: &super::Weights,
                _: usize,
                _: super::TierBlock<'_>,
            ) -> Result<(), crate::GpuError> {
                Ok(())
            }
            fn block_bytes(&self) -> usize {
                0
            }
        }
        let gpu = Gpu::with_device(0).expect("a GPU");
        let weights =
            super::Weights::load_where(gpu.stream(), file, |_| false).expect("no weights");
        let shape = TierShape {
            hidden: 32,
            n_used: 1,
            rows: 1,
            act: TierAct::F32,
        };
        let mut work = super::TierWork::open(gpu, weights, Box::new(Empty(dropped)), shape, 1)
            .expect("empty work");
        let layout = work.page_layout(1).expect("a layout");
        let link = super::super::link::tests::remote_with_wire(
            work.gpu().context(),
            layout.bytes().expect("page bytes"),
            layout.crossings(),
            stalled,
        );
        work.bind(layout, 0, link.own_end());
        work.link = Some(link);
        work
    }

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_remote_reset_releases_waits_before_its_drain() {
        let (path, file) = super::super::node::tests::empty_file("remote-reset");
        let mut work = remote_work(&file, Arc::new(AtomicBool::new(false)), false);
        work.enqueue(0, 0, 1).expect("a go wait");
        let result = work.reset_remote(0);
        work.link
            .as_ref()
            .expect("a page")
            .release(RELEASE)
            .expect("release for cleanup");
        work.sync().expect("drained for cleanup");
        assert!(result.is_ok(), "{result:?}");
        drop(work);
        drop(file);
        std::fs::remove_file(path).expect("remove the fixture");
    }

    /// A family whose tier opens only in the loading process refuses a
    /// tier on a node by name, in one sentence, before it opens a card.
    #[test]
    fn a_tier_on_a_node_is_refused_by_name() {
        let t = TierOpen {
            card: 1,
            name: "GB10".into(),
            device: None,
            node: Some("127.0.0.1:7340".parse().expect("an address")),
        };
        let e = t
            .open_here("deepseek41 open_tier")
            .err()
            .expect("a refusal")
            .to_string();
        assert!(
            e.contains(
                "the expert tier GB10 on the node at 127.0.0.1:7340: this family's tier opens on \
                 a card of the loading process only"
            ),
            "{e}"
        );
    }

    /// A set's rows are ascending ids below the stack, each once; a map's
    /// set of a tier is that tier's entries in its slot order, and a layer
    /// outside it is refused.
    #[test]
    fn a_set_is_the_maps_tier_rows() {
        let rows = vec![0, HOST, TIER, HOST, HOST, TIER, 0, TIER | 1];
        let map = SlotMap::from_rows(3..5, 4, rows).expect("two rows of 4");
        let set = TierSet::of_map(&map, 0).expect("the map's set");
        assert_eq!(set.ids(3), Some(&[2][..]));
        assert_eq!(set.ids(4), Some(&[1, 3][..]));
        assert_eq!(set.on_tier(3).expect("layer 3"), 1);
        assert_eq!(set.on_tier(4).expect("layer 4"), 2);
        assert!(set.on_tier(5).is_err());
        assert_eq!(set.experts(), 3);
        assert!(
            TierSet::of_map(&map, 1).is_err(),
            "a tier the map does not name"
        );
        let set = TierSet::new(3..5, 8, vec![vec![2, 5], vec![]]).expect("a set");
        assert_eq!(set.on_tier(4).expect("layer 4"), 0);
        assert!(TierSet::new(0..1, 8, vec![vec![3, 3]]).is_err());
        assert!(TierSet::new(0..1, 8, vec![vec![4, 2]]).is_err());
        assert!(TierSet::new(0..1, 8, vec![vec![8]]).is_err());
        assert!(TierSet::new(0..2, 8, vec![vec![1]]).is_err());
    }

    /// Each tier's set is its own entries alone: tier 1's ids are not tier
    /// 0's.
    #[test]
    fn each_tiers_set_is_its_own() {
        let t = |tier: usize, slot: u32| Slot::Tier { tier, slot }.entry().expect("a tier slot");
        let map = SlotMap::from_rows(0..1, 5, vec![t(1, 0), 0, t(0, 0), t(1, 1), HOST])
            .expect("a row over two tiers");
        assert_eq!(
            TierSet::of_map(&map, 0).expect("tier 0").ids(0),
            Some(&[2][..])
        );
        assert_eq!(
            TierSet::of_map(&map, 1).expect("tier 1").ids(0),
            Some(&[0, 3][..])
        );
    }

    const V41: TierShape = TierShape {
        hidden: 4096,
        n_used: 6,
        rows: 2,
        act: TierAct::Q8,
    };

    /// V4.1's tier page of one tier: two rows of six slots over 4096 values,
    /// the q8_1 codes of 16 super-blocks (512 u64) and their 32 scales; every
    /// field apart and inside the page, the flag words each on a line of
    /// their own — and every offset the one-tier page's: the flag words at
    /// 0, 64, 128, 192, 512 and 576, the images from byte 768.
    // PIN(2026-10-05): page::MAX_ROWS 2→4 moves the tier's flag lines —
    // TIER_FLAGS 128·4+128=640, Prog 512, Fault 576 — so the words'
    // offsets and the images' start (payload_off 768) moved with them.
    #[test]
    fn v41_tier_page_keeps_its_fields_apart() {
        let l = TierLayout::new(V41, 512, 32, 1).expect("V4.1's tier page");
        let i = l.image(0).expect("tier 0");
        assert_eq!((i.sel, i.q3, i.d8), (0, 64, 64 + 1024));
        assert!(i.sel + i.n_used <= i.q3 && i.q3 + i.q3_words <= i.d8);
        assert_eq!(l.payload_off(), 768);
        assert!(l.image_off(1) >= l.image_off(0) + 4 * i.words());
        assert!(l.rows_off(0) >= l.image_off(1) + 4 * i.words());
        assert!(l.rows_off(1) >= l.rows_off(0) + 4 * 6 * 4096);
        assert!(l.bytes().unwrap() >= l.rows_off(1) + 4 * 6 * 4096);
        let mut offs = vec![l.word_off(0, TWord::Prog), l.word_off(0, TWord::Fault)];
        for r in 0..2 {
            offs.extend([l.word_off(0, TWord::Go(r)), l.word_off(0, TWord::Cnt(r))]);
        }
        offs.sort_unstable();
        assert_eq!(offs, [0, 64, 128, 192, 512, 576]);
        assert!(offs.iter().all(|&o| o + 64 <= l.payload_off()));
        assert!(l.image(1).is_err());
        // PIN(2026-10-05): 3 rows moved inside 1..=MAX_ROWS (4); the refusal
        // probe is the bound's new outside, 5.
        assert!(TierLayout::new(TierShape { rows: 5, ..V41 }, 512, 32, 1).is_err());
        assert!(TierLayout::new(V41, 512, 32, 0).is_err());
        assert!(TierLayout::new(V41, 512, 32, 9).is_err());
    }

    /// A release interrupted after any store leaves a loss marker before
    /// the tier can serve stale images and a resumed stage can take their done.
    #[test]
    fn a_tier_release_marks_stale_rows_before_the_stage_can_take_them() {
        for rows in [1, super::MAX_ROWS] {
            for tiers in [1, super::MAX_TIERS] {
                let layout = TierLayout::new(TierShape { rows, ..V41 }, 512, 32, tiers)
                    .expect("a tier page");
                for tier in 0..tiers {
                    let page: Vec<AtomicU32> = (0..layout.payload_off() / 4)
                        .map(|_| AtomicU32::new(0))
                        .collect();
                    for off in layout.counters(tier) {
                        page[off / 4].store(RELEASE, Ordering::Release);
                        for row in 0..rows {
                            let go = &page[layout.word_off(tier, TWord::Go(row)) / 4];
                            let done = &page[layout.word_off(tier, TWord::Cnt(row)) / 4];
                            if go.load(Ordering::Acquire) >= RELEASE {
                                // The released tier finishes a stale layer while
                                // release is paused; the stage takes its done.
                                done.fetch_add(1, Ordering::Release);
                                done.fetch_sub(1, Ordering::AcqRel);
                                assert!(
                                    done.load(Ordering::Acquire) >= RELEASE >> 1,
                                    "tier {tier} row {row}: stale rows accepted after release at {off}"
                                );
                            }
                        }
                    }
                    for row in 0..rows {
                        for w in [TWord::Cnt(row), TWord::Go(row)] {
                            assert_eq!(
                                page[layout.word_off(tier, w) / 4].load(Ordering::Acquire),
                                RELEASE
                            );
                        }
                    }
                }
            }
        }
    }

    /// A release over the loopback takes every counter of the page in one
    /// call. The released tier card runs on: after each release the tier's
    /// done steps by three on the tier cards' page and each end passes once.
    /// Released together, the done is stopped on both ends before any
    /// counter is raised and carries nothing; released one counter at a
    /// time, the passes between the go and the done would see the done three
    /// ahead and stop its end by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_release_takes_a_tiers_counters_in_one_hold_of_the_pump() {
        let stage = Gpu::with_device(0).expect("a Gpu on device 0");
        let tier = Gpu::with_device(0).expect("a second Gpu on device 0");
        let layout = TierLayout::new(V41, 512, 32, 1).expect("V4.1's tier page");
        let mut page =
            TierPage::new(&stage, &tier, layout, TierLink::Loopback).expect("a loopback page");
        let done = layout.word_off(0, TWord::Cnt(0));
        page.link.run_on(done, 3);
        assert_eq!(page.release(), Ok(()));
        assert_eq!(page.link.broken(), None);
        assert_eq!(page.link.load(done), RELEASE);
    }

    /// A page of three tiers: each tier's flag lines its own, past the
    /// last of them the images; each tier's places apart from every other
    /// tier's and before the shared activation, which every tier reads at
    /// one offset.
    #[test]
    fn a_page_of_three_tiers_keeps_each_tiers_fields_apart() {
        let l = TierLayout::new(V41, 512, 32, 3).expect("three tiers");
        let mut offs = Vec::new();
        for t in 0..3 {
            offs.extend([l.word_off(t, TWord::Prog), l.word_off(t, TWord::Fault)]);
            for r in 0..2 {
                offs.extend([l.word_off(t, TWord::Go(r)), l.word_off(t, TWord::Cnt(r))]);
            }
        }
        offs.sort_unstable();
        assert!(offs.windows(2).all(|w| w[1] - w[0] >= 64), "{offs:?}");
        assert!(offs.iter().all(|&o| o + 64 <= l.payload_off()));
        assert_eq!(l.word_off(2, TWord::Go(0)), 2 * TIER_FLAGS);
        let images: Vec<_> = (0..3).map(|t| l.image(t).expect("a tier")).collect();
        for (t, i) in images.iter().enumerate() {
            assert_eq!((i.q3, i.d8), (images[0].q3, images[0].d8), "tier {t}");
            assert!(i.sel + i.n_used <= i.q3, "tier {t}");
        }
        assert!(
            images
                .windows(2)
                .all(|w| w[0].sel + w[0].n_used <= w[1].sel)
        );
        assert!(l.image(3).is_err());
    }

    /// GLM's tier page of one tier: two rows of eight slots over the 4096
    /// f32 of the normed activation, the activation at the first 256-byte
    /// boundary past the places, no codes and no scales; an f32 image handed
    /// codes or scales is refused.
    #[test]
    fn an_f32_tier_page_keeps_its_fields_apart() {
        let glm = TierShape {
            hidden: 4096,
            n_used: 8,
            rows: 2,
            act: TierAct::F32,
        };
        let l = TierLayout::new(glm, 0, 0, 1).expect("GLM's tier page");
        let i = l.image(0).expect("tier 0");
        assert_eq!(
            (i.sel, i.x, i.x_len, i.q3_words, i.d8_len),
            (0, 64, 4096, 0, 0)
        );
        assert_eq!(i.words(), 64 + 4096);
        assert_eq!(l.wire_bytes(), [16_644, 131_084]);
        assert!(l.rows_off(0) >= l.image_off(1) + 4 * i.words());
        assert!(l.bytes().unwrap() >= l.rows_off(1) + 4 * 8 * 4096);
        assert!(TierLayout::new(glm, 512, 0, 1).is_err());
        assert!(TierLayout::new(glm, 0, 32, 1).is_err());
        assert_eq!(TierAct::F32.copies() + 1, TierAct::Q8.copies());
    }

    /// Qwen3.8's tier page of one tier: one row of up to four columns of ten
    /// slots over 2,560 f32 — the places 40 words from 0, the activation's
    /// four columns from the first 256-byte boundary past them, the routed
    /// rows forty slots — and at one column every offset the one-column
    /// page's; a width of 0 or past the host's columns is refused; the
    /// graph slots are the step's, the pair's, the quad's and one per width
    /// 2..=8, the existing slots where they were.
    #[test]
    fn a_tier_page_of_four_columns_keeps_its_fields_apart() {
        let q38 = TierShape {
            hidden: 2560,
            n_used: 10,
            rows: 1,
            act: TierAct::F32,
        };
        let l = TierLayout::with_cols(q38, 4, 0, 0, 1).expect("Qwen3.8's tier page");
        let i = l.image(0).expect("tier 0");
        assert_eq!(
            (i.sel, i.x, i.x_len, i.cols, i.n_used),
            (0, 64, 4 * 2560, 4, 10)
        );
        assert!(i.sel + 4 * i.n_used <= i.x);
        assert_eq!(l.wire_bytes(), [41_220, 409_612]);
        assert!(l.rows_off(0) >= l.image_off(0) + 4 * i.words());
        assert!(l.bytes().unwrap() >= l.rows_off(0) + 4 * 40 * 2560);
        assert_eq!(
            TierLayout::with_cols(q38, 1, 0, 0, 1).expect("one column"),
            TierLayout::new(q38, 0, 0, 1).expect("the one-column page")
        );
        assert!(TierLayout::with_cols(q38, 0, 0, 0, 1).is_err());
        assert!(TierLayout::with_cols(q38, 9, 0, 0, 1).is_err());
        assert_eq!(chain_slot(Chain::Step).ok(), Some(0));
        assert_eq!(chain_slot(Chain::Pair).ok(), Some(1));
        assert_eq!(chain_slot(Chain::Quad).ok(), Some(1 + DEFER_MAX_COLS));
        assert_eq!(chain_slot(Chain::Cols(4)).ok(), Some(4));
        assert!(chain_slot(Chain::Cols(1)).is_err());
        assert!(chain_slot(Chain::Cols(9)).is_err());
        let chains = [Chain::Step, Chain::Pair, Chain::Quad]
            .into_iter()
            .chain((2..=DEFER_MAX_COLS).map(Chain::Cols));
        let mut seen = [false; TIER_CHAINS];
        for c in chains {
            let slot = chain_slot(c).expect("every chain the tier serves has a slot");
            assert!(
                slot < TIER_CHAINS,
                "{c:?} at slot {slot}, past the {TIER_CHAINS} graphs"
            );
            assert!(!seen[slot], "{c:?} shares slot {slot}");
            seen[slot] = true;
        }
    }
}
