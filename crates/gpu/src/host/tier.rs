//! The expert tiers: further cards hung under the host tier ([`TierCard`]).
//! The stage card keeps its plan byte for byte; per hybrid layer each tier
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
//! The tiers' words live in one host-mapped page ([`TierPage`], laid out by
//! [`TierLayout`]) the host tier owns. Per tier it holds that tier's flag
//! lines: per row a go counter the stage card adds one to and the tier takes
//! back, and a counter the tier adds one to and the stage card's wait takes
//! back — the host tier's counter rule, one writer at a time — then the
//! progress word, which the tier adds one to per layer served and the host
//! reads, and the tier's fault copy. Per row it holds the image the stage
//! card's handoff writes — each tier's places of the routed slots, then the
//! stage card's q8_1 activation, which every tier reads — and the rows the
//! tiers' downs write, one per routed slot, each tier its own slots, which
//! the stage card's join reads in place.
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
//! counters too, so every stream drains.

use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use model::placement::Machine;

use super::page::MAX_ROWS;
use super::slots::{MAX_TIERS, Slot, SlotMap};
use super::step::{Boundary, Chain, RELEASE};
use crate::fault::{FAULT_NONE, Fault};
use crate::graph::{Graph, MappedHost, cu, mem_batch, op_add, op_barrier_sys, op_wait_geq};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError, Q8Act};
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
/// and the device the plan resolved it to, by which it opens
/// ([`Gpu::open_card`]).
#[derive(Clone, Debug)]
pub struct TierOpen {
    pub card: usize,
    pub name: String,
    pub device: Option<model::placement::workstation::DeviceId>,
}

impl TierOpen {
    /// The expert tier cards `machine` names, in tier order: each card's
    /// index in [`Machine::all_cards`] (after the stage cards), its name and
    /// its device. Empty without a tier.
    #[must_use]
    pub fn of_machine(machine: &Machine) -> Vec<TierOpen> {
        machine
            .tiers
            .iter()
            .enumerate()
            .map(|(i, t)| TierOpen {
                card: machine.cards.len() + i,
                name: t.name.clone(),
                device: t.device,
            })
            .collect()
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

/// The tier page: the host-mapped memory every tier card and the stage card
/// reach at one address ([`TierLayout`]), allocated in the stage card's
/// context, and the stage card's windows over it. The host tier owns it and
/// drops it after its tier cards, whose graphs and windows address it.
///
/// Field order is drop order: the windows before the page.
pub struct TierPage {
    stage: Vec<StageRow>,
    page: MappedHost,
    layout: TierLayout,
}

impl TierPage {
    /// The page laid out by `layout`, in `stage`'s context, every tier's
    /// fault copy clean, with the stage card's windows. Load-time only;
    /// `stage`'s context is current on return.
    pub(super) fn new(stage: &Gpu, layout: TierLayout) -> Result<TierPage, GpuError> {
        let ctx = stage.context();
        let bytes = layout
            .bytes()
            .ok_or_else(|| GpuError::shape("TierPage::new", "the page's size passes usize"))?;
        let page = MappedHost::new(ctx, bytes, "cuMemHostAlloc (tier page)")?;
        for t in 0..layout.tiers {
            // SAFETY: tier t's fault copy's two words lie in its flag lines,
            // inside the page.
            unsafe {
                page.host_at(layout.word_off(t, TWord::Fault))
                    .cast::<u32>()
                    .write_volatile(FAULT_NONE);
            }
        }
        let img = layout.image;
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
                        image: window::<u32>(page.dev_at(layout.image_off(r)), img.words(), ctx),
                        rows: window::<f32>(
                            page.dev_at(layout.rows_off(r)),
                            img.n_used * img.cols * img.hidden,
                            ctx,
                        ),
                    }
                }
            })
            .collect();
        Ok(TierPage {
            stage: stage_rows,
            page,
            layout,
        })
    }

    /// The page's layout.
    #[must_use]
    pub fn layout(&self) -> TierLayout {
        self.layout
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
        let mut ops = [op_barrier_sys(); 5 + MAX_TIERS];
        ops[1] = crate::graph::op_write(lyr, layer);
        for (op, &t) in ops[2..].iter_mut().zip(tiers) {
            *op = op_add(self.page.dev_at(self.layout.word_off(t, TWord::Go(row))), 1);
        }
        ops[3 + k] = op_add(gen_, 1);
        ops[4 + k] = op_add(seq, 1);
        mem_batch(
            stream,
            &mut ops[..5 + k],
            "cuStreamBatchMemOp_v2 (hybrid go, tier)",
        )
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
        let mut ops = [op_wait_geq(host, 1); 2 + 2 * MAX_TIERS];
        ops[1 + k] = op_add(host, u32::MAX);
        for (i, &t) in tiers.iter().enumerate() {
            let cnt = self.page.dev_at(self.layout.word_off(t, TWord::Cnt(row)));
            ops[1 + i] = op_wait_geq(cnt, 1);
            ops[2 + k + i] = op_add(cnt, u32::MAX);
        }
        mem_batch(
            stream,
            &mut ops[..2 + 2 * k],
            "cuStreamBatchMemOp_v2 (hybrid wait, tier)",
        )
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

    /// The device address of tier `tier`'s word `w`.
    fn dev(&self, tier: usize, w: TWord) -> sys::CUdeviceptr {
        self.page.dev_at(self.layout.word_off(tier, w))
    }

    /// Tier `tier`'s word `w` as the host reads and writes it.
    fn word(&self, tier: usize, w: TWord) -> &AtomicU32 {
        self.word_at(self.layout.word_off(tier, w))
    }

    fn word_at(&self, off: usize) -> &AtomicU32 {
        self.page
            .atomic_u32(off)
            .expect("every tier flag word lies in the page's flag lines")
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
/// other row is written. The staging holds the tier's columns
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

/// One row's routed rows on the page as the tier's down writes them (the
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

/// The tier graphs' slots: the step, the pair, and one per [`Chain::Cols`]
/// width `2..=DEFER_MAX_COLS` ([`chain_slot`]).
const TIER_CHAINS: usize = 1 + DEFER_MAX_COLS;

/// An expert tier's card: its `Gpu` (its own stream and fault word), its
/// resident stacks, its tier on the host tier's page, one captured graph per
/// chain, and the architecture's computation. Owned by the host tier
/// ([`super::HostTier::attach_tiers`]), which binds it to the page; every
/// call that reads or writes the page takes it.
///
/// Field order is drop order: the graphs before the buffers they address,
/// the windows and the weights before the card.
pub struct TierCard {
    graphs: [Option<Graph>; TIER_CHAINS],
    /// Per chain, the (layer, row) services its graph was captured for, in
    /// go order.
    captured: [Vec<(usize, usize)>; TIER_CHAINS],
    experts: Box<dyn TierExperts>,
    module: tier_kernels::LoadedModule,
    /// The staged activation and places a layer's kernels read, copied in
    /// from the row's image.
    act: Staged,
    sel: DeviceBuffer<u32>,
    /// Per row, the page's routed rows in this card's context: made when the
    /// host tier binds the card to its page ([`TierCard::bind`]).
    rows: Vec<TierRows>,
    /// The card's tier on the page: its index in the host tier's tiers.
    index: usize,
    shape: TierShape,
    /// Columns a row's image carries ([`TierCard::open_cols`]).
    cols: usize,
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

    /// [`TierCard::open`] of rows of `cols` columns (`1..=DEFER_MAX_COLS`,
    /// else refused by name): the staging holds `cols` columns' activation
    /// and places, the page's rows `cols` columns' ([`TierLayout::with_cols`]),
    /// and the card serves [`Chain::Cols`] up to `cols` wide beside the step
    /// and the pair.
    pub fn open_cols(
        gpu: Gpu,
        name: String,
        weights: Weights,
        set: TierSet,
        experts: Box<dyn TierExperts>,
        shape: TierShape,
        cols: usize,
    ) -> Result<TierCard, GpuError> {
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
        // The shape is checked once here, as the page the card binds to
        // will be laid out.
        let (q3, d8) = act.planes();
        TierLayout::with_cols(shape, cols, q3, d8, 1)?;
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; its launcher checks the launch contract.
        let module = unsafe { crate::shared_module!(tier_kernels, &ctx)? };
        let layer_hits = vec![0; set.layers().len()];
        Ok(TierCard {
            graphs: std::array::from_fn(|_| None),
            captured: std::array::from_fn(|_| Vec::new()),
            experts,
            module,
            sel: DeviceBuffer::zeroed(stream, cols * shape.n_used)?,
            act,
            rows: Vec::new(),
            index: 0,
            shape,
            cols,
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

    /// Where the architecture's block rows reach the set's rows
    /// ([`TierExperts::block_rows`]).
    #[must_use]
    pub fn block_rows(&self) -> BlockRows {
        self.experts.block_rows()
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

    /// The layout of a page of `tiers` tiers this card's staging reads.
    pub(super) fn page_layout(&self, tiers: usize) -> Result<TierLayout, GpuError> {
        let (q3, d8) = self.act.planes();
        TierLayout::with_cols(self.shape, self.cols, q3, d8, tiers)
    }

    /// Columns a row's image carries.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `chain`'s columns, refused by name past the tier's own.
    fn chain_cols(&self, chain: Chain) -> Result<usize, GpuError> {
        let m = chain.cols();
        if m <= self.cols {
            Ok(m)
        } else {
            Err(GpuError::shape(
                WHAT,
                format!(
                    "a {chain:?} go of {m} columns to the tier {}, opened for {}",
                    self.name, self.cols
                ),
            ))
        }
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

    /// Nodes per tier layer of a captured tier graph whose image carries
    /// `act`: the go wait, the copies in ([`TierAct::copies`]), the
    /// architecture's launches (`launches`) and the signal; the graph's last
    /// layer adds the fault copy.
    #[must_use]
    pub const fn nodes_per_layer(act: TierAct, launches: usize) -> usize {
        1 + act.copies() + launches + 1
    }

    /// Nodes of `chain`'s captured tier graph; `None` before its first
    /// replay captured it, or for a chain the tier does not serve.
    #[must_use]
    pub fn graph_nodes(&self, chain: Chain) -> Option<usize> {
        let i = chain_slot(chain).ok()?;
        self.graphs[i].as_ref().map(Graph::node_count)
    }

    /// The tier's fault copy as `page` holds it: what the tier's last fault
    /// copy saw.
    pub(super) fn fault_copy(&self, page: &TierPage) -> Option<Fault> {
        let (w, s) = (
            page.word(self.index, TWord::Fault).load(Ordering::Acquire),
            page.word_at(page.layout.word_off(self.index, TWord::Fault) + 4)
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

    /// Bind the card to `page` as tier `index`: its windows over the page's
    /// routed rows in its own context, and `stage`'s stream and fault word
    /// kept. The page was allocated in `stage`'s context, and the card's
    /// staging copies and its waits take the page's device address from
    /// there, so the page must sit at that address in this card's context too
    /// (unified addressing), else the call is refused by name; a page laid
    /// out for another shape is refused too. The host tier's load-time call;
    /// `stage`'s context is current on return.
    pub(super) fn bind(
        &mut self,
        page: &TierPage,
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
        let ctx = self.gpu.context();
        ctx.bind_to_thread()?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: this card's context is current on this thread (bound
        // above), `dev` is a live local the call writes, and the page's first
        // byte is the start of its live mapped allocation; the flags must be 0.
        let rc =
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, page.page.host_at(0).cast(), 0) };
        cu(rc, "cuMemHostGetDevicePointer_v2 (tier page, tier context)")?;
        if dev != page.page.dev_at(0) {
            return Err(GpuError::shape(
                BIND,
                format!(
                    "the tier page is at {:#x} in the stage card's context and at {dev:#x} in the \
                     tier {}'s: the tier's copies and the stage's waits need one address",
                    page.page.dev_at(0),
                    self.name
                ),
            ));
        }
        let img = layout.image;
        self.rows = (0..layout.rows)
            .map(|r| {
                // SAFETY: row r's routed rows (`n_used · cols · hidden` f32)
                // lie inside the page, apart from every other row's (the
                // layout's strides); the host tier frees the page only after
                // its cards.
                TierRows(unsafe {
                    window::<f32>(
                        page.page.dev_at(layout.rows_off(r)),
                        img.n_used * img.cols * img.hidden,
                        ctx,
                    )
                })
            })
            .collect();
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
        let Some(graph) = self.graphs[i].as_ref().filter(|_| self.captured[i] == list) else {
            return Ok(Pass::Feed);
        };
        self.gpu.context().bind_to_thread()?;
        graph.launch(self.gpu.stream())?;
        self.issue(list.len());
        self.rebind_stage()?;
        Ok(Pass::Graph)
    }

    /// Capture `chain`'s tier graph for `list` over `page`, replacing the
    /// one held, after a fed pass has settled: the stage card's chain waits
    /// on neither the host nor the tier any more. Nothing is launched.
    pub(super) fn capture(
        &mut self,
        page: &TierPage,
        chain: Chain,
        list: &[(usize, usize)],
    ) -> Result<(), GpuError> {
        let i = chain_slot(chain)?;
        let cols = self.chain_cols(chain)?;
        self.graphs[i] = None;
        self.captured[i].clear();
        let Some(last) = list.len().checked_sub(1) else {
            return Ok(());
        };
        self.gpu.context().bind_to_thread()?;
        let mut parts = self.parts(page);
        let graph = Graph::capture(parts.gpu.stream(), |_| {
            for (k, &(layer, row)) in list.iter().enumerate() {
                parts.enqueue_layer(layer, row, cols, k == last)?;
            }
            Ok(())
        })?;
        self.graphs[i] = Some(graph);
        self.captured[i] = list.to_vec();
        self.rebind_stage()
    }

    /// Enqueue layer `layer` of row `row` of `chain` on the tier's stream
    /// now, for an eager chain or a fed pass ([`Pass::Feed`]), over the
    /// chain's columns, its fault copy with it; the host then expects one
    /// layer more. A chain wider than the tier's rows is refused by name.
    pub(super) fn enqueue_eager(
        &mut self,
        page: &TierPage,
        chain: Chain,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let cols = self.chain_cols(chain)?;
        self.gpu.context().bind_to_thread()?;
        self.parts(page).enqueue_layer(layer, row, cols, true)?;
        self.issue(1);
        self.rebind_stage()
    }

    /// The parts one tier layer's enqueue reads and writes, over `page`.
    fn parts<'a>(&'a mut self, page: &'a TierPage) -> Parts<'a> {
        let TierCard {
            experts,
            module,
            act,
            sel,
            rows,
            index,
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
            index: *index,
            weights,
            gpu,
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

    /// The layers the tier has served since load, as its progress word on
    /// `page` says.
    pub(super) fn progress(&self, page: &TierPage) -> u32 {
        page.word(self.index, TWord::Prog).load(Ordering::Acquire)
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
            "the expert tier on {} served {} of the {} layers asked of it and signalled nothing more in {waited:.1?}: the card is lost",
            self.name,
            self.progress(page),
            self.issued
        )
    }

    /// Release every wait of both cards on this tier's words that is still
    /// pending, for good: the tier's counters and go counters go far past
    /// anything a step subtracts.
    pub(super) fn release(&self, page: &TierPage) {
        for r in 0..page.layout.rows {
            page.word(self.index, TWord::Go(r))
                .store(RELEASE, Ordering::Release);
            page.word(self.index, TWord::Cnt(r))
                .store(RELEASE, Ordering::Release);
        }
    }

    /// Whether the tier's words hold what a drained step leaves: no go and
    /// no signal pending, every layer asked of it served.
    pub(super) fn at_rest(&self, page: &TierPage) -> bool {
        (0..page.layout.rows).all(|r| {
            page.word(self.index, TWord::Go(r)).load(Ordering::Acquire) == 0
                && page.word(self.index, TWord::Cnt(r)).load(Ordering::Acquire) == 0
        }) && !self.behind(page)
    }

    /// Back to a fresh tier's words once the tier's stream has drained:
    /// counters and go counters at 0, the host's count at the tier's
    /// progress, the tier's fault word and its copy clean. The host tier's
    /// reset calls it.
    pub(super) fn reset(&mut self, page: &TierPage) -> Result<(), GpuError> {
        self.gpu.stream().synchronize()?;
        for r in 0..page.layout.rows {
            page.word(self.index, TWord::Go(r))
                .store(0, Ordering::Release);
            page.word(self.index, TWord::Cnt(r))
                .store(0, Ordering::Release);
        }
        self.issued = self.progress(page);
        self.clear_fault(page)
    }

    /// The tier's fault word and its copy back to clean.
    pub(super) fn clear_fault(&mut self, page: &TierPage) -> Result<(), GpuError> {
        self.gpu.clear_fault()?;
        self.gpu.stream().synchronize()?;
        page.word(self.index, TWord::Fault)
            .store(FAULT_NONE, Ordering::Release);
        page.word_at(page.layout.word_off(self.index, TWord::Fault) + 4)
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

/// The tier graph's slot of `chain`: the step 0, the pair 1, a
/// [`Chain::Cols`] of `m` columns `m` for `2..=DEFER_MAX_COLS`; any other
/// width is refused by name.
fn chain_slot(chain: Chain) -> Result<usize, GpuError> {
    match chain {
        Chain::Step => Ok(0),
        Chain::Pair => Ok(1),
        Chain::Cols(m) if (2..=DEFER_MAX_COLS).contains(&m) => Ok(m),
        Chain::Cols(m) => Err(GpuError::shape(
            WHAT,
            format!("a chain of {m} columns: the tier serves 2..={DEFER_MAX_COLS} a row"),
        )),
    }
}

const _: () = assert!(TIER_CHAINS == DEFER_MAX_COLS + 1);

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

/// The card's parts one tier layer's enqueue reads and writes.
struct Parts<'a> {
    experts: &'a mut dyn TierExperts,
    module: &'a tier_kernels::LoadedModule,
    act: &'a mut Staged,
    sel: &'a mut DeviceBuffer<u32>,
    rows: &'a mut [TierRows],
    page: &'a TierPage,
    index: usize,
    weights: &'a Weights,
    gpu: &'a Gpu,
}

impl Parts<'_> {
    /// Layer `layer` of row `row` on the tier's stream over the image's
    /// first `cols` columns: wait for the row's go and take it back, copy
    /// those columns of the row's image — the activation (the q8_1 codes and
    /// their scales, or the f32 values), this tier's places — into the
    /// staging, the architecture's launches into the row's routed rows, then
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
        let (page, t) = (self.page, self.index);
        let img = page.layout.image(t)?;
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
        let go = page.dev(t, TWord::Go(row));
        mem_batch(
            stream,
            &mut [op_wait_geq(go, 1), op_add(go, u32::MAX)],
            "cuStreamBatchMemOp_v2 (tier go)",
        )?;
        let at = page.layout.image_off(row);
        let staged = match &*self.act {
            Staged::Q8(a) => a.q3.len() * 2 == img.q3_words && a.d8.len() == img.d8_len,
            Staged::F32(x) => x.len() == img.x_len && img.q3_words + img.d8_len == 0,
        };
        if !staged || self.sel.len() != img.n_used * img.cols {
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
                copy_in(
                    stream,
                    a.q3.cu_deviceptr(),
                    page.page.dev_at(at + 4 * img.q3),
                    4 * part(img.q3_words),
                )?;
                copy_in(
                    stream,
                    a.d8.cu_deviceptr(),
                    page.page.dev_at(at + 4 * img.d8),
                    4 * part(img.d8_len),
                )?;
            }
            Staged::F32(x) => copy_in(
                stream,
                x.cu_deviceptr(),
                page.page.dev_at(at + 4 * img.x),
                4 * part(img.x_len),
            )?,
        }
        copy_in(
            stream,
            self.sel.cu_deviceptr(),
            page.page.dev_at(at + 4 * img.sel),
            4 * img.n_used * cols,
        )?;
        let r = &mut self.rows[row];
        self.experts.enqueue_layer(
            self.gpu,
            self.weights,
            layer,
            TierIo {
                act: self.act.input(),
                sel: self.sel,
                rows: &mut r.0,
                cols,
            },
        )?;
        if copy_fault {
            let ctx = self.gpu.context();
            // SAFETY: the fault copy's two words lie in this tier's flag
            // lines; the window is used for this launch only and never
            // dropped as a buffer (its raw parts are, below).
            let mut out = unsafe { window::<u32>(page.dev(t, TWord::Fault), 2, ctx) };
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
                op_add(page.dev(t, TWord::Cnt(row)), 1),
                op_add(page.dev(t, TWord::Prog), 1),
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
    // tier's context of at least `bytes`, `src` a span of the tier page
    // inside its allocation; both outlive every graph that captured the copy.
    let rc = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
    cu(rc, "cuMemcpyDtoDAsync_v2 (tier stage-in)")
}

#[cfg(test)]
mod tests {
    use super::{TIER_FLAGS, TWord, TierAct, TierLayout, TierSet, TierShape, chain_slot};
    use crate::host::slots::{HOST, Slot, SlotMap, TIER};
    use crate::host::step::Chain;

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
    /// 0, 64, 128, 192, 256 and 320, the images from byte 512.
    #[test]
    fn v41_tier_page_keeps_its_fields_apart() {
        let l = TierLayout::new(V41, 512, 32, 1).expect("V4.1's tier page");
        let i = l.image(0).expect("tier 0");
        assert_eq!((i.sel, i.q3, i.d8), (0, 64, 64 + 1024));
        assert!(i.sel + i.n_used <= i.q3 && i.q3 + i.q3_words <= i.d8);
        assert_eq!(l.payload_off(), 512);
        assert!(l.image_off(1) >= l.image_off(0) + 4 * i.words());
        assert!(l.rows_off(0) >= l.image_off(1) + 4 * i.words());
        assert!(l.rows_off(1) >= l.rows_off(0) + 4 * 6 * 4096);
        assert!(l.bytes().unwrap() >= l.rows_off(1) + 4 * 6 * 4096);
        let mut offs = vec![l.word_off(0, TWord::Prog), l.word_off(0, TWord::Fault)];
        for r in 0..2 {
            offs.extend([l.word_off(0, TWord::Go(r)), l.word_off(0, TWord::Cnt(r))]);
        }
        offs.sort_unstable();
        assert_eq!(offs, [0, 64, 128, 192, 256, 320]);
        assert!(offs.iter().all(|&o| o + 64 <= l.payload_off()));
        assert!(l.image(1).is_err());
        assert!(TierLayout::new(TierShape { rows: 3, ..V41 }, 512, 32, 1).is_err());
        assert!(TierLayout::new(V41, 512, 32, 0).is_err());
        assert!(TierLayout::new(V41, 512, 32, 9).is_err());
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
    /// graph slots are the step's, the pair's and one per width 2..=8.
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
        assert_eq!(chain_slot(Chain::Cols(4)).ok(), Some(4));
        assert!(chain_slot(Chain::Cols(1)).is_err());
        assert!(chain_slot(Chain::Cols(9)).is_err());
    }
}
