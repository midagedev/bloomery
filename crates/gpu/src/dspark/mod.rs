//! The DSpark draft on the card, common to every family that drafts with
//! one: [`DsparkBody`] runs fc over the target's tapped rows, keeps the
//! context ring's slots, stages a block, and runs the head chain (norm,
//! projection, the Markov loop, the draft-to-target id map) with one graph
//! per width. The layers between the embedded block and the head are a
//! family's ([`DraftLayers`]): V4.1's live in `bloomery-gpu-deepseek41`, the
//! dense Qwen3-shaped layer in [`dense`]. The draft's geometry is
//! [`bloomery_runtime::dspark::DsparkShape`].

use cuda_core::{CudaStream, DeviceBuffer};
use runtime::dspark::{DraftVocab, DsparkShape};

use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::{DeviceTensor, Gpu, GpuError, Graph, WindowMut};

pub mod dense;
pub mod head;
pub mod ring;
pub mod stage;

use head::{Head, HeadWeights, MARKOV_RANK};
use ring::Positions;
use stage::{Inbox, put_f32};

const WHAT: &str = "dspark";

/// Q8_0 rows on the card in the form the Q8_0 row kernels take
/// (`q8f32::enqueue_q8_0_gemv_mcol`): the quants and the f16 scales apart.
pub struct Q8Rows {
    /// The quants, four to a word, row-major.
    pub qs: DeviceTensor<u32>,
    /// One f16 scale (its bits) per 32 values.
    pub d: DeviceTensor<u16>,
}

/// fc over the taps and the hidden norm after it: rows of
/// `taps.len() * hidden` floats in, `hidden` out.
pub struct FcWeights {
    /// fc's rows.
    pub fc: Q8Rows,
    /// The hidden norm's gain.
    pub norm: DeviceTensor<f32>,
    /// The hidden norm's epsilon.
    pub eps: f32,
}

/// A family's draft layers: everything between the embedded block rows and
/// the head's input, and the context rows' append. Each call that launches
/// is preceded by its stage call, outside any capture, so the per-width and
/// append graphs replay with this call's positions. The owner records its
/// graphs at load, before any stage call: the launches read their staged
/// inputs from device addresses and stand from the layers' construction.
pub trait DraftLayers {
    /// The most rows a block holds: the widest block these layers' scratch
    /// takes. The owner records one block graph per row count in
    /// `1..=MAX_ROWS`.
    const MAX_ROWS: usize;

    /// The most context rows one append graph takes. The owner records one
    /// append graph per row count in `1..=APPEND_GROUP`; an append of more
    /// rows runs eager, a group at a time.
    const APPEND_GROUP: usize;

    /// The draft's hidden width: the rows [`DraftLayers::enqueue_block`]
    /// reads and writes.
    fn hidden(&self) -> usize;

    /// Stage the next [`DraftLayers::append_kv`]: `n` context rows at
    /// positions `first..first + n`. Never inside a capture.
    fn stage_kv(&mut self, gpu: &Gpu, first: u32, n: usize) -> Result<(), GpuError>;

    /// Append the staged context rows to every layer's KV. Each row of `x` is
    /// already fc over the taps, then the hidden norm,
    /// [`DraftLayers::hidden`] floats; the layer's own K/V projections and
    /// rotation are its own. Capturable.
    fn append_kv(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, n: usize) -> Result<(), GpuError>;

    /// Stage the next [`DraftLayers::enqueue_block`]: `m` rows after `last`,
    /// the token at the target's position `pos`, over `ctx` visible context
    /// rows. Under [`EmbedSource::Target`] the layers embed the rows here
    /// from `last` and the mask id; under [`EmbedSource::OwnHost`] the owner
    /// has written them to the block's input. Never inside a capture.
    fn stage_block(
        &mut self,
        gpu: &Gpu,
        last: u32,
        pos: u32,
        m: usize,
        ctx: usize,
    ) -> Result<(), GpuError>;

    /// The staged block pass: `m` rows through every layer, each attending to
    /// the visible context rows and to the block's rows under the file's
    /// mask. Leaves the last layer's output in `x`, `m` rows, before the
    /// head's norm. Capturable: the owner records one graph per `m`.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError>;

    /// Forget every context row: a new generation.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// Whether the captured graphs replay what the layers would enqueue now.
    /// The owner asks before every graph replay, block and append, and never
    /// before an eager pass. Layers whose captures bake in a setting a caller
    /// can change refuse by name; the default says the captures always hold.
    fn replayable(&self) -> Result<(), GpuError> {
        Ok(())
    }

    /// Card bytes these layers hold at `ctx` positions.
    fn card_bytes(&self, ctx: usize) -> u64;
}

/// Where the block's embedded rows come from.
#[derive(Debug)]
pub enum EmbedSource {
    /// The target's embedding rows: the layers embed them in
    /// [`DraftLayers::stage_block`].
    Target,
    /// The draft file's own table, kept on the host; the owner gathers the
    /// rows a pass needs there and widens them to f32.
    OwnHost(HostEmbed),
}

/// A draft's own embedding table on the host: `rows * cols` bf16 values,
/// row-major.
#[derive(Debug)]
pub struct HostEmbed {
    /// Ids the table holds (the target's vocabulary).
    pub rows: usize,
    /// The hidden width.
    pub cols: usize,
    /// The table's bf16 bits.
    pub bf16: Vec<u16>,
}

/// The target's output matrix as the draft's head borrows it: a copy on the
/// draft's card in the target file's type, each arm holding exactly its
/// planes. The owner runs the types it has a row kernel for and refuses any
/// other by name.
pub enum TargetHead {
    /// Q6_K superblocks: the matrix's words.
    Q6K(DeviceTensor<u32>),
    /// Q8_0 rows in the form the row kernel takes. Refused by name until a
    /// family's borrowed head needs it.
    Q8_0(Q8Rows),
}

/// Where the head's projection comes from.
pub enum HeadSource {
    /// The target's output matrix, borrowed (a draft over the target's full
    /// vocabulary).
    Target(TargetHead),
    /// The draft's own projection over its head's rows.
    Own(Q8Rows),
}

/// How a [`DsparkBody`] call reaches the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submit {
    /// A replay of the graph captured at load for the call's shape, after
    /// [`DraftLayers::replayable`] says the captures hold. An append of more
    /// than [`DraftLayers::APPEND_GROUP`] rows has no graph and runs eager
    /// either way.
    Graph,
    /// Every launch enqueued one by one: the gates' twin of a replay.
    Eager,
}

/// The graphs [`DsparkBody::new`] captures.
struct Graphs {
    /// The block pass of `m` rows at `m - 1`: the layers, the head and the
    /// Markov loop.
    block: Vec<Graph>,
    /// The append of `n` rows at `n - 1`: fc, the hidden norm and the layers'
    /// context append.
    append: Vec<Graph>,
}

/// Where the owner takes a block's embedded rows from, resolved at load.
enum Embed {
    /// The layers embed in [`DraftLayers::stage_block`].
    Layers,
    /// The owner gathers rows of the draft's own table.
    Host(HostEmbed),
}

/// The owner's own device buffers ([`DsparkBody::own_held`]), split as a
/// load's held count splits them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwnHeld {
    /// Every weight buffer, in bytes: fc's quants and scales, its norm,
    /// then the head's ([`Head::weight_buffers`]).
    pub weights: Vec<u64>,
    /// The working buffers, in all.
    pub working: u64,
}

/// The common draft over a family's layers `L`: fc over the target's tapped
/// rows, the context ring's positions, a block's staging, the head chain and
/// the per-shape graphs.
///
/// One sequence, in this order:
///
/// 1. [`DsparkBody::reset`]: no committed position.
/// 2. [`DsparkBody::append_as`] of the tapped rows in order from position 0,
///    or from `to` after [`DsparkBody::skip`]: a draft that keeps a window
///    needs only the last window's rows of a long prompt.
/// 3. [`DsparkBody::propose_as`] after the token at the committed position.
/// 4. The target's verify pass, then [`DsparkBody::append_as`] of the rows it
///    kept, then step 3 again.
///
/// A rejected proposal leaves nothing behind: only appended rows enter the
/// context. Every call enqueues on the engine stream; an append returns
/// without waiting and a proposal returns once its ids are on the host. A
/// call writes its inputs into a pinned image, copies it to the card in one
/// transfer right before the replay (outside the graph, so the graphs hold
/// kernels only), stages the layers ([`DraftLayers::stage_kv`],
/// [`DraftLayers::stage_block`]) and replays.
pub struct DsparkBody<L: DraftLayers> {
    /// Declared first: dropped before the weights and buffers they name.
    graphs: Graphs,
    layers: L,
    head: Head,
    fc: FcWeights,
    embed: Embed,
    shape: DsparkShape,
    positions: Positions,
    /// Floats of one tapped row: `taps.len() * hidden` of the target.
    feat: usize,
    /// The draft's hidden width.
    hidden: usize,
    /// The most rows one append takes.
    capacity: usize,
    /// The rows between fc and the head: an append's normed context rows,
    /// and a block's rows in and out of the layers.
    x: DeviceBuffer<f32>,
    /// fc's output before its norm, one group's rows.
    fc_out: DeviceBuffer<f32>,
    /// An append's tapped rows, token-major.
    feats: Inbox,
    /// A block's inputs: the id after which it runs at word 0, then under
    /// [`EmbedSource::OwnHost`] its `hidden`-wide embedded rows.
    block_in: Inbox,
}

impl<L: DraftLayers> DsparkBody<L> {
    /// Both counts at least 1: a draft that holds no block or no append has
    /// nothing to capture.
    const COUNTS: () = assert!(
        L::MAX_ROWS >= 1 && L::APPEND_GROUP >= 1,
        "DraftLayers::MAX_ROWS and APPEND_GROUP are at least 1"
    );

    /// The owner over `layers` for a draft of `shape`: its buffers, and one
    /// graph per block row count `1..=L::MAX_ROWS` and per append row count
    /// `1..=L::APPEND_GROUP`. Each shape and weight set a launch could not
    /// read is refused by name. Load-time only.
    ///
    /// The captures run [`DraftLayers::enqueue_block`] and
    /// [`DraftLayers::append_kv`] before any stage call, so the layers launch
    /// from construction.
    pub fn new(
        gpu: &Gpu,
        shape: &DsparkShape,
        layers: L,
        fc: FcWeights,
        head: HeadWeights,
        embed: EmbedSource,
    ) -> Result<DsparkBody<L>, GpuError> {
        let () = Self::COUNTS;
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        shape.check().map_err(|e| refuse(e.to_string()))?;
        if shape.markov_rank != MARKOV_RANK {
            return Err(refuse(format!(
                "a Markov rank of {}; the head's kernel takes {MARKOV_RANK}",
                shape.markov_rank
            )));
        }
        // The owner maps the head's ids through the shape's map and the head's
        // Markov step through its device map: the two must be one map.
        match (&shape.vocab, &head.to_target) {
            (DraftVocab::Full { .. }, None) => {}
            (DraftVocab::Reduced { to_target, n_target }, Some(map)) => {
                if head.markov_w1.rows() != *n_target {
                    return Err(refuse(format!(
                        "markov_w1 of {} rows for a reduced vocabulary over the target's {n_target} \
                         ids",
                        head.markov_w1.rows()
                    )));
                }
                if map.to_host_vec(gpu.stream())? != *to_target {
                    return Err(refuse(
                        "the head's draft-to-target map differs from the shape's".into(),
                    ));
                }
            }
            (DraftVocab::Full { .. }, Some(_)) => {
                return Err(refuse(
                    "a draft-to-target map on the head of a full vocabulary".into(),
                ));
            }
            (DraftVocab::Reduced { .. }, None) => {
                return Err(refuse(
                    "a reduced vocabulary whose head carries no draft-to-target map".into(),
                ));
            }
        }
        if shape.rows(1) > L::MAX_ROWS {
            return Err(refuse(format!(
                "a block of {} rows for the shortest proposal; the layers hold {}",
                shape.rows(1),
                L::MAX_ROWS
            )));
        }
        let hidden = layers.hidden();
        let feat = shape.taps.len() * shape.hidden;
        if hidden == 0 || !hidden.is_multiple_of(32) || !feat.is_multiple_of(32) {
            return Err(refuse(format!(
                "rows of {hidden} values from {feat} tapped: both multiples of 32"
            )));
        }
        let (qs, d) = (&fc.fc.qs, &fc.fc.d);
        if (qs.rows(), qs.cols()) != (hidden, feat / 4)
            || (d.rows(), d.cols()) != (hidden, feat / 32)
            || fc.norm.buf().len() != hidden
        {
            return Err(refuse(format!(
                "fc of {}x{} words and {}x{} scales, a norm of {} values; {hidden} rows over \
                 {feat} tapped values need {hidden}x{} words, {hidden}x{} scales, {hidden} gains",
                qs.rows(),
                qs.cols(),
                d.rows(),
                d.cols(),
                fc.norm.buf().len(),
                feat / 4,
                feat / 32
            )));
        }
        let embed = match embed {
            EmbedSource::Target => Embed::Layers,
            EmbedSource::OwnHost(table) => {
                let rows = shape.vocab.n_target();
                if table.rows != rows
                    || table.cols != hidden
                    || table.bf16.len() != rows.saturating_mul(hidden)
                {
                    return Err(refuse(format!(
                        "an embedding table of {}x{} ({} values); {rows} ids of {hidden} values \
                         need {}",
                        table.rows,
                        table.cols,
                        table.bf16.len(),
                        rows.saturating_mul(hidden)
                    )));
                }
                Embed::Host(table)
            }
        };
        let head = Head::new(gpu, head, hidden, L::MAX_ROWS)?;
        if head.n_vocab() != shape.vocab.rows() {
            return Err(refuse(format!(
                "a projection of {} rows for a draft vocabulary of {}",
                head.n_vocab(),
                shape.vocab.rows()
            )));
        }
        let group = L::APPEND_GROUP;
        let capacity = shape.window.map_or(group, |w| w.div_ceil(group) * group);
        let block_words = 1 + match embed {
            Embed::Layers => 0,
            Embed::Host(_) => L::MAX_ROWS * hidden,
        };
        let s = gpu.stream();
        let x = DeviceBuffer::zeroed(s, capacity.max(L::MAX_ROWS) * hidden)?;
        let fc_out = DeviceBuffer::zeroed(s, group * hidden)?;
        let feats = Inbox::new(gpu, capacity * feat)?;
        let block_in = Inbox::new(gpu, block_words)?;
        let mut body = DsparkBody {
            graphs: Graphs {
                block: Vec::new(),
                append: Vec::new(),
            },
            layers,
            head,
            fc,
            embed,
            shape: shape.clone(),
            positions: Positions::new(shape.window)?,
            feat,
            hidden,
            capacity,
            x,
            fc_out,
            feats,
            block_in,
        };
        for m in 1..=L::MAX_ROWS {
            let g = gpu.capture(|_| body.enqueue_block(gpu, m))?;
            body.counted(&g, Head::launches(m))?;
            body.graphs.block.push(g);
        }
        for n in 1..=group {
            let g = gpu.capture(|_| body.enqueue_append(gpu, n))?;
            body.counted(&g, body.fc_launches(n))?;
            body.graphs.append.push(g);
        }
        Ok(body)
    }

    /// Refuse a capture with fewer nodes than the owner's own launches: the
    /// layers' launches are theirs to count, the owner's are not optional.
    fn counted(&self, g: &Graph, own: usize) -> Result<(), GpuError> {
        if g.node_count() >= own {
            return Ok(());
        }
        Err(GpuError::shape(
            WHAT,
            format!(
                "a capture of {} nodes holds fewer than the owner's {own} launches",
                g.node_count()
            ),
        ))
    }

    /// Enqueue a block of `m` rows: the layers, then the head's norm,
    /// projection and Markov loop. Capturable.
    fn enqueue_block(&mut self, gpu: &Gpu, m: usize) -> Result<(), GpuError> {
        self.layers.enqueue_block(gpu, &mut self.x, m)?;
        self.head.enqueue_logits(gpu, &self.x, m)?;
        let first = self.block_in.u32s(0, 1)?;
        self.head.enqueue_markov(gpu, m, &first)
    }

    /// Enqueue an append of `n` rows over the staged tapped rows: fc and the
    /// hidden norm a group at a time into `x`, then the layers' append.
    /// Capturable.
    fn enqueue_append(&mut self, gpu: &Gpu, n: usize) -> Result<(), GpuError> {
        let s = gpu.stream();
        let (group, hidden) = (L::APPEND_GROUP, self.hidden);
        for r0 in (0..n).step_by(group) {
            let rows = group.min(n - r0);
            let feat = self.feats.f32s(r0 * self.feat, rows * self.feat)?;
            gpu.q8f32().enqueue_q8_0_gemv_mcol(
                s,
                Q8_0GemvMcolArgs {
                    qs: &self.fc.fc.qs,
                    d: &self.fc.fc.d,
                    x: &feat,
                    m: rows,
                    out: GemvOut::TokenMajor,
                    y: &mut self.fc_out,
                },
            )?;
            let mut normed = WindowMut::of_mut(&mut self.x, 4 * r0 * hidden, rows * hidden)?;
            gpu.elem().enqueue_rms_norm(
                s,
                &self.fc_out,
                self.fc.norm.buf(),
                self.fc.eps,
                hidden,
                rows,
                &mut normed,
            )?;
        }
        self.layers.append_kv(gpu, &self.x, n)
    }

    /// Commit positions before `to` without their rows: a reader that keeps
    /// only a window's rows skips the older ones. A block that would read a
    /// skipped position before appended rows overwrite it is refused by
    /// [`DsparkBody::propose_as`]. Refused for a `to` below the committed
    /// position.
    pub fn skip(&mut self, to: u32) -> Result<(), GpuError> {
        self.positions.skip(to)
    }

    /// Context rows from tapped target rows (`shape.taps.len() * shape.hidden`
    /// floats each, at most a window of rows), the first at position `first`,
    /// which is the committed position. Returns the rows appended. Enqueues on the engine stream
    /// and returns without waiting.
    pub fn append_as(&mut self, gpu: &Gpu, first: u32, taps: &[f32]) -> Result<usize, GpuError> {
        self.append_as_with(gpu, first, taps, Submit::Graph)
    }

    /// [`DsparkBody::append_as`], submitted as `submit` says. An append of
    /// more than [`DraftLayers::APPEND_GROUP`] rows has no graph and runs
    /// eager either way.
    pub fn append_as_with(
        &mut self,
        gpu: &Gpu,
        first: u32,
        taps: &[f32],
        submit: Submit,
    ) -> Result<usize, GpuError> {
        let n = taps.len() / self.feat;
        // Rows past the window would stand in slots the same call's earlier
        // rows hold.
        let limit = self
            .shape
            .window
            .map_or(self.capacity, |w| w.min(self.capacity));
        if n == 0 || n * self.feat != taps.len() || n > limit {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} values are not 1..={limit} rows of {}",
                    taps.len(),
                    self.feat
                ),
            ));
        }
        self.positions.check_append(first, n)?;
        let graph = match submit {
            Submit::Graph => self.graphs.append.get(n - 1),
            Submit::Eager => None,
        };
        if graph.is_some() {
            self.layers.replayable()?;
        }
        let s = gpu.stream();
        put_f32(&mut self.feats.host_mut()?[..n * self.feat], taps);
        self.feats.upload(s, n * self.feat)?;
        self.layers.stage_kv(gpu, first, n)?;
        match graph {
            Some(g) => g.launch(s)?,
            None => self.enqueue_append(gpu, n)?,
        }
        self.positions.append(first, n)?;
        Ok(n)
    }

    /// One block after `last`, the token at position `pos` (the committed
    /// position): it writes `width` ids in the target's vocabulary to the
    /// front of `out`, and when `p` is not empty each id's probability under
    /// the draft, after the Markov term, to the front of `p`. `width` is in
    /// `1..=shape.ids_per_block()`, and its block rows within
    /// [`DraftLayers::MAX_ROWS`]. An empty `p` asks for no probabilities and
    /// launches nothing for them. Blocks until the ids are on the host.
    /// Returns the ids written.
    pub fn propose_as(
        &mut self,
        gpu: &Gpu,
        pos: u32,
        last: u32,
        width: usize,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, GpuError> {
        self.propose_as_with(gpu, pos, last, width, out, p, Submit::Graph)
    }

    /// [`DsparkBody::propose_as`], submitted as `submit` says.
    #[allow(
        clippy::too_many_arguments,
        reason = "the contract's propose_as arguments and the submission: 8 with self, where R8 asks a struct past 8"
    )]
    pub fn propose_as_with(
        &mut self,
        gpu: &Gpu,
        pos: u32,
        last: u32,
        width: usize,
        out: &mut [u32],
        p: &mut [f32],
        submit: Submit,
    ) -> Result<usize, GpuError> {
        self.positions.check_block(pos)?;
        let refuse = |detail: String| GpuError::shape(WHAT, detail);
        if width == 0 || width > self.shape.ids_per_block() {
            return Err(refuse(format!(
                "a proposal of {width} ids; 1..={}",
                self.shape.ids_per_block()
            )));
        }
        let m = self.shape.rows(width);
        if m > L::MAX_ROWS {
            return Err(refuse(format!(
                "a proposal of {width} ids is a block of {m} rows; the layers hold {}",
                L::MAX_ROWS
            )));
        }
        if out.len() < width || (!p.is_empty() && p.len() < width) {
            return Err(refuse(format!(
                "{} ids and {} probabilities of room for a proposal of {width}",
                out.len(),
                p.len()
            )));
        }
        let n_target = self.shape.vocab.n_target();
        if usize::try_from(last).map_or(true, |l| l >= n_target) {
            return Err(refuse(format!(
                "the token {last} before a block, past the target's {n_target} ids"
            )));
        }
        let graph = match submit {
            Submit::Graph => self.graphs.block.get(m - 1),
            Submit::Eager => None,
        };
        if graph.is_some() {
            self.layers.replayable()?;
        }
        let s = gpu.stream();
        let host = self.block_in.host_mut()?;
        host[0] = last;
        let words = match &self.embed {
            Embed::Layers => 1,
            Embed::Host(table) => {
                let rows = 1 + m * self.hidden;
                put_embedded(&mut host[1..rows], table, last, self.shape.mask_id);
                rows
            }
        };
        self.block_in.upload(s, words)?;
        self.layers
            .stage_block(gpu, last, pos, m, self.positions.ring_rows())?;
        if let Embed::Host(_) = self.embed {
            let rows = self.block_in.f32s(1, m * self.hidden)?;
            let mut x = WindowMut::of_mut(&mut self.x, 0, m * self.hidden)?;
            x.copy_from_device_async(&rows, s)?;
        }
        match graph {
            Some(g) => g.launch(s)?,
            None => self.enqueue_block(gpu, m)?,
        }
        let asked = !p.is_empty();
        let (ids, probs) = if asked {
            self.head.enqueue_p(gpu, m)?;
            let (ids, probs) = self.head.tokens_p(s, m)?;
            (ids, Some(probs))
        } else {
            (self.head.tokens(s, m)?, None)
        };
        // A row 0 that is only the anchor drafts no id: the ids are the last
        // `width` rows'.
        let skip = m - width;
        for (i, &d) in ids[skip..].iter().enumerate() {
            out[i] = self
                .shape
                .vocab
                .target(d)
                .map_err(|e| refuse(e.to_string()))?;
        }
        if let Some(probs) = probs {
            p[..width].copy_from_slice(&probs[skip..]);
        }
        Ok(width)
    }

    /// The gate's seat: commit `committed` positions as if that many rows
    /// had been appended, with no launch, for a gate that seats the layers'
    /// context from an oracle ([`DsparkBody::layers_mut`]) instead of
    /// appending it. Refused by name unless nothing is committed, and for 0.
    pub fn seat_positions(&mut self, committed: u32) -> Result<(), GpuError> {
        let held = self.positions.committed();
        if held != 0 || committed == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a seat of {committed} positions after {held} committed; a seat takes a \
                     reset draft and 1.. positions"
                ),
            ));
        }
        let n = usize::try_from(committed)
            .map_err(|_| GpuError::shape(WHAT, format!("a seat of {committed} positions")))?;
        self.positions.append(0, n)
    }

    /// Forget the sequence: no committed position, the layers' context
    /// cleared. Waits for the card.
    pub fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.layers.reset(gpu)?;
        gpu.stream().synchronize()?;
        self.positions.reset();
        Ok(())
    }

    /// Positions committed so far.
    #[must_use]
    pub fn committed(&self) -> u32 {
        self.positions.committed()
    }

    /// Card bytes the draft holds at `ctx` positions: the layers', and the
    /// owner's weights and buffers. Captured graphs are not counted.
    #[must_use]
    pub fn card_bytes(&self, ctx: usize) -> u64 {
        let own = self.own_held();
        self.layers.card_bytes(ctx) + own.weights.iter().sum::<u64>() + own.working
    }

    /// The owner's own device buffers, counted from the buffers themselves
    /// where they live: fc and the head's weights, the row buffers, the
    /// inboxes' card sides and the head's scratch. The layers' are theirs.
    #[must_use]
    pub fn own_held(&self) -> OwnHeld {
        let mut weights = vec![
            self.fc.fc.qs.buf().num_bytes() as u64,
            self.fc.fc.d.buf().num_bytes() as u64,
            self.fc.norm.buf().num_bytes() as u64,
        ];
        weights.extend(self.head.weight_buffers());
        let working = (self.x.num_bytes()
            + self.fc_out.num_bytes()
            + self.feats.device_bytes()
            + self.block_in.device_bytes()) as u64
            + self.head.scratch_bytes();
        OwnHeld { weights, working }
    }

    /// The layers.
    #[must_use]
    pub fn layers(&self) -> &L {
        &self.layers
    }

    /// The layers, writable: the gates' way to seat an oracle's rows. The
    /// graphs read the layers' buffers in place, so a replay sees what is
    /// seated.
    pub fn layers_mut(&mut self) -> &mut L {
        &mut self.layers
    }

    /// The head, for the gates' readouts.
    #[must_use]
    pub fn head(&self) -> &Head {
        &self.head
    }

    /// The head, writable: the gates' way to run its launches over rows of
    /// their own.
    pub fn head_mut(&mut self) -> &mut Head {
        &mut self.head
    }

    /// The `m` rows the layers left in the owner's `x`, before the head's
    /// norm, token-major. Blocking; gate use.
    pub fn block_rows_to_host(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        if !(1..=L::MAX_ROWS).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a readout of {m} rows; the layers hold 1..={}", L::MAX_ROWS),
            ));
        }
        Ok(self.x.to_host_vec(stream)?[..m * self.hidden].to_vec())
    }

    /// The captured block pass of `m` rows.
    #[must_use]
    pub fn graph_block(&self, m: usize) -> Option<&Graph> {
        m.checked_sub(1).and_then(|i| self.graphs.block.get(i))
    }

    /// The captured append of `n` rows.
    #[must_use]
    pub fn graph_append(&self, n: usize) -> Option<&Graph> {
        n.checked_sub(1).and_then(|i| self.graphs.append.get(i))
    }

    /// Launches of the head over a block of `m` rows: the norm, the
    /// quantizer, the projection and the Markov loop's two a row.
    #[must_use]
    pub fn head_launches(&self, m: usize) -> usize {
        Head::launches(m)
    }

    /// Launches of the owner's part of an append of `n` rows: fc and the
    /// hidden norm for each group of rows.
    #[must_use]
    pub fn fc_launches(&self, n: usize) -> usize {
        2 * n.div_ceil(L::APPEND_GROUP)
    }
}

/// A block's `m` embedded rows as f32 bits into `host` (`m · cols` words):
/// row 0 is the table's row of `last`, the others the mask token's. bf16
/// widens to f32 exactly, as `bits << 16`.
fn put_embedded(host: &mut [u32], table: &HostEmbed, last: u32, mask: u32) {
    let cols = table.cols;
    let row = |id: u32| {
        let at = id as usize * cols;
        &table.bf16[at..at + cols]
    };
    for (i, out) in host.chunks_exact_mut(cols).enumerate() {
        let src = row(if i == 0 { last } else { mask });
        for (w, &b) in out.iter_mut().zip(src) {
            *w = u32::from(b) << 16;
        }
    }
}
