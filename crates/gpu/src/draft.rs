//! The types a draft walk is given: how it runs ([`WalkMode`]), which
//! projection its head runs ([`DraftHead`]), where its hidden rows come from
//! ([`Hidden`]), the rows it walks ([`Feed`], [`DraftFeed`]), the positions
//! its target arenas hold ([`Held`]) and the names of its row bounds
//! ([`WalkBounds`]). Each draft program — Qwen3.8's MTP layer and GLM's
//! NextN layer — walks these.
//!
//! The walk of several resident sequences' rows at once is this module's
//! too ([`SlotDraft`]): [`GpuModel::draft_store_slots`] runs one store walk
//! of every slot's rows, and [`GpuModel::draft_chain_slots`] a round of
//! every slot's window chains, each depth one walk over every chain that
//! reaches it ([`ChainBlock`]) and the round read back once. The program
//! gives the phases: each sequence's rows checked and staged, the row-wise
//! launches over all of them, each sequence's append and attention over its
//! own store, the head and the readback.

use crate::media::CallRows;
use crate::model::slots::slot_twice;
use crate::model::{GpuModel, MAX_PASS_ROWS, Slots, StepMode};
use crate::weights::Weights;
use crate::{Gpu, GpuError};

/// Copy one final hidden row to row 0 of a sequence-owned arena, ordered on
/// `stream`. The caller validates the source's position record first.
pub fn retain_hidden_row(
    stream: &cuda_core::CudaStream,
    source: &cuda_core::DeviceBuffer<f32>,
    first: usize,
    width: usize,
    target: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let byte_off = first
        .checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| GpuError::shape("draft::retain_hidden_row", "hidden row offset overflow"))?;
    let source = crate::tensor::Window::<f32>::of(source, byte_off, width)?;
    let mut target = crate::tensor::WindowMut::<f32>::of_mut(target, 0, width)?;
    target.copy_from_device_async(&source, stream)?;
    Ok(())
}

/// How a walk runs: the program's own walk, or its chain of walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkMode {
    /// Enqueued launch by launch.
    Eager,
    /// Replayed from its capture, captured on first use.
    Graph,
    /// Enqueued launch by launch through the store's append and no further:
    /// the rows' keys and values, which are all a later row reads of them (a
    /// row's input is its token and the target's hidden row, never the
    /// draft's output). No flash, feed-forward block or head, no readback
    /// and no own row: the prompt's warmup. The store's bits are a
    /// [`WalkMode::Eager`] walk's. A body with no such walk refuses it by
    /// name; a chain never takes it.
    Store,
}

impl From<StepMode> for WalkMode {
    /// A step's mode as a walk's: the draft's walks with a head run as the
    /// target's steps do.
    fn from(m: StepMode) -> WalkMode {
        match m {
            StepMode::Eager => WalkMode::Eager,
            StepMode::Graph => WalkMode::Graph,
        }
    }
}

/// Which projection a draft's head runs. Whether the load opened a row list
/// is the load's fact; a program that opens none holds [`DraftHead::Full`]
/// alone and refuses the other by name ([`DraftHead::check_opened`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftHead {
    /// The target's `output`: every token of the vocabulary.
    Full,
    /// The load's row list: the target's `output` rows of its ids.
    Rows,
}

impl DraftHead {
    /// Refused by name: the row-list head on a load that opened the full head
    /// alone (`opened_rows` false).
    ///
    /// # Errors
    /// [`GpuError::State`] naming `what` for [`DraftHead::Rows`] when the load
    /// opened no row list.
    pub fn check_opened(self, what: &'static str, opened_rows: bool) -> Result<(), GpuError> {
        if self == DraftHead::Rows && !opened_rows {
            return Err(GpuError::state(
                what,
                "a row-list head: this draft was opened with the full head",
            ));
        }
        Ok(())
    }
}

/// Where a walk's hidden rows come from; `A` is the arena kind the caller's
/// body names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hidden<'a, A> {
    /// Rows written from the host, as many values a row as the caller's body
    /// says: the row at position 0 reads a zero row, the target holding
    /// nothing before it.
    Host(&'a [f32]),
    /// The target's final hidden rows in the arena `walk`, from its row
    /// `first` on, as the target's last call of that arena left them.
    Target { walk: A, first: usize },
}

/// A walk's rows: `tokens` at positions `pos0 ..`, each beside its hidden
/// row (the target's at the position before it). `media` is the prompt
/// call's span rows by absolute position: a row the lookup holds replaces
/// the token's embedding row at its position, as the target's media call
/// took it. Only a prompt call that holds an image carries it; a body whose
/// walk cannot take it refuses it by name.
#[derive(Clone, Copy, Debug)]
pub struct Feed<'a, A> {
    pub tokens: &'a [u32],
    pub pos0: u32,
    pub hidden: Hidden<'a, A>,
    pub media: Option<&'a CallRows<'a>>,
}

/// What a draft program is fed one walk: the rows of a [`Feed`], or the
/// draft's own last row.
#[derive(Clone, Copy, Debug)]
pub enum DraftFeed<'a, A> {
    /// `tokens` at positions `pos0 ..`, each beside its hidden row.
    Rows(Feed<'a, A>),
    /// One row at `pos0`: the last walk's last row's token and streams. A
    /// program whose proposal holds one id has no such walk and refuses it by
    /// name ([`DraftFeed::target_rows`]).
    Own { pos0: u32 },
}

impl<'a, A> DraftFeed<'a, A> {
    /// The position of the feed's first row.
    #[must_use]
    pub fn pos0(&self) -> u32 {
        match self {
            DraftFeed::Rows(f) => f.pos0,
            DraftFeed::Own { pos0 } => *pos0,
        }
    }

    /// How many rows the feed walks: its tokens, or the one own row.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            DraftFeed::Rows(f) => f.tokens.len(),
            DraftFeed::Own { .. } => 1,
        }
    }

    /// The feed's rows, for a program that walks the target's rows alone: the
    /// own row is refused by name ([`refuse_own`]).
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming `what` for [`DraftFeed::Own`].
    pub fn target_rows(self, what: &'static str) -> Result<Feed<'a, A>, GpuError> {
        match self {
            DraftFeed::Rows(f) => Ok(f),
            DraftFeed::Own { pos0 } => Err(refuse_own(
                what,
                &format!("a walk of the draft's own row at position {pos0}"),
            )),
        }
    }
}

/// The refusal by name of the draft's own row ([`DraftFeed::Own`]) by a
/// program whose walks read the target's hidden rows alone, `detail` naming
/// the call.
#[must_use]
pub fn refuse_own(what: &'static str, detail: &str) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "{detail}: this program's walk reads the target's hidden rows and its proposal holds \
             one id, so it has no own row (DraftFeed::Own)"
        ),
    )
}

/// The names of a draft program's row bounds: the most rows one walk takes
/// in each mode. A program implements it with its own values; the names, and
/// the refusal of a walk past them ([`refuse_rows`]), are this module's.
pub trait WalkBounds {
    /// The most rows one eager walk takes.
    const WALK_ROWS: usize;
    /// The most rows a captured walk takes; 0 for a program that captures
    /// none.
    const GRAPH_ROWS: usize;
    /// The most rows one store walk ([`WalkMode::Store`]) takes.
    const STORE_ROWS: usize;
}

/// The most rows one walk of `P` takes in `mode`.
#[must_use]
pub const fn most_rows<P: WalkBounds>(mode: WalkMode) -> usize {
    match mode {
        WalkMode::Eager => P::WALK_ROWS,
        WalkMode::Graph => P::GRAPH_ROWS,
        WalkMode::Store => P::STORE_ROWS,
    }
}

/// Refused by name: a walk of `m` rows outside `1..=` [`most_rows`] of `P`'s
/// `mode`.
///
/// # Errors
/// [`GpuError::Shape`] naming `what`.
pub fn refuse_rows<P: WalkBounds>(
    what: &'static str,
    mode: WalkMode,
    m: usize,
) -> Result<(), GpuError> {
    let most = most_rows::<P>(mode);
    if !(1..=most).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("a walk of {m} rows; the {mode:?} walk takes 1..={most}"),
        ));
    }
    Ok(())
}

/// The positions a target arena's rows hold, row `r` position `first + r`:
/// what the call that last wrote the arena left there, less the positions a
/// cut took back since. No rows: the arena holds no position of the
/// sequence. A [`Hidden::Target`] feed must read the positions recorded here
/// rather than stale rows left in the arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Held {
    first: u32,
    rows: u32,
}

impl Held {
    /// No position.
    pub const NONE: Held = Held { first: 0, rows: 0 };

    /// Rows `0 .. rows` at positions `first ..`. Construction does not read
    /// rows; [`Held::holds`] refuses a read past the position width.
    #[must_use]
    pub const fn at(first: u32, rows: u32) -> Held {
        Held { first, rows }
    }

    /// The positions from `pos` on taken back.
    pub fn cut(&mut self, pos: u32) {
        self.rows = self.rows.min(pos.saturating_sub(self.first));
    }

    /// The first position and the number of rows it holds.
    #[must_use]
    pub const fn parts(self) -> (u32, u32) {
        (self.first, self.rows)
    }

    /// Whether rows `row .. row + m` hold positions `at .. at + m`.
    /// Row arithmetic must fit `usize`, and every read position must fit
    /// `u32`. An empty range may name the boundary just after the held rows.
    #[must_use]
    pub fn holds(self, row: usize, m: usize, at: u32) -> bool {
        let Some(end) = row.checked_add(m) else {
            return false;
        };
        let (Ok(row), Ok(m), Ok(end)) = (u32::try_from(row), u32::try_from(m), u32::try_from(end))
        else {
            return false;
        };
        end <= self.rows
            && self.first.checked_add(row) == Some(at)
            && (m == 0 || at.checked_add(m - 1).is_some())
    }

    /// The positions held, for a refusal's words.
    #[must_use]
    pub fn shown(self) -> String {
        match self.rows {
            0 => "no position".to_string(),
            rows => format!(
                "positions {}..{}",
                self.first,
                u64::from(self.first) + u64::from(rows)
            ),
        }
    }

    /// Check the target rows of a [`Hidden::Target`] feed before any copy.
    /// `rows` is the arena's row capacity, and `pos0` the walk's first token
    /// position: its hidden rows must start at position `pos0 - 1`.
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming `what` and `walk` for rows outside the
    /// arena, position 0 (which needs a host zero row), or positions the
    /// arena does not hold, including overflowing row or position ranges.
    pub fn check_read(
        self,
        what: &'static str,
        walk: impl std::fmt::Debug,
        rows: usize,
        (first, m): (usize, usize),
        pos0: u32,
    ) -> Result<(), GpuError> {
        let Some(end) = first.checked_add(m).filter(|&end| end <= rows) else {
            return Err(GpuError::shape(
                what,
                format!("{m} rows from row {first} of the {walk:?} arena's {rows}"),
            ));
        };
        let at = pos0.checked_sub(1).ok_or_else(|| {
            GpuError::shape(
                what,
                format!(
                    "the {walk:?} arena's rows for a walk at position 0: the target \
                     holds no row before it (a host zero row is position 0's)"
                ),
            )
        })?;
        if !self.holds(first, m, at) {
            return Err(GpuError::shape(
                what,
                format!(
                    "rows {first}..{end} of the {walk:?} arena as positions {at}..{} for \
                     a walk from {pos0}: the arena holds {}",
                    u128::from(at) + m as u128,
                    self.shown()
                ),
            ));
        }
        Ok(())
    }
}

/// A walk's readback: each row's token and, for a program whose readback
/// holds it, the draft's probability of it among the head's rows (empty
/// for one whose readback holds the ids alone).
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    pub p: Vec<f32>,
}

/// One slot's chain of a round of several slots' chains
/// ([`GpuModel::draft_chain_slots`]): the slot, the refresh, the own walks.
pub type SlotChain<'a, A> = (usize, Feed<'a, A>, usize);

/// One sequence's chain in a round of several sequences' chains
/// ([`SlotDraft::stage_chain`]): the round's block `block`, its refresh's
/// arena rows `at .. at + m` from position `pos0`, then `own` own walks, and
/// the first of its readback pairs (one a walk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainBlock {
    pub block: usize,
    pub at: usize,
    pub m: usize,
    pub pos0: u32,
    pub own: usize,
    pub pair: usize,
}

impl ChainBlock {
    /// Positions of the sequence its store holds after the chain: through
    /// the refresh.
    #[must_use]
    pub fn end(&self) -> usize {
        self.pos0 as usize + self.m
    }

    /// The pairs its walks fill: one a walk.
    #[must_use]
    pub fn pairs(&self) -> usize {
        1 + self.own
    }
}

/// The rows walk `j` of a round of `blocks`' chains runs, a block's at a
/// time for every block whose chain reaches it, in `blocks`' order, each
/// with its first arena row and its rows: walk 0 each block's refresh at
/// its own rows, a later walk each block's one own row at the next arena
/// row.
pub fn chain_rows(
    blocks: &[ChainBlock],
    j: usize,
) -> impl Iterator<Item = (ChainBlock, usize, usize)> + '_ {
    blocks
        .iter()
        .filter(move |b| b.own >= j)
        .enumerate()
        .map(move |(next, &b)| if j == 0 { (b, b.at, b.m) } else { (b, next, 1) })
}

/// A [`Slots`] body whose draft walks several resident sequences' rows at
/// once, over the target's arenas `A`: the phases [`GpuModel`]'s
/// [`GpuModel::draft_store_slots`] and [`GpuModel::draft_chain_slots`] run.
/// A phase marked "live" acts on the live sequence, which the round makes
/// each block's in turn; the others run once over every block. Each launch
/// computes a row from that row's inputs alone, so a sequence's store and
/// proposal are those its own walks leave.
pub trait SlotDraft<A>: Slots {
    /// The draft program whose bounds a walk of several sequences holds to:
    /// a store walk its [`WalkBounds::STORE_ROWS`], a round's walk its
    /// [`WalkBounds::WALK_ROWS`].
    type Program: WalkBounds;
    /// The most readback pairs (one a chain a walk) a round reads back.
    const ROUND_PAIRS: usize;

    /// One sequence's store rows staged at rows `at ..` of the store walk
    /// (live): `feed` checked as a store walk into `head` and its hidden rows
    /// copied in; nothing but the walk's scratch moves.
    type StoreBlock<'f>: Copy;

    /// See [`SlotDraft::StoreBlock`].
    fn stage_store<'f>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        feed: Feed<'f, A>,
        head: DraftHead,
        at: usize,
    ) -> Result<Self::StoreBlock<'f>, GpuError>;

    /// The store walk's row-wise launches once over every block, the rows'
    /// tokens `ids` in the blocks' order.
    fn store_rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        ids: &[u32],
        blocks: &[Self::StoreBlock<'_>],
    ) -> Result<(), GpuError>;

    /// Block `b`'s rows appended to its sequence's store (live).
    fn append_store(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        b: Self::StoreBlock<'_>,
    ) -> Result<(), GpuError>;

    /// One sequence's chain checked and staged (live): `walk` its `(own,
    /// head, mode)`, `refresh`'s rows then `own` own walks into `head`, as
    /// the round's `(block, at, pair)`: block `block` at arena rows `at ..`,
    /// its readback pairs from `pair`. Refused by name as the
    /// program's chain refuses it; nothing but the round's scratch moves.
    fn stage_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        refresh: Feed<'_, A>,
        walk: (usize, DraftHead, WalkMode),
        at: (usize, usize, usize),
    ) -> Result<ChainBlock, GpuError>;

    /// The round's staged host words copied to the card ahead of its walks.
    fn upload_chains(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// Walk `j` of the round up to its attention, over the rows
    /// [`chain_rows`] names.
    fn chain_in(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        blocks: &[ChainBlock],
        j: usize,
    ) -> Result<(), GpuError>;

    /// A walk's `rows`, `(at, m)`: arena rows `at .. at + m` of block `b`,
    /// over its own store (live).
    fn attend(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        b: &ChainBlock,
        rows: (usize, usize),
    ) -> Result<(), GpuError>;

    /// Walk `j` after its attention, through the head, each block's last
    /// row's readback pair filled; the round's last walk closes the
    /// readback with the card's fault word.
    fn chain_out(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        blocks: &[ChainBlock],
        j: usize,
        head: DraftHead,
    ) -> Result<(), GpuError>;

    /// The round ended for `b`'s sequence (live): its store holds the
    /// sequence through the refresh ([`ChainBlock::end`]).
    fn end_chain(&mut self, b: &ChainBlock) -> Result<(), GpuError>;

    /// The round read back once: each block's proposal, in `blocks`' order.
    /// A fault any walk of the round raised is [`GpuError::Fault`].
    fn read_chains(&mut self, gpu: &Gpu, blocks: &[ChainBlock]) -> Result<Vec<Proposal>, GpuError>;
}

/// `f` on `b` with `seq` live, exchanged in and back out
/// ([`Slots::swap_seq`]), or with the live sequence when `None`.
fn on_seq<B: Slots, T>(
    b: &mut B,
    gpu: &Gpu,
    seq: Option<&mut B::Seq>,
    f: impl FnOnce(&mut B) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    match seq {
        None => f(b),
        Some(seq) => {
            b.swap_seq(gpu, seq)?;
            let r = f(b);
            b.swap_seq(gpu, seq)?;
            r
        }
    }
}

/// Refused by name: a slot named twice among `slots`.
fn refuse_twice(
    what: &'static str,
    slots: impl Iterator<Item = usize> + Clone,
) -> Result<(), GpuError> {
    for (i, slot) in slots.clone().enumerate() {
        if slots.clone().take(i).any(|s| s == slot) {
            return Err(slot_twice(what, slot));
        }
    }
    Ok(())
}

/// How many chains of `sizes` (each chain's rows and readback pairs, in
/// order) one round takes: as many as fit `most` rows and pairs, and the
/// first always.
fn round_len(
    sizes: impl Iterator<Item = (usize, usize)>,
    (most_rows, most_pairs): (usize, usize),
) -> usize {
    let (mut rows, mut pairs, mut n) = (0, 0, 0);
    for (r, p) in sizes {
        if n > 0 && (rows + r > most_rows || pairs + p > most_pairs) {
            break;
        }
        (rows, pairs, n) = (rows + r, pairs + p, n + 1);
    }
    n
}

impl<B: Slots> GpuModel<B>
where
    B::Seq: 'static,
{
    /// One store walk of several resident sequences' rows: each `(slot,
    /// feed)`'s rows checked and staged with its slot selected, the walk's
    /// row-wise launches once over every slot's rows, then each slot's
    /// append into its own store. Each slot's store and the positions it
    /// holds come out as its own store walk leaves them, and the selected
    /// slot stays selected. Refused by name: a slot named twice, rows past
    /// the program's [`WalkBounds::STORE_ROWS`], a poisoned model, and as
    /// the program refuses each slot's walk, before anything but the walk's
    /// scratch moves. A fault a launch raises stays on the card's fault
    /// word, which the next readback names.
    pub fn draft_store_slots<A>(
        &mut self,
        walks: &[(usize, Feed<'_, A>)],
        head: DraftHead,
    ) -> Result<(), GpuError>
    where
        B: SlotDraft<A>,
        A: Copy,
    {
        const WHAT: &str = "draft store_slots";
        self.refuse_if_poisoned(WHAT)?;
        refuse_twice(WHAT, walks.iter().map(|&(slot, _)| slot))?;
        let home = self.selected();
        let r = self.store_slots_at(walks, head);
        let back = self.select_slot(home);
        self.note_fault(WHAT, r.and(back))
    }

    /// [`GpuModel::draft_store_slots`]'s walk, its checks done.
    fn store_slots_at<A>(
        &mut self,
        walks: &[(usize, Feed<'_, A>)],
        head: DraftHead,
    ) -> Result<(), GpuError>
    where
        B: SlotDraft<A>,
        A: Copy,
    {
        const WHAT: &str = "draft store_slots";
        let mut blocks = Vec::with_capacity(walks.len());
        let mut ids = Vec::new();
        for &(slot, feed) in walks {
            let at = ids.len();
            let m = feed.tokens.len();
            let most = <B::Program as WalkBounds>::STORE_ROWS;
            if at + m > most {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{m} rows from row {at} of a store walk of at most {most} rows"),
                ));
            }
            self.select_slot(slot)?;
            let (gpu, w, body) = self.body_parts(WHAT)?;
            blocks.push(body.stage_store(gpu, w, feed, head, at)?);
            ids.extend_from_slice(feed.tokens);
        }
        if blocks.is_empty() {
            return Ok(());
        }
        let (gpu, w, body) = self.body_parts(WHAT)?;
        body.store_rows(gpu, w, &ids, &blocks)?;
        for (&(slot, _), &b) in walks.iter().zip(&blocks) {
            self.select_slot(slot)?;
            let (gpu, w, body) = self.body_parts(WHAT)?;
            body.append_store(gpu, w, b)?;
        }
        Ok(())
    }

    /// Several slots' window chains of the draft at once, one readback a
    /// round: each `(slot, refresh, own)` the chain the program runs on that
    /// slot alone, the chains' walks of each depth run as one walk over
    /// every slot's rows, its attention each slot's over its own store, and
    /// each slot's proposal and store come out as its own chain leaves them.
    /// Chains past one walk's rows (the program's [`WalkBounds::WALK_ROWS`])
    /// or readback ([`SlotDraft::ROUND_PAIRS`]) run in rounds of them
    /// (`round_len`), in order, and more chains than one call binds
    /// ([`MAX_PASS_ROWS`]) in calls of that many. The model stands with slot 0 selected
    /// ([`GpuModel::on_slots`]). Refused by name: a slot named twice, a
    /// poisoned model, and as the program refuses each chain, before its
    /// round's first walk; earlier rounds have run. A fault any walk raised
    /// poisons every slot its call named.
    pub fn draft_chain_slots<A>(
        &mut self,
        chains: &[SlotChain<'_, A>],
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Vec<Proposal>, GpuError>
    where
        B: SlotDraft<A>,
        A: Copy,
    {
        const WHAT: &str = "draft chain_slots";
        self.refuse_if_poisoned(WHAT)?;
        refuse_twice(WHAT, chains.iter().map(|&(slot, ..)| slot))?;
        if chains.len() > MAX_PASS_ROWS {
            let mut out = Vec::with_capacity(chains.len());
            for part in chains.chunks(MAX_PASS_ROWS) {
                out.extend(self.draft_chain_slots(part, head, mode)?);
            }
            return Ok(out);
        }
        let slots: Vec<usize> = chains.iter().map(|&(slot, ..)| slot).collect();
        self.on_slots(&slots, WHAT, |gpu, w, body, parked| {
            let mut seqs = Vec::with_capacity(chains.len());
            let mut next = 0..parked.len();
            for &(slot, ..) in chains {
                seqs.push(match slot {
                    0 => None,
                    _ => Some(next.next().ok_or(GpuError::state(
                        WHAT,
                        "a parked sequence for every slot but slot 0",
                    ))?),
                });
            }
            let most = (<B::Program as WalkBounds>::WALK_ROWS, B::ROUND_PAIRS);
            let mut out = Vec::with_capacity(chains.len());
            let mut start = 0;
            while start < chains.len() {
                let sizes = chains[start..]
                    .iter()
                    .map(|(_, feed, own)| (feed.tokens.len(), 1 + own));
                let end = start + round_len(sizes, most);
                let round = (&chains[start..end], &seqs[start..end]);
                out.extend(chain_round(body, gpu, w, parked, round, head, mode)?);
                start = end;
            }
            Ok(out)
        })
    }
}

/// One round of several slots' chains ([`GpuModel::draft_chain_slots`]),
/// the homes canonical: each chain's sequence the live one (`None`) or
/// `parked`'s entry its `seqs` index names. Each chain is checked and staged
/// with its sequence live, then each walk runs once over every chain it
/// reaches, each chain's attention with its sequence live, and the round is
/// read back once.
fn chain_round<A: Copy, B: SlotDraft<A>>(
    body: &mut B,
    gpu: &Gpu,
    w: &Weights,
    parked: &mut [&mut B::Seq],
    (chains, seqs): (&[SlotChain<'_, A>], &[Option<usize>]),
    head: DraftHead,
    mode: WalkMode,
) -> Result<Vec<Proposal>, GpuError> {
    let mut blocks: Vec<ChainBlock> = Vec::with_capacity(chains.len());
    let (mut at, mut pair) = (0, 0);
    for (block, (&(_, feed, own), &seq)) in chains.iter().zip(seqs).enumerate() {
        let seq = seq.map(|i| &mut *parked[i]);
        let b = on_seq(body, gpu, seq, |body| {
            body.stage_chain(gpu, w, feed, (own, head, mode), (block, at, pair))
        })?;
        at += b.m;
        pair += b.pairs();
        blocks.push(b);
    }
    body.upload_chains(gpu)?;
    let walks = blocks.iter().map(ChainBlock::pairs).max().unwrap_or(0);
    for j in 0..walks {
        body.chain_in(gpu, w, &blocks, j)?;
        for (b, at, m) in chain_rows(&blocks, j) {
            let seq = seqs[b.block].map(|i| &mut *parked[i]);
            on_seq(body, gpu, seq, |body| body.attend(gpu, w, &b, (at, m)))?;
        }
        body.chain_out(gpu, w, &blocks, j, head)?;
    }
    for (b, &seq) in blocks.iter().zip(seqs) {
        let seq = seq.map(|i| &mut *parked[i]);
        on_seq(body, gpu, seq, |body| body.end_chain(b))?;
    }
    body.read_chains(gpu, &blocks)
}

#[cfg(test)]
mod tests {
    use super::{ChainBlock, Held, chain_rows, refuse_twice, round_len};
    use crate::GpuError;

    /// A round's depths: walk 0 runs every block's refresh at its own rows,
    /// a later walk one row for each block whose chain still walks, packed
    /// in the blocks' order; a slot named twice is refused by name; a round
    /// takes chains up to its rows and pairs, and its first chain always.
    #[test]
    fn a_round_walks_each_chain_to_its_depth() {
        let block = |block, at, m, own, pair| ChainBlock {
            block,
            at,
            m,
            pos0: 10,
            own,
            pair,
        };
        let blocks = [
            block(0, 0, 3, 2, 0),
            block(1, 3, 1, 0, 3),
            block(2, 4, 2, 1, 4),
        ];
        let rows = |j| {
            chain_rows(&blocks, j)
                .map(|(b, at, m)| (b.block, at, m))
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(0), [(0, 0, 3), (1, 3, 1), (2, 4, 2)]);
        assert_eq!(rows(1), [(0, 0, 1), (2, 1, 1)]);
        assert_eq!(rows(2), [(0, 0, 1)]);
        assert!(rows(3).is_empty());
        assert_eq!((blocks[0].end(), blocks[0].pairs()), (13, 3));
        refuse_twice("round", [0, 2, 1].into_iter()).expect("three slots");
        let e = refuse_twice("round", [0, 2, 0].into_iter()).expect_err("slot 0 twice");
        assert!(matches!(e, GpuError::Shape { what: "round", .. }), "{e}");
        let round = |sizes: &[(usize, usize)]| round_len(sizes.iter().copied(), (6, 4));
        assert_eq!(round(&[(3, 1), (3, 2), (1, 1)]), 2);
        assert_eq!(round(&[(2, 2), (2, 2), (1, 1)]), 2);
        assert_eq!(round(&[(9, 9), (1, 1)]), 1);
    }

    #[test]
    fn held_rows_respect_row_and_position_widths_and_cuts() {
        let held = Held::at(1, 1);
        assert!(held.holds(0, 1, 1));
        assert!(!held.holds(usize::MAX, 1, 0));
        assert!(!held.holds(1, usize::MAX, 2));
        assert!(held.holds(1, 0, 2));
        assert!(!held.holds(2, 0, 3));
        assert!(Held::NONE.holds(0, 0, 0));
        assert!(!Held::NONE.holds(0, 1, 0));
        assert!(Held::at(u32::MAX, 1).holds(0, 1, u32::MAX));
        assert!(!Held::at(u32::MAX, 2).holds(0, 2, u32::MAX));
        assert!(!Held::at(u32::MAX, 2).holds(1, 1, 0));
        assert!(Held::at(u32::MAX, 2).shown().contains("4294967297"));
        if let Ok(past_u32) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(!Held::at(0, u32::MAX).holds(past_u32, 0, 0));
            assert!(!Held::at(0, u32::MAX).holds(0, past_u32, 0));
        }
        let mut held = Held::at(5, 3);
        held.cut(7);
        assert_eq!(held.parts(), (5, 2));
        assert!(held.holds(0, 2, 5));
        assert!(!held.holds(2, 1, 7));
        held.cut(4);
        assert_eq!(held.parts(), (5, 0));
        held.cut(9);
        assert_eq!(held.parts(), (5, 0));
    }

    #[test]
    fn target_reads_refuse_unheld_or_overflowing_ranges_before_copy() {
        const WHAT: &str = "draft read";
        let held = Held::at(1, 1);
        let e = held
            .check_read(WHAT, "Step", 0, (0, 1), 2)
            .expect_err("a held row outside the arena");
        assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        held.check_read(WHAT, "Step", 2, (0, 1), 2)
            .expect("the held row");
        for (first, m) in [
            (0, 3),
            (usize::MAX, 1),
            (1, usize::MAX),
            (usize::MAX, usize::MAX),
        ] {
            let e = held
                .check_read(WHAT, "Step", 2, (first, m), 1)
                .expect_err("rows outside the arena");
            assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        }
        let e = held
            .check_read(WHAT, "Step", usize::MAX, (0, usize::MAX), 2)
            .expect_err("a position end past usize");
        assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        for (held, rows, range, pos0) in [
            (held, 2, (0, 1), 0),
            (Held::NONE, 2, (0, 1), 2),
            (held, 2, (1, 1), 3),
            (Held::at(u32::MAX - 1, 3), 3, (0, 3), u32::MAX),
        ] {
            let e = held
                .check_read(WHAT, "Step", rows, range, pos0)
                .expect_err("a target read the arena cannot provide");
            assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
            if pos0 == u32::MAX {
                assert!(e.to_string().contains("4294967297"));
            }
        }
        let mut held = Held::at(5, 2);
        held.cut(6);
        held.check_read(WHAT, "Pair", 2, (0, 1), 6)
            .expect("the kept row remains readable");
        assert!(held.check_read(WHAT, "Pair", 2, (0, 2), 6).is_err());
    }
}
