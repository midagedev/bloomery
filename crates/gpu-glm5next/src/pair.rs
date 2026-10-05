//! The verify of two rows behind [`Rows`]: row 0 the current token at `pos`,
//! row 1 the drafted one at `pos + 1`, each through the one-token step's
//! launches in the step's order, the two rows one layer apart on the one
//! stream (`runtime::sched`'s point `(2, 1, Step)`, [`Chain::Pair`]) so the
//! host serves one row's layer while the card runs the other's. Row 1's
//! latent layers read the rows row 0 appended at the same layer just before
//! them; row 0 reads nothing row 1 writes. So the two rows' logits, every
//! latent row, pool key and conv ring slot, and the state after each row are
//! bit for bit two steps in turn.
//!
//! The KDA state is the one store a row cannot take back by position: row 0
//! writes the committed lane `c` in place, row 1 reads it and writes lane `c +
//! 1` (`linear::delta`'s `kda_delta_lanes` at row base 1). The verify then
//! waits for its commit ([`Lanes`]): keeping row 0 alone leaves the word at
//! `c`, keeping both moves it to `c + 1`, and either copies nothing. Every
//! other call — a step, a prompt, a checkpoint, another verify — is refused
//! by name until the commit (`GpuModel::rollback` to the first position not
//! kept, the verify's end when both are kept). The latent rows, the index
//! rows and the conv ring are indexed by position: a taken-back row 1 is
//! written again by the next step at `pos + 1` before any launch reads it,
//! the pool key it completed too.
//!
//! Only a load of two KDA lanes verifies (`place::KdaLanes::Two`: the NextN
//! load, [`Body::open_placed_lanes`] at two); on a load of one lane a verify
//! is refused by name before anything moves, its plan and its capture alike.
//!
//! A pass of two slots ([`SlotRows`], `GpuModel::step_slots`) walks the same
//! two rows one layer apart, each row a plain step of its own slot: its
//! embedding at its slot's position, its KDA launches in place on its
//! slot's committed lane (row base 0, whatever the load's lanes), its latent
//! rows and conv ring in its slot's stores, and its final streams in its
//! slot's own step row ([`binding`]), so each row is bit for bit its slot's
//! step alone. The rows in flight are the load's, so a load of one lane runs
//! it as a load of two does. The host half plans each row from its own
//! slot's sequence: the stores standing at its position, its waiting cut
//! carried out from its own checkpoints, its held position and step row
//! moved.

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::linear::delta::row_lane;
use bloomery_gpu::model::{ChainBody, Rows, SlotRange, SlotRows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, capturing};
use cuda_core::CudaStream;
use runtime::sched;
use runtime::seqstate::{Kept, Why};

use model::arch::glm5next::place::KdaLanes;

use super::nextn::{Held, Wrote};
use super::{
    Body, GlmSlot, LANES, Parts, Plant, RowScratch, Scratch, SeqParts, StepInput, Store, cut_into,
    held_at, shape,
};
use crate::program;

/// The rows one pass runs one layer apart, the load's rows in flight, whose
/// buffers every load holds whatever its KDA lanes: a verify's two (a row a
/// lane of the one sequence) or a pass of two slots' (a row a slot).
pub const PAIR_ROWS: usize = 2;

// A verify keeps the state after each of its rows in a lane of its own.
const _: () = assert!(PAIR_ROWS == LANES && LANES >= 1 && LANES <= u32::MAX as usize);

/// The host's side of the KDA lanes: the committed lane every launch reads
/// (the lane word's value), and the verify waiting for its commit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Lanes {
    committed: u32,
    waiting: Option<Waiting>,
}

/// A verify of `rows` rows from `pos0` whose chain was planned: its rows
/// stand in lanes `c .. c + rows` until the commit.
#[derive(Clone, Copy, Debug)]
struct Waiting {
    pos0: u32,
    rows: u32,
}

impl Lanes {
    /// The lane the lane word holds.
    pub(crate) fn committed(&self) -> u32 {
        self.committed
    }

    /// What a cut to at most `n` of a model standing at `pos` keeps while a
    /// verify waits and ran whole (the stores hold its rows, `held`): every
    /// position when `n` reaches them, `n` itself when it lies inside the
    /// verify's rows (the commit's rule); `None` otherwise, for the
    /// checkpoints' rule.
    pub(crate) fn kept(&self, n: u32, pos: u32, held: u32) -> Option<Kept> {
        let w = self.waiting?;
        if held != pos || held != w.pos0 + w.rows {
            return None;
        }
        let (at, why) = if n >= held {
            (held, Why::Current)
        } else if n > w.pos0 {
            (n, Why::Rule)
        } else {
            return None;
        };
        Some(Kept {
            asked: n,
            held,
            at,
            why,
        })
    }

    /// Refused by name while a verify waits for its commit: a call at `pos`
    /// would read a lane the commit has not named.
    pub(crate) fn refuse_if_waiting(&self, pos: u32) -> Result<(), GpuError> {
        match self.waiting {
            None => Ok(()),
            Some(w) => Err(shape(format!(
                "a call at position {pos} while the verify of {} rows at {} waits for its commit \
                 (GpuModel::rollback to the first position not kept, {} when every row is)",
                w.rows,
                w.pos0,
                w.pos0 + w.rows
            ))),
        }
    }
}

/// A row's inputs at position `p` into its buffers `r`: the embedding's
/// four stream copies `streams` into stream buffer 0, the position, the
/// visible counts and the live count.
fn write_row(
    stream: &CudaStream,
    r: &mut RowScratch,
    streams: &[f32],
    p: u32,
) -> Result<(), GpuError> {
    r.streams[0].copy_from_host(stream, streams)?;
    r.pos.copy_from_host(stream, &[p])?;
    r.vis.copy_from_host(stream, &[0, p + 1])?;
    r.cnt.copy_from_host(stream, &[p + 1])?;
    Ok(())
}

impl Body {
    /// One row's inputs at `input.pos` into row `row`'s buffers: the
    /// embedding's four stream copies, the position, the visible counts and
    /// the live count. Row 0 first carries out a waiting cut
    /// ([`Body::apply_cut`]). Once the copies are sent the stores count the
    /// position, and row 0's buffers — the step arena — hold it.
    pub(super) fn refresh_row(
        &mut self,
        stream: &CudaStream,
        input: &StepInput,
        row: usize,
    ) -> Result<(), GpuError> {
        if row == 0 {
            self.apply_cut(stream)?;
        }
        let p = input.pos;
        let r = self
            .s
            .row_mut(row)
            .ok_or_else(|| shape(format!("row {row} of a pass on a load of {PAIR_ROWS} rows")))?;
        write_row(stream, r, &self.embd.streams, p)?;
        self.held = p + 1;
        if row == 0 {
            self.wrote.step = super::nextn::Held::at(p, 1);
        }
        Ok(())
    }

    /// The verify's host half: `tokens[r]` at `pos + r` into row `r`'s
    /// buffers, in row order, then the verify left waiting for its commit and
    /// the pair arena holding its rows. Refused by name before anything
    /// moves: a load of one KDA lane, another row count, the taps armed (a
    /// tap holds one row), an id past the vocabulary, a verify already
    /// waiting, a position other than the stores' or whose rows pass them, a
    /// failure planted before the launch.
    pub(super) fn plan_pair(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        self.refuse_one_lane()?;
        if tokens.len() != PAIR_ROWS {
            return Err(shape(format!(
                "a verify of {} rows; the lanes hold {PAIR_ROWS}",
                tokens.len()
            )));
        }
        if self.taps.is_some() {
            return Err(shape(
                "a verify with the taps armed: a tap holds one row's streams".to_string(),
            ));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.embd.n_vocab) {
            return Err(shape(format!(
                "token {t} is past the {} embedding rows",
                self.embd.n_vocab
            )));
        }
        let carried = self.hybrid.boundary().rows();
        if carried < PAIR_ROWS {
            return Err(shape(format!(
                "a verify of {PAIR_ROWS} rows on a host boundary of {carried}: the load makes the \
                 boundary with a row for each of the verify's rows"
            )));
        }
        if pos as usize + PAIR_ROWS > self.ctx {
            return Err(shape(format!(
                "a verify of {PAIR_ROWS} rows at position {pos} in stores of {}",
                self.ctx
            )));
        }
        self.stores_at(pos)?;
        self.planted(Plant::BeforeLaunch)?;
        for (row, (&token, p)) in tokens.iter().zip(pos..).enumerate() {
            let input = self.decode_input(token, p)?;
            self.refresh_row(stream, &input, row)?;
        }
        self.s.lanes.waiting = Some(Waiting {
            pos0: pos,
            rows: PAIR_ROWS as u32,
        });
        self.wrote.pair = super::nextn::Held::at(pos, PAIR_ROWS as u32);
        Ok(())
    }

    /// Refused by name on a load of one KDA lane: a verify's row 1 writes a
    /// lane the load does not hold.
    fn refuse_one_lane(&self) -> Result<(), GpuError> {
        match self.lanes {
            KdaLanes::Two => Ok(()),
            KdaLanes::One => Err(shape(format!(
                "a verify of {PAIR_ROWS} rows on a load of one KDA lane: only a load that \
                 verifies holds the second (Body::open_placed_lanes at KdaLanes::Two, or the \
                 NextN load)"
            ))),
        }
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit:
    /// the lane word to the lane the last kept row wrote, or where it stands
    /// when that is row 0's. With no verify waiting, or a `pos` outside its
    /// rows, the cut [`Body::cut`] takes.
    pub(super) fn commit(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let Some(w) = self.s.lanes.waiting else {
            return self.cut(pos);
        };
        let end = w.pos0 + w.rows;
        if pos <= w.pos0 || pos > end || self.held != end {
            return self.cut(pos);
        }
        let c = self.s.lanes.committed;
        let lane = row_lane(c, pos - w.pos0 - 1, self.lanes.count() as u32);
        if lane != c {
            self.s.lane.copy_from_host(gpu.stream(), &[lane])?;
        }
        self.s.lanes = Lanes {
            committed: lane,
            waiting: None,
        };
        self.held = pos;
        self.wrote.cut(pos);
        if let Some(n) = self.nextn.as_deref_mut() {
            n.cut(pos);
        }
        Ok(())
    }

    /// A cut to `pos`: nothing at the fed position; the empty model at 0;
    /// else the checkpoint at `pos`, copied back into the committed lane at
    /// the next step. A cut into the rows of a verify that ran whole is its
    /// commit, which moves the lane word and so needs the card
    /// (`GpuModel::rollback`, [`Body::commit`]); any other position is
    /// refused by name ([`Body::kept`] says what a cut keeps). A cut drops a
    /// verify waiting below it.
    pub(super) fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
        if let Some(w) = self.s.lanes.waiting
            && pos > w.pos0
            && self.held == w.pos0 + w.rows
        {
            return Err(shape(format!(
                "back to position {pos} inside the verify of {} rows at {} without the card: its \
                 commit moves the lane word (GpuModel::rollback)",
                w.rows, w.pos0
            )));
        }
        self.ckpt.cut(pos, self.held)?;
        self.s.lanes.waiting = None;
        self.held = pos;
        self.wrote.cut(pos);
        if let Some(n) = self.nextn.as_deref_mut() {
            n.cut(pos);
        }
        Ok(())
    }

    /// The committed lane of every KDA layer's state.
    #[must_use]
    pub fn lane(&self) -> u32 {
        self.s.lanes.committed
    }
}

impl Rows for Body {
    const MAX_ROWS: usize = PAIR_ROWS;
    const CHAIN: Chain = Chain::Pair;

    /// [`Body::plan_pair`].
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.plan_pair(stream, tokens, pos)
    }

    /// The verify's walk into its two heads, row `r` into `heads[r]`. Outside
    /// a capture it needs its plan ([`Body::plan_pair`]); a capture records
    /// the launches, which read every per-row value from the rows' words.
    /// Refused by name on a load of one KDA lane, a capture too.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        self.refuse_one_lane()?;
        let [a, b] = heads else {
            return Err(shape(format!(
                "{} heads; the verify runs {PAIR_ROWS}",
                heads.len()
            )));
        };
        if self.s.lanes.waiting.is_none() && !capturing(gpu.stream())? {
            return Err(shape(
                "a verify with no plan (Rows::plan_rows before the pass)".to_string(),
            ));
        }
        let (parts, hybrid) = self.parts();
        program::walk_pair(gpu, w, parts, hybrid, [a, b])?;
        // A NextN load keeps row 0's streams past the step that writes its
        // buffers next (`nextn::GlmArena::Pair`): one copy more in the pass.
        if let Some(n) = self.nextn.as_deref_mut() {
            let fin = program::final_streams(self.cfg.len());
            n.pair0_mut()
                .copy_from_device_async(&self.s.row0.streams[fin], gpu.stream())?;
        }
        self.planted(Plant::AfterLaunch)
    }
}

// ------------------------------------------------- a pass of several slots

/// The rows the last plan of a pass of several slots wrote, each row's slot
/// and position: what an enqueue outside a capture runs on, and only on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotsPlanned {
    at: [(usize, u32); PAIR_ROWS],
    n: usize,
}

impl SlotsPlanned {
    fn of(rows: &[SlotRange]) -> SlotsPlanned {
        let mut at = [(0, 0); PAIR_ROWS];
        for (a, r) in at.iter_mut().zip(rows) {
            *a = (r.slot, r.pos0);
        }
        SlotsPlanned { at, n: rows.len() }
    }
}

/// The chain a pass of `slots` slots runs: [`sched::slot_lanes`]' point on a
/// page of one column a row — the step for one slot, the pair for two;
/// `None` for a point the walk does not lay.
fn slots_chain(slots: usize) -> Option<Chain> {
    let o = sched::slot_lanes(slots).ok()?.overlap;
    Chain::of(o.units, o.cols, 1)
}

/// One exchange of handles of a pass of several slots' final-stream binding
/// ([`binding`]).
#[derive(Clone, Copy, Debug)]
enum Swap {
    /// Row 0's final streams with row 1's.
    Rows,
    /// Row `row`'s final streams with the step row of the pass's parked
    /// sequence `seq`.
    Parked { row: usize, seq: usize },
}

/// The exchanges that put each row's slot's own step row of final streams
/// (its [`GlmSlot`] row 0, which the draft's walks and the sequence's state
/// read) in that row's buffer for a pass of `rows`, in order: the live
/// sequence's is row 0's buffer, so a live slot in row 1 first exchanges the
/// two rows', then each parked slot's goes into its row. Pointer moves
/// only; [`exchange`] undoes them in reverse order. A pass of at most
/// [`PAIR_ROWS`] slots ([`Body::refuse_slots`]) takes at most two.
fn binding(rows: &[SlotRange]) -> [Option<Swap>; PAIR_ROWS] {
    let mut out = [None; PAIR_ROWS];
    let rows_first = rows
        .get(1)
        .is_some_and(|r| r.slot == 0)
        .then_some(Swap::Rows);
    let parked = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.slot != 0)
        .enumerate()
        .map(|(seq, (row, _))| Swap::Parked { row, seq });
    for (o, sw) in out.iter_mut().zip(rows_first.into_iter().chain(parked)) {
        *o = Some(sw);
    }
    out
}

/// Carry out the exchanges `swaps` ([`binding`]) between the rows' buffers
/// `s` and the pass's `parked` sequences on stream buffer `fin`, in order,
/// or undo them, in reverse when `undo`.
fn exchange(
    s: &mut Scratch,
    parked: &mut [&mut GlmSlot],
    fin: usize,
    swaps: &[Option<Swap>; PAIR_ROWS],
    undo: bool,
) -> Result<(), GpuError> {
    let mut one = |sw: Swap| -> Result<(), GpuError> {
        match sw {
            Swap::Rows => {
                let Scratch { row0, row1, .. } = &mut *s;
                std::mem::swap(&mut row0.streams[fin], &mut row1.streams[fin]);
            }
            Swap::Parked { row, seq } => {
                let r = s
                    .row_mut(row)
                    .ok_or_else(|| shape(format!("row {row} of a pass of several slots")))?;
                let own = parked
                    .get_mut(seq)
                    .and_then(|p| p.rows.first_mut())
                    .ok_or_else(|| shape(format!("parked sequence {seq}'s step row")))?;
                std::mem::swap(&mut r.streams[fin], own);
            }
        }
        Ok(())
    };
    if undo {
        swaps.iter().rev().flatten().try_for_each(|&sw| one(sw))
    } else {
        swaps.iter().flatten().try_for_each(|&sw| one(sw))
    }
}

/// `e` with `what` before its detail, when it is a shape refusal.
fn prefixed(e: GpuError, what: &str) -> GpuError {
    match e {
        GpuError::Shape { what: at, detail } => GpuError::Shape {
            what: at,
            detail: format!("{what}: {detail}"),
        },
        e => e,
    }
}

/// One sequence's host side a pass of several slots plans a row of, lent
/// apart: the live sequence's fields or a parked [`GlmSlot`]'s.
struct SeqHost<'a> {
    stores: &'a mut [Store],
    lanes: &'a Lanes,
    held: &'a mut u32,
    ckpt: &'a mut Checkpoints,
    wrote: &'a mut Wrote,
}

impl<'a> SeqHost<'a> {
    fn parked(p: &'a mut GlmSlot) -> SeqHost<'a> {
        SeqHost {
            stores: &mut p.stores,
            lanes: &p.lanes,
            held: &mut p.held,
            ckpt: &mut p.ckpt,
            wrote: &mut p.wrote,
        }
    }
}

impl Body {
    /// Refused by name before anything moves, a pass of several slots of
    /// `rows` whose busy slots other than slot 0 are `parked`: on a NextN
    /// load, with the taps armed or a route trace attached, at a point the
    /// walk does not lay, a slot of other than one row, while the live
    /// sequence's verify waits for its commit, and `parked` not the pass's
    /// other sequences.
    fn refuse_slots(&self, parked: &[&mut GlmSlot], rows: &[SlotRange]) -> Result<(), GpuError> {
        if self.nextn.is_some() {
            return Err(shape(
                "a pass of several slots on a NextN load: each of its slots is drafted, and a \
                 pass of plain steps tells no slot's draft what it ran"
                    .to_string(),
            ));
        }
        if self.taps.is_some() {
            return Err(shape(
                "a pass of several slots with the taps armed: a tap holds one row's streams"
                    .to_string(),
            ));
        }
        if self.hybrid.route_traced() {
            return Err(shape(
                "a pass of several slots with a route trace attached: the trace records one \
                 sequence's positions"
                    .to_string(),
            ));
        }
        if slots_chain(rows.len()).is_none() {
            return Err(shape(format!(
                "a pass of {} slots: the walk lays one or {PAIR_ROWS}, a row a slot",
                rows.len()
            )));
        }
        if let Some(r) = rows.iter().find(|r| r.rows.len() != 1) {
            return Err(shape(format!(
                "slot {}'s {} rows in a pass of several slots: each slot's row is one plain \
                 step; a verify of one slot's rows is Rows::step_rows",
                r.slot,
                r.rows.len()
            )));
        }
        let want = rows.iter().filter(|r| r.slot != 0).count();
        if parked.len() != want || parked.iter().any(|p| p.rows.is_empty()) {
            return Err(shape(format!(
                "{} parked sequences for a pass whose other busy slots are {want}",
                parked.len()
            )));
        }
        self.s
            .lanes
            .refuse_if_waiting(self.held)
            .map_err(|e| prefixed(e, "a pass of several slots"))
    }

    /// Each row's slot's sequence standing at its row's position, its token
    /// in the vocabulary and its position in the stores (`tokens[i]` row
    /// `i`'s), refused by name naming the slot.
    fn refuse_slot_rows(
        &self,
        parked: &[&mut GlmSlot],
        rows: &[SlotRange],
        tokens: &[u32; PAIR_ROWS],
    ) -> Result<(), GpuError> {
        let mut parked = parked.iter();
        for (r, &t) in rows.iter().zip(tokens) {
            let named = |e| prefixed(e, &format!("slot {}", r.slot));
            if t as usize >= self.embd.n_vocab {
                return Err(named(shape(format!(
                    "token {t} is past the {} embedding rows",
                    self.embd.n_vocab
                ))));
            }
            if r.pos0 as usize >= self.ctx {
                return Err(named(shape(format!(
                    "a step at position {} in stores of {}",
                    r.pos0, self.ctx
                ))));
            }
            let (lanes, held) = if r.slot == 0 {
                (&self.s.lanes, self.held)
            } else {
                let p = parked
                    .next()
                    .ok_or_else(|| shape(format!("slot {}'s parked sequence", r.slot)))?;
                (&p.lanes, p.held)
            };
            held_at(lanes, held, r.pos0).map_err(named)?;
        }
        Ok(())
    }

    /// Each row's inputs into its buffer, as the step's refresh writes row
    /// 0's, and its slot's sequence moved as the step moves the live one:
    /// its waiting cut carried out, its stores holding the row's position,
    /// its step row holding it. The rows' buffers are bound ([`binding`]).
    fn write_slot_rows(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
        tokens: &[u32; PAIR_ROWS],
    ) -> Result<(), GpuError> {
        let Body {
            embd,
            s,
            stores,
            held,
            ckpt,
            wrote,
            ..
        } = self;
        let Scratch {
            row0, row1, lanes, ..
        } = s;
        let mut live = Some(SeqHost {
            stores,
            lanes,
            held,
            ckpt,
            wrote,
        });
        let mut parked = parked.iter_mut();
        for ((r, &t), buf) in rows.iter().zip(tokens).zip([row0, row1]) {
            let seq = if r.slot == 0 {
                live.take()
            } else {
                parked.next().map(|p| SeqHost::parked(p))
            }
            .ok_or_else(|| shape(format!("slot {}'s sequence in the pass", r.slot)))?;
            embd.fill(t)?;
            let lane = seq.lanes.committed();
            cut_into(stream, seq.ckpt, seq.stores, lane, *seq.held)?;
            write_row(stream, buf, &embd.streams, r.pos0)?;
            *seq.held = r.pos0 + 1;
            seq.wrote.step = Held::at(r.pos0, 1);
        }
        Ok(())
    }

    /// The pass of `rows` walked with each row bound to its slot's lane word
    /// and stores ([`Parts`]), row `r` into `heads[r]`. The rows' buffers
    /// are bound ([`binding`]).
    fn walk_slot_rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        let Body {
            hybrid,
            cfg,
            names,
            dims,
            k,
            s,
            stores,
            slots,
            card,
            ..
        } = self;
        let Scratch {
            row0, row1, lane, ..
        } = s;
        let mut live = Some(SeqParts {
            lane: &*lane,
            stores,
        });
        let mut parked = parked.iter_mut();
        let mut seqs = [None, None];
        for (seq, r) in seqs.iter_mut().zip(rows) {
            *seq = Some(
                if r.slot == 0 {
                    live.take()
                } else {
                    parked.next().map(|p| SeqParts {
                        lane: &p.lane,
                        stores: &mut p.stores,
                    })
                }
                .ok_or_else(|| shape(format!("slot {}'s sequence in the pass", r.slot)))?,
            );
        }
        let [first, other] = seqs;
        let first = first.ok_or_else(|| shape("a pass of no slot".to_string()))?;
        let parts = Parts {
            k,
            d: dims,
            cfg,
            names,
            s: row0,
            idle: row1,
            row: 0,
            lane: first.lane,
            stores: first.stores,
            other,
            slots,
            card,
            taps: None,
        };
        program::walk_slots(gpu, w, parts, hybrid, heads)
    }
}

impl SlotRows for Body {
    /// Two rows in flight, a row a slot: the step port's pair until its
    /// rows carry columns.
    const MAX_ROWS: usize = PAIR_ROWS;

    /// [`sched::slot_lanes`]' point over the busy slots: the step for one,
    /// the pair for two. A count the walk does not lay is refused by name
    /// before any capture or launch of it ([`Body::refuse_slots`]), so no
    /// replay of one is served and its arm is never read.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        slots_chain(rows.len()).unwrap_or(Chain::Step)
    }

    /// Each row planned from its own slot's sequence (module doc): refused
    /// by name before anything moves as [`Body::refuse_slots`] and each
    /// row's slot ([`Body::refuse_slot_rows`]) refuse, and on a failure
    /// planted before the launch; then each row's inputs written and its
    /// slot's sequence moved, every row's final streams its own slot's step
    /// row's while they are written.
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        self.refuse_slots(parked, rows)?;
        let mut tokens = [0; PAIR_ROWS];
        for (t, r) in tokens.iter_mut().zip(rows) {
            *t = *ids.get(r.rows.start).ok_or_else(|| {
                shape(format!(
                    "slot {}'s id past the pass's {}",
                    r.slot,
                    ids.len()
                ))
            })?;
        }
        self.refuse_slot_rows(parked, rows, &tokens)?;
        self.planted(Plant::BeforeLaunch)?;
        let fin = program::final_streams(self.cfg.len());
        let swaps = binding(rows);
        exchange(&mut self.s, parked, fin, &swaps, false)?;
        let wrote = self.write_slot_rows(stream, parked, rows, &tokens);
        exchange(&mut self.s, parked, fin, &swaps, true)?;
        wrote?;
        self.planned = Some(SlotsPlanned::of(rows));
        Ok(())
    }

    /// The pass's walk, each row bound to its own slot's lane word, stores
    /// and step row, row `r` into `heads[r]`. Outside a capture it needs its
    /// plan ([`SlotRows::plan_slots`] of the same rows); a capture records
    /// the launches, which read every per-row value from the rows' words and
    /// every slot's buffers by address. Refused by name as
    /// [`Body::refuse_slots`] refuses, and for other than a head a row.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        self.refuse_slots(parked, rows)?;
        if heads.len() != rows.len() {
            return Err(shape(format!(
                "{} heads for a pass of {} rows",
                heads.len(),
                rows.len()
            )));
        }
        let captured = capturing(gpu.stream())?;
        if !captured && self.planned != Some(SlotsPlanned::of(rows)) {
            return Err(shape(
                "a pass of several slots with no plan of its rows (SlotRows::plan_slots before \
                 the pass)"
                    .to_string(),
            ));
        }
        let fin = program::final_streams(self.cfg.len());
        let swaps = binding(rows);
        exchange(&mut self.s, parked, fin, &swaps, false)?;
        let walked = self.walk_slot_rows(gpu, w, heads, parked, rows);
        exchange(&mut self.s, parked, fin, &swaps, true)?;
        walked?;
        if !captured {
            self.planned = None;
        }
        self.planted(Plant::AfterLaunch)
    }
}
